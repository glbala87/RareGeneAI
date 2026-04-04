"""Structural variant VCF parser.

Parses SV VCFs from long-read callers (Sniffles2, Jasmine, cuteSV)
and short-read callers (Manta, DELLY, Lumpy). Handles the diverse
INFO field conventions across callers.

Extracts SVTYPE, SVLEN, END, breakpoint coordinates, read support,
and genotype for each structural variant.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

from loguru import logger

from raregeneai.config.settings import SVConfig
from raregeneai.models.data_models import StructuralVariant, SVType, Zygosity


# Map SVTYPE strings from various callers to SVType enum
_SVTYPE_MAP = {
    "DEL": SVType.DEL,
    "DUP": SVType.DUP,
    "INV": SVType.INV,
    "INS": SVType.INS,
    "BND": SVType.BND,
    "CNV": SVType.CNV,
    "DUP:TANDEM": SVType.DUP_TANDEM,
    "DUP/TANDEM": SVType.DUP_TANDEM,
    "DEL:ME": SVType.DEL_ME,
    "INS:ME": SVType.INS_ME,
    "TRA": SVType.BND,  # Some callers use TRA for translocations
}


class SVParser:
    """Parse structural variant VCF files."""

    def __init__(self, config: SVConfig | None = None):
        self.config = config or SVConfig()

    def parse(
        self,
        sv_vcf_path: str | Path,
        sample_id: str | None = None,
    ) -> list[StructuralVariant]:
        """Parse SV VCF file into StructuralVariant objects.

        Supports:
          - Sniffles2 / Sniffles output
          - Jasmine (merged SV calls)
          - cuteSV
          - Manta
          - DELLY
          - Generic SV VCF (VCFv4.x with SVTYPE INFO)

        Args:
            sv_vcf_path: Path to SV VCF file.
            sample_id: Specific sample to extract (None = first).

        Returns:
            Filtered list of StructuralVariant objects.
        """
        sv_vcf_path = Path(sv_vcf_path)
        if not sv_vcf_path.exists():
            raise FileNotFoundError(f"SV VCF not found: {sv_vcf_path}")

        logger.info(f"Parsing SV VCF: {sv_vcf_path}")

        svs = self._parse_vcf(sv_vcf_path, sample_id)
        filtered = self._apply_filters(svs)

        logger.info(
            f"Parsed {len(svs)} SVs, {len(filtered)} passed filters "
            f"(DEL={sum(1 for s in filtered if s.is_deletion)}, "
            f"DUP={sum(1 for s in filtered if s.is_duplication)}, "
            f"INV={sum(1 for s in filtered if s.sv_type == SVType.INV)}, "
            f"INS={sum(1 for s in filtered if s.sv_type == SVType.INS)}, "
            f"BND={sum(1 for s in filtered if s.sv_type == SVType.BND)})"
        )
        return filtered

    def _parse_vcf(
        self, vcf_path: Path, sample_id: str | None
    ) -> list[StructuralVariant]:
        """Parse VCF lines into StructuralVariant objects."""
        open_fn = gzip.open if str(vcf_path).endswith(".gz") else open
        svs = []
        samples: list[str] = []
        sample_idx = 0
        caller = self._detect_caller(vcf_path)

        with open_fn(vcf_path, "rt") as f:
            for line in f:
                line = line.strip()

                # Header: detect caller from meta-information
                if line.startswith("##"):
                    if "sniffles" in line.lower():
                        caller = "sniffles"
                    elif "jasmine" in line.lower():
                        caller = "jasmine"
                    elif "manta" in line.lower():
                        caller = "manta"
                    elif "delly" in line.lower():
                        caller = "delly"
                    elif "cutesv" in line.lower():
                        caller = "cutesv"
                    continue

                # Column header
                if line.startswith("#CHROM"):
                    fields = line.split("\t")
                    samples = fields[9:] if len(fields) > 9 else []
                    if sample_id and sample_id in samples:
                        sample_idx = samples.index(sample_id)
                    continue

                if not line:
                    continue

                sv = self._parse_sv_record(line, samples, sample_idx, caller)
                if sv:
                    svs.append(sv)

        return svs

    def _parse_sv_record(
        self,
        line: str,
        samples: list[str],
        sample_idx: int,
        caller: str,
    ) -> StructuralVariant | None:
        """Parse a single VCF line into a StructuralVariant."""
        fields = line.split("\t")
        if len(fields) < 8:
            return None

        chrom = fields[0]
        pos = int(fields[1])
        sv_id = fields[2] if fields[2] != "." else ""
        ref = fields[3]
        alt = fields[4]
        qual = float(fields[5]) if fields[5] != "." else 0.0
        filt = fields[6]

        info = self._parse_info(fields[7])

        # Extract SVTYPE
        svtype_str = info.get("SVTYPE", "")
        if not svtype_str:
            # Try to infer from ALT field
            if alt.startswith("<") and alt.endswith(">"):
                svtype_str = alt[1:-1]
            elif "]" in alt or "[" in alt:
                svtype_str = "BND"
            else:
                return None  # Not an SV record

        sv_type = _SVTYPE_MAP.get(svtype_str.upper(), SVType.UNKNOWN)
        if sv_type == SVType.UNKNOWN:
            return None

        # Extract END and SVLEN
        end = int(info.get("END", pos))
        svlen = abs(int(info.get("SVLEN", "0").lstrip("-")))
        if svlen == 0 and end > pos:
            svlen = end - pos

        # BND: extract mate chromosome/position
        chrom2 = info.get("CHR2", "")
        pos2 = int(info.get("POS2", "0"))
        if not chrom2 and sv_type == SVType.BND:
            chrom2, pos2 = self._parse_bnd_alt(alt)

        # Genotype extraction
        zygosity = Zygosity.UNKNOWN
        genotype = ""
        support_reads = 0
        ref_reads = 0
        allele_fraction = 0.0

        if len(fields) > 9:
            fmt_fields = fields[8].split(":")
            sample_data = fields[9 + sample_idx].split(":") if len(fields) > 9 + sample_idx else []
            fmt_dict = dict(zip(fmt_fields, sample_data))

            genotype = fmt_dict.get("GT", "./.")
            zygosity = self._parse_zygosity(genotype)

            # Read support varies by caller
            support_reads = self._extract_support_reads(fmt_dict, info, caller)
            ref_reads = self._extract_ref_reads(fmt_dict, info, caller)

            total = support_reads + ref_reads
            allele_fraction = support_reads / total if total > 0 else 0.0

        # Precision
        confidence = "PRECISE" if "PRECISE" in fields[7] else "IMPRECISE"

        return StructuralVariant(
            chrom=chrom,
            pos=pos,
            end=end,
            sv_type=sv_type,
            sv_len=svlen,
            sv_id=sv_id,
            quality=qual,
            filter_status=filt,
            chrom2=chrom2,
            pos2=pos2,
            zygosity=zygosity,
            genotype=genotype,
            sample_id=samples[sample_idx] if samples else "",
            support_reads=support_reads,
            ref_reads=ref_reads,
            allele_fraction=allele_fraction,
            caller=caller,
            confidence=confidence,
        )

    def _apply_filters(self, svs: list[StructuralVariant]) -> list[StructuralVariant]:
        """Apply quality and size filters."""
        filtered = []
        for sv in svs:
            # Size filter
            if sv.sv_len < self.config.min_sv_size and sv.sv_type != SVType.BND:
                continue
            if sv.sv_len > self.config.max_sv_size:
                continue

            # Quality filter
            if sv.quality < self.config.min_sv_quality and sv.quality > 0:
                continue

            # Read support filter
            if sv.support_reads < self.config.min_support_reads and sv.support_reads > 0:
                continue

            # Must carry at least one alt allele
            if sv.zygosity == Zygosity.HOM_REF:
                continue

            # FILTER field
            if sv.filter_status not in ("PASS", ".", ""):
                continue

            filtered.append(sv)

        return filtered

    # ── Helper methods ─────────────────────────────────────────────────────

    @staticmethod
    def _parse_info(info_str: str) -> dict[str, str]:
        """Parse VCF INFO field into dict."""
        info = {}
        for field in info_str.split(";"):
            if "=" in field:
                key, val = field.split("=", 1)
                info[key] = val
            else:
                info[field] = "1"  # Flag fields
        return info

    @staticmethod
    def _parse_zygosity(gt_str: str) -> Zygosity:
        """Parse genotype string to Zygosity."""
        sep = "|" if "|" in gt_str else "/"
        alleles = gt_str.split(sep)
        try:
            a = [int(x) for x in alleles if x != "."]
        except ValueError:
            return Zygosity.UNKNOWN

        if not a:
            return Zygosity.UNKNOWN
        if len(a) == 1:
            return Zygosity.HEMIZYGOUS if a[0] > 0 else Zygosity.HOM_REF
        if all(x == 0 for x in a):
            return Zygosity.HOM_REF
        if all(x > 0 for x in a):
            return Zygosity.HOM_ALT
        return Zygosity.HET

    @staticmethod
    def _parse_bnd_alt(alt: str) -> tuple[str, int]:
        """Parse BND ALT field to extract mate chrom and pos.

        BND formats:  ]chr2:321681]T  or  T[chr2:321681[
        """
        match = re.search(r"[\[\]](\w+):(\d+)[\[\]]", alt)
        if match:
            return match.group(1), int(match.group(2))
        return "", 0

    @staticmethod
    def _extract_support_reads(
        fmt: dict[str, str], info: dict[str, str], caller: str
    ) -> int:
        """Extract supporting read count (caller-specific fields)."""
        # Try FORMAT fields first (most common)
        for key in ("DV", "SR", "AD", "RV", "AO"):
            if key in fmt:
                val = fmt[key]
                if key == "AD":
                    # AD = ref,alt
                    parts = val.split(",")
                    return int(parts[1]) if len(parts) > 1 else 0
                try:
                    return int(val)
                except ValueError:
                    continue

        # Try INFO fields
        for key in ("SUPPORT", "RE", "SR", "PE"):
            if key in info:
                try:
                    return int(info[key])
                except ValueError:
                    continue

        return 0

    @staticmethod
    def _extract_ref_reads(
        fmt: dict[str, str], info: dict[str, str], caller: str
    ) -> int:
        """Extract reference read count."""
        for key in ("DR", "RR", "RO"):
            if key in fmt:
                try:
                    return int(fmt[key])
                except ValueError:
                    continue
        if "AD" in fmt:
            parts = fmt["AD"].split(",")
            try:
                return int(parts[0])
            except (ValueError, IndexError):
                pass
        return 0

    @staticmethod
    def _detect_caller(vcf_path: Path) -> str:
        """Try to detect caller from filename."""
        name = vcf_path.stem.lower()
        for caller in ("sniffles", "jasmine", "manta", "delly", "cutesv", "lumpy"):
            if caller in name:
                return caller
        return "unknown"
