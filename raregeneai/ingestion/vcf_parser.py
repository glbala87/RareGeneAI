"""VCF file parsing and variant extraction using cyvcf2.

Handles multi-sample VCFs, multi-allelic splitting, left-alignment,
and quality filtering. Outputs structured Variant objects.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from raregeneai.config.settings import IngestionConfig
from raregeneai.models.data_models import Variant, Zygosity


class VCFParser:
    """Parse VCF files into structured variant records."""

    def __init__(self, config: IngestionConfig | None = None):
        self.config = config or IngestionConfig()

    def parse(self, vcf_path: str | Path, sample_id: str | None = None) -> list[Variant]:
        """Parse VCF file and return list of Variant objects.

        Args:
            vcf_path: Path to VCF/VCF.gz file.
            sample_id: Specific sample to extract (None = first sample).

        Returns:
            List of quality-filtered Variant objects.
        """
        vcf_path = Path(vcf_path)
        if not vcf_path.exists():
            raise FileNotFoundError(f"VCF file not found: {vcf_path}")

        logger.info(f"Parsing VCF: {vcf_path}")

        # Normalize first if configured
        normalized_path = vcf_path
        if self.config.split_multiallelic or self.config.left_align:
            normalized_path = self._normalize_vcf(vcf_path)

        variants = self._extract_variants(normalized_path, sample_id)
        logger.info(f"Extracted {len(variants)} variants after QC filtering")
        return variants

    def parse_to_dataframe(self, vcf_path: str | Path, sample_id: str | None = None) -> pd.DataFrame:
        """Parse VCF and return as pandas DataFrame."""
        variants = self.parse(vcf_path, sample_id)
        records = []
        for v in variants:
            records.append({
                "chrom": v.chrom,
                "pos": v.pos,
                "ref": v.ref,
                "alt": v.alt,
                "variant_key": v.variant_key,
                "quality": v.quality,
                "zygosity": v.zygosity.value,
                "genotype": v.genotype,
                "depth": v.depth,
                "gq": v.gq,
                "ad_ref": v.allele_depth_ref,
                "ad_alt": v.allele_depth_alt,
                "sample_id": v.sample_id,
                "is_snv": v.is_snv,
                "is_indel": v.is_indel,
            })
        return pd.DataFrame(records)

    def _normalize_vcf(self, vcf_path: Path) -> Path:
        """Normalize VCF: split multi-allelic and left-align using bcftools."""
        output_path = vcf_path.parent / f"{vcf_path.stem}.normalized.vcf.gz"

        if output_path.exists():
            logger.info(f"Using existing normalized VCF: {output_path}")
            return output_path

        cmd_parts = ["bcftools", "norm"]

        if self.config.split_multiallelic:
            cmd_parts.extend(["-m", "-any"])

        if self.config.left_align:
            cmd_parts.extend(["--fasta-ref", ""])  # Placeholder - needs ref genome

        cmd_parts.extend(["-o", str(output_path), "-O", "z", str(vcf_path)])

        try:
            logger.info(f"Normalizing VCF with bcftools: {' '.join(cmd_parts)}")
            subprocess.run(cmd_parts, check=True, capture_output=True, text=True)
            # Index the output
            subprocess.run(["bcftools", "index", str(output_path)], check=True, capture_output=True)
            return output_path
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"bcftools normalization failed ({e}), using original VCF")
            return vcf_path

    def _extract_variants(self, vcf_path: Path, sample_id: str | None) -> list[Variant]:
        """Extract variants using cyvcf2."""
        try:
            from cyvcf2 import VCF
        except ImportError:
            logger.warning("cyvcf2 not available, falling back to pure-Python parser")
            return self._extract_variants_fallback(vcf_path, sample_id)

        vcf = VCF(str(vcf_path))
        samples = vcf.samples

        if not samples:
            raise ValueError("VCF contains no samples")

        sample_idx = 0
        if sample_id:
            if sample_id not in samples:
                raise ValueError(f"Sample '{sample_id}' not found. Available: {samples}")
            sample_idx = samples.index(sample_id)

        actual_sample_id = samples[sample_idx]
        logger.info(f"Extracting variants for sample: {actual_sample_id}")

        variants = []
        total = 0
        filtered = 0

        for record in vcf:
            total += 1

            for alt_idx, alt in enumerate(record.ALT):
                gt = record.genotypes[sample_idx]
                alleles = gt[:2]
                phased = gt[2] if len(gt) > 2 else False

                # Determine zygosity
                alt_allele = alt_idx + 1
                if alleles[0] == alt_allele and alleles[1] == alt_allele:
                    zygosity = Zygosity.HOM_ALT
                elif alt_allele in alleles:
                    zygosity = Zygosity.HET
                else:
                    continue  # This alt not present in this sample

                # Extract quality metrics
                dp = self._get_format_int(record, sample_idx, "DP")
                gq = self._get_format_int(record, sample_idx, "GQ")
                ad = self._get_format_array(record, sample_idx, "AD")

                ad_ref = ad[0] if ad and len(ad) > 0 else 0
                ad_alt_val = ad[alt_allele] if ad and len(ad) > alt_allele else 0

                # Quality filtering
                if dp < self.config.min_dp:
                    filtered += 1
                    continue
                if gq < self.config.min_gq:
                    filtered += 1
                    continue

                gt_str = f"{alleles[0]}{'|' if phased else '/'}{alleles[1]}"

                # Extract INFO fields for fallback annotation
                info_dict = {}
                try:
                    info_str = str(record).split("\t")[7] if len(str(record).split("\t")) > 7 else ""
                    for field in info_str.split(";"):
                        if "=" in field:
                            k, v = field.split("=", 1)
                            info_dict[k] = v
                except Exception:
                    pass

                variant = Variant(
                    chrom=record.CHROM,
                    pos=record.POS,
                    ref=record.REF,
                    alt=alt,
                    variant_id=record.ID or "",
                    quality=record.QUAL or 0.0,
                    zygosity=zygosity,
                    genotype=gt_str,
                    depth=dp,
                    gq=gq,
                    allele_depth_ref=ad_ref,
                    allele_depth_alt=ad_alt_val,
                    sample_id=actual_sample_id,
                    info_fields=info_dict,
                )
                variants.append(variant)

        logger.info(f"Total records: {total}, Passed QC: {len(variants)}, Filtered: {filtered}")
        return variants

    def _extract_variants_fallback(self, vcf_path: Path, sample_id: str | None) -> list[Variant]:
        """Pure-Python VCF parser fallback when cyvcf2 is unavailable."""
        import gzip

        open_fn = gzip.open if str(vcf_path).endswith(".gz") else open
        variants = []
        samples = []
        sample_idx = 0

        with open_fn(vcf_path, "rt") as f:
            for line in f:
                line = line.strip()
                if line.startswith("##"):
                    continue
                if line.startswith("#CHROM"):
                    fields = line.split("\t")
                    samples = fields[9:]
                    if sample_id and sample_id in samples:
                        sample_idx = samples.index(sample_id)
                    continue
                if not line:
                    continue

                fields = line.split("\t")
                if len(fields) < 10:
                    continue

                chrom, pos, vid, ref, alt_str, qual, filt, info, fmt = fields[:9]
                sample_data = fields[9 + sample_idx]

                fmt_fields = fmt.split(":")
                sample_values = sample_data.split(":")
                fmt_dict = dict(zip(fmt_fields, sample_values))

                for alt in alt_str.split(","):
                    gt_str = fmt_dict.get("GT", "./.")
                    sep = "|" if "|" in gt_str else "/"
                    alleles = gt_str.split(sep)

                    try:
                        a1, a2 = int(alleles[0]), int(alleles[1])
                    except (ValueError, IndexError):
                        continue

                    alt_idx = alt_str.split(",").index(alt) + 1
                    if alt_idx not in (a1, a2):
                        continue

                    if a1 == alt_idx and a2 == alt_idx:
                        zygosity = Zygosity.HOM_ALT
                    else:
                        zygosity = Zygosity.HET

                    dp = int(fmt_dict.get("DP", "0"))
                    gq = int(fmt_dict.get("GQ", "0"))
                    ad = fmt_dict.get("AD", "0,0").split(",")
                    ad_ref = int(ad[0]) if len(ad) > 0 else 0
                    ad_alt_val = int(ad[alt_idx]) if len(ad) > alt_idx else 0

                    if dp < self.config.min_dp or gq < self.config.min_gq:
                        continue

                    # Parse INFO fields
                    info_dict = {}
                    for inf in info.split(";"):
                        if "=" in inf:
                            k, v = inf.split("=", 1)
                            info_dict[k] = v

                    variant = Variant(
                        chrom=chrom,
                        pos=int(pos),
                        ref=ref,
                        alt=alt,
                        variant_id=vid if vid != "." else "",
                        quality=float(qual) if qual != "." else 0.0,
                        zygosity=zygosity,
                        genotype=gt_str,
                        depth=dp,
                        gq=gq,
                        allele_depth_ref=ad_ref,
                        allele_depth_alt=ad_alt_val,
                        sample_id=samples[sample_idx] if samples else "",
                        info_fields=info_dict,
                    )
                    variants.append(variant)

        return variants

    @staticmethod
    def _get_format_int(record, sample_idx: int, field: str) -> int:
        try:
            val = record.format(field)
            if val is not None:
                v = val[sample_idx]
                if isinstance(v, np.ndarray):
                    v = v[0]
                return int(v) if v >= 0 else 0
        except Exception:
            pass
        return 0

    @staticmethod
    def _get_format_array(record, sample_idx: int, field: str) -> list[int]:
        try:
            val = record.format(field)
            if val is not None:
                arr = val[sample_idx]
                return [int(x) for x in arr if x >= 0]
        except Exception:
            pass
        return []
