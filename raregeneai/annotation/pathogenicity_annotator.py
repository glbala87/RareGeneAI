"""Pathogenicity score annotation: CADD, REVEL, SpliceAI, ClinVar.

Supports both local database files and remote API queries.
"""

from __future__ import annotations

from pathlib import Path

import requests
from loguru import logger

from raregeneai.config.settings import AnnotationConfig
from raregeneai.models.data_models import AnnotatedVariant


class PathogenicityAnnotator:
    """Annotate variants with pathogenicity prediction scores."""

    def __init__(self, config: AnnotationConfig | None = None):
        self.config = config or AnnotationConfig()

    def annotate(self, variants: list[AnnotatedVariant]) -> list[AnnotatedVariant]:
        """Add CADD, REVEL, SpliceAI, and ClinVar annotations."""
        logger.info(f"Annotating {len(variants)} variants with pathogenicity scores")

        for var in variants:
            self._annotate_cadd(var)
            self._annotate_revel(var)
            self._annotate_spliceai(var)
            self._annotate_clinvar(var)

        return variants

    # ── CADD ───────────────────────────────────────────────────────────────
    def _annotate_cadd(self, var: AnnotatedVariant) -> None:
        """Query CADD score (local TSV or remote API)."""
        if self.config.cadd_snv_path and Path(self.config.cadd_snv_path).exists():
            self._cadd_local(var)
        elif self.config.use_remote_api:
            self._cadd_remote(var)

    def _cadd_local(self, var: AnnotatedVariant) -> None:
        """Query local CADD tabix file."""
        try:
            import pysam

            path = (
                self.config.cadd_snv_path
                if var.variant.is_snv
                else self.config.cadd_indel_path
            )
            if not path or not Path(path).exists():
                return

            tbx = pysam.TabixFile(path)
            for row in tbx.fetch(
                var.variant.chrom,
                var.variant.pos - 1,
                var.variant.pos,
            ):
                fields = row.split("\t")
                if len(fields) >= 6 and fields[2] == var.variant.ref and fields[3] == var.variant.alt:
                    var.cadd_raw = float(fields[4])
                    var.cadd_phred = float(fields[5])
                    break
        except Exception as e:
            logger.debug(f"CADD local query failed: {e}")

    def _cadd_remote(self, var: AnnotatedVariant) -> None:
        """Query CADD API."""
        try:
            url = (
                f"https://cadd.gs.washington.edu/api/v1.0/"
                f"{self.config.vep_assembly}/{var.variant.chrom}:{var.variant.pos}"
                f":{var.variant.ref}:{var.variant.alt}"
            )
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    entry = data[0] if isinstance(data, list) else data
                    var.cadd_phred = entry.get("PHRED")
                    var.cadd_raw = entry.get("RawScore")
        except Exception as e:
            logger.debug(f"CADD API query failed: {e}")

    # ── REVEL ──────────────────────────────────────────────────────────────
    def _annotate_revel(self, var: AnnotatedVariant) -> None:
        """Query REVEL score for missense variants."""
        if "missense" not in var.consequence.lower():
            return

        if self.config.revel_path and Path(self.config.revel_path).exists():
            self._revel_local(var)

    def _revel_local(self, var: AnnotatedVariant) -> None:
        """Query local REVEL tabix file."""
        try:
            import pysam

            tbx = pysam.TabixFile(self.config.revel_path)
            for row in tbx.fetch(
                var.variant.chrom, var.variant.pos - 1, var.variant.pos
            ):
                fields = row.split(",")
                if len(fields) >= 7:
                    var.revel_score = float(fields[6])
                    break
        except Exception as e:
            logger.debug(f"REVEL local query failed: {e}")

    # ── SpliceAI ───────────────────────────────────────────────────────────
    def _annotate_spliceai(self, var: AnnotatedVariant) -> None:
        """Query SpliceAI scores."""
        if self.config.spliceai_path and Path(self.config.spliceai_path).exists():
            self._spliceai_local(var)

    def _spliceai_local(self, var: AnnotatedVariant) -> None:
        """Query local SpliceAI VCF."""
        try:
            import pysam

            tbx = pysam.TabixFile(self.config.spliceai_path)
            for row in tbx.fetch(
                var.variant.chrom, var.variant.pos - 1, var.variant.pos
            ):
                fields = row.split("\t")
                if len(fields) >= 8:
                    info = fields[7]
                    scores = self._parse_spliceai_info(info)
                    if scores:
                        var.spliceai_max = max(scores)
                        break
        except Exception as e:
            logger.debug(f"SpliceAI local query failed: {e}")

    @staticmethod
    def _parse_spliceai_info(info: str) -> list[float]:
        """Extract SpliceAI delta scores from INFO field."""
        for field in info.split(";"):
            if field.startswith("SpliceAI="):
                parts = field.split("|")
                if len(parts) >= 8:
                    try:
                        return [float(parts[i]) for i in range(2, 6)]
                    except ValueError:
                        pass
        return []

    # ── ClinVar ────────────────────────────────────────────────────────────
    def _annotate_clinvar(self, var: AnnotatedVariant) -> None:
        """Query ClinVar clinical significance."""
        if self.config.clinvar_path and Path(self.config.clinvar_path).exists():
            self._clinvar_local(var)
        elif self.config.use_remote_api:
            self._clinvar_remote(var)

    def _clinvar_local(self, var: AnnotatedVariant) -> None:
        """Query local ClinVar VCF."""
        try:
            import pysam

            tbx = pysam.TabixFile(self.config.clinvar_path)
            for row in tbx.fetch(
                var.variant.chrom, var.variant.pos - 1, var.variant.pos
            ):
                fields = row.split("\t")
                if len(fields) >= 8 and fields[3] == var.variant.ref and fields[4] == var.variant.alt:
                    info = fields[7]
                    var.clinvar_significance = self._parse_info_field(info, "CLNSIG") or ""
                    var.clinvar_review_status = self._parse_info_field(info, "CLNREVSTAT") or ""
                    var.clinvar_id = fields[2]
                    break
        except Exception as e:
            logger.debug(f"ClinVar local query failed: {e}")

    def _clinvar_remote(self, var: AnnotatedVariant) -> None:
        """Query ClinVar via NCBI E-utilities."""
        try:
            url = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                f"?db=clinvar&term={var.variant.chrom}[chr]+AND+"
                f"{var.variant.pos}[chrpos37]&retmode=json"
            )
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                ids = data.get("esearchresult", {}).get("idlist", [])
                if ids:
                    var.clinvar_id = ids[0]
                    self._fetch_clinvar_details(var, ids[0])
        except Exception as e:
            logger.debug(f"ClinVar API query failed: {e}")

    def _fetch_clinvar_details(self, var: AnnotatedVariant, clinvar_id: str) -> None:
        """Fetch ClinVar details from NCBI."""
        try:
            url = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                f"?db=clinvar&id={clinvar_id}&retmode=json"
            )
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                result = data.get("result", {}).get(clinvar_id, {})
                var.clinvar_significance = result.get(
                    "clinical_significance", {}).get("description", "")
                var.clinvar_review_status = result.get(
                    "clinical_significance", {}).get("review_status", "")
        except Exception:
            pass

    @staticmethod
    def _parse_info_field(info: str, key: str) -> str | None:
        for field in info.split(";"):
            if field.startswith(f"{key}="):
                return field.split("=", 1)[1]
        return None
