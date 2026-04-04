"""Variant Effect Predictor (VEP) annotation interface.

Supports both local VEP installation and Ensembl REST API.
Handles batch processing for large VCFs.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import requests
from loguru import logger

from raregeneai.config.settings import AnnotationConfig
from raregeneai.models.data_models import (
    AnnotatedVariant,
    FunctionalImpact,
    Variant,
)

IMPACT_MAP = {
    "HIGH": FunctionalImpact.HIGH,
    "MODERATE": FunctionalImpact.MODERATE,
    "LOW": FunctionalImpact.LOW,
    "MODIFIER": FunctionalImpact.MODIFIER,
}

VEP_REST_URL = "https://rest.ensembl.org/vep/human/region"


class VEPAnnotator:
    """Annotate variants with VEP functional consequences."""

    def __init__(self, config: AnnotationConfig | None = None):
        self.config = config or AnnotationConfig()

    def annotate(self, variants: list[Variant]) -> list[AnnotatedVariant]:
        """Annotate variants with VEP.

        Tries local VEP first, falls back to REST API, then to
        VCF INFO field extraction (for pre-annotated VCFs).
        """
        if not variants:
            return []

        logger.info(f"Annotating {len(variants)} variants with VEP")

        if self.config.use_remote_api:
            result = self._annotate_rest_api(variants)
        else:
            result = self._annotate_local(variants)

        # Fallback: fill missing gene symbols from VCF INFO field
        # This handles pre-annotated VCFs and cases where VEP fails
        n_empty = sum(1 for a in result if not a.gene_symbol)
        if n_empty > 0:
            self._fill_from_vcf_info(result, variants)
            n_filled = n_empty - sum(1 for a in result if not a.gene_symbol)
            if n_filled > 0:
                logger.info(f"Filled {n_filled} gene symbols from VCF INFO fields")

        return result

    def _fill_from_vcf_info(
        self, annotated: list[AnnotatedVariant], original_variants: list[Variant],
    ) -> None:
        """Extract gene symbol and consequence from VCF INFO field as fallback.

        Looks for common INFO tags: GENE, SYMBOL, GENEINFO, ANN, CSQ.
        """
        GENE_KEYS = ["GENE", "SYMBOL", "Gene_Name", "GENEINFO", "Gene"]
        CSQ_KEYS = ["CSQ", "Consequence", "ANN", "Effect"]

        CONSEQUENCE_TO_IMPACT = {
            "stop_gained": FunctionalImpact.HIGH,
            "frameshift_variant": FunctionalImpact.HIGH,
            "splice_donor_variant": FunctionalImpact.HIGH,
            "splice_acceptor_variant": FunctionalImpact.HIGH,
            "start_lost": FunctionalImpact.HIGH,
            "transcript_ablation": FunctionalImpact.HIGH,
            "missense_variant": FunctionalImpact.MODERATE,
            "inframe_insertion": FunctionalImpact.MODERATE,
            "inframe_deletion": FunctionalImpact.MODERATE,
            "synonymous_variant": FunctionalImpact.LOW,
            "intron_variant": FunctionalImpact.MODIFIER,
            "intergenic_variant": FunctionalImpact.MODIFIER,
        }

        for ann in annotated:
            if ann.gene_symbol:
                continue  # Already annotated by VEP

            info = ann.variant.info_fields
            if not info:
                continue

            # Extract gene symbol
            for key in GENE_KEYS:
                if key in info:
                    gene = info[key].split(":")[0].split("|")[0].strip()
                    if gene:
                        ann.gene_symbol = gene
                        break

            # Extract consequence
            for key in CSQ_KEYS:
                if key in info:
                    csq = info[key].split("|")[0].split(",")[0].strip()
                    if csq:
                        ann.consequence = csq
                        ann.impact = CONSEQUENCE_TO_IMPACT.get(
                            csq, FunctionalImpact.MODIFIER
                        )
                        break

    def _annotate_rest_api(self, variants: list[Variant]) -> list[AnnotatedVariant]:
        """Annotate using Ensembl REST API (batch mode)."""
        annotated = []
        batch_size = min(self.config.batch_size, 200)  # API limit

        for i in range(0, len(variants), batch_size):
            batch = variants[i : i + batch_size]
            batch_results = self._query_vep_rest(batch)

            for variant, vep_data in zip(batch, batch_results):
                ann = self._parse_vep_result(variant, vep_data)
                annotated.append(ann)

            logger.debug(f"Annotated batch {i // batch_size + 1}")

        return annotated

    def _query_vep_rest(self, variants: list[Variant]) -> list[dict]:
        """Send batch query to VEP REST API."""
        regions = []
        for v in variants:
            # VEP region format: chr:start:end:allele:strand
            if v.is_snv:
                regions.append(f"{v.chrom} {v.pos} {v.pos} {v.ref}/{v.alt} 1")
            elif len(v.alt) > len(v.ref):
                # Insertion
                ins_seq = v.alt[len(v.ref):]
                regions.append(f"{v.chrom} {v.pos + 1} {v.pos} {ins_seq}/- 1")
            else:
                # Deletion
                del_start = v.pos + 1
                del_end = v.pos + len(v.ref) - len(v.alt)
                regions.append(f"{v.chrom} {del_start} {del_end} -/- 1")

        payload = {"regions": regions}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        try:
            resp = requests.post(
                VEP_REST_URL,
                json=payload,
                headers=headers,
                timeout=120,
            )
            resp.raise_for_status()
            results = resp.json()

            # Map results back to variants
            result_map = {}
            for r in results:
                key = f"{r.get('seq_region_name', '')}-{r.get('start', '')}"
                result_map[key] = r

            mapped = []
            for v in variants:
                key = f"{v.chrom}-{v.pos}"
                mapped.append(result_map.get(key, {}))

            return mapped

        except Exception as e:
            logger.error(f"VEP REST API error: {e}")
            return [{}] * len(variants)

    def _annotate_local(self, variants: list[Variant]) -> list[AnnotatedVariant]:
        """Annotate using local VEP installation."""
        # Write variants to temp VCF
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as tmp:
            tmp.write("##fileformat=VCFv4.2\n")
            tmp.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
            for v in variants:
                tmp.write(f"{v.chrom}\t{v.pos}\t.\t{v.ref}\t{v.alt}\t.\t.\t.\n")
            tmp_path = tmp.name

        output_path = tmp_path + ".vep.json"

        cmd = [
            self.config.vep_executable,
            "--input_file", tmp_path,
            "--output_file", output_path,
            "--format", "vcf",
            "--json",
            "--offline",
            "--assembly", self.config.vep_assembly,
            "--dir_cache", self.config.vep_cache_dir,
            "--everything",
            "--no_stats",
            "--force_overwrite",
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"Local VEP failed: {e}")
            return [AnnotatedVariant(variant=v) for v in variants]

        # Parse VEP JSON output
        vep_results = {}
        with open(output_path) as f:
            for line in f:
                data = json.loads(line)
                key = f"{data.get('seq_region_name', '')}-{data.get('start', '')}"
                vep_results[key] = data

        annotated = []
        for v in variants:
            key = f"{v.chrom}-{v.pos}"
            ann = self._parse_vep_result(v, vep_results.get(key, {}))
            annotated.append(ann)

        # Cleanup temp files
        Path(tmp_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)

        return annotated

    def _parse_vep_result(self, variant: Variant, vep_data: dict) -> AnnotatedVariant:
        """Parse VEP result into AnnotatedVariant."""
        if not vep_data:
            return AnnotatedVariant(variant=variant)

        # Get most severe transcript consequence
        transcript_consequences = vep_data.get("transcript_consequences", [])
        if not transcript_consequences:
            return AnnotatedVariant(variant=variant)

        # Pick canonical or most severe
        best = None
        for tc in transcript_consequences:
            if tc.get("canonical", 0) == 1:
                best = tc
                break
        if not best:
            best = transcript_consequences[0]

        consequence_terms = best.get("consequence_terms", [])
        consequence = ",".join(consequence_terms) if consequence_terms else ""
        impact_str = best.get("impact", "MODIFIER")

        return AnnotatedVariant(
            variant=variant,
            gene_symbol=best.get("gene_symbol", ""),
            gene_id=best.get("gene_id", ""),
            transcript_id=best.get("transcript_id", ""),
            hgvs_c=best.get("hgvsc", ""),
            hgvs_p=best.get("hgvsp", ""),
            consequence=consequence,
            impact=IMPACT_MAP.get(impact_str, FunctionalImpact.MODIFIER),
        )
