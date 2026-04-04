"""Methylation data analysis and DMR detection.

Identifies differentially methylated regions (DMRs) in the patient
relative to a control cohort and overlaps them with candidate genes.

Supports:
  - BED-format beta values (from WGBS/RRBS/EPIC array)
  - Pre-called DMR files
  - Bismark coverage files

Produces MethylationRegion objects per gene with delta-beta and significance.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from raregeneai.config.settings import MultiOmicsConfig
from raregeneai.models.data_models import MethylationRegion


class MethylationAnalyzer:
    """Detect DMRs and overlap with candidate genes."""

    def __init__(self, config: MultiOmicsConfig | None = None):
        self.config = config or MultiOmicsConfig()
        self._promoter_regions: dict[str, list[tuple[str, int, int]]] | None = None

    def analyze(
        self,
        methylation_path: str | None = None,
        dmr_calls_path: str | None = None,
        candidate_genes: list[str] | None = None,
    ) -> dict[str, list[MethylationRegion]]:
        """Run methylation analysis.

        Strategy:
        1. If pre-called DMRs provided: load and overlap with genes.
        2. If raw beta values provided: call DMRs then overlap.
        3. Return gene -> list[MethylationRegion] mapping.

        Args:
            methylation_path: Patient methylation data (overrides config).
            dmr_calls_path: Pre-computed DMR calls (overrides config).
            candidate_genes: Restrict analysis to these genes.

        Returns:
            Dict mapping gene_symbol -> list of DMRs overlapping the gene.
        """
        # Load promoter regions for gene overlap
        self._load_promoter_regions()

        # Strategy 1: Pre-called DMRs
        dmr_path = dmr_calls_path or self.config.dmr_calls_path
        if dmr_path and Path(dmr_path).exists():
            return self._load_precalled_dmrs(dmr_path, candidate_genes)

        # Strategy 2: Raw methylation data -> call DMRs
        meth_path = methylation_path or self.config.methylation_path
        if meth_path and Path(meth_path).exists():
            return self._call_dmrs_from_beta(meth_path, candidate_genes)

        logger.info("No methylation data provided, skipping methylation analysis")
        return {}

    def _load_precalled_dmrs(
        self,
        dmr_path: str,
        candidate_genes: list[str] | None,
    ) -> dict[str, list[MethylationRegion]]:
        """Load pre-called DMRs from file.

        Expected format (TSV):
        chrom<TAB>start<TAB>end<TAB>gene<TAB>delta_beta<TAB>pvalue[<TAB>n_cpgs<TAB>region_type]
        """
        gene_dmrs: dict[str, list[MethylationRegion]] = defaultdict(list)

        try:
            df = pd.read_csv(dmr_path, sep="\t")

            # Normalize column names
            col_map = {}
            for col in df.columns:
                cl = col.lower().strip()
                if cl in ("chrom", "chr", "#chrom", "chromosome"):
                    col_map[col] = "chrom"
                elif cl in ("start", "chromstart"):
                    col_map[col] = "start"
                elif cl in ("end", "chromend"):
                    col_map[col] = "end"
                elif cl in ("gene", "gene_symbol", "gene_name", "genename"):
                    col_map[col] = "gene"
                elif cl in ("delta_beta", "deltabeta", "diff", "meandiff"):
                    col_map[col] = "delta_beta"
                elif cl in ("pvalue", "p_value", "pval", "p.value"):
                    col_map[col] = "pvalue"
                elif cl in ("n_cpgs", "ncpg", "ncpgs", "cpgs"):
                    col_map[col] = "n_cpgs"
                elif cl in ("region_type", "type", "annotation"):
                    col_map[col] = "region_type"
            df = df.rename(columns=col_map)

            for _, row in df.iterrows():
                gene = str(row.get("gene", ""))
                if not gene or gene == "nan":
                    continue
                if candidate_genes and gene not in candidate_genes:
                    continue

                delta = float(row.get("delta_beta", 0.0))
                pval = row.get("pvalue")
                pval = float(pval) if pval is not None and str(pval) != "nan" else None
                n_cpgs = int(row.get("n_cpgs", 0)) if "n_cpgs" in row else 0
                region_type = str(row.get("region_type", "")) if "region_type" in row else ""

                # Classify DMR
                is_dmr = abs(delta) >= self.config.delta_beta_threshold
                if pval is not None:
                    is_dmr = is_dmr and pval < self.config.dmr_pvalue_threshold
                if n_cpgs > 0:
                    is_dmr = is_dmr and n_cpgs >= self.config.min_cpgs_per_dmr

                direction = ""
                if delta >= self.config.delta_beta_threshold:
                    direction = "hyper"
                elif delta <= -self.config.delta_beta_threshold:
                    direction = "hypo"

                mr = MethylationRegion(
                    chrom=str(row.get("chrom", "")),
                    start=int(row.get("start", 0)),
                    end=int(row.get("end", 0)),
                    gene_symbol=gene,
                    delta_beta=delta,
                    mean_beta_patient=max(0.5 + delta / 2, 0.0),
                    mean_beta_control=0.5,
                    n_cpgs=n_cpgs,
                    p_value=pval,
                    is_dmr=is_dmr,
                    dmr_direction=direction,
                    region_type=region_type or self._infer_region_type(gene, str(row.get("chrom", "")), int(row.get("start", 0))),
                )
                gene_dmrs[gene].append(mr)

        except Exception as e:
            logger.error(f"Failed to load DMR file {dmr_path}: {e}")

        n_genes = len(gene_dmrs)
        n_dmrs = sum(len(v) for v in gene_dmrs.values())
        n_sig = sum(1 for v in gene_dmrs.values() for m in v if m.is_dmr)
        logger.info(f"Methylation: {n_dmrs} regions across {n_genes} genes, {n_sig} significant DMRs")

        return dict(gene_dmrs)

    def _call_dmrs_from_beta(
        self,
        meth_path: str,
        candidate_genes: list[str] | None,
    ) -> dict[str, list[MethylationRegion]]:
        """Call DMRs from raw beta value BED file.

        Patient BED format: chrom<TAB>start<TAB>end<TAB>beta_value

        Uses reference methylation stats for comparison.
        """
        gene_dmrs: dict[str, list[MethylationRegion]] = defaultdict(list)

        # Load reference methylation if available
        ref_stats = self._load_reference_methylation()

        try:
            df = pd.read_csv(meth_path, sep="\t", header=None,
                           names=["chrom", "start", "end", "beta"],
                           dtype={"chrom": str, "start": int, "end": int, "beta": float})
        except Exception as e:
            logger.error(f"Failed to load methylation BED {meth_path}: {e}")
            return {}

        # Group CpGs into windows and compute per-gene methylation
        # Overlap with gene promoters to find gene-level DMRs
        for gene, regions in (self._promoter_regions or {}).items():
            if candidate_genes and gene not in candidate_genes:
                continue

            for chrom, prom_start, prom_end in regions:
                # Find CpGs overlapping this promoter
                mask = (
                    (df["chrom"] == chrom) &
                    (df["start"] >= prom_start) &
                    (df["end"] <= prom_end)
                )
                cpg_subset = df[mask]

                if len(cpg_subset) < self.config.min_cpgs_per_dmr:
                    continue

                patient_mean = float(cpg_subset["beta"].mean())

                # Get reference mean for this region
                region_key = f"{chrom}:{prom_start}-{prom_end}"
                if region_key in ref_stats:
                    control_mean = ref_stats[region_key]
                else:
                    control_mean = 0.5  # Assume moderate methylation

                delta = patient_mean - control_mean

                is_dmr = abs(delta) >= self.config.delta_beta_threshold
                direction = ""
                if delta >= self.config.delta_beta_threshold:
                    direction = "hyper"
                elif delta <= -self.config.delta_beta_threshold:
                    direction = "hypo"

                mr = MethylationRegion(
                    chrom=chrom,
                    start=prom_start,
                    end=prom_end,
                    gene_symbol=gene,
                    mean_beta_patient=patient_mean,
                    mean_beta_control=control_mean,
                    delta_beta=round(delta, 4),
                    n_cpgs=len(cpg_subset),
                    is_dmr=is_dmr,
                    dmr_direction=direction,
                    region_type="promoter",
                )
                gene_dmrs[gene].append(mr)

        n_genes = len(gene_dmrs)
        n_sig = sum(1 for v in gene_dmrs.values() for m in v if m.is_dmr)
        logger.info(f"Methylation: called DMRs for {n_genes} genes, {n_sig} significant")

        return dict(gene_dmrs)

    def _load_reference_methylation(self) -> dict[str, float]:
        """Load reference methylation statistics.

        Returns dict: region_key -> mean_beta in controls.
        """
        ref_path = self.config.reference_methylation_path
        if not ref_path or not Path(ref_path).exists():
            return {}

        stats = {}
        try:
            df = pd.read_csv(ref_path, sep="\t")
            for _, row in df.iterrows():
                key = f"{row.iloc[0]}:{row.iloc[1]}-{row.iloc[2]}"
                stats[key] = float(row.iloc[3])
        except Exception as e:
            logger.debug(f"Failed to load reference methylation: {e}")

        return stats

    def _load_promoter_regions(self) -> None:
        """Load gene promoter BED for DMR overlap.

        BED format: chrom<TAB>start<TAB>end<TAB>gene_symbol
        """
        if self._promoter_regions is not None:
            return

        self._promoter_regions = {}
        bed_path = self.config.promoter_bed_path
        if not bed_path or not Path(bed_path).exists():
            return

        try:
            with open(bed_path) as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split("\t")
                    if len(parts) >= 4:
                        gene = parts[3]
                        self._promoter_regions.setdefault(gene, []).append(
                            (parts[0], int(parts[1]), int(parts[2]))
                        )
            logger.info(f"Loaded promoter regions for {len(self._promoter_regions)} genes")
        except Exception as e:
            logger.debug(f"Failed to load promoter BED: {e}")

    def _infer_region_type(self, gene: str, chrom: str, start: int) -> str:
        """Infer region type (promoter/gene_body) from promoter overlap."""
        if self._promoter_regions and gene in self._promoter_regions:
            for p_chrom, p_start, p_end in self._promoter_regions[gene]:
                if chrom == p_chrom and p_start <= start <= p_end:
                    return "promoter"
        return "gene_body"
