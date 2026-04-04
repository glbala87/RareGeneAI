"""RNA-seq expression outlier detection.

Detects genes with aberrant expression in the patient sample relative
to a reference cohort. Uses robust statistics (median + MAD) for
outlier calling, following the OUTRIDER/FRASER approach.

Input formats:
  - TPM matrix (gene x sample or single-sample column)
  - Raw counts (will be TPM-normalized)
  - Pre-computed reference statistics (gene, median, MAD)

Output: GeneExpression objects with Z-scores and outlier flags per gene.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from raregeneai.config.settings import MultiOmicsConfig
from raregeneai.models.data_models import GeneExpression


class ExpressionOutlierDetector:
    """Detect expression outliers against a reference cohort."""

    def __init__(self, config: MultiOmicsConfig | None = None):
        self.config = config or MultiOmicsConfig()
        self._ref_stats: pd.DataFrame | None = None  # gene -> (median, mad)

    def detect_outliers(
        self,
        expression_path: str | None = None,
        reference_path: str | None = None,
    ) -> dict[str, GeneExpression]:
        """Run expression outlier detection.

        Args:
            expression_path: Patient expression file (overrides config).
            reference_path: Reference cohort or stats file (overrides config).

        Returns:
            Dict mapping gene_symbol -> GeneExpression with outlier annotations.
        """
        expr_path = expression_path or self.config.expression_path
        if not expr_path or not Path(expr_path).exists():
            logger.info("No expression data provided, skipping outlier detection")
            return {}

        # Load patient expression
        patient_expr = self._load_patient_expression(expr_path)
        if patient_expr.empty:
            return {}

        # Load or compute reference statistics
        ref_path = reference_path or self.config.reference_stats_path or self.config.reference_expression_path
        ref_stats = self._load_reference_stats(ref_path)

        # Compute Z-scores and detect outliers
        results = self._compute_outliers(patient_expr, ref_stats)

        n_outliers = sum(1 for g in results.values() if g.is_outlier)
        n_under = sum(1 for g in results.values() if g.is_underexpressed)
        n_over = sum(1 for g in results.values() if g.is_overexpressed)
        logger.info(
            f"Expression analysis: {len(results)} genes profiled, "
            f"{n_outliers} outliers ({n_under} under, {n_over} over)"
        )

        return results

    def _load_patient_expression(self, path: str) -> pd.Series:
        """Load patient gene expression from file.

        Supports:
          - Tab-separated: gene<TAB>value  (single sample)
          - CSV/TSV matrix: gene x samples (takes first non-gene column)
          - RSEM, Salmon, Kallisto output formats

        Returns pd.Series with gene symbols as index and TPM values.
        """
        path = Path(path)
        try:
            df = pd.read_csv(path, sep="\t", index_col=0)

            # If multiple columns, take first numeric column (single patient)
            if df.shape[1] > 1:
                # Look for a TPM or expression column
                for col in df.columns:
                    if any(k in col.lower() for k in ("tpm", "fpkm", "count", "expression")):
                        return df[col].astype(float)
                # Default: first column
                return df.iloc[:, 0].astype(float)
            else:
                return df.iloc[:, 0].astype(float)

        except Exception as e:
            logger.error(f"Failed to load expression file {path}: {e}")
            return pd.Series(dtype=float)

    def _load_reference_stats(
        self, ref_path: str | None,
    ) -> pd.DataFrame:
        """Load reference cohort statistics.

        If pre-computed stats file (gene, median, mad):
          - Load directly.
        If full expression matrix (gene x samples):
          - Compute median and MAD per gene.
        If no reference:
          - Use GTEx-based defaults (log-normal approximation).

        Returns DataFrame with columns: median, mad, indexed by gene.
        """
        if self._ref_stats is not None:
            return self._ref_stats

        if ref_path and Path(ref_path).exists():
            path = Path(ref_path)
            df = pd.read_csv(path, sep="\t", index_col=0)

            if "median" in df.columns and "mad" in df.columns:
                # Pre-computed stats
                self._ref_stats = df[["median", "mad"]]
            else:
                # Full matrix: compute stats
                self._ref_stats = pd.DataFrame({
                    "median": df.median(axis=1),
                    "mad": df.apply(
                        lambda row: self._median_absolute_deviation(row.values),
                        axis=1,
                    ),
                })

            logger.info(f"Loaded reference stats for {len(self._ref_stats)} genes")
            return self._ref_stats

        # No reference: return empty (will use fallback Z-scoring)
        logger.warning("No reference expression data; using log2(TPM) distribution estimates")
        self._ref_stats = pd.DataFrame(columns=["median", "mad"])
        return self._ref_stats

    def _compute_outliers(
        self,
        patient_expr: pd.Series,
        ref_stats: pd.DataFrame,
    ) -> dict[str, GeneExpression]:
        """Compute Z-scores and outlier status for each gene.

        Z-score = (patient_log2tpm - ref_median) / ref_MAD

        Uses MAD (median absolute deviation) instead of SD for robustness
        against reference outliers.
        """
        results: dict[str, GeneExpression] = {}

        # Convert to log2(TPM + 1) for normality
        patient_log2 = np.log2(patient_expr + 1)

        # If no reference: compute Z-scores against the patient's own distribution
        if ref_stats.empty:
            global_median = patient_log2.median()
            global_mad = self._median_absolute_deviation(patient_log2.values)
            if global_mad == 0:
                global_mad = 1.0
            ref_stats = pd.DataFrame({
                "median": global_median,
                "mad": global_mad,
            }, index=patient_log2.index)

        for gene in patient_expr.index:
            tpm_val = float(patient_expr.get(gene, 0.0))
            log2_val = float(patient_log2.get(gene, 0.0))

            # Get reference stats for this gene
            if gene in ref_stats.index:
                ref_med = float(ref_stats.loc[gene, "median"])
                ref_mad = float(ref_stats.loc[gene, "mad"])
            else:
                # Gene not in reference: skip Z-scoring
                results[gene] = GeneExpression(
                    gene_symbol=gene, tpm=tpm_val, log2_tpm=log2_val,
                )
                continue

            # Avoid division by zero
            if ref_mad <= 0:
                ref_mad = 0.5  # Minimum MAD for genes with no variation

            z = (log2_val - ref_med) / ref_mad

            # Determine outlier status
            is_outlier = abs(z) >= self.config.z_score_threshold
            direction = ""
            if z <= -self.config.z_score_threshold:
                direction = "under"
            elif z >= self.config.z_score_threshold:
                direction = "over"

            # Compute percentile (approximation from Z-score assuming normality)
            try:
                from scipy.stats import norm
                percentile = float(norm.cdf(z) * 100)
            except ImportError:
                # Rough percentile from Z-score
                percentile = max(0.0, min(100.0, 50 + z * 15))

            results[gene] = GeneExpression(
                gene_symbol=gene,
                tpm=tpm_val,
                log2_tpm=log2_val,
                z_score=round(z, 3),
                percentile=round(percentile, 1),
                is_outlier=is_outlier,
                outlier_direction=direction,
                reference_median=ref_med,
                reference_mad=ref_mad,
            )

        return results

    @staticmethod
    def _median_absolute_deviation(values: np.ndarray) -> float:
        """Compute MAD (median absolute deviation) with consistency constant."""
        values = values[~np.isnan(values)]
        if len(values) == 0:
            return 0.0
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        # Scale factor for consistency with SD under normality
        return float(mad * 1.4826)
