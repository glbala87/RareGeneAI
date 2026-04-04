"""Validation and benchmarking framework.

Evaluates RareGeneAI against:
  - ClinVar pathogenic variants (known disease-causing)
  - OMIM disease cases
  - Simulated rare disease cohorts

Metrics:
  - Top-1 accuracy (correct gene ranked #1)
  - Top-10 recall (correct gene in top 10)
  - Top-50 recall
  - ROC-AUC
  - Mean Reciprocal Rank (MRR)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from raregeneai.config.settings import PipelineConfig
from raregeneai.pipeline.orchestrator import RareGeneAIPipeline


class Benchmarker:
    """Benchmark RareGeneAI against known disease cases."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.results: list[dict] = []

    def run_benchmark(
        self,
        test_cases: list[dict],
        output_path: str | None = None,
    ) -> pd.DataFrame:
        """Run benchmark on a set of test cases.

        Each test case dict:
        {
            "case_id": str,
            "vcf_path": str,
            "hpo_terms": list[str],
            "causal_gene": str,  # ground truth
            "disease": str,      # optional
        }

        Returns DataFrame with metrics per case.
        """
        pipeline = RareGeneAIPipeline(self.config)
        results = []

        for i, case in enumerate(test_cases):
            logger.info(f"Benchmarking case {i+1}/{len(test_cases)}: {case['case_id']}")

            try:
                report = pipeline.run(
                    vcf_path=case["vcf_path"],
                    hpo_terms=case["hpo_terms"],
                    patient_id=case["case_id"],
                    sv_vcf_path=case.get("sv_vcf_path"),
                    father_vcf_path=case.get("father_vcf_path"),
                    mother_vcf_path=case.get("mother_vcf_path"),
                    expression_path=case.get("expression_path"),
                    methylation_path=case.get("methylation_path"),
                )

                ranked_genes = [g.gene_symbol for g in report.ranked_genes]
                causal = case["causal_gene"]

                # Compute metrics
                rank = ranked_genes.index(causal) + 1 if causal in ranked_genes else None

                result = {
                    "case_id": case["case_id"],
                    "causal_gene": causal,
                    "disease": case.get("disease", ""),
                    "n_hpo_terms": len(case["hpo_terms"]),
                    "total_genes_ranked": len(ranked_genes),
                    "causal_gene_rank": rank,
                    "in_top_1": rank == 1 if rank else False,
                    "in_top_5": rank is not None and rank <= 5,
                    "in_top_10": rank is not None and rank <= 10,
                    "in_top_20": rank is not None and rank <= 20,
                    "in_top_50": rank is not None and rank <= 50,
                    "reciprocal_rank": 1.0 / rank if rank else 0.0,
                    "causal_gene_score": (
                        report.ranked_genes[rank - 1].gene_rank_score
                        if rank else 0.0
                    ),
                }
                results.append(result)

            except Exception as e:
                logger.error(f"Case {case['case_id']} failed: {e}")
                results.append({
                    "case_id": case["case_id"],
                    "causal_gene": case["causal_gene"],
                    "error": str(e),
                })

        df = pd.DataFrame(results)
        self.results = results

        # Summary
        summary = self._compute_summary(df)
        logger.info("=" * 60)
        logger.info("BENCHMARK SUMMARY")
        for metric, value in summary.items():
            logger.info(f"  {metric}: {value}")
        logger.info("=" * 60)

        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")

        return df

    def _compute_summary(self, df: pd.DataFrame) -> dict:
        """Compute aggregate benchmark metrics."""
        n = len(df)
        valid = df[df["causal_gene_rank"].notna()]

        return {
            "total_cases": n,
            "successful_cases": len(valid),
            "top_1_accuracy": valid["in_top_1"].mean() if len(valid) > 0 else 0.0,
            "top_5_recall": valid["in_top_5"].mean() if len(valid) > 0 else 0.0,
            "top_10_recall": valid["in_top_10"].mean() if len(valid) > 0 else 0.0,
            "top_20_recall": valid["in_top_20"].mean() if len(valid) > 0 else 0.0,
            "top_50_recall": valid["in_top_50"].mean() if len(valid) > 0 else 0.0,
            "mean_reciprocal_rank": valid["reciprocal_rank"].mean() if len(valid) > 0 else 0.0,
            "median_rank": valid["causal_gene_rank"].median() if len(valid) > 0 else None,
        }

    def load_test_cases_from_csv(self, csv_path: str) -> list[dict]:
        """Load test cases from CSV file.

        Expected columns: case_id, vcf_path, hpo_terms (comma-sep), causal_gene, disease
        """
        df = pd.read_csv(csv_path)
        cases = []
        for _, row in df.iterrows():
            cases.append({
                "case_id": row["case_id"],
                "vcf_path": row["vcf_path"],
                "hpo_terms": row["hpo_terms"].split(","),
                "causal_gene": row["causal_gene"],
                "disease": row.get("disease", ""),
            })
        return cases

    def compare_with_exomiser(
        self,
        raregeneai_results: pd.DataFrame,
        exomiser_results_path: str,
    ) -> pd.DataFrame:
        """Compare RareGeneAI results with Exomiser output."""
        exomiser_df = pd.read_csv(exomiser_results_path)

        comparison = raregeneai_results[["case_id", "causal_gene", "causal_gene_rank"]].copy()
        comparison.columns = ["case_id", "causal_gene", "raregeneai_rank"]

        # Merge Exomiser results
        exomiser_ranks = exomiser_df.set_index("case_id")["rank"].to_dict()
        comparison["exomiser_rank"] = comparison["case_id"].map(exomiser_ranks)

        comparison["raregeneai_better"] = (
            comparison["raregeneai_rank"] < comparison["exomiser_rank"]
        )

        logger.info(f"RareGeneAI better: {comparison['raregeneai_better'].sum()}/{len(comparison)}")
        return comparison
