"""Multi-omics evidence integrator.

Combines genomic (SNV/SV), transcriptomic (RNA-seq), and epigenomic
(methylation) evidence into a unified per-gene confidence score.

Key concepts:
  - Evidence layers: count of distinct data types supporting the gene
  - Concordance: agreement in direction across layers
    (e.g. LoF variant + underexpression + promoter hypermethylation)
  - Multi-omics score: weighted combination with concordance bonus

Genes supported by multiple independent evidence layers receive
a significant confidence boost, reflecting the higher probability
of true causality when orthogonal data converge.
"""

from __future__ import annotations

import math
from collections import defaultdict

from loguru import logger

from raregeneai.config.settings import MultiOmicsConfig
from raregeneai.models.data_models import (
    GeneCandidate,
    GeneExpression,
    FunctionalImpact,
    MethylationRegion,
    MultiOmicsEvidence,
)

from .expression_outlier import ExpressionOutlierDetector
from .methylation_analyzer import MethylationAnalyzer


class MultiOmicsIntegrator:
    """Integrate multi-omics evidence for gene prioritization."""

    def __init__(self, config: MultiOmicsConfig | None = None):
        self.config = config or MultiOmicsConfig()
        self.expression_detector = ExpressionOutlierDetector(self.config)
        self.methylation_analyzer = MethylationAnalyzer(self.config)

    def analyze(
        self,
        candidate_genes: list[str],
        expression_path: str | None = None,
        methylation_path: str | None = None,
        dmr_calls_path: str | None = None,
        reference_expression_path: str | None = None,
    ) -> dict[str, MultiOmicsEvidence]:
        """Run multi-omics analysis for candidate genes.

        Args:
            candidate_genes: Genes to analyze (from variant prioritization).
            expression_path: Patient RNA-seq TPM file.
            methylation_path: Patient methylation data.
            dmr_calls_path: Pre-computed DMR calls.
            reference_expression_path: Reference cohort expression.

        Returns:
            Dict mapping gene_symbol -> MultiOmicsEvidence.
        """
        if not self.config.enabled:
            return {}

        logger.info(f"Running multi-omics integration for {len(candidate_genes)} candidate genes")

        # 1. Expression outlier detection
        expression_results = self.expression_detector.detect_outliers(
            expression_path=expression_path,
            reference_path=reference_expression_path,
        )

        # 2. Methylation DMR analysis
        methylation_results = self.methylation_analyzer.analyze(
            methylation_path=methylation_path,
            dmr_calls_path=dmr_calls_path,
            candidate_genes=candidate_genes,
        )

        # 3. Integrate per gene
        results: dict[str, MultiOmicsEvidence] = {}
        for gene in candidate_genes:
            evidence = self._integrate_gene(
                gene,
                expression_results.get(gene),
                methylation_results.get(gene, []),
            )
            results[gene] = evidence

        # Summary
        n_expr = sum(1 for e in results.values() if e.has_expression_outlier)
        n_meth = sum(1 for e in results.values() if e.has_dmr)
        n_multi = sum(1 for e in results.values() if e.n_evidence_layers >= 2)
        n_concordant = sum(1 for e in results.values() if e.is_concordant)

        logger.info(
            f"Multi-omics results: {n_expr} expression outliers, "
            f"{n_meth} with DMRs, {n_multi} with 2+ evidence layers, "
            f"{n_concordant} concordant"
        )

        return results

    def _integrate_gene(
        self,
        gene_symbol: str,
        expression: GeneExpression | None,
        methylation_regions: list[MethylationRegion],
    ) -> MultiOmicsEvidence:
        """Integrate evidence for a single gene.

        Computes:
          - Per-layer scores (expression, methylation)
          - Evidence layer count
          - Concordance assessment
          - Combined multi-omics confidence score
        """
        evidence = MultiOmicsEvidence(gene_symbol=gene_symbol)
        layers: list[str] = []

        # ── Expression score ──────────────────────────────────────────────
        if expression and expression.is_outlier:
            evidence.expression = expression
            evidence.has_expression_outlier = True
            layers.append("expression")

            # Score based on Z-score magnitude
            z = abs(expression.z_score)
            if z >= self.config.z_score_strong_threshold:
                evidence.expression_score = min(0.6 + (z - 3) * 0.1, 1.0)
            elif z >= self.config.z_score_threshold:
                evidence.expression_score = 0.3 + (z - 2) * 0.3
            else:
                evidence.expression_score = 0.0
        elif expression:
            evidence.expression = expression

        # ── Methylation score ─────────────────────────────────────────────
        sig_dmrs = [m for m in methylation_regions if m.is_dmr]
        if sig_dmrs:
            evidence.methylation_regions = methylation_regions
            evidence.has_dmr = True
            evidence.has_promoter_dmr = any(m.region_type == "promoter" for m in sig_dmrs)
            layers.append("methylation")

            # Score based on delta_beta magnitude and promoter involvement
            max_delta = max(abs(m.delta_beta) for m in sig_dmrs)
            promoter_bonus = 0.2 if evidence.has_promoter_dmr else 0.0

            if max_delta >= self.config.delta_beta_strong_threshold:
                evidence.methylation_score = min(0.5 + promoter_bonus + (max_delta - 0.3) * 0.5, 1.0)
            elif max_delta >= self.config.delta_beta_threshold:
                evidence.methylation_score = 0.3 + promoter_bonus
            else:
                evidence.methylation_score = 0.0
        else:
            evidence.methylation_regions = methylation_regions

        # ── Concordance assessment ────────────────────────────────────────
        concordance = self._assess_concordance(evidence)
        evidence.is_concordant = concordance["is_concordant"]
        evidence.concordance_description = concordance["description"]
        evidence.concordance_bonus = concordance["bonus"]

        # Genomic layer is implicitly present (these are candidate genes)
        layers.insert(0, "genomic")

        evidence.evidence_layers = layers
        evidence.n_evidence_layers = len(layers)

        # ── Combined multi-omics score ────────────────────────────────────
        evidence.multi_omics_score = self._compute_multi_omics_score(evidence)

        return evidence

    def _assess_concordance(self, evidence: MultiOmicsEvidence) -> dict:
        """Assess whether evidence layers agree in direction.

        Concordant patterns (higher confidence):
          - LoF variant + underexpression
          - LoF variant + underexpression + promoter hypermethylation
          - Gain-of-function + overexpression
          - Promoter hypermethylation + underexpression

        Discordant patterns (lower confidence but still informative):
          - LoF variant + overexpression (possible compensatory)
          - Hypomethylation + underexpression

        Returns dict with is_concordant, description, bonus.
        """
        expr = evidence.expression
        has_under = expr is not None and expr.is_underexpressed
        has_over = expr is not None and expr.is_overexpressed

        promoter_dmrs = [m for m in evidence.methylation_regions
                        if m.is_dmr and m.region_type == "promoter"]
        has_hyper = any(m.is_hypermethylated for m in promoter_dmrs)
        has_hypo = any(m.is_hypomethylated for m in promoter_dmrs)

        # Check concordance patterns
        if has_under and has_hyper:
            return {
                "is_concordant": True,
                "description": "Underexpression concordant with promoter hypermethylation (silencing)",
                "bonus": 0.3,
            }

        if has_under and evidence.has_dmr:
            return {
                "is_concordant": True,
                "description": "Underexpression supported by aberrant methylation",
                "bonus": 0.2,
            }

        if has_over and has_hypo:
            return {
                "is_concordant": True,
                "description": "Overexpression concordant with promoter hypomethylation (activation)",
                "bonus": 0.25,
            }

        if has_under:
            return {
                "is_concordant": False,
                "description": "Underexpression without methylation support",
                "bonus": 0.0,
            }

        if has_hyper:
            return {
                "is_concordant": False,
                "description": "Promoter hypermethylation without expression data",
                "bonus": 0.05,
            }

        return {"is_concordant": False, "description": "", "bonus": 0.0}

    def _compute_multi_omics_score(self, evidence: MultiOmicsEvidence) -> float:
        """Compute the unified multi-omics confidence score.

        Formula:
          score = w_expr * expression_score
                + w_meth * methylation_score
                + w_conc * concordance_bonus
                + w_layers * layer_bonus

        Where layer_bonus scales with number of supporting evidence types:
          1 layer (genomic only) = 0.0
          2 layers = 0.5
          3 layers = 0.8
          4+ layers = 1.0
        """
        cfg = self.config

        # Layer count bonus (rewards multi-layer convergence)
        n = evidence.n_evidence_layers
        if n <= 1:
            layer_bonus = 0.0
        elif n == 2:
            layer_bonus = 0.5
        elif n == 3:
            layer_bonus = 0.8
        else:
            layer_bonus = 1.0

        raw_score = (
            cfg.w_expression * evidence.expression_score
            + cfg.w_methylation * evidence.methylation_score
            + cfg.w_concordance * evidence.concordance_bonus
            + cfg.w_layer_count * layer_bonus
        )

        # Apply concordance multiplier
        if evidence.is_concordant:
            raw_score *= cfg.concordance_multiplier

        return min(raw_score, 1.0)

    def enrich_candidates(
        self,
        candidates: list[GeneCandidate],
        multi_omics: dict[str, MultiOmicsEvidence],
    ) -> list[GeneCandidate]:
        """Enrich GeneCandidate objects with multi-omics evidence.

        Updates evidence_summary with multi-omics features for the
        ranking model, and attaches the full MultiOmicsEvidence object
        for downstream explanation.
        """
        for candidate in candidates:
            evidence = multi_omics.get(candidate.gene_symbol)
            if not evidence:
                continue

            candidate.multi_omics = evidence
            candidate.multi_omics_score = evidence.multi_omics_score
            candidate.n_evidence_layers = evidence.n_evidence_layers

            # Update evidence_summary with multi-omics features
            candidate.evidence_summary.update({
                "multi_omics_score": evidence.multi_omics_score,
                "n_evidence_layers": evidence.n_evidence_layers,
                "has_expression_outlier": evidence.has_expression_outlier,
                "expression_score": evidence.expression_score,
                "expression_z_score": (
                    evidence.expression.z_score if evidence.expression else 0.0
                ),
                "has_dmr": evidence.has_dmr,
                "has_promoter_dmr": evidence.has_promoter_dmr,
                "methylation_score": evidence.methylation_score,
                "is_concordant": evidence.is_concordant,
                "concordance_bonus": evidence.concordance_bonus,
            })

        n_enriched = sum(1 for c in candidates if c.multi_omics_score > 0)
        logger.info(f"Enriched {n_enriched} candidates with multi-omics evidence")
        return candidates
