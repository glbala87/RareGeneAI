"""Variant-level composite scoring model.

Computes a weighted composite score from:
  - Pathogenicity (CADD, REVEL, SpliceAI)
  - Rarity (gnomAD AF)
  - Functional impact (consequence severity)
  - Inheritance model compatibility
  - Regulatory impact (non-coding: ENCODE, conservation, DL, gene mapping)

Formula: Score = w1*pathogenicity + w2*rarity + w3*impact + w4*inheritance + w5*regulatory
All sub-scores normalized to [0, 1].
"""

from __future__ import annotations

import math

from loguru import logger

from raregeneai.config.settings import ScoringConfig
from raregeneai.models.data_models import (
    AnnotatedVariant,
    FunctionalImpact,
    InheritanceMode,
    Zygosity,
)


class VariantScorer:
    """Compute composite variant prioritization scores."""

    def __init__(self, config: ScoringConfig | None = None):
        self.config = config or ScoringConfig()

    def score_variants(self, variants: list[AnnotatedVariant]) -> list[AnnotatedVariant]:
        """Score all variants and populate score fields.

        Each variant gets:
          - pathogenicity_score: normalized pathogenicity
          - rarity_score: rarity based on allele frequency
          - functional_score: consequence severity
          - noncoding_impact_score: regulatory impact (non-coding variants)
          - composite_score: weighted combination
        """
        for var in variants:
            var.pathogenicity_score = self._pathogenicity_score(var)
            var.rarity_score = self._rarity_score(var)
            var.functional_score = self._functional_impact_score(var)

            inheritance_score = self._inheritance_score(var)
            regulatory_score = self._regulatory_impact_score(var)

            var.composite_score = (
                self.config.w_pathogenicity * var.pathogenicity_score
                + self.config.w_rarity * var.rarity_score
                + self.config.w_functional_impact * var.functional_score
                + self.config.w_inheritance * inheritance_score
                + self.config.w_regulatory * regulatory_score
            )

        n_scored = sum(1 for v in variants if v.composite_score > 0)
        n_noncoding_scored = sum(
            1 for v in variants if v.is_noncoding and v.composite_score > 0
        )
        logger.info(
            f"Scored {n_scored}/{len(variants)} variants with composite > 0 "
            f"({n_noncoding_scored} non-coding)"
        )
        return variants

    def _pathogenicity_score(self, var: AnnotatedVariant) -> float:
        """Compute normalized pathogenicity from available scores.

        Combines CADD, REVEL, SpliceAI, and regulatory DL scores
        using max-of-normalized approach:
          - CADD PHRED: scaled 0-40 -> 0-1
          - REVEL: already 0-1 (missense only)
          - SpliceAI: already 0-1
          - Enformer/Sei: already 0-1 (non-coding)

        ClinVar pathogenic/likely_pathogenic gets a boost.
        """
        scores = []

        # CADD PHRED score (0-40 scale, normalize to 0-1)
        if var.cadd_phred is not None:
            cadd_norm = min(var.cadd_phred / 40.0, 1.0)
            scores.append(cadd_norm)

        # REVEL score (already 0-1, only for missense)
        if var.revel_score is not None:
            scores.append(var.revel_score)

        # SpliceAI (0-1)
        if var.spliceai_max is not None:
            scores.append(var.spliceai_max)

        # Deep learning regulatory scores (non-coding)
        if var.enformer_score is not None:
            scores.append(var.enformer_score)
        if var.sei_score is not None:
            scores.append(var.sei_score)

        if not scores:
            # No pathogenicity data: use impact as proxy
            return self._impact_to_pathogenicity(var)

        base_score = max(scores)

        # ClinVar boost
        clinvar_lower = var.clinvar_significance.lower()
        if "pathogenic" in clinvar_lower and "likely" not in clinvar_lower:
            base_score = max(base_score, 0.95)
        elif "likely_pathogenic" in clinvar_lower or "likely pathogenic" in clinvar_lower:
            base_score = max(base_score, 0.85)
        elif "benign" in clinvar_lower:
            base_score = min(base_score, 0.1)

        return base_score

    def _impact_to_pathogenicity(self, var: AnnotatedVariant) -> float:
        """Estimate pathogenicity from functional impact when no scores available."""
        if var.impact == FunctionalImpact.HIGH:
            return 0.7
        elif var.impact == FunctionalImpact.MODERATE:
            return 0.4
        elif var.impact == FunctionalImpact.LOW:
            return 0.1
        # For non-coding with regulatory annotation, give moderate baseline
        if var.is_noncoding and var.has_regulatory_annotation:
            return 0.25
        return 0.0

    def _rarity_score(self, var: AnnotatedVariant) -> float:
        """Compute population-adjusted rarity score.

        Uses the effective_af (max of local population AF and gnomAD AF)
        to avoid false positives from founder variants that are globally
        rare but locally common.

        For founder variants:
          - Pathogenic founders: maintain high rarity (disease-relevant)
          - Non-pathogenic founders: penalize rarity (likely benign locally)

        Novel variants (absent in all databases) get maximum rarity.
        """
        effective = var.effective_af

        # Novel in all databases
        if effective == 0.0:
            return 1.0

        # Hard filter: AF > threshold = not rare
        if effective > self.config.gnomad_af_threshold:
            return 0.0

        # Founder variant special handling
        if var.is_founder_variant:
            clinvar_lower = var.clinvar_significance.lower()
            if "pathogenic" in clinvar_lower:
                # Known pathogenic founder: softer decay (keep as candidate)
                return math.exp(-500 * effective)
            else:
                # Non-pathogenic founder: stronger decay (penalize)
                return math.exp(-2000 * effective)

        # Standard exponential decay
        return math.exp(-1000 * effective)

    def _functional_impact_score(self, var: AnnotatedVariant) -> float:
        """Score variant by functional consequence severity.

        Uses the impact_scores mapping from config for fine-grained scoring.
        For non-coding variants, the base consequence score is boosted by
        regulatory context (enhancer/promoter annotation, conservation).
        """
        base_score = 0.0

        # Try specific consequence type first
        for consequence in var.consequence.split(","):
            consequence = consequence.strip()
            if consequence in self.config.impact_scores:
                base_score = max(base_score, self.config.impact_scores[consequence])

        # Fallback to broad impact category if no consequence matched
        if base_score == 0.0:
            impact_map = {
                FunctionalImpact.HIGH: 0.90,
                FunctionalImpact.MODERATE: 0.55,
                FunctionalImpact.LOW: 0.10,
                FunctionalImpact.MODIFIER: 0.02,
            }
            base_score = impact_map.get(var.impact, 0.0)

        # For non-coding variants: boost score based on regulatory context
        if var.is_noncoding and var.has_regulatory_annotation:
            regulatory_boost = 0.0
            # Strong promoter/enhancer context elevates the consequence
            if var.regulatory_class in ("promoter", "enhancer"):
                regulatory_boost = 0.20
            elif var.regulatory_class in ("tf_binding_site", "TFBS"):
                regulatory_boost = 0.15
            # High conservation in regulatory region is additional evidence
            if var.phastcons_score is not None and var.phastcons_score > 0.7:
                regulatory_boost += 0.10

            base_score = min(base_score + regulatory_boost, 0.85)

        return base_score

    def _regulatory_impact_score(self, var: AnnotatedVariant) -> float:
        """Score the non-coding regulatory impact of a variant.

        For non-coding variants: uses the precomputed regulatory_score
        from the RegulatoryAnnotator (combines splicing, regulatory regions,
        conservation, DL scores, and gene mapping confidence).

        For coding variants: returns 0.0 (no regulatory component).

        Returns:
            Regulatory impact score in [0, 1].
        """
        if not var.is_noncoding:
            # Coding variants can still have splicing impact
            if var.spliceai_max is not None and var.spliceai_max > 0.2:
                return var.spliceai_max * 0.5  # Partial regulatory credit for splicing
            return 0.0

        # Use precomputed regulatory score from annotation pipeline
        if var.regulatory_score > 0:
            return var.regulatory_score

        # If no regulatory annotation: estimate from available data
        score = 0.0

        # SpliceAI contribution
        if var.spliceai_max is not None and var.spliceai_max > 0.1:
            score = max(score, var.spliceai_max)

        # Conservation contribution
        if var.phastcons_score is not None and var.phastcons_score > 0.5:
            score = max(score, var.phastcons_score * 0.6)

        # Gene mapping confidence (higher = more likely functional)
        if var.gene_mapping_score > 0.3:
            score = max(score, var.gene_mapping_score * 0.4)

        return score

    def _inheritance_score(self, var: AnnotatedVariant) -> float:
        """Score variant by inheritance model.

        If trio analysis has been run (inheritance_score > 0), uses the
        pre-computed trio-aware score from InheritanceAnalyzer.
        Otherwise falls back to zygosity-based scoring.

        Trio-aware weights (from InheritanceAnalyzer):
          De novo LoF:           1.00
          De novo missense:      0.90
          Compound het LoF+LoF:  0.95
          Compound het LoF+mis:  0.85
          Hom recessive LoF:     0.90
          X-linked hemi LoF:     0.95
          Inherited dominant:    0.40
          Unknown het:           0.30
        """
        # Use trio-computed score if available
        if var.inheritance_score > 0:
            return var.inheritance_score

        # Fallback: zygosity-only
        zygosity = var.variant.zygosity

        if zygosity == Zygosity.HOM_ALT:
            return 0.5  # Could be recessive, but unconfirmed without trio
        if zygosity == Zygosity.HET:
            return 0.3  # Could be anything without trio
        if zygosity == Zygosity.HEMIZYGOUS:
            return 0.8
        return 0.2

    def filter_variants(
        self,
        variants: list[AnnotatedVariant],
        min_score: float = 0.1,
        require_rare: bool = True,
        require_coding: bool = False,
    ) -> list[AnnotatedVariant]:
        """Filter variants by score and quality thresholds."""
        filtered = []
        for var in variants:
            if var.composite_score < min_score:
                continue
            if require_rare and not var.is_rare:
                continue
            if require_coding and var.impact == FunctionalImpact.MODIFIER:
                continue
            filtered.append(var)

        logger.info(f"Filtered to {len(filtered)}/{len(variants)} variants")
        return filtered
