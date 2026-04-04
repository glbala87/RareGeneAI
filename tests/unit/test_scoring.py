"""Unit tests for variant scoring model."""

import pytest

from raregeneai.config.settings import ScoringConfig
from raregeneai.models.data_models import (
    AnnotatedVariant,
    FunctionalImpact,
    Variant,
    Zygosity,
)
from raregeneai.scoring.variant_scorer import VariantScorer


@pytest.fixture
def scorer():
    return VariantScorer()


def _make_variant(
    cadd=None, revel=None, spliceai=None,
    gnomad_af=None, clinvar="",
    impact=FunctionalImpact.MODERATE,
    consequence="missense_variant",
    zygosity=Zygosity.HET,
):
    v = Variant(chrom="1", pos=100, ref="A", alt="G", zygosity=zygosity)
    return AnnotatedVariant(
        variant=v,
        cadd_phred=cadd,
        revel_score=revel,
        spliceai_max=spliceai,
        gnomad_af=gnomad_af,
        clinvar_significance=clinvar,
        impact=impact,
        consequence=consequence,
    )


class TestVariantScorer:
    def test_high_pathogenicity_high_score(self, scorer):
        var = _make_variant(cadd=35.0, revel=0.9)
        scored = scorer.score_variants([var])
        assert scored[0].pathogenicity_score > 0.8

    def test_clinvar_pathogenic_boost(self, scorer):
        var = _make_variant(cadd=20.0, clinvar="Pathogenic")
        scored = scorer.score_variants([var])
        assert scored[0].pathogenicity_score >= 0.95

    def test_clinvar_benign_caps_score(self, scorer):
        var = _make_variant(cadd=30.0, clinvar="Benign")
        scored = scorer.score_variants([var])
        assert scored[0].pathogenicity_score <= 0.1

    def test_novel_variant_max_rarity(self, scorer):
        var = _make_variant(gnomad_af=None)
        scored = scorer.score_variants([var])
        assert scored[0].rarity_score == 1.0

    def test_common_variant_zero_rarity(self, scorer):
        var = _make_variant(gnomad_af=0.05)
        scored = scorer.score_variants([var])
        assert scored[0].rarity_score == 0.0

    def test_rare_variant_high_rarity(self, scorer):
        var = _make_variant(gnomad_af=0.0001)
        scored = scorer.score_variants([var])
        assert scored[0].rarity_score > 0.5

    def test_lof_high_impact_score(self, scorer):
        var = _make_variant(impact=FunctionalImpact.HIGH, consequence="stop_gained")
        scored = scorer.score_variants([var])
        assert scored[0].functional_score >= 0.9

    def test_synonymous_low_impact(self, scorer):
        var = _make_variant(
            impact=FunctionalImpact.LOW, consequence="synonymous_variant"
        )
        scored = scorer.score_variants([var])
        assert scored[0].functional_score <= 0.1

    def test_composite_score_bounded(self, scorer):
        var = _make_variant(cadd=30.0, gnomad_af=0.0001)
        scored = scorer.score_variants([var])
        assert 0.0 <= scored[0].composite_score <= 1.0

    def test_homozygous_inheritance_bonus(self, scorer):
        var_het = _make_variant(zygosity=Zygosity.HET)
        var_hom = _make_variant(zygosity=Zygosity.HOM_ALT)
        scorer.score_variants([var_het, var_hom])
        # HOM_ALT should get higher inheritance score
        assert var_hom.composite_score >= var_het.composite_score

    def test_filter_removes_common(self, scorer):
        var_rare = _make_variant(gnomad_af=0.0001, cadd=25.0)
        var_common = _make_variant(gnomad_af=0.05, cadd=25.0)
        scorer.score_variants([var_rare, var_common])
        filtered = scorer.filter_variants([var_rare, var_common], require_rare=True)
        assert len(filtered) == 1
        assert filtered[0].variant.variant_key == var_rare.variant.variant_key
