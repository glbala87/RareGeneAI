"""Unit tests for explainability module."""

import pytest

from raregeneai.explainability.explainer import Explainer
from raregeneai.models.data_models import (
    ACMGClassification,
    AnnotatedVariant,
    FunctionalImpact,
    GeneCandidate,
    Variant,
    Zygosity,
)


@pytest.fixture
def explainer():
    return Explainer()


def _make_annotated_variant(**kwargs):
    defaults = dict(
        chrom="1", pos=100, ref="A", alt="G",
        zygosity=Zygosity.HET,
    )
    v = Variant(**defaults)
    return AnnotatedVariant(variant=v, **kwargs)


class TestExplainer:
    def test_explain_gene_returns_string(self, explainer):
        var = _make_annotated_variant(
            gene_symbol="BRCA1",
            cadd_phred=28.0,
            consequence="missense_variant",
            impact=FunctionalImpact.MODERATE,
        )
        candidate = GeneCandidate(
            gene_symbol="BRCA1",
            variants=[var],
            gene_rank_score=0.85,
            phenotype_score=0.7,
            confidence=0.9,
        )
        explanation = explainer.explain_gene(candidate)
        assert "BRCA1" in explanation
        assert "0.85" in explanation

    def test_explain_variant_cadd(self, explainer):
        var = _make_annotated_variant(cadd_phred=30.0, consequence="stop_gained")
        text = explainer._explain_variant(var)
        assert "CADD=30.0" in text
        assert "deleterious" in text

    def test_explain_variant_novel(self, explainer):
        var = _make_annotated_variant(gnomad_af=None)
        text = explainer._explain_variant(var)
        assert "novel" in text.lower()


class TestACMGClassification:
    def test_pathogenic_classification(self, explainer):
        var = _make_annotated_variant(
            impact=FunctionalImpact.HIGH,
            clinvar_significance="Pathogenic",
            gnomad_af=None,
            cadd_phred=35.0,
        )
        cls = explainer.classify_acmg(var)
        assert cls == ACMGClassification.PATHOGENIC

    def test_benign_high_af(self, explainer):
        var = _make_annotated_variant(
            impact=FunctionalImpact.LOW,
            gnomad_af=0.10,
            clinvar_significance="Benign",
        )
        cls = explainer.classify_acmg(var)
        assert cls == ACMGClassification.BENIGN

    def test_vus_moderate_evidence(self, explainer):
        var = _make_annotated_variant(
            impact=FunctionalImpact.MODERATE,
            gnomad_af=0.005,
            cadd_phred=18.0,
        )
        cls = explainer.classify_acmg(var)
        assert cls == ACMGClassification.VUS
