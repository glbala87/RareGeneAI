"""Unit tests for clinical decision support layer.

Tests:
  - ACMG/AMP variant classification (all criteria + combining rules)
  - ACMG SF v3.2 actionable gene flagging
  - Pharmacogenomic drug-gene interactions
  - Clinical recommendation generation
  - Audit trail completeness
  - GeneCandidate enrichment
"""

import pytest

from raregeneai.clinical.acmg_classifier import ACMGClassifier, ACMGResult
from raregeneai.clinical.clinical_decision import (
    ACMG_SF_GENES,
    PHARMACOGENOMIC_GENES,
    ClinicalDecisionEngine,
    ClinicalInsight,
)
from raregeneai.models.data_models import (
    ACMGClassification,
    AnnotatedVariant,
    FunctionalImpact,
    GeneCandidate,
    Variant,
    Zygosity,
)


def _v(
    consequence="missense_variant",
    impact=FunctionalImpact.MODERATE,
    cadd=None, revel=None, spliceai=None,
    gnomad_af=None, clinvar="",
    gene="GENE_A",
    is_de_novo=False, is_compound_het=False, is_hom_recessive=False,
    compound_het_partner="",
    **kwargs,
):
    return AnnotatedVariant(
        variant=Variant(chrom="chr1", pos=100, ref="A", alt="G",
                       zygosity=Zygosity.HET, depth=30, gq=40),
        gene_symbol=gene,
        consequence=consequence,
        impact=impact,
        cadd_phred=cadd,
        revel_score=revel,
        spliceai_max=spliceai,
        gnomad_af=gnomad_af,
        clinvar_significance=clinvar,
        is_de_novo=is_de_novo,
        is_compound_het=is_compound_het,
        compound_het_partner_key=compound_het_partner,
        is_hom_recessive=is_hom_recessive,
        **kwargs,
    )


# ── ACMG Classifier Tests ────────────────────────────────────────────────────

class TestACMGClassifier:
    @pytest.fixture
    def classifier(self):
        return ACMGClassifier()

    # ── Pathogenic criteria ───────────────────────────────────────────

    def test_pvs1_lof_variant(self, classifier):
        var = _v(consequence="stop_gained", impact=FunctionalImpact.HIGH)
        result = classifier.classify(var)
        codes = result.pathogenic_criteria
        assert "PVS1" in codes

    def test_pvs1_frameshift(self, classifier):
        var = _v(consequence="frameshift_variant", impact=FunctionalImpact.HIGH)
        result = classifier.classify(var)
        assert "PVS1" in result.pathogenic_criteria

    def test_pvs1_not_missense(self, classifier):
        var = _v(consequence="missense_variant", impact=FunctionalImpact.MODERATE)
        result = classifier.classify(var)
        assert "PVS1" not in result.pathogenic_criteria

    def test_ps1_clinvar_pathogenic(self, classifier):
        var = _v(clinvar="Pathogenic")
        result = classifier.classify(var)
        assert "PS1" in result.pathogenic_criteria

    def test_pm2_novel_variant(self, classifier):
        var = _v(gnomad_af=None)
        result = classifier.classify(var)
        assert "PM2" in result.pathogenic_criteria

    def test_pm2_extremely_rare(self, classifier):
        var = _v(gnomad_af=0.00005)
        result = classifier.classify(var)
        assert "PM2" in result.pathogenic_criteria

    def test_pm3_compound_het(self, classifier):
        var = _v(is_compound_het=True, compound_het_partner="chr1-200-C-T")
        result = classifier.classify(var)
        assert "PM3" in result.pathogenic_criteria

    def test_pp1_de_novo(self, classifier):
        var = _v(is_de_novo=True)
        result = classifier.classify(var)
        assert "PP1" in result.pathogenic_criteria

    def test_pp3_computational_evidence(self, classifier):
        var = _v(cadd=30.0, revel=0.8)
        result = classifier.classify(var)
        assert "PP3" in result.pathogenic_criteria

    # ── Benign criteria ───────────────────────────────────────────────

    def test_ba1_common_variant(self, classifier):
        var = _v(gnomad_af=0.10)
        result = classifier.classify(var)
        assert "BA1" in result.benign_criteria
        assert result.classification == ACMGClassification.BENIGN

    def test_bs1_elevated_frequency(self, classifier):
        var = _v(gnomad_af=0.02)
        result = classifier.classify(var)
        assert "BS1" in result.benign_criteria

    def test_bp4_computational_benign(self, classifier):
        var = _v(cadd=5.0, revel=0.05)
        result = classifier.classify(var)
        assert "BP4" in result.benign_criteria

    def test_bp6_clinvar_benign(self, classifier):
        var = _v(clinvar="Benign")
        result = classifier.classify(var)
        assert "BP6" in result.benign_criteria

    def test_bp7_synonymous_no_splice(self, classifier):
        var = _v(consequence="synonymous_variant", impact=FunctionalImpact.LOW, spliceai=0.01)
        result = classifier.classify(var)
        assert "BP7" in result.benign_criteria

    # ── ACMG Combining Rules ──────────────────────────────────────────

    def test_pathogenic_pvs1_plus_ps1(self, classifier):
        """PVS1 + PS1 (Very Strong + Strong) = Pathogenic."""
        var = _v(
            consequence="stop_gained", impact=FunctionalImpact.HIGH,
            clinvar="Pathogenic", gnomad_af=None,
        )
        result = classifier.classify(var)
        assert result.classification == ACMGClassification.PATHOGENIC

    def test_likely_pathogenic_pvs1_plus_pm(self, classifier):
        """PVS1 + PM2 (Very Strong + Moderate) = Likely Pathogenic."""
        var = _v(
            consequence="frameshift_variant", impact=FunctionalImpact.HIGH,
            gnomad_af=None,
        )
        result = classifier.classify(var)
        assert result.classification in (
            ACMGClassification.PATHOGENIC,
            ACMGClassification.LIKELY_PATHOGENIC,
        )

    def test_vus_insufficient_evidence(self, classifier):
        """Single weak supporting evidence = VUS."""
        var = _v(cadd=20.0, gnomad_af=0.005)
        result = classifier.classify(var)
        assert result.classification == ACMGClassification.VUS

    def test_benign_ba1_overrides_all(self, classifier):
        """BA1 stand-alone = Benign, even with some pathogenic evidence."""
        var = _v(
            consequence="missense_variant",
            cadd=28.0, revel=0.7,
            gnomad_af=0.08,  # BA1 triggered
        )
        result = classifier.classify(var)
        assert result.classification == ACMGClassification.BENIGN

    # ── Audit trail ───────────────────────────────────────────────────

    def test_evidence_audit_trail(self, classifier):
        var = _v(
            consequence="stop_gained", impact=FunctionalImpact.HIGH,
            gnomad_af=None, cadd=35.0,
        )
        result = classifier.classify(var)

        # All criteria should be in evidence list (applied or not)
        codes = [e.code for e in result.evidence]
        assert "PVS1" in codes
        assert "BA1" in codes

        # Applied criteria should have justification
        for e in result.applied_evidence:
            assert e.justification != ""
            assert e.direction in ("pathogenic", "benign")

    def test_summary_contains_classification(self, classifier):
        var = _v(consequence="stop_gained", impact=FunctionalImpact.HIGH)
        result = classifier.classify(var)
        assert "ACMG Classification" in result.summary

    def test_is_reportable_for_pathogenic(self, classifier):
        var = _v(
            consequence="stop_gained", impact=FunctionalImpact.HIGH,
            clinvar="Pathogenic", gnomad_af=None,
        )
        result = classifier.classify(var)
        assert result.is_reportable


# ── Clinical Decision Engine Tests ────────────────────────────────────────────

class TestClinicalDecisionEngine:
    @pytest.fixture
    def engine(self):
        return ClinicalDecisionEngine()

    def test_acmg_sf_gene_flagged(self, engine):
        var = _v(gene="BRCA1", consequence="stop_gained", impact=FunctionalImpact.HIGH)
        candidate = GeneCandidate(gene_symbol="BRCA1", variants=[var])
        results = engine.analyze([candidate])

        insight = results["BRCA1"]
        assert insight.is_acmg_sf_gene
        assert insight.sf_condition == "Hereditary breast/ovarian cancer"
        assert insight.sf_category == "cancer"

    def test_pgx_gene_flagged(self, engine):
        var = _v(gene="CYP2D6")
        candidate = GeneCandidate(gene_symbol="CYP2D6", variants=[var])
        results = engine.analyze([candidate])

        insight = results["CYP2D6"]
        assert insight.has_pgx_relevance
        assert "codeine" in insight.pgx_drugs
        assert insight.pgx_level == "A"

    def test_diagnostic_significance(self, engine):
        var = _v(
            gene="SCN1A",
            consequence="stop_gained", impact=FunctionalImpact.HIGH,
            clinvar="Pathogenic", gnomad_af=None,
        )
        candidate = GeneCandidate(gene_symbol="SCN1A", variants=[var])
        results = engine.analyze([candidate])

        insight = results["SCN1A"]
        assert insight.clinical_significance == "Diagnostic"
        assert insight.n_pathogenic_variants >= 1

    def test_recommendation_includes_confirmation(self, engine):
        var = _v(gene="TP53", consequence="missense_variant", clinvar="Pathogenic")
        candidate = GeneCandidate(gene_symbol="TP53", variants=[var])
        results = engine.analyze([candidate])

        rec = results["TP53"].clinical_recommendation
        assert "confirm" in rec.lower() or "review" in rec.lower()
        assert "board-certified" in rec.lower()

    def test_recommendation_for_de_novo_lof(self, engine):
        var = _v(
            gene="GENE_X",
            consequence="frameshift_variant", impact=FunctionalImpact.HIGH,
            gnomad_af=None, is_de_novo=True,
        )
        candidate = GeneCandidate(
            gene_symbol="GENE_X", variants=[var],
            has_de_novo_lof=True,
        )
        results = engine.analyze([candidate])

        rec = results["GENE_X"].clinical_recommendation
        assert "de novo" in rec.lower()

    def test_vus_recommendation(self, engine):
        var = _v(gene="GENE_Y", cadd=18.0, gnomad_af=0.005)
        candidate = GeneCandidate(gene_symbol="GENE_Y", variants=[var])
        results = engine.analyze([candidate])

        insight = results["GENE_Y"]
        assert insight.clinical_significance == "Research"
        assert "reclassification" in insight.clinical_recommendation.lower()

    def test_audit_timestamp_present(self, engine):
        var = _v(gene="GENE_Z")
        candidate = GeneCandidate(gene_symbol="GENE_Z", variants=[var])
        results = engine.analyze([candidate])

        insight = results["GENE_Z"]
        assert insight.analysis_timestamp != ""
        assert insight.classification_method != ""
        assert insight.analyst_review_required is True

    def test_enrich_candidates(self, engine):
        var = _v(gene="BRCA2", consequence="stop_gained", impact=FunctionalImpact.HIGH,
                clinvar="Pathogenic", gnomad_af=None)
        candidate = GeneCandidate(gene_symbol="BRCA2", variants=[var], evidence_summary={})
        results = engine.analyze([candidate])
        enriched = engine.enrich_candidates([candidate], results)

        ev = enriched[0].evidence_summary
        assert ev["is_acmg_sf_gene"] is True
        assert ev["acmg_class"] == "Pathogenic"
        assert ev["clinical_significance"] == "Diagnostic"


# ── Reference Data Tests ──────────────────────────────────────────────────────

class TestReferenceData:
    def test_acmg_sf_gene_count(self):
        """ACMG SF v3.2 should have ~60+ genes."""
        assert len(ACMG_SF_GENES) >= 55

    def test_acmg_sf_categories(self):
        categories = {g["category"] for g in ACMG_SF_GENES.values()}
        assert "cardiovascular" in categories
        assert "cancer" in categories
        assert "metabolic" in categories

    def test_acmg_sf_key_genes_present(self):
        assert "BRCA1" in ACMG_SF_GENES
        assert "BRCA2" in ACMG_SF_GENES
        assert "TP53" in ACMG_SF_GENES
        assert "MLH1" in ACMG_SF_GENES
        assert "LDLR" in ACMG_SF_GENES
        assert "SCN5A" in ACMG_SF_GENES
        assert "FBN1" in ACMG_SF_GENES

    def test_pgx_gene_count(self):
        """Should have 10+ pharmacogenomic genes."""
        assert len(PHARMACOGENOMIC_GENES) >= 10

    def test_pgx_key_genes_present(self):
        assert "CYP2D6" in PHARMACOGENOMIC_GENES
        assert "CYP2C19" in PHARMACOGENOMIC_GENES
        assert "DPYD" in PHARMACOGENOMIC_GENES
        assert "G6PD" in PHARMACOGENOMIC_GENES
        assert "HLA-B" in PHARMACOGENOMIC_GENES

    def test_pgx_drugs_populated(self):
        for gene, data in PHARMACOGENOMIC_GENES.items():
            assert len(data["drugs"]) > 0, f"{gene} has no drugs"
            assert data["level"] in ("A", "B"), f"{gene} has invalid level"

    def test_scn1a_pgx_sodium_blockers(self):
        """SCN1A PGx: sodium channel blockers contraindicated in Dravet."""
        pgx = PHARMACOGENOMIC_GENES["SCN1A"]
        assert "carbamazepine" in pgx["drugs"]
        assert "Dravet" in pgx["action"]
