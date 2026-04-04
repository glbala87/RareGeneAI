"""Unit tests for non-coding variant prioritization.

Tests the full non-coding pipeline:
  - Regulatory annotation (ENCODE, ChromHMM, conservation)
  - Variant-to-gene mapping (eQTL, Hi-C, ABC)
  - Regulatory impact scoring
  - Gene ranking with non-coding features
  - Explainability for non-coding variants
"""

import pytest

from raregeneai.annotation.regulatory_annotator import (
    CHROMHMM_STATES,
    CCRE_TYPE_MAP,
    RegulatoryAnnotator,
)
from raregeneai.config.settings import RegulatoryConfig, ScoringConfig
from raregeneai.explainability.explainer import Explainer
from raregeneai.models.data_models import (
    ACMGClassification,
    AnnotatedVariant,
    FunctionalImpact,
    GeneCandidate,
    Variant,
    Zygosity,
)
from raregeneai.ranking.gene_ranker import GeneRanker
from raregeneai.scoring.variant_scorer import VariantScorer


def _make_variant(
    chrom="chr1", pos=1000000, ref="A", alt="G",
    zygosity=Zygosity.HET, **kwargs,
) -> AnnotatedVariant:
    """Helper to create an AnnotatedVariant with defaults."""
    v = Variant(chrom=chrom, pos=pos, ref=ref, alt=alt, zygosity=zygosity,
                depth=30, gq=40)
    return AnnotatedVariant(variant=v, **kwargs)


def _make_noncoding_variant(
    consequence="intron_variant",
    regulatory_class="",
    chromhmm_state="",
    phastcons=None,
    spliceai=None,
    cadd=None,
    gnomad_af=None,
    regulatory_score=0.0,
    target_gene="",
    gene_mapping_method="",
    gene_mapping_score=0.0,
    enformer=None,
    **kwargs,
) -> AnnotatedVariant:
    """Helper to create a non-coding AnnotatedVariant."""
    return _make_variant(
        consequence=consequence,
        impact=FunctionalImpact.MODIFIER,
        regulatory_class=regulatory_class,
        chromhmm_state=chromhmm_state,
        phastcons_score=phastcons,
        spliceai_max=spliceai,
        cadd_phred=cadd,
        gnomad_af=gnomad_af,
        regulatory_score=regulatory_score,
        target_gene_symbol=target_gene,
        gene_mapping_method=gene_mapping_method,
        gene_mapping_score=gene_mapping_score,
        enformer_score=enformer,
        **kwargs,
    )


# ── Data Model Tests ──────────────────────────────────────────────────────────

class TestNoncodingProperties:
    def test_is_noncoding_intron(self):
        var = _make_noncoding_variant(consequence="intron_variant")
        assert var.is_noncoding

    def test_is_noncoding_intergenic(self):
        var = _make_noncoding_variant(consequence="intergenic_variant")
        assert var.is_noncoding

    def test_is_noncoding_regulatory(self):
        var = _make_noncoding_variant(consequence="regulatory_region_variant")
        assert var.is_noncoding

    def test_is_noncoding_utr(self):
        var = _make_noncoding_variant(consequence="5_prime_UTR_variant")
        assert var.is_noncoding

    def test_is_not_noncoding_missense(self):
        var = _make_variant(consequence="missense_variant", impact=FunctionalImpact.MODERATE)
        assert not var.is_noncoding

    def test_has_regulatory_annotation(self):
        var = _make_noncoding_variant(regulatory_class="enhancer")
        assert var.has_regulatory_annotation

    def test_no_regulatory_annotation(self):
        var = _make_noncoding_variant()
        assert not var.has_regulatory_annotation

    def test_effective_gene_symbol_noncoding(self):
        var = _make_noncoding_variant(
            consequence="intergenic_variant",
            target_gene="BRCA1",
            gene_symbol="",
        )
        assert var.effective_gene_symbol == "BRCA1"

    def test_effective_gene_symbol_coding(self):
        var = _make_variant(
            gene_symbol="TP53",
            consequence="missense_variant",
            impact=FunctionalImpact.MODERATE,
        )
        assert var.effective_gene_symbol == "TP53"

    def test_gene_candidate_has_splicing_variant(self):
        var = _make_noncoding_variant(spliceai=0.7)
        gc = GeneCandidate(gene_symbol="TEST", variants=[var])
        assert gc.has_splicing_variant

    def test_gene_candidate_no_splicing(self):
        var = _make_noncoding_variant(spliceai=0.1)
        gc = GeneCandidate(gene_symbol="TEST", variants=[var])
        assert not gc.has_splicing_variant


# ── Regulatory Annotator Tests ────────────────────────────────────────────────

class TestRegulatoryAnnotator:
    @pytest.fixture
    def annotator(self):
        return RegulatoryAnnotator(RegulatoryConfig(enabled=True))

    def test_disabled_returns_unchanged(self):
        cfg = RegulatoryConfig(enabled=False)
        annotator = RegulatoryAnnotator(cfg)
        var = _make_noncoding_variant()
        result = annotator.annotate([var])
        assert result[0].regulatory_score == 0.0

    def test_skips_coding_variants(self, annotator):
        var = _make_variant(
            consequence="missense_variant",
            impact=FunctionalImpact.MODERATE,
        )
        result = annotator.annotate([var])
        assert result[0].regulatory_score == 0.0

    def test_compute_regulatory_score_splicing(self, annotator):
        var = _make_noncoding_variant(spliceai=0.8)
        annotator._compute_regulatory_score(var)
        # Splicing should contribute to regulatory score
        assert var.regulatory_score > 0.2

    def test_compute_regulatory_score_enhancer_conserved(self, annotator):
        var = _make_noncoding_variant(
            regulatory_class="enhancer",
            phastcons=0.95,
            chromhmm_state="7_Enh",
        )
        annotator._compute_regulatory_score(var)
        # region=0.75*0.20 + conservation=0.95*0.15 = 0.2925
        assert var.regulatory_score > 0.25

    def test_compute_regulatory_score_promoter_mapped(self, annotator):
        var = _make_noncoding_variant(
            regulatory_class="promoter",
            chromhmm_state="1_TssA",
            target_gene="GENE1",
            gene_mapping_method="eqtl",
            gene_mapping_score=0.8,
        )
        annotator._compute_regulatory_score(var)
        # region=0.85*0.20 + mapping=0.8*0.15 = 0.29
        assert var.regulatory_score > 0.25

    def test_estimate_dl_score_from_conservation(self, annotator):
        var = _make_noncoding_variant(
            phastcons=0.9,
            chromhmm_state="7_Enh",
            regulatory_class="enhancer",
        )
        annotator._estimate_regulatory_dl_score(var)
        assert var.enformer_score is not None
        assert var.enformer_score > 0.2

    def test_chromhmm_state_mapping(self):
        # Verify ChromHMM states have correct scores
        assert "1_TssA" in CHROMHMM_STATES
        assert CHROMHMM_STATES["1_TssA"][1] > 0.8  # Active TSS should be high
        assert CHROMHMM_STATES["7_Enh"][1] > 0.7  # Enhancer should be high
        assert CHROMHMM_STATES["15_Quies"][1] < 0.05  # Quiescent should be low

    def test_ccre_type_mapping(self):
        assert "PLS" in CCRE_TYPE_MAP
        assert CCRE_TYPE_MAP["PLS"][0] == "promoter"
        assert CCRE_TYPE_MAP["dELS"][0] == "enhancer"


# ── Variant Scoring Tests (Non-coding) ────────────────────────────────────────

class TestVariantScorerNoncoding:
    @pytest.fixture
    def scorer(self):
        return VariantScorer()

    def test_regulatory_impact_score_noncoding(self, scorer):
        var = _make_noncoding_variant(regulatory_score=0.7)
        score = scorer._regulatory_impact_score(var)
        assert score == 0.7

    def test_regulatory_impact_score_coding_zero(self, scorer):
        var = _make_variant(
            consequence="missense_variant",
            impact=FunctionalImpact.MODERATE,
        )
        score = scorer._regulatory_impact_score(var)
        assert score == 0.0

    def test_regulatory_impact_score_coding_splicing(self, scorer):
        var = _make_variant(
            consequence="missense_variant",
            impact=FunctionalImpact.MODERATE,
            spliceai_max=0.6,
        )
        score = scorer._regulatory_impact_score(var)
        assert score > 0.0  # Partial credit for splicing

    def test_composite_includes_regulatory(self, scorer):
        var = _make_noncoding_variant(
            regulatory_score=0.8,
            cadd=25.0,
            gnomad_af=None,
        )
        scorer.score_variants([var])
        # Regulatory weight is 0.15, so regulatory contribution should be ~0.12
        assert var.composite_score > 0.1

    def test_noncoding_enhancer_boosted_impact(self, scorer):
        # Enhancer variant with conservation should get boosted functional score
        var = _make_noncoding_variant(
            consequence="regulatory_region_variant",
            regulatory_class="enhancer",
            phastcons=0.9,
        )
        score = scorer._functional_impact_score(var)
        # Base score for regulatory_region_variant is 0.35, boost = 0.30
        assert score > 0.35

    def test_noncoding_consequence_scores(self, scorer):
        # Verify non-coding consequences are properly scored
        regulatory_var = _make_noncoding_variant(consequence="regulatory_region_variant")
        intergenic_var = _make_noncoding_variant(consequence="intergenic_variant")
        tfbs_var = _make_noncoding_variant(consequence="TFBS_ablation")

        reg_score = scorer._functional_impact_score(regulatory_var)
        inter_score = scorer._functional_impact_score(intergenic_var)
        tfbs_score = scorer._functional_impact_score(tfbs_var)

        # TFBS_ablation > regulatory_region > intergenic
        assert tfbs_score > reg_score > inter_score

    def test_pathogenicity_includes_enformer(self, scorer):
        var = _make_noncoding_variant(enformer=0.8)
        score = scorer._pathogenicity_score(var)
        assert score >= 0.8

    def test_filter_includes_noncoding_when_enabled(self, scorer):
        coding = _make_variant(
            consequence="missense_variant",
            impact=FunctionalImpact.MODERATE,
        )
        noncoding = _make_noncoding_variant(regulatory_score=0.5)
        coding.composite_score = 0.5
        noncoding.composite_score = 0.3

        # Default: require_coding=False, should include non-coding
        filtered = scorer.filter_variants([coding, noncoding], min_score=0.1)
        assert len(filtered) == 2

    def test_filter_excludes_noncoding_when_coding_required(self, scorer):
        coding = _make_variant(
            consequence="missense_variant",
            impact=FunctionalImpact.MODERATE,
        )
        noncoding = _make_noncoding_variant()
        coding.composite_score = 0.5
        noncoding.composite_score = 0.3

        filtered = scorer.filter_variants(
            [coding, noncoding], min_score=0.1, require_coding=True
        )
        assert len(filtered) == 1


# ── Gene Ranking Tests (Non-coding) ──────────────────────────────────────────

class TestGeneRankerNoncoding:
    @pytest.fixture
    def ranker(self):
        return GeneRanker()

    def test_noncoding_variants_grouped_by_target_gene(self, ranker):
        # Non-coding variant mapped to GENE_A via eQTL
        nc_var = _make_noncoding_variant(
            gene_symbol="",
            target_gene="GENE_A",
            gene_mapping_method="eqtl",
            gene_mapping_score=0.9,
        )
        nc_var.composite_score = 0.5

        groups = ranker._group_by_gene([nc_var])
        assert "GENE_A" in groups

    def test_mixed_coding_noncoding_ranking(self, ranker):
        # Gene A: has coding LoF variant
        coding_var = _make_variant(
            gene_symbol="GENE_A",
            gene_id="ENSG1",
            consequence="stop_gained",
            impact=FunctionalImpact.HIGH,
            cadd_phred=35.0,
        )
        coding_var.composite_score = 0.8

        # Gene B: has non-coding regulatory variant with strong evidence
        nc_var = _make_noncoding_variant(
            gene_symbol="",
            gene_id="ENSG2",
            target_gene="GENE_B",
            gene_mapping_method="abc",
            gene_mapping_score=0.95,
            regulatory_score=0.85,
            regulatory_class="enhancer",
            phastcons=0.98,
            spliceai=None,
        )
        nc_var.composite_score = 0.6

        phenotype_scores = {"GENE_A": 0.8, "GENE_B": 0.6}
        ranked = ranker.rank([coding_var, nc_var], phenotype_scores)

        assert len(ranked) == 2
        # Both genes should have non-zero scores
        for gene in ranked:
            assert gene.gene_rank_score > 0

    def test_evidence_summary_includes_regulatory(self, ranker):
        nc_var = _make_noncoding_variant(
            gene_symbol="GENE_X",
            gene_id="ENSG_X",
            regulatory_class="enhancer",
            regulatory_score=0.7,
            phastcons=0.9,
        )
        nc_var.composite_score = 0.5

        phenotype_scores = {"GENE_X": 0.5}
        ranked = ranker.rank([nc_var], phenotype_scores)

        assert len(ranked) == 1
        ev = ranked[0].evidence_summary
        assert ev["max_regulatory_score"] == 0.7
        assert ev["has_enhancer_variant"] is True
        assert ev["max_conservation_score"] == 0.9
        assert ev["n_noncoding_variants"] == 1


# ── Explainer Tests (Non-coding) ─────────────────────────────────────────────

class TestExplainerNoncoding:
    @pytest.fixture
    def explainer(self):
        return Explainer()

    def test_explain_noncoding_variant(self, explainer):
        var = _make_noncoding_variant(
            consequence="regulatory_region_variant",
            regulatory_class="enhancer",
            chromhmm_state_name="Enhancers",
            phastcons=0.95,
            target_gene="BRCA1",
            gene_mapping_method="eqtl",
            eqtl_tissue="Brain_Cortex",
            gene_mapping_score=0.85,
            regulatory_score=0.72,
        )
        text = explainer._explain_variant(var)

        assert "REGULATORY" in text
        assert "enhancer" in text
        assert "BRCA1" in text
        assert "eQTL" in text
        assert "0.72" in text  # regulatory score

    def test_explain_splicing_variant(self, explainer):
        var = _make_noncoding_variant(
            consequence="intron_variant",
            spliceai=0.8,
        )
        text = explainer._explain_variant(var)
        assert "SpliceAI=0.80" in text
        assert "strong splice disruption" in text

    def test_acmg_splicing_evidence(self, explainer):
        var = _make_noncoding_variant(
            consequence="intron_variant",
            spliceai=0.6,
            gnomad_af=None,
        )
        classification = explainer.classify_acmg(var)
        # Strong SpliceAI + novel should give at least LP
        assert classification in (
            ACMGClassification.LIKELY_PATHOGENIC,
            ACMGClassification.PATHOGENIC,
        )

    def test_acmg_regulatory_evidence(self, explainer):
        var = _make_noncoding_variant(
            consequence="regulatory_region_variant",
            regulatory_score=0.8,
            phastcons=0.95,
            gnomad_af=None,
            regulatory_class="enhancer",
        )
        classification = explainer.classify_acmg(var)
        # Novel + regulatory + conservation support should give at least VUS
        assert classification != ACMGClassification.BENIGN

    def test_explain_gene_with_regulatory(self, explainer):
        var = _make_noncoding_variant(
            consequence="regulatory_region_variant",
            regulatory_class="promoter",
            target_gene="TEST_GENE",
            gene_mapping_method="abc",
            gene_mapping_score=0.9,
            regulatory_score=0.75,
        )
        candidate = GeneCandidate(
            gene_symbol="TEST_GENE",
            variants=[var],
            gene_rank_score=0.65,
            confidence=0.8,
            max_regulatory_score=0.75,
            has_regulatory_variant=True,
        )
        text = explainer.explain_gene(candidate)
        assert "TEST_GENE" in text
        assert "promoter" in text or "Regulatory" in text
