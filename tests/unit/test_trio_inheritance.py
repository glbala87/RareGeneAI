"""Unit tests for trio-based inheritance modeling.

Tests:
  - De novo variant detection
  - Compound heterozygous identification (in trans)
  - Homozygous recessive confirmation
  - X-linked inheritance
  - Variant tagging (in-place mutation of AnnotatedVariant)
  - Inheritance scoring (de novo LoF > de novo missense > comp het > hom rec)
  - Gene-level inheritance summary
  - Integration with variant scorer and gene ranker
  - Explainer trio output
"""

import pytest

from raregeneai.config.settings import ScoringConfig
from raregeneai.explainability.explainer import Explainer
from raregeneai.models.data_models import (
    AnnotatedVariant,
    FunctionalImpact,
    GeneCandidate,
    InheritanceMode,
    Variant,
    Zygosity,
)
from raregeneai.ranking.gene_ranker import GeneRanker
from raregeneai.scoring.inheritance_analyzer import (
    INHERITANCE_WEIGHTS,
    InheritanceAnalyzer,
)
from raregeneai.scoring.variant_scorer import VariantScorer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _v(chrom="chr1", pos=100, ref="A", alt="G", zyg=Zygosity.HET,
       gene="GENE_A", impact=FunctionalImpact.HIGH, **kwargs):
    """Create an AnnotatedVariant with defaults."""
    return AnnotatedVariant(
        variant=Variant(chrom=chrom, pos=pos, ref=ref, alt=alt, zygosity=zyg,
                        depth=30, gq=40),
        gene_symbol=gene,
        impact=impact,
        consequence="stop_gained" if impact == FunctionalImpact.HIGH else "missense_variant",
        **kwargs,
    )


# ── De Novo Detection Tests ──────────────────────────────────────────────────

class TestDeNovo:
    @pytest.fixture
    def analyzer(self):
        return InheritanceAnalyzer()

    def test_detect_de_novo(self, analyzer):
        """Variant in proband absent in both parents = de novo."""
        proband = [_v(pos=100)]
        father = [_v(pos=200)]  # Different variant
        mother = [_v(pos=300)]  # Different variant

        results = analyzer.analyze_trio(proband, father, mother)

        assert len(results["de_novo"]) == 1
        assert proband[0].is_de_novo
        assert proband[0].inheritance_class == "de_novo"

    def test_de_novo_lof_highest_score(self, analyzer):
        """De novo LoF should get the maximum inheritance score."""
        proband = [_v(pos=100, impact=FunctionalImpact.HIGH)]
        father = []
        mother = []

        analyzer.analyze_trio(proband, father, mother)

        assert proband[0].is_de_novo
        assert proband[0].inheritance_score == INHERITANCE_WEIGHTS["de_novo_lof"]
        assert proband[0].inheritance_score == 1.0

    def test_de_novo_missense_lower_than_lof(self, analyzer):
        """De novo missense should score lower than de novo LoF."""
        proband = [_v(pos=100, impact=FunctionalImpact.MODERATE)]
        analyzer.analyze_trio(proband, [], [])

        assert proband[0].is_de_novo
        assert proband[0].inheritance_score == INHERITANCE_WEIGHTS["de_novo_missense"]
        assert proband[0].inheritance_score < INHERITANCE_WEIGHTS["de_novo_lof"]

    def test_not_de_novo_if_in_parent(self, analyzer):
        """Variant present in father is NOT de novo."""
        shared_var = _v(pos=100)
        father_var = _v(pos=100)  # Same position/alleles

        proband = [shared_var]
        father = [father_var]
        mother = []

        results = analyzer.analyze_trio(proband, father, mother)

        assert len(results["de_novo"]) == 0
        assert not proband[0].is_de_novo


# ── Compound Heterozygous Tests ──────────────────────────────────────────────

class TestCompoundHet:
    @pytest.fixture
    def analyzer(self):
        return InheritanceAnalyzer()

    def test_detect_compound_het_in_trans(self, analyzer):
        """Two het variants from different parents = compound het."""
        v1 = _v(pos=100, gene="SCN1A")  # Will be paternal
        v2 = _v(pos=200, ref="C", alt="T", gene="SCN1A")  # Will be maternal

        proband = [v1, v2]
        father = [_v(pos=100)]  # Father has v1
        mother = [_v(pos=200, ref="C", alt="T")]  # Mother has v2

        results = analyzer.analyze_trio(proband, father, mother)

        assert len(results["compound_het"]) == 1
        assert v1.is_compound_het
        assert v2.is_compound_het
        assert v1.compound_het_partner_key == v2.variant.variant_key
        assert v2.compound_het_partner_key == v1.variant.variant_key

    def test_compound_het_not_if_same_parent(self, analyzer):
        """Two het variants from the same parent = NOT compound het (in cis)."""
        v1 = _v(pos=100, gene="GENE_X")
        v2 = _v(pos=200, ref="C", alt="T", gene="GENE_X")

        proband = [v1, v2]
        # Both from father (not in trans)
        father = [_v(pos=100), _v(pos=200, ref="C", alt="T")]
        mother = []

        results = analyzer.analyze_trio(proband, father, mother)

        assert len(results["compound_het"]) == 0
        assert not v1.is_compound_het
        assert not v2.is_compound_het

    def test_compound_het_scoring_lof_plus_missense(self, analyzer):
        """Compound het LoF + missense should score appropriately."""
        v1 = _v(pos=100, gene="GENE_A", impact=FunctionalImpact.HIGH)
        v2 = _v(pos=200, ref="C", alt="T", gene="GENE_A", impact=FunctionalImpact.MODERATE)

        proband = [v1, v2]
        father = [_v(pos=100, impact=FunctionalImpact.HIGH)]
        mother = [_v(pos=200, ref="C", alt="T", impact=FunctionalImpact.MODERATE)]

        analyzer.analyze_trio(proband, father, mother)

        gene_summary = analyzer.compute_gene_inheritance_score("GENE_A", proband)
        assert gene_summary["has_compound_het"]
        assert gene_summary["inheritance_score"] >= INHERITANCE_WEIGHTS["compound_het_lof_mis"]


# ── Homozygous Recessive Tests ────────────────────────────────────────────────

class TestHomozygousRecessive:
    @pytest.fixture
    def analyzer(self):
        return InheritanceAnalyzer()

    def test_detect_hom_recessive(self, analyzer):
        """HOM_ALT in proband + HET in both parents = homozygous recessive."""
        proband = [_v(pos=100, zyg=Zygosity.HOM_ALT)]
        father = [_v(pos=100, zyg=Zygosity.HET)]
        mother = [_v(pos=100, zyg=Zygosity.HET)]

        results = analyzer.analyze_trio(proband, father, mother)

        assert len(results["homozygous_recessive"]) == 1
        assert proband[0].is_hom_recessive
        assert proband[0].inheritance_class == "hom_recessive"
        assert proband[0].parent_of_origin == "both"

    def test_hom_recessive_lof_high_score(self, analyzer):
        proband = [_v(pos=100, zyg=Zygosity.HOM_ALT, impact=FunctionalImpact.HIGH)]
        father = [_v(pos=100, zyg=Zygosity.HET)]
        mother = [_v(pos=100, zyg=Zygosity.HET)]

        analyzer.analyze_trio(proband, father, mother)

        assert proband[0].inheritance_score == INHERITANCE_WEIGHTS["hom_recessive_lof"]

    def test_not_hom_recessive_if_parent_hom(self, analyzer):
        """If one parent is HOM_ALT, not classic recessive pattern."""
        proband = [_v(pos=100, zyg=Zygosity.HOM_ALT)]
        father = [_v(pos=100, zyg=Zygosity.HOM_ALT)]  # Father is also HOM
        mother = [_v(pos=100, zyg=Zygosity.HET)]

        results = analyzer.analyze_trio(proband, father, mother)

        assert len(results["homozygous_recessive"]) == 0
        assert not proband[0].is_hom_recessive


# ── No Trio (Zygosity-only) Tests ─────────────────────────────────────────────

class TestNoTrio:
    @pytest.fixture
    def analyzer(self):
        return InheritanceAnalyzer()

    def test_no_parents_tags_zygosity_only(self, analyzer):
        proband = [_v(pos=100, zyg=Zygosity.HET)]
        analyzer.analyze_trio(proband)  # No parents

        assert proband[0].inheritance_class == "unknown_het"
        assert proband[0].inheritance_score == INHERITANCE_WEIGHTS["unknown_het"]

    def test_no_parents_hom_scores_moderate(self, analyzer):
        proband = [_v(pos=100, zyg=Zygosity.HOM_ALT)]
        analyzer.analyze_trio(proband)

        assert proband[0].inheritance_class == "unknown_hom"
        assert proband[0].inheritance_score == INHERITANCE_WEIGHTS["unknown_hom"]


# ── Gene-level Inheritance Summary ────────────────────────────────────────────

class TestGeneInheritanceSummary:
    @pytest.fixture
    def analyzer(self):
        return InheritanceAnalyzer()

    def test_gene_with_de_novo_lof(self, analyzer):
        v = _v(pos=100, impact=FunctionalImpact.HIGH)
        analyzer.analyze_trio([v], [], [])

        summary = analyzer.compute_gene_inheritance_score("GENE_A", [v])
        assert summary["has_de_novo"]
        assert summary["has_de_novo_lof"]
        assert summary["trio_analyzed"]
        assert summary["inheritance_class"] == "de_novo_lof"
        assert summary["inheritance_score"] == 1.0

    def test_gene_with_mixed_variants(self, analyzer):
        v1 = _v(pos=100, impact=FunctionalImpact.HIGH)  # De novo LoF
        v2 = _v(pos=200, ref="C", alt="T", impact=FunctionalImpact.MODERATE)  # Inherited

        proband = [v1, v2]
        father = [_v(pos=200, ref="C", alt="T")]  # v2 is inherited
        mother = []

        analyzer.analyze_trio(proband, father, mother)
        summary = analyzer.compute_gene_inheritance_score("GENE_A", proband)

        assert summary["has_de_novo_lof"]
        assert summary["inheritance_score"] == 1.0  # Max from de novo LoF

    def test_gene_no_trio_data(self, analyzer):
        v = _v(pos=100, zyg=Zygosity.HET)
        v.inheritance_class = "unknown_het"
        v.inheritance_score = 0.3

        summary = analyzer.compute_gene_inheritance_score("GENE_A", [v])
        assert not summary["trio_analyzed"]
        assert summary["inheritance_score"] == 0.3


# ── Inheritance Score Ordering ────────────────────────────────────────────────

class TestInheritanceScoreOrdering:
    """Verify the clinical hierarchy of inheritance scores."""

    def test_de_novo_lof_is_maximum(self):
        assert INHERITANCE_WEIGHTS["de_novo_lof"] == 1.0

    def test_de_novo_lof_gt_de_novo_missense(self):
        assert INHERITANCE_WEIGHTS["de_novo_lof"] > INHERITANCE_WEIGHTS["de_novo_missense"]

    def test_de_novo_gt_compound_het(self):
        assert INHERITANCE_WEIGHTS["de_novo_missense"] > INHERITANCE_WEIGHTS["compound_het_lof_mis"]

    def test_compound_het_lof_lof_gt_lof_mis(self):
        assert INHERITANCE_WEIGHTS["compound_het_lof_lof"] > INHERITANCE_WEIGHTS["compound_het_lof_mis"]

    def test_hom_recessive_lof_high(self):
        assert INHERITANCE_WEIGHTS["hom_recessive_lof"] >= 0.9

    def test_inherited_dominant_lower_than_biallelic(self):
        assert INHERITANCE_WEIGHTS["inherited_dominant"] < INHERITANCE_WEIGHTS["compound_het_other"]

    def test_unknown_het_lowest_practical(self):
        assert INHERITANCE_WEIGHTS["unknown_het"] <= 0.3


# ── Variant Scorer Integration ────────────────────────────────────────────────

class TestVariantScorerTrio:
    def test_scorer_uses_trio_inheritance_score(self):
        """VariantScorer should use pre-computed trio score when available."""
        scorer = VariantScorer()
        var = _v(pos=100, impact=FunctionalImpact.HIGH)
        var.inheritance_score = 1.0  # Pre-computed by InheritanceAnalyzer
        var.is_de_novo = True

        result = scorer._inheritance_score(var)
        assert result == 1.0  # Should use the pre-computed score

    def test_scorer_fallback_without_trio(self):
        """VariantScorer should use zygosity fallback when no trio."""
        scorer = VariantScorer()
        var = _v(pos=100, zyg=Zygosity.HET)
        # No trio data: inheritance_score = 0.0

        result = scorer._inheritance_score(var)
        assert result == 0.3  # HET fallback

    def test_composite_score_boosted_by_de_novo(self):
        """De novo LoF should significantly boost composite score."""
        scorer = VariantScorer()

        de_novo_lof = _v(pos=100, impact=FunctionalImpact.HIGH, zyg=Zygosity.HET)
        de_novo_lof.inheritance_score = 1.0
        de_novo_lof.is_de_novo = True
        de_novo_lof.cadd_phred = 35.0

        inherited = _v(pos=200, ref="C", alt="T", impact=FunctionalImpact.HIGH, zyg=Zygosity.HET)
        inherited.inheritance_score = 0.4
        inherited.cadd_phred = 35.0

        scorer.score_variants([de_novo_lof, inherited])

        assert de_novo_lof.composite_score > inherited.composite_score


# ── Gene Ranker Integration ───────────────────────────────────────────────────

class TestGeneRankerTrio:
    def test_de_novo_lof_gene_ranks_highest(self):
        """Gene with de novo LoF should rank above gene with inherited variant."""
        ranker = GeneRanker()

        # Gene A: de novo LoF
        v_denovo = _v(pos=100, gene="GENE_A", impact=FunctionalImpact.HIGH)
        v_denovo.is_de_novo = True
        v_denovo.inheritance_class = "de_novo"
        v_denovo.inheritance_score = 1.0
        v_denovo.composite_score = 0.8

        # Gene B: inherited missense
        v_inherited = _v(pos=200, ref="C", alt="T", gene="GENE_B",
                        impact=FunctionalImpact.MODERATE)
        v_inherited.inheritance_class = "inherited_dominant"
        v_inherited.inheritance_score = 0.4
        v_inherited.composite_score = 0.7

        phenotype_scores = {"GENE_A": 0.5, "GENE_B": 0.5}
        ranked = ranker.rank([v_denovo, v_inherited], phenotype_scores)

        assert ranked[0].gene_symbol == "GENE_A"
        assert ranked[0].has_de_novo_lof
        assert ranked[0].inheritance_score == 1.0

    def test_evidence_summary_includes_trio(self):
        ranker = GeneRanker()

        v = _v(pos=100, gene="GENE_X", impact=FunctionalImpact.HIGH)
        v.is_de_novo = True
        v.inheritance_class = "de_novo"
        v.inheritance_score = 1.0
        v.composite_score = 0.7

        ranked = ranker.rank([v], {"GENE_X": 0.6})

        ev = ranked[0].evidence_summary
        assert ev["has_de_novo"] is True
        assert ev["has_de_novo_lof"] is True
        assert ev["trio_analyzed"] is True
        assert ev["inheritance_score"] == 1.0

    def test_compound_het_gene_ranked(self):
        ranker = GeneRanker()

        v1 = _v(pos=100, gene="GENE_Y", impact=FunctionalImpact.HIGH)
        v1.is_compound_het = True
        v1.inheritance_class = "compound_het"
        v1.inheritance_score = 0.95

        v2 = _v(pos=200, ref="C", alt="T", gene="GENE_Y",
               impact=FunctionalImpact.MODERATE)
        v2.is_compound_het = True
        v2.inheritance_class = "compound_het"
        v2.inheritance_score = 0.75
        v2.compound_het_partner_key = v1.variant.variant_key
        v1.compound_het_partner_key = v2.variant.variant_key

        v1.composite_score = 0.7
        v2.composite_score = 0.5

        ranked = ranker.rank([v1, v2], {"GENE_Y": 0.5})

        assert ranked[0].has_compound_het
        assert ranked[0].evidence_summary["has_compound_het"] is True


# ── Explainer Trio Tests ──────────────────────────────────────────────────────

class TestExplainerTrio:
    def test_explain_de_novo_lof(self):
        explainer = Explainer()
        var = _v(pos=100, impact=FunctionalImpact.HIGH)
        var.is_de_novo = True
        var.inheritance_score = 1.0

        candidate = GeneCandidate(
            gene_symbol="GENE_A",
            variants=[var],
            gene_rank_score=0.9,
            confidence=0.95,
            has_de_novo=True,
            has_de_novo_lof=True,
            trio_analyzed=True,
            inheritance_score=1.0,
            inheritance_modes=[InheritanceMode.AUTOSOMAL_DOMINANT],
        )
        text = explainer.explain_gene(candidate)

        assert "DE NOVO LOSS-OF-FUNCTION" in text
        assert "very strong" in text.lower()

    def test_explain_compound_het(self):
        explainer = Explainer()

        v1 = _v(pos=100, impact=FunctionalImpact.HIGH)
        v1.is_compound_het = True
        v1.compound_het_partner_key = "chr1-200-C-T"

        v2 = _v(pos=200, ref="C", alt="T", impact=FunctionalImpact.MODERATE)
        v2.is_compound_het = True
        v2.compound_het_partner_key = "chr1-100-A-G"

        candidate = GeneCandidate(
            gene_symbol="GENE_B",
            variants=[v1, v2],
            gene_rank_score=0.7,
            confidence=0.8,
            has_compound_het=True,
            trio_analyzed=True,
            inheritance_score=0.85,
        )
        text = explainer.explain_gene(candidate)

        assert "COMPOUND HETEROZYGOUS" in text
        assert "biallelic" in text.lower()

    def test_explain_hom_recessive(self):
        explainer = Explainer()
        var = _v(pos=100, zyg=Zygosity.HOM_ALT)
        var.is_hom_recessive = True

        candidate = GeneCandidate(
            gene_symbol="GENE_C",
            variants=[var],
            gene_rank_score=0.6,
            confidence=0.7,
            has_hom_recessive=True,
            trio_analyzed=True,
            inheritance_score=0.75,
        )
        text = explainer.explain_gene(candidate)

        assert "HOMOZYGOUS RECESSIVE" in text
        assert "carriers" in text.lower()

    def test_explain_variant_de_novo_tag(self):
        explainer = Explainer()
        var = _v(pos=100)
        var.is_de_novo = True
        text = explainer._explain_variant(var)
        assert "DE NOVO" in text

    def test_explain_variant_compound_het_tag(self):
        explainer = Explainer()
        var = _v(pos=100)
        var.is_compound_het = True
        var.compound_het_partner_key = "chr1-200-C-T"
        text = explainer._explain_variant(var)
        assert "COMPOUND HET" in text
        assert "chr1-200-C-T" in text

    def test_explain_variant_hom_recessive_tag(self):
        explainer = Explainer()
        var = _v(pos=100, zyg=Zygosity.HOM_ALT)
        var.is_hom_recessive = True
        text = explainer._explain_variant(var)
        assert "HOM RECESSIVE" in text

    def test_explain_variant_inherited_paternal(self):
        explainer = Explainer()
        var = _v(pos=100)
        var.parent_of_origin = "paternal"
        text = explainer._explain_variant(var)
        assert "paternal" in text
