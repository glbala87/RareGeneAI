"""Unit tests for population-specific frequency and founder variant detection.

Tests:
  - Population-adjusted rarity scoring
  - QGP/local database lookup
  - Founder variant detection (enrichment-based)
  - Known Middle Eastern founder gene list
  - effective_af property (max of local + global)
  - is_rare property with population data
  - Integration with VariantScorer
  - Explainer population output
"""

import pytest

from raregeneai.config.settings import PopulationConfig, ScoringConfig
from raregeneai.models.data_models import (
    AnnotatedVariant,
    FunctionalImpact,
    Variant,
    Zygosity,
)
from raregeneai.population.population_annotator import (
    ME_FOUNDER_GENES,
    PopulationAnnotator,
)
from raregeneai.scoring.variant_scorer import VariantScorer


def _v(gnomad_af=None, local_af=None, clinvar="", gene="GENE_A",
       impact=FunctionalImpact.MODERATE, **kwargs):
    """Helper to create AnnotatedVariant with frequency data."""
    return AnnotatedVariant(
        variant=Variant(chrom="chr1", pos=100, ref="A", alt="G",
                       zygosity=Zygosity.HET, depth=30, gq=40),
        gene_symbol=gene,
        impact=impact,
        consequence="missense_variant",
        gnomad_af=gnomad_af,
        local_af=local_af,
        clinvar_significance=clinvar,
        **kwargs,
    )


# ── effective_af and is_rare Property Tests ───────────────────────────────────

class TestEffectiveAF:
    def test_novel_both_databases(self):
        var = _v(gnomad_af=None, local_af=None)
        assert var.effective_af == 0.0
        assert var.is_rare
        assert var.is_novel

    def test_gnomad_only(self):
        var = _v(gnomad_af=0.005, local_af=None)
        assert var.effective_af == 0.005
        assert var.is_rare

    def test_local_only(self):
        var = _v(gnomad_af=None, local_af=0.003)
        assert var.effective_af == 0.003
        assert var.is_rare

    def test_uses_max_of_local_and_global(self):
        """Population-adjusted AF = max(local, global)."""
        var = _v(gnomad_af=0.001, local_af=0.05)
        assert var.effective_af == 0.05  # Local is higher
        assert not var.is_rare  # 5% = not rare

    def test_globally_rare_locally_common_not_rare(self):
        """Variant rare in gnomAD (0.0001) but common in QGP (2%)
        should NOT be considered rare for a Qatari patient."""
        var = _v(gnomad_af=0.0001, local_af=0.02)
        assert not var.is_rare

    def test_locally_rare_globally_common_not_rare(self):
        """Variant common globally should still be filtered even if
        absent from local database."""
        var = _v(gnomad_af=0.05, local_af=0.001)
        assert not var.is_rare

    def test_novel_requires_both_absent(self):
        var1 = _v(gnomad_af=None, local_af=None)
        assert var1.is_novel

        var2 = _v(gnomad_af=None, local_af=0.001)
        assert not var2.is_novel

        var3 = _v(gnomad_af=0.001, local_af=None)
        assert not var3.is_novel


# ── Population Annotator Tests ────────────────────────────────────────────────

class TestPopulationAnnotator:
    def test_disabled_returns_unchanged(self):
        cfg = PopulationConfig(enabled=False)
        annotator = PopulationAnnotator(cfg)
        var = _v()
        result = annotator.annotate([var])
        assert result[0].local_af is None

    def test_load_tsv_database(self, tmp_path):
        """Load a local frequency TSV and annotate variants."""
        db_file = tmp_path / "qgp_af.tsv"
        db_file.write_text(
            "#chrom\tpos\tref\talt\taf\tac\tan\thom\n"
            "chr1\t100\tA\tG\t0.03\t60\t2000\t2\n"
            "chr1\t200\tC\tT\t0.001\t2\t2000\t0\n"
        )

        cfg = PopulationConfig(
            enabled=True,
            qgp_af_path=str(db_file),
            population="QGP",
        )
        annotator = PopulationAnnotator(cfg)

        var = _v(gnomad_af=0.0001)  # Rare globally
        annotator.annotate([var])

        assert var.local_af == 0.03
        assert var.local_ac == 60
        assert var.local_an == 2000
        assert var.local_hom_count == 2
        assert var.local_population == "QGP"

    def test_enrichment_ratio_computed(self, tmp_path):
        db_file = tmp_path / "local_af.tsv"
        db_file.write_text("chr1\t100\tA\tG\t0.02\t40\t2000\t1\n")

        cfg = PopulationConfig(enabled=True, local_af_path=str(db_file))
        annotator = PopulationAnnotator(cfg)

        var = _v(gnomad_af=0.0002)  # 0.02% globally
        annotator.annotate([var])

        # Enrichment = 0.02 / 0.0002 = 100x
        assert var.population_af_ratio is not None
        assert var.population_af_ratio == pytest.approx(100.0, rel=0.01)

    def test_variant_not_in_local_db(self, tmp_path):
        db_file = tmp_path / "local_af.tsv"
        db_file.write_text("chr2\t999\tT\tC\t0.05\t100\t2000\t5\n")

        cfg = PopulationConfig(enabled=True, local_af_path=str(db_file), population="QGP")
        annotator = PopulationAnnotator(cfg)

        var = _v()  # chr1-100, not in DB
        annotator.annotate([var])
        assert var.local_af is None
        assert var.local_population == "QGP"  # Population label still set


# ── Founder Variant Detection Tests ───────────────────────────────────────────

class TestFounderDetection:
    def test_founder_by_enrichment(self, tmp_path):
        """Variant enriched 10x+ in local pop vs gnomAD = founder."""
        db_file = tmp_path / "qgp.tsv"
        db_file.write_text("chr1\t100\tA\tG\t0.005\t10\t2000\t0\n")

        cfg = PopulationConfig(
            enabled=True,
            qgp_af_path=str(db_file),
            population="QGP",
            founder_enrichment_threshold=10.0,
            founder_local_af_min=0.001,
            founder_global_af_max=0.001,
        )
        annotator = PopulationAnnotator(cfg)

        var = _v(gnomad_af=0.0001)  # 0.01% globally, 0.5% locally = 50x
        annotator.annotate([var])

        assert var.is_founder_variant
        assert var.founder_enrichment >= 10.0

    def test_not_founder_if_common_globally(self, tmp_path):
        """Variant common globally is NOT a founder (just common everywhere)."""
        db_file = tmp_path / "qgp.tsv"
        db_file.write_text("chr1\t100\tA\tG\t0.05\t100\t2000\t5\n")

        cfg = PopulationConfig(
            enabled=True,
            qgp_af_path=str(db_file),
            founder_global_af_max=0.001,
        )
        annotator = PopulationAnnotator(cfg)

        var = _v(gnomad_af=0.03)  # 3% globally
        annotator.annotate([var])

        assert not var.is_founder_variant

    def test_not_founder_if_rare_locally(self, tmp_path):
        """Variant rare in local pop is NOT a founder (just rare)."""
        db_file = tmp_path / "qgp.tsv"
        db_file.write_text("chr1\t100\tA\tG\t0.0001\t1\t10000\t0\n")

        cfg = PopulationConfig(
            enabled=True,
            qgp_af_path=str(db_file),
            founder_local_af_min=0.001,
        )
        annotator = PopulationAnnotator(cfg)

        var = _v(gnomad_af=0.00005)
        annotator.annotate([var])

        assert not var.is_founder_variant

    def test_founder_by_known_me_gene(self, tmp_path):
        """Known ME founder gene (MEFV) with local enrichment = founder."""
        db_file = tmp_path / "qgp.tsv"
        db_file.write_text("chr1\t100\tA\tG\t0.002\t4\t2000\t0\n")

        cfg = PopulationConfig(
            enabled=True,
            qgp_af_path=str(db_file),
            population="QGP",
        )
        annotator = PopulationAnnotator(cfg)

        var = _v(gnomad_af=0.0005, gene="MEFV")
        annotator.annotate([var])

        assert var.is_founder_variant

    def test_me_founder_genes_list(self):
        """Known Middle Eastern founder genes should be populated."""
        assert "MEFV" in ME_FOUNDER_GENES
        assert "GJB2" in ME_FOUNDER_GENES
        assert "HBB" in ME_FOUNDER_GENES
        assert len(ME_FOUNDER_GENES) >= 15

    def test_known_founders_database(self, tmp_path):
        db_file = tmp_path / "qgp.tsv"
        db_file.write_text("chr1\t100\tA\tG\t0.001\t2\t2000\t0\n")

        founders_file = tmp_path / "known_founders.tsv"
        founders_file.write_text("chr1\t100\tA\tG\tGENE_A\tDisease_X\n")

        cfg = PopulationConfig(
            enabled=True,
            qgp_af_path=str(db_file),
            known_founders_path=str(founders_file),
        )
        annotator = PopulationAnnotator(cfg)

        var = _v(gnomad_af=0.01)  # Not enriched enough by criteria
        annotator.annotate([var])

        # But it's in the known founders DB
        assert var.is_founder_variant


# ── VariantScorer Population Integration ──────────────────────────────────────

class TestVariantScorerPopulation:
    @pytest.fixture
    def scorer(self):
        return VariantScorer()

    def test_rarity_novel_max(self, scorer):
        var = _v(gnomad_af=None, local_af=None)
        assert scorer._rarity_score(var) == 1.0

    def test_rarity_common_zero(self, scorer):
        var = _v(gnomad_af=0.05, local_af=0.05)
        assert scorer._rarity_score(var) == 0.0

    def test_rarity_globally_rare_locally_common_penalized(self, scorer):
        """Globally rare (0.01%) but locally common (5%) → NOT rare."""
        var = _v(gnomad_af=0.0001, local_af=0.05)
        score = scorer._rarity_score(var)
        assert score == 0.0  # effective_af = 0.05 > threshold

    def test_founder_pathogenic_keeps_rarity(self, scorer):
        """Pathogenic founder variant should maintain higher rarity."""
        var = _v(gnomad_af=0.0001, local_af=0.005, clinvar="Pathogenic")
        var.is_founder_variant = True
        score = scorer._rarity_score(var)
        assert score > 0.0  # Should still have some rarity

    def test_founder_non_pathogenic_penalized(self, scorer):
        """Non-pathogenic founder gets stronger decay (penalized)."""
        var_founder = _v(gnomad_af=0.0001, local_af=0.005)
        var_founder.is_founder_variant = True

        var_regular = _v(gnomad_af=0.005, local_af=None)

        founder_score = scorer._rarity_score(var_founder)
        regular_score = scorer._rarity_score(var_regular)

        # Founder penalized more than regular variant at same effective AF
        assert founder_score < regular_score

    def test_population_adjusted_composite(self, scorer):
        """Composite score should differ when local AF is considered."""
        # Same variant, different population context
        var_no_pop = _v(gnomad_af=0.0001, cadd_phred=25.0)
        var_with_pop = _v(gnomad_af=0.0001, local_af=0.03, cadd_phred=25.0)

        scorer.score_variants([var_no_pop])
        scorer.score_variants([var_with_pop])

        # Variant common locally should have lower composite
        assert var_with_pop.composite_score < var_no_pop.composite_score


# ── Explainer Population Tests ────────────────────────────────────────────────

class TestExplainerPopulation:
    def test_explain_founder_variant(self):
        from raregeneai.explainability.explainer import Explainer

        explainer = Explainer()
        var = _v(
            gnomad_af=0.0001,
            local_af=0.005,
            local_population="QGP",
        )
        var.is_founder_variant = True
        var.founder_enrichment = 50.0

        text = explainer._explain_variant(var)

        assert "QGP" in text
        assert "FOUNDER" in text
        assert "50x" in text

    def test_explain_local_af(self):
        from raregeneai.explainability.explainer import Explainer

        explainer = Explainer()
        var = _v(gnomad_af=0.001, local_af=0.01, local_population="GME")

        text = explainer._explain_variant(var)
        assert "GME" in text
        assert "gnomAD" in text

    def test_explain_novel_with_population(self):
        from raregeneai.explainability.explainer import Explainer

        explainer = Explainer()
        var = _v(gnomad_af=None, local_af=None)
        text = explainer._explain_variant(var)
        assert "novel" in text.lower()
        assert "local" in text.lower()
