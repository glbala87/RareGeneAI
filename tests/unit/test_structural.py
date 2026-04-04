"""Unit tests for structural variant prioritization.

Tests:
  - SV VCF parsing (Sniffles/Jasmine format)
  - SV annotation (gene overlap, dosage, regulatory)
  - SV scoring (composite score computation)
  - SV-to-SNV integration bridge
  - Unified gene ranking with SVs
  - SV explanation generation
"""

import tempfile
from pathlib import Path

import pytest

from raregeneai.config.settings import SVConfig
from raregeneai.models.data_models import (
    AnnotatedSV,
    AnnotatedVariant,
    FunctionalImpact,
    GeneCandidate,
    StructuralVariant,
    SVType,
    Variant,
    Zygosity,
)
from raregeneai.structural.sv_annotator import SVAnnotator
from raregeneai.structural.sv_integration import (
    enrich_candidates_with_sv,
    sv_to_annotated_variants,
    _sv_impact,
)
from raregeneai.structural.sv_parser import SVParser


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_sv(
    chrom="chr1", pos=1000000, end=1100000, sv_type=SVType.DEL,
    sv_len=100000, quality=50.0, zygosity=Zygosity.HET,
    support_reads=15, ref_reads=25, **kwargs,
) -> StructuralVariant:
    return StructuralVariant(
        chrom=chrom, pos=pos, end=end, sv_type=sv_type,
        sv_len=sv_len, quality=quality, zygosity=zygosity,
        support_reads=support_reads, ref_reads=ref_reads,
        allele_fraction=support_reads / (support_reads + ref_reads),
        filter_status="PASS", caller="sniffles", **kwargs,
    )


def _make_annotated_sv(
    sv=None,
    overlapping_genes=None,
    fully_deleted_genes=None,
    pli=None, hi=None, ts=None, loeuf=None,
    gnomad_sv_af=None, clinvar="",
    disrupted_enhancers=0, disrupted_tads=None,
    **kwargs,
) -> AnnotatedSV:
    if sv is None:
        sv = _make_sv()
    return AnnotatedSV(
        sv=sv,
        overlapping_genes=overlapping_genes or [],
        fully_deleted_genes=fully_deleted_genes or [],
        pli_score=pli,
        hi_score=hi,
        ts_score=ts,
        loeuf_score=loeuf,
        gnomad_sv_af=gnomad_sv_af,
        clinvar_sv_significance=clinvar,
        disrupted_enhancers=disrupted_enhancers,
        disrupted_tads=disrupted_tads or [],
        **kwargs,
    )


def _make_test_sv_vcf(tmp_path: Path) -> Path:
    """Create a minimal SV VCF file for testing."""
    content = """##fileformat=VCFv4.2
##source=Sniffles2
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of SV">
##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="Length of SV">
##INFO=<ID=END,Number=1,Type=Integer,Description="End position">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DV,Number=1,Type=Integer,Description="Alt reads">
##FORMAT=<ID=DR,Number=1,Type=Integer,Description="Ref reads">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE
chr1\t1000000\tSV001\tN\t<DEL>\t50\tPASS\tSVTYPE=DEL;SVLEN=-50000;END=1050000;PRECISE\tGT:DV:DR\t0/1:15:25
chr2\t5000000\tSV002\tN\t<DUP>\t40\tPASS\tSVTYPE=DUP;SVLEN=200000;END=5200000;IMPRECISE\tGT:DV:DR\t0/1:10:30
chr3\t10000000\tSV003\tN\t<INV>\t60\tPASS\tSVTYPE=INV;SVLEN=80000;END=10080000;PRECISE\tGT:DV:DR\t1/1:40:2
chrX\t50000000\tSV004\tN\t<DEL>\t30\tPASS\tSVTYPE=DEL;SVLEN=-5000;END=50005000;PRECISE\tGT:DV:DR\t0/1:8:20
chr5\t1000\tSV005\tN\t<DEL>\t3\tLowQual\tSVTYPE=DEL;SVLEN=-100;END=1100\tGT:DV:DR\t0/1:1:30
"""
    vcf_path = tmp_path / "test_svs.vcf"
    vcf_path.write_text(content)
    return vcf_path


# ── Data Model Tests ──────────────────────────────────────────────────────────

class TestSVModels:
    def test_sv_variant_key(self):
        sv = _make_sv()
        assert sv.variant_key == "chr1-1000000-DEL-100000"

    def test_sv_is_deletion(self):
        sv = _make_sv(sv_type=SVType.DEL)
        assert sv.is_deletion
        assert not sv.is_duplication

    def test_sv_is_duplication(self):
        sv = _make_sv(sv_type=SVType.DUP)
        assert sv.is_duplication
        assert not sv.is_deletion

    def test_sv_size_category(self):
        small = _make_sv(sv_len=500)
        medium = _make_sv(sv_len=50000)
        large = _make_sv(sv_len=500000)
        very_large = _make_sv(sv_len=5000000)

        assert small.size_category == "small"
        assert medium.size_category == "medium"
        assert large.size_category == "large"
        assert very_large.size_category == "very_large"

    def test_sv_is_large(self):
        assert not _make_sv(sv_len=500000).is_large
        assert _make_sv(sv_len=1500000).is_large

    def test_annotated_sv_is_rare(self):
        ann = _make_annotated_sv(gnomad_sv_af=None)
        assert ann.is_rare
        assert ann.is_novel

    def test_annotated_sv_not_rare(self):
        ann = _make_annotated_sv(gnomad_sv_af=0.05)
        assert not ann.is_rare

    def test_annotated_sv_n_genes(self):
        ann = _make_annotated_sv(overlapping_genes=["A", "B", "C"])
        assert ann.n_genes_affected == 3

    def test_annotated_sv_dosage_sensitive(self):
        ann = _make_annotated_sv(pli=0.99)
        assert ann.has_dosage_sensitive_gene

    def test_annotated_sv_not_dosage_sensitive(self):
        ann = _make_annotated_sv(pli=0.1)
        assert not ann.has_dosage_sensitive_gene

    def test_gene_candidate_has_sv(self):
        ann = _make_annotated_sv(overlapping_genes=["GENE_A"])
        gc = GeneCandidate(gene_symbol="GENE_A", has_sv=True, structural_variants=[ann])
        assert gc.has_sv


# ── SV Parser Tests ───────────────────────────────────────────────────────────

class TestSVParser:
    @pytest.fixture
    def parser(self):
        return SVParser()

    def test_parse_sv_vcf(self, parser, tmp_path):
        vcf_path = _make_test_sv_vcf(tmp_path)
        svs = parser.parse(vcf_path)

        # SV005 filtered (LowQual + only 1 support read)
        assert len(svs) >= 3

    def test_parse_svtype(self, parser, tmp_path):
        vcf_path = _make_test_sv_vcf(tmp_path)
        svs = parser.parse(vcf_path)

        types = {sv.sv_type for sv in svs}
        assert SVType.DEL in types

    def test_parse_genotype(self, parser, tmp_path):
        vcf_path = _make_test_sv_vcf(tmp_path)
        svs = parser.parse(vcf_path)

        het_svs = [sv for sv in svs if sv.zygosity == Zygosity.HET]
        assert len(het_svs) >= 2

    def test_parse_read_support(self, parser, tmp_path):
        vcf_path = _make_test_sv_vcf(tmp_path)
        svs = parser.parse(vcf_path)

        # First SV should have DV=15
        del_svs = [sv for sv in svs if sv.sv_type == SVType.DEL]
        assert any(sv.support_reads == 15 for sv in del_svs)

    def test_filter_low_quality(self, parser, tmp_path):
        vcf_path = _make_test_sv_vcf(tmp_path)
        svs = parser.parse(vcf_path)

        # LowQual SV should be filtered
        ids = [sv.sv_id for sv in svs]
        assert "SV005" not in ids

    def test_parse_sv_length(self, parser, tmp_path):
        vcf_path = _make_test_sv_vcf(tmp_path)
        svs = parser.parse(vcf_path)

        del_svs = [sv for sv in svs if sv.sv_id == "SV001"]
        if del_svs:
            assert del_svs[0].sv_len == 50000

    def test_parse_empty_vcf(self, parser, tmp_path):
        vcf_path = tmp_path / "empty.vcf"
        vcf_path.write_text(
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n"
        )
        svs = parser.parse(vcf_path)
        assert len(svs) == 0

    def test_parse_bnd_alt(self):
        chrom, pos = SVParser._parse_bnd_alt("]chr2:321681]T")
        assert chrom == "chr2"
        assert pos == 321681

    def test_parse_zygosity(self):
        assert SVParser._parse_zygosity("0/1") == Zygosity.HET
        assert SVParser._parse_zygosity("1/1") == Zygosity.HOM_ALT
        assert SVParser._parse_zygosity("0/0") == Zygosity.HOM_REF
        assert SVParser._parse_zygosity("./.") == Zygosity.UNKNOWN


# ── SV Annotator Tests ────────────────────────────────────────────────────────

class TestSVAnnotator:
    @pytest.fixture
    def annotator(self):
        return SVAnnotator(SVConfig(enabled=True))

    def test_compute_gene_overlap_score_full_deletion(self, annotator):
        ann = _make_annotated_sv(
            fully_deleted_genes=["GENE_A"],
            overlapping_genes=["GENE_A"],
        )
        annotator._compute_sv_scores(ann)
        assert ann.gene_overlap_score >= 0.5

    def test_compute_gene_overlap_score_partial(self, annotator):
        ann = _make_annotated_sv(
            overlapping_genes=["GENE_A"],
            partially_overlapping_genes=["GENE_A"],
        )
        annotator._compute_sv_scores(ann)
        assert 0.3 <= ann.gene_overlap_score <= 0.8

    def test_compute_dosage_score_pli_high(self, annotator):
        ann = _make_annotated_sv(
            sv=_make_sv(sv_type=SVType.DEL),
            overlapping_genes=["GENE_A"],
            pli=0.99,
        )
        annotator._compute_sv_scores(ann)
        assert ann.dosage_sensitivity_score > 0.9

    def test_compute_dosage_score_dup_ts(self, annotator):
        ann = _make_annotated_sv(
            sv=_make_sv(sv_type=SVType.DUP),
            overlapping_genes=["GENE_A"],
            ts=3.0,
        )
        annotator._compute_sv_scores(ann)
        assert ann.dosage_sensitivity_score > 0.8

    def test_compute_rarity_novel(self, annotator):
        ann = _make_annotated_sv(gnomad_sv_af=None)
        annotator._compute_sv_scores(ann)
        assert ann.sv_rarity_score == 1.0

    def test_compute_rarity_common(self, annotator):
        ann = _make_annotated_sv(gnomad_sv_af=0.05)
        annotator._compute_sv_scores(ann)
        assert ann.sv_rarity_score == 0.0

    def test_compute_clinical_score_pathogenic(self, annotator):
        ann = _make_annotated_sv(clinvar="Pathogenic")
        annotator._compute_sv_scores(ann)
        assert ann.sv_composite_score > 0.0

    def test_compute_composite_all_evidence(self, annotator):
        ann = _make_annotated_sv(
            sv=_make_sv(sv_type=SVType.DEL),
            overlapping_genes=["GENE_A"],
            fully_deleted_genes=["GENE_A"],
            pli=0.99,
            gnomad_sv_af=None,
            clinvar="Pathogenic",
        )
        annotator._compute_sv_scores(ann)

        # High overlap + high dosage + novel + pathogenic = high score
        assert ann.sv_composite_score > 0.6

    def test_regulatory_disruption_estimate(self, annotator):
        ann = _make_annotated_sv(sv=_make_sv(sv_len=200000))
        annotator._estimate_regulatory_disruption(ann)
        assert ann.disrupted_enhancers > 0

    def test_annotate_batch(self, annotator):
        svs = [
            _make_sv(sv_type=SVType.DEL, sv_len=50000),
            _make_sv(sv_type=SVType.DUP, sv_len=100000, pos=5000000, end=5100000),
        ]
        annotated = annotator.annotate(svs)
        assert len(annotated) == 2


# ── SV Integration Tests ─────────────────────────────────────────────────────

class TestSVIntegration:
    def test_sv_to_annotated_variants(self):
        ann = _make_annotated_sv(
            overlapping_genes=["GENE_A", "GENE_B"],
            fully_deleted_genes=["GENE_A"],
        )
        ann.sv_composite_score = 0.7
        ann.sv_pathogenicity_score = 0.8
        ann.gene_overlap_score = 0.9
        ann.sv_rarity_score = 1.0

        variants = sv_to_annotated_variants([ann])
        assert len(variants) == 2  # One per gene

        gene_a_vars = [v for v in variants if v.gene_symbol == "GENE_A"]
        assert len(gene_a_vars) == 1
        assert gene_a_vars[0].impact == FunctionalImpact.HIGH
        assert gene_a_vars[0].composite_score == 0.7
        assert gene_a_vars[0].variant.alt == "<DEL>"

    def test_sv_impact_full_deletion(self):
        sv = _make_sv(sv_type=SVType.DEL)
        assert _sv_impact(sv, fully_deleted=True) == FunctionalImpact.HIGH

    def test_sv_impact_partial_dup(self):
        sv = _make_sv(sv_type=SVType.DUP)
        assert _sv_impact(sv, fully_deleted=False) == FunctionalImpact.MODERATE

    def test_sv_impact_translocation(self):
        sv = _make_sv(sv_type=SVType.BND)
        assert _sv_impact(sv, fully_deleted=False) == FunctionalImpact.HIGH

    def test_enrich_candidates_with_sv(self):
        ann = _make_annotated_sv(
            overlapping_genes=["GENE_A"],
            fully_deleted_genes=["GENE_A"],
            pli=0.98,
        )
        ann.sv_composite_score = 0.8
        ann.gene_overlap_score = 0.9
        ann.dosage_sensitivity_score = 0.95
        ann.regulatory_disruption_score = 0.3

        candidate = GeneCandidate(
            gene_symbol="GENE_A",
            gene_rank_score=0.5,
            confidence=0.7,
            evidence_summary={},
        )

        enriched = enrich_candidates_with_sv([candidate], [ann])
        assert enriched[0].has_sv
        assert enriched[0].sv_fully_deleted
        assert enriched[0].sv_dosage_sensitive
        assert enriched[0].max_sv_score == 0.8
        assert enriched[0].evidence_summary["max_sv_dosage_score"] == 0.95

    def test_sv_snv_unified_ranking(self):
        """Test that SVs and SNVs rank together correctly."""
        from raregeneai.ranking.gene_ranker import GeneRanker

        ranker = GeneRanker()

        # Gene A: SNV with high CADD
        snv = AnnotatedVariant(
            variant=Variant(chrom="chr1", pos=100, ref="A", alt="G", zygosity=Zygosity.HET),
            gene_symbol="GENE_A", gene_id="ENSG1",
            consequence="missense_variant", impact=FunctionalImpact.MODERATE,
            composite_score=0.5,
        )

        # Gene B: whole-gene deletion (converted from SV)
        sv_var = AnnotatedVariant(
            variant=Variant(chrom="chr2", pos=1000000, ref="N", alt="<DEL>", zygosity=Zygosity.HET),
            gene_symbol="GENE_B", gene_id="ENSG2",
            consequence="whole_gene_deletion", impact=FunctionalImpact.HIGH,
            composite_score=0.85,
        )

        phenotype_scores = {"GENE_A": 0.6, "GENE_B": 0.8}
        ranked = ranker.rank([snv, sv_var], phenotype_scores)

        assert len(ranked) == 2
        # Gene B (SV) should rank higher due to higher composite + HIGH impact
        assert ranked[0].gene_symbol == "GENE_B"


# ── Explainer Tests (SV) ─────────────────────────────────────────────────────

class TestExplainerSV:
    def test_explain_sv_variant(self):
        from raregeneai.explainability.explainer import Explainer

        explainer = Explainer()
        var = AnnotatedVariant(
            variant=Variant(chrom="chr1", pos=1000000, ref="N", alt="<DEL>"),
            gene_symbol="TEST_GENE",
            consequence="whole_gene_deletion",
            impact=FunctionalImpact.HIGH,
            composite_score=0.8,
        )
        text = explainer._explain_variant(var)
        assert "SV" in text
        assert "<DEL>" in text
        assert "WHOLE GENE DISRUPTION" in text

    def test_explain_gene_with_sv(self):
        from raregeneai.explainability.explainer import Explainer

        explainer = Explainer()
        ann = _make_annotated_sv(
            sv=_make_sv(sv_type=SVType.DEL, sv_len=150000),
            overlapping_genes=["TEST_GENE"],
            fully_deleted_genes=["TEST_GENE"],
            pli=0.99,
        )
        ann.sv_composite_score = 0.85

        candidate = GeneCandidate(
            gene_symbol="TEST_GENE",
            gene_rank_score=0.80,
            confidence=0.90,
            has_sv=True,
            sv_fully_deleted=True,
            structural_variants=[ann],
        )
        text = explainer.explain_gene(candidate)
        assert "WHOLE GENE DELETION" in text
        assert "dosage-sensitive" in text
        assert "DEL" in text
