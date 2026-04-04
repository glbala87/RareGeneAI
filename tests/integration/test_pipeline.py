"""Integration test for the full RareGeneAI pipeline."""

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def test_vcf(tmp_path):
    """Create a minimal test VCF."""
    vcf_content = """##fileformat=VCFv4.2
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype quality">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tPROBAND
chr2\t166049061\t.\tG\tA\t99\tPASS\t.\tGT:DP:GQ:AD\t0/1:50:99:25,25
chr17\t43094464\t.\tG\tA\t99\tPASS\t.\tGT:DP:GQ:AD\t0/1:40:95:20,20
chr1\t1000000\t.\tA\tG\t30\tPASS\t.\tGT:DP:GQ:AD\t0/1:30:30:15,15
"""
    vcf_path = tmp_path / "test.vcf"
    vcf_path.write_text(vcf_content)
    return str(vcf_path)


class TestPipelineIntegration:
    def test_vcf_parsing(self, test_vcf):
        """Test VCF ingestion produces variants."""
        from raregeneai.ingestion.vcf_parser import VCFParser

        parser = VCFParser()
        variants = parser.parse(test_vcf)
        assert len(variants) >= 1  # At least some pass QC

    def test_hpo_parsing(self):
        """Test HPO term validation."""
        from raregeneai.ingestion.hpo_parser import HPOParser

        parser = HPOParser()
        phenotype = parser.parse_phenotype(
            patient_id="TEST_001",
            hpo_terms=["HP:0001250", "HP:0002878", "HP:0001263"],
        )
        assert phenotype.patient_id == "TEST_001"
        assert len(phenotype.hpo_terms) >= 1

    def test_variant_scoring(self):
        """Test variant scoring produces bounded scores."""
        from raregeneai.models.data_models import AnnotatedVariant, FunctionalImpact, Variant
        from raregeneai.scoring.variant_scorer import VariantScorer

        scorer = VariantScorer()
        var = AnnotatedVariant(
            variant=Variant(chrom="1", pos=100, ref="A", alt="G"),
            cadd_phred=25.0,
            gnomad_af=0.0001,
            impact=FunctionalImpact.MODERATE,
            consequence="missense_variant",
        )
        scored = scorer.score_variants([var])
        assert 0.0 <= scored[0].composite_score <= 1.0
        assert scored[0].pathogenicity_score > 0

    def test_gene_ranking(self):
        """Test gene ranking produces sorted candidates."""
        from raregeneai.models.data_models import (
            AnnotatedVariant, FunctionalImpact, Variant, PatientPhenotype, HPOTerm,
        )
        from raregeneai.ranking.gene_ranker import GeneRanker

        ranker = GeneRanker()

        vars1 = [AnnotatedVariant(
            variant=Variant(chrom="1", pos=100, ref="A", alt="G"),
            gene_symbol="GENE_A", gene_id="ENSG1",
            cadd_phred=30.0, impact=FunctionalImpact.HIGH,
            composite_score=0.8,
        )]
        vars2 = [AnnotatedVariant(
            variant=Variant(chrom="1", pos=200, ref="C", alt="T"),
            gene_symbol="GENE_B", gene_id="ENSG2",
            cadd_phred=15.0, impact=FunctionalImpact.LOW,
            composite_score=0.3,
        )]

        phenotype_scores = {"GENE_A": 0.8, "GENE_B": 0.2}

        ranked = ranker.rank(vars1 + vars2, phenotype_scores)
        assert len(ranked) == 2
        assert ranked[0].gene_symbol == "GENE_A"
        assert ranked[0].gene_rank_score >= ranked[1].gene_rank_score

    def test_explainer(self):
        """Test explanation generation."""
        from raregeneai.explainability.explainer import Explainer
        from raregeneai.models.data_models import (
            AnnotatedVariant, FunctionalImpact, GeneCandidate, Variant,
        )

        explainer = Explainer()
        var = AnnotatedVariant(
            variant=Variant(chrom="1", pos=100, ref="A", alt="G"),
            cadd_phred=28.0,
            consequence="missense_variant",
            impact=FunctionalImpact.MODERATE,
        )
        candidate = GeneCandidate(
            gene_symbol="TEST_GENE",
            variants=[var],
            gene_rank_score=0.75,
            confidence=0.85,
        )
        explanation = explainer.explain_gene(candidate)
        assert "TEST_GENE" in explanation
        assert len(explanation) > 50
