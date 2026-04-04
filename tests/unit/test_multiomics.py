"""Unit tests for multi-omics integration module.

Tests:
  - Expression outlier detection
  - Methylation DMR analysis
  - Multi-omics evidence integration
  - Concordance assessment
  - Multi-omics scoring
  - Gene candidate enrichment
  - Explainer multi-omics output
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raregeneai.config.settings import MultiOmicsConfig
from raregeneai.models.data_models import (
    GeneCandidate,
    GeneExpression,
    MethylationRegion,
    MultiOmicsEvidence,
)
from raregeneai.multiomics.expression_outlier import ExpressionOutlierDetector
from raregeneai.multiomics.integrator import MultiOmicsIntegrator
from raregeneai.multiomics.methylation_analyzer import MethylationAnalyzer


# ── Data Model Tests ──────────────────────────────────────────────────────────

class TestMultiOmicsModels:
    def test_gene_expression_underexpressed(self):
        ge = GeneExpression(
            gene_symbol="TP53", z_score=-3.5,
            is_outlier=True, outlier_direction="under",
        )
        assert ge.is_underexpressed
        assert not ge.is_overexpressed

    def test_gene_expression_overexpressed(self):
        ge = GeneExpression(
            gene_symbol="MYC", z_score=4.0,
            is_outlier=True, outlier_direction="over",
        )
        assert ge.is_overexpressed
        assert not ge.is_underexpressed

    def test_gene_expression_normal(self):
        ge = GeneExpression(gene_symbol="GAPDH", z_score=0.5)
        assert not ge.is_outlier
        assert not ge.is_underexpressed

    def test_methylation_region_hyper(self):
        mr = MethylationRegion(
            gene_symbol="BRCA1", delta_beta=0.35,
            is_dmr=True, dmr_direction="hyper",
            chrom="chr17", start=100, end=200,
        )
        assert mr.is_hypermethylated
        assert not mr.is_hypomethylated
        assert mr.length == 100

    def test_methylation_region_hypo(self):
        mr = MethylationRegion(
            gene_symbol="GENE_X", delta_beta=-0.3,
            is_dmr=True, dmr_direction="hypo",
        )
        assert mr.is_hypomethylated

    def test_multi_omics_evidence(self):
        moe = MultiOmicsEvidence(
            gene_symbol="TEST",
            n_evidence_layers=3,
            evidence_layers=["genomic", "expression", "methylation"],
            multi_omics_score=0.75,
            is_concordant=True,
        )
        assert moe.n_evidence_layers == 3
        assert moe.is_concordant

    def test_gene_candidate_multi_omics_fields(self):
        gc = GeneCandidate(
            gene_symbol="TEST",
            multi_omics_score=0.8,
            n_evidence_layers=3,
        )
        assert gc.multi_omics_score == 0.8
        assert gc.n_evidence_layers == 3


# ── Expression Outlier Tests ──────────────────────────────────────────────────

class TestExpressionOutlier:
    @pytest.fixture
    def detector(self):
        return ExpressionOutlierDetector(MultiOmicsConfig())

    def test_no_expression_file_returns_empty(self, detector):
        result = detector.detect_outliers(expression_path="/nonexistent/file.tsv")
        assert result == {}

    def test_detect_outliers_with_reference(self, tmp_path):
        # Create patient expression (TPM values)
        patient_file = tmp_path / "patient_expr.tsv"
        patient_file.write_text("gene\ttpm\nGENE_A\t0.1\nGENE_B\t500.0\nGENE_C\t10.0\n")

        # Reference stats are in log2(TPM+1) space to match the Z-score computation
        # log2(0.1+1)=0.14, log2(500+1)=8.97, log2(10+1)=3.46
        # GENE_A ref median=3.0, so patient log2=0.14, Z=(0.14-3.0)/0.5=-5.7 -> under
        # GENE_B ref median=3.5, so patient log2=8.97, Z=(8.97-3.5)/0.5=10.9 -> over
        ref_file = tmp_path / "ref_stats.tsv"
        ref_file.write_text("gene\tmedian\tmad\nGENE_A\t3.0\t0.5\nGENE_B\t3.5\t0.5\nGENE_C\t3.2\t0.8\n")

        cfg = MultiOmicsConfig(z_score_threshold=2.0)
        detector = ExpressionOutlierDetector(cfg)

        results = detector.detect_outliers(
            expression_path=str(patient_file),
            reference_path=str(ref_file),
        )

        assert "GENE_A" in results
        assert "GENE_B" in results

        # GENE_A: log2(0.1+1)=0.14 vs median 3.0 -> strongly underexpressed
        gene_a = results["GENE_A"]
        assert gene_a.z_score < -2.0
        assert gene_a.is_outlier
        assert gene_a.outlier_direction == "under"

        # GENE_B: log2(500+1)=8.97 vs median 3.5 -> strongly overexpressed
        gene_b = results["GENE_B"]
        assert gene_b.z_score > 2.0
        assert gene_b.is_outlier
        assert gene_b.outlier_direction == "over"

    def test_detect_outliers_self_reference(self, tmp_path):
        """When no reference provided, use patient's own distribution."""
        patient_file = tmp_path / "patient.tsv"
        # Most genes around 10 TPM, one outlier at 1000
        lines = ["gene\ttpm"]
        for i in range(50):
            lines.append(f"GENE_{i}\t{10 + i * 0.5}")
        lines.append("OUTLIER_GENE\t5000.0")
        patient_file.write_text("\n".join(lines) + "\n")

        detector = ExpressionOutlierDetector(MultiOmicsConfig())
        results = detector.detect_outliers(expression_path=str(patient_file))

        assert "OUTLIER_GENE" in results
        assert results["OUTLIER_GENE"].is_outlier
        assert results["OUTLIER_GENE"].outlier_direction == "over"

    def test_median_absolute_deviation(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        mad = ExpressionOutlierDetector._median_absolute_deviation(values)
        # MAD should be robust against the outlier 100.0
        assert 0.5 < mad < 5.0  # Much smaller than SD


# ── Methylation Analyzer Tests ────────────────────────────────────────────────

class TestMethylationAnalyzer:
    def test_load_precalled_dmrs(self, tmp_path):
        dmr_file = tmp_path / "dmrs.tsv"
        dmr_file.write_text(
            "chrom\tstart\tend\tgene\tdelta_beta\tpvalue\tn_cpgs\tregion_type\n"
            "chr1\t1000\t2000\tGENE_A\t0.35\t0.001\t10\tpromoter\n"
            "chr2\t5000\t6000\tGENE_B\t-0.25\t0.01\t8\tgene_body\n"
            "chr3\t9000\t10000\tGENE_C\t0.10\t0.5\t3\tpromoter\n"
        )

        cfg = MultiOmicsConfig(delta_beta_threshold=0.2, dmr_pvalue_threshold=0.05)
        analyzer = MethylationAnalyzer(cfg)
        result = analyzer.analyze(dmr_calls_path=str(dmr_file))

        # GENE_A: delta=0.35, p=0.001 -> significant DMR
        assert "GENE_A" in result
        gene_a_dmrs = result["GENE_A"]
        assert len(gene_a_dmrs) == 1
        assert gene_a_dmrs[0].is_dmr
        assert gene_a_dmrs[0].dmr_direction == "hyper"
        assert gene_a_dmrs[0].region_type == "promoter"

        # GENE_B: delta=-0.25, p=0.01 -> significant DMR (hypo)
        assert "GENE_B" in result
        assert result["GENE_B"][0].is_dmr
        assert result["GENE_B"][0].dmr_direction == "hypo"

        # GENE_C: delta=0.10 < threshold -> not a DMR
        assert "GENE_C" in result
        assert not result["GENE_C"][0].is_dmr

    def test_filter_by_candidate_genes(self, tmp_path):
        dmr_file = tmp_path / "dmrs.tsv"
        dmr_file.write_text(
            "chrom\tstart\tend\tgene\tdelta_beta\tpvalue\n"
            "chr1\t1000\t2000\tGENE_A\t0.35\t0.001\n"
            "chr2\t5000\t6000\tGENE_B\t-0.30\t0.01\n"
        )

        analyzer = MethylationAnalyzer(MultiOmicsConfig())
        result = analyzer.analyze(
            dmr_calls_path=str(dmr_file),
            candidate_genes=["GENE_A"],
        )

        assert "GENE_A" in result
        assert "GENE_B" not in result

    def test_no_data_returns_empty(self):
        analyzer = MethylationAnalyzer(MultiOmicsConfig())
        result = analyzer.analyze()
        assert result == {}


# ── Multi-omics Integrator Tests ──────────────────────────────────────────────

class TestMultiOmicsIntegrator:
    @pytest.fixture
    def integrator(self):
        return MultiOmicsIntegrator(MultiOmicsConfig())

    def test_integrate_gene_expression_outlier_only(self, integrator):
        expr = GeneExpression(
            gene_symbol="GENE_A", tpm=0.1, log2_tpm=0.14,
            z_score=-3.5, is_outlier=True, outlier_direction="under",
        )
        result = integrator._integrate_gene("GENE_A", expr, [])

        assert result.has_expression_outlier
        assert result.expression_score > 0.5
        assert result.n_evidence_layers == 2  # genomic + expression
        assert "expression" in result.evidence_layers

    def test_integrate_gene_dmr_only(self, integrator):
        dmrs = [MethylationRegion(
            gene_symbol="GENE_A", delta_beta=0.35,
            is_dmr=True, dmr_direction="hyper",
            region_type="promoter", n_cpgs=10,
        )]
        result = integrator._integrate_gene("GENE_A", None, dmrs)

        assert result.has_dmr
        assert result.has_promoter_dmr
        assert result.methylation_score > 0.3
        assert result.n_evidence_layers == 2  # genomic + methylation
        assert "methylation" in result.evidence_layers

    def test_integrate_gene_concordant(self, integrator):
        """Underexpression + promoter hypermethylation = concordant."""
        expr = GeneExpression(
            gene_symbol="GENE_A", z_score=-3.0,
            is_outlier=True, outlier_direction="under",
        )
        dmrs = [MethylationRegion(
            gene_symbol="GENE_A", delta_beta=0.40,
            is_dmr=True, dmr_direction="hyper",
            region_type="promoter", n_cpgs=15,
        )]
        result = integrator._integrate_gene("GENE_A", expr, dmrs)

        assert result.n_evidence_layers == 3  # genomic + expression + methylation
        assert result.is_concordant
        assert result.concordance_bonus > 0.2
        assert "silencing" in result.concordance_description.lower()
        # Multi-omics score should be boosted by concordance multiplier
        assert result.multi_omics_score > 0.4

    def test_integrate_gene_discordant(self, integrator):
        """Overexpression + no methylation = not concordant."""
        expr = GeneExpression(
            gene_symbol="GENE_A", z_score=3.0,
            is_outlier=True, outlier_direction="over",
        )
        result = integrator._integrate_gene("GENE_A", expr, [])

        assert result.has_expression_outlier
        assert not result.is_concordant

    def test_three_layer_convergence_high_score(self, integrator):
        """Gene with genomic + expression + methylation = high confidence."""
        expr = GeneExpression(
            gene_symbol="GENE_X", z_score=-4.0,
            is_outlier=True, outlier_direction="under",
        )
        dmrs = [MethylationRegion(
            gene_symbol="GENE_X", delta_beta=0.45,
            is_dmr=True, dmr_direction="hyper",
            region_type="promoter", n_cpgs=20,
        )]
        result = integrator._integrate_gene("GENE_X", expr, dmrs)

        assert result.n_evidence_layers == 3
        assert result.is_concordant
        assert result.multi_omics_score > 0.5

    def test_no_omics_data_low_score(self, integrator):
        """Gene with only genomic evidence = low multi-omics score."""
        result = integrator._integrate_gene("GENE_Y", None, [])

        assert result.n_evidence_layers == 1  # genomic only
        assert result.multi_omics_score == 0.0

    def test_enrich_candidates(self, integrator):
        candidates = [
            GeneCandidate(gene_symbol="GENE_A", gene_rank_score=0.5, evidence_summary={}),
            GeneCandidate(gene_symbol="GENE_B", gene_rank_score=0.4, evidence_summary={}),
        ]
        multi_omics = {
            "GENE_A": MultiOmicsEvidence(
                gene_symbol="GENE_A",
                multi_omics_score=0.8,
                n_evidence_layers=3,
                has_expression_outlier=True,
                expression_score=0.7,
                expression=GeneExpression(gene_symbol="GENE_A", z_score=-3.0),
            ),
        }

        enriched = integrator.enrich_candidates(candidates, multi_omics)

        assert enriched[0].multi_omics_score == 0.8
        assert enriched[0].n_evidence_layers == 3
        assert enriched[0].evidence_summary["has_expression_outlier"] is True
        assert enriched[0].evidence_summary["expression_score"] == 0.7

        # GENE_B has no multi-omics data
        assert enriched[1].multi_omics_score == 0.0


# ── Concordance Assessment Tests ──────────────────────────────────────────────

class TestConcordance:
    @pytest.fixture
    def integrator(self):
        return MultiOmicsIntegrator(MultiOmicsConfig())

    def test_under_plus_hyper_concordant(self, integrator):
        evidence = MultiOmicsEvidence(
            gene_symbol="TEST",
            expression=GeneExpression(
                gene_symbol="TEST", z_score=-3.0,
                is_outlier=True, outlier_direction="under",
            ),
            has_expression_outlier=True,
            methylation_regions=[MethylationRegion(
                gene_symbol="TEST", delta_beta=0.3,
                is_dmr=True, dmr_direction="hyper",
                region_type="promoter",
            )],
            has_dmr=True,
        )
        result = integrator._assess_concordance(evidence)
        assert result["is_concordant"]
        assert result["bonus"] > 0.2

    def test_over_plus_hypo_concordant(self, integrator):
        evidence = MultiOmicsEvidence(
            gene_symbol="TEST",
            expression=GeneExpression(
                gene_symbol="TEST", z_score=3.5,
                is_outlier=True, outlier_direction="over",
            ),
            has_expression_outlier=True,
            methylation_regions=[MethylationRegion(
                gene_symbol="TEST", delta_beta=-0.3,
                is_dmr=True, dmr_direction="hypo",
                region_type="promoter",
            )],
            has_dmr=True,
        )
        result = integrator._assess_concordance(evidence)
        assert result["is_concordant"]
        assert "activation" in result["description"].lower()

    def test_no_expression_not_concordant(self, integrator):
        evidence = MultiOmicsEvidence(gene_symbol="TEST")
        result = integrator._assess_concordance(evidence)
        assert not result["is_concordant"]


# ── Multi-omics Score Computation Tests ───────────────────────────────────────

class TestMultiOmicsScoring:
    @pytest.fixture
    def integrator(self):
        return MultiOmicsIntegrator(MultiOmicsConfig())

    def test_score_increases_with_layers(self, integrator):
        # 1 layer
        e1 = MultiOmicsEvidence(gene_symbol="G1", n_evidence_layers=1)
        s1 = integrator._compute_multi_omics_score(e1)

        # 2 layers with expression
        e2 = MultiOmicsEvidence(
            gene_symbol="G2", n_evidence_layers=2,
            expression_score=0.6, has_expression_outlier=True,
        )
        s2 = integrator._compute_multi_omics_score(e2)

        # 3 layers with expression + methylation
        e3 = MultiOmicsEvidence(
            gene_symbol="G3", n_evidence_layers=3,
            expression_score=0.6, methylation_score=0.5,
            has_expression_outlier=True, has_dmr=True,
        )
        s3 = integrator._compute_multi_omics_score(e3)

        assert s1 < s2 < s3

    def test_concordance_multiplier_applied(self, integrator):
        base = MultiOmicsEvidence(
            gene_symbol="G1", n_evidence_layers=3,
            expression_score=0.6, methylation_score=0.5,
        )
        conc = MultiOmicsEvidence(
            gene_symbol="G2", n_evidence_layers=3,
            expression_score=0.6, methylation_score=0.5,
            is_concordant=True, concordance_bonus=0.3,
        )

        s_base = integrator._compute_multi_omics_score(base)
        s_conc = integrator._compute_multi_omics_score(conc)

        assert s_conc > s_base

    def test_score_bounded_at_one(self, integrator):
        extreme = MultiOmicsEvidence(
            gene_symbol="G1", n_evidence_layers=4,
            expression_score=1.0, methylation_score=1.0,
            is_concordant=True, concordance_bonus=0.5,
        )
        score = integrator._compute_multi_omics_score(extreme)
        assert score <= 1.0


# ── Explainer Multi-omics Tests ──────────────────────────────────────────────

class TestExplainerMultiOmics:
    def test_explain_gene_with_multi_omics(self):
        from raregeneai.explainability.explainer import Explainer

        explainer = Explainer()
        candidate = GeneCandidate(
            gene_symbol="TEST_GENE",
            gene_rank_score=0.85,
            confidence=0.90,
            multi_omics=MultiOmicsEvidence(
                gene_symbol="TEST_GENE",
                multi_omics_score=0.75,
                n_evidence_layers=3,
                evidence_layers=["genomic", "expression", "methylation"],
                has_expression_outlier=True,
                expression=GeneExpression(
                    gene_symbol="TEST_GENE", tpm=0.05,
                    z_score=-4.2, is_outlier=True, outlier_direction="under",
                ),
                expression_score=0.8,
                has_dmr=True,
                has_promoter_dmr=True,
                methylation_regions=[MethylationRegion(
                    gene_symbol="TEST_GENE", delta_beta=0.4,
                    is_dmr=True, dmr_direction="hyper",
                    region_type="promoter",
                )],
                methylation_score=0.7,
                is_concordant=True,
                concordance_description="Underexpression concordant with promoter hypermethylation (silencing)",
            ),
            multi_omics_score=0.75,
            n_evidence_layers=3,
        )
        text = explainer.explain_gene(candidate)

        assert "Multi-omics" in text
        assert "3 evidence layers" in text
        assert "under" in text
        assert "CONCORDANT" in text
        assert "silencing" in text.lower()

    def test_explain_gene_no_multi_omics(self):
        from raregeneai.explainability.explainer import Explainer

        explainer = Explainer()
        candidate = GeneCandidate(
            gene_symbol="GENE_X",
            gene_rank_score=0.5,
            confidence=0.6,
        )
        text = explainer.explain_gene(candidate)

        # Should NOT contain multi-omics section
        assert "Multi-omics" not in text
