"""Master pipeline orchestrator.

Coordinates the full RareGeneAI workflow:
  VCF → Ingestion → Annotation → Phenotype Matching →
  Scoring → Ranking → Explanation → Report
"""

from __future__ import annotations

import time
from pathlib import Path

from loguru import logger

from raregeneai.annotation.annotation_engine import AnnotationEngine
from raregeneai.clinical.clinical_decision import ClinicalDecisionEngine
from raregeneai.config.settings import PipelineConfig
from raregeneai.explainability.explainer import Explainer
from raregeneai.ingestion.hpo_parser import HPOParser
from raregeneai.ingestion.pedigree_parser import PedigreeParser
from raregeneai.ingestion.vcf_parser import VCFParser
from raregeneai.models.data_models import ClinicalReport, Pedigree, PatientPhenotype
from raregeneai.phenotype.gene_phenotype_matcher import GenePhenotypeMatcher
from raregeneai.ranking.gene_ranker import GeneRanker
from raregeneai.reporting.report_generator import ReportGenerator
from raregeneai.scoring.inheritance_analyzer import InheritanceAnalyzer
from raregeneai.scoring.variant_scorer import VariantScorer
from raregeneai.knowledge_graph.graph_scorer import KnowledgeGraphScorer
from raregeneai.multiomics.integrator import MultiOmicsIntegrator
from raregeneai.structural.sv_annotator import SVAnnotator
from raregeneai.structural.sv_integration import enrich_candidates_with_sv, sv_to_annotated_variants
from raregeneai.structural.sv_parser import SVParser
from raregeneai.utils.logging import setup_logging


class RareGeneAIPipeline:
    """End-to-end rare disease gene prioritization pipeline."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        setup_logging(self.config.log_level)

        # Initialize modules
        self.vcf_parser = VCFParser(self.config.ingestion)
        self.hpo_parser = HPOParser(self.config.phenotype)
        self.ped_parser = PedigreeParser()
        self.annotation_engine = AnnotationEngine(self.config.annotation)
        self.phenotype_matcher = GenePhenotypeMatcher(self.config.phenotype)
        self.variant_scorer = VariantScorer(self.config.scoring)
        self.gene_ranker = GeneRanker(self.config.ranking)
        self.explainer = Explainer()
        self.report_generator = ReportGenerator(self.config.report)

        # SV modules
        self.sv_parser = SVParser(self.config.sv)
        self.sv_annotator = SVAnnotator(self.config.sv)

        # Multi-omics module
        self.multiomics_integrator = MultiOmicsIntegrator(self.config.multiomics)

        # Inheritance analyzer (for trio analysis)
        self.inheritance_analyzer = InheritanceAnalyzer()

        # Knowledge graph scorer
        self.kg_scorer = KnowledgeGraphScorer(self.config.knowledge_graph)

        # Clinical decision support
        self.clinical_engine = ClinicalDecisionEngine()

    def run(
        self,
        vcf_path: str,
        hpo_terms: list[str],
        patient_id: str = "PATIENT_001",
        sample_id: str | None = None,
        ped_path: str | None = None,
        sv_vcf_path: str | None = None,
        father_vcf_path: str | None = None,
        mother_vcf_path: str | None = None,
        expression_path: str | None = None,
        methylation_path: str | None = None,
        output_dir: str | None = None,
    ) -> ClinicalReport:
        """Execute the full RareGeneAI pipeline.

        Args:
            vcf_path: Path to input VCF file (SNVs/indels, proband).
            hpo_terms: List of HPO term IDs (e.g., ["HP:0001250"]).
            patient_id: Patient identifier.
            sample_id: VCF sample ID to analyze.
            ped_path: Optional PED file for trio/family analysis.
            sv_vcf_path: Optional SV VCF (Sniffles/Jasmine/Manta output).
            father_vcf_path: Optional father VCF for trio inheritance analysis.
            mother_vcf_path: Optional mother VCF for trio inheritance analysis.
            expression_path: Optional RNA-seq expression (TPM per gene).
            methylation_path: Optional methylation data (BED or DMR calls).
            output_dir: Output directory for report and intermediate files.

        Returns:
            ClinicalReport with ranked genes and explanations.
        """
        start_time = time.time()
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("RareGeneAI Pipeline Started")
        logger.info(f"VCF: {vcf_path}")
        if sv_vcf_path:
            logger.info(f"SV VCF: {sv_vcf_path}")
        logger.info(f"HPO terms: {hpo_terms}")
        logger.info(f"Patient: {patient_id}")
        logger.info("=" * 60)

        # ── Step 1: Data Ingestion ─────────────────────────────────────
        logger.info("[1/8] Ingesting VCF and phenotype data...")

        variants = self.vcf_parser.parse(vcf_path, sample_id)
        patient_phenotype = self.hpo_parser.parse_phenotype(
            patient_id=patient_id,
            hpo_terms=hpo_terms,
        )

        pedigree = None
        if ped_path:
            pedigrees = self.ped_parser.parse(ped_path)
            pedigree = pedigrees[0] if pedigrees else None

        logger.info(f"Ingested {len(variants)} SNV/indel variants, "
                    f"{len(patient_phenotype.hpo_terms)} HPO terms")

        # ── Step 2: Variant Annotation ─────────────────────────────────
        logger.info("[2/8] Annotating SNV/indel variants...")

        annotated_variants = self.annotation_engine.annotate_variants(variants)

        # ── Step 3: Trio Inheritance Analysis ──────────────────────────
        if father_vcf_path or mother_vcf_path:
            logger.info("[3/10] Running trio inheritance analysis...")

            father_variants = None
            mother_variants = None

            if father_vcf_path:
                raw_father = self.vcf_parser.parse(father_vcf_path)
                father_variants = self.annotation_engine.vep.annotate(raw_father)
                logger.info(f"Father: {len(father_variants)} variants")

            if mother_vcf_path:
                raw_mother = self.vcf_parser.parse(mother_vcf_path)
                mother_variants = self.annotation_engine.vep.annotate(raw_mother)
                logger.info(f"Mother: {len(mother_variants)} variants")

            trio_results = self.inheritance_analyzer.analyze_trio(
                proband_variants=annotated_variants,
                father_variants=father_variants,
                mother_variants=mother_variants,
                pedigree=pedigree,
            )

            n_denovo = len(trio_results["de_novo"])
            n_comphet = len(trio_results["compound_het"])
            n_homrec = len(trio_results["homozygous_recessive"])
            logger.info(
                f"Trio results: {n_denovo} de novo, {n_comphet} compound het pairs, "
                f"{n_homrec} homozygous recessive"
            )
        else:
            logger.info("[3/10] No parental VCFs provided, using zygosity-based inheritance")
            # Tag all variants with zygosity-only scores
            self.inheritance_analyzer.analyze_trio(annotated_variants)

        # ── Step 4: Variant Scoring ────────────────────────────────────
        logger.info("[4/10] Scoring SNV/indel variants...")

        scored_variants = self.variant_scorer.score_variants(annotated_variants)

        # ── Step 5: Filter by quality ──────────────────────────────────
        logger.info("[5/10] Filtering variants...")

        filtered_variants = self.variant_scorer.filter_variants(
            scored_variants,
            min_score=0.05,
            require_rare=True,
            require_coding=not self.config.include_noncoding,
        )

        n_noncoding_kept = sum(1 for v in filtered_variants if v.is_noncoding)
        n_regulatory_kept = sum(1 for v in filtered_variants if v.has_regulatory_annotation)
        logger.info(
            f"Retained {len(filtered_variants)} SNV/indel variants "
            f"({n_noncoding_kept} non-coding, {n_regulatory_kept} with regulatory annotation)"
        )

        # ── Step 5: Structural Variant Analysis ────────────────────────
        total_svs = 0
        annotated_svs = []

        if sv_vcf_path and self.config.sv.enabled:
            logger.info("[6/10] Parsing and annotating structural variants...")

            raw_svs = self.sv_parser.parse(sv_vcf_path, sample_id)
            total_svs = len(raw_svs)

            if raw_svs:
                annotated_svs = self.sv_annotator.annotate(raw_svs)

                # Convert SVs to AnnotatedVariant format and merge
                sv_as_variants = sv_to_annotated_variants(annotated_svs)
                filtered_variants = filtered_variants + sv_as_variants

                logger.info(
                    f"SV analysis: {total_svs} SVs parsed, "
                    f"{len(annotated_svs)} annotated, "
                    f"{len(sv_as_variants)} gene-level records added to ranking"
                )
        else:
            logger.info("[6/10] No SV VCF provided, skipping SV analysis")

        # ── Step 6: Phenotype Matching ─────────────────────────────────
        logger.info("[7/10] Computing phenotype scores...")

        candidate_genes = list({v.effective_gene_symbol for v in filtered_variants if v.effective_gene_symbol})
        phenotype_scores = self.phenotype_matcher.score_candidates(
            candidate_genes, patient_phenotype
        )

        # ── Step 7: Gene Ranking ───────────────────────────────────────
        logger.info("[8/10] Ranking genes (unified SNV + SV)...")

        ranked_genes = self.gene_ranker.rank(
            filtered_variants,
            phenotype_scores,
            patient_phenotype,
        )

        # Enrich with SV-specific evidence for reporting
        if annotated_svs:
            ranked_genes = enrich_candidates_with_sv(ranked_genes, annotated_svs)

        # ── Step 9: Knowledge Graph Scoring ───────────────────────────
        if self.config.knowledge_graph.enabled:
            logger.info("[9/11] Computing knowledge graph scores...")

            patient_hpo_ids = [t.id for t in patient_phenotype.hpo_terms]
            gene_names = [g.gene_symbol for g in ranked_genes]

            kg_results = self.kg_scorer.score_genes(patient_hpo_ids, gene_names)

            if kg_results:
                ranked_genes = self.kg_scorer.enrich_candidates(ranked_genes, kg_results)

                # Apply KG boost to ranking
                kg_weight = self.config.knowledge_graph.kg_weight_in_ranking
                for g in ranked_genes:
                    g.gene_rank_score = (
                        g.gene_rank_score * (1.0 - kg_weight)
                        + g.kg_score * kg_weight
                    )
                ranked_genes.sort(key=lambda g: g.gene_rank_score, reverse=True)

                n_kg = sum(1 for g in ranked_genes if g.kg_score > 0.01)
                logger.info(f"KG scoring: {n_kg} genes with graph-based evidence")
        else:
            logger.info("[9/11] Knowledge graph scoring disabled")

        # ── Step 10: Multi-omics Integration ───────────────────────────
        expr_path = expression_path or self.config.multiomics.expression_path
        meth_path = methylation_path or self.config.multiomics.methylation_path
        dmr_path = self.config.multiomics.dmr_calls_path

        if self.config.multiomics.enabled and (expr_path or meth_path or dmr_path):
            logger.info("[10/11] Integrating multi-omics evidence...")

            gene_names = [g.gene_symbol for g in ranked_genes]
            multi_omics = self.multiomics_integrator.analyze(
                candidate_genes=gene_names,
                expression_path=expr_path,
                methylation_path=meth_path,
                dmr_calls_path=dmr_path,
            )

            ranked_genes = self.multiomics_integrator.enrich_candidates(
                ranked_genes, multi_omics
            )

            # Re-sort: boost genes with multi-omics support
            mo_weight = self.config.multiomics.multiomics_weight_in_ranking
            for g in ranked_genes:
                g.gene_rank_score = (
                    g.gene_rank_score * (1.0 - mo_weight)
                    + g.multi_omics_score * mo_weight
                )
            ranked_genes.sort(key=lambda g: g.gene_rank_score, reverse=True)

            n_mo = sum(1 for g in ranked_genes if g.n_evidence_layers >= 2)
            logger.info(f"Multi-omics: {n_mo} genes with 2+ evidence layers")
        else:
            logger.info("[10/11] No multi-omics data provided, skipping")

        # ── Step 11: Clinical Decision Support ─────────────────────────
        logger.info("[11/12] Running clinical decision support...")

        clinical_insights = self.clinical_engine.analyze(ranked_genes)
        ranked_genes = self.clinical_engine.enrich_candidates(ranked_genes, clinical_insights)

        n_diagnostic = sum(
            1 for i in clinical_insights.values()
            if i.clinical_significance in ("Diagnostic", "Likely Diagnostic")
        )
        n_sf = sum(1 for i in clinical_insights.values() if i.is_acmg_sf_gene)
        n_pgx = sum(1 for i in clinical_insights.values() if i.has_pgx_relevance)
        logger.info(
            f"Clinical insights: {n_diagnostic} diagnostic, "
            f"{n_sf} secondary findings, {n_pgx} pharmacogenomic"
        )

        # ── Step 12: Report Generation ─────────────────────────────────
        logger.info("[12/12] Generating clinical report...")

        report_path = output_dir / f"{patient_id}_report.html"
        report = self.report_generator.generate(
            patient_phenotype=patient_phenotype,
            ranked_genes=ranked_genes,
            total_variants=len(variants),
            total_genes=len(candidate_genes),
            pedigree=pedigree,
            output_path=str(report_path),
        )
        report.total_svs_analyzed = total_svs

        # Save intermediate data
        variants_df = self.annotation_engine.to_dataframe(scored_variants)
        variants_df.to_parquet(output_dir / f"{patient_id}_variants.parquet")

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"Pipeline completed in {elapsed:.1f}s")
        logger.info(f"Analyzed: {len(variants)} SNVs + {total_svs} SVs")
        logger.info(f"Top 5 genes: {[g.gene_symbol for g in ranked_genes[:5]]}")
        logger.info(f"Report: {report_path}")
        logger.info("=" * 60)

        return report
