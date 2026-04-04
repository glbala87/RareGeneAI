"""Explainable AI layer for clinical interpretability.

Generates human-readable explanations for gene rankings using:
  - SHAP values (for ML models)
  - Rule-based evidence summaries
  - Phenotype match justifications
  - Variant-level explanations

Critical for clinical use: every ranking must be justified.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from raregeneai.models.data_models import (
    ACMGClassification,
    AnnotatedVariant,
    FunctionalImpact,
    GeneCandidate,
    PatientPhenotype,
)


class Explainer:
    """Generate explanations for gene rankings and variant prioritization."""

    FEATURE_DESCRIPTIONS = {
        "max_variant_score": "variant pathogenicity composite score",
        "max_cadd": "CADD pathogenicity prediction",
        "max_revel": "REVEL missense pathogenicity",
        "min_af": "population allele frequency (rarity)",
        "has_lof": "loss-of-function variant presence",
        "has_clinvar_pathogenic": "ClinVar pathogenic classification",
        "phenotype_score": "phenotype similarity to known gene-disease associations",
        "n_variants": "number of qualifying variants",
        "n_inheritance_modes": "compatible inheritance patterns",
        "is_known_disease_gene": "known disease gene status",
        # Non-coding regulatory features
        "n_noncoding_variants": "non-coding variants affecting gene regulation",
        "max_regulatory_score": "regulatory impact score (ENCODE/Roadmap/DL)",
        "max_spliceai": "SpliceAI splice disruption prediction",
        "has_regulatory_variant": "variant in annotated regulatory region",
        "has_enhancer_variant": "variant disrupting enhancer element",
        "has_promoter_variant": "variant disrupting promoter region",
        "max_gene_mapping_score": "variant-to-gene mapping confidence (eQTL/Hi-C/ABC)",
        "max_conservation_score": "sequence conservation in regulatory region",
        # Structural variant features
        "has_sv": "structural variant affecting gene",
        "max_sv_score": "structural variant pathogenicity score",
        "sv_fully_deleted": "gene fully deleted by structural variant",
        "sv_dosage_sensitive": "gene is dosage-sensitive (pLI/ClinGen)",
        "max_sv_gene_overlap": "gene disruption by structural variant",
        "max_sv_dosage_score": "dosage sensitivity of affected gene",
        "max_sv_regulatory_disruption": "regulatory element disruption by SV",
        # Multi-omics features
        "multi_omics_score": "multi-omics confidence score",
        "n_evidence_layers": "number of supporting evidence layers",
        "has_expression_outlier": "gene has aberrant expression (RNA-seq)",
        "expression_score": "expression outlier severity",
        "has_dmr": "gene overlaps differentially methylated region",
        "has_promoter_dmr": "promoter region is differentially methylated",
        "methylation_score": "methylation aberration severity",
        "is_concordant": "multi-omics evidence is directionally concordant",
        # Trio inheritance features
        "has_de_novo": "variant arose de novo (absent in both parents)",
        "has_de_novo_lof": "de novo loss-of-function variant (strongest signal)",
        "has_compound_het": "compound heterozygous (biallelic, variants in trans)",
        "has_hom_recessive": "homozygous recessive (confirmed by parental carriers)",
        "trio_inheritance_score": "trio-based inheritance model confidence",
        "trio_analyzed": "trio/family data was used for inheritance analysis",
        # Knowledge graph features
        "kg_score": "knowledge graph proximity to patient phenotype",
        "kg_ppi_neighbors": "protein interaction partners among candidates",
        "kg_n_diseases": "known disease associations in knowledge graph",
        "kg_n_pathways": "pathway memberships connecting to phenotype",
        "kg_has_direct_hpo_link": "gene directly linked to patient HPO term",
    }

    def explain_gene(
        self,
        candidate: GeneCandidate,
        patient_phenotype: PatientPhenotype | None = None,
        model=None,
    ) -> str:
        """Generate comprehensive explanation for a gene's ranking.

        Returns a clinician-readable explanation string.
        """
        parts = []

        # Header
        parts.append(
            f"Gene {candidate.gene_symbol} ranked with score "
            f"{candidate.gene_rank_score:.3f} (confidence: {candidate.confidence:.1%})"
        )

        # Phenotype match
        if candidate.phenotype_score > 0:
            parts.append(
                f"  Phenotype match: {candidate.phenotype_score:.3f} — "
                f"gene has known associations to patient's clinical features"
            )

        # Variant evidence
        for var in candidate.variants:
            var_explanation = self._explain_variant(var)
            parts.append(f"  {var_explanation}")

        # Inheritance (trio-aware)
        if candidate.trio_analyzed:
            inh_parts = []
            if candidate.has_de_novo_lof:
                inh_parts.append("DE NOVO LOSS-OF-FUNCTION (very strong evidence)")
            elif candidate.has_de_novo:
                inh_parts.append("DE NOVO variant (strong evidence)")
            if candidate.has_compound_het:
                # Find the pair
                ch_vars = [v for v in candidate.variants if v.is_compound_het]
                if len(ch_vars) >= 2:
                    impacts = [v.impact.value for v in ch_vars[:2]]
                    inh_parts.append(
                        f"COMPOUND HETEROZYGOUS (biallelic, {impacts[0]}+{impacts[1]})"
                    )
                else:
                    inh_parts.append("COMPOUND HETEROZYGOUS (biallelic)")
            if candidate.has_hom_recessive:
                inh_parts.append("HOMOZYGOUS RECESSIVE (both parents carriers)")
            if inh_parts:
                parts.append("  Trio inheritance: " + "; ".join(inh_parts))
                parts.append(
                    f"  Inheritance score: {candidate.inheritance_score:.2f}"
                )
        if candidate.inheritance_modes:
            modes = ", ".join(m.value for m in candidate.inheritance_modes)
            parts.append(f"  Compatible inheritance: {modes}")

        # Structural variant evidence
        if candidate.has_sv:
            sv_parts = []
            if candidate.sv_fully_deleted:
                sv_parts.append("WHOLE GENE DELETION")
            for sv in candidate.structural_variants[:3]:
                sv_desc = (
                    f"{sv.sv.sv_type.value} ({sv.sv.sv_len:,}bp) "
                    f"score={sv.sv_composite_score:.3f}"
                )
                if sv.has_dosage_sensitive_gene:
                    sv_desc += " [dosage-sensitive]"
                if sv.known_syndrome:
                    sv_desc += f" [{sv.known_syndrome}]"
                if sv.n_genes_affected > 1:
                    sv_desc += f" ({sv.n_genes_affected} genes affected)"
                sv_parts.append(sv_desc)
            parts.append("  Structural variants: " + "; ".join(sv_parts))

        # Multi-omics evidence
        if candidate.multi_omics and candidate.multi_omics_score > 0:
            mo = candidate.multi_omics
            mo_parts = [f"Score={mo.multi_omics_score:.3f}"]
            mo_parts.append(f"{mo.n_evidence_layers} evidence layers ({', '.join(mo.evidence_layers)})")

            if mo.has_expression_outlier and mo.expression:
                direction = mo.expression.outlier_direction
                z = mo.expression.z_score
                mo_parts.append(f"expression {direction} (Z={z:+.1f}, TPM={mo.expression.tpm:.1f})")

            if mo.has_dmr:
                dmr_types = []
                for m in mo.methylation_regions:
                    if m.is_dmr:
                        dmr_types.append(f"{m.dmr_direction} {m.region_type} (delta={m.delta_beta:+.2f})")
                if dmr_types:
                    mo_parts.append("DMRs: " + "; ".join(dmr_types[:3]))

            if mo.is_concordant:
                mo_parts.append(f"CONCORDANT: {mo.concordance_description}")

            parts.append("  Multi-omics: " + " | ".join(mo_parts))

        # Knowledge graph evidence
        if candidate.kg_score > 0.01:
            kg_parts = [f"Score={candidate.kg_score:.3f} (rank #{candidate.kg_rank})"]
            if candidate.kg_connected_diseases:
                kg_parts.append(f"diseases: {', '.join(candidate.kg_connected_diseases[:3])}")
            if candidate.kg_connected_pathways:
                kg_parts.append(f"pathways: {', '.join(candidate.kg_connected_pathways[:3])}")
            if candidate.kg_ppi_neighbors > 0:
                kg_parts.append(f"{candidate.kg_ppi_neighbors} PPI neighbors among candidates")
            if candidate.kg_paths:
                kg_parts.append(f"path: {candidate.kg_paths[0]}")
            parts.append("  Knowledge graph: " + " | ".join(kg_parts))

        # Known disease associations
        if candidate.known_diseases:
            diseases = ", ".join(candidate.known_diseases[:3])
            parts.append(f"  Known disease associations: {diseases}")

        # SHAP-based explanation (if ML model available)
        if model is not None:
            shap_explanation = self._explain_with_shap(candidate, model)
            if shap_explanation:
                parts.append(f"  Model drivers: {shap_explanation}")

        explanation = "\n".join(parts)
        candidate.explanation = explanation
        return explanation

    def _explain_variant(self, var: AnnotatedVariant) -> str:
        """Generate explanation for a single variant."""
        parts = []

        # Detect if this is an SV-derived variant
        is_sv = var.variant.alt.startswith("<") and var.variant.alt.endswith(">")

        # Variant identity
        if is_sv:
            parts.append(
                f"SV {var.variant.alt} at {var.variant.chrom}:{var.variant.pos}: "
                f"{var.consequence}"
            )
        else:
            parts.append(
                f"Variant {var.variant.variant_key}: "
                f"{var.hgvs_p or var.hgvs_c or var.consequence}"
            )

        # Impact
        if is_sv:
            if "whole_gene" in var.consequence:
                parts.append(f"WHOLE GENE DISRUPTION ({var.consequence})")
            else:
                parts.append(f"STRUCTURAL ({var.consequence})")
        elif var.impact == FunctionalImpact.HIGH:
            parts.append(f"HIGH impact ({var.consequence})")
        elif var.impact == FunctionalImpact.MODERATE:
            parts.append(f"MODERATE impact ({var.consequence})")
        elif var.is_noncoding and var.has_regulatory_annotation:
            parts.append(f"REGULATORY ({var.consequence})")

        # Pathogenicity scores
        score_parts = []
        if var.cadd_phred is not None:
            severity = "deleterious" if var.cadd_phred >= 20 else "moderate"
            score_parts.append(f"CADD={var.cadd_phred:.1f} ({severity})")
        if var.revel_score is not None:
            severity = "pathogenic" if var.revel_score >= 0.5 else "benign"
            score_parts.append(f"REVEL={var.revel_score:.3f} ({severity})")
        if var.spliceai_max is not None and var.spliceai_max > 0.2:
            severity = "strong" if var.spliceai_max >= 0.5 else "moderate"
            score_parts.append(f"SpliceAI={var.spliceai_max:.2f} ({severity} splice disruption)")

        if score_parts:
            parts.append(", ".join(score_parts))

        # Non-coding regulatory context
        if var.is_noncoding:
            reg_parts = []

            # Regulatory region
            if var.regulatory_class:
                region_desc = var.regulatory_class.replace("_", " ")
                reg_parts.append(f"in {region_desc}")
                if var.regulatory_feature_id:
                    reg_parts[-1] += f" ({var.regulatory_feature_id})"

            # ChromHMM state
            if var.chromhmm_state_name:
                reg_parts.append(f"chromatin: {var.chromhmm_state_name}")

            # Conservation
            if var.phastcons_score is not None and var.phastcons_score > 0.5:
                level = "highly conserved" if var.phastcons_score > 0.9 else "conserved"
                reg_parts.append(f"PhastCons={var.phastcons_score:.2f} ({level})")
            if var.gerp_score is not None and var.gerp_score > 2.0:
                reg_parts.append(f"GERP={var.gerp_score:.1f}")

            # Deep learning scores
            if var.enformer_score is not None and var.enformer_score > 0.3:
                reg_parts.append(f"Enformer={var.enformer_score:.2f}")

            # Gene mapping
            if var.target_gene_symbol:
                method_desc = {
                    "eqtl": f"eQTL in {var.eqtl_tissue}" if var.eqtl_tissue else "eQTL",
                    "abc": "ABC model",
                    "chromatin_interaction": "Hi-C",
                    "nearest": "nearest gene",
                }.get(var.gene_mapping_method, var.gene_mapping_method)
                confidence = f"conf={var.gene_mapping_score:.2f}"
                reg_parts.append(
                    f"targets {var.target_gene_symbol} via {method_desc} ({confidence})"
                )

            if reg_parts:
                parts.append("Regulatory: " + ", ".join(reg_parts))

            # Regulatory impact score
            if var.regulatory_score > 0:
                parts.append(f"Regulatory impact={var.regulatory_score:.3f}")

        # Frequency (population-aware)
        if var.is_novel:
            parts.append("novel (absent in gnomAD and local databases)")
        elif var.gnomad_af is not None or var.local_af is not None:
            freq_parts = []
            if var.gnomad_af is not None:
                freq_parts.append(f"gnomAD AF={var.gnomad_af:.2e}")
            if var.local_af is not None:
                pop_label = var.local_population or "local"
                freq_parts.append(f"{pop_label} AF={var.local_af:.2e}")
            if var.is_founder_variant:
                freq_parts.append(
                    f"FOUNDER VARIANT ({var.founder_enrichment:.0f}x enrichment)"
                )
            parts.append(", ".join(freq_parts))

        # Trio inheritance tag
        if var.is_de_novo:
            parts.append("DE NOVO")
        elif var.is_compound_het:
            parts.append(f"COMPOUND HET (partner: {var.compound_het_partner_key})")
        elif var.is_hom_recessive:
            parts.append("HOM RECESSIVE (parental carriers)")
        elif var.parent_of_origin:
            parts.append(f"inherited ({var.parent_of_origin})")

        # ClinVar
        if var.clinvar_significance:
            parts.append(f"ClinVar: {var.clinvar_significance}")

        return " | ".join(parts)

    def _explain_with_shap(self, candidate: GeneCandidate, model) -> str:
        """Generate SHAP-based feature importance explanation."""
        try:
            import shap

            features = pd.DataFrame([candidate.evidence_summary])
            feature_cols = [
                "max_variant_score", "max_cadd", "max_revel", "min_af",
                "has_lof", "has_clinvar_pathogenic", "phenotype_score",
                "n_variants", "n_inheritance_modes", "is_known_disease_gene",
                "n_noncoding_variants", "max_regulatory_score", "max_spliceai",
                "has_regulatory_variant", "has_enhancer_variant",
                "has_promoter_variant", "max_gene_mapping_score",
                "max_conservation_score",
                "has_sv", "max_sv_score", "sv_fully_deleted",
                "sv_dosage_sensitive", "max_sv_gene_overlap",
                "max_sv_dosage_score", "max_sv_regulatory_disruption",
                "multi_omics_score", "n_evidence_layers",
                "has_expression_outlier", "expression_score",
                "has_dmr", "has_promoter_dmr", "methylation_score",
                "is_concordant",
                "has_de_novo", "has_de_novo_lof", "has_compound_het",
                "has_hom_recessive", "trio_inheritance_score",
                "trio_analyzed",
                "kg_score", "kg_ppi_neighbors", "kg_n_diseases",
                "kg_n_pathways", "kg_has_direct_hpo_link",
            ]

            # Ensure all columns exist
            for col in feature_cols:
                if col not in features.columns:
                    features[col] = 0

            features = features[feature_cols].fillna(0)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Class 1 (pathogenic)

            # Get top contributing features
            shap_vals = shap_values[0]
            feature_importance = sorted(
                zip(feature_cols, shap_vals),
                key=lambda x: abs(x[1]),
                reverse=True,
            )

            explanations = []
            for feat, val in feature_importance[:3]:
                direction = "increases" if val > 0 else "decreases"
                desc = self.FEATURE_DESCRIPTIONS.get(feat, feat)
                explanations.append(f"{desc} {direction} ranking ({val:+.3f})")

            return "; ".join(explanations)

        except Exception as e:
            logger.debug(f"SHAP explanation failed: {e}")
            return ""

    def explain_ranking(
        self,
        candidates: list[GeneCandidate],
        patient_phenotype: PatientPhenotype | None = None,
        model=None,
    ) -> list[dict]:
        """Generate explanations for entire ranked list."""
        explanations = []

        for i, candidate in enumerate(candidates):
            text = self.explain_gene(candidate, patient_phenotype, model)
            explanations.append({
                "rank": i + 1,
                "gene": candidate.gene_symbol,
                "score": candidate.gene_rank_score,
                "confidence": candidate.confidence,
                "explanation": text,
                "evidence": candidate.evidence_summary,
                "n_variants": candidate.n_variants,
            })

        return explanations

    def classify_acmg(self, var: AnnotatedVariant) -> ACMGClassification:
        """Apply simplified ACMG classification criteria.

        This is a simplified implementation. Full ACMG classification
        requires expert review and additional evidence types.
        """
        evidence_for_pathogenic = 0
        evidence_for_benign = 0

        # PVS1: Null variant in gene where LoF is known mechanism
        if var.impact == FunctionalImpact.HIGH:
            evidence_for_pathogenic += 4  # Very strong

        # PS1/PM5: Known pathogenic in ClinVar
        clinvar_lower = var.clinvar_significance.lower()
        if "pathogenic" in clinvar_lower and "likely" not in clinvar_lower:
            evidence_for_pathogenic += 3  # Strong
        elif "likely pathogenic" in clinvar_lower:
            evidence_for_pathogenic += 2  # Moderate

        # PM2: Absent from population databases
        if var.is_novel:
            evidence_for_pathogenic += 2  # Moderate
        elif var.gnomad_af is not None and var.gnomad_af < 0.0001:
            evidence_for_pathogenic += 1  # Supporting

        # PP3: Computational evidence (coding and non-coding)
        if var.cadd_phred is not None and var.cadd_phred >= 25:
            evidence_for_pathogenic += 1  # Supporting
        if var.revel_score is not None and var.revel_score >= 0.7:
            evidence_for_pathogenic += 1  # Supporting
        # PP3 non-coding: strong SpliceAI prediction
        if var.spliceai_max is not None and var.spliceai_max >= 0.5:
            evidence_for_pathogenic += 2  # Moderate (strong splice disruption)
        elif var.spliceai_max is not None and var.spliceai_max >= 0.2:
            evidence_for_pathogenic += 1  # Supporting
        # PP3 non-coding: regulatory impact from DL models
        if var.regulatory_score >= 0.7 and var.is_noncoding:
            evidence_for_pathogenic += 1  # Supporting (regulatory disruption)
        # PP3 non-coding: high conservation in regulatory region
        if (var.is_noncoding and var.phastcons_score is not None
                and var.phastcons_score > 0.9 and var.has_regulatory_annotation):
            evidence_for_pathogenic += 1  # Supporting

        # BA1: High AF = benign
        if var.gnomad_af is not None and var.gnomad_af > 0.05:
            evidence_for_benign += 8  # Stand-alone

        # BP4: Computational evidence for benign
        if var.cadd_phred is not None and var.cadd_phred < 10:
            evidence_for_benign += 1
        if var.revel_score is not None and var.revel_score < 0.15:
            evidence_for_benign += 1

        # ClinVar benign
        if "benign" in clinvar_lower:
            evidence_for_benign += 3

        # Classification logic
        if evidence_for_benign >= 8:
            return ACMGClassification.BENIGN
        elif evidence_for_benign >= 4:
            return ACMGClassification.LIKELY_BENIGN
        elif evidence_for_pathogenic >= 6:
            return ACMGClassification.PATHOGENIC
        elif evidence_for_pathogenic >= 4:
            return ACMGClassification.LIKELY_PATHOGENIC
        else:
            return ACMGClassification.VUS
