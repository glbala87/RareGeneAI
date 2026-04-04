"""Gene ranking engine with multi-modal evidence fusion.

Supports three ranking strategies:
  1. Rule-based: Weighted scoring formula
  2. XGBoost: Gradient-boosted tree model
  3. LightGBM: Light gradient boosting

Combines variant scores, phenotype scores, and gene-level features
into a unified gene ranking with confidence scores.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from raregeneai.config.settings import RankingConfig
from raregeneai.models.data_models import (
    AnnotatedVariant,
    FunctionalImpact,
    GeneCandidate,
    InheritanceMode,
    PatientPhenotype,
)
from raregeneai.scoring.inheritance_analyzer import InheritanceAnalyzer


class GeneRanker:
    """Rank candidate genes using multi-modal evidence fusion."""

    def __init__(self, config: RankingConfig | None = None):
        self.config = config or RankingConfig()
        self.inheritance_analyzer = InheritanceAnalyzer()
        self._model = None

    def rank(
        self,
        annotated_variants: list[AnnotatedVariant],
        phenotype_scores: dict[str, float],
        patient_phenotype: PatientPhenotype | None = None,
    ) -> list[GeneCandidate]:
        """Rank genes by multi-modal evidence.

        Steps:
        1. Group variants by gene
        2. Compute gene-level features
        3. Combine with phenotype scores
        4. Apply ranking model
        5. Return sorted GeneCandidate list

        Args:
            annotated_variants: Scored, annotated variants.
            phenotype_scores: gene_symbol -> phenotype similarity score.
            patient_phenotype: Optional patient phenotype for explanation.

        Returns:
            Sorted list of GeneCandidate objects (highest rank first).
        """
        # Step 1: Group variants by gene
        gene_variants = self._group_by_gene(annotated_variants)
        logger.info(f"Ranking {len(gene_variants)} candidate genes")

        # Step 2: Build gene candidates with features
        candidates = []
        for gene_symbol, variants in gene_variants.items():
            candidate = self._build_candidate(
                gene_symbol, variants, phenotype_scores
            )
            candidates.append(candidate)

        # Step 3: Apply ranking model
        if self.config.model_type == "rule_based":
            candidates = self._rank_rule_based(candidates)
        else:
            candidates = self._rank_ml(candidates)

        # Step 4: Sort by rank score
        candidates.sort(key=lambda c: c.gene_rank_score, reverse=True)

        # Step 5: Assign confidence and trim
        candidates = self._assign_confidence(candidates)
        top_n = self.config.top_n_genes
        candidates = candidates[:top_n]

        logger.info(
            f"Top gene: {candidates[0].gene_symbol} "
            f"(score={candidates[0].gene_rank_score:.4f})"
            if candidates else "No candidates"
        )

        return candidates

    def _group_by_gene(
        self, variants: list[AnnotatedVariant]
    ) -> dict[str, list[AnnotatedVariant]]:
        """Group variants by gene symbol.

        For non-coding variants mapped to a target gene via eQTL/Hi-C/ABC,
        uses the target gene symbol for grouping (via effective_gene_symbol).
        """
        groups: dict[str, list[AnnotatedVariant]] = defaultdict(list)
        for var in variants:
            gene = var.effective_gene_symbol
            if gene:
                groups[gene].append(var)
        return dict(groups)

    def _build_candidate(
        self,
        gene_symbol: str,
        variants: list[AnnotatedVariant],
        phenotype_scores: dict[str, float],
    ) -> GeneCandidate:
        """Build a GeneCandidate with aggregated features."""
        # Aggregate variant scores
        max_variant_score = max((v.composite_score for v in variants), default=0.0)
        max_cadd = max((v.cadd_phred or 0.0 for v in variants), default=0.0)
        max_revel = max((v.revel_score or 0.0 for v in variants), default=0.0)
        min_af = min(
            (v.gnomad_af for v in variants if v.gnomad_af is not None),
            default=None,
        )
        has_lof = any(v.impact == FunctionalImpact.HIGH for v in variants)
        has_clinvar_path = any(
            "pathogenic" in v.clinvar_significance.lower() for v in variants
        )

        # Non-coding / regulatory aggregation
        noncoding_variants = [v for v in variants if v.is_noncoding]
        n_noncoding = len(noncoding_variants)
        max_regulatory = max(
            (v.regulatory_score for v in noncoding_variants), default=0.0
        )
        max_spliceai = max(
            (v.spliceai_max or 0.0 for v in variants), default=0.0
        )
        has_regulatory = any(v.has_regulatory_annotation for v in variants)
        has_enhancer = any(v.regulatory_class == "enhancer" for v in variants)
        has_promoter = any(v.regulatory_class == "promoter" for v in variants)
        max_gene_mapping = max(
            (v.gene_mapping_score for v in noncoding_variants), default=0.0
        )
        max_conservation = max(
            (v.phastcons_score or 0.0 for v in noncoding_variants), default=0.0
        )

        # Inheritance analysis (trio-aware)
        inheritance_modes = self.inheritance_analyzer.infer_inheritance_modes(
            gene_symbol, variants
        )
        inh_evidence = self.inheritance_analyzer.compute_gene_inheritance_score(
            gene_symbol, variants
        )

        # Phenotype score
        pheno_score = phenotype_scores.get(gene_symbol, 0.0)

        gene_id = variants[0].gene_id if variants else ""

        candidate = GeneCandidate(
            gene_symbol=gene_symbol,
            gene_id=gene_id,
            variants=variants,
            max_variant_score=max_variant_score,
            phenotype_score=pheno_score,
            inheritance_modes=inheritance_modes,
            is_known_disease_gene=pheno_score > 0.1,
            max_regulatory_score=max_regulatory,
            has_regulatory_variant=has_regulatory,
            n_noncoding_variants=n_noncoding,
            # Trio inheritance fields
            has_de_novo=inh_evidence.get("has_de_novo", False),
            has_de_novo_lof=inh_evidence.get("has_de_novo_lof", False),
            has_compound_het=inh_evidence.get("has_compound_het", False),
            has_hom_recessive=inh_evidence.get("has_hom_recessive", False),
            inheritance_score=inh_evidence.get("inheritance_score", 0.0),
            trio_analyzed=inh_evidence.get("trio_analyzed", False),
        )

        # Store features for ML model (expanded with non-coding features)
        candidate.evidence_summary = {
            "n_variants": len(variants),
            "max_variant_score": max_variant_score,
            "max_cadd": max_cadd,
            "max_revel": max_revel,
            "min_af": min_af,
            "has_lof": has_lof,
            "has_clinvar_pathogenic": has_clinvar_path,
            "phenotype_score": pheno_score,
            "n_inheritance_modes": len(inheritance_modes),
            "is_known_disease_gene": pheno_score > 0.1,
            # Non-coding regulatory features
            "n_noncoding_variants": n_noncoding,
            "max_regulatory_score": max_regulatory,
            "max_spliceai": max_spliceai,
            "has_regulatory_variant": has_regulatory,
            "has_enhancer_variant": has_enhancer,
            "has_promoter_variant": has_promoter,
            "max_gene_mapping_score": max_gene_mapping,
            "max_conservation_score": max_conservation,
            # Trio inheritance features
            "has_de_novo": inh_evidence.get("has_de_novo", False),
            "has_de_novo_lof": inh_evidence.get("has_de_novo_lof", False),
            "has_compound_het": inh_evidence.get("has_compound_het", False),
            "has_hom_recessive": inh_evidence.get("has_hom_recessive", False),
            "inheritance_score": inh_evidence.get("inheritance_score", 0.0),
            "trio_analyzed": inh_evidence.get("trio_analyzed", False),
            "inheritance_class": inh_evidence.get("inheritance_class", ""),
        }

        return candidate

    def _rank_rule_based(self, candidates: list[GeneCandidate]) -> list[GeneCandidate]:
        """Rule-based ranking using weighted scoring formula.

        Gene Score = 0.30 * max_variant_score
                   + 0.25 * phenotype_score
                   + 0.10 * clinvar_bonus
                   + 0.08 * lof_bonus
                   + 0.07 * rarity_bonus
                   + 0.10 * regulatory_bonus
                   + 0.05 * splicing_bonus
                   + 0.05 * conservation_bonus
        """
        for c in candidates:
            ev = c.evidence_summary

            clinvar_bonus = 1.0 if ev.get("has_clinvar_pathogenic") else 0.0
            lof_bonus = 1.0 if ev.get("has_lof") else 0.0
            rarity_bonus = 1.0 if ev.get("min_af") is None or (ev.get("min_af") or 0) < 0.001 else 0.0

            # Non-coding regulatory bonuses
            regulatory_bonus = ev.get("max_regulatory_score", 0.0)
            splicing_bonus = min(ev.get("max_spliceai", 0.0), 1.0)
            conservation_bonus = ev.get("max_conservation_score", 0.0)

            # Structural variant bonuses
            sv_bonus = ev.get("max_sv_score", 0.0)
            sv_dosage_bonus = 1.0 if ev.get("sv_dosage_sensitive") else 0.0

            # Trio inheritance bonuses (highest impact features)
            de_novo_lof_bonus = 1.0 if ev.get("has_de_novo_lof") else 0.0
            de_novo_bonus = 1.0 if ev.get("has_de_novo") and not ev.get("has_de_novo_lof") else 0.0
            biallelic_bonus = 1.0 if (ev.get("has_compound_het") or ev.get("has_hom_recessive")) else 0.0
            trio_inh_score = ev.get("inheritance_score", 0.0)

            c.gene_rank_score = (
                0.20 * c.max_variant_score
                + 0.18 * c.phenotype_score
                + 0.06 * clinvar_bonus
                + 0.05 * lof_bonus
                + 0.05 * rarity_bonus
                + 0.06 * regulatory_bonus
                + 0.03 * splicing_bonus
                + 0.03 * conservation_bonus
                + 0.08 * sv_bonus
                + 0.04 * sv_dosage_bonus
                # Trio inheritance (total 0.22)
                + 0.10 * de_novo_lof_bonus    # De novo LoF = strongest signal
                + 0.05 * de_novo_bonus        # De novo non-LoF
                + 0.04 * biallelic_bonus      # Compound het or hom recessive
                + 0.03 * trio_inh_score       # Continuous inheritance score
            )

        return candidates

    def _rank_ml(self, candidates: list[GeneCandidate]) -> list[GeneCandidate]:
        """ML-based ranking using XGBoost or LightGBM."""
        # Build feature matrix
        features = self._extract_features(candidates)

        model = self._get_model()
        if model is not None:
            try:
                predictions = model.predict_proba(features)[:, 1]
                for i, c in enumerate(candidates):
                    c.gene_rank_score = float(predictions[i])
                return candidates
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}; falling back to rule-based")

        # No trained model: use rule-based as fallback
        return self._rank_rule_based(candidates)

    def _extract_features(self, candidates: list[GeneCandidate]) -> pd.DataFrame:
        """Extract feature matrix for ML model (expanded with non-coding features)."""
        records = []
        for c in candidates:
            ev = c.evidence_summary
            records.append({
                "max_variant_score": ev.get("max_variant_score", 0.0),
                "max_cadd": ev.get("max_cadd", 0.0),
                "max_revel": ev.get("max_revel", 0.0),
                "min_af": ev.get("min_af") if ev.get("min_af") is not None else 0.0,
                "has_lof": int(ev.get("has_lof", False)),
                "has_clinvar_pathogenic": int(ev.get("has_clinvar_pathogenic", False)),
                "phenotype_score": ev.get("phenotype_score", 0.0),
                "n_variants": ev.get("n_variants", 0),
                "n_inheritance_modes": ev.get("n_inheritance_modes", 0),
                "is_known_disease_gene": int(ev.get("is_known_disease_gene", False)),
                # Non-coding regulatory features
                "n_noncoding_variants": ev.get("n_noncoding_variants", 0),
                "max_regulatory_score": ev.get("max_regulatory_score", 0.0),
                "max_spliceai": ev.get("max_spliceai", 0.0),
                "has_regulatory_variant": int(ev.get("has_regulatory_variant", False)),
                "has_enhancer_variant": int(ev.get("has_enhancer_variant", False)),
                "has_promoter_variant": int(ev.get("has_promoter_variant", False)),
                "max_gene_mapping_score": ev.get("max_gene_mapping_score", 0.0),
                "max_conservation_score": ev.get("max_conservation_score", 0.0),
                # Structural variant features
                "has_sv": int(ev.get("has_sv", False)),
                "max_sv_score": ev.get("max_sv_score", 0.0),
                "sv_fully_deleted": int(ev.get("sv_fully_deleted", False)),
                "sv_dosage_sensitive": int(ev.get("sv_dosage_sensitive", False)),
                "max_sv_gene_overlap": ev.get("max_sv_gene_overlap", 0.0),
                "max_sv_dosage_score": ev.get("max_sv_dosage_score", 0.0),
                "max_sv_regulatory_disruption": ev.get("max_sv_regulatory_disruption", 0.0),
                # Multi-omics features
                "multi_omics_score": ev.get("multi_omics_score", 0.0),
                "n_evidence_layers": ev.get("n_evidence_layers", 0),
                "has_expression_outlier": int(ev.get("has_expression_outlier", False)),
                "expression_score": ev.get("expression_score", 0.0),
                "has_dmr": int(ev.get("has_dmr", False)),
                "has_promoter_dmr": int(ev.get("has_promoter_dmr", False)),
                "methylation_score": ev.get("methylation_score", 0.0),
                "is_concordant": int(ev.get("is_concordant", False)),
                # Trio inheritance features
                "has_de_novo": int(ev.get("has_de_novo", False)),
                "has_de_novo_lof": int(ev.get("has_de_novo_lof", False)),
                "has_compound_het": int(ev.get("has_compound_het", False)),
                "has_hom_recessive": int(ev.get("has_hom_recessive", False)),
                "trio_inheritance_score": ev.get("inheritance_score", 0.0),
                "trio_analyzed": int(ev.get("trio_analyzed", False)),
                # Knowledge graph features
                "kg_score": ev.get("kg_score", 0.0),
                "kg_ppi_neighbors": ev.get("kg_ppi_neighbors", 0),
                "kg_n_diseases": ev.get("kg_n_diseases", 0),
                "kg_n_pathways": ev.get("kg_n_pathways", 0),
                "kg_has_direct_hpo_link": int(ev.get("kg_has_direct_hpo_link", False)),
            })
        return pd.DataFrame(records)

    def _get_model(self):
        """Load pretrained ML model.

        Supports both:
          - New bundle format: {"model": ..., "feature_columns": ..., "training_metrics": ...}
          - Legacy format: bare XGBClassifier object
        """
        if self._model is not None:
            return self._model

        if self.config.pretrained_model_path:
            model_path = Path(self.config.pretrained_model_path)
            if model_path.exists():
                try:
                    with open(model_path, "rb") as f:
                        loaded = pickle.load(f)

                    if isinstance(loaded, dict) and "model" in loaded:
                        # New bundle format
                        self._model = loaded["model"]
                        metrics = loaded.get("training_metrics", {})
                        cv_auc = metrics.get("cv_roc_auc", "N/A")
                        logger.info(
                            f"Loaded pretrained model from {model_path} "
                            f"(CV ROC-AUC: {cv_auc})"
                        )
                    else:
                        # Legacy bare model
                        self._model = loaded
                        logger.info(f"Loaded pretrained model from {model_path}")

                    return self._model
                except Exception as e:
                    logger.error(f"Failed to load model: {e}")

        return None

    def _assign_confidence(self, candidates: list[GeneCandidate]) -> list[GeneCandidate]:
        """Assign confidence scores based on rank score distribution."""
        if not candidates:
            return candidates

        scores = [c.gene_rank_score for c in candidates]
        max_score = max(scores) if scores else 1.0

        for c in candidates:
            if max_score > 0:
                c.confidence = c.gene_rank_score / max_score
            else:
                c.confidence = 0.0

        return candidates

    def train_model(
        self,
        training_features: pd.DataFrame,
        training_labels: np.ndarray,
        save_path: str | None = None,
    ) -> None:
        """Train the ML ranking model on labeled data.

        Args:
            training_features: Feature matrix (from _extract_features).
            training_labels: Binary labels (1=causal, 0=not).
            save_path: Where to save the trained model.
        """
        if self.config.model_type == "xgboost":
            import xgboost as xgb

            self._model = xgb.XGBClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                eval_metric="logloss",
                use_label_encoder=False,
            )
        elif self.config.model_type == "lightgbm":
            import lightgbm as lgb

            self._model = lgb.LGBMClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
            )
        else:
            logger.warning("Rule-based model does not require training")
            return

        logger.info(f"Training {self.config.model_type} model on {len(training_labels)} samples")
        self._model.fit(training_features, training_labels)

        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(self._model, f)
            logger.info(f"Model saved to {save_path}")
