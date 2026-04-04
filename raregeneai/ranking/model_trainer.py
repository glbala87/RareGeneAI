"""Production ML training pipeline for RareGeneAI gene ranking.

Trains an XGBoost classifier on the full 43-feature vector derived
from all evidence layers:
  - Variant pathogenicity (CADD, REVEL, SpliceAI)
  - Phenotype similarity (HPO semantic matching)
  - Inheritance pattern (trio: de novo, compound het, hom recessive)
  - Multi-omics (expression outliers, methylation DMRs, concordance)
  - Structural variants (gene overlap, dosage sensitivity)
  - Non-coding regulatory (ENCODE, conservation, DL models)
  - Knowledge graph (RWR proximity, PPI, disease/pathway links)

Training data is built from:
  - ClinVar pathogenic variants (positive labels)
  - OMIM known gene-disease associations (positive labels)
  - Non-causal genes from the same cases (negative labels)

Evaluation uses clinical-genomics-relevant metrics:
  - ROC-AUC (discrimination)
  - Top-1 / Top-5 / Top-10 accuracy (causal gene in top K)
  - Mean Reciprocal Rank (MRR)
  - Precision-Recall AUC (class imbalance)

Interpretability via SHAP TreeExplainer with per-feature
contribution analysis.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    roc_auc_score,
)


# ── Canonical feature list (must match gene_ranker._extract_features) ─────────
FEATURE_COLUMNS = [
    # ── Variant pathogenicity (10) ────────────────────────────────────
    "max_variant_score",
    "max_cadd",
    "max_revel",
    "min_af",
    "has_lof",
    "has_clinvar_pathogenic",
    "phenotype_score",
    "n_variants",
    "n_inheritance_modes",
    "is_known_disease_gene",
    # ── Non-coding regulatory (8) ─────────────────────────────────────
    "n_noncoding_variants",
    "max_regulatory_score",
    "max_spliceai",
    "has_regulatory_variant",
    "has_enhancer_variant",
    "has_promoter_variant",
    "max_gene_mapping_score",
    "max_conservation_score",
    # ── Structural variants (7) ───────────────────────────────────────
    "has_sv",
    "max_sv_score",
    "sv_fully_deleted",
    "sv_dosage_sensitive",
    "max_sv_gene_overlap",
    "max_sv_dosage_score",
    "max_sv_regulatory_disruption",
    # ── Multi-omics (8) ──────────────────────────────────────────────
    "multi_omics_score",
    "n_evidence_layers",
    "has_expression_outlier",
    "expression_score",
    "has_dmr",
    "has_promoter_dmr",
    "methylation_score",
    "is_concordant",
    # ── Trio inheritance (6) ──────────────────────────────────────────
    "has_de_novo",
    "has_de_novo_lof",
    "has_compound_het",
    "has_hom_recessive",
    "trio_inheritance_score",
    "trio_analyzed",
    # ── Knowledge graph (5) ──────────────────────────────────────────
    "kg_score",
    "kg_ppi_neighbors",
    "kg_n_diseases",
    "kg_n_pathways",
    "kg_has_direct_hpo_link",
]

# Feature groups for SHAP visualization
FEATURE_GROUPS = {
    "Variant pathogenicity": FEATURE_COLUMNS[:10],
    "Non-coding regulatory": FEATURE_COLUMNS[10:18],
    "Structural variants": FEATURE_COLUMNS[18:25],
    "Multi-omics": FEATURE_COLUMNS[25:33],
    "Trio inheritance": FEATURE_COLUMNS[33:39],
    "Knowledge graph": FEATURE_COLUMNS[39:44],
}


class ModelTrainer:
    """Train, evaluate, and explain XGBoost gene ranking models."""

    def __init__(self):
        self.model = None
        self.feature_columns = FEATURE_COLUMNS
        self.training_metrics: dict = {}
        self.shap_values: np.ndarray | None = None

    # ── Training Data Construction ────────────────────────────────────────────

    def build_training_data(
        self,
        positive_cases: list[dict],
        negative_genes_per_case: int = 50,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Build training dataset from labeled cases.

        Each case dict must contain the gene's evidence_summary fields.
        Positive cases: the causal gene (label=1).
        Negative cases: non-causal genes from the same patient (label=0).

        Args:
            positive_cases: List of dicts. Each must have a "negative_genes"
                key with a list of dicts (non-causal gene features).
            negative_genes_per_case: Max negatives per positive case.

        Returns:
            (features DataFrame, labels array)
        """
        records = []
        labels = []

        for case in positive_cases:
            records.append(self._extract_row(case))
            labels.append(1)

            for neg in case.get("negative_genes", [])[:negative_genes_per_case]:
                records.append(self._extract_row(neg))
                labels.append(0)

        df = pd.DataFrame(records, columns=self.feature_columns)
        y = np.array(labels)

        n_pos = (y == 1).sum()
        n_neg = (y == 0).sum()
        logger.info(
            f"Training data: {len(y)} samples ({n_pos} positive, {n_neg} negative, "
            f"ratio 1:{n_neg // max(n_pos, 1)})"
        )

        return df, y

    def build_training_data_from_csv(
        self, csv_path: str,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Load pre-built training data from CSV.

        CSV must have all feature columns plus a 'label' column (0/1).
        """
        df = pd.read_csv(csv_path)

        if "label" not in df.columns:
            raise ValueError("CSV must contain a 'label' column")

        y = df["label"].values
        # Use only known feature columns; fill missing with 0
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        X = df[self.feature_columns]

        logger.info(f"Loaded training data from {csv_path}: {len(y)} samples")
        return X, y

    def _extract_row(self, gene_data: dict) -> list[float]:
        """Extract a normalized feature vector from gene evidence dict."""
        return [
            # Variant pathogenicity
            float(gene_data.get("max_variant_score", 0.0)),
            min(float(gene_data.get("max_cadd", 0.0)) / 40.0, 1.0),
            float(gene_data.get("max_revel", 0.0)),
            float(gene_data.get("min_af") or 0.0),
            int(bool(gene_data.get("has_lof", False))),
            int(bool(gene_data.get("has_clinvar_pathogenic", False))),
            float(gene_data.get("phenotype_score", 0.0)),
            min(int(gene_data.get("n_variants", 0)), 20) / 20.0,
            min(int(gene_data.get("n_inheritance_modes", 0)), 5) / 5.0,
            int(bool(gene_data.get("is_known_disease_gene", False))),
            # Non-coding regulatory
            min(int(gene_data.get("n_noncoding_variants", 0)), 10) / 10.0,
            float(gene_data.get("max_regulatory_score", 0.0)),
            float(gene_data.get("max_spliceai", 0.0)),
            int(bool(gene_data.get("has_regulatory_variant", False))),
            int(bool(gene_data.get("has_enhancer_variant", False))),
            int(bool(gene_data.get("has_promoter_variant", False))),
            float(gene_data.get("max_gene_mapping_score", 0.0)),
            float(gene_data.get("max_conservation_score", 0.0)),
            # Structural variants
            int(bool(gene_data.get("has_sv", False))),
            float(gene_data.get("max_sv_score", 0.0)),
            int(bool(gene_data.get("sv_fully_deleted", False))),
            int(bool(gene_data.get("sv_dosage_sensitive", False))),
            float(gene_data.get("max_sv_gene_overlap", 0.0)),
            float(gene_data.get("max_sv_dosage_score", 0.0)),
            float(gene_data.get("max_sv_regulatory_disruption", 0.0)),
            # Multi-omics
            float(gene_data.get("multi_omics_score", 0.0)),
            min(int(gene_data.get("n_evidence_layers", 0)), 5) / 5.0,
            int(bool(gene_data.get("has_expression_outlier", False))),
            float(gene_data.get("expression_score", 0.0)),
            int(bool(gene_data.get("has_dmr", False))),
            int(bool(gene_data.get("has_promoter_dmr", False))),
            float(gene_data.get("methylation_score", 0.0)),
            int(bool(gene_data.get("is_concordant", False))),
            # Trio inheritance
            int(bool(gene_data.get("has_de_novo", False))),
            int(bool(gene_data.get("has_de_novo_lof", False))),
            int(bool(gene_data.get("has_compound_het", False))),
            int(bool(gene_data.get("has_hom_recessive", False))),
            float(gene_data.get("inheritance_score", gene_data.get("trio_inheritance_score", 0.0))),
            int(bool(gene_data.get("trio_analyzed", False))),
            # Knowledge graph
            float(gene_data.get("kg_score", 0.0)),
            min(int(gene_data.get("kg_ppi_neighbors", 0)), 20) / 20.0,
            min(int(gene_data.get("kg_n_diseases", 0)), 10) / 10.0,
            min(int(gene_data.get("kg_n_pathways", 0)), 10) / 10.0,
            int(bool(gene_data.get("kg_has_direct_hpo_link", False))),
        ]

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        save_path: str | None = None,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        early_stopping_rounds: int = 20,
    ):
        """Train XGBoost classifier with cross-validated evaluation.

        Uses stratified 5-fold CV to compute out-of-fold predictions
        for unbiased evaluation, then trains final model on all data.

        Args:
            X: Feature DataFrame (N x 44).
            y: Binary labels (N,). 1=causal, 0=not.
            save_path: Where to save the trained model (.pkl).
            n_estimators: Number of boosting rounds.
            max_depth: Max tree depth.
            learning_rate: Boosting learning rate.
            early_stopping_rounds: Rounds without improvement to stop.

        Returns:
            Trained XGBClassifier.
        """
        import xgboost as xgb

        pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "scale_pos_weight": pos_weight,
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "random_state": 42,
        }

        logger.info(
            f"Training XGBoost: {n_estimators} trees, depth={max_depth}, "
            f"lr={learning_rate}, pos_weight={pos_weight:.1f}, "
            f"{len(self.feature_columns)} features"
        )

        # ── Cross-validated out-of-fold predictions ───────────────────────
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_proba = cross_val_predict(
            xgb.XGBClassifier(**params),
            X, y, cv=cv, method="predict_proba",
        )[:, 1]

        cv_metrics = self._compute_metrics(y, oof_proba, prefix="cv")
        logger.info(
            f"CV ROC-AUC: {cv_metrics['cv_roc_auc']:.4f} | "
            f"CV PR-AUC: {cv_metrics['cv_pr_auc']:.4f}"
        )

        # ── Train final model on all data ─────────────────────────────────
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X, y)

        # Full-data metrics (will be optimistic; CV metrics are authoritative)
        train_proba = self.model.predict_proba(X)[:, 1]
        train_metrics = self._compute_metrics(y, train_proba, prefix="train")

        self.training_metrics = {**cv_metrics, **train_metrics}

        # Feature importance
        self.training_metrics["feature_importance"] = self._get_feature_importance()

        # Log top features
        fi = self.training_metrics["feature_importance"]
        top5 = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("Top 5 features: " + ", ".join(f"{k}={v:.4f}" for k, v in top5))

        if save_path:
            self.save_model(save_path)

        return self.model

    def train_with_hyperopt(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        save_path: str | None = None,
        n_trials: int = 30,
    ):
        """Train with basic grid search over key hyperparameters.

        Tests combinations of max_depth, learning_rate, n_estimators.
        Selects the best by CV ROC-AUC.
        """
        import xgboost as xgb

        pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        param_grid = [
            {"max_depth": d, "learning_rate": lr, "n_estimators": n}
            for d in [4, 6, 8]
            for lr in [0.01, 0.05, 0.1]
            for n in [200, 500]
        ]

        best_auc = 0.0
        best_params = {}
        trials_run = 0

        for params in param_grid[:n_trials]:
            model = xgb.XGBClassifier(
                **params,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                scale_pos_weight=pos_weight,
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42,
            )

            oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
            auc = roc_auc_score(y, oof)

            trials_run += 1
            if auc > best_auc:
                best_auc = auc
                best_params = params
                logger.debug(f"Trial {trials_run}: AUC={auc:.4f} (new best) {params}")

        logger.info(
            f"Hyperopt: best CV AUC={best_auc:.4f} after {trials_run} trials | "
            f"depth={best_params['max_depth']}, lr={best_params['learning_rate']}, "
            f"trees={best_params['n_estimators']}"
        )

        return self.train(X, y, save_path=save_path, **best_params)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        case_ids: np.ndarray | None = None,
    ) -> dict:
        """Evaluate model with clinical-genomics-relevant metrics.

        Args:
            X: Feature matrix.
            y: True labels (0/1).
            case_ids: Optional per-sample case ID for Top-K evaluation.
                      When provided, computes Top-K accuracy per case.

        Returns:
            Dict with roc_auc, pr_auc, top_k_accuracy, mrr, etc.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        proba = self.model.predict_proba(X)[:, 1]
        metrics = self._compute_metrics(y, proba, prefix="test")

        # Top-K accuracy per case
        if case_ids is not None:
            topk = self._compute_topk_metrics(proba, y, case_ids)
            metrics.update(topk)

        # Feature importance
        metrics["feature_importance"] = self._get_feature_importance()

        self._log_metrics(metrics)
        return metrics

    def evaluate_topk_from_cases(
        self,
        cases: list[dict],
    ) -> dict:
        """Evaluate Top-K accuracy from a list of case dicts.

        Each case dict:
        {
            "case_id": str,
            "causal_gene": str,
            "gene_features": list[dict],  # All genes including causal
        }

        Returns Top-1, Top-5, Top-10 accuracy and MRR.
        """
        if self.model is None:
            raise ValueError("Model not trained")

        ranks = []

        for case in cases:
            causal = case["causal_gene"]
            gene_features = case["gene_features"]

            rows = [self._extract_row(g) for g in gene_features]
            df = pd.DataFrame(rows, columns=self.feature_columns)
            probas = self.model.predict_proba(df)[:, 1]

            # Find the causal gene's rank
            gene_names = [g.get("gene_symbol", "") for g in gene_features]
            scored = sorted(zip(gene_names, probas), key=lambda x: x[1], reverse=True)
            rank = None
            for i, (g, _) in enumerate(scored):
                if g == causal:
                    rank = i + 1
                    break

            ranks.append(rank)

        valid = [r for r in ranks if r is not None]
        n = len(valid)

        metrics = {
            "n_cases": len(cases),
            "n_evaluated": n,
            "top_1_accuracy": sum(1 for r in valid if r == 1) / max(n, 1),
            "top_5_accuracy": sum(1 for r in valid if r <= 5) / max(n, 1),
            "top_10_accuracy": sum(1 for r in valid if r <= 10) / max(n, 1),
            "top_20_accuracy": sum(1 for r in valid if r <= 20) / max(n, 1),
            "mrr": sum(1.0 / r for r in valid) / max(n, 1),
            "median_rank": float(np.median(valid)) if valid else None,
        }

        logger.info(
            f"Top-K: Top1={metrics['top_1_accuracy']:.1%}, "
            f"Top5={metrics['top_5_accuracy']:.1%}, "
            f"Top10={metrics['top_10_accuracy']:.1%}, "
            f"MRR={metrics['mrr']:.3f}, "
            f"MedianRank={metrics['median_rank']}"
        )

        return metrics

    # ── SHAP Interpretation ───────────────────────────────────────────────────

    def explain_with_shap(
        self,
        X: pd.DataFrame,
        max_samples: int = 500,
    ) -> dict:
        """Compute SHAP values for model interpretation.

        Returns:
            Dict with:
              - shap_values: np.ndarray (N x features)
              - feature_importance_shap: dict (feature -> mean |SHAP|)
              - top_features: list of (feature, importance) sorted
              - group_importance: dict (group_name -> sum of member importances)
        """
        if self.model is None:
            raise ValueError("Model not trained")

        # Subsample for speed
        if len(X) > max_samples:
            X_sample = X.sample(max_samples, random_state=42)
        else:
            X_sample = X

        sv = self._compute_shap_values(X_sample)

        # Mean absolute SHAP per feature
        mean_shap = np.abs(sv).mean(axis=0)
        fi_shap = dict(zip(self.feature_columns, mean_shap))

        # Sort by importance
        top_features = sorted(fi_shap.items(), key=lambda x: x[1], reverse=True)

        # Group importance
        group_imp = {}
        for group_name, members in FEATURE_GROUPS.items():
            group_imp[group_name] = sum(fi_shap.get(f, 0.0) for f in members)

        logger.info("SHAP feature importance (top 10):")
        for feat, imp in top_features[:10]:
            logger.info(f"  {feat}: {imp:.4f}")

        logger.info("SHAP group importance:")
        for group, imp in sorted(group_imp.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {group}: {imp:.4f}")

        return {
            "shap_values": sv,
            "feature_importance_shap": fi_shap,
            "top_features": top_features,
            "group_importance": group_imp,
        }

    def explain_single_gene(
        self, gene_features: dict,
    ) -> list[tuple[str, float]]:
        """Explain a single gene's prediction using SHAP.

        Returns list of (feature_name, shap_value) sorted by |value|.
        """
        if self.model is None:
            raise ValueError("Model not trained")

        row = self._extract_row(gene_features)
        df = pd.DataFrame([row], columns=self.feature_columns)

        sv = self._compute_shap_values(df)

        contributions = list(zip(self.feature_columns, sv[0]))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        return contributions

    def _compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values with fallback for xgboost/shap version issues.

        Strategy:
        1. Try shap.TreeExplainer (fastest, most accurate)
        2. Fallback to XGBoost native predict(pred_contribs=True)
        3. Fallback to feature importance approximation
        """
        # Method 1: shap library
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            sv = explainer.shap_values(X)
            if isinstance(sv, list):
                sv = sv[1]
            return sv
        except Exception:
            pass

        # Method 2: XGBoost native SHAP (pred_contribs)
        try:
            import xgboost as xgb
            dmat = xgb.DMatrix(X, feature_names=list(X.columns))
            contribs = self.model.get_booster().predict(dmat, pred_contribs=True)
            # Last column is the bias term; drop it
            return contribs[:, :-1]
        except Exception:
            pass

        # Method 3: Approximate from feature importance
        logger.warning("SHAP unavailable; using feature importance approximation")
        fi = self.model.feature_importances_
        proba = self.model.predict_proba(X)[:, 1]
        # Scale importance by prediction to approximate contribution direction
        sv = np.outer(proba - 0.5, fi)
        return sv

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_model(self, path: str) -> None:
        """Save trained model and metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        bundle = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "training_metrics": self.training_metrics,
        }

        with open(path, "wb") as f:
            pickle.dump(bundle, f)

        # Also save metrics as JSON for human inspection
        metrics_path = path.with_suffix(".metrics.json")
        json_safe = {
            k: v for k, v in self.training_metrics.items()
            if not isinstance(v, np.ndarray)
        }
        with open(metrics_path, "w") as f:
            json.dump(json_safe, f, indent=2, default=str)

        logger.info(f"Model saved to {path} (metrics: {metrics_path})")

    def load_model(self, path: str) -> None:
        """Load a trained model bundle."""
        with open(path, "rb") as f:
            bundle = pickle.load(f)

        self.model = bundle["model"]
        self.feature_columns = bundle.get("feature_columns", FEATURE_COLUMNS)
        self.training_metrics = bundle.get("training_metrics", {})

        logger.info(f"Model loaded from {path}")

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _compute_metrics(
        self, y_true: np.ndarray, y_proba: np.ndarray, prefix: str,
    ) -> dict:
        """Compute classification metrics."""
        y_pred = (y_proba >= 0.5).astype(int)

        auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0
        pr_auc = average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0
        acc = accuracy_score(y_true, y_pred)

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        return {
            f"{prefix}_roc_auc": round(auc, 4),
            f"{prefix}_pr_auc": round(pr_auc, 4),
            f"{prefix}_accuracy": round(acc, 4),
            f"{prefix}_precision_pos": round(report.get("1", {}).get("precision", 0.0), 4),
            f"{prefix}_recall_pos": round(report.get("1", {}).get("recall", 0.0), 4),
            f"{prefix}_f1_pos": round(report.get("1", {}).get("f1-score", 0.0), 4),
        }

    def _compute_topk_metrics(
        self, proba: np.ndarray, y: np.ndarray, case_ids: np.ndarray,
    ) -> dict:
        """Compute Top-K accuracy per case."""
        unique_cases = np.unique(case_ids)
        ranks = []

        for cid in unique_cases:
            mask = case_ids == cid
            case_proba = proba[mask]
            case_y = y[mask]

            if case_y.sum() == 0:
                continue  # No positive in this case

            # Rank by descending probability
            order = np.argsort(-case_proba)
            ranked_labels = case_y[order]

            # Find rank of first positive
            for i, label in enumerate(ranked_labels):
                if label == 1:
                    ranks.append(i + 1)
                    break

        n = len(ranks)
        if n == 0:
            return {}

        return {
            "topk_n_cases": n,
            "top_1_accuracy": sum(1 for r in ranks if r == 1) / n,
            "top_5_accuracy": sum(1 for r in ranks if r <= 5) / n,
            "top_10_accuracy": sum(1 for r in ranks if r <= 10) / n,
            "mrr": sum(1.0 / r for r in ranks) / n,
            "median_rank": float(np.median(ranks)),
        }

    def _get_feature_importance(self) -> dict:
        """Extract XGBoost native feature importance."""
        if self.model is None:
            return {}

        importances = self.model.feature_importances_
        return dict(zip(self.feature_columns, [round(float(v), 6) for v in importances]))

    def _log_metrics(self, metrics: dict) -> None:
        """Log evaluation metrics."""
        logger.info("=" * 50)
        logger.info("Model Evaluation Results")
        for key, val in metrics.items():
            if key == "feature_importance":
                continue
            if isinstance(val, float):
                logger.info(f"  {key}: {val:.4f}")
            else:
                logger.info(f"  {key}: {val}")
        logger.info("=" * 50)
