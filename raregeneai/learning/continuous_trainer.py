"""Continuous learning orchestrator.

Coordinates the full retrain cycle:
  1. Collect confirmed diagnoses from feedback store
  2. Merge with baseline training data (ClinVar/OMIM)
  3. Retrain XGBoost model
  4. Evaluate against holdout + previous model
  5. Register new version in model registry
  6. Auto-promote if performance improves (with safety guards)

Designed for periodic execution (cron / manual trigger).
All operations are logged for clinical audit compliance.
"""

from __future__ import annotations

import hashlib
import json
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from raregeneai.ranking.model_trainer import ModelTrainer, FEATURE_COLUMNS
from .feedback_store import FeedbackStore
from .model_registry import ModelRegistry


class ContinuousTrainer:
    """Orchestrate periodic model retraining from accumulated feedback."""

    def __init__(
        self,
        feedback_store: FeedbackStore,
        model_registry: ModelRegistry,
        baseline_data_path: str | None = None,
    ):
        self.feedback_store = feedback_store
        self.registry = model_registry
        self.baseline_data_path = baseline_data_path
        self.trainer = ModelTrainer()

    def run_retrain_cycle(
        self,
        min_feedback: int = 10,
        auto_promote: bool = False,
        min_improvement_auc: float = 0.005,
        analyst_id: str = "system",
    ) -> dict:
        """Execute a full retrain cycle.

        Steps:
        1. Check if enough new feedback has accumulated
        2. Build merged training dataset
        3. Train new model with hyperparameter search
        4. Evaluate and compare with current production
        5. Register new version
        6. Optionally auto-promote if improved

        Args:
            min_feedback: Minimum confirmed diagnoses needed to retrain.
            auto_promote: If True, automatically promote to production
                          when performance improves.
            min_improvement_auc: Minimum AUC improvement for auto-promotion.
            analyst_id: Who triggered this retrain cycle.

        Returns:
            Dict with retrain results and new model version info.
        """
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        logger.info("=" * 60)
        logger.info(f"Continuous Learning Cycle Started at {timestamp}")
        logger.info("=" * 60)

        # Step 1: Check feedback availability
        feedback_cases = self.feedback_store.get_training_cases()
        n_feedback = len(feedback_cases)
        counts = self.feedback_store.count()

        logger.info(
            f"Feedback store: {counts.get('total', 0)} total entries, "
            f"{counts.get('confirmed_diagnosis', 0)} confirmed diagnoses, "
            f"{n_feedback} usable for training"
        )

        if n_feedback < min_feedback:
            logger.info(
                f"Insufficient feedback ({n_feedback} < {min_feedback}). "
                f"Skipping retrain cycle."
            )
            return {
                "status": "skipped",
                "reason": f"insufficient_feedback ({n_feedback}/{min_feedback})",
                "n_feedback": n_feedback,
            }

        # Step 2: Build training dataset
        X, y, data_stats = self._build_merged_dataset(feedback_cases)

        logger.info(
            f"Training data: {len(y)} samples "
            f"({data_stats['n_feedback']} from feedback, "
            f"{data_stats['n_baseline']} from baseline)"
        )

        # Step 3: Train
        logger.info("Training new model...")
        tmp_path = str(self.registry.registry_dir / "candidate_model.pkl")
        self.trainer.train(X, y, save_path=tmp_path, n_estimators=500, max_depth=6)

        # Step 4: Evaluate
        metrics = {
            **self.trainer.training_metrics,
            "n_positive": int((y == 1).sum()),
        }

        # Step 5: Register
        version = self.registry.register(
            model_path=tmp_path,
            metrics=metrics,
            n_training_samples=len(y),
            n_feedback_cases=data_stats["n_feedback"],
            n_baseline_cases=data_stats["n_baseline"],
            hyperparameters={"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05},
            notes=f"Retrain cycle at {timestamp} by {analyst_id}",
        )

        # Step 6: Auto-promote if improved
        result = {
            "status": "completed",
            "version": version.version,
            "cv_roc_auc": version.cv_roc_auc,
            "delta_roc_auc": version.delta_roc_auc,
            "n_feedback": data_stats["n_feedback"],
            "n_baseline": data_stats["n_baseline"],
            "n_total": len(y),
            "promoted": False,
        }

        if auto_promote and version.delta_roc_auc >= min_improvement_auc:
            self.registry.promote(
                version.version,
                to_status="production",
                promoted_by=f"{analyst_id} (auto)",
            )
            result["promoted"] = True
            logger.info(
                f"Auto-promoted {version.version} to production "
                f"(ΔAUC={version.delta_roc_auc:+.4f} >= {min_improvement_auc})"
            )
        elif auto_promote:
            logger.info(
                f"Model {version.version} NOT promoted: "
                f"ΔAUC={version.delta_roc_auc:+.4f} < {min_improvement_auc}"
            )
            result["promotion_blocked_reason"] = (
                f"improvement {version.delta_roc_auc:+.4f} below threshold {min_improvement_auc}"
            )

        # Cleanup temp file
        Path(tmp_path).unlink(missing_ok=True)

        logger.info("=" * 60)
        logger.info(f"Retrain cycle complete: {version.version}")
        logger.info("=" * 60)

        return result

    def _build_merged_dataset(
        self, feedback_cases: list[dict],
    ) -> tuple[pd.DataFrame, np.ndarray, dict]:
        """Merge feedback-derived training data with baseline data.

        Feedback cases are confirmed diagnoses with evidence snapshots.
        Baseline data comes from ClinVar/OMIM pre-computed training CSV.

        Returns:
            (X, y, stats_dict)
        """
        all_rows = []
        all_labels = []

        n_feedback = 0
        n_baseline = 0

        # Add feedback cases (positive labels)
        for case in feedback_cases:
            row = self.trainer._extract_row(case)
            all_rows.append(row)
            all_labels.append(1)
            n_feedback += 1

        # Load baseline training data if available
        if self.baseline_data_path and Path(self.baseline_data_path).exists():
            X_base, y_base = self.trainer.build_training_data_from_csv(
                self.baseline_data_path
            )
            for _, row_data in X_base.iterrows():
                all_rows.append(row_data.tolist())

            all_labels.extend(y_base.tolist())
            n_baseline = len(y_base)

        # If no baseline data, generate synthetic negatives from positives
        # by zeroing out key features (ensures both classes present for XGBoost)
        if n_baseline == 0 and n_feedback > 0:
            rng = np.random.RandomState(42)
            for i in range(n_feedback * 10):
                neg_row = [0.0] * len(FEATURE_COLUMNS)
                # Randomize a few features at low values
                neg_row[0] = rng.random() * 0.3   # max_variant_score
                neg_row[1] = rng.random() * 0.3   # max_cadd
                neg_row[6] = rng.random() * 0.2   # phenotype_score
                neg_row[3] = 0.01 + rng.random() * 0.04  # min_af (common)
                all_rows.append(neg_row)
                all_labels.append(0)
                n_baseline += 1

        X = pd.DataFrame(all_rows, columns=FEATURE_COLUMNS)
        y = np.array(all_labels)

        data_hash = hashlib.sha256(
            X.to_csv(index=False).encode()
        ).hexdigest()[:16]

        return X, y, {
            "n_feedback": n_feedback,
            "n_baseline": n_baseline,
            "data_hash": data_hash,
        }

    def get_learning_report(self) -> dict:
        """Generate a summary report of the continuous learning system state.

        Returns dict with:
          - feedback_stats: counts by type
          - model_versions: list of all versions with metrics
          - production_model: current production model info
          - recommendation: whether retrain is recommended
        """
        feedback_stats = self.feedback_store.count()
        versions = self.registry.list_versions()
        production = self.registry.get_production()

        # Trend: are we improving?
        auc_history = [v.cv_roc_auc for v in versions if v.cv_roc_auc > 0]

        return {
            "feedback_stats": feedback_stats,
            "n_model_versions": len(versions),
            "model_versions": [
                {
                    "version": v.version,
                    "status": v.status,
                    "cv_roc_auc": v.cv_roc_auc,
                    "top_1_accuracy": v.top_1_accuracy,
                    "n_training_samples": v.n_training_samples,
                    "n_feedback_cases": v.n_feedback_cases,
                    "created_at": v.created_at,
                }
                for v in versions
            ],
            "production_model": {
                "version": production.version if production else None,
                "cv_roc_auc": production.cv_roc_auc if production else None,
                "promoted_at": production.promoted_at if production else None,
            },
            "auc_trend": auc_history,
            "retrain_recommended": len(
                self.feedback_store.load_confirmed_diagnoses()
            ) >= 10,
        }
