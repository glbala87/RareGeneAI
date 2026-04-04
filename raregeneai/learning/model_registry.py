"""Model version registry with audit trail.

Tracks every model version produced by the continuous learning system:
  - Model artifacts (pickle bundles)
  - Training metadata (data size, hyperparameters)
  - Performance metrics (ROC-AUC, Top-K)
  - Promotion status (candidate → staging → production)
  - Rollback capability

The registry is a JSON manifest file alongside versioned model files.
This design supports CAP/CLIA requirements for traceable model lineage.
"""

from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path

from pydantic import BaseModel, Field
from loguru import logger


class ModelVersion(BaseModel):
    """Metadata for a single model version."""
    version: str = ""                    # Semantic version: "v1.0", "v1.1", ...
    created_at: str = ""
    model_path: str = ""                 # Path to .pkl bundle

    # Training provenance
    n_training_samples: int = 0
    n_positive_samples: int = 0
    n_feedback_cases: int = 0            # How many came from feedback
    n_baseline_cases: int = 0            # How many from ClinVar/OMIM baseline
    training_data_hash: str = ""         # SHA256 of training data for reproducibility

    # Hyperparameters
    hyperparameters: dict = Field(default_factory=dict)

    # Performance metrics
    cv_roc_auc: float = 0.0
    cv_pr_auc: float = 0.0
    top_1_accuracy: float = 0.0
    top_5_accuracy: float = 0.0
    top_10_accuracy: float = 0.0
    mrr: float = 0.0

    # Comparison with previous version
    delta_roc_auc: float = 0.0           # Improvement over previous
    delta_top_1: float = 0.0

    # Status
    status: str = "candidate"            # "candidate", "staging", "production", "retired"
    promoted_at: str = ""
    promoted_by: str = ""
    retirement_reason: str = ""

    # Audit
    notes: str = ""


class ModelRegistry:
    """Version-controlled model registry."""

    MANIFEST_FILE = "model_registry.json"

    def __init__(self, registry_dir: str | Path):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._manifest: list[ModelVersion] = []
        self._load_manifest()

    def _manifest_path(self) -> Path:
        return self.registry_dir / self.MANIFEST_FILE

    def _load_manifest(self) -> None:
        path = self._manifest_path()
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            self._manifest = [ModelVersion(**v) for v in data]
        else:
            self._manifest = []

    def _save_manifest(self) -> None:
        path = self._manifest_path()
        with open(path, "w") as f:
            json.dump([v.model_dump() for v in self._manifest], f, indent=2)

    def register(
        self,
        model_path: str,
        metrics: dict,
        n_training_samples: int = 0,
        n_feedback_cases: int = 0,
        n_baseline_cases: int = 0,
        hyperparameters: dict | None = None,
        notes: str = "",
    ) -> ModelVersion:
        """Register a new model version.

        Copies the model artifact into the registry directory,
        assigns a version number, and records metrics.
        """
        version_num = len(self._manifest) + 1
        version_str = f"v{version_num}.0"
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")

        # Copy model artifact to registry
        src = Path(model_path)
        dest = self.registry_dir / f"model_{version_str}.pkl"
        shutil.copy2(src, dest)

        # Compare with previous production model
        prev = self.get_production()
        delta_auc = metrics.get("cv_roc_auc", 0.0) - (prev.cv_roc_auc if prev else 0.0)
        delta_top1 = metrics.get("top_1_accuracy", 0.0) - (prev.top_1_accuracy if prev else 0.0)

        version = ModelVersion(
            version=version_str,
            created_at=timestamp,
            model_path=str(dest),
            n_training_samples=n_training_samples,
            n_positive_samples=metrics.get("n_positive", 0),
            n_feedback_cases=n_feedback_cases,
            n_baseline_cases=n_baseline_cases,
            hyperparameters=hyperparameters or {},
            cv_roc_auc=metrics.get("cv_roc_auc", 0.0),
            cv_pr_auc=metrics.get("cv_pr_auc", 0.0),
            top_1_accuracy=metrics.get("top_1_accuracy", 0.0),
            top_5_accuracy=metrics.get("top_5_accuracy", 0.0),
            top_10_accuracy=metrics.get("top_10_accuracy", 0.0),
            mrr=metrics.get("mrr", 0.0),
            delta_roc_auc=round(delta_auc, 4),
            delta_top_1=round(delta_top1, 4),
            status="candidate",
            notes=notes,
        )

        self._manifest.append(version)
        self._save_manifest()

        logger.info(
            f"Registered model {version_str}: "
            f"AUC={version.cv_roc_auc:.4f} (Δ={delta_auc:+.4f}), "
            f"Top1={version.top_1_accuracy:.1%}, "
            f"trained on {n_training_samples} samples "
            f"({n_feedback_cases} feedback + {n_baseline_cases} baseline)"
        )

        return version

    def promote(
        self, version: str, to_status: str = "production", promoted_by: str = "",
    ) -> None:
        """Promote a model version (candidate → staging → production).

        When promoting to production, the previous production model is retired.
        """
        target = self._find_version(version)
        if not target:
            raise ValueError(f"Version {version} not found")

        valid_transitions = {
            "candidate": ["staging", "production"],
            "staging": ["production", "retired"],
            "production": ["retired"],
        }
        allowed = valid_transitions.get(target.status, [])
        if to_status not in allowed:
            raise ValueError(
                f"Cannot promote {version} from '{target.status}' to '{to_status}'. "
                f"Allowed: {allowed}"
            )

        # Retire current production if promoting to production
        if to_status == "production":
            current = self.get_production()
            if current and current.version != version:
                current.status = "retired"
                current.retirement_reason = f"Replaced by {version}"

        target.status = to_status
        target.promoted_at = datetime.datetime.now().isoformat(timespec="seconds")
        target.promoted_by = promoted_by

        self._save_manifest()
        logger.info(f"Model {version} promoted to {to_status} by {promoted_by or 'system'}")

    def rollback(self, to_version: str, reason: str = "") -> None:
        """Rollback production to a previous version."""
        target = self._find_version(to_version)
        if not target:
            raise ValueError(f"Version {to_version} not found")

        current = self.get_production()
        if current:
            current.status = "retired"
            current.retirement_reason = f"Rolled back to {to_version}: {reason}"

        target.status = "production"
        target.promoted_at = datetime.datetime.now().isoformat(timespec="seconds")

        self._save_manifest()
        logger.info(f"Rolled back to {to_version}: {reason}")

    def get_production(self) -> ModelVersion | None:
        """Get the current production model."""
        for v in reversed(self._manifest):
            if v.status == "production":
                return v
        return None

    def get_latest(self) -> ModelVersion | None:
        """Get the most recently registered model."""
        return self._manifest[-1] if self._manifest else None

    def list_versions(self) -> list[ModelVersion]:
        """List all model versions."""
        return list(self._manifest)

    def should_retrain(
        self, min_new_feedback: int = 10, min_improvement: float = 0.01,
    ) -> bool:
        """Determine if retraining is warranted based on accumulated feedback.

        Criteria:
        1. At least min_new_feedback new confirmed diagnoses since last training
        2. No model exists yet (first training)
        """
        if not self._manifest:
            return True  # No model at all

        latest = self.get_latest()
        if not latest:
            return True

        # Count feedback entries since last training (approximation)
        # In production, compare feedback timestamps vs model creation time
        return True  # Always allow retraining; gate on feedback count externally

    def _find_version(self, version: str) -> ModelVersion | None:
        for v in self._manifest:
            if v.version == version:
                return v
        return None
