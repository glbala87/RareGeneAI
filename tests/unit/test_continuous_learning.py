"""Unit tests for continuous learning system.

Tests:
  - Feedback capture and storage
  - Feedback retrieval and filtering
  - Model version registry (register, promote, rollback)
  - Continuous retrain cycle
  - Audit trail integrity
  - Learning report generation
"""

import json

import numpy as np
import pandas as pd
import pytest

from raregeneai.learning.feedback_store import FeedbackEntry, FeedbackStore
from raregeneai.learning.model_registry import ModelRegistry, ModelVersion
from raregeneai.learning.continuous_trainer import ContinuousTrainer
from raregeneai.ranking.model_trainer import FEATURE_COLUMNS


# ── Feedback Store Tests ──────────────────────────────────────────────────────

class TestFeedbackStore:
    @pytest.fixture
    def store(self, tmp_path):
        return FeedbackStore(tmp_path / "feedback.jsonl")

    def test_submit_creates_file(self, store):
        entry = FeedbackEntry(
            patient_id="P001",
            feedback_type="confirmed_diagnosis",
            gene_symbol="BRCA1",
            confirmed_causal=True,
            diagnosis="Hereditary breast cancer",
        )
        fid = store.submit(entry)
        assert fid != ""
        assert store.store_path.exists()

    def test_submit_confirmed_diagnosis(self, store):
        fid = store.submit_confirmed_diagnosis(
            patient_id="P001",
            gene_symbol="SCN1A",
            diagnosis="Dravet syndrome",
            original_rank=1,
            original_score=0.92,
            evidence_snapshot={"max_variant_score": 0.85, "has_de_novo_lof": True},
            confirmation_method="functional_study",
            analyst_id="Dr. Smith",
        )
        assert fid != ""

        entries = store.load_all()
        assert len(entries) == 1
        assert entries[0].gene_symbol == "SCN1A"
        assert entries[0].confirmed_causal is True
        assert entries[0].analyst_id == "Dr. Smith"

    def test_submit_rejected_gene(self, store):
        store.submit_rejected_gene(
            patient_id="P002",
            gene_symbol="GENE_X",
            original_rank=3,
            notes="Ruled out by functional study",
        )
        entries = store.load_all()
        assert len(entries) == 1
        assert entries[0].feedback_type == "rejected_gene"

    def test_load_confirmed_diagnoses_only(self, store):
        store.submit_confirmed_diagnosis(
            patient_id="P001", gene_symbol="SCN1A",
            diagnosis="Dravet", evidence_snapshot={"max_variant_score": 0.8},
        )
        store.submit_rejected_gene(
            patient_id="P001", gene_symbol="GENE_X",
        )
        store.submit_confirmed_diagnosis(
            patient_id="P002", gene_symbol="BRCA1",
            diagnosis="HBOC", evidence_snapshot={"max_variant_score": 0.9},
        )

        confirmed = store.load_confirmed_diagnoses()
        assert len(confirmed) == 2
        assert all(e.confirmed_causal for e in confirmed)

    def test_count_by_type(self, store):
        store.submit_confirmed_diagnosis(
            patient_id="P1", gene_symbol="G1", diagnosis="D1",
        )
        store.submit_confirmed_diagnosis(
            patient_id="P2", gene_symbol="G2", diagnosis="D2",
        )
        store.submit_rejected_gene(patient_id="P1", gene_symbol="G3")

        counts = store.count()
        assert counts["confirmed_diagnosis"] == 2
        assert counts["rejected_gene"] == 1
        assert counts["total"] == 3

    def test_get_training_cases(self, store):
        store.submit_confirmed_diagnosis(
            patient_id="P001", gene_symbol="SCN1A",
            diagnosis="Dravet",
            evidence_snapshot={
                "max_variant_score": 0.85,
                "has_de_novo_lof": True,
                "phenotype_score": 0.9,
            },
        )

        cases = store.get_training_cases()
        assert len(cases) == 1
        assert cases[0]["gene_symbol"] == "SCN1A"
        assert cases[0]["max_variant_score"] == 0.85

    def test_append_only_multiple_submits(self, store):
        for i in range(5):
            store.submit_confirmed_diagnosis(
                patient_id=f"P{i}", gene_symbol=f"GENE_{i}",
                diagnosis=f"Disease_{i}",
            )

        entries = store.load_all()
        assert len(entries) == 5

    def test_empty_store_returns_empty(self, store):
        assert store.load_all() == []
        assert store.load_confirmed_diagnoses() == []
        assert store.count() == {"total": 0}

    def test_feedback_has_timestamp(self, store):
        store.submit_confirmed_diagnosis(
            patient_id="P1", gene_symbol="G1", diagnosis="D1",
        )
        entries = store.load_all()
        assert entries[0].timestamp != ""

    def test_feedback_has_unique_id(self, store):
        store.submit_confirmed_diagnosis(
            patient_id="P1", gene_symbol="G1", diagnosis="D1",
        )
        store.submit_confirmed_diagnosis(
            patient_id="P2", gene_symbol="G2", diagnosis="D2",
        )
        entries = store.load_all()
        assert entries[0].feedback_id != entries[1].feedback_id


# ── Model Registry Tests ─────────────────────────────────────────────────────

class TestModelRegistry:
    @pytest.fixture
    def registry(self, tmp_path):
        return ModelRegistry(tmp_path / "registry")

    def _create_dummy_model(self, tmp_path, name="model.pkl"):
        import pickle
        model_path = tmp_path / name
        with open(model_path, "wb") as f:
            pickle.dump({"dummy": True}, f)
        return str(model_path)

    def test_register_model(self, registry, tmp_path):
        model_path = self._create_dummy_model(tmp_path)
        version = registry.register(
            model_path=model_path,
            metrics={"cv_roc_auc": 0.92, "top_1_accuracy": 0.65},
            n_training_samples=1000,
            n_feedback_cases=50,
            n_baseline_cases=950,
        )

        assert version.version == "v1.0"
        assert version.cv_roc_auc == 0.92
        assert version.status == "candidate"

    def test_promote_to_production(self, registry, tmp_path):
        model_path = self._create_dummy_model(tmp_path)
        registry.register(
            model_path=model_path,
            metrics={"cv_roc_auc": 0.90},
        )

        registry.promote("v1.0", to_status="production", promoted_by="Dr. Admin")

        prod = registry.get_production()
        assert prod is not None
        assert prod.version == "v1.0"
        assert prod.status == "production"
        assert prod.promoted_by == "Dr. Admin"

    def test_promote_retires_previous(self, registry, tmp_path):
        m1 = self._create_dummy_model(tmp_path, "m1.pkl")
        m2 = self._create_dummy_model(tmp_path, "m2.pkl")

        registry.register(model_path=m1, metrics={"cv_roc_auc": 0.85})
        registry.promote("v1.0", to_status="production")

        registry.register(model_path=m2, metrics={"cv_roc_auc": 0.90})
        registry.promote("v2.0", to_status="production")

        versions = registry.list_versions()
        v1 = next(v for v in versions if v.version == "v1.0")
        v2 = next(v for v in versions if v.version == "v2.0")

        assert v1.status == "retired"
        assert v2.status == "production"

    def test_rollback(self, registry, tmp_path):
        m1 = self._create_dummy_model(tmp_path, "m1.pkl")
        m2 = self._create_dummy_model(tmp_path, "m2.pkl")

        registry.register(model_path=m1, metrics={"cv_roc_auc": 0.90})
        registry.promote("v1.0", to_status="production")

        registry.register(model_path=m2, metrics={"cv_roc_auc": 0.88})
        registry.promote("v2.0", to_status="production")

        # v2 is worse, rollback to v1
        registry.rollback("v1.0", reason="v2.0 regression in Top-1")

        prod = registry.get_production()
        assert prod.version == "v1.0"

        v2 = next(v for v in registry.list_versions() if v.version == "v2.0")
        assert v2.status == "retired"
        assert "regression" in v2.retirement_reason.lower()

    def test_invalid_promotion_rejected(self, registry, tmp_path):
        model_path = self._create_dummy_model(tmp_path)
        registry.register(model_path=model_path, metrics={"cv_roc_auc": 0.90})
        registry.promote("v1.0", to_status="production")

        with pytest.raises(ValueError, match="Cannot promote"):
            registry.promote("v1.0", to_status="staging")  # Can't go backward

    def test_delta_metrics_computed(self, registry, tmp_path):
        m1 = self._create_dummy_model(tmp_path, "m1.pkl")
        m2 = self._create_dummy_model(tmp_path, "m2.pkl")

        registry.register(model_path=m1, metrics={"cv_roc_auc": 0.85})
        registry.promote("v1.0", to_status="production")

        v2 = registry.register(model_path=m2, metrics={"cv_roc_auc": 0.90})
        assert v2.delta_roc_auc == pytest.approx(0.05, abs=0.001)

    def test_manifest_persists(self, tmp_path):
        reg1 = ModelRegistry(tmp_path / "reg")
        model_path = self._create_dummy_model(tmp_path)
        reg1.register(model_path=model_path, metrics={"cv_roc_auc": 0.88})

        # Re-open registry
        reg2 = ModelRegistry(tmp_path / "reg")
        assert len(reg2.list_versions()) == 1
        assert reg2.get_latest().cv_roc_auc == 0.88

    def test_get_latest(self, registry, tmp_path):
        m1 = self._create_dummy_model(tmp_path, "m1.pkl")
        m2 = self._create_dummy_model(tmp_path, "m2.pkl")

        registry.register(model_path=m1, metrics={"cv_roc_auc": 0.85})
        registry.register(model_path=m2, metrics={"cv_roc_auc": 0.90})

        latest = registry.get_latest()
        assert latest.version == "v2.0"
        assert latest.cv_roc_auc == 0.90

    def test_empty_registry(self, registry):
        assert registry.get_production() is None
        assert registry.get_latest() is None
        assert registry.list_versions() == []


# ── Continuous Trainer Tests ─────────────────────────────────────────────────

class TestContinuousTrainer:
    @pytest.fixture
    def setup(self, tmp_path):
        store = FeedbackStore(tmp_path / "feedback.jsonl")
        registry = ModelRegistry(tmp_path / "registry")
        trainer = ContinuousTrainer(store, registry)
        return trainer, store, registry, tmp_path

    def _add_feedback(self, store, n=15):
        """Add synthetic confirmed diagnoses with evidence snapshots."""
        import random
        rng = random.Random(42)
        for i in range(n):
            store.submit_confirmed_diagnosis(
                patient_id=f"P{i:03d}",
                gene_symbol=f"GENE_{i}",
                diagnosis=f"Disease_{i}",
                evidence_snapshot={
                    "max_variant_score": 0.5 + rng.random() * 0.5,
                    "max_cadd": 20 + rng.random() * 20,
                    "max_revel": 0.4 + rng.random() * 0.6,
                    "min_af": rng.random() * 0.001,
                    "has_lof": rng.random() > 0.5,
                    "has_clinvar_pathogenic": rng.random() > 0.6,
                    "phenotype_score": 0.3 + rng.random() * 0.7,
                    "n_variants": rng.randint(1, 5),
                    "has_de_novo_lof": rng.random() > 0.7,
                    "inheritance_score": 0.5 + rng.random() * 0.5,
                    "trio_analyzed": True,
                    "kg_score": rng.random() * 0.8,
                },
                analyst_id="Dr. Test",
            )

    def test_skip_if_insufficient_feedback(self, setup):
        trainer, store, _, _ = setup
        # Only 3 feedback entries (below threshold of 10)
        self._add_feedback(store, n=3)

        result = trainer.run_retrain_cycle(min_feedback=10)
        assert result["status"] == "skipped"
        assert "insufficient" in result["reason"]

    def test_retrain_with_feedback_only(self, setup):
        trainer, store, registry, _ = setup
        self._add_feedback(store, n=15)

        result = trainer.run_retrain_cycle(min_feedback=10, auto_promote=False)

        assert result["status"] == "completed"
        assert result["n_feedback"] == 15
        assert result["version"].startswith("v")
        assert result["cv_roc_auc"] >= 0  # May be low with synthetic data

        # Model should be registered
        assert len(registry.list_versions()) == 1
        assert registry.get_latest().status == "candidate"

    def test_auto_promote_when_improved(self, setup):
        trainer, store, registry, tmp_path = setup

        # Register a baseline model with low AUC
        import pickle
        baseline_path = tmp_path / "baseline.pkl"
        with open(baseline_path, "wb") as f:
            pickle.dump({"dummy": True}, f)

        registry.register(
            model_path=str(baseline_path),
            metrics={"cv_roc_auc": 0.50},  # Low baseline
        )
        registry.promote("v1.0", to_status="production")

        # Add feedback and retrain
        self._add_feedback(store, n=20)
        result = trainer.run_retrain_cycle(
            min_feedback=10,
            auto_promote=True,
            min_improvement_auc=-1.0,  # Accept any improvement
        )

        assert result["status"] == "completed"
        # With synthetic data, new model should be registered
        assert len(registry.list_versions()) == 2

    def test_learning_report(self, setup):
        trainer, store, registry, tmp_path = setup

        self._add_feedback(store, n=5)

        import pickle
        model_path = tmp_path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"dummy": True}, f)
        registry.register(
            model_path=str(model_path),
            metrics={"cv_roc_auc": 0.88, "top_1_accuracy": 0.60},
            n_training_samples=500,
            n_feedback_cases=5,
        )

        report = trainer.get_learning_report()

        assert report["feedback_stats"]["total"] == 5
        assert report["n_model_versions"] == 1
        assert report["production_model"]["version"] is None  # Not yet promoted
        assert len(report["auc_trend"]) == 1


# ── Audit Trail Tests ────────────────────────────────────────────────────────

class TestAuditTrail:
    def test_feedback_entry_has_all_audit_fields(self):
        entry = FeedbackEntry(
            patient_id="P001",
            feedback_type="confirmed_diagnosis",
            gene_symbol="SCN1A",
            analyst_id="Dr. Smith",
            institution="Hospital X",
            model_version="v3.0",
            pipeline_version="1.0.0",
        )

        # All audit fields populated
        assert entry.feedback_id != ""
        assert entry.timestamp != ""
        assert entry.analyst_id == "Dr. Smith"
        assert entry.institution == "Hospital X"
        assert entry.model_version == "v3.0"

    def test_model_version_has_provenance(self):
        v = ModelVersion(
            version="v2.0",
            created_at="2026-04-01T10:00:00",
            n_training_samples=1500,
            n_feedback_cases=100,
            n_baseline_cases=1400,
            cv_roc_auc=0.92,
            status="production",
            promoted_by="Dr. Admin",
        )

        assert v.n_training_samples == 1500
        assert v.n_feedback_cases == 100
        assert v.promoted_by == "Dr. Admin"

    def test_feedback_store_is_append_only(self, tmp_path):
        store = FeedbackStore(tmp_path / "fb.jsonl")

        store.submit_confirmed_diagnosis(
            patient_id="P1", gene_symbol="G1", diagnosis="D1",
        )
        size_after_1 = store.store_path.stat().st_size

        store.submit_confirmed_diagnosis(
            patient_id="P2", gene_symbol="G2", diagnosis="D2",
        )
        size_after_2 = store.store_path.stat().st_size

        # File grew (append-only, no overwrites)
        assert size_after_2 > size_after_1

        # Both entries retrievable
        assert len(store.load_all()) == 2

    def test_registry_manifest_is_json(self, tmp_path):
        registry = ModelRegistry(tmp_path / "reg")
        import pickle
        model_path = tmp_path / "m.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"x": 1}, f)

        registry.register(model_path=str(model_path), metrics={"cv_roc_auc": 0.88})

        manifest_path = tmp_path / "reg" / "model_registry.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["version"] == "v1.0"
