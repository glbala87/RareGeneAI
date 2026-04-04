"""Unit tests for XGBoost ML ranking pipeline.

Tests:
  - Feature extraction (44 features from all evidence layers)
  - Training data construction
  - XGBoost training + CV evaluation
  - Hyperparameter tuning
  - Top-K accuracy (Top-1, Top-5, Top-10, MRR)
  - SHAP interpretation (feature + group importance)
  - Model persistence (save/load bundle)
  - Gene ranker ML integration
  - Single-gene SHAP explanation
"""

import pickle
import tempfile

import numpy as np
import pandas as pd
import pytest

from raregeneai.ranking.model_trainer import (
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    ModelTrainer,
)


# ── Synthetic training data ───────────────────────────────────────────────────

def _make_positive_case(**overrides) -> dict:
    """Create a positive (causal gene) feature dict."""
    base = {
        "gene_symbol": "CAUSAL_GENE",
        "max_variant_score": 0.85,
        "max_cadd": 32.0,
        "max_revel": 0.82,
        "min_af": 0.0001,
        "has_lof": True,
        "has_clinvar_pathogenic": True,
        "phenotype_score": 0.75,
        "n_variants": 2,
        "n_inheritance_modes": 1,
        "is_known_disease_gene": True,
        "max_spliceai": 0.0,
        "has_de_novo": True,
        "has_de_novo_lof": True,
        "inheritance_score": 1.0,
        "trio_analyzed": True,
        "kg_score": 0.6,
        "kg_ppi_neighbors": 3,
        "kg_n_diseases": 2,
        "multi_omics_score": 0.7,
        "n_evidence_layers": 3,
        "has_expression_outlier": True,
        "expression_score": 0.8,
    }
    base.update(overrides)
    return base


def _make_negative_case(**overrides) -> dict:
    """Create a negative (non-causal gene) feature dict."""
    base = {
        "gene_symbol": "BYSTANDER",
        "max_variant_score": 0.2,
        "max_cadd": 12.0,
        "max_revel": 0.15,
        "min_af": 0.02,
        "has_lof": False,
        "has_clinvar_pathogenic": False,
        "phenotype_score": 0.1,
        "n_variants": 1,
        "n_inheritance_modes": 0,
        "is_known_disease_gene": False,
        "has_de_novo": False,
        "has_de_novo_lof": False,
        "inheritance_score": 0.3,
        "trio_analyzed": True,
        "kg_score": 0.05,
        "multi_omics_score": 0.0,
    }
    base.update(overrides)
    return base


def _build_synthetic_dataset(n_cases: int = 50, neg_per_case: int = 20):
    """Build a synthetic training dataset with realistic class separation."""
    rng = np.random.RandomState(42)
    cases = []

    for i in range(n_cases):
        pos = _make_positive_case(
            max_variant_score=0.6 + rng.random() * 0.4,
            phenotype_score=0.5 + rng.random() * 0.5,
            max_cadd=20 + rng.random() * 20,
            has_de_novo_lof=rng.random() > 0.5,
            kg_score=0.3 + rng.random() * 0.7,
        )

        negs = []
        for j in range(neg_per_case):
            negs.append(_make_negative_case(
                max_variant_score=rng.random() * 0.4,
                phenotype_score=rng.random() * 0.3,
                max_cadd=rng.random() * 15,
                kg_score=rng.random() * 0.2,
            ))

        pos["negative_genes"] = negs
        cases.append(pos)

    return cases


# ── Feature Column Tests ─────────────────────────────────────────────────────

class TestFeatureColumns:
    def test_feature_count(self):
        assert len(FEATURE_COLUMNS) == 44

    def test_feature_groups_cover_all(self):
        all_grouped = []
        for members in FEATURE_GROUPS.values():
            all_grouped.extend(members)
        assert len(all_grouped) == len(FEATURE_COLUMNS)
        assert set(all_grouped) == set(FEATURE_COLUMNS)

    def test_feature_groups_exist(self):
        assert "Variant pathogenicity" in FEATURE_GROUPS
        assert "Trio inheritance" in FEATURE_GROUPS
        assert "Multi-omics" in FEATURE_GROUPS
        assert "Knowledge graph" in FEATURE_GROUPS


# ── Training Data Construction Tests ──────────────────────────────────────────

class TestTrainingData:
    def test_build_training_data(self):
        trainer = ModelTrainer()
        cases = _build_synthetic_dataset(n_cases=5, neg_per_case=10)
        X, y = trainer.build_training_data(cases, negative_genes_per_case=10)

        assert X.shape[1] == 44  # All features
        assert len(y) == 5 + 5 * 10  # 5 positive + 50 negative
        assert (y == 1).sum() == 5
        assert (y == 0).sum() == 50

    def test_feature_normalization(self):
        trainer = ModelTrainer()
        row = trainer._extract_row({
            "max_cadd": 40.0,  # Should normalize to 1.0
            "n_variants": 30,  # Should cap at 20/20 = 1.0
            "n_evidence_layers": 10,  # Should cap at 5/5 = 1.0
        })
        df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

        assert df["max_cadd"].iloc[0] == 1.0  # 40/40
        assert df["n_variants"].iloc[0] == 1.0  # capped

    def test_build_from_csv(self, tmp_path):
        csv_file = tmp_path / "train.csv"
        cols = FEATURE_COLUMNS + ["label"]
        n = 20
        rng = np.random.RandomState(42)
        data = rng.random((n, len(FEATURE_COLUMNS)))
        labels = np.array([1] * 5 + [0] * 15)

        df = pd.DataFrame(data, columns=FEATURE_COLUMNS)
        df["label"] = labels
        df.to_csv(csv_file, index=False)

        trainer = ModelTrainer()
        X, y = trainer.build_training_data_from_csv(str(csv_file))

        assert X.shape == (n, 44)
        assert (y == 1).sum() == 5

    def test_missing_features_filled_with_zero(self, tmp_path):
        csv_file = tmp_path / "sparse.csv"
        df = pd.DataFrame({
            "max_variant_score": [0.8, 0.1],
            "phenotype_score": [0.9, 0.1],
            "label": [1, 0],
        })
        df.to_csv(csv_file, index=False)

        trainer = ModelTrainer()
        X, y = trainer.build_training_data_from_csv(str(csv_file))

        assert X.shape[1] == 44
        # Missing columns should be 0
        assert X["has_de_novo_lof"].iloc[0] == 0


# ── XGBoost Training Tests ───────────────────────────────────────────────────

class TestXGBoostTraining:
    @pytest.fixture
    def trained_trainer(self):
        trainer = ModelTrainer()
        cases = _build_synthetic_dataset(n_cases=30, neg_per_case=10)
        X, y = trainer.build_training_data(cases, negative_genes_per_case=10)
        trainer.train(X, y, n_estimators=50, max_depth=4)
        return trainer, X, y

    def test_train_produces_model(self, trained_trainer):
        trainer, X, y = trained_trainer
        assert trainer.model is not None

    def test_cv_roc_auc_above_random(self, trained_trainer):
        trainer, X, y = trained_trainer
        assert trainer.training_metrics["cv_roc_auc"] > 0.5

    def test_feature_importance_populated(self, trained_trainer):
        trainer, X, y = trained_trainer
        fi = trainer.training_metrics.get("feature_importance", {})
        assert len(fi) == 44
        # At least one feature should have nonzero importance
        assert max(fi.values()) > 0

    def test_predictions_are_probabilities(self, trained_trainer):
        trainer, X, y = trained_trainer
        proba = trainer.model.predict_proba(X)[:, 1]
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_positive_cases_score_higher(self, trained_trainer):
        """Positive cases should have higher mean probability than negatives."""
        trainer, X, y = trained_trainer
        proba = trainer.model.predict_proba(X)[:, 1]

        pos_mean = proba[y == 1].mean()
        neg_mean = proba[y == 0].mean()
        assert pos_mean > neg_mean


# ── Evaluation + Top-K Tests ─────────────────────────────────────────────────

class TestEvaluation:
    @pytest.fixture
    def trained_setup(self):
        trainer = ModelTrainer()
        cases = _build_synthetic_dataset(n_cases=30, neg_per_case=10)
        X, y = trainer.build_training_data(cases, negative_genes_per_case=10)
        trainer.train(X, y, n_estimators=50, max_depth=4)
        return trainer, X, y, cases

    def test_evaluate_returns_metrics(self, trained_setup):
        trainer, X, y, _ = trained_setup
        metrics = trainer.evaluate(X, y)

        assert "test_roc_auc" in metrics
        assert "test_pr_auc" in metrics
        assert "feature_importance" in metrics
        assert metrics["test_roc_auc"] > 0.5

    def test_evaluate_with_case_ids(self, trained_setup):
        trainer, X, y, cases = trained_setup

        # Build case_ids: first sample of each case is positive
        case_ids = []
        idx = 0
        for i, case in enumerate(cases):
            case_ids.append(i)  # Positive
            idx += 1
            for _ in case.get("negative_genes", [])[:10]:
                case_ids.append(i)
                idx += 1

        case_ids = np.array(case_ids[:len(y)])
        metrics = trainer.evaluate(X, y, case_ids=case_ids)

        assert "top_1_accuracy" in metrics
        assert "top_5_accuracy" in metrics
        assert "top_10_accuracy" in metrics
        assert "mrr" in metrics
        # With good features, Top-10 should be decent
        assert metrics["top_10_accuracy"] > 0.3

    def test_evaluate_topk_from_cases(self, trained_setup):
        trainer, _, _, _ = trained_setup

        test_cases = []
        rng = np.random.RandomState(99)
        for i in range(10):
            causal = _make_positive_case(
                gene_symbol=f"CAUSAL_{i}",
                max_variant_score=0.7 + rng.random() * 0.3,
            )
            genes = [causal]
            for j in range(15):
                genes.append(_make_negative_case(gene_symbol=f"NEG_{i}_{j}"))

            test_cases.append({
                "case_id": f"CASE_{i}",
                "causal_gene": f"CAUSAL_{i}",
                "gene_features": genes,
            })

        metrics = trainer.evaluate_topk_from_cases(test_cases)

        assert metrics["n_cases"] == 10
        assert "top_1_accuracy" in metrics
        assert "mrr" in metrics
        assert metrics["mrr"] > 0


# ── SHAP Interpretation Tests ─────────────────────────────────────────────────

class TestSHAP:
    @pytest.fixture
    def trained_setup(self):
        trainer = ModelTrainer()
        cases = _build_synthetic_dataset(n_cases=30, neg_per_case=10)
        X, y = trainer.build_training_data(cases, negative_genes_per_case=10)
        trainer.train(X, y, n_estimators=50, max_depth=4)
        return trainer, X, y

    def test_shap_returns_values(self, trained_setup):
        trainer, X, y = trained_setup
        result = trainer.explain_with_shap(X, max_samples=50)

        assert "shap_values" in result
        assert "feature_importance_shap" in result
        assert "top_features" in result
        assert "group_importance" in result

    def test_shap_values_shape(self, trained_setup):
        trainer, X, y = trained_setup
        result = trainer.explain_with_shap(X, max_samples=50)

        sv = result["shap_values"]
        assert sv.shape[1] == 44  # One SHAP value per feature

    def test_shap_feature_importance_all_present(self, trained_setup):
        trainer, X, y = trained_setup
        result = trainer.explain_with_shap(X, max_samples=50)

        fi = result["feature_importance_shap"]
        assert len(fi) == 44

    def test_shap_group_importance(self, trained_setup):
        trainer, X, y = trained_setup
        result = trainer.explain_with_shap(X, max_samples=50)

        groups = result["group_importance"]
        assert "Variant pathogenicity" in groups
        assert "Trio inheritance" in groups
        assert "Knowledge graph" in groups
        # Pathogenicity group should generally be important
        assert groups["Variant pathogenicity"] > 0

    def test_explain_single_gene(self, trained_setup):
        trainer, X, y = trained_setup

        gene = _make_positive_case()
        contributions = trainer.explain_single_gene(gene)

        assert len(contributions) == 44
        # Should be sorted by |value|
        assert abs(contributions[0][1]) >= abs(contributions[-1][1])


# ── Model Persistence Tests ──────────────────────────────────────────────────

class TestModelPersistence:
    def test_save_and_load(self, tmp_path):
        trainer = ModelTrainer()
        cases = _build_synthetic_dataset(n_cases=20, neg_per_case=5)
        X, y = trainer.build_training_data(cases)
        trainer.train(X, y, n_estimators=20, max_depth=3)

        model_path = str(tmp_path / "model.pkl")
        trainer.save_model(model_path)

        # Load in new trainer
        loaded = ModelTrainer()
        loaded.load_model(model_path)

        assert loaded.model is not None
        assert loaded.feature_columns == FEATURE_COLUMNS
        assert "cv_roc_auc" in loaded.training_metrics

        # Predictions should match
        p1 = trainer.model.predict_proba(X)[:, 1]
        p2 = loaded.model.predict_proba(X)[:, 1]
        np.testing.assert_array_almost_equal(p1, p2)

    def test_metrics_json_saved(self, tmp_path):
        trainer = ModelTrainer()
        cases = _build_synthetic_dataset(n_cases=10, neg_per_case=5)
        X, y = trainer.build_training_data(cases)
        trainer.train(X, y, n_estimators=20, max_depth=3)

        model_path = str(tmp_path / "model.pkl")
        trainer.save_model(model_path)

        import json
        metrics_path = tmp_path / "model.metrics.json"
        assert metrics_path.exists()

        with open(metrics_path) as f:
            metrics = json.load(f)
        assert "cv_roc_auc" in metrics


# ── Gene Ranker ML Integration Tests ─────────────────────────────────────────

class TestGeneRankerML:
    def test_ranker_loads_bundle_format(self, tmp_path):
        """Gene ranker should load the new bundle format."""
        from raregeneai.ranking.gene_ranker import GeneRanker
        from raregeneai.config.settings import RankingConfig

        # Train a model
        trainer = ModelTrainer()
        cases = _build_synthetic_dataset(n_cases=20, neg_per_case=5)
        X, y = trainer.build_training_data(cases)
        trainer.train(X, y, n_estimators=20, max_depth=3)

        model_path = str(tmp_path / "model.pkl")
        trainer.save_model(model_path)

        # Load via gene ranker
        config = RankingConfig(
            model_type="xgboost",
            pretrained_model_path=model_path,
        )
        ranker = GeneRanker(config)
        model = ranker._get_model()

        assert model is not None

    def test_ranker_loads_legacy_format(self, tmp_path):
        """Gene ranker should still load bare model objects."""
        import xgboost as xgb
        from raregeneai.ranking.gene_ranker import GeneRanker
        from raregeneai.config.settings import RankingConfig

        # Save bare model (legacy format)
        model = xgb.XGBClassifier(n_estimators=10, max_depth=2, use_label_encoder=False)
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.random((50, 44)), columns=FEATURE_COLUMNS)
        y = np.array([1] * 10 + [0] * 40)
        model.fit(X, y)

        model_path = str(tmp_path / "legacy.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        config = RankingConfig(model_type="xgboost", pretrained_model_path=model_path)
        ranker = GeneRanker(config)
        loaded = ranker._get_model()

        assert loaded is not None
