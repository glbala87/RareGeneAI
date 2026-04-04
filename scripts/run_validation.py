#!/usr/bin/env python3
"""RareGeneAI Validation Suite — Train, Evaluate, and Report ROC-AUC.

Generates a HARD synthetic validation cohort with realistic feature
overlap between causal and non-causal genes. Includes:
  - Easy cases (clear LoF + de novo + strong phenotype)
  - Medium cases (missense VUS + moderate phenotype match)
  - Hard cases (weak evidence, novel gene, no trio)
  - Distractors (high CADD bystander variants in known disease genes)

Usage:
    python scripts/run_validation.py
    python scripts/run_validation.py --n-cases 500 --save-model models/v1.pkl
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raregeneai.ranking.model_trainer import (
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    ModelTrainer,
)


def _make_row(rng, label: int, difficulty: str = "medium") -> dict:
    """Generate a single gene feature vector.

    Difficulty levels control how much overlap exists between
    positive and negative distributions:
      easy:   clear separation (LoF + de novo + strong HPO)
      medium: moderate overlap (missense + partial HPO match)
      hard:   high overlap (weak evidence, no trio)
      distractor: negative that looks like a positive
    """
    if label == 1:
        # ── Positive (causal gene) ─────────────────────────────────────
        if difficulty == "easy":
            cadd = rng.uniform(28, 40)
            revel = rng.uniform(0.7, 0.99)
            pheno = rng.uniform(0.7, 1.0)
            af = 0.0
            has_lof = rng.random() < 0.7
            has_clinvar = rng.random() < 0.5
            has_denovo_lof = rng.random() < 0.5
            kg = rng.uniform(0.5, 0.9)
            expr = rng.uniform(0.5, 0.9) if rng.random() < 0.4 else 0.0
        elif difficulty == "hard":
            cadd = rng.uniform(15, 28)
            revel = rng.uniform(0.3, 0.7)
            pheno = rng.uniform(0.15, 0.5)
            af = rng.uniform(0, 0.003)
            has_lof = rng.random() < 0.1
            has_clinvar = False
            has_denovo_lof = False
            kg = rng.uniform(0.0, 0.3)
            expr = rng.uniform(0.2, 0.5) if rng.random() < 0.15 else 0.0
        else:  # medium
            cadd = rng.uniform(20, 35)
            revel = rng.uniform(0.45, 0.85)
            pheno = rng.uniform(0.3, 0.8)
            af = rng.choice([0.0, rng.uniform(0, 0.001)], p=[0.6, 0.4])
            has_lof = rng.random() < 0.35
            has_clinvar = rng.random() < 0.2
            has_denovo_lof = rng.random() < 0.15
            kg = rng.uniform(0.15, 0.6)
            expr = rng.uniform(0.3, 0.7) if rng.random() < 0.25 else 0.0

        has_denovo = has_denovo_lof or (rng.random() < 0.2)
        has_ch = not has_denovo and rng.random() < 0.2
        has_hr = not has_denovo and not has_ch and rng.random() < 0.15
        inh = 1.0 if has_denovo_lof else (0.9 if has_denovo else (0.85 if has_ch else (0.75 if has_hr else 0.3)))
        trio = rng.random() < 0.5
        is_known = rng.random() < 0.5
        mv = 0.3 * max(cadd/40, revel) + 0.2 * (1.0 if af == 0 else np.exp(-1000*af)) + 0.15 * (0.9 if has_lof else 0.55) + 0.05 * inh
        mv = np.clip(mv + rng.normal(0, 0.08), 0.05, 1.0)

    else:
        # ── Negative (non-causal gene) ─────────────────────────────────
        if difficulty == "distractor":
            # Distractor: high CADD bystander in a known disease gene
            cadd = rng.uniform(22, 35)
            revel = rng.uniform(0.3, 0.7)
            pheno = rng.uniform(0.2, 0.6)
            af = rng.uniform(0.001, 0.02)
            has_lof = rng.random() < 0.15
            has_clinvar = False
            is_known = rng.random() < 0.4
            kg = rng.uniform(0.1, 0.5)
            expr = 0.0
        else:
            cadd = rng.uniform(0, 25)
            revel = rng.uniform(0, 0.5)
            pheno = rng.uniform(0, 0.3)
            af = rng.choice([0.0, rng.uniform(0, 0.005), rng.uniform(0.005, 0.05)], p=[0.15, 0.35, 0.5])
            has_lof = rng.random() < 0.04
            has_clinvar = rng.random() < 0.02
            is_known = rng.random() < 0.12
            kg = rng.uniform(0, 0.2)
            expr = rng.uniform(0, 0.2) if rng.random() < 0.03 else 0.0

        has_denovo_lof = False
        has_denovo = False
        has_ch = False
        has_hr = False
        inh = rng.uniform(0.2, 0.4)
        trio = rng.random() < 0.3
        mv = rng.uniform(0.0, 0.4)

    has_expr = expr > 0.1
    has_dmr = label == 1 and rng.random() < 0.12
    meth_score = rng.uniform(0.3, 0.7) if has_dmr else 0.0
    conc = has_expr and has_dmr and rng.random() < 0.5
    n_layers = 1 + int(has_expr) + int(has_dmr)
    mo = 0.35*expr + 0.25*meth_score + 0.25*(0.3 if conc else 0) + 0.15*min(n_layers/3, 1)

    return {
        "max_variant_score": mv,
        "max_cadd": cadd, "max_revel": revel, "min_af": af,
        "has_lof": has_lof, "has_clinvar_pathogenic": has_clinvar,
        "phenotype_score": pheno, "n_variants": rng.randint(1, 5),
        "n_inheritance_modes": rng.randint(0 if label == 0 else 1, 3),
        "is_known_disease_gene": is_known,
        "n_noncoding_variants": rng.randint(0, 4),
        "max_regulatory_score": rng.uniform(0, 0.25),
        "max_spliceai": rng.uniform(0, 0.2 if label == 0 else 0.5),
        "has_regulatory_variant": rng.random() < 0.08,
        "has_enhancer_variant": False, "has_promoter_variant": False,
        "max_gene_mapping_score": rng.uniform(0, 0.2),
        "max_conservation_score": rng.uniform(0.1 if label == 0 else 0.3, 0.7 if label == 0 else 1.0),
        "has_sv": False, "max_sv_score": 0.0, "sv_fully_deleted": False,
        "sv_dosage_sensitive": False, "max_sv_gene_overlap": 0.0,
        "max_sv_dosage_score": 0.0, "max_sv_regulatory_disruption": 0.0,
        "multi_omics_score": mo, "n_evidence_layers": n_layers,
        "has_expression_outlier": has_expr, "expression_score": expr,
        "has_dmr": has_dmr, "has_promoter_dmr": has_dmr and rng.random() < 0.5,
        "methylation_score": meth_score, "is_concordant": conc,
        "has_de_novo": has_denovo, "has_de_novo_lof": has_denovo_lof,
        "has_compound_het": has_ch, "has_hom_recessive": has_hr,
        "trio_inheritance_score": inh if label == 1 else rng.uniform(0.2, 0.4),
        "trio_analyzed": trio,
        "kg_score": kg, "kg_ppi_neighbors": rng.randint(0, 6),
        "kg_n_diseases": rng.randint(0, 4 if is_known else 1),
        "kg_n_pathways": rng.randint(0, 4),
        "kg_has_direct_hpo_link": is_known and rng.random() < 0.5 and label == 1,
    }


def generate_cohort(n_cases=300, neg_per_case=30, seed=42):
    """Generate train/test cohort with difficulty mix.

    Mix: 30% easy, 40% medium, 30% hard (positives)
         80% standard negatives, 20% distractors
    """
    rng = np.random.RandomState(seed)
    rows, labels, case_ids = [], [], []
    case_gene_lists = []

    difficulties = rng.choice(["easy", "medium", "hard"], size=n_cases, p=[0.3, 0.4, 0.3])

    for i in range(n_cases):
        # Positive
        pos = _make_row(rng, label=1, difficulty=difficulties[i])
        pos["gene_symbol"] = f"CAUSAL_{i}"
        rows.append(pos)
        labels.append(1)
        case_ids.append(i)

        genes_for_case = [pos]

        # Negatives
        for j in range(neg_per_case):
            diff = "distractor" if rng.random() < 0.2 else "medium"
            neg = _make_row(rng, label=0, difficulty=diff)
            neg["gene_symbol"] = f"NEG_{i}_{j}"
            rows.append(neg)
            labels.append(0)
            case_ids.append(i)
            genes_for_case.append(neg)

        case_gene_lists.append({
            "case_id": f"CASE_{i:04d}",
            "causal_gene": f"CAUSAL_{i}",
            "gene_features": genes_for_case,
        })

    return rows, np.array(labels), np.array(case_ids), case_gene_lists


def run_validation(n_cases=300, save_model=None):
    print("=" * 70)
    print("  RareGeneAI Validation Suite (Hard Mode)")
    print("  30% easy / 40% medium / 30% hard cases + 20% distractors")
    print("=" * 70)
    print()

    t0 = time.time()
    trainer = ModelTrainer()

    # ── Generate data ─────────────────────────────────────────────────
    print(f"[1/5] Generating {n_cases} synthetic cases (30 genes each)...")
    all_rows, all_labels, all_case_ids, case_lists = generate_cohort(n_cases, seed=42)

    # Train/test split by case
    split = int(0.7 * n_cases)
    train_mask = all_case_ids < split
    test_mask = ~train_mask

    X_all = pd.DataFrame([trainer._extract_row(r) for r in all_rows], columns=FEATURE_COLUMNS)
    y_all = all_labels

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]
    test_case_ids = all_case_ids[test_mask]

    print(f"       Train: {len(y_train)} samples ({(y_train==1).sum()} pos, {(y_train==0).sum()} neg)")
    print(f"       Test:  {len(y_test)} samples ({(y_test==1).sum()} pos, {(y_test==0).sum()} neg)")
    print()

    # ── Train ─────────────────────────────────────────────────────────
    print("[2/5] Training XGBoost (5-fold stratified CV)...")
    trainer.train(X_train, y_train, save_path=save_model, n_estimators=500, max_depth=6, learning_rate=0.05)
    cv_auc = trainer.training_metrics["cv_roc_auc"]
    cv_pr = trainer.training_metrics["cv_pr_auc"]
    print()

    # ── Holdout evaluation ────────────────────────────────────────────
    print("[3/5] Evaluating on holdout test set...")
    test_metrics = trainer.evaluate(X_test, y_test, case_ids=test_case_ids)
    print()

    # ── Top-K per case ────────────────────────────────────────────────
    print("[4/5] Computing Top-K accuracy per case...")
    test_cases = case_lists[split:]
    topk = trainer.evaluate_topk_from_cases(test_cases)
    print()

    # ── SHAP ──────────────────────────────────────────────────────────
    print("[5/5] SHAP feature importance...")
    shap_result = trainer.explain_with_shap(X_train, max_samples=500)
    print()

    # ── Report ────────────────────────────────────────────────────────
    elapsed = time.time() - t0

    print("=" * 70)
    print("  VALIDATION RESULTS")
    print("=" * 70)
    print()
    print("  ┌────────────────────────────────────────────────────┐")
    print(f"  │  Cross-Validated ROC-AUC:     {cv_auc:.4f}              │")
    print(f"  │  Cross-Validated PR-AUC:      {cv_pr:.4f}              │")
    print(f"  │  Holdout Test ROC-AUC:        {test_metrics['test_roc_auc']:.4f}              │")
    print(f"  │  Holdout Test PR-AUC:         {test_metrics['test_pr_auc']:.4f}              │")
    print("  ├────────────────────────────────────────────────────┤")
    print(f"  │  Top-1  Accuracy:             {topk.get('top_1_accuracy',0):.1%}               │")
    print(f"  │  Top-5  Accuracy:             {topk.get('top_5_accuracy',0):.1%}               │")
    print(f"  │  Top-10 Accuracy:             {topk.get('top_10_accuracy',0):.1%}               │")
    print(f"  │  Top-20 Accuracy:             {topk.get('top_20_accuracy',0):.1%}               │")
    print(f"  │  Mean Reciprocal Rank:        {topk.get('mrr',0):.4f}              │")
    print(f"  │  Median Rank:                 {topk.get('median_rank','N/A')}                   │")
    print("  ├────────────────────────────────────────────────────┤")
    print(f"  │  Precision (pathogenic):      {test_metrics.get('test_precision_pos',0):.4f}              │")
    print(f"  │  Recall (pathogenic):         {test_metrics.get('test_recall_pos',0):.4f}              │")
    print(f"  │  F1 (pathogenic):             {test_metrics.get('test_f1_pos',0):.4f}              │")
    print("  └────────────────────────────────────────────────────┘")
    print()

    # Feature group importance
    print("  SHAP Feature Group Importance:")
    print("  " + "─" * 50)
    groups = shap_result["group_importance"]
    total_imp = sum(groups.values()) or 1.0
    for group, imp in sorted(groups.items(), key=lambda x: x[1], reverse=True):
        pct = imp / total_imp * 100
        bar_len = int(pct / 2.5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {group:<25s} {bar} {pct:5.1f}%")
    print()

    # Top 10 features
    print("  Top 10 Individual Features:")
    print("  " + "─" * 50)
    for feat, imp in shap_result["top_features"][:10]:
        print(f"    {feat:<38s} {imp:.4f}")
    print()

    print(f"  Completed in {elapsed:.1f}s")
    if save_model:
        print(f"  Model saved to: {save_model}")
    print("=" * 70)

    return {
        "cv_roc_auc": cv_auc, "cv_pr_auc": cv_pr,
        "test_roc_auc": test_metrics["test_roc_auc"],
        "test_pr_auc": test_metrics["test_pr_auc"],
        **topk, "groups": groups,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RareGeneAI Validation Suite")
    parser.add_argument("--n-cases", type=int, default=300, help="Number of synthetic cases")
    parser.add_argument("--save-model", type=str, default=None, help="Save trained model path")
    args = parser.parse_args()
    run_validation(n_cases=args.n_cases, save_model=args.save_model)
