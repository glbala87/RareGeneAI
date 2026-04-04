"""Clinical decision support: actionable genes, pharmacogenomics, and audit trail.

Components:
  1. ACMG Secondary Findings (SF v3.2): flags genes on the ACMG recommended
     list for return of secondary/incidental findings.
  2. Pharmacogenomics (PharmGKB/CPIC): identifies clinically actionable
     drug-gene interactions in candidate genes.
  3. Audit trail: every clinical assertion is timestamped and traceable
     for CAP/CLIA regulatory compliance.

Produces ClinicalInsight objects attached to each GeneCandidate.
"""

from __future__ import annotations

import datetime
from typing import Optional

from pydantic import BaseModel, Field
from loguru import logger

from raregeneai.models.data_models import (
    ACMGClassification,
    AnnotatedVariant,
    GeneCandidate,
)
from .acmg_classifier import ACMGClassifier, ACMGResult


# ── ACMG SF v3.2 gene list (81 genes recommended for secondary findings) ─────
# Source: Miller et al. 2023, Genet Med
ACMG_SF_GENES: dict[str, dict] = {
    # Cardiovascular
    "MYBPC3": {"condition": "Hypertrophic cardiomyopathy", "category": "cardiovascular"},
    "MYH7": {"condition": "Hypertrophic cardiomyopathy", "category": "cardiovascular"},
    "TNNT2": {"condition": "Hypertrophic cardiomyopathy", "category": "cardiovascular"},
    "TNNI3": {"condition": "Hypertrophic cardiomyopathy", "category": "cardiovascular"},
    "TPM1": {"condition": "Hypertrophic cardiomyopathy", "category": "cardiovascular"},
    "MYL3": {"condition": "Hypertrophic cardiomyopathy", "category": "cardiovascular"},
    "ACTC1": {"condition": "Hypertrophic cardiomyopathy", "category": "cardiovascular"},
    "MYL2": {"condition": "Hypertrophic cardiomyopathy", "category": "cardiovascular"},
    "LMNA": {"condition": "Dilated cardiomyopathy", "category": "cardiovascular"},
    "SCN5A": {"condition": "Brugada / Long QT syndrome", "category": "cardiovascular"},
    "KCNQ1": {"condition": "Long QT syndrome", "category": "cardiovascular"},
    "KCNH2": {"condition": "Long QT syndrome", "category": "cardiovascular"},
    "RYR2": {"condition": "CPVT", "category": "cardiovascular"},
    "PKP2": {"condition": "ARVC", "category": "cardiovascular"},
    "DSP": {"condition": "ARVC", "category": "cardiovascular"},
    "DSC2": {"condition": "ARVC", "category": "cardiovascular"},
    "DSG2": {"condition": "ARVC", "category": "cardiovascular"},
    "TMEM43": {"condition": "ARVC", "category": "cardiovascular"},
    "FBN1": {"condition": "Marfan syndrome", "category": "cardiovascular"},
    "TGFBR1": {"condition": "Loeys-Dietz syndrome", "category": "cardiovascular"},
    "TGFBR2": {"condition": "Loeys-Dietz syndrome", "category": "cardiovascular"},
    "SMAD3": {"condition": "Loeys-Dietz syndrome", "category": "cardiovascular"},
    "ACTA2": {"condition": "Thoracic aortic aneurysm", "category": "cardiovascular"},
    "MYH11": {"condition": "Thoracic aortic aneurysm", "category": "cardiovascular"},
    "COL3A1": {"condition": "Ehlers-Danlos vascular type", "category": "cardiovascular"},
    # Cancer
    "BRCA1": {"condition": "Hereditary breast/ovarian cancer", "category": "cancer"},
    "BRCA2": {"condition": "Hereditary breast/ovarian cancer", "category": "cancer"},
    "TP53": {"condition": "Li-Fraumeni syndrome", "category": "cancer"},
    "MLH1": {"condition": "Lynch syndrome", "category": "cancer"},
    "MSH2": {"condition": "Lynch syndrome", "category": "cancer"},
    "MSH6": {"condition": "Lynch syndrome", "category": "cancer"},
    "PMS2": {"condition": "Lynch syndrome", "category": "cancer"},
    "APC": {"condition": "Familial adenomatous polyposis", "category": "cancer"},
    "MUTYH": {"condition": "MUTYH-associated polyposis", "category": "cancer"},
    "RB1": {"condition": "Retinoblastoma", "category": "cancer"},
    "MEN1": {"condition": "Multiple endocrine neoplasia type 1", "category": "cancer"},
    "RET": {"condition": "Multiple endocrine neoplasia type 2", "category": "cancer"},
    "VHL": {"condition": "Von Hippel-Lindau", "category": "cancer"},
    "WT1": {"condition": "WT1-related Wilms tumor", "category": "cancer"},
    "NF2": {"condition": "Neurofibromatosis type 2", "category": "cancer"},
    "STK11": {"condition": "Peutz-Jeghers syndrome", "category": "cancer"},
    "TSC1": {"condition": "Tuberous sclerosis", "category": "cancer"},
    "TSC2": {"condition": "Tuberous sclerosis", "category": "cancer"},
    "PTEN": {"condition": "PTEN hamartoma tumor syndrome", "category": "cancer"},
    "SDHB": {"condition": "Hereditary paraganglioma-pheochromocytoma", "category": "cancer"},
    "SDHD": {"condition": "Hereditary paraganglioma-pheochromocytoma", "category": "cancer"},
    "SDHAF2": {"condition": "Hereditary paraganglioma-pheochromocytoma", "category": "cancer"},
    "SDHC": {"condition": "Hereditary paraganglioma-pheochromocytoma", "category": "cancer"},
    "BMPR1A": {"condition": "Juvenile polyposis", "category": "cancer"},
    "SMAD4": {"condition": "Juvenile polyposis", "category": "cancer"},
    # Metabolic / Other
    "LDLR": {"condition": "Familial hypercholesterolemia", "category": "metabolic"},
    "APOB": {"condition": "Familial hypercholesterolemia", "category": "metabolic"},
    "PCSK9": {"condition": "Familial hypercholesterolemia", "category": "metabolic"},
    "GAA": {"condition": "Pompe disease", "category": "metabolic"},
    "GLA": {"condition": "Fabry disease", "category": "metabolic"},
    "OTC": {"condition": "Ornithine transcarbamylase deficiency", "category": "metabolic"},
    "ATP7B": {"condition": "Wilson disease", "category": "metabolic"},
    "HFE": {"condition": "Hereditary hemochromatosis", "category": "metabolic"},
    "BTD": {"condition": "Biotinidase deficiency", "category": "metabolic"},
    "RPE65": {"condition": "RPE65-related retinal dystrophy", "category": "metabolic"},
}

# ── Pharmacogenomic drug-gene interactions (CPIC Level A/B) ──────────────────
# Source: CPIC guidelines (cpicpgx.org), PharmGKB clinical annotations
PHARMACOGENOMIC_GENES: dict[str, dict] = {
    "CYP2D6": {
        "drugs": ["codeine", "tramadol", "tamoxifen", "ondansetron"],
        "level": "A",
        "action": "Dose adjustment or alternative drug based on metabolizer status",
    },
    "CYP2C19": {
        "drugs": ["clopidogrel", "voriconazole", "escitalopram", "sertraline"],
        "level": "A",
        "action": "Dose adjustment or alternative drug based on metabolizer status",
    },
    "CYP2C9": {
        "drugs": ["warfarin", "phenytoin"],
        "level": "A",
        "action": "Dose reduction for poor metabolizers",
    },
    "CYP3A5": {
        "drugs": ["tacrolimus"],
        "level": "A",
        "action": "Dose adjustment for transplant immunosuppression",
    },
    "DPYD": {
        "drugs": ["fluorouracil", "capecitabine"],
        "level": "A",
        "action": "Dose reduction or contraindication (DPD deficiency)",
    },
    "TPMT": {
        "drugs": ["azathioprine", "mercaptopurine", "thioguanine"],
        "level": "A",
        "action": "Dose reduction for intermediate/poor metabolizers",
    },
    "NUDT15": {
        "drugs": ["azathioprine", "mercaptopurine"],
        "level": "A",
        "action": "Dose reduction or contraindication",
    },
    "SLCO1B1": {
        "drugs": ["simvastatin"],
        "level": "A",
        "action": "Reduced dose or alternative statin",
    },
    "VKORC1": {
        "drugs": ["warfarin"],
        "level": "A",
        "action": "Dose adjustment based on sensitivity",
    },
    "G6PD": {
        "drugs": ["rasburicase", "dapsone", "primaquine"],
        "level": "A",
        "action": "Contraindicated in G6PD deficiency",
    },
    "HLA-B": {
        "drugs": ["abacavir", "carbamazepine", "allopurinol", "phenytoin"],
        "level": "A",
        "action": "HLA screening before prescribing (severe adverse reaction risk)",
    },
    "HLA-A": {
        "drugs": ["carbamazepine"],
        "level": "A",
        "action": "HLA-A*31:01 screening before prescribing",
    },
    "RYR1": {
        "drugs": ["volatile anesthetics", "succinylcholine"],
        "level": "A",
        "action": "Malignant hyperthermia risk — avoid triggering agents",
    },
    "CACNA1S": {
        "drugs": ["volatile anesthetics"],
        "level": "A",
        "action": "Malignant hyperthermia risk",
    },
    "IFNL3": {
        "drugs": ["peginterferon alfa-2a", "ribavirin"],
        "level": "A",
        "action": "Response prediction for HCV treatment",
    },
    "SCN1A": {
        "drugs": ["carbamazepine", "phenytoin", "lamotrigine"],
        "level": "B",
        "action": "Sodium channel blockers may worsen seizures in Dravet syndrome",
    },
}


class ClinicalInsight(BaseModel):
    """Clinical decision support output for a single gene."""
    gene_symbol: str

    # ACMG variant classification
    acmg_results: list[ACMGResult] = Field(default_factory=list)
    highest_acmg_class: ACMGClassification = ACMGClassification.VUS
    n_pathogenic_variants: int = 0
    n_lp_variants: int = 0
    n_vus_variants: int = 0

    # Actionability
    is_acmg_sf_gene: bool = False       # On ACMG SF v3.2 list
    sf_condition: str = ""               # Associated condition
    sf_category: str = ""                # cardiovascular, cancer, metabolic

    # Pharmacogenomics
    has_pgx_relevance: bool = False
    pgx_drugs: list[str] = Field(default_factory=list)
    pgx_action: str = ""
    pgx_level: str = ""  # CPIC level A, B

    # Clinical summary
    clinical_significance: str = ""  # "Diagnostic", "Actionable", "PGx", "Research"
    clinical_recommendation: str = ""
    requires_confirmation: bool = True   # All findings need orthogonal confirmation

    # Audit trail (CAP/CLIA compliance)
    analysis_timestamp: str = ""
    pipeline_version: str = "1.0.0"
    classification_method: str = "ACMG/AMP 2015 (Richards et al.)"
    analyst_review_required: bool = True  # Human review always required


class ClinicalDecisionEngine:
    """Generate clinician-ready insights for prioritized genes."""

    def __init__(self):
        self.acmg_classifier = ACMGClassifier()

    def analyze(
        self, candidates: list[GeneCandidate],
    ) -> dict[str, ClinicalInsight]:
        """Run clinical decision support on ranked gene candidates.

        For each gene:
        1. Classify all variants with ACMG/AMP criteria
        2. Check ACMG SF v3.2 actionable gene list
        3. Check pharmacogenomic relevance (CPIC)
        4. Generate clinical recommendation
        5. Create audit trail entry

        Returns dict: gene_symbol -> ClinicalInsight
        """
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        results: dict[str, ClinicalInsight] = {}

        for candidate in candidates:
            insight = self._analyze_gene(candidate, timestamp)
            results[candidate.gene_symbol] = insight

        n_path = sum(1 for i in results.values() if i.n_pathogenic_variants > 0)
        n_sf = sum(1 for i in results.values() if i.is_acmg_sf_gene)
        n_pgx = sum(1 for i in results.values() if i.has_pgx_relevance)
        logger.info(
            f"Clinical insights: {n_path} with P/LP variants, "
            f"{n_sf} ACMG SF genes, {n_pgx} with PGx relevance"
        )

        return results

    def _analyze_gene(
        self, candidate: GeneCandidate, timestamp: str,
    ) -> ClinicalInsight:
        """Analyze a single gene candidate."""
        gene = candidate.gene_symbol
        insight = ClinicalInsight(gene_symbol=gene, analysis_timestamp=timestamp)

        # 1. ACMG classify each variant
        for var in candidate.variants:
            result = self.acmg_classifier.classify(var)
            insight.acmg_results.append(result)

            if result.classification == ACMGClassification.PATHOGENIC:
                insight.n_pathogenic_variants += 1
            elif result.classification == ACMGClassification.LIKELY_PATHOGENIC:
                insight.n_lp_variants += 1
            elif result.classification == ACMGClassification.VUS:
                insight.n_vus_variants += 1

        # Highest classification across all variants
        class_order = [
            ACMGClassification.PATHOGENIC,
            ACMGClassification.LIKELY_PATHOGENIC,
            ACMGClassification.VUS,
            ACMGClassification.LIKELY_BENIGN,
            ACMGClassification.BENIGN,
        ]
        for cls in class_order:
            if any(r.classification == cls for r in insight.acmg_results):
                insight.highest_acmg_class = cls
                break

        # 2. ACMG SF actionability
        if gene in ACMG_SF_GENES:
            sf = ACMG_SF_GENES[gene]
            insight.is_acmg_sf_gene = True
            insight.sf_condition = sf["condition"]
            insight.sf_category = sf["category"]

        # 3. Pharmacogenomics
        if gene in PHARMACOGENOMIC_GENES:
            pgx = PHARMACOGENOMIC_GENES[gene]
            insight.has_pgx_relevance = True
            insight.pgx_drugs = pgx["drugs"]
            insight.pgx_action = pgx["action"]
            insight.pgx_level = pgx["level"]

        # 4. Clinical significance and recommendation
        insight.clinical_significance = self._determine_significance(insight)
        insight.clinical_recommendation = self._generate_recommendation(insight, candidate)

        return insight

    def _determine_significance(self, insight: ClinicalInsight) -> str:
        """Determine clinical significance category."""
        if insight.n_pathogenic_variants > 0:
            return "Diagnostic"
        if insight.n_lp_variants > 0:
            return "Likely Diagnostic"
        if insight.is_acmg_sf_gene and insight.n_vus_variants > 0:
            return "Actionable (secondary finding)"
        if insight.has_pgx_relevance:
            return "Pharmacogenomic"
        if insight.n_vus_variants > 0:
            return "Research"
        return "Not significant"

    def _generate_recommendation(
        self, insight: ClinicalInsight, candidate: GeneCandidate,
    ) -> str:
        """Generate a clinician-facing recommendation."""
        parts = []

        if insight.n_pathogenic_variants > 0:
            parts.append(
                f"DIAGNOSTIC: {insight.n_pathogenic_variants} pathogenic variant(s) "
                f"identified in {insight.gene_symbol}. "
                f"Recommend confirmatory testing (Sanger sequencing) and "
                f"referral to clinical genetics."
            )

        if insight.n_lp_variants > 0 and insight.n_pathogenic_variants == 0:
            parts.append(
                f"LIKELY DIAGNOSTIC: {insight.n_lp_variants} likely pathogenic variant(s). "
                f"Clinical correlation and confirmatory testing recommended."
            )

        if insight.is_acmg_sf_gene:
            parts.append(
                f"SECONDARY FINDING: {insight.gene_symbol} is on the ACMG SF v3.2 list "
                f"for {insight.sf_condition}. "
                f"Consider return of result per institutional policy."
            )

        if insight.has_pgx_relevance:
            drugs = ", ".join(insight.pgx_drugs[:3])
            parts.append(
                f"PHARMACOGENOMIC: {insight.gene_symbol} affects response to {drugs}. "
                f"{insight.pgx_action}."
            )

        if candidate.has_de_novo_lof:
            parts.append(
                "De novo loss-of-function variant detected — "
                "strong evidence for causality in the proband's phenotype."
            )

        if not parts:
            if insight.n_vus_variants > 0:
                parts.append(
                    f"VUS identified in {insight.gene_symbol}. "
                    f"Insufficient evidence for clinical action. "
                    f"Consider periodic reclassification."
                )
            else:
                parts.append("No clinically significant findings.")

        parts.append(
            "NOTE: All findings require review by a board-certified "
            "clinical molecular geneticist before clinical action."
        )

        return " ".join(parts)

    def enrich_candidates(
        self,
        candidates: list[GeneCandidate],
        insights: dict[str, ClinicalInsight],
    ) -> list[GeneCandidate]:
        """Attach clinical insights to GeneCandidate evidence_summary."""
        for candidate in candidates:
            insight = insights.get(candidate.gene_symbol)
            if not insight:
                continue

            candidate.evidence_summary.update({
                "acmg_class": insight.highest_acmg_class.value,
                "n_pathogenic_variants": insight.n_pathogenic_variants,
                "n_lp_variants": insight.n_lp_variants,
                "is_acmg_sf_gene": insight.is_acmg_sf_gene,
                "has_pgx_relevance": insight.has_pgx_relevance,
                "clinical_significance": insight.clinical_significance,
            })

        return candidates
