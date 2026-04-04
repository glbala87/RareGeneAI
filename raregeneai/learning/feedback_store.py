"""Feedback capture and storage for continuous learning.

Captures clinician feedback on RareGeneAI diagnoses:
  - Confirmed diagnosis (causal gene identified)
  - Rejected candidate (gene ruled out)
  - VUS reclassification
  - False positive/negative reports

Each feedback entry is immutable (append-only) with a full audit trail
for clinical compliance. The store is a simple JSON-lines file that
supports concurrent appends without corruption.

The feedback corpus drives periodic model retraining to improve
ranking accuracy over time.
"""

from __future__ import annotations

import datetime
import json
import uuid
from pathlib import Path

from pydantic import BaseModel, Field
from loguru import logger


class FeedbackEntry(BaseModel):
    """Single feedback record from a clinician."""
    # Identity
    feedback_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: str = Field(
        default_factory=lambda: datetime.datetime.now().isoformat(timespec="seconds")
    )

    # Case reference
    patient_id: str = ""
    report_id: str = ""

    # Feedback content
    feedback_type: str = ""  # "confirmed_diagnosis", "rejected_gene", "vus_reclassified", "false_negative"
    gene_symbol: str = ""
    was_ranked: bool = False      # Was this gene in the ranked output?
    original_rank: int = 0        # Rank in the original RareGeneAI output
    original_score: float = 0.0   # Score in the original output

    # For confirmed diagnosis
    confirmed_causal: bool = False
    diagnosis: str = ""            # Disease name / OMIM ID
    confirmation_method: str = ""  # "functional_study", "segregation", "sanger", "clinical"

    # For VUS reclassification
    old_classification: str = ""
    new_classification: str = ""

    # Evidence snapshot (gene features at time of analysis)
    evidence_snapshot: dict = Field(default_factory=dict)

    # Audit
    analyst_id: str = ""
    institution: str = ""
    notes: str = ""

    # Model version that produced the original ranking
    model_version: str = ""
    pipeline_version: str = ""


class FeedbackStore:
    """Append-only feedback store backed by JSON-lines file."""

    def __init__(self, store_path: str | Path):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def submit(self, entry: FeedbackEntry) -> str:
        """Record a feedback entry. Returns the feedback_id."""
        line = entry.model_dump_json() + "\n"

        with open(self.store_path, "a") as f:
            f.write(line)

        logger.info(
            f"Feedback recorded: {entry.feedback_type} for {entry.gene_symbol} "
            f"(patient={entry.patient_id}, id={entry.feedback_id})"
        )
        return entry.feedback_id

    def submit_confirmed_diagnosis(
        self,
        patient_id: str,
        gene_symbol: str,
        diagnosis: str,
        original_rank: int = 0,
        original_score: float = 0.0,
        evidence_snapshot: dict | None = None,
        confirmation_method: str = "clinical",
        analyst_id: str = "",
        model_version: str = "",
        **kwargs,
    ) -> str:
        """Convenience: record a confirmed diagnosis."""
        entry = FeedbackEntry(
            patient_id=patient_id,
            feedback_type="confirmed_diagnosis",
            gene_symbol=gene_symbol,
            confirmed_causal=True,
            diagnosis=diagnosis,
            was_ranked=original_rank > 0,
            original_rank=original_rank,
            original_score=original_score,
            evidence_snapshot=evidence_snapshot or {},
            confirmation_method=confirmation_method,
            analyst_id=analyst_id,
            model_version=model_version,
            **kwargs,
        )
        return self.submit(entry)

    def submit_rejected_gene(
        self,
        patient_id: str,
        gene_symbol: str,
        original_rank: int = 0,
        analyst_id: str = "",
        notes: str = "",
        **kwargs,
    ) -> str:
        """Convenience: record a rejected candidate gene."""
        entry = FeedbackEntry(
            patient_id=patient_id,
            feedback_type="rejected_gene",
            gene_symbol=gene_symbol,
            was_ranked=original_rank > 0,
            original_rank=original_rank,
            analyst_id=analyst_id,
            notes=notes,
            **kwargs,
        )
        return self.submit(entry)

    def load_all(self) -> list[FeedbackEntry]:
        """Load all feedback entries from the store."""
        if not self.store_path.exists():
            return []

        entries = []
        with open(self.store_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entries.append(FeedbackEntry(**data))
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"Skipping malformed feedback line: {e}")

        return entries

    def load_confirmed_diagnoses(self) -> list[FeedbackEntry]:
        """Load only confirmed diagnosis entries (for training data)."""
        return [
            e for e in self.load_all()
            if e.feedback_type == "confirmed_diagnosis" and e.confirmed_causal
        ]

    def count(self) -> dict[str, int]:
        """Count feedback entries by type."""
        all_entries = self.load_all()
        counts: dict[str, int] = {}
        for e in all_entries:
            counts[e.feedback_type] = counts.get(e.feedback_type, 0) + 1
        counts["total"] = len(all_entries)
        return counts

    def get_training_cases(self) -> list[dict]:
        """Convert confirmed diagnoses to training cases for model retraining.

        Returns list of dicts compatible with ModelTrainer.build_training_data():
        Each dict has the gene's evidence_snapshot (features) as a positive case.
        """
        confirmed = self.load_confirmed_diagnoses()
        cases = []
        for entry in confirmed:
            if entry.evidence_snapshot:
                case = dict(entry.evidence_snapshot)
                case["gene_symbol"] = entry.gene_symbol
                case["_feedback_id"] = entry.feedback_id
                case["_patient_id"] = entry.patient_id
                cases.append(case)
        return cases
