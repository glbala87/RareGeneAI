"""HPO term parsing, validation, and standardization.

Accepts HPO terms in multiple formats, validates against the HPO ontology,
and returns structured PatientPhenotype objects.
"""

from __future__ import annotations

import re
from pathlib import Path

import requests
from loguru import logger

from raregeneai.config.settings import PhenotypeConfig
from raregeneai.models.data_models import HPOTerm, PatientPhenotype

HPO_ID_PATTERN = re.compile(r"HP:\d{7}")


class HPOParser:
    """Parse and validate HPO phenotype terms."""

    def __init__(self, config: PhenotypeConfig | None = None):
        self.config = config or PhenotypeConfig()
        self._ontology: dict[str, HPOTerm] | None = None

    def parse_phenotype(
        self,
        patient_id: str,
        hpo_terms: list[str],
        negated_terms: list[str] | None = None,
        age_of_onset: str | None = None,
        sex: str | None = None,
    ) -> PatientPhenotype:
        """Parse and validate HPO terms into a PatientPhenotype.

        Args:
            patient_id: Patient identifier.
            hpo_terms: List of HPO IDs (e.g., ["HP:0001250", "HP:0002878"]).
            negated_terms: HPO terms explicitly NOT present.
            age_of_onset: HPO onset term or free text.
            sex: Patient sex.

        Returns:
            Validated PatientPhenotype object.
        """
        validated = []
        for term_str in hpo_terms:
            term = self._validate_term(term_str)
            if term:
                validated.append(term)
            else:
                logger.warning(f"Invalid/unknown HPO term: {term_str}")

        validated_neg = []
        for term_str in (negated_terms or []):
            term = self._validate_term(term_str)
            if term:
                validated_neg.append(term)

        logger.info(
            f"Patient {patient_id}: {len(validated)} valid HPO terms, "
            f"{len(validated_neg)} negated terms"
        )

        return PatientPhenotype(
            patient_id=patient_id,
            hpo_terms=validated,
            negated_terms=validated_neg,
            age_of_onset=age_of_onset,
            sex=sex,
        )

    def parse_from_file(self, filepath: str | Path) -> list[PatientPhenotype]:
        """Parse phenotype file (tab-separated: patient_id, hpo_terms).

        Format: patient_id<TAB>HP:0001250,HP:0002878[<TAB>sex<TAB>onset]
        """
        filepath = Path(filepath)
        patients = []

        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split("\t")
                patient_id = parts[0]
                hpo_ids = [t.strip() for t in parts[1].split(",")]
                sex = parts[2] if len(parts) > 2 else None
                onset = parts[3] if len(parts) > 3 else None

                patient = self.parse_phenotype(
                    patient_id=patient_id,
                    hpo_terms=hpo_ids,
                    sex=sex,
                    age_of_onset=onset,
                )
                patients.append(patient)

        return patients

    def _validate_term(self, term_str: str) -> HPOTerm | None:
        """Validate an HPO term against the ontology."""
        term_str = term_str.strip()

        # Extract HP:XXXXXXX pattern
        match = HPO_ID_PATTERN.search(term_str)
        if not match:
            return None

        hpo_id = match.group(0)

        # Try local ontology first
        ontology = self._load_ontology()
        if ontology and hpo_id in ontology:
            return ontology[hpo_id]

        # Fall back to remote API
        return self._fetch_term_remote(hpo_id)

    def _load_ontology(self) -> dict[str, HPOTerm] | None:
        """Load HPO ontology from OBO file using pronto."""
        if self._ontology is not None:
            return self._ontology

        obo_path = Path(self.config.hpo_obo_path)
        if not obo_path.exists():
            logger.info("HPO OBO file not found locally, will use remote API")
            self._ontology = {}
            return self._ontology

        try:
            import pronto
            ont = pronto.Ontology(str(obo_path))
            self._ontology = {}

            for term in ont.terms():
                if term.obsolete:
                    continue
                parents = [str(p.id) for p in term.superclasses(distance=1) if p.id != term.id]
                children = [str(c.id) for c in term.subclasses(distance=1) if c.id != term.id]

                self._ontology[str(term.id)] = HPOTerm(
                    id=str(term.id),
                    name=str(term.name),
                    definition=str(term.definition) if term.definition else "",
                    is_obsolete=term.obsolete,
                    parents=parents,
                    children=children,
                )

            logger.info(f"Loaded {len(self._ontology)} HPO terms from ontology")
            return self._ontology

        except ImportError:
            logger.warning("pronto not installed, using remote API for HPO validation")
            self._ontology = {}
            return self._ontology
        except Exception as e:
            logger.error(f"Failed to load HPO ontology: {e}")
            self._ontology = {}
            return self._ontology

    def _fetch_term_remote(self, hpo_id: str) -> HPOTerm | None:
        """Fetch HPO term from the HPO API."""
        try:
            url = f"https://ontology.jax.org/api/hp/terms/{hpo_id}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return HPOTerm(
                    id=hpo_id,
                    name=data.get("name", ""),
                    definition=data.get("definition", ""),
                )
        except Exception as e:
            logger.debug(f"Remote HPO lookup failed for {hpo_id}: {e}")

        # Return minimal term if ID format is valid
        if HPO_ID_PATTERN.match(hpo_id):
            return HPOTerm(id=hpo_id, name="Unknown")
        return None
