"""Gene-phenotype matching engine.

Maps patient HPO terms to candidate genes using semantic similarity
and gene-phenotype association databases (HPO annotations, OMIM, Orphanet).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import requests
from loguru import logger

from raregeneai.config.settings import PhenotypeConfig
from raregeneai.models.data_models import AnnotatedVariant, HPOTerm, PatientPhenotype

from .semantic_similarity import SemanticSimilarity


class GenePhenotypeMatcher:
    """Match patient phenotypes to candidate genes."""

    def __init__(self, config: PhenotypeConfig | None = None):
        self.config = config or PhenotypeConfig()
        self.sim = SemanticSimilarity(config)
        self._gene_to_hpo: dict[str, list[str]] = {}
        self._loaded = False

    def load(self) -> None:
        """Load ontology and gene-phenotype associations."""
        if self._loaded:
            return

        # Load HPO ontology for semantic similarity
        obo_path = Path(self.config.hpo_obo_path)
        if obo_path.exists():
            self.sim.load_ontology(str(obo_path))
        else:
            logger.warning("HPO OBO file not found; will use precomputed associations only")

        # Load gene-phenotype associations
        self._load_gene_phenotype_associations()
        self._loaded = True

    def _load_gene_phenotype_associations(self) -> None:
        """Load gene-to-HPO mappings from HPO annotation file.

        Expected format (genes_to_phenotype.txt from HPO):
        gene_id<TAB>gene_symbol<TAB>hpo_id<TAB>hpo_name<TAB>...
        """
        gp_path = Path(self.config.gene_phenotype_path)

        if gp_path.exists():
            with open(gp_path) as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    fields = line.strip().split("\t")
                    if len(fields) >= 3:
                        gene_symbol = fields[1]
                        hpo_id = fields[2]
                        self._gene_to_hpo.setdefault(gene_symbol, []).append(hpo_id)

            logger.info(f"Loaded phenotype associations for {len(self._gene_to_hpo)} genes")
        else:
            logger.info("Gene-phenotype file not found; fetching from HPO downloads")
            self._fetch_gene_phenotype_remote()

    def _fetch_gene_phenotype_remote(self) -> None:
        """Download gene-phenotype associations from HPO."""
        try:
            url = "https://purl.obolibrary.org/obo/hp/hpoa/genes_to_phenotype.txt"
            resp = requests.get(url, timeout=60, stream=True)
            resp.raise_for_status()

            for line in resp.iter_lines(decode_unicode=True):
                if not line or line.startswith("#"):
                    continue
                fields = line.split("\t")
                if len(fields) >= 3:
                    gene_symbol = fields[1]
                    hpo_id = fields[2]
                    self._gene_to_hpo.setdefault(gene_symbol, []).append(hpo_id)

            # Cache locally
            gp_path = Path(self.config.gene_phenotype_path)
            gp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(gp_path, "w") as f:
                f.write(resp.text)

            logger.info(f"Downloaded phenotype associations for {len(self._gene_to_hpo)} genes")

        except Exception as e:
            logger.error(f"Failed to download gene-phenotype data: {e}")

    def score_gene(
        self,
        gene_symbol: str,
        patient_phenotype: PatientPhenotype,
    ) -> float:
        """Compute phenotype similarity score for a single gene.

        Score formula:
        score = BMA_similarity(patient_hpo, gene_hpo)

        Where BMA = Best Match Average across patient HPO terms.

        Returns:
            Phenotype score in [0, 1].
        """
        self.load()

        patient_hpo_ids = [t.id for t in patient_phenotype.hpo_terms]
        gene_hpo_ids = self._gene_to_hpo.get(gene_symbol, [])

        if not gene_hpo_ids:
            return 0.0

        # Compute semantic similarity
        score = self.sim.phenotype_set_similarity(patient_hpo_ids, gene_hpo_ids)

        # Bonus for known disease gene
        if gene_symbol in self._gene_to_hpo:
            n_associations = len(gene_hpo_ids)
            association_bonus = min(0.1, n_associations * 0.005)
            score = min(1.0, score + association_bonus)

        return score

    def score_candidates(
        self,
        candidate_genes: list[str],
        patient_phenotype: PatientPhenotype,
    ) -> dict[str, float]:
        """Score all candidate genes against patient phenotype.

        Returns:
            Dict mapping gene_symbol -> phenotype_score.
        """
        self.load()

        scores = {}
        for gene in candidate_genes:
            scores[gene] = self.score_gene(gene, patient_phenotype)

        # Normalize to [0, 1]
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {g: s / max_score for g, s in scores.items()}

        n_scored = sum(1 for s in scores.values() if s > self.config.min_phenotype_score)
        logger.info(
            f"Phenotype scoring: {n_scored}/{len(candidate_genes)} genes above threshold "
            f"({self.config.min_phenotype_score})"
        )

        return scores

    def get_phenotype_explanation(
        self,
        gene_symbol: str,
        patient_phenotype: PatientPhenotype,
    ) -> list[dict]:
        """Generate detailed phenotype match explanation.

        Returns list of best HPO term matches with similarity scores.
        """
        self.load()

        patient_hpo_ids = [t.id for t in patient_phenotype.hpo_terms]
        gene_hpo_ids = self._gene_to_hpo.get(gene_symbol, [])

        if not gene_hpo_ids:
            return []

        explanations = []
        for p_term in patient_phenotype.hpo_terms:
            best_score = 0.0
            best_gene_term = ""

            for g_hpo in gene_hpo_ids:
                sim = self.sim.term_similarity(p_term.id, g_hpo)
                if sim > best_score:
                    best_score = sim
                    best_gene_term = g_hpo

            if best_score > 0:
                explanations.append({
                    "patient_hpo": p_term.id,
                    "patient_hpo_name": p_term.name,
                    "matched_gene_hpo": best_gene_term,
                    "similarity": round(best_score, 4),
                })

        explanations.sort(key=lambda x: x["similarity"], reverse=True)
        return explanations
