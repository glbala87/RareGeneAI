"""Ontology-based semantic similarity for HPO phenotype matching.

Implements Resnik, Lin, and Jiang-Conrath information content measures
for computing phenotype-to-gene similarity scores.
"""

from __future__ import annotations

import math
from collections import defaultdict

import networkx as nx
from loguru import logger

from raregeneai.config.settings import PhenotypeConfig


class SemanticSimilarity:
    """Compute semantic similarity between HPO terms using information content."""

    def __init__(self, config: PhenotypeConfig | None = None):
        self.config = config or PhenotypeConfig()
        self._graph: nx.DiGraph | None = None
        self._ic: dict[str, float] = {}
        self._ancestors_cache: dict[str, set[str]] = {}

    def load_ontology(self, obo_path: str | None = None) -> None:
        """Load HPO ontology and compute information content."""
        path = obo_path or self.config.hpo_obo_path

        try:
            import pronto

            ont = pronto.Ontology(path)
            self._graph = nx.DiGraph()

            for term in ont.terms():
                if term.obsolete:
                    continue
                self._graph.add_node(str(term.id), name=str(term.name))
                for parent in term.superclasses(distance=1):
                    if parent.id != term.id:
                        self._graph.add_edge(str(term.id), str(parent.id))

            logger.info(f"Loaded HPO graph: {self._graph.number_of_nodes()} terms, "
                       f"{self._graph.number_of_edges()} edges")

            self._compute_ic()

        except ImportError:
            logger.warning("pronto not available; building minimal graph from gene-phenotype file")
            self._build_minimal_graph()

    def _build_minimal_graph(self) -> None:
        """Build a minimal HPO graph from gene-phenotype associations."""
        self._graph = nx.DiGraph()
        self._graph.add_node("HP:0000001", name="All")

    def _compute_ic(self) -> None:
        """Compute information content for each HPO term.

        IC(t) = -log2(p(t))
        where p(t) = |descendants(t)| / |total_terms|
        """
        if not self._graph:
            return

        total = self._graph.number_of_nodes()
        if total == 0:
            return

        for node in self._graph.nodes():
            # Count descendants (all terms reachable by following reverse edges)
            descendants = nx.descendants(self._graph, node) if self._graph.has_node(node) else set()
            n_desc = len(descendants) + 1  # Include the term itself
            prob = n_desc / total
            self._ic[node] = -math.log2(prob) if prob > 0 else 0.0

        max_ic = max(self._ic.values()) if self._ic else 1.0
        logger.info(f"Computed IC for {len(self._ic)} terms (max IC={max_ic:.2f})")

    def get_ancestors(self, term_id: str) -> set[str]:
        """Get all ancestors of an HPO term (including itself)."""
        if term_id in self._ancestors_cache:
            return self._ancestors_cache[term_id]

        if not self._graph or not self._graph.has_node(term_id):
            return {term_id}

        ancestors = nx.ancestors(self._graph, term_id)
        ancestors.add(term_id)
        self._ancestors_cache[term_id] = ancestors
        return ancestors

    def most_informative_common_ancestor(self, term1: str, term2: str) -> tuple[str, float]:
        """Find the Most Informative Common Ancestor (MICA).

        Returns (term_id, IC_value) of the common ancestor with highest IC.
        """
        ancestors1 = self.get_ancestors(term1)
        ancestors2 = self.get_ancestors(term2)
        common = ancestors1 & ancestors2

        if not common:
            return "HP:0000001", 0.0

        mica = max(common, key=lambda t: self._ic.get(t, 0.0))
        return mica, self._ic.get(mica, 0.0)

    def resnik_similarity(self, term1: str, term2: str) -> float:
        """Resnik similarity: IC of MICA.

        sim_Resnik(t1, t2) = IC(MICA(t1, t2))
        """
        _, ic = self.most_informative_common_ancestor(term1, term2)
        return ic

    def lin_similarity(self, term1: str, term2: str) -> float:
        """Lin similarity: normalized IC.

        sim_Lin(t1, t2) = 2 * IC(MICA) / (IC(t1) + IC(t2))
        """
        _, mica_ic = self.most_informative_common_ancestor(term1, term2)
        ic1 = self._ic.get(term1, 0.0)
        ic2 = self._ic.get(term2, 0.0)

        denom = ic1 + ic2
        if denom == 0:
            return 0.0
        return 2 * mica_ic / denom

    def jiang_conrath_similarity(self, term1: str, term2: str) -> float:
        """Jiang-Conrath similarity (converted to similarity from distance).

        dist_JC(t1, t2) = IC(t1) + IC(t2) - 2*IC(MICA)
        sim_JC = 1 / (1 + dist_JC)
        """
        _, mica_ic = self.most_informative_common_ancestor(term1, term2)
        ic1 = self._ic.get(term1, 0.0)
        ic2 = self._ic.get(term2, 0.0)

        distance = ic1 + ic2 - 2 * mica_ic
        return 1.0 / (1.0 + distance)

    def term_similarity(self, term1: str, term2: str) -> float:
        """Compute similarity between two HPO terms using configured method."""
        method = self.config.similarity_method
        if method == "resnik":
            return self.resnik_similarity(term1, term2)
        elif method == "lin":
            return self.lin_similarity(term1, term2)
        elif method == "jc":
            return self.jiang_conrath_similarity(term1, term2)
        else:
            return self.resnik_similarity(term1, term2)

    def phenotype_set_similarity(
        self,
        patient_terms: list[str],
        gene_terms: list[str],
    ) -> float:
        """Compute similarity between patient phenotype and gene phenotype profile.

        Uses Best-Match Average (BMA):
        sim(P, G) = (1/|P|) * sum_p max_g sim(p, g)

        This captures how well the gene's known phenotype profile explains
        the patient's observed phenotypes.
        """
        if not patient_terms or not gene_terms:
            return 0.0

        total = 0.0
        for p_term in patient_terms:
            best_match = max(
                (self.term_similarity(p_term, g_term) for g_term in gene_terms),
                default=0.0,
            )
            total += best_match

        return total / len(patient_terms)
