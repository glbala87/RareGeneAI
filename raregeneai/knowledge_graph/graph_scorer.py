"""Knowledge graph-based gene scoring via graph propagation.

Implements three graph-based ranking algorithms:
  1. Random Walk with Restart (RWR): Personalized diffusion from
     phenotype seed nodes through the heterogeneous graph.
  2. Personalized PageRank: Variant of PageRank seeded on patient HPO nodes.
  3. Network diffusion: Heat diffusion from seed nodes.

All methods rank genes by their proximity/connectivity to the patient's
phenotype nodes through gene-disease, gene-pathway, and PPI edges.

The key biological insight: a gene that is close in the knowledge graph
to the patient's phenotype terms — via disease associations, shared
pathways, or interacting proteins — is more likely to be causal.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from loguru import logger

from raregeneai.config.settings import KnowledgeGraphConfig
from raregeneai.models.data_models import GeneCandidate

from .graph_builder import (
    GENE_PREFIX,
    HPO_PREFIX,
    DISEASE_PREFIX,
    PATHWAY_PREFIX,
    KnowledgeGraphBuilder,
)


class KnowledgeGraphScorer:
    """Score genes using graph propagation from phenotype seeds."""

    def __init__(self, config: KnowledgeGraphConfig | None = None):
        self.config = config or KnowledgeGraphConfig()
        self.builder = KnowledgeGraphBuilder(self.config)
        self._propagation_scores: dict[str, float] | None = None

    def score_genes(
        self,
        patient_hpo_ids: list[str],
        candidate_genes: list[str],
    ) -> dict[str, dict]:
        """Score candidate genes by graph proximity to patient phenotype.

        Args:
            patient_hpo_ids: Patient's HPO term IDs (e.g. ["HP:0001250"]).
            candidate_genes: Gene symbols to score.

        Returns:
            Dict mapping gene_symbol -> {
                "kg_score": float,  # Propagation score (0-1)
                "kg_rank": int,
                "paths": list[str],  # Explanatory paths
                "connected_diseases": list[str],
                "connected_pathways": list[str],
                "ppi_neighbors": int,
            }
        """
        if not self.config.enabled:
            return {}

        G = self.builder.graph

        if G.number_of_nodes() == 0:
            logger.warning("Knowledge graph is empty; skipping KG scoring")
            return {}

        # Build seed vector from patient HPO terms
        seed_nodes = self._build_seed_nodes(patient_hpo_ids, G)
        if not seed_nodes:
            logger.warning("No patient HPO terms found in knowledge graph")
            return {}

        logger.info(
            f"KG scoring: {len(seed_nodes)} seed HPO nodes, "
            f"{len(candidate_genes)} candidate genes, "
            f"graph has {G.number_of_nodes()} nodes / {G.number_of_edges()} edges"
        )

        # Run graph propagation
        if self.config.algorithm == "rwr":
            raw_scores = self._random_walk_with_restart(G, seed_nodes)
        elif self.config.algorithm == "pagerank":
            raw_scores = self._personalized_pagerank(G, seed_nodes)
        else:
            raw_scores = self._network_diffusion(G, seed_nodes)

        # Extract and normalize gene scores
        results = self._extract_gene_scores(
            G, raw_scores, candidate_genes, patient_hpo_ids
        )

        n_scored = sum(1 for v in results.values() if v["kg_score"] > self.config.min_kg_score)
        logger.info(f"KG scoring complete: {n_scored}/{len(candidate_genes)} genes scored above threshold")

        return results

    def _build_seed_nodes(
        self, hpo_ids: list[str], G: nx.Graph,
    ) -> dict[str, float]:
        """Map patient HPO IDs to graph seed nodes with uniform weight."""
        seeds = {}
        for hpo_id in hpo_ids:
            node = f"{HPO_PREFIX}{hpo_id}"
            if node in G:
                seeds[node] = 1.0 / len(hpo_ids)

        # If some HPO terms not directly in graph, try ancestors/children
        if len(seeds) < len(hpo_ids):
            for hpo_id in hpo_ids:
                node = f"{HPO_PREFIX}{hpo_id}"
                if node not in G:
                    # Check if any neighbor of a gene shares this HPO
                    # This handles ontology-level fuzzy matching
                    pass  # Handled by the phenotype matcher upstream

        return seeds

    # ── Algorithm 1: Random Walk with Restart ─────────────────────────────────

    def _random_walk_with_restart(
        self, G: nx.Graph, seeds: dict[str, float],
    ) -> dict[str, float]:
        """Random Walk with Restart from phenotype seed nodes.

        At each step, the walker either:
          - Follows a random edge (weighted) with probability (1 - r)
          - Returns to a seed node with probability r

        After convergence, each node has a stationary probability
        reflecting its proximity to the seed phenotype nodes.

        This is the core algorithm used by tools like PRINCE and RWRH.
        """
        r = self.config.restart_probability
        max_iter = self.config.max_iterations
        tol = self.config.convergence_threshold

        nodes = list(G.nodes())
        n = len(nodes)
        if n == 0:
            return {}

        node_idx = {node: i for i, node in enumerate(nodes)}

        # Build normalized adjacency matrix
        A = nx.to_numpy_array(G, nodelist=nodes, weight="weight")

        # Row-normalize (stochastic matrix)
        row_sums = A.sum(axis=1)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        W = A / row_sums[:, np.newaxis]

        # Restart vector (seed distribution)
        p0 = np.zeros(n)
        for node, weight in seeds.items():
            if node in node_idx:
                p0[node_idx[node]] = weight

        # Normalize restart vector
        p0_sum = p0.sum()
        if p0_sum > 0:
            p0 /= p0_sum

        # Iterative propagation
        p = p0.copy()
        for iteration in range(max_iter):
            p_new = (1 - r) * W.T @ p + r * p0

            # Check convergence
            diff = np.abs(p_new - p).sum()
            p = p_new

            if diff < tol:
                logger.debug(f"RWR converged in {iteration + 1} iterations (diff={diff:.2e})")
                break

        # Map back to node names
        scores = {}
        for i, node in enumerate(nodes):
            scores[node] = float(p[i])

        return scores

    # ── Algorithm 2: Personalized PageRank ────────────────────────────────────

    def _personalized_pagerank(
        self, G: nx.Graph, seeds: dict[str, float],
    ) -> dict[str, float]:
        """NetworkX personalized PageRank with HPO seed personalization.

        Uses the built-in efficient implementation.
        """
        try:
            scores = nx.pagerank(
                G,
                alpha=self.config.damping,
                personalization=seeds,
                weight="weight",
                max_iter=self.config.max_iterations,
                tol=self.config.convergence_threshold,
            )
            return scores
        except Exception as e:
            logger.warning(f"PageRank failed: {e}; falling back to RWR")
            return self._random_walk_with_restart(G, seeds)

    # ── Algorithm 3: Network Diffusion ────────────────────────────────────────

    def _network_diffusion(
        self, G: nx.Graph, seeds: dict[str, float],
    ) -> dict[str, float]:
        """Heat diffusion from seed nodes through the graph.

        Uses the exponential of the negative graph Laplacian:
        H(t) = exp(-t * L) * h0

        Where L is the graph Laplacian and h0 is the seed heat vector.
        """
        from scipy.linalg import expm

        nodes = list(G.nodes())
        n = len(nodes)
        node_idx = {node: i for i, node in enumerate(nodes)}

        A = nx.to_numpy_array(G, nodelist=nodes, weight="weight")
        D = np.diag(A.sum(axis=1))
        L = D - A  # Graph Laplacian

        # Heat vector (initial distribution)
        h0 = np.zeros(n)
        for node, weight in seeds.items():
            if node in node_idx:
                h0[node_idx[node]] = weight

        # Diffusion: H = exp(-t * L) * h0, with t=1.0
        t = 1.0
        H = expm(-t * L)
        h = H @ h0

        scores = {}
        for i, node in enumerate(nodes):
            scores[node] = float(max(h[i], 0.0))

        return scores

    # ── Score Extraction and Explanation ───────────────────────────────────────

    def _extract_gene_scores(
        self,
        G: nx.Graph,
        raw_scores: dict[str, float],
        candidate_genes: list[str],
        patient_hpo_ids: list[str],
    ) -> dict[str, dict]:
        """Extract gene-level scores and build explanatory paths."""
        # Get all gene scores for normalization
        all_gene_scores = {}
        for node, score in raw_scores.items():
            if node.startswith(GENE_PREFIX):
                gene = node[len(GENE_PREFIX):]
                all_gene_scores[gene] = score

        # Normalize to [0, 1]
        max_score = max(all_gene_scores.values()) if all_gene_scores else 1.0
        if max_score == 0:
            max_score = 1.0

        # Rank all genes
        sorted_genes = sorted(all_gene_scores.items(), key=lambda x: x[1], reverse=True)
        gene_ranks = {gene: rank + 1 for rank, (gene, _) in enumerate(sorted_genes)}

        results = {}
        for gene in candidate_genes:
            gene_node = f"{GENE_PREFIX}{gene}"
            raw = all_gene_scores.get(gene, 0.0)
            normalized = raw / max_score

            # Find connected diseases and pathways
            connected_diseases = self._find_connected_entities(G, gene_node, "disease")
            connected_pathways = self._find_connected_entities(G, gene_node, "pathway")

            # Count PPI neighbors among candidate genes
            ppi_neighbors = self._count_ppi_neighbors(G, gene_node, candidate_genes)

            # Build explanatory paths
            paths = self._find_explanatory_paths(
                G, gene_node, patient_hpo_ids, raw_scores
            )

            results[gene] = {
                "kg_score": round(normalized, 6),
                "kg_rank": gene_ranks.get(gene, 0),
                "paths": paths[:self.config.top_paths_to_report],
                "connected_diseases": connected_diseases[:5],
                "connected_pathways": connected_pathways[:5],
                "ppi_neighbors": ppi_neighbors,
            }

        return results

    def _find_connected_entities(
        self, G: nx.Graph, gene_node: str, entity_type: str,
    ) -> list[str]:
        """Find diseases/pathways directly connected to a gene."""
        if gene_node not in G:
            return []

        entities = []
        prefix = DISEASE_PREFIX if entity_type == "disease" else PATHWAY_PREFIX

        for neighbor in G.neighbors(gene_node):
            if neighbor.startswith(prefix):
                name = G.nodes[neighbor].get("name", neighbor)
                entities.append(name)

        # Also check 2-hop connections (gene -> disease -> phenotype)
        if entity_type == "disease":
            for n1 in G.neighbors(gene_node):
                if G.nodes.get(n1, {}).get("type") == "disease":
                    continue
                for n2 in G.neighbors(n1):
                    if n2.startswith(prefix):
                        name = G.nodes[n2].get("name", n2)
                        if name not in entities:
                            entities.append(name)

        return entities

    def _count_ppi_neighbors(
        self, G: nx.Graph, gene_node: str, candidate_genes: list[str],
    ) -> int:
        """Count how many candidate genes are PPI neighbors."""
        if gene_node not in G:
            return 0

        candidate_set = {f"{GENE_PREFIX}{g}" for g in candidate_genes}
        count = 0
        for neighbor in G.neighbors(gene_node):
            if neighbor in candidate_set:
                edge_data = G.edges[gene_node, neighbor]
                if edge_data.get("edge_type") == "ppi":
                    count += 1
        return count

    def _find_explanatory_paths(
        self, G: nx.Graph, gene_node: str,
        patient_hpo_ids: list[str],
        scores: dict[str, float],
    ) -> list[str]:
        """Find the most informative paths from gene to patient HPO terms.

        Returns human-readable path descriptions like:
        "GENE_A -> OMIM:123456 (Epilepsy) -> HP:0001250 (Seizures)"
        """
        paths = []

        for hpo_id in patient_hpo_ids[:5]:  # Limit to top 5 HPO terms
            hpo_node = f"{HPO_PREFIX}{hpo_id}"

            if gene_node not in G or hpo_node not in G:
                continue

            try:
                sp = nx.shortest_path(G, gene_node, hpo_node)
                if len(sp) <= 5:  # Only report short paths (biologically meaningful)
                    path_desc = self._format_path(G, sp)
                    score = scores.get(hpo_node, 0.0)
                    paths.append(f"{path_desc} (score={score:.4f})")
            except nx.NetworkXNoPath:
                continue

        # Sort by path length (shorter = more direct = more informative)
        paths.sort(key=len)
        return paths

    @staticmethod
    def _format_path(G: nx.Graph, path: list[str]) -> str:
        """Format a graph path into a human-readable string."""
        parts = []
        for node in path:
            data = G.nodes.get(node, {})
            node_type = data.get("type", "unknown")

            if node_type == "gene":
                parts.append(data.get("symbol", node))
            elif node_type == "phenotype":
                name = data.get("name", "")
                hpo = data.get("hpo_id", node)
                parts.append(f"{hpo} ({name})" if name else hpo)
            elif node_type == "disease":
                name = data.get("name", "")
                omim = data.get("omim_id", node)
                parts.append(f"OMIM:{omim} ({name})" if name else node)
            elif node_type == "pathway":
                name = data.get("name", "")
                parts.append(f"Pathway:{name}" if name else node)
            else:
                parts.append(node)

        return " -> ".join(parts)

    def enrich_candidates(
        self,
        candidates: list[GeneCandidate],
        kg_results: dict[str, dict],
    ) -> list[GeneCandidate]:
        """Enrich GeneCandidate objects with KG evidence."""
        for candidate in candidates:
            kg = kg_results.get(candidate.gene_symbol)
            if not kg:
                continue

            candidate.kg_score = kg["kg_score"]
            candidate.kg_rank = kg["kg_rank"]
            candidate.kg_paths = kg["paths"]
            candidate.kg_connected_diseases = kg["connected_diseases"]
            candidate.kg_connected_pathways = kg["connected_pathways"]
            candidate.kg_ppi_neighbors = kg["ppi_neighbors"]

            candidate.evidence_summary.update({
                "kg_score": kg["kg_score"],
                "kg_rank": kg["kg_rank"],
                "kg_ppi_neighbors": kg["ppi_neighbors"],
                "kg_n_diseases": len(kg["connected_diseases"]),
                "kg_n_pathways": len(kg["connected_pathways"]),
                "kg_has_direct_hpo_link": any("HP:" in p for p in kg["paths"]),
            })

        n_enriched = sum(1 for c in candidates if c.kg_score > 0)
        logger.info(f"Enriched {n_enriched} candidates with knowledge graph scores")
        return candidates
