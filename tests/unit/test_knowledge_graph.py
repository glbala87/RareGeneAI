"""Unit tests for knowledge graph-based gene prioritization.

Tests:
  - Graph construction (nodes, edges, types)
  - Random walk with restart
  - Personalized PageRank
  - Gene scoring and ranking
  - Path finding and explanation
  - Candidate enrichment
  - Explainer KG output
"""

import pytest
import networkx as nx

from raregeneai.config.settings import KnowledgeGraphConfig
from raregeneai.models.data_models import GeneCandidate, InheritanceMode
from raregeneai.knowledge_graph.graph_builder import (
    KnowledgeGraphBuilder,
    GENE_PREFIX,
    HPO_PREFIX,
    DISEASE_PREFIX,
    PATHWAY_PREFIX,
)
from raregeneai.knowledge_graph.graph_scorer import KnowledgeGraphScorer


# ── Test graph fixture ────────────────────────────────────────────────────────

def _build_test_graph() -> nx.Graph:
    """Build a small knowledge graph for testing.

    Topology:
      GENE_A -- HP:0001250 (Seizures)
      GENE_A -- OMIM:607208 (Dravet syndrome)
      OMIM:607208 -- HP:0001250
      OMIM:607208 -- HP:0001263 (Developmental regression)

      GENE_B -- HP:0001263
      GENE_B -- KEGG:hsa04721 (Synaptic vesicle cycle)

      GENE_A -- GENE_B (PPI)

      GENE_C -- HP:0004322 (Short stature) [unrelated phenotype]
      GENE_C -- KEGG:hsa04110 (Cell cycle)

      GENE_D (no connections to patient phenotype)
    """
    G = nx.Graph()

    # Gene nodes
    G.add_node(f"{GENE_PREFIX}GENE_A", type="gene", symbol="GENE_A")
    G.add_node(f"{GENE_PREFIX}GENE_B", type="gene", symbol="GENE_B")
    G.add_node(f"{GENE_PREFIX}GENE_C", type="gene", symbol="GENE_C")
    G.add_node(f"{GENE_PREFIX}GENE_D", type="gene", symbol="GENE_D")

    # HPO nodes
    G.add_node(f"{HPO_PREFIX}HP:0001250", type="phenotype", hpo_id="HP:0001250", name="Seizures")
    G.add_node(f"{HPO_PREFIX}HP:0001263", type="phenotype", hpo_id="HP:0001263", name="Developmental regression")
    G.add_node(f"{HPO_PREFIX}HP:0004322", type="phenotype", hpo_id="HP:0004322", name="Short stature")

    # Disease nodes
    G.add_node(f"{DISEASE_PREFIX}OMIM:607208", type="disease", omim_id="607208", name="Dravet syndrome")

    # Pathway nodes
    G.add_node(f"{PATHWAY_PREFIX}hsa04721", type="pathway", pathway_id="hsa04721", name="Synaptic vesicle cycle")
    G.add_node(f"{PATHWAY_PREFIX}hsa04110", type="pathway", pathway_id="hsa04110", name="Cell cycle")

    # Gene-phenotype edges
    G.add_edge(f"{GENE_PREFIX}GENE_A", f"{HPO_PREFIX}HP:0001250", weight=1.0, edge_type="gene_phenotype")
    G.add_edge(f"{GENE_PREFIX}GENE_B", f"{HPO_PREFIX}HP:0001263", weight=1.0, edge_type="gene_phenotype")
    G.add_edge(f"{GENE_PREFIX}GENE_C", f"{HPO_PREFIX}HP:0004322", weight=1.0, edge_type="gene_phenotype")

    # Gene-disease edges
    G.add_edge(f"{GENE_PREFIX}GENE_A", f"{DISEASE_PREFIX}OMIM:607208", weight=0.8, edge_type="gene_disease")

    # Disease-phenotype edges
    G.add_edge(f"{DISEASE_PREFIX}OMIM:607208", f"{HPO_PREFIX}HP:0001250", weight=0.7, edge_type="disease_phenotype")
    G.add_edge(f"{DISEASE_PREFIX}OMIM:607208", f"{HPO_PREFIX}HP:0001263", weight=0.7, edge_type="disease_phenotype")

    # PPI edges
    G.add_edge(f"{GENE_PREFIX}GENE_A", f"{GENE_PREFIX}GENE_B", weight=0.5, edge_type="ppi")

    # Pathway edges
    G.add_edge(f"{GENE_PREFIX}GENE_B", f"{PATHWAY_PREFIX}hsa04721", weight=0.4, edge_type="pathway")
    G.add_edge(f"{GENE_PREFIX}GENE_C", f"{PATHWAY_PREFIX}hsa04110", weight=0.4, edge_type="pathway")

    return G


# ── Graph Builder Tests ───────────────────────────────────────────────────────

class TestGraphBuilder:
    def test_build_empty_graph(self):
        cfg = KnowledgeGraphConfig(hpo_gene_path=None)
        builder = KnowledgeGraphBuilder(cfg)
        G = builder.build()
        assert isinstance(G, nx.Graph)

    def test_node_type_queries(self):
        cfg = KnowledgeGraphConfig()
        builder = KnowledgeGraphBuilder(cfg)
        builder._graph = _build_test_graph()

        genes = builder.get_gene_nodes()
        assert len(genes) == 4
        assert f"{GENE_PREFIX}GENE_A" in genes

        phenos = builder.get_phenotype_nodes()
        assert len(phenos) == 3

        diseases = builder.get_disease_nodes()
        assert len(diseases) == 1

        pathways = builder.get_pathway_nodes()
        assert len(pathways) == 2

    def test_get_neighbors(self):
        cfg = KnowledgeGraphConfig()
        builder = KnowledgeGraphBuilder(cfg)
        builder._graph = _build_test_graph()

        # GENE_A neighbors: HP:0001250, OMIM:607208, GENE_B
        neighbors = builder.get_neighbors(f"{GENE_PREFIX}GENE_A")
        assert len(neighbors) == 3

        # PPI neighbors only
        ppi = builder.get_neighbors(f"{GENE_PREFIX}GENE_A", edge_type="ppi")
        assert len(ppi) == 1
        assert f"{GENE_PREFIX}GENE_B" in ppi

    def test_shortest_path(self):
        cfg = KnowledgeGraphConfig()
        builder = KnowledgeGraphBuilder(cfg)
        builder._graph = _build_test_graph()

        path = builder.shortest_path(f"{GENE_PREFIX}GENE_A", f"{HPO_PREFIX}HP:0001250")
        assert len(path) == 2  # Direct connection

        # GENE_B to HP:0001250 goes through GENE_A or OMIM
        path = builder.shortest_path(f"{GENE_PREFIX}GENE_B", f"{HPO_PREFIX}HP:0001250")
        assert len(path) >= 2

    def test_shortest_path_no_connection(self):
        cfg = KnowledgeGraphConfig()
        builder = KnowledgeGraphBuilder(cfg)
        builder._graph = _build_test_graph()

        path = builder.shortest_path(f"{GENE_PREFIX}GENE_D", f"{HPO_PREFIX}HP:0001250")
        assert path == []

    def test_add_dynamic_edge(self):
        cfg = KnowledgeGraphConfig()
        builder = KnowledgeGraphBuilder(cfg)
        builder._graph = _build_test_graph()

        initial_edges = builder.graph.number_of_edges()
        builder.add_gene_phenotype_edge("NEW_GENE", "HP:0001250")
        assert builder.graph.number_of_edges() == initial_edges + 1

    def test_load_from_file(self, tmp_path):
        """Test loading gene-phenotype from file."""
        gp_file = tmp_path / "genes_to_phenotype.txt"
        gp_file.write_text(
            "#Format\n"
            "1\tGENE_X\tHP:0001250\tSeizures\n"
            "2\tGENE_Y\tHP:0001263\tDevelopmental regression\n"
            "3\tGENE_X\tHP:0001263\tDevelopmental regression\n"
        )

        cfg = KnowledgeGraphConfig(hpo_gene_path=str(gp_file))
        builder = KnowledgeGraphBuilder(cfg)
        G = builder.build()

        assert G.number_of_nodes() >= 4  # 2 genes + 2 HPOs
        assert G.has_edge(f"{GENE_PREFIX}GENE_X", f"{HPO_PREFIX}HP:0001250")


# ── Graph Scorer Tests ────────────────────────────────────────────────────────

class TestGraphScorer:
    @pytest.fixture
    def scorer(self):
        cfg = KnowledgeGraphConfig(algorithm="rwr", restart_probability=0.4)
        s = KnowledgeGraphScorer(cfg)
        s.builder._graph = _build_test_graph()
        return s

    def test_rwr_gene_a_highest(self, scorer):
        """GENE_A is directly connected to patient HPO -> should score highest."""
        results = scorer.score_genes(
            patient_hpo_ids=["HP:0001250", "HP:0001263"],
            candidate_genes=["GENE_A", "GENE_B", "GENE_C", "GENE_D"],
        )

        assert "GENE_A" in results
        assert "GENE_B" in results
        assert results["GENE_A"]["kg_score"] > results["GENE_C"]["kg_score"]

    def test_rwr_gene_b_through_ppi(self, scorer):
        """GENE_B connects to HP:0001263 directly and to HP:0001250 through PPI with GENE_A."""
        results = scorer.score_genes(
            patient_hpo_ids=["HP:0001250", "HP:0001263"],
            candidate_genes=["GENE_A", "GENE_B", "GENE_C"],
        )

        # GENE_B should score higher than GENE_C (which only connects to unrelated HP:0004322)
        assert results["GENE_B"]["kg_score"] > results["GENE_C"]["kg_score"]

    def test_rwr_disconnected_gene_low(self, scorer):
        """GENE_D has no connections -> should have minimal score."""
        results = scorer.score_genes(
            patient_hpo_ids=["HP:0001250"],
            candidate_genes=["GENE_A", "GENE_D"],
        )

        assert results["GENE_D"]["kg_score"] < results["GENE_A"]["kg_score"]

    def test_pagerank_algorithm(self):
        """Test personalized PageRank produces valid scores."""
        cfg = KnowledgeGraphConfig(algorithm="pagerank")
        scorer = KnowledgeGraphScorer(cfg)
        scorer.builder._graph = _build_test_graph()

        results = scorer.score_genes(
            patient_hpo_ids=["HP:0001250"],
            candidate_genes=["GENE_A", "GENE_B", "GENE_C"],
        )

        # All genes should have a score
        for gene in ["GENE_A", "GENE_B", "GENE_C"]:
            assert gene in results
            assert results[gene]["kg_score"] >= 0.0

    def test_scores_normalized_to_01(self, scorer):
        """All KG scores should be in [0, 1]."""
        results = scorer.score_genes(
            patient_hpo_ids=["HP:0001250", "HP:0001263"],
            candidate_genes=["GENE_A", "GENE_B", "GENE_C", "GENE_D"],
        )

        for gene, data in results.items():
            assert 0.0 <= data["kg_score"] <= 1.0, f"{gene} score out of range: {data['kg_score']}"

    def test_kg_rank_assigned(self, scorer):
        results = scorer.score_genes(
            patient_hpo_ids=["HP:0001250"],
            candidate_genes=["GENE_A", "GENE_B", "GENE_C"],
        )

        ranks = [results[g]["kg_rank"] for g in ["GENE_A", "GENE_B", "GENE_C"]]
        # All ranks should be positive integers
        assert all(r > 0 for r in ranks)

    def test_connected_diseases_found(self, scorer):
        results = scorer.score_genes(
            patient_hpo_ids=["HP:0001250"],
            candidate_genes=["GENE_A"],
        )

        # GENE_A is connected to OMIM:607208 (Dravet syndrome)
        assert len(results["GENE_A"]["connected_diseases"]) > 0

    def test_ppi_neighbors_counted(self, scorer):
        results = scorer.score_genes(
            patient_hpo_ids=["HP:0001250"],
            candidate_genes=["GENE_A", "GENE_B"],
        )

        # GENE_A and GENE_B are PPI neighbors
        assert results["GENE_A"]["ppi_neighbors"] >= 1

    def test_explanatory_paths(self, scorer):
        results = scorer.score_genes(
            patient_hpo_ids=["HP:0001250"],
            candidate_genes=["GENE_A"],
        )

        paths = results["GENE_A"]["paths"]
        assert len(paths) > 0
        # Path should mention GENE_A and HP:0001250
        assert any("GENE_A" in p for p in paths)

    def test_no_hpo_in_graph_returns_empty(self, scorer):
        results = scorer.score_genes(
            patient_hpo_ids=["HP:9999999"],  # Non-existent
            candidate_genes=["GENE_A"],
        )

        assert results == {}

    def test_empty_graph_returns_empty(self):
        cfg = KnowledgeGraphConfig()
        scorer = KnowledgeGraphScorer(cfg)
        scorer.builder._graph = nx.Graph()  # Empty

        results = scorer.score_genes(
            patient_hpo_ids=["HP:0001250"],
            candidate_genes=["GENE_A"],
        )

        assert results == {}


# ── Candidate Enrichment Tests ────────────────────────────────────────────────

class TestCandidateEnrichment:
    def test_enrich_candidates(self):
        cfg = KnowledgeGraphConfig()
        scorer = KnowledgeGraphScorer(cfg)
        scorer.builder._graph = _build_test_graph()

        kg_results = scorer.score_genes(
            patient_hpo_ids=["HP:0001250", "HP:0001263"],
            candidate_genes=["GENE_A", "GENE_B"],
        )

        candidates = [
            GeneCandidate(gene_symbol="GENE_A", gene_rank_score=0.5, evidence_summary={}),
            GeneCandidate(gene_symbol="GENE_B", gene_rank_score=0.4, evidence_summary={}),
        ]

        enriched = scorer.enrich_candidates(candidates, kg_results)

        assert enriched[0].kg_score > 0
        assert enriched[0].evidence_summary.get("kg_score", 0) > 0
        assert "kg_n_diseases" in enriched[0].evidence_summary

    def test_enrichment_preserves_unmatched(self):
        cfg = KnowledgeGraphConfig()
        scorer = KnowledgeGraphScorer(cfg)

        candidates = [
            GeneCandidate(gene_symbol="UNKNOWN_GENE", gene_rank_score=0.3, evidence_summary={}),
        ]

        enriched = scorer.enrich_candidates(candidates, {})
        assert enriched[0].kg_score == 0.0


# ── Graph Propagation Property Tests ─────────────────────────────────────────

class TestPropagationProperties:
    """Verify that graph propagation satisfies expected biological properties."""

    @pytest.fixture
    def scorer(self):
        cfg = KnowledgeGraphConfig(algorithm="rwr", restart_probability=0.4)
        s = KnowledgeGraphScorer(cfg)
        s.builder._graph = _build_test_graph()
        return s

    def test_direct_connection_beats_indirect(self, scorer):
        """Gene directly connected to patient HPO should score higher
        than gene connected only through intermediaries."""
        results = scorer.score_genes(
            patient_hpo_ids=["HP:0001250"],
            candidate_genes=["GENE_A", "GENE_C"],
        )

        # GENE_A is directly connected to HP:0001250
        # GENE_C is connected to HP:0004322 (not in patient phenotype)
        assert results["GENE_A"]["kg_score"] > results["GENE_C"]["kg_score"]

    def test_more_hpo_connections_higher_score(self, scorer):
        """Gene connected to more patient HPO terms should score higher."""
        # GENE_A connects to HP:0001250 directly and HP:0001263 via OMIM
        # GENE_C connects only to HP:0004322 (not queried)
        results = scorer.score_genes(
            patient_hpo_ids=["HP:0001250", "HP:0001263"],
            candidate_genes=["GENE_A", "GENE_C"],
        )

        assert results["GENE_A"]["kg_score"] > results["GENE_C"]["kg_score"]

    def test_ppi_neighbor_of_high_scorer_gets_boost(self, scorer):
        """Gene with PPI connection to a high-scoring gene should
        score higher than isolated gene."""
        results = scorer.score_genes(
            patient_hpo_ids=["HP:0001250"],
            candidate_genes=["GENE_B", "GENE_D"],
        )

        # GENE_B has PPI with GENE_A (which scores high for HP:0001250)
        # GENE_D is isolated
        assert results["GENE_B"]["kg_score"] > results["GENE_D"]["kg_score"]


# ── Explainer KG Tests ────────────────────────────────────────────────────────

class TestExplainerKG:
    def test_explain_gene_with_kg(self):
        from raregeneai.explainability.explainer import Explainer

        explainer = Explainer()
        candidate = GeneCandidate(
            gene_symbol="SCN1A",
            gene_rank_score=0.85,
            confidence=0.90,
            kg_score=0.78,
            kg_rank=2,
            kg_paths=["SCN1A -> HP:0001250 (Seizures) (score=0.0042)"],
            kg_connected_diseases=["Dravet syndrome"],
            kg_connected_pathways=["Sodium channel complex"],
            kg_ppi_neighbors=3,
        )
        text = explainer.explain_gene(candidate)

        assert "Knowledge graph" in text
        assert "0.780" in text  # kg_score
        assert "Dravet" in text
        assert "PPI" in text or "ppi" in text.lower()

    def test_explain_gene_no_kg(self):
        from raregeneai.explainability.explainer import Explainer

        explainer = Explainer()
        candidate = GeneCandidate(
            gene_symbol="GENE_X",
            gene_rank_score=0.5,
            confidence=0.6,
            kg_score=0.0,
        )
        text = explainer.explain_gene(candidate)

        assert "Knowledge graph" not in text
