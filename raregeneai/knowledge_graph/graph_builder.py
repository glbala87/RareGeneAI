"""Knowledge graph construction from biomedical databases.

Builds a heterogeneous graph with four node types:
  - GENE: protein-coding genes
  - PHENOTYPE: HPO terms (patient symptoms)
  - DISEASE: OMIM disease entries
  - PATHWAY: KEGG/Reactome pathway groups

Five edge types connect these nodes:
  - gene-phenotype: HPO gene annotations
  - gene-disease: OMIM morbidmap associations
  - disease-phenotype: OMIM clinical features mapped to HPO
  - gene-gene (PPI): STRING protein interactions
  - gene-pathway: KEGG/Reactome membership

The graph is stored as a weighted NetworkX DiGraph where edge weights
reflect biological confidence of the relationship.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import requests
from loguru import logger

from raregeneai.config.settings import KnowledgeGraphConfig


# Node type prefixes for the heterogeneous graph
GENE_PREFIX = "gene:"
HPO_PREFIX = "hpo:"
DISEASE_PREFIX = "disease:"
PATHWAY_PREFIX = "pathway:"


class KnowledgeGraphBuilder:
    """Construct a biomedical knowledge graph from reference databases."""

    def __init__(self, config: KnowledgeGraphConfig | None = None):
        self.config = config or KnowledgeGraphConfig()
        self._graph: nx.Graph | None = None

    @property
    def graph(self) -> nx.Graph:
        if self._graph is None:
            self._graph = self.build()
        return self._graph

    def build(self) -> nx.Graph:
        """Build the complete knowledge graph.

        Returns a NetworkX undirected weighted graph.
        """
        G = nx.Graph()

        n_gp = self._add_gene_phenotype_edges(G)
        n_gd = self._add_gene_disease_edges(G)
        n_ppi = self._add_ppi_edges(G)
        n_pw = self._add_pathway_edges(G)

        logger.info(
            f"Knowledge graph built: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges "
            f"(gene-HPO={n_gp}, gene-disease={n_gd}, PPI={n_ppi}, pathway={n_pw})"
        )

        self._graph = G
        return G

    # ── Gene-Phenotype edges (HPO annotations) ────────────────────────────────

    def _add_gene_phenotype_edges(self, G: nx.Graph) -> int:
        """Load gene-HPO associations.

        File format (genes_to_phenotype.txt):
        ncbi_gene_id<TAB>gene_symbol<TAB>hpo_id<TAB>hpo_name<TAB>...
        """
        count = 0
        path = self.config.hpo_gene_path

        if path and Path(path).exists():
            with open(path) as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    fields = line.strip().split("\t")
                    if len(fields) >= 3:
                        gene = fields[1]
                        hpo_id = fields[2]

                        gene_node = f"{GENE_PREFIX}{gene}"
                        hpo_node = f"{HPO_PREFIX}{hpo_id}"

                        G.add_node(gene_node, type="gene", symbol=gene)
                        G.add_node(hpo_node, type="phenotype", hpo_id=hpo_id,
                                  name=fields[3] if len(fields) > 3 else "")

                        G.add_edge(gene_node, hpo_node,
                                  weight=self.config.w_gene_phenotype,
                                  edge_type="gene_phenotype")
                        count += 1
        else:
            count = self._fetch_gene_phenotype_remote(G)

        return count

    def _fetch_gene_phenotype_remote(self, G: nx.Graph) -> int:
        """Fetch gene-phenotype associations from HPO downloads."""
        count = 0
        try:
            url = "https://purl.obolibrary.org/obo/hp/hpoa/genes_to_phenotype.txt"
            resp = requests.get(url, timeout=60, stream=True)
            resp.raise_for_status()

            for line in resp.iter_lines(decode_unicode=True):
                if not line or line.startswith("#"):
                    continue
                fields = line.split("\t")
                if len(fields) >= 3:
                    gene = fields[1]
                    hpo_id = fields[2]

                    gene_node = f"{GENE_PREFIX}{gene}"
                    hpo_node = f"{HPO_PREFIX}{hpo_id}"

                    G.add_node(gene_node, type="gene", symbol=gene)
                    G.add_node(hpo_node, type="phenotype", hpo_id=hpo_id)
                    G.add_edge(gene_node, hpo_node,
                              weight=self.config.w_gene_phenotype,
                              edge_type="gene_phenotype")
                    count += 1

            logger.info(f"Fetched {count} gene-phenotype associations from HPO")
        except Exception as e:
            logger.warning(f"Failed to fetch gene-phenotype data: {e}")

        return count

    # ── Gene-Disease edges (OMIM) ─────────────────────────────────────────────

    def _add_gene_disease_edges(self, G: nx.Graph) -> int:
        """Load gene-disease associations from OMIM morbidmap.

        morbidmap.txt format:
        disorder_info<TAB>gene_symbols<TAB>mim_number<TAB>...
        """
        count = 0
        path = self.config.omim_morbidmap_path

        if not path or not Path(path).exists():
            return count

        with open(path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                fields = line.strip().split("\t")
                if len(fields) >= 3:
                    disorder = fields[0].strip()
                    genes_str = fields[1].strip()
                    mim = fields[2].strip()

                    disease_node = f"{DISEASE_PREFIX}OMIM:{mim}"
                    G.add_node(disease_node, type="disease", omim_id=mim,
                              name=disorder)

                    for gene in genes_str.split(","):
                        gene = gene.strip()
                        if not gene:
                            continue
                        gene_node = f"{GENE_PREFIX}{gene}"
                        G.add_node(gene_node, type="gene", symbol=gene)
                        G.add_edge(gene_node, disease_node,
                                  weight=self.config.w_gene_disease,
                                  edge_type="gene_disease")
                        count += 1

        return count

    # ── PPI edges (STRING) ────────────────────────────────────────────────────

    def _add_ppi_edges(self, G: nx.Graph) -> int:
        """Load protein-protein interactions from STRING.

        STRING format:
        protein1<SPACE>protein2<SPACE>combined_score

        Protein IDs are like '9606.ENSP00000269305'. We map to gene symbols
        when possible, otherwise use the protein ID.
        """
        count = 0
        path = self.config.string_ppi_path

        if not path or not Path(path).exists():
            return count

        min_score = self.config.string_min_score

        with open(path) as f:
            header = f.readline()  # Skip header
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue

                p1, p2, score_str = parts[0], parts[1], parts[2]
                try:
                    score = int(score_str)
                except ValueError:
                    continue

                if score < min_score:
                    continue

                # Extract gene name from STRING ID (9606.ENSPXXX -> gene lookup)
                # In production, use a protein-to-gene mapping file
                g1 = self._string_id_to_gene(p1)
                g2 = self._string_id_to_gene(p2)

                if g1 and g2 and g1 != g2:
                    n1 = f"{GENE_PREFIX}{g1}"
                    n2 = f"{GENE_PREFIX}{g2}"
                    G.add_node(n1, type="gene", symbol=g1)
                    G.add_node(n2, type="gene", symbol=g2)

                    # Normalize STRING score to 0-1
                    norm_weight = (score / 1000.0) * self.config.w_ppi
                    G.add_edge(n1, n2, weight=norm_weight, edge_type="ppi",
                              string_score=score)
                    count += 1

        return count

    @staticmethod
    def _string_id_to_gene(string_id: str) -> str:
        """Extract gene identifier from STRING protein ID.

        STRING IDs: '9606.ENSP00000269305' or 'GENE_SYMBOL'
        """
        # If it looks like a gene symbol already (no dots, short)
        if "." not in string_id and len(string_id) < 20:
            return string_id

        # Strip species prefix
        parts = string_id.split(".")
        if len(parts) == 2:
            return parts[1]  # Return the protein ID portion
        return string_id

    # ── Pathway edges (KEGG / Reactome) ───────────────────────────────────────

    def _add_pathway_edges(self, G: nx.Graph) -> int:
        """Load gene-pathway associations.

        Expected format (TSV):
        gene_symbol<TAB>pathway_id<TAB>pathway_name

        Genes sharing a pathway get indirect connections through
        the pathway node (gene -> pathway <- gene).
        """
        count = 0

        for path in [self.config.kegg_pathway_path, self.config.reactome_pathway_path]:
            if not path or not Path(path).exists():
                continue

            with open(path) as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    fields = line.strip().split("\t")
                    if len(fields) >= 2:
                        gene = fields[0]
                        pw_id = fields[1]
                        pw_name = fields[2] if len(fields) > 2 else pw_id

                        gene_node = f"{GENE_PREFIX}{gene}"
                        pw_node = f"{PATHWAY_PREFIX}{pw_id}"

                        G.add_node(gene_node, type="gene", symbol=gene)
                        G.add_node(pw_node, type="pathway",
                                  pathway_id=pw_id, name=pw_name)
                        G.add_edge(gene_node, pw_node,
                                  weight=self.config.w_pathway,
                                  edge_type="pathway")
                        count += 1

        return count

    # ── Graph utilities ───────────────────────────────────────────────────────

    def get_gene_nodes(self) -> list[str]:
        """Return all gene node IDs."""
        return [n for n, d in self.graph.nodes(data=True) if d.get("type") == "gene"]

    def get_phenotype_nodes(self) -> list[str]:
        return [n for n, d in self.graph.nodes(data=True) if d.get("type") == "phenotype"]

    def get_disease_nodes(self) -> list[str]:
        return [n for n, d in self.graph.nodes(data=True) if d.get("type") == "disease"]

    def get_pathway_nodes(self) -> list[str]:
        return [n for n, d in self.graph.nodes(data=True) if d.get("type") == "pathway"]

    def get_neighbors(self, node: str, edge_type: str | None = None) -> list[str]:
        """Get neighbors of a node, optionally filtered by edge type."""
        if node not in self.graph:
            return []
        neighbors = []
        for n in self.graph.neighbors(node):
            if edge_type is None:
                neighbors.append(n)
            else:
                edge_data = self.graph.edges[node, n]
                if edge_data.get("edge_type") == edge_type:
                    neighbors.append(n)
        return neighbors

    def shortest_path(self, source: str, target: str) -> list[str]:
        """Find shortest path between two nodes."""
        try:
            return nx.shortest_path(self.graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def add_gene_phenotype_edge(self, gene: str, hpo_id: str, weight: float | None = None) -> None:
        """Add a single gene-phenotype edge (for dynamic graph updates)."""
        gene_node = f"{GENE_PREFIX}{gene}"
        hpo_node = f"{HPO_PREFIX}{hpo_id}"
        w = weight or self.config.w_gene_phenotype
        self.graph.add_node(gene_node, type="gene", symbol=gene)
        self.graph.add_node(hpo_node, type="phenotype", hpo_id=hpo_id)
        self.graph.add_edge(gene_node, hpo_node, weight=w, edge_type="gene_phenotype")
