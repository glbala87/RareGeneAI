"""Central configuration for RareGeneAI pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = DATA_DIR / "output"
MODELS_DIR = PROJECT_ROOT / "models"


# ── Pydantic config models ────────────────────────────────────────────────────
class IngestionConfig(BaseModel):
    """VCF ingestion parameters."""
    min_gq: int = Field(default=20, description="Minimum genotype quality")
    min_dp: int = Field(default=10, description="Minimum read depth")
    split_multiallelic: bool = True
    left_align: bool = True
    output_format: str = Field(default="parquet", description="parquet | json")


class PopulationConfig(BaseModel):
    """Population-specific allele frequency and founder variant configuration."""
    enabled: bool = True

    # Patient population label
    population: str = Field(
        default="", description="Population code: QGP, GME, QASP, saudi, emirati, or gnomAD pop (mid, sas, ...)"
    )

    # ── Local/regional frequency databases ─────────────────────────────
    # QGP (Qatar Genome Programme) - tabix-indexed VCF or TSV
    qgp_af_path: Optional[str] = None
    # Greater Middle East (GME) Variome
    gme_af_path: Optional[str] = None
    # Other local databases (generic TSV: chrom, pos, ref, alt, af, ac, an, hom)
    local_af_path: Optional[str] = None
    local_db_name: str = ""  # Display name for custom DB

    # ── Founder variant detection ──────────────────────────────────────
    # Criteria: enriched in local pop AND rare globally
    founder_enrichment_threshold: float = 10.0   # local_af/gnomad_af >= 10x
    founder_local_af_min: float = 0.001          # Must appear in local pop (AF >= 0.1%)
    founder_global_af_max: float = 0.001         # Must be rare globally (AF < 0.1%)

    # Known founder variant database (chrom<TAB>pos<TAB>ref<TAB>alt<TAB>gene<TAB>disease)
    known_founders_path: Optional[str] = None

    # ── Population-adjusted scoring ────────────────────────────────────
    # When local AF is available, use it to adjust rarity score
    use_population_adjusted_rarity: bool = True
    # Penalty factor for variants common in local population
    local_common_penalty: float = 0.8  # Rarity reduced by 80% if AF > threshold locally
    # AF threshold in local pop to consider "locally common"
    local_af_threshold: float = 0.01


class AnnotationConfig(BaseModel):
    """Variant annotation sources."""
    vep_executable: str = "vep"
    vep_cache_dir: str = str(REFERENCE_DIR / "vep_cache")
    vep_assembly: str = "GRCh38"
    gnomad_path: Optional[str] = None
    clinvar_path: Optional[str] = None
    cadd_snv_path: Optional[str] = None
    cadd_indel_path: Optional[str] = None
    revel_path: Optional[str] = None
    spliceai_path: Optional[str] = None
    use_remote_api: bool = True
    batch_size: int = 1000

    # Non-coding regulatory annotation sources
    regulatory: "RegulatoryConfig" = Field(default_factory=lambda: RegulatoryConfig())

    # Population-specific frequency sources
    population: "PopulationConfig" = Field(default_factory=lambda: PopulationConfig())


class RegulatoryConfig(BaseModel):
    """Non-coding variant annotation configuration."""
    enabled: bool = True

    # ENCODE / Roadmap Epigenomics regulatory regions (BED files)
    encode_cres_path: Optional[str] = None  # ENCODE cCRE registry
    roadmap_chromhmm_path: Optional[str] = None  # Roadmap 15-state ChromHMM
    ensembl_regulatory_path: Optional[str] = None  # Ensembl Regulatory Build

    # Conservation scores (tabix-indexed)
    phastcons_path: Optional[str] = None  # PhastCons 100-way
    phylop_path: Optional[str] = None  # PhyloP 100-way
    gerp_path: Optional[str] = None  # GERP++ rejected substitutions

    # Variant-to-gene mapping databases
    gtex_eqtl_path: Optional[str] = None  # GTEx significant eQTL pairs
    hic_path: Optional[str] = None  # Hi-C chromatin interaction data
    abc_model_path: Optional[str] = None  # Activity-by-Contact model predictions
    tad_path: Optional[str] = None  # TAD boundary definitions

    # Tissue/cell type specificity (for Roadmap / ENCODE lookups)
    target_tissues: list[str] = Field(default_factory=lambda: [
        "E003",  # H1 Cell Line
        "E017",  # IMR90 Fetal Lung
        "E070",  # Brain Hippocampus Middle
        "E071",  # Brain Hippocampus
        "E073",  # Brain Dorsolateral Prefrontal Cortex
        "E082",  # Fetal Brain Female
        "E081",  # Fetal Brain Male
        "E125",  # NH-A Astrocytes
    ])

    # Score thresholds
    min_regulatory_score: float = 0.2  # Minimum to retain non-coding variant
    spliceai_high_threshold: float = 0.5  # Strong splicing disruption
    spliceai_moderate_threshold: float = 0.2  # Moderate splicing disruption
    enformer_threshold: float = 0.3  # Enformer effect threshold
    abc_score_threshold: float = 0.015  # ABC model significance threshold
    eqtl_pvalue_threshold: float = 5e-8  # eQTL significance threshold
    conservation_phastcons_threshold: float = 0.5  # Conservation threshold

    # Gene mapping maximum distance (for nearest-gene fallback)
    max_distance_to_gene: int = 1_000_000  # 1 Mb window for eQTL/HiC

    # Scoring weights for regulatory impact sub-components
    w_splicing: float = 0.30  # SpliceAI weight in regulatory score
    w_regulatory_region: float = 0.20  # ENCODE/Roadmap annotation
    w_conservation: float = 0.15  # PhastCons/PhyloP/GERP
    w_deep_learning: float = 0.20  # Enformer/Sei/DeepBind
    w_gene_mapping: float = 0.15  # Confidence of gene linkage


class PhenotypeConfig(BaseModel):
    """Phenotype matching parameters."""
    hpo_obo_path: str = str(REFERENCE_DIR / "hp.obo")
    gene_phenotype_path: str = str(REFERENCE_DIR / "genes_to_phenotype.txt")
    similarity_method: str = Field(default="resnik", description="resnik | lin | jc")
    ic_method: str = Field(default="annotation", description="annotation | graph")
    min_phenotype_score: float = 0.1


class ScoringConfig(BaseModel):
    """Variant scoring weights."""
    # Weights for composite score (coding + non-coding combined)
    w_pathogenicity: float = 0.30
    w_rarity: float = 0.20
    w_functional_impact: float = 0.15
    w_phenotype: float = 0.15
    w_inheritance: float = 0.05
    w_regulatory: float = 0.15  # Non-coding regulatory impact weight

    gnomad_af_threshold: float = 0.01
    cadd_phred_threshold: float = 15.0
    revel_threshold: float = 0.5

    impact_scores: dict = Field(default_factory=lambda: {
        # ── Coding consequences ────────────────────────────────────────
        "transcript_ablation": 1.0,
        "splice_acceptor_variant": 0.95,
        "splice_donor_variant": 0.95,
        "stop_gained": 0.95,
        "frameshift_variant": 0.90,
        "stop_lost": 0.85,
        "start_lost": 0.85,
        "inframe_insertion": 0.60,
        "inframe_deletion": 0.60,
        "missense_variant": 0.55,
        "protein_altering_variant": 0.50,
        "splice_region_variant": 0.40,
        "synonymous_variant": 0.05,
        # ── Non-coding consequences (scored by regulatory context) ─────
        "regulatory_region_ablation": 0.75,
        "TFBS_ablation": 0.70,
        "regulatory_region_amplification": 0.45,
        "TFBS_amplification": 0.40,
        "TF_binding_site_variant": 0.40,
        "regulatory_region_variant": 0.35,
        "5_prime_UTR_variant": 0.30,
        "3_prime_UTR_variant": 0.25,
        "mature_miRNA_variant": 0.40,
        "non_coding_transcript_exon_variant": 0.20,
        "non_coding_transcript_variant": 0.10,
        "upstream_gene_variant": 0.08,
        "downstream_gene_variant": 0.06,
        "intron_variant": 0.05,
        "intergenic_variant": 0.02,
    })


class SVConfig(BaseModel):
    """Structural variant analysis configuration."""
    enabled: bool = True

    # AnnotSV integration
    annotsv_executable: str = "AnnotSV"
    annotsv_annotations_dir: Optional[str] = None

    # Reference databases (tabix-indexed)
    gnomad_sv_path: Optional[str] = None  # gnomAD-SV sites VCF
    dgv_path: Optional[str] = None        # Database of Genomic Variants
    decipher_path: Optional[str] = None   # DECIPHER CNV syndromes
    clinvar_sv_path: Optional[str] = None  # ClinVar SVs

    # Dosage sensitivity databases
    clingen_hi_path: Optional[str] = None  # ClinGen haploinsufficiency
    clingen_ts_path: Optional[str] = None  # ClinGen triplosensitivity
    pli_scores_path: Optional[str] = None  # gnomAD pLI scores per gene
    loeuf_scores_path: Optional[str] = None  # gnomAD LOEUF per gene

    # Gene/region overlap (BED files)
    gene_bed_path: Optional[str] = None    # Gene coordinates BED
    exon_bed_path: Optional[str] = None    # Exon coordinates BED
    tad_bed_path: Optional[str] = None     # TAD boundaries BED
    regulatory_bed_path: Optional[str] = None  # Regulatory regions BED

    # Filtering thresholds
    min_sv_size: int = 50           # Minimum SV size to consider (bp)
    max_sv_size: int = 100_000_000  # Maximum SV size (100 Mb)
    min_sv_quality: float = 5.0     # Minimum QUAL
    min_support_reads: int = 3      # Minimum supporting reads

    # Population frequency threshold
    gnomad_sv_af_threshold: float = 0.01  # Max AF to consider rare

    # Scoring weights for SV composite score
    w_sv_gene_overlap: float = 0.30    # Gene disruption weight
    w_sv_dosage: float = 0.25          # Dosage sensitivity weight
    w_sv_rarity: float = 0.20          # Population rarity weight
    w_sv_regulatory: float = 0.15      # Regulatory disruption weight
    w_sv_clinical: float = 0.10        # Clinical database weight

    # SV contribution to unified gene ranking
    sv_weight_in_ranking: float = 0.15  # Weight of SV evidence in final gene score


class MultiOmicsConfig(BaseModel):
    """Multi-omics integration configuration."""
    enabled: bool = True

    # ── RNA-seq expression ─────────────────────────────────────────────
    # Patient expression matrix: gene<TAB>tpm (or gene<TAB>count)
    expression_path: Optional[str] = None
    expression_format: str = Field(default="tpm", description="tpm | counts | fpkm")
    # Reference cohort for outlier detection (gene x sample matrix)
    reference_expression_path: Optional[str] = None
    # Pre-computed gene-level reference stats: gene<TAB>median<TAB>mad
    reference_stats_path: Optional[str] = None

    # Outlier detection thresholds
    z_score_threshold: float = 2.0          # |Z| > 2 = outlier
    z_score_strong_threshold: float = 3.0   # |Z| > 3 = strong outlier
    min_expression_tpm: float = 0.5         # Genes below this in reference are low-expressed
    underexpression_percentile: float = 5.0  # Bottom 5% = underexpressed
    overexpression_percentile: float = 95.0  # Top 5% = overexpressed

    # ── Methylation ────────────────────────────────────────────────────
    # Patient methylation: chrom<TAB>start<TAB>end<TAB>beta
    methylation_path: Optional[str] = None
    methylation_format: str = Field(default="bed", description="bed | bismark | array")
    # Reference methylation stats: region<TAB>mean_beta<TAB>sd
    reference_methylation_path: Optional[str] = None
    # Pre-computed DMR calls: chrom<TAB>start<TAB>end<TAB>gene<TAB>delta_beta<TAB>pval
    dmr_calls_path: Optional[str] = None
    # Gene promoter regions BED for DMR overlap
    promoter_bed_path: Optional[str] = None

    # DMR thresholds
    delta_beta_threshold: float = 0.2       # |delta_beta| > 0.2 = DMR
    delta_beta_strong_threshold: float = 0.3
    dmr_pvalue_threshold: float = 0.05
    min_cpgs_per_dmr: int = 3

    # ── Integration weights ────────────────────────────────────────────
    w_expression: float = 0.35       # Expression outlier weight
    w_methylation: float = 0.25      # Methylation DMR weight
    w_concordance: float = 0.25      # Concordance bonus weight
    w_layer_count: float = 0.15      # Number of supporting layers weight

    # Concordance bonus: multiplier when multiple evidence types agree
    concordance_multiplier: float = 1.3

    # Multi-omics contribution to unified gene ranking
    multiomics_weight_in_ranking: float = 0.12


class KnowledgeGraphConfig(BaseModel):
    """Knowledge graph-based prioritization configuration."""
    enabled: bool = True

    # ── Data sources ───────────────────────────────────────────────────
    # Gene-phenotype associations (HPO annotations)
    hpo_gene_path: Optional[str] = None  # genes_to_phenotype.txt
    # Gene-disease associations (OMIM)
    omim_genemap_path: Optional[str] = None  # OMIM genemap2.txt
    omim_morbidmap_path: Optional[str] = None  # OMIM morbidmap.txt
    # Protein-protein interactions
    string_ppi_path: Optional[str] = None  # STRING protein.links.v12.0.txt
    string_min_score: int = 700  # Minimum STRING combined score (0-1000)
    # Pathway data
    kegg_pathway_path: Optional[str] = None  # KEGG gene-pathway mapping
    reactome_pathway_path: Optional[str] = None  # Reactome annotations

    # ── Graph propagation parameters ───────────────────────────────────
    algorithm: str = Field(
        default="rwr", description="rwr | pagerank | diffusion"
    )
    # Random Walk with Restart
    restart_probability: float = 0.4  # Probability of returning to seed
    max_iterations: int = 100
    convergence_threshold: float = 1e-6
    # Damping factor for PageRank
    damping: float = 0.85

    # ── Edge weights ───────────────────────────────────────────────────
    w_gene_phenotype: float = 1.0   # Gene-HPO edge weight
    w_gene_disease: float = 0.8     # Gene-OMIM disease edge weight
    w_disease_phenotype: float = 0.7  # Disease-HPO edge weight
    w_ppi: float = 0.5              # Protein-protein interaction edge weight
    w_pathway: float = 0.4          # Gene-pathway co-membership edge weight

    # ── Scoring ────────────────────────────────────────────────────────
    kg_weight_in_ranking: float = 0.10  # KG contribution to final gene score
    min_kg_score: float = 0.01  # Minimum KG score to report
    top_paths_to_report: int = 3  # Number of explanatory paths to keep


class RankingConfig(BaseModel):
    """Gene ranking model parameters."""
    model_type: str = Field(default="xgboost", description="xgboost | lightgbm | rule_based")
    pretrained_model_path: Optional[str] = None
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    top_n_genes: int = 50
    min_confidence: float = 0.1


class ReportConfig(BaseModel):
    """Clinical report generation."""
    output_format: str = Field(default="html", description="html | pdf")
    template_dir: str = str(PROJECT_ROOT / "templates")
    include_acmg: bool = True
    include_evidence_summary: bool = True
    lab_name: str = "Clinical Genomics Laboratory"
    report_version: str = "1.0"


class PipelineConfig(BaseModel):
    """Master pipeline configuration."""
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    annotation: AnnotationConfig = Field(default_factory=AnnotationConfig)
    phenotype: PhenotypeConfig = Field(default_factory=PhenotypeConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    sv: SVConfig = Field(default_factory=SVConfig)
    multiomics: MultiOmicsConfig = Field(default_factory=MultiOmicsConfig)
    knowledge_graph: KnowledgeGraphConfig = Field(default_factory=KnowledgeGraphConfig)

    genome_build: str = "GRCh38"
    n_threads: int = 4
    log_level: str = "INFO"
    output_dir: str = str(OUTPUT_DIR)

    # Non-coding analysis toggle
    include_noncoding: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)


# Resolve forward references
AnnotationConfig.model_rebuild()
