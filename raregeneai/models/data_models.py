"""Core data models for RareGeneAI pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SVType(str, Enum):
    """Structural variant types."""
    DEL = "DEL"
    DUP = "DUP"
    INV = "INV"
    INS = "INS"
    BND = "BND"  # Translocation breakend
    CNV = "CNV"
    DUP_TANDEM = "DUP:TANDEM"
    DEL_ME = "DEL:ME"     # Mobile element deletion
    INS_ME = "INS:ME"     # Mobile element insertion
    UNKNOWN = "UNKNOWN"


class InheritanceMode(str, Enum):
    AUTOSOMAL_DOMINANT = "AD"
    AUTOSOMAL_RECESSIVE = "AR"
    X_LINKED_DOMINANT = "XLD"
    X_LINKED_RECESSIVE = "XLR"
    MITOCHONDRIAL = "MT"
    UNKNOWN = "UNKNOWN"


class Zygosity(str, Enum):
    HOM_REF = "HOM_REF"
    HET = "HET"
    HOM_ALT = "HOM_ALT"
    HEMIZYGOUS = "HEMI"
    UNKNOWN = "UNKNOWN"


class ACMGClassification(str, Enum):
    PATHOGENIC = "Pathogenic"
    LIKELY_PATHOGENIC = "Likely Pathogenic"
    VUS = "Variant of Uncertain Significance"
    LIKELY_BENIGN = "Likely Benign"
    BENIGN = "Benign"


class FunctionalImpact(str, Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    MODIFIER = "MODIFIER"


# ── Variant-level models ──────────────────────────────────────────────────────
class Variant(BaseModel):
    """Single variant with core VCF fields."""
    chrom: str
    pos: int
    ref: str
    alt: str
    variant_id: str = ""
    quality: float = 0.0

    # Genotype info
    zygosity: Zygosity = Zygosity.UNKNOWN
    genotype: str = ""
    depth: int = 0
    gq: int = 0
    allele_depth_ref: int = 0
    allele_depth_alt: int = 0

    # VCF INFO fields (carried through for fallback annotation)
    info_fields: dict = Field(default_factory=dict)

    # Sample info
    sample_id: str = ""

    @property
    def variant_key(self) -> str:
        return f"{self.chrom}-{self.pos}-{self.ref}-{self.alt}"

    @property
    def is_snv(self) -> bool:
        return len(self.ref) == 1 and len(self.alt) == 1

    @property
    def is_indel(self) -> bool:
        return len(self.ref) != len(self.alt)


class AnnotatedVariant(BaseModel):
    """Variant with full annotation layer."""
    variant: Variant

    # Gene-level
    gene_symbol: str = ""
    gene_id: str = ""  # Ensembl ID
    transcript_id: str = ""
    hgvs_c: str = ""
    hgvs_p: str = ""
    consequence: str = ""
    impact: FunctionalImpact = FunctionalImpact.MODIFIER

    # Population frequency
    gnomad_af: Optional[float] = None
    gnomad_af_popmax: Optional[float] = None
    gnomad_hom_count: int = 0

    # Population-specific frequency (QGP / local cohort)
    local_af: Optional[float] = None  # AF in patient's population (e.g. QGP)
    local_ac: int = 0  # Allele count in local cohort
    local_an: int = 0  # Total alleles in local cohort
    local_hom_count: int = 0
    local_population: str = ""  # Population label (e.g. "QGP", "GME", "QASP")
    population_af_ratio: Optional[float] = None  # local_af / gnomad_af (enrichment)
    is_founder_variant: bool = False  # Enriched in local pop, rare globally
    founder_enrichment: float = 0.0  # Fold-enrichment over global AF

    # Pathogenicity scores
    cadd_phred: Optional[float] = None
    cadd_raw: Optional[float] = None
    revel_score: Optional[float] = None
    spliceai_max: Optional[float] = None

    # Clinical
    clinvar_significance: str = ""
    clinvar_review_status: str = ""
    clinvar_id: str = ""
    omim_id: str = ""

    # ── Non-coding / Regulatory annotations ──────────────────────────────────
    # Regulatory region classification
    regulatory_class: str = ""  # enhancer, promoter, TFBS, silencer, insulator, open_chromatin
    regulatory_feature_id: str = ""  # ENCODE / Ensembl Regulatory Build ID
    epigenome_tissue: str = ""  # Tissue/cell type with peak signal

    # Deep learning regulatory scores (0-1, higher = more disruptive)
    deepbind_score: Optional[float] = None  # TF binding disruption
    enformer_score: Optional[float] = None  # Enformer gene expression change
    sei_score: Optional[float] = None  # Sei regulatory activity
    regulatory_score: float = 0.0  # Unified non-coding impact (0-1)

    # Chromatin state (Roadmap / ChromHMM)
    chromhmm_state: str = ""  # e.g., "1_TssA", "7_Enh", "13_ReprPC"
    chromhmm_state_name: str = ""  # e.g., "Active TSS", "Enhancers"

    # Conservation / constraint
    phastcons_score: Optional[float] = None  # PhastCons (0-1, higher = more conserved)
    phylop_score: Optional[float] = None  # PhyloP (positive = conserved, negative = accelerated)
    gerp_score: Optional[float] = None  # GERP++ RS score

    # Variant-to-gene mapping (critical for non-coding)
    target_gene_symbol: str = ""  # Gene regulated by this non-coding variant
    target_gene_id: str = ""  # Ensembl ID of target gene
    gene_mapping_method: str = ""  # eqtl, chromatin_interaction, tad, nearest, abc
    gene_mapping_score: float = 0.0  # Confidence of variant-to-gene link (0-1)
    eqtl_tissue: str = ""  # GTEx tissue with significant eQTL
    eqtl_pvalue: Optional[float] = None  # eQTL p-value
    eqtl_beta: Optional[float] = None  # eQTL effect size
    hic_score: Optional[float] = None  # Hi-C interaction score
    abc_score: Optional[float] = None  # Activity-by-Contact model score

    # ── Trio inheritance classification ──────────────────────────────────────
    inheritance_class: str = ""  # "de_novo", "compound_het", "hom_recessive", "x_linked", "inherited_dominant", ""
    is_de_novo: bool = False
    is_compound_het: bool = False  # Part of a compound-het pair
    compound_het_partner_key: str = ""  # variant_key of the other allele
    is_hom_recessive: bool = False  # Homozygous recessive confirmed by trio
    is_inherited: bool = False  # Present in at least one parent
    parent_of_origin: str = ""  # "paternal", "maternal", "both", ""

    # Computed scores
    pathogenicity_score: float = 0.0
    rarity_score: float = 0.0
    functional_score: float = 0.0
    noncoding_impact_score: float = 0.0  # Aggregated non-coding regulatory impact
    inheritance_score: float = 0.0  # Trio-aware inheritance score
    composite_score: float = 0.0

    @property
    def is_noncoding(self) -> bool:
        """True if variant is in a non-coding region."""
        noncoding_consequences = {
            "intron_variant", "intergenic_variant", "upstream_gene_variant",
            "downstream_gene_variant", "regulatory_region_variant",
            "TF_binding_site_variant", "TFBS_ablation", "TFBS_amplification",
            "regulatory_region_ablation", "regulatory_region_amplification",
            "non_coding_transcript_exon_variant", "non_coding_transcript_variant",
            "mature_miRNA_variant", "5_prime_UTR_variant", "3_prime_UTR_variant",
        }
        return any(c.strip() in noncoding_consequences for c in self.consequence.split(","))

    @property
    def has_regulatory_annotation(self) -> bool:
        """True if variant has any regulatory annotation data."""
        return bool(self.regulatory_class or self.chromhmm_state or self.regulatory_score > 0)

    @property
    def effective_gene_symbol(self) -> str:
        """Gene symbol, preferring target gene for non-coding variants."""
        if self.is_noncoding and self.target_gene_symbol:
            return self.target_gene_symbol
        return self.gene_symbol

    @property
    def is_rare(self) -> bool:
        """Rare if both global AND local population AF are below threshold."""
        if self.gnomad_af is None and self.local_af is None:
            return True
        # Use the most specific frequency available
        af = self.effective_af
        return af < 0.01

    @property
    def effective_af(self) -> float:
        """Best available allele frequency, preferring population-specific.

        For rare disease in a specific population, the local AF is more
        informative than global gnomAD AF. A variant common in QGP but
        rare in gnomAD should NOT be treated as rare for a Qatari patient.
        """
        if self.local_af is not None:
            # When local AF available, use max of local and global
            # This prevents false positives from founder variants
            gnomad = self.gnomad_af or 0.0
            return max(self.local_af, gnomad)
        return self.gnomad_af or 0.0

    @property
    def is_novel(self) -> bool:
        return (
            (self.gnomad_af is None or self.gnomad_af == 0.0)
            and (self.local_af is None or self.local_af == 0.0)
        )


# ── Gene-level models ─────────────────────────────────────────────────────────
class GeneCandidate(BaseModel):
    """Candidate gene with aggregated evidence."""
    gene_symbol: str
    gene_id: str = ""

    # Variants in this gene
    variants: list[AnnotatedVariant] = Field(default_factory=list)

    # Scores
    max_variant_score: float = 0.0
    phenotype_score: float = 0.0
    gene_rank_score: float = 0.0
    confidence: float = 0.0

    # Inheritance
    inheritance_modes: list[InheritanceMode] = Field(default_factory=list)
    inheritance_compatible: bool = False

    # Trio-based inheritance evidence
    has_de_novo: bool = False
    has_de_novo_lof: bool = False  # De novo + HIGH impact (strongest signal)
    has_compound_het: bool = False
    has_hom_recessive: bool = False
    inheritance_score: float = 0.0  # Trio-aware inheritance contribution (0-1)
    trio_analyzed: bool = False

    # OMIM / disease association
    known_diseases: list[str] = Field(default_factory=list)
    is_known_disease_gene: bool = False

    # Non-coding / regulatory evidence
    max_regulatory_score: float = 0.0
    has_regulatory_variant: bool = False
    n_noncoding_variants: int = 0

    # Structural variant evidence
    structural_variants: list["AnnotatedSV"] = Field(default_factory=list)
    max_sv_score: float = 0.0
    has_sv: bool = False
    sv_fully_deleted: bool = False  # Gene entirely removed by a DEL
    sv_dosage_sensitive: bool = False

    # Multi-omics evidence
    multi_omics: Optional["MultiOmicsEvidence"] = None
    multi_omics_score: float = 0.0
    n_evidence_layers: int = 0

    # Knowledge graph evidence
    kg_score: float = 0.0  # Graph propagation score (0-1)
    kg_rank: int = 0  # Rank from KG analysis
    kg_paths: list[str] = Field(default_factory=list)  # Top paths explaining KG score
    kg_connected_diseases: list[str] = Field(default_factory=list)
    kg_connected_pathways: list[str] = Field(default_factory=list)
    kg_ppi_neighbors: int = 0  # Number of PPI neighbors among candidates

    # Explainability
    explanation: str = ""
    evidence_summary: dict = Field(default_factory=dict)

    @property
    def n_variants(self) -> int:
        return len(self.variants)

    @property
    def has_lof_variant(self) -> bool:
        return any(
            v.impact in (FunctionalImpact.HIGH,)
            for v in self.variants
        )

    @property
    def has_splicing_variant(self) -> bool:
        return any(
            (v.spliceai_max is not None and v.spliceai_max >= 0.5)
            for v in self.variants
        )


# ── Phenotype models ──────────────────────────────────────────────────────────
class HPOTerm(BaseModel):
    """Human Phenotype Ontology term."""
    id: str  # HP:0000001 format
    name: str = ""
    definition: str = ""
    is_obsolete: bool = False
    parents: list[str] = Field(default_factory=list)
    children: list[str] = Field(default_factory=list)


class PatientPhenotype(BaseModel):
    """Patient phenotype profile."""
    patient_id: str
    hpo_terms: list[HPOTerm] = Field(default_factory=list)
    negated_terms: list[HPOTerm] = Field(default_factory=list)
    age_of_onset: Optional[str] = None
    sex: Optional[str] = None


# ── Pedigree models ───────────────────────────────────────────────────────────
class PedigreeMember(BaseModel):
    """Single family member in a pedigree."""
    individual_id: str
    family_id: str = ""
    father_id: Optional[str] = None
    mother_id: Optional[str] = None
    sex: str = "unknown"  # male, female, unknown
    affected: bool = False
    vcf_sample_id: Optional[str] = None


class Pedigree(BaseModel):
    """Family pedigree for inheritance analysis."""
    family_id: str
    members: list[PedigreeMember] = Field(default_factory=list)
    proband_id: str = ""

    @property
    def proband(self) -> Optional[PedigreeMember]:
        for m in self.members:
            if m.individual_id == self.proband_id:
                return m
        return None

    @property
    def is_trio(self) -> bool:
        return len(self.members) == 3

    @property
    def affected_members(self) -> list[PedigreeMember]:
        return [m for m in self.members if m.affected]


# ── Report model ──────────────────────────────────────────────────────────────
class ClinicalReport(BaseModel):
    """Final clinical report output."""
    patient_id: str
    report_id: str = ""
    report_date: str = ""

    phenotype: PatientPhenotype
    pedigree: Optional[Pedigree] = None

    ranked_genes: list[GeneCandidate] = Field(default_factory=list)
    total_variants_analyzed: int = 0
    total_genes_analyzed: int = 0
    total_svs_analyzed: int = 0

    pipeline_version: str = "1.0.0"
    genome_build: str = "GRCh38"
    analysis_parameters: dict = Field(default_factory=dict)


# ── Structural Variant models ─────────────────────────────────────────────────
class StructuralVariant(BaseModel):
    """Raw structural variant from SV VCF (Sniffles/Jasmine/Manta/DELLY)."""
    chrom: str
    pos: int
    end: int = 0
    sv_type: SVType = SVType.UNKNOWN
    sv_len: int = 0  # Absolute length in bp
    sv_id: str = ""
    quality: float = 0.0
    filter_status: str = "PASS"

    # Breakpoint detail
    chrom2: str = ""  # For BND / translocations
    pos2: int = 0

    # Genotype
    zygosity: Zygosity = Zygosity.UNKNOWN
    genotype: str = ""
    sample_id: str = ""

    # Read support (from SV callers)
    support_reads: int = 0  # Number of supporting reads
    ref_reads: int = 0
    allele_fraction: float = 0.0

    # Caller metadata
    caller: str = ""  # sniffles, jasmine, manta, delly
    confidence: str = ""  # PRECISE or IMPRECISE

    @property
    def variant_key(self) -> str:
        return f"{self.chrom}-{self.pos}-{self.sv_type.value}-{self.sv_len}"

    @property
    def is_deletion(self) -> bool:
        return self.sv_type in (SVType.DEL, SVType.DEL_ME)

    @property
    def is_duplication(self) -> bool:
        return self.sv_type in (SVType.DUP, SVType.DUP_TANDEM, SVType.CNV)

    @property
    def is_large(self) -> bool:
        """SVs >1 Mb are considered large."""
        return self.sv_len > 1_000_000

    @property
    def size_category(self) -> str:
        if self.sv_len < 1_000:
            return "small"
        elif self.sv_len < 100_000:
            return "medium"
        elif self.sv_len < 1_000_000:
            return "large"
        return "very_large"


class AnnotatedSV(BaseModel):
    """Structural variant with full annotation layer."""
    sv: StructuralVariant

    # ── Gene overlap ──────────────────────────────────────────────────────
    overlapping_genes: list[str] = Field(default_factory=list)
    fully_deleted_genes: list[str] = Field(default_factory=list)
    partially_overlapping_genes: list[str] = Field(default_factory=list)
    exons_affected: int = 0
    coding_overlap_bp: int = 0

    # ── AnnotSV classification ────────────────────────────────────────────
    annotsv_class: str = ""  # "pathogenic", "likely_pathogenic", "benign", "VUS"
    annotsv_rank: int = 0   # 1-5 (AnnotSV ranking)

    # ── Dosage sensitivity ────────────────────────────────────────────────
    hi_score: Optional[float] = None    # Haploinsufficiency score (ClinGen/DECIPHER)
    ts_score: Optional[float] = None    # Triplosensitivity score
    pli_score: Optional[float] = None   # gnomAD pLI (prob. of LoF intolerance)
    loeuf_score: Optional[float] = None # gnomAD LOEUF (obs/exp LoF upper bound)

    # ── Population frequency ──────────────────────────────────────────────
    dgv_frequency: Optional[float] = None   # Database of Genomic Variants
    gnomad_sv_af: Optional[float] = None    # gnomAD-SV allele frequency
    dbvar_id: str = ""

    # ── Regulatory disruption ─────────────────────────────────────────────
    disrupted_tads: list[str] = Field(default_factory=list)
    disrupted_enhancers: int = 0
    disrupted_promoters: int = 0
    disrupted_ctcf_sites: int = 0
    regulatory_disruption_score: float = 0.0

    # ── Clinical databases ────────────────────────────────────────────────
    clinvar_sv_significance: str = ""
    decipher_id: str = ""
    known_syndrome: str = ""

    # ── Computed scores ───────────────────────────────────────────────────
    gene_overlap_score: float = 0.0     # Gene disruption severity (0-1)
    dosage_sensitivity_score: float = 0.0  # Dosage intolerance (0-1)
    sv_pathogenicity_score: float = 0.0    # Overall SV pathogenicity (0-1)
    sv_rarity_score: float = 0.0           # Population rarity (0-1)
    sv_composite_score: float = 0.0        # Final composite score

    @property
    def is_rare(self) -> bool:
        if self.gnomad_sv_af is None and self.dgv_frequency is None:
            return True
        af = self.gnomad_sv_af or self.dgv_frequency or 0.0
        return af < 0.01

    @property
    def is_novel(self) -> bool:
        return self.gnomad_sv_af is None and self.dgv_frequency is None

    @property
    def n_genes_affected(self) -> int:
        return len(self.overlapping_genes)

    @property
    def has_dosage_sensitive_gene(self) -> bool:
        return (
            (self.pli_score is not None and self.pli_score > 0.9)
            or (self.hi_score is not None and self.hi_score >= 3)
        )


# ── Multi-omics models ───────────────────────────────────────────────────────
class GeneExpression(BaseModel):
    """Per-gene RNA-seq expression data for a single sample."""
    gene_symbol: str
    gene_id: str = ""
    tpm: float = 0.0           # Transcripts per million
    raw_count: int = 0         # Raw read count
    log2_tpm: float = 0.0      # log2(TPM + 1)

    # Outlier detection against reference cohort
    z_score: float = 0.0       # Z-score vs reference distribution
    percentile: float = 50.0   # Percentile rank in reference
    is_outlier: bool = False
    outlier_direction: str = ""  # "under", "over", or ""
    reference_median: float = 0.0
    reference_mad: float = 0.0  # Median absolute deviation

    @property
    def is_underexpressed(self) -> bool:
        return self.is_outlier and self.outlier_direction == "under"

    @property
    def is_overexpressed(self) -> bool:
        return self.is_outlier and self.outlier_direction == "over"


class MethylationRegion(BaseModel):
    """Differentially methylated region (DMR) overlapping a gene."""
    chrom: str = ""
    start: int = 0
    end: int = 0
    gene_symbol: str = ""

    # Methylation measurements
    mean_beta_patient: float = 0.0   # Patient mean beta (0-1)
    mean_beta_control: float = 0.5   # Control cohort mean beta
    delta_beta: float = 0.0          # Patient - control
    n_cpgs: int = 0                  # Number of CpGs in region

    # Statistical significance
    p_value: Optional[float] = None
    fdr: Optional[float] = None

    # Classification
    is_dmr: bool = False
    dmr_direction: str = ""  # "hyper", "hypo", or ""
    region_type: str = ""    # "promoter", "gene_body", "shore", "shelf", "enhancer"

    @property
    def is_hypermethylated(self) -> bool:
        return self.is_dmr and self.dmr_direction == "hyper"

    @property
    def is_hypomethylated(self) -> bool:
        return self.is_dmr and self.dmr_direction == "hypo"

    @property
    def length(self) -> int:
        return self.end - self.start


class MultiOmicsEvidence(BaseModel):
    """Aggregated multi-omics evidence for a single gene.

    Integrates genomic (SNV/SV), transcriptomic (RNA-seq), and
    epigenomic (methylation) evidence layers.
    """
    gene_symbol: str

    # ── Expression evidence ────────────────────────────────────────────
    expression: Optional[GeneExpression] = None
    expression_score: float = 0.0  # Outlier severity (0-1)
    has_expression_outlier: bool = False

    # ── Methylation evidence ───────────────────────────────────────────
    methylation_regions: list[MethylationRegion] = Field(default_factory=list)
    methylation_score: float = 0.0  # DMR severity (0-1)
    has_dmr: bool = False
    has_promoter_dmr: bool = False

    # ── Evidence layer integration ─────────────────────────────────────
    n_evidence_layers: int = 0   # Count of supporting data types (0-4)
    evidence_layers: list[str] = Field(default_factory=list)  # e.g. ["genomic","expression","methylation"]
    multi_omics_score: float = 0.0   # Combined confidence (0-1)
    concordance_bonus: float = 0.0   # Bonus for concordant evidence direction

    # Direction concordance: does expression + methylation + variant agree?
    # e.g. LoF variant + underexpression + promoter hypermethylation = concordant
    is_concordant: bool = False
    concordance_description: str = ""
