# RareGeneAI — Full Methodology and Execution Guide

## 1. System Overview

RareGeneAI is a production-grade rare disease gene prioritization system
that takes WGS/WES VCF input, integrates phenotype (HPO terms) and
multi-modal evidence, and ranks candidate genes for clinical diagnosis.

```
                          INPUTS
    ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ Proband │  │ Father   │  │ Mother   │  │ SV VCF   │
    │ VCF     │  │ VCF      │  │ VCF      │  │(optional)│
    └────┬────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
         │            │             │              │
    ┌────┴────┐  ┌────┴─────────────┴────┐   ┌────┴─────┐
    │ HPO     │  │ Trio (optional)       │   │ RNA-seq  │
    │ Terms   │  │                       │   │ Methyl.  │
    └────┬────┘  └───────────────────────┘   └────┬─────┘
         │                                        │
         └──────────────┬─────────────────────────┘
                        │
                  ┌─────▼─────┐
                  │ RareGeneAI│
                  │ 12-Step   │
                  │ Pipeline  │
                  └─────┬─────┘
                        │
                  ┌─────▼──────────────────────────────┐
                  │ OUTPUT                              │
                  │ ● Ranked gene list with scores      │
                  │ ● ACMG/AMP variant classification   │
                  │ ● Clinical report (HTML/PDF)        │
                  │ ● SHAP explanations                 │
                  │ ● Actionable findings (SF, PGx)     │
                  └────────────────────────────────────┘
```

**Codebase: 40 source files | 11,900 lines | 311 tests | 19 modules**

---

## 2. The 12-Step Pipeline

```
Step  1: Data Ingestion       — Parse VCF (cyvcf2), validate HPO terms, load PED
Step  2: Variant Annotation   — VEP + gnomAD + QGP/population AF (parallel)
Step  3: Trio Inheritance     — De novo / compound het / hom recessive detection
Step  4: Variant Scoring      — Composite: pathogenicity + rarity + impact + inheritance + regulatory
Step  5: Variant Filtering    — Quality, rarity, coding/non-coding thresholds
Step  6: SV Analysis          — Parse SV VCF, annotate gene overlap + dosage + regulatory
Step  7: Phenotype Matching   — HPO semantic similarity (Resnik/Lin IC)
Step  8: Gene Ranking         — XGBoost (44 features) or rule-based fusion
Step  9: Knowledge Graph      — Random Walk with Restart from patient HPO nodes
Step 10: Multi-omics          — Expression outliers + methylation DMRs + concordance
Step 11: Clinical Decision    — ACMG/AMP classification + ACMG SF + pharmacogenomics
Step 12: Report Generation    — HTML/PDF clinical report with evidence badges
```

---

## 3. Module-by-Module Methodology

### 3.1 Data Ingestion (`ingestion/`)

**VCF Parser** (`vcf_parser.py`):
- Uses cyvcf2 (C-backed) for high-performance VCF parsing
- Falls back to pure-Python parser when cyvcf2 unavailable
- Extracts: GT, DP, GQ, AD per variant per sample
- Filters: min_gq=20, min_dp=10 (configurable)
- Normalizes: splits multi-allelic, left-aligns via bcftools

**HPO Parser** (`hpo_parser.py`):
- Validates HPO IDs against the ontology (pronto library)
- Falls back to HPO REST API for validation
- Supports negated phenotypes (NOT present)

**Pedigree Parser** (`pedigree_parser.py`):
- Standard PED format (6-column tab-separated)
- Identifies proband, parents, affected status

### 3.2 Variant Annotation (`annotation/`)

Runs in parallel where possible (gnomAD + population run concurrently):

**VEP Annotator** (`vep_annotator.py`):
- Local VEP installation or Ensembl REST API
- Extracts: gene, consequence, impact, HGVS, transcript
- Picks canonical transcript

**Frequency Annotator** (`frequency_annotator.py`):
- gnomAD v4 (local tabix or GraphQL API)
- Fields: AF, AF_popmax, homozygote_count

**Population Annotator** (`population/population_annotator.py`):
- QGP, GME, or custom local cohort frequencies
- Founder variant detection: local_af/gnomad_af >= 10x enrichment
- 18 known Middle Eastern founder genes (MEFV, GJB2, HBB, etc.)
- Population-adjusted effective_af = max(local_af, gnomad_af)

**Pathogenicity Annotator** (`pathogenicity_annotator.py`):
- CADD (local tabix or API)
- REVEL (missense only)
- SpliceAI (splice disruption)
- ClinVar (local VCF or NCBI API)

**Regulatory Annotator** (`regulatory_annotator.py`):
- ENCODE cCREs (enhancer/promoter/insulator classification)
- Roadmap ChromHMM 15-state model
- Conservation: PhastCons, PhyloP, GERP++
- Variant-to-gene mapping: ABC model > eQTL (GTEx) > Hi-C > nearest
- Deep learning score estimation (Enformer/Sei proxy)

### 3.3 Variant Scoring (`scoring/`)

**Composite Score Formula:**
```
composite = 0.30 × pathogenicity
          + 0.20 × rarity
          + 0.15 × functional_impact
          + 0.15 × regulatory_impact
          + 0.15 × phenotype
          + 0.05 × inheritance
```

**Pathogenicity** = max(CADD/40, REVEL, SpliceAI, Enformer) + ClinVar boost

**Rarity** = exp(-1000 × effective_af), where effective_af = max(local_af, gnomad_af)
- Founder variants: pathogenic founders use softer decay (exp(-500×AF))
- Non-pathogenic founders get harsher decay (exp(-2000×AF))

**Inheritance Score** (trio-aware):
```
De novo LoF:           1.00    (strongest signal in rare disease)
De novo missense:      0.90
Compound het LoF+LoF:  0.95
Compound het LoF+mis:  0.85
Hom recessive LoF:     0.90
X-linked hemi LoF:     0.95
Inherited dominant:    0.40
No trio (HET):         0.30
```

### 3.4 Structural Variant Analysis (`structural/`)

**SV Parser** (`sv_parser.py`):
- Parses Sniffles2, Jasmine, Manta, DELLY, cuteSV output
- Extracts: SVTYPE, SVLEN, END, read support, genotype
- Filters: min size=50bp, min quality=5, min support reads=3

**SV Annotator** (`sv_annotator.py`):
- Gene overlap: full deletion vs partial overlap
- Dosage sensitivity: pLI, LOEUF, ClinGen HI/TS scores
- Population frequency: gnomAD-SV, DGV
- Regulatory disruption: TAD boundaries, enhancers, CTCF sites
- Clinical: ClinVar SVs, DECIPHER syndromes
- AnnotSV batch integration (if installed)

**SV Score:**
```
sv_composite = 0.30 × gene_overlap + 0.25 × dosage_sensitivity
             + 0.20 × rarity + 0.15 × regulatory_disruption
             + 0.10 × clinical_databases
```

### 3.5 Phenotype Matching (`phenotype/`)

**Semantic Similarity** (`semantic_similarity.py`):
- Information Content via HPO ontology DAG
- Three methods: Resnik (default), Lin, Jiang-Conrath
- IC(t) = -log2(|descendants(t)| / |total_terms|)

**Gene-Phenotype Matcher** (`gene_phenotype_matcher.py`):
- Best-Match Average: BMA(patient, gene) = (1/|P|) × Σ_p max_g sim(p,g)
- Gene-HPO associations from HPO annotation database
- Downloads automatically from HPO if not cached locally

### 3.6 Gene Ranking (`ranking/`)

**44-Feature XGBoost Model** (`gene_ranker.py` + `model_trainer.py`):

Feature groups:
```
Variant pathogenicity  (10): max_cadd, max_revel, has_lof, has_clinvar_pathogenic, ...
Non-coding regulatory   (8): max_spliceai, has_enhancer_variant, max_conservation, ...
Structural variants     (7): has_sv, sv_dosage_sensitive, max_sv_gene_overlap, ...
Multi-omics             (8): expression_score, methylation_score, is_concordant, ...
Trio inheritance        (6): has_de_novo_lof, has_compound_het, trio_inheritance_score, ...
Knowledge graph         (5): kg_score, kg_ppi_neighbors, kg_n_diseases, ...
```

Training: 5-fold stratified CV, scale_pos_weight for class imbalance
Evaluation: ROC-AUC, PR-AUC, Top-1/5/10 accuracy, MRR
Interpretation: SHAP TreeExplainer (with XGBoost native fallback)

**Rule-based fallback** (when no trained model):
```
gene_score = 0.20 × variant_score + 0.18 × phenotype + 0.10 × de_novo_lof_bonus
           + 0.08 × sv_bonus + 0.08 × regulatory + 0.06 × clinvar + ...
```

### 3.7 Knowledge Graph (`knowledge_graph/`)

**Graph Construction** (`graph_builder.py`):
- 4 node types: Gene, Phenotype (HPO), Disease (OMIM), Pathway (KEGG/Reactome)
- 5 edge types: gene-phenotype, gene-disease, disease-phenotype, PPI (STRING), gene-pathway

**Graph Propagation** (`graph_scorer.py`):
- Random Walk with Restart (default): restart_prob=0.4, seed=patient HPO nodes
- Also: Personalized PageRank, Network Diffusion
- Extracts: per-gene score, connected diseases/pathways, PPI neighbors, explanatory paths

### 3.8 Multi-omics Integration (`multiomics/`)

**Expression Outlier Detection** (`expression_outlier.py`):
- Z-score against reference cohort (median + MAD for robustness)
- Outlier: |Z| > 2.0 (configurable)
- Handles TPM, counts, FPKM input formats

**Methylation Analysis** (`methylation_analyzer.py`):
- Loads pre-called DMRs or calls from raw beta values
- Overlaps with gene promoter regions
- Hyper/hypo classification with delta_beta threshold

**Evidence Integration** (`integrator.py`):
- Concordance detection: underexpression + promoter hypermethylation = silencing
- Multi-omics score with concordance multiplier (1.3x)
- Layer counting: genomic + expression + methylation = 3 layers

### 3.9 Clinical Decision Support (`clinical/`)

**ACMG/AMP Classifier** (`acmg_classifier.py`):
- 18 evidence criteria: PVS1, PS1, PS3, PM1-PM5, PP1, PP3, PP5, BA1, BS1-BS2, BP1, BP4, BP6-BP7
- Table 5 combining rules for Pathogenic/LP/VUS/LB/Benign
- Full audit trail: each criterion has code, strength, direction, justification

**Clinical Decision Engine** (`clinical_decision.py`):
- ACMG SF v3.2: 60 actionable genes (cardiovascular, cancer, metabolic)
- CPIC pharmacogenomics: 16 genes, 40+ drugs (CYP2D6, DPYD, HLA-B, etc.)
- Clinical significance: Diagnostic / Likely Diagnostic / Actionable / PGx / Research
- Recommendation text with mandatory "board-certified review required"

### 3.10 Continuous Learning (`learning/`)

**Feedback Store** (`feedback_store.py`):
- Append-only JSON-lines file
- Captures: confirmed diagnosis, rejected gene, VUS reclassification
- Each entry: UUID, timestamp, analyst, evidence snapshot, model version

**Model Registry** (`model_registry.py`):
- Versioned model artifacts with JSON manifest
- Lifecycle: candidate → staging → production → retired
- Rollback capability
- Delta metrics vs previous version

**Continuous Trainer** (`continuous_trainer.py`):
- Collects confirmed diagnoses from feedback store
- Merges with ClinVar/OMIM baseline training data
- Retrains XGBoost, evaluates, registers new version
- Auto-promote if ΔAUC > threshold

---

## 4. Execution Guide

### 4.1 Installation

```bash
cd RareGeneAI

# Install Python package
pip install -e .

# Verify installation
raregeneai --version
python -m pytest tests/ -q
```

### 4.2 Download Reference Data

```bash
# Download HPO ontology, gene-phenotype associations, ClinVar
bash scripts/download_references.sh data/reference/

# Install VEP cache (optional, for local VEP)
vep_install --AUTO cf --ASSEMBLY GRCh38 --SPECIES homo_sapiens
```

### 4.3 Basic Analysis (Minimum Input)

```bash
# Just VCF + HPO terms
raregeneai analyze \
  --vcf patient.vcf.gz \
  --hpo HP:0001250 \
  --hpo HP:0002878 \
  --hpo HP:0001263 \
  --output results/
```

This runs the full 12-step pipeline with remote API annotation.
Output: `results/PATIENT_001_report.html` + `results/PATIENT_001_variants.parquet`

### 4.4 Full Multi-Modal Analysis (All Inputs)

```bash
raregeneai analyze \
  --vcf proband.vcf.gz \
  --father-vcf father.vcf.gz \
  --mother-vcf mother.vcf.gz \
  --sv-vcf proband.sv.vcf.gz \
  --expression rnaseq_tpm.tsv \
  --methylation dmr_calls.tsv \
  --hpo HP:0001250 \
  --hpo HP:0002878 \
  --ped family.ped \
  --config config.yaml \
  --top-n 30 \
  --output results/
```

### 4.5 Configuration

Create `config.yaml` (or use `raregeneai init-config`):

```yaml
genome_build: GRCh38
n_threads: 4
include_noncoding: true

annotation:
  vep_assembly: GRCh38
  use_remote_api: true
  population:
    enabled: true
    population: QGP
    qgp_af_path: data/reference/qgp_af.tsv

scoring:
  w_pathogenicity: 0.30
  w_rarity: 0.20
  gnomad_af_threshold: 0.01

ranking:
  model_type: xgboost
  pretrained_model_path: models/clinical_v1.pkl
  top_n_genes: 50

knowledge_graph:
  enabled: true
  algorithm: rwr
  restart_probability: 0.4

multiomics:
  enabled: true
  z_score_threshold: 2.0
```

### 4.6 Web Interface

```bash
# Launch Streamlit UI (browser-based)
raregeneai ui
# Opens http://localhost:8501

# Or launch REST API
uvicorn raregeneai.ui.api:app --host 0.0.0.0 --port 8000
# POST /analyze with VCF file + HPO terms
```

### 4.7 Docker Deployment

```bash
# Build image
docker build -f docker/Dockerfile -t raregeneai:latest .

# Run analysis
docker run -v $(pwd)/data:/app/data raregeneai:latest analyze \
  --vcf /app/data/patient.vcf.gz \
  --hpo HP:0001250

# Run with docker-compose (API + UI)
cd docker && docker-compose up
# API: http://localhost:8000
# UI:  http://localhost:8501
```

### 4.8 Nextflow Pipeline (HPC/Cloud)

```bash
nextflow run nextflow/main.nf \
  --vcf patient.vcf.gz \
  --hpo "HP:0001250,HP:0002878" \
  --outdir results/ \
  -profile docker
```

### 4.9 Train the ML Model

```bash
# Generate synthetic benchmark + train
python scripts/clinical_validation.py \
  --n-bystanders 100 \
  --save-model models/clinical_v1.pkl

# Or train on your own labeled data (CSV)
python -c "
from raregeneai.ranking.model_trainer import ModelTrainer
trainer = ModelTrainer()
X, y = trainer.build_training_data_from_csv('training_data.csv')
trainer.train(X, y, save_path='models/my_model.pkl')
trainer.explain_with_shap(X)
"
```

### 4.10 Submit Clinician Feedback

```python
from raregeneai.learning.feedback_store import FeedbackStore

store = FeedbackStore("data/feedback/feedback.jsonl")

# Confirmed diagnosis
store.submit_confirmed_diagnosis(
    patient_id="P001",
    gene_symbol="SCN1A",
    diagnosis="Dravet syndrome",
    original_rank=1,
    confirmation_method="sanger",
    analyst_id="Dr. Smith",
)

# Periodically retrain
from raregeneai.learning.continuous_trainer import ContinuousTrainer
from raregeneai.learning.model_registry import ModelRegistry

registry = ModelRegistry("models/registry")
trainer = ContinuousTrainer(store, registry)
trainer.run_retrain_cycle(min_feedback=20, auto_promote=True)
```

### 4.11 Run Tests

```bash
# All 311 tests
python -m pytest tests/ -v

# Specific module
python -m pytest tests/unit/test_trio_inheritance.py -v

# With coverage
python -m pytest tests/ --cov=raregeneai --cov-report=html
```

---

## 5. Output Files

| File | Content |
|------|---------|
| `{patient}_report.html` | Clinical report with ranked genes, evidence badges, ACMG classification |
| `{patient}_variants.parquet` | Full annotated variant table (all columns) |
| `models/clinical_v1.pkl` | Trained XGBoost model bundle (model + metrics + features) |
| `models/clinical_v1.metrics.json` | Model performance metrics (human-readable) |
| `data/feedback/feedback.jsonl` | Clinician feedback log (append-only) |
| `models/registry/model_registry.json` | Model version manifest |

---

## 6. Input File Formats

| File | Format | Required |
|------|--------|----------|
| Proband VCF | VCF 4.x (.vcf or .vcf.gz) | Yes |
| HPO terms | CLI arguments (HP:XXXXXXX) | Yes |
| Father VCF | VCF 4.x | No (trio) |
| Mother VCF | VCF 4.x | No (trio) |
| SV VCF | Sniffles/Manta/DELLY VCF | No |
| PED file | PLINK PED (6-column tab) | No |
| RNA-seq | gene<TAB>tpm (TSV) | No |
| Methylation | chrom<TAB>start<TAB>end<TAB>gene<TAB>delta_beta<TAB>pvalue (TSV) | No |
| Population AF | chrom<TAB>pos<TAB>ref<TAB>alt<TAB>af<TAB>ac<TAB>an<TAB>hom (TSV) | No |
| Config | YAML | No (defaults used) |

---

## 7. Architecture Diagram

```
raregeneai/
├── annotation/          # VEP, gnomAD, CADD, ClinVar, regulatory, population
├── clinical/            # ACMG/AMP classifier, SF v3.2, pharmacogenomics
├── config/              # Pydantic settings, default YAML
├── explainability/      # SHAP, rule-based explanations, ACMG audit
├── ingestion/           # VCF parser, HPO parser, PED parser
├── knowledge_graph/     # Graph builder (HPO/OMIM/STRING/KEGG), RWR scorer
├── learning/            # Feedback store, model registry, continuous trainer
├── models/              # Pydantic data models (Variant, Gene, SV, etc.)
├── multiomics/          # RNA-seq outliers, methylation DMRs, integrator
├── phenotype/           # Semantic similarity, gene-phenotype matching
├── pipeline/            # 12-step orchestrator
├── population/          # QGP/GME/local AF, founder variant detection
├── ranking/             # XGBoost trainer (44 features), gene ranker
├── reporting/           # HTML/PDF clinical report generator
├── scoring/             # Variant composite scorer, inheritance analyzer
├── structural/          # SV parser, annotator, integration bridge
├── ui/                  # Streamlit app, FastAPI REST API
├── utils/               # Logging, caching, parallel processing
└── validation/          # Benchmarker framework
```
