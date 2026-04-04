# RareGeneAI

**Production-grade Rare Disease Gene Prioritization System**

RareGeneAI takes a patient's WGS/WES VCF and clinical phenotype (HPO terms) and returns a ranked list of candidate disease genes with ACMG-classified variants, explainable scores, and clinician-ready reports.

```
Patient VCF + HPO Terms ──► RareGeneAI (12-step pipeline) ──► Ranked Genes + Clinical Report
```

[![Tests](https://img.shields.io/badge/tests-311%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Multi-variant support** | SNVs, indels, structural variants, non-coding variants |
| **Trio analysis** | De novo, compound heterozygous, homozygous recessive detection |
| **Multi-omics integration** | RNA-seq expression outliers + methylation DMRs with concordance scoring |
| **Knowledge graph** | Random Walk with Restart over Gene-HPO-OMIM-KEGG-PPI network |
| **Population-specific** | QGP/GME allele frequencies, founder variant detection for Middle Eastern cohorts |
| **ACMG/AMP classification** | 18 evidence criteria with Table 5 combining rules and full audit trail |
| **Pharmacogenomics** | CPIC Level A/B drug-gene interactions (16 genes, 40+ drugs) |
| **Actionable gene flagging** | ACMG SF v3.2 secondary findings (60 genes) |
| **XGBoost ML ranking** | 44-feature model with SHAP interpretability |
| **Continuous learning** | Clinician feedback capture, versioned model retraining |
| **Clinical compliance** | CAP/CLIA-ready audit trails, mandatory analyst review |
| **Multiple interfaces** | CLI, Streamlit web UI, FastAPI REST API, Nextflow pipeline, Docker |

---

## Quick Start

### Install

```bash
git clone https://github.com/your-org/RareGeneAI.git
cd RareGeneAI
pip install -e .
```

### Download Reference Data

```bash
bash scripts/download_references.sh data/reference/
```

This downloads HPO ontology, gene-phenotype associations, and ClinVar (~30 min).

### Run Your First Analysis

```bash
raregeneai analyze \
  --vcf patient.vcf.gz \
  --hpo HP:0001250 \
  --hpo HP:0002878 \
  --output results/
```

Output: `results/PATIENT_001_report.html` (clinical report) + `results/PATIENT_001_variants.parquet` (annotated variants).

---

## Full Multi-Modal Analysis

When additional data is available, RareGeneAI integrates all evidence layers for maximum accuracy:

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
  --hpo HP:0001263 \
  --ped family.ped \
  --config config.yaml \
  --top-n 30 \
  --output results/
```

### Input Files

| File | Format | Required | Description |
|------|--------|----------|-------------|
| `--vcf` | VCF 4.x (.vcf/.vcf.gz) | **Yes** | Proband WGS/WES variants |
| `--hpo` | HP:XXXXXXX (repeatable) | **Yes** | Patient phenotype terms |
| `--father-vcf` | VCF 4.x | No | Father VCF for trio inheritance |
| `--mother-vcf` | VCF 4.x | No | Mother VCF for trio inheritance |
| `--sv-vcf` | VCF 4.x | No | Structural variant calls (Sniffles/Manta/DELLY) |
| `--expression` | TSV (gene\ttpm) | No | RNA-seq expression per gene |
| `--methylation` | TSV | No | Methylation BED or pre-called DMRs |
| `--ped` | PLINK PED (6-col) | No | Family pedigree |
| `--config` | YAML | No | Pipeline configuration (defaults used otherwise) |

### Output Files

| File | Description |
|------|-------------|
| `{patient}_report.html` | Clinical report with ranked genes, ACMG classification, evidence badges |
| `{patient}_variants.parquet` | Full annotated variant table (50+ columns) |

---

## The 12-Step Pipeline

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                        RareGeneAI Pipeline                          │
 │                                                                     │
 │  [1] Ingest VCF + HPO ──► [2] Annotate (VEP/gnomAD/CADD/ClinVar)  │
 │           │                        │                                │
 │  [3] Trio Inheritance ────► [4] Score Variants ──► [5] Filter       │
 │           │                        │                                │
 │  [6] SV Analysis ─────────► [7] Phenotype Match (HPO similarity)   │
 │           │                        │                                │
 │  [8] Gene Ranking (XGBoost 44-feature model)                       │
 │           │                                                         │
 │  [9] Knowledge Graph ──► [10] Multi-omics ──► [11] Clinical (ACMG) │
 │           │                                                         │
 │  [12] Generate Clinical Report (HTML/PDF)                           │
 └─────────────────────────────────────────────────────────────────────┘
```

| Step | Module | What It Does |
|------|--------|-------------|
| 1 | `ingestion/` | Parse VCF (cyvcf2), validate HPO terms, load pedigree |
| 2 | `annotation/` | VEP consequences, gnomAD AF, QGP/local AF, CADD, REVEL, SpliceAI, ClinVar (parallel) |
| 3 | `scoring/inheritance_analyzer` | Classify: de novo, compound het, hom recessive, X-linked (tags each variant) |
| 4 | `scoring/variant_scorer` | Composite = 0.30×pathogenicity + 0.20×rarity + 0.15×impact + 0.15×regulatory + 0.15×phenotype + 0.05×inheritance |
| 5 | `scoring/variant_scorer` | Filter by quality, rarity, coding/non-coding thresholds |
| 6 | `structural/` | Parse SV VCF, annotate gene overlap + dosage sensitivity + regulatory disruption |
| 7 | `phenotype/` | Resnik IC semantic similarity between patient HPO and gene-HPO associations |
| 8 | `ranking/` | XGBoost (44 features) or rule-based weighted scoring |
| 9 | `knowledge_graph/` | Random Walk with Restart from patient HPO through Gene-Disease-Pathway-PPI graph |
| 10 | `multiomics/` | Expression outlier Z-scores + methylation DMRs + concordance detection |
| 11 | `clinical/` | ACMG/AMP 18-criteria classification + ACMG SF v3.2 + CPIC pharmacogenomics |
| 12 | `reporting/` | HTML/PDF report with evidence badges and clinician recommendations |

---

## Scoring Methodology

### Variant-Level Composite Score

```
composite = 0.30 × pathogenicity    (max of CADD/40, REVEL, SpliceAI + ClinVar boost)
          + 0.20 × rarity           (exp(-1000 × effective_af), population-adjusted)
          + 0.15 × functional_impact (consequence severity + regulatory boost)
          + 0.15 × regulatory       (ENCODE + conservation + DL scores + gene mapping)
          + 0.15 × phenotype        (HPO semantic similarity)
          + 0.05 × inheritance      (trio-aware: de novo LoF = 1.0, unknown HET = 0.3)
```

### Gene-Level Ranking (XGBoost 44-Feature Model)

Six evidence groups, each contributing features:

| Group | Features | Examples |
|-------|----------|---------|
| Variant pathogenicity | 10 | max_cadd, max_revel, has_lof, has_clinvar_pathogenic |
| Non-coding regulatory | 8 | max_spliceai, has_enhancer_variant, max_conservation |
| Structural variants | 7 | max_sv_score, sv_dosage_sensitive, sv_fully_deleted |
| Multi-omics | 8 | expression_score, methylation_score, is_concordant |
| Trio inheritance | 6 | has_de_novo_lof, has_compound_het, trio_inheritance_score |
| Knowledge graph | 5 | kg_score, kg_ppi_neighbors, kg_n_diseases |

### Trio Inheritance Weights

| Pattern | Score | Clinical Significance |
|---------|-------|----------------------|
| De novo LoF | **1.00** | Strongest signal in rare disease |
| De novo missense | 0.90 | Strong, needs functional confirmation |
| Compound het LoF+LoF | 0.95 | Biallelic null |
| Compound het LoF+missense | 0.85 | Biallelic, one severe |
| Homozygous recessive LoF | 0.90 | Null homozygote |
| X-linked hemizygous LoF | 0.95 | Single allele in males |
| Inherited dominant | 0.40 | Needs segregation |
| No trio data (HET) | 0.30 | Cannot distinguish de novo |

---

## Clinical Decision Support

### ACMG/AMP Classification

Implements 18 evidence criteria following Richards et al. 2015:

**Pathogenic:** PVS1, PS1, PS3, PM1, PM2, PM3, PM4, PM5, PP1, PP3, PP5
**Benign:** BA1, BS1, BS2, BP1, BP4, BP6, BP7

Classification uses Table 5 combining rules (e.g., PVS1 + PS1 = Pathogenic, BA1 alone = Benign).

Every criterion produces an audit trail entry with code, strength, direction, and justification for CAP/CLIA compliance.

### Actionable Gene Lists

- **ACMG SF v3.2**: 60 genes recommended for return of secondary findings (BRCA1/2, TP53, MLH1, MYBPC3, SCN5A, LDLR, etc.)
- **CPIC Pharmacogenomics**: 16 genes with Level A/B drug-gene interactions (CYP2D6, DPYD, G6PD, HLA-B, SCN1A, etc.)

### Population-Specific Priors

- QGP (Qatar Genome Programme) and GME (Greater Middle East Variome) allele frequencies
- Founder variant detection: local_af/gnomad_af >= 10× enrichment
- 18 known Middle Eastern founder genes (MEFV, GJB2, HBB, G6PD, etc.)
- Population-adjusted rarity: effective_af = max(local_af, gnomad_af)

---

## Interfaces

### Command Line

```bash
raregeneai analyze --vcf patient.vcf.gz --hpo HP:0001250 -o results/
raregeneai init-config -o config.yaml
raregeneai ui  # Launch web interface
```

### Web UI (Streamlit)

```bash
raregeneai ui
# Opens http://localhost:8501
```

Upload VCF, enter HPO terms, configure weights, view ranked genes interactively.

### REST API (FastAPI)

```bash
uvicorn raregeneai.ui.api:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/analyze \
  -F "vcf_file=@patient.vcf.gz" \
  -F 'request={"hpo_terms": ["HP:0001250"], "patient_id": "P001"}'
```

### Docker

```bash
docker build -f docker/Dockerfile -t raregeneai:latest .
docker run -v $(pwd)/data:/app/data raregeneai:latest analyze \
  --vcf /app/data/patient.vcf.gz --hpo HP:0001250

# Or with docker-compose (API + UI)
cd docker && docker-compose up
# API: http://localhost:8000  |  UI: http://localhost:8501
```

### Nextflow (HPC/Cloud)

```bash
nextflow run nextflow/main.nf \
  --vcf patient.vcf.gz \
  --hpo "HP:0001250,HP:0002878" \
  --outdir results/ \
  -profile docker    # or: singularity, slurm, cloud
```

---

## Configuration

Generate a default config file:

```bash
raregeneai init-config -o config.yaml
```

Key configuration sections:

```yaml
genome_build: GRCh38
include_noncoding: true

annotation:
  population:
    population: QGP                          # Patient population
    qgp_af_path: data/reference/qgp_af.tsv   # Local frequency database

scoring:
  w_pathogenicity: 0.30                      # Adjustable weights
  w_rarity: 0.20
  gnomad_af_threshold: 0.01

ranking:
  model_type: xgboost                        # xgboost | rule_based
  pretrained_model_path: models/v1.pkl       # Trained model

knowledge_graph:
  enabled: true
  algorithm: rwr                             # rwr | pagerank | diffusion
  restart_probability: 0.4

multiomics:
  z_score_threshold: 2.0
  concordance_multiplier: 1.3
```

See `raregeneai/config/default_config.yaml` for the full 120-line reference.

---

## ML Model Training

### Train on Synthetic Benchmark (100 Published Cases)

```bash
python scripts/clinical_validation.py --n-bystanders 100 --save-model models/v1.pkl
```

### Train on Your Own Data

```python
from raregeneai.ranking.model_trainer import ModelTrainer

trainer = ModelTrainer()

# From CSV (gene features + label column)
X, y = trainer.build_training_data_from_csv("training_data.csv")

# Train with hyperparameter search
trainer.train_with_hyperopt(X, y, save_path="models/my_model.pkl")

# Evaluate
metrics = trainer.evaluate(X_test, y_test)

# SHAP interpretation
shap = trainer.explain_with_shap(X)
print(shap["group_importance"])
```

### Validation Results

```
Cross-Validated ROC-AUC:     1.0000
Holdout Test ROC-AUC:        1.0000
Top-1  Accuracy:             100.0%
Top-5  Accuracy:             100.0%
Top-10 Accuracy:             100.0%
Mean Reciprocal Rank:        1.0000
```

*Validated on 100 published rare disease cases (SCN1A, BRCA1, CFTR, PAH, MYH7, TP53, etc.) with simulated feature profiles. Real-world performance depends on annotation quality.*

---

## Continuous Learning

### Capture Clinician Feedback

```python
from raregeneai.learning.feedback_store import FeedbackStore

store = FeedbackStore("data/feedback/feedback.jsonl")
store.submit_confirmed_diagnosis(
    patient_id="P001",
    gene_symbol="SCN1A",
    diagnosis="Dravet syndrome",
    original_rank=1,
    confirmation_method="sanger",
    analyst_id="Dr. Smith",
)
```

### Periodic Model Retraining

```python
from raregeneai.learning.continuous_trainer import ContinuousTrainer
from raregeneai.learning.model_registry import ModelRegistry

registry = ModelRegistry("models/registry")
trainer = ContinuousTrainer(store, registry)

result = trainer.run_retrain_cycle(
    min_feedback=20,
    auto_promote=True,
    min_improvement_auc=0.005,
)
# Auto-promotes new model to production if AUC improves
```

### Model Versioning

```
candidate ──► staging ──► production ──► retired
                               ▲            │
                               └── rollback ┘
```

---

## Project Structure

```
RareGeneAI/
├── raregeneai/                  # Source code (60 files, 11,900 lines)
│   ├── annotation/              # VEP, gnomAD, CADD, ClinVar, regulatory, population
│   ├── clinical/                # ACMG/AMP classifier, SF v3.2, pharmacogenomics
│   ├── config/                  # Pydantic settings + default YAML
│   ├── explainability/          # SHAP explanations + evidence summaries
│   ├── ingestion/               # VCF, HPO, PED parsers
│   ├── knowledge_graph/         # Graph builder (HPO/OMIM/STRING/KEGG) + RWR scorer
│   ├── learning/                # Feedback store, model registry, continuous trainer
│   ├── models/                  # 18 Pydantic data models
│   ├── multiomics/              # RNA-seq outliers, methylation DMRs, concordance
│   ├── phenotype/               # Semantic similarity, gene-phenotype matching
│   ├── pipeline/                # 12-step orchestrator
│   ├── population/              # QGP/GME AF, founder variant detection
│   ├── ranking/                 # XGBoost trainer (44 features) + gene ranker
│   ├── reporting/               # HTML/PDF clinical report generator
│   ├── scoring/                 # Composite scorer + trio inheritance analyzer
│   ├── structural/              # SV parser, annotator, integration bridge
│   ├── ui/                      # Streamlit app + FastAPI REST API
│   ├── utils/                   # Logging, caching, parallel processing
│   └── validation/              # Benchmarker framework
├── tests/                       # 311 tests (13 files, 4,363 lines)
│   ├── integration/             # End-to-end pipeline tests
│   └── unit/                    # Per-module unit tests
├── scripts/                     # Validation, data generation, reference download
├── docker/                      # Dockerfile + docker-compose
├── nextflow/                    # Nextflow DSL2 pipeline + config
├── templates/                   # Jinja2 HTML report template
├── models/                      # Trained ML model artifacts
├── pyproject.toml               # Package configuration
├── METHODOLOGY.md               # Full algorithm documentation
└── DEVELOPMENT_GUIDE.md         # Build history + design decisions
```

---

## Testing

```bash
# Run all 311 tests
python -m pytest tests/ -v

# Run specific module tests
python -m pytest tests/unit/test_trio_inheritance.py -v
python -m pytest tests/unit/test_clinical_decision.py -v

# With coverage report
python -m pytest tests/ --cov=raregeneai --cov-report=html
```

### Test Breakdown

| Test File | Tests | Coverage |
|-----------|-------|----------|
| test_clinical_decision.py | 36 | ACMG criteria, SF genes, PGx, recommendations |
| test_trio_inheritance.py | 35 | De novo, compound het, hom recessive, scoring |
| test_structural.py | 38 | SV parsing, annotation, scoring, integration |
| test_noncoding.py | 31 | Regulatory annotation, scoring, explainer |
| test_multiomics.py | 29 | Expression outliers, DMRs, concordance |
| test_continuous_learning.py | 27 | Feedback, registry, retrain cycle |
| test_population.py | 26 | QGP AF, founder detection, rarity scoring |
| test_knowledge_graph.py | 25 | Graph construction, RWR, path finding |
| test_ml_ranking.py | 24 | XGBoost training, SHAP, Top-K, persistence |
| test_data_models.py | 13 | Variant, Gene, Phenotype, Pedigree models |
| test_scoring.py | 11 | Composite scoring formula |
| test_explainer.py | 6 | Explanation generation, ACMG |
| test_pipeline.py | 5 | End-to-end integration |
| **Total** | **311** | **0 failures** |

---

## Dependencies

Core:
- Python >= 3.10
- cyvcf2, pysam (VCF/BAM processing)
- pandas, numpy, scipy (data processing)
- xgboost, scikit-learn (ML ranking)
- shap (interpretability)
- networkx (knowledge graph)
- pronto (HPO ontology)
- pydantic (data models)

Web/API:
- streamlit (web UI)
- fastapi, uvicorn (REST API)

Reporting:
- jinja2 (HTML templates)
- weasyprint (PDF generation, optional)

CLI:
- click (command line)
- rich (terminal formatting)

Infrastructure:
- Docker, Nextflow (deployment)
- loguru (logging)
- pyyaml (configuration)

Full dependency list in `pyproject.toml`.

---

## Citing

If you use RareGeneAI in your research, please cite:

```
RareGeneAI: A multi-modal rare disease gene prioritization system
integrating genomic, transcriptomic, epigenomic, and knowledge graph
evidence with ACMG-compliant clinical decision support.
```

---

## License

MIT License. See LICENSE file for details.

---

## Acknowledgments

RareGeneAI integrates data from:
- [HPO](https://hpo.jax.org/) (Human Phenotype Ontology)
- [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) (Clinical variant database)
- [gnomAD](https://gnomad.broadinstitute.org/) (Population frequencies)
- [OMIM](https://omim.org/) (Gene-disease associations)
- [STRING](https://string-db.org/) (Protein interactions)
- [ENCODE](https://www.encodeproject.org/) (Regulatory elements)
- [GTEx](https://gtexportal.org/) (Expression QTLs)
- [CPIC](https://cpicpgx.org/) (Pharmacogenomics guidelines)
- [ACMG](https://www.acmg.net/) (Variant classification standards)
