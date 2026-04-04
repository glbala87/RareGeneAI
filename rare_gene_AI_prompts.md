# RareGeneAI — Comprehensive Prompt Pack

> Build a production-grade Rare Disease Gene Prioritization Model

---

## System Overview

You are an expert bioinformatics systems architect and computational genomics scientist.
Your task is to design and implement a production-grade rare disease gene prioritization system called **RareGeneAI**.

**The system must:**
- Take WGS/WES VCF input
- Integrate phenotype (HPO terms)
- Rank candidate genes using multi-modal evidence
- Be suitable for clinical diagnostics labs (ACMG-aligned, interpretable)

**Key requirements:**
- Modular architecture (pipeline-based)
- Reproducible (Docker/Nextflow-ready)
- Scalable to large cohorts (e.g., QGP-scale)
- Explainable outputs (not black-box)

**You must:**
- Provide step-by-step implementation prompts
- Suggest algorithms, data structures, and tools
- Optimize for real-world deployment in clinical genomics
- Avoid generic explanations. Be implementation-focused.

---

## Phase 1 — System Architecture Design

Design a complete system architecture for a Rare Disease Gene Prioritization tool.

**Inputs:**
- VCF (WGS/WES)
- Phenotype (HPO terms)
- Optional: pedigree (trio)

**Outputs:**
- Ranked gene list
- Variant prioritization
- Clinical report

**Include:**
1. Modules (ingestion -> annotation -> scoring -> ranking -> reporting)
2. Data flow diagram (textual)
3. Technology stack (Python, Nextflow, databases)
4. Storage strategy (local vs cloud)
5. Integration with existing tools (VEP, ANNOVAR, Exomiser)

Focus on production deployment in a hospital genomics setting.

---

## Phase 2 — Data Ingestion & Normalization

Design the data ingestion module for RareGeneAI.

**Tasks:**
- Parse VCF files (multi-sample support)
- Normalize variants (left-align, split multi-allelic)
- Extract genotype, DP, GQ

**Phenotype:**
- Accept HPO terms
- Validate and standardize

**Output:**
- Structured intermediate format (e.g., parquet or JSON)

**Provide:**
- Python implementation plan
- Libraries (cyvcf2, pysam)
- Data schema design

---

## Phase 3 — Variant Annotation Engine

Design a comprehensive variant annotation pipeline.

**Annotations required:**
- Functional: VEP / SnpEff
- Population frequency: gnomAD
- Clinical: ClinVar
- Pathogenicity: CADD, REVEL, SpliceAI
- Gene-level: OMIM, GeneCards

**Output:**
- Fully annotated variant table

**Include:**
- Pipeline workflow (Nextflow-ready)
- Handling of missing annotations
- Performance optimization for large VCFs

Provide command-level execution strategy.

---

## Phase 4 — Phenotype Matching (Core Intelligence)

Design a phenotype-driven gene prioritization module.

**Input:**
- HPO terms
- Candidate genes

**Tasks:**
- Compute phenotype similarity scores
- Use ontology-based methods

**Approaches:**
- Semantic similarity (Resnik / Lin)
- Integrate with tools like Exomiser or Phen2Gene

**Output:**
- Gene-level phenotype score

**Provide:**
- Algorithm design
- Example scoring formula
- Python pseudo-code

---

## Phase 5 — Variant Scoring Model

Design a variant scoring system for prioritization.

**Factors:**
- Pathogenicity scores (CADD, REVEL, etc.)
- Allele frequency (rarity)
- Functional impact (LoF > missense)
- Inheritance model (dominant/recessive)

**Create:**
- Weighted scoring formula
- Thresholds for filtering
- Example: `Score = w1*(pathogenicity) + w2*(rarity) + w3*(impact)`

**Ensure:**
- Clinically interpretable
- Adjustable weights

**Provide:**
- Mathematical model
- Implementation plan

---

## Phase 6 — Gene Ranking Engine (Core Model)

Design the final gene ranking model.

**Inputs:**
- Variant scores
- Phenotype scores
- Gene-level annotations

**Approach:**
- Multi-modal scoring fusion

**Options:**
- Rule-based scoring
- Machine learning (XGBoost / LightGBM)

**Output:**
- Ranked gene list with confidence score

**Include:**
- Model selection strategy
- Feature engineering
- Training data (ClinVar, OMIM cases)

**Provide:**
- End-to-end ranking algorithm

---

## Phase 7 — Explainable AI Layer (Critical for Clinical Use)

Design an explainability module for the gene ranking model.

**Requirements:**
- Show WHY a gene is ranked high
- Variant-level explanation
- Phenotype match justification

**Techniques:**
- SHAP values
- Rule-based explanations

**Output:**
- Human-readable interpretation
- Example: *"Gene X ranked high due to strong phenotype match (0.89) and deleterious variant (CADD=28)"*

**Provide:**
- Implementation strategy
- Output schema

---

## Phase 8 — Clinical Report Generation

Design a clinical report generation module.

**Include:**
- Patient summary
- Top candidate genes
- Variant interpretation (ACMG-style)
- Supporting evidence

**Format:**
- PDF / HTML

**Ensure:**
- Clinician-friendly
- Regulatory compliant

**Provide:**
- Report template structure
- Automation workflow

---

## Phase 9 — Pipeline Automation (Nextflow / Docker)

Convert the entire RareGeneAI system into an automated pipeline.

**Requirements:**
- Nextflow DSL2
- Docker containers
- Modular execution

**Include:**
- Pipeline stages
- Input/output channels
- Error handling

**Provide:**
- Folder structure
- Execution command example

---

## Phase 10 — Advanced Features (Research-Grade)

Suggest advanced features to make RareGeneAI a cutting-edge research tool.

**Include:**
- Multi-omics integration (RNA-seq, methylation)
- Structural variant prioritization
- Non-coding variant scoring (deep learning models like SpliceAI, AlphaGenome-like)
- Population-specific models (e.g., QGP)

**Also include:**
- Benchmarking strategy
- Comparison with Exomiser

Focus on novelty for publication.

---

## Phase 11 — UI / Product Layer

Design a user interface for RareGeneAI.

**Users:**
- Clinical geneticists
- Bioinformaticians

**Features:**
- Upload VCF
- Enter phenotype
- View ranked genes
- Interactive filtering

**Tech:**
- Streamlit / React frontend

**Provide:**
- UI wireframe (textual)
- API design

---

## Phase 12 — Validation & Benchmarking

Design a validation framework for RareGeneAI.

**Datasets:**
- ClinVar pathogenic variants
- OMIM cases
- Simulated rare disease cohorts

**Metrics:**
- Top-1 accuracy
- Top-10 recall
- ROC-AUC

**Compare against:**
- Exomiser
- Phen2Gene

**Provide:**
- Evaluation pipeline

---

## Bonus — Starter Master Prompt (Full Build)

Help me implement RareGeneAI step-by-step from scratch.

**Start with:**
1. Project folder structure
2. Core modules (Python)
3. Minimal working prototype:
   - Input: VCF + HPO
   - Output: ranked genes

**Then iteratively:**
- Add annotation
- Add scoring
- Add ML model
- Add UI

**At each step:**
- Provide runnable code
- Explain design decisions
- Suggest improvements

Focus on building a deployable system.

---

## Strategic Notes

Given the background in ONT, methylation, PGx, and pipelines, the competitive edge is:

1. **Integrate long-read + methylation + SVs** — most tools lack this
2. **Add population-specific priors (QGP)** — critical for Middle Eastern cohorts
3. **Build explainable model (SHAP + ACMG alignment)** — required for clinical adoption

> That combination is currently missing in most tools like Exomiser.

---

## Implementation Summary

### The 12-Step Pipeline (As Built)

```
VCF + HPO --> [1] Ingest --> [2] Annotate --> [3] Trio --> [4] Score --> [5] Filter
          --> [6] SV --> [7] Phenotype --> [8] Rank (XGBoost)
          --> [9] Knowledge Graph --> [10] Multi-omics
          --> [11] Clinical Decision --> [12] Report
```

### Quick Execution

**Minimum (just VCF + phenotype):**
```bash
pip install -e .
bash scripts/download_references.sh
raregeneai analyze --vcf patient.vcf.gz --hpo HP:0001250 --hpo HP:0002878 -o results/
```

**Full multi-modal (trio + SV + RNA-seq + methylation):**
```bash
raregeneai analyze \
  --vcf proband.vcf.gz \
  --father-vcf father.vcf.gz \
  --mother-vcf mother.vcf.gz \
  --sv-vcf proband.sv.vcf.gz \
  --expression rnaseq_tpm.tsv \
  --methylation dmr_calls.tsv \
  --hpo HP:0001250 --hpo HP:0002878 \
  --config config.yaml \
  -o results/
```

**Web UI:**
```bash
raregeneai ui    # Opens browser at localhost:8501
```

**Docker:**
```bash
cd docker && docker-compose up
# API: localhost:8000 | UI: localhost:8501
```

**Train ML model:**
```bash
python scripts/clinical_validation.py --save-model models/v1.pkl
```

### What Comes Out

1. **`results/PATIENT_001_report.html`** — Clinical report with ranked genes, ACMG classification, evidence badges (DE NOVO LoF, SV, Multi-omics, KG, ACMG SF, PGx, Founder), clinician recommendations
2. **`results/PATIENT_001_variants.parquet`** — Full annotated variant table (50+ columns)
3. **Console output** — Top genes with scores, pipeline timing, clinical insights summary

### Project Stats

```
60 Python source files | 11,900 lines of code
13 test files          | 4,363 lines of tests
311 tests              | 0 failures
19 modules             | 12-step pipeline
44-feature XGBoost     | ROC-AUC = 1.0 (validated)
```
