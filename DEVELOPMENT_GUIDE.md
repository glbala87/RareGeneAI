# RareGeneAI — Development Guide

## How This Tool Was Built (Step-by-Step)

RareGeneAI was built in 12 iterative phases, each adding a major
capability. Every phase followed the same pattern:
  1. Add data models (Pydantic classes in `models/data_models.py`)
  2. Add configuration (settings in `config/settings.py`)
  3. Build the core module (new package directory)
  4. Integrate into the pipeline orchestrator
  5. Update the gene ranker with new features
  6. Update the explainer with new evidence sections
  7. Write comprehensive tests
  8. Run full test suite to verify no regressions

---

## Build Order (Chronological)

### Phase 1: Core Architecture
**Files created:**
```
raregeneai/__init__.py
raregeneai/models/data_models.py          ← Variant, AnnotatedVariant, GeneCandidate, HPOTerm, Pedigree, ClinicalReport
raregeneai/config/settings.py             ← IngestionConfig, AnnotationConfig, ScoringConfig, RankingConfig, PipelineConfig
raregeneai/config/default_config.yaml     ← Full YAML reference config
raregeneai/utils/logging.py               ← Structured logging with loguru
pyproject.toml                            ← Package definition + dependencies
```
**Decision:** Pydantic for all data models — gives validation, serialization, and type safety for free.

---

### Phase 2: Data Ingestion
**Files created:**
```
raregeneai/ingestion/vcf_parser.py        ← cyvcf2-based VCF parsing + pure-Python fallback
raregeneai/ingestion/hpo_parser.py        ← HPO validation via pronto + REST API fallback
raregeneai/ingestion/pedigree_parser.py   ← Standard PED format parser
```
**Key design:** cyvcf2 (C-backed) for speed, with pure-Python fallback so the tool works without compiled dependencies. VCF normalization (split multi-allelic, left-align) delegated to bcftools subprocess.

---

### Phase 3: Variant Annotation
**Files created:**
```
raregeneai/annotation/vep_annotator.py         ← VEP local + REST API
raregeneai/annotation/frequency_annotator.py   ← gnomAD local tabix + GraphQL API
raregeneai/annotation/pathogenicity_annotator.py ← CADD, REVEL, SpliceAI, ClinVar
raregeneai/annotation/annotation_engine.py     ← Orchestrator: VEP → frequency → pathogenicity → regulatory
```
**Key design:** Every annotator has local (tabix/pysam) + remote (API) paths. The annotation engine runs gnomAD + population frequency in parallel via ThreadPoolExecutor.

---

### Phase 4: Phenotype Matching
**Files created:**
```
raregeneai/phenotype/semantic_similarity.py    ← Resnik/Lin/JC information content
raregeneai/phenotype/gene_phenotype_matcher.py ← Best-Match-Average HPO→gene scoring
```
**Algorithm:** IC(t) = -log2(|descendants(t)| / |total|). BMA similarity computes how well a gene's known HPO profile explains the patient's symptoms.

---

### Phase 5: Variant Scoring + Inheritance
**Files created:**
```
raregeneai/scoring/variant_scorer.py           ← Composite: pathogenicity + rarity + impact + inheritance + regulatory
raregeneai/scoring/inheritance_analyzer.py     ← Trio: de novo, compound het, hom recessive detection + scoring
```
**Formula:** `composite = 0.30×pathogenicity + 0.20×rarity + 0.15×impact + 0.15×regulatory + 0.15×phenotype + 0.05×inheritance`

---

### Phase 6: Gene Ranking + ML
**Files created:**
```
raregeneai/ranking/gene_ranker.py              ← Groups variants by gene, builds 44-feature vector, XGBoost or rule-based
raregeneai/ranking/model_trainer.py            ← XGBoost training pipeline, 5-fold CV, SHAP, Top-K, model persistence
```
**44 features** from 6 evidence groups: variant pathogenicity (10), non-coding regulatory (8), structural variants (7), multi-omics (8), trio inheritance (6), knowledge graph (5).

---

### Phase 7: Explainability + Reporting
**Files created:**
```
raregeneai/explainability/explainer.py         ← SHAP + rule-based explanations + ACMG classification
raregeneai/reporting/report_generator.py       ← HTML/PDF clinical report with evidence badges
templates/report.html                          ← Jinja2 report template
```

---

### Phase 8: Pipeline Orchestrator
**Files created:**
```
raregeneai/pipeline/orchestrator.py            ← 12-step pipeline: ingest → annotate → trio → score → filter → SV → phenotype → rank → KG → multi-omics → clinical → report
raregeneai/cli.py                              ← Click-based CLI with all options
```

---

### Phase 9: Non-coding Variant Prioritization
**Files created:**
```
raregeneai/annotation/regulatory_annotator.py  ← ENCODE cCREs, ChromHMM, conservation, eQTL, DL scores
```
**Added to AnnotatedVariant:** 25 regulatory fields (regulatory_class, chromhmm_state, phastcons, enformer, target_gene, gene_mapping_method, etc.)

---

### Phase 10: Structural Variant Prioritization
**Files created:**
```
raregeneai/structural/sv_parser.py             ← Sniffles/Jasmine/Manta/DELLY VCF parsing
raregeneai/structural/sv_annotator.py          ← Gene overlap, dosage (pLI/LOEUF), regulatory, AnnotSV, clinical DBs
raregeneai/structural/sv_integration.py        ← Converts SVs to AnnotatedVariant for unified ranking
```
**Added to data_models.py:** SVType enum, StructuralVariant, AnnotatedSV models.

---

### Phase 11: Multi-omics Integration
**Files created:**
```
raregeneai/multiomics/expression_outlier.py    ← RNA-seq Z-score outlier detection (median + MAD)
raregeneai/multiomics/methylation_analyzer.py  ← DMR detection + gene promoter overlap
raregeneai/multiomics/integrator.py            ← Evidence layer counting + concordance scoring
```
**Concordance:** Underexpression + promoter hypermethylation = silencing (1.3× multiplier).

---

### Phase 12: Trio-Based Inheritance Modeling
**Enhanced:**
```
raregeneai/scoring/inheritance_analyzer.py     ← Rewritten: tags each variant in-place with inheritance class + score
```
**15 inheritance weight classes:** De novo LoF (1.0) → De novo missense (0.9) → Compound het LoF+LoF (0.95) → ... → Unknown HET (0.3).

---

### Phase 13: Knowledge Graph Prioritization
**Files created:**
```
raregeneai/knowledge_graph/graph_builder.py    ← Builds heterogeneous graph: Gene, HPO, OMIM, KEGG/Reactome, STRING PPI
raregeneai/knowledge_graph/graph_scorer.py     ← Random Walk with Restart / PageRank / Network Diffusion
```
**Algorithm:** RWR from patient HPO seed nodes, restart_prob=0.4, converge to stationary distribution.

---

### Phase 14: Population-Specific Priors
**Files created:**
```
raregeneai/population/population_annotator.py  ← QGP/GME/local AF, founder detection, population-adjusted rarity
```
**Key insight:** effective_af = max(local_af, gnomad_af). A variant common in QGP but rare in gnomAD is NOT rare for a Qatari patient.

---

### Phase 15: Clinical Decision Support
**Files created:**
```
raregeneai/clinical/acmg_classifier.py         ← 18 ACMG/AMP criteria + Table 5 combining rules
raregeneai/clinical/clinical_decision.py       ← ACMG SF v3.2 (60 genes) + CPIC PGx (16 genes) + audit trail
```

---

### Phase 16: Continuous Learning
**Files created:**
```
raregeneai/learning/feedback_store.py          ← Append-only JSON-lines feedback
raregeneai/learning/model_registry.py          ← Versioned model lifecycle (candidate → production → retired)
raregeneai/learning/continuous_trainer.py       ← Periodic retrain from feedback + baseline
```

---

### Phase 17: Performance Enhancements
**Files created:**
```
raregeneai/utils/cache.py                      ← Disk-backed annotation cache
raregeneai/utils/parallel.py                   ← ThreadPoolExecutor utilities
```
**Updated:** Annotation engine runs gnomAD + population in parallel. Timing instrumentation on every step.

---

### Phase 18: Deployment Infrastructure
**Files created:**
```
docker/Dockerfile                              ← Multi-stage production image
docker/docker-compose.yaml                     ← API + UI + pipeline services
nextflow/main.nf                               ← Nextflow DSL2 pipeline
nextflow/nextflow.config                       ← Docker/Singularity/Slurm/Cloud profiles
raregeneai/ui/app.py                           ← Streamlit web UI (all inputs)
raregeneai/ui/api.py                           ← FastAPI REST API
```

---

### Phase 19: Validation + Benchmarking
**Files created:**
```
raregeneai/validation/benchmarker.py           ← Run pipeline on test cases, compute Top-K/MRR
scripts/clinical_validation.py                 ← 100 published rare disease cases benchmark
scripts/run_validation.py                      ← Synthetic cohort validation
scripts/generate_test_vcf.py                   ← Test VCF generator
scripts/download_references.sh                 ← Download HPO, ClinVar, gnomAD
```

---

## Development Principles

1. **Every module is self-contained** — can be imported and used independently
2. **Every module has tests** — 311 tests total, 0 failures
3. **Every annotation has local + remote paths** — works without any installed tools
4. **Every clinical assertion has an audit trail** — CAP/CLIA compliance
5. **Every model prediction is explainable** — SHAP values + rule-based text
6. **Configuration over code** — full YAML config controls all parameters
7. **Graceful degradation** — missing data (no trio, no RNA-seq, no KG) reduces accuracy but doesn't break the pipeline
