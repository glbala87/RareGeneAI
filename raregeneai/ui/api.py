"""FastAPI REST API for RareGeneAI.

Endpoints:
  POST /analyze - Run full pipeline
  GET  /status/{job_id} - Check job status
  GET  /results/{job_id} - Get results
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="RareGeneAI API",
    version="1.0.0",
    description="Rare Disease Gene Prioritization REST API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store (use Redis/DB in production)
_jobs: dict[str, dict] = {}


class AnalysisRequest(BaseModel):
    hpo_terms: list[str] = Field(..., description="HPO term IDs")
    patient_id: str = "PATIENT_001"
    sample_id: str | None = None
    genome_build: str = "GRCh38"
    top_n: int = 20


class GeneResult(BaseModel):
    rank: int
    gene_symbol: str
    score: float
    confidence: float
    phenotype_score: float
    n_variants: int
    explanation: str


class AnalysisResponse(BaseModel):
    job_id: str
    patient_id: str
    total_variants: int
    total_genes: int
    ranked_genes: list[GeneResult]


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    request: AnalysisRequest,
    vcf_file: UploadFile = File(...),
):
    """Run the full gene prioritization pipeline."""
    from raregeneai.config.settings import PipelineConfig
    from raregeneai.pipeline.orchestrator import RareGeneAIPipeline

    # Save uploaded VCF
    suffix = ".vcf.gz" if vcf_file.filename.endswith(".gz") else ".vcf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await vcf_file.read()
        tmp.write(content)
        vcf_path = tmp.name

    # Configure and run
    config = PipelineConfig()
    config.genome_build = request.genome_build
    config.ranking.top_n_genes = request.top_n

    pipeline = RareGeneAIPipeline(config)

    try:
        report = pipeline.run(
            vcf_path=vcf_path,
            hpo_terms=request.hpo_terms,
            patient_id=request.patient_id,
            sample_id=request.sample_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(vcf_path).unlink(missing_ok=True)

    job_id = str(uuid.uuid4())[:8]

    results = []
    for i, gene in enumerate(report.ranked_genes, 1):
        results.append(GeneResult(
            rank=i,
            gene_symbol=gene.gene_symbol,
            score=gene.gene_rank_score,
            confidence=gene.confidence,
            phenotype_score=gene.phenotype_score,
            n_variants=gene.n_variants,
            explanation=gene.explanation,
        ))

    response = AnalysisResponse(
        job_id=job_id,
        patient_id=report.patient_id,
        total_variants=report.total_variants_analyzed,
        total_genes=report.total_genes_analyzed,
        ranked_genes=results,
    )

    _jobs[job_id] = response.model_dump()
    return response


@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Get results for a completed analysis."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
