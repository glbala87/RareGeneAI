"""FastAPI REST API for RareGeneAI.

Production-ready API with:
  - JWT/OAuth2 authentication with DB-backed user verification
  - CORS lockdown (configurable origins)
  - Input validation (VCF, HPO, patient IDs)
  - Secure temp file handling
  - PostgreSQL-backed job persistence with ownership
  - HIPAA audit logging
  - Prometheus metrics (admin-only)
  - Comprehensive health checks

Endpoints:
  POST /auth/token         - Get JWT access token
  POST /auth/register      - Register new user (admin only)
  POST /analyze            - Run full pipeline (analyst+)
  GET  /jobs/{job_id}      - Get job status/results (owner, analyst, or admin)
  GET  /health             - Basic health check
  GET  /health/detailed    - Detailed component health (admin)
  GET  /metrics            - Prometheus metrics (admin)
  GET  /compliance/check   - HIPAA compliance check (admin)
  GET  /retention/report   - Data retention report (admin)
  POST /retention/purge    - Run retention purge (admin)
"""

from __future__ import annotations

import json
import os
import time
import uuid

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from raregeneai.database.audit import AuditService
from raregeneai.database.connection import get_db, init_db
from raregeneai.database.encryption import FieldEncryptor
from raregeneai.database.models import AnalysisJob, UserAccount
from raregeneai.observability.health import HealthChecker
from raregeneai.observability.metrics import (
    PrometheusMiddleware,
    metrics,
    metrics_endpoint,
)
from raregeneai.security.auth import (
    User,
    UserRole,
    create_access_token,
    get_current_user,
    hash_password,
    require_admin,
    require_analyst,
    verify_password,
)
from raregeneai.security.rate_limit import (
    ANALYZE_MAX,
    ANALYZE_WINDOW,
    LOGIN_MAX,
    LOGIN_WINDOW,
    REGISTER_MAX,
    REGISTER_WINDOW,
    rate_limiter,
)
from raregeneai.security.secure_temp import SecureTempFile
from raregeneai.security.validation import (
    validate_hpo_terms,
    validate_patient_id,
    validate_sample_id,
    validate_vcf_upload,
)

# ── App setup ────────────────────────────────────────────────────────────────

_start_time = time.time()

app = FastAPI(
    title="RareGeneAI API",
    version="1.1.0",
    description="Rare Disease Gene Prioritization REST API",
    docs_url="/docs" if os.environ.get("RAREGENEAI_ENABLE_DOCS", "false") == "true" else None,
    redoc_url=None,
)

# CORS - locked down to configured origins
_allowed_origins = os.environ.get(
    "RAREGENEAI_ALLOWED_ORIGINS", "http://localhost:8501"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _allowed_origins],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
    allow_credentials=True,
)

# Prometheus metrics middleware
app.add_middleware(PrometheusMiddleware)

# Initialize database and encryption on startup
_encryptor: FieldEncryptor | None = None


@app.on_event("startup")
async def startup():
    global _encryptor
    init_db()
    _encryptor = FieldEncryptor()  # Will fail-fast in production if key missing
    metrics.set_app_info(version="1.1.0")
    logger.info("RareGeneAI API started")


def _get_encryptor() -> FieldEncryptor:
    if _encryptor is None:
        raise RuntimeError("Application not initialized")
    return _encryptor


# ── Request/Response models ──────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    hpo_terms: list[str] = Field(..., description="HPO term IDs (HP:NNNNNNN)")
    patient_id: str = Field(default="PATIENT_001", min_length=1, max_length=128)
    sample_id: str | None = None
    genome_build: str = Field(default="GRCh38", pattern="^GRCh3[78]$")
    top_n: int = Field(default=20, ge=1, le=200)


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


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=128)
    password: str = Field(..., min_length=12, max_length=256)
    role: str = Field(default="viewer", pattern="^(admin|analyst|viewer)$")
    institution: str = ""


# ── Auth endpoints ───────────────────────────────────────────────────────────

@app.post("/auth/token", response_model=TokenResponse)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """Authenticate and receive a JWT token."""
    rate_limiter.check("login", request, max_requests=LOGIN_MAX, window_seconds=LOGIN_WINDOW)

    user = db.query(UserAccount).filter(
        UserAccount.username == form_data.username,
        UserAccount.is_active == True,  # noqa: E712
    ).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        AuditService.log_auth_event(
            db,
            username=form_data.username,
            action="login",
            success=False,
            request=request,
            detail="Invalid credentials",
        )
        metrics.auth_attempts_total.labels(result="failure").inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(data={"sub": user.username, "role": user.role})

    AuditService.log_auth_event(
        db, username=user.username, action="login", success=True, request=request,
    )
    metrics.auth_attempts_total.labels(result="success").inc()

    return TokenResponse(access_token=token)


@app.post("/auth/register", status_code=status.HTTP_201_CREATED)
async def register_user(
    req: RegisterRequest,
    request: Request,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Register a new user (admin only)."""
    rate_limiter.check("register", request, max_requests=REGISTER_MAX, window_seconds=REGISTER_WINDOW)

    existing = db.query(UserAccount).filter(
        UserAccount.username == req.username
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists",
        )

    new_user = UserAccount(
        username=req.username,
        hashed_password=hash_password(req.password),
        role=req.role,
        institution=req.institution,
    )
    db.add(new_user)

    AuditService.log(
        db,
        username=current_user.username,
        action="user_register",
        resource_type="user_account",
        resource_id=req.username,
        request=request,
    )

    return {"message": f"User '{req.username}' created with role '{req.role}'"}


# ── Analysis endpoints ───────────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    request_obj: Request,
    analysis_request: AnalysisRequest,
    vcf_file: UploadFile = File(...),
    current_user: User = Depends(require_analyst),
    db: Session = Depends(get_db),
):
    """Run the full gene prioritization pipeline."""
    rate_limiter.check(
        "analyze", request_obj, max_requests=ANALYZE_MAX,
        window_seconds=ANALYZE_WINDOW, identifier=current_user.username,
    )

    from raregeneai.compliance.retention import RetentionService
    from raregeneai.config.settings import PipelineConfig
    from raregeneai.pipeline.orchestrator import RareGeneAIPipeline

    encryptor = _get_encryptor()

    # Validate inputs
    validated_patient_id = validate_patient_id(analysis_request.patient_id)
    validated_sample_id = validate_sample_id(analysis_request.sample_id)
    validated_hpo = validate_hpo_terms(analysis_request.hpo_terms)
    vcf_content = await validate_vcf_upload(vcf_file)

    # Resolve user_id from database
    db_user = db.query(UserAccount).filter(
        UserAccount.username == current_user.username
    ).first()
    user_id = db_user.id if db_user else None

    # Create job record
    job_id = str(uuid.uuid4())[:8]
    job = AnalysisJob(
        job_id=job_id,
        patient_id_encrypted=encryptor.encrypt(validated_patient_id),
        sample_id_encrypted=encryptor.encrypt(validated_sample_id or ""),
        genome_build=analysis_request.genome_build,
        hpo_terms=json.dumps(validated_hpo),
        top_n=analysis_request.top_n,
        status="running",
        user_id=user_id,
    )

    # Set retention deadline
    RetentionService().set_retention_deadline(db, job)
    db.add(job)
    db.flush()

    # Audit: PHI access
    AuditService.log_phi_access(
        db,
        username=current_user.username,
        patient_id=validated_patient_id,
        job_id=job_id,
        access_type="create",
        request=request_obj,
    )

    metrics.active_jobs.inc()
    metrics.jobs_total.labels(status="created").inc()

    # Run pipeline with secure temp file
    suffix = ".vcf.gz" if vcf_file.filename.endswith(".gz") else ".vcf"

    try:
        async with SecureTempFile(suffix=suffix, content=vcf_content) as vcf_path:
            config = PipelineConfig()
            config.genome_build = analysis_request.genome_build
            config.ranking.top_n_genes = analysis_request.top_n

            pipeline = RareGeneAIPipeline(config)
            report = pipeline.run(
                vcf_path=str(vcf_path),
                hpo_terms=validated_hpo,
                patient_id=validated_patient_id,
                sample_id=validated_sample_id,
            )
    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)[:1000]
        metrics.active_jobs.dec()
        metrics.errors_total.labels(type="pipeline_error", component="api").inc()
        logger.error(f"Pipeline failed for job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis pipeline failed. Check server logs for details.",
        )

    metrics.active_jobs.dec()

    # Build response
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

    # Persist results
    job.status = "completed"
    job.total_variants = report.total_variants_analyzed
    job.total_genes = report.total_genes_analyzed
    job.results_json = json.dumps([r.model_dump() for r in results])

    metrics.variants_processed_total.inc(report.total_variants_analyzed)
    metrics.genes_analyzed_total.inc(report.total_genes_analyzed)
    metrics.pipeline_runs_total.labels(status="success").inc()

    return response


@app.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get job status and results.

    Ownership rules:
      - Admin: can view any job
      - Analyst: can view own jobs
      - Viewer: can view own jobs (without PHI)
    """
    job = db.query(AnalysisJob).filter(
        AnalysisJob.job_id == job_id,
        AnalysisJob.is_purged == False,  # noqa: E712
    ).first()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Ownership check: non-admins can only access their own jobs
    if current_user.role != UserRole.ADMIN:
        db_user = db.query(UserAccount).filter(
            UserAccount.username == current_user.username
        ).first()
        if not db_user or job.user_id != db_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this job",
            )

    encryptor = _get_encryptor()

    # Audit PHI access
    AuditService.log_phi_access(
        db,
        username=current_user.username,
        patient_id=encryptor.decrypt(job.patient_id_encrypted),
        job_id=job_id,
        access_type="view",
        request=request,
    )

    result = {
        "job_id": job.job_id,
        "status": job.status,
        "genome_build": job.genome_build,
        "total_variants": job.total_variants,
        "total_genes": job.total_genes,
        "created_at": job.created_at.isoformat() if job.created_at else None,
    }

    if job.status == "completed" and job.results_json:
        result["ranked_genes"] = json.loads(job.results_json)

    if job.status == "failed":
        result["error"] = "Analysis failed. Contact administrator."

    # Decrypt patient ID for analyst+ roles
    if current_user.role in (UserRole.ADMIN, UserRole.ANALYST):
        result["patient_id"] = encryptor.decrypt(job.patient_id_encrypted)

    return result


# ── Health & monitoring endpoints ────────────────────────────────────────────

@app.get("/health")
async def health():
    """Basic liveness check (unauthenticated)."""
    return {"status": "ok", "version": "1.1.0"}


@app.get("/health/detailed")
async def health_detailed(current_user: User = Depends(require_admin)):
    """Detailed health check with component status (admin only)."""
    checker = HealthChecker(version="1.1.0", start_time=_start_time)
    health_report = checker.check_all()

    status_code = 200 if health_report.status.value == "healthy" else 503
    from starlette.responses import JSONResponse
    return JSONResponse(content=health_report.to_dict(), status_code=status_code)


@app.get("/metrics")
async def get_metrics(request: Request, current_user: User = Depends(require_admin)):
    """Prometheus metrics endpoint (admin only)."""
    return metrics_endpoint(request)


# ── Compliance endpoints ─────────────────────────────────────────────────────

@app.get("/compliance/check")
async def compliance_check(current_user: User = Depends(require_admin)):
    """Run HIPAA compliance checks (admin only)."""
    from raregeneai.compliance.hipaa import HIPAAComplianceChecker

    checker = HIPAAComplianceChecker()
    report = checker.run_all_checks()
    return {
        "passed": report.passed,
        "summary": report.summary(),
        "critical_failures": report.critical_failures,
        "warnings": report.warnings,
        "checks": [
            {
                "name": c.name,
                "category": c.category,
                "passed": c.passed,
                "detail": c.detail,
                "severity": c.severity,
            }
            for c in report.checks
        ],
    }


@app.get("/retention/report")
async def retention_report(
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Get data retention status report (admin only)."""
    from raregeneai.compliance.retention import RetentionService
    service = RetentionService()
    return service.get_retention_report(db)


@app.post("/retention/purge")
async def retention_purge(
    request: Request,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Run data retention purge (admin only)."""
    from raregeneai.compliance.retention import RetentionService

    service = RetentionService()
    jobs_purged = service.purge_expired_jobs(db)
    audit_purged = service.purge_expired_audit_logs(db)

    AuditService.log(
        db,
        username=current_user.username,
        action="retention_purge",
        detail=f"Purged {jobs_purged} jobs, {audit_purged} audit entries",
        request=request,
    )

    return {
        "jobs_purged": jobs_purged,
        "audit_entries_purged": audit_purged,
    }
