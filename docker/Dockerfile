# RareGeneAI Docker Image
# =======================
# Multi-stage build for production deployment
# Security-hardened with non-root user and minimal attack surface

# ── Stage 1: Build dependencies ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies for cyvcf2 and pysam
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make \
    zlib1g-dev libbz2-dev liblzma-dev libcurl4-openssl-dev \
    libhts-dev tabix bcftools libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir --prefix=/install .


# ── Stage 2: Production image ─────────────────────────────────────────────────
FROM python:3.11-slim AS production

LABEL maintainer="RareGeneAI Team"
LABEL description="Rare Disease Gene Prioritization System"
LABEL version="1.1.0"

WORKDIR /app

# Runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    zlib1g libbz2-1.0 liblzma5 libcurl4 \
    tabix bcftools libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r raregeneai && useradd -r -g raregeneai -d /app -s /sbin/nologin raregeneai

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy application
COPY raregeneai/ /app/raregeneai/
COPY templates/ /app/templates/
COPY pyproject.toml /app/

# Create data directories with proper ownership
RUN mkdir -p /app/data/reference /app/data/cache /app/data/output /app/models \
    /app/tmp/secure \
    && chown -R raregeneai:raregeneai /app

# Environment
ENV PYTHONUNBUFFERED=1
ENV RAREGENEAI_HOME=/app
ENV RAREGENEAI_SECURE_TEMP_DIR=/app/tmp/secure
ENV RAREGENEAI_LOG_DIR=/app/logs

# Switch to non-root user
USER raregeneai

# Healthcheck via HTTP endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default entrypoint
ENTRYPOINT ["python", "-m", "raregeneai.cli"]
CMD ["--help"]
