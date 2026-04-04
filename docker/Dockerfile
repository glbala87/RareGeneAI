# RareGeneAI Docker Image
# =======================
# Multi-stage build for production deployment

# ── Stage 1: Build dependencies ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies for cyvcf2 and pysam
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make \
    zlib1g-dev libbz2-dev liblzma-dev libcurl4-openssl-dev \
    libhts-dev tabix bcftools \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir --prefix=/install .


# ── Stage 2: Production image ─────────────────────────────────────────────────
FROM python:3.11-slim AS production

LABEL maintainer="RareGeneAI Team"
LABEL description="Rare Disease Gene Prioritization System"
LABEL version="1.0.0"

WORKDIR /app

# Runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    zlib1g libbz2-1.0 liblzma5 libcurl4 \
    tabix bcftools \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy application
COPY raregeneai/ /app/raregeneai/
COPY templates/ /app/templates/
COPY pyproject.toml /app/

# Create data directories
RUN mkdir -p /app/data/reference /app/data/cache /app/data/output /app/models

# Environment
ENV PYTHONUNBUFFERED=1
ENV RAREGENEAI_HOME=/app

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "from raregeneai import __version__; print(__version__)"

# Default entrypoint
ENTRYPOINT ["python", "-m", "raregeneai.cli"]
CMD ["--help"]
