# ==============================================================================
# DeepLog-Infinity: Production Multi-Stage Hardened Dockerfile
# Base: Python 3.11 Slim (Debian Bookworm)
# Security: Non-Root Execution (UID 10001), Minimal Attack Surface, Healthchecked
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build & Dependency Pre-Warming
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build-time dependencies required for C-extensions and eBPF tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for isolated layer caching
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Pre-warm Python dependencies before copying source code for optimal Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy package metadata and source code for local installation
COPY setup.py README.md ./
COPY src/ ./src/

# Install DeepLog in editable/production wheel format into virtualenv
RUN pip install --no-cache-dir --no-deps -e .

# ------------------------------------------------------------------------------
# Stage 2: Hardened Runtime Container
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="DeepLog Infinity Team <architecture@deeplog.ai>" \
      org.opencontainers.image.title="DeepLog Mission Control Cockpit" \
      org.opencontainers.image.description="Autonomous Neuro-Immune System for Microsecond Distributed Observability" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    DEEPLOG_ENV=production \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    PROMETHEUS_METRICS_PORT=8000

# Install minimal runtime system utilities and security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    procps \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Security Hardening: Dedicated non-privileged user and group
RUN groupadd -g 10001 deeplog && \
    useradd -u 10001 -g deeplog -m -s /bin/bash deeplog

# Copy pre-built virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Copy application code, datasets, scripts, and model configurations with non-root ownership
COPY --chown=deeplog:deeplog app.py setup.py README.md requirements.txt ./
COPY --chown=deeplog:deeplog src/ ./src/
COPY --chown=deeplog:deeplog dataset/ ./dataset/
COPY --chown=deeplog:deeplog deploy/ ./deploy/
COPY --chown=deeplog:deeplog results/ ./results/
COPY --chown=deeplog:deeplog scripts/ ./scripts/

# Ensure runtime directories for data persistence and logging exist with correct ownership
RUN mkdir -p /app/data /app/logs /app/results && \
    chown -R deeplog:deeplog /app

# Switch to non-root user
USER deeplog

# Expose Streamlit Mission Control Cockpit (8501) and Prometheus Metrics Exporter (8000)
EXPOSE 8501 8000

# Enterprise Healthcheck probe against Streamlit core health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default Entrypoint launches DeepLog Mission Control Cockpit
ENTRYPOINT ["streamlit", "run", "app.py"]
CMD ["--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
