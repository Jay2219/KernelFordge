# ==========================================
# KernelForge Backend Dockerfile
# Python 3.11 with FastAPI & Uvicorn
# ==========================================

FROM python:3.11-slim

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy requirements and install dependencies
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy source code and scripts
COPY src/ ./src/
COPY benchmarks/ ./benchmarks/
COPY modal_runner.py sandbox_exec.py ./
COPY .env.example .env

# Expose FastAPI port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Launch Gateway
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

