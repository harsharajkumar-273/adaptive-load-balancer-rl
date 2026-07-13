# Dockerfile
# Multi-stage build to keep the image slim
FROM python:3.10-slim AS builder

WORKDIR /app

# Install compilation dependencies (required for psutil compiling if wheel is not available)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies in a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.10-slim

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code
COPY src/ /app/src/

# Expose ports (Gateway on 8000, Nodes on 8001-8005)
EXPOSE 8000 8001 8002 8003 8004 8005

# Environment variables defaults
ENV PYTHONUNBUFFERED=1
ENV RUNNING_IN_DOCKER=true

# Default entrypoint
CMD ["python", "src/main.py"]
