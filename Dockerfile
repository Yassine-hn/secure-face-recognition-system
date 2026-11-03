# Multi-stage / optimized Dockerfile for Secure Face Recognition System (SFRS)
# Compatible with local Docker run, Docker Compose, Hugging Face Spaces, and Render

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    SFRS_MODE=development \
    HOME=/home/appuser

# Install system dependencies for OpenCV, MediaPipe, SQLCipher, and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libsqlcipher-dev \
    sqlcipher \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user with UID 1000 (standard for Hugging Face Spaces)
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data /app/models && \
    chown -R appuser:appuser /app /home/appuser

WORKDIR /app

# Upgrade pip and install CPU-only PyTorch to minimize image size (~1GB instead of 5GB+ CUDA)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose standard web port
EXPOSE 7860

# Health check probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/_stcore/health || exit 1

# Launch the Streamlit Web Application
CMD ["sh", "-c", "streamlit run web/app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true"]
