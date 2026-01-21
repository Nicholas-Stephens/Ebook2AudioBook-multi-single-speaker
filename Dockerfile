# Multi-stage build for IndexTTS
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    git-lfs \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Enable Git LFS
RUN git lfs install

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Upgrade pip and install uv package manager (recommended by IndexTTS)
RUN pip3 install --no-cache-dir --upgrade pip uv

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Final runtime image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    git-lfs \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Enable Git LFS
RUN git lfs install

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/books \
             /app/output/chunks \
             /app/output/final \
             /app/cache \
             /app/voice_samples \
             /app/index-tts/checkpoints

# Install IndexTTS from local directory
RUN pip3 install --no-cache-dir -e /app/index-tts || echo "IndexTTS installation will complete on first run"

# Set environment variables - include index-tts in PYTHONPATH
ENV PYTHONPATH="/app:/app/index-tts:$PYTHONPATH"
ENV HF_HOME="/app/cache/huggingface"
ENV HF_HUB_CACHE="/app/index-tts/checkpoints/hf_cache"
ENV TRANSFORMERS_CACHE="/app/cache/transformers"
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
# Set to "cpu" for CPU-only mode, or "cuda:0" for GPU
ENV DEVICE=cuda:0
ENV CUDA_VISIBLE_DEVICES=0

# For CPU-only mode, uncomment these:
# ENV DEVICE=cpu
# ENV CUDA_VISIBLE_DEVICES=''

# Download IndexTTS models during build (auto-setup everything)
RUN python3 /app/download_models.py /app/index-tts/checkpoints || echo "⚠️ Model download will complete on first run if automatic download fails"

# Expose the port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the application
CMD ["python3", "app.py"]