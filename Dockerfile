# --- Base image: lightweight Python, no bloat ---
FROM python:3.11-slim

WORKDIR /app

# Some ML libs (tokenizers, lxml) occasionally need a compiler for edge-case builds.
# Small cost, avoids obscure build failures.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy just requirements first so Docker can cache this layer
# (rebuilds skip reinstalling deps if requirements.txt hasn't changed)
COPY requirements.txt .

# Install CPU-only torch/torchvision FIRST, from PyTorch's CPU wheel index.
# This must happen before `pip install -r requirements.txt`, otherwise pip
# will grab the default CUDA build (several GB, won't fit HF's free tier).
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install everything else. pip sees torch/torchvision already satisfied
# at the exact pinned version and won't touch them again.
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the actual application code
COPY . .

# Pre-download and cache the sentence-transformer models AT BUILD TIME,
# not on first request — matches what download_models.py already does for you.
RUN python download_models.py

# Hugging Face Spaces expects the app on port 7860
EXPOSE 7860

# Run the FastAPI app directly via uvicorn (skips run.py's pip-install-at-runtime
# and --reload, which are dev-only conveniences with no place in a container)
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
