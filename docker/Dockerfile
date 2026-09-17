# Training/experimentation environment. CPU-only (Apple Silicon MPS acceleration is
# only available when running natively on macOS, not inside this Linux container —
# training scripts fall back to CPU here; see README for the native-MPS path).
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
# CPU-only torch wheels explicitly: the default PyPI linux wheel pulls in the full
# CUDA toolkit (~1GB+) even though this container never has a GPU.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -e ".[dev]"

COPY configs ./configs
COPY scripts ./scripts

ENTRYPOINT ["python"]
