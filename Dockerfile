# BotaniSense for Immich — self-contained image (backend + built SPA).
#
#   docker compose up --build     # recommended; see compose.yml
#
# BioCLIP-2 weights (~4.3 GB) are NOT baked in: they download from
# HuggingFace on first pipeline run into $HF_HOME (a mounted volume),
# so the image stays ~2 GB and rebuilds don't invalidate the model
# cache.
#
# KNN reclassification is data-optional: faiss-cpu + duckdb ship in
# the image, but the feature stays off until KNN_INVENTORY_DIR points
# at an index. See scripts/download_knn_index.py for the official
# imageomics index (~20 GB).

# ---- Stage 1: build the SPA ------------------------------------------------

FROM node:20-alpine AS frontend
WORKDIR /build
# Fetch deps first so source edits don't invalidate the npm layer.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# tsc -b && vite build -> /build/dist
RUN npm run build

# ---- Stage 2: runtime -------------------------------------------------------

# 3.12 (not 3.13): hdbscan ships cp312 manylinux wheels, so nothing in
# the dependency set needs a compiler.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

# libgl1 + libglib2.0-0: ultralytics imports cv2, whose wheel needs
# them at runtime. libgomp1: torch's OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch/torchvision: UV_TORCH_BACKEND makes uv resolve the
# PyTorch CPU index instead of the default CUDA build — ~800 MB instead
# of several GB of NVIDIA wheels. This intentionally bypasses uv.lock
# for the torch flavor only; nothing in the codebase is CUDA-aware
# (BioCLIP-2 and YOLO both run on CPU).
ENV UV_TORCH_BACKEND=cpu \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8765 \
    HF_HOME=/app/data/huggingface

WORKDIR /app
# config.toml / .env / data/ / YOLO weights are all resolved relative
# to the CWD — the app must run from /app. Secrets come in via the
# environment (compose env_file), never baked into the image.
COPY pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml

# Warm model downloads so first scan doesn't need to fetch them, and so
# a read-only /app still works — yolo.py loads weights by bare filename
# from the CWD, and ultralytics resolves its weights dir to
# $CWD/weights (settings weights_dir="weights"), which is root-owned at
# runtime. set_classes() also imports OpenAI CLIP and downloads its
# ViT-B/32 text-encoder weights (~340 MB into /app/weights/clip), so
# doing it here both proves the `clip` dep is present and bakes the
# weights into the image world-readable.
RUN python -c "from ultralytics import YOLOWorld; m = YOLOWorld('yolov8s-world.pt'); m.set_classes(['plant'])"

COPY --from=frontend /build/dist frontend/dist
COPY app.py pipeline.py pipeline_worker.py storage.py clusters.py \
     knn.py knn_io.py immich_client.py run.py yolo.py face.py ./
# YuNet face weights — face.py loads by bare filename from the CWD
# (same read-only-/app constraint as yolov8s-world.pt above).
COPY face_detection_yunet_2023mar.onnx ./

RUN useradd --uid 1000 --create-home botanisense \
    && mkdir -p /app/data \
    && chown botanisense:botanisense /app/data

# The image's default /app/data is uid-1000-owned (named volumes and
# `--user 1000` runs inherit that), but the container STARTS as root:
# the entrypoint fixes the ownership of a root-owned ./data bind mount
# (Compose auto-creates it as root) and then re-execs this CMD via
# setpriv as botanisense. uvicorn therefore never serves as root.
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod 0755 /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

EXPOSE 8765
# /api/health answers 200 once the app is up (it never loads models at
# startup — the heavy imports live in the pipeline child process).
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=5 \
    CMD ["python", "-c", "import urllib.request, sys; sys.exit(0) if urllib.request.urlopen('http://127.0.0.1:8765/api/health', timeout=4).status == 200 else sys.exit(1)"]

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8765"]
