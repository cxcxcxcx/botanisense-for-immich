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

FROM node:24-alpine AS frontend
WORKDIR /build
# Fetch deps first so source edits don't invalidate the npm layer.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# BotaniSense account integration (sign-in, Diary tab, Send to
# BotaniSense) is compiled out unless this is 1 — see
# frontend/src/lib/features.ts.
ARG VITE_BOTANISENSE_ENABLED=0
ENV VITE_BOTANISENSE_ENABLED=$VITE_BOTANISENSE_ENABLED
# tsc -b && vite build -> /build/dist
RUN npm run build

# ---- Stage 2: runtime -------------------------------------------------------

# 3.12 (not 3.13): hdbscan ships cp312 manylinux wheels, so nothing in
# the dependency set needs a compiler.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

# libgomp1: torch's OpenMP runtime. (cv2 is opencv-python-headless,
# which needs no libgl1/libglib.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch/torchvision: UV_TORCH_BACKEND makes uv resolve the
# PyTorch CPU index instead of the default CUDA build — ~800 MB instead
# of several GB of NVIDIA wheels. This intentionally bypasses uv.lock
# for the torch flavor only; nothing in the codebase is CUDA-aware
# (BioCLIP-2 runs on CPU; the detector is a CPU onnxruntime graph).
ENV UV_TORCH_BACKEND=cpu \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8765 \
    HF_HOME=/app/data/huggingface

WORKDIR /app
# config.toml / .env / data/ / models/ are all resolved relative
# to the CWD — the app must run from /app. Secrets come in via the
# environment (compose env_file), never baked into the image.
COPY pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml

COPY --from=frontend /build/dist frontend/dist
COPY app.py pipeline.py pipeline_worker.py storage.py clusters.py \
     map_clusters.py location_guess.py \
     knn.py knn_io.py immich_client.py run.py face.py \
     mobile_engine.py yolo_obj6.py yolo_rtmdet.py ./
# YuNet face weights — face.py loads by bare filename from the CWD,
# and the default RTMDet detector from models/ — both
# baked in because /app is read-only at runtime. The detector weights
# are CC BY-NC 4.0, so their license and model card ship alongside.
COPY face_detection_yunet_2023mar.onnx ./
COPY models/obj6_rtmdet_ep97.onnx models/LICENSE models/README.md models/

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
