"""BioCLIP-2 on Core ML — the macOS-native image tower for the teacher.

`CoreMLEngine` IS a `TreeOfLifeClassifier`: same 867k-row text table, same
names, same `format_*` / `predict` — only the image encoder is swapped for
a Core ML export of BioCLIP-2's ViT-L/14 visual tower
(scripts/export_teacher_coreml.py), run on the ANE/GPU. The embeddings live
in the teacher's space (golden cosine ≥ .999), so stored embeddings and
predictions stay comparable with `ENGINE=teacher` rows.

Package layout ($COREML_PACKAGE, default data/models/bioclip-2-coreml):

    encoder_b16.mlpackage   static batch 16, fp16, macOS 15+
    encoder_b16.mlmodelc    compiled on first load and kept (compiling a
                            580 MB package costs seconds every load)
    golden.npz              real crops already through `preprocess_img`
                            + torch fp32 reference embeddings
    manifest.json           batch, logit_scale, versions, gate numbers

The torch weights are never loaded: the text side is the precomputed
table, so the only model parameter the classifier needs is logit_scale
(from the manifest). Resident ≈ 2.5 GB label table + the Core ML model,
versus ~5.9 GB for the torch teacher.

Selection (`select_engine`): ENGINE=coreml requires the package; ENGINE=
auto picks coreml on macOS when the package and coremltools are present,
otherwise the torch teacher. A load or golden-gate failure falls back to
the teacher with one `[model]` log line saying why.

Download (`ensure_package`): on macOS with coremltools, a missing package
is fetched once from Hugging Face (HF_REPO at the pinned HF_REVISION,
~625 MB) and its sha256 checked before it is moved into place, so a
present manifest.json always means a complete, verified package. The
fetch runs as `python -m coreml_engine fetch <dir>` in a child whose env
drops HF_HUB_OFFLINE — pipeline_worker sets that flag before
huggingface_hub is imported, and the hub reads it once at import.
COREML_AUTO_DOWNLOAD=0 turns the download off.
"""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_log = logging.getLogger("pipeline")

COS_MEAN_GATE = 0.999
COS_MIN_GATE = 0.998
_COMPUTE_UNITS = {"all": "ALL", "cpu_gpu": "CPU_AND_GPU", "cpu": "CPU_ONLY"}

HF_REPO = "cxcxcxcx/bioclip-2-coreml"
HF_REVISION = "19254e4bec083c46c28e1148faa174deeb03f4a3"
# sha256 of encoder_b16.mlpackage at HF_REVISION (same as its manifest).
PACKAGE_SHA256 = "fc11f66c798cb4dca7d89a315f27c926d5917153ec7d63ccba87bf10ef3670f8"
_FETCH_PATTERNS = ["encoder_b16.mlpackage/*", "golden.npz", "manifest.json", "LICENSE"]


def default_package_dir() -> str:
    return os.environ.get("COREML_PACKAGE", "data/models/bioclip-2-coreml")


def coreml_available() -> bool:
    try:
        import coremltools  # noqa: F401
    except ImportError:
        return False
    return True


def unavailable_reason(*, system: str | None = None,
                       package_dir: str | Path | None = None,
                       have_coremltools: bool | None = None) -> str | None:
    """Why the Core ML engine can't load here, or None if it can."""
    system = system or platform.system()
    pkg = Path(package_dir or default_package_dir())
    if have_coremltools is None:
        have_coremltools = coreml_available()
    if system != "Darwin":
        return "not macOS"
    if not have_coremltools:
        return "coremltools not installed (uv sync --extra mac)"
    if not (pkg / "manifest.json").exists():
        return f"no package at {pkg}"
    return None


def select_engine(engine: str, *, system: str | None = None,
                  package_dir: str | Path | None = None,
                  have_coremltools: bool | None = None,
                  quiet: bool = False) -> str:
    """Map $ENGINE to the engine that should actually load: 'coreml',
    'teacher' or the mobile-* value unchanged. Pure apart from the
    defaults, so the auto rules are unit-testable on Linux."""
    if engine not in ("auto", "coreml"):
        return engine
    why = unavailable_reason(system=system, package_dir=package_dir,
                             have_coremltools=have_coremltools)
    if why is None:
        return "coreml"
    if engine == "coreml" and not quiet:
        _log.info("[model] ENGINE=coreml unavailable (%s) — using BioCLIP-2 torch", why)
    return "teacher"


def status_path() -> Path:
    """Where the pipeline worker records which engine it actually loaded
    for ENGINE=auto/coreml (the choice is made in the worker, at model
    load) — /api/health reads it back."""
    return Path(default_package_dir()).parent / "engine_loaded.json"


def record_loaded(engine: str, **info) -> None:
    """Best-effort write of status_path(); never raises."""
    try:
        p = status_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"engine": engine, **info,
                                 "at": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    except OSError:
        pass


def read_loaded() -> dict | None:
    try:
        return json.loads(status_path().read_text())
    except (OSError, ValueError):
        return None


def sha256_dir(path: Path) -> str:
    """Hash of every file under `path` (relative names + bytes, sorted) —
    the manifest's `sha256` (scripts/export_teacher_coreml.py)."""
    import hashlib

    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(f.relative_to(path)).encode())
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def fetch_package(dest: str | Path, *, expected_sha256: str = PACKAGE_SHA256) -> None:
    """Download the package from HF into `dest`: fetched into
    `<dest>.partial`, verified, then renamed, so `dest` only ever holds
    a complete package. Raises on a hash mismatch."""
    from huggingface_hub import snapshot_download

    dest = Path(dest)
    part = dest.with_name(dest.name + ".partial")
    # Resumes an interrupted fetch: snapshot_download skips files whose
    # size/etag already match.
    snapshot_download(HF_REPO, revision=HF_REVISION, local_dir=part,
                      allow_patterns=_FETCH_PATTERNS)
    manifest = json.loads((part / "manifest.json").read_text())
    got = sha256_dir(part / manifest.get("file", "encoder_b16.mlpackage"))
    if got != expected_sha256 or got != manifest.get("sha256"):
        shutil.rmtree(part, ignore_errors=True)
        raise RuntimeError(f"Core ML package sha256 mismatch: got {got[:12]}, "
                           f"expected {expected_sha256[:12]}")
    shutil.rmtree(part / ".cache", ignore_errors=True)
    if dest.exists():
        shutil.rmtree(dest)
    part.rename(dest)


def will_download(package_dir: str | Path | None = None, *, system: str | None = None,
                  have_coremltools: bool | None = None) -> bool:
    """The package is missing only because nothing has fetched it yet:
    ensure_package() will download it at the next model load."""
    pkg = Path(package_dir or default_package_dir())
    if (pkg / "manifest.json").exists():
        return False
    if (system or platform.system()) != "Darwin":
        return False
    if not (coreml_available() if have_coremltools is None else have_coremltools):
        return False
    return os.environ.get("COREML_AUTO_DOWNLOAD", "1").strip().lower() not in (
        "0", "false", "no")


def ensure_package(package_dir: str | Path | None = None, *, system: str | None = None,
                   have_coremltools: bool | None = None) -> bool:
    """True if a verified package is in place, downloading it first when
    it's missing and this machine could use it. Never raises."""
    pkg = Path(package_dir or default_package_dir())
    if (pkg / "manifest.json").exists():
        return True
    if not will_download(pkg, system=system, have_coremltools=have_coremltools):
        return False
    _log.info("[model] downloading the Core ML encoder (%s, ~625 MB, once) into %s",
              HF_REPO, pkg)
    env = {k: v for k, v in os.environ.items()
           if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "coreml_engine", "fetch", str(pkg)],
            cwd=Path(__file__).parent, env=env, capture_output=True, text=True,
            timeout=3600,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        _log.info("[model] Core ML download failed (%s) — using BioCLIP-2 torch", e)
        return False
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or ["?"]
        _log.info("[model] Core ML download failed (%s) — using BioCLIP-2 torch", tail[0])
        return False
    _log.info("[model] Core ML encoder downloaded and verified in %.0fs",
              time.monotonic() - t0)
    return (pkg / "manifest.json").exists()


def encode_batched(predict, x: np.ndarray, batch: int) -> np.ndarray:
    """Run a static-batch encoder over any number of inputs: chunk by
    `batch`, zero-pad the tail chunk, drop the padded rows."""
    out = []
    for i in range(0, len(x), batch):
        chunk = x[i:i + batch]
        n = len(chunk)
        if n < batch:
            pad = np.zeros((batch - n, *chunk.shape[1:]), dtype=chunk.dtype)
            chunk = np.concatenate([chunk, pad])
        out.append(np.asarray(predict(chunk), dtype=np.float32)[:n])
    if not out:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(out)


def golden_gate(emb: np.ndarray, ref: np.ndarray) -> tuple[bool, float, float]:
    cos = (emb * ref).sum(1) / (np.linalg.norm(emb, axis=1) * np.linalg.norm(ref, axis=1))
    mean, low = float(cos.mean()), float(cos.min())
    return mean >= COS_MEAN_GATE and low >= COS_MIN_GATE, mean, low


def _compiled_path(package_dir: Path, manifest: dict) -> Path:
    """The .mlmodelc next to the .mlpackage, compiling it once if absent."""
    pkg = package_dir / manifest.get("file", "encoder_b16.mlpackage")
    compiled = pkg.with_suffix(".mlmodelc")
    if compiled.exists():
        return compiled
    import shutil

    import coremltools as ct

    _log.info("[model] compiling %s (once; kept as %s)", pkg.name, compiled.name)
    shutil.move(ct.utils.compile_model(str(pkg)), str(compiled))
    return compiled


def _make_class():
    # Defined lazily: importing bioclip pulls torch/open_clip, and this
    # module must stay cheap to import for select_engine().
    import torch
    from bioclip.predict import BIOCLIP_MODEL_STR, TreeOfLifeClassifier, preprocess_img

    class _LogitScaleOnly(torch.nn.Module):
        """Stands in for the open_clip model: create_probabilities only
        reads `self.model.logit_scale.exp()`."""

        def __init__(self, scale: float):
            super().__init__()
            self.logit_scale = torch.nn.Parameter(
                torch.tensor(math.log(scale)), requires_grad=False
            )

    class CoreMLEngine(TreeOfLifeClassifier):
        def __init__(self, package_dir: str | Path | None = None,
                     compute_units: str | None = None):
            self.package_dir = Path(package_dir or default_package_dir())
            self._manifest = json.loads((self.package_dir / "manifest.json").read_text())
            self.batch = int(self._manifest.get("batch", 16))
            cu = (compute_units or os.environ.get("COREML_COMPUTE_UNITS") or "all").lower()
            self.compute_units = _COMPUTE_UNITS.get(cu, "ALL")
            self._mlmodel = None
            super().__init__(device="cpu")

        def load_pretrained_model(self, model_str: str = BIOCLIP_MODEL_STR,
                                  pretrained_str: str | None = None):
            self.model_str = model_str or BIOCLIP_MODEL_STR
            self.pretrained_str = pretrained_str
            self.model = _LogitScaleOnly(float(self._manifest.get("logit_scale", 100.0)))
            self.preprocess = preprocess_img
            import coremltools as ct

            path = _compiled_path(self.package_dir, self._manifest)
            t0 = time.monotonic()
            self._mlmodel = ct.models.CompiledMLModel(
                str(path), compute_units=getattr(ct.ComputeUnit, self.compute_units)
            )
            _log.info("[model] Core ML encoder %s loaded in %.1fs (compute_units=%s)",
                      path.name, time.monotonic() - t0, self.compute_units)

        def _predict(self, chunk: np.ndarray) -> np.ndarray:
            return self._mlmodel.predict({"images": chunk})["emb"]

        def encode_preprocessed(self, x: np.ndarray) -> np.ndarray:
            return encode_batched(self._predict, x.astype(np.float32, copy=False), self.batch)

        def self_check(self) -> tuple[bool, float, float]:
            """Golden gate at load: the package's own reference crops must
            reproduce the torch embeddings, or the engine is not used."""
            golden = self.package_dir / "golden.npz"
            if not golden.exists():
                return True, float("nan"), float("nan")
            g = np.load(golden)
            return golden_gate(self.encode_preprocessed(g["images"]), g["emb"])

        @torch.no_grad()
        def create_image_features(self, images, normalize: bool = True):
            x = torch.stack([self.preprocess(img) for img in images]).numpy()
            feats = torch.from_numpy(self.encode_preprocessed(x))
            if normalize:
                return torch.nn.functional.normalize(feats, dim=-1)
            return feats

        def version_key(self) -> str:
            m = self._manifest
            return (f"coreml/{m.get('file', '?')}/{str(m.get('sha256', '?'))[:8]}"
                    f"/{self.compute_units}")

    return CoreMLEngine


_CLS = None


def CoreMLEngine(*args, **kwargs):  # noqa: N802 — factory standing in for the class
    global _CLS
    if _CLS is None:
        _CLS = _make_class()
    return _CLS(*args, **kwargs)


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "fetch":
        fetch_package(argv[1])
        return 0
    print("usage: python -m coreml_engine fetch <package_dir>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
