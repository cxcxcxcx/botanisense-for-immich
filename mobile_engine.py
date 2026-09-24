"""BioCLIP-Mobile engine — server-side drop-in for TreeOfLifeClassifier.

Implements the exact classifier surface `pipeline.py` touches (see the
duck-type note in docs/web_model_ab_plan.md §3):

    create_image_features(pil_images, normalize=True)  -> torch (B, 768)
    get_txt_embeddings()                               -> torch (768, N)
    get_current_txt_names()                            -> [[[sci…7], common], …]
    create_probabilities(feats, txt)                   -> torch (B, N)
    format_species_probs / format_grouped_probs        -> storage row dicts
    predict(images=[paths], rank, k)                   -> list of row dicts
    device                                             -> "cpu"

so the species / kingdom / bbox-bioclip stages run unmodified against a
BioCLIP-Mobile package (bioclip-mobile-v0 = MobileCLIP-S0 tower + trained
512→768 head, distilling BioCLIP-2's space).

Differences from the teacher that live here on purpose:

- **Preprocess contract** (manifest.json `input`): short side → 256 bicubic,
  center crop 256×256, /255 — NO mean/std normalization. This is NOT the
  teacher's squash-to-224 CLIP transform; getting it wrong fails the
  package's golden gate loudly (cos ~0.9), which is the point of the gate.
- **Text tower is a table**: labels.f16.npy is (12501, 768), unit rows.
  get_txt_embeddings() returns it TRANSPOSED (768, N) — the same layout as
  the teacher's txt_emb_species.npy, which pipeline's dim guard reads via
  `.shape[0]` (=768, the embed dim).
- **logit scale fixed at 100** (CLIP's exp(logit_scale)); probs are softmax
  over the 12501-row table, same calibration idea as the teacher's 785k
  softmax — NOT comparable numerically across the two engines.

Everything heavy is lazy: the onnxruntime session and the label table load
on first use, so `import mobile_engine` is cheap and the browse-only
server never pays for the engine unless ENGINE is set.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

_log = logging.getLogger("pipeline")

# CLIP logit scale (exp(logit_scale)); the teacher's create_probabilities
# uses the model's own learned value — BioCLIP-2 ships 100.0. The mobile
# head was distilled to the teacher's cosine space, so 100 keeps score
# sharpness in the same regime.
_LOGIT_SCALE = 100.0
_INPUT_SIZE = 256

# Lineage fields in bioclip's txt_names positional order. labels.json
# spells class as "klass" (json can't use the python keyword).
_TOL_RANK_FIELDS = (
    ("kingdom", "kingdom"),
    ("phylum", "phylum"),
    ("class", "klass"),
    ("order", "order"),
    ("family", "family"),
    ("genus", "genus"),
    ("species", "species"),
)


def default_package_dir() -> str:
    """Package dir from $MOBILE_PACKAGE (ENGINE selects the flavor; the
    dir decides what actually loads). Default matches the compose.s0.yml
    layout and the repo's gitignored data/models/ checkout path."""
    return os.environ.get("MOBILE_PACKAGE", "data/models/bioclip-mobile-v0")


def resolve_engine_name() -> str:
    """$ENGINE selector: 'teacher' (default) keeps the BioCLIP-2 path;
    'mobile-<flavor>' builds a MobileEngine. Returns the normalized
    engine id used in logs and /api/health."""
    raw = (os.environ.get("ENGINE") or "teacher").strip().lower()
    return raw or "teacher"


class MobileEngine:
    """ONNX BioCLIP-Mobile encoder + precomputed label-table head."""

    def __init__(self, package_dir: str | Path | None = None):
        self.package_dir = Path(package_dir or default_package_dir())
        self.device = "cpu"
        self._session = None
        self._pre = None          # torchvision preprocess, lazy
        self._txt = None          # (768, N) float32 torch tensor
        self._names = None        # [[[sci…7], common], …]
        self._rows = None         # raw labels.json rows
        self._version = None      # identity key string
        manifest_path = self.package_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"MobileEngine package dir {self.package_dir} has no "
                f"manifest.json — set MOBILE_PACKAGE to an exported "
                f"bioclip-mobile package"
            )
        self._manifest = json.loads(manifest_path.read_text())

    # ---------- identity -------------------------------------------------

    def _encoder_path(self) -> Path:
        """Which ONNX graph to serve. fp32 on CPU: x86 has no fast fp16
        kernels, so the fp16 graph (needed on-device for size) runs ~30%
        slower server-side (measured 22.3 vs 28.8 img/s batch-16 on the
        deploy box). All shipped packages carry both graphs."""
        fp32 = self.package_dir / "encoder.fp32.onnx"
        return fp32 if fp32.exists() else self.package_dir / "encoder.fp16.onnx"

    def version_key(self) -> str:
        """Identity string for logs / album_meta markers: package +
        code_rev + encoder md5 prefix + labels md5 prefix. Changing ANY
        component invalidates every stored embedding/prediction — the
        shadow-DB setup script records this into album_meta."""
        if self._version is None:
            enc = self._encoder_path()
            lab = self.package_dir / "labels.f16.npy"

            def _md5(p: Path) -> str:
                h = hashlib.md5()
                with p.open("rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                return h.hexdigest()

            self._version = (
                f"{self._manifest.get('package', self.package_dir.name)}/"
                f"{self._manifest.get('code_rev', '?')}/"
                f"{_md5(enc)[:8]}/labels:{_md5(lab)[:8]}"
            )
        return self._version

    # ---------- lazy model state -----------------------------------------

    def _get_session(self):
        if self._session is None:
            import onnxruntime as ort

            model_path = self._encoder_path()
            _log.info(
                "[model] loading BioCLIP-Mobile ONNX %s (%.1f MB)",
                model_path, model_path.stat().st_size / 1e6,
            )
            # Threads pinned like the teacher path (torch.set_num_threads
            # analogy): the run child is forked from uvicorn and inherits
            # whatever affinity it had — be explicit instead.
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = min(8, os.cpu_count() or 8)
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            _log.info(
                "[model] BioCLIP-Mobile ready  version=%s",
                self.version_key(),
            )
        return self._session

    def _get_txt(self):
        """(768, N) float32 unit-column table as a torch tensor — the
        transposed twin of labels.f16.npy's (N, 768) unit rows."""
        if self._txt is None:
            import torch

            table = np.load(self.package_dir / "labels.f16.npy").astype(
                np.float32
            )
            table /= np.linalg.norm(table, axis=1, keepdims=True)
            self._txt = torch.from_numpy(np.ascontiguousarray(table.T))
        return self._txt

    def _get_rows(self) -> list[dict]:
        if self._rows is None:
            self._rows = json.loads(
                (self.package_dir / "labels.json").read_text()
            )
        return self._rows

    def _get_names(self) -> list[list]:
        """bioclip's txt_names shape: each entry is [scientific_names,
        common_name] where scientific_names is the 7-item lineage list
        (kingdom…species). pipeline._kingdom_bins reads name_ary[0][0]
        as the kingdom string; format_* maps the fields positionally."""
        if self._names is None:
            self._names = [
                [
                    [
                        row.get("kingdom", ""),
                        row.get("phylum", ""),
                        row.get("klass", ""),
                        row.get("order", ""),
                        row.get("family", ""),
                        row.get("genus", ""),
                        row.get("species", ""),
                    ],
                    row.get("common", "") or "",
                ]
                for row in self._get_rows()
            ]
        return self._names

    # ---------- classifier surface ---------------------------------------

    def _get_pre(self):
        """The package's REAL preprocess, verified bit-exact against
        golden.npz: mobileclip's inference transform = torchvision
        Resize(short side 256, BILINEAR) + CenterCrop(256) + ToTensor
        (/255, CHW). The manifest's `input` string says "bicubic", but
        that is a doc bug — the export built golden x with this exact
        transform, and a PIL-bicubic hand-implementation drops embedding
        cosine from ~.99999 to ~.9989 (fails the golden gate).
        """
        if self._pre is None:
            from torchvision import transforms
            from torchvision.transforms import InterpolationMode

            self._pre = transforms.Compose([
                transforms.Resize(
                    _INPUT_SIZE, interpolation=InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(_INPUT_SIZE),
                transforms.ToTensor(),
            ])
        return self._pre

    def preprocess(self, img: Image.Image):
        """RGB PIL → (3, 256, 256) float32 CHW in [0, 1] (torch tensor,
        mobileclip transform exactly)."""
        return self._get_pre()(img.convert("RGB"))

    def create_image_features(self, images, normalize: bool = True):
        """Encode PIL images → (B, 768) torch tensor (L2-normalized when
        `normalize`, matching the teacher's contract; the ONNX already
        emits unit vectors — the renorm is a cheap guarantee)."""
        import torch

        batch = torch.stack([self.preprocess(img) for img in images])
        out = self._get_session().run(
            None, {"images": batch.numpy().astype(np.float32)}
        )[0]
        feats = torch.from_numpy(np.asarray(out, dtype=np.float32))
        if normalize:
            feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats

    def get_txt_embeddings(self):
        return self._get_txt()

    def get_current_txt_names(self):
        return self._get_names()

    def create_probabilities(self, img_features, txt_features=None):
        """softmax(logit_scale · feats @ txt) — txt defaults to the full
        table (the teacher requires both, but defaulting keeps ad-hoc
        callers like run_knn_reclassify_bbox working)."""
        import torch

        txt = self._get_txt() if txt_features is None else txt_features
        logits = _LOGIT_SCALE * img_features @ txt
        return torch.nn.functional.softmax(logits, dim=1)

    def _classification_dict(self, idx: int, rank_name: str) -> dict:
        """bioclip's create_classification_dict equivalent: the row's
        lineage entry for `rank_name` + common_name, plus the three
        lineage fields storage.replace_predictions persists on every
        row (kingdom/family/genus)."""
        names, common = self._get_names()[idx]
        fields = [f for f, _ in _TOL_RANK_FIELDS]
        d: dict = {}
        if rank_name == "species":
            d["species"] = f"{names[5]} {names[6]}".strip()
        else:
            d[rank_name] = names[fields.index(rank_name)]
        d["common_name"] = common
        for extra in ("kingdom", "family", "genus"):
            if extra not in d:
                d[extra] = names[fields.index(extra)]
        return d

    def format_species_probs(self, image_key: str, probs, k: int = 5) -> list[dict]:
        """Top-k species rows — same dict shape the teacher produces
        (file_name + classification fields + score)."""
        k = min(k, probs.shape[0])
        topk = probs.topk(k)
        out = []
        for idx, prob in zip(topk.indices.tolist(), topk.values.tolist()):
            item = {"file_name": image_key}
            item.update(self._classification_dict(idx, "species"))
            item["score"] = float(prob)
            out.append(item)
        return out

    def format_grouped_probs(
        self, image_key: str, probs, rank, min_prob: float = 1e-9, k: int = 5
    ) -> list[dict]:
        """Sum probs into same-value buckets at `rank` (bioclip's grouped
        formatting) and return the top-k buckets as row dicts. Used by
        the kingdom stage (rank='kingdom') and any non-species run rank.
        Buckets span at most 12501 rows — a plain python walk beats a
        vectorized bincount at this size."""
        rank_name = getattr(rank, "name", str(rank)).lower()
        fields = [f for f, _ in _TOL_RANK_FIELDS]
        if rank_name not in fields:
            raise ValueError(f"unsupported rank {rank_name!r}")
        field = dict(_TOL_RANK_FIELDS)[rank_name]
        names = self._get_names()
        col = fields.index(field)
        buckets: dict[str, float] = {}
        first_idx: dict[str, int] = {}
        for i, p in enumerate(probs.tolist()):
            if p <= min_prob:
                continue
            value = names[i][0][col]
            if not value:
                continue
            buckets[value] = buckets.get(value, 0.0) + p
            first_idx.setdefault(value, i)
        top = sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)[:k]
        out = []
        for value, score in top:
            item = {"file_name": image_key}
            item.update(self._classification_dict(first_idx[value], rank_name))
            # The bucket's own value wins over the representative row's
            # (identical by construction, but explicit for rank='species'
            # where _classification_dict joins genus+species).
            item[rank_name] = value
            item["score"] = float(score)
            out.append(item)
        return out

    def predict(self, images, rank, k: int = 5, min_prob: float = 1e-9) -> list[dict]:
        """bioclip's predict() shape for FILE PATHS (the bbox-bioclip
        stage passes crop paths, not PIL images). Rows are keyed by the
        path string, which _group_predictions relies on."""
        if isinstance(images, (str, Path)):
            images = [images]
        feats = self.create_image_features(
            [Image.open(p) for p in images], normalize=True
        )
        probs = self.create_probabilities(feats)
        rank_name = getattr(rank, "name", str(rank)).lower()
        out: list[dict] = []
        for i, key in enumerate(images):
            if rank_name == "species":
                out.extend(self.format_species_probs(key, probs[i], k))
            else:
                out.extend(
                    self.format_grouped_probs(key, probs[i], rank, min_prob, k)
                )
        return out
