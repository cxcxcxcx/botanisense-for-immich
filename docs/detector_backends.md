# Detection backends

The yolo stage picks its detector from `YOLO_BACKEND` at process start
(`pipeline._yolo_backend_module`, mirrored read-only in
`app._yolo_backend_name` for the health badge). All backends expose the
same interface:

    get_model() / unload_model() / detect(images, conf=...) -> [[{…}], …]

with each detection `{"label", "score", "x", "y", "w", "h"}` normalized
to the source image (x/y = top-left). `DECODE_PREVIEW_MAX_SIZE` on the
module tells the pipeline what preview size to decode at.

| `YOLO_BACKEND` | Module | Weights | Params | ONNX (deploy) | License |
|---|---|---|---|---|---|
| `rtmdet` (default) | `yolo_rtmdet.py` | RTMDet-Ins-tiny, trained obj6 (5 classes) | 6.47 M | 20.5 MB fp32, `models/` | Apache-2.0 base; weights TBD |
| `obj6` (opt-in) | `yolo_obj6.py` | YOLO26n-seg, trained obj6 (5 classes) | 3.05 M | 5.7 MB fp16, not shipped | AGPL-3.0 |

The YOLO-World open-vocab backend (`world`, `yolo.py`) was removed
together with the `ultralytics` dependency; `YOLO_BACKEND=world` now
fails at the yolo stage with a pointer here.

Both trained exports share the obj6 5-class table
(`plant, person, hand, flower, tree`) and the same product semantics;
they differ in architecture, decode contract and score calibration.

## Configuration

Nothing to set for the default: the RTMDet fp32 ONNX lives at
`models/obj6_rtmdet_ep97.onnx` and baked into the image at
`/app/models/` (`yolo_rtmdet.DEFAULT_MODEL_PATH`, resolved from the
CWD). `YOLO_RTMDET_ONNX` overrides the path.

To run your own Ultralytics export instead (AGPL-3.0 weights):

```yaml
environment:
  YOLO_BACKEND: obj6
  YOLO_OBJ6_ONNX: /app/obj6/2026_11_20_obj6_control_5c.fp16.onnx
volumes:
  - ./data/models:/app/obj6:ro
```

The obj6 backend is plain numpy + onnxruntime — Ultralytics itself is
not needed at runtime, only to train/export the weights. An fp16 RTMDet
graph (`obj6_rtmdet_ep97.fp16.onnx`, 10.3 MB) exists for on-device use;
on CPU it is app-level identical and no faster, so the image ships fp32.

## Score calibration is NOT portable between backends

This is the one thing to get right when adding an export. The pipeline
passes `conf=0.05` (the old YOLO-World default); both trained backends floor
that at their own operating point:

* `yolo_obj6.CONF_FLOOR = 0.25` — the Ultralytics export's scores are
  sharply separated (median ~0.59 per box).
* `yolo_rtmdet.SCORE_THR = 0.35` — the mmdet export's scores are much
  flatter (median ~0.11 on the same photos). At mmdet's own
  `score_thr=0.05` the box set is ~100/image of mostly noise.

Measured on the 81-image obj6 val split (GT-matched at IoU ≥ 0.5, the
threshold sweep is in the modeling repo's `RESULT.md`):

| backend | threshold | P | R | F1 | boxes/img | images with ≥1 GT crop |
|---|---|---|---|---|---|---|
| obj6 | 0.25 (**production**) | .534 | .515 | .524 | 6.0 | 66/77 |
| rtmdet | 0.05 (mmdet test_cfg) | .054 | .828 | .101 | 100 | 76/77 |
| rtmdet | 0.30 | .460 | .594 | .519 | 8.0 | 74/77 |
| rtmdet | 0.35 (**shipped**) | .549 | .531 | **.540** | 6.0 | 70/77 |
| rtmdet | 0.40 | .620 | .481 | .542 | 4.8 | 67/77 |

The shipped 0.35 is chosen to match obj6's crop budget (6.0
boxes/image) rather than to maximise F1: at the same budget RTMDet
wins on P, F1 and per-image coverage. The floor matters because
`pipeline` passes `conf=0.05`, which both trained backends ignore.

## Measured trade-offs

* **Accuracy.** COCO box AP50 on the 81-image val split: obj6 .5720,
  RTMDet .5558 (masks .5100 / .4899). The gap is concentrated in dense
  multi-plant photos (the mined `new_37` subset: box AP50 .4178 vs
  .4754 — RTMDet is the better one there); on the `old_44` subset
  obj6 wins. What this looks like in the app: on a normal
  single-plant photo the boxes agree; on a photo with 15–20 separate
  pots the obj6 export finds slightly more (`aecaae90`: 18 vs 15 at
  the shipped floors), and occasionally splits one clump into several
  overlapping boxes.

  Per class at the thresholds the pipeline actually applies (val,
  GT-matched at IoU ≥ 0.5), obj6 → rtmdet:

  | class | floor | P | R |
  |---|---|---|---|
  | person | 0.50 (`PERSON_MIN_CONF`) | .892 → .933 | **.733 → .622** |
  | hand | 0.05 (`HAND_MIN_CONF`) | .650 → .538 | .565 → .609 |
  | plant | 0.15 (`BBOX_BIOCLIP_MIN`) | .493 → **.550** | .484 → .516 |
  | flower | 0.15 | .521 → .506 | .457 → **.506** |
  | tree | 0.15 | **.506 → .482** | .506 → .506 |

  The person-recall drop is the only user-facing regression
  (`has_person`): on the full 24k library it moves 667 photos from
  flagged to unflagged and 183 the other way. The plant crop stage is
  roughly neutral: 102.8k crop boxes vs 103.6k, median 3/image for both,
  photos with at least one crop 23,121 vs 22,969 (RTMDet loses all crops
  on 504 photos, gains them on 656).

  Library-wide box agreement is lower than val suggests (64% of obj6
  crop boxes have an IoU≥0.5 RTMDet partner, 94% label agreement) —
  these are mostly photos outside the training distribution, which is
  where the two models diverge most.
* **Speed.** ~76–85 ms/img vs ~67–71 ms/img at 960 px on CPU (~1.15×).
  fp16 buys no CPU time in ORT (casts on the way in/out) — ship fp32.
* **Size.** 20.5 MB fp32 ONNX vs 5.7 MB fp16 (21 MB fp32) — ~2× the
  parameters.

## Why RTMDet is the default

Ultralytics-trained weights are AGPL-3.0-derived, and so was any image
that linked `ultralytics` (the old YOLO-World backend made it a runtime
dependency). AGPL §7 forbids adding a non-commercial /
no-derivative-training clause on top. RTMDet (mmdet) is Apache-2.0, so
this backend's code carries no copyleft (license terms for the trained
weights are still to be decided), and the image no
longer installs `ultralytics` (or its `clip` fork) at all. Nothing else
in the stack blocks it: BioCLIP-2 is MIT, TreeOfLife-200M is CC0, the
repo is MIT.

## Export + parity procedure

Run in the modeling repo
(`botanisense_modeling/plant_yolo/scripts/`, mmdet venv):

```bash
# 1. export the box branch only (the mask head is unused by the app)
.venv-mmdet/bin/python scripts/export_rtmdet_onnx.py \
    --ckpt out/train_obj6_rtmdet/obj6_rtmdet_best_ep97.pth \
    --out  out/train_obj6_rtmdet/obj6_rtmdet_ep97.onnx

# 2. two-layer parity gate: raw-tensor frame parity vs torch, then a
#    COCOeval reproduction of the scored checkpoint's numbers
.venv-mmdet/bin/python scripts/verify_rtmdet_onnx.py \
    --onnx out/train_obj6_rtmdet/obj6_rtmdet_ep97.onnx \
    --ckpt out/train_obj6_rtmdet/obj6_rtmdet_best_ep97.pth \
    --data out/coco/obj6 \
    --ref  out/train_obj6_rtmdet/ep97_pred.bbox.json
```

The parity gate must reproduce the reference COCO eval exactly; a small
mAP drift means the numpy decode disagrees with mmdet, not that the
model is worse.

## In-app A/B

`scripts/yolo_backend_ab.py` runs two trained backends over the same
library photos (read-only; nothing is written) through the product's
real path — `fetch_thumbs` → `detect` → `_merge_overlapping_boxes` — and
reports per-label counts plus crop-set agreement at IoU ≥ 0.5:

```bash
YOLO_OBJ6_ONNX=$PWD/data/models/2026_11_20_obj6_control_5c.fp16.onnx \
.venv/bin/python scripts/yolo_backend_ab.py \
    --a obj6 --b rtmdet --db data/state.db \
    --assets b19a9b47,aecaae90 --random 40 --csv data/yolo_ab.csv
```

It selects backends by name (`--a`/`--b`, both from the table above), so
`YOLO_BACKEND` does not need to be set.

## Shadow instance

`compose.rtmdet-ab.yml` runs the teacher classifier + RTMDet against a
throwaway copy of the production DB on :8775. `YOLO_BACKEND=rtmdet` also
drives `scripts/yolo_backfill.py`, which is the way to re-detect an
existing library without a pipeline run.
