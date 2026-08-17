# Person-flag classifier — design + maintenance notes

This directory contains a logistic regression classifier that predicts
whether a photo contains a person we **care about** (foreground,
face-visible, identifiable, or interactive subject) — based on
the 768-d BioCLIP-2 embedding of the photo.

It is stored at `data/person_classifier.joblib` and is the production
signal for the `has_person` column. YOLO remains the upstream detector
(it finds the bounding boxes), but the classifier decides the final
flag.

---

## Why we built it

YOLO-World fired on `person` for many photos that don't actually
contain a real person (hand-only, jacket-shaped, photo-of-a-photo,
empty background), and it *missed* many real persons when the photo
was flowers-with-person, kids-in-a-garden, or other low-contrast
compositions. We tried:

1. Raising the YOLO threshold from 0.10 → 0.50 — caught some FPs but
   dropped real persons like the child on the tricycle.
2. Hand-suppression rule (containment ≥ 0.85) — caught most hand FPs
   but didn't help recall on the missed persons.
3. **The BioCLIP-2 classifier** — best of both: catches real persons
   YOLO misses, suppresses the use-case-irrelevant ones YOLO flags.

---

## Training data (auto-derived, no manual labels)

From `storage.py` + `pipeline.py`:

**Positives** (`actual_person = 1`):
- `assets.has_person = 1`
- `MAX(detections.score) WHERE label='person' >= 0.90`
- **AND** one of:
  - `MAX(detections.score) WHERE label='face' >= 0.50` (face box → real person)
  - No hand box with `score >= 0.05` has `≥ 0.85` containment inside
    any person box (rules out hand silhouettes)

**Negatives** (`actual_person = 0`):
- `assets.has_person = 0`
- `taxon_scores.taxon_name = 'Plantae' AND score >= 0.50` (plant
  candidates — extremely high precision for no-person)

Training uses `LogisticRegression(C=1.0, class_weight='balanced')`
to compensate for the ~30:1 imbalance.

---

## How we know it works

### 1. Population-weighted evaluation (`person_label_aggregate_weighted.py`)

70 hand-labeled photos across 7 strata (hp=1 ∩ model-low, hp=1 ∩
model-high, hp=0 ∩ Plantae, hp=0 ∩ model-would-flip, etc.),
re-weighted to reflect actual population sizes in the DB:

| Target | Detector | Precision | Recall | F1 |
|---|---|---|---|---|
| `actual_person` | YOLO | 0.924 | 0.525 | 0.669 |
| `actual_person` | Classifier | **1.000** | **0.602** | **0.751** |
| `use_case_relevant` | YOLO | 0.728 | 0.669 | 0.697 |
| `use_case_relevant` | Classifier | **0.958** | **0.964** | **0.961** |

### 2. Train/holdout sanity check (`person_classifier_holdout.py`)

Train logistic regression on half of the 70 labels, evaluate on the
other half (5 random seeds × 2 folds = 10 runs each):

| Target | Method | F1 |
|---|---|---|
| `actual_person` | Half-train | 0.826 |
| `actual_person` | Full production | 0.845 |
| `use_case_relevant` | Half-train | 0.884 |
| `use_case_relevant` | Full production | 0.951 |

The half-train model reaches F1 0.83–0.88 from just 35 human labels,
which means **BioCLIP embeddings carry a learnable signal** and the
production model's F1 0.95 is not overfit.

### 3. Failure cases (from labels' `notes` field)

The `notes` field on each labeled photo captures *why* a detector was
wrong. The dominant failure modes:

- `hand_only` (YOLO false positive, classifier suppresses)
- `pure_<thing>` (YOLO false positive on similar-textured object)
- `background_walkers` (real person, but not use-case-relevant)
- `cropped_person_<edge>` (real person partially out of frame)
- `tiny_<thing>` (real person too small to care about)

---

## The labeling workflow (re-runnable)

If you want to relabel or extend:

```bash
# 1. Sample — writes data/label_sample/batch_NN.yaml
PYTHONPATH=. python scripts/person_label_sample.py \
    --per-stratum 15 --batch-size 5

# 2. Label — for each batch, spawn an agent with the prompt
cat scripts/person_label_prompt.md   # read this first

# 3. Aggregate — equal-weighted (per-stratum) and population-weighted
PYTHONPATH=. python scripts/person_label_aggregate.py
PYTHONPATH=. python scripts/person_label_aggregate_weighted.py

# 4. Sanity check — train/holdout
PYTHONPATH=. python scripts/person_classifier_holdout.py
```

Each batch YAML has these fields, where the labeler fills in the last
4:

```yaml
- stratum: 1_yolo_only_disagree
  asset_id: <uuid>
  filename: <original_file_name>
  thumb_path: data/thumbs/<uuid>.jpg
  yolo_max: 0.7
  yolo_person: 1
  model_p: 0.05
  actual_person: 0|1      # any human, anywhere in the photo?
  use_case_relevant: 0|1  # foreground / face / identifiable / interactive?
  notes: "free-text short tag"
  sublabel: "optional fine-grain"
```

The labeling spec is `scripts/person_label_prompt.md` — it includes a
decision table with worked examples and is the source of truth for
what counts as a "person we care about."

---

## Recommended flip rule for production

`storage.apply_classifier_flips(conn, *, model_path)` (see
`person_apply_flips.py`):

```
For every asset with a BioCLIP embedding:
  p = sigmoid(embedding · coef + intercept)

  if p >= 0.5 AND has_person == 0:
      # model says use-case-relevant person but YOLO missed it
      has_person = 1
      reason = "classifier_flip_up"

  # (Down-flips from has_person=1 → 0 are NOT automatic; they need
  # a stricter rule because the 70-label sample has zero such cases
  # where the classifier was right and YOLO was wrong on the use-case
  # signal. If we want them, lower the threshold to ~0.2 and
  # gate on yolo_max < 0.65.)
```

After flipping, re-run `/api/results` (cache invalidates via the
existing `_invalidate_results_cache("all:")` helper).

---

## Files in this design

| File | Purpose |
|---|---|
| `person_classifier_eval.py` | Trains and writes `data/person_classifier.joblib` |
| `person_label_sample.py` | Builds stratified sample batches for manual labeling |
| `person_label_prompt.md` | Labeling spec — what counts as actual_person / use_case_relevant |
| `person_label_aggregate.py` | Equal-weighted metrics across the labeled batches |
| `person_label_aggregate_weighted.py` | Population-weighted metrics + bootstrap 95% CIs |
| `person_classifier_holdout.py` | Train/holdout sanity check on human labels |
| `person_apply_flips.py` | Apply classifier flips to `has_person` in the DB |
| `data/person_classifier.joblib` | Trained model: `{coef, intercept, trained_at, n_pos, n_neg}` |
| `data/label_sample/batch_*.yaml` | Hand-labeled samples |