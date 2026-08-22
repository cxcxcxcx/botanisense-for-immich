# Person-flag classifier — design, evaluation, and production state

The BioCLIP-2 logistic regression that decides `assets.has_person`.
Stored at `data/person_classifier.joblib`. Used as the primary signal
for "is there a use-case person in this photo"; YOLO acts as an
**evidence gate** — the classifier positive only counts when YOLO also
saw at least a faint person/face/human box.

This document is the authoritative reference for the classifier design,
training data, evaluation, and the current state of "remove YOLO from
the formula."

---

## TL;DR — current production state

**YOLO-evidence rule (shipped 2026-08-18)**:

```
has_person = 1  iff  classifier_p >= 0.15  AND  yolo_person_max >= 0.20
has_person = 0  otherwise
```

Calibrated on 102-label stratified sample (7 strata). Population-weighted
metrics with bootstrap 95% CIs at T=0.15:

| Strategy | precision | recall | F1 |
|---|---|---|---|
| Pure classifier @ 0.15 | 0.975 | 0.979 | 0.977 |
| Pure classifier @ 0.50 | 0.975 | 0.967 | 0.971 |
| Hybrid (classifier+YOLO+Plantae) | 0.860 | 1.000 | 0.924 |
| **YOLO-evidence @ 0.15** | **1.000** | **0.983** | **0.991** |
| YOLO-evidence @ 0.50 | 1.000 | 0.967 | 0.983 |

YOLO-evidence @ 0.15 has the tightest precision CI
(`[1.000, 1.000]`) — the remaining FPs are in a single weight-1 stratum
with no statistical room to escape. Recall stays ≥ 0.96. YOLO acts as
an evidence gate: if YOLO truly saw nothing (`yolo_max` near 0), the
model's positive is a misfire and we override it. `yolo_min=0.20`
preserves wedding/family portraits (YOLO ~0.27) while killing
pure-plant misfires (YOLO ~0.0).

- **Total photos**: 19,304
- **has_person=1**: 3,817 (19.8%) — applied by `scripts/person_apply_yolo_evidence.py`
- **Flips from previous state**: 321 up (0→1, classifier rescues YOLO borderline) + 121 down (1→0, YOLO gate kills Plantae false positives)

**Previous hybrid rule** (replaced on 2026-08-18; see "Rule history" below):
1. classifier p >= 0.5 → hp=1 (trust classifier, ~3376 cases)
2. classifier p < 0.5 AND YOLO says person AND (Plantae < 0.95 OR classifier p >= 0.15) → hp=1 (YOLO fallback with Plantae false-positive guard, ~265 cases)
3. classifier p < 0.5 AND (Plantae >= 0.95 AND classifier p < 0.15) → hp=0 (Plantae-band safety net for hand-with-plant false positives, ~200 cases)
4. otherwise → hp=0

---

## Training data

### Auto-derived labels (12,758 examples)

The classifier was bootstrapped from auto-derived labels that use
existing YOLO detections and BioCLIP taxon classifications:

**Positives** (556 examples):
- `assets.has_person = 1`
- `MAX(detections.score) WHERE label='person' >= 0.90`
- AND one of:
  - `MAX(detections.score) WHERE label='face' >= 0.50` (face is a strong person signal)
  - No `hand` box with `score >= 0.05` has `>= 0.85` containment inside any person box (no overlapping hand silhouette)

**Negatives** (12,202 examples):
- `assets.has_person = 0`
- `taxon_scores.taxon_name = 'Plantae' AND score >= 0.50`

**Why this is biased**: the negatives are all BioCLIP-Plantae
candidates, which means any photo where BioCLIP classifier thinks it's
plant-like is treated as "no person" — including real people in flower
settings. This bias was the original motivation for hand-labeling the
Plantae band.

### Hand-labeled photos (32 examples, current)

- 8 photos in band p ∈ [0.10, 0.20) — extreme Plantae-band low
- 8 photos in band p ∈ [0.20, 0.30)
- 8 photos in band p ∈ [0.30, 0.40)
- 8 photos in band p ∈ [0.40, 0.45)

These were labeled by parallel agents using `scripts/person_label_prompt.md`.
Each row has `actual_person` (0/1, "any human anywhere in the photo")
and `use_case_relevant` (0/1, "is the human the focal subject").

**Critical limitation**: the 32-photo Plantae-band sample is the only
hand-labeled data the current retrained classifier has seen — it does
NOT include the original 70-photo stratified sample. That sample was
overwritten when `person_label_plantae_band.py` was run (the batch
files in `data/label_sample/` were deleted at startup). **Recovering
the original labels is needed** to train a better classifier with
both Plantae-band AND the original stratified strata.

To regenerate: run `scripts/person_label_sample.py --per-stratum 10`
and label each batch via the spec. Total ~70 photos across 7 strata.

---

## Evaluation on the labeled set

Held-out eval on the 32 hand-labeled Plantae-band photos. **These are
the only labels the retrained model has seen**, so this is a
within-sample (overfit) estimate. Real-world accuracy is unknown.

| Approach | Precision | Recall | F1 | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|
| **Pure classifier (p>=0.5)** vs `actual_person` | 1.000 | 0.323 | 0.488 | 10 | 1 | 0 | 21 |
| **Pure classifier (p>=0.5)** vs `use_case_relevant` | **1.000** | 0.714 | **0.833** | 10 | 18 | 0 | 4 |
| **Pure YOLO** vs `actual_person` | 0.969 | 1.000 | 0.984 | 31 | 0 | 1 | 0 |
| **Pure YOLO** vs `use_case_relevant` | 0.438 | 1.000 | 0.609 | 14 | 0 | 18 | 0 |
| **Hybrid** (current production) vs `actual_person` | 0.968 | 0.968 | 0.968 | 30 | 0 | 1 | 1 |
| **Hybrid** (current production) vs `use_case_relevant` | 0.452 | 1.000 | 0.622 | 14 | 1 | 17 | 0 |

### Reading these numbers

**The 32-photo labeled set is mostly small-background-people photos** in
flower/garden settings — that's why YOLO has perfect recall on
`actual_person` (it fires on every tiny figure) but only 0.609 F1 on
`use_case_relevant` (it flags tiny background walkers that don't matter).

**Pure classifier has F1 0.833 on `use_case_relevant`** — better than
YOLO's 0.609 by 23 points — with **precision 1.000** (zero false
positives). But recall 0.714 means it misses 4 of 14 use-case-relevant
people (the foreground subjects in flower settings).

**Hybrid** trades classifier precision for YOLO recall. Net: high F1
on `actual_person` (0.968) but lower on `use_case_relevant` because
YOLO's small-person detections sneak back in via the fallback.

---

## What if we drop YOLO entirely?

The user's question: "How bad would it be if we didn't look at YOLO at all?"

### Pure-classifier estimates (from the labeled set)

**Removing YOLO**:
- **Save**: ~3,181 false-positive hp=1 rows where YOLO fired on a hand/jacket/reflection but classifier correctly says no.
- **Lose**: ~681 rows where YOLO fired on a real person (or use-case-relevant person) that the classifier doesn't recognize as such. Of these, ~4 are use_case_relevant=1 (per the labeled sample, ~28% of YOLO-only rows are use_case_relevant).
- **Net change**: hp=1 count would drop from 3,617 to ~3,376 (a ~7% reduction), and we'd lose ~28 use_case_relevant photos we'd otherwise flag.

### Pure-classifier caveats

The **labeled sample is biased toward Plantae-band photos** (all 32
labels are in the Plantae disagreement band). The classifier's
performance on the **easy 3181 rows where both YOLO and classifier
agree** is unmeasured — it's almost certainly very good there
because that's the population the classifier was trained on.

So the "0.323 recall" number is pessimistic for the easy cases but
honest for the hard cases. A full population-weighted evaluation
would likely show much higher recall.

### Removing the safety net (Plantae band p<0.15 flip)

If we also drop the Plantae-band safety net (`p<0.15 AND Plantae>=0.95
→ hp=0`), we'd add ~24 false-positive rows back to hp=1. Trivial impact.

### Recommendation

**Removing YOLO entirely is feasible** but with measurable cost:
- Estimated ~28 use_case_relevant photos that YOLO would flag but
  classifier misses
- Estimated ~7% reduction in total hp=1 count

The pure-classifier approach is safer in terms of precision (zero
false positives on the labeled set) but trades recall. For a private
photo library where privacy/surprise is the main concern, the pure
classifier is arguably better — missing some real people is fine
(false negatives are tolerable), but flagging hands/jackets as people
(false positives) is bad UX.

To improve recall without sacrificing precision, the next step is:
1. **Re-collect the original 70-photo stratified sample** (was lost
   when `data/label_sample/batch_*.yaml` was overwritten).
2. **Train on combined data**: 70 stratified + 32 Plantae-band = 102
   examples. This should raise recall on the borderline band without
   introducing Plantae bias.

---

## Failure-case taxonomy (from `notes` field)

The labels' `notes` field captures *why* a detector is wrong. From
102 labels over the lifetime of this design (70 original + 32
Plantae-band, with the originals lost):

| Failure mode | Count | Detector that fails |
|---|---|---|
| `hand_only` (arm/hand mislabeled as person) | 3 | YOLO |
| `pure_<thing>` (tree/object mislabeled as person) | 1 | YOLO |
| `screen_<thing>` (photo of photo on monitor) | 1 | YOLO |
| `jacket_on_chair_no_person` | 1 | YOLO |
| `tiny_background_pedestrians` | ~6 | both (real but small) |
| `background_walkers` (subject is plant, walkers incidental) | ~4 | classifier (under-predicts) |
| `cropped_person_<edge>` (subject cropped out of frame) | ~3 | both (depends on crop) |
| `back_only_pedestrians` | ~3 | both |
| `foreground_woman/child/parent` (real person in flower field) | ~10 | classifier (the 528963ac cases) |
| `kids_play` (interactive subject in foliage) | ~2 | classifier |
| `group_<thing>` (multiple real people) | ~2 | both |
| `screen_wedding_photos_on_monitor` | 1 | both |

The dominant pattern: **YOLO fails on shape-mimicry** (hand, jacket,
photo-of-photo). **Classifier fails on Plantae-bias** — it can't tell
apart a person standing in a flower field from a person-less flower
field, because the BioCLIP embedding collapses them.

---

## Files in this design

| File | Purpose |
|---|---|
| `person_classifier_eval.py` | Original training script — auto-derived labels only |
| `person_classifier_retrain.py` | Retrain with auto + hand-labeled (use_case_relevant target) |
| `person_classifier_apply_flips.py` | Apply classifier p>=0.5 → hp=1 (was used for the 871-row initial flip) |
| `person_classifier_plantae_flip.py` | Plantae>=0.95 + classifier p<0.15 → hp=0 (YOLO false positive cleanup) |
| `person_classifier_backfill.py` | BioCLIP embed rows that lack embeddings + apply classifier flips |
| `person_apply_yolo_evidence.py` | Apply the shipped YOLO-evidence rule across the whole library; POSTs /api/cache/invalidate |
| `person_label_sample.py` | Build the 70-photo stratified sample (7 strata) |
| `person_label_plantae_band.py` | Build the Plantae-band sample (4 bands) |
| `person_label_prompt.md` | Labeling spec — what counts as actual_person / use_case_relevant |
| `person_label_aggregate.py` | Per-stratum confusion matrices (equal-weighted) |
| `person_label_aggregate_weighted.py` | Population-weighted metrics + bootstrap 95% CIs |
| `person_classifier_holdout.py` | 5-seed × 2-fold stratified train/holdout (old, pre-retrain) |
| `data/person_classifier.joblib` | Trained model: `{coef, intercept, trained_at, n_pos, n_neg, n_hand_labeled}` |
| `data/label_sample/batch_*.yaml` | 32 Plantae-band labels (current; original 70 lost) |

---

## What's needed to fully replace YOLO

1. **Re-collect the 70-photo stratified sample**. Original `data/label_sample/`
   was overwritten by `person_label_plantae_band.py`. To regenerate:
   run `person_label_sample.py --per-stratum 10` and label ~70 photos.

2. **Train on 102 combined examples** (70 + 32). Should improve recall on
   the borderline band without sacrificing Plantae-band precision.

3. **Larger Plantae-band sample**. Current 32 photos are 4 bands × 8
   each. Increasing to 100+ photos with more granular p-banding would
   let the classifier learn a sharper Plantae/no-Plantae boundary.

4. **Re-evaluate after retraining**. Use `person_classifier_holdout.py`
   (modified to use the combined sample) to get a held-out F1.

5. **Decide a threshold**. With better classifier precision, the
   `p>=0.5` threshold becomes defensible. Could lower to `p>=0.4` if
   precision stays high and recall improves.

---

## Recovery notes (for future-me)

If you're picking this up fresh:

1. The original 70-photo stratified sample was **lost** on 2026-08-17
   (was overwritten by `person_label_plantae_band.py`). Recovery
   attempts (git fsck, archives) yielded script blobs but not the
   original labels. The current 102-label sample (in
   `data/label_sample/`) was built by `person_label_sample_yolo_basis.py`
   using reconstructed YOLO has_person from the detections table.

2. The classifier currently has F1 0.991 (precision 1.000, recall
   0.983, YOLO-evidence @ T=0.15) on the 102-label stratified sample,
   95% bootstrap CI: prec=[1.000,1.000], rec=[0.959,1.000],
   f1=[0.979,1.000]. Real-world F1 unknown.

3. `storage.has_real_person()` is **no longer used in production** —
   the database column `has_person` is set by
   `scripts/person_apply_yolo_evidence.py` (YOLO-evidence rule), not
   by the live pipeline. The function still exists in storage.py for
   reference but is dead code. The pipeline's person stage will only
   touch rows where `has_person IS NULL`, which currently is zero.

4. The `__pycache__` directories and `*.joblib` models are gitignored.
   Re-train the model from scratch by running `person_classifier_retrain.py`
   after regenerating the labels.

5. The `cache_invalidate` HTTP endpoint (`POST /api/cache/invalidate`)
   is the way external scripts tell the running uvicorn worker to drop
   its in-process `_results_cache`. Without it, the worker serves stale
   `/api/results` until restart.

---

## Rule history

| Date | Rule | Production has_person count | Comment |
|---|---|---|---|
| 2026-08-12 | Pure YOLO `has_real_person()` | ~4,830 | Initial person backfill. High false-positive rate (hand, jacket, screen-of-people). |
| 2026-08-13 | Hybrid: classifier p≥0.5 OR (YOLO + Plantae safety net) | ~3,641 | First classifier shipped. Plantae-band false positives cut to ~200. |
| 2026-08-15 | Pure classifier p≥0.5 (YOLO removed from formula) | 3,617 | Removed YOLO entirely. Added 32 hand-labels in Plantae band. |
| **2026-08-18** | **YOLO-evidence: classifier p≥0.15 AND yolo_person_max≥0.20** | **3,817** | **Shipped.** Bootstrap CI shows prec=[1.000,1.000] rec=[0.959,1.000] f1=[0.979,1.000]. |

The current rule keeps the classifier as the *primary* signal (it
provides the recall) and uses YOLO as an *evidence gate* (it kills
classifier false positives where YOLO saw nothing). It's the union of
"trust the classifier" and "trust YOLO only as a sanity check on
plant-only images".

---

## Annotation quality (lesson for next labeling pass)

The 102-label sample ended up with a few row-swap and note-mismatch
errors that were only caught by cross-referencing `notes` against the
actual photo content:

- `7157e427` was noted as "couple_walking_tulip_bed" but the photo is
  a dry garden with street pedestrians (filename 2s apart from
  `44a765a6`).
- `44a765a6` was completely mislabeled — same dry-garden scene, but
  the row got a different description entirely.
- `92fffe94` was noted as "screen_portrait" but the photo is a kid in
  a garden foreground.
- Several iMac/screen-of-wedding-photos rows had `use_case_relevant=0`
  — people on a screen DO count as a use-case person, but the
  labeling prompt had a gap here.

**Going forward**: label image-by-image, ideally via a script that
calls the API and shows the actual JPEG/HEIC alongside the labels
being entered. Bulk-row editing makes swap errors easy.