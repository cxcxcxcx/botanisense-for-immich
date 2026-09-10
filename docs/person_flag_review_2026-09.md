# Person-flag review — box heuristics vs BioCLIP classifier (2026-09)

Persisted so the visual verdicts below never need to be re-derived. This
review decided the **classifier veto** (`pipeline.PERSON_VETO_P = 0.05`,
applied by `pipeline.apply_person_veto` at every `has_real_person` stamp
site and by `scripts/person_apply_clf_veto.py` across existing rows).

## Background

Two `has_person` signals exist and fail on **different, complementary
populations**:

- **Box heuristics** (`storage.has_real_person()`: person box ≥ 0.5 +
  hand-area/containment suppression + YuNet face rescue) — wrong on
  hand-holding-X photos: YOLO-World reads the arm + object as one big
  `person` box and never emits `hand` (same shape-mimicry failure as its
  dead `face` prompt). A plant library is full of these.
- **BioCLIP whole-image classifier** (`data/person_classifier.joblib`,
  logreg over the persisted embedding) — right on hand-holding-X, wrong
  on person-in-garden scenes (its Plantae-bias FN, documented in
  `scripts/PERSON_CLASSIFIER.md`).

The classifier veto combines them: the box stack keeps the 0→1
direction (existence + rescue); a confident classifier negative
(p < 0.05) vetoes it. Missing embedding or missing model file ⇒ no veto.

## Method

2026-09-10, fleet state after the 2026-09-09 face-rescue backfill
(4,778 hp=1). For all 4,847 assets with a person/human box ≥ 0.5, both
rules were computed: 86.8% agreement; 616 disagreements (current=1 /
classifier-rule=0, of which 350 have no embedding) and 24 (current=0 /
classifier-rule=1). Random samples were fetched as thumbnail contact
sheets and judged by eye:

- Sheet A — 12 of the 266 embedded "current=1, classifier=0" with p < 0.05
- Sheet B — 12 of the same set with p ∈ [0.05, 0.15)
- Sheet C — 12 of the 24 "current=0, classifier=1"
- Sheet D — 6 of the 350 no-embedding "current=1, classifier=0"

## Verdicts

### Sheet A — p < 0.05, box rule says person, classifier says no

**12/12 classifier right.** Every one is hand-only or person-less:

| asset | p | content |
|---|---|---|
| 60e26b73… | 0.014 | potted plant, no person |
| 4b0b0cf5… | 0.021 | fence |
| fc7b3f62… | 0.028 | hand reaching for a plant |
| ca04dac3… | 0.019 | hand holding a cucumber |
| c2403bbe… | 0.037 | garden bushes |
| f8222b74… | 0.034 | hand pointing at a seedling |
| 3f6a866f… | 0.007 | hand holding a maple leaf |
| e3b995e7… | 0.015 | street, no visible person |
| c62a7628… | 0.019 | garden bed, distant incidental figure |
| 05dd4b1f… | 0.002 | nursery aisle, no person |
| 4236b2f7… | 0.044 | park grasses |
| 045fb496… | 0.043 | foot + hand at trail edge (body parts, no person) |

(`d7b693e9` — hand holding a succulent pot, YOLO person 0.754 — is the
same class, p = 0.014.)

### Sheet B — p ∈ [0.05, 0.15), same disagreement direction

**~6/12 real people** (classifier wrong — its Plantae bias), ~6/12
hand-only/screenshot/reflection (classifier right):

- Real people (must keep hp=1): c84c609f p=0.061 + 32adc702 p=0.063
  (child in lupine field, ×2), ab65f47d p=0.078 (person with backpack
  under maple), 478e1cff p=0.055 (person by garden arch), 22a73612
  p=0.065 (people in sunflower field), 3b469d19 p=0.143 (two people on
  mosaic stairs), 2123b68e p=0.129 (stroller behind tulips).
- Correctly not-person: 06590a04 (webpage screenshot), 05199f9a
  (people reflected in TV), d8d21164 (hand + clematis), 7b8fbfaf
  (feet at planter edge), a32850b3 (tiny distant figure).

**Key separation**: every real person here has p ≥ 0.055; every
hand-only photo in sheet A has p ≤ 0.044. The veto threshold 0.05 sits
in that gap — sheet B rows keep hp=1 under the veto, and only the
p < 0.05 band is zeroed.

### Sheet C — current=0, classifier-rule=1 (24 rows, sampled 12)

Mixed — the classifier is NOT trustworthy in this band:

- Real people the box stack missed (the 24 include the YuNet-residual
  "no visible face" FNs): e0559d22 p=0.255 (girl's head with flower,
  top-down), 318ac7a9 p=0.971 (child walking away), 382cb346 p=0.441
  (bride from behind), ec4b57bd p=0.403 (hands playing in leaves —
  likely kids).
- Hand-only FPs at HIGH p (why classifier-primary was rejected):
  147bf4b2 p=0.641 (hands + mushrooms), 00e2284d p=0.643 (hands with
  ring over garden), 73906124 p=0.571 (hand with keychain), 088dfd72
  p=0.294 (hand holding password paper), 5380dc68 p=0.278 (hand holding
  a leek), f413cff9 p=0.181 (hand holding cup), 9794c01f p=0.366
  (screenshot of nails), 63bc49af p=0.742 (hand reaching into roses).

### Sheet D — current=1, no embedding (350 rows, sampled 6)

**~5/6 real people** (babies/toddlers in gardens — 05253fd6, 3a89f308,
49fbcebc, 7b441edb, 71b283c2; 54beccc2 is a hand at frame edge). The
no-embedding population is mostly genuinely-person photos, so the
fallback (no embedding ⇒ no veto) is correct, and a classifier-primary
rule with a "no embedding ⇒ trust YOLO" fallback would also have been
viable here — the veto wins on sheets A+B+C combined.

## Veto safety measurement

- All 19 YuNet-rescued rows (visually confirmed people, 2026-09-09):
  min p = 0.110 — **zero veto exposure on verified rescues**.
- Exactly 102 hp=1 rows have p < 0.05 → swept to 0 by
  `scripts/person_apply_clf_veto.py` (fleet: 4,778 → 4,676).

## Why not the alternatives

- **Pure box heuristics** (pre-review state): ~100+ hand-holding FPs
  like `d7b693e9` stayed in the with-people grid.
- **Classifier-primary** (the 2026-08-18 rule `p ≥ 0.15 AND yolo_max ≥
  0.20`, abandoned by the 2026-09-07 yolo-single-pass merge): sheet C
  shows it adds hand FPs at p 0.18–0.64, and sheet B shows it zeroes
  ~half of person-in-garden photos. Its published F1 0.991 predates
  both the 09-07 pipeline merge and the 09-09 rescue backfill.
- **Retraining with current-era labels** (hand-holding-X negatives +
  person-in-garden positives, ~150 images) remains the best end state
  if the veto's residual FPs bother us; the labels above are the seed
  set.
