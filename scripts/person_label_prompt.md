# Person-flag labelling task — instructions for agent

You are labeling photos to evaluate a "has person" classifier. Each
photo has been pre-detected by YOLO and scored by a BioCLIP-embedding
classifier. Your job is to provide ground truth using these fields:

```yaml
- actual_person: 0 | 1        # is there any human in this photo at all?
  use_case_relevant: 0 | 1    # is the human relevant to our use cases? (see below)
  notes: "free-text short tag, e.g. 'hand_only', 'tiny_back_only', 'reflection_only', 'kid_play', 'group_photo', 'selfie'"
  sublabel: "optional fine-grain: 'foreground' | 'background' | 'cropped' | 'face' | 'back_only' | 'tiny'"
```

## Use case — what we ACTUALLY care about

We are building a private photo library tool. The "use case" for
marking a photo as `has_person` is one of:

- **Foreground subject**: a person who is the clear subject of the photo.
  They occupy a meaningful fraction of the frame and the photo was
  probably taken *of* them (or *with* them).
- **Identifiable / face visible**: a face (or near-face) is visible
  enough that the photo has privacy implications — recognisable, not
  pixelated, not a silhouette. We want to flag these so we don't
  accidentally expose them in a public demo.
- **Interactive activity**: a person is doing something that matters
  to the photo (riding a bike, holding the plant, posing with the
  dog, etc.).

If NONE of the above apply, set `use_case_relevant: 0`.

## Concrete examples (use these as your mental model)

| Scenario | actual_person | use_case_relevant | notes |
|---|---|---|---|
| A portrait of your child on a tricycle | 1 | 1 | "foreground_child" |
| Same photo, but child is in the corner, mostly off-frame | 1 | 0 | "cropped_corner" |
| Group photo at a dinner table | 1 | 1 | "group_photo" |
| A back-of-head passer-by on a street photo | 1 | 0 | "back_only_pedestrian" |
| Person in a window reflection, semi-transparent | 1 | 0 | "reflection_only" |
| A tiny person in the background, no face | 1 | 0 | "tiny_background" |
| Hand holding a leaf | 0 | 0 | "hand_only" (YOLO said person; actual=no person at all) |
| Pure plant photo, no person anywhere | 0 | 0 | "pure_plant" |
| Selfie | 1 | 1 | "selfie" |
| Child on a balance bike — clearly the subject | 1 | 1 | "kid_play_focal" |

## Decision order (do this in your head for each photo)

1. **Look at the photo.** What is actually in it?
2. **actual_person**: is there any human, even tiny / partial / back-of-head / in reflection? If yes → 1, else → 0.
3. **use_case_relevant**: would we want this photo flagged for "this has people we care about" purposes?
   - Big foreground person → 1
   - Visible face → 1
   - Person doing something interactive / pose / group → 1
   - Tiny background, back-only, partial silhouette, reflection → 0
   - Hand only, no other body part visible → 0
4. **notes**: one short snake_case tag for the failure mode (or success mode). Be specific.
5. **sublabel**: optional, but useful for quick grouping in the final report.

## What to do

1. Read your batch file: `data/label_sample/batch_NN.yaml`
2. For each row, view the thumb at `data/thumbs/<asset_id>.jpg`
3. Fill in the four fields: `actual_person`, `use_case_relevant`, `notes`, `sublabel`
4. **Write the file back**, keeping all other fields exactly as-is. Do NOT change `asset_id`, `filename`, `thumb_path`, `yolo_max`, `yolo_person`, `model_p`, `stratum`, or `i`.

## Output format

Each row after labelling should look like:

```yaml
- i: 0
  stratum: 1_yolo_only_disagree
  asset_id: 00000000-0000-0000-0000-000000000000
  filename: 2025-01-01 12.00.00.jpg
  thumb_path: data/thumbs/00000000-0000-0000-0000-000000000000.jpg
  yolo_max: 0.62
  yolo_person: 1
  model_p: 0.05
  actual_person: 1           # <-- YOUR ANSWER
  use_case_relevant: 0       # <-- YOUR ANSWER
  notes: "tiny_back_only"    # <-- YOUR ANSWER
  sublabel: "background"     # <-- YOUR ANSWER (optional)
```

Do not invent rows. Do not skip rows. If a thumb is missing or unreadable, write `actual_person: null` and `notes: "missing_thumb"` and continue.

Return a short summary: how many you labeled, how many were missing/unclear, anything that surprised you.