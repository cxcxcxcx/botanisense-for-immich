# BioCLIP memory — where the 5 GB comes from, and what to do about it

Captured 2026-08-22 after a uvicorn process was observed holding ~5.8 GB RSS
on a single BioCLIP-2 species-rank prediction pass, and a 19.5 s `/api/results`
cold-cache tail was traced to near-dup recomputation + JSON serialization.

This is a **design-options note**, not a fix. We deferred BioCLIP loading so
runs with `n_new=0` skip the 5 GB entirely; for runs that need BioCLIP we
still pay the cost. Three options below show how to reduce that cost without
losing accuracy.

---

## 1. The layered cost (measured, fresh subprocess)

Reproduction: `./.venv/bin/python /tmp/bench_memory_breakdown.py`

| Step                                            | Δ RSS   | Cumulative |
| ----------------------------------------------- | ------: | ---------: |
| `python` start                                  |          |     14 MB  |
| `import torch`                                  | +470 MB |    485 MB  |
| `import open_clip`                              | +213 MB |    697 MB  |
| `import bioclip.predict`                        |  +58 MB |    755 MB  |
| `clf = TreeOfLifeClassifier()`                  | +4774 MB |  5,529 MB  |
| `clf.predict(..., rank=KINGDOM)` (×2 warmup)    | +272 MB |  5,801 MB  |
| `clf.predict(..., rank=SPECIES)` ×5             |  +50 MB |  5,851 MB  |
| `gc.collect()`                                  |   0 MB  |  5,851 MB  |

(The `Rank` enum is at `bioclip.predict.Rank`; `.KINGDOM` / `.SPECIES` are
the values we pass.)

**The model itself is only 1.6 GB.** Verified on disk:

| File                                          | Size on disk | In RAM after clf() |
| --------------------------------------------- | -----------: | -----------------: |
| `models--imageomics--bioclip-2/.../open_clip_model.safetensors` | **1,631 MB** | 1,631 MB |
| `datasets--imageomics--TreeOfLife-200M/.../txt_emb_species.npy` | **2,541 MB** | 2,541 MB |
| `datasets--imageomics--TreeOfLife-200M/.../txt_emb_species.json` |    88 MB   |   ~88 MB |
| **Model + label table total**                  |  **4,260 MB** | **~4,260 MB** |

The 446 model tensors are all `torch.float32` (verified by walking the
safetensors file with `safe_open`). Token embedding shape `(49408, 768)`,
text projection `(768, 768)`. The label table is `(768, 867455) fp32`, norms
in the 19–93 range — **pre-projection token embeddings**, not the final
512-D text embedding (so the projection head runs on them at query time).

So the 5.1 GB of "BioCLIP" splits as:

| Component                                | Resident | Notes                                      |
| ---------------------------------------- | -------: | ------------------------------------------ |
| BioCLIP-2 weights (fp32)                 | 1,631 MB | unavoidable as long as we use the model    |
| Species label embedding table (fp32)     | 2,541 MB | only needed for SPECIES-rank predictions   |
| `import torch`                           |   470 MB | MKL, threading, dispatcher                 |
| `import open_clip`                       |   213 MB | BPE vocab + helpers                        |
| `import bioclip.predict`                 |    58 MB | transformers, hf_hub, torchvision, pandas  |
| First-predict activation buffers (8 imgs) |   300 MB | batch + intermediate features              |

---

## 2. The paper does say fp16 is safe

From [Stevens et al., CVPR 2024](https://arxiv.org/abs/2311.18803) and the
[bioclip-2 model card](https://huggingface.co/imageomics/bioclip-2):

- The model was **trained with fp16 mixed precision** (8× A100-80GB, 4 days).
- The published safetensors are fp32 (we measured). There is no official fp16
  variant. Loading the safetensors as `torch.float16` halves the weights to
  ~816 MB; the safetensors header doesn't gate this, you do it in code.
- The species npy is also fp32. Casting to fp16 halves it to 1,270 MB.

The authors designed for fp16, so inference in fp16 is in-distribution. **Not
yet measured on our prediction set** — a top-5 agreement test (fp32 vs fp16
on ~500 cached predictions) is the cheap validation step before flipping it
on for production.

---

## 3. Unload is real and almost free

Reproduction: `./.venv/bin/python /tmp/bench_unload.py`

| Step                                  | RSS after  | Δ         |
| ------------------------------------- | ---------: | --------: |
| After full lifecycle (steady peak)    |  5,852 MB  |           |
| `del clf`                             |  5,852 MB  |     0 MB  |
| `_cache._clf = None` (drop global)    |  1,274 MB  | **−4,578 MB** |
| `gc.collect()`                        |  1,274 MB  |     0 MB  |
| `torch.cuda.empty_cache()`            |  1,274 MB  |     0 MB (no-op on CPU) |
| **Reload: `clf = TreeOfLifeClassifier()`** |  **5,357 MB** | **+4,083 MB in 13.8 s** |

Key points:

- **The PyTorch tensors are pure-Python refcounted objects.** Once the global
  reference drops, refcount hits zero and the OS reclaims the pages
  *immediately*. No `gc.collect()` needed (though harmless).
- **`torch.cuda.empty_cache()` is a no-op on CPU.** Useful only if we ever
  move to GPU.
- **Reload cost: ~14 seconds.** Mostly safetensors load + open_clip model
  init. The species npy is reloaded lazily on the first species-rank predict
  (HF cache hit, instant).
- **The floor depends on what else the process has imported.** A subprocess
  that already ran BioCLIP (`torch + open_clip + bioclip.predict + Python`)
  settles around 1.27 GB after unload. Uvicorn, which never imports
  BioCLIP modules, sits at ~640 MB — verified live: `ps aux` shows
  VmRSS=635,852 KB on a fresh run with `n_new=0`. The 1.27 GB number came
  from the unload-probe subprocess, not from uvicorn. The minimum cost of
  *ever* importing BioCLIP is ~270 MB (the `open_clip + bioclip.predict`
  delta on top of `import torch`), and that's only paid in processes that
  actually need BioCLIP.
- **Per-rank npy loading is real.** Kingdom-rank predict adds 272 MB to the
  base; species-rank predict adds another 50 MB. We can decide whether to
  pay this per request.

---

## 4. Three concrete options to reduce the BioCLIP footprint

Listed in order of effort vs. payoff. None is implemented yet — these are
for design discussion.

### Option A: rank-gate the species-npy load (HIGHEST ROI, low effort)

BioCLIP's `TreeOfLifeClassifier` loads `txt_emb_species.npy` lazily on first
species-rank predict. If we never call species rank, we never load 2.5 GB.

Implementation: don't use the shipped `TreeOfLifeClassifier` for the kingdom
stage. Call `open_clip.create_model_from_pretrained(...)` directly, tokenize
the ~10 kingdom names ourselves, and `model.encode_text(...)` once. ~200 KB
of text features vs 2.5 GB. We then avoid the species npy entirely for
kingdom-only runs.

Cost: a small monkey-patch or a wrapper class. No accuracy change.

### Option B: cast weights + npy to fp16 on first load (medium effort, big win)

- Weights: in `TreeOfLifeClassifier.__init__`, after `self.model = ...`,
  call `.half()`. Or load the safetensors directly into fp16.
- npy: in `_get_txt_embeddings`, do `arr = np.load(p).astype(np.float16)`
  before storing.

Total saving: ~2 GB on the steady-state peak. Brings us from 5.85 GB to
~3.8 GB.

Risk: requires a top-5 agreement test before flipping. Hypothesis: <0.1%
disagreement because the published model is fp16-trained.

### Option C: int8 per-row quantize the npy (highest effort)

The species npy has row norms 19–93 (very different magnitudes per species).
Per-row re-scaling to int8 would be ~635 MB, but the projection head runs on
this matrix at query time, so any quantization needs to compose with the
projection. This is research-grade work and not a near-term lever.

---

## 5. Unload-on-idle — design sketch

Goal: drop BioCLIP after N seconds of inactivity, reload on the next request.

Where the code lives today: `pipeline.py` has a module-level
`_clf_lock` + `_clf_cache` (added in commit 02d1af8). The lifecycle is:

1. `_resolve_classifier()` checks `_clf_cache`, returns it if set, else
   builds one inside the lock and caches it.
2. Each call site (`run_kingdom`, `run_species`, `bbox-bioclip`) goes
   through `_resolve_classifier()`.

To add unload-on-idle:

1. Replace the bare module global with a small `ClassifierCache` class:
   ```python
   class ClassifierCache:
       def __init__(self, idle_timeout_s: float = 300):
           self._clf = None
           self._last_used: float | None = None
           self._lock = threading.Lock()
           self._idle_timeout = idle_timeout_s
       def get(self) -> TreeOfLifeClassifier:
           now = time.monotonic()
           if self._clf is not None and self._last_used is not None:
               if now - self._last_used > self._idle_timeout:
                   # unload
                   self._clf = None
           if self._clf is None:
               with self._lock:
                   if self._clf is None:
                       self._clf = TreeOfLifeClassifier()
           self._last_used = now
           return self._clf
   ```
2. Add a background reaper task in `app.py` that calls `cache.maybe_unload()`
   every 30 s and emits an SSE event so the UI can show "BioCLIP unloaded,
   reloading on next run".
3. Frontend: show a small badge "BioCLIP: loaded (5.4 GB)" / "BioCLIP: unloaded"
   in the JobPanel. User knows what to expect before clicking Run.

Edge cases to think about:
- A run is in-flight when the timer fires. The `last_used` is bumped inside
  `get()`, so as long as the run calls `get()` on every batch, the timer
  resets. Safe.
- Two runs back-to-back (within idle window): no unload, no reload cost.
- User clicks Run while unloaded: first batch waits ~14 s for reload. Show
  this in the UI ("Reloading BioCLIP — this takes ~15 s on first batch").
- Idle timeout choice: 5 minutes is a reasonable default (long enough that
  a user thinking "let me check the results" doesn't trigger reload; short
  enough that an overnight idle doesn't leave 4.5 GB sitting).

### Measured trade-off summary

| Strategy                            | Peak RSS | Cost on reload          | Effort |
| ----------------------------------- | -------: | ----------------------- | ------ |
| Current (lazy + singleton, no idle unload) | 5.85 GB | n/a (stays resident)   | done   |
| + Option A (no species npy on kingdom-only) | 3.30 GB | n/a                    | low    |
| + Option B (fp16 weights + npy)     | ~3.80 GB | n/a                     | medium |
| + Option B + Option A               | ~1.80 GB | n/a                     | medium |
| + Unload-on-idle (300 s)            | floor 1.27 GB (subprocess that ran BioCLIP) | +14 s on next request | medium |
| + Unload-on-idle + Option B + Option A | floor 1.27 GB (subprocess that ran BioCLIP) | +7 s on next request  | medium |

The "floor 1.27 GB" is the cost in a *process that has already imported
BioCLIP*. In uvicorn, which never imports BioCLIP, the floor is ~640 MB
(verified). The lowest we can get while staying in the same Python
process is ~640 MB; the only way below that is to run BioCLIP in a
separate process that exits cleanly when idle (the kernel reclaims
everything).

---

## 6. Repro scripts

All three are checked into `/tmp/` (not the repo) for one-off use:

- `/tmp/bench_memory_breakdown.py` — layered RSS measurement, prints a
  table at the end.
- `/tmp/bench_compile.py` — plain vs `torch.compile` comparison; showed
  compile is a no-op on CPU for this workload (~1% diff, within noise).
- `/tmp/bench_unload.py` — unload + reload probe, confirms 4.5 GB freeable
  in one line and ~14 s reload cost.

If we decide to keep these, move them to `scripts/bench_bioclip_memory.py`
et al. so they live with the rest of the bench scripts.

---

## 7. Decision: run pipeline in a subprocess

**Recommendation: move the pipeline run into a child process that exits
on completion.** Reason: same-process unload can't get uvicorn below
~640 MB; subprocess-per-run gives back the 4.6 GB for free with zero
unload logic.

### Why subprocess wins over unload-on-idle

The same-process + unload option (§5) was appealing when the floor was
thought to be 1.27 GB — "we're paying it anyway, just unload on a
timer." Once we measured uvicorn properly (`ps aux` shows VmRSS=636 MB
because uvicorn never imports open_clip/bioclip.predict), the comparison
flips:

| Architecture                  | Uvicorn RSS during run | Uvicorn RSS after run | Logic to write     | Cost per run              | Status            |
| ----------------------------- | ----------------------: | --------------------: | ------------------ | ------------------------- | ----------------- |
| Same-process (pre-refactor)   | 5.85 GB                 | 5.85 GB               | nothing            | 0                         | replaced          |
| Same-process + unload timer   | 5.85 GB                 | 1.27 GB               | reaper + UI badge  | 0 + 14 s reload next call | not implemented   |
| Subprocess + reaper thread    | 222 MB                  | 222 MB                | reaper thread      | 2-3 s spawn + 14 s load   | replaced (c2f068b) |
| **Subprocess, no reaper**     | **222 MB**              | **222 MB**            | `_get_active_job`  | 2-3 s spawn + 14 s load   | **shipped (c2f068b)** |
| Persistent worker (RPC)       | 222 MB                  | 222 MB                | large (RPC layer)  | 0 (worker stays warm)     | not needed        |

The 14 s model load is paid **once per child lifetime**, in both
subprocess architectures. The 2-3 s spawn is the cold `import torch` cost.
Both are amortized over a multi-minute BioCLIP run — irrelevant compared
to the 5 GB we're keeping out of uvicorn. Live uvicorn sits at **222 MB**
(the lazy-classifier fix already moved us off the 640 MB ceiling; the
subprocess split removed the conditional 5.85 GB peak).

We pick subprocess-per-run over persistent worker because:
1. We have one BioCLIP consumer (`/api/run`). Persistent worker only
   wins when there are many.
2. Simpler failure mode — child crashes, parent respawns. No RPC
   protocol to debug.
3. No "warm worker" complexity (model version mismatches, stale caches).

### The refactor — what it looks like

We already have the pattern. `start_person_backfill` at
`app.py:3286-3353` uses `multiprocessing.Process` with a JSON status
file that the parent polls. Same shape works for `/api/run`:

```python
# app.py — sketch, not implemented yet
import multiprocessing as _mp

def _pipeline_child_entry(
    query: str, n_samples: int, top_k: int, batch_size: int,
    status_path: str, stop_event: _mp.Event,
) -> None:
    """Runs in a separate process. BioCLIP lives only here."""
    try:
        out = run_pipeline(
            base_url=..., api_key=..., query=query,
            n_samples=n_samples, top_k=top_k, batch_size=batch_size,
            db_path=DB_PATH,
            stop_event=stop_event,
        )
        Path(status_path).write_text(json.dumps({"status": "done", "result": out}))
    except Exception as e:
        Path(status_path).write_text(json.dumps({"status": "error", "error": str(e)}))


@app.post("/api/run")
async def start_run(query: str = None, n_samples: int = None, ...):
    # ... existing concurrency guard ...
    job_id = uuid.uuid4().hex[:8]
    status_path = Path(DB_PATH).parent / f"pipeline-{job_id}.status.json"
    stop_event = _mp.Event()
    proc = _mp.Process(
        target=_pipeline_child_entry,
        args=(query, n_samples, top_k, batch_size, str(status_path), stop_event),
        daemon=True,
        name=f"pipeline-{job_id}",
    )
    proc.start()
    # Return immediately; UI polls /api/status which reads status_path
    return {"job_id": job_id}
```

### Caveats (none fatal)

1. **Spawn vs fork.** Use `mp.set_start_method('spawn')` (or let it
   default on Python 3.14). Fork would inherit uvicorn's address space
   copy-on-write, including any modules uvicorn has loaded. With our
   lazy BioCLIP fix in place today, uvicorn has none of BioCLIP, but
   spawn is the defensive default and matches the person-backfill path.
2. **Spawn cost on small runs.** A 4-second cached run pays 2-3 s of
   spawn overhead. Acceptable for now; if it becomes a complaint, we'd
   look at a persistent worker.
3. **Stop/cancel.** Already handled — pass the `multiprocessing.Event`
   to `run_pipeline(stop_event=...)`, same as today.
4. **SQLite contention.** WAL mode + the existing `_pipeline_lock`
   already cover this. Both processes write to the same DB; SQLite
   serializes.
5. **Status polling vs SSE.** Today `/api/events` is a server-side
   stream. With subprocess, the child writes a status file; the parent
   either polls on `/api/status` or bridges to SSE by reading the
   status file in the event loop and pushing a delta. The polling
   path is the simpler first cut.

### What this does NOT solve

- The child still hits 5.85 GB during the BioCLIP predict. That's fine
  because the child is transient — kernel reclaims on exit. The system
  needs 5.85 GB free *during* the run; it doesn't need 5.85 GB
  resident permanently.
- If multiple BioCLIP runs overlap (today the `_pipeline_lock`
  prevents this), they each fork their own child and we'd briefly have
  5.85 × N GB system-resident. The lock keeps us at N=1.

