"""Generate an HTML page of model-vs-label disagreements for review.

Renders to frontend/dist/disagreements.html. Open the file directly,
or serve frontend/dist/ (the built SPA dir) and open
  http://<host>:<port>/disagreements.html

Shows every labeled photo as a card with:
  - thumb image
  - model_p, yolo_person, yolo_max, label, use_case_relevant
  - stratum, notes, sublabel
  - TP / FP / TN / FN badge based on chosen threshold

For each threshold T:
  - TP: model p >= T, label use_case_relevant = 1
  - TN: model p < T,  label use_case_relevant = 0
  - FP: model p >= T, label use_case_relevant = 0  <-- WRONG
  - FN: model p < T,  label use_case_relevant = 1  <-- MISSED

Usage:
  python scripts/person_label_vs_model_html.py --threshold 0.15
  python scripts/person_label_vs_model_html.py --threshold 0.50 --target actual_person
"""
from __future__ import annotations

import argparse
import html
import time
from pathlib import Path

import yaml

ROOT = Path("data/label_sample")
OUT = Path("frontend/dist/disagreements.html")


def load_labeled(target: str) -> list[dict]:
    rows = []
    for p in sorted(ROOT.glob("batch_*.yaml")):
        rows.extend(yaml.safe_load(p.read_text()) or [])
    return [r for r in rows if r.get(target) is not None]


def classify(r, threshold, target, yolo_min=0.0):
    p = r.get("model_p")
    ym = r.get("yolo_max") or 0.0
    truth = int(r[target])
    pred = 1 if (p is not None and p >= threshold and ym >= yolo_min) else 0
    if pred == 1 and truth == 1: return "TP"
    if pred == 0 and truth == 0: return "TN"
    if pred == 1 and truth == 0: return "FP"
    if pred == 0 and truth == 1: return "FN"
    return "??"


CSS = """
  body { font-family: ui-sans-serif, system-ui, sans-serif; background:#0d1117; color:#c9d1d9; margin:0; padding:24px; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .subtitle { color:#8b949e; font-size:12px; margin-bottom:16px; }
  .controls { position:sticky; top:0; background:#0d1117; padding:12px 0; border-bottom:1px solid #30363d; margin-bottom:16px; display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
  .controls label { font-size:13px; }
  .controls input[type=number] { background:#0d1117; color:#c9d1d9; border:1px solid #30363d; padding:4px 8px; width:80px; }
  .controls select { background:#0d1117; color:#c9d1d9; border:1px solid #30363d; padding:4px 8px; }
  .summary { display:flex; gap:12px; font-size:13px; flex-wrap:wrap; }
  .pill { padding:4px 10px; border-radius:999px; border:1px solid #30363d; }
  .pill.TP { color:#3fb950; border-color:#3fb95040; }
  .pill.TN { color:#58a6ff; border-color:#58a6ff40; }
  .pill.FP { color:#f85149; border-color:#f8514940; background:#f8514910; }
  .pill.FN { color:#d29922; border-color:#d2992240; background:#d2992210; }
  .grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap:14px; }
  .card { background:#161b22; border:1px solid #30363d; border-radius:6px; overflow:hidden; }
  .card.FP { border-color:#f85149; }
  .card.FN { border-color:#d29922; }
  .thumb { width:100%; aspect-ratio:1.3/1; object-fit:cover; background:#000; display:block; }
  .body { padding:8px 10px; font-size:12px; line-height:1.4; }
  .badge { display:inline-block; padding:2px 6px; border-radius:4px; font-weight:600; font-size:11px; margin-right:6px; }
  .badge.TP { background:#3fb95030; color:#3fb950; }
  .badge.TN { background:#58a6ff30; color:#58a6ff; }
  .badge.FP { background:#f8514930; color:#f85149; }
  .badge.FN { background:#d2992230; color:#d29922; }
  .meta { color:#8b949e; font-size:11px; margin-top:6px; word-break:break-all; }
  .truth-row { display:flex; gap:8px; margin-top:6px; font-size:11px; }
  .truth-row b { color:#c9d1d9; }
  .notes { color:#d29922; margin-top:4px; font-style:italic; }
  .stratum { color:#58a6ff; font-family: ui-monospace, monospace; font-size:10px; margin-top:4px; word-break:break-all; }
  .filter-btns { display:flex; gap:6px; }
  .filter-btns button { background:#0d1117; color:#c9d1d9; border:1px solid #30363d; padding:4px 10px; border-radius:4px; cursor:pointer; font-size:12px; }
  .filter-btns button.active { background:#1f6feb; color:#fff; border-color:#1f6feb; }
  .hidden { display:none !important; }
"""


def render_card(r, outcome, threshold, target):
    p = r.get("model_p")
    p_str = f"{p:.3f}" if p is not None else "n/a"
    yp = r.get("yolo_person")
    yp_str = str(int(yp)) if yp is not None else "n/a"
    ym = r.get("yolo_max")
    ym_str = f"{ym:.3f}" if ym is not None else "n/a"
    notes = r.get("notes") or ""
    sub = r.get("sublabel") or ""
    ucr = r.get("use_case_relevant")
    ucr_str = str(int(ucr)) if ucr is not None else "n/a"
    filename = html.escape(r.get("filename") or "")
    stratum = html.escape(r.get("stratum") or "")
    notes_esc = html.escape(notes)
    sub_esc = html.escape(sub)
    # Use server's /thumb endpoint so we don't have to expose data/thumbs.
    aid = html.escape(r["asset_id"])

    return f"""
    <div class="card {outcome}" data-outcome="{outcome}">
      <img class="thumb" src="/thumb/{aid}?size=preview" loading="lazy" alt="">
      <div class="body">
        <div><span class="badge {outcome}">{outcome}</span>
          <span style="color:#8b949e">T={threshold:.2f}</span></div>
        <div class="truth-row">
          <span><b>label</b> {target}={int(r[target])}</span>
          <span><b>ucr</b> {ucr_str}</span>
        </div>
        <div class="meta">
          <b>model_p</b> {p_str} &nbsp;
          <b>yolo_person</b> {yp_str} &nbsp;
          <b>yolo_max</b> {ym_str}
        </div>
        {f'<div class="notes">{notes_esc}</div>' if notes else ''}
        {f'<div class="meta">sublabel: {sub_esc}</div>' if sub else ''}
        <div class="stratum">{stratum}</div>
        <div class="meta">{aid}</div>
      </div>
    </div>
    """


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--yolo-min", type=float, default=0.0,
                    help="YOLO-evidence rule: positive requires yolo_person_max >= this. Default 0 = pure classifier.")
    ap.add_argument("--target", default="use_case_relevant",
                    choices=["actual_person", "use_case_relevant"])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rows = load_labeled(args.target)
    if not rows:
        raise SystemExit(f"No rows labeled with {args.target}")

    # Sort: FP first (most-confident wrong), then FN (most-regrettable misses),
    # then TP, then TN.
    def sort_key(r):
        out = classify(r, args.threshold, args.target, args.yolo_min)
        p = r.get("model_p") or 0.0
        if out == "FP":
            return (0, -p)        # most confident FP first
        if out == "FN":
            return (1, p)         # lowest p first (most confident FN)
        if out == "TP":
            return (2, -p)
        return (3, p)
    rows_sorted = sorted(rows, key=sort_key)

    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    cards_html = []
    for r in rows_sorted:
        out = classify(r, args.threshold, args.target, args.yolo_min)
        counts[out] += 1
        cards_html.append(render_card(r, out, args.threshold, args.target))

    n = len(rows_sorted)
    tp, tn, fp, fn = counts["TP"], counts["TN"], counts["FP"], counts["FN"]
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    body = f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<title>label vs model — disagreements (built {time.strftime('%Y-%m-%d %H:%M:%S')})</title>
<style>{CSS}</style></head>
<body>
<h1>label vs model — disagreements</h1>
<div class="subtitle">
  Threshold T = classifier_p ≥ T → predicted positive.
  Target = <b>{args.target}</b> (the human ground truth).
  Sample = {n} rows from data/label_sample/ (all strata).
  <br>
  <span style="color:#d29922">
    ⚠ Numbers below are <b>sample-weighted</b> (each of {n} rows counts once).
    Population-weighted PR curve is in scripts/person_pr_curve.py —
    stratum sizes differ 50× so weighted F1 can be 0.2 different.
  </span>
  {("<br><span style=\"color:#3fb950\">Rule: hp=1 iff model_p &gt;= " + f"{args.threshold}" + " AND yolo_person_max &gt;= " + f"{args.yolo_min}</span>") if args.yolo_min > 0 else ("<br><span style=\"color:#58a6ff\">Rule: pure classifier, model_p &gt;= " + f"{args.threshold}</span>")}
</div>
<div class="controls">
  <div class="summary">
    <span class="pill TP">TP {tp}</span>
    <span class="pill TN">TN {tn}</span>
    <span class="pill FP">FP {fp}</span>
    <span class="pill FN">FN {fn}</span>
    <span class="pill">precision {prec:.3f}</span>
    <span class="pill">recall {rec:.3f}</span>
    <span class="pill">F1 {f1:.3f}</span>
  </div>
  <div class="filter-btns">
    <button data-filter="ALL" class="active">All</button>
    <button data-filter="FP">FP only</button>
    <button data-filter="FN">FN only</button>
    <button data-filter="TP">TP only</button>
    <button data-filter="TN">TN only</button>
  </div>
</div>
<div class="grid" id="grid">
{''.join(cards_html)}
</div>
<script>
  const btns = document.querySelectorAll('.filter-btns button');
  btns.forEach(b => b.addEventListener('click', () => {{
    btns.forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    const f = b.dataset.filter;
    document.querySelectorAll('.card').forEach(c => {{
      c.classList.toggle('hidden', f !== 'ALL' && c.dataset.outcome !== f);
    }});
  }}));
</script>
</body></html>
"""
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(body)
    print(f"Wrote {args.out}  ({n} cards, FP={fp} FN={fn} prec={prec:.3f} rec={rec:.3f} F1={f1:.3f})")
    print(f"Open: http://<server>/{Path(args.out).name}?v={int(time.time())}")


if __name__ == "__main__":
    main()