#!/usr/bin/env python3
"""Put the per-prompt CSVs written by the original eval.py into the results tree used by
eval_hpsv3.py / eval_summarize.py.

  <src>/<model>/<epoch>/seed_<s>/scores_pickapic.csv   (written by eval.py)
    -> <dst>/<model>/<epoch>/seed_<s>/scores_evalpy.csv    verbatim copy of eval.py's file
    -> <dst>/<model>/<epoch>/seed_<s>/scores_partial.csv   same numbers, column 'hps' named 'hps_v2'
"""
import csv
import os
import shutil
import sys

src, dst = sys.argv[1], sys.argv[2]
COLS = ["prompt_id", "prompt", "model_name", "epoch", "seed", "guidance_scale", "num_inference_steps"]
METRICS = [("ps", "ps"), ("hps", "hps_v2"), ("clip", "clip"), ("aes", "aes"), ("image_reward", "image_reward")]


def num(x):
    x = x.strip()
    if x.startswith("tensor("):
        x = x[7:].split(",")[0].rstrip(")")
    return float(x)


n = 0
for root, _, files in os.walk(src):
    if "scores_pickapic.csv" not in files:
        continue
    rel = os.path.relpath(root, src)
    out = os.path.join(dst, rel)
    os.makedirs(out, exist_ok=True)
    shutil.copy(os.path.join(root, "scores_pickapic.csv"), os.path.join(out, "scores_evalpy.csv"))
    with open(os.path.join(root, "scores_pickapic.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    with open(os.path.join(out, "scores_partial.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS + [b for _, b in METRICS])
        for r in rows:
            w.writerow([r[c] for c in COLS] + [num(r[a]) for a, _ in METRICS])
    print(f"{rel}: {len(rows)} rows, {len({r['prompt_id'] for r in rows})} distinct prompts")
    n += 1
if n == 0:
    print("no scores_pickapic.csv found")
    sys.exit(1)
