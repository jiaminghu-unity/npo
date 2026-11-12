#!/usr/bin/env python3
"""Print the result tables of results/README.md from the files in this folder.

Usage (from the repo root):  python results/make_tables.py
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
M = ["ps", "hps", "clip", "aes", "image_reward", "hpsv3"]
HEAD = "PickScore | HPS (v2) | CLIP | Aesthetics | ImageReward | HPSv3"


def tables(summaries, order=None):
    rows = {}
    for path in summaries:
        with open(path) as f:
            rows.update(json.load(f))
    keys = sorted(rows, key=order) if order else list(rows)
    print(f"Average score, mean ± std over 5 seeds\n\n| model / step | {HEAD} |\n|---|" + "---|" * len(M))
    for k in keys:
        print(f"| {k} | " + " | ".join(f"{rows[k]['avg'][m]['mean']:.4f} ± {rows[k]['avg'][m]['std']:.4f}" for m in M) + " |")
    print(f"\nWin rate against the original model (%), mean ± std over 5 seeds\n\n| model / step | {HEAD} |\n|---|" + "---|" * len(M))
    for k in keys:
        w = rows[k]["win_rate_vs_base"]
        if w[M[0]] is None:
            continue
        print(f"| {k} | " + " | ".join(f"{100 * w[m]['mean']:.1f} ± {100 * w[m]['std']:.1f}" for m in M) + " |")


def sana_order(k):
    name, step = k.split("/")
    return (["Orig._Sana", "DPO", "NPO-randjudge", "NPO-avgrank"].index(name), 0 if step == "base" else int(step))


print("### Sana 0.6B (eval_sana.py)\n")
tables(sorted(glob.glob(os.path.join(HERE, "sana", "tests", "*", "summary.json"))), sana_order)
print("\n### SD1.5 baselines (original eval.py; `none` = the official checkpoint)\n")
tables([os.path.join(HERE, "sd15", "eval_py_5seeds", "summary.json")])

print("\n### Training-time eval gain (256 PartiPrompts, mean over the five reward models of the "
      "standardized improvement over step 0)\n")
runs = [("NPO random judge", "sana-newnash-randjudge-v1"), ("NPO avg rank", "sana-newnash-avgrank-v1"),
        ("offline DPO", "sana-dpo-offline-v1")]
gains = {}
for label, run in runs:
    with open(os.path.join(HERE, "sana", "training", run, "eval.jsonl")) as f:
        gains[label] = {r["step"]: r["gain"] for r in map(json.loads, filter(str.strip, f))}
steps = sorted(s for s in set().union(*[set(g) for g in gains.values()]) if s and s % 40 == 0 or s in (270, 420))
print("| step | " + " | ".join(label for label, _ in runs) + " |\n|---|" + "---|" * len(runs))
for s in steps:
    print(f"| {s} | " + " | ".join(f"{gains[label][s]:+.4f}" if s in gains[label] else "–" for label, _ in runs) + " |")
print("\nLast evaluated step: " + ", ".join(f"{label} {max(gains[label])}" for label, _ in runs))
