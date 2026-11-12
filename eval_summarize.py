#!/usr/bin/env python
"""Merge the outputs of eval_sana.py and eval_hpsv3.py and print the tables of eval.py:
average of every metric per model and win rate against the base model, per seed and over seeds.

Reported metrics: the five of eval.py unchanged (ps, hps, clip, aes, image_reward; hps is the HPSv2
model of utils/hps_utils.py as before) plus one added column, hpsv3 (user request 2026-11-12).
Writes scores_pickapic.csv (per-prompt, same columns as eval.py plus hpsv3) in every result folder
and <result_dir>/summary.json + summary.md.
"""
import argparse
import csv
import json
import os
import statistics
import sys

REPORT = ["ps", "hps", "clip", "aes", "image_reward", "hpsv3"]
BASE_EPOCH = "base"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", default="eval_result_rebuttal")
    args = ap.parse_args()
    rd = os.path.abspath(args.result_dir)

    data = {}   # (model_dir, epoch) -> seed -> {prompt_id: row dict}
    for root, _, files in os.walk(rd):
        if "scores_partial.csv" not in files:
            continue
        model_dir, epoch, seed_dir = os.path.relpath(root, rd).split(os.sep)
        with open(os.path.join(root, "scores_partial.csv")) as f:
            rows = list(csv.DictReader(f))
        v3 = {}
        if os.path.exists(os.path.join(root, "hpsv3.csv")):
            with open(os.path.join(root, "hpsv3.csv")) as f:
                v3 = {r["prompt_id"]: float(r["hps_v3"]) for r in csv.DictReader(f)}
        merged = {}
        for r in rows:
            m = {"ps": float(r["ps"]), "hps": float(r["hps_v2"]), "clip": float(r["clip"]),
                 "aes": float(r["aes"]), "image_reward": float(r["image_reward"]),
                 "hpsv3": v3.get(r["prompt_id"], float("nan"))}
            merged[r["prompt_id"]] = (r, m)
        with open(os.path.join(root, "scores_pickapic.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["prompt_id", "prompt", "model_name", "epoch", "seed", "guidance_scale",
                        "num_inference_steps", "ps", "hps", "clip", "aes", "image_reward", "hpsv3"])
            for r, m in merged.values():
                w.writerow([r["prompt_id"], r["prompt"], r["model_name"], r["epoch"], r["seed"], r["guidance_scale"],
                            r["num_inference_steps"], m["ps"], m["hps"], m["clip"], m["aes"], m["image_reward"],
                            m["hpsv3"]])
        data.setdefault((model_dir, epoch), {})[seed_dir] = {k: v[1] for k, v in merged.items()}

    base_key = next((k for k in data if k[1] == BASE_EPOCH), None)
    if base_key is None:
        print("no base model results found")
        return 1
    summary = {}
    for key in sorted(data, key=lambda k: (k != base_key, k)):
        per_seed = {}
        for seed, rows in sorted(data[key].items()):
            base = data[base_key].get(seed, {})
            ids = [i for i in rows if i in base]
            avg = {m: sum(rows[i][m] for i in ids) / len(ids) for m in REPORT}
            win = None
            if key != base_key:   # tie rule of eval.py: strict except image_reward
                win = {m: sum((rows[i][m] >= base[i][m]) if m == "image_reward" else (rows[i][m] > base[i][m])
                              for i in ids) / len(ids) for m in REPORT}
            per_seed[seed] = {"n": len(ids), "avg": avg, "win_rate_vs_base": win}
        seeds = list(per_seed)

        def agg(field, m):
            vals = [per_seed[s][field][m] for s in seeds if per_seed[s][field] is not None]
            if not vals:
                return None
            return {"mean": statistics.mean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}
        summary["/".join(key)] = {"seeds": seeds, "per_seed": per_seed,
                                  "avg": {m: agg("avg", m) for m in REPORT},
                                  "win_rate_vs_base": {m: agg("win_rate_vs_base", m) for m in REPORT}}

    with open(os.path.join(rd, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    lines = [f"Metrics: {', '.join(REPORT)} (the five of eval.py, hps = HPSv2 as before, plus hpsv3 = HPSv3)", "",
             "Average score, mean ± std over seeds", "",
             "| model | seeds | " + " | ".join(REPORT) + " |", "|---|---|" + "---|" * len(REPORT)]
    for k, s in summary.items():
        lines.append(f"| {k} | {len(s['seeds'])} | " + " | ".join(
            f"{s['avg'][m]['mean']:.4f} ± {s['avg'][m]['std']:.4f}" for m in REPORT) + " |")
    lines += ["", "Win rate against the base model (share of the prompts where the model scores higher), "
              "mean ± std over seeds", "", "| model | " + " | ".join(REPORT) + " |", "|---|" + "---|" * len(REPORT)]
    for k, s in summary.items():
        if s["win_rate_vs_base"][REPORT[0]] is None:
            continue
        lines.append(f"| {k} | " + " | ".join(
            f"{100 * s['win_rate_vs_base'][m]['mean']:.1f}% ± {100 * s['win_rate_vs_base'][m]['std']:.1f}" for m in REPORT) + " |")
    text = "\n".join(lines)
    with open(os.path.join(rd, "summary.md"), "w") as f:
        f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
