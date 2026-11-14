#!/usr/bin/env python
"""Score the images saved by eval_sana.py with HPSv3 (run in the HPSv3 python environment).

For every <result_dir>/<model>/<epoch>/seed_<seed>/scores_partial.csv it scores the matching images
in <save_dir>/<model>/<epoch>/seed_<seed>/<prompt_id>.png and writes hpsv3.csv (prompt_id, hps_v3)
next to it. Folders that already have a complete hpsv3.csv are skipped, so the base model is only
scored once across tests.
"""
import argparse
import csv
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", default="eval_images")
    ap.add_argument("--result_dir", default="eval_result_rebuttal")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    save_dir, result_dir = os.path.abspath(args.save_dir), os.path.abspath(args.result_dir)

    todo = []
    for root, _, files in os.walk(result_dir):
        if "scores_partial.csv" not in files:
            continue
        with open(os.path.join(root, "scores_partial.csv")) as f:
            rows = list(csv.DictReader(f))
        out = os.path.join(root, "hpsv3.csv")
        if os.path.exists(out):
            with open(out) as f:
                if len(list(csv.DictReader(f))) == len(rows):
                    continue
        todo.append((root, rows, out))
    print(f"{len(todo)} folder(s) to score with HPSv3", flush=True)
    if not todo:
        return 0

    from hpsv3 import HPSv3RewardInferencer
    inferencer = HPSv3RewardInferencer(device=args.device)

    for root, rows, out in sorted(todo):
        rel = os.path.relpath(root, result_dir)
        paths = [os.path.join(save_dir, rel, f"{r['prompt_id']}.png") for r in rows]
        prompts = [r["prompt"] for r in rows]
        scores = []
        for s in range(0, len(rows), args.batch_size):
            with torch.no_grad():
                # note the argument order of the hpsv3 package: reward(image_paths, prompts)
                rewards = inferencer.reward(image_paths=paths[s:s + args.batch_size],
                                            prompts=prompts[s:s + args.batch_size])
            scores.extend(float(r[0].item()) for r in rewards)   # r = (mu, sigma); mu is the score
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["prompt_id", "hps_v3"])
            w.writerows([r["prompt_id"], s] for r, s in zip(rows, scores))
        print(f"{rel}: mean HPSv3 {sum(scores) / len(scores):.4f} ({len(scores)} images)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
