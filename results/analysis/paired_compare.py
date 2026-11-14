#!/usr/bin/env python3
"""Paired comparison of two tested checkpoints on the same prompts and seeds.

Both models were tested on the same 500 prompts with the same 5 seeds, so their per-prompt scores can be
compared directly instead of through their win rates against the original model. For every metric:
  - delta      : mean(score A - score B) over prompts (each prompt = mean over its 5 seeds);
  - 95% CI     : percentile bootstrap over the 500 prompts (10,000 resamples);
  - p          : two-sided sign-flip permutation test on the per-prompt differences (10,000 flips);
  - A wins     : share of the 2,500 prompt-seed pairs where A scores higher than B.

Usage (from the repo root):
  python results/analysis/paired_compare.py <label A> <folder A> <label B> <folder B>
where a folder is e.g. results/sana/tests/test1-20261112/NPO-avgrank/90 (it contains seed_0 ... seed_4).
Without arguments it prints every Online-DPO vs NPO avg-rank comparison at equal steps found in results/sana/tests.
"""
import csv
import glob
import os
import random
import sys

M = [("ps", "PickScore"), ("hps", "HPS (v2)"), ("clip", "CLIP"), ("aes", "Aesthetics"),
     ("image_reward", "ImageReward"), ("hpsv3", "HPSv3")]
HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(HERE, "..", "sana", "tests")


def load(folder):
    out = {}
    for s in range(5):
        with open(os.path.join(folder, f"seed_{s}", "scores_pickapic.csv"), newline="") as f:
            for r in csv.DictReader(f):
                out[(r["prompt_id"], s)] = {k: float(r[k]) for k, _ in M}
    return out


def compare(label_a, folder_a, label_b, folder_b, n_boot=10000, seed=0):
    a, b = load(folder_a), load(folder_b)
    keys = sorted(set(a) & set(b))
    prompts = sorted({k[0] for k in keys})
    rng = random.Random(seed)
    print(f"\n**{label_a} minus {label_b}** ({len(prompts)} prompts x 5 seeds)\n")
    print("| metric | " + label_a + " | " + label_b + " | delta | 95% CI | p | " + label_a + " wins |")
    print("|---|---|---|---|---|---|---|")
    for k, name in M:
        per_prompt = []
        for p in prompts:
            d = [a[(p, s)][k] - b[(p, s)][k] for s in range(5) if (p, s) in a and (p, s) in b]
            per_prompt.append(sum(d) / len(d))
        n = len(per_prompt)
        delta = sum(per_prompt) / n
        boots = sorted(sum(per_prompt[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
        lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
        extreme = sum(abs(sum(x if rng.random() < 0.5 else -x for x in per_prompt) / n) >= abs(delta)
                      for _ in range(n_boot))
        p_val = (extreme + 1) / (n_boot + 1)
        wins = sum(a[q][k] > b[q][k] for q in keys) / len(keys)
        mean_a = sum(a[q][k] for q in keys) / len(keys)
        mean_b = sum(b[q][k] for q in keys) / len(keys)
        p_txt = "<0.001" if p_val < 0.001 else f"{p_val:.3f}"
        print(f"| {name} | {mean_a:.4f} | {mean_b:.4f} | {delta:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p_txt} | {100 * wins:.1f}% |")


def main():
    if len(sys.argv) == 5:
        compare(*sys.argv[1:5])
        return
    npo = {os.path.basename(p): p for p in glob.glob(os.path.join(TESTS, "*", "NPO-avgrank", "*"))}
    odpo = {os.path.basename(p): p for p in glob.glob(os.path.join(TESTS, "*", "Online-DPO-avgrank", "*"))}
    for step in sorted(set(npo) & set(odpo), key=int):
        compare(f"NPO avg rank @{step}", npo[step], f"Online-DPO @{step}", odpo[step])


if __name__ == "__main__":
    main()
