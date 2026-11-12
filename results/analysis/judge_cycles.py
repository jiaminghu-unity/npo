#!/usr/bin/env python3
"""How often do the five reward models (PickScore, HPSv2, CLIP, Aesthetics, ImageReward) form a
majority cycle among candidate images of the same prompt?

The random-judge preference used in training is P(a > b) = share of the five judges that score a
above b. It is intransitive whenever the majority relation has a cycle (a beats b, b beats c, c beats a).
Candidates here are the 5 seeds of one model for each of the 500 Pick-a-Pic test prompts, read from the
per-prompt scores of the tests in results/sana/tests/.

Usage (from the repo root):  python results/analysis/judge_cycles.py
"""
import csv
import itertools
import os
import random

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sana", "tests")
JUDGES = ["ps", "hps", "clip", "aes", "image_reward"]
SETS = [
    ("original Sana, 5 seeds", "test3-npo-step140/Orig._Sana/base"),
    ("NPO random judge step 130, 5 seeds", "test3-npo-step140/NPO-randjudge/130"),
    ("NPO avg rank step 140, 5 seeds", "test3-npo-step140/NPO-avgrank/140"),
]


def load(rel):
    out = {}
    for s in range(5):
        with open(os.path.join(ROOT, rel, f"seed_{s}", "scores_pickapic.csv"), newline="") as f:
            for r in csv.DictReader(f):
                out.setdefault(r["prompt_id"], []).append([float(r[j]) for j in JUDGES])
    return {k: v for k, v in out.items() if len(v) == 5}


def analyse(cands_by_prompt, label):
    n_tri = cyc_tri = prompts_cyc = no_winner = n_pairs = split = 0
    for cands in cands_by_prompt.values():
        n, k = len(cands), len(cands[0])

        def maj(i, j):   # a majority of the judges prefers candidate i to candidate j
            return sum(cands[i][q] > cands[j][q] for q in range(k)) * 2 > k

        for i, j in itertools.combinations(range(n), 2):
            v = sum(cands[i][q] > cands[j][q] for q in range(k))
            n_pairs += 1
            split += v not in (0, k)
        c = 0
        for i, j, l in itertools.combinations(range(n), 3):
            n_tri += 1
            c += (maj(i, j) and maj(j, l) and maj(l, i)) or (maj(j, i) and maj(l, j) and maj(i, l))
        cyc_tri += c
        prompts_cyc += c > 0
        no_winner += not any(all(maj(i, j) for j in range(n) if j != i) for i in range(n))
    p = len(cands_by_prompt)
    print(f"| {label} | {n} | {100 * split / n_pairs:.1f}% | {100 * cyc_tri / n_tri:.2f}% ({cyc_tri}/{n_tri}) | "
          f"{100 * prompts_cyc / p:.1f}% | {100 * no_winner / p:.1f}% |")


def main():
    print("| candidates | per prompt | pairs where the judges are not unanimous | cyclic triples | "
          "prompts with at least one cycle | prompts with no majority winner |")
    print("|---|---|---|---|---|---|")
    data = {label: load(rel) for label, rel in SETS}
    for label, _ in SETS:
        analyse(data[label], label)
    a, b = data[SETS[0][0]], data[SETS[1][0]]
    analyse({k: a[k] + b[k] for k in a if k in b}, "original + random judge pooled")
    random.seed(0)
    analyse({i: [[random.random() for _ in range(5)] for _ in range(5)] for i in range(20000)},
            "reference: 5 independent random judges")


if __name__ == "__main__":
    main()
