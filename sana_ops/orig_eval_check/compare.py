#!/usr/bin/env python3
"""Compare the original eval.py with eval_sana.py --model_type sd15 (SD1.5, seed 0, 500 prompts).

  A    = original eval.py end to end                      (out/A_eval_result/<model>/<epoch>/seed_0/scores_pickapic.csv)
  B    = eval_sana.py's scorer on the images A generated  (out/B_my_scorer/<model>/<epoch>/seed_0/scores_partial.csv)
  MINE = the test already reported (eval_sana.py end to end, batch 32), seed 0
"""
import csv
import os
import sys

M = ["ps", "hps", "clip", "aes", "image_reward"]
HERE = sys.argv[1]            # ~/npo_sana_dev/tests/orig-eval-check/out
MINE = sys.argv[2]            # folder with <model>/<epoch>/seed_0/scores_pickapic.csv of the reported test
MODELS = [("Orig._SD1.5", "base", "base"), ("Diffusion-DPO", "none", "official"), ("Diffusion-KTO", "none", "official")]


def num(x):
    x = x.strip()
    if x.startswith("tensor("):
        x = x[7:].split(",")[0].rstrip(")")
    return float(x)


def load(path, rename=None):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        out[r["prompt_id"]] = {m: num(r[(rename or {}).get(m, m)]) for m in M}
    return out


def avg(d, m):
    return sum(v[m] for v in d.values()) / len(d)


def win(d, base, m):
    ids = [i for i in d if i in base]
    return 100 * sum((d[i][m] >= base[i][m]) if m == "image_reward" else (d[i][m] > base[i][m]) for i in ids) / len(ids)


A, B, MY = {}, {}, {}
for name, ep_a, ep_my in MODELS:
    A[name] = load(os.path.join(HERE, "A_eval_result", name, ep_a, "seed_0", "scores_pickapic.csv"))
    B[name] = load(os.path.join(HERE, "B_my_scorer", name, ep_a, "seed_0", "scores_partial.csv"), {"hps": "hps_v2"})
    MY[name] = load(os.path.join(MINE, name, ep_my, "seed_0", "scores_pickapic.csv"))
    print(f"{name}: rows A={len(A[name])} B={len(B[name])} MINE={len(MY[name])}")

print("\n1) Same images, two scorers (A = original Selector calls, B = eval_sana.py scorer): per-image |difference|")
print("| model | metric | mean abs diff | max abs diff | mean score |")
print("|---|---|---|---|---|")
for name, _, _ in MODELS:
    for m in M:
        ids = [i for i in A[name] if i in B[name]]
        d = [abs(A[name][i][m] - B[name][i][m]) for i in ids]
        print(f"| {name} | {m} | {sum(d) / len(d):.2e} | {max(d):.2e} | {avg(A[name], m):.4f} |")

print("\n2) End to end, seed 0: original eval.py (A) vs the reported test (MINE)")
print("| model | source | " + " | ".join(M) + " |")
print("|---|---|" + "---|" * len(M))
for name, _, _ in MODELS:
    print(f"| {name} avg | eval.py | " + " | ".join(f"{avg(A[name], m):.4f}" for m in M) + " |")
    print(f"| {name} avg | reported | " + " | ".join(f"{avg(MY[name], m):.4f}" for m in M) + " |")
for name, _, _ in MODELS[1:]:
    print(f"| {name} win% | eval.py | " + " | ".join(f"{win(A[name], A['Orig._SD1.5'], m):.1f}" for m in M) + " |")
    print(f"| {name} win% | reported | " + " | ".join(f"{win(MY[name], MY['Orig._SD1.5'], m):.1f}" for m in M) + " |")

print("\n3) Per-prompt |difference| between A and MINE (different generation batch size: 16 vs 32)")
print("| model | metric | mean abs diff | max abs diff | prompts with diff > 1e-3 |")
print("|---|---|---|---|---|")
for name, _, _ in MODELS:
    for m in M:
        ids = [i for i in A[name] if i in MY[name]]
        d = [abs(A[name][i][m] - MY[name][i][m]) for i in ids]
        print(f"| {name} | {m} | {sum(d) / len(d):.2e} | {max(d):.2e} | {sum(x > 1e-3 for x in d)}/{len(d)} |")
