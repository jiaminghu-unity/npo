# Results: Sana 0.6B (flow matching) runs and SD1.5 baselines

Snapshot of 2026-11-13, about 17:20 UTC. The two NPO runs were stopped on 2026-11-13 09:53 UTC at steps 225
(random judge) and 227 (average rank): their test results were flat from step 170 to step 210 (tests at 170,
190, 200, 210). Step 200, the end of the learning-rate warmup, is taken as the final checkpoint of both runs.
The offline DPO run finished earlier (early stop at step 420: 15 evaluations without a new best). The Online-DPO
control (started 2026-11-13 00:27 UTC) is still training (step about 79; first test at step 70).

## What was run

| run | method | data / judge | GPUs |
|---|---|---|---|
| `sana-newnash-randjudge-v1` | Diff.-NPO (`newnash`) | online, 8 candidates per prompt; at every training step one of the five reward models is drawn at random as the judge | 4 x A100 |
| `sana-newnash-avgrank-v1` | Diff.-NPO (`newnash`) | online, 8 candidates per prompt; judge = average rank over the five reward models (as in the paper) | 4 x A100 |
| `sana-onlinedpo-avgrank-v1` | Online-DPO (`online_dpo`) | identical to the avg-rank NPO run (same seed, sampler, judge and hyper-parameters); only the loss differs | 4 x A100 |
| `sana-dpo-offline-v1` | offline flow-matching DPO | 294,418 human-labelled Pick-a-Pic v1 pairs (label 0/1, SFW subset) | 2 x A100 |

Model: `Efficient-Large-Model/Sana_600M_512px_diffusers`. The configuration mirrors the SD1.5 `newnash`
launcher (`launchers/sd15.sh`): 2048 pairs per optimizer step, lr 1e-8 scaled by the batch (2.048e-5),
200 warmup steps, beta 5000, tau/eta = 0.025/0.05, hard update of the previous policy, 10 sampling steps.
Only what flow matching requires differs (velocity target, flow-matching scheduler, Sana pipeline, CFG 4.5).
Code: `train_nash_sana.py`, `launchers/sana_*.sh`, cluster scripts in `sana_ops/`.

## Test protocol

500 prompts of `pickapic_v1` `test_unique`, one image per prompt and model, seeds 0-4, the same seed for
every model; metrics PickScore, HPS (v2), CLIP, Aesthetics, ImageReward as in `eval.py`, plus HPSv3
(not used in training). Win rate = share of prompts where the model scores higher than the original model.

- Sana models: `eval_sana.py` (the Sana version of `eval.py`), then `eval_hpsv3.py`, `eval_summarize.py`.
- SD1.5 models: the original `eval.py`, unchanged (arguments as in `eval.sh`); HPSv3 is computed from the
  images that `eval.py` saved. `sd15/eval_py_5seeds/*/scores_evalpy.csv` is the file written by `eval.py`,
  `evalpy_final_summary_seed*.txt` is its printed summary.

## Folders

- `sana/training/<run>/` — `metrics.jsonl` (every training step) and `eval.jsonl` (every 10 steps).
- `sana/tests/<tag>/` — `summary.md`, `summary.json` and per-prompt scores of every tested checkpoint.
- `sd15/eval_py_5seeds/` — original SD1.5, official Diffusion-DPO and official Diffusion-KTO.
- `sd15/eval_py_vs_eval_sana_check/` — seed-0 comparison of `eval.py` with `eval_sana.py --model_type sd15`.
- `analysis/judge_cycles.*` — how often the five reward models form majority cycles among candidates.
- `make_tables.py` — prints the tables below from these files.

## Compute (measured)

| | Diff.-NPO on Sana | offline DPO on Sana |
|---|---|---|
| GPU-seconds per optimizer step (2048 pairs) | 2976 (12.4 min on 4 GPUs) | 262 (2.2 min on 2 GPUs) |
| transformer forward passes per training pair | 166 | 4 |

## Tables

### Sana 0.6B (eval_sana.py)

Average score, mean ± std over 5 seeds

| model / step | PickScore | HPS (v2) | CLIP | Aesthetics | ImageReward | HPSv3 |
|---|---|---|---|---|---|---|
| Orig._Sana/base | 0.2218 ± 0.0004 | 0.2847 ± 0.0006 | 0.3552 ± 0.0014 | 6.0293 ± 0.0173 | 1.0974 ± 0.0149 | 7.9932 ± 0.1561 |
| DPO/60 | 0.2218 ± 0.0004 | 0.2848 ± 0.0006 | 0.3553 ± 0.0012 | 6.0291 ± 0.0150 | 1.1013 ± 0.0156 | 8.0291 ± 0.1617 |
| DPO/120 | 0.2219 ± 0.0003 | 0.2849 ± 0.0005 | 0.3554 ± 0.0013 | 6.0294 ± 0.0166 | 1.1022 ± 0.0132 | 8.0388 ± 0.1534 |
| DPO/180 | 0.2219 ± 0.0004 | 0.2848 ± 0.0006 | 0.3554 ± 0.0013 | 6.0324 ± 0.0142 | 1.1057 ± 0.0188 | 8.0169 ± 0.1467 |
| DPO/200 | 0.2219 ± 0.0003 | 0.2849 ± 0.0005 | 0.3555 ± 0.0013 | 6.0337 ± 0.0151 | 1.1047 ± 0.0166 | 8.0438 ± 0.1442 |
| DPO/270 | 0.2220 ± 0.0004 | 0.2849 ± 0.0005 | 0.3553 ± 0.0012 | 6.0321 ± 0.0155 | 1.1004 ± 0.0209 | 8.0567 ± 0.1469 |
| DPO/380 | 0.2220 ± 0.0004 | 0.2849 ± 0.0005 | 0.3555 ± 0.0016 | 6.0306 ± 0.0160 | 1.1071 ± 0.0150 | 8.0658 ± 0.1545 |
| Online-DPO-avgrank/70 | 0.2220 ± 0.0003 | 0.2853 ± 0.0005 | 0.3559 ± 0.0011 | 6.0411 ± 0.0189 | 1.1168 ± 0.0180 | 8.0324 ± 0.1348 |
| NPO-randjudge/90 | 0.2221 ± 0.0003 | 0.2856 ± 0.0006 | 0.3563 ± 0.0013 | 6.0449 ± 0.0157 | 1.1248 ± 0.0154 | 8.0650 ± 0.1480 |
| NPO-randjudge/130 | 0.2222 ± 0.0003 | 0.2856 ± 0.0005 | 0.3564 ± 0.0010 | 6.0498 ± 0.0155 | 1.1251 ± 0.0205 | 8.0684 ± 0.1437 |
| NPO-randjudge/170 | 0.2222 ± 0.0003 | 0.2857 ± 0.0006 | 0.3563 ± 0.0014 | 6.0462 ± 0.0121 | 1.1240 ± 0.0148 | 8.0679 ± 0.1355 |
| NPO-randjudge/190 | 0.2223 ± 0.0003 | 0.2859 ± 0.0005 | 0.3567 ± 0.0014 | 6.0480 ± 0.0149 | 1.1321 ± 0.0141 | 8.1066 ± 0.1280 |
| NPO-randjudge/200 | 0.2223 ± 0.0003 | 0.2859 ± 0.0005 | 0.3567 ± 0.0015 | 6.0486 ± 0.0118 | 1.1377 ± 0.0108 | 8.1117 ± 0.1365 |
| NPO-randjudge/210 | 0.2223 ± 0.0003 | 0.2857 ± 0.0006 | 0.3563 ± 0.0014 | 6.0536 ± 0.0145 | 1.1257 ± 0.0160 | 8.0713 ± 0.1480 |
| NPO-avgrank/90 | 0.2222 ± 0.0003 | 0.2857 ± 0.0004 | 0.3563 ± 0.0012 | 6.0499 ± 0.0169 | 1.1277 ± 0.0175 | 8.0598 ± 0.1277 |
| NPO-avgrank/140 | 0.2224 ± 0.0003 | 0.2860 ± 0.0005 | 0.3567 ± 0.0012 | 6.0573 ± 0.0114 | 1.1386 ± 0.0141 | 8.0995 ± 0.1299 |
| NPO-avgrank/170 | 0.2224 ± 0.0003 | 0.2862 ± 0.0005 | 0.3570 ± 0.0013 | 6.0566 ± 0.0141 | 1.1408 ± 0.0125 | 8.1179 ± 0.1412 |
| NPO-avgrank/190 | 0.2225 ± 0.0003 | 0.2861 ± 0.0004 | 0.3567 ± 0.0012 | 6.0584 ± 0.0154 | 1.1342 ± 0.0085 | 8.1096 ± 0.1406 |
| NPO-avgrank/200 | 0.2225 ± 0.0003 | 0.2862 ± 0.0004 | 0.3567 ± 0.0012 | 6.0607 ± 0.0150 | 1.1397 ± 0.0105 | 8.1241 ± 0.1292 |
| NPO-avgrank/210 | 0.2225 ± 0.0003 | 0.2862 ± 0.0004 | 0.3568 ± 0.0012 | 6.0593 ± 0.0147 | 1.1409 ± 0.0127 | 8.1142 ± 0.1472 |

Win rate against the original model (%), mean ± std over 5 seeds

| model / step | PickScore | HPS (v2) | CLIP | Aesthetics | ImageReward | HPSv3 |
|---|---|---|---|---|---|---|
| DPO/60 | 57.5 ± 0.8 | 54.6 ± 3.1 | 50.2 ± 3.2 | 49.1 ± 1.7 | 53.7 ± 1.4 | 54.4 ± 3.3 |
| DPO/120 | 57.4 ± 1.8 | 57.2 ± 1.7 | 52.0 ± 1.3 | 50.2 ± 2.1 | 54.3 ± 3.1 | 54.9 ± 2.2 |
| DPO/180 | 57.1 ± 1.6 | 55.0 ± 3.5 | 49.6 ± 1.5 | 51.5 ± 3.0 | 55.2 ± 2.1 | 54.2 ± 3.0 |
| DPO/200 | 57.1 ± 2.3 | 56.7 ± 0.8 | 49.6 ± 1.6 | 50.4 ± 1.7 | 56.4 ± 1.9 | 54.8 ± 2.1 |
| DPO/270 | 59.0 ± 1.6 | 56.6 ± 2.1 | 50.8 ± 2.6 | 51.0 ± 2.2 | 53.0 ± 3.4 | 54.8 ± 1.0 |
| DPO/380 | 56.9 ± 2.6 | 54.1 ± 3.0 | 50.8 ± 1.2 | 50.6 ± 1.4 | 54.8 ± 1.4 | 55.9 ± 2.4 |
| Online-DPO-avgrank/70 | 61.1 ± 2.8 | 67.0 ± 1.9 | 53.0 ± 1.6 | 58.8 ± 2.2 | 62.4 ± 0.6 | 54.5 ± 3.5 |
| NPO-randjudge/90 | 59.9 ± 1.7 | 69.2 ± 1.4 | 54.0 ± 2.5 | 60.3 ± 1.4 | 63.8 ± 1.3 | 55.8 ± 2.6 |
| NPO-randjudge/130 | 63.8 ± 1.6 | 68.4 ± 2.1 | 54.2 ± 3.2 | 62.2 ± 2.7 | 62.2 ± 2.7 | 58.2 ± 2.3 |
| NPO-randjudge/170 | 63.4 ± 2.1 | 67.6 ± 2.3 | 53.6 ± 1.4 | 60.5 ± 2.2 | 62.3 ± 1.9 | 56.8 ± 3.0 |
| NPO-randjudge/190 | 63.2 ± 2.4 | 69.2 ± 1.7 | 56.1 ± 1.3 | 60.8 ± 2.2 | 64.3 ± 1.7 | 59.1 ± 2.0 |
| NPO-randjudge/200 | 61.4 ± 2.0 | 68.9 ± 2.7 | 55.0 ± 2.6 | 59.5 ± 1.0 | 65.4 ± 2.1 | 58.9 ± 1.8 |
| NPO-randjudge/210 | 65.3 ± 1.7 | 67.1 ± 2.2 | 54.3 ± 1.6 | 61.7 ± 1.8 | 61.4 ± 2.0 | 56.3 ± 3.0 |
| NPO-avgrank/90 | 62.4 ± 2.8 | 68.8 ± 2.5 | 53.8 ± 1.6 | 62.6 ± 0.9 | 63.3 ± 1.0 | 55.9 ± 1.1 |
| NPO-avgrank/140 | 66.0 ± 3.3 | 69.7 ± 1.9 | 55.5 ± 2.3 | 62.9 ± 3.2 | 64.3 ± 2.4 | 57.8 ± 1.5 |
| NPO-avgrank/170 | 65.7 ± 3.4 | 71.0 ± 2.0 | 56.0 ± 1.5 | 62.4 ± 1.8 | 65.3 ± 2.5 | 60.2 ± 1.6 |
| NPO-avgrank/190 | 67.2 ± 3.5 | 69.8 ± 2.7 | 54.3 ± 1.0 | 63.6 ± 1.9 | 64.0 ± 1.5 | 59.4 ± 1.5 |
| NPO-avgrank/200 | 66.2 ± 2.9 | 71.3 ± 1.5 | 53.9 ± 1.4 | 62.4 ± 1.8 | 64.8 ± 2.2 | 59.8 ± 1.9 |
| NPO-avgrank/210 | 67.0 ± 3.0 | 70.2 ± 1.0 | 54.9 ± 1.2 | 62.0 ± 2.4 | 64.4 ± 0.9 | 59.1 ± 2.2 |

### SD1.5 baselines (original eval.py; `none` = the official checkpoint)

Average score, mean ± std over 5 seeds

| model / step | PickScore | HPS (v2) | CLIP | Aesthetics | ImageReward | HPSv3 |
|---|---|---|---|---|---|---|
| Orig._SD1.5/base | 0.2068 ± 0.0010 | 0.2680 ± 0.0026 | 0.3340 ± 0.0034 | 5.4946 ± 0.0680 | 0.0470 ± 0.1485 | 5.7062 ± 0.5886 |
| Diffusion-DPO/none | 0.2108 ± 0.0013 | 0.2720 ± 0.0022 | 0.3404 ± 0.0024 | 5.5975 ± 0.0653 | 0.2568 ± 0.1531 | 6.6361 ± 0.5434 |
| Diffusion-KTO/none | 0.2119 ± 0.0005 | 0.2782 ± 0.0013 | 0.3427 ± 0.0009 | 5.7060 ± 0.0377 | 0.6145 ± 0.0667 | 7.6032 ± 0.2212 |

Win rate against the original model (%), mean ± std over 5 seeds

| model / step | PickScore | HPS (v2) | CLIP | Aesthetics | ImageReward | HPSv3 |
|---|---|---|---|---|---|---|
| Diffusion-DPO/none | 76.0 ± 1.8 | 68.8 ± 3.6 | 55.8 ± 2.2 | 63.5 ± 1.8 | 62.2 ± 1.4 | 67.8 ± 1.2 |
| Diffusion-KTO/none | 73.1 ± 3.4 | 81.3 ± 4.1 | 57.0 ± 2.9 | 68.8 ± 3.4 | 76.6 ± 2.7 | 75.5 ± 5.0 |

### Training-time eval gain (256 PartiPrompts, mean over the five reward models of the standardized improvement over step 0)

| step | NPO random judge | NPO avg rank | Online-DPO avg rank | offline DPO |
|---|---|---|---|---|
| 10 | +0.0017 | +0.0038 | +0.0023 | -0.0000 |
| 20 | +0.0055 | +0.0073 | +0.0030 | +0.0017 |
| 30 | +0.0085 | +0.0098 | +0.0069 | +0.0001 |
| 40 | +0.0178 | +0.0155 | +0.0107 | +0.0014 |
| 50 | +0.0133 | +0.0187 | +0.0100 | +0.0048 |
| 60 | +0.0145 | +0.0250 | +0.0150 | +0.0077 |
| 70 | +0.0205 | +0.0228 | +0.0181 | +0.0045 |
| 80 | +0.0201 | +0.0290 | – | +0.0016 |
| 90 | +0.0216 | +0.0295 | – | -0.0001 |
| 100 | +0.0155 | +0.0317 | – | -0.0000 |
| 120 | +0.0263 | +0.0297 | – | +0.0034 |
| 140 | +0.0288 | +0.0367 | – | +0.0055 |
| 160 | +0.0366 | +0.0441 | – | +0.0043 |
| 180 | +0.0372 | +0.0433 | – | +0.0117 |
| 200 | +0.0383 | +0.0509 | – | +0.0033 |
| 220 | +0.0315 | +0.0551 | – | +0.0113 |
| 240 | – | – | – | +0.0103 |
| 260 | – | – | – | +0.0157 |
| 270 | – | – | – | +0.0186 |
| 280 | – | – | – | +0.0102 |
| 300 | – | – | – | +0.0165 |
| 320 | – | – | – | +0.0119 |
| 340 | – | – | – | +0.0090 |
| 360 | – | – | – | +0.0118 |
| 380 | – | – | – | +0.0119 |
| 400 | – | – | – | +0.0105 |
| 420 | – | – | – | +0.0057 |

Last evaluated step: NPO random judge 220, NPO avg rank 220, Online-DPO avg rank 70, offline DPO 420

### Majority cycles among the five reward models (`analysis/judge_cycles.py`)

| candidates | per prompt | pairs where the judges are not unanimous | cyclic triples | prompts with at least one cycle | prompts with no majority winner |
|---|---|---|---|---|---|
| original Sana, 5 seeds | 5 | 83.0% | 3.26% (163/5000) | 22.2% | 9.0% |
| NPO random judge step 130, 5 seeds | 5 | 83.9% | 2.68% (134/5000) | 18.2% | 7.0% |
| NPO avg rank step 140, 5 seeds | 5 | 84.4% | 3.14% (157/5000) | 20.2% | 10.4% |
| original + random judge pooled | 10 | 83.5% | 2.47% (1483/60000) | 59.2% | 12.2% |
| reference: 5 independent random judges | 5 | 93.8% | 6.98% (13961/200000) | 39.1% | 20.0% |
