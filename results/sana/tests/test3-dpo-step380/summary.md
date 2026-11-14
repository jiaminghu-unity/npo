Metrics: ps, hps, clip, aes, image_reward, hpsv3 (the five of eval.py, hps = HPSv2 as before, plus hpsv3 = HPSv3)

Average score, mean ± std over seeds

| model | seeds | ps | hps | clip | aes | image_reward | hpsv3 |
|---|---|---|---|---|---|---|---|
| Orig._Sana/base | 5 | 0.2218 ± 0.0004 | 0.2847 ± 0.0006 | 0.3552 ± 0.0014 | 6.0293 ± 0.0173 | 1.0974 ± 0.0149 | 7.9932 ± 0.1561 |
| DPO/270 | 5 | 0.2220 ± 0.0004 | 0.2849 ± 0.0005 | 0.3553 ± 0.0012 | 6.0321 ± 0.0155 | 1.1004 ± 0.0209 | 8.0567 ± 0.1469 |
| DPO/380 | 5 | 0.2220 ± 0.0004 | 0.2849 ± 0.0005 | 0.3555 ± 0.0016 | 6.0306 ± 0.0160 | 1.1071 ± 0.0150 | 8.0658 ± 0.1545 |

Win rate against the base model (share of the prompts where the model scores higher), mean ± std over seeds

| model | ps | hps | clip | aes | image_reward | hpsv3 |
|---|---|---|---|---|---|---|
| DPO/270 | 59.0% ± 1.6 | 56.6% ± 2.1 | 50.8% ± 2.6 | 51.0% ± 2.2 | 53.0% ± 3.4 | 54.8% ± 1.0 |
| DPO/380 | 56.9% ± 2.6 | 54.1% ± 3.0 | 50.8% ± 1.2 | 50.6% ± 1.4 | 54.8% ± 1.4 | 55.9% ± 2.4 |
