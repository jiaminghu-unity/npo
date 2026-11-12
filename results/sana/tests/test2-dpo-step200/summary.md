Metrics: ps, hps, clip, aes, image_reward, hpsv3 (the five of eval.py, hps = HPSv2 as before, plus hpsv3 = HPSv3)

Average score, mean ± std over seeds

| model | seeds | ps | hps | clip | aes | image_reward | hpsv3 |
|---|---|---|---|---|---|---|---|
| Orig._Sana/base | 5 | 0.2218 ± 0.0004 | 0.2847 ± 0.0006 | 0.3552 ± 0.0014 | 6.0293 ± 0.0173 | 1.0974 ± 0.0149 | 7.9932 ± 0.1561 |
| DPO/180 | 5 | 0.2219 ± 0.0004 | 0.2848 ± 0.0006 | 0.3554 ± 0.0013 | 6.0324 ± 0.0142 | 1.1057 ± 0.0188 | 8.0169 ± 0.1467 |
| DPO/200 | 5 | 0.2219 ± 0.0003 | 0.2849 ± 0.0005 | 0.3555 ± 0.0013 | 6.0337 ± 0.0151 | 1.1047 ± 0.0166 | 8.0438 ± 0.1442 |

Win rate against the base model (share of the prompts where the model scores higher), mean ± std over seeds

| model | ps | hps | clip | aes | image_reward | hpsv3 |
|---|---|---|---|---|---|---|
| DPO/180 | 57.1% ± 1.6 | 55.0% ± 3.5 | 49.6% ± 1.5 | 51.5% ± 3.0 | 55.2% ± 2.1 | 54.2% ± 3.0 |
| DPO/200 | 57.1% ± 2.3 | 56.7% ± 0.8 | 49.6% ± 1.6 | 50.4% ± 1.7 | 56.4% ± 1.9 | 54.8% ± 2.1 |
