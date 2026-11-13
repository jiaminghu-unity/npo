Metrics: ps, hps, clip, aes, image_reward, hpsv3 (the five of eval.py, hps = HPSv2 as before, plus hpsv3 = HPSv3)

Average score, mean ± std over seeds

| model | seeds | ps | hps | clip | aes | image_reward | hpsv3 |
|---|---|---|---|---|---|---|---|
| Orig._Sana/base | 5 | 0.2218 ± 0.0004 | 0.2847 ± 0.0006 | 0.3552 ± 0.0014 | 6.0293 ± 0.0173 | 1.0974 ± 0.0149 | 7.9932 ± 0.1561 |
| NPO-avgrank/200 | 5 | 0.2225 ± 0.0003 | 0.2862 ± 0.0004 | 0.3567 ± 0.0012 | 6.0607 ± 0.0150 | 1.1397 ± 0.0105 | 8.1241 ± 0.1292 |
| NPO-randjudge/200 | 5 | 0.2223 ± 0.0003 | 0.2859 ± 0.0005 | 0.3567 ± 0.0015 | 6.0486 ± 0.0118 | 1.1377 ± 0.0108 | 8.1117 ± 0.1365 |

Win rate against the base model (share of the prompts where the model scores higher), mean ± std over seeds

| model | ps | hps | clip | aes | image_reward | hpsv3 |
|---|---|---|---|---|---|---|
| NPO-avgrank/200 | 66.2% ± 2.9 | 71.3% ± 1.5 | 53.9% ± 1.4 | 62.4% ± 1.8 | 64.8% ± 2.2 | 59.8% ± 1.9 |
| NPO-randjudge/200 | 61.4% ± 2.0 | 68.9% ± 2.7 | 55.0% ± 2.6 | 59.5% ± 1.0 | 65.4% ± 2.1 | 58.9% ± 1.8 |
