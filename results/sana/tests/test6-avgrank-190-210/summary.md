Metrics: ps, hps, clip, aes, image_reward, hpsv3 (the five of eval.py, hps = HPSv2 as before, plus hpsv3 = HPSv3)

Average score, mean ± std over seeds

| model | seeds | ps | hps | clip | aes | image_reward | hpsv3 |
|---|---|---|---|---|---|---|---|
| Orig._Sana/base | 5 | 0.2218 ± 0.0004 | 0.2847 ± 0.0006 | 0.3552 ± 0.0014 | 6.0293 ± 0.0173 | 1.0974 ± 0.0149 | 7.9932 ± 0.1561 |
| NPO-avgrank/190 | 5 | 0.2225 ± 0.0003 | 0.2861 ± 0.0004 | 0.3567 ± 0.0012 | 6.0584 ± 0.0154 | 1.1342 ± 0.0085 | 8.1096 ± 0.1406 |
| NPO-avgrank/210 | 5 | 0.2225 ± 0.0003 | 0.2862 ± 0.0004 | 0.3568 ± 0.0012 | 6.0593 ± 0.0147 | 1.1409 ± 0.0127 | 8.1142 ± 0.1472 |

Win rate against the base model (share of the prompts where the model scores higher), mean ± std over seeds

| model | ps | hps | clip | aes | image_reward | hpsv3 |
|---|---|---|---|---|---|---|
| NPO-avgrank/190 | 67.2% ± 3.5 | 69.8% ± 2.7 | 54.3% ± 1.0 | 63.6% ± 1.9 | 64.0% ± 1.5 | 59.4% ± 1.5 |
| NPO-avgrank/210 | 67.0% ± 3.0 | 70.2% ± 1.0 | 54.9% ± 1.2 | 62.0% ± 2.4 | 64.4% ± 0.9 | 59.1% ± 2.2 |
