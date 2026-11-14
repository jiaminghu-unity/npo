Metrics: ps, hps, clip, aes, image_reward, hpsv3 (the five of eval.py, hps = HPSv2 as before, plus hpsv3 = HPSv3)

Average score, mean ± std over seeds

| model | seeds | ps | hps | clip | aes | image_reward | hpsv3 |
|---|---|---|---|---|---|---|---|
| Orig._Sana/base | 5 | 0.2218 ± 0.0004 | 0.2847 ± 0.0006 | 0.3552 ± 0.0014 | 6.0293 ± 0.0173 | 1.0974 ± 0.0149 | 7.9932 ± 0.1561 |
| NPO-avgrank/140 | 5 | 0.2224 ± 0.0003 | 0.2860 ± 0.0005 | 0.3567 ± 0.0012 | 6.0573 ± 0.0114 | 1.1386 ± 0.0141 | 8.0995 ± 0.1299 |
| NPO-randjudge/130 | 5 | 0.2222 ± 0.0003 | 0.2856 ± 0.0005 | 0.3564 ± 0.0010 | 6.0498 ± 0.0155 | 1.1251 ± 0.0205 | 8.0684 ± 0.1437 |

Win rate against the base model (share of the prompts where the model scores higher), mean ± std over seeds

| model | ps | hps | clip | aes | image_reward | hpsv3 |
|---|---|---|---|---|---|---|
| NPO-avgrank/140 | 66.0% ± 3.3 | 69.7% ± 1.9 | 55.5% ± 2.3 | 62.9% ± 3.2 | 64.3% ± 2.4 | 57.8% ± 1.5 |
| NPO-randjudge/130 | 63.8% ± 1.6 | 68.4% ± 2.1 | 54.2% ± 3.2 | 62.2% ± 2.7 | 62.2% ± 2.7 | 58.2% ± 2.3 |
