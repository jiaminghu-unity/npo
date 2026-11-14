Metrics: ps, hps, clip, aes, image_reward, hpsv3 (the five of eval.py, hps = HPSv2 as before, plus hpsv3 = HPSv3)

Average score, mean ± std over seeds

| model | seeds | ps | hps | clip | aes | image_reward | hpsv3 |
|---|---|---|---|---|---|---|---|
| Orig._Sana/base | 5 | 0.2218 ± 0.0004 | 0.2847 ± 0.0006 | 0.3552 ± 0.0014 | 6.0293 ± 0.0173 | 1.0974 ± 0.0149 | 7.9932 ± 0.1561 |
| Online-DPO-avgrank/70 | 5 | 0.2220 ± 0.0003 | 0.2853 ± 0.0005 | 0.3559 ± 0.0011 | 6.0411 ± 0.0189 | 1.1168 ± 0.0180 | 8.0324 ± 0.1348 |

Win rate against the base model (share of the prompts where the model scores higher), mean ± std over seeds

| model | ps | hps | clip | aes | image_reward | hpsv3 |
|---|---|---|---|---|---|---|
| Online-DPO-avgrank/70 | 61.1% ± 2.8 | 67.0% ± 1.9 | 53.0% ± 1.6 | 58.8% ± 2.2 | 62.4% ± 0.6 | 54.5% ± 3.5 |
