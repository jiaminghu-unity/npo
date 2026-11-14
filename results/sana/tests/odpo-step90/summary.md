Metrics: ps, hps, clip, aes, image_reward, hpsv3 (the five of eval.py, hps = HPSv2 as before, plus hpsv3 = HPSv3)

Average score, mean ± std over seeds

| model | seeds | ps | hps | clip | aes | image_reward | hpsv3 |
|---|---|---|---|---|---|---|---|
| Orig._Sana/base | 5 | 0.2218 ± 0.0004 | 0.2847 ± 0.0006 | 0.3552 ± 0.0014 | 6.0293 ± 0.0173 | 1.0974 ± 0.0149 | 7.9932 ± 0.1561 |
| Online-DPO-avgrank/90 | 5 | 0.2220 ± 0.0003 | 0.2853 ± 0.0005 | 0.3561 ± 0.0013 | 6.0423 ± 0.0173 | 1.1181 ± 0.0196 | 8.0409 ± 0.1366 |

Win rate against the base model (share of the prompts where the model scores higher), mean ± std over seeds

| model | ps | hps | clip | aes | image_reward | hpsv3 |
|---|---|---|---|---|---|---|
| Online-DPO-avgrank/90 | 61.5% ± 1.3 | 65.9% ± 2.1 | 53.3% ± 1.9 | 59.9% ± 2.1 | 61.8% ± 2.1 | 56.0% ± 3.2 |
