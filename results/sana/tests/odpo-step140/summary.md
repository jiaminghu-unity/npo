Metrics: ps, hps, clip, aes, image_reward, hpsv3 (the five of eval.py, hps = HPSv2 as before, plus hpsv3 = HPSv3)

Average score, mean ± std over seeds

| model | seeds | ps | hps | clip | aes | image_reward | hpsv3 |
|---|---|---|---|---|---|---|---|
| Orig._Sana/base | 5 | 0.2218 ± 0.0004 | 0.2847 ± 0.0006 | 0.3552 ± 0.0014 | 6.0293 ± 0.0173 | 1.0974 ± 0.0149 | 7.9932 ± 0.1561 |
| Online-DPO-avgrank/140 | 5 | 0.2222 ± 0.0003 | 0.2855 ± 0.0005 | 0.3563 ± 0.0015 | 6.0452 ± 0.0150 | 1.1225 ± 0.0148 | 8.0662 ± 0.1476 |

Win rate against the base model (share of the prompts where the model scores higher), mean ± std over seeds

| model | ps | hps | clip | aes | image_reward | hpsv3 |
|---|---|---|---|---|---|---|
| Online-DPO-avgrank/140 | 63.2% ± 2.0 | 66.0% ± 2.7 | 53.6% ± 2.0 | 60.6% ± 0.7 | 60.6% ± 1.3 | 57.3% ± 1.9 |
