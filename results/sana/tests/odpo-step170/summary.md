Metrics: ps, hps, clip, aes, image_reward, hpsv3 (the five of eval.py, hps = HPSv2 as before, plus hpsv3 = HPSv3)

Average score, mean ± std over seeds

| model | seeds | ps | hps | clip | aes | image_reward | hpsv3 |
|---|---|---|---|---|---|---|---|
| Orig._Sana/base | 5 | 0.2218 ± 0.0004 | 0.2847 ± 0.0006 | 0.3552 ± 0.0014 | 6.0293 ± 0.0173 | 1.0974 ± 0.0149 | 7.9932 ± 0.1561 |
| Online-DPO-avgrank/170 | 5 | 0.2222 ± 0.0003 | 0.2857 ± 0.0005 | 0.3561 ± 0.0013 | 6.0465 ± 0.0160 | 1.1295 ± 0.0156 | 8.0891 ± 0.1363 |

Win rate against the base model (share of the prompts where the model scores higher), mean ± std over seeds

| model | ps | hps | clip | aes | image_reward | hpsv3 |
|---|---|---|---|---|---|---|
| Online-DPO-avgrank/170 | 63.4% ± 1.7 | 67.4% ± 2.5 | 54.5% ± 1.1 | 60.7% ± 0.7 | 63.7% ± 1.7 | 59.6% ± 2.8 |
