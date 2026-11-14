Metrics: ps, hps, clip, aes, image_reward, hpsv3 (the five of eval.py, hps = HPSv2 as before, plus hpsv3 = HPSv3)

Average score, mean ± std over seeds

| model | seeds | ps | hps | clip | aes | image_reward | hpsv3 |
|---|---|---|---|---|---|---|---|
| Orig._Sana/base | 5 | 0.2218 ± 0.0004 | 0.2847 ± 0.0006 | 0.3552 ± 0.0014 | 6.0293 ± 0.0173 | 1.0974 ± 0.0149 | 7.9932 ± 0.1561 |
| NPO-avgrank/170 | 5 | 0.2224 ± 0.0003 | 0.2862 ± 0.0005 | 0.3570 ± 0.0013 | 6.0566 ± 0.0141 | 1.1408 ± 0.0125 | 8.1179 ± 0.1412 |
| NPO-randjudge/170 | 5 | 0.2222 ± 0.0003 | 0.2857 ± 0.0006 | 0.3563 ± 0.0014 | 6.0462 ± 0.0121 | 1.1240 ± 0.0148 | 8.0679 ± 0.1355 |

Win rate against the base model (share of the prompts where the model scores higher), mean ± std over seeds

| model | ps | hps | clip | aes | image_reward | hpsv3 |
|---|---|---|---|---|---|---|
| NPO-avgrank/170 | 65.7% ± 3.4 | 71.0% ± 2.0 | 56.0% ± 1.5 | 62.4% ± 1.8 | 65.3% ± 2.5 | 60.2% ± 1.6 |
| NPO-randjudge/170 | 63.4% ± 2.1 | 67.6% ± 2.3 | 53.6% ± 1.4 | 60.5% ± 2.2 | 62.3% ± 1.9 | 56.8% ± 3.0 |
