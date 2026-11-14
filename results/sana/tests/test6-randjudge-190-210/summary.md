Metrics: ps, hps, clip, aes, image_reward, hpsv3 (the five of eval.py, hps = HPSv2 as before, plus hpsv3 = HPSv3)

Average score, mean ± std over seeds

| model | seeds | ps | hps | clip | aes | image_reward | hpsv3 |
|---|---|---|---|---|---|---|---|
| Orig._Sana/base | 5 | 0.2218 ± 0.0004 | 0.2847 ± 0.0006 | 0.3552 ± 0.0014 | 6.0293 ± 0.0173 | 1.0974 ± 0.0149 | 7.9932 ± 0.1561 |
| NPO-randjudge/190 | 5 | 0.2223 ± 0.0003 | 0.2859 ± 0.0005 | 0.3567 ± 0.0014 | 6.0480 ± 0.0149 | 1.1321 ± 0.0141 | 8.1066 ± 0.1280 |
| NPO-randjudge/210 | 5 | 0.2223 ± 0.0003 | 0.2857 ± 0.0006 | 0.3563 ± 0.0014 | 6.0536 ± 0.0145 | 1.1257 ± 0.0160 | 8.0713 ± 0.1480 |

Win rate against the base model (share of the prompts where the model scores higher), mean ± std over seeds

| model | ps | hps | clip | aes | image_reward | hpsv3 |
|---|---|---|---|---|---|---|
| NPO-randjudge/190 | 63.2% ± 2.4 | 69.2% ± 1.7 | 56.1% ± 1.3 | 60.8% ± 2.2 | 64.3% ± 1.7 | 59.1% ± 2.0 |
| NPO-randjudge/210 | 65.3% ± 1.7 | 67.1% ± 2.2 | 54.3% ± 1.6 | 61.7% ± 1.8 | 61.4% ± 2.0 | 56.3% ± 3.0 |
