Metrics: ps, hps, clip, aes, image_reward, hpsv3 (the five of eval.py, hps = HPSv2 as before, plus hpsv3 = HPSv3)

Average score, mean ± std over seeds

| model | seeds | ps | hps | clip | aes | image_reward | hpsv3 |
|---|---|---|---|---|---|---|---|
| Orig._SD1.5/base | 5 | 0.2068 ± 0.0010 | 0.2680 ± 0.0026 | 0.3340 ± 0.0034 | 5.4946 ± 0.0680 | 0.0470 ± 0.1485 | 5.7062 ± 0.5886 |
| Diffusion-DPO/none | 5 | 0.2108 ± 0.0013 | 0.2720 ± 0.0022 | 0.3404 ± 0.0024 | 5.5975 ± 0.0653 | 0.2568 ± 0.1531 | 6.6361 ± 0.5434 |
| Diffusion-KTO/none | 5 | 0.2119 ± 0.0005 | 0.2782 ± 0.0013 | 0.3427 ± 0.0009 | 5.7060 ± 0.0377 | 0.6145 ± 0.0667 | 7.6032 ± 0.2212 |

Win rate against the base model (share of the prompts where the model scores higher), mean ± std over seeds

| model | ps | hps | clip | aes | image_reward | hpsv3 |
|---|---|---|---|---|---|---|
| Diffusion-DPO/none | 76.0% ± 1.8 | 68.8% ± 3.6 | 55.8% ± 2.2 | 63.5% ± 1.8 | 62.2% ± 1.4 | 67.8% ± 1.2 |
| Diffusion-KTO/none | 73.1% ± 3.4 | 81.3% ± 4.1 | 57.0% ± 2.9 | 68.8% ± 3.4 | 76.6% ± 2.7 | 75.5% ± 5.0 |
