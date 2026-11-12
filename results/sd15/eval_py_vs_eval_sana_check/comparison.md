Orig._SD1.5: rows A=500 B=500 MINE=500
Diffusion-DPO: rows A=500 B=500 MINE=500
Diffusion-KTO: rows A=500 B=500 MINE=500

1) Same images, two scorers (A = original Selector calls, B = eval_sana.py scorer): per-image |difference|
| model | metric | mean abs diff | max abs diff | mean score |
|---|---|---|---|---|
| Orig._SD1.5 | ps | 3.49e-08 | 2.38e-07 | 0.2052 |
| Orig._SD1.5 | hps | 5.11e-05 | 4.54e-04 | 0.2648 |
| Orig._SD1.5 | clip | 1.81e-04 | 1.67e-03 | 0.3282 |
| Orig._SD1.5 | aes | 1.36e-07 | 9.54e-07 | 5.3847 |
| Orig._SD1.5 | image_reward | 1.37e-03 | 4.24e-02 | -0.1750 |
| Diffusion-DPO | ps | 3.29e-08 | 1.49e-07 | 0.2088 |
| Diffusion-DPO | hps | 5.32e-05 | 5.87e-04 | 0.2691 |
| Diffusion-DPO | clip | 1.80e-04 | 1.91e-03 | 0.3363 |
| Diffusion-DPO | aes | 1.43e-07 | 9.54e-07 | 5.4894 |
| Diffusion-DPO | image_reward | 1.53e-03 | 3.28e-02 | 0.0299 |
| Diffusion-KTO | ps | 3.18e-08 | 2.09e-07 | 0.2114 |
| Diffusion-KTO | hps | 5.16e-05 | 4.38e-04 | 0.2764 |
| Diffusion-KTO | clip | 1.83e-04 | 1.97e-03 | 0.3437 |
| Diffusion-KTO | aes | 1.10e-07 | 9.54e-07 | 5.6464 |
| Diffusion-KTO | image_reward | 1.13e-03 | 2.02e-02 | 0.5389 |

2) End to end, seed 0: original eval.py (A) vs the reported test (MINE)
| model | source | ps | hps | clip | aes | image_reward |
|---|---|---|---|---|---|---|
| Orig._SD1.5 avg | eval.py | 0.2052 | 0.2648 | 0.3282 | 5.3847 | -0.1750 |
| Orig._SD1.5 avg | reported | 0.2053 | 0.2649 | 0.3281 | 5.3898 | -0.1760 |
| Diffusion-DPO avg | eval.py | 0.2088 | 0.2691 | 0.3363 | 5.4894 | 0.0299 |
| Diffusion-DPO avg | reported | 0.2088 | 0.2691 | 0.3365 | 5.4938 | 0.0316 |
| Diffusion-KTO avg | eval.py | 0.2114 | 0.2764 | 0.3437 | 5.6464 | 0.5389 |
| Diffusion-KTO avg | reported | 0.2114 | 0.2764 | 0.3439 | 5.6475 | 0.5326 |
| Diffusion-DPO win% | eval.py | 73.2 | 70.6 | 59.2 | 63.0 | 62.6 |
| Diffusion-DPO win% | reported | 74.0 | 72.0 | 59.0 | 64.4 | 63.4 |
| Diffusion-KTO win% | eval.py | 77.2 | 84.4 | 61.6 | 73.2 | 80.0 |
| Diffusion-KTO win% | reported | 77.6 | 84.2 | 62.8 | 73.6 | 80.0 |

3) Per-prompt |difference| between A and MINE (different generation batch size: 16 vs 32)
| model | metric | mean abs diff | max abs diff | prompts with diff > 1e-3 |
|---|---|---|---|---|
| Orig._SD1.5 | ps | 3.52e-04 | 7.22e-03 | 33/500 |
| Orig._SD1.5 | hps | 5.32e-04 | 9.79e-03 | 56/500 |
| Orig._SD1.5 | clip | 2.09e-03 | 4.16e-02 | 219/500 |
| Orig._SD1.5 | aes | 2.59e-02 | 5.27e-01 | 476/500 |
| Orig._SD1.5 | image_reward | 3.61e-02 | 1.83e+00 | 468/500 |
| Diffusion-DPO | ps | 4.28e-04 | 1.29e-02 | 48/500 |
| Diffusion-DPO | hps | 5.69e-04 | 1.76e-02 | 67/500 |
| Diffusion-DPO | clip | 2.10e-03 | 3.13e-02 | 228/500 |
| Diffusion-DPO | aes | 2.81e-02 | 3.60e-01 | 481/500 |
| Diffusion-DPO | image_reward | 3.11e-02 | 5.87e-01 | 469/500 |
| Diffusion-KTO | ps | 2.87e-04 | 5.54e-03 | 27/500 |
| Diffusion-KTO | hps | 4.85e-04 | 6.87e-03 | 52/500 |
| Diffusion-KTO | clip | 1.59e-03 | 2.13e-02 | 221/500 |
| Diffusion-KTO | aes | 2.01e-02 | 3.21e-01 | 483/500 |
| Diffusion-KTO | image_reward | 2.90e-02 | 8.62e-01 | 461/500 |
