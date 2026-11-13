#!/usr/bin/env python3
"""Upload Sana transformer checkpoints from the GCS bucket to Hugging Face, one private repo per checkpoint
(the account's convention: weights at the repo root, repo name starts with the training step).

    ~/hf-venv/bin/python hf_upload.py                # uploads every entry of SPECS that is not there yet
    ~/hf-venv/bin/python hf_upload.py <repo-name>    # only that one

The token is read from the Hugging Face login of this machine (~/.cache/huggingface/token) or HF_TOKEN.
Add finished runs / newly tested checkpoints to SPECS.
"""
import os
import shutil
import subprocess
import sys

from huggingface_hub import HfApi

G = "gs://prism-training-apac/jiaming/npo-sana"
BASE = "Efficient-Large-Model/Sana_600M_512px_diffusers"
TMP = os.path.expanduser("~/hf_upload_tmp")
PROTO = ("500 prompts of Pick-a-Pic v1 `test_unique`, seeds 0-4, guidance 4.5, 20 steps; "
         "win rate = share of prompts scored higher than the original Sana 0.6B")
NPO = ("Diff.-NPO (`newnash`), online: 8 candidates per prompt sampled from the current model with 10 steps, "
       "best / worst candidate form the training pair; 2048 pairs per optimizer step, lr 2.048e-5 "
       "(200 warmup steps), beta 5000, tau/eta = 0.025/0.05, hard update of the previous policy")
DPO = ("offline flow-matching DPO on 294,418 human-labelled Pick-a-Pic v1 pairs; 2048 pairs per optimizer step, "
       "lr 2.048e-5 (200 warmup steps), beta 5000")

ODPO = ("Online-DPO (`online_dpo`): the same online pipeline as the Diff.-NPO average-rank run (8 candidates per prompt "
        "sampled from the current model with 10 steps, best / worst by average rank over PickScore / HPSv2 / CLIP / "
        "Aesthetics / ImageReward, same seed and hyper-parameters); only the loss differs (DPO loss against the original "
        "model, beta 5000)")

# repo name, bucket path of the 'transformer' folder, title, method text, results text
SPECS = [
    ("sana-130-newnash-randjudge", f"{G}/tests/_pinned/randjudge-step130/transformer",
     "Diff.-NPO, random judge, step 130",
     NPO + ". Judge: at every training step one of PickScore / HPSv2 / CLIP / Aesthetics / ImageReward is drawn at random.",
     "Intermediate checkpoint (the run was still training, inside the learning-rate warmup).\n\n"
     "| | PickScore | HPS (v2) | CLIP | Aesthetics | ImageReward | HPSv3 |\n|---|---|---|---|---|---|---|\n"
     "| average score | 0.2222 | 0.2856 | 0.3564 | 6.0498 | 1.1251 | 8.0684 |\n"
     "| original Sana | 0.2218 | 0.2847 | 0.3552 | 6.0293 | 1.0974 | 7.9932 |\n"
     "| win rate (%) | 63.8 | 68.4 | 54.2 | 62.2 | 62.2 | 58.2 |"),
    ("sana-140-newnash-avgrank", f"{G}/tests/_pinned/avgrank-step140/transformer",
     "Diff.-NPO, average-rank judge, step 140",
     NPO + ". Judge: average rank over PickScore / HPSv2 / CLIP / Aesthetics / ImageReward.",
     "Intermediate checkpoint (the run was still training, inside the learning-rate warmup).\n\n"
     "| | PickScore | HPS (v2) | CLIP | Aesthetics | ImageReward | HPSv3 |\n|---|---|---|---|---|---|---|\n"
     "| average score | 0.2224 | 0.2860 | 0.3567 | 6.0573 | 1.1386 | 8.0995 |\n"
     "| original Sana | 0.2218 | 0.2847 | 0.3552 | 6.0293 | 1.0974 | 7.9932 |\n"
     "| win rate (%) | 66.0 | 69.7 | 55.5 | 62.9 | 64.3 | 57.8 |"),
    ("sana-170-newnash-randjudge", f"{G}/tests/_pinned/randjudge-step170/transformer",
     "Diff.-NPO, random judge, step 170",
     NPO + ". Judge: at every training step one of PickScore / HPSv2 / CLIP / Aesthetics / ImageReward is drawn at random.",
     "Intermediate checkpoint (the run was still training, inside the learning-rate warmup).\n\n"
     "| | PickScore | HPS (v2) | CLIP | Aesthetics | ImageReward | HPSv3 |\n|---|---|---|---|---|---|---|\n"
     "| average score | 0.2222 | 0.2857 | 0.3563 | 6.0462 | 1.1240 | 8.0679 |\n"
     "| original Sana | 0.2218 | 0.2847 | 0.3552 | 6.0293 | 1.0974 | 7.9932 |\n"
     "| win rate (%) | 63.4 | 67.6 | 53.6 | 60.5 | 62.3 | 56.8 |"),
    ("sana-170-newnash-avgrank", f"{G}/tests/_pinned/avgrank-step170/transformer",
     "Diff.-NPO, average-rank judge, step 170",
     NPO + ". Judge: average rank over PickScore / HPSv2 / CLIP / Aesthetics / ImageReward.",
     "Intermediate checkpoint (the run was still training, inside the learning-rate warmup).\n\n"
     "| | PickScore | HPS (v2) | CLIP | Aesthetics | ImageReward | HPSv3 |\n|---|---|---|---|---|---|---|\n"
     "| average score | 0.2224 | 0.2862 | 0.3570 | 6.0566 | 1.1408 | 8.1179 |\n"
     "| original Sana | 0.2218 | 0.2847 | 0.3552 | 6.0293 | 1.0974 | 7.9932 |\n"
     "| win rate (%) | 65.7 | 71.0 | 56.0 | 62.4 | 65.3 | 60.2 |"),
    ("sana-200-newnash-randjudge", f"{G}/tests/_pinned/randjudge-step200/transformer",
     "Diff.-NPO, random judge, step 200",
     NPO + ". Judge: at every training step one of PickScore / HPSv2 / CLIP / Aesthetics / ImageReward is drawn at random.",
     "Intermediate checkpoint (end of the learning-rate warmup; the run was still training).\n\n"
     "| | PickScore | HPS (v2) | CLIP | Aesthetics | ImageReward | HPSv3 |\n|---|---|---|---|---|---|---|\n"
     "| average score | 0.2223 | 0.2859 | 0.3567 | 6.0486 | 1.1377 | 8.1117 |\n"
     "| original Sana | 0.2218 | 0.2847 | 0.3552 | 6.0293 | 1.0974 | 7.9932 |\n"
     "| win rate (%) | 61.4 | 68.9 | 55.0 | 59.5 | 65.4 | 58.9 |"),
    ("sana-200-newnash-avgrank", f"{G}/tests/_pinned/avgrank-step200/transformer",
     "Diff.-NPO, average-rank judge, step 200",
     NPO + ". Judge: average rank over PickScore / HPSv2 / CLIP / Aesthetics / ImageReward.",
     "Intermediate checkpoint (end of the learning-rate warmup; the run was still training).\n\n"
     "| | PickScore | HPS (v2) | CLIP | Aesthetics | ImageReward | HPSv3 |\n|---|---|---|---|---|---|---|\n"
     "| average score | 0.2225 | 0.2862 | 0.3567 | 6.0607 | 1.1397 | 8.1241 |\n"
     "| original Sana | 0.2218 | 0.2847 | 0.3552 | 6.0293 | 1.0974 | 7.9932 |\n"
     "| win rate (%) | 66.2 | 71.3 | 53.9 | 62.4 | 64.8 | 59.8 |"),
    ("sana-190-newnash-randjudge", f"{G}/tests/_pinned/randjudge-step190/transformer",
     "Diff.-NPO, random judge, step 190",
     NPO + ". Judge: at every training step one of PickScore / HPSv2 / CLIP / Aesthetics / ImageReward is drawn at random.",
     "Intermediate checkpoint (the run was still training).\n\n"
     "| | PickScore | HPS (v2) | CLIP | Aesthetics | ImageReward | HPSv3 |\n|---|---|---|---|---|---|---|\n"
     "| average score | 0.2223 | 0.2859 | 0.3567 | 6.0480 | 1.1321 | 8.1066 |\n"
     "| original Sana | 0.2218 | 0.2847 | 0.3552 | 6.0293 | 1.0974 | 7.9932 |\n"
     "| win rate (%) | 63.2 | 69.2 | 56.1 | 60.8 | 64.3 | 59.1 |"),
    ("sana-210-newnash-randjudge", f"{G}/tests/_pinned/randjudge-step210/transformer",
     "Diff.-NPO, random judge, step 210",
     NPO + ". Judge: at every training step one of PickScore / HPSv2 / CLIP / Aesthetics / ImageReward is drawn at random.",
     "Intermediate checkpoint (the run was still training).\n\n"
     "| | PickScore | HPS (v2) | CLIP | Aesthetics | ImageReward | HPSv3 |\n|---|---|---|---|---|---|---|\n"
     "| average score | 0.2223 | 0.2857 | 0.3563 | 6.0536 | 1.1257 | 8.0713 |\n"
     "| original Sana | 0.2218 | 0.2847 | 0.3552 | 6.0293 | 1.0974 | 7.9932 |\n"
     "| win rate (%) | 65.3 | 67.1 | 54.3 | 61.7 | 61.4 | 56.3 |"),
    ("sana-190-newnash-avgrank", f"{G}/tests/_pinned/avgrank-step190/transformer",
     "Diff.-NPO, average-rank judge, step 190",
     NPO + ". Judge: average rank over PickScore / HPSv2 / CLIP / Aesthetics / ImageReward.",
     "Intermediate checkpoint (the run was still training).\n\n"
     "| | PickScore | HPS (v2) | CLIP | Aesthetics | ImageReward | HPSv3 |\n|---|---|---|---|---|---|---|\n"
     "| average score | 0.2225 | 0.2861 | 0.3567 | 6.0584 | 1.1342 | 8.1096 |\n"
     "| original Sana | 0.2218 | 0.2847 | 0.3552 | 6.0293 | 1.0974 | 7.9932 |\n"
     "| win rate (%) | 67.2 | 69.8 | 54.3 | 63.6 | 64.0 | 59.4 |"),
    ("sana-210-newnash-avgrank", f"{G}/tests/_pinned/avgrank-step210/transformer",
     "Diff.-NPO, average-rank judge, step 210",
     NPO + ". Judge: average rank over PickScore / HPSv2 / CLIP / Aesthetics / ImageReward.",
     "Intermediate checkpoint (the run was still training).\n\n"
     "| | PickScore | HPS (v2) | CLIP | Aesthetics | ImageReward | HPSv3 |\n|---|---|---|---|---|---|---|\n"
     "| average score | 0.2225 | 0.2862 | 0.3568 | 6.0593 | 1.1409 | 8.1142 |\n"
     "| original Sana | 0.2218 | 0.2847 | 0.3552 | 6.0293 | 1.0974 | 7.9932 |\n"
     "| win rate (%) | 67.0 | 70.2 | 54.9 | 62.0 | 64.4 | 59.1 |"),
    ("sana-220-newnash-randjudge", f"{G}/tests/_pinned/randjudge-step220/transformer",
     "Diff.-NPO, random judge, step 220",
     NPO + ". Judge: at every training step one of PickScore / HPSv2 / CLIP / Aesthetics / ImageReward is drawn at random.",
     "This step was not tested. The tests of the same run at steps 170, 190, 200 and 210 were flat; step 200: win rate (%) PickScore 61.4, HPS (v2) 68.9, CLIP 55.0, Aesthetics 59.5, ImageReward 65.4, HPSv3 58.9."),
    ("sana-224-newnash-randjudge", f"{G}/tests/_pinned/randjudge-final-step224/transformer",
     "Diff.-NPO, random judge, step 224 (last checkpoint)",
     NPO + ". Judge: at every training step one of PickScore / HPSv2 / CLIP / Aesthetics / ImageReward is drawn at random.",
     "Last checkpoint: the run was stopped at step 225 because its test results had been flat since step 170. This step was not tested. The tests of the same run at steps 170, 190, 200 and 210 were flat; step 200: win rate (%) PickScore 61.4, HPS (v2) 68.9, CLIP 55.0, Aesthetics 59.5, ImageReward 65.4, HPSv3 58.9."),
    ("sana-220-newnash-avgrank", f"{G}/tests/_pinned/avgrank-step220/transformer",
     "Diff.-NPO, average-rank judge, step 220 (best training-time eval)",
     NPO + ". Judge: average rank over PickScore / HPSv2 / CLIP / Aesthetics / ImageReward.",
     "Step 220 had the best training-time evaluation of the run. This step was not tested. The tests of the same run at steps 170, 190, 200 and 210 were flat; step 200: win rate (%) PickScore 66.2, HPS (v2) 71.3, CLIP 53.9, Aesthetics 62.4, ImageReward 64.8, HPSv3 59.8."),
    ("sana-226-newnash-avgrank", f"{G}/tests/_pinned/avgrank-final-step226/transformer",
     "Diff.-NPO, average-rank judge, step 226 (last checkpoint)",
     NPO + ". Judge: average rank over PickScore / HPSv2 / CLIP / Aesthetics / ImageReward.",
     "Last checkpoint: the run was stopped at step 227 because its test results had been flat since step 170. This step was not tested. The tests of the same run at steps 170, 190, 200 and 210 were flat; step 200: win rate (%) PickScore 66.2, HPS (v2) 71.3, CLIP 53.9, Aesthetics 62.4, ImageReward 64.8, HPSv3 59.8."),
    ("sana-70-onlinedpo-avgrank", f"{G}/tests/_pinned/onlinedpo-step70/transformer",
     "Online-DPO, average-rank judge, step 70",
     ODPO + ".",
     "Intermediate checkpoint (the run was still training, inside the learning-rate warmup).\n\n"
     "| | PickScore | HPS (v2) | CLIP | Aesthetics | ImageReward | HPSv3 |\n|---|---|---|---|---|---|---|\n"
     "| average score | 0.2220 | 0.2853 | 0.3559 | 6.0411 | 1.1168 | 8.0324 |\n"
     "| original Sana | 0.2218 | 0.2847 | 0.3552 | 6.0293 | 1.0974 | 7.9932 |\n"
     "| win rate (%) | 61.1 | 67.0 | 53.0 | 58.8 | 62.4 | 54.5 |"),
    ("sana-270-dpo-offline", f"{G}/tests/_pinned/dpo-best-270/transformer",
     "offline DPO, step 270 (best training-time eval)",
     DPO + ". Step 270 had the best training-time evaluation of the run (420 steps, early stop).",
     "| | PickScore | HPS (v2) | CLIP | Aesthetics | ImageReward | HPSv3 |\n|---|---|---|---|---|---|---|\n"
     "| average score | 0.2220 | 0.2849 | 0.3553 | 6.0321 | 1.1004 | 8.0567 |\n"
     "| original Sana | 0.2218 | 0.2847 | 0.3552 | 6.0293 | 1.0974 | 7.9932 |\n"
     "| win rate (%) | 59.0 | 56.6 | 50.8 | 51.0 | 53.0 | 54.8 |"),
    ("sana-420-dpo-offline", f"{G}/sana-dpo-offline-v1/final/transformer",
     "offline DPO, step 420 (final)",
     DPO + ". Final weights: the run stopped early at step 420 (15 evaluations without a new best).",
     "This step was not tested. Step 380 of the same run: win rate (%) PickScore 56.9, HPS (v2) 54.1, CLIP 50.8, "
     "Aesthetics 50.6, ImageReward 54.8, HPSv3 55.9."),
]

CARD = """---
library_name: diffusers
base_model: {base}
tags:
- text-to-image
- sana
- flow-matching
---
# Sana 0.6B transformer: {title}

Transformer weights (fp32) fine-tuned from `{base}`; everything else (VAE, text encoder, scheduler) is the
original pipeline.

**Training.** {method}

## Use

```python
import torch
from diffusers import SanaPipeline, SanaTransformer2DModel

pipe = SanaPipeline.from_pretrained("{base}", torch_dtype=torch.float32)
pipe.transformer = SanaTransformer2DModel.from_pretrained("{repo}", torch_dtype=torch.float32)
pipe.text_encoder.to(torch.bfloat16)
pipe.to("cuda")
image = pipe("a photo of a cat", guidance_scale=4.5, num_inference_steps=20, height=512, width=512).images[0]
```

## Test

{proto}.

{results}
"""


def main():
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    user = api.whoami()["name"]
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, src, title, method, results in SPECS:
        if only and name != only:
            continue
        repo = f"{user}/{name}"
        if api.repo_exists(repo) and "diffusion_pytorch_model.safetensors" in api.list_repo_files(repo):
            print(f"SKIP {repo}: already uploaded", flush=True)
            continue
        d = os.path.join(TMP, name)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        subprocess.run(["gcloud", "storage", "cp", "--quiet", f"{src}/*", d + "/"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        files = sorted(os.listdir(d))
        assert files == ["config.json", "diffusion_pytorch_model.safetensors"], files
        with open(os.path.join(d, "README.md"), "w") as f:
            f.write(CARD.format(base=BASE, title=title, method=method, repo=repo, proto=PROTO, results=results))
        api.create_repo(repo, private=True, exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=d, commit_message=f"Sana 0.6B: {title}")
        size = os.path.getsize(os.path.join(d, "diffusion_pytorch_model.safetensors"))
        print(f"DONE {repo}: {size / 1e9:.2f} GB, private, files {api.list_repo_files(repo)}", flush=True)
        shutil.rmtree(d, ignore_errors=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
