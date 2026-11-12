#!/usr/bin/env python
"""Sana version of eval.py: evaluate fine-tuned Sana transformers against the base model.

Everything follows eval.py except the model loading / generation call (SanaPipeline + transformer
weights instead of StableDiffusionPipeline + UNet weights):
  - prompts: pickapic-anonymous/pickapic_v1, split test_unique (500 prompts), used as they are;
  - one image per prompt and model, the base model and every checkpoint use the same seed;
  - metrics: PickScore, HPS, CLIP, Aesthetics, ImageReward; averages and win rate vs the base model;
  - images are cached under <save_dir>/<model>/<epoch>/seed_<seed>/<prompt_id>.png;
  - per-prompt scores go to <result_dir>/<model>/<epoch>/seed_<seed>/.

--model_type sd15 evaluates Stable Diffusion 1.5 UNet checkpoints instead (e.g. the official
Diffusion-KTO / Diffusion-DPO weights), with the generation call and defaults of eval.py (fp16 pipeline,
safety checker off, guidance 7.5, 50 steps); everything else is shared with the Sana path.

HPSv3 (added at the user's request, on top of the five metrics of eval.py): it needs its own python
environment (it pins another transformers version), so it is computed by eval_hpsv3.py from the saved
images, and eval_summarize.py merges the two and prints the tables. The column written here as hps_v2
is the HPS of eval.py (the HPSv2 model of utils/hps_utils.py); the summary reports it as "hps".

Several seeds can be evaluated in one call (--seeds 0 1 2 3 4); the models are loaded once.
"""
import argparse
import csv
import hashlib
import os
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
BASELINE_NAMES = {"sana": "Orig. Sana", "sd15": "Orig. SD1.5"}
BASE_MODELS = {"sana": "Efficient-Large-Model/Sana_600M_512px_diffusers",
               "sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5"}
METRICS = ("ps", "hps_v2", "clip", "aes", "image_reward")
ORACLE_OF = {"ps": "pickscore", "hps_v2": "hps", "clip": "clip", "aes": "aes", "image_reward": "im"}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Sana transformer checkpoints against the base pipeline")
    p.add_argument("--model_type", type=str, default="sana", choices=["sana", "sd15"])
    p.add_argument("--original_model", type=str, default=None, help="Base pipeline (default depends on --model_type).")
    p.add_argument("--transformer_checkpoints", type=str, nargs="*", default=[],
                   help="sana: folders that contain a 'transformer' subfolder (Diffusers format).")
    p.add_argument("--unet_checkpoints", type=str, nargs="*", default=[],
                   help="sd15: Hugging Face repos or folders that contain a 'unet' subfolder (as in eval.py).")
    p.add_argument("--epochs", type=str, nargs="*", default=[],
                   help="One label per checkpoint (e.g. the training step); used as a folder name.")
    p.add_argument("--names", type=str, nargs="*", default=None,
                   help="Display names, length 1 + number of checkpoints; index 0 is the base model.")
    p.add_argument("--num_samples", type=int, default=None, help="Number of prompts (default: all 500).")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--guidance_scale", type=float, default=None, help="Default: 4.5 for Sana, 7.5 for SD1.5 (eval.py).")
    p.add_argument("--num_inference_steps", type=int, default=None, help="Default: 20 for Sana, 50 for SD1.5 (eval.py).")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--save_dir", type=str, default="eval_images")
    p.add_argument("--result_dir", type=str, default="eval_result_rebuttal")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def sanitize_path_component(component: str) -> str:
    return re.sub(r"[^\w\-.]", "_", component)


def prompt_to_id(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]


def load_test_prompts():
    """Captions of pickapic_v1 test_unique, in dataset order (only that split is downloaded)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    repo = "pickapic-anonymous/pickapic_v1"
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                   if f.startswith("data/test_unique-"))
    prompts = []
    for f in files:
        prompts.extend(pq.read_table(hf_hub_download(repo, f, repo_type="dataset"), columns=["caption"])["caption"].to_pylist())
    return prompts


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    os.chdir(REPO_ROOT)                     # the metric wrappers use repo-relative paths
    sys.path.insert(0, str(REPO_ROOT))
    args.save_dir = os.path.abspath(args.save_dir)
    args.result_dir = os.path.abspath(args.result_dir)
    from diffusers import SanaPipeline, SanaTransformer2DModel
    from train_nash_sana import BatchedScorer, build_oracles

    sd = args.model_type == "sd15"
    ckpts = args.unet_checkpoints if sd else args.transformer_checkpoints
    args.original_model = args.original_model or BASE_MODELS[args.model_type]
    if args.guidance_scale is None:
        args.guidance_scale = 7.5 if sd else 4.5
    if args.num_inference_steps is None:
        args.num_inference_steps = 50 if sd else 20
    if len(args.epochs) != len(ckpts):
        raise ValueError("--epochs needs one label per checkpoint")
    names = [BASELINE_NAMES[args.model_type]] + [f"Model_{i + 1}" for i in range(len(ckpts))]
    if args.names is not None:
        if len(args.names) != len(names):
            raise ValueError(f"--names expects {len(names)} entries (got {len(args.names)}).")
        names = args.names
    epochs = ["base"] + [sanitize_path_component(str(e)) for e in args.epochs]
    M = len(names)

    if sd:
        # as eval.py: fp16 pipeline, safety checker off, fine-tuned UNets swapped in
        from diffusers import StableDiffusionPipeline, UNet2DConditionModel
        pipe = StableDiffusionPipeline.from_pretrained(args.original_model, torch_dtype=torch.float16,
                                                       use_safetensors=True).to(args.device)
        pipe.safety_checker = None
        pipe.set_progress_bar_config(disable=True)
        nets = [pipe.unet] + [UNet2DConditionModel.from_pretrained(p, subfolder="unet", torch_dtype=torch.float16)
                              .to(args.device).eval() for p in ckpts]
        net_attr, size_kwargs = "unet", {}
    else:
        # Same precision set-up as training: transformer and VAE fp32, Gemma text encoder bf16.
        pipe = SanaPipeline.from_pretrained(args.original_model, torch_dtype=torch.float32)
        pipe.text_encoder.to(torch.bfloat16)
        pipe.vae.to(torch.float32)
        pipe.to(args.device)
        pipe.set_progress_bar_config(disable=True)
        nets = [pipe.transformer]
        for p in ckpts:
            sub = "transformer" if (Path(p) / "transformer").exists() else None
            nets.append(SanaTransformer2DModel.from_pretrained(p, subfolder=sub, torch_dtype=torch.float32)
                        .to(args.device).eval())
        net_attr, size_kwargs = "transformer", {"height": args.resolution, "width": args.resolution}
    print(f"Loaded {M} model(s): {names}", flush=True)

    scorer = BatchedScorer(build_oracles(list(ORACLE_OF.values()), torch.device(args.device)))

    prompts = load_test_prompts()
    if args.num_samples is not None:
        prompts = prompts[:args.num_samples]
    print(f"{len(prompts)} test prompts", flush=True)

    from PIL import Image

    def image_path(i, seed, prompt):
        return os.path.join(args.save_dir, sanitize_path_component(names[i]), epochs[i], f"seed_{seed}",
                            f"{prompt_to_id(prompt)}.png")

    for seed in args.seeds:
        rows = [[] for _ in range(M)]
        totals = {m: [0.0] * M for m in METRICS}
        wins = {m: [0] * M for m in METRICS}
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start:start + args.batch_size]
            images = [[None] * len(batch) for _ in range(M)]
            for i in range(M):
                need = []
                for j, prompt in enumerate(batch):
                    try:
                        images[i][j] = Image.open(image_path(i, seed, prompt)).convert("RGB")
                    except Exception:
                        need.append(j)
                if need:
                    setattr(pipe, net_attr, nets[i])
                    # one generator per image, all with the same seed (as eval.py)
                    gens = [torch.Generator(device=args.device).manual_seed(seed) for _ in need]
                    out = pipe(prompt=[batch[j] for j in need], generator=gens,
                               guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps,
                               **size_kwargs).images
                    for k, j in enumerate(need):
                        path = image_path(i, seed, batch[j])
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        out[k].save(path)
                        images[i][j] = out[k]
            # scores[oracle] has shape (prompts in batch, M)
            flat = [images[i][j] for j in range(len(batch)) for i in range(M)]
            scores = scorer(flat, batch, M)
            for j, prompt in enumerate(batch):
                for i in range(M):
                    vals = {m: float(scores[ORACLE_OF[m]][j][i]) for m in METRICS}
                    rows[i].append([prompt_to_id(prompt), prompt, names[i], epochs[i], seed, args.guidance_scale,
                                    args.num_inference_steps] + [vals[m] for m in METRICS])
                    for m in METRICS:
                        totals[m][i] += vals[m]
                        if i > 0:   # same tie rule as eval.py: strict for all but image_reward
                            base = float(scores[ORACLE_OF[m]][j][0])
                            wins[m][i] += int(vals[m] >= base) if m == "image_reward" else int(vals[m] > base)
            print(f"seed {seed}: {min(start + args.batch_size, len(prompts))}/{len(prompts)} prompts", flush=True)

        for i in range(M):
            d = os.path.join(args.result_dir, sanitize_path_component(names[i]), epochs[i], f"seed_{seed}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "scores_partial.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["prompt_id", "prompt", "model_name", "epoch", "seed", "guidance_scale",
                            "num_inference_steps"] + list(METRICS))
                w.writerows(rows[i])
        n = len(prompts)
        print(f"\n===== seed {seed}: summary without HPSv3 ({n} prompts) =====")
        for i in range(M):
            print(f"Average - {names[i]:<28}: " + " | ".join(f"{m} {totals[m][i] / n:.4f}" for m in METRICS))
        for i in range(1, M):
            print(f"WinRate vs base - {names[i]:<20}: " + " | ".join(f"{m} {wins[m][i] / n:.2%}" for m in METRICS))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
