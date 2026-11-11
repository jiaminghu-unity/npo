import os
# TODO: change cache dir
os.environ["HF_HOME"] = "/network/rit/lab/wang_lab_cs/jmh/diffusion-nash-main/DiffusionDPO/cache"
os.environ["DIFFUSERS_DISABLE_PROGRESS_BAR"] = "1"
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import argparse
from typing import List

import torch
import hashlib
import csv
import re
from datasets import load_dataset
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from tqdm import tqdm
from PIL import Image

# === Metric selectors ===
from utils.pickscore_utils import Selector as PS_Selector
from utils.hps_utils import Selector as HPS_Selector
from utils.clip_utils import Selector as CLIP_Selector
from utils.aes_utils import Selector as AES_Selector
import ImageReward as RM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple diffusion models (UNet checkpoints) against a baseline pipeline"
    )
    parser.add_argument(
        "--original_model",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
        help="Hugging Face repo or local path for the BASE pipeline (SDXL or SD1.5)",
    )
    parser.add_argument(
        "--unet_checkpoints",
        type=str,
        nargs="*",
        default=[],
        help=(
            "List of paths to fine-tuned UNet checkpoints. "
            "Each path should contain an 'unet' subfolder (Diffusers format)."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=str,
        nargs="*",
        default=[],
        help=(
            "List of paths to fine-tuned UNet checkpoints. "
            "Each path should contain an 'unet' subfolder (Diffusers format)."
        ),
    )
    parser.add_argument(
        "--names",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional display names. If provided, must have length == 1 + len(unet_checkpoints). "
            "Index 0 corresponds to the baseline pipeline UNet."
        ),
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/mnt/hdfs/test/dataset/pickapic/data",
        help="Path to Pick-a-Pic dataset directory",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of prompts to evaluate (default: all)",
    )
    parser.add_argument(
        "--device_orig",
        type=str,
        default=None,
        help="Device for the original/base pipeline (defaults to --device)",
    )
    parser.add_argument(
        "--device_trained",
        type=str,
        default=None,
        help="Device for trained UNet(s) (defaults to --device)",
    )
    parser.add_argument(
        "--metrics_device",
        type=str,
        default=None,
        help="Device for metric models (defaults to --device_orig or --device)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for determinism")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Guidance scale. Default is 5 for SDXL, 7.5 for SD1.5.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Inference steps. Default is 30 for SDXL, 50 for SD1.5.",
    )
    parser.add_argument(
        "--no_baseline",
        action="store_true",
        help="If set, do not include the baseline UNet; only evaluate provided UNet checkpoints.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="eval_images",
        help="Directory to save/reuse generated images under <unet_name>/<epoch>/<seed>/",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of prompts to process per generation batch.",
    )
    return parser.parse_args()


def is_sdxl(model_id_or_path: str) -> bool:
    key = model_id_or_path.lower()
    return "stable-diffusion-xl" in key or "sdxl" in key


def build_pipeline(base: str, device: str):
    """Load an SDXL or SD1.5 pipeline in fp16 and return it on device."""
    if is_sdxl(base):
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to(device)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            base, torch_dtype=torch.float16, use_safetensors=True
        ).to(device)
    pipe.set_progress_bar_config(disable=True)
    # Disable safety checker for deterministic throughput
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    return pipe


def sanitize_path_component(component: str) -> str:
    """Sanitize a string to be safe for filesystem path components."""
    return re.sub(r"[^\w\-.]", "_", component)


def prompt_to_id(prompt: str) -> str:
    """Deterministic short id for a prompt string for filename purposes."""
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]


def load_unets(checkpoints: List[str], device: str, epochs: List[str]) -> List[UNet2DConditionModel]:
    unets = []
    for p, epoch in zip(checkpoints, epochs):
        if epoch != 'none':
            unet = UNet2DConditionModel.from_pretrained(
                p, subfolder=f'checkpoints/checkpoint-epoch-{epoch}/unet', torch_dtype=torch.float16
            )
        else:
            unet = UNet2DConditionModel.from_pretrained(
                p, subfolder='unet', torch_dtype=torch.float16
            )

        unets.append(unet)
    return unets


def main():
    args = parse_args()
    torch.set_grad_enabled(False)

    # Defaults based on base model type
    if args.guidance_scale is None:
        args.guidance_scale = 5.0 if is_sdxl(args.original_model) else 7.5
    if args.num_inference_steps is None:
        args.num_inference_steps = 30 if is_sdxl(args.original_model) else 50

    # Resolve devices
    device_orig = args.device_orig or args.device
    device_trained = args.device_trained or args.device
    metrics_device = args.metrics_device or device_orig

    # Build pipelines on separate devices (if needed)
    pipe_orig = build_pipeline(args.original_model, device_orig)
    pipe_trained = None
    if len(args.unet_checkpoints) > 0:
        pipe_trained = build_pipeline(args.original_model, device_trained)

    # Build UNet list (baseline first unless --no_baseline)
    names: List[str] = []

    has_baseline = True
    if has_baseline:
        names.append("Orig. SDXL" if is_sdxl(args.original_model) else "Orig. SD1.5")

    extra_unets = load_unets(args.unet_checkpoints, device_trained, args.epochs)

    # Default names for extras
    for i in range(len(extra_unets)):
        names.append(f"Model_{i+1}")

    # If user provided names, validate & override
    if args.names is not None:
        expected = (1 if has_baseline else 0) + len(extra_unets)
        if len(args.names) != expected:
            raise ValueError(
                f"--names expects {expected} entries (got {len(args.names)})."
            )
        names = args.names

    print(f"Loaded {len(extra_unets) + (1 if has_baseline else 0)} model(s).")
    print(f"Model names: {names}")

    # Metric selectors
    ps_selector = PS_Selector(metrics_device)
    hps_selector = HPS_Selector(metrics_device)
    clip_selector = CLIP_Selector(metrics_device)
    aes_selector = AES_Selector(metrics_device)
    image_reward_model = RM.load("ImageReward-v1.0", device=metrics_device, download_root='temp_root')

    gen_orig = torch.Generator(device=device_orig).manual_seed(args.seed)
    gen_trained = torch.Generator(device=device_trained).manual_seed(args.seed)

    def generate_batch(prompts: List[str]):
        """Generate images for a batch of prompts across baseline and extra UNets.
        Returns a list of lists: per-prompt list of images aligned with model order (baseline first if present).
        """
        B = len(prompts)
        base_save_dir = args.save_dir or "eval_images"
        baseline_images = [None] * B if has_baseline else []
        extras_images = [[None] * B for _ in range(len(extra_unets))]

        # Baseline generation/caching
        if has_baseline:
            baseline_display_name = names[0] if len(names) > 0 else "baseline"
            model_dir = sanitize_path_component(baseline_display_name)
            need_indices = []
            out_paths = []
            for j, prompt in enumerate(prompts):
                prompt_id = prompt_to_id(prompt)
                epoch_dir = "base"
                seed_dir = f"seed_{args.seed}"
                out_path = os.path.join(base_save_dir, model_dir, epoch_dir, seed_dir, f"{prompt_id}.png")
                out_paths.append(out_path)
                try:
                    img0 = Image.open(out_path).convert("RGB")
                    baseline_images[j] = img0
                except Exception:
                    need_indices.append(j)
            if len(need_indices) > 0:
                prompts_to_generate = [prompts[j] for j in need_indices]
                # One generator per batch element to preserve determinism vs single-image calls
                gens = [torch.Generator(device=device_orig).manual_seed(args.seed) for _ in prompts_to_generate]
                pipe_orig.to(device_orig)
                imgs_generated = pipe_orig(
                    prompt=prompts_to_generate,
                    generator=gens,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                ).images
                # pipe_orig.to("cpu")
                for k, j in enumerate(need_indices):
                    os.makedirs(os.path.dirname(out_paths[j]), exist_ok=True)
                    imgs_generated[k].save(out_paths[j])
                    baseline_images[j] = imgs_generated[k]

        # Extra UNets generation/caching
        if len(extra_unets) > 0 and pipe_trained is not None:
            for idx, unet in enumerate(extra_unets):
                name_index = idx + (1 if has_baseline else 0)
                display_name = names[name_index] if name_index < len(names) else os.path.basename(os.path.normpath(args.unet_checkpoints[idx]))
                model_dir = sanitize_path_component(display_name)
                epoch_value = str(args.epochs[idx]) if idx < len(args.epochs) else "unknown"
                epoch_dir = sanitize_path_component(epoch_value)
                need_indices = []
                out_paths = []
                for j, prompt in enumerate(prompts):
                    prompt_id = prompt_to_id(prompt)
                    seed_dir = f"seed_{args.seed}"
                    out_path = os.path.join(base_save_dir, model_dir, epoch_dir, seed_dir, f"{prompt_id}.png")
                    out_paths.append(out_path)
                    try:
                        img = Image.open(out_path).convert("RGB")
                        extras_images[idx][j] = img
                    except Exception:
                        need_indices.append(j)
                if len(need_indices) > 0:
                    prompts_to_generate = [prompts[j] for j in need_indices]
                    gens = [torch.Generator(device=device_trained).manual_seed(args.seed) for _ in prompts_to_generate]
                    unet = unet.to(device_trained)
                    pipe_trained.unet = unet
                    imgs_generated = pipe_trained(
                        prompt=prompts_to_generate,
                        generator=gens,
                        guidance_scale=args.guidance_scale,
                        num_inference_steps=args.num_inference_steps,
                    ).images
                    # offload unet to cpu
                    # unet.to("cpu")
                    for k, j in enumerate(need_indices):
                        os.makedirs(os.path.dirname(out_paths[j]), exist_ok=True)
                        imgs_generated[k].save(out_paths[j])
                        extras_images[idx][j] = imgs_generated[k]

        torch.cuda.empty_cache()

        # Assemble per-prompt lists in model order
        batch_imgs = []
        for j in range(B):
            imgs = []
            if has_baseline:
                imgs.append(baseline_images[j])
            for idx in range(len(extra_unets)):
                imgs.append(extras_images[idx][j])
            batch_imgs.append(imgs)
        return batch_imgs

    def generate_all(prompt: str):
        """Return a list of PIL images, one per model across devices."""
        images = []
        prompt_id = prompt_to_id(prompt)
        base_save_dir = args.save_dir or "eval_images"
        if has_baseline:
            baseline_display_name = names[0] if len(names) > 0 else "baseline"
            model_dir = sanitize_path_component(baseline_display_name)
            epoch_dir = "base"
            seed_dir = f"seed_{args.seed}"
            out_path = os.path.join(base_save_dir, model_dir, epoch_dir, seed_dir, f"{prompt_id}.png")
            try:
                img0 = Image.open(out_path).convert("RGB")
            except Exception as e:
                print(f"Failed to load {out_path} due to {e}")
                gen_orig.manual_seed(args.seed)
                pipe_orig.to(device_orig)
                img0 = pipe_orig(
                    prompt=prompt,
                    generator=gen_orig,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                ).images[0]
                # pipe_orig.to("cpu")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                img0.save(out_path)

            images.append(img0)
        if len(extra_unets) > 0 and pipe_trained is not None:
            for idx, unet in enumerate(extra_unets):
                name_index = idx + (1 if has_baseline else 0)
                if name_index < len(names):
                    display_name = names[name_index]
                else:
                    display_name = os.path.basename(os.path.normpath(args.unet_checkpoints[idx]))
                model_dir = sanitize_path_component(display_name)
                epoch_value = str(args.epochs[idx]) if idx < len(args.epochs) else "unknown"
                epoch_dir = sanitize_path_component(epoch_value)
                seed_dir = f"seed_{args.seed}"
                out_path = os.path.join(base_save_dir, model_dir, epoch_dir, seed_dir, f"{prompt_id}.png")
                try:
                    img = Image.open(out_path).convert("RGB")
                except Exception as e:
                    print(f"Failed to load {out_path} due to {e}")
                    unet = unet.to(device_trained)
                    pipe_trained.unet = unet
                    gen_trained.manual_seed(args.seed)
                    img = pipe_trained(
                        prompt=prompt,
                        generator=gen_trained,
                        guidance_scale=args.guidance_scale,
                        num_inference_steps=args.num_inference_steps,
                    ).images[0]
                    # offload the unet to cpu
                    # unet.to("cpu")
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    img.save(out_path)

                images.append(img)
        torch.cuda.empty_cache()
        return images

    # Dataset
    dataset = load_dataset(
        "pickapic-anonymous/pickapic_v1",
        split='test_unique',
        token='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'
    )

    # Accumulators per metric (length = num_models)
    M = (1 if has_baseline else 0) + len(extra_unets)
    totals = {m: [0.0] * M for m in ("ps", "hps", "clip", "aes", "image_reward")}
    wins_vs_baseline = {m: [0] * M for m in ("ps", "hps", "clip", "aes", "image_reward")}
    # Note: wins_vs_baseline[m][0] will count baseline wins if baseline is present

    processed = 0
    prompt_buffer: List[str] = []

    def process_prompt_batch(batch_prompts: List[str]):
        nonlocal processed
        if len(batch_prompts) == 0:
            return
        batch_imgs = generate_batch(batch_prompts)
        base_save_dir = args.save_dir or "eval_images"
        for b_idx, prompt in enumerate(batch_prompts):
            imgs = batch_imgs[b_idx]
            # Scores: each returns a list[float] of length M aligned with imgs
            ps_scores = ps_selector.score(imgs, prompt)
            hps_scores = hps_selector.score(imgs, prompt)
            clip_scores = clip_selector.score(imgs, prompt)
            aes_scores = aes_selector.score(imgs, prompt)
            image_reward_scores = image_reward_model.score(prompt, imgs)

            for i in range(M):
                totals["ps"][i] += ps_scores[i]
                totals["hps"][i] += hps_scores[i]
                totals["clip"][i] += clip_scores[i]
                totals["aes"][i] += aes_scores[i]
                totals["image_reward"][i] += image_reward_scores[i]

            # Persist per-prompt scores for each experiment under the same directory as images
            pid = prompt_to_id(prompt)
            for i in range(M):
                # Derive experiment directory structure
                if has_baseline and i == 0:
                    display_name = names[0] if len(names) > 0 else "baseline"
                    model_dir = sanitize_path_component(display_name)
                    epoch_dir = "base"
                else:
                    unet_index = i - (1 if has_baseline else 0)
                    display_name = names[i] if i < len(names) else f"Model_{i}"
                    model_dir = sanitize_path_component(display_name)
                    epoch_value = str(args.epochs[unet_index]) if unet_index < len(args.epochs) else "unknown"
                    epoch_dir = sanitize_path_component(epoch_value)
                seed_dir = f"seed_{args.seed}"
                exp_dir = os.path.join('eval_result_rebuttal', model_dir, epoch_dir, seed_dir)
                os.makedirs(exp_dir, exist_ok=True)

                csv_path = os.path.join(exp_dir, "scores_pickapic.csv")
                is_new = not os.path.exists(csv_path)
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    if is_new:
                        writer.writerow([
                            "prompt_id",
                            "prompt",
                            "model_name",
                            "epoch",
                            "seed",
                            "guidance_scale",
                            "num_inference_steps",
                            "ps",
                            "hps",
                            "clip",
                            "aes",
                            "image_reward",
                        ])
                    epoch_str = epoch_dir
                    writer.writerow([
                        pid,
                        prompt,
                        display_name,
                        epoch_str,
                        args.seed,
                        args.guidance_scale,
                        args.num_inference_steps,
                        ps_scores[i],
                        hps_scores[i],
                        clip_scores[i],
                        aes_scores[i],
                        image_reward_scores[i],
                    ])

            # Win-rate vs baseline (index 0) only if baseline is present
            if has_baseline and M > 1:
                for i in range(1, M):
                    wins_vs_baseline["ps"][i] += int(ps_scores[i] > ps_scores[0])
                    wins_vs_baseline["hps"][i] += int(hps_scores[i] > hps_scores[0])
                    wins_vs_baseline["clip"][i] += int(clip_scores[i] > clip_scores[0])
                    wins_vs_baseline["aes"][i] += int(aes_scores[i] > aes_scores[0])
                    wins_vs_baseline["image_reward"][i] += int(image_reward_scores[i] >= image_reward_scores[0])
                    # Count baseline wins too (ties count as baseline loss)
                    wins_vs_baseline["ps"][0] += int(ps_scores[0] >= ps_scores[i])
                    wins_vs_baseline["hps"][0] += int(hps_scores[0] >= hps_scores[i])
                    wins_vs_baseline["clip"][0] += int(clip_scores[0] >= clip_scores[i])
                    wins_vs_baseline["aes"][0] += int(aes_scores[0] >= aes_scores[i])
                    wins_vs_baseline["image_reward"][0] += int(image_reward_scores[0] >= image_reward_scores[i])


            processed += 1

            if processed % 32 == 0:
                print("\n===== Running Summary =====")
                print(f"Samples: {processed}")
                for i in range(M):
                    avg_ps = totals["ps"][i] / processed
                    avg_hps = totals["hps"][i] / processed
                    avg_clip = totals["clip"][i] / processed
                    avg_aes = totals["aes"][i] / processed
                    avg_image_reward = totals["image_reward"][i] / processed

                    print(
                        f"Average - {names[i]:<14}: PS {avg_ps:.4f} | HPS {avg_hps:.4f} | CLIP {avg_clip:.4f} | AES {avg_aes:.4f} | Image Reward {avg_image_reward:.4f}"
                    )
                if has_baseline and M > 1:
                    for i in range(1, M):
                        wr_ps = wins_vs_baseline["ps"][i] / processed
                        wr_hps = wins_vs_baseline["hps"][i] / processed
                        wr_clip = wins_vs_baseline["clip"][i] / processed
                        wr_aes = wins_vs_baseline["aes"][i] / processed
                        wr_image_reward = wins_vs_baseline["image_reward"][i] / processed
                        print(
                            f"WinRate vs {names[0]:<14} - {names[i]:<14}: PS {wr_ps:>.2%} | HPS {wr_hps:>.2%} | CLIP {wr_clip:>.2%} | AES {wr_aes:>.2%} | Image Reward {wr_image_reward:>.2%}"
                        )

    bsz = max(1, int(args.batch_size))
    for ex in tqdm(dataset, desc="Evaluating"):
        if args.num_samples is not None and processed >= args.num_samples:
            break
        prompt = ex["caption"]
        prompt_buffer.append(prompt)
        if args.num_samples is not None:
            remain = args.num_samples - processed - len(prompt_buffer) + 0
            target_bsz = min(bsz, max(1, args.num_samples - processed))
        else:
            target_bsz = bsz
        if len(prompt_buffer) >= target_bsz:
            process_prompt_batch(prompt_buffer[:target_bsz])
            prompt_buffer = prompt_buffer[target_bsz:]

    # Flush remaining prompts
    if len(prompt_buffer) > 0 and (args.num_samples is None or processed < args.num_samples):
        if args.num_samples is not None:
            remain = args.num_samples - processed
            if remain > 0:
                process_prompt_batch(prompt_buffer[:remain])
        else:
            process_prompt_batch(prompt_buffer)

    # Final summary
    if processed == 0:
        print("No samples were processed. Please check dataset path and models.")
        return

    print("\n===== Final Summary =====")
    print(f"Samples: {processed}")
    for i in range(M):
        avg_ps = totals["ps"][i] / processed
        avg_hps = totals["hps"][i] / processed
        avg_clip = totals["clip"][i] / processed
        avg_aes = totals["aes"][i] / processed
        avg_image_reward = totals["image_reward"][i] / processed

        print(
            f"Average - {names[i]:<14}: PS {avg_ps:.4f} | HPS {avg_hps:.4f} | CLIP {avg_clip:.4f} | AES {avg_aes:.4f} | Image Reward {avg_image_reward:.4f}"
        )
    if has_baseline and M > 1:
        for i in range(1, M):
            wr_ps = wins_vs_baseline["ps"][i] / processed
            wr_hps = wins_vs_baseline["hps"][i] / processed
            wr_clip = wins_vs_baseline["clip"][i] / processed
            wr_aes = wins_vs_baseline["aes"][i] / processed
            wr_image_reward = wins_vs_baseline["image_reward"][i] / processed
            print(
                f"WinRate vs {names[0]:<14} - {names[i]:<14}: PS {wr_ps:>.2%} | HPS {wr_hps:>.2%} | CLIP {wr_clip:>.2%} | AES {wr_aes:>.2%} | Image Reward {wr_image_reward:>.2%}"
            )


if __name__ == "__main__":
    main()
