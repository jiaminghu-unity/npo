#!/usr/bin/env python
# coding=utf-8
"""Online Nash preference optimization ("newnash") for flow-matching models (Sana).

This is the flow-matching port of the `newnash` branch of train_nash.py.

Per optimizer step (repeated `gradient_accumulation_steps` times):
  1. sample a batch of prompts;
  2. pi_prev generates K candidate images per prompt (different initial noise per candidate,
     because the flow ODE sampler is deterministic);
  3. all reward models score all candidates (logged); a judge decides winner / loser:
       --nash_judge_mode random   : ONE reward model, drawn at random, ranks the candidates
       --nash_judge_mode avg_rank : average rank over all reward models (old behaviour)
  4. the pair (y_w, y_l) gets the same timestep and the same noise, and we compute the
     flow-matching error  L = || v_theta(x_t, t) - (eps - x_0) ||^2  under pi, pi_ref, pi_prev:
       log_ratio      = -0.5 * (L_w - L_l)
       h_t            = log_ratio - (tau/eta) * log_ratio_ref - ((eta-tau)/eta) * log_ratio_prev
       loss (newnash) = -log sigmoid(beta * h_t)
  5. pi_prev is refreshed from pi every `nash_prev_update_steps` optimizer steps.

All randomness (prompts, initial noise, judge, timesteps) is derived from (seed, global_step,
micro_step), so a run resumed from a checkpoint continues exactly where it stopped.
"""

import argparse
import copy
import inspect
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent
SUPPORTED_ORACLES = ("pickscore", "hps", "clip", "aes", "im")


# --------------------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="newnash for flow-matching models (Sana)")
    # model
    p.add_argument("--pretrained_model_name_or_path", type=str,
                   default="Efficient-Large-Model/Sana_600M_512px_diffusers")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--max_sequence_length", type=int, default=300)
    p.add_argument("--cache_dir", type=str, default=None)

    # prompts
    p.add_argument("--prompts_file", type=str, default=None,
                   help="Optional .txt (one prompt per line) or .json (list). Overrides --dataset_name.")
    p.add_argument("--dataset_name", type=str, default="CarperAI/pickapic_v1_no_images_training_sfw")
    p.add_argument("--caption_column", type=str, default="caption")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--max_prompts", type=int, default=None)
    p.add_argument("--max_prompt_chars", type=int, default=0, help="If >0, drop longer training prompts.")
    p.add_argument("--dedupe_prompts", action="store_true",
                   help="Train on unique prompts. Default (as in train_nash.py): one entry per dataset row, "
                        "so frequent prompts are drawn more often.")
    p.add_argument("--eval_prompts_file", type=str, default=None)
    p.add_argument("--eval_dataset_name", type=str, default="nateraw/parti-prompts")
    p.add_argument("--eval_caption_column", type=str, default="Prompt")
    p.add_argument("--eval_split", type=str, default="train")
    p.add_argument("--eval_num_prompts", type=int, default=128)

    # objective
    p.add_argument("--train_method", type=str, default="newnash",
                   choices=["newnash", "nash", "online_dpo", "dpo"],
                   help="newnash / nash / online_dpo sample their own pairs online. dpo is the offline "
                        "Diffusion-DPO baseline: it trains on human-labelled image pairs (--pairs_dir).")
    p.add_argument("--pairs_dir", type=str, default="~/npo_data/pickapic_pairs",
                   help="dpo only: local folder with the Arrow shards made by prepare_pickapic_pairs.py.")
    p.add_argument("--pairs_gcs", type=str, default=None,
                   help="dpo only: gs:// prefix to download --pairs_dir from if it is not there yet.")
    p.add_argument("--dataloader_num_workers", type=int, default=10)
    p.add_argument("--beta_dpo", type=float, default=5000.0,
                   help="Temperature inside the sigmoid (newnash / online_dpo).")
    p.add_argument("--nash_tau", type=float, default=0.1, help="INPO tau (KL weight toward pi_ref).")
    p.add_argument("--nash_eta", type=float, default=1.0, help="INPO eta (OMD stepsize).")
    p.add_argument("--nash_prev_update_steps", type=int, default=1,
                   help="Refresh pi_prev from pi every this many optimizer steps.")
    p.add_argument("--nash_prev_hard_update", action="store_true",
                   help="Copy pi into pi_prev. Default: soft update prev <- mu*prev + (1-mu)*pi, "
                        "mu = min(0.001*step, 0.5) (same as train_nash.py).")
    p.add_argument("--latent_source", type=str, default="reencode", choices=["reencode", "direct"],
                   help="reencode (as in train_nash.py): winner/loser images go through the training image "
                        "transforms and the VAE encoder. direct: use the sampler's final latents.")
    p.add_argument("--no_hflip", action="store_true",
                   help="Disable the random horizontal flip of the winner/loser images (reencode only).")
    p.add_argument("--timesteps_per_pair", type=int, default=1,
                   help="How many (timestep, noise) draws each winner/loser pair is trained on per step. "
                        "1 = same as train_nash.py. Larger values are nearly free (sampling dominates the cost).")
    p.add_argument("--proportion_empty_prompts", type=float, default=0.2,
                   help="Probability of training a pair with the empty prompt (keeps CFG working).")

    # online sampling
    p.add_argument("--nash_num_candidates", type=int, default=8)
    p.add_argument("--nash_num_inference_steps", type=int, default=20,
                   help="Sampler steps for the online candidates (train_nash.py used 10 DDIM steps).")
    p.add_argument("--nash_guidance_scale", type=float, default=4.5)
    p.add_argument("--sample_autocast", action="store_true",
                   help="Run the online sampler under bf16 autocast (faster, slightly different images).")

    # judges
    p.add_argument("--nash_oracles", type=str, default="pickscore,hps,clip,aes,im",
                   help=f"Comma-separated reward models. Supported: {','.join(SUPPORTED_ORACLES)}")
    p.add_argument("--nash_judge_mode", type=str, default="random", choices=["random", "avg_rank"],
                   help="random: one reward model drawn at random decides winner/loser. "
                        "avg_rank: average rank over all reward models.")
    p.add_argument("--nash_judge_granularity", type=str, default="step",
                   choices=["step", "microbatch", "prompt"],
                   help="How often a new random judge is drawn: once per optimizer step (default), "
                        "once per micro-batch, or once per prompt.")

    # optimisation
    p.add_argument("--train_batch_size", type=int, default=8, help="Prompts (= pairs) per micro-batch.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_train_steps", type=int, default=2000)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--scale_lr", action="store_true",
                   help="Scale the learning rate by gradient accumulation steps * batch size * number of GPUs "
                        "(same rule as train_nash.py).")
    p.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    p.add_argument("--lr_warmup_steps", type=int, default=500)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_weight_decay", type=float, default=1e-2)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=50.0)
    p.add_argument("--train_autocast", action="store_true",
                   help="Run training forward passes under bf16 autocast. Default is fp32, because the "
                        "DPO signal is a small difference of errors.")
    p.add_argument("--weighting_scheme", type=str, default="none",
                   choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
                   help="Timestep sampling density (same options as the diffusers Sana trainer).")
    p.add_argument("--logit_mean", type=float, default=0.0)
    p.add_argument("--logit_std", type=float, default=1.0)
    p.add_argument("--mode_scale", type=float, default=1.29)

    # bookkeeping
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="tmp-sana-newnash")
    p.add_argument("--gcs_dir", type=str, default=None,
                   help="Optional gs://... prefix; checkpoints and logs are mirrored there.")
    p.add_argument("--resume", type=str, default="auto", choices=["auto", "none"])
    p.add_argument("--checkpointing_steps", type=int, default=10)
    p.add_argument("--keep_checkpoints", type=int, default=2)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--early_stop_patience", type=int, default=0,
                   help="If >0: stop when this many consecutive evals bring no new best gain "
                        "(covers both convergence and collapse; the best weights are kept in <output>/best).")
    p.add_argument("--early_stop_min_steps", type=int, default=0,
                   help="Never early-stop before this step (e.g. the end of the learning-rate warmup).")
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--eval_num_inference_steps", type=int, default=20)
    p.add_argument("--save_candidate_images_steps", type=int, default=0,
                   help="If >0, save the candidates of the first prompt every N steps (debugging).")
    args = p.parse_args()

    names = [s.strip().lower() for s in args.nash_oracles.split(",") if s.strip()]
    unknown = [n for n in names if n not in SUPPORTED_ORACLES]
    if unknown or not names:
        raise ValueError(f"Bad --nash_oracles {args.nash_oracles!r}; supported: {SUPPORTED_ORACLES}")
    args.oracle_names = names
    if args.nash_num_candidates < 2:
        raise ValueError("--nash_num_candidates must be >= 2")
    return args


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------
RANK = 0


def log(msg):
    tag = f"[rank {RANK}] " if RANK else ""
    print(f"[{time.strftime('%m/%d %H:%M:%S')}] {tag}{msg}", flush=True)


def mix_seed(*parts):
    """Deterministic 63-bit seed from integers (stable across processes and restarts)."""
    state = np.random.SeedSequence([int(x) & 0xFFFFFFFF for x in parts]).generate_state(1, dtype=np.uint64)
    return int(state[0]) >> 1


def run_cmd(cmd, check=False):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            log(f"[WARN] command failed ({r.returncode}): {' '.join(cmd)}\n{r.stderr[-800:]}")
            if check:
                raise RuntimeError("command failed")
        return r
    except FileNotFoundError as e:
        log(f"[WARN] {e}")
        if check:
            raise
        return None


def gcs_upload(local, remote):
    """Mirror a local file/dir to GCS (best effort)."""
    local = str(local)
    if os.path.isdir(local):
        return run_cmd(["gcloud", "storage", "rsync", "-r", "--quiet", local, remote])
    return run_cmd(["gcloud", "storage", "cp", "--quiet", local, remote])


def ensure_utils_package():
    """The reward-model wrappers import each other as a `utils` package (utils/hps_utils.py, ...).
    In this repo they sit at the top level, so build a `utils/` package of symlinks if needed."""
    utils_dir = REPO_ROOT / "utils"
    if (utils_dir / "hps_utils.py").exists():
        return
    utils_dir.mkdir(exist_ok=True)
    (utils_dir / "__init__.py").touch()
    for name in ["hps_utils.py", "clip_utils.py", "pickscore_utils.py", "aes_utils.py",
                 "tokenizer_hps.py", "hps_bpe.txt.gz", "open_clip", "aesthetics_model"]:
        src, dst = REPO_ROOT / name, utils_dir / name
        if src.exists() and not dst.exists():
            os.symlink(os.path.relpath(src, utils_dir), dst)
    log(f"built reward-model package at {utils_dir}")


def load_prompt_list(prompts_file, dataset_name, column, split, cache_dir, max_chars, max_prompts, seed,
                     dedupe=True):
    if prompts_file:
        text = Path(prompts_file).read_text()
        prompts = json.loads(text) if prompts_file.endswith(".json") else text.splitlines()
    else:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        prompts = ds[column]
        if "pickapic" in dataset_name and "label_0" in ds.column_names:
            # same filter as train_nash.py: eliminate no-decisions (0.5-0.5 labels)
            keep = [l in (0, 1) for l in ds["label_0"]]
            log(f"Eliminated {len(keep) - sum(keep)}/{len(keep)} split decisions for Pick-a-pic")
            prompts = [c for c, k in zip(prompts, keep) if k]
    seen, out = set(), []
    for c in prompts:
        if not isinstance(c, str):
            continue
        c = c.strip()
        if not c or (max_chars and len(c) > max_chars) or (dedupe and c in seen):
            continue
        seen.add(c)
        out.append(c)
    out.sort()
    random.Random(seed).shuffle(out)
    if max_prompts:
        out = out[:max_prompts]
    return out


# --------------------------------------------------------------------------------------
# Offline human-labelled pairs (Diffusion-DPO baseline)
# --------------------------------------------------------------------------------------
def ensure_pairs(args):
    root = Path(os.path.expanduser(args.pairs_dir))
    if not (root / ".complete").exists():
        if not args.pairs_gcs:
            raise RuntimeError(f"{root} has no prepared pairs and --pairs_gcs is not set")
        root.mkdir(parents=True, exist_ok=True)
        log(f"downloading the image pairs from {args.pairs_gcs} to {root} ...")
        t0 = time.time()
        run_cmd(["gcloud", "storage", "rsync", "-r", "--quiet", args.pairs_gcs, str(root)], check=True)
        (root / ".complete").touch()
        log(f"pairs downloaded in {time.time() - t0:.0f}s")
    return root


class PairData:
    """Rows (caption, jpg_0, jpg_1, label_0) stored as memory-mapped Arrow IPC files."""

    def __init__(self, root):
        manifest = json.loads((Path(root) / "manifest.json").read_text())
        self.files = [(str(Path(root) / e["file"]), e["rows"]) for e in manifest["files"] if e["rows"] > 0]
        self.offsets = np.cumsum([0] + [r for _, r in self.files])
        self.n = int(self.offsets[-1])
        self._tables = {}

    def row(self, i):
        import pyarrow as pa
        fi = int(np.searchsorted(self.offsets, i, side="right") - 1)
        if fi not in self._tables:   # opened lazily in each DataLoader worker
            self._tables[fi] = pa.ipc.open_file(pa.memory_map(self.files[fi][0], "r")).read_all()
        return self._tables[fi].slice(i - int(self.offsets[fi]), 1).to_pylist()[0]


class PairBatches(torch.utils.data.Dataset):
    """Item k is the k-th micro-batch of this GPU after `start_step`: (pixels (B, 2, 3, H, W), captions),
    with the human-preferred image first. The dataset is read in shuffled epochs (a fixed permutation per
    epoch), and the position only depends on (step, micro-batch, rank), so a resumed run continues the
    same data order. Image transforms are those of train.py: Resize, CenterCrop, RandomHorizontalFlip
    (each image independently), ToTensor, Normalize."""

    def __init__(self, data, args, rank, world_size, start_step):
        self.data, self.args, self.rank, self.world_size, self.start_step = data, args, rank, world_size, start_step
        self._perm = {}

    def __len__(self):
        return max(self.args.max_train_steps - self.start_step, 0) * self.args.gradient_accumulation_steps

    def _index(self, pos):
        epoch, off = divmod(pos, self.data.n)
        if epoch not in self._perm:
            self._perm = {epoch: np.random.default_rng(mix_seed(self.args.seed, 8, epoch)).permutation(self.data.n)}
        return int(self._perm[epoch][off])

    def __getitem__(self, k):
        import io
        from PIL import Image
        from torchvision import transforms
        from torchvision.transforms import functional as TF
        a = self.args
        step = self.start_step + k // a.gradient_accumulation_steps
        micro = k % a.gradient_accumulation_steps
        pos0 = ((step * a.gradient_accumulation_steps + micro) * self.world_size + self.rank) * a.train_batch_size
        rng = random.Random(mix_seed(a.seed, 6, step, micro, self.rank))
        base = transforms.Compose([
            transforms.Resize(a.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(a.resolution)])

        def load(b):
            im = base(Image.open(io.BytesIO(b)).convert("RGB"))
            if not a.no_hflip and rng.random() < 0.5:
                im = TF.hflip(im)
            return TF.normalize(TF.to_tensor(im), [0.5], [0.5])

        pixels, captions = [], []
        for p in range(pos0, pos0 + a.train_batch_size):
            r = self.data.row(self._index(p))
            im0, im1 = load(r["jpg_0"]), load(r["jpg_1"])
            pixels.append(torch.stack([im0, im1] if r["label_0"] == 1 else [im1, im0]))
            captions.append(r["caption"])
        return torch.stack(pixels), captions


# --------------------------------------------------------------------------------------
# Reward models ("oracles")
# --------------------------------------------------------------------------------------
class ImageRewardSelector:
    def __init__(self, device, cache_dir=None):
        import ImageReward as RM
        self.device = device
        self.model = RM.load("ImageReward-v1.0", device=device, download_root=cache_dir)

    def score(self, images, prompt):
        out = self.model.score(prompt, images)
        return out if isinstance(out, (list, tuple)) else [out]


def build_oracles(names, device, cache_dir=None):
    ensure_utils_package()
    try:  # open_clip/factory.py has a stray `from turtle import forward`, which needs tkinter
        import tkinter  # noqa: F401
    except ImportError:
        import types
        stub = types.ModuleType("turtle")
        stub.forward = None
        sys.modules.setdefault("turtle", stub)
    selectors = {}
    for name in names:
        t0 = time.time()
        if name == "hps":
            from utils.hps_utils import Selector as _Sel
            sel = _Sel(device)
        elif name == "clip":
            from utils.clip_utils import Selector as _Sel
            sel = _Sel(device)
        elif name == "pickscore":
            from utils.pickscore_utils import Selector as _Sel
            sel = _Sel(device)
        elif name == "aes":
            from utils.aes_utils import Selector as _Sel
            sel = _Sel(device)
        elif name == "im":
            sel = ImageRewardSelector(device, cache_dir)
        else:
            raise RuntimeError("unreachable")
        selectors[name] = sel
        log(f"loaded reward model '{name}' in {time.time() - t0:.1f}s")
    return selectors


class BatchedScorer:
    """Scores many (image, prompt) pairs per forward pass instead of one image at a time.

    Same models, same preprocessing and same formulas as the Selector classes; only the batching
    differs. The first call checks each oracle against its original `Selector.score` and falls
    back to the original (slow) path for any oracle whose batched numbers do not match."""

    TOL = {"pickscore": 2e-3, "hps": 2e-3, "clip": 2e-3, "aes": 2e-2, "im": 2e-2}

    def __init__(self, selectors, num_threads=8):
        from concurrent.futures import ThreadPoolExecutor
        self.selectors = selectors
        self.pool = ThreadPoolExecutor(num_threads)
        self.verified = {}

    def _prep(self, fn, pils, device):
        return torch.stack(list(self.pool.map(fn, pils))).to(device)

    def _fast(self, name, pils, prompts, k):
        sel = self.selectors[name]
        dev = sel.device
        b = len(prompts)
        if name in ("hps", "clip"):
            img = self._prep(sel.preprocess_val, pils, dev)
            txt = sel.tokenizer(prompts).to(dev)
            out = sel.model(img, txt)
            f_i, f_t = out["image_features"].float(), out["text_features"].float()
            s = (f_i.view(b, k, -1) * f_t[:, None, :]).sum(-1)
        elif name == "pickscore":
            ii = sel.processor(images=pils, padding=True, truncation=True, max_length=77, return_tensors="pt").to(dev)
            ti = sel.processor(text=prompts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(dev)
            f_i = sel.model.get_image_features(**ii)
            f_i = f_i / torch.norm(f_i, dim=-1, keepdim=True)
            f_t = sel.model.get_text_features(**ti)
            f_t = f_t / torch.norm(f_t, dim=-1, keepdim=True)
            s = (f_i.view(b, k, -1) * f_t[:, None, :]).sum(-1)
        elif name == "aes":
            s = torch.tensor(sel.score(pils, None)).view(b, k)
        elif name == "im":
            m = sel.model
            img = self._prep(m.preprocess, pils, dev)
            ti = m.blip.tokenizer(prompts, padding="max_length", truncation=True, max_length=35,
                                  return_tensors="pt").to(dev)
            ids = ti.input_ids.repeat_interleave(k, dim=0)
            att = ti.attention_mask.repeat_interleave(k, dim=0)
            emb = m.blip.visual_encoder(img)
            atts = torch.ones(emb.size()[:-1], dtype=torch.long, device=dev)
            out = m.blip.text_encoder(ids, attention_mask=att, encoder_hidden_states=emb,
                                      encoder_attention_mask=atts, return_dict=True)
            r = m.mlp(out.last_hidden_state[:, 0, :].float())
            s = ((r - m.mean) / m.std).view(b, k)
        else:
            raise RuntimeError("unreachable")
        return s.detach().float().cpu().numpy()

    def _slow(self, name, pils, prompts, k):
        sel = self.selectors[name]
        return np.array([[float(x) for x in _as_list(sel.score(pils[i * k:(i + 1) * k], p))]
                         for i, p in enumerate(prompts)], dtype=np.float32)

    @torch.no_grad()
    def __call__(self, pils, prompts, k):
        """pils: B*k images (prompt b owns [b*k, (b+1)*k)). -> {oracle: array (B, k)}, higher is better."""
        out = {}
        for name in self.selectors:
            if name not in self.verified:
                try:
                    fast = self._fast(name, pils[:k], prompts[:1], k)
                    slow = self._slow(name, pils[:k], prompts[:1], k)
                    diff = float(np.abs(fast - slow).max())
                    self.verified[name] = diff <= self.TOL[name]
                    log(f"batched scorer '{name}': max |batched - original| = {diff:.2e} -> "
                        + ("OK" if self.verified[name] else "MISMATCH, using the original path"))
                except Exception as e:
                    self.verified[name] = False
                    log(f"[WARN] batched scorer '{name}' failed ({type(e).__name__}: {e}); using the original path")
            fn = self._fast if self.verified[name] else self._slow
            out[name] = fn(name, pils, prompts, k)
        return out


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple, np.ndarray)) else [x]


def pick_winner_loser(scores, judge):
    """scores: {oracle: [K floats]}. judge: an oracle name, or None for average rank over all oracles."""
    k = len(next(iter(scores.values())))
    if judge is not None:
        s = scores[judge]
        w = max(range(k), key=lambda i: (s[i], -i))
        l = min(range(k), key=lambda i: (s[i], -i))
    else:
        ranks_accum = [0.0] * k
        for s in scores.values():
            order = sorted(range(k), key=lambda i: s[i], reverse=True)
            for rank_idx, img_idx in enumerate(order, start=1):
                ranks_accum[img_idx] += rank_idx
        w = min(range(k), key=lambda i: (ranks_accum[i], i))
        l = max(range(k), key=lambda i: (ranks_accum[i], i))
    if w == l:  # all candidates tied
        l = (w + 1) % k
    return w, l


# --------------------------------------------------------------------------------------
# Sana helpers
# --------------------------------------------------------------------------------------
class SanaBundle:
    """Frozen pieces shared by sampling and training: text encoder, VAE, schedulers."""

    def __init__(self, args, device):
        from diffusers import FlowMatchEulerDiscreteScheduler, SanaPipeline
        self.args, self.device = args, device
        pipe = SanaPipeline.from_pretrained(args.pretrained_model_name_or_path,
                                            torch_dtype=torch.float32, cache_dir=args.cache_dir)
        pipe.text_encoder.to(torch.bfloat16)   # Gemma2 is meant to run in bf16
        pipe.vae.to(torch.float32)             # DC-AE is kept in fp32
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        pipe.text_encoder.requires_grad_(False).eval()
        pipe.vae.requires_grad_(False).eval()
        self.pipe = pipe
        self.scaling_factor = pipe.vae.config.scaling_factor
        # Same "complex human instruction" prefix the pipeline uses by default at inference.
        self.chi = inspect.signature(SanaPipeline.__call__).parameters["complex_human_instruction"].default
        self.train_sched = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler", cache_dir=args.cache_dir)
        self.train_sigmas = self.train_sched.sigmas.to(device=device, dtype=torch.float32)
        self.train_timesteps = self.train_sched.timesteps.to(device=device)
        self.latent_shape = (pipe.transformer.config.in_channels,
                             args.resolution // pipe.vae_scale_factor, args.resolution // pipe.vae_scale_factor)

    @torch.no_grad()
    def encode(self, prompts):
        emb, mask, nemb, nmask = self.pipe.encode_prompt(
            prompts, do_classifier_free_guidance=True, negative_prompt="", device=self.device,
            num_images_per_prompt=1, max_sequence_length=self.args.max_sequence_length,
            complex_human_instruction=self.chi, clean_caption=False)
        return emb.float(), mask, nemb.float(), nmask

    @torch.no_grad()
    def sample_latents(self, model, emb, mask, nemb, nmask, num_per_prompt, steps, guidance, seed, autocast=False):
        """Returns latents of shape (B*num_per_prompt, C, h, w); prompt b owns rows [b*K, (b+1)*K)."""
        was_training = model.training
        model.eval()
        self.pipe.transformer = model
        gen = torch.Generator(device=self.device).manual_seed(seed)
        n = emb.shape[0] * num_per_prompt
        init = torch.randn((n, *self.latent_shape), generator=gen, device=self.device, dtype=torch.float32)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            out = self.pipe(
                prompt=None, negative_prompt=None, prompt_embeds=emb, prompt_attention_mask=mask,
                negative_prompt_embeds=nemb, negative_prompt_attention_mask=nmask,
                height=self.args.resolution, width=self.args.resolution,
                num_inference_steps=steps, guidance_scale=guidance,
                num_images_per_prompt=num_per_prompt, latents=init, generator=gen,
                output_type="latent", use_resolution_binning=False)
        lat = out.images if hasattr(out, "images") else out[0]
        model.train(was_training)
        return lat.float()

    @torch.no_grad()
    def encode_images(self, pils, flip_rng=None):
        """PIL images -> latents through the training image transforms of train_nash.py
        (Resize, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize) and the VAE encoder."""
        from torchvision import transforms
        from torchvision.transforms import functional as TF
        res = self.args.resolution
        base = transforms.Compose([
            transforms.Resize(res, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(res)])
        tensors = []
        for im in pils:
            im = base(im)
            if flip_rng is not None and flip_rng.random() < 0.5:
                im = TF.hflip(im)
            tensors.append(TF.normalize(TF.to_tensor(im), [0.5], [0.5]))
        x = torch.stack(tensors).to(self.device, dtype=self.pipe.vae.dtype)
        return (self.pipe.vae.encode(x).latent * self.scaling_factor).float()

    @torch.no_grad()
    def encode_pixels(self, x, chunk=8):   # DC-AE encoding is memory hungry at 512px
        vae = self.pipe.vae
        out = [vae.encode(c.to(self.device, dtype=vae.dtype)).latent for c in x.split(chunk)]
        return (torch.cat(out) * self.scaling_factor).float()

    @torch.no_grad()
    def decode(self, latents, chunk=16):
        pils = []
        vae = self.pipe.vae
        for c in latents.split(chunk):
            img = vae.decode(c.to(vae.dtype) / self.scaling_factor, return_dict=False)[0]
            pils.extend(self.pipe.image_processor.postprocess(img.float(), output_type="pil"))
        return pils


def flow_errors(model, x_t, timesteps, emb, mask, target, autocast=False):
    """Per-sample flow-matching error || v(x_t, t) - target ||^2 (mean over C,H,W), in fp32."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        pred = model(hidden_states=x_t, encoder_hidden_states=emb, encoder_attention_mask=mask,
                     timestep=timesteps, return_dict=False)[0]
    return (pred.float() - target.float()).pow(2).mean(dim=[1, 2, 3])


# --------------------------------------------------------------------------------------
# Evaluation on a fixed prompt set with fixed seeds
# --------------------------------------------------------------------------------------
@torch.no_grad()
def run_eval(args, bundle, model, scorer, eval_prompts, out_dir, tag):
    t0 = time.time()
    vals = {n: [] for n in scorer.selectors}
    grid_imgs = []
    for start in range(0, len(eval_prompts), args.eval_batch_size):
        prompts = eval_prompts[start:start + args.eval_batch_size]
        emb, mask, nemb, nmask = bundle.encode(prompts)
        lat = bundle.sample_latents(model, emb, mask, nemb, nmask, 1, args.eval_num_inference_steps,
                                    args.nash_guidance_scale, seed=mix_seed(args.seed, 987654321, start),
                                    autocast=args.sample_autocast)
        pils = bundle.decode(lat)
        for n, s in scorer(pils, prompts, 1).items():
            vals[n].extend(s.reshape(-1).tolist())
        if len(grid_imgs) < 16:
            grid_imgs.extend(pils[:16 - len(grid_imgs)])
    means = {n: float(np.mean(v)) for n, v in vals.items()}
    stds = {n: float(np.std(v)) for n, v in vals.items()}
    try:
        from diffusers.utils import make_image_grid
        os.makedirs(out_dir, exist_ok=True)
        small = [im.resize((256, 256)) for im in grid_imgs]
        while len(small) % 4:
            small.append(small[-1])
        make_image_grid(small, rows=len(small) // 4, cols=4).save(os.path.join(out_dir, f"{tag}.jpg"), quality=90)
    except Exception as e:  # image grid is a convenience only
        log(f"[WARN] could not save eval grid: {e}")
    log(f"[eval {tag}] n={len(eval_prompts)} " + " ".join(f"{n}={v:.4f}" for n, v in means.items())
        + f" ({time.time() - t0:.0f}s)")
    return means, stds


# --------------------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------------------
def save_checkpoint(args, step, transformer, prev, optimizer, lr_scheduler, extra):
    root = Path(args.output_dir) / "checkpoints"
    d = root / f"step-{step:06d}"
    tmp = root / f".tmp-step-{step:06d}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    transformer.save_pretrained(tmp / "transformer")
    torch.save({"prev": prev.state_dict(), "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(), "global_step": step, "extra": extra,
                "args": vars(args)}, tmp / "state.pt")
    shutil.rmtree(d, ignore_errors=True)
    tmp.rename(d)
    (root / "latest.json").write_text(json.dumps({"step": step, "dir": d.name}))
    if args.gcs_dir:
        gcs_upload(d, f"{args.gcs_dir}/checkpoints/{d.name}")
        gcs_upload(root / "latest.json", f"{args.gcs_dir}/checkpoints/latest.json")
    # prune old checkpoints (local and remote)
    kept = sorted(p for p in root.glob("step-*") if p.is_dir())
    for old in kept[:-args.keep_checkpoints]:
        shutil.rmtree(old, ignore_errors=True)
        if args.gcs_dir:
            run_cmd(["gcloud", "storage", "rm", "-r", "--quiet", f"{args.gcs_dir}/checkpoints/{old.name}"])
    log(f"saved checkpoint {d}")


def find_checkpoint(args):
    root = Path(args.output_dir) / "checkpoints"
    latest = root / "latest.json"
    if not latest.exists() and args.gcs_dir:
        root.mkdir(parents=True, exist_ok=True)
        r = run_cmd(["gcloud", "storage", "cp", "--quiet", f"{args.gcs_dir}/checkpoints/latest.json", str(latest)])
        if r is None or r.returncode != 0:
            latest.unlink(missing_ok=True)
            return None
    if not latest.exists():
        return None
    info = json.loads(latest.read_text())
    d = root / info["dir"]
    if not (d / "state.pt").exists():
        if not args.gcs_dir:
            return None
        d.mkdir(parents=True, exist_ok=True)
        r = run_cmd(["gcloud", "storage", "rsync", "-r", "--quiet", f"{args.gcs_dir}/checkpoints/{info['dir']}", str(d)])
        if r is None or r.returncode != 0 or not (d / "state.pt").exists():
            return None
        # bring back the logs of the interrupted run as well
        for name in ["metrics.jsonl", "eval.jsonl"]:
            run_cmd(["gcloud", "storage", "cp", "--quiet", f"{args.gcs_dir}/{name}", str(Path(args.output_dir) / name)])
    return d


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    args = parse_args()
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(REPO_ROOT)  # reward-model wrappers use paths relative to the repo root
    sys.path.insert(0, str(REPO_ROOT))
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- multi-GPU: one process per GPU (possibly on different machines), gradients are averaged once per optimizer step.
    # The launcher sets WORLD_SIZE / RANK / MASTER_ADDR / MASTER_PORT. Rank 0 logs, evaluates, checkpoints.
    global RANK
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = RANK = int(os.environ.get("RANK", "0"))
    is_main = rank == 0
    dist = None
    if world_size > 1:
        import datetime
        import torch.distributed as dist
        dist.init_process_group(
            "nccl", init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ.get('MASTER_PORT', '29500')}",
            rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=60))
        log(f"joined process group: rank {rank} of {world_size}")

    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import compute_density_for_timestep_sampling

    log("args: " + json.dumps({k: v for k, v in vars(args).items()}, default=str))

    # ---- prompts ----
    offline = args.train_method == "dpo"
    pair_data = None
    if offline:
        pair_data = PairData(ensure_pairs(args))
        train_prompts = []
        log(f"offline DPO: {pair_data.n} human-labelled image pairs in {len(pair_data.files)} files")
    else:
        train_prompts = load_prompt_list(args.prompts_file, args.dataset_name, args.caption_column, args.split,
                                         args.cache_dir, args.max_prompt_chars, args.max_prompts, args.seed,
                                         dedupe=args.dedupe_prompts)
    eval_prompts = load_prompt_list(args.eval_prompts_file, args.eval_dataset_name, args.eval_caption_column,
                                    args.eval_split, args.cache_dir, 400,
                                    args.eval_num_prompts, 12345) if args.eval_steps > 0 else []
    log(f"{len(train_prompts)} training prompts ({len(set(train_prompts))} unique), {len(eval_prompts)} eval prompts")

    # ---- models: pi (trainable), pi_ref (frozen init), pi_prev (sampler / previous iterate) ----
    bundle = SanaBundle(args, device)
    transformer = bundle.pipe.transformer
    ref = copy.deepcopy(transformer).requires_grad_(False).eval()
    prev = copy.deepcopy(transformer).requires_grad_(False).eval()
    transformer.requires_grad_(True).train()
    n_params = sum(p.numel() for p in transformer.parameters())
    log(f"transformer params: {n_params / 1e6:.1f}M, latent shape {bundle.latent_shape}, "
        f"train scheduler shift={getattr(bundle.train_sched.config, 'shift', None)}")

    selectors = build_oracles(args.oracle_names, device, args.cache_dir)
    scorer = BatchedScorer(selectors)

    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps
                              * args.train_batch_size * world_size)
        log(f"scaled learning rate: {args.learning_rate:.4e}")
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2),
                                  weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                                 num_warmup_steps=args.lr_warmup_steps, num_training_steps=args.max_train_steps)

    # ---- resume ----
    global_step = 0
    judge_counts = {n: 0 for n in args.oracle_names}
    if args.resume == "auto":
        ckpt = find_checkpoint(args)
        if ckpt is not None:
            from diffusers import SanaTransformer2DModel
            loaded = SanaTransformer2DModel.from_pretrained(ckpt / "transformer", torch_dtype=torch.float32)
            transformer.load_state_dict(loaded.state_dict())
            del loaded
            state = torch.load(ckpt / "state.pt", map_location="cpu", weights_only=False)
            prev.load_state_dict(state["prev"])
            optimizer.load_state_dict(state["optimizer"])
            lr_scheduler.load_state_dict(state["lr_scheduler"])
            global_step = state["global_step"]
            judge_counts.update(state.get("extra", {}).get("judge_counts", {}))
            del state
            log(f"resumed from {ckpt} at step {global_step}")

    model_fwd = transformer
    train_params = [p_ for p_ in transformer.parameters() if p_.requires_grad]

    def average_gradients():
        """Average the accumulated gradients over all GPUs: one all-reduce per optimizer step.
        (DistributedDataParallel would sync the full 2.4GB gradient on every backward call.)"""
        flat = torch.cat([(p_.grad if p_.grad is not None else torch.zeros_like(p_)).reshape(-1)
                          for p_ in train_params])
        dist.all_reduce(flat)
        flat /= world_size
        offset = 0
        for p_ in train_params:
            n = p_.numel()
            if p_.grad is None:
                p_.grad = flat[offset:offset + n].view_as(p_).clone()
            else:
                p_.grad.copy_(flat[offset:offset + n].view_as(p_))
            offset += n

    metrics_path = Path(args.output_dir) / "metrics.jsonl"
    eval_path = Path(args.output_dir) / "eval.jsonl"
    writer = None
    if is_main:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(args.output_dir, "tb"))
        except Exception as e:
            log(f"[WARN] tensorboard unavailable: {e}")

    def sync_logs():
        if args.gcs_dir:
            for name in ["metrics.jsonl", "eval.jsonl"]:
                if (Path(args.output_dir) / name).exists():
                    gcs_upload(Path(args.output_dir) / name, f"{args.gcs_dir}/{name}")
            if (Path(args.output_dir) / "eval").exists():
                gcs_upload(Path(args.output_dir) / "eval", f"{args.gcs_dir}/eval")

    def read_eval_rows():
        if not eval_path.exists():
            return []
        return [json.loads(x) for x in eval_path.read_text().splitlines() if x.strip()]

    def do_eval(step):
        """Evaluate pi on the fixed prompts. Returns the number of evals since the best one.

        gain = average over reward models of (mean score - mean score at step 0) / (std over images at
        step 0), i.e. the improvement in units of the image-to-image spread, averaged over all judges."""
        means, stds = run_eval(args, bundle, transformer, scorer, eval_prompts,
                               os.path.join(args.output_dir, "eval"), f"step-{step:06d}")
        rows = read_eval_rows()
        base = next((r for r in rows if r["step"] == 0), None)
        row = {"step": step, **means}
        if base is None:
            row.update({f"std/{n}": v for n, v in stds.items()})
            base = row
        row["gain"] = float(np.mean([(means[n] - base[n]) / max(base[f"std/{n}"], 1e-8) for n in means]))
        best_before = max([r["gain"] for r in rows if r["step"] > 0], default=-float("inf"))
        is_best = step > 0 and row["gain"] > best_before
        row["is_best"] = is_best
        with open(eval_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if writer:
            for n, v in means.items():
                writer.add_scalar(f"eval/{n}", v, step)
            writer.add_scalar("eval/gain", row["gain"], step)
        if is_best:
            best_dir = Path(args.output_dir) / "best"
            transformer.save_pretrained(best_dir / "transformer")
            (best_dir / "best.json").write_text(json.dumps(row))
            if args.gcs_dir:
                gcs_upload(best_dir, f"{args.gcs_dir}/best")
        rows.append(row)
        later = [r for r in rows if r["step"] > 0]
        best_i = max(range(len(later)), key=lambda i: later[i]["gain"]) if later else 0
        since_best = len(later) - 1 - best_i if later else 0
        log(f"[eval step {step}] gain={row['gain']:+.4f}" + (" (new best, weights saved)" if is_best else
            f" (best {later[best_i]['gain']:+.4f} at step {later[best_i]['step']}, {since_best} evals ago)" if later else ""))
        sync_logs()
        return since_best

    if is_main and args.eval_steps > 0 and global_step == 0:
        do_eval(0)

    K = args.nash_num_candidates
    B = args.train_batch_size
    tau, eta = args.nash_tau, args.nash_eta
    num_train_timesteps = bundle.train_sched.config.num_train_timesteps
    log(f"***** training: method={args.train_method} judge={args.nash_judge_mode}/{args.nash_judge_granularity} "
        f"oracles={args.oracle_names} pairs/step={B * args.gradient_accumulation_steps * world_size} "
        f"(gpus={world_size}) K={K} *****")

    pair_iter = None
    if offline:
        loader = torch.utils.data.DataLoader(
            PairBatches(pair_data, args, rank, world_size, global_step), batch_size=None, shuffle=False,
            num_workers=args.dataloader_num_workers, prefetch_factor=4 if args.dataloader_num_workers else None)
        pair_iter = iter(loader)

    stop_training = False
    while global_step < args.max_train_steps:
        if world_size > 1:   # rank 0 decides about early stopping; tell everyone
            flag = torch.tensor([1.0 if stop_training else 0.0], device=device)
            dist.all_reduce(flag)
            stop_training = flag.item() > 0
        if stop_training:
            break
        t_step = time.time()
        times = {"sample": 0.0, "score": 0.0, "train": 0.0, "sync": 0.0}
        acc = {}          # metric sums over micro-batches

        def add(name, value, weight=1.0):
            s, w = acc.get(name, (0.0, 0.0))
            acc[name] = (s + float(value) * weight, w + weight)

        step_judges = []
        optimizer.zero_grad(set_to_none=True)
        for micro in range(args.gradient_accumulation_steps):
            rng = np.random.default_rng(mix_seed(args.seed, 1, global_step, micro, rank))
            if offline:
                # ---------- offline DPO: a micro-batch of human-labelled pairs (winner first) ----------
                t0 = time.time()
                pixels, prompts = next(pair_iter)
                times["sample"] += time.time() - t0          # time spent waiting for data
                t0 = time.time()
                emb, mask, nemb, nmask = bundle.encode(prompts)
                x0 = bundle.encode_pixels(torch.cat([pixels[:, 0], pixels[:, 1]], dim=0))   # (2B, C, h, w)
                del pixels
                step_judges.append("human")
            else:
                # ---------- 1. prompts ----------
                prompts = [train_prompts[i] for i in rng.choice(len(train_prompts), size=B, replace=False)]

                # ---------- 2. pi_prev generates K candidates per prompt ----------
                t0 = time.time()
                emb, mask, nemb, nmask = bundle.encode(prompts)
                cand = bundle.sample_latents(prev, emb, mask, nemb, nmask, K, args.nash_num_inference_steps,
                                             args.nash_guidance_scale, seed=mix_seed(args.seed, 2, global_step, micro, rank),
                                             autocast=args.sample_autocast)
                pils = bundle.decode(cand)
                torch.cuda.synchronize()
                times["sample"] += time.time() - t0

                # ---------- 3. reward models score, the judge picks winner / loser ----------
                t0 = time.time()
                w_rows, l_rows = [], []
                all_scores = scorer(pils, prompts, K)            # {oracle: (B, K)}
                for b, prompt in enumerate(prompts):
                    scores = {n: all_scores[n][b].tolist() for n in args.oracle_names}
                    if args.nash_judge_mode == "avg_rank":
                        judge = None
                    else:
                        key = {"step": (global_step,), "microbatch": (global_step, micro),
                               "prompt": (global_step, micro, b, rank)}[args.nash_judge_granularity]
                        judge = args.oracle_names[mix_seed(args.seed, 3, *key) % len(args.oracle_names)]
                    w, l = pick_winner_loser(scores, judge)
                    w_rows.append(b * K + w)
                    l_rows.append(b * K + l)
                    step_judges.append(judge or "avg_rank")
                    if judge is not None:
                        judge_counts[judge] += world_size
                    for n, s in scores.items():
                        add(f"oracle_mean/{n}", np.mean(s))
                        add(f"oracle_winner/{n}", s[w])
                        add(f"oracle_loser/{n}", s[l])
                        add(f"oracle_agree/{n}", float(s[w] > s[l]))
                times["score"] += time.time() - t0

                if args.save_candidate_images_steps and micro == 0 and global_step % args.save_candidate_images_steps == 0:
                    d = Path(args.output_dir) / "candidates" / f"step-{global_step:06d}"
                    d.mkdir(parents=True, exist_ok=True)
                    for k in range(K):
                        tag = "W" if k == w_rows[0] else ("L" if k == l_rows[0] else "_")
                        pils[k].save(d / f"cand{k}_{tag}.jpg", quality=90)
                    (d / "prompt.txt").write_text(f"{prompts[0]}\njudge={step_judges[0]}\n")
                # ---------- 4. flow-matching newnash loss ----------
                t0 = time.time()
                if args.latent_source == "reencode":
                    flip_rng = None if args.no_hflip else random.Random(mix_seed(args.seed, 6, global_step, micro, rank))
                    x0 = bundle.encode_images([pils[i] for i in w_rows] + [pils[i] for i in l_rows], flip_rng)
                else:
                    x0 = torch.cat([cand[w_rows], cand[l_rows]], dim=0)                  # (2B, C, h, w)
                del cand, pils
            # Each pair can be noised several times (different t and eps): this reuses the expensive
            # samples for more gradient signal; the loss is averaged over the repeats.
            for rep in range(args.timesteps_per_pair):
                gen = torch.Generator(device=device).manual_seed(mix_seed(args.seed, 4, global_step, micro, rank, rep))
                noise = torch.randn((B, *bundle.latent_shape), generator=gen, device=device).repeat(2, 1, 1, 1)
                torch.manual_seed(mix_seed(args.seed, 5, global_step, micro, rank, rep) % (2 ** 31))
                u = compute_density_for_timestep_sampling(weighting_scheme=args.weighting_scheme, batch_size=B,
                                                          logit_mean=args.logit_mean, logit_std=args.logit_std,
                                                          mode_scale=args.mode_scale)
                idx = (u * num_train_timesteps).long().clamp(0, num_train_timesteps - 1).to(device)
                timesteps = bundle.train_timesteps[idx].repeat(2)                        # same t for w and l
                sigmas = bundle.train_sigmas[idx].repeat(2).view(-1, 1, 1, 1)
                x_t = (1.0 - sigmas) * x0 + sigmas * noise
                target = noise - x0

                # conditioning: same prompt for w and l; a fraction of pairs is trained unconditionally
                drop = torch.tensor(rng.random(B) < args.proportion_empty_prompts, device=device)
                c_emb = torch.where(drop.view(-1, 1, 1), nemb, emb).repeat(2, 1, 1)
                c_mask = torch.where(drop.view(-1, 1), nmask, mask).repeat(2, 1)

                model_losses = flow_errors(model_fwd, x_t, timesteps, c_emb, c_mask, target, args.train_autocast)
                model_losses_w, model_losses_l = model_losses.chunk(2)
                model_diff = model_losses_w - model_losses_l
                with torch.no_grad():
                    ref_losses = flow_errors(ref, x_t, timesteps, c_emb, c_mask, target, args.train_autocast)
                    ref_losses_w, ref_losses_l = ref_losses.chunk(2)
                    ref_diff = ref_losses_w - ref_losses_l
                    if args.train_method in ("newnash", "nash"):
                        prev_losses = flow_errors(prev, x_t, timesteps, c_emb, c_mask, target, args.train_autocast)
                        prev_losses_w, prev_losses_l = prev_losses.chunk(2)
                        prev_diff = prev_losses_w - prev_losses_l

                if args.train_method in ("online_dpo", "dpo"):
                    inside_term = -0.5 * args.beta_dpo * (model_diff - ref_diff)
                    loss = -F.logsigmoid(inside_term).mean()
                else:
                    log_ratio = -0.5 * model_diff
                    log_ratio_ref = -0.5 * ref_diff
                    log_ratio_prev = -0.5 * prev_diff
                    h_t = log_ratio - (tau / eta) * log_ratio_ref - ((eta - tau) / eta) * log_ratio_prev
                    add("h_t", h_t.mean().item())
                    if args.train_method == "newnash":
                        inside_term = args.beta_dpo * h_t
                        loss = -F.logsigmoid(inside_term).mean()
                    else:  # nash: squared regression to 1/(2 eta)
                        inside_term = h_t
                        loss = (h_t - 1.0 / (2.0 * eta)).pow(2).mean()
                    add("prev_diff", prev_diff.mean().item())
                    add("prev_losses_w", prev_losses_w.mean().item())
                    add("prev_losses_l", prev_losses_l.mean().item())

                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {global_step}: {loss.item()}")
                (loss / (args.gradient_accumulation_steps * args.timesteps_per_pair)).backward()

                add("loss", loss.item())
                add("implicit_acc", (inside_term > 0).float().mean().item())
                add("inside_abs", inside_term.detach().abs().mean().item())   # |beta*h|: >>1 means saturated
                add("pair_diff_std", (model_diff - ref_diff).detach().std().item())
                add("model_diff", model_diff.mean().item())
                add("ref_diff", ref_diff.mean().item())
                add("model_losses_w", model_losses_w.mean().item())
                add("model_losses_l", model_losses_l.mean().item())
                add("ref_losses_w", ref_losses_w.mean().item())
                add("ref_losses_l", ref_losses_l.mean().item())

            torch.cuda.synchronize()
            times["train"] += time.time() - t0

        # ---------- optimizer step ----------
        if world_size > 1:
            t0 = time.time()
            average_gradients()
            torch.cuda.synchronize()
            times["sync"] = time.time() - t0
        grad_norm = torch.nn.utils.clip_grad_norm_(transformer.parameters(), args.max_grad_norm).item()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        # ---------- 5. refresh pi_prev (same rule as train_nash.py) ----------
        if not offline and global_step % args.nash_prev_update_steps == 0:
            with torch.no_grad():
                src_sd, prev_sd = transformer.state_dict(), prev.state_dict()
                mu = 0.0 if args.nash_prev_hard_update else min(0.001 * global_step, 0.5)
                for k_, v in prev_sd.items():
                    if mu > 0.0 and torch.is_floating_point(v):
                        v.mul_(mu).add_(src_sd[k_], alpha=1.0 - mu)
                    else:
                        v.copy_(src_sd[k_])

        # ---------- logging (metrics are averaged over all GPUs) ----------
        if world_size > 1:
            keys = sorted(acc)
            packed = torch.tensor([list(acc[k_]) for k_ in keys], device=device, dtype=torch.float64)
            dist.all_reduce(packed)
            acc = {k_: (packed[i, 0].item(), packed[i, 1].item()) for i, k_ in enumerate(keys)}
        if not is_main:
            continue
        m = {k_: s / max(w_, 1e-12) for k_, (s, w_) in acc.items()}
        m.update({"step": global_step, "grad_norm": grad_norm, "lr": lr_scheduler.get_last_lr()[0],
                  "time_step": time.time() - t_step, "time_sample": times["sample"],
                  "time_score": times["score"], "time_train": times["train"], "time_sync": times["sync"],
                  "judges": sorted(set(step_judges))})
        with open(metrics_path, "a") as f:
            f.write(json.dumps(m) + "\n")
        if writer:
            for k_, v in m.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(k_, v, global_step)
        log(f"[step {global_step}/{args.max_train_steps}] loss={m['loss']:.4f} acc={m['implicit_acc']:.3f} |bh|={m['inside_abs']:.2f} "
            f"model_diff={m['model_diff']:.2e} ref_diff={m['ref_diff']:.2e} gnorm={grad_norm:.2f} "
            f"lr={m['lr']:.2e} judge={','.join(m['judges'])} | "
            + " ".join(f"{n}={m[f'oracle_mean/{n}']:.3f}" for n in args.oracle_names if f"oracle_mean/{n}" in m)
            + f" | {m['time_step']:.0f}s (sample {times['sample']:.0f} score {times['score']:.0f} train {times['train']:.0f} sync {times['sync']:.0f})")

        if args.eval_steps > 0 and (global_step % args.eval_steps == 0 or global_step == args.max_train_steps):
            since_best = do_eval(global_step)
            if (args.early_stop_patience > 0 and since_best >= args.early_stop_patience
                    and global_step >= args.early_stop_min_steps):
                log(f"early stop: no new best eval gain for {since_best} evals")
                stop_training = True
        if global_step % args.checkpointing_steps == 0 or global_step == args.max_train_steps or stop_training:
            save_checkpoint(args, global_step, transformer, prev, optimizer, lr_scheduler,
                            {"judge_counts": judge_counts})
            sync_logs()

    if is_main:
        final = Path(args.output_dir) / "final" / "transformer"
        transformer.save_pretrained(final)
        if args.gcs_dir:
            gcs_upload(final, f"{args.gcs_dir}/final/transformer")
            sync_logs()
        log(f"done. judge counts: {judge_counts}. final transformer at {final}")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
