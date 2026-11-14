#!/bin/bash
# Offline flow-DPO baseline on Sana (Diffusion-DPO on human-labelled Pick-a-Pic v1 pairs), to compare
# with the online NPO runs. Same optimiser settings as launchers/sana_submit.sh:
#   2048 pairs per optimizer step (NUM_GPUS x train_batch_size 16 x gradient_accumulation_steps),
#   lr 1e-8 with --scale_lr (2.048e-5), constant_with_warmup 200, beta_dpo 5000, max_grad_norm 50,
#   20% empty prompts, random horizontal flip, same evaluation (256 PartiPrompts, same seeds).
# Differences: the pairs are human-labelled images (no sampling, no reward models during training) and
# the loss is the DPO loss (pi vs pi_ref only).
#
#   NUM_GPUS=2 NODE_RESOURCE=npo_c bash launchers/sana_submit_dpo.sh <run_name> [extra args]
set -euo pipefail
RUN_NAME="${1:?usage: sana_submit_dpo.sh <run_name> [extra args]}"
shift
export NUM_GPUS="${NUM_GPUS:-2}"
export GCS_ROOT="${GCS_ROOT:-gs://prism-training-apac/jiaming/npo-sana}"
ACCUM=$((2048 / (NUM_GPUS * 16)))
exec bash "$(dirname "$0")/sana_submit.sh" "$RUN_NAME" \
  --train_method dpo \
  --train_batch_size 16 --gradient_accumulation_steps "$ACCUM" \
  --pairs_gcs "${PAIRS_GCS:-$GCS_ROOT/data/pickapic_v1_sfw_decided}" \
  --checkpointing_steps 10 \
  "$@"
