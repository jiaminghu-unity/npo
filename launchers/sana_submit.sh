#!/bin/bash
# Submit a Sana newnash run to the Ray cluster (run this on the Ray head, from the repo root).
#
#   NUM_GPUS=4 bash launchers/sana_submit.sh <run_name> [extra train_nash_sana.py args]
#
# The training arguments below mirror the SD1.5 newnash launcher (launchers/sd15.sh):
#   4 GPUs x train_batch_size 8 x gradient_accumulation_steps 64 = 2048 pairs per optimizer step,
#   learning_rate 1e-8 with --scale_lr (= 2.048e-5), constant_with_warmup with 200 warmup steps,
#   nash_tau 0.025 / nash_eta 0.05, --nash_prev_hard_update, beta_dpo 5000 (script default),
#   max_grad_norm 50 (script default), 8 candidates per prompt, 20% empty prompts.
# If NUM_GPUS is not 4, change --gradient_accumulation_steps so the product stays 2048.
# NODE_RESOURCE=npo_gpu runs on the dedicated nodes made by ~/npo_sana_dev/nodes/make_node.sh.
#
# Checkpoints / logs go to $GCS_ROOT/<run_name>; a preempted run resumes from there.
# Follow it with:   ray job logs -f <run_name>      Stop it with:   ray job stop <run_name>
set -euo pipefail

RUN_NAME="${1:?usage: sana_submit.sh <run_name> [extra args]}"
shift
NUM_GPUS="${NUM_GPUS:-4}"
GCS_ROOT="${GCS_ROOT:-gs://prism-training-us/jiaming/npo-sana}"
export RAY_API_SERVER_ADDRESS="${RAY_API_SERVER_ADDRESS:-http://127.0.0.1:8265}"

ray job submit --no-wait --submission-id "$RUN_NAME" \
  --runtime-env-json '{"working_dir": ".", "excludes": [".git", "output.log", "__pycache__", "utils", "tmp-*", "results", "sana_ops"]}' \
  -- python3 launchers/sana_ray_job.py --num_gpus "$NUM_GPUS" ${NODE_RESOURCE:+--node_resource "$NODE_RESOURCE"} \
  --train_method newnash \
  --train_batch_size 8 \
  --gradient_accumulation_steps 64 \
  --lr_scheduler constant_with_warmup --lr_warmup_steps 200 \
  --learning_rate 1e-8 --scale_lr \
  --nash_tau 0.025 --nash_eta 0.05 \
  --nash_prev_hard_update \
  --nash_oracles pickscore,hps,clip,aes,im \
  --nash_judge_mode random --nash_judge_granularity step \
  --nash_guidance_scale 4.5 --sample_autocast \
  --checkpointing_steps 2 --eval_steps 10 --eval_num_prompts 256 \
  --early_stop_patience 15 --early_stop_min_steps 250 \
  --output_dir "runs/$RUN_NAME" --gcs_dir "$GCS_ROOT/$RUN_NAME" \
  "$@"
