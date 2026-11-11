export HF_HOME=/network/rit/lab/wang_lab_cs/jmh/ddk_cache
export TRANSFORMERS_CACHE=/network/rit/lab/wang_lab_cs/jmh/ddk_cache
export HF_DATASETS_CACHE=/network/rit/lab/wang_lab_cs/jmh/ddk_cache/datasets
export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export DATASET_NAME="pickapic-anonymous/pickapic_v1"
export HF_TOKEN=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

# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~24 hours / 2000 steps

nohup accelerate launch --config_file config/accelerate_config_01234567.yaml train_nash.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=8 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=64 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=200 \
  --learning_rate=1e-8 --scale_lr \
  --checkpointing_steps 10 \
  --cache_dir="/network/rit/lab/wang_lab_cs/jmh/diffusion-nash-main/DiffusionDPO/cache" \
  --nash_tau 0.025 \
  --nash_eta 0.05 \
  --train_method='newnash' \
  --nash_prev_hard_update \
  --output_dir="tmp-sd15-newnash_1half_hard" \
  --report_to="wandb" \
  > output.log 2>&1 & 

  tmux   

  --nash_prev_update_steps=99999999 \
  --unet_init="/network/rit/lab/wang_lab_cs/jmh/diffusion-nash-main/DiffusionDPO/tmp-sd15-loop/checkpoint-20-loop-tau0p0167-eta0p05" \


nohup accelerate launch --config_file config/accelerate_config_01234567.yaml train_nash.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=8 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=32 \
  --max_train_steps=1000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
  --learning_rate=1e-8 --scale_lr \
  --checkpointing_steps 10 \
  --resume_from_checkpoint='checkpoint-250-tau0p0167-eta0p05' \
  --cache_dir="/network/rit/lab/wang_lab_cs/jmh/diffusion-nash-main/DiffusionDPO/cache" \
  --nash_tau 3.34 \
  --nash_eta 10 \
  --train_method='nash' \
  --output_dir="tmp-sd15-online_dpo" \
  > output.log 2>&1 & 

nohup accelerate launch --config_file config/accelerate_config_01234567.yaml train_nash.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=8 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=32 \
  --max_train_steps=2000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=300 \
  --learning_rate=5e-7 --scale_lr \
  --checkpointing_steps 50 \
  --resume_from_checkpoint='checkpoint-epoch-1' \
  --cache_dir="/network/rit/lab/wang_lab_cs/jmh/diffusion-nash-main/DiffusionDPO/cache" \
  --nash_tau 0.0167 \
  --nash_eta 0.05 \
  --train_method='nash' \
  --output_dir="tmp-sd15" \
  > output.log 2>&1 & 

nohup accelerate launch --config_file config/accelerate_config_01234567.yaml train_nash.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=8 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=32 \
  --max_train_steps=2000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=300 \
  --learning_rate=5e-7 --scale_lr \
  --checkpointing_steps 50 \
  --resume_from_checkpoint='checkpoint-epoch-2' \
  --cache_dir="/network/rit/lab/wang_lab_cs/jmh/diffusion-nash-main/DiffusionDPO/cache" \
  --nash_tau 0.0167 \
  --nash_eta 0.05 \
  --train_method='nash' \
  --output_dir="tmp-sd15" \
  > output.log 2>&1 & 