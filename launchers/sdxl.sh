export HF_HOME=/network/rit/lab/wang_lab_cs/jmh/ddk_cache
export TRANSFORMERS_CACHE=/network/rit/lab/wang_lab_cs/jmh/ddk_cache
export HF_DATASETS_CACHE=/network/rit/lab/wang_lab_cs/jmh/ddk_cache/datasets
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="pickapic-anonymous/pickapic_v1"
export HF_TOKEN=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

# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~30 hours / 200 steps
  nohup accelerate launch --config_file config/accelerate_config_01234567.yaml train_nash.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE \
  --dataset_name=$DATASET_NAME \
  --sdxl \
  --train_batch_size=4 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=64 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=200 \
  --learning_rate=1e-8 --scale_lr \
  --checkpointing_steps 10 \
  --cache_dir="/network/rit/lab/wang_lab_cs/jmh/diffusion-nash-main/DiffusionDPO/cache" \
  --nash_tau 0.025 \
  --nash_eta 0.05 \
  --train_method='online_dpo' \
  --output_dir="tmp-sdxl-online_dpo_1half" \
  --report_to="wandb" \
  > output.log 2>&1 & 





accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=1 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=128 \
  --max_train_steps=2000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=200 \
  --learning_rate=1e-8 --scale_lr \
  --cache_dir="/export/share/datasets/vision_language/pick_a_pic_v2/" \
  --checkpointing_steps 200 \
  --beta_dpo 5000 \
   --sdxl  \
  --output_dir="tmp-sdxl"



  
