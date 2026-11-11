export MODEL_PATH='tmp-sd15-loop/checkpoint-250-loop-tau33p4-eta100p0/unet' # Please change to your ckpt path, such as path/to/checkpoint-xx
export NAME='NASH_15' # Name cannot contain space
export DEVICE='cuda:2'
export SEED=0
export HF_HOME='/network/rit/lab/wang_lab_cs/jmh/ddk_cache' # change to your HF_HOME
python eval.py \
    --original_model "stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --unet_checkpoints $MODEL_PATH \
    --epochs 2 \
    --names "Orig. 15" $NAME \
    --device_orig $DEVICE \
    --device_trained $DEVICE \
    --metrics_device $DEVICE \
    --batch_size 16 \
    --seed $SEED 