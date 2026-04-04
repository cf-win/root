export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

DATASET=Amazon_Beauty
BASE_MODEL=/root/autodl-tmp/Llama
DATA_PATH=/root/autodl-tmp/
OUTPUT_DIR=/root/autodl-tmp/ckpt/$DATASET/
TARGET_K=10

torchrun --nproc_per_node=1 --master_port=3325  lora_finetune.py \
    --base_model $BASE_MODEL\
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --target_k $TARGET_K \
    --per_device_batch_size 16 \
    --learning_rate 1e-4 \
    --epochs 1 \
    --tasks seqrec \
    --train_prompt_sample_num 1 \
    --train_data_sample_num -1 \
    --index_file .index.json\
    --wandb_run_name test\
    --temperature 1.0
