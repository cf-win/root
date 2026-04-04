export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

DATASET=Amazon_Beauty
DATA_PATH=/root/autodl-tmp/
CKPT_PATH=/root/autodl-tmp/ckpt/$DATASET/
OUTPUT_DIR=/root/autodl-tmp/ckpt/$DATASET/
RESULTS_FILE=/root/autodl-tmp/results/$DATASET/ddp.json
BASE_MODEL=/root/autodl-tmp/Llama

torchrun --nproc_per_node=1 --master_port=4324 test_ddp.py \
    --ckpt_path $CKPT_PATH \
    --base_model $BASE_MODEL\
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 1 \
    --num_beams 10 \
    --sample_num -1 \
    --test_prompt_ids 0 \
    --index_file .index.json

