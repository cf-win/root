export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

DATASET=Beauty_5
# 使用 1.5B 模型快速验证
BASE_MODEL=/root/autodl-tmp/Qwen/Qwen2.5-1.5B-Instruct
# 数据路径指向标准格式的数据目录
DATA_PATH=/root/autodl-tmp/
OUTPUT_DIR=/root/autodl-tmp/ckpt/$DATASET/
GRAPH_TOKEN_PATH=/root/autodl-tmp/token/graph_token/${DATASET}_graph_tokens.pt
BEHAVIOR_TOKEN_PATH=/root/autodl-tmp/token/behavior_token/${DATASET}_behavior_tokens.pt

torchrun --nproc_per_node=1 train_t2rec.py \
    --base_model $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --graph_token_path $GRAPH_TOKEN_PATH \
    --behavior_token_path $BEHAVIOR_TOKEN_PATH \
    --per_device_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --epochs 1 \
    --logging_step 10 \
    --task rec_train \
    --train_prompt_sample_num 1 \
    --train_data_sample_num 0 \
    --index_file .index.json \
    --temperature 1.0 \
    --lambda_anomaly 0.0 \
    --lambda_risk 2.0 \
    --probe_dim 64 \
    --max_his_len 20 \
    --top_k 10 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --bf16 \
    --deepspeed ./config/ds_z2_bf16.json
