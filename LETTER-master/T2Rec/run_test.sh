export CUDA_VISIBLE_DEVICES=0

DATASET=Beauty_5
# 使用 1.5B 模型
BASE_MODEL=/root/autodl-tmp/Qwen/Qwen2.5-1.5B-Instruct
# 数据路径指向标准格式的数据目录
DATA_PATH=/root/autodl-tmp/
CKPT_PATH=/root/autodl-tmp/ckpt/$DATASET/checkpoint-20530/
GRAPH_TOKEN_PATH=/root/autodl-tmp/token/graph_token/${DATASET}_graph_tokens.pt
BEHAVIOR_TOKEN_PATH=/root/autodl-tmp/token/behavior_token/${DATASET}_behavior_tokens.pt
RESULTS_FILE=/root/autodl-tmp/results/${DATASET}_test.json

python test_t2rec.py \
    --base_model $BASE_MODEL \
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --graph_token_path $GRAPH_TOKEN_PATH \
    --behavior_token_path $BEHAVIOR_TOKEN_PATH \
    --results_file $RESULTS_FILE \
    --task simple_rec \
    --test_batch_size 2 \
    --num_beams 10 \
    --index_file .index.json \
    --max_his_len 20 \
    --sample_num -1 \
    --risk_threshold 0.5 \
    --threshold_min 0.1 \
    --threshold_max 0.9 \
    --threshold_step 0.01 \
    --probe_dim 64 \
    --lora
