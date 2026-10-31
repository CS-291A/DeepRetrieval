#!/bin/sh
# common.sh: loads code/.env, defaults DEEPRETRIEVAL_DATA_DIR, creates exp_log/, cd code/.
# GPUs: all visible ones are used; export CUDA_VISIBLE_DEVICES first to restrict them.
. "$(dirname "$0")/../../common.sh" || exit 1

DATASET=$1
MODEL_NAME=$2
DATA_PATH="$DEEPRETRIEVAL_DATA_DIR"/local_index_search/no_reason/BM25/$DATASET/test.parquet
SAVE_DIR=../results/no_reason

python src/eval/BM25/baselines/model_generate/gpt.py \
    --data_path "$DATA_PATH" \
    --dataset "$DATASET" \
    --save_dir $SAVE_DIR \
    --model_name "$MODEL_NAME"
