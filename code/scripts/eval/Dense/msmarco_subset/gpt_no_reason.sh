#!/bin/sh
# common.sh: loads code/.env, defaults DEEPRETRIEVAL_DATA_DIR, creates exp_log/, cd code/.
# GPUs: all visible ones are used; export CUDA_VISIBLE_DEVICES first to restrict them.
. "$(dirname "$0")/../../../common.sh" || exit 1

DATASET=$1
DATA_PATH="$DEEPRETRIEVAL_DATA_DIR"/local_index_search/no_reason/$DATASET/dense/test.parquet
SAVE_DIR=../results_dense/no_reason

python src/eval/Dense/baselines/model_generate_no_reasoning/msmarco_subset/gpt-4o.py \
    --data_path "$DATA_PATH" \
    --dataset "$DATASET" \
    --save_dir $SAVE_DIR