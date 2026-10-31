#!/bin/sh
# common.sh: loads code/.env, defaults DEEPRETRIEVAL_DATA_DIR, creates exp_log/, cd code/.
# GPUs: all visible ones are used; export CUDA_VISIBLE_DEVICES first to restrict them.
. "$(dirname "$0")/../common.sh" || exit 1

# /shared/eng/pj20/lmr_model/ctgov_3b_transfer_1/actor/global_step_100
# --model_path Qwen/Qwen2.5-3B-Instruct \
python src/eval/SearchEngine/eval_ctgov.py \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --data_path "$DEEPRETRIEVAL_DATA_DIR"/search_engine/no_reason/SearchEngine/ctgov/test_full.parquet \
    --model_name ctgov_3b_base_no_reason