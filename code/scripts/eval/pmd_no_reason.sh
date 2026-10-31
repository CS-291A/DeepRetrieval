#!/bin/sh
# common.sh: loads code/.env, defaults DEEPRETRIEVAL_DATA_DIR, creates exp_log/, cd code/.
# GPUs: all visible ones are used; export CUDA_VISIBLE_DEVICES first to restrict them.
. "$(dirname "$0")/../common.sh" || exit 1

python src/eval/SearchEngine/eval_pubmed.py \
    --model_path /shared/eng/pj20/lmr_model/pubmed_search_3b_no_reason/actor/global_step_900 \
    --data_path "$DEEPRETRIEVAL_DATA_DIR"/search_engine/no_reason/SearchEngine/pmd_no_reason/test_full.parquet \
    --model_name pubmed_no_reason_900
