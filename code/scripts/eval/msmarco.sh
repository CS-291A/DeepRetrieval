#!/bin/sh
# common.sh: loads code/.env, defaults DEEPRETRIEVAL_DATA_DIR, creates exp_log/, cd code/.
# GPUs: all visible ones are used; export CUDA_VISIBLE_DEVICES first to restrict them.
. "$(dirname "$0")/../common.sh" || exit 1

# --model_path /shared/eng/pj20/lmr_model/nq_serini_3b/actor/global_step_400 \
# --model_path Qwen/Qwen2.5-3B-Instruct \
python src/eval/BM25/msmarco.py \
    --model_path /shared/eng/pj20/lmr_model/msmarco_health_sparse_recall/actor/global_step_650 \
    --model_name msmarco_health_sparse_recall_step_650 \
    --data_path "$DEEPRETRIEVAL_DATA_DIR"/local_index_search/msmarco_health/sparse/test.parquet