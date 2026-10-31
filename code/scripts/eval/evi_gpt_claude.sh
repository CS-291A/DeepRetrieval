#!/bin/sh
# common.sh: loads code/.env, defaults DEEPRETRIEVAL_DATA_DIR, creates exp_log/, cd code/.
# GPUs: all visible ones are used; export CUDA_VISIBLE_DEVICES first to restrict them.
. "$(dirname "$0")/../common.sh" || exit 1

python src/eval/evidence_gpt_claude.py \
    --data_path "$DEEPRETRIEVAL_DATA_DIR"/local_index_search/nq_serini/test.parquet \
    --model_name nq-gpt4o-no-block \
    --batch_size 8

# python src/eval/evidence_gpt_claude.py \
#     --data_path "$DEEPRETRIEVAL_DATA_DIR"/local_index_search/nq_serini/test.parquet \
#     --model_name nq-claude3 \
#     --batch_size 8