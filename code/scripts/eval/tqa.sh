#!/bin/sh
# common.sh: loads code/.env, defaults DEEPRETRIEVAL_DATA_DIR, creates exp_log/, cd code/.
# GPUs: all visible ones are used; export CUDA_VISIBLE_DEVICES first to restrict them.
. "$(dirname "$0")/../common.sh" || exit 1

# --model_path /shared/eng/pj20/lmr_model/triviaqa_3b_nq_1800/actor/global_step_1200 \
# --model_path Qwen/Qwen2.5-3B-Instruct \
python src/eval/BM25/triviaqa.py \
    --model_path /shared/eng/pj20/lmr_model/triviaqa_3b_no_reason/actor/global_step_750 \
    --model_name triviaqa_serini_ours_no_reason \
    --data_path "$DEEPRETRIEVAL_DATA_DIR"/local_index_search/triviaqa/test.parquet