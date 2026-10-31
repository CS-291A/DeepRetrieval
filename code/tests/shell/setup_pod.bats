#!/usr/bin/env bats
# Tests for code/scripts/setup_pod.sh. External commands (git, uv, nvidia-smi, curl) are
# stubs from helpers.bash that record their arguments, so the script runs for real
# against a temporary --workspace whose path contains a space.

bats_require_minimum_version 1.5.0

load helpers

setup() {
  clear_dr_env
  make_stubs
  WS="${BATS_TEST_TMPDIR}/work space"
  mkdir -p "$WS"
  DEST="$WS/DeepRetrieval"
  ENV_FILE="$DEST/code/.env"
  PATH="$STUB_BIN:$PATH"
  export PATH
}

setup_pod() {
  "${TEST_BASH:-bash}" "$SETUP_POD" --workspace "$WS" "$@"
}

@test "--help prints usage and exits 0" {
  run -0 setup_pod --help
  [[ $output == *"Usage:"* ]]
  [[ $output == *"--datasets"* ]]
}

@test "unknown option is a usage error" {
  run -2 setup_pod --bogus
  [[ $output == *"unknown option: --bogus"* ]]
}

@test "--gpus must be a positive integer" {
  run -2 setup_pod --gpus 'two'
  [[ $output == *"--gpus"* ]]
}

@test "a relative --data-dir is a usage error" {
  run -2 setup_pod --data-dir data
  [[ $output == *"absolute"* ]]
}

@test "a missing workspace directory fails" {
  run -1 "${TEST_BASH:-bash}" "$SETUP_POD" --workspace "$WS/nope"
  [[ $output == *"nope"* ]]
}

@test "non-interactive run writes .env with mode 600 and the pod settings" {
  run -0 setup_pod --non-interactive
  [[ -f $ENV_FILE ]]
  [[ $(file_mode "$ENV_FILE") == 600 ]]
  grep -qx "DEEPRETRIEVAL_DATA_DIR='$WS/data'" "$ENV_FILE"
  grep -qx "HF_HOME='$WS/.cache/huggingface'" "$ENV_FILE"
  grep -qx "UV_CACHE_DIR='$WS/.cache/uv'" "$ENV_FILE"
  grep -qx "UV_PYTHON_INSTALL_DIR='$WS/.cache/uv-python'" "$ENV_FILE"
  grep -qx "RAY_ENABLE_UV_RUN_RUNTIME_ENV='0'" "$ENV_FILE"
  grep -qx "VERL_LOGGING_LEVEL='INFO'" "$ENV_FILE"
  # Missing optional secrets are skipped, not written empty, and named in the output.
  [[ $output == *"PUBMED_API_KEY: not set"* ]]
  run ! grep -q 'PUBMED_API_KEY' "$ENV_FILE"
}

@test "full run clones, syncs, downloads pubmed_32 and verifies" {
  run -0 setup_pod --non-interactive
  grep -qF "git clone -- https://github.com/objones25/DeepRetrieval.git $DEST" "$CALLS"
  grep -qF "uv --directory $DEST/code sync --extra gpu --extra retrieval" "$CALLS"
  grep -qF "download_datasets.py --categories search_engine --datasets pubmed_32 --output_dir $WS/data" "$CALLS"
  grep -qF 'nvidia-smi -L' "$CALLS"
  grep -qF 'require_triton_toolchain()' "$CALLS"
  [[ -f "$WS/data/search_engine/pubmed_32/train.parquet" ]]
}

@test "secrets from the environment are written but never printed" {
  PUBMED_API_KEY='pm-SECRET-1' WANDB_API_KEY='wb-SECRET-2' HF_TOKEN='hf-SECRET-3' \
    run -0 setup_pod --non-interactive
  grep -qx "PUBMED_API_KEY='pm-SECRET-1'" "$ENV_FILE"
  grep -qx "WANDB_API_KEY='wb-SECRET-2'" "$ENV_FILE"
  grep -qx "HF_TOKEN='hf-SECRET-3'" "$ENV_FILE"
  [[ $output != *SECRET* ]]
  [[ $output == *"PUBMED_API_KEY: set (environment)"* ]]
}

@test "environment values take precedence over an existing .env" {
  mkdir -p "$DEST/.git" "$DEST/code"
  printf "PUBMED_API_KEY='old-SECRET'\nDEEPRETRIEVAL_DATA_DIR='%s'\n" "$WS/old" >"$ENV_FILE"
  PUBMED_API_KEY='new-SECRET' DEEPRETRIEVAL_DATA_DIR="$WS/newdata" \
    run -0 setup_pod --non-interactive
  grep -qx "PUBMED_API_KEY='new-SECRET'" "$ENV_FILE"
  grep -qx "DEEPRETRIEVAL_DATA_DIR='$WS/newdata'" "$ENV_FILE"
  [[ $output != *SECRET* ]]
}

@test "--data-dir beats the environment" {
  DEEPRETRIEVAL_DATA_DIR="$WS/envdata" run -0 setup_pod --non-interactive --data-dir "$WS/flag data"
  grep -qx "DEEPRETRIEVAL_DATA_DIR='$WS/flag data'" "$ENV_FILE"
  [[ -f "$WS/flag data/search_engine/pubmed_32/test.parquet" ]]
}

@test "second run is idempotent: reuses .env secrets, updates the repo, skips the download" {
  PUBMED_API_KEY='pm-SECRET-1' run -0 setup_pod --non-interactive
  first=$(cat "$ENV_FILE")
  : >"$CALLS"
  run -0 setup_pod --non-interactive
  [[ $output != *SECRET* ]]
  grep -qx "PUBMED_API_KEY='pm-SECRET-1'" "$ENV_FILE"
  [[ $output == *"PUBMED_API_KEY: set (existing .env)"* ]]
  [[ $(file_mode "$ENV_FILE") == 600 ]]
  # Only the timestamp comment may differ.
  diff <(grep -v '^#' <<<"$first") <(grep -v '^#' "$ENV_FILE")
  run ! grep -q 'git clone' "$CALLS"
  grep -qF "git -C $DEST pull --ff-only" "$CALLS"
  run ! grep -q 'download_datasets.py' "$CALLS"
}

@test "unrelated keys in an existing .env are preserved" {
  mkdir -p "$DEST/.git" "$DEST/code"
  printf "OPENAI_API_KEY='keep-me'\n" >"$ENV_FILE"
  run -0 setup_pod --non-interactive
  grep -qx "OPENAI_API_KEY='keep-me'" "$ENV_FILE"
  [[ $output != *keep-me* ]]
}

@test "--force downloads even when the data is present" {
  run -0 setup_pod --non-interactive
  : >"$CALLS"
  run -0 setup_pod --non-interactive --force
  grep -q 'download_datasets.py' "$CALLS"
}

@test "--require-pubmed-key fails non-interactively, naming the variable, before any work" {
  run -1 setup_pod --non-interactive --require-pubmed-key
  [[ $output == *"PUBMED_API_KEY"* ]]
  [[ ! -e $CALLS ]] || run ! grep -q 'uv ' "$CALLS"
  [[ ! -e $ENV_FILE ]]
}

@test "--require-pubmed-key passes when the key is in the environment" {
  PUBMED_API_KEY='pm-SECRET-1' run -0 setup_pod --non-interactive --require-pubmed-key
  [[ $output != *SECRET* ]]
}

@test "stdin that is not a terminal counts as non-interactive" {
  run -1 setup_pod --require-pubmed-key </dev/null
  [[ $output == *"PUBMED_API_KEY"* ]]
}

@test "a secret containing a quote is rejected without echoing it" {
  PUBMED_API_KEY="bad'SECRET" run -1 setup_pod --non-interactive
  [[ $output == *"PUBMED_API_KEY"* ]]
  [[ $output != *SECRET* ]]
}

@test "--datasets and --categories take comma-separated lists" {
  run -0 setup_pod --non-interactive --categories search_engine --datasets pubmed_32,ctgov_32
  grep -qF -- "--categories search_engine --datasets pubmed_32 ctgov_32 --output_dir" "$CALLS"
  [[ -f "$WS/data/search_engine/ctgov_32/train.parquet" ]]
}

@test "a download that produces no parquet files fails naming the dataset" {
  STUB_DOWNLOAD_NOOP=1 run -1 setup_pod --non-interactive
  [[ $output == *"pubmed_32"* ]]
}

@test "a failing uv sync stops the script before the download" {
  STUB_UV_FAIL=sync run -1 setup_pod --non-interactive
  [[ $output == *"uv sync"* ]]
  run ! grep -q 'download_datasets.py' "$CALLS"
}

@test "a failing toolchain check stops the script" {
  STUB_UV_FAIL='-c' run -1 setup_pod --non-interactive
  [[ $output == *"toolchain"* ]]
}

@test "fewer GPUs than --gpus warns but succeeds" {
  STUB_GPUS=1 run -0 setup_pod --non-interactive --gpus 2
  [[ $output == *"WARNING"*"1 GPU"* ]]
}

@test "nvidia-smi failing stops the script" {
  STUB_NVIDIA_FAIL=1 run -1 setup_pod --non-interactive
  [[ $output == *"nvidia-smi"* ]]
}

@test "a dirty checkout is not updated" {
  mkdir -p "$DEST/.git" "$DEST/code"
  STUB_GIT_DIRTY=1 run -0 setup_pod --non-interactive
  [[ $output == *"uncommitted"* ]]
  run ! grep -q 'pull' "$CALLS"
}

@test "--branch checks the branch out on an existing clone" {
  mkdir -p "$DEST/.git" "$DEST/code"
  run -0 setup_pod --non-interactive --branch audit-round-2
  grep -qF "git -C $DEST checkout audit-round-2" "$CALLS"
}

@test "--branch is passed to a fresh clone" {
  run -0 setup_pod --non-interactive --branch audit-round-2 --repo https://example.com/r.git
  grep -qF "git clone --branch audit-round-2 -- https://example.com/r.git $DEST" "$CALLS"
}

@test "uv is installed onto the volume when missing" {
  nouv="${BATS_TEST_TMPDIR}/nouv bin"
  mkdir -p "$nouv"
  ln -s "$STUB_BIN/git" "$STUB_BIN/nvidia-smi" "$STUB_BIN/curl" "$nouv/"
  PATH="$nouv:/usr/bin:/bin" run -0 setup_pod --non-interactive
  [[ -x "$WS/.local/bin/uv" ]]
  grep -qF 'https://astral.sh/uv/' "$CALLS"
  grep -qF 'sync --extra gpu --extra retrieval' "$CALLS"
}

@test "--dry-run prints the actions, runs nothing and writes nothing" {
  PUBMED_API_KEY='pm-SECRET-1' run -0 setup_pod --dry-run
  [[ $output == *"would run:"*"git clone"* ]]
  [[ $output == *"would run:"*"sync --extra gpu --extra retrieval"* ]]
  [[ $output == *"would run:"*"download_datasets.py"* ]]
  [[ $output == *"would write"*".env"* ]]
  [[ $output != *SECRET* ]]
  [[ ! -e $CALLS ]]
  [[ ! -e $DEST ]]
}

@test "prompt_secret returns the typed value and does not echo it" {
  # Sourced in a child bash so the script's set -e and traps stay out of bats.
  run -0 --separate-stderr "${TEST_BASH:-bash}" -c \
    'source "$1"; prompt_secret PUBMED_API_KEY' _ "$SETUP_POD" <<<'pm-SECRET-9'
  [[ $output == 'pm-SECRET-9' ]]
  [[ $stderr == *"PUBMED_API_KEY (optional, input hidden"* ]]
  [[ $stderr != *SECRET* ]]
}
