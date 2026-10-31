#!/usr/bin/env bash
#
# setup_pod.sh: prepare a Runpod GPU pod to train DeepRetrieval, keeping everything
# that is slow to rebuild (repo, venv, uv cache, managed Python, HF cache, datasets)
# on the network volume so that a fresh pod on the same volume is ready in minutes.
#
# Runs ON the pod. Idempotent: every step checks what is already there.
# Usage: see usage() below, or run with --help.
# Requires: bash 3.2+, git, curl (only when uv is missing), nvidia-smi; installs uv.
# Targets Ubuntu 22.04 (bash 5.1) on the pod; tested on macOS bash 3.2 and 5.3.
# bash-target: 3.2
# audit-bash disable-file=B402 # asked for as a shell script: it is glue around git, uv, nvidia-smi
#
# Exit status: 0 success, 1 failure, 2 usage error.
# All diagnostics go to stderr. Secret values are never printed or logged.

# shellcheck disable=SC2310 # run_cmd and the predicates are called under || / if on purpose:
# their status is the wrapped command's, and every such call handles it explicitly.
set -Eeuo pipefail
shopt -s inherit_errexit 2>/dev/null || true # bash 4.4+: $(...) inherits set -e

readonly SCRIPT_NAME=${0##*/}
readonly DEFAULT_REPO=https://github.com/objones25/DeepRetrieval.git
# uv that wrote code/uv.lock; any later uv reads it too (--uv-version to change).
readonly DEFAULT_UV_VERSION=0.13.6
readonly SECRET_KEYS='PUBMED_API_KEY WANDB_API_KEY HF_TOKEN'
readonly SETTING_KEYS='DEEPRETRIEVAL_DATA_DIR HF_HOME UV_CACHE_DIR UV_PYTHON_INSTALL_DIR RAY_ENABLE_UV_RUN_RUNTIME_ENV VERL_LOGGING_LEVEL'
readonly TOOLCHAIN_CHECK='from verl.utils.toolchain import require_triton_toolchain; require_triton_toolchain()'

# ---- logging ----

log() {
  local now
  now=$(date '+%H:%M:%S')
  printf '%s %s: %s\n' "$now" "$SCRIPT_NAME" "$*" >&2
}
warn() { log "WARNING: $*"; }
die() {
  log "ERROR: $*"
  exit 1
}

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME [options]

Prepares this pod for DeepRetrieval training, reusing the network volume:
  1. checks the workspace is a mounted volume and nvidia-smi sees the GPUs;
  2. installs uv onto the volume if missing;
  3. clones the repo onto the volume, or fast-forwards an existing clean checkout;
  4. writes code/.env (mode 600): secrets from the environment, else the existing
     .env, else a prompt (interactive only); plus the pod's cache/data settings;
  5. uv sync --extra gpu --extra retrieval (venv and caches on the volume);
  6. downloads the datasets unless they are already present;
  7. checks the Triton toolchain and the datasets' train/test parquet files.

Options:
  --workspace DIR      volume root (default /workspace; only /workspace is checked
                       to be a mount point)
  --repo URL           git remote to clone (default $DEFAULT_REPO)
  --branch NAME        branch to clone or check out (default: the remote's default
                       branch on clone, the current branch on update)
  --dest DIR           checkout location (default WORKSPACE/DeepRetrieval)
  --data-dir DIR       absolute DEEPRETRIEVAL_DATA_DIR (default: \$DEEPRETRIEVAL_DATA_DIR,
                       else the value in code/.env, else WORKSPACE/data)
  --categories LIST    comma-separated dataset categories (default search_engine)
  --datasets LIST      comma-separated datasets (default pubmed_32)
  --force              download the datasets even if they are present
  --gpus N             GPUs expected; fewer is a warning (default 2)
  --non-interactive    never prompt; optional secrets that are missing are skipped
  --require-pubmed-key fail if PUBMED_API_KEY ends up unset
  --uv-version VER     uv to install when missing (default $DEFAULT_UV_VERSION)
  -n, --dry-run        print the actions without executing them or writing files
  -h, --help           show this help and exit

Secrets (all optional): PUBMED_API_KEY (NCBI 10 req/s instead of 3), WANDB_API_KEY,
HF_TOKEN (gated models such as meta-llama/Llama-3.2-3B-Instruct). Set them as Runpod
pod environment variables/secrets for unattended runs. Prompts happen only when stdin
is a terminal and --non-interactive is not given.
Exit status: 0 success, 1 failure, 2 usage error.
EOF
}

usage_error() {
  printf '%s: %s\n' "$SCRIPT_NAME" "$*" >&2
  printf "Try '%s --help'.\n" "$SCRIPT_NAME" >&2
  exit 2
}

# ---- cleanup and error reporting ----

tmpdir=''
tmp_env=''
code_dir=''
env_file=''
# resolve_settings fills val_<KEY> and src_<KEY> for each key via printf -v.
val_DEEPRETRIEVAL_DATA_DIR=''
src_DEEPRETRIEVAL_DATA_DIR=''

cleanup() {
  if [[ -n $tmp_env ]]; then rm -f -- "$tmp_env" || true; fi
  if [[ -n $tmpdir ]]; then rm -rf -- "${tmpdir:?}" || true; fi
}

on_error() {
  local status=$?
  log "ERROR: line ${BASH_LINENO[0]}: '$BASH_COMMAND' exited with status $status"
}

trap cleanup EXIT
trap on_error ERR

# ---- argument parsing ----

workspace=/workspace
repo=$DEFAULT_REPO
branch=''
dest=''
data_dir_flag=''
categories=search_engine
datasets=pubmed_32
force=0
gpus=2
non_interactive=0
require_pubmed=0
uv_version=$DEFAULT_UV_VERSION
dry_run=0

# need_value OPTION ARGC: an option that takes a value must have one.
need_value() {
  # audit-bash disable=B307 # $2 is the caller's $#, never outside input
  if (($2 < 2)); then usage_error "$1 needs an argument"; fi
}

parse_args() {
  while (($# > 0)); do
    case $1 in
      -h | --help)
        usage
        exit 0
        ;;
      -n | --dry-run) dry_run=1 ;;
      --force) force=1 ;;
      --non-interactive) non_interactive=1 ;;
      --require-pubmed-key) require_pubmed=1 ;;
      --workspace | --repo | --branch | --dest | --data-dir | --categories | --datasets | --gpus | --uv-version)
        need_value "$1" "$#"
        set_option "${1#--}" "$2"
        shift
        ;;
      --*=*)
        local name=${1%%=*}
        case $name in
          --workspace | --repo | --branch | --dest | --data-dir | --categories | --datasets | --gpus | --uv-version)
            set_option "${name#--}" "${1#*=}"
            ;;
          *) usage_error "unknown option: $name" ;;
        esac
        ;;
      *) usage_error "unknown option: $1" ;;
    esac
    shift
  done

  case $gpus in
    '' | *[!0-9]* | 0*) usage_error "--gpus must be a positive integer, got '$gpus'" ;;
  esac
  if [[ -n $data_dir_flag && $data_dir_flag != /* ]]; then
    usage_error "--data-dir must be an absolute path, got '$data_dir_flag'"
  fi
  local list
  for list in "$categories" "$datasets"; do
    case $list in
      '' | *[!A-Za-z0-9_.,-]* | ,* | *, | *,,*)
        usage_error "--categories/--datasets take comma-separated names, got '$list'"
        ;;
    esac
  done
}

set_option() {
  case $1 in
    workspace) workspace=$2 ;;
    repo) repo=$2 ;;
    branch) branch=$2 ;;
    dest) dest=$2 ;;
    data-dir) data_dir_flag=$2 ;;
    categories) categories=$2 ;;
    datasets) datasets=$2 ;;
    gpus) gpus=$2 ;;
    uv-version) uv_version=$2 ;;
  esac
}

# ---- helpers ----

run_cmd() {
  # Runs a command, or only prints it in dry-run mode. Never pass secrets as arguments.
  if ((dry_run)); then
    printf 'would run:' >&2
    printf ' %q' "$@" >&2
    printf '\n' >&2
  else
    "$@"
  fi
}

# env_file_value KEY: prints KEY's value from the existing code/.env (quotes stripped);
# returns 1 when the file or the key is missing or the value is empty.
env_file_value() {
  local key=$1 line value
  [[ -f $env_file ]] || return 1
  line=$(grep -E "^[[:space:]]*(export[[:space:]]+)?${key}=" "$env_file" | sed -n '$p') || return 1
  [[ -n $line ]] || return 1
  value=${line#*"${key}="}
  case $value in
    \'*\') value=${value#\'} value=${value%\'} ;;
    \"*\") value=${value#\"} value=${value%\"} ;;
  esac
  [[ -n $value ]] || return 1
  printf '%s' "$value"
}

default_for() {
  case $1 in
    DEEPRETRIEVAL_DATA_DIR) printf '%s' "$workspace/data" ;;
    HF_HOME) printf '%s' "$workspace/.cache/huggingface" ;;
    UV_CACHE_DIR) printf '%s' "$workspace/.cache/uv" ;;
    # uv-managed CPython must live on the volume too: the venv links to it.
    UV_PYTHON_INSTALL_DIR) printf '%s' "$workspace/.cache/uv-python" ;;
    RAY_ENABLE_UV_RUN_RUNTIME_ENV) printf '0' ;;
    VERL_LOGGING_LEVEL) printf 'INFO' ;;
    *) printf '' ;;
  esac
}

missing_secret_note() {
  case $1 in
    PUBMED_API_KEY) printf '%s' "PubMed rewards use NCBI's keyless limit of 3 requests/s (a key allows 10)" ;;
    WANDB_API_KEY) printf '%s' "launchers log to W&B (trainer.logger=['wandb']); set it or export WANDB_MODE=offline" ;;
    HF_TOKEN) printf '%s' "gated models (meta-llama/Llama-3.2-3B-Instruct in scripts/train/pubmed_32.sh) cannot be downloaded" ;;
  esac
}

# prompt_secret KEY: reads a value from the terminal without echo; prints it.
prompt_secret() {
  local reply=''
  printf '%s (optional, input hidden, Enter to skip): ' "$1" >&2
  IFS= read -r -s reply || reply=''
  printf '\n' >&2
  printf '%s' "$reply"
}

# ---- steps ----

check_workspace() {
  [[ -d $workspace ]] || die "workspace $workspace does not exist; attach a network volume at /workspace or pass --workspace DIR"
  workspace=$(cd -- "$workspace" && pwd) || die "cannot enter workspace $workspace"
  if [[ $workspace != /workspace ]]; then
    log "workspace: $workspace (not checking that it is a mount point)"
    return 0
  fi
  local root_mnt ws_mnt
  if command -v mountpoint >/dev/null 2>&1; then
    mountpoint -q /workspace || die "/workspace is not a mount point; create the pod with a network volume mounted at /workspace"
  else
    root_mnt=$(df -P / | awk 'NR==2 {print $6}') || die "df failed"
    ws_mnt=$(df -P /workspace | awk 'NR==2 {print $6}') || die "df failed"
    [[ $ws_mnt == /workspace && $root_mnt != /workspace ]] \
      || die "/workspace is not a mount point; create the pod with a network volume mounted at /workspace"
  fi
  log "workspace: /workspace is a mounted volume"
}

check_gpus() {
  local listing count
  if ((dry_run)); then
    run_cmd nvidia-smi -L
    return 0
  fi
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found: this pod has no NVIDIA driver; pick a GPU pod"
  listing=$(nvidia-smi -L) || die "nvidia-smi -L failed: no usable NVIDIA GPU in this pod"
  count=$(printf '%s\n' "$listing" | awk '/^GPU / {n++} END {print n + 0}')
  if ((count < gpus)); then
    warn "nvidia-smi lists $count GPU(s), fewer than --gpus $gpus; scripts/train/pubmed_32.sh needs 2 (trainer.n_gpus_per_node=2, tensor_model_parallel_size=2)"
  else
    log "GPUs: $count visible"
  fi
}

# Resolves every setting and secret into val_KEY / src_KEY. Fails before any slow work
# when a required value is missing.
resolve_settings() {
  local interactive=0 key value src note
  if ((non_interactive == 0)) && [[ -t 0 ]]; then interactive=1; fi

  for key in $SETTING_KEYS $SECRET_KEYS; do
    value='' src=''
    if [[ $key == DEEPRETRIEVAL_DATA_DIR && -n $data_dir_flag ]]; then
      value=$data_dir_flag src='--data-dir'
    elif [[ -n ${!key:-} ]]; then
      value=${!key} src=environment
    elif value=$(env_file_value "$key"); then
      src='existing .env'
    elif [[ " $SECRET_KEYS " == *" $key "* ]]; then
      value='' src=''
      if ((interactive)) && ((dry_run == 0)); then
        value=$(prompt_secret "$key")
        if [[ -n $value ]]; then src=entered; fi
      fi
    else
      value=$(default_for "$key") src=default
    fi

    case $value in
      *\'* | *$'\n'* | *$'\r'*)
        die "$key contains a quote or a line break; the .env format cannot store it (value not shown)"
        ;;
    esac
    printf -v "val_$key" '%s' "$value"
    printf -v "src_$key" '%s' "$src"
  done

  if [[ $val_DEEPRETRIEVAL_DATA_DIR != /* ]]; then
    die "DEEPRETRIEVAL_DATA_DIR must be absolute, got '$val_DEEPRETRIEVAL_DATA_DIR' ($src_DEEPRETRIEVAL_DATA_DIR)"
  fi

  for key in $SETTING_KEYS; do
    value="val_$key" src="src_$key"
    log "$key=${!value} (${!src})"
  done
  for key in $SECRET_KEYS; do
    value="val_$key" src="src_$key"
    if [[ -n ${!value} ]]; then
      log "$key: set (${!src})"
    elif [[ $key == PUBMED_API_KEY ]] && ((require_pubmed)); then
      die "PUBMED_API_KEY is required (--require-pubmed-key) but not set: export it (Runpod pod env var or secret) or run interactively"
    else
      note=$(missing_secret_note "$key") || die "missing_secret_note failed"
      log "$key: not set - $note"
    fi
  done

  # This process (uv, download_datasets.py) uses the same settings as the launchers.
  for key in $SETTING_KEYS $SECRET_KEYS; do
    value="val_$key"
    if [[ -n ${!value} ]]; then export "$key=${!value}"; fi
  done
}

ensure_uv() {
  local v installer
  if command -v uv >/dev/null 2>&1; then
    if ((dry_run)); then
      v=$(command -v uv) || die "command -v uv failed"
      log "uv: found at $v"
      return 0
    fi
    v=$(uv --version) || die "uv --version failed"
    log "uv: $v"
    return 0
  fi
  log "uv not found; installing uv $uv_version into $workspace/.local/bin"
  installer=$tmpdir/uv-install.sh
  run_cmd curl -LsSf --retry 3 -o "$installer" "https://astral.sh/uv/$uv_version/install.sh" \
    || die "downloading the uv installer failed"
  run_cmd env UV_INSTALL_DIR="$workspace/.local/bin" UV_NO_MODIFY_PATH=1 sh "$installer" \
    || die "the uv installer failed"
  if ((dry_run == 0)); then
    command -v uv >/dev/null 2>&1 || die "uv still not on PATH after installing it into $workspace/.local/bin"
  fi
}

sync_repo() {
  local dirty
  if [[ ! -e $dest ]]; then
    local clone_args=(clone)
    if [[ -n $branch ]]; then clone_args+=(--branch "$branch"); fi
    run_cmd git "${clone_args[@]}" -- "$repo" "$dest" || die "git clone of $repo into $dest failed"
    return 0
  fi
  [[ -d $dest/.git ]] || die "$dest exists but is not a git checkout; move it away or pass --dest DIR"
  dirty=$(git -C "$dest" status --porcelain --untracked-files=no) || die "git status failed in $dest"
  if [[ -n $dirty ]]; then
    warn "$dest has uncommitted changes to tracked files; not updating it (commit or stash them and rerun)"
    return 0
  fi
  run_cmd git -C "$dest" fetch --prune origin || die "git fetch failed in $dest"
  if [[ -n $branch ]]; then
    run_cmd git -C "$dest" checkout "$branch" || die "git checkout $branch failed in $dest"
  fi
  run_cmd git -C "$dest" pull --ff-only \
    || die "git pull --ff-only failed in $dest (local commits diverge from the remote?); resolve it by hand and rerun"
}

write_env_file() {
  local key value managed now
  # shellcheck disable=SC2086 # the key lists are split into words on purpose
  managed=$(printf '%s|' $SETTING_KEYS $SECRET_KEYS)
  managed=${managed%|}
  if ((dry_run)); then
    local names=''
    for key in $SETTING_KEYS $SECRET_KEYS; do
      value="val_$key"
      if [[ -n ${!value} ]]; then names="$names $key"; fi
    done
    log "would write $env_file (mode 600) with:$names"
    return 0
  fi
  [[ -d $code_dir ]] || die "$code_dir does not exist; is $dest a DeepRetrieval checkout?"
  now=$(date '+%Y-%m-%d %H:%M:%S')
  # mktemp creates the file 0600; same directory so the mv below is atomic.
  tmp_env=$(mktemp "$code_dir/.env.tmp.XXXXXX") || die "cannot create a temp file in $code_dir"
  chmod 600 "$tmp_env" || die "chmod 600 $tmp_env failed"
  {
    printf '# Written by scripts/setup_pod.sh on %s. Loaded by scripts/common.sh;\n' "$now"
    printf '# variables already set in the environment win. Secrets: keep this file mode 600.\n'
    for key in $SETTING_KEYS $SECRET_KEYS; do
      value="val_$key"
      if [[ -n ${!value} ]]; then printf "%s='%s'\n" "$key" "${!value}"; fi
    done
    if [[ -f $env_file ]]; then
      # Keep KEY=VALUE lines this script does not manage (e.g. OPENAI_API_KEY).
      grep -v -E "^[[:space:]]*(#|\$)|^[[:space:]]*(export[[:space:]]+)?($managed)=" "$env_file" \
        || [[ $? -eq 1 ]]
    fi
  } >"$tmp_env" || die "writing $tmp_env failed"
  mv -f -- "$tmp_env" "$env_file" || die "cannot move the new .env into place at $env_file"
  tmp_env=''
  log "wrote $env_file (mode 600)"
}

uv_sync() {
  run_cmd mkdir -p -- "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HOME" "$DEEPRETRIEVAL_DATA_DIR" \
    || die "cannot create the cache/data directories on $workspace"
  log "uv sync (first run on a new volume downloads torch/vllm/flash-attn: several GB)"
  run_cmd uv --directory "$code_dir" sync --extra gpu --extra retrieval \
    || die "uv sync --extra gpu --extra retrieval failed in $code_dir (see uv's message above)"
}

# missing_datasets: prints the requested datasets that lack a train.parquet and a
# test.parquet under DATA_DIR/<category>/<dataset>/ (any depth), space-separated.
missing_datasets() {
  local ds cat dir found train test missing=''
  local IFS=,
  for ds in $datasets; do
    found=0
    for cat in $categories; do
      dir=$DEEPRETRIEVAL_DATA_DIR/$cat/$ds
      [[ -d $dir ]] || continue
      train=$(find "$dir" -type f -name train.parquet) || return 1
      test=$(find "$dir" -type f -name test.parquet) || return 1
      if [[ -n $train && -n $test ]]; then found=1; fi
    done
    if ((found == 0)); then missing="$missing $ds"; fi
  done
  printf '%s' "${missing# }"
}

download_data() {
  local missing cat_args ds_args
  missing=$(missing_datasets) || die "cannot inspect $DEEPRETRIEVAL_DATA_DIR"
  if [[ -z $missing ]] && ((force == 0)); then
    log "datasets present in $DEEPRETRIEVAL_DATA_DIR ($datasets); skipping download (--force to redo)"
    return 0
  fi
  IFS=, read -r -a cat_args <<<"$categories"
  IFS=, read -r -a ds_args <<<"$datasets"
  log "downloading $datasets ($categories) into $DEEPRETRIEVAL_DATA_DIR"
  run_cmd uv --directory "$code_dir" run python download_datasets.py \
    --categories ${cat_args[@]+"${cat_args[@]}"} --datasets ${ds_args[@]+"${ds_args[@]}"} \
    --output_dir "$DEEPRETRIEVAL_DATA_DIR" || die "download_datasets.py failed"
  if ((dry_run)); then return 0; fi
  # download_datasets.py exits 0 even when files fail, so check the result.
  missing=$(missing_datasets) || die "cannot inspect $DEEPRETRIEVAL_DATA_DIR"
  if [[ -n $missing ]]; then
    die "after download_datasets.py, no train.parquet + test.parquet under $DEEPRETRIEVAL_DATA_DIR/{$categories}/ for: $missing (see its log above)"
  fi
}

check_toolchain() {
  run_cmd uv --directory "$code_dir" run python -c "$TOOLCHAIN_CHECK" \
    || die "Triton toolchain check failed (see the message above); the venv must use uv-managed Python (python-preference = only-managed)"
  if ((dry_run == 0)); then log "Triton toolchain: OK"; fi
}

main() {
  parse_args "$@"
  if ((dry_run)); then log "dry run: printing actions, executing nothing, writing nothing"; fi

  check_workspace
  dest=${dest:-$workspace/DeepRetrieval}
  code_dir=$dest/code
  env_file=$code_dir/.env
  PATH="$workspace/.local/bin:$PATH"
  export PATH

  # Fast checks first, so a wrong pod or a missing required key fails in seconds.
  check_gpus
  resolve_settings

  tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/$SCRIPT_NAME.XXXXXX") || die "mktemp failed"
  ensure_uv
  sync_repo
  write_env_file
  uv_sync
  download_data
  check_toolchain

  if ((dry_run)); then
    log "dry run complete"
    return 0
  fi
  log "ready. Train with:  cd $code_dir && sh scripts/train/pubmed_32.sh"
  log "launchers source scripts/common.sh, which loads $env_file"
}

# main and exit are read together, so a `git pull` that rewrites this file while it
# runs cannot make bash execute the new text.
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  main "$@"
  exit
fi
