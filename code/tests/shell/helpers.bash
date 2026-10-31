# Shared helpers for the bats suites in this directory.
# Run the suites with:  npx --yes bats@1.13.0 code/tests/shell   (or plain `bats` if installed)
# TEST_BASH picks the bash that runs setup_pod.sh (default: bash on PATH), e.g.
#   TEST_BASH=/bin/bash npx --yes bats@1.13.0 code/tests/shell     # macOS bash 3.2

SCRIPTS_DIR="$(cd "${BATS_TEST_DIRNAME}/../../scripts" && pwd)"
SETUP_POD="${SCRIPTS_DIR}/setup_pod.sh"
COMMON_SH="${SCRIPTS_DIR}/common.sh"

# Octal permission bits of a file, on BSD (macOS) or GNU (Linux) stat.
file_mode() {
  stat -f '%Lp' "$1" 2>/dev/null || stat -c '%a' "$1"
}

# Writes stub executables for the external commands setup_pod.sh runs (git, uv,
# nvidia-smi, curl) into $STUB_BIN. Every call is appended to $CALLS, one line each.
# Knobs (environment): STUB_GPUS (number of GPUs nvidia-smi lists, default 2),
# STUB_NVIDIA_FAIL=1, STUB_UV_FAIL=<word> (uv exits 1 when an argument equals it),
# STUB_DOWNLOAD_NOOP=1 (download_datasets.py creates nothing), STUB_GIT_DIRTY=1.
make_stubs() {
  STUB_BIN="${BATS_TEST_TMPDIR}/stub bin"
  CALLS="${BATS_TEST_TMPDIR}/calls.log"
  export STUB_BIN CALLS
  mkdir -p "$STUB_BIN"

  cat >"$STUB_BIN/git" <<'EOF'
#!/usr/bin/env bash
set -eu
printf 'git %s\n' "$*" >>"$CALLS"
if [[ $1 == clone ]]; then
  dest=${!#}
  mkdir -p "$dest/.git" "$dest/code/scripts"
  exit 0
fi
if [[ $1 == -C ]]; then shift 2; fi
case $1 in
  rev-parse) printf '.git\n' ;;
  status) if [[ ${STUB_GIT_DIRTY:-0} == 1 ]]; then printf ' M code/x.py\n'; fi ;;
  remote) printf 'https://github.com/objones25/DeepRetrieval.git\n' ;;
  *) : ;;
esac
EOF

  cat >"$STUB_BIN/uv" <<'EOF'
#!/usr/bin/env bash
set -eu
printf 'uv %s\n' "$*" >>"$CALLS"
for a in "$@"; do
  if [[ -n ${STUB_UV_FAIL:-} && $a == "$STUB_UV_FAIL" ]]; then exit 1; fi
done
if [[ " $* " == *" download_datasets.py "* && ${STUB_DOWNLOAD_NOOP:-0} != 1 ]]; then
  out='' cats=() dss=() mode=''
  for a in "$@"; do
    case $a in
      --output_dir) mode=out ;;
      --categories) mode=cat ;;
      --datasets) mode=ds ;;
      --*) mode='' ;;
      *)
        case $mode in
          out) out=$a ;;
          cat) cats+=("$a") ;;
          ds) dss+=("$a") ;;
        esac
        ;;
    esac
  done
  for c in "${cats[@]}"; do
    for d in "${dss[@]}"; do
      mkdir -p "$out/$c/$d"
      : >"$out/$c/$d/train.parquet"
      : >"$out/$c/$d/test.parquet"
    done
  done
fi
EOF

  cat >"$STUB_BIN/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
set -eu
printf 'nvidia-smi %s\n' "$*" >>"$CALLS"
if [[ ${STUB_NVIDIA_FAIL:-0} == 1 ]]; then exit 9; fi
i=0
while ((i < ${STUB_GPUS:-2})); do
  printf 'GPU %d: NVIDIA A100-SXM4-80GB (UUID: GPU-%d)\n' "$i" "$i"
  i=$((i + 1))
done
EOF

  # curl -o FILE URL: writes an installer that drops a uv stub into $UV_INSTALL_DIR.
  cat >"$STUB_BIN/curl" <<'EOF'
#!/usr/bin/env bash
set -eu
printf 'curl %s\n' "$*" >>"$CALLS"
out=''
while (($# > 0)); do
  if [[ $1 == -o ]]; then out=$2; shift; fi
  shift
done
cat >"$out" <<'INSTALLER'
mkdir -p "$UV_INSTALL_DIR"
cp "$STUB_BIN/uv" "$UV_INSTALL_DIR/uv"
INSTALLER
EOF

  chmod +x "$STUB_BIN"/*
}

# Clears every variable the scripts read so the host environment cannot leak in.
clear_dr_env() {
  unset PUBMED_API_KEY WANDB_API_KEY HF_TOKEN DEEPRETRIEVAL_DATA_DIR HF_HOME \
    UV_CACHE_DIR UV_PYTHON_INSTALL_DIR RAY_ENABLE_UV_RUN_RUNTIME_ENV VERL_LOGGING_LEVEL \
    VIRTUAL_ENV CUDA_VISIBLE_DEVICES
}
