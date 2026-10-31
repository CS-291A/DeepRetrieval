#!/bin/sh
# Sourced, not executed (no execute bit): the shebang only names the dialect for tools.
#
# common.sh: shared setup for the launchers in scripts/train and scripts/eval.
#
# Sourced, never executed, as the first command of every launcher:
#     . "$(dirname "$0")/../common.sh" || exit 1
# Launchers run under `sh` (dash on Ubuntu), so this file is POSIX sh: no bash-isms,
# no `local`, and its variables are prefixed _dr_ and unset at the end.
#
# In order, it:
#   1. finds code/ (the directory holding scripts/common.sh) from the launcher's $0;
#   2. loads code/.env when present and exports its KEY=VALUE lines, except that a
#      variable already set to a non-empty value in the environment keeps that value;
#   3. defaults DEEPRETRIEVAL_DATA_DIR=<code>/data (the Python default) and
#      RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 (otherwise Ray under `uv run` ships a copy of the
#      working dir that honours .gitignore, and workers cannot see data/);
#   4. puts code/.venv/bin first on PATH when no virtualenv is active;
#   5. creates code/exp_log/ (launchers tee their logs into it) and cd's to code/.
# CUDA_VISIBLE_DEVICES is never set here: export it before launching to pick GPUs.
# Prints one summary line to stderr. Secret values are never printed.
# On failure it prints the reason and returns 1 (the launcher then exits 1).

_dr_fail() {
  printf 'common.sh: error: %s\n' "$*" >&2
  return 1
}

# ---- 1. find code/ ----
_dr_code=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd) || {
  _dr_fail "cannot resolve the launcher directory from \$0='$0'"
  return 1
}
while [ ! -f "$_dr_code/scripts/common.sh" ]; do
  case $_dr_code in
    / | '' | .)
      _dr_fail "no scripts/common.sh above the launcher ($0); run launchers from the repo"
      return 1
      ;;
  esac
  _dr_code=$(dirname -- "$_dr_code")
done

# ---- 2. load code/.env without overriding the environment ----
_dr_env_file=$_dr_code/.env
_dr_env_state=absent
if [ -f "$_dr_env_file" ]; then
  # Accept only comments, blank lines and KEY=VALUE where VALUE is single-quoted,
  # double-quoted without $ ` or \, or a bare word, so sourcing it runs nothing.
  _dr_q="'"
  _dr_ok='^[[:space:]]*(#.*)?$|^[[:space:]]*(export[[:space:]]+)?[A-Za-z_][A-Za-z0-9_]*='
  _dr_ok="$_dr_ok(${_dr_q}[^${_dr_q}]*${_dr_q}|\"[^\"\$\`\\\\]*\"|[A-Za-z0-9_./:@,+=-]*)[[:space:]]*\$"
  _dr_bad=$(grep -n -v -E "$_dr_ok" "$_dr_env_file" | sed -n 1p | cut -d: -f1)
  if [ -n "$_dr_bad" ]; then
    _dr_fail "$_dr_env_file line $_dr_bad is not KEY=VALUE (quote the value; no \$, backticks or backslashes); not loading it"
    return 1
  fi
  _dr_keys=$(sed -n -E 's/^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=.*/\2/p' "$_dr_env_file")
  _dr_keep=''
  # eval is POSIX sh's only indirect expansion; the keys are identifiers validated
  # by the grep above, and values are only ever referenced, never interpolated.
  for _dr_k in $_dr_keys; do
    eval "_dr_v=\${$_dr_k-}"
    if [ -n "$_dr_v" ]; then
      eval "_dr_saved_$_dr_k=\$_dr_v"
      _dr_keep="$_dr_keep $_dr_k"
    fi
  done
  set -a
  # shellcheck disable=SC1090 # path is computed at runtime; content validated above
  . "$_dr_env_file"
  set +a
  for _dr_k in $_dr_keep; do
    eval "$_dr_k=\$_dr_saved_$_dr_k"
    export "${_dr_k?}"
    unset "_dr_saved_$_dr_k"
  done
  _dr_env_state=loaded
fi

# ---- 3. defaults ----
: "${DEEPRETRIEVAL_DATA_DIR:=$_dr_code/data}"
: "${RAY_ENABLE_UV_RUN_RUNTIME_ENV:=0}"
export DEEPRETRIEVAL_DATA_DIR RAY_ENABLE_UV_RUN_RUNTIME_ENV

# ---- 4. project virtualenv ----
if [ -z "${VIRTUAL_ENV:-}" ] && [ -x "$_dr_code/.venv/bin/python" ]; then
  VIRTUAL_ENV=$_dr_code/.venv
  PATH=$VIRTUAL_ENV/bin:$PATH
  export VIRTUAL_ENV PATH
fi

# ---- 5. log dir and working directory ----
mkdir -p -- "$_dr_code/exp_log" || {
  _dr_fail "cannot create $_dr_code/exp_log"
  return 1
}
cd -- "$_dr_code" || {
  _dr_fail "cannot cd to $_dr_code"
  return 1
}

printf 'common.sh: code=%s DEEPRETRIEVAL_DATA_DIR=%s .env=%s CUDA_VISIBLE_DEVICES=%s\n' \
  "$_dr_code" "$DEEPRETRIEVAL_DATA_DIR" "$_dr_env_state" \
  "${CUDA_VISIBLE_DEVICES:-<unset: all GPUs>}" >&2
if [ ! -d "$DEEPRETRIEVAL_DATA_DIR" ]; then
  printf 'common.sh: warning: DEEPRETRIEVAL_DATA_DIR=%s does not exist (run scripts/setup_pod.sh or download_datasets.py)\n' \
    "$DEEPRETRIEVAL_DATA_DIR" >&2
fi

unset _dr_q _dr_code _dr_env_file _dr_env_state _dr_ok _dr_bad _dr_keys _dr_keep _dr_k _dr_v
unset -f _dr_fail
