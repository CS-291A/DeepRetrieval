#!/usr/bin/env bats
# Tests for code/scripts/common.sh, sourced by every launcher under scripts/train and
# scripts/eval. Launchers run under `sh` (dash on Ubuntu), so each behaviour is checked
# under dash and bash. A fake code tree with a space in its path holds a copy of
# common.sh and a launcher that prints what it sees.

bats_require_minimum_version 1.5.0

load helpers

setup() {
  clear_dr_env
  CODE="${BATS_TEST_TMPDIR}/my code"
  mkdir -p "$CODE/scripts/train/nested"
  cp "$COMMON_SH" "$CODE/scripts/common.sh"
  for f in "$CODE/scripts/train/show.sh" "$CODE/scripts/train/nested/show.sh"; do
    if [[ $f == */nested/* ]]; then rel='../..'; else rel='..'; fi
    cat >"$f" <<EOF
. "\$(dirname "\$0")/${rel}/common.sh" || exit 1
printf 'PWD=%s\n' "\$(pwd)"
printf 'DATA=%s\n' "\$DEEPRETRIEVAL_DATA_DIR"
printf 'RAY=%s\n' "\$RAY_ENABLE_UV_RUN_RUNTIME_ENV"
printf 'PUBMED=%s\n' "\${PUBMED_API_KEY-<unset>}"
printf 'HF_HOME=%s\n' "\${HF_HOME-<unset>}"
printf 'EXPORTED_PUBMED=%s\n' "\$(sh -c 'printf %s "\${PUBMED_API_KEY-<unset>}"')"
printf 'PATH=%s\n' "\$PATH"
printf 'VIRTUAL_ENV=%s\n' "\${VIRTUAL_ENV-<unset>}"
printf 'CUDA=%s\n' "\${CUDA_VISIBLE_DEVICES-<unset>}"
EOF
  done
  SHOW="$CODE/scripts/train/show.sh"
  cd "$BATS_TEST_TMPDIR" || return 1
}

shells() {
  printf '%s\n' /bin/dash bash
}

@test "defaults: data dir is <code>/data, Ray uv runtime env off, cwd is code/, exp_log created" {
  for sh in $(shells); do
    run -0 --separate-stderr "$sh" "$SHOW"
    [[ $output == *"PWD=$CODE"$'\n'* ]]
    [[ $output == *"DATA=$CODE/data"$'\n'* ]]
    [[ $output == *"RAY=0"$'\n'* ]]
    [[ $output == *"PUBMED=<unset>"* ]]
    [[ -d "$CODE/exp_log" ]]
    rm -rf "$CODE/exp_log"
  done
}

@test "works from a nested launcher directory" {
  for sh in $(shells); do
    run -0 --separate-stderr "$sh" "$CODE/scripts/train/nested/show.sh"
    [[ $output == *"PWD=$CODE"$'\n'* ]]
  done
}

@test "works when invoked with a relative path from code/" {
  cd "$CODE"
  for sh in $(shells); do
    run -0 --separate-stderr "$sh" scripts/train/show.sh
    [[ $output == *"DATA=$CODE/data"$'\n'* ]]
  done
}

@test "loads and exports values from code/.env" {
  printf "PUBMED_API_KEY='pm-SECRET-1'\nHF_HOME='/vol/hf home'\nDEEPRETRIEVAL_DATA_DIR='/vol/data'\n" >"$CODE/.env"
  for sh in $(shells); do
    run -0 --separate-stderr "$sh" "$SHOW"
    [[ $output == *"PUBMED=pm-SECRET-1"* ]]
    [[ $output == *"EXPORTED_PUBMED=pm-SECRET-1"* ]]
    [[ $output == *"HF_HOME=/vol/hf home"* ]]
    [[ $output == *"DATA=/vol/data"$'\n'* ]]
    # Diagnostics never contain secret values.
    [[ $stderr != *SECRET* ]]
  done
}

@test "variables already set in the environment win over code/.env" {
  printf "PUBMED_API_KEY='from-file'\nDEEPRETRIEVAL_DATA_DIR='/file/data'\nRAY_ENABLE_UV_RUN_RUNTIME_ENV='0'\n" >"$CODE/.env"
  for sh in $(shells); do
    PUBMED_API_KEY='from-env' DEEPRETRIEVAL_DATA_DIR='/env/data' RAY_ENABLE_UV_RUN_RUNTIME_ENV=1 \
      run -0 --separate-stderr "$sh" "$SHOW"
    [[ $output == *"PUBMED=from-env"* ]]
    [[ $output == *"DATA=/env/data"$'\n'* ]]
    [[ $output == *"RAY=1"$'\n'* ]]
  done
}

@test "CUDA_VISIBLE_DEVICES from the environment is left alone" {
  for sh in $(shells); do
    CUDA_VISIBLE_DEVICES=1 run -0 --separate-stderr "$sh" "$SHOW"
    [[ $output == *"CUDA=1"* ]]
    run -0 --separate-stderr "$sh" "$SHOW"
    [[ $output == *"CUDA=<unset>"* ]]
  done
}

@test "puts code/.venv/bin first on PATH when no venv is active" {
  mkdir -p "$CODE/.venv/bin"
  : >"$CODE/.venv/bin/python"
  chmod +x "$CODE/.venv/bin/python"
  for sh in $(shells); do
    run -0 --separate-stderr "$sh" "$SHOW"
    [[ $output == *"PATH=$CODE/.venv/bin:"* ]]
    [[ $output == *"VIRTUAL_ENV=$CODE/.venv"* ]]
    VIRTUAL_ENV=/other/venv run -0 --separate-stderr "$sh" "$SHOW"
    [[ $output != *"PATH=$CODE/.venv/bin:"* ]]
  done
}

@test "a .env line that is not KEY=VALUE fails with a message instead of running it" {
  printf "PUBMED_API_KEY='x'\nrm -rf /\n" >"$CODE/.env"
  for sh in $(shells); do
    run -1 --separate-stderr "$sh" "$SHOW"
    [[ $stderr == *".env"*"line 2"* ]]
  done
}

@test "comments, blank lines and export prefixes in .env are accepted" {
  printf "# comment\n\nexport HF_HOME='/vol/hf'\n" >"$CODE/.env"
  for sh in $(shells); do
    run -0 --separate-stderr "$sh" "$SHOW"
    [[ $output == *"HF_HOME=/vol/hf"* ]]
  done
}
