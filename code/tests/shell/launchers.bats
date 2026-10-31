#!/usr/bin/env bats
# Checks every training/eval launcher under code/scripts/{train,eval}:
#   static:     sources common.sh, no relative data/ paths, no CUDA_VISIBLE_DEVICES;
#   behaviour:  run under dash with python stubbed out, every data path argument is
#               under $DEEPRETRIEVAL_DATA_DIR and CUDA_VISIBLE_DEVICES is passed through.

bats_require_minimum_version 1.5.0

load helpers

launchers() {
  find "$SCRIPTS_DIR/train" "$SCRIPTS_DIR/eval" -type f -name '*.sh' | LC_ALL=C sort
}

# Fills the array `files` with the launcher paths (safe for paths with spaces).
load_launchers() {
  files=()
  local f
  while IFS= read -r f; do files+=("$f"); done < <(launchers)
}

@test "there are launchers to check" {
  n=$(launchers | wc -l)
  ((n >= 50))
}

@test "no launcher hardcodes CUDA_VISIBLE_DEVICES" {
  load_launchers
  run grep -nE 'CUDA_VISIBLE_DEVICES[[:space:]]*=' "${files[@]}"
  [[ $status -eq 1 ]] || {
    printf 'still set in:\n%s\n' "$output"
    false
  }
}

@test "no launcher uses a relative data/ path" {
  load_launchers
  run grep -nE '(^|[=[:space:]"])data/' "${files[@]}"
  [[ $status -eq 1 ]] || {
    printf 'relative data/ in:\n%s\n' "$output"
    false
  }
}

@test "every launcher sources common.sh first, via a path that resolves" {
  bad=''
  while IFS= read -r f; do
    first=$(grep -vE '^[[:space:]]*(#|$)' "$f" | sed -n 1p)
    rel=$(sed -n 's#^\. "\$(dirname "\$0")/\(.*\)/common\.sh" || exit 1$#\1#p' <<<"$first")
    if [[ -z $rel || ! -f "$(dirname "$f")/$rel/common.sh" ]]; then
      bad+="$f: $first"$'\n'
    fi
  done < <(launchers)
  [[ -z $bad ]] || {
    printf '%s' "$bad"
    false
  }
}

@test "launchers run under dash with data paths under DEEPRETRIEVAL_DATA_DIR" {
  clear_dr_env
  code="${BATS_TEST_TMPDIR}/code copy"
  mkdir -p "$code"
  cp -R "$SCRIPTS_DIR" "$code/scripts"
  bin="${BATS_TEST_TMPDIR}/bin"
  mkdir -p "$bin"
  argv_log="${BATS_TEST_TMPDIR}/argv.log"
  cat >"$bin/python" <<'EOF'
#!/bin/sh
printf 'CUDA=%s\n' "${CUDA_VISIBLE_DEVICES-<unset>}" >>"$ARGV_LOG"
for a in "$@"; do printf 'ARG %s\n' "$a" >>"$ARGV_LOG"; done
EOF
  chmod +x "$bin/python"
  cp "$bin/python" "$bin/python3"
  dd="${BATS_TEST_TMPDIR}/the data"
  count=0
  while IFS= read -r f; do
    rel=${f#"$SCRIPTS_DIR"/}
    : >"$argv_log"
    ARGV_LOG="$argv_log" DEEPRETRIEVAL_DATA_DIR="$dd" CUDA_VISIBLE_DEVICES=0,1 PATH="$bin:$PATH" \
      run --separate-stderr /bin/dash "$code/scripts/$rel" scifact gpt-4o
    [[ $status -eq 0 ]] || {
      printf '%s exited %s: %s\n' "$rel" "$status" "$stderr"
      false
    }
    grep -qx 'CUDA=0,1' "$argv_log" || {
      printf '%s did not run python with CUDA_VISIBLE_DEVICES=0,1\n' "$rel"
      false
    }
    # Every argument naming a parquet file or index under data must be absolute under $dd.
    while IFS= read -r line; do
      value=${line#ARG }
      value=${value#*=}
      if [[ $value == *.parquet && $value != "$dd"/* ]]; then
        printf '%s passes a data path outside DEEPRETRIEVAL_DATA_DIR: %s\n' "$rel" "$line"
        false
      fi
    done < <(grep '^ARG ' "$argv_log")
    count=$((count + 1))
  done < <(launchers)
  ((count >= 50))
  [[ -d "$code/exp_log" ]]
}
