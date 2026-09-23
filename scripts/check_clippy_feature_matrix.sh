#!/usr/bin/env bash
# check_clippy_feature_matrix.sh — strict clippy over apr-cli's FEATURE AXES, not only its default build (#3837).
#
# WHY. `ci / lint` runs clippy without `--features cuda`, and the cuda jobs run tests, never clippy, so no gate had
# ever linted code behind `--features cuda`: 62 findings accumulated invisibly in aprender-train and
# aprender-compute. That is #2370's mechanism on a different axis (there: toolchain version; here: feature flags).
#
# THE AXES. Each axis's flags are a bash ARRAY and are passed to cargo as separate words. #3837 recorded the trap:
# a feature string in an unquoted variable is ONE argument under zsh, and cargo exits 1 with
# "unexpected argument '--features cuda'" — which reads exactly like a lint failure. The self-test below proves the
# flags reach cargo, by planting a finding only the cuda axis can see.
#
#   default   (no flags)
#   cuda      --features cuda     no CUDA toolkit is needed: the driver is dlopen'd (libloading), nothing links CUDA
#
# EXCLUDED, by name and ticket, never by omission:
#   no-default   --no-default-features does not COMPILE (8 errors) — #4041 decides whether it is a supported
#                configuration; it joins this list when it is.
#
# --self-test plants `x as u32` on a u32 inside cuda-only code and requires: cuda axis RED, default axis GREEN.
# The restore is trapped BEFORE the plant and uses `git checkout --`, so an interrupted run leaves no planted file.
#
# Exit: 0 every axis clean · 1 an axis has findings · 2 could not check.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 2
cd "$ROOT" || exit 2
PROG=check_clippy_feature_matrix
AXES=(default cuda)
flags_for() { # prints nothing; fills the global array F for axis $1
  case "$1" in
    default) F=() ;;
    cuda)    F=(--features cuda) ;;
    *) echo "$PROG: unknown axis $1" >&2; return 2 ;;
  esac
}
EXCLUDED="no-default (#4041: does not compile)"

# An axis that names a feature apr-cli does not declare would fail for the wrong reason: refuse up front.
grep -qE '^cuda *=' crates/apr-cli/Cargo.toml || { echo "$PROG: apr-cli declares no 'cuda' feature — the axis table is stale" >&2; exit 2; }

run_axis() { # run_axis <axis> <log> -> clippy's rc
  local F=()
  flags_for "$1" || return 2
  cargo clippy -p apr-cli --lib "${F[@]}" -- -D warnings > "$2" 2>&1
}

check() { # -> 0 all clean, 1 findings; prints one line per axis
  local axis rc fails=0 log
  for axis in "${AXES[@]}"; do
    log="$TMP/$axis.log"
    run_axis "$axis" "$log"; rc=$?
    if grep -q "unexpected argument" "$log"; then
      echo "  BROKE $axis: cargo rejected the flags ($(grep -m1 'unexpected argument' "$log")) — not a lint result"; return 2
    fi
    if [ "$rc" -eq 0 ]; then
      echo "  ok    $axis: 0 findings"
    else
      echo "  FAIL  $axis: rc $rc, $(grep -c '^error' "$log") error line(s); first: $(grep -m1 '^error' "$log")"
      fails=1
    fi
  done
  return "$fails"
}

TMP=$(mktemp -d) || exit 2
PLANT_FILE=crates/aprender-train/src/autograd/cuda_forward/matmul_f16.rs
cleanup() { [ -n "${PLANTED:-}" ] && git checkout -- "$PLANT_FILE" 2>/dev/null; rm -rf "$TMP"; }
trap cleanup EXIT

if [ "${1:-}" = "--self-test" ]; then
  git diff --quiet -- "$PLANT_FILE" || { echo "$PROG: $PLANT_FILE has local edits; the self-test will not plant over them" >&2; exit 2; }
  grep -q '#!\[cfg(feature = "cuda")\]\|^#\[cfg(feature = "cuda")\]' "$PLANT_FILE" || { echo "$PROG: $PLANT_FILE is no longer cuda-gated; move the plant" >&2; exit 2; }
  PLANTED=1
  printf '\n#[cfg(feature = "cuda")]\n#[allow(dead_code)]\nfn planted_3837_self_test(x: u32) -> u32 {\n    x as u32\n}\n' >> "$PLANT_FILE"
  out=$(check); rc=$?
  printf '%s\n' "$out"
  git checkout -- "$PLANT_FILE"; PLANTED=
  if grep -q '^  FAIL  cuda' <<< "$out" && grep -q '^  ok    default' <<< "$out"; then
    echo "$PROG --self-test: PASS — the planted cuda-only finding turned ONLY the cuda axis red"; exit 0
  fi
  echo "$PROG --self-test: FAIL (rc $rc) — want cuda RED and default GREEN"; exit 1
fi

echo "$PROG: axes ${AXES[*]}; excluded: $EXCLUDED"
out=$(check); rc=$?
printf '%s\n' "$out"
case $rc in
  0) echo "$PROG: PASS" ;;
  1) echo "$PROG: FAIL" ;;
  *) echo "$PROG: could not check" ;;
esac
exit "$rc"
