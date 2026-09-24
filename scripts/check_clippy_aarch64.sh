#!/usr/bin/env bash
# check_clippy_aarch64.sh — strict clippy ON AN aarch64 HOST (#4134). Axis: default; cuda excluded by ticket, below.
#
# WHY. Every clippy gate we own ran on x86_64 (make tier1/2/3, sovereign-ci lint, toolchain-ceiling), so code
# that is compiled only on arm, or bindings used only on the x86 side of a cfg, was never linted where it builds.
# #4134 measured the result on gx10: `cargo clippy -p apr-cli --lib -- -D warnings` failed in aprender-compute on
# four bindings read only inside `#[cfg(target_arch = "x86_64")]` branches, while x86 clippy was clean. gx10 and
# yoga build the release's aarch64 binaries. This is #2370's shape (the toolchain axis) on the target axis.
#
# WHERE IT RUNS. .github/workflows/silicon-nightly.yml, job aarch64-cuda-sm121, on gx10 (self-test first). A run
# on any other architecture is REFUSED (exit 2), never reported green: an x86 pass proves nothing about arm.
#
# THE AXES, each passed to cargo as separate words (the #3837 trap: one string would be one argument):
#   default   cargo clippy -p apr-cli --lib -- -D warnings
#
# EXCLUDED, by name and ticket, never by omission:
#   cuda      --features cuda is RED on main on EVERY architecture (x86 included): `unused import GemmOp` in
#             aprender-compute and ~61 findings in aprender-train, all fixed by #3837 (PR #4091), not yet merged.
#             Measured on gx10 at a1d42003b: the default axis is 0 findings, the cuda axis stops at GemmOp. Add
#             `cuda` to AXES when #4091 lands — gx10's release binary is a cuda build, so that axis matters here.
#
# --self-test plants an aarch64-ONLY unused binding in aprender-compute and requires every axis RED; the restore is
# trapped BEFORE the plant and uses `git checkout --`, and the plant file must have no local edits. Being
# `#[cfg(target_arch = "aarch64")]`, the plant cannot turn an x86 run red, which is the property being guarded.
#
# Exit: 0 every axis clean · 1 an axis has findings · 2 could not check (not aarch64, cargo rejected the flags).
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 2
cd "$ROOT" || exit 2
PROG=check_clippy_aarch64
AXES=(default)
EXCLUDED="cuda (#3837/PR #4091: RED on every arch on main)"
flags_for() { # fills the global array F for axis $1
  case "$1" in
    default) F=() ;;
    cuda)    F=(--features cuda) ;;
    *) echo "$PROG: unknown axis $1" >&2; return 2 ;;
  esac
}

arch=$(uname -m)
if [ "$arch" != "aarch64" ] && [ "$arch" != "arm64" ]; then
  echo "$PROG: could not check — this host is $arch; the gate measures aarch64 and runs on gx10 (silicon-nightly)"
  exit 2
fi

TMP=$(mktemp -d) || exit 2
PLANT_FILE=crates/aprender-compute/src/lib.rs
cleanup() {
  [ -n "${PLANTED:-}" ] && git checkout -- "$PLANT_FILE" 2>/dev/null
  rm -rf "$TMP"
}
trap cleanup EXIT

check() { # -> 0 all clean, 1 findings, 2 could not check; one line per axis
  local axis rc fails=0 log F
  for axis in "${AXES[@]}"; do
    log="$TMP/$axis.log"
    flags_for "$axis" || return 2
    cargo clippy -p apr-cli --lib "${F[@]}" -- -D warnings > "$log" 2>&1; rc=$?
    if grep -q "unexpected argument" "$log"; then
      echo "  BROKE $axis: cargo rejected the flags ($(grep -m1 'unexpected argument' "$log")) — not a lint result"; return 2
    fi
    if [ "$rc" -eq 0 ]; then
      echo "  ok    $axis: 0 findings"
    else
      echo "  FAIL  $axis: rc $rc, $(grep -c '^error' "$log") error line(s); first: $(grep -m1 -A3 '^error' "$log" | tr '\n' ' ' | cut -c1-240)"
      fails=1
    fi
  done
  return "$fails"
}

if [ "${1:-}" = "--self-test" ]; then
  git diff --quiet -- "$PLANT_FILE" || { echo "$PROG: $PLANT_FILE has local edits; the self-test will not plant over them" >&2; exit 2; }
  PLANTED=1
  printf '\n#[cfg(target_arch = "aarch64")]\n#[allow(dead_code)]\nfn planted_4134_self_test(x: u32) -> u32 {\n    let unused_on_arm = x;\n    0\n}\n' >> "$PLANT_FILE"
  out=$(check); rc=$?
  printf '%s\n' "$out"
  git checkout -- "$PLANT_FILE"; PLANTED=
  if grep -q '^  FAIL  default' <<< "$out" && grep -q 'unused_on_arm' "$TMP/default.log"; then
    echo "$PROG --self-test: PASS — the planted aarch64-only unused binding turned the default axis RED"; exit 0
  fi
  echo "$PROG --self-test: FAIL (rc $rc) — want the default axis RED on the planted unused_on_arm"; exit 1
fi

echo "$PROG: $arch, axes ${AXES[*]}; excluded: $EXCLUDED"
out=$(check); rc=$?
printf '%s\n' "$out"
case $rc in
  0) echo "$PROG: PASS" ;;
  1) echo "$PROG: FAIL" ;;
  *) echo "$PROG: could not check" ;;
esac
exit "$rc"
