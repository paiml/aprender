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
#   default     (no flags)
#   cuda        --features cuda     no CUDA toolkit is needed: the driver is dlopen'd (libloading), nothing links CUDA
#   full        --features full     pulls in cuda-batch, training-gpu, code, xet and trueno-explain
#   dev         --features dev
#   dhat-heap   --features dhat-heap
#
# EXCLUDED, by name and ticket, never by omission:
#   wgpu         --features wgpu does not COMPILE (5 errors) — #4056 (P1)
#   no-default   --no-default-features does not COMPILE (8 errors) — #4041 decides whether it is a supported
#                configuration; it joins this list when it is.
#
# THE UNIVERSE IS DERIVED (#3837's "enumerate the other feature axes"). apr-cli's feature table is read from
# `cargo metadata`, and every feature must be linted by some axis — directly, or because an axis's feature set
# (plus `default` unless the axis disables it) implies it — or be EXCLUDED with a ticket. A new feature that no axis
# reaches is a refusal naming it, before any clippy runs. --self-test plants one to prove it.
#
# --self-test plants `x as u32` on a u32 inside cuda-only code and requires: cuda axis RED, default axis GREEN.
# The restore is trapped BEFORE the plant and uses `git checkout --`, so an interrupted run leaves no planted file.
#
# Exit: 0 every axis clean · 1 an axis has findings · 2 could not check.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 2
cd "$ROOT" || exit 2
PROG=check_clippy_feature_matrix
AXES=(default cuda full dev dhat-heap)
flags_for() { # prints nothing; fills the global array F for axis $1
  case "$1" in
    default)   F=() ;;
    cuda)      F=(--features cuda) ;;
    full)      F=(--features full) ;;
    dev)       F=(--features dev) ;;
    dhat-heap) F=(--features dhat-heap) ;;
    *) echo "$PROG: unknown axis $1" >&2; return 2 ;;
  esac
}
# Features no axis may claim, each with its ticket (the no-default MODE is excluded above, not a feature).
EXCLUDED_FEATURES="wgpu=#4056"
EXCLUDED="wgpu (#4056: does not compile), no-default (#4041: does not compile)"

# Every apr-cli feature must be reached by an axis or excluded with a ticket: derived from cargo metadata.
coverage() { # -> 0 every feature accounted for; 1 names each unreached feature; 2 metadata unreadable
  local specs=() axis F
  for axis in "${AXES[@]}"; do flags_for "$axis" || return 2; specs+=("$axis=${F[*]:-}"); done
  cargo metadata --no-deps --format-version 1 2>/dev/null | python3 -c '
import json, sys
excluded = dict(x.split("=", 1) for x in sys.argv[1].split(",") if x)
axes = [a.split("=", 1) for a in sys.argv[2:]]
pkgs = [p for p in json.load(sys.stdin)["packages"] if p["name"] == "apr-cli"]
if not pkgs: print("  refused: apr-cli is not in cargo metadata"); sys.exit(2)
feats = pkgs[0]["features"]
def closure(start):
    seen, stack = set(), list(start)
    while stack:
        f = stack.pop()
        if f in seen or f not in feats: continue
        seen.add(f); stack += [x for x in feats[f] if "/" not in x and not x.startswith("dep:")]
    return seen
covered = {}
for name, flags in axes:
    words = flags.split()
    start = [] if "--no-default-features" in words else ["default"]
    if "--features" in words: start += words[words.index("--features") + 1].split(",")
    for f in closure(start): covered.setdefault(f, name)
missing = [f for f in sorted(feats) if f != "default" and f not in covered and f not in excluded]
for f in missing: print(f"  UNREACHED feature `{f}`: no axis lints it and it is not excluded with a ticket")
for f in sorted(excluded):
    if f not in feats: print(f"  STALE exclusion `{f}` ({excluded[f]}): apr-cli no longer declares it"); missing.append(f)
reached = sum(1 for f in feats if f in covered and f != "default")
print(f"  coverage: {len(feats) - 1} features; {reached} reached by an axis, {len(excluded)} excluded with a ticket")
sys.exit(1 if missing else 0)
' "$EXCLUDED_FEATURES" "${specs[@]}"
}

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
MANIFEST=crates/apr-cli/Cargo.toml
cleanup() {
  [ -n "${PLANTED:-}" ] && git checkout -- "$PLANT_FILE" 2>/dev/null
  [ -n "${PLANTED_FEATURE:-}" ] && git checkout -- "$MANIFEST" 2>/dev/null
  rm -rf "$TMP"
}
trap cleanup EXIT

if [ "${1:-}" = "--self-test" ]; then
  # Plant 1: a feature no axis reaches must be refused BY NAME, before any clippy runs.
  git diff --quiet -- "$MANIFEST" || { echo "$PROG: $MANIFEST has local edits; the self-test will not plant over them" >&2; exit 2; }
  PLANTED_FEATURE=1
  sed -i 's/^\[features\]$/[features]\nplanted-3837-unreached = []/' "$MANIFEST"
  grep -q '^planted-3837-unreached = \[\]' "$MANIFEST" || { echo "$PROG: could not plant the feature — the [features] anchor moved" >&2; exit 2; }
  cout=$(coverage); crc=$?
  git checkout -- "$MANIFEST"; PLANTED_FEATURE=
  printf '%s\n' "$cout"
  if [ "$crc" -ne 1 ] || ! grep -q 'UNREACHED feature `planted-3837-unreached`' <<< "$cout"; then
    echo "$PROG --self-test: FAIL (rc $crc) — an unreached feature was not refused by name"; exit 1
  fi
  echo "  ok    an unreached feature is refused by name, before any clippy"
  # Plant 2: a cuda-only finding reddens ONLY the cuda-reaching axes.
  git diff --quiet -- "$PLANT_FILE" || { echo "$PROG: $PLANT_FILE has local edits; the self-test will not plant over them" >&2; exit 2; }
  grep -q '#!\[cfg(feature = "cuda")\]\|^#\[cfg(feature = "cuda")\]' "$PLANT_FILE" || { echo "$PROG: $PLANT_FILE is no longer cuda-gated; move the plant" >&2; exit 2; }
  PLANTED=1
  printf '\n#[cfg(feature = "cuda")]\n#[allow(dead_code)]\nfn planted_3837_self_test(x: u32) -> u32 {\n    x as u32\n}\n' >> "$PLANT_FILE"
  out=$(check); rc=$?
  printf '%s\n' "$out"
  git checkout -- "$PLANT_FILE"; PLANTED=
  if grep -q '^  FAIL  cuda' <<< "$out" && grep -q '^  FAIL  full' <<< "$out" && grep -q '^  ok    default' <<< "$out" \
     && grep -q '^  ok    dev' <<< "$out" && grep -q '^  ok    dhat-heap' <<< "$out"; then
    echo "$PROG --self-test: PASS — the planted cuda-only finding turned exactly the cuda-reaching axes (cuda, full) red"; exit 0
  fi
  echo "$PROG --self-test: FAIL (rc $rc) — want cuda and full RED, default/dev/dhat-heap GREEN"; exit 1
fi

echo "$PROG: axes ${AXES[*]}; excluded: $EXCLUDED"
cout=$(coverage); crc=$?
printf '%s\n' "$cout"
[ "$crc" -eq 0 ] || { echo "$PROG: FAIL — the axis table does not reach every apr-cli feature"; exit "$crc"; }
out=$(check); rc=$?
printf '%s\n' "$out"
case $rc in
  0) echo "$PROG: PASS" ;;
  1) echo "$PROG: FAIL" ;;
  *) echo "$PROG: could not check" ;;
esac
exit "$rc"
