#!/usr/bin/env bash
# check_format_sovereignty.sh -- Poka-Yoke: prove the `apr-format` leaf crate is
# DEPENDENCY-SOVEREIGN (issue #2231 / contracts/apr-format-leaf-sovereignty-v1.yaml).
#
# The whole point of extracting the `.apr` container into `apr-format` is that a
# consumer can `cargo add apr-format` and read/write `.apr` files WITHOUT pulling
# `aprender-core` (~138 deps) and the trueno/wgpu/CUDA GPU stack. This guard
# parses the crate's resolved dependency closure (ALL features) via
# `cargo metadata` and fails if it intersects the forbidden ML/GPU/tokenizer set.
#
# It is a DISCRIMINATING guard (not a tautology): the same logic is run as a
# regression fixture against a known-sovereign crate (`aprender-quant`, must
# PASS) and a known-non-sovereign crate (`aprender-core`, must FAIL -- it really
# does pull trueno/wgpu). If the guard cannot tell those two apart it is broken.
#
# The dependency-closure computation uses `cargo tree` (normal+build edges) — the
# sovereign-ci image ships cargo but not python3, so no external interpreter.
#
# Usage: bash scripts/check_format_sovereignty.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

# Forbidden = any ML / GPU / tokenizer / framework crate. A match means the leaf
# is no longer sovereign and a consumer is forced to pull the framework again.
FORBIDDEN="trueno aprender-compute aprender-gpu aprender-core wgpu naga cudarc cust candle-core candle-nn tch torch-sys"

. "${SCRIPT_DIR}/cargo_classify.sh" || exit 1

# The classifier is a new surface in this script, so it is re-mutated HERE
# rather than inheriting check_facade_compat.sh's green -- extending a guard's
# scope requires re-mutating in the new scope. Text and committed fixtures
# only, so it runs without cargo and without the container.
if [ "${1:-}" = "--self-test" ]; then
    cargo_classify_selftest || { printf '\nSELF-TEST FAILED\n'; exit 1; }
    printf '\nSELF-TEST PASSED (classifier)\n'
    exit 0
fi

PUBLISH_LOG="$(mktemp)"
trap 'rm -f "${PUBLISH_LOG}"' EXIT

# Emit one crate name per line for the transitive consumer closure of crate $1.
# Uses `cargo tree` (normal + build edges = exactly what a downstream `cargo add`
# pulls; dev-deps excluded) rather than `cargo metadata | python3`, because the
# sovereign-ci image ships cargo but NOT python3 (a python3 dependency silently
# no-opped the detector and broke the negative-control fixture, #2236).
closure_of() {
  local crate="$1"
  cargo tree --package "${crate}" --all-features --edges normal,build --prefix none 2>/dev/null \
    | sed 's/[[:space:]].*//' \
    | grep -v '^$' \
    | sort -u
}

# Return the forbidden crates present in crate $1's closure (space-separated).
forbidden_in() {
  local crate="$1"
  local clo
  clo="$(closure_of "${crate}")"
  local hit=""
  local f
  for f in ${FORBIDDEN}; do
    if grep -qx "${f}" <<< "${clo}" ; then
      hit="${hit} ${f}"
    fi
  done
  printf '%s' "${hit# }"
}

echo "== check_format_sovereignty: apr-format leaf dependency sovereignty (#2231) =="
# Armed on the NORMAL path too: no workflow invokes this script's --self-test,
# and a case table nothing runs is the vacuous-scan class.
cargo_classify_selftest --quiet || exit 1

# ---- Primary assertion: apr-format must be sovereign --------------------------
APR_HIT="$(forbidden_in apr-format)"
if [ -n "${APR_HIT}" ]; then
  echo "FAIL: apr-format pulls forbidden ML/GPU/framework crate(s):${APR_HIT}" >&2
  echo "      The sovereign leaf must not depend on the framework it was carved out of." >&2
  exit 1
fi
echo "PASS: apr-format closure is sovereign (no trueno/wgpu/cuda*/candle/tch/aprender-core)"

# ---- Regression fixture (positive): aprender-quant must also PASS -------------
QUANT_HIT="$(forbidden_in aprender-quant)"
if [ -n "${QUANT_HIT}" ]; then
  echo "FAIL[fixture]: aprender-quant unexpectedly pulls:${QUANT_HIT}" >&2
  echo "      The positive fixture regressed; either the guard or aprender-quant changed." >&2
  exit 1
fi
echo "PASS[fixture+]: aprender-quant is sovereign (positive control)"

# ---- Regression fixture (negative): aprender-core MUST FAIL -------------------
# aprender-core genuinely pulls trueno/wgpu -- if the guard reports it as
# sovereign the detector is broken (a tautology that can never fail).
CORE_HIT="$(forbidden_in aprender-core)"
if [ -z "${CORE_HIT}" ]; then
  echo "FAIL[fixture]: guard reports aprender-core as sovereign -- detector is BROKEN." >&2
  echo "      aprender-core pulls trueno/wgpu; the guard must flag it." >&2
  exit 1
fi
echo "PASS[fixture-]: aprender-core correctly flagged as NON-sovereign (pulls:${CORE_HIT})"

# ---- Publish dry-run: catch dev-dep cycles that path-deps mask ---------------
# (See feedback_crates_io_devdep_publish_cycles: a full cascade fails on cycles
#  cargo builds locally but crates.io rejects. --no-verify keeps it fast.
#  --allow-dirty: this gate runs in dirty worktrees too -- it checks the manifest
#  graph, not working-tree cleanliness.)
echo "== cargo publish -p apr-format --dry-run --no-verify (dev-dep cycle check) =="
# ONE invocation, output captured, `rc` read DIRECTLY. This was
# `cargo publish ... | tee "${PUBLISH_LOG}" | tail -3`, so the `if` tested the
# PIPELINE's status -- `tail`'s, made survivable only by `set -o pipefail`. The
# repo has shipped that construct as a live defect twice (#2336, #2360).
set +e
cargo publish -p apr-format --dry-run --no-verify --allow-dirty > "${PUBLISH_LOG}" 2>&1
PUB_RC=$?
set -e
tail -3 "${PUBLISH_LOG}"
if [ "${PUB_RC}" -eq 0 ]; then
  echo "PASS: apr-format publish dry-run clean (no dependency cycle)"
elif [ "$( classify_cargo_failure "${PUBLISH_LOG}" )" = 'ENV' ]; then
  # A dry-run publish talks to the registry, so this block is the one in the
  # repo MOST likely to die of the environment -- and it named a dependency
  # cycle for any non-zero exit. ENV still exits 1.
  report_cargo_env_failure "${PUBLISH_LOG}" \
    'whether apr-format has a dev-dep publish cycle' >&2
  exit 1
else
  echo "FAIL: apr-format publish dry-run failed -- see ${PUBLISH_LOG}" >&2
  exit 1
fi

echo "== check_format_sovereignty: ALL CHECKS PASS =="
