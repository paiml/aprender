#!/usr/bin/env bash
# Local gate (FLOW-009 LG-1..LG-4, LG-9): the x86-main sections of the full GitHub
# gate, run on this host with the pinned toolchain, a warm target dir, and the LG-3/LG-4
# env. The section list is read from ci.yml, so it cannot drift from CI.
# Usage: scripts/local_gate.sh [--self-test]
set -euo pipefail

# The `--sections '...'` argument of the x86-main driver call in ci.yml.
sections_of() {
    awk '/fat_driver.py run/ {f=1} f && /--sections/ {gsub(/.*--sections .?/, ""); gsub(/.?$/, ""); print; exit}' "$1"
}

# LG-3: one number for cargo jobs, rayon and nextest = floor(cores / concurrent gates).
threads_for() { echo $(($1 / $2)); }

if [ "${1:-}" = "--self-test" ]; then
    fail=0
    [ "$(threads_for 32 2)" = 16 ] || { echo "FAIL threads 32/2" >&2; fail=1; }
    [ "$(threads_for 48 2)" = 24 ] || { echo "FAIL threads 48/2" >&2; fail=1; }
    [ "$(threads_for 7 2)" = 3 ] || { echo "FAIL threads floor 7/2" >&2; fail=1; }
    got="$(sections_of "$(dirname "$0")/../.github/workflows/ci.yml")"
    case "$got" in sov.\*,guard-tree,*) ;; *) echo "FAIL sections_of: '$got'" >&2; fail=1 ;; esac
    [ "$fail" = 0 ] && echo "local_gate self-test: ok"
    exit "$fail"
fi

root="$(git rev-parse --show-toplevel)"
cd "$root"
repo=aprender
build="${GATE_BUILD_ROOT:-/mnt/nvme-raid0/build}/$repo"
mkdir -p "$build"
gates="${GATE_CONCURRENT:-2}"
n="$(threads_for "$(nproc)" "$gates")"
[ "$n" -ge 1 ] || n=1

# LG-2: warm dir, one gate at a time per dir.
exec 9>"$build/target.lock"
flock 9
export CARGO_TARGET_DIR="$build/target"
# LG-3, LG-4.
export CARGO_BUILD_JOBS="$n" RAYON_NUM_THREADS="$n" NEXTEST_TEST_THREADS="$n"
export CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0

sections="$(sections_of .github/workflows/ci.yml)"
[ -n "$sections" ] || { echo "local_gate: no --sections found in ci.yml" >&2; exit 2; }
results="$build/results.json"
rm -f "${results:?}"
rc=0
bash scripts/ci/install_pinned_toolchain.sh || rc=$?
if [ "$rc" = 0 ]; then
    python3 scripts/ci/fat_driver.py run --sections "$sections" --results "$results" || rc=$?
fi

# LG-9: step times to the ledger, kept even when a section is red.
if [ -f "$results" ]; then
    ledger="$build/gate-steps.jsonl"
    cargo run -q -p aprender-ci-tools -- gate-steps "$results" >>"$ledger" || echo "local_gate: step table not written" >&2
    echo "local_gate: step times appended to $ledger"
fi
exit "$rc"
