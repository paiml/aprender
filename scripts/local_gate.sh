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
# Hermetic registry: a host ~/.cargo/registry may be a symlink into /dev/shm that
# dangles inside the section containers (guard-cargo m5). Keep the toolchain home
# where it is and point cargo's registry/cache at the build filesystem.
# cargo-free guards need user-scope tools (check-jsonschema); a non-login ssh PATH lacks them.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
export CARGO_HOME="$build/cargo-home"
mkdir -p "$CARGO_HOME"
# LG-3, LG-4.
export CARGO_BUILD_JOBS="$n" RAYON_NUM_THREADS="$n" NEXTEST_TEST_THREADS="$n"
export CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0

# The guards read GH_TOKEN / GITHUB_TOKEN (cargo-deny advisory clone, gh reads); CI sets them.
if [ -z "${GITHUB_TOKEN:-}" ]; then
    GITHUB_TOKEN="$(gh auth token 2>/dev/null || true)"
fi
[ -n "${GITHUB_TOKEN:-}" ] || echo "local_gate: no GITHUB_TOKEN (gh auth token empty); token-reading guards will fail" >&2
export GITHUB_TOKEN GH_TOKEN="${GH_TOKEN:-$GITHUB_TOKEN}"
# The base-differential guards need origin/main and its merge-base. A linked worktree's
# .git is a file the section containers cannot follow, so the gate judges the HEAD commit
# in a full private clone (hermetic: uncommitted edits are not part of the verdict).
git fetch -q origin main
head_sha="$(git rev-parse HEAD)"
clone="$build/src"
rm -rf "${clone:?}"
git clone -q --no-hardlinks "$root" "$clone"
git -C "$clone" fetch -q "$root" "refs/remotes/origin/main:refs/remotes/origin/main"
# What a pull_request job checks out: refs/pull/N/merge, i.e. HEAD merged onto origin/main
# (the base-differential guards name origin/main as the first parent of that merge).
git -C "$clone" checkout -q --detach refs/remotes/origin/main
git -C "$clone" -c user.name=gate -c user.email=gate@localhost merge -q --no-ff -m "gate: merge $head_sha" "$head_sha" ||
    { echo "local_gate: HEAD does not merge cleanly onto origin/main; update the branch" >&2; exit 2; }
cd "$clone"
# Serialise the shared-state installs the parallel sections would otherwise race on.
rustup toolchain install stable --profile minimal --target x86_64-pc-windows-gnu >/dev/null 2>&1 || true
cargo fetch --locked >/dev/null 2>&1 || true

sections="$(sections_of .github/workflows/ci.yml)"
[ -n "$sections" ] || { echo "local_gate: no --sections found in ci.yml" >&2; exit 2; }
# Same filesystem as the clone: the driver hard-links objects into each section tree.
export RUNNER_TEMP="$build/tmp"
mkdir -p "$RUNNER_TEMP"
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
