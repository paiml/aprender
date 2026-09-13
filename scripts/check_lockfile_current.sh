#!/usr/bin/env bash
# check_lockfile_current.sh — Cargo.lock must match the manifests.
#
# WHY THIS EXISTS
# ---------------
# Cargo.lock was last committed on 2026-08-01 (0.63.0). Manifests changed on
# 2026-08-10 and 2026-08-11 without it. For fifteen days, on a clean checkout of
# main, `cargo metadata` ALONE -- no build, no test -- rewrote the lockfile:
#
#     1 file changed, 87 insertions(+), 1436 deletions(-)
#
# Reproduced identically in three independent worktrees, so it was the tree and
# not one machine.
#
# Nothing caught it because every CI job runs cargo WITHOUT --locked: cargo
# silently updates the lock in place and carries on green. The jobs that do pass
# --locked are the ones nobody runs on a PR:
#
#     .github/workflows/binary-release.yml:116  cross build ... --locked
#     .github/workflows/binary-release.yml:118  cargo build ... --locked
#
# That is the RELEASE path, and on main it failed:
#
#     error: cannot update the lock file ... because --locked was passed
#
# So the guard that mattered only ran at the moment it was most expensive to
# fail, which is the same shape as every other defect in this class: the check
# did not scan the surface where the decision is made.
#
# WHY `cargo metadata` AND NOT A BUILD
# ------------------------------------
# Resolution is the whole question here; codegen is not. `cargo metadata
# --locked` resolves the full workspace graph and fails on exactly the condition
# we care about, in about a second, with no compilation and no network.
#
#   bash scripts/check_lockfile_current.sh              # check
#   bash scripts/check_lockfile_current.sh --self-test  # case table

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

. "${REPO_ROOT}/scripts/cargo_classify.sh" || exit 1

# ---------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
    fails=0
    TD="$(mktemp -d)" || exit 1
    case "$TD" in
        /tmp/*|/var/folders/*) : ;;
        *) printf 'FAIL: mktemp -d gave %s, refusing to use it\n' "${TD:-<empty>}"; exit 1 ;;
    esac
    trap 'rm -rf "${TD:?}"' EXIT

    mkdir -p "$TD/src"
    {
        printf '[package]\n'
        printf 'name = "lockfile-probe"\n'
        printf 'version = "0.0.0"\n'
        printf 'edition = "2021"\n'
    } > "$TD/Cargo.toml"
    printf 'pub fn probe() -> u8 { 1 }\n' > "$TD/src/lib.rs"

    # Row 1: a lock generated from THESE manifests must satisfy --locked.
    ( cd "$TD" && cargo metadata --format-version 1 >/dev/null 2>&1 )
    if ( cd "$TD" && cargo metadata --format-version 1 --locked >/dev/null 2>&1 ); then
        printf 'ok    row 1 a current lockfile passes --locked\n'
    else
        printf 'FAIL  row 1 a freshly generated lockfile did NOT pass --locked\n'; fails=1
    fi

    # Row 2 is the control, and it is the whole point: make the manifest and the
    # lock disagree, and --locked MUST refuse. Without this, row 1 passes even if
    # --locked never rejected anything.
    printf 'anyhow = "1"\n' >> "$TD/Cargo.toml"
    printf '[dependencies]\nanyhow = "1"\n' >> "$TD/Cargo.toml"
    if ( cd "$TD" && cargo metadata --format-version 1 --locked >/dev/null 2>&1 ); then
        printf 'FAIL  row 2 --locked ACCEPTED a lockfile that predates a new dependency\n'; fails=1
    else
        printf 'ok    row 2 --locked refuses a stale lockfile\n'
    fi

    # Rows 1-2 probe `--locked` itself. The ENV/CODE classifier is a DIFFERENT
    # surface -- it decides whether a non-zero `cargo metadata` gets to be named
    # "the lockfile is stale" -- so it is re-mutated here rather than inheriting
    # rows 1-2's green.
    cargo_classify_selftest || fails=1

    [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED\n'; exit 1; }
    printf '\nSELF-TEST PASSED (2/2 locked + classifier table above)\n'
    exit 0
fi

printf '=== Cargo.lock must match the manifests (check_lockfile_current.sh) ===\n'

cd "$REPO_ROOT" || exit 1

if [ ! -f Cargo.lock ]; then
    printf 'FAIL: Cargo.lock does not exist. This guard is checking nothing.\n'
    exit 1
fi

ERR="$(mktemp)" || exit 1
trap 'rm -f "${ERR:?}"' EXIT

cargo metadata --format-version 1 --locked > /dev/null 2> "$ERR"
rc=$?

# `rc` was read correctly here from the start; the residual is the CLAIM. Any
# non-zero exit was named "the committed Cargo.lock does not match the
# manifests", and `cargo metadata` also exits non-zero when it cannot reach the
# registry, cannot take the package cache lock, or cannot write. On 2026-08-27
# that reading -- from a sibling guard -- blocked every open PR in this repo.
if [ "$rc" -ne 0 ] && [ "$( classify_cargo_failure "$ERR" )" = 'ENV' ]; then
    report_cargo_env_failure "$ERR" 'whether Cargo.lock matches the manifests'
    exit 1
fi

if [ "$rc" -ne 0 ]; then
    printf '\nFAIL: the committed Cargo.lock does not match the manifests.\n\n'
    sed 's|^|  |' < "$ERR" | head -6
    printf '\nFix: run `cargo metadata --format-version 1 >/dev/null` and COMMIT the\n'
    printf 'resulting Cargo.lock. Every PR that changes a Cargo.toml must carry the\n'
    printf 'lockfile update with it -- ordinary cargo commands rewrite it silently and\n'
    printf 'stay green, so only --locked (i.e. the release path) ever notices.\n'
    exit 1
fi

printf 'PASS: `cargo metadata --locked` resolves the workspace.\n'
exit 0
