#!/usr/bin/env bash
# isolate_home.sh - nextest setup script: every test runs with HOME in a throwaway dir (EXT-001 EXT-02).
#
# WHY: the writers resolve their stores through $HOME, not a variable of their own. entrenar's global
# store is dirs::home_dir()/.entrenar/experiments.db (aprender-train config/train/loader), pacha's
# registry is ~/.pacha (aprender-registry, aprender-orchestrate crypto.rs reads `HOME` itself), and
# apr-cli's runs/monitor/experiment verbs default to ~/.entrenar. A test that reaches a default
# constructor therefore writes into the developer's real registry. The v0 sample of that registry was
# 3,940 experiments, 3,607 of them named `.tmp%` (EXT-001 section 1, T2).
#
# nextest runs this once per run and exports every KEY=VALUE line written to $NEXTEST_ENV into each test.
# ENTRENAR_HOME and APR_HOME have no reader yet. They are set so that a reader added later is born
# isolated. XDG_* move with HOME because the dirs crate prefers them over $HOME when they are set.
# CARGO_HOME and RUSTUP_HOME are pinned to the REAL values first: tests that shell out to `cargo`
# must still find the toolchain, and neither directory is a store this spec guards.
#
# FALSIFY-EXT-001: crates/aprender-train/src/ext_isolation_tests.rs fails under nextest when HOME is
# not the dir this script marked.
set -euo pipefail

if [ -z "${NEXTEST_ENV:-}" ]; then
    echo "isolate_home.sh: NEXTEST_ENV is unset; this runs only as a nextest setup script" >&2
    exit 2
fi
real_home="${HOME:?isolate_home.sh: HOME is unset}"
base="${TMPDIR:-/tmp}"

# nextest has no teardown hook, so each run's dir outlives it. Prune the dirs of runs that have ended:
# .ext-001-owner holds the nextest pid (this script's parent). A dir is removed only when that file is a
# number and `ps` finds no such process. A missing or unreadable owner, or a live one (a concurrent run),
# keeps the dir, so a mistake here leaks a dir and never deletes one in use. `ps -p` rather than /proc:
# it also works on the macOS runners and for another user's live process.
for d in "${base%/}"/ext001-home.*; do
    [ -d "$d" ] || continue
    IFS= read -r owner 2> /dev/null < "$d/.ext-001-owner" || continue
    case "$owner" in "" | *[!0-9]*) continue ;; esac
    ps -p "$owner" > /dev/null 2>&1 && continue
    rm -rf -- "${d:?}"
done
iso="$(mktemp -d "${base%/}/ext001-home.XXXXXXXX")"
# The dir must outlive this script (the tests use it), so it is removed only if the script fails first.
trap 'rm -rf -- "${iso:?}"' EXIT
case "$iso" in
    "${real_home%/}" | "${real_home%/}"/*)
        echo "isolate_home.sh: temp dir $iso is inside the real HOME $real_home" >&2
        exit 2
        ;;
esac
mkdir -p "$iso/.config" "$iso/.cache" "$iso/.local/share" "$iso/.local/state" "$iso/.entrenar" "$iso/.apr"
printf '%s\n' "$real_home" > "$iso/.ext-001-isolated"
printf '%s\n' "$PPID" > "$iso/.ext-001-owner"

{
    printf 'CARGO_HOME=%s\n' "${CARGO_HOME:-"$real_home/.cargo"}"
    printf 'RUSTUP_HOME=%s\n' "${RUSTUP_HOME:-"$real_home/.rustup"}"
    printf 'EXT001_REAL_HOME=%s\n' "$real_home"
    printf 'HOME=%s\n' "$iso"
    printf 'ENTRENAR_HOME=%s\n' "$iso/.entrenar"
    printf 'APR_HOME=%s\n' "$iso/.apr"
    printf 'XDG_CONFIG_HOME=%s\n' "$iso/.config"
    printf 'XDG_CACHE_HOME=%s\n' "$iso/.cache"
    printf 'XDG_DATA_HOME=%s\n' "$iso/.local/share"
    printf 'XDG_STATE_HOME=%s\n' "$iso/.local/state"
} >> "${NEXTEST_ENV:?}"
trap - EXIT
