#!/usr/bin/env bash
# check_nextest_config_keys.sh — nextest must not IGNORE any key in
# .config/nextest.toml.
#
# WHY THIS EXISTS
# ---------------
# `.config/nextest.toml` carried
#
#     [profile.ci]
#     slow-warning = "60s"
#
# There is no `slow-warning` key in nextest. The real one is `slow-timeout`.
# nextest does not fail on an unknown key — it prints a warning and carries on:
#
#     warning: in config file .config/nextest.toml, ignoring unknown
#              configuration key: profile.ci.slow-warning
#
# So `profile.ci` had NO slow-timeout at all. Consequence, measured on the
# merge_group run for #2502 (job 95162862834):
#
#     12:50:58  Starting 80806 tests across 69 binaries
#     13:36:07  ##[error]The operation was canceled.
#
# 45 minutes inside nextest, killed by the JOB's `timeout-minutes: 85`, naming
# no test. The queue entry was evicted 31 seconds later. `main` did not move.
# Without a per-test timeout there is nothing to name the culprit, and the whole
# job dies anonymously instead.
#
# The warning was on line 353 of that job's log, and of every workspace-test log
# before it. A diagnostic nobody reads is not a diagnostic — so this makes it a
# hard failure.
#
# HOW
# ---
# nextest only validates config when it runs, so this runs it: a throwaway
# three-line crate, with the REPO's config passed via `--config-file`. Costs one
# trivial rustc invocation, not a workspace build, and exercises the real parser
# rather than a regex of our own guessing at nextest's schema.
#
#   bash scripts/check_nextest_config_keys.sh              # check
#   bash scripts/check_nextest_config_keys.sh --self-test  # case table

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${REPO_ROOT}/.config/nextest.toml"

if ! command -v cargo-nextest >/dev/null 2>&1 && ! cargo nextest --version >/dev/null 2>&1; then
    printf 'SKIP: cargo-nextest not installed; install with `cargo install cargo-nextest --locked`.\n' >&2
    [ "${CI:-}" = "true" ] && exit 1
    exit 0
fi

# Run nextest on a throwaway crate with the given config; echo any ignored-key
# warnings. Nothing else about the run matters.
ignored_keys_for() {
    local cfg="$1" dir
    dir="$(mktemp -d)" || return 1
    case "$dir" in
        /tmp/*|/var/folders/*) : ;;
        *) printf 'BADTMP %s\n' "${dir:-<empty>}"; return 1 ;;
    esac
    mkdir -p "$dir/src"
    # Line by line rather than a heredoc: bashrs parses an embedded heredoc as
    # shell, so TOML `name = "x"` gets reported as SC1007.
    {
        printf '[package]\n'
        printf 'name = "nextest-config-probe"\n'
        printf 'version = "0.0.0"\n'
        printf 'edition = "2021"\n'
    } > "$dir/Cargo.toml"
    printf 'pub fn probe() -> u8 { 1 }\n' > "$dir/src/lib.rs"

    ( cd "$dir" && cargo nextest run --config-file "$cfg" --profile ci 2>&1 ) \
        | grep -F 'ignoring unknown configuration key' \
        | sed -e 's/.*ignoring unknown configuration key: //' -e 's/[[:space:]]*$//'

    rm -rf "${dir:?refusing to rm an empty path}"
}

# ---------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
    fails=0
    SD="$(mktemp -d)" || exit 1
    trap 'rm -rf "${SD:?}"' EXIT

    # Row 1: the exact historical defect must be REPORTED.
    printf '[profile.ci]\nslow-warning = "60s"\n' > "$SD/bad.toml"
    got="$(ignored_keys_for "$SD/bad.toml")"
    if [ "$got" = "profile.ci.slow-warning" ]; then
        printf 'ok    row 1 the dead key that cost #2502 is reported\n'
    else
        printf 'FAIL  row 1 got [%s], expected profile.ci.slow-warning\n' "$got"; fails=1
    fi

    # Row 2 is the control. Without it row 1 passes even if this reported every
    # key it saw -- and then the real config could never go green.
    printf '[profile.ci]\nslow-timeout = { period = "60s", terminate-after = 10 }\n' > "$SD/good.toml"
    got="$(ignored_keys_for "$SD/good.toml")"
    if [ -z "$got" ]; then
        printf 'ok    row 2 the CORRECT key is not reported\n'
    else
        printf 'FAIL  row 2 reported [%s] for a valid config\n' "$got"; fails=1
    fi

    [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED\n'; exit 1; }
    printf '\nSELF-TEST PASSED (2/2)\n'
    exit 0
fi

printf '=== nextest must ignore no key in .config/nextest.toml (check_nextest_config_keys.sh) ===\n'

if [ ! -f "$CONFIG" ]; then
    printf 'FAIL: %s does not exist. This guard is scanning nothing.\n' "$CONFIG"
    exit 1
fi

# Vacuity: a config nextest cannot parse at all, or an empty one, would report
# zero ignored keys and look like a pass.
if ! grep -q '^\[profile\.ci\]' "$CONFIG"; then
    printf 'FAIL (vacuity): %s has no [profile.ci] section.\n' "$CONFIG"
    printf 'CI runs `--profile ci`; if that section is gone the scan proves nothing.\n'
    exit 1
fi

FOUND="$(ignored_keys_for "$CONFIG")"

if [ -n "$FOUND" ]; then
    printf '\nFAIL: nextest is IGNORING these keys:\n\n'
    printf '%s\n' "$FOUND" | sed 's|^|  |'
    printf '\nA key nextest ignores is a setting that does not exist. Check the\n'
    printf 'spelling against `cargo nextest show-config` / the nextest docs --\n'
    printf '`slow-warning` was one of these, and it cost a merge-queue eviction.\n'
    exit 1
fi

printf 'PASS: every key is understood.\n'
exit 0
