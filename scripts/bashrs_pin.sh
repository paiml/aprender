#!/usr/bin/env bash
# bashrs_pin.sh -- the bashrs that CI pins (tools.toml), never the one on PATH.
#
#   . scripts/bashrs_pin.sh || exit 1
#   bashrs_pin_resolve || exit 1      # exports BASHRS=<abs path> BASHRS_PIN=<version>
#   "$BASHRS" lint foo.sh
#
# A bashrs count is only comparable to a baseline measured by the SAME bashrs.
# tools.toml pins CI at one version, and the fleet correctly runs whatever the
# nightly installs (lambda went 7.4.1 -> 7.4.2 on 2026-09-25 while CI stayed on
# 7.4.1). A guard that reads `command -v bashrs` therefore measures a different
# instrument on each box and blames the tree for it.
#
# So the binary is resolved from the pin, not from PATH:
#
#   root = $BASHRS_PIN_ROOT if set, else the first writable of
#          ${TOOL_PIN_BASE:-/mnt/nvme-raid0/tmp/tool-pins}/bashrs-<pin> and
#          $RUNNER_TEMP/tool-pins/bashrs-<pin>
#   bin  = $root/bin/bashrs, installed on first use with
#          cargo install bashrs --version =<pin> --locked --root "$root"
#          (under flock, so concurrent jobs on one host build it once)
#
# and every failure is LOUD, one line on stderr, return 4:
#   * tools.toml names no [bashrs] version
#   * the binary is absent and BASHRS_PIN_NO_INSTALL=1, or the install failed
#   * `$bin --version` is not exactly "bashrs <pin>"   <- the mismatch case
# There is NO fallback to PATH in any branch. A pin that falls back to the
# ambient tool is the defect this file exists to remove.
#
# Sourced library: option-neutral (no `set`), fails by return status
# (CLAUDE.md, check_sourced_libs_option_neutral.sh).
#
# Env for tests: TOOL_PIN_TOML (default <repo>/tools.toml), BASHRS_PIN_ROOT,
# TOOL_PIN_BASE, RUNNER_TEMP, BASHRS_PIN_NO_INSTALL=1.
#
# Refs: tools.toml, scripts/check_tool_versions.sh, PMAT-1066.

_BASHRS_PIN_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || return 1

# tool_pin_version <tool> <tools.toml> -> the `version = "..."` in [<tool>]
tool_pin_version() {
    awk -v want="[$1]" '
        /^[[:space:]]*\[/ { sec = $0; gsub(/[[:space:]]/, "", sec); next }
        sec == want && /^[[:space:]]*version[[:space:]]*=/ {
            v = $0; sub(/^[^=]*=[[:space:]]*"/, "", v); sub(/".*$/, "", v); print v; exit
        }' "$2" 2>/dev/null
}

# _bashrs_pin_fail <fmt> <args...> -- one line on stderr, kept in BASHRS_PIN_ERR too
_bashrs_pin_fail() {
    local fmt=$1
    shift
    # shellcheck disable=SC2059
    BASHRS_PIN_ERR=$(printf "bashrs_pin: $fmt (no PATH fallback)" "$@")
    printf '%s\n' "$BASHRS_PIN_ERR" >&2
    return 4
}

# _bashrs_pin_root <pin> -> the first base whose bashrs-<pin> already holds the
# binary or can be created. The default base is lambda's layout; a clean-room
# runner without it (aprender#4429: both ratchets red with "cannot create")
# falls back to the job's RUNNER_TEMP. That costs one install per job there,
# and it is still the PINNED binary, version-checked by the caller: a fallback
# of WHERE, never of WHICH instrument, and never a skip.
_bashrs_pin_root() {
    local base bases
    bases=("${TOOL_PIN_BASE:-/mnt/nvme-raid0/tmp/tool-pins}")
    # --- TOOLPIN-FALLBACK-BEGIN ---
    [ -n "${RUNNER_TEMP:-}" ] && bases+=("$RUNNER_TEMP/tool-pins")
    # --- TOOLPIN-FALLBACK-END ---
    for base in "${bases[@]}"; do
        if [ -x "$base/bashrs-$1/bin/bashrs" ] || mkdir -p -- "$base/bashrs-$1" 2>/dev/null; then
            printf '%s\n' "$base/bashrs-$1"
            return 0
        fi
    done
    return 1
}

bashrs_pin_resolve() {
    local toml pin root bin live lock
    BASHRS_PIN_ERR=""
    toml=${TOOL_PIN_TOML:-$_BASHRS_PIN_DIR/../tools.toml}
    pin=$(tool_pin_version bashrs "$toml")
    if [ -z "$pin" ]; then
        _bashrs_pin_fail '%s names no [bashrs] version; cannot resolve the CI pin' "$toml"
        return 4
    fi
    if [ -n "${BASHRS_PIN_ROOT:-}" ]; then
        root=$BASHRS_PIN_ROOT
    elif ! root=$(_bashrs_pin_root "$pin"); then
        _bashrs_pin_fail 'no writable root for pinned bashrs %s under %s or RUNNER_TEMP=%s; set BASHRS_PIN_ROOT' \
            "$pin" "${TOOL_PIN_BASE:-/mnt/nvme-raid0/tmp/tool-pins}" "${RUNNER_TEMP:-unset}"
        return 4
    fi
    bin=$root/bin/bashrs
    if [ ! -x "$bin" ]; then
        if [ "${BASHRS_PIN_NO_INSTALL:-0}" = 1 ]; then
            _bashrs_pin_fail 'pinned bashrs %s is not installed at %s and BASHRS_PIN_NO_INSTALL=1' "$pin" "$bin"
            return 4
        fi
        lock="$root.lock"
        if ! mkdir -p -- "$root" 2>/dev/null; then
            _bashrs_pin_fail 'cannot create %s to install pinned bashrs %s; set BASHRS_PIN_ROOT' "$root" "$pin"
            return 4
        fi
        printf 'bashrs_pin: installing pinned bashrs %s into %s\n' "$pin" "$root" >&2
        # flock: two jobs on one host must not race one --root. The test for -x
        # is repeated under the lock, so the loser finds the winner's binary.
        if ! flock "$lock" bash -c '
                [ -x "$1/bin/bashrs" ] && exit 0
                cargo install bashrs --version "=$2" --locked --root "$1" --target-dir "$1/.build" --quiet
                rc=$?; rm -rf -- "${1:?}/.build"; exit "$rc"' _ "$root" "$pin" >&2; then
            _bashrs_pin_fail 'cargo install bashrs --version =%s --root %s FAILED' "$pin" "$root"
            return 4
        fi
    fi
    live=$("$bin" --version 2>/dev/null)
    live=${live%%$'\n'*}
    # --- BASHRS-PIN-CMP-BEGIN ---
    if [ "$live" != "bashrs $pin" ]; then
        _bashrs_pin_fail '%s reports "%s", tools.toml pins bashrs %s -- refusing to measure with a mismatched instrument' \
            "$bin" "$live" "$pin"
        return 4
    fi
    # --- BASHRS-PIN-CMP-END ---
    BASHRS=$bin
    BASHRS_PIN=$pin
    export BASHRS BASHRS_PIN
    return 0
}
