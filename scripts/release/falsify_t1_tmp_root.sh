#!/usr/bin/env bash
# falsify_t1_tmp_root.sh - check_release_models_t1.sh must build its fixtures under
# $RUNNER_TEMP (else $TMPDIR, else /tmp), never bare `mktemp -d` (R34-3: a shared-/tmp
# sweep deleted a fixture mid-run, merge_group shard red).
#   bash scripts/release/falsify_t1_tmp_root.sh --self-test | --mutants
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUBJECT="${T1_SUBJECT:-$REPO_ROOT/scripts/check_release_models_t1.sh}"
# line -> where the fixture root lands when RUNNER_TEMP=$1
root_for() {
    local line root
    line=$(awk '/^TMP=\$\(mktemp/ {print; exit}' "$1") || return 2
    [ -n "$line" ] || return 2
    root=$(RUNNER_TEMP="$2" TMPDIR="" bash -c "${line%% || *}; printf '%s' \"\$TMP\"; rmdir \"\$TMP\"" 2> /dev/null) || return 2
    printf '%s\n' "$root"
}
check() {
    local rt got
    rt=$(mktemp -d) || return 2
    got=$(root_for "$SUBJECT" "$rt") || { rmdir -- "$rt"; return 2; }
    rmdir -- "$rt" 2> /dev/null
    case "$got" in "$rt"/*) printf 'ok   fixtures under RUNNER_TEMP\n'; return 0 ;; *) printf 'FAIL fixtures at %s, not under RUNNER_TEMP\n' "$got"; return 1 ;; esac
}
case "${1:-}" in
    --self-test) check ;;
    --mutants)
        m=$(mktemp) || exit 2
        sed 's|mktemp -d -p "[^"]*"|mktemp -d|' "$SUBJECT" > "$m"
        if T1_SUBJECT="$m" bash "$0" --self-test > /dev/null; then rm -- "$m"; printf 'FAIL bare mktemp -d survived\n'; exit 1; fi
        rm -- "$m"; printf 'ok   bare mktemp -d is caught\n' ;;
    *) check ;;
esac
