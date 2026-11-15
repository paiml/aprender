#!/usr/bin/env bash
# check_cleanroom_presha.sh -- the release train runs clean-room on the bump commit's sha BEFORE the
# tag (#6500). Extracts cleanroom_on_sha() from scripts/release/autopilot.sh, runs it against stubs,
# and asserts: green + tested-sha == sha passes; a red job, a different tested-sha, a failed dispatch
# and a missing run each die; and the tag step calls it ahead of cut_tag.
#   check_cleanroom_presha.sh [<autopilot.sh>]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
AP_FILE="${1:-$ROOT/scripts/release/autopilot.sh}"
SHA=0123456789abcdef0123456789abcdef01234567
OTHER=fedcba9876543210fedcba9876543210fedcba98
bad=0
row() { if [ "$1" -eq 0 ]; then printf 'ok    %s\n' "$2"; else printf 'FAIL  %s\n' "$2" >&2; bad=1; fi; }

fn=$(awk '/^cleanroom_on_sha\(\) \{/,/^\}/' "$AP_FILE")
if [ -z "$fn" ]; then
    printf 'FAIL  %s has no cleanroom_on_sha(): no clean-room runs on the sha before the tag\n' "$AP_FILE" >&2
    exit 1
fi

# run_case <job-conclusion|none> <tested-sha> <dispatch-rc> -- prints DIE/SAY lines, rc of the function
run_case() {
    local concl=$1 tested=$2 drc=$3 d
    d=$(mktemp -d) || exit 2
    (
        AP="$d"; LOG="$d/log"; INFRA=paiml/infra; AP_SETTLE=0; AP_POLL=0
        die() { printf 'DIE %s\n' "$*"; exit 1; }
        say() { printf 'SAY %s\n' "$*"; }
        rw_read() { :; }
        sleep() { :; }
        gh() {
            case "$1 $2" in
                "workflow run") return "$drc" ;;
                "run list") [ "$concl" = none ] && return 0; echo 77 ;;
                "run view")
                    if [ "$3" = --job ]; then printf 'x tested-sha: %s\n' "$tested"; else echo "$concl 99"; fi ;;
            esac
        }
        eval "$fn"
        cleanroom_on_sha "$SHA"
    ) 2>&1
    local rc=$?
    [ -s "$d/cleanroom-presha-run-id" ] && echo "RECORDED $(cat "$d/cleanroom-presha-run-id")"
    [ -n "${d:-}" ] && [ "${d#/}" != "$d" ] && rm -rf -- "${d:?}"
    return "$rc"
}

out=$(run_case success "$SHA" 0); rc=$?
{ [ "$rc" -eq 0 ] && grep -q 'RECORDED 77' <<< "$out"; }; row $? "green job, tested-sha == sha -> passes and records the run id"
out=$(run_case failure "$SHA" 0); rc=$?
{ [ "$rc" -ne 0 ] && grep -q '^DIE' <<< "$out"; }; row $? "red job -> dies, no tag"
out=$(run_case success "$OTHER" 0); rc=$?
{ [ "$rc" -ne 0 ] && grep -q 'not exactly' <<< "$out"; }; row $? "green job that tested another sha -> dies"
out=$(run_case success "$SHA" 1); rc=$?
{ [ "$rc" -ne 0 ] && grep -q 'dispatch' <<< "$out"; }; row $? "failed dispatch -> dies"
out=$(run_case none "$SHA" 0); rc=$?
{ [ "$rc" -ne 0 ] && grep -q 'no clean-room.yml run' <<< "$out"; }; row $? "no run appears -> dies"

# order: the call sits in the tag step, ahead of cut_tag's call
call=$(grep -n '^  cleanroom_on_sha "\$MC"' "$AP_FILE" | head -1 | cut -d: -f1)
cut=$(grep -n '^  cut_tag "\$V" "\$T" "\$MC"' "$AP_FILE" | head -1 | cut -d: -f1)
{ [ -n "$call" ] && [ -n "$cut" ] && [ "$call" -lt "$cut" ]; }; row $? "tag step calls cleanroom_on_sha before cut_tag"

[ "$bad" -eq 0 ] && { echo PASS; exit 0; }
echo FAIL >&2; exit 1
