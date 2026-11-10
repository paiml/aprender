#!/usr/bin/env bash
# check_pretag_cleanroom.sh -- aprender#6500: the tag must not be cut unless clean-room (aprender) was
# green on the release COMMIT, with every tested-sha equal to that commit. Extracts pretag_cleanroom()
# from scripts/release/autopilot.sh and runs it against a stub `gh`, once per outcome:
#   green + exact sha -> returns 0 and records the run id
#   job red | sha differs | no run appears | dispatch fails -> die (no tag)
# Also requires the call to sit before cut_tag in the non-rehearsal tag branch.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
AP_SRC="$ROOT/scripts/release/autopilot.sh"
MC=0123456789abcdef0123456789abcdef01234567
OTHER=fedcba9876543210fedcba9876543210fedcba98
pass=0; fail=0
ok()  { printf '  ok   %s\n' "$1"; pass=$((pass+1)); }
bad() { printf '  FAIL %s\n' "$1"; fail=$((fail+1)); }

fn=$(awk '/^pretag_cleanroom\(\) \{/,/^\}/' "$AP_SRC")
[ -n "$fn" ] || { echo "pretag_cleanroom() missing from autopilot.sh"; exit 1; }

# run_case <dispatch-rc> <run-id|none> <job-conclusion|none> <tested-sha>  -> prints "rc=N" and any DIE line
run_case() {
    local d; d=$(mktemp -d) || return 2
    local out
    out=$(
        export AP="$d" LOG="$d/log" INFRA=paiml/infra AP_SETTLE=0 AP_POLL=0
        : > "$LOG"
        die() { echo "DIE: $*"; exit 1; }
        say() { :; }
        sleep() { :; }
        gh() {
            case "$1 $2" in
                "workflow run") return "${DISPATCH_RC}" ;;
            esac
            case "$1 $2" in
                "run list") [ "$RUN_ID" = none ] || echo "$RUN_ID" ;;
                "run view")
                    if [ "$3" = "--job" ]; then
                        echo "tested-sha: $TESTED"
                    elif [ "$JOB" != none ]; then
                        echo "$JOB 777"
                    fi ;;
            esac
            return 0
        }
        export -f gh die say sleep 2>/dev/null || true
        eval "$fn"
        pretag_cleanroom "$MC"
        echo "RETURNED"
        [ -s "$d/pretag-cleanroom-run-id" ] && echo "RECORDED $(cat "$d/pretag-cleanroom-run-id")"
    )
    rm -rf -- "${d:?}"
    printf '%s\n' "$out"
}

mk() { DISPATCH_RC=$1 RUN_ID=$2 JOB=$3 TESTED=$4 run_case; }
# `gh workflow run` rc comes from DISPATCH_RC; keep the stub simple
export DISPATCH_RC=0

o=$(mk 0 4242 success "$MC")
grep -q RETURNED <<<"$o" && grep -q 'RECORDED 4242' <<<"$o" && ok "green-on-exact-sha-passes" || bad "green-on-exact-sha-passes: $o"
o=$(mk 0 4242 failure "$MC")
grep -q 'DIE:.*concluded .failure' <<<"$o" && ! grep -q RETURNED <<<"$o" && ok "red-job-stops" || bad "red-job-stops: $o"
o=$(mk 0 4242 success "$OTHER")
grep -q 'DIE:.*not exactly' <<<"$o" && ! grep -q RETURNED <<<"$o" && ok "wrong-tested-sha-stops" || bad "wrong-tested-sha-stops: $o"
o=$(mk 0 none success "$MC")
grep -q 'DIE:.*no pre-tag clean-room run appeared' <<<"$o" && ok "no-run-stops" || bad "no-run-stops: $o"
o=$(mk 1 4242 success "$MC")
grep -q 'DIE:.*dispatch.*failed' <<<"$o" && ok "dispatch-failure-stops" || bad "dispatch-failure-stops: $o"

# structure: the call precedes cut_tag in the non-rehearsal branch
pl=$(grep -n '^  pretag_cleanroom "\$MC"$' "$AP_SRC" | head -1 | cut -d: -f1)
cl=$(awk -v p="${pl:-0}" 'NR>p && /^  cut_tag "\$V" "\$T" "\$MC"$/ {print NR; exit}' "$AP_SRC")
if [ -n "${pl:-}" ] && [ -n "${cl:-}" ] && [ "$pl" -lt "$cl" ]; then ok "call-precedes-cut_tag"; else bad "call-precedes-cut_tag (pl=${pl:-} cl=${cl:-})"; fi

echo
[ "$fail" -eq 0 ] && { echo "check_pretag_cleanroom: OK ($pass cases)"; exit 0; }
echo "check_pretag_cleanroom: FAIL ($fail of $((pass+fail)))"; exit 1
