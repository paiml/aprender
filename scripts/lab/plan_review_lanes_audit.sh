#!/usr/bin/env bash
# plan_review_lanes_audit.sh — LAB: a plan spec's quorum "Lanes (...)" line must say ADVISORY when every lane is one model family
#
# THE RULE. A plan-review quorum whose lanes are all one family (every id starts with the same word, e.g. gemini-*)
#   is not independent review. The spec must label that line "Lanes (ADVISORY: single family ...)". An unlabeled
#   single-family line reads as a real quorum, which it is not. Tickets aprender#6625 #6626 #6627 (EPIC-0.74/0.71/0.75).
#
# FINDING (exit 3). A "**Lanes (" line in docs/specifications/*.md whose ids span ONE family and lacks "ADVISORY".
# NOT RUN (exit 4). No spec file with a Lanes line was found: zero files read is not green.
# LAB ONLY (C324): runs nightly via .github/workflows/plan-review-lanes-nightly.yml, never in a PR, merge-queue or release job.
#
# USAGE  plan_review_lanes_audit.sh [SPEC_DIR]     audit (default docs/specifications)
#        plan_review_lanes_audit.sh --self-test    planted violation RED, labeled line GREEN, mixed families GREEN, empty dir NOT RUN
set -euo pipefail

families() { # families <lanes-line>: distinct leading word of every model id (name-with-digit-or-dash), one per line
    grep -oE '[a-z][a-z0-9_]*-[a-z0-9][a-z0-9._-]*' <<< "$1" | grep -E '[0-9]' | sed -E 's/-.*//' | sort -u
}

audit() { # audit <dir>
    local dir="$1" f line nfam files=0 lines=0 bad=0 adv=0
    for f in "$dir"/*.md; do
        [ -e "$f" ] || continue
        files=$((files + 1))
        while IFS= read -r line; do
            lines=$((lines + 1))
            nfam=$(families "$line" | grep -c . || true)
            if [ "$nfam" -le 1 ]; then
                if grep -q 'ADVISORY' <<< "$line"; then adv=$((adv + 1)); printf 'advisory  %s\n' "$f"
                else bad=$((bad + 1)); printf 'FINDING   %s: single-family lanes not labeled ADVISORY: %s\n' "$f" "${line:0:120}"; fi
            fi
        done < <(grep -E '^\*\*Lanes \(' "$f" || true)
    done
    printf 'scanned %s file(s), %s Lanes line(s): %s labeled advisory, %s FINDING\n' "$files" "$lines" "$adv" "$bad"
    [ "$lines" -gt 0 ] || { echo "NOT RUN: no Lanes line found under $dir (zero lines read is not green)"; return 4; }
    [ "$bad" -eq 0 ] || return 3
}

self_test() {
    local t rc fails=0; t=$(mktemp -d); trap 'rm -rf -- "${t:?}"' RETURN
    expect() { # expect <name> <want rc> <dir>
        rc=0; audit "$3" >/dev/null 2>&1 || rc=$?
        if [ "$rc" = "$2" ]; then echo "ok   $1 (rc=$rc)"; else echo "FAIL $1: want rc=$2 got rc=$rc"; fails=$((fails + 1)); fi
    }
    mkdir -p "$t/red" "$t/green" "$t/mixed" "$t/empty"
    echo '**Lanes (decision quorum):** gemini-3.1-pro-high, gemini-3.8-flash-high' > "$t/red/p.md"
    echo '**Lanes (ADVISORY: single family, all gemini):** gemini-3.1-pro-high, gemini-3.8-flash-high' > "$t/green/p.md"
    echo '**Lanes (decision quorum):** gemini-3.1-pro-high, claude-opus-5-5' > "$t/mixed/p.md"
    echo 'no lanes here' > "$t/empty/p.md"
    expect "planted unlabeled single-family is a FINDING" 3 "$t/red"
    expect "labeled advisory line is green" 0 "$t/green"
    expect "mixed families need no label" 0 "$t/mixed"
    expect "no Lanes line is NOT RUN" 4 "$t/empty"
    [ "$fails" -eq 0 ] || { echo "$fails FAIL"; return 1; }
    echo "plan_review_lanes_audit self-test ok: 4 cases"
}

case "${1:-}" in
    --self-test) self_test ;;
    *) audit "${1:-docs/specifications}" ;;
esac
