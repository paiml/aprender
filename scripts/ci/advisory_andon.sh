#!/usr/bin/env bash
# advisory_andon.sh -- main-resident advisories are an andon plus a P0 ticket, never a PR block
# (PMAT-5792, contracts/advisory-gate-diff-v1.yaml main_resident_advisory). Runs ONLY on a
# schedule or push to main, never on a PR or merge_group path (C324: no third state there).
#
#   advisory_andon.sh keys FILE      finding keys of a cargo-deny NDJSON file (advisory_diff.sh keys)
#   advisory_andon.sh due CREATED NOW   exit 0 if the ticket is older than 4h (label p0-overdue), else 1
#   advisory_andon.sh emit FILE RC  open one P0 issue per main-resident finding; label p0-overdue past 4h
#   advisory_andon.sh --self-test
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUE_SECS=$((4 * 3600))

due() { # ISO created, ISO now
    local c n
    c=$(date -u -d "$1" +%s) n=$(date -u -d "$2" +%s)
    [ $((n - c)) -gt "$DUE_SECS" ]
}

GH="${GH:-gh}"
emit() { # NDJSON-FILE cargo-deny-rc
    case "$2" in 0 | 1) ;; *) echo "COULD-NOT-CHECK: cargo deny rc $2" >&2; return 75 ;; esac
    local keys k found now
    keys=$(bash "$HERE/advisory_diff.sh" keys "$1") || return 75
    now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    "$GH" label create advisory-andon --color B60205 2>/dev/null || true
    "$GH" label create P0 --color B60205 2>/dev/null || true
    "$GH" label create p0-overdue --color B60205 2>/dev/null || true
    while IFS= read -r k; do
        [ -n "$k" ] || continue
        found=$("$GH" issue list --label advisory-andon --state open --search "in:title \"$k\"" --json number,createdAt,labels \
            --jq '.[0] | if . == null then "" else "\(.number) \(.createdAt) \([.labels[].name] | index("p0-overdue") != null)" end') || return 75
        if [ -z "$found" ]; then
            "$GH" issue create --label advisory-andon --label P0 --title "ADVISORY ANDON: $k" \
                --body "Advisory $k is on main. It does not block PRs (PMAT-5792); fix within 4h of this issue's creation." || return 75
        else
            set -- $found
            if due "$2" "$now" && [ "$3" = false ]; then
                "$GH" issue edit "$1" --add-label p0-overdue || return 75
                "$GH" issue comment "$1" --body "Open past the 4h P0 window." || return 75
            fi
        fi
    done <<<"$keys"
}

self_test() {
    local bad=0 rc
    chk() { rc=0; due "$2" "$3" || rc=$?; [ "$rc" -eq "$4" ] || { echo "SELF-TEST FAIL: $1 want $4 got $rc" >&2; bad=1; }; }
    chk 'under 4h is not due' 2026-10-20T10:00:00Z 2026-10-20T13:59:00Z 1
    chk 'exactly 4h is not due' 2026-10-20T10:00:00Z 2026-10-20T14:00:00Z 1
    chk 'over 4h is due' 2026-10-20T10:00:00Z 2026-10-20T14:00:01Z 0
    local d; d=$(mktemp -d)
    # stub gh: logs argv; `issue list` answers from $STUB_LIST
    printf '#!/usr/bin/env bash\necho "$*" >>"%s/calls"\n[ "$1 $2" = "issue list" ] && printf "%%s" "$STUB_LIST"\nexit 0\n' "$d" >"$d/gh"; chmod +x "$d/gh"
    printf '{"type":"diagnostic","fields":{"code":"vulnerability","severity":"error","advisory":{"id":"RUSTSEC-1"},"graphs":[{"Krate":{"name":"c","version":"1"}}]}}\n' >"$d/f.json"
    ec() { # name want-rc want-grep(or -) rc list
        : >"$d/calls"; rc=0; GH="$d/gh" STUB_LIST="$5" emit "$d/f.json" "$4" >/dev/null 2>&1 || rc=$?
        [ "$rc" -eq "$2" ] || { echo "SELF-TEST FAIL: $1 rc want $2 got $rc" >&2; bad=1; }
        if [ "$3" != - ]; then grep -q -- "$3" "$d/calls" || { echo "SELF-TEST FAIL: $1 missing call $3" >&2; bad=1; }; fi
    }
    ec 'new finding opens a P0 issue' 0 'issue create' 1 ''
    ec 'tool error is cannot-check' 75 - 2 ''
    ec 'old open issue gets p0-overdue' 0 'add-label p0-overdue' 1 '7 2020-01-01T00:00:00Z false'
    ec 'fresh open issue is left alone' 0 - 1 "7 $(date -u +%Y-%m-%dT%H:%M:%SZ) false"
    : >"$d/calls"; GH="$d/gh" STUB_LIST="7 $(date -u +%Y-%m-%dT%H:%M:%SZ) false" emit "$d/f.json" 1 >/dev/null 2>&1 || true
    if grep -q 'issue create\|add-label' "$d/calls"; then echo "SELF-TEST FAIL: fresh issue was touched" >&2; bad=1; fi
    rm -rf -- "${d:?}"
    [ "$bad" -eq 0 ] && echo "advisory_andon self-test: 7 cases ok"
    return "$bad"
}

case "${1:-}" in
    keys) bash "$HERE/advisory_diff.sh" keys "$2" ;;
    due) due "$2" "$3" ;;
    emit) emit "$2" "$3" ;;
    --self-test) self_test ;;
    *) echo "usage: $0 keys FILE | due CREATED NOW | --self-test" >&2; exit 2 ;;
esac
