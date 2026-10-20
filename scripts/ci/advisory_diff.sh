#!/usr/bin/env bash
# advisory_diff.sh -- PR advisory verdict = findings the PR introduces (PMAT-5792, contract advisory-gate-diff-v1).
#
#   advisory_diff.sh keys FILE            print sorted finding keys "code|id|crate@version" from cargo-deny NDJSON
#   advisory_diff.sh verdict PR_JSON PR_RC MAIN_JSON MAIN_RC
#                                         exit 0: nothing new; 1: PR introduces findings; 75: could not check
#   advisory_diff.sh --self-test          case table
#
# Fail closed (exit 75, never a pass): rc 0 with findings, rc 1 with none, rc other than 0/1, unparsable JSON,
# a missing file. The key is code|advisory-id|crate@version so a new crate under an id main already carries,
# and a yanked finding (no id), are both seen. cargo-deny writes NDJSON diagnostics to STDERR; the caller
# redirects it to FILE.
set -euo pipefail

CANT=75

keys() { # FILE -> keys on stdout; nonzero when the file does not parse
    jq -R -r 'fromjson? | select(.type == "diagnostic" or has("fields"))
      | .fields | select(.severity == "error" or .code == "yanked")
      | "\(.code // "?")|\(.advisory.id // "-")|\(.graphs[0].Krate.name // "?")@\(.graphs[0].Krate.version // "?")"' "$1" | LC_ALL=C sort -u
}

parses() { # every non-blank line is JSON
    jq -R -n -e '[inputs | select(length > 0) | fromjson] | length >= 0' "$1" >/dev/null 2>&1
}

side() { # JSON RC -> echoes keys; returns CANT when the side cannot be trusted
    local f=$1 rc=$2 k
    [ -f "$f" ] || { echo "COULD-NOT-CHECK: $f missing" >&2; return "$CANT"; }
    case "$rc" in 0 | 1) ;; *) echo "COULD-NOT-CHECK: cargo-deny rc=$rc" >&2; return "$CANT" ;; esac
    parses "$f" || { echo "COULD-NOT-CHECK: $f is not NDJSON" >&2; return "$CANT"; }
    k=$(keys "$f")
    local e; e=$(jq -R -r 'fromjson? | .fields? | select(.severity == "error") | .code' "$f" | head -c1)
    if [ "$rc" -eq 1 ] && [ -z "$e" ]; then echo "COULD-NOT-CHECK: rc=1 but no findings parsed" >&2; return "$CANT"; fi
    if [ "$rc" -eq 0 ] && [ -n "$e" ]; then echo "COULD-NOT-CHECK: rc=0 but error findings parsed" >&2; return "$CANT"; fi
    printf '%s\n' "$k"
}

verdict() {
    local pr main new
    pr=$(side "$1" "$2") || return "$CANT"
    main=$(side "$3" "$4") || return "$CANT"
    new=$(comm -23 <(printf '%s\n' "$pr" | sed '/^$/d') <(printf '%s\n' "$main" | sed '/^$/d'))
    if [ -n "$new" ]; then
        echo "PR introduces advisories:" >&2
        printf '%s\n' "$new" >&2
        return 1
    fi
    if [ -n "$(printf '%s' "$main" | tr -d '\n')" ]; then
        echo "ANDON: advisories already on main (not blocking this PR): $(printf '%s' "$main" | tr '\n' ' ')" >&2
    fi
    return 0
}

self_test() {
    local d bad=0
    d=$(mktemp -d)
    # shellcheck disable=SC2064
    trap "rm -rf -- '${d:?}'" RETURN
    row() { printf '{"type":"diagnostic","fields":{"code":"%s","severity":"%s","advisory":{"id":"%s"},"graphs":[{"Krate":{"name":"%s","version":"%s"}}]}}\n' "$@"; }
    row vulnerability error RUSTSEC-2026-0378 wasmtime 48.0.6 >"$d/a.json"
    row vulnerability error RUSTSEC-2026-0378 wasmtime 48.0.6 >"$d/a2.json"; row vulnerability error RUSTSEC-2026-0379 wasmtime-x 48.0.6 >>"$d/a2.json"
    row vulnerability error RUSTSEC-2026-0378 wasmtime 48.0.6 >"$d/a3.json"; row vulnerability error RUSTSEC-2026-0378 wasmtime-y 1.0.0 >>"$d/a3.json"
    : >"$d/empty.json"
    printf 'garbage\n' >"$d/bad.json"
    printf '{"type":"diagnostic","fields":{"code":"yanked","severity":"warning","graphs":[{"Krate":{"name":"y","version":"1"}}]}}\n' >"$d/yank.json"
    t() { # name want pr-file pr-rc main-file main-rc
        local rc=0
        verdict "$d/$3" "$4" "$d/$5" "$6" >/dev/null 2>&1 || rc=$?
        [ "$rc" -eq "$2" ] || { echo "SELF-TEST FAIL: $1 wanted rc=$2 got $rc" >&2; bad=1; }
    }
    t 'same advisory on both sides: pass'            0  a.json 1 a.json 1
    t 'both clean: pass'                             0  empty.json 0 empty.json 0
    t 'PR adds a new id: block'                      1  a2.json 1 a.json 1
    t 'PR adds a crate under an id main has: block'  1  a3.json 1 a.json 1
    t 'PR adds a yanked crate: block'                1  yank.json 0 empty.json 0
    t 'main has it, PR fixed it: pass'               0  empty.json 0 a.json 1
    t 'rc 0 but findings parsed: cannot check'       75 a.json 0 empty.json 0
    t 'rc 1 but nothing parsed: cannot check'        75 empty.json 1 empty.json 0
    t 'tool error rc 2: cannot check'                75 empty.json 2 empty.json 0
    t 'garbage JSON: cannot check'                   75 bad.json 1 empty.json 0
    t 'main side missing: cannot check'              75 a.json 1 nope.json 1
    [ "$bad" -eq 0 ] && echo "advisory_diff self-test: 11 cases ok"
    return "$bad"
}

case "${1:-}" in
    --self-test) self_test ;;
    keys) keys "$2" ;;
    verdict) verdict "$2" "$3" "$4" "$5" ;;
    *) echo "usage: $0 keys F | verdict PR PRRC MAIN MAINRC | --self-test" >&2; exit 2 ;;
esac
