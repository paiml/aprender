#!/usr/bin/env bash
# ci_docs_only.sh -- the ONE docs-only decision the whole ci.yml run reads (#3668).
#
# A roadmap-only PR used to pay ~26 minutes of guard-cargo and ~6 of determinism
# for a diff neither of them reads. The `changes` job runs this once; guard-cargo
# and determinism skip on `docs_only=true`, guard-tree runs the docs readers
# guard-cargo would have run, and `gate` accepts those skips ONLY when this said
# true (scripts/check_ci_gate_docs_only_rule.sh runs gate's block).
#
# The definition is NOT re-implemented here. docs-only is exactly what
# scripts/ci_test_tier.sh already calls it (#3658): every touched path under
# docs/roadmaps/ or docs/audits/ AND present at HEAD. This script asks that
# oracle and answers true only on its docs-only verdict. Everything else answers
# false: another event, an empty diff, a missing diff file, an oracle that
# errors or prints something else. It FAILS CLOSED -- the answer is always
# printed and the exit is 0, so a broken decision runs the full CI, never less.
#
#   ci_docs_only.sh --event EVENT --diff-from FILE   -> docs_only=true|false, reason=...
#   ci_docs_only.sh --self-test                      -> the case table
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
TIER="${CI_DOCS_ONLY_TIER_SCRIPT:-$ROOT/scripts/ci_test_tier.sh}"

answer() { printf 'docs_only=%s\nreason=%s\n' "$1" "$2"; exit 0; }

decide() { # decide EVENT FILE -> prints the answer, always exit 0
    local evt=$1 diff=$2 out
    [ "$evt" = pull_request ] || answer false "event '$evt': only a pull_request is judged docs-only; every other event runs everything"
    [ -f "$diff" ] || answer false "no diff file '$diff' -- cannot judge, so everything runs"
    [ -s "$diff" ] || answer false "empty diff -- nothing to classify, so everything runs"
    # Pre-filter, narrower than the oracle and so only ever in the fail-closed
    # direction: a path outside the two directories is an immediate false, and a
    # code PR never pays the oracle's crate selection for a question it lost.
    # The oracle alone can say true.
    # grep reads the FILE, never a pipe (check_no_pipe_into_grep_q.sh): `grep -q`
    # exiting early SIGPIPEs its writer, and under pipefail that reads as no-match.
    local outside
    outside=$(grep -vE '^(docs/(roadmaps|audits)/|$)' "$diff") || outside=''
    if [ -n "$outside" ]; then
        answer false "a touched path is outside docs/roadmaps/ and docs/audits/: ${outside%%$'\n'*}"
    fi
    out=$(bash "$TIER" --event pull_request --diff-from "$diff" 2>&1) \
        || answer false "ci_test_tier.sh failed ($(printf '%s' "$out" | tail -1 | cut -c1-120)) -- fail closed"
    if grep -qx 'tier=none' <<< "$out" && grep -q '^reason=.*docs-only' <<< "$out"; then
        answer true "all $(grep -c . "$diff") touched path(s) are docs/roadmaps/** or docs/audits/** present at HEAD (ci_test_tier.sh docs-only, #3658)"
    fi
    answer false "not docs-only per ci_test_tier.sh: tier=$(printf '%s\n' "$out" | sed -n 's/^tier=//p' | head -1)"
}

self_test() {
    local td bad=0 n=0 rm au
    td=$(mktemp -d "${TMPDIR:-/tmp}/docs-only.XXXXXX") || exit 2
    rmtree() { case "${1:-}" in ''|/) return 0 ;; *) [ -d "$1" ] && rm -rf -- "$1" ;; esac; return 0; }
    trap 'rmtree "${td:-}"' EXIT
    # row WANT DESCRIPTION EVENT FILE [TIER-SCRIPT]
    row() {
        local want=$1 desc=$2 evt=$3 f=$4 got
        n=$((n + 1))
        got=$(CI_DOCS_ONLY_TIER_SCRIPT="${5:-$TIER}" bash "$0" --event "$evt" --diff-from "$f" | sed -n 's/^docs_only=//p')
        if [ "$got" = "$want" ]; then printf 'ok    row %-2s %-5s %s\n' "$n" "$want" "$desc"
        else printf 'FAIL  row %-2s wanted %s, got "%s": %s\n' "$n" "$want" "$got" "$desc"; bad=1; fi
    }
    # Real paths at HEAD of this tree, so the oracle's present-at-HEAD test passes.
    rm=$(git -C "$ROOT" ls-files 'docs/roadmaps/*' | head -1)
    au=$(git -C "$ROOT" ls-files 'docs/audits/*' | head -1)
    { [ -n "$rm" ] && [ -n "$au" ]; } || { echo "ENV   no tracked docs/roadmaps or docs/audits file to build fixtures from" >&2; exit 2; }
    printf '%s\n%s\n' "$rm" "$au" > "$td/docs.txt"
    printf '%s\ncrates/aprender-core/src/lib.rs\n' "$rm" > "$td/mixed.txt"
    printf '%s\n.github/workflows/ci.yml\n' "$rm" > "$td/wf.txt"
    printf 'docs/roadmaps/entries/PMAT-DELETED-3668.yaml\n' > "$td/deleted.txt"
    printf 'docs/specifications/aprender-monorepo-consolidation.md\n' > "$td/spec.txt"
    : > "$td/empty.txt"
    printf '#!/usr/bin/env bash\necho boom >&2; exit 3\n' > "$td/tier-dies.sh"
    printf '#!/usr/bin/env bash\nprintf "tier=none\\nreason=x: empty diff\\n"\n' > "$td/tier-none-not-docs.sh"
    printf '#!/usr/bin/env bash\nprintf "tier=none\\nreason=x: docs-only\\n"\n' > "$td/tier-says-docs.sh"
    row true  "roadmap + audit only, both present at HEAD"                       pull_request "$td/docs.txt"
    row false "roadmap + one .rs file (the control: everything runs)"            pull_request "$td/mixed.txt"
    row false "roadmap + ci.yml (a workflow edit is never docs-only)"           pull_request "$td/wf.txt"
    row false "a DELETED roadmap path (a tree reader asserts cited paths exist)" pull_request "$td/deleted.txt"
    row false "docs/specifications/ is read by tests, not docs-only"            pull_request "$td/spec.txt"
    row false "empty diff"                                                       pull_request "$td/empty.txt"
    row false "missing diff file"                                                pull_request "$td/nope.txt"
    row false "merge_group with a docs-only diff (only pull_request is judged)"  merge_group  "$td/docs.txt"
    row false "push with a docs-only diff"                                       push         "$td/docs.txt"
    row false "the oracle ERRORS -> fail closed"                                 pull_request "$td/docs.txt" "$td/tier-dies.sh"
    row false "the oracle says tier=none for a non-docs reason"                  pull_request "$td/docs.txt" "$td/tier-none-not-docs.sh"
    row false "(discriminator) a non-docs path is false even if the oracle says docs-only" pull_request "$td/mixed.txt" "$td/tier-says-docs.sh"
    row true  "(discriminator) on a docs diff the oracle's docs-only verdict is what flips it" pull_request "$td/docs.txt" "$td/tier-says-docs.sh"
    [ "$bad" = 0 ] && { echo "SELF-TEST PASSED ($n rows)"; return 0; }
    echo "SELF-TEST FAILED" >&2; return 1
}

EVENT="" DIFF=""
while [ $# -gt 0 ]; do
    case "$1" in
        --event) EVENT=${2:-}; shift 2 ;;
        --diff-from) DIFF=${2:-}; shift 2 ;;
        --self-test) self_test; exit $? ;;
        -h|--help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) printf 'usage: %s --event EVENT --diff-from FILE | --self-test\n' "$0" >&2; exit 2 ;;
    esac
done
[ -n "$EVENT" ] || { printf 'usage: %s --event EVENT --diff-from FILE | --self-test\n' "$0" >&2; exit 2; }
decide "$EVENT" "$DIFF"
