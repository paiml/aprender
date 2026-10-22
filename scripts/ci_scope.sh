#!/usr/bin/env bash
# ci_scope.sh -- decide whether a pull_request needs the fat CI jobs at all.
#
#   ci_scope.sh classify <draft:true|false>   < changed paths, one per line
#       prints `skip=1 reason=draft`, `skip=1 reason=docs-only` or `skip=0 reason=full`
#   ci_scope.sh --self-test                    case table + wiring checks on ci.yml
#
# WHY. CI demand: a draft PR and a PR that only touches inert documentation each cost
# a full x86-main + three workspace-test shards. Skipping happens INSIDE ci.yml, never
# with a workflow-level `paths-ignore`: a filtered-out workflow never reports, and the
# required checks (`ci / gate`, `gate`, `workspace-test`) would sit at "expected"
# forever. Here the verdict jobs still run and report success on a skipped scope.
#
# SCOPE. pull_request ONLY. merge_group and push never reach this script's verdict
# (the `scope` job is `if: github.event_name == 'pull_request'`), so a queue build or
# a main push is never skipped or cancelled by it. A draft gets a fresh run when it is
# marked ready (`ready_for_review` is in the trigger types).
#
# DOCS-ONLY is an ALLOWLIST of trees no test or gate reads, found by searching the
# repo for readers (docs/audits, docs/roadmaps, docs/specifications, docs/findings,
# docs/BEATS.md, README.md, CLAUDE.md are READ by gates and are therefore NOT here; neither is any
# docs/**/*.md, which check_no_claim_literals.sh and check_perf_claims_cite_receipts.sh scan).
# Every changed path must match; any other path, an empty list or a `..` is `full`.
# Growing the list is a gate-content change: prove no reader first.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2
DOCS_ONLY_RE='^docs/images/[^.][^[:space:]]*\.(png|jpg|jpeg|gif|svg|webp)$'

classify() { # classify <draft> ; paths on stdin
    local draft=${1:-false} n=0 p
    if [ "$draft" = true ]; then cat >/dev/null; echo "skip=1 reason=draft"; return 0; fi
    while IFS= read -r p; do
        [ -n "$p" ] || continue
        n=$((n + 1))
        case "$p" in *..*) echo "skip=0 reason=full"; return 0 ;; esac
        printf '%s\n' "$p" | grep -Eq "$DOCS_ONLY_RE" || { echo "skip=0 reason=full"; return 0; }
    done
    [ "$n" -gt 0 ] && echo "skip=1 reason=docs-only" || echo "skip=0 reason=full"
}

self_test() {
    local fail=0 got
    t() { # t NAME WANT DRAFT PATHS...
        local name=$1 want=$2 draft=$3; shift 3
        got=$(printf '%s\n' "$@" | classify "$draft")
        [ "$got" = "$want" ] || { echo "FAIL $name: want [$want] got [$got]"; fail=1; }
    }
    t "draft skips" "skip=1 reason=draft" true crates/a/src/lib.rs
    t "draft, no paths" "skip=1 reason=draft" true
    t "images only" "skip=1 reason=docs-only" false docs/images/x.png docs/images/y.svg
    t "docs archive md is gated" "skip=0 reason=full" false docs/archive/a.md
    t "images md is gated" "skip=0 reason=full" false docs/images/README.md
    t "postmortem md is gated" "skip=0 reason=full" false docs/postmortems/2026-10-19.md
    t "code" "skip=0 reason=full" false crates/a/src/lib.rs
    t "images + code" "skip=0 reason=full" false docs/images/a.png crates/a/src/lib.rs
    t "README is gated" "skip=0 reason=full" false README.md
    t "CLAUDE.md is gated" "skip=0 reason=full" false CLAUDE.md
    t "BEATS is gated" "skip=0 reason=full" false docs/BEATS.md
    t "roadmap entries are gated" "skip=0 reason=full" false docs/roadmaps/entries/PMAT-1.yaml
    t "specifications are gated" "skip=0 reason=full" false docs/specifications/a.md
    t "audits are read" "skip=0 reason=full" false docs/audits/surface_audit.csv
    t "findings ledger is gated" "skip=0 reason=full" false docs/findings/w1.jsonl
    t "workflow file" "skip=0 reason=full" false .github/workflows/ci.yml
    t "contracts" "skip=0 reason=full" false contracts/a-v1.yaml
    t "dot-dot escape" "skip=0 reason=full" false docs/images/../../Cargo.toml
    t "hidden file under docs" "skip=0 reason=full" false docs/images/.hidden
    t "prefix lookalike" "skip=0 reason=full" false docs/images-x/a.png
    t "empty diff is full" "skip=0 reason=full" false
    local wf="$REPO_ROOT/.github/workflows/ci.yml" bad="" n
    # No yq on the clean-room runner (run 39721317355): read the YAML with awk. Full-line comments are
    # stripped first (the ci.yml comments name `paths`). blk KEY INDENT = body of the top-level (INDENT 0)
    # or second-level (INDENT 2) key KEY, up to the next key at the same or lower indent.
    if [ ! -r "$wf" ] || ! grep -q '^jobs:' "$wf"; then
        echo "FAIL wiring: cannot read ci.yml"; fail=1
    else
        nc() { grep -vE '^[[:space:]]*#' "$wf"; }
        top() { nc | awk -v k="$1" '$0 ~ "^"k":" {f=1; next} f && /^[^ ]/ {f=0} f'; }
        job() { top jobs | awk -v k="$1" '$0 == "  "k":" || $0 == "  \""k"\":" {f=1; next} f && /^  [^ ]/ {f=0} f'; }
        ev() { top on | awk -v k="$1" '$0 ~ "^  "k":" {f=1; next} f && /^  [^ ]/ {f=0} f'; }
        top on | grep -qE '^  pull_request:' || bad="$bad; no pull_request trigger"
        ev pull_request | grep -qE '^    paths(-ignore)?:' && bad="$bad; pull_request has a paths filter (stuck required checks)"
        ev pull_request | grep -qE '^    types:.*ready_for_review' || bad="$bad; pull_request types lack ready_for_review"
        for n in push merge_group; do
            ev "$n" | grep -qE '^    paths(-ignore)?:' && bad="$bad; $n has a paths filter"
        done
        job scope | grep -qF "if: \${{ github.event_name == 'pull_request' }}" || bad="$bad; scope job missing or not pull_request-only"
        for n in x86-main workspace-test-shard; do
            job "$n" | grep -E '^    needs:' | grep -qF scope && job "$n" | grep -E '^    if:' | grep -qF "needs.scope.outputs.skip != '1'" || bad="$bad; $n does not honour scope"
        done
        for n in ci-gate gate workspace-test; do
            job "$n" | grep -E '^    needs:' | grep -qF scope || bad="$bad; $n does not need scope"
            job "$n" | grep -qF 'needs.scope.outputs.skip' || bad="$bad; $n does not read scope"
        done
        top concurrency | grep -E '^  cancel-in-progress:' | grep -qF pull_request || bad="$bad; concurrency no longer cancels superseded PR runs"
        grep -qF 'git diff --no-renames --name-only' "$wf" || bad="$bad; scope diff may report renames by new path only (fails open)"
    fi
    [ -z "$bad" ] || { printf 'FAIL wiring: %s\n' "${bad#; }"; fail=1; }
    [ "$fail" -eq 0 ] && echo "ci_scope: self-test ok"
    return "$fail"
}

case "${1:-}" in
    classify) shift; classify "${1:-false}" ;;
    --self-test) self_test ;;
    *) echo "usage: $0 classify <draft> | --self-test" >&2; exit 2 ;;
esac
