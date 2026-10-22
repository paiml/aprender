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
    local wf="${CI_SCOPE_WF:-$REPO_ROOT/.github/workflows/ci.yml}" bad="" n body line
    # No yq on the clean-room runner (run 39721317355): read the YAML with awk. Full-line and trailing
    # ` #` comments are stripped first (ci.yml comments name `paths`, and a trailing comment must not
    # satisfy a check). Results are captured and matched with [[ ]], never piped into `grep -q`.
    # top KEY = body of top-level KEY; job J / ev E = body of that job / trigger; jl J KEY = the job-level
    # (4-space) KEY line only, so a step-level `if:` or `needs:` cannot stand in for it.
    if [ ! -r "$wf" ] || ! grep -q '^jobs:' "$wf"; then
        echo "FAIL wiring: cannot read ci.yml"; fail=1
    else
        nc() { sed -E -e '/^[[:space:]]*#/d' -e 's/[[:space:]]+#.*$//' -e 's/[[:space:]]+$//' "$wf"; }
        top() { nc | awk -v k="$1" '$0 == k":" {f=1; next} f && /^[^ ]/ {f=0} f'; }
        job() { top jobs | awk -v k="$1" '$0 == "  "k":" || $0 == "  \""k"\":" {f=1; next} f && /^  [^ ]/ {f=0} f'; }
        ev() { top on | awk -v k="$1" '$0 == "  "k":" {f=1; next} f && /^  [^ ]/ {f=0} f'; }
        jl() { job "$1" | awk -v k="$2" 'index($0, "    "k": ") == 1 {print; exit}'; }
        has_scope() { [[ "$1" =~ [\[,][[:space:]]*scope[[:space:]]*[],] ]]; }
        [ "$(nc | grep -cE '^on:$')" = 1 ] || bad="$bad; on: is not a block mapping"
        for n in push pull_request merge_group; do
            [ "$(top on | awk -v k="$n" '$0 == "  "k":" {c++} END {print c+0}')" = 1 ] || bad="$bad; trigger $n is not a plain block key"
            body="$(ev "$n")"
            [[ "$body" =~ (^|[^A-Za-z_-])paths(-ignore)?[[:space:]]*: ]] && bad="$bad; $n has a paths filter (stuck required checks)"
        done
        line="$(ev pull_request | awk '/^    types:/ {print; exit}')"
        [[ "$line" =~ ^[[:space:]]+types:[[:space:]]*\[.*ready_for_review.*\]$ ]] || bad="$bad; pull_request types lack ready_for_review"
        [ "$(jl scope if)" = "    if: \${{ github.event_name == 'pull_request' }}" ] || bad="$bad; scope job missing or not pull_request-only"
        for n in x86-main workspace-test-shard; do
            has_scope "$(jl "$n" needs)" && [[ "$(jl "$n" if)" == *"needs.scope.outputs.skip != '1'"* ]] || bad="$bad; $n does not honour scope"
        done
        has_scope "$(jl workspace-test needs)" && [[ "$(jl workspace-test if)" == *"needs.scope.outputs.skip != '1'"* ]] || bad="$bad; workspace-test does not honour scope"
        for n in ci-gate gate; do
            has_scope "$(jl "$n" needs)" || bad="$bad; $n does not need scope"
            body="$(job "$n")"
            [[ "$body" == *'SCOPE_SKIP: ${{ needs.scope.outputs.skip }}'* && "$body" == *'"${SCOPE_SKIP:-}" = 1'* ]] || bad="$bad; $n does not read scope"
        done
        [ "$(top concurrency | awk '/^  cancel-in-progress:/ {print; exit}')" = "  cancel-in-progress: \${{ github.event_name == 'pull_request' || github.event_name == 'merge_group' }}" ] || bad="$bad; concurrency no longer cancels superseded PR runs (exact expression pinned)"
        grep -qF 'git diff --no-renames --name-only' "$wf" || bad="$bad; scope diff may report renames by new path only (fails open)"
    fi
    [ -z "$bad" ] || { printf 'FAIL wiring: %s\n' "${bad#; }"; fail=1; }
    # MUTANTS: each slip-through of a weaker reader must turn the wiring RED on a mutated copy of ci.yml.
    if [ -z "${CI_SCOPE_WF:-}" ]; then
        local md mn mexp
        md="$(mktemp -d)"
        mutant() { # LABEL SED-EXPR
            sed -E "$2" "$wf" > "$md/ci.yml"
            if cmp -s "$wf" "$md/ci.yml"; then echo "FAIL mutant $1: sed changed nothing"; fail=1
            elif CI_SCOPE_WF="$md/ci.yml" bash "${BASH_SOURCE[0]}" --self-test > /dev/null 2>&1; then echo "FAIL mutant $1: self-test stayed green"; fail=1
            else echo "ok    red  mutant $1"; fi
        }
        mutant paths-filter-pr '0,/^    types:/s//    paths-ignore: [docs\/**]\n    types:/'
        mutant paths-oneline-push 's/^  push:$/  push: {paths: [x]}/'
        mutant paths-flow-under-push '/^  push:$/{n;s/^(    branches:.*)$/\1\n    paths: [x]/}'
        mutant no-ready_for_review 's/, ready_for_review//'
        mutant scope-if "s/^(    if: .\{\{ github.event_name == 'pull_request') \}\}/\1 || true }}/"
        mutant cancel-false-trailing-comment 's/^  cancel-in-progress: .*/  cancel-in-progress: false # pull_request/'
        mutant cancel-true 's/^  cancel-in-progress: .*/  cancel-in-progress: true/'
        mutant no-renames 's/git diff --no-renames --name-only/git diff --name-only/'
        mutant x86-main-no-scope-clause "/^  x86-main:\$/,/^  [a-z]/s/ && needs.scope.outputs.skip != '1'//"
        mutant shard-needs-comment 's/^(  workspace-test-shard:)$/\1/;/^  workspace-test-shard:$/,/^  [a-z]/s/^    needs: \[scope\]/    needs: [mg-reuse] # [scope]/'
        mutant gate-needs-scope-dropped '/^  gate:$/,/^  [a-z]/s/, scope\]/]/'
        mutant ci-gate-reads-scope-dropped '/^  ci-gate:$/,/^  [a-z]/s/SCOPE_SKIP: .*/SCOPE_SKIP: ""/'
        mutant workspace-test-if-dropped "/^  workspace-test:\$/,/^  [a-z]/s/ && needs.scope.outputs.skip != '1'//"
        rm -rf "${md:?}"
    fi
    [ "$fail" -eq 0 ] && echo "ci_scope: self-test ok"
    return "$fail"
}

case "${1:-}" in
    classify) shift; classify "${1:-false}" ;;
    --self-test) self_test ;;
    *) echo "usage: $0 classify <draft> | --self-test" >&2; exit 2 ;;
esac
