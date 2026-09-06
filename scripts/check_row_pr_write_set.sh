#!/usr/bin/env bash
# check_row_pr_write_set.sh — a PP-066 ROW PR never writes a shared file
# (PMAT-1062, DAG row G-11, epic #2873; driver R2 of 2026-09-06).
#
# WHY THIS EXISTS
# ---------------
# Eight armed row PRs each carried a hand-edit of docs/specifications/pp-066-dag.yaml
# (status), docs/roadmaps/roadmap.yaml (pmat work complete) or the README's counts.
# Every merge then made the other seven DIRTY and each needed a rebuild — a queue that
# merges ~1 PR/hour spent its throughput on contention over three files no row owns.
# So a row PR's write set is {crate code, tests, contracts/, its own receipt, its book
# page} and nothing shared: the DAG's status is DERIVED from the receipt
# (scripts/lib/dag_status.py), the roadmap and the README counts are written by ONE
# orchestrator docs commit after each merge.
#
#   bash scripts/check_row_pr_write_set.sh --base <ref> --head <ref> [--branch <name>] [--event <name>] [--dag <yaml>] [--readme <md>]
#   bash scripts/check_row_pr_write_set.sh --self-test
#
# Two rules, two universes (the G-11 review quorum, 2026-09-06: a row PR opened from a
# branch NOT named agent/<id> must not walk through):
#   * the DAG and the release spec are the ORCHESTRATOR's on EVERY branch: only an
#     orchestrator branch (agent/pp-066-*, agent/pr-triage*) may write
#     docs/specifications/pp-066-dag.yaml or docs/specifications/PP-066-release-spec.md
#     (its §5.0 block is rendered from the DAG);
#   * a ROW PR — head branch `agent/<id>` whose <id> is a row of the DAG — may in
#     addition not write docs/roadmaps/roadmap.yaml (other work mints tickets there)
#     or a README.md line carrying a count claim (N workspace crates / N contracts /
#     N CLI commands — the same extractor patterns scripts/check_readme_claims.sh reads).
# On the merge_group and push shapes there is no head branch to read, so the guard
# REPORTs that the write set was judged on the pull_request run — which IS required:
# job guard-runner-labels is in the local `gate` job's needs and the ruleset "Green
# Main" requires the context `gate` — and exits 0, stated, not silent.
# Renames are never collapsed (--no-renames): a rename's SOURCE path is a write.
# ci.yml itself is guarded by scripts/check_guards_are_wired.sh (a deleted step is RED).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROG=check_row_pr_write_set
DAG_DEFAULT="docs/specifications/pp-066-dag.yaml"
COUNT_RE='[0-9]+\*{0,2}( +[a-z]+){0,2} +(workspace crates?|contracts?|CLI commands?)\b'   # = check_readme_claims.sh's claim extractors: a line they do not read is not a claim
ORCHESTRATOR_RE='^agent/(pp-066-|pr-triage)'

usage() { printf 'usage: %s --base <ref> --head <ref> [--branch <name>] [--event <name>] [--dag <yaml>] [--readme <md>] | --self-test\n' "$PROG" >&2; exit 2; }

row_ids() { # row_ids <repo> <base> <dag relpath> -> one id per line; the DAG is read at the BASE (a PR that renames or deletes it still has a base)
    git -C "$1" show "$2:$3" 2>/dev/null | python3 -c '
import sys, yaml
d = yaml.safe_load(sys.stdin) or {}
for r in d.get("rows") or []:
    print(r.get("id"))
'
}

judge() { # judge <repo> <base> <head> <branch> <event> <dag> <readme> -> 0 clean / 1 RED / 2 usage-env
    local repo=$1 base=$2 head=$3 branch=$4 event=$5 dag=$6 readme=$7 rc=0 id f
    case "$event" in
        merge_group|push)
            printf 'REPORT %s: the %s shape carries no head branch; the write set was judged on the pull_request run (a required check). Not a verdict.\n' "$PROG" "$event"; return 0 ;;
    esac
    git -C "$repo" cat-file -e "$base:$dag" 2>/dev/null || { printf '%s: ENV - %s is missing at the base %.9s (the box cannot answer)\n' "$PROG" "$dag" "$base" >&2; return 2; }
    local changed; changed=$(git -C "$repo" diff --no-renames --name-only "$base" "$head" --) || { printf '%s: ENV - git diff %s %s failed\n' "$PROG" "$base" "$head" >&2; return 2; }
    if printf '%s' "$branch" | grep -qE -- "$ORCHESTRATOR_RE"; then
        printf 'ok    %s: `%s` is an orchestrator branch; it owns the DAG, the spec block, the roadmap and the README counts\n' "$PROG" "$branch"; return 0
    fi
    # rule 1: the DAG and the spec are orchestrator-only on EVERY branch
    for f in docs/specifications/pp-066-dag.yaml docs/specifications/PP-066-release-spec.md; do
        if printf '%s\n' "$changed" | grep -qxF -- "$f"; then
            printf 'FAIL  %s: branch %s writes %s — orchestrator-only (agent/pp-066-*); a row'\''s status is derived from its receipt and the spec block is rendered\n' "$PROG" "${branch:-<none>}" "$f"; rc=1
        fi
    done
    # rule 2: a ROW PR (agent/<id>, <id> a DAG row) may not write the roadmap or a README count line
    case "$branch" in
        agent/*) id=${branch#agent/} ;;
        *) [ "$rc" = 0 ] && printf 'ok    %s: `%s` is not an agent/<row> branch; the roadmap/README-count rule binds row PRs only\n' "$PROG" "${branch:-<none>}"; return "$rc" ;;
    esac
    if ! row_ids "$repo" "$base" "$dag" | grep -qxF -- "$id"; then
        [ "$rc" = 0 ] && printf 'ok    %s: `%s` names no DAG row; the roadmap/README-count rule binds row PRs only\n' "$PROG" "$branch"; return "$rc"
    fi
    if printf '%s\n' "$changed" | grep -qxF -- docs/roadmaps/roadmap.yaml; then
        printf 'FAIL  %s: row PR %s writes docs/roadmaps/roadmap.yaml — pmat work complete and the ticket edits are the orchestrator docs commit'\''s\n' "$PROG" "$branch"; rc=1
    fi
    if printf '%s\n' "$changed" | grep -qxF -- "$readme"; then
        local hits; hits=$(git -C "$repo" diff "$base" "$head" -- "$readme" | grep -E '^[-+][^-+]' | grep -E -- "$COUNT_RE" || true)
        if [ -n "$hits" ]; then
            printf 'FAIL  %s: row PR %s edits a README count line (the orchestrator docs commit regenerates counts; check_readme_claims.sh lets the README lag, never overstate):\n' "$PROG" "$branch"
            printf '%s\n' "$hits" | head -6 | sed 's|^|        |'; rc=1
        fi
    fi
    [ "$rc" = 0 ] && printf 'PASS  %s: row PR %s writes no shared file (%s changed path(s))\n' "$PROG" "$branch" "$(printf '%s\n' "$changed" | grep -c . || true)"
    return "$rc"
}

# ---------------------------------------------------------------------------
# --self-test: a fixture repo, both polarities per rule
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
    TD=$(mktemp -d "${TMPDIR:-/tmp}/rowpr-selftest.XXXXXX")
    safe_rm_scratch() {
        local victim=${1:-} must=${2:-}
        [ -n "$victim" ] || return 0
        [ -n "$must" ]   || return 0
        [ "$victim" != "/" ] || return 0
        case "$victim" in
          *"$must"*) if [ -n "$victim" ] && [ "$victim" != "/" ]; then rm -rf -- "$victim"; fi ;;
          *) return 0 ;;
        esac
    }
    cleanup() { safe_rm_scratch "$TD" 'rowpr-selftest.'; }
    trap cleanup EXIT
    R="$TD/repo"; mkdir -p "$R/docs/specifications" "$R/docs/roadmaps" "$R/crates/x/src"
    ( cd "$R" && git init -q . && git config user.email t@t && git config user.name t && git config core.hooksPath /dev/null )
    printf 'rows:\n- {id: G-11, pmat_id: PMAT-1062}\n- {id: R-0, pmat_id: PMAT-989}\n' > "$R/docs/specifications/pp-066-dag.yaml"
    printf -- '- id: PMAT-1\n  title: a\n' > "$R/docs/roadmaps/roadmap.yaml"
    printf '# spec\n' > "$R/docs/specifications/PP-066-release-spec.md"
    printf '# apr\n\n**78** workspace crates and **1812** provable contracts across 111 CLI commands.\n\nprose line\n' > "$R/README.md"
    printf 'fn main() {}\n' > "$R/crates/x/src/lib.rs"
    ( cd "$R" && git add -A && git commit -qm base )
    BASE=$(git -C "$R" rev-parse HEAD)
    n=0; red=0
    row() { # row <want rc> <label> <branch> <event> <shell mutating the tree>
        local want=$1 label=$2 branch=$3 event=$4 mut=$5 rc=0
        n=$((n + 1))
        ( cd "$R" && git checkout -q --detach "$BASE" && bash -c "$mut" && git add -A && git commit -qm "$label" --allow-empty ) >/dev/null 2>&1
        judge "$R" "$BASE" "$(git -C "$R" rev-parse HEAD)" "$branch" "$event" docs/specifications/pp-066-dag.yaml README.md > "$TD/out.$n" 2>&1 || rc=$?
        if [ "$rc" = "$want" ]; then printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
        else printf 'FAIL  row %-2s rc=%s (wanted %s)  %s\n' "$n" "$rc" "$want" "$label"; sed 's/^/        /' "$TD/out.$n"; red=1; fi
    }
    row 0 "row PR touching crate code only: PASS"                                   agent/G-11 pull_request 'echo "// x" >> crates/x/src/lib.rs'
    row 1 "row PR writing pp-066-dag.yaml: RED naming the file (the registered mutation)" agent/G-11 pull_request 'echo "- {id: Z-1}" >> docs/specifications/pp-066-dag.yaml'
    row 1 "row PR writing roadmap.yaml: RED"                                          agent/R-0  pull_request 'echo "- id: PMAT-2" >> docs/roadmaps/roadmap.yaml'
    row 1 "row PR writing the release spec (rendered block): RED"                     agent/R-0  pull_request 'echo "x" >> docs/specifications/PP-066-release-spec.md'
    row 1 "row PR bumping a README count line: RED naming the line"                   agent/G-11 pull_request 'sed -i "s/1812/1813/" README.md'
    row 0 "row PR editing README prose (no count line): PASS"                         agent/G-11 pull_request 'sed -i "s/prose line/prose line edited/" README.md'
    row 0 "orchestrator branch writing the DAG, roadmap and README counts: not a row PR" agent/pp-066-spec pull_request 'echo "- {id: Z-1}" >> docs/specifications/pp-066-dag.yaml; sed -i "s/1812/1813/" README.md'
    row 0 "a fix/ branch writing the roadmap: not an agent branch"                    fix/thing  pull_request 'echo "- id: PMAT-2" >> docs/roadmaps/roadmap.yaml'
    row 1 "a fix/ branch writing the DAG: RED on EVERY non-orchestrator branch (quorum Q1)" fix/thing pull_request 'echo "- {id: Z-1}" >> docs/specifications/pp-066-dag.yaml'
    row 1 "an agent/<not-a-row> branch writing the spec: RED"                          agent/nope pull_request 'echo "x" >> docs/specifications/PP-066-release-spec.md'
    row 1 "a row PR RENAMING the DAG away: RED (the source path is a write; --no-renames)" agent/G-11 pull_request 'git mv docs/specifications/pp-066-dag.yaml docs/specifications/moved.yaml'
    row 0 "merge_group shape: REPORT (judged on the pull_request run), exit 0"        agent/G-11 merge_group 'echo "- {id: Z-1}" >> docs/specifications/pp-066-dag.yaml'
    row 0 "push shape: REPORT, exit 0"                                                agent/G-11 push        'echo "- {id: Z-1}" >> docs/specifications/pp-066-dag.yaml'
    for i in 12 13; do grep -q '^REPORT' "$TD/out.$i" || { printf 'FAIL  row %-2s printed no REPORT line: a silent skip\n' "$i"; red=1; }; done
    grep -q 'pp-066-dag.yaml' "$TD/out.11" || { printf 'FAIL  row 11 did not name the renamed file\n'; red=1; }
    n2=$((n + 1)); rc=0; judge "$R" "$BASE" "$BASE" agent/G-11 pull_request docs/specifications/nope.yaml README.md >"$TD/out.$n2" 2>&1 || rc=$?
    if [ "$rc" = 2 ]; then printf 'ok    row %-2s rc=2  a DAG missing at the base is ENV (exit 2), never a pass\n' "$n2"; else printf 'FAIL  row %-2s rc=%s (wanted 2)  a missing DAG\n' "$n2" "$rc"; red=1; fi
    printf '%s/%s rows\n' "$((n2 - red))" "$n2"
    [ "$red" = 0 ] || exit 1
    exit 0
fi

BASE=""; HEAD_REF="HEAD"; BRANCH="${GITHUB_HEAD_REF:-}"; EVENT="${GITHUB_EVENT_NAME:-pull_request}"; DAG="$DAG_DEFAULT"; README="README.md"
while [ $# -gt 0 ]; do
    case "$1" in
        --base) BASE=$2; shift 2 ;;
        --head) HEAD_REF=$2; shift 2 ;;
        --branch) BRANCH=$2; shift 2 ;;
        --event) EVENT=$2; shift 2 ;;
        --dag) DAG=$2; shift 2 ;;   # a path RELATIVE to the repo root; read at the base commit
        --readme) README=$2; shift 2 ;;
        *) usage ;;
    esac
done
[ -n "$BRANCH" ] || BRANCH=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || true)
if [ -z "$BASE" ]; then
    # shellcheck source=scripts/lib/resolve_base.sh
    REPO_ROOT="$ROOT"; . "$ROOT/scripts/lib/resolve_base.sh" || exit 2
    resolve_base "$HEAD_REF" || { printf '%s: no base can be named; pass --base\n' "$PROG" >&2; exit 2; }
    BASE="$BASE_REF"; printf '               comparand: %s (%s)\n' "${BASE:0:9}" "$BASE_HOW"
fi
judge "$ROOT" "$BASE" "$HEAD_REF" "$BRANCH" "$EVENT" "$DAG" "$README"
