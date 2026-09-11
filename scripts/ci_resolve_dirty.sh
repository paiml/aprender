#!/usr/bin/env bash
# shellcheck disable=SC2297,SC2086,SC1117,SC2016
# bashrs disable-file=SEC014,BRS0008,REL003
set -euo pipefail

PROG="${0##*/}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

APPLY=0
SELFTEST=0
declare -a PRS=()
REPO=""

while [ $# -gt 0 ]; do
    case "$1" in
        --apply) APPLY=1; shift ;;
        --selftest) SELFTEST=1; shift ;;
        --pr) PRS+=("$2"); shift 2 ;;
        --repo) REPO="$2"; shift 2 ;;
        *) printf "Unknown arg: %s\n" "$1" >&2; exit 2 ;;
    esac
done

if [ "$SELFTEST" -eq 1 ]; then
    tmp_td_name=$(mktemp)
    mktemp -d "${TMPDIR:-/tmp}/ci_resolve_selftest.XXXXXX" > "$tmp_td_name"
    read -r td < "$tmp_td_name"
    rm -f "$tmp_td_name"
    [ -z "$td" ] && exit 2

    cleanup() {
        local victim="${td:-}"
        if [[ -n "$victim" && "$victim" != "/" && "$victim" == *ci_resolve_selftest.* ]]; then
            rm -rf -- "${victim:?}"
        fi
    }
    trap cleanup EXIT

    export GIT_TERMINAL_PROMPT=0
    
    mkdir -p "$td/.empty-template"
    
    repo1="$td/repo1"
    git -C "$td" init -q --template="$td/.empty-template" repo1
    git -C "$repo1" config user.email test@example.com
    git -C "$repo1" config user.name test
    
    driver_script="$REPO_ROOT/scripts/lib/roadmap_merge.py"
    
    mkdir -p "$repo1/docs/roadmaps"
    printf 'roadmap:\n- id: PMAT-1\n  title: one\n' > "$repo1/docs/roadmaps/roadmap.yaml"
    
    echo "docs/roadmaps/roadmap.yaml merge=roadmap" > "$repo1/.gitattributes"
    git -C "$repo1" add docs/roadmaps/roadmap.yaml .gitattributes
    git -C "$repo1" commit -q -m "base"
    
    git -C "$repo1" branch -q main
    
    git -C "$repo1" checkout -q -b branch1
    printf 'roadmap:\n- id: PMAT-1\n  title: one\n- id: PMAT-3\n  title: three\n' > "$repo1/docs/roadmaps/roadmap.yaml"
    git -C "$repo1" commit -q -am "branch1"
    
    git -C "$repo1" checkout -q main
    printf 'roadmap:\n- id: PMAT-1\n  title: one\n- id: PMAT-2\n  title: two\n' > "$repo1/docs/roadmaps/roadmap.yaml"
    git -C "$repo1" commit -q -am "main"
    
    git -C "$repo1" checkout -q branch1
    if git -C "$repo1" merge --no-commit --no-ff main >/dev/null 2>&1; then
        echo "FAIL: expected conflict without driver"
        exit 1
    fi
    git -C "$repo1" merge --abort
    
    if ! git -C "$repo1" -c merge.roadmap.driver="python3 $driver_script %O %A %B" merge --no-commit --no-ff main >/dev/null 2>&1; then
        echo "FAIL: expected successful merge with driver"
        exit 1
    fi
    
    if ! grep -q "PMAT-2" "$repo1/docs/roadmaps/roadmap.yaml" || ! grep -q "PMAT-3" "$repo1/docs/roadmaps/roadmap.yaml"; then
        echo "FAIL: merged file missing IDs"
        exit 1
    fi
    
    tmp_l2=$(mktemp)
    tmp_l3=$(mktemp)
    { grep -n "PMAT-2" "$repo1/docs/roadmaps/roadmap.yaml" | cut -d: -f1 ; } > "$tmp_l2"
    { grep -n "PMAT-3" "$repo1/docs/roadmaps/roadmap.yaml" | cut -d: -f1 ; } > "$tmp_l3"
    read -r line2 < "$tmp_l2"
    read -r line3 < "$tmp_l3"
    rm -f "$tmp_l2" "$tmp_l3"
    
    if [ "$line2" -gt "$line3" ]; then
        echo "FAIL: IDs not sorted"
        exit 1
    fi
    git -C "$repo1" merge --abort
    
    git -C "$repo1" checkout -q -b branch2 main
    printf 'roadmap:\n- id: PMAT-1\n  title: one\n- id: PMAT-2\n  title: two_edited\n' > "$repo1/docs/roadmaps/roadmap.yaml"
    git -C "$repo1" commit -q -am "branch2"
    
    git -C "$repo1" checkout -q main
    printf 'roadmap:\n- id: PMAT-1\n  title: one\n- id: PMAT-2\n  title: two_edited_differently\n' > "$repo1/docs/roadmaps/roadmap.yaml"
    git -C "$repo1" commit -q -am "main edit"
    
    git -C "$repo1" checkout -q branch2
    if git -C "$repo1" -c merge.roadmap.driver="python3 $driver_script %O %A %B" merge --no-commit --no-ff main >/dev/null 2>&1; then
        echo "FAIL: expected conflict for same-id different edit"
        exit 1
    fi
    git -C "$repo1" merge --abort
    
    echo "PASS selftest"
    exit 0
fi

declare -a gh_cmd=("gh" "pr" "list" "--json" "number,mergeStateStatus,autoMergeRequest,headRefName,isDraft")
if [ -n "$REPO" ]; then
    gh_cmd+=("--repo" "$REPO")
fi

gh_out=$(mktemp)
"${gh_cmd[@]}" > "$gh_out"

declare -a jq_args=("-r")
if [ "${#PRS[@]}" -gt 0 ]; then
    export PRS_JSON
    PRS_JSON=$(printf '%s\n' "${PRS[@]}" | jq -R . | jq -s .)
    jq_args+=('.[] | select(.mergeStateStatus == "DIRTY" and (env.PRS_JSON | fromjson | index(.number | tostring))) | [.number, .headRefName] | @tsv')
else
    jq_args+=('.[] | select(.mergeStateStatus == "DIRTY") | [.number, .headRefName] | @tsv')
fi

DRIVER_SCRIPT="$REPO_ROOT/scripts/lib/roadmap_merge.py"

tmp_td_wt=$(mktemp)
tmp_head_sha=$(mktemp)
tmp_conf=$(mktemp)
tmp_dirty=$(mktemp)

while read -r pr_number head_ref; do
    [ -z "$pr_number" ] && continue
    
    mktemp -d "${TMPDIR:-/tmp}/ci_resolve_wt.XXXXXX" > "$tmp_td_wt"
    read -r td < "$tmp_td_wt"
    [ -z "$td" ] && continue
    
    git worktree add -q "$td" "origin/$head_ref"
    
    git -C "$td" rev-parse HEAD > "$tmp_head_sha"
    read -r head_sha < "$tmp_head_sha"
    
    if [ "$APPLY" -eq 1 ]; then
        if git -C "$td" -c merge.roadmap.driver="python3 $DRIVER_SCRIPT %O %A %B" merge --no-edit origin/main >/dev/null 2>&1; then
            git -C "$td" commit --amend -q -m "merge origin/main (roadmap 3-way by id)"
            printf "git push origin HEAD:%s\n" "$head_ref"
        else
            { git -C "$td" diff --name-only --diff-filter=U | tr '\n' ' ' ; } > "$tmp_conf"
            read -r conflicts < "$tmp_conf" || true
            printf "conflict pr=%s files=%s\n" "$pr_number" "${conflicts:-}"
            git -C "$td" merge --abort || true
        fi
    else
        dirty_files=0
        if ! git -C "$td" merge --no-commit --no-ff origin/main >/dev/null 2>&1; then
            { git -C "$td" diff --name-only --diff-filter=U | wc -l | tr -d ' ' ; } > "$tmp_dirty"
            read -r dirty_files < "$tmp_dirty"
            git -C "$td" merge --abort || true
        fi
        printf "plan pr=%s head=%s dirty-files=%s\n" "$pr_number" "$head_sha" "$dirty_files"
    fi
    
    git worktree remove -f "${td:?}"
    
done < <(jq "${jq_args[@]}" < "$gh_out")

rm -f "$gh_out" "$tmp_td_wt" "$tmp_head_sha" "$tmp_conf" "$tmp_dirty"
