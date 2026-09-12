#!/usr/bin/env bash
# ci_resolve_dirty.sh — replay origin/main into every DIRTY pull request through
# the roadmap.yaml 3-way-by-id merge driver (PMAT-3118; spec §6.4).
#
# A squash-merge from the queue rewrites all of docs/roadmaps/roadmap.yaml, so
# every stacked branch goes DIRTY on that one file. `scripts/lib/roadmap_merge.py`
# merges it by ENTRY ID instead of by line, which resolves the whole class. This
# script finds the DIRTY PRs and drives that merge.
#
#   bash scripts/ci_resolve_dirty.sh                 # plan: one line per DIRTY PR
#   bash scripts/ci_resolve_dirty.sh --apply         # merge and push
#   bash scripts/ci_resolve_dirty.sh --apply --no-push # merge only, kept in local branch
#   bash scripts/ci_resolve_dirty.sh --pr 123 --pr 124   # restrict to those PRs
#   bash scripts/ci_resolve_dirty.sh --list-only     # just the selection, no worktree
#   bash scripts/ci_resolve_dirty.sh --selftest      # hermetic case table, no gh
#
# THE DRIVER IS REGISTERED PER INVOCATION, never in the shared config:
# `git -c merge.roadmap.driver="python3 scripts/lib/roadmap_merge.py %O %A %B"`.
# A `git config` write would leak into every other worktree sharing the .git.
#
# Exit: 0 clean · 1 a selftest row failed · 2 usage/env.
set -euo pipefail

# bashrs disable-file=PERF002
# bashrs disable-file=BRS0021
# bashrs disable-file=SEC014
# bashrs disable-file=SC2086
# bashrs disable-file=SC2154

PROG="${0##*/}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DRIVER_SCRIPT="$REPO_ROOT/scripts/lib/roadmap_merge.py"
DRIVER_CONFIG_KEY="merge.roadmap.driver"

APPLY=0
NO_PUSH=0
SELFTEST=0
LIST_ONLY=0
REPO=""
declare -a PRS=()

usage() {
    printf 'usage: %s [--apply] [--no-push] [--list-only] [--pr N]... [--repo O/R]  or  %s --selftest\n' \
        "$PROG" "$PROG" >&2
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --apply) APPLY=1; shift ;;
        --no-push) NO_PUSH=1; shift ;;
        --list-only) LIST_ONLY=1; shift ;;
        --selftest|--self-test) SELFTEST=1; shift ;;
        --pr) [ $# -ge 2 ] || usage; PRS+=("$2"); shift 2 ;;
        --repo) [ $# -ge 2 ] || usage; REPO="$2"; shift 2 ;;
        --help|-h) usage ;;
        *) printf '%s: unknown arg: %s\n' "$PROG" "$1" >&2; usage ;;
    esac
done

# --------------------------------------------------------------------------
# The `gh` seam. CI_RESOLVE_DIRTY_PRS_JSON=<file> substitutes a canned
# `gh pr list --json …` payload for the live call, so the DIRTY/--pr filter is
# testable without a network call. --selftest additionally puts a failing `gh`
# shim on PATH, so a regression that reaches for gh is loud, not silent.
# --------------------------------------------------------------------------
pr_list_json() {
    if [ -n "${CI_RESOLVE_DIRTY_PRS_JSON:-}" ]; then
        cat -- "$CI_RESOLVE_DIRTY_PRS_JSON"
        return 0
    fi
    local -a gh_cmd=(gh pr list --limit 200 --json
        number,mergeStateStatus,autoMergeRequest,headRefName,isDraft)
    if [ -n "$REPO" ]; then
        gh_cmd+=(--repo "$REPO")
    fi
    "${gh_cmd[@]}"
}

# select_dirty -> "NUMBER<TAB>HEADREF" for every DIRTY PR, restricted to --pr
# when given. `.number` must be BOUND before it is used inside index(): in
# `$want | index(.number|tostring)` the `.` is $want (the array), which is why
# the first version died with `Cannot index array with string "number"`.
select_dirty() {
    local json want
    json=$(pr_list_json)
    if [ "${#PRS[@]}" -gt 0 ]; then
        want=$(printf '%s\n' "${PRS[@]}" | jq -R . | jq -sc .)
        printf '%s' "$json" | jq -r --argjson want "$want" '
            .[]
            | select(.mergeStateStatus == "DIRTY")
            | . as $p
            | select($want | index($p.number | tostring))
            | [$p.number, $p.headRefName] | @tsv'
    else
        printf '%s' "$json" | jq -r '
            .[]
            | select(.mergeStateStatus == "DIRTY")
            | [.number, .headRefName] | @tsv'
    fi
}

# --------------------------------------------------------------------------
# --selftest
# --------------------------------------------------------------------------
ST_N=0
ST_RED=0

st_row() { # st_row RC LABEL [detail...]
    local line
    ST_N=$((ST_N + 1))
    if [ "$1" -eq 0 ]; then
        printf 'PASS  row %-2s %s\n' "$ST_N" "$2"
    else
        printf 'FAIL  row %-2s %s\n' "$ST_N" "$2"
        ST_RED=$((ST_RED + 1))
        shift 2
        for line in "$@"; do printf '%s\n' "$line" | sed 's/^/        /'; done
    fi
}

# fixture_merge DEFAULTBRANCH — the whole driver story in a throwaway repo
# created with `git init -q -b main` UNDER the given init.defaultBranch. The
# explicit -b is the fix for the 128 (`A branch named 'main' already exists`)
# that `git init` + `git branch main` hit whenever init.defaultBranch=main.
fixture_merge() {
    local dflt=$1 repo="$td/repo-$dflt" rm_yaml bad=0 l2 l3
    git -c "init.defaultBranch=$dflt" init -q -b main \
        --template="$td/.empty-template" "$repo"
    git -C "$repo" config user.email test@example.com
    git -C "$repo" config user.name "ci_resolve_dirty selftest"

    mkdir -p "$repo/docs/roadmaps"
    rm_yaml="$repo/docs/roadmaps/roadmap.yaml"
    printf 'roadmap:\n- id: PMAT-1\n  title: one\n' > "$rm_yaml"
    printf 'docs/roadmaps/roadmap.yaml merge=roadmap\n' > "$repo/.gitattributes"
    git -C "$repo" add docs/roadmaps/roadmap.yaml .gitattributes
    git -C "$repo" commit -q -m base

    git -C "$repo" checkout -q -b branch1
    printf 'roadmap:\n- id: PMAT-1\n  title: one\n- id: PMAT-3\n  title: three\n' > "$rm_yaml"
    git -C "$repo" commit -q -am branch1
    git -C "$repo" checkout -q main
    printf 'roadmap:\n- id: PMAT-1\n  title: one\n- id: PMAT-2\n  title: two\n' > "$rm_yaml"
    git -C "$repo" commit -q -am main-append
    git -C "$repo" checkout -q branch1

    if git -C "$repo" merge --no-commit --no-ff main >/dev/null 2>&1; then
        printf 'no-driver merge did NOT conflict (the fixture proves nothing)\n'
        bad=1
    fi
    git -C "$repo" merge --abort >/dev/null 2>&1 || true

    if git -C "$repo" -c "$DRIVER_CONFIG_KEY=python3 $DRIVER_SCRIPT %O %A %B" \
            merge --no-commit --no-ff main >/dev/null 2>&1; then
        l2=$(grep -n 'PMAT-2' "$rm_yaml" | head -1 | cut -d: -f1)
        l3=$(grep -n 'PMAT-3' "$rm_yaml" | head -1 | cut -d: -f1)
        if [ -z "$l2" ] || [ -z "$l3" ]; then
            printf 'merged roadmap.yaml lost an id (PMAT-2=%s PMAT-3=%s)\n' "${l2:-none}" "${l3:-none}"
            bad=1
        elif [ "$l2" -gt "$l3" ]; then
            printf 'merged ids not ascending: PMAT-2 at %s after PMAT-3 at %s\n' "$l2" "$l3"
            bad=1
        fi
    else
        printf 'driver merge failed where it must succeed (append vs append)\n'
        bad=1
    fi
    git -C "$repo" merge --abort >/dev/null 2>&1 || true

    git -C "$repo" checkout -q -b branch2 main
    printf 'roadmap:\n- id: PMAT-1\n  title: one\n- id: PMAT-2\n  title: two_here\n' > "$rm_yaml"
    git -C "$repo" commit -q -am branch2
    git -C "$repo" checkout -q main
    printf 'roadmap:\n- id: PMAT-1\n  title: one\n- id: PMAT-2\n  title: two_there\n' > "$rm_yaml"
    git -C "$repo" commit -q -am main-edit
    git -C "$repo" checkout -q branch2
    if git -C "$repo" -c "$DRIVER_CONFIG_KEY=python3 $DRIVER_SCRIPT %O %A %B" \
            merge --no-commit --no-ff main >/dev/null 2>&1; then
        printf 'same-id different-edit merged silently (must conflict)\n'
        bad=1
    fi
    git -C "$repo" merge --abort >/dev/null 2>&1 || true
    return "$bad"
}

canned_prs() { # canned_prs FILE
    cat > "$1" <<'JSON'
[
  {"number": 101, "mergeStateStatus": "DIRTY",  "headRefName": "feat/one",  "isDraft": false, "autoMergeRequest": null},
  {"number": 102, "mergeStateStatus": "CLEAN",  "headRefName": "feat/two",  "isDraft": false, "autoMergeRequest": null},
  {"number": 103, "mergeStateStatus": "DIRTY",  "headRefName": "feat/three","isDraft": false, "autoMergeRequest": null}
]
JSON
}

fixture_apply_setup() { # fixture_apply_setup ID
    local t=$1
    local origin="$td/origin_$t.git"
    local repo="$td/repo_$t"
    git init -q --bare "$origin"
    git clone -q "$origin" "$repo"
    git -C "$repo" config user.name "Self Test"
    git -C "$repo" config user.email "test@example.com"
    git -C "$repo" checkout -q -b main
    mkdir -p "$repo/docs/roadmaps"
    printf 'roadmap:\n- id: PMAT-1\n' > "$repo/docs/roadmaps/roadmap.yaml"
    printf 'docs/roadmaps/roadmap.yaml merge=roadmap\n' > "$repo/.gitattributes"
    git -C "$repo" add docs/roadmaps/roadmap.yaml .gitattributes
    git -C "$repo" commit -q -m "initial"
    git -C "$repo" push -q origin main
    git -C "$repo" checkout -q -b feat/pr-104
    printf 'roadmap:\n- id: PMAT-1\n- id: PMAT-2\n' > "$repo/docs/roadmaps/roadmap.yaml"
    git -C "$repo" commit -q -am "pr commit"
    local pr_head
    pr_head=$(git -C "$repo" rev-parse HEAD)
    git -C "$repo" push -q origin feat/pr-104
    git -C "$repo" checkout -q main
    printf 'roadmap:\n- id: PMAT-1\n- id: PMAT-3\n' > "$repo/docs/roadmaps/roadmap.yaml"
    git -C "$repo" commit -q -am "main commit"
    local main_head
    main_head=$(git -C "$repo" rev-parse HEAD)
    git -C "$repo" push -q origin main
    printf '[{"number": 104, "mergeStateStatus": "DIRTY", "headRefName": "feat/pr-104", "isDraft": false, "autoMergeRequest": null}]\n' > "$repo/prs.json"
    printf '%s\n' "$origin" > "$td/fa_origin_$t"
    printf '%s\n' "$repo" > "$td/fa_repo_$t"
    printf '%s\n' "$pr_head" > "$td/fa_pr_head_$t"
    printf '%s\n' "$main_head" > "$td/fa_main_head_$t"
}

selftest() {
    local out detail
    td=$(mktemp -d "${TMPDIR:-/tmp}/ci_resolve_selftest.XXXXXX") || return 2
    cleanup() {
        local victim="${td:-}"
        case "$victim" in
            *ci_resolve_selftest.*)
                if [ -n "$victim" ] && [ "$victim" != "/" ]; then rm -rf -- "${victim:?}"; fi ;;
            *) return 0 ;;
        esac
    }
    trap cleanup EXIT
    export GIT_TERMINAL_PROMPT=0
    mkdir -p "$td/.empty-template" "$td/bin"

    # A `gh` that cannot succeed, first on PATH: --selftest must never reach
    # the network, and a regression that does is loud.
    GH_SHIM_MARKER="$td/gh-invoked"
    export GH_SHIM_MARKER
    cat > "$td/bin/gh" <<'SHIM'
#!/usr/bin/env bash
printf 'gh invoked: %s\n' "$*" >> "$GH_SHIM_MARKER"
exit 97
SHIM
    chmod +x "$td/bin/gh"
    PATH="$td/bin:$PATH"
    export PATH

    # rows 1-2: the driver fixture under BOTH init.defaultBranch values. The
    # 128 regression only reproduced under =main, so both are rows of record.
    # `set -e` is re-armed INSIDE the capture subshell on purpose: a bare
    # `fixture_merge … || st_row 1` runs the fixture in a tested context, where
    # errexit is suspended, so a git command that dies mid-fixture (the 128 this
    # phase fixes) would be swallowed and the row would still report PASS.
    local dflt frc
    for dflt in main master; do
        frc=0
        detail=$(set -e; fixture_merge "$dflt" 2>&1) || frc=$?
        if [ "$frc" -eq 0 ] && [ -z "$detail" ]; then
            st_row 0 "3-way-by-id driver fixture under init.defaultBranch=$dflt"
        else
            st_row 1 "3-way-by-id driver fixture under init.defaultBranch=$dflt" \
                "fixture rc=$frc" "$detail"
        fi
    done

    # rows 3-4: the --pr / DIRTY filter, on a canned `gh pr list` payload.
    canned_prs "$td/prs.json"
    local rc=0
    out=$(CI_RESOLVE_DIRTY_PRS_JSON="$td/prs.json" bash "$REPO_ROOT/scripts/$PROG" \
            --list-only --pr 103 2>&1) || rc=$?
    if [ "$rc" -eq 0 ] && [ "$out" = "$(printf '103\tfeat/three')" ]; then
        st_row 0 '--pr 103 selects exactly that DIRTY PR (canned gh payload)'
    else
        st_row 1 '--pr 103 selects exactly that DIRTY PR (canned gh payload)' "rc=$rc" "got: $out"
    fi

    rc=0
    out=$(CI_RESOLVE_DIRTY_PRS_JSON="$td/prs.json" bash "$REPO_ROOT/scripts/$PROG" \
            --list-only 2>&1) || rc=$?
    if [ "$rc" -eq 0 ] && [ "$out" = "$(printf '101\tfeat/one\n103\tfeat/three')" ]; then
        st_row 0 'no --pr selects both DIRTY PRs, never the CLEAN one'
    else
        st_row 1 'no --pr selects both DIRTY PRs, never the CLEAN one' "rc=$rc" "got: $out"
    fi

    # row a: --apply pushes
    fixture_apply_setup "a"
    local origin_a repo_a pr_head_a main_head_a merge_sha_a p1_a p2_a
    read -r origin_a < "$td/fa_origin_a"
    read -r repo_a < "$td/fa_repo_a"
    read -r pr_head_a < "$td/fa_pr_head_a"
    read -r main_head_a < "$td/fa_main_head_a"
    rc=0
    out=$(cd "$repo_a" && CI_RESOLVE_DIRTY_PRS_JSON="$repo_a/prs.json" bash "$REPO_ROOT/scripts/$PROG" --apply 2>&1) || rc=$?
    merge_sha_a=$(git -C "$origin_a" rev-parse feat/pr-104)
    p1_a=$(git -C "$origin_a" log -1 --format="%P" "$merge_sha_a" | awk '{print $1}')
    p2_a=$(git -C "$origin_a" log -1 --format="%P" "$merge_sha_a" | awk '{print $2}')
    if [ "$rc" -eq 0 ] && [ "$p1_a" = "$pr_head_a" ] && [ "$p2_a" = "$main_head_a" ] && \
            echo "$out" | grep -q "pushed pr=104 merge=$merge_sha_a onto=$pr_head_a branch=feat/pr-104"; then
        st_row 0 '--apply pushes to origin'
    else
        st_row 1 '--apply pushes to origin' "rc=$rc" "out=$out"
    fi

    # row b: --apply --no-push leaves resolve/<pr> branch
    fixture_apply_setup "b"
    local origin_b repo_b pr_head_b main_head_b local_sha_b remote_sha_b p1_b p2_b
    read -r origin_b < "$td/fa_origin_b"
    read -r repo_b < "$td/fa_repo_b"
    read -r pr_head_b < "$td/fa_pr_head_b"
    read -r main_head_b < "$td/fa_main_head_b"
    rc=0
    out=$(cd "$repo_b" && CI_RESOLVE_DIRTY_PRS_JSON="$repo_b/prs.json" bash "$REPO_ROOT/scripts/$PROG" --apply --no-push 2>&1) || rc=$?
    remote_sha_b=$(git -C "$origin_b" rev-parse feat/pr-104)
    local_sha_b=$(git -C "$repo_b" rev-parse resolve/104)
    p1_b=$(git -C "$repo_b" log -1 --format="%P" "$local_sha_b" | awk '{print $1}')
    p2_b=$(git -C "$repo_b" log -1 --format="%P" "$local_sha_b" | awk '{print $2}')
    if [ "$rc" -eq 0 ] && [ "$remote_sha_b" = "$pr_head_b" ] && [ "$p1_b" = "$pr_head_b" ] && [ "$p2_b" = "$main_head_b" ] && \
            echo "$out" | grep -q "merged pr=104 merge=$local_sha_b ref=resolve/104"; then
        st_row 0 '--apply --no-push leaves local branch and does not push'
    else
        st_row 1 '--apply --no-push leaves local branch and does not push' "rc=$rc" "out=$out" "remote_sha_b=$remote_sha_b"
    fi

    # row c: mutation - neutralising push makes apply fail
    fixture_apply_setup "c"
    local origin_c repo_c
    read -r origin_c < "$td/fa_origin_c"
    read -r repo_c < "$td/fa_repo_c"
    chmod -R a-w "$origin_c"
    rc=0
    out=$(cd "$repo_c" && CI_RESOLVE_DIRTY_PRS_JSON="$repo_c/prs.json" bash "$REPO_ROOT/scripts/$PROG" --apply 2>&1) || rc=$?
    chmod -R u+w "$origin_c"
    if [ "$rc" -ne 0 ] && echo "$out" | grep -q "push-failed pr=104"; then
        st_row 0 'apply fails when push fails (mutation)'
    else
        st_row 1 'apply fails when push fails (mutation)' "rc=$rc" "out=$out"
    fi

    # row 5: gh never ran in rows 3-4 — AND the shim that proves it does engage.
    if [ -e "$GH_SHIM_MARKER" ]; then
        st_row 1 'gh is never invoked under --selftest (PATH shim engages)' \
            "$(cat -- "$GH_SHIM_MARKER")"
    else
        gh --version >/dev/null 2>&1 || true
        if [ -e "$GH_SHIM_MARKER" ]; then
            st_row 0 'gh is never invoked under --selftest (PATH shim engages)'
        else
            st_row 1 'gh is never invoked under --selftest (PATH shim engages)' \
                'the PATH shim never ran: rows 3-4 proved nothing about gh'
        fi
        rm -f -- "$GH_SHIM_MARKER"
    fi

    printf '%s/%s rows, %s failed\n' "$((ST_N - ST_RED))" "$ST_N" "$ST_RED"
    [ "$ST_RED" -eq 0 ]
}

if [ "$SELFTEST" -eq 1 ]; then
    selftest
    exit $?
fi

if [ "$LIST_ONLY" -eq 1 ]; then
    select_dirty
    exit 0
fi

# --------------------------------------------------------------------------
# plan / apply
# --------------------------------------------------------------------------
resolve_one() { # resolve_one NUMBER HEADREF
    local pr_number=$1 head_ref=$2 td_wt head_sha conflicts dirty_files=0 merge_sha
    td_wt=$(mktemp -d "${TMPDIR:-/tmp}/ci_resolve_wt.XXXXXX")
    git worktree add -q "$td_wt" "origin/$head_ref"
    head_sha=$(git -C "$td_wt" rev-parse HEAD)

    if [ "$APPLY" -eq 1 ]; then
        if git -C "$td_wt" -c "$DRIVER_CONFIG_KEY=python3 $DRIVER_SCRIPT %O %A %B" \
                merge --no-edit -m "merge origin/main (roadmap 3-way by id)" \
                origin/main >/dev/null 2>&1; then
            merge_sha=$(git -C "$td_wt" rev-parse HEAD)
            if [ "$NO_PUSH" -eq 1 ]; then
                git branch -f "resolve/$pr_number" "$merge_sha"
                printf 'merged pr=%s merge=%s ref=resolve/%s — push with: git push origin resolve/%s:%s\n' \
                    "$pr_number" "$merge_sha" "$pr_number" "$pr_number" "$head_ref"
            else
                if git -C "$td_wt" push origin HEAD:"$head_ref" >/dev/null 2>&1; then
                    printf 'pushed pr=%s merge=%s onto=%s branch=%s\n' \
                        "$pr_number" "$merge_sha" "$head_sha" "$head_ref"
                else
                    printf 'push-failed pr=%s branch=%s\n' "$pr_number" "$head_ref"
                    return 1
                fi
            fi
        else
            conflicts=$(git -C "$td_wt" diff --name-only --diff-filter=U | tr '\n' ' ')
            printf 'conflict pr=%s files=%s\n' "$pr_number" "${conflicts:-unknown}"
            git -C "$td_wt" merge --abort >/dev/null 2>&1 || true
        fi
    else
        if ! git -C "$td_wt" -c "$DRIVER_CONFIG_KEY=python3 $DRIVER_SCRIPT %O %A %B" \
                merge --no-commit --no-ff origin/main >/dev/null 2>&1; then
            dirty_files=$(git -C "$td_wt" diff --name-only --diff-filter=U | wc -l | tr -d ' ')
            git -C "$td_wt" merge --abort >/dev/null 2>&1 || true
        fi
        printf 'plan pr=%s head=%s dirty-files=%s\n' "$pr_number" "$head_sha" "$dirty_files"
    fi

    git worktree remove -f "${td_wt:?}"
}

while IFS=$'\t' read -r pr_number head_ref; do
    [ -n "$pr_number" ] || continue
    resolve_one "$pr_number" "$head_ref"
done < <(select_dirty)
