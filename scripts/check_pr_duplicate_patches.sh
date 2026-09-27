#!/usr/bin/env bash
# check_pr_duplicate_patches.sh - DUP-001 (#4536): a PR may not carry a change
# that another OPEN PR already carries.
#
# Why: the 0.70 train carried the same fixes in two and three open PRs at once
# (a fix on its owner PR, copied into a fold, cherry-picked again into the
# release car). Each copy is reviewed, queued and CI'd separately, and when
# they diverge nobody knows which one is the fix. Target: 0 duplicated
# changes across open PRs.
#
# How: every commit and every hunk of the PR (commits in base...head, minus
# merges and minus commits already on base under another sha, plus the PR's
# cumulative diff) is hashed with `git patch-id --stable`, which ignores the
# commit message, line numbers and whitespace but not the file path. Any id
# shared with another open PR fails the check, naming both PRs, both shas,
# and the file + hunk. The engine is scripts/lib/pr_dup_patches.py.
#
# Not counted: generated outputs (Cargo.lock, roadmap.yaml, contracts census/
# nt/ttl, README hunks inside a generated <!-- X_START -->..<!-- X_END -->
# block) -- two honest regens produce identical bytes -- and whitespace-only
# hunks.
#
# Exemption: a PR body line `stacked-on: #N` exempts that pair, both ways.
# There is no other exemption; a fold closes its sources, it does not sit
# beside them.
#
# Usage:
#   check_pr_duplicate_patches.sh --pr N   check open PR N against every other open PR
#   check_pr_duplicate_patches.sh          same, N read from $GITHUB_EVENT_PATH
#                                          (pull_request events); with no PR under
#                                          test (push, merge_group, a local run) it
#                                          prints why and exits 0
#   check_pr_duplicate_patches.sh --self-test   run the must-match / must-not-match case table
#
# Exit: 0 no duplicate; 1 duplicate found (or a case-table row failed);
#       2 usage error, or a PR under test whose data could not be read.
set -euo pipefail

TD=""  # global: the EXIT trap runs after the function that set it returned
HERE="$(cd "$(dirname "$0")" && pwd)"
ENGINE="$HERE/lib/pr_dup_patches.py"

usage() {
    sed -n '/^# Usage:/,/^# Exit:/p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 2
}

# ---------------------------------------------------------------- real mode --

pr_from_event() {
    local ev="${GITHUB_EVENT_PATH:-}"
    [ -n "$ev" ] && [ -f "$ev" ] || return 0
    python3 -c 'import json,sys; print((json.load(open(sys.argv[1])).get("pull_request") or {}).get("number") or "")' "$ev"
}

check_pr() {
    local pr="$1" repo list base
    repo="$(git rev-parse --show-toplevel)"
    TD="$(mktemp -d)"; td="$TD"
    trap 'rm -rf -- "${TD:?}"' EXIT
    list="$td/open.json"
    if ! gh pr list --state open --limit 500 \
            --json number,headRefOid,baseRefName,body > "$list"; then
        echo "DUP-001: cannot list open PRs (gh failed) -- refusing to pass unread" >&2
        exit 2
    fi
    if [ "$(git -C "$repo" rev-parse --is-shallow-repository)" = true ]; then
        git -C "$repo" fetch --quiet --no-tags --unshallow origin
    fi
    # One fetch for every open PR head and every base branch they target.
    local -a refspecs
    mapfile -t refspecs < <(python3 -c '
import json,sys
prs=json.load(open(sys.argv[1]))
for b in sorted({p["baseRefName"] for p in prs}): print(f"+refs/heads/{b}:refs/dup001/base/{b}")
for p in prs: n = p["number"]; print(f"+refs/pull/{n}/head:refs/dup001/pr/{n}")
' "$list")
    if [ "${#refspecs[@]}" -eq 0 ]; then
        echo "DUP-001: no refs to fetch from the open-PR list -- refusing to pass unread" >&2
        exit 2
    fi
    git -C "$repo" fetch --quiet --no-tags origin "${refspecs[@]}"
    base="$(python3 -c '
import json,sys
pr=int(sys.argv[2])
m=[p for p in json.load(open(sys.argv[1])) if p["number"]==pr]
print(m[0]["baseRefName"] if m else "")
' "$list" "$pr")"
    if [ -z "$base" ]; then
        echo "DUP-001: PR #$pr is not in the open-PR list -- nothing to check it as" >&2
        exit 2
    fi
    python3 -c '
import json,sys
prs=json.load(open(sys.argv[1])); pr=int(sys.argv[2])
def d(p): return {"number":p["number"],"head":p["headRefOid"],"body":p.get("body") or "","base":"refs/dup001/base/"+p["baseRefName"]}
subj=[d(p) for p in prs if p["number"]==pr][0]
json.dump({"base":subj["base"],"subject":subj,"others":[d(p) for p in prs if p["number"]!=pr]},open(sys.argv[3],"w"))
' "$list" "$pr" "$td/spec.json"
    local n rc=0
    n="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["others"]))' "$td/spec.json")"
    python3 "$ENGINE" check "$repo" "$td/spec.json" || rc=$?
    case "$rc" in
        0) echo "PASS: DUP-001 PR #$pr shares no patch-id with the $n other open PR(s)" ;;
        1) echo "FAIL: DUP-001 PR #$pr carries changes another open PR already carries (above)." \
                "Drop the copy, close the PR it was folded from, or declare 'stacked-on: #N'." >&2 ;;
        *) echo "DUP-001: engine error rc=$rc" >&2; exit 2 ;;
    esac
    exit "$rc"
}

# ---------------------------------------------------------------- self-test --

# Each row builds branches in a scratch repo off `main`, runs the engine with
# the named subject/others, and compares exit status and the reported level.
# must-match rows expect exit 1; must-not-match rows expect exit 0.
self_test() {
    local r fail=0 total=0
    TD="$(mktemp -d)"; td="$TD"
    trap 'rm -rf -- "${TD:?}"' EXIT
    r="$td/repo"
    git init --quiet -b main "$r"
    g() { git -C "$r" -c user.name=t -c user.email=t@t -c commit.gpgsign=false \
              -c core.hooksPath=/dev/null "$@"; }
    # (git commit/cherry-pick print a summary even with --quiet on some paths)
    commit() { g add -A && g commit --quiet -m "$1"; }
    seq 1 40 > "$r/a.txt"; seq 1 40 > "$r/b.txt"
    printf 'x\n<!-- CONTRACT_COUNT_START -->10<!-- CONTRACT_COUNT_END -->\ny\n' > "$r/README.md"
    mkdir -p "$r/docs/roadmaps"; echo 'n: 1' > "$r/docs/roadmaps/roadmap.yaml"
    commit init

    # A: the original fix (two hunks in a.txt)
    g checkout --quiet -b A main
    sed -i -e 's/^5$/five/' -e 's/^30$/thirty/' "$r/a.txt"; commit "fix: a"
    local fix; fix="$(g rev-parse HEAD)"

    # copy: the same commit, same message, on another branch
    g checkout --quiet -b copy main; g cherry-pick --quiet "$fix" >/dev/null
    # pick: cherry-picked, message reworded, plus an unrelated later commit
    g checkout --quiet -b pick main; g cherry-pick --quiet "$fix" >/dev/null
    g commit --quiet --amend >/dev/null -m "chore: totally different words"
    sed -i 's/^20$/twenty/' "$r/b.txt"; commit "other work"
    # edit: copy then edit -- the hunk at line 5 kept, line 30 changed, b.txt added
    g checkout --quiet -b edit main
    sed -i -e 's/^5$/five/' -e 's/^30$/THIRTY/' "$r/a.txt"; sed -i 's/^9$/nine/' "$r/b.txt"
    commit "fix: a, edited"
    # squash: the fix squashed into one larger commit with other work
    g checkout --quiet -b squash main
    sed -i -e 's/^5$/five/' -e 's/^30$/thirty/' "$r/a.txt"; sed -i 's/^33$/x33/' "$r/b.txt"
    commit "big squash"
    # regen: an identical generated regen on two PRs (roadmap + README block)
    g checkout --quiet -b regen1 main
    echo 'n: 2' > "$r/docs/roadmaps/roadmap.yaml"
    sed -i 's/START -->10</START -->11</' "$r/README.md"; commit "regen"
    g checkout --quiet -b regen2 main
    echo 'n: 2' > "$r/docs/roadmaps/roadmap.yaml"
    sed -i 's/START -->10</START -->11</' "$r/README.md"
    sed -i 's/^12$/twelve/' "$r/b.txt"; commit "regen + own work"
    # readme-prose: the same README edit OUTSIDE a generated block IS a duplicate
    g checkout --quiet -b prose1 main; sed -i 's/^x$/intro/' "$r/README.md"; commit "prose"
    g checkout --quiet -b prose2 main; sed -i 's/^x$/intro/' "$r/README.md"; commit "prose again"
    # samefile-elsewhere: the same edit in a DIFFERENT file is not a duplicate
    g checkout --quiet -b elsewhere main; sed -i 's/^5$/five/' "$r/b.txt"; commit "b five"
    # unrelated work
    g checkout --quiet -b unrelated main; sed -i 's/^25$/x25/' "$r/b.txt"; commit "unrelated"
    # stack: B stacked on A carries A's commit by construction
    g checkout --quiet -b stack A; sed -i 's/^38$/x38/' "$r/b.txt"; commit "on top of A"
    # landed: A's fix already on main under another sha; a PR re-carrying it is
    # not a duplicate of a second PR that merged main
    g checkout --quiet -b main2 main; g cherry-pick --quiet "$fix" >/dev/null
    g commit --quiet --amend >/dev/null -m "landed"
    g checkout --quiet -b landedA main; g cherry-pick --quiet "$fix" >/dev/null
    g commit --quiet --amend >/dev/null -m "re-carried"
    g checkout --quiet -b landedB main2; sed -i 's/^2$/two/' "$r/b.txt"; commit "after landing"
    # whitespace-only: identical whitespace hunks on two PRs are ignored
    g checkout --quiet -b ws1 main; sed -i 's/^17$/17 /' "$r/a.txt"; commit ws
    g checkout --quiet -b ws2 main; sed -i 's/^17$/17 /' "$r/a.txt"; commit "ws again"

    # row <expect-rc> <expect-level|-> <name> <base> <subject> <subject-body> <other> <other-body>
    row() {
        local want="$1" level="$2" name="$3" base="$4" s="$5" sb="$6" o="$7" ob="$8" out rc=0
        total=$((total + 1))
        python3 -c '
import json,sys
b,s,sb,o,ob,f=sys.argv[1:]
json.dump({"base":b,"subject":{"number":1,"head":s,"body":sb},"others":[{"number":2,"head":o,"body":ob}]},open(f,"w"))
' "$base" "$s" "$sb" "$o" "$ob" "$td/spec.json"
        out="$(python3 "$ENGINE" check "$r" "$td/spec.json" 2>&1)" || rc=$?
        if [ "$rc" != "$want" ]; then
            echo "  FAIL [$name] rc=$rc want=$want: $out"; fail=$((fail + 1)); return
        fi
        if [ "$level" != - ] && ! printf '%s\n' "$out" | grep -q "PR #1 $level .*duplicates PR #2"; then
            echo "  FAIL [$name] no '$level' row naming both PRs: $out"; fail=$((fail + 1)); return
        fi
        echo "  ok   [$name] rc=$rc"
    }
    echo "must-match:"
    row 1 commit "pure copy"                        main copy "" A ""
    row 1 commit "cherry-pick, different message"   main pick "" A ""
    row 1 hunk   "copy then edit (hunk kept)"       main edit "" A ""
    row 1 hunk   "squashed into a bigger commit"    main squash "" A ""
    row 1 hunk   "README prose outside a block"     main prose2 "" prose1 ""
    row 1 commit "stack declared by the WRONG pr#"  main stack "stacked-on: #9" A ""
    echo "must-not-match:"
    row 0 - "identical generated regen"             main regen2 "" regen1 ""
    row 0 - "same edit, different file"             main elsewhere "" A ""
    row 0 - "unrelated work"                        main unrelated "" A ""
    row 0 - "declared stack (subject body)"         main stack "stacked-on: #2" A ""
    row 0 - "declared stack (other body)"           main A "" stack "Stacked-on: #1"
    row 0 - "fix already on base under other sha"   main2 landedA "" landedB ""
    row 0 - "whitespace-only hunk"                  main ws2 "" ws1 ""

    if [ "$fail" -ne 0 ]; then
        echo "FAIL: DUP-001 self-test $fail of $total row(s) failed"; exit 1
    fi
    echo "PASS: DUP-001 self-test $total/$total rows"
}

case "${1:-}" in
    --self-test) [ "$#" -eq 1 ] || usage; self_test ;;
    --pr) [ "$#" -eq 2 ] || usage
          case "$2" in ''|*[!0-9]*) usage ;; esac
          check_pr "$2" ;;
    -h|--help) usage ;;
    "") pr="$(pr_from_event)"
        if [ -z "$pr" ]; then
            echo "DUP-001: no pull request under test (event: ${GITHUB_EVENT_NAME:-none}); nothing to compare"
            exit 0
        fi
        check_pr "$pr" ;;
    *) usage ;;
esac
