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
# How: every commit and every hunk of the PR (commits in base...head minus
# merges, plus the PR's cumulative diff; minus any hunk base already gained
# since the PR forked, under whatever sha) is hashed with `git patch-id --stable`, which ignores the
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
    # A failed fetch is rc 2 like a failed gh read: under `set -e` it would exit with
    # git's own status (1 or 128), and rc 1 means "duplicate found" (quorum, 2026-09-27).
    if [ "$(git -C "$repo" rev-parse --is-shallow-repository)" = true ]; then
        git -C "$repo" fetch --quiet --no-tags --unshallow origin || {
            echo "DUP-001: cannot unshallow the checkout (git fetch failed) -- refusing to pass unread" >&2
            exit 2; }
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
    git -C "$repo" fetch --quiet --no-tags origin "${refspecs[@]}" || {
        echo "DUP-001: cannot fetch the open PR heads (git fetch failed) -- refusing to pass unread" >&2
        exit 2; }
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
    seq 1 40 > "$r/a.txt"; seq 1 40 > "$r/b.txt"; seq 1 300 > "$r/c.txt"
    printf 'x\n<!-- CONTRACT_COUNT_START -->10<!-- CONTRACT_COUNT_END -->\ny\n' > "$r/README.md"
    cp "$r/README.md" "$r/notes.md"
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
    # many: 25 separate hunks carried by two PRs -- more than the detail cap
    g checkout --quiet -b many1 main; sed -i '0~10s/$/ edited/' "$r/c.txt"; commit many
    g checkout --quiet -b many2 main; sed -i '0~10s/$/ edited/' "$r/c.txt"; commit "many again"
    # landedC: a second stale PR that ALSO still carries the since-landed fix
    g checkout --quiet -b landedC main; g cherry-pick --quiet "$fix" >/dev/null
    g commit --quiet --amend >/dev/null -m "also re-carried"
    sed -i 's/^3$/three/' "$r/b.txt"; commit "own work"
    # twostep: the same end state as A's line-5 hunk, reached in two commits --
    # only the PR's cumulative diff carries the finished hunk
    g checkout --quiet -b twostep main; sed -i 's/^5$/fiv/' "$r/a.txt"; commit "step 1"
    sed -i 's/^fiv$/five/' "$r/a.txt"; commit "step 2"
    # block-elsewhere: a START/END block outside README.md is NOT generated
    g checkout --quiet -b blk1 main; sed -i 's/START -->10</START -->11</' "$r/notes.md"; commit blk
    g checkout --quiet -b blk2 main; sed -i 's/START -->10</START -->11</' "$r/notes.md"; commit "blk again"
    # whitespace-only: identical whitespace hunks on two PRs are ignored
    g checkout --quiet -b ws1 main; sed -i 's/^17$/17 /' "$r/a.txt"; commit ws
    g checkout --quiet -b ws2 main; sed -i 's/^17$/17 /' "$r/a.txt"; commit "ws again"

    # row <expect-rc> <expect-level|-> <name> <base> <subject> <subject-body> <other> <other-body>
    #     [<output must match ERE|-> [<output must NOT match ERE>]]
    row() {
        local want="$1" level="$2" name="$3" base="$4" s="$5" sb="$6" o="$7" ob="$8"
        local must="${9:--}" mustnt="${10:-}" out rc=0
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
        if [ "$must" != - ] && ! printf '%s\n' "$out" | grep -Eq -- "$must"; then
            echo "  FAIL [$name] output lacks /$must/: $out"; fail=$((fail + 1)); return
        fi
        if [ -n "$mustnt" ] && printf '%s\n' "$out" | grep -Eq -- "$mustnt"; then
            echo "  FAIL [$name] output has /$mustnt/: $out"; fail=$((fail + 1)); return
        fi
        echo "  ok   [$name] rc=$rc"
    }
    echo "must-match:"
    row 1 commit "pure copy"                        main copy "" A ""
    row 1 commit "cherry-pick, different message"   main pick "" A ""
    row 1 hunk   "copy then edit (hunk kept)"       main edit "" A ""
    row 1 hunk   "squashed into a bigger commit"    main squash "" A ""
    row 1 hunk   "README prose outside a block"     main prose2 "" prose1 ""
    row 1 commit "stack declared by the WRONG pr#"  main stack "stacked-on: #9" A "" \
        'SUMMARY: PR #1 vs PR #2: 1 commit\(s\), .* 1 under the SAME sha.*stacked-on: #2'
    row 1 commit "copy is summarized, not called a stack" main pick "" A "" \
        'SUMMARY: PR #1 vs PR #2: 1 commit\(s\), 2 hunk\(s\) in 1 file\(s\)$' 'SAME sha'
    row 1 hunk   "detail capped, summary counts all" main many2 "" many1 "" \
        'SUMMARY: PR #1 vs PR #2: 1 commit\(s\), 30 hunk\(s\) in 1 file\(s\); 11 more row\(s\) not printed'
    row 1 hunk   "same end state in two commits"    main twostep "" A ""
    row 1 hunk   "START/END block outside README"   main blk2 "" blk1 ""
    echo "must-not-match:"
    row 0 - "identical generated regen"             main regen2 "" regen1 ""
    row 0 - "same edit, different file"             main elsewhere "" A ""
    row 0 - "unrelated work"                        main unrelated "" A ""
    row 0 - "declared stack (subject body)"         main stack "stacked-on: #2" A ""
    row 0 - "declared stack (other body)"           main A "" stack "Stacked-on: #1"
    row 0 - "fix already on base under other sha"   main2 landedA "" landedB ""
    row 0 - "two stale PRs both carry a landed fix" main2 landedA "" landedC ""
    row 0 - "whitespace-only hunk"                  main ws2 "" ws1 ""

    echo "output and wrapper:"
    # The cap: exactly DETAIL_PER_PAIR detail rows for a pair that has more.
    local n rc
    python3 -c '
import json,sys
json.dump({"base":"main","subject":{"number":1,"head":"many2","body":""},"others":[{"number":2,"head":"many1","body":""}]},open(sys.argv[1],"w"))
' "$td/spec.json"
    n="$(python3 "$ENGINE" check "$r" "$td/spec.json" | grep -c '^DUP-001: PR #1 ')" || true
    total=$((total + 1))
    if [ "$n" = 20 ]; then echo "  ok   [detail rows capped at 20] n=$n"
    else echo "  FAIL [detail rows capped at 20] n=$n"; fail=$((fail + 1)); fi
    # gh cannot list the open PRs: refuse (2), never pass unread (0) or blame the PR (1).
    mkdir -p "$td/bin"; printf '#!/bin/sh\nexit 1\n' > "$td/bin/gh"; chmod +x "$td/bin/gh"
    rc=0; (cd "$r" && PATH="$td/bin:$PATH" bash "$HERE/check_pr_duplicate_patches.sh" --pr 1) >/dev/null 2>&1 || rc=$?
    total=$((total + 1))
    if [ "$rc" = 2 ]; then echo "  ok   [gh fails -> rc 2] rc=$rc"
    else echo "  FAIL [gh fails -> rc 2] rc=$rc"; fail=$((fail + 1)); fi
    # gh answers but the fetch fails (the scratch repo has no `origin`): rc 2, never
    # rc 1 (a duplicate) and never 0.
    mkdir -p "$td/binf"
    printf '#!/bin/sh\necho %s\n' "'[{\"number\":1,\"headRefOid\":\"x\",\"baseRefName\":\"main\",\"body\":\"\"}]'" > "$td/binf/gh"
    chmod +x "$td/binf/gh"
    rc=0; (cd "$r" && PATH="$td/binf:$PATH" bash "$HERE/check_pr_duplicate_patches.sh" --pr 1) >/dev/null 2>&1 || rc=$?
    total=$((total + 1))
    if [ "$rc" = 2 ]; then echo "  ok   [git fetch fails -> rc 2] rc=$rc"
    else echo "  FAIL [git fetch fails -> rc 2] rc=$rc"; fail=$((fail + 1)); fi
    # No PR under test (push / merge_group): nothing to compare, and it says so.
    printf '{"ref":"refs/heads/main"}' > "$td/event.json"
    local out; rc=0
    out="$(cd "$r" && GITHUB_EVENT_PATH="$td/event.json" GITHUB_EVENT_NAME=push PATH="$td/bin:$PATH" \
           bash "$HERE/check_pr_duplicate_patches.sh" 2>&1)" || rc=$?
    total=$((total + 1))
    if [ "$rc" = 0 ] && [[ "$out" == *"no pull request under test (event: push)"* ]]; then
        echo "  ok   [no PR in event -> rc 0, says why] rc=$rc"
    else echo "  FAIL [no PR in event -> rc 0, says why] rc=$rc: $out"; fail=$((fail + 1)); fi
    # A pull_request event names the PR: it reaches the gh read (and so refuses here).
    printf '{"pull_request":{"number":7}}' > "$td/event.json"; rc=0
    (cd "$r" && GITHUB_EVENT_PATH="$td/event.json" PATH="$td/bin:$PATH" \
        bash "$HERE/check_pr_duplicate_patches.sh") >/dev/null 2>&1 || rc=$?
    total=$((total + 1))
    if [ "$rc" = 2 ]; then echo "  ok   [PR read from event -> checked] rc=$rc"
    else echo "  FAIL [PR read from event -> checked] rc=$rc"; fail=$((fail + 1)); fi

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
