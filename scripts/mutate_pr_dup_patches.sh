#!/usr/bin/env bash
# mutate_pr_dup_patches.sh - the mutation set for DUP-001 (#4536):
# scripts/check_pr_duplicate_patches.sh and its engine scripts/lib/pr_dup_patches.py.
#
# Each mutant is a named single-line edit that breaks exactly one rule the guard
# states. It is KILLED when `check_pr_duplicate_patches.sh --self-test` goes RED on
# it; a SURVIVOR is a rule the guard states and nothing tests. Target: 100%.
#
# Three things that have fooled mutation runs here before, and what stops each:
#   * an edit that matched no text and "passed": every mutant's old text must occur
#     EXACTLY ONCE, and the mutated file must differ, or the run fails;
#   * a broken tree read as a killed mutant: the unmutated baseline must be GREEN
#     in the same copied tree first;
#   * a status read through a pipe: every rc below is read from the command itself.
#
# Usage: mutate_pr_dup_patches.sh [--list]
# Exit:  0 every mutant killed; 1 a survivor, a stale mutant, or a red baseline.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TD=""
trap '[ -n "$TD" ] && rm -rf -- "${TD:?}"' EXIT

# id <TAB> file (under scripts/) <TAB> old text <TAB> new text
mutants() {
    cat <<'TSV'
generated-paths-counted	lib/pr_dup_patches.py	            if path in GENERATED_PATHS:	            if False:
readme-block-counted	lib/pr_dup_patches.py	(path == README and gen.hunk(old_rev, new_rev, h))	(False)
readme-block-any-file	lib/pr_dup_patches.py	(path == README and gen.hunk(old_rev, new_rev, h))	(gen.hunk(old_rev, new_rev, h))
whitespace-hunks-counted	lib/pr_dup_patches.py	                if blank_only(h) or	                if False or
every-hunk-is-whitespace	lib/pr_dup_patches.py	    return side("-") == side("+")	    return True
landed-hunks-counted	lib/pr_dup_patches.py	        if pid is None or pid in landed:	        if pid is None:
cumulative-diff-dropped	lib/pr_dup_patches.py	    sources.append(("cumulative", mb, head,	    (("cumulative", mb, head,
commit-level-dropped	lib/pr_dup_patches.py	            out.setdefault(pid, []).append(("commit", label, "", ""))	            pass
stack-subject-ignored	lib/pr_dup_patches.py	        if o["number"] in mine_stack or subject["number"] in stacked_on(o.get("body")):	        if subject["number"] in stacked_on(o.get("body")):
stack-other-ignored	lib/pr_dup_patches.py	        if o["number"] in mine_stack or subject["number"] in stacked_on(o.get("body")):	        if o["number"] in mine_stack:
stack-any-number	lib/pr_dup_patches.py	        if o["number"] in mine_stack or subject["number"] in stacked_on(o.get("body")):	        if mine_stack or stacked_on(o.get("body")):
stack-case-sensitive	lib/pr_dup_patches.py	STACKED_ON = re.compile(r"^\s*stacked-on:\s*#(\d+)\s*$", re.I | re.M)	STACKED_ON = re.compile(r"^\s*stacked-on:\s*#(\d+)\s*$", re.M)
duplicate-exits-zero	lib/pr_dup_patches.py	    return 1 if rows else 0	    return 0
same-sha-always	lib/pr_dup_patches.py	            if r["sha"] == r["other_sha"]:	            if True:
same-sha-never	lib/pr_dup_patches.py	            if r["sha"] == r["other_sha"]:	            if False:
detail-uncapped	lib/pr_dup_patches.py	        if n > per_pair:	        if False:
summary-dropped	lib/pr_dup_patches.py	    for other, s in sorted(summarize(rows).items()):	    for other, s in []:
hunk-count-wrong	lib/pr_dup_patches.py	            s["hunks"] += 1	            s["hunks"] += 2
gh-failure-passes	check_pr_duplicate_patches.sh	        echo "DUP-001: cannot list open PRs (gh failed) -- refusing to pass unread" >&2	        echo "DUP-001: cannot list open PRs (gh failed)" >&2; exit 0
no-pr-fails	check_pr_duplicate_patches.sh	            exit 0	            exit 1
event-pr-ignored	check_pr_duplicate_patches.sh	    python3 -c 'import json,sys; print((json.load(open(sys.argv[1])).get("pull_request") or {}).get("number") or "")' "$ev"	    echo ""
TSV
}

if [ "${1:-}" = --list ]; then mutants | cut -f1; exit 0; fi
[ "$#" -eq 0 ] || { echo "usage: $0 [--list]" >&2; exit 1; }

TD="$(mktemp -d)"
fresh_tree() {
    rm -rf -- "${TD:?}/t"
    mkdir -p "$TD/t/lib"
    cp "$HERE/check_pr_duplicate_patches.sh" "$TD/t/"
    cp "$HERE/lib/pr_dup_patches.py" "$TD/t/lib/"
}

fresh_tree
if ! bash "$TD/t/check_pr_duplicate_patches.sh" --self-test >"$TD/base.log" 2>&1; then
    echo "FAIL: the UNMUTATED self-test is red in the copied tree -- no mutant can be scored"
    tail -5 "$TD/base.log"; exit 1
fi

killed=0 total=0 bad=0
while IFS=$'\t' read -r id file old new; do
    total=$((total + 1))
    fresh_tree
    f="$TD/t/$file"
    rc=0
    python3 - "$f" "$old" "$new" <<'PY' || rc=$?
import sys
p, old, new = sys.argv[1:]
s = open(p).read()
n = s.count(old)
if n != 1:
    print(f"stale: old text occurs {n} times"); sys.exit(3)
t = s.replace(old, new)
if t == s:
    print("stale: mutation changed nothing"); sys.exit(3)
open(p, "w").write(t)
PY
    if [ "$rc" -ne 0 ]; then
        echo "  STALE    $id"; bad=$((bad + 1)); continue
    fi
    rc=0
    bash "$TD/t/check_pr_duplicate_patches.sh" --self-test >"$TD/m.log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then
        killed=$((killed + 1)); echo "  killed   $id ($(grep -m1 -o 'FAIL \[[^]]*\]' "$TD/m.log" || echo "rc=$rc"))"
    else
        echo "  SURVIVED $id"; bad=$((bad + 1))
    fi
done < <(mutants)

echo "DUP-001 mutation score: $killed/$total killed"
[ "$bad" -eq 0 ] && [ "$killed" -eq "$total" ]
