#!/usr/bin/env bash
# mutate_pr_dup_patches.sh - the mutation set for DUP-001 (#4536),
# scripts/check_pr_duplicate_patches.sh.
#
# Each mutant is a named single edit that breaks exactly one rule the guard
# states. It is KILLED when `check_pr_duplicate_patches.sh --self-test` goes RED
# on it; a SURVIVOR is a rule the guard states and nothing tests. Target: 100%.
#
# Three things that have fooled mutation runs here before, and what stops each:
#   * an edit that matched no text and "passed": every mutant's old text must
#     occur EXACTLY ONCE, and the mutated file must differ, or the run fails;
#   * a broken tree read as a killed mutant: the unmutated baseline must be
#     GREEN in the same copied tree first;
#   * a status read through a pipe: every rc below is read from the command.
# Mutants run one after another, never in parallel: a timed-out run would read
# as a kill, and that is the fail-open direction.
#
# Usage: mutate_pr_dup_patches.sh [--list]
# Exit:  0 every mutant killed; 1 a survivor, a stale mutant, or a red baseline.
set -euo pipefail
shopt -u patsub_replacement 2>/dev/null || true  # bash 5.2: `&` in a replacement is literal

HERE="$(cd "$(dirname "$0")" && pwd)"
TD=""
trap '[ -n "$TD" ] && rm -rf -- "${TD:?}"' EXIT

# id <TAB> old text <TAB> new text, all in check_pr_duplicate_patches.sh
mutants() {
    cat <<'TSV'
generated-paths-counted	if (path in GEN) return	if (0) return
witness-prefix-counted	if (index(path, PRE[i]) == 1) return	if (0) return
readme-block-counted	if (path == "README.md" && readme_generated()) return	if (0) return
readme-block-any-file	if (path == "README.md" && readme_generated()) return	if (readme_generated()) return
whitespace-hunks-counted	if (blank_only()) return	if (0) return
whitespace-not-stripped	s = substr(L[k], 2); gsub(/[[:space:]]/, "", s)	s = substr(L[k], 2)
every-hunk-is-whitespace	    return r == a	    return 1
landed-hunks-counted	if (!($1 in PID) || (PID[$1] in LANDED)) next	if (!($1 in PID)) next
landed-quoted-path-filtered	if LC_ALL=C grep -q '^"' "$w/paths"; then	if false; then
cumulative-diff-dropped	gitp diff "${DIFF_FLAGS[@]}" "$mb" "$head" -- >> "$w/raw"	true
commit-level-dropped	printf "%s\tcommit\t%s\t\t\n", a[1], a[2]	next
stack-subject-ignored	case ",${stk[$1]}," in *",${num[$2]},"*) return 0 ;; esac	:
stack-other-ignored	case ",${stk[$2]}," in *",${num[$1]},"*) return 0 ;; esac	:
stack-any-number	in *",${num[$2]},"*) return 0	in *,[0-9]*) return 0
stack-case-sensitive	[[:space:]]*$"; "i"))	[[:space:]]*$"; ""))
duplicate-exits-zero	[ -s "$rows" ] || return 0	return 0
same-sha-always	if ($5 == $9) SAME[pair]++	SAME[pair]++
same-sha-never	if ($5 == $9) SAME[pair]++	if (0) SAME[pair]++
detail-uncapped	if (SHOWN[pair] <= cap) {	if (1) {
summary-dropped	more, stack > sumf	more, stack > "/dev/null"
hunk-count-wrong	else { H[pair]++;	else { H[pair] += 2;
gh-failure-passes	die2 "cannot list open PRs (gh failed)"	exit 0
fetch-failure-passes	|| die2 "cannot fetch the open PR heads (git fetch failed)"	|| exit 0
no-pr-passes	            exit 2	            exit 0
event-pr-ignored	jq -r '.pull_request.number // empty' "$ev"	true
engine-failure-passes	[ "$rc" -eq 0 ] || die2 "the engine stopped (rc $rc)"	[ "$rc" -eq 0 ] || return 0
unlisted-pr-passes	[ "$s" -ge 0 ] || die2 "PR #$subj is not in the open-PR list"	[ "$s" -ge 0 ] || exit 0
all-mode-skips	for ((j = i + 1; j < ${#num[@]}; j++)); do pair "$i" "$j"; done	:
TSV
}

if [ "${1:-}" = --list ]; then mutants | cut -f1; exit 0; fi
[ "$#" -eq 0 ] || { echo "usage: $0 [--list]" >&2; exit 1; }

TD="$(mktemp -d)"
SRC="$HERE/check_pr_duplicate_patches.sh"
T="$TD/check_pr_duplicate_patches.sh"

cp "$SRC" "$T"
if ! bash "$T" --self-test > "$TD/base.log" 2>&1; then
    echo "FAIL: the UNMUTATED self-test is red in the copied tree -- no mutant can be scored"
    tail -n 5 "$TD/base.log"; exit 1
fi

killed=0 total=0 bad=0
src="$(cat "$SRC")"
while IFS=$'\t' read -r id old new; do
    total=$((total + 1))
    n="$(grep -oF -- "$old" "$SRC" | wc -l)"
    if [ "$n" -ne 1 ]; then
        echo "  STALE    [$id] old text occurs $n times"; bad=$((bad + 1)); continue
    fi
    printf '%s\n' "${src/"$old"/"$new"}" > "$T"
    if cmp -s "$SRC" "$T"; then
        echo "  STALE    [$id] the mutated file is unchanged"; bad=$((bad + 1)); continue
    fi
    rc=0
    timeout 600 bash "$T" --self-test > "$TD/$id.log" 2>&1 || rc=$?
    case "$rc" in
        0) echo "  SURVIVED [$id]"; bad=$((bad + 1)) ;;
        1) echo "  killed   [$id] $(grep -m1 'FAIL \[' "$TD/$id.log" | cut -c1-120)"
           killed=$((killed + 1)) ;;
        *) echo "  ERROR    [$id] self-test rc=$rc (124 = timed out); not counted as a kill"
           tail -n 3 "$TD/$id.log"; bad=$((bad + 1)) ;;
    esac
done < <(mutants)

if [ "$bad" -ne 0 ]; then
    echo "FAIL: DUP-001 mutation $killed/$total killed, $bad survived, stale or errored"; exit 1
fi
echo "PASS: DUP-001 mutation $killed/$total killed"
