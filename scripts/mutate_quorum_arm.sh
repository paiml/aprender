#!/usr/bin/env bash
# mutate_quorum_arm.sh - the mutation set for scripts/pr_review_quorum_arm.sh.
#
# PR-REVIEW-SKILL-002 v2 S13.10. The rule S6.4 applies to the receipt guard applies here
# with more force, not less: the receipt guard decides whether a HUMAN is shown a
# rejection, and this script decides whether a merge happens with nobody watching. A
# surviving mutant is a refusal path the arm script STATES and nothing TESTS - under
# autonomy, that is a merge nobody authorised.
#
# It is a SIBLING of scripts/mutate-guard.sh and not an extension of it, deliberately:
# the two sets mutate different files, run different bats files, and take different wall
# times, and merging them would put one 51-minute sweep in front of a 15-minute one for
# no gain. The MECHANICS are identical, down to the failure modes, because those were
# learned the expensive way and are not worth relearning:
#
#  1. A mutant that changed nothing reports a kill it never earned. Every mutation below
#     asserts the new text is present and the file actually differs before a test runs.
#  2. A mutant in a bare tempdir aborts on a fail-closed fixture lookup, so 0/N are
#     killed and the PROBE is wrong rather than the script. Each mutant gets a complete
#     tree and the baseline is proven GREEN in that same tree first.
#  3. A status read through a pipe is the last command's status. Every exit code here is
#     read from the command itself.
#
# The `drop` and `flip` sites are DERIVED by scanning for `refuse Q<n>`, never listed, so
# a refusal added to the arm script is mutated the next time this runs without anybody
# remembering. The `text` sites are listed and each must occur EXACTLY ONCE.
#
# EXCLUDED ON PURPOSE: the two `case` guards around `rm -rf` and the EXIT trap. They are
# destructive-op guards, not refusal branches; no receipt can reach them, so no fixture
# can kill them, and including them would put a permanent survivor in a score S13.10
# fixes at one.
#
# EXCLUDED FOR THE SAME REASON, and MEASURED rather than assumed: control 1's own
# `|| return 1`. Control 1 is synthesized inline, so its only failure mode is a defect in
# phase A - and every phase-A defect is already a `drop` or `flip` in this set. Under a
# single-mutation model no test can distinguish `|| true` there, and run 1 on 2026-08-31
# reported exactly that: `control-1-unarmed` SURVIVED. The arm script was restructured so
# both controls share ONE arming line (`run_all_quorum_controls || exit 1`), which the
# three seed tests DO reach; that line is `controls-unarmed` below and it is killed.
#
# USAGE
#   scripts/mutate_quorum_arm.sh                run the set, print the table
#   scripts/mutate_quorum_arm.sh --list         print the catalogue and exit
#   scripts/mutate_quorum_arm.sh --jobs N       parallel mutants (default: 12)
#   scripts/mutate_quorum_arm.sh --only <id>    one mutant (debugging)
#
# EXIT: 0 only when every mutant in the set was killed.

set -euo pipefail

PROG=${0##*/}
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$HERE/.." && pwd)
ARM_REL=scripts/pr_review_quorum_arm.sh
GUARD_REL=scripts/check_pr_review_receipt.sh
BATS_REL=tests/pr-review-quorum.bats
ARM="$ROOT/$ARM_REL"

JOBS=12
ONLY=''
LIST_ONLY=0
RUN_ONE=''

while [ "$#" -gt 0 ]; do
  case $1 in
    --list)  LIST_ONLY=1; shift ;;
    --jobs)  JOBS=${2:?--jobs needs a number}; shift 2 ;;
    --only)  ONLY=${2:?--only needs a mutant id}; shift 2 ;;
    --run-one) RUN_ONE=${2:?internal}; WORKDIR=${3:?internal}; shift 3 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "$PROG: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

for t in bats jq awk minisign check-jsonschema git; do
  command -v "$t" >/dev/null 2>&1 || {
    echo "$PROG: FAIL - $t is not on PATH. A mutation score that could not be measured" >&2
    echo "  must not be reported as one (S13.10: this number is a one, not a floor)." >&2
    exit 1; }
done
[ -f "$ARM" ] || { echo "$PROG: FAIL - no arm script at $ARM" >&2; exit 1; }

safe_rmtree() {
  cleanup_dir=${1:-}
  case "$cleanup_dir" in
    */prrev-qmutate.*) ;;
    *) return 0 ;;
  esac
  if [ -z "$cleanup_dir" ] || [ "$cleanup_dir" = "/" ]; then
    return 0
  fi
  rm -rf -- "$cleanup_dir"
  return 0
}

# ---------------------------------------------------------------------------
# THE CATALOGUE.  id <TAB> op <TAB> line <TAB> old <TAB> new
# ---------------------------------------------------------------------------
text_mutations() {
  cat <<'TSV'
tools-absent-is-a-skip	if [ -n "$MISSING_TOOLS" ]; then	if false; then
phase-a-never-runs	  phase_a "$1" "$2" || return 1	  phase_a "$1" "$2" || true
phase-b-never-runs	  phase_b "$1" "$2" || return 1	  phase_b "$1" "$2" || true
control-acceptance-ignored	  if phase_a "$dir" "$ctx"; then	  if phase_a "$dir" "$ctx" && false; then
control-class-unasserted	  if [ "$REFUSE_CLASS" != "$want" ]; then	  if false; then
control-reason-unasserted	    *"$want_reason"*) ;;	    *) ;;
control-fixture-absence-ignored	  if [ ! -f "$seed/receipt.intoto.jsonl" ]; then	  if false; then
controls-unarmed	run_all_quorum_controls || exit 1	run_all_quorum_controls || true
control-2-unarmed	  seeded_quorum_control single-vendor single-vendor Q5 "distinct vendor" || return 1	  seeded_quorum_control single-vendor single-vendor Q5 "distinct vendor" || true
refusal-exits-zero	exit "$REFUSAL_EXIT"	exit 0
explain-mode-ignored	  if [ "$MODE" = explain ]; then	  if false; then
idempotence-check-dropped	  if [ "$already" = true ]; then	  if false; then
mechanism-entry-lost	scripts/check_pr_review_receipt.sh	scripts/no-such-guard-here.sh
evidence-prefix-empty	EVIDENCE_PREFIX_TEMPLATE='evidence/pr-review/@@PR@@/'	EVIDENCE_PREFIX_TEMPLATE='@@TRUNCATE@@
evidence-prefix-never-matches	EVIDENCE_PREFIX_TEMPLATE='evidence/pr-review/@@PR@@/'	EVIDENCE_PREFIX_TEMPLATE='no-such-region/@@TRUNCATE@@
guard-shape-never-matches	      scripts/check_*.sh|*/scripts/check_*.sh|scripts/mutate*.sh|*/scripts/mutate*.sh)	      zzz-no-such-shape)
guard-shape-matches-everything	      scripts/check_*.sh|*/scripts/check_*.sh|scripts/mutate*.sh|*/scripts/mutate*.sh)	      *)
gate-second-spelling-dropped	  gate_conclusion=$(jq -r '.checks["ci / gate"] // .checks["gate"] // "absent"' "$ctx")	  gate_conclusion=$(jq -r '.checks["ci / gate"] // "absent"' "$ctx")
kill-switch-read-from-the-pr-tree	  if git -C "$REPO" cat-file -e "refs/remotes/origin/main:.github/pr-review-autonomy.disabled" 2>/dev/null; then	  if git -C "$REPO" cat-file -e "HEAD:.github/pr-review-autonomy.disabled" 2>/dev/null; then
receipt-guard-status-ignored	  if [ "$rc" -ne 0 ]; then	  if false; then
TSV
}

catalogue() {
  awk '
    /refuse Q[0-9]/ {
      n += 1
      printf "refuse-%02d-drop\tdrop\t%d\trefuse Q\ttrue Q\n", n, NR
      if (index($0, "|| return 1") == 0) {
        printf "MUTATION SITE %d HAS NO `|| return 1` TERMINATOR\n", NR > "/dev/stderr"
        bad = 1
      }
      printf "refuse-%02d-flip\tflip\t%d\t|| return 1\t&& return 1\n", n, NR
    }
    END {
      if (bad) { exit 3 }
      if (n < 45) {
        printf "ONLY %d refusal sites found; the arm script has always had more\n", n > "/dev/stderr"
        exit 3
      }
    }' "$ARM"

  while IFS=$'\t' read -r t_id t_old t_new; do
    [ -n "$t_id" ] || continue
    t_count=$(awk -v o="$t_old" 'index($0,o) { n += 1 } END { print n + 0 }' "$ARM")
    if [ "$t_count" -ne 1 ]; then
      echo "$PROG: FAIL - text mutation '$t_id' matches $t_count lines, not 1." >&2
      echo "  A catalogue entry that matches nothing mutates nothing and reports a kill" >&2
      echo "  it never earned. Fix the entry or the script; do not leave it stale." >&2
      exit 1
    fi
    t_ln=$(awk -v o="$t_old" 'index($0,o) { print NR; exit }' "$ARM")
    printf '%s\ttext\t%s\t%s\t%s\n' "$t_id" "$t_ln" "$t_old" "$t_new"
  done < <(text_mutations)
}

apply() {
  local ln=$1 old=$2 new=$3 src=$4 dst=$5 trunc=0
  case $new in *@@TRUNCATE@@) trunc=1; new=${new%@@TRUNCATE@@} ;; esac
  awk -v n="$ln" -v o="$old" -v w="$new" -v trunc="$trunc" '
    NR == n {
      i = index($0, o)
      if (i == 0) { print "SITE NOT PRESENT AT LINE " n > "/dev/stderr"; exit 3 }
      if (trunc == 1) { $0 = substr($0, 1, i - 1) w "'"'"'" }
      else            { $0 = substr($0, 1, i - 1) w substr($0, i + length(o)) }
    }
    { print }' "$src" > "$dst"
}

# A COMPLETE tree. The receipt guard is COPIED and never mutated: it is the delegate,
# it has its own 185-mutant set, and mutating it here would credit this set with kills
# that belong to that one.
build_tree() {
  local d=$1
  mkdir -p "$d/scripts" "$d/tests/fixtures"
  cp -a "$ROOT/schemas" "$d/schemas"
  cp -a "$ROOT/tests/fixtures/pr-review" "$d/tests/fixtures/pr-review"
  cp -a "$ROOT/$BATS_REL" "$d/$BATS_REL"
  cp -a "$ARM" "$d/$ARM_REL"
  cp -a "$ROOT/$GUARD_REL" "$d/$GUARD_REL"
}

if [ -n "$RUN_ONE" ]; then
  spec="$WORKDIR/specs/$RUN_ONE"
  [ -f "$spec" ] || { echo "no spec for $RUN_ONE" >&2; exit 1; }
  IFS=$'\t' read -r m_id m_op m_line m_old m_new < "$spec"
  tree="$WORKDIR/trees/$m_id"
  safe_rmtree "$tree"
  build_tree "$tree"
  set +e
  apply "$m_line" "$m_old" "$m_new" "$ARM" "$tree/$ARM_REL"
  apply_rc=$?
  set -e

  mutated_line=$(awk -v n="$m_line" 'NR == n { print; exit }' "$tree/$ARM_REL")
  status_note=ok
  if [ "$apply_rc" -ne 0 ]; then
    status_note='MUTATION-SITE-NOT-PRESENT-AT-THAT-LINE'
  elif cmp -s "$ARM" "$tree/$ARM_REL"; then
    status_note='MUTATION-DID-NOT-CHANGE-THE-FILE'
  elif ! grep -qF -- "${m_new%@@TRUNCATE@@}" <<<"$mutated_line"; then
    # HERESTRING, not a pipe: `producer | grep -q` returns 141 on SIGPIPE having
    # already MATCHED, which here would report a landed mutation as absent.
    status_note='MUTATED-TEXT-ABSENT-FROM-THE-LINE'
  fi
  if [ "$status_note" != ok ]; then
    printf '%s\t%s\t%s\tINVALID\t-\t-\t%s\t%s\n' \
      "$m_id" "$m_op" "$m_line" "$status_note" "$mutated_line" \
      > "$WORKDIR/results/$m_id"
    exit 0
  fi

  log="$WORKDIR/logs/$m_id.log"
  set +e
  ( cd "$tree" && bats "$BATS_REL" ) > "$log" 2>&1
  rc=$?
  set -e
  nfail=$(awk '/^not ok /  { n += 1 } END { print n + 0 }' "$log")
  first=$(awk '/^not ok /  { sub(/^not ok [0-9]+ /, ""); print; exit }' "$log")
  [ -n "$first" ] || first='-'
  if [ "$rc" -ne 0 ]; then verdict=KILLED; else verdict=SURVIVED; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$m_id" "$m_op" "$m_line" "$verdict" "$rc" "$nfail" "$first" "$mutated_line" \
    > "$WORKDIR/results/$m_id"
  if [ "$verdict" = KILLED ]; then safe_rmtree "$tree"; fi
  exit 0
fi

CAT=$(catalogue)
if [ "$LIST_ONLY" = 1 ]; then
  printf '%s\n' "$CAT"
  printf '%s mutants\n' "$(printf '%s\n' "$CAT" | grep -c .)" >&2
  exit 0
fi

WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/prrev-qmutate.XXXXXX")
case "$WORKDIR" in
  */prrev-qmutate.*) ;;
  *) echo "$PROG: FAIL - refusing to use scratch dir $WORKDIR" >&2; exit 1 ;;
esac
export WORKDIR
cleanup_workdir() { safe_rmtree "${WORKDIR:-}"; }
trap cleanup_workdir EXIT
mkdir -p "$WORKDIR/specs" "$WORKDIR/results" "$WORKDIR/logs" "$WORKDIR/trees"

echo "== baseline: the unmutated arm script in a mutant tree =="
build_tree "$WORKDIR/trees/baseline"
set +e
( cd "$WORKDIR/trees/baseline" && bats "$BATS_REL" ) > "$WORKDIR/logs/baseline.log" 2>&1
base_rc=$?
set -e
base_ok=$(awk '/^ok /     { n += 1 } END { print n + 0 }' "$WORKDIR/logs/baseline.log")
base_bad=$(awk '/^not ok / { n += 1 } END { print n + 0 }' "$WORKDIR/logs/baseline.log")
if [ "$base_rc" -ne 0 ]; then
  echo "$PROG: FAIL - the UNMUTATED arm script does not pass in a mutant tree" >&2
  echo "  ($base_ok ok, $base_bad not ok, exit $base_rc). Every kill below would be a" >&2
  echo "  kill of the harness, not of a mutant. Log: $WORKDIR/logs/baseline.log" >&2
  cp -a "$WORKDIR/logs/baseline.log" "${TMPDIR:-/tmp}/prrev-qmutate-baseline.log" || true
  exit 1
fi
printf 'baseline GREEN: %s tests, 0 failures\n\n' "$base_ok"
safe_rmtree "$WORKDIR/trees/baseline"

n_specs=0
while IFS= read -r row; do
  [ -n "$row" ] || continue
  id=${row%%$'\t'*}
  if [ -n "$ONLY" ] && [ "$id" != "$ONLY" ]; then continue; fi
  printf '%s\n' "$row" > "$WORKDIR/specs/$id"
  n_specs=$((n_specs + 1))
done <<EOF
$CAT
EOF
[ "$n_specs" -gt 0 ] || { echo "$PROG: FAIL - no mutants selected" >&2; exit 1; }
printf '== %s mutants, %s at a time ==\n' "$n_specs" "$JOBS"

find "$WORKDIR/specs" -maxdepth 1 -type f -printf '%f\n' | sort > "$WORKDIR/ids.txt"
set +e
xargs -a "$WORKDIR/ids.txt" -P "$JOBS" -I{} "$HERE/mutate_quorum_arm.sh" --run-one {} "$WORKDIR"
sweep_rc=$?
set -e
[ "$sweep_rc" -eq 0 ] || { echo "$PROG: FAIL - the sweep itself failed (exit $sweep_rc)" >&2; exit 1; }

REPORT=${PRREV_QUORUM_MUTATION_REPORT:-${TMPDIR:-/tmp}/prrev-quorum-mutation-report.tsv}
{
  printf 'id\top\tline\tverdict\trc\tfailing_tests\tkilled_by\tmutated_line\n'
  cat "$WORKDIR"/results/*
} > "$WORKDIR/report.tsv"

attempted=$(awk 'NR > 1 { n += 1 } END { print n + 0 }' "$WORKDIR/report.tsv")
killed=$(awk -F'\t'   'NR > 1 && $4 == "KILLED"   { n += 1 } END { print n + 0 }' "$WORKDIR/report.tsv")
survived=$(awk -F'\t' 'NR > 1 && $4 == "SURVIVED" { n += 1 } END { print n + 0 }' "$WORKDIR/report.tsv")
invalid=$(awk -F'\t'  'NR > 1 && $4 == "INVALID"  { n += 1 } END { print n + 0 }' "$WORKDIR/report.tsv")

awk -F'\t' 'NR > 1 { printf "%-28s %-5s L%-4s %-9s %s\n", $1, $2, $3, $4, $7 }' \
  "$WORKDIR/report.tsv" | sort

cp -a "$WORKDIR/report.tsv" "$REPORT"

printf '\n== quorum_arm_mutation_score ==\n'
printf 'attempted %s   killed %s   survived %s   invalid %s\n' \
  "$attempted" "$killed" "$survived" "$invalid"
printf 'report: %s\n' "$REPORT"

if [ "$attempted" -eq 0 ]; then
  echo "$PROG: FAIL - zero mutants attempted. A mutation set that matches nothing" >&2
  echo "  passes vacuously, which is what S3.D already calls DEGRADED, not clean." >&2
  exit 1
fi
if [ "$invalid" -ne 0 ]; then
  echo "$PROG: FAIL - $invalid mutation(s) did not change the arm script." >&2
  echo "  A mutant that edits nothing scores a kill it never earned." >&2
  exit 1
fi
if [ "$survived" -ne 0 ]; then
  echo "$PROG: FAIL - $survived mutant(s) SURVIVED." >&2
  echo "  Each is a refusal path the arm script states and no fixture tests: a receipt" >&2
  echo "  could break it and the table would stay green. Under S13 that is a merge" >&2
  echo "  nobody authorised. Logs kept under $WORKDIR/logs." >&2
  trap - EXIT
  exit 1
fi
printf 'quorum_arm_mutation_score = 100%% (%s/%s)\n' "$killed" "$attempted"
