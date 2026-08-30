#!/usr/bin/env bash
# mutate-guard.sh - the mutation set for scripts/check_pr_review_receipt.sh.
#
# PR-REVIEW-SKILL-002 v2 S6.4: "Mechanically flip each validation branch and drop each
# required-field check. Target: 100% kill." S8 records `guard_mutation_score` as 100%
# with "ratchet rule: zero - no ratchet, it is a one". This is the ONE place S7's
# narrowness does not apply, because the guard is what every other verdict rests on.
#
# A surviving mutant is not a scoring detail. It is a validation rule the guard STATES
# and nothing TESTS: the receipt could break that rule and the fixture table would stay
# green. That is the same defect class as a pass-grep that cannot fail.
#
# WHAT A MUTANT IS
#
#   drop  <site>   the rejection at that site never fires. The guard becomes permissive
#                  on exactly one rule. Killed by a case that must be RED going GREEN.
#   flip  <site>   the branch's sense is inverted: `|| return 1` becomes `&& return 1`,
#                  so a receipt that SATISFIES the rule is rejected and one that breaks
#                  it is accepted. Killed by a RED case going GREEN or, where the branch
#                  has a valid direction, by a discrimination row going RED.
#   text  <site>   a named single-line edit to the control machinery, which is not a
#                  uniform `reject` site and cannot be derived by scanning.
#
# The `drop` and `flip` sites are DERIVED by scanning the guard for `reject B<n>`, never
# listed here, so a rule added to the guard is mutated the next time this runs without
# anyone remembering to add it. The `text` sites are listed, and each is asserted to
# occur EXACTLY ONCE in the guard - a stale entry fails the run instead of silently
# mutating nothing.
#
# THREE WAYS THIS HAS BEEN FOOLED BEFORE, AND WHAT STOPS EACH
#
#  1. A mutant that changed nothing and reported the suite "34 passed" - the target had
#     been reformatted and the edit matched no text. Every mutation here ASSERTS the new
#     text is present and the file actually differs before a single test runs.
#  2. Mutants in a bare tempdir: the guard's fail-closed positive-control lookup aborted
#     every run before it validated anything, so 0/15 were killed and the probe, not the
#     guard, was wrong. Each mutant gets a COMPLETE tree - guard, schemas, fixtures,
#     bats file - and the baseline is proven GREEN in that same tree first.
#  3. Reading a status through a pipe. Every exit code below is read from the command
#     itself, never from the tail of a pipeline.
#
# EXCLUDED FROM THE SET, ON PURPOSE: the two `case` guards around `rm -rf` (the scratch
# directory and the fixture-repo destination) and the EXIT trap. They are destructive-op
# guards, not validation branches: no receipt can reach them, so no fixture can kill
# them, and including them would put a permanent survivor in a score S8 fixes at one.
#
# USAGE
#   scripts/mutate-guard.sh                 run the whole set, print the table
#   scripts/mutate-guard.sh --list          print the catalogue and exit
#   scripts/mutate-guard.sh --jobs N        parallel mutants (default: 12)
#   scripts/mutate-guard.sh --only <id>     run one mutant (debugging)
#
# EXIT: 0 only when every mutant in the set was killed.

set -euo pipefail

PROG=${0##*/}
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$HERE/.." && pwd)
GUARD_REL=scripts/check_pr_review_receipt.sh
BATS_REL=tests/pr-review.bats
SCAN_REL=scripts/pr_review_duplication_scan.sh
GUARD="$ROOT/$GUARD_REL"

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
    -h|--help) sed -n '2,50p' "$0"; exit 0 ;;
    *) echo "$PROG: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

for t in bats jq awk minisign check-jsonschema git; do
  command -v "$t" >/dev/null 2>&1 || {
    echo "$PROG: FAIL - $t is not on PATH. A mutation score that could not be measured" >&2
    echo "  must not be reported as one (S8: guard_mutation_score is a one, not a floor)." >&2
    exit 1; }
done
[ -f "$GUARD" ] || { echo "$PROG: FAIL - no guard at $GUARD" >&2; exit 1; }

# ---------------------------------------------------------------------------
# safe_rmtree <path>
#
# Every rm -rf in this file goes through here. It expands into rm -rf, so it is gated
# twice: the path must still look like the scratch directory this run created, and must
# be neither empty nor the root. Returns 0 whatever happens, so a trap body calling it
# cannot rewrite the script's own exit status.
# ---------------------------------------------------------------------------
safe_rmtree() {
  cleanup_dir=${1:-}
  case "$cleanup_dir" in
    */prrev-mutate.*) ;;
    *) return 0 ;;
  esac
  if [ -z "$cleanup_dir" ] || [ "$cleanup_dir" = "/" ]; then
    return 0
  fi
  rm -rf -- "$cleanup_dir"
  return 0
}

# ---------------------------------------------------------------------------
# THE CATALOGUE
#
# Emitted as TSV on stdout:  id <TAB> op <TAB> line <TAB> old <TAB> new
# `old` and `new` are the exact substrings swapped ON THAT LINE. Deriving the line
# numbers on every run is what keeps the set honest: it cannot fall behind the guard.
# ---------------------------------------------------------------------------

# Named single-line edits to the control machinery. Each is  <old> <TAB> <new>  and
# each must match exactly one line of the guard.
#
# NO BACKSLASH MAY APPEAR IN AN ANCHOR. The anchors reach awk through `-v`, which
# expands escape sequences in the VALUE: an anchor written `CRUX_SURFACE_RE='(#\[arg`
# arrives as `CRUX_SURFACE_RE='(#[arg` and matches nothing. That fails loudly here
# ("matches 0 lines, not 1") rather than silently mutating nothing, which is the
# only reason it is a nuisance and not a survivor. Anchor on the part of the
# assignment before the first backslash.
text_mutations() {
  cat <<'TSV'
tools-absent-is-a-skip	if [ -n "$MISSING_TOOLS" ]; then	if false; then
repo-fallback-dropped	if [ -z "$REPO" ]; then	if false; then
control-acceptance-ignored	  if validate_receipt "$dir"; then	  if validate_receipt "$dir" && false; then
control-class-unasserted	  if [ "$REJECT_CLASS" != "$want" ]; then	  if false; then
control-reason-unasserted	    *"$want_reason"*) ;;	    *) ;;
control-fixture-absence-ignored	  if [ ! -f "$seed/receipt.intoto.jsonl" ]; then	  if false; then
control-1-unarmed	"$PC1" || exit 1	"$PC1" || true
control-2-unarmed	seeded_control self-review     self-review     B2 "reviewer_actor.id ="        || exit 1	seeded_control self-review     self-review     B2 "reviewer_actor.id ="        || true
control-3-unarmed	seeded_control findings-digest findings-digest B1 "findings_ref.sha256"        || exit 1	seeded_control findings-digest findings-digest B1 "findings_ref.sha256"        || true
control-4-unarmed	seeded_control cost-missing    cost-missing    B1 "predicate.cost must carry"  || exit 1	seeded_control cost-missing    cost-missing    B1 "predicate.cost must carry"  || true
zero-receipts-is-a-pass	if [ "$#" -eq 0 ]; then	if false; then
batch-status-not-accumulated	    RC=1	    RC=$RC
final-status-always-zero	exit "$RC"	exit 0
path-predicate-verdict-not-propagated	match_cuda_path    "${2?--match-path needs an argument}";        exit $?	match_cuda_path    "${2?--match-path needs an argument}";        exit 1
message-predicate-verdict-not-propagated	match_cuda_message "${2?--match-message needs an argument}";     exit $?	match_cuda_message "${2?--match-message needs an argument}";     exit 1
comparative-predicate-verdict-not-propagated	match_comparative  "${2?--match-comparative needs an argument}"; exit $?	match_comparative  "${2?--match-comparative needs an argument}"; exit 1
cuda-path-regex-never-matches	CUDA_PATH_RE='(^crates/aprender-gpu/)	CUDA_PATH_RE='(^$)@@TRUNCATE@@
cuda-path-regex-matches-everything	CUDA_PATH_RE='(^crates/aprender-gpu/)	CUDA_PATH_RE='(.)@@TRUNCATE@@
cuda-message-regex-never-matches	CUDA_MSG_RE='(sm_[0-9]+)	CUDA_MSG_RE='(^$)@@TRUNCATE@@
cuda-message-regex-matches-everything	CUDA_MSG_RE='(sm_[0-9]+)	CUDA_MSG_RE='(.)@@TRUNCATE@@
shipped-surface-predicate-verdict-not-propagated	match_shipped_surface "${2?--match-shipped-surface needs an argument}"; exit $?	match_shipped_surface "${2?--match-shipped-surface needs an argument}"; exit 1
crux-surface-predicate-verdict-not-propagated	match_crux_surface    "${2?--match-crux-surface needs an argument}";    exit $?	match_crux_surface    "${2?--match-crux-surface needs an argument}";    exit 1
mutation-trigger-predicate-verdict-not-propagated	match_mutation_trigger "${2?--match-mutation-trigger needs an argument}"; exit $?	match_mutation_trigger "${2?--match-mutation-trigger needs an argument}"; exit 1
target-predicate-verdict-not-propagated	match_target       "${2?--match-target needs an argument}";      exit $?	match_target       "${2?--match-target needs an argument}";      exit 1
rs-published-predicate-verdict-not-propagated	match_rs_published "${2?--match-rs-published needs an argument}"; exit $?	match_rs_published "${2?--match-rs-published needs an argument}"; exit 1
rs-published-regex-never-matches	RS_PUBLISHED_RE='(println!	RS_PUBLISHED_RE='(^$)@@TRUNCATE@@
rs-published-regex-matches-everything	RS_PUBLISHED_RE='(println!	RS_PUBLISHED_RE='(.)@@TRUNCATE@@
rs-line-test-not-applied	  case "$1" in *.rs) match_rs_published "$2" || return 1 ;; esac	  case "$1" in *.rs) true ;; esac
docs-prose-back-in-b4-scope	src/*.rs|book/*.md) return 0 ;;	src/*.rs|book/*.md|docs/*.md) return 0 ;;
book-removed-from-b4-scope	crates/*/src/*.rs|src/*.rs|book/*.md) return 0 ;;	crates/*/src/*.rs|src/*.rs) return 0 ;;
comparative-competitor-list-never-matches	COMPETITOR_RE='(ollama	COMPETITOR_RE='(^$)@@TRUNCATE@@
comparative-competitor-list-matches-everything	COMPETITOR_RE='(ollama	COMPETITOR_RE='(.)@@TRUNCATE@@
comparative-gap-bound-unbounded	){0,5}	){0,99}
comparative-left-boundary-dropped	RATIO_LEFT_RE='(^|	RATIO_LEFT_RE='(^|.)@@TRUNCATE@@
comparative-mult-sign-ascii-only	MULT_RE='(x|	MULT_RE='(x)@@TRUNCATE@@
target-suppressor-matches-everything	TARGET_RE='(	TARGET_RE='(.)@@TRUNCATE@@
target-suppressor-never-matches	TARGET_RE='(	TARGET_RE='(^$)@@TRUNCATE@@
crux-surface-regex-never-matches	CRUX_SURFACE_RE='(#	CRUX_SURFACE_RE='(^$)@@TRUNCATE@@
crux-surface-regex-matches-everything	CRUX_SURFACE_RE='(#	CRUX_SURFACE_RE='(.)@@TRUNCATE@@
mutation-trigger-regex-never-matches	MUTATION_TRIGGER_RE='(^|/)scripts	MUTATION_TRIGGER_RE='(^$)@@TRUNCATE@@
mutation-trigger-regex-matches-everything	MUTATION_TRIGGER_RE='(^|/)scripts	MUTATION_TRIGGER_RE='(.)@@TRUNCATE@@
self-review-misclassified-B1	reject B2 "reviewer_actor.id = author_actor.id	reject B1 "reviewer_actor.id = author_actor.id
comparator-misclassified-B1	reject B4 "$claim"	reject B1 "$claim"
stale-index-misclassified-B1	reject B6 "index_commit $idx is not an ancestor	reject B1 "index_commit $idx is not an ancestor
TSV
}

catalogue() {
  # --- derived: one drop and one flip per `reject B<n>` site ----------------
  awk '
    /reject B[0-9]/ {
      n += 1
      printf "reject-%02d-drop\tdrop\t%d\treject B\ttrue B\n", n, NR
      if (index($0, "|| return 1") == 0) {
        printf "MUTATION SITE %d HAS NO `|| return 1` TERMINATOR\n", NR > "/dev/stderr"
        bad = 1
      }
      printf "reject-%02d-flip\tflip\t%d\t|| return 1\t&& return 1\n", n, NR
    }
    END {
      if (bad) { exit 3 }
      if (n < 55) {
        printf "ONLY %d reject sites found; the guard has always had more\n", n > "/dev/stderr"
        exit 3
      }
    }' "$GUARD"

  # --- named: the control machinery ----------------------------------------
  # Plain variables, not `local`: bashrs mis-parses the awk block above as leaving this
  # function, and reports SC2168 on a `local` that is plainly inside one. The false
  # positive is not worth arguing with - catalogue() runs once, before anything else.
  while IFS=$'\t' read -r t_id t_old t_new; do
    [ -n "$t_id" ] || continue
    t_count=$(awk -v o="$t_old" 'index($0,o) { n += 1 } END { print n + 0 }' "$GUARD")
    if [ "$t_count" -ne 1 ]; then
      echo "$PROG: FAIL - text mutation '$t_id' matches $t_count lines of the guard, not 1." >&2
      echo "  A catalogue entry that matches nothing mutates nothing and reports a kill" >&2
      echo "  it never earned. Fix the entry or the guard; do not leave it stale." >&2
      exit 1
    fi
    t_ln=$(awk -v o="$t_old" 'index($0,o) { print NR; exit }' "$GUARD")
    printf '%s\ttext\t%s\t%s\t%s\n' "$t_id" "$t_ln" "$t_old" "$t_new"
  done < <(text_mutations)
}

# ---------------------------------------------------------------------------
# apply <lineno> <old> <new> <src> <dst>
#
# Substitutes on ONE line and proves it happened. `@@TRUNCATE@@` in <new> means "and
# discard the rest of the line", which is how a regex assignment is replaced wholesale
# without writing the whole pattern into this file twice.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# build_tree <dest> - a COMPLETE tree the mutant can be tested in.
# Not just the guard: mutant #2 in the list at the top of this file was a bare tempdir,
# and the fail-closed fixture lookup aborted every run before it validated anything.
# ---------------------------------------------------------------------------
build_tree() {
  local d=$1
  mkdir -p "$d/scripts" "$d/tests/fixtures"
  cp -a "$ROOT/schemas" "$d/schemas"
  cp -a "$ROOT/tests/fixtures/pr-review" "$d/tests/fixtures/pr-review"
  cp -a "$ROOT/$BATS_REL" "$d/$BATS_REL"
  cp -a "$GUARD" "$d/$GUARD_REL"
  # The duplication scanner is NOT mutated here - it is a producer, not a guard, and no
  # verdict rests on it. It is copied because tests/pr-review.bats exercises it, and a
  # mutant tree missing it would fail the baseline for a reason that has nothing to do
  # with the mutant. That is failure mode 2 at the top of this file, exactly.
  cp -a "$ROOT/$SCAN_REL" "$d/$SCAN_REL"
}

# ---------------------------------------------------------------------------
# --run-one: one mutant, in its own tree. Writes <workdir>/results/<id>.
# ---------------------------------------------------------------------------
if [ -n "$RUN_ONE" ]; then
  spec="$WORKDIR/specs/$RUN_ONE"
  [ -f "$spec" ] || { echo "no spec for $RUN_ONE" >&2; exit 1; }
  IFS=$'\t' read -r m_id m_op m_line m_old m_new < "$spec"
  tree="$WORKDIR/trees/$m_id"
  safe_rmtree "$tree"
  build_tree "$tree"
  set +e
  apply "$m_line" "$m_old" "$m_new" "$GUARD" "$tree/$GUARD_REL"
  apply_rc=$?
  set -e

  # PROVE THE MUTATION LANDED. A mutant that changed nothing scores a kill it never
  # earned, and a green suite over an unmutated file is the loudest lie in this repo.
  mutated_line=$(awk -v n="$m_line" 'NR == n { print; exit }' "$tree/$GUARD_REL")
  status_note=ok
  if [ "$apply_rc" -ne 0 ]; then
    status_note='MUTATION-SITE-NOT-PRESENT-AT-THAT-LINE'
  elif cmp -s "$GUARD" "$tree/$GUARD_REL"; then
    status_note='MUTATION-DID-NOT-CHANGE-THE-FILE'
  elif ! grep -qF -- "${m_new%@@TRUNCATE@@}" <<<"$mutated_line"; then
    # HERESTRING, not a pipe: `producer | grep -q` can return 141 on SIGPIPE having
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
  # set +e around the measurement: under `set -e` a failing bats would abort this
  # script BEFORE $? is read, and a mutant that killed the suite would be recorded as
  # nothing at all. The status is read from bats itself, never from a pipeline tail.
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
  # Trees are large; keep only the ones that tell us something.
  if [ "$verdict" = KILLED ]; then safe_rmtree "$tree"; fi
  exit 0
fi

# ---------------------------------------------------------------------------
# The sweep.
# ---------------------------------------------------------------------------
CAT=$(catalogue)
if [ "$LIST_ONLY" = 1 ]; then
  printf '%s\n' "$CAT"
  printf '%s mutants\n' "$(printf '%s\n' "$CAT" | grep -c .)" >&2
  exit 0
fi

WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/prrev-mutate.XXXXXX")
case "$WORKDIR" in
  */prrev-mutate.*) ;;
  *) echo "$PROG: FAIL - refusing to use scratch dir $WORKDIR" >&2; exit 1 ;;
esac
export WORKDIR
cleanup_workdir() { safe_rmtree "${WORKDIR:-}"; }
trap cleanup_workdir EXIT
mkdir -p "$WORKDIR/specs" "$WORKDIR/results" "$WORKDIR/logs" "$WORKDIR/trees"

# --- the baseline, FIRST. -------------------------------------------------
# An unmutated tree must be GREEN here, or every "kill" below is a kill of the harness.
echo "== baseline: the unmutated guard in a mutant tree =="
build_tree "$WORKDIR/trees/baseline"
set +e
( cd "$WORKDIR/trees/baseline" && bats "$BATS_REL" ) > "$WORKDIR/logs/baseline.log" 2>&1
base_rc=$?
set -e
base_ok=$(awk '/^ok /     { n += 1 } END { print n + 0 }' "$WORKDIR/logs/baseline.log")
base_bad=$(awk '/^not ok / { n += 1 } END { print n + 0 }' "$WORKDIR/logs/baseline.log")
if [ "$base_rc" -ne 0 ]; then
  echo "$PROG: FAIL - the UNMUTATED guard does not pass in a mutant tree" >&2
  echo "  ($base_ok ok, $base_bad not ok, exit $base_rc). Every kill below would be a" >&2
  echo "  kill of the harness, not of a mutant. Log: $WORKDIR/logs/baseline.log" >&2
  cp -a "$WORKDIR/logs/baseline.log" "${TMPDIR:-/tmp}/prrev-mutate-baseline.log" || true
  exit 1
fi
printf 'baseline GREEN: %s tests, 0 failures\n\n' "$base_ok"
safe_rmtree "$WORKDIR/trees/baseline"

# --- write one spec per mutant -------------------------------------------
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
xargs -a "$WORKDIR/ids.txt" -P "$JOBS" -I{} "$HERE/mutate-guard.sh" --run-one {} "$WORKDIR"
sweep_rc=$?                              # xargs' own status, read directly
set -e
[ "$sweep_rc" -eq 0 ] || { echo "$PROG: FAIL - the sweep itself failed (exit $sweep_rc)" >&2; exit 1; }

# --- report ---------------------------------------------------------------
REPORT=${PRREV_MUTATION_REPORT:-${TMPDIR:-/tmp}/prrev-mutation-report.tsv}
{
  printf 'id\top\tline\tverdict\trc\tfailing_tests\tkilled_by\tmutated_line\n'
  cat "$WORKDIR"/results/* 
} > "$WORKDIR/report.tsv"

attempted=$(awk 'NR > 1 { n += 1 } END { print n + 0 }' "$WORKDIR/report.tsv")
killed=$(awk -F'\t'   'NR > 1 && $4 == "KILLED"   { n += 1 } END { print n + 0 }' "$WORKDIR/report.tsv")
survived=$(awk -F'\t' 'NR > 1 && $4 == "SURVIVED" { n += 1 } END { print n + 0 }' "$WORKDIR/report.tsv")
invalid=$(awk -F'\t'  'NR > 1 && $4 == "INVALID"  { n += 1 } END { print n + 0 }' "$WORKDIR/report.tsv")

awk -F'\t' 'NR > 1 { printf "%-40s %-5s L%-4s %-9s %s\n", $1, $2, $3, $4, $7 }' \
  "$WORKDIR/report.tsv" | sort

cp -a "$WORKDIR/report.tsv" "$REPORT"

printf '\n== guard_mutation_score ==\n'
printf 'attempted %s   killed %s   survived %s   invalid %s\n' \
  "$attempted" "$killed" "$survived" "$invalid"
printf 'report: %s\n' "$REPORT"

if [ "$attempted" -eq 0 ]; then
  echo "$PROG: FAIL - zero mutants attempted. A mutation set that matches nothing" >&2
  echo "  passes vacuously, which is exactly what S3.D calls DEGRADED, not clean." >&2
  exit 1
fi
if [ "$invalid" -ne 0 ]; then
  echo "$PROG: FAIL - $invalid mutation(s) did not change the guard." >&2
  echo "  A mutant that edits nothing scores a kill it never earned." >&2
  exit 1
fi
if [ "$survived" -ne 0 ]; then
  echo "$PROG: FAIL - $survived mutant(s) SURVIVED." >&2
  echo "  Each is a validation rule the guard states and no fixture tests: the receipt" >&2
  echo "  could break it and the table would stay green. S8 fixes guard_mutation_score" >&2
  echo "  at 100% with no ratchet. Logs kept under $WORKDIR/logs." >&2
  trap - EXIT
  exit 1
fi
printf 'guard_mutation_score = 100%% (%s/%s)\n' "$killed" "$attempted"
