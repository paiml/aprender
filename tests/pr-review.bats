#!/usr/bin/env bats
#
# PR-REVIEW-SKILL-002 v2 S6.3 — the fixture table, exercised by bats-core.
#
# Every row of S6.3 is a COMMITTED fixture under tests/fixtures/pr-review/, and every
# fixture is asserted to produce the verdict the spec states AND, when it is rejected,
# the blocking class contracts/pr-review-skill-v2.yaml names for it. Asserting only the
# exit code would let a fixture pass for the wrong reason: rows 9 and 10 both live on
# the same head commit and both exit 1, but one is B6 and the other is B1, and an early
# version of the guard returned B1 for both because of a jq quirk. The class assertion
# is what caught it.
#
# Rows 6, 7 and 14 are DISCRIMINATION cases and must stay GREEN. Without them "refuse
# every receipt" reads green — the over-reach a discrimination case already caught in
# PERF-055 and in the #2766 delta-gate work.

setup_file() {
  REPO_ROOT=$(cd -- "$BATS_TEST_DIRNAME/.." && pwd)
  export REPO_ROOT
  export GUARD="$REPO_ROOT/scripts/check_pr_review_receipt.sh"
  export FIX="$REPO_ROOT/tests/fixtures/pr-review"

  # The receipts name commits in a purpose-built repository, not in aprender: for any
  # commit reachable from aprender's own origin/main, `git merge-base origin/main X` is
  # X itself, so row 10's check would pass vacuously and row 1's diff would be empty.
  # See tests/fixtures/pr-review/make-fixture-repo.sh for the topology and why.
  export FIXTURE_REPO="$BATS_FILE_TMPDIR/fixture-repo"
  "$FIX/make-fixture-repo.sh" "$FIXTURE_REPO" >/dev/null
}

setup() {
  export PR_REVIEW_REPO="$FIXTURE_REPO"
  export PR_REVIEW_PUBKEY="$FIX/keys/pr-review-test.pub"
  # Each test gets its own scratch directory; nothing is written into the fixture tree.
  WORK="$BATS_TEST_TMPDIR/work"
  mkdir -p "$WORK"
}

# seed_controls <dir> - a private, writable copy of the committed control set.
seed_controls() {
  mkdir -p "$1"
  cp -r "$FIX/positive-control/." "$1/"
}

# assert_row <fixture-dir-name> <GREEN|RED> [expected-blocking-class]
assert_row() {
  local name=$1 want=$2 class=${3:-}
  run "$GUARD" "$FIX/$name"
  if [ "$want" = GREEN ]; then
    [ "$status" -eq 0 ] || {
      echo "expected GREEN (exit 0), got exit $status:"; echo "$output"; return 1; }
    [[ "$output" == *"ACCEPT"* ]] || { echo "no ACCEPT line:"; echo "$output"; return 1; }
  else
    [ "$status" -eq 1 ] || {
      echo "expected RED (exit 1), got exit $status:"; echo "$output"; return 1; }
    [[ "$output" == *"REJECT"* ]] || { echo "no REJECT line:"; echo "$output"; return 1; }
    if [ -n "$class" ]; then
      [[ "$output" == *"[$class]"* ]] || {
        echo "expected blocking class $class, output was:"; echo "$output"; return 1; }
    fi
  fi
}

# --- S6.1 --------------------------------------------------------------------

@test "S6.1 all four positive controls fire before anything is validated" {
  run "$GUARD" "$FIX/row-07-honest-docs-only-all-not-triggered"
  [ "$status" -eq 0 ]
  # Four controls at increasing depth. A schema-depth control alone stays green with
  # every semantic branch below it deleted; these three reach branches no S6.3 row does.
  [[ "$output" =~ positive-control[[:space:]]+schema-invalid[[:space:]]+fired[[:space:]]\(B1 ]]
  [[ "$output" =~ positive-control[[:space:]]+self-review[[:space:]]+fired[[:space:]]\(B2 ]]
  [[ "$output" =~ positive-control[[:space:]]+findings-digest[[:space:]]+fired[[:space:]]\(B1 ]]
  [[ "$output" =~ positive-control[[:space:]]+cost-missing[[:space:]]+fired[[:space:]]\(B1 ]]
}

@test "S6.1 the guard refuses to run at all when a positive-control fixture is gone" {
  PR_REVIEW_POSITIVE_CONTROL_DIR="$WORK/absent" run "$GUARD" \
    "$FIX/row-07-honest-docs-only-all-not-triggered"
  [ "$status" -eq 1 ]
  [[ "$output" == *"positive-control fixture missing"* ]]
  # An honest receipt must NOT be accepted when a control cannot run: without the
  # controls, a green verdict is a count of files.
  [[ "$output" != *"ACCEPT"* ]]
}

@test "S6.1 a control rejected under the wrong CLASS is a MISFIRE, not a pass" {
  # Seed the self-review control with the findings-digest receipt. It is still
  # rejected - but under B1, not the B2 it exists to prove.
  seed_controls "$WORK/pc"
  cp "$FIX/positive-control/findings-digest/receipt.intoto.jsonl" \
     "$FIX/positive-control/findings-digest/receipt.intoto.jsonl.minisig" \
     "$FIX/positive-control/findings-digest/findings.sarif" "$WORK/pc/self-review/"
  PR_REVIEW_POSITIVE_CONTROL_DIR="$WORK/pc" run "$GUARD" \
    "$FIX/row-07-honest-docs-only-all-not-triggered"
  [ "$status" -eq 1 ]
  [[ "$output" == *"POSITIVE CONTROL MISFIRED"* ]]
  [[ "$output" != *"ACCEPT"* ]]
}

@test "S6.1 a control rejected under the right class but the wrong BRANCH is a MISFIRE" {
  # findings-digest and cost-missing are BOTH B1. Swapping one for the other keeps the
  # class correct and changes the branch, so only the reason assertion can catch it.
  # This is not hypothetical: with the class alone asserted, deleting the in-toto schema
  # gate left the schema control firing on the signature branch and the mutation SURVIVED.
  seed_controls "$WORK/pc2"
  cp "$FIX/positive-control/cost-missing/receipt.intoto.jsonl" \
     "$FIX/positive-control/cost-missing/receipt.intoto.jsonl.minisig" \
     "$FIX/positive-control/cost-missing/findings.sarif" "$WORK/pc2/findings-digest/"
  PR_REVIEW_POSITIVE_CONTROL_DIR="$WORK/pc2" run "$GUARD" \
    "$FIX/row-07-honest-docs-only-all-not-triggered"
  [ "$status" -eq 1 ]
  [[ "$output" == *"POSITIVE CONTROL MISFIRED"* ]]
  [[ "$output" == *"wrong branch"* ]]
  [[ "$output" != *"ACCEPT"* ]]
}

# --- S6.3, the fourteen rows -------------------------------------------------

@test "row 1  cuda not-triggered on a diff touching src/cuda/            RED  B1" {
  assert_row row-01-cuda-not-triggered-on-cuda-diff RED B1
}

@test "row 2  mutation.attempted 0 with status consulted                 RED  B1" {
  assert_row row-02-mutation-attempted-zero RED B1
}

@test "row 3  cited finding with an empty excerpt                        RED  B1" {
  assert_row row-03-cited-empty-excerpt RED B1
}

@test "row 4  comparative claim with no comparator command or hash       RED  B4" {
  assert_row row-04-comparative-claim-no-comparator RED B4
}

@test "row 5  pmat unreachable with verdict PASS                         RED  B1" {
  assert_row row-05-unreachable-pmat-verdict-pass RED B1
}

@test "row 6  pmat unreachable with verdict DEGRADED                   GREEN     [discrimination]" {
  assert_row row-06-unreachable-pmat-verdict-degraded GREEN
}

@test "row 7  honest docs-only PR, all consultations not-triggered     GREEN     [discrimination]" {
  assert_row row-07-honest-docs-only-all-not-triggered GREEN
}

@test "row 8  reviewer_actor equals author_actor                         RED  B2" {
  assert_row row-08-self-review RED B2
}

@test "row 9  index_commit not an ancestor of head_sha, verdict PASS     RED  B6" {
  assert_row row-09-stale-index-verdict-pass RED B6
}

@test "row 10 base_sha is not git merge-base origin/main head_sha        RED  B1" {
  assert_row row-10-base-sha-not-merge-base RED B1
}

@test "row 11 finding with an empty failure_scenario                     RED  B1" {
  assert_row row-11-empty-failure-scenario RED B1
}

@test "row 12 excerpt_sha256 is not sha256(excerpt)                      RED  B1" {
  assert_row row-12-excerpt-digest-mismatch RED B1
}

@test "row 13 valid receipt, invalid signature                           RED  B1" {
  assert_row row-13-invalid-signature RED B1
}

@test "row 14 complete GPU review, all four consulted, findings        GREEN     [discrimination]" {
  assert_row row-14-complete-gpu-review GREEN
}

@test "row 15 finding carrying no grounding mark at all                  RED  B1" {
  # Not one of S6.3's fourteen. Owed to PRREV-003 by contracts/pr-review-skill-v2.yaml,
  # whose falsification test F-PRREV-001 is recorded LIVE-PENDING on exactly this case:
  # rows 3, 11 and 12 cover a malformed mark, but nothing covered a MISSING one — the
  # single S8 metric (unmarked_claims = 0) the fourteen rows left asserted.
  assert_row row-15-finding-with-no-grounding-mark RED B1
}

# --- rules that are not rows -------------------------------------------------

@test "a missing receipt is RED, not skipped" {
  mkdir -p "$WORK/empty-dir"
  run "$GUARD" "$WORK/empty-dir"
  [ "$status" -eq 1 ]
  [[ "$output" == *"a missing receipt is RED, not skipped"* ]]
}

@test "a receipt directory that does not exist is RED, not skipped" {
  run "$GUARD" "$WORK/does-not-exist"
  [ "$status" -eq 1 ]
  [[ "$output" == *"no such receipt directory"* ]]
}

@test "a run over zero receipts is not a pass" {
  run "$GUARD"
  [ "$status" -eq 1 ]
  [[ "$output" == *"no receipt directory given"* ]]
}

@test "findings.sarif present but receipt absent is still RED" {
  mkdir -p "$WORK/sarif-only"
  cp "$FIX/row-07-honest-docs-only-all-not-triggered/findings.sarif" "$WORK/sarif-only/"
  run "$GUARD" "$WORK/sarif-only"
  [ "$status" -eq 1 ]
  [[ "$output" == *"receipt.intoto.jsonl is missing"* ]]
}

@test "one bad receipt in a batch fails the whole run" {
  # A per-receipt loop that forgets to accumulate its status reports the LAST result.
  run "$GUARD" "$FIX/row-07-honest-docs-only-all-not-triggered" \
                "$FIX/row-08-self-review" \
                "$FIX/row-14-complete-gpu-review"
  [ "$status" -eq 1 ]
  [[ "$output" == *"REJECT"* ]]
  [[ "$output" == *"ACCEPT"* ]]
}

# --- the regex case tables ---------------------------------------------------
#
# Any regex in this repository ships a must-match / must-not-match table. The patterns
# have been wrong six times; a table caught every one and review caught none.

run_case_table() {  # run_case_table <table-basename> <guard-flag>
  local table="$FIX/$1-cases.tsv" flag=$2 expect subject why rc fails=0 rows=0
  while IFS=$'\t' read -r expect subject why; do
    case "$expect" in ''|'#'*) continue ;; esac
    rows=$((rows + 1))
    "$GUARD" "$flag" "$subject" >/dev/null 2>&1
    rc=$?
    if [ "$expect" = MATCH ] && [ "$rc" -ne 0 ]; then
      echo "MISS   expected MATCH,    got no match: $subject"; fails=$((fails + 1))
    elif [ "$expect" = NO-MATCH ] && [ "$rc" -eq 0 ]; then
      echo "SPURIOUS expected NO-MATCH, got a match: $subject"; fails=$((fails + 1))
    fi
  done < "$table"
  # A table that matched nothing is the vacuous-pass shape: assert it had rows.
  [ "$rows" -ge 10 ] || { echo "case table $table has only $rows rows"; return 1; }
  echo "$rows rows checked"
  [ "$fails" -eq 0 ]
}

@test "S3.B CUDA path trigger matches its case table, both polarities" {
  run run_case_table cuda-path --match-path
  [ "$status" -eq 0 ] || { echo "$output"; false; }
}

@test "S3.B CUDA message trigger matches its case table, both polarities" {
  run run_case_table cuda-message --match-message
  [ "$status" -eq 0 ] || { echo "$output"; false; }
}

@test "S3.C.1 comparative-claim detection matches its case table, both polarities" {
  run run_case_table comparative-claim --match-comparative
  [ "$status" -eq 0 ] || { echo "$output"; false; }
}

# --- the fixture repository itself -------------------------------------------

@test "the fixture repository has the topology the receipts were written against" {
  # make-fixture-repo.sh asserts merge-base, ancestry, the CUDA path in the diff, and
  # that the generated SHAs equal expected-shas.txt. If git ever changes object hashing
  # the committed receipts would describe a different repository, and rows 1, 9, 10 and
  # 14 would silently test nothing.
  run "$FIX/make-fixture-repo.sh" "$WORK/topology-check"
  [ "$status" -eq 0 ] || { echo "$output"; false; }
}

@test "every S6.3 row plus the contract's owed row has exactly one fixture directory" {
  local n
  n=$(find "$FIX" -maxdepth 1 -type d -name 'row-*' | wc -l)
  [ "$n" -eq 15 ] || { echo "expected 15 row fixtures (14 from S6.3 + row 15 owed by the contract), found $n"; false; }
  local i
  for i in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15; do
    find "$FIX" -maxdepth 1 -type d -name "row-$i-*" | grep -q . \
      || { echo "no fixture directory for row $i"; false; }
  done
}

@test "every fixture carries all three artifacts the guard reads" {
  local d missing=""
  for d in "$FIX"/row-*/; do
    [ -f "$d/receipt.intoto.jsonl" ]         || missing="$missing $d:receipt"
    [ -f "$d/findings.sarif" ]               || missing="$missing $d:sarif"
    [ -f "$d/receipt.intoto.jsonl.minisig" ] || missing="$missing $d:signature"
  done
  [ -z "$missing" ] || { echo "incomplete fixtures:$missing"; false; }
}
