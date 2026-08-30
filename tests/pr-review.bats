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

# assert_row <fixture-dir-name> <GREEN|RED> [<expected-class> <expected-reason-substring>]
#
# The REASON is asserted, not only the class. B1 covers thirty-odd branches, so a case
# that trips a DIFFERENT B1 branch than the one it exists to pin still reports B1 and
# still exits 1 - it passes for the wrong reason, and the mutant that dropped its
# branch lives.
#
# HOW LOAD-BEARING THAT IS, MEASURED RATHER THAN ASSUMED. A counter-sweep ran the whole
# mutation set against a copy of this file with every assertion made reason-BLIND:
# 110/119 killed, so NINE mutants die only because the reason is asserted. They sit on
# seven guard branches - head_sha absent (184), base_sha absent (185), an unresolvable
# head (214), a non-numeric mutation.attempted (253), index_commit absent (286) or
# unresolvable (288), and a cited finding with no excerpt_sha256 (313) - every one of
# which falls through to a NEIGHBOURING B1 branch that rejects with a different reason.
# Class-only, all nine read as a correct rejection.
#
# The first candidate tested was NOT one of them, which is why this says "measured".
# Dropping the empty-excerpt check (guard line 312) was predicted to leave row 3
# rejected on the digest branch; it does not. Row 3's excerpt_sha256 is sha256("") on
# purpose, so with the check gone the receipt is ACCEPTED and the exit code alone kills
# the mutant. The prediction was wrong; the counter-sweep is what stands.
assert_row() {
  local name=$1 want=$2 class=${3:-} reason=${4:-}
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
    if [ -n "$reason" ]; then
      [[ "$output" == *"$reason"* ]] || {
        echo "expected the rejection reason to contain:"; echo "  $reason"
        echo "output was:"; echo "$output"; return 1; }
    fi
  fi
}

# --- branch probes (PRREV-004) -----------------------------------------------
#
# The fifteen rows of S6.3 pin fifteen branches. The guard has more than fifty, and
# S6.4 requires 100% kill on the mutation set - so every remaining branch needs a case
# that trips IT, or dropping it is a rule nothing tests.
#
# A probe is DERIVED from a committed row by a single jq edit and RE-SIGNED with the
# committed TEST-ONLY key, because the signature is verified before any semantic branch
# is reached: an unsigned probe would be rejected at the signature and would pin nothing.
# The derivation is committed, the base bytes are committed, and the key is committed,
# so a probe is reproducible from this tree - it is a shorter way of writing a fixture,
# not a weaker one.

PROBE_KEY() { printf '%s' "$FIX/keys/pr-review-test-TEST-ONLY.key"; }

# make_probe <name> <base-row> <receipt-jq> [<sarif-jq>] -> echoes the probe directory
make_probe() {
  local name=$1 base=$2 rjq=$3 sjq=${4:-.}
  local d="$WORK/probe-$name" src="$FIX/$base"
  mkdir -p "$d"
  jq -c "$sjq" "$src/findings.sarif" > "$d/findings.sarif" || return 1
  local fsha
  fsha=$(sha256sum "$d/findings.sarif" | cut -d' ' -f1)
  # findings_ref.sha256 is recomputed FIRST so a probe that edits the SARIF stays
  # internally consistent; a probe that means to break the digest overrides it after.
  jq -c --arg fsha "$fsha" ".predicate.findings_ref.sha256 = \$fsha | ($rjq)" \
     "$src/receipt.intoto.jsonl" > "$d/receipt.intoto.jsonl" || return 1
  minisign -q -S -s "$(PROBE_KEY)" -m "$d/receipt.intoto.jsonl" \
    -t "PRREV-004 branch probe: $name" -c "TEST ONLY probe signature" </dev/null || return 1
  printf '%s' "$d"
}

# assert_probe <name> <base-row> <class> <reason-substring> <receipt-jq> [<sarif-jq>]
assert_probe() {
  local name=$1 base=$2 class=$3 reason=$4 rjq=$5 sjq=${6:-.} d
  d=$(make_probe "$name" "$base" "$rjq" "$sjq") || { echo "probe $name could not be built"; return 1; }
  run "$GUARD" "$d"
  [ "$status" -eq 1 ] || {
    echo "probe $name: expected RED (exit 1), got exit $status:"; echo "$output"; return 1; }
  [[ "$output" == *"[$class]"* ]] || {
    echo "probe $name: expected blocking class $class, output was:"; echo "$output"; return 1; }
  [[ "$output" == *"$reason"* ]] || {
    echo "probe $name: expected the rejection reason to contain:"; echo "  $reason"
    echo "output was:"; echo "$output"; return 1; }
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
  # It must misfire ON THE CLASS, and say which class it expected and which it got.
  # Asserting only "MISFIRED" left the class comparison entirely untested: with that
  # comparison deleted this control still misfires - on the REASON - and the mutant
  # survived the whole sweep. Measured, not argued.
  [[ "$output" == *"Expected the receipt to be rejected under B2"* ]]
  [[ "$output" == *"it was rejected under B1"* ]]
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
  assert_row row-01-cuda-not-triggered-on-cuda-diff RED B1 \
    "consultations.cuda is not-triggered but its S3.B trigger fires on this diff (path src/cuda/kernel.cu)"
}

@test "row 2  mutation.attempted 0 with status consulted                 RED  B1" {
  assert_row row-02-mutation-attempted-zero RED B1 \
    "mutation.status is consulted with attempted=0"
}

@test "row 3  cited finding with an empty excerpt                        RED  B1" {
  assert_row row-03-cited-empty-excerpt RED B1 \
    "a cited finding has an empty excerpt"
}

@test "row 4  comparative claim with no comparator command or hash       RED  B4" {
  assert_row row-04-comparative-claim-no-comparator RED B4 \
    "is missing comparator field(s): command, env_sha256, artifact_sha256"
}

@test "row 5  pmat unreachable with verdict PASS                         RED  B1" {
  assert_row row-05-unreachable-pmat-verdict-pass RED B1 \
    "consultations.pmat is unreachable but the verdict is PASS"
}

@test "row 6  pmat unreachable with verdict DEGRADED                   GREEN     [discrimination]" {
  assert_row row-06-unreachable-pmat-verdict-degraded GREEN
}

@test "row 7  honest docs-only PR, all consultations not-triggered     GREEN     [discrimination]" {
  assert_row row-07-honest-docs-only-all-not-triggered GREEN
}

@test "row 8  reviewer_actor equals author_actor                         RED  B2" {
  assert_row row-08-self-review RED B2 \
    "reviewer_actor.id = author_actor.id ="
}

@test "row 9  index_commit not an ancestor of head_sha, verdict PASS     RED  B6" {
  assert_row row-09-stale-index-verdict-pass RED B6 \
    "is not an ancestor of head"
}

@test "row 10 base_sha is not git merge-base origin/main head_sha        RED  B1" {
  assert_row row-10-base-sha-not-merge-base RED B1 \
    "is not git merge-base origin/main"
}

@test "row 11 finding with an empty failure_scenario                     RED  B1" {
  assert_row row-11-empty-failure-scenario RED B1 \
    "a finding has an empty failure_scenario"
}

@test "row 12 excerpt_sha256 is not sha256(excerpt)                      RED  B1" {
  assert_row row-12-excerpt-digest-mismatch RED B1 \
    "records excerpt_sha256 0000000000000000000000000000000000000000000000000000000000000000 but sha256(excerpt)"
}

@test "row 13 valid receipt, invalid signature                           RED  B1" {
  assert_row row-13-invalid-signature RED B1 \
    "signature does not verify against"
}

@test "row 14 complete GPU review, all four consulted, findings        GREEN     [discrimination]" {
  assert_row row-14-complete-gpu-review GREEN
}

@test "row 15 finding carrying no grounding mark at all                  RED  B1" {
  # Not one of S6.3's fourteen. Owed to PRREV-003 by contracts/pr-review-skill-v2.yaml,
  # whose falsification test F-PRREV-001 is recorded LIVE-PENDING on exactly this case:
  # rows 3, 11 and 12 cover a malformed mark, but nothing covered a MISSING one — the
  # single S8 metric (unmarked_claims = 0) the fourteen rows left asserted.
  assert_row row-15-finding-with-no-grounding-mark RED B1 \
    "a finding carries no properties.grounding"
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
  [ "$n" -eq 17 ] || { echo "expected 17 row fixtures (14 from S6.3, row 15 owed by the contract, rows 16-17 owed by PRREV-007 F4), found $n"; false; }
  local i
  for i in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17; do
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

# =============================================================================
# BRANCH PROBES (PRREV-004, S6.4)
#
# One case per validation branch the fifteen rows do not reach. Each exists because
# the mutation sweep in scripts/mutate-guard.sh found the corresponding mutant ALIVE:
# a rule the guard states and nothing tests. They are named for the branch, not for a
# spec row, because they are not spec rows.
# =============================================================================

# --- artifact presence, shape, and the offline schema gate -------------------

@test "probe findings.sarif absent                                     RED  B1" {
  local d="$WORK/probe-no-sarif"
  mkdir -p "$d"
  cp "$FIX/row-07-honest-docs-only-all-not-triggered/receipt.intoto.jsonl" \
     "$FIX/row-07-honest-docs-only-all-not-triggered/receipt.intoto.jsonl.minisig" "$d/"
  run "$GUARD" "$d"
  [ "$status" -eq 1 ]
  [[ "$output" == *"[B1]"* ]]
  [[ "$output" == *"findings.sarif is missing"* ]]
}

@test "probe receipt holding two JSON records is not JSON Lines        RED  B1" {
  # JSON Lines with two records parses as two Statements. Which one was signed, and
  # which one is the review? A file that cannot answer that is not a receipt.
  local d="$WORK/probe-two-records" src="$FIX/row-07-honest-docs-only-all-not-triggered"
  mkdir -p "$d"
  cp "$src/findings.sarif" "$d/"
  cat "$src/receipt.intoto.jsonl" "$src/receipt.intoto.jsonl" > "$d/receipt.intoto.jsonl"
  cp "$src/receipt.intoto.jsonl.minisig" "$d/"
  run "$GUARD" "$d"
  [ "$status" -eq 1 ]
  [[ "$output" == *"[B1]"* ]]
  [[ "$output" == *"holds 2 JSON record/s"* ]]
}

@test "probe receipt that is not parseable JSON                        RED  B1" {
  local d="$WORK/probe-receipt-not-json" src="$FIX/row-07-honest-docs-only-all-not-triggered"
  mkdir -p "$d"
  cp "$src/findings.sarif" "$src/receipt.intoto.jsonl.minisig" "$d/"
  printf 'this is not JSON at all' > "$d/receipt.intoto.jsonl"
  run "$GUARD" "$d"
  [ "$status" -eq 1 ]
  [[ "$output" == *"[B1]"* ]]
  [[ "$output" == *"receipt.intoto.jsonl is not parseable JSON"* ]]
}

@test "probe findings.sarif that is not parseable JSON                 RED  B1" {
  local d="$WORK/probe-sarif-not-json" src="$FIX/row-07-honest-docs-only-all-not-triggered"
  mkdir -p "$d"
  cp "$src/receipt.intoto.jsonl" "$src/receipt.intoto.jsonl.minisig" "$d/"
  printf 'not json either' > "$d/findings.sarif"
  run "$GUARD" "$d"
  [ "$status" -eq 1 ]
  [[ "$output" == *"[B1]"* ]]
  [[ "$output" == *"findings.sarif is not parseable JSON"* ]]
}

@test "probe findings.sarif that parses but fails the SARIF schema     RED  B1" {
  # Well-formed JSON, wrong shape. The receipt is untouched and still correctly
  # signed, so only the SARIF schema gate can reject this - which is the point:
  # dropping that gate must not leave a malformed findings file reading green.
  local d="$WORK/probe-sarif-schema" src="$FIX/row-07-honest-docs-only-all-not-triggered"
  mkdir -p "$d"
  cp "$src/receipt.intoto.jsonl" "$src/receipt.intoto.jsonl.minisig" "$d/"
  printf '{"version":"2.1.0","runs":"this must be an array"}\n' > "$d/findings.sarif"
  run "$GUARD" "$d"
  [ "$status" -eq 1 ]
  [[ "$output" == *"[B1]"* ]]
  [[ "$output" == *"fails schemas/sarif-2.1.0.json"* ]]
}

# --- signature material ------------------------------------------------------

@test "probe the public key is absent                                  RED  B1" {
  # An unverifiable signature is not a verified one. Without this branch the guard
  # would fall through to minisign, which reports a missing key as a failed
  # verification - the same words for "the key is gone" and "this receipt is forged".
  PR_REVIEW_PUBKEY="$WORK/there-is-no-key-here.pub" \
    run "$GUARD" "$FIX/row-07-honest-docs-only-all-not-triggered"
  [ "$status" -eq 1 ]
  [[ "$output" == *"[B1]"* ]]
  [[ "$output" == *"is absent; an unverifiable signature is not a verified one"* ]]
}

@test "probe the receipt carries no signature at all                   RED  B1" {
  local d="$WORK/probe-unsigned" src="$FIX/row-07-honest-docs-only-all-not-triggered"
  mkdir -p "$d"
  cp "$src/receipt.intoto.jsonl" "$src/findings.sarif" "$d/"
  run "$GUARD" "$d"
  [ "$status" -eq 1 ]
  [[ "$output" == *"[B1]"* ]]
  [[ "$output" == *"receipt is unsigned"* ]]
}

# --- the predicate's own identity -------------------------------------------

@test "probe predicateType is some other attestation                   RED  B1" {
  # The in-toto Statement schema constrains predicateType only to a TypeURI: a SLSA
  # provenance, a VSA, or anything else at all validates against it. Only this branch
  # makes the receipt be a PR REVIEW.
  assert_probe predicate-type row-07-honest-docs-only-all-not-triggered B1 \
    "predicateType is 'https://slsa.dev/verification_summary/v1'" \
    '.predicateType = "https://slsa.dev/verification_summary/v1"'
}

@test "probe attestation_level claims more than L1-self                RED  B1" {
  # R1: SLSA Build L3 requires an isolated builder the tenant cannot influence. A
  # skill invoked by the authoring agent is self-attestation. A receipt that claims
  # otherwise is the exact enforcement theatre the spec rejects, so it is refused.
  assert_probe attestation-level row-07-honest-docs-only-all-not-triggered B1 \
    "attestation_level is 'SLSA-BUILD-L3'" \
    '.predicate.attestation_level = "SLSA-BUILD-L3"'
}

@test "probe head_sha is absent                                        RED  B1" {
  assert_probe head-absent row-07-honest-docs-only-all-not-triggered B1 \
    "predicate.head_sha is absent" 'del(.predicate.head_sha)'
}

@test "probe base_sha is absent                                        RED  B1" {
  assert_probe base-absent row-07-honest-docs-only-all-not-triggered B1 \
    "predicate.base_sha is absent" 'del(.predicate.base_sha)'
}

@test "probe the subject digest is not the head the predicate reviews  RED  B1" {
  # in-toto binds the attestation to subject[].digest. If the predicate reviews a
  # different commit from the one the statement is ABOUT, the signature attests to
  # a review of something else.
  assert_probe subject-digest row-07-honest-docs-only-all-not-triggered B1 \
    "the sha1 digest of subject 0 is" \
    '.subject[0].digest.sha1 = "0000000000000000000000000000000000000000"'
}

@test "probe verdict outside the four defined values                   RED  B1" {
  assert_probe verdict-outside row-07-honest-docs-only-all-not-triggered B1 \
    "verdict 'PROBABLY-FINE' is outside" '.predicate.verdict = "PROBABLY-FINE"'
}

@test "probe findings_ref.path points somewhere else                   RED  B1" {
  assert_probe findings-ref-path row-07-honest-docs-only-all-not-triggered B1 \
    "findings_ref.path is 'somewhere-else.sarif'" \
    '.predicate.findings_ref.path = "somewhere-else.sarif"'
}

@test "probe author_actor.id is absent                                 RED  B1" {
  assert_probe author-absent row-07-honest-docs-only-all-not-triggered B1 \
    "author_actor.id is absent" 'del(.predicate.author_actor.id)'
}

@test "probe reviewer_actor.id is absent                               RED  B1" {
  # S5's separation cannot be checked against an absent reviewer, and an absent
  # reviewer is indistinguishable from no review at all.
  assert_probe reviewer-absent row-07-honest-docs-only-all-not-triggered B1 \
    "reviewer_actor.id is absent" 'del(.predicate.reviewer_actor.id)'
}

# --- the diff boundary -------------------------------------------------------

@test "probe head_sha does not resolve in the repository               RED  B1" {
  assert_probe head-unresolvable row-07-honest-docs-only-all-not-triggered B1 \
    "does not resolve to a commit in" \
    '.predicate.head_sha = "0000000000000000000000000000000000000000"
     | .subject[0].digest.sha1 = "0000000000000000000000000000000000000000"'
}

@test "probe head_sha shares no history with origin/main               RED  B1" {
  # A resolvable head with NO merge base at all: `git merge-base` exits non-zero and
  # prints nothing. Without this branch that empty output would be compared against
  # base_sha, and a receipt claiming base_sha "" would be the one that passed.
  local repo="$WORK/orphan-repo"
  cp -r "$FIXTURE_REPO" "$repo"
  git -C "$repo" checkout -q --orphan unrelated
  git -C "$repo" rm -rq --cached .
  rm -rf "$repo/src" "$repo/docs" "$repo/crates" "$repo/README.md"
  printf 'an unrelated root\n' > "$repo/UNRELATED.md"
  git -C "$repo" add -A
  GIT_AUTHOR_NAME=prrev GIT_AUTHOR_EMAIL=prrev@fixture.invalid \
  GIT_COMMITTER_NAME=prrev GIT_COMMITTER_EMAIL=prrev@fixture.invalid \
    git -C "$repo" commit -q -m "O1 an orphan root sharing no history with main"
  local o1
  o1=$(git -C "$repo" rev-parse HEAD)
  # Prove the topology this probe needs, rather than assuming it.
  run git -C "$repo" merge-base refs/remotes/origin/main "$o1"
  [ "$status" -ne 0 ]

  local d
  d=$(make_probe orphan-head row-07-honest-docs-only-all-not-triggered \
      ".predicate.head_sha = \"$o1\" | .subject[0].digest.sha1 = \"$o1\"")
  PR_REVIEW_REPO="$repo" run "$GUARD" "$d"
  [ "$status" -eq 1 ]
  [[ "$output" == *"[B1]"* ]]
  [[ "$output" == *"cannot compute merge-base(origin/main,"* ]]
}

# --- consultation status vocabulary -----------------------------------------

@test "probe a consultation with no status at all                      RED  B1" {
  # S3.0: an omitted consultation must not be indistinguishable from one that ran
  # and found nothing.
  assert_probe status-absent row-07-honest-docs-only-all-not-triggered B1 \
    "consultations.crux.status is absent" 'del(.predicate.consultations.crux.status)'
}

@test "probe a consultation status outside the three-state vocabulary  RED  B1" {
  # "skipped" is the word that hides the difference S3.0 exists to make visible.
  assert_probe status-invalid row-07-honest-docs-only-all-not-triggered B1 \
    "consultations.crux.status is 'skipped'" \
    '.predicate.consultations.crux.status = "skipped"'
}

# --- the mutation consultation's own counts ---------------------------------

@test "probe mutation.attempted is not a count                         RED  B1" {
  # Row 2 pins attempted = 0. A non-numeric attempted skips the > 0 comparison
  # entirely, so "attempted": "lots" would otherwise sail past the vacuity check.
  assert_probe attempted-non-numeric row-14-complete-gpu-review B1 \
    "attempted is 'lots'" '.predicate.consultations.mutation.attempted = "lots"'
}

@test "probe mutation.killed is not a count                            RED  B1" {
  assert_probe killed-non-numeric row-14-complete-gpu-review B1 \
    "mutation.killed is 'all of them', not a count" \
    '.predicate.consultations.mutation.killed = "all of them"'
}

# --- the pmat index -----------------------------------------------------------

@test "probe pmat consulted with no index_commit recorded              RED  B1" {
  # The 66-commit drift scar: an unrecorded index cannot be SHOWN to describe this PR,
  # and row 9 only pins the case where it was recorded and was stale.
  assert_probe index-absent row-14-complete-gpu-review B1 \
    "index_commit is absent" 'del(.predicate.consultations.pmat.index_commit)'
}

@test "probe pmat index_commit does not resolve in the repository      RED  B1" {
  assert_probe index-unresolvable row-14-complete-gpu-review B1 \
    "index_commit 0000000000000000000000000000000000000000 does not resolve" \
    '.predicate.consultations.pmat.index_commit = "0000000000000000000000000000000000000000"'
}

@test "probe index_is_ancestor is MISREPORTED as false                 RED  B1" {
  # The index really IS fresh here; the receipt says it is not. A receipt that
  # misreports its own ancestry is invalid whatever the verdict, because the field is
  # what every downstream reader trusts instead of recomputing.
  assert_probe index-misreported row-14-complete-gpu-review B1 \
    "index_is_ancestor is recorded as false but merge-base --is-ancestor" \
    '.predicate.consultations.pmat.index_is_ancestor = false'
}

# --- grounding marks (S1) ----------------------------------------------------

@test "probe a grounding mark outside the three categories             RED  B1" {
  # S1: "There is no fourth category." Row 15 pins a MISSING mark; this pins an
  # invented one, which is how a fourth category would actually arrive.
  assert_probe grounding-fourth-category row-14-complete-gpu-review B1 \
    "the grounding of a finding is outside { cited, measured, asserted }" '.' \
    '.runs[0].results[0].properties.grounding = "inferred"'
}

@test "probe a cited finding with an empty source                      RED  B1" {
  # S6.3 row 3 reads "empty source OR excerpt"; the committed row 3 is the excerpt
  # half. This is the source half, and dropping the check survived without it.
  assert_probe cited-empty-source row-14-complete-gpu-review B1 \
    "a cited finding has an empty source" '.' \
    '.runs[1].results[0].properties.source = ""'
}

@test "probe a cited finding with no excerpt_sha256                    RED  B1" {
  # S1.1: cited is VERIFIED, not just labelled. With no digest there is nothing to
  # verify against, and the digest-comparison branch would compare against nothing.
  assert_probe cited-no-digest row-14-complete-gpu-review B1 \
    "a cited finding has no excerpt_sha256" '.' \
    'del(.runs[1].results[0].properties.excerpt_sha256)'
}

@test "probe an asserted finding classed blocking                      RED  B1" {
  # S1: reviewer judgement is "permitted, visibly distinct, never blocking".
  assert_probe asserted-blocking row-14-complete-gpu-review B1 \
    "a finding marked asserted is classed blocking" '.' \
    '.runs[0].results[0].properties.grounding = "asserted"
     | .runs[0].results[0].properties.precision_class = "blocking"'
}

# --- comparative claims (S3.C.1) ---------------------------------------------

@test "probe a competitor ratio stated in a finding but never recorded RED  B4" {
  # Row 4 pins a RECORDED claim with an incomplete comparator. This pins the other
  # half, and the more likely one: the ratio appears in the review's prose and the
  # comparative_claims list is empty. That is the never-ran-Ollama shape with one
  # extra step, and without this branch it is the shape that passes.
  assert_probe comparative-unrecorded row-14-complete-gpu-review B4 \
    "consultations.crux.comparative_claims is empty" \
    '.predicate.consultations.crux.comparative_claims = []'
}

# --- the guard's own preconditions -------------------------------------------

@test "a tool the guard needs and cannot find is a REJECTION, not a skip" {
  # "There is no variable that turns a check off." A gate that cannot execute its own
  # checks must not report green - and must NAME the tool it could not find, rather
  # than failing further down with a message about the receipt. The PATH below holds
  # bash only, because the guard's own #!/usr/bin/env bash must still resolve: a PATH
  # with nothing on it would test env(1), not the guard.
  local pd="$WORK/bash-only-path"
  mkdir -p "$pd"
  ln -sf "$(command -v bash)" "$pd/bash"
  PATH="$pd" run "$GUARD" "$FIX/row-07-honest-docs-only-all-not-triggered"
  [ "$status" -eq 1 ]
  [[ "$output" == *"cannot run:"* ]]
  [[ "$output" == *"minisign"* ]]
  [[ "$output" != *"ACCEPT"* ]]
}

@test "no PR_REVIEW_REPO and no git repository is a REJECTION" {
  # Falling through with REPO unset would run every `git -C ""` against the caller's
  # current directory, so the merge-base boundary would be computed in some other
  # repository entirely - and would still print a class and a reason.
  cd "$WORK"
  PR_REVIEW_REPO="" run "$GUARD" "$FIX/row-07-honest-docs-only-all-not-triggered"
  [ "$status" -eq 1 ]
  [[ "$output" == *"not in a git repository and PR_REVIEW_REPO is unset"* ]]
  [[ "$output" != *"ACCEPT"* ]]
}

@test "S6.1 a positive control that is ACCEPTED fails the whole run" {
  # The control set's own reason for existing. Seeded with an HONEST receipt, the
  # self-review control is accepted - and the guard must refuse to validate anything,
  # because a green verdict from a guard whose controls do not fire is a count of files.
  seed_controls "$WORK/pc3"
  cp "$FIX/row-07-honest-docs-only-all-not-triggered/receipt.intoto.jsonl" \
     "$FIX/row-07-honest-docs-only-all-not-triggered/receipt.intoto.jsonl.minisig" \
     "$FIX/row-07-honest-docs-only-all-not-triggered/findings.sarif" "$WORK/pc3/self-review/"
  PR_REVIEW_POSITIVE_CONTROL_DIR="$WORK/pc3" run "$GUARD" \
    "$FIX/row-07-honest-docs-only-all-not-triggered"
  [ "$status" -eq 1 ]
  [[ "$output" == *"POSITIVE CONTROL FAILED"* ]]
  # ANCHORED: the failure message itself contains the word ACCEPTED, so the usual
  # substring test would pass here for the wrong reason. Only a line that BEGINS with
  # ACCEPT is a receipt this guard accepted. Matched from a herestring, never from a
  # pipe: `producer | grep -q` can return 141 on SIGPIPE despite having matched.
  run grep -c '^ACCEPT' <<<"$output"
  [ "$output" -eq 0 ]
}

@test "S6.1 the schema-depth control is measured against the VENDORED schema" {
  # S6.2 vendors the schemas so the gate does not depend on an external service. Point
  # PR_REVIEW_SCHEMA_DIR at a permissive in-toto schema and control 1 no longer reaches
  # the branch it names: it is still rejected - on the SIGNATURE branch - and still
  # under B1. Only the reason assertion can tell those apart, and the run must stop.
  mkdir -p "$WORK/permissive"
  printf '{"$schema":"https://json-schema.org/draft/2020-12/schema"}\n' \
    > "$WORK/permissive/in-toto-statement-v1.json"
  cp "$REPO_ROOT/schemas/sarif-2.1.0.json" "$WORK/permissive/"
  PR_REVIEW_SCHEMA_DIR="$WORK/permissive" run "$GUARD" \
    "$FIX/row-07-honest-docs-only-all-not-triggered"
  [ "$status" -eq 1 ]
  [[ "$output" == *"POSITIVE CONTROL MISFIRED"* ]]
  run grep -c '^ACCEPT' <<<"$output"
  [ "$output" -eq 0 ]
}

@test "S6.1 EVERY control stops the run when it misfires, not just the first" {
  # Each control is armed by its own `|| exit 1`. A missing one on any single control
  # would let that control misfire and the run carry on to ACCEPT - and the three
  # controls sit at three different depths, so no other test reaches all of them.
  # Each pairing below swaps in a receipt that is rejected on a DIFFERENT branch.
  local pair victim donor n=0
  for pair in "self-review:findings-digest" \
              "findings-digest:cost-missing" \
              "cost-missing:self-review"; do
    victim=${pair%%:*}; donor=${pair##*:}
    n=$((n + 1))
    seed_controls "$WORK/every-$victim"
    cp "$FIX/positive-control/$donor/receipt.intoto.jsonl" \
       "$FIX/positive-control/$donor/receipt.intoto.jsonl.minisig" \
       "$FIX/positive-control/$donor/findings.sarif" "$WORK/every-$victim/$victim/"
    PR_REVIEW_POSITIVE_CONTROL_DIR="$WORK/every-$victim" run "$GUARD" \
      "$FIX/row-07-honest-docs-only-all-not-triggered"
    [ "$status" -eq 1 ] || {
      echo "control $victim seeded with $donor: expected exit 1, got $status"
      echo "$output"; return 1; }
    [[ "$output" == *"POSITIVE CONTROL MISFIRED ($victim)"* ]] || {
      echo "control $victim seeded with $donor did not misfire:"; echo "$output"; return 1; }
    run grep -c '^ACCEPT' <<<"$output"
    [ "$output" -eq 0 ] || {
      echo "control $victim misfired and the run still accepted a receipt"; return 1; }
  done
  [ "$n" -eq 3 ]
}

# =============================================================================
# S3.A DUPLICATION COVERAGE (PRREV-009, from PRREV-007's backtest finding F4)
#
# F4, measured: `duplication_hits` is blind to 48.8% of the diff it was designed for.
# pmat's semantic index is Rust-only, so shell, python and yaml are outside semantic
# reach; and prior art on an UNMERGED SIBLING BRANCH is invisible by construction, so
# PERF-055's prior art was found only because #2742 had merged 17 hours earlier.
#
# The repair has two halves and both are tested here:
#   the RECORD  - a coverage map, a horizon and a needle count in the receipt, with the
#                 guard rejecting an unrecorded claim and refusing PASS over an
#                 unsearched surface (rows 16/17 and the probes below);
#   the MECHANISM - scripts/pr_review_duplication_scan.sh, which actually searches the
#                 non-Rust surface and the sibling branches. A record with no mechanism
#                 behind it is a field that will be filled in by hand with whatever
#                 reads green.
# =============================================================================

@test "row 16 a duplication surface that could not be searched, PASS    RED  B1" {
  assert_row row-16-duplication-surface-unsearched-verdict-pass RED B1 \
    "duplication_coverage could not search [shell, python, config, docs, other] and the verdict is PASS"
}

@test "row 17 the SAME unsearched surface, verdict DEGRADED          GREEN     [discrimination]" {
  # Without this row the rule would punish the honest receipt exactly as hard as the
  # silent one - and a coverage field that costs you a PASS whatever you write in it
  # is a field that learns to stay empty.
  assert_row row-17-duplication-surface-unsearched-verdict-degraded GREEN
}

@test "probe pmat consulted with duplication_hits not an array         RED  B1" {
  assert_probe dup-hits-not-array row-14-complete-gpu-review B1 \
    "duplication_hits is not an array" \
    '.predicate.consultations.pmat.duplication_hits = "nothing like this exists"'
}

@test "probe pmat consulted with NO duplication_coverage at all        RED  B1" {
  # The core of F4. A receipt with hits [] and no coverage map cannot distinguish
  # "searched everywhere and found nothing" from "searched the Rust half".
  assert_probe dup-cov-absent row-14-complete-gpu-review B1 \
    "duplication_coverage is absent" \
    'del(.predicate.consultations.pmat.duplication_coverage)'
}

@test "probe duplication_coverage missing one surface                  RED  B1" {
  # Deleting `shell` is the exact shape of the defect: the surface pmat cannot see is
  # the surface most likely to go unrecorded.
  assert_probe dup-cov-missing-shell row-14-complete-gpu-review B1 \
    "duplication_coverage records no verdict for: shell" \
    'del(.predicate.consultations.pmat.duplication_coverage.shell)'
}

@test "probe duplication_coverage with a method outside the vocabulary RED  B1" {
  # "yes" is not a coverage method. Without the vocabulary check any string reads as
  # coverage, and `"shell": "best effort"` would pass while meaning nothing.
  assert_probe dup-cov-bad-method row-14-complete-gpu-review B1 \
    "holds a method outside { semantic, lexical, none }: shell=yes" \
    '.predicate.consultations.pmat.duplication_coverage.shell = "yes"'
}

@test "probe duplication_horizon absent                                RED  B1" {
  assert_probe dup-horizon-absent row-14-complete-gpu-review B1 \
    "duplication_horizon is absent or is not a non-empty array" \
    'del(.predicate.consultations.pmat.duplication_horizon)'
}

@test "probe duplication_horizon that is a string, not an array        RED  B1" {
  # A scalar horizon is the shape a hand-written receipt takes. It parses, it reads
  # like a statement, and nothing can be counted against it.
  assert_probe dup-horizon-scalar row-14-complete-gpu-review B1 \
    "duplication_horizon is absent or is not a non-empty array" \
    '.predicate.consultations.pmat.duplication_horizon = "HEAD"'
}

@test "probe horizon_branches_total that is not a number               RED  B1" {
  assert_probe dup-horizon-total-nan row-14-complete-gpu-review B1 \
    "must both be whole numbers with 0 <= scanned <= total" \
    '.predicate.consultations.pmat.horizon_branches_total = "many"'
}

@test "probe a FRACTIONAL horizon count, which would skip both rules   RED  B1" {
  # Not a hypothetical. `[ "2.5" -lt 40 ]` is a bash ERROR, not a false comparison: the
  # `if` reads false and BOTH horizon rules below it are skipped, so without the
  # whole-number clause this receipt is ACCEPTED under a PASS despite covering a
  # fraction of 40 branches. Written 2.5 and not 2.0 for a measured reason: jq's number
  # type does not carry the trailing zero, `2.0 | floor == 2.0` is true, and the first
  # draft of this probe was rejected by the CAPPED-horizon rule instead - passing while
  # testing nothing about the branch it names.
  assert_probe dup-horizon-fractional row-14-complete-gpu-review B1 \
    "must both be whole numbers" \
    '.predicate.verdict = "PASS"
     | .predicate.consultations.pmat.horizon_branches_total = 40
     | .predicate.consultations.pmat.horizon_branches_scanned = 2.5'
}

@test "probe horizon_branches_scanned greater than the total           RED  B1" {
  # A sweep cannot cover more branches than exist. The inequality is what makes the
  # denominator load-bearing rather than decorative.
  assert_probe dup-horizon-over row-14-complete-gpu-review B1 \
    "must both be whole numbers with 0 <= scanned <= total" \
    '.predicate.consultations.pmat.horizon_branches_total = 4
     | .predicate.consultations.pmat.horizon_branches_scanned = 9'
}

@test "probe sibling horizon claimed but zero branches scanned         RED  B1" {
  # The `attempted: 0` shape one field over: a sweep that ran over nothing found
  # nothing, and S3.D already calls that DEGRADED rather than clean for mutation.
  assert_probe dup-horizon-vacuous row-14-complete-gpu-review B1 \
    "with horizon_branches_scanned=0 of 6" \
    '.predicate.consultations.pmat.horizon_branches_total = 6
     | .predicate.consultations.pmat.horizon_branches_scanned = 0'
}

@test "probe a CAPPED horizon sweep still reading PASS                 RED  B1" {
  # --max-branches / --horizon since are legitimate; reading PASS over the branches
  # they skipped is not. The skipped branches were not searched, and unsearched is
  # DEGRADED. Measured cost of the uncapped sweep on this repository: 18.6 s over 772
  # branches (evidence/prrev-009/coverage-measurements.txt), so the cap is a choice.
  assert_probe dup-horizon-capped-pass row-14-complete-gpu-review B1 \
    "covered 2 of 40 sibling branches and the verdict is PASS" \
    '.predicate.verdict = "PASS"
     | .predicate.consultations.pmat.horizon_branches_total = 40
     | .predicate.consultations.pmat.horizon_branches_scanned = 2'
}

@test "probe a capped horizon sweep marked DEGRADED                  GREEN     [discrimination]" {
  # The same partial sweep, honestly labelled, is accepted. Without this the rule
  # would forbid the partial sweep instead of forbidding the silence about it.
  local d
  d=$(make_probe dup-horizon-capped-degraded row-14-complete-gpu-review \
      '.predicate.verdict = "DEGRADED"
       | .predicate.consultations.pmat.horizon_branches_total = 40
       | .predicate.consultations.pmat.horizon_branches_scanned = 2')
  run "$GUARD" "$d"
  [ "$status" -eq 0 ] || { echo "$output"; false; }
  [[ "$output" == *"ACCEPT"* ]]
}

@test "probe symbols_searched absent                                   RED  B1" {
  # Record-only is not unenforced. Without the needle count, precision cannot be
  # judged at all: a scan of one needle and a scan of two hundred read identically.
  assert_probe dup-symbols-absent row-14-complete-gpu-review B1 \
    "symbols_searched must be a number" \
    'del(.predicate.consultations.pmat.symbols_searched)'
}

@test "probe a complete duplication block with real hits             GREEN     [discrimination]" {
  # The widest acceptance case for this rule: coverage across every surface, a stated
  # horizon that was swept in full, a needle count, and hits actually recorded. If the
  # guard refuses this it refuses a correct scan.
  local d
  d=$(make_probe dup-complete row-14-complete-gpu-review \
      '.predicate.consultations.pmat.horizon_branches_total = 772
       | .predicate.consultations.pmat.horizon_branches_scanned = 772
       | .predicate.consultations.pmat.symbols_searched = 4
       | .predicate.consultations.pmat.duplication_hits = [
           {"needle":"check_pr_review_wiring.sh","kind":"filename","where":"branch",
            "ref":"origin/feat/prrev-006-wiring","path":"scripts/check_pr_review_wiring.sh",
            "line":0,"method":"lexical"} ]')
  run "$GUARD" "$d"
  [ "$status" -eq 0 ] || { echo "$output"; false; }
  [[ "$output" == *"ACCEPT"* ]]
}

# =============================================================================
# THE MECHANISM: scripts/pr_review_duplication_scan.sh
#
# The rules above make the guard reject a receipt that does not RECORD its coverage.
# They cannot make a scan happen. Without a mechanism the coverage map is a field an
# agent fills in with whatever reads green, which is a worse artifact than the empty
# `duplication_hits: []` it replaces - it is the same silence with a signature on it.
#
# So the scanner is exercised against a purpose-built repository, on both halves of F4
# and in both polarities. The repository is built here rather than in
# tests/fixtures/pr-review/ because it needs an UNMERGED SIBLING BRANCH, and adding one
# to the committed fixture repo would change nothing about rows 1-17 while making their
# provenance harder to read.
# =============================================================================

SCAN="$BATS_TEST_DIRNAME/../scripts/pr_review_duplication_scan.sh"

# make_scan_repo <dir> - main carries a SHELL definition; an unmerged sibling and the
# PR under review both add the same new path. Deterministic identity and dates.
make_scan_repo() {
  local d=$1
  mkdir -p "$d"
  export GIT_AUTHOR_NAME=scanfix GIT_AUTHOR_EMAIL=scan@fixture.invalid
  export GIT_COMMITTER_NAME=scanfix GIT_COMMITTER_EMAIL=scan@fixture.invalid
  export GIT_AUTHOR_DATE="2026-02-02T00:00:00+0000" GIT_COMMITTER_DATE="2026-02-02T00:00:00+0000"
  git -C "$d" init -q -b main .
  git -C "$d" config core.hooksPath /dev/null
  git -C "$d" config commit.gpgsign false
  mkdir -p "$d/scripts" "$d/src"
  # The prior art, in SHELL. pmat's semantic index cannot return this file; that is the
  # whole of F4(a), and it is why the needle must be searched lexically.
  printf '#!/usr/bin/env bash\nrender_band_receipt() {\n  printf receipt\n}\n' \
    > "$d/scripts/existing_helper.sh"
  printf 'baseline\n' > "$d/README.md"
  git -C "$d" add -A && git -C "$d" commit -q -m "M1 baseline: a shell helper"
  git -C "$d" update-ref refs/remotes/origin/main refs/heads/main

  # The unmerged sibling: it already adds the file the PR is about to add.
  git -C "$d" checkout -q -b sibling main
  printf '#!/usr/bin/env bash\nsweep_band_matrix() { :; }\n' > "$d/scripts/shared_helper.sh"
  git -C "$d" add -A && git -C "$d" commit -q -m "S1 sibling: shared_helper.sh, not merged"
  git -C "$d" update-ref refs/remotes/origin/sibling refs/heads/sibling

  # The pull request under review, forked from the same base.
  git -C "$d" checkout -q -b pr main
  printf '#!/usr/bin/env bash\nrender_band_receipt() { :; }\n' > "$d/scripts/shared_helper.sh"
  printf 'pub fn render_band_receipt() {}\n' > "$d/src/newmod.rs"
  git -C "$d" add -A && git -C "$d" commit -q -m "P1 pr: re-implement the helper"
  git -C "$d" checkout -q main
}

# run_symbol_table - the needle-extraction case table. A FUNCTION invoked through
# bats' `run`, because bats runs a test body under `set -e` and every NO-MATCH row
# exits 1 by design: called inline, the first NO-MATCH row aborts the test before a
# single polarity has been checked. Measured, not guessed - that is exactly how this
# failed the first time it ran.
run_symbol_table() {
  local table="$FIX/duplication-symbol-cases.tsv"
  local expect subject want why out rc rows=0 fails=0
  while IFS=$'\t' read -r expect subject want why; do
    case "$expect" in ''|'#'*) continue ;; esac
    rows=$((rows + 1))
    out=$("$SCAN" --extract-symbol "$subject" 2>/dev/null)
    rc=$?
    if [ "$expect" = MATCH ]; then
      if [ "$rc" -ne 0 ]; then
        echo "MISS      expected $want, got no match: $subject"; fails=$((fails + 1)); continue
      fi
      # The NAME is asserted, not merely that something matched: a pattern can match
      # the right line and capture the wrong group, and that is invisible to a
      # match/no-match table.
      if [ "$out" != "$want" ]; then
        echo "WRONG-NAME expected '$want', got '$out': $subject"; fails=$((fails + 1))
      fi
    elif [ "$expect" = NO-MATCH ] && [ "$rc" -eq 0 ]; then
      echo "SPURIOUS  expected no match, got '$out': $subject"; fails=$((fails + 1))
    fi
  done < "$table"
  [ "$rows" -ge 25 ] || { echo "case table has only $rows rows"; return 1; }
  echo "$rows rows checked"
  [ "$fails" -eq 0 ]
}

@test "S3.A the needle extraction matches its case table, both polarities" {
  run run_symbol_table
  [ "$status" -eq 0 ] || { echo "$output"; false; }
}

@test "F4a the scan reaches a SHELL definition, which pmat's index cannot" {
  local d="$WORK/scanrepo-a"
  make_scan_repo "$d"
  local base head out
  base=$(git -C "$d" rev-parse main)
  head=$(git -C "$d" rev-parse pr)
  out=$("$SCAN" --repo "$d" --base "$base" --head "$head" --horizon none) || {
    echo "$out"; false; }
  # The prior art lives in scripts/existing_helper.sh - a .sh file, outside any semantic
  # index this repository has. If this assertion fails, duplication_hits is Rust-only
  # again and F4(a) is back.
  run jq -r '[.duplication_hits[] | select(.path | endswith(".sh"))] | length' <<<"$out"
  [ "$output" -ge 1 ] || { echo "no shell-file hit in: $out"; false; }
  run jq -r '[.duplication_hits[] | select(.needle == "render_band_receipt" and .path == "scripts/existing_helper.sh")] | length' <<<"$out"
  [ "$output" -eq 1 ] || { echo "the shell definition was not attributed: $out"; false; }
}

@test "F4b the scan sees prior art on an UNMERGED sibling branch     [discrimination]" {
  local d="$WORK/scanrepo-b"
  make_scan_repo "$d"
  local base head none all
  base=$(git -C "$d" rev-parse main)
  head=$(git -C "$d" rev-parse pr)

  # A. what S3.A does today: the index cannot see off this branch.
  none=$("$SCAN" --repo "$d" --base "$base" --head "$head" --horizon none)
  run jq -r '[.duplication_hits[] | select(.where == "branch")] | length' <<<"$none"
  [ "$output" -eq 0 ] || { echo "expected no branch hits with --horizon none: $none"; false; }
  run jq -r '.duplication_coverage.sibling_branches' <<<"$none"
  [ "$output" = none ]
  run jq -r '.horizon_branches_scanned' <<<"$none"
  [ "$output" -eq 0 ]

  # B. the repair. SAME diff, SAME script; the horizon is the only variable.
  all=$("$SCAN" --repo "$d" --base "$base" --head "$head")
  run jq -r '[.duplication_hits[] | select(.where == "branch" and .ref == "origin/sibling" and .path == "scripts/shared_helper.sh")] | length' <<<"$all"
  [ "$output" -eq 1 ] || { echo "the sibling branch's prior art was not found: $all"; false; }
  run jq -r '.duplication_coverage.sibling_branches' <<<"$all"
  [ "$output" = lexical ]
  run jq -r '.horizon_branches_total' <<<"$all"
  [ "$output" -eq 1 ]
  run jq -r '.horizon_branches_scanned' <<<"$all"
  [ "$output" -eq 1 ]
}

@test "the scan's own output satisfies every rule the guard enforces on it" {
  # A mechanism whose output the guard would reject is two artifacts that disagree.
  # This is the join between them, and it is asserted rather than assumed.
  local d="$WORK/scanrepo-c"
  make_scan_repo "$d"
  local base head out
  base=$(git -C "$d" rev-parse main)
  head=$(git -C "$d" rev-parse pr)
  out=$("$SCAN" --repo "$d" --base "$base" --head "$head" --rust-semantic)

  run jq -e '(.duplication_hits | type == "array")
             and (["rust","shell","python","config","docs","other","sibling_branches"]
                  - (.duplication_coverage | keys) | length == 0)
             and (.duplication_coverage | to_entries
                  | map(.value as $v | ["semantic","lexical","none"] | index($v) != null) | all)
             and (.duplication_horizon | (type == "array") and (length > 0) and (map(type == "string") | all))
             and (.horizon_branches_total | type == "number")
             and (.horizon_branches_scanned | type == "number")
             and (.horizon_branches_scanned <= .horizon_branches_total)
             and (.symbols_searched | type == "number")' <<<"$out"
  [ "$status" -eq 0 ] || { echo "the scan output would be REJECTED by the guard: $out"; false; }

  # --rust-semantic is a claim about the CALLER, not about this script: without it the
  # honest value is `lexical`, because on its own the scan is a name match.
  run jq -r '.duplication_coverage.rust' <<<"$out"
  [ "$output" = semantic ]
  out=$("$SCAN" --repo "$d" --base "$base" --head "$head")
  run jq -r '.duplication_coverage.rust' <<<"$out"
  [ "$output" = lexical ]
}

@test "a clone with no branch but main records sibling_branches: none, not lexical" {
  # The shallow/CI-checkout case. `git for-each-ref refs/remotes/origin` enumerates what
  # THIS clone has fetched, so a checkout holding only main has a horizon of zero
  # branches - and "swept 0 of 0 in full" satisfies every count rule in the guard while
  # having looked nowhere. The scan degrades the METHOD instead, and the guard's existing
  # `none` may not read PASS rule turns that into DEGRADED.
  local d="$WORK/scanrepo-e"
  make_scan_repo "$d"
  git -C "$d" update-ref -d refs/remotes/origin/sibling
  local out
  out=$("$SCAN" --repo "$d" --base "$(git -C "$d" rev-parse main)" --head "$(git -C "$d" rev-parse pr)")
  run jq -r '.duplication_coverage.sibling_branches' <<<"$out"
  [ "$output" = none ] || { echo "expected none, got '$output': $out"; false; }
  run jq -r '.horizon_refs.local_origin_refs' <<<"$out"
  [ "$output" -eq 1 ]
  # With the sibling ref present the SAME repo records lexical - the discrimination arm,
  # without which "always say none" would read green here.
  make_scan_repo "$WORK/scanrepo-f"
  out=$("$SCAN" --repo "$WORK/scanrepo-f" --base "$(git -C "$WORK/scanrepo-f" rev-parse main)" \
        --head "$(git -C "$WORK/scanrepo-f" rev-parse pr)")
  run jq -r '.duplication_coverage.sibling_branches' <<<"$out"
  [ "$output" = lexical ]
  run jq -r '.horizon_refs.local_origin_refs' <<<"$out"
  [ "$output" -eq 2 ]
}

@test "the scan REFUSES rather than printing a coverage map it did not earn" {
  # A scan that could not run must not print `lexical` for anything. Three ways in:
  local d="$WORK/scanrepo-d"
  make_scan_repo "$d"
  run "$SCAN" --repo "$d" --base "$(git -C "$d" rev-parse main)"
  [ "$status" -eq 1 ]
  [[ "$output" == *"--head is required"* ]]

  run "$SCAN" --repo "$d" --base deadbeefdeadbeefdeadbeefdeadbeefdeadbeef --head "$(git -C "$d" rev-parse pr)"
  [ "$status" -eq 1 ]
  [[ "$output" == *"does not resolve"* ]]

  run "$SCAN" --repo "$d" --base "$(git -C "$d" rev-parse main)" --head "$(git -C "$d" rev-parse pr)" --horizon everything
  [ "$status" -eq 1 ]
  [[ "$output" == *"--horizon must be all, since or none"* ]]
}
