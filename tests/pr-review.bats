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
# The twenty-two committed rows pin twenty-two branches. The guard has more than seventy, and
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
  run "$GUARD" "$FIX/row-07-honest-docs-only-pmat-consulted"
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
    "$FIX/row-07-honest-docs-only-pmat-consulted"
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
    "$FIX/row-07-honest-docs-only-pmat-consulted"
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
    "$FIX/row-07-honest-docs-only-pmat-consulted"
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

@test "row 7  honest docs-only PR, pmat consulted, rest not-triggered  GREEN     [discrimination]" {
  # S6.3 writes this row as "all consultations not-triggered". S3.A and S8.4 both make
  # pmat's trigger UNCONDITIONAL, so the spec contradicts itself and the two normative
  # statements win over the illustrative row. The fixture previously carried
  # `pmat: not-triggered` with a trigger_reason reading "pmat is unconditional;
  # not-triggered is never correct for it" — it stated the rule it exempted, and the
  # guard accepted it. Row 19 is the RED half; this stays the GREEN one, because a
  # docs-only PR must still be accepted.
  assert_row row-07-honest-docs-only-pmat-consulted GREEN
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

# --- PRREV-008: the four defects the backtest measured ------------------------
#
# S9 step 7 is the acceptance test for the whole spec, and it failed on three of its
# four named cases. These six rows are the fixtures the fixes require. Each was RED
# against the previous guard for no reason at all — it ACCEPTED every one of them.

@test "row 16 the diff publishes a ratio, nothing records it           RED  B4" {
  # F1, the worst of the four: `match_comparative` had ONE call site, inside a loop
  # over findings THE REVIEWER WROTE. A blocking class that asks the reviewer whether
  # the reviewer should be blocked is circular. Measured with a signed discrimination
  # pair — identical diff, ACCEPT when the receipt was silent, REJECT when it was not.
  assert_row row-16-comparative-claim-only-in-the-diff RED B4 \
    "the diff publishes a comparative claim on a user-facing surface"
}

@test "row 17 the same ratio, RECORDED with a comparator             GREEN     [discrimination]" {
  # Without this row, "block every PR that names a competitor" reads green — and B4
  # would forbid the honest path it exists to require.
  assert_row row-17-comparative-claim-recorded GREEN
}

@test "row 18 cuda consulted with queries: []                         RED  B1" {
  # F2: the guard rejected the analogous `mutation.attempted: 0` and accepted this.
  # S8 sets vacuous_consultations = 0; it was enforced for one consultation in four.
  assert_row row-18-cuda-consulted-no-queries RED B1 \
    "cuda.status is consulted with queries: []"
}

@test "row 19 pmat not-triggered on a diff changing Rust source       RED  B1" {
  # F3: S3.A says "Trigger: unconditional" and S8.4 repeats it. Every other
  # consultation in this receipt is honest, so exactly one rule can reject it.
  assert_row row-19-pmat-not-triggered-on-a-code-diff RED B1 \
    "consultations.pmat is not-triggered, but S3.A makes pmat unconditional"
}

@test "row 20 mutation not-triggered on a diff changing Rust source   RED  B1" {
  # F2's other half: mutation's emptiness was checked and its trigger was not, so the
  # consultation could be skipped outright by writing three words.
  assert_row row-20-mutation-not-triggered-on-a-code-diff RED B1 \
    "consultations.mutation is not-triggered but S3.D triggers on this diff"
}

@test "row 21 crux not-triggered on a diff publishing a ratio         RED  B1" {
  # S3.C.1 lives under S3.C, so a comparative claim IS a crux trigger. Without this,
  # a reviewer could put crux beyond the reach of every claim rule.
  assert_row row-21-crux-not-triggered-on-a-claim-diff RED B1 \
    "consultations.crux is not-triggered but its S3.C trigger fires on this diff"
}

@test "row 22 the printed ratio fires, the quoted one two lines up does not  RED  B4" {
  # B4's SCOPE, measured rather than assumed. One .rs file carries the SAME ratio twice:
  # first in a plain `//` comment that quotes the withdrawn claim in order to name it,
  # then inside a `format!` a user reads. The comment is FIRST, and the guard rejects on
  # the first match — so a reason naming the `format!` line is the proof that the comment
  # was skipped. If the comment fired, it would be the line quoted back here.
  assert_row row-22-printed-ratio-not-the-quoted-one RED B4 \
    'format!("apr sustains 2.93x Ollama'
  # ...and the half that says it discriminates: the quoted line must not appear at all.
  run "$GUARD" "$FIX/row-22-printed-ratio-not-the-quoted-one"
  [[ "$output" != *"The book published"* ]] || {
    echo "the merely-quoted comment fired; B4 would block a withdrawal with no honest exit"
    echo "$output"; return 1; }
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
  cp "$FIX/row-07-honest-docs-only-pmat-consulted/findings.sarif" "$WORK/sarif-only/"
  run "$GUARD" "$WORK/sarif-only"
  [ "$status" -eq 1 ]
  [[ "$output" == *"receipt.intoto.jsonl is missing"* ]]
}

@test "one bad receipt in a batch fails the whole run" {
  # A per-receipt loop that forgets to accumulate its status reports the LAST result.
  run "$GUARD" "$FIX/row-07-honest-docs-only-pmat-consulted" \
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

@test "S3.C.1 the two spellings the backtest measured as MISSES are asserted" {
  # The table above would still pass if it silently shrank. These two rows are the
  # reason PRREV-008 exists: `36.9x over FasterTransformer` is the spelling
  # APR-PERF-GATE-001 §0.1 uses and #2763 hardened its own guard to catch, and the
  # previous pattern allowed a ZERO-word gap where #2763 measured five. Asserted in
  # the harness's own voice so a future narrowing is loud rather than shorter.
  "$GUARD" --match-comparative '36.9x over FasterTransformer'
  "$GUARD" --match-comparative '2x speedup versus Ollama'
  "$GUARD" --match-comparative '3.2x faster than HuggingFace transformers'
  # ...and the bound is a BOUND: six gap words must not match, or the pattern is not
  # #2763's measured rule any more, it is a wildcard with a competitor list.
  run "$GUARD" --match-comparative 'the 2x speedup we would need on six separate kernels before llama'
  [ "$status" -ne 0 ]
}

@test "the shipped surface B4 scans matches its case table, both polarities" {
  run run_case_table shipped-surface --match-shipped-surface
  [ "$status" -eq 0 ] || { echo "$output"; false; }
}

@test "the .rs line test matches its case table, both polarities" {
  # The path says the file ships; this says the LINE is read. Both halves are needed:
  # over 300 commits of origin/main every comparative claim added to a plain // comment
  # was one this repository was WITHDRAWING, and B4's only remedy is a comparator log.
  run run_case_table rs-published --match-rs-published
  [ "$status" -eq 0 ] || { echo "$output"; false; }
}

@test "B4 does not block a claim it has no honest remedy for" {
  # The two verbatim lines from origin/main that removed docs/ prose and plain comments
  # from B4's scope. ce712eae0 quotes a fabricated ratio in order to ban it; there is no
  # comparator log for a number nobody measured, so a block there can only be satisfied
  # by inventing one. Asserted here rather than left to the tables, because this is the
  # S7 admission rule in force, not a spelling detail.
  run "$GUARD" --match-shipped-surface docs/benchmarking-gate-spec.md
  [ "$status" -ne 0 ]
  run "$GUARD" --match-rs-published '    // #2696: this printed "Performance: 800+ tok/s (2.8x Ollama)"'
  [ "$status" -ne 0 ]
  # ...while the surface the scar was published on still fires.
  run "$GUARD" --match-shipped-surface book/src/tools/apr-cli.md
  [ "$status" -eq 0 ]
  run "$GUARD" --match-rs-published '    println!("851.8 tok/s = 2.93x Ollama");'
  [ "$status" -eq 0 ]
}

@test "S3.C surface trigger matches its case table, both polarities" {
  run run_case_table crux-surface --match-crux-surface
  [ "$status" -eq 0 ] || { echo "$output"; false; }
}

@test "S3.D mutation trigger matches its case table, both polarities" {
  run run_case_table mutation-trigger --match-mutation-trigger
  [ "$status" -eq 0 ] || { echo "$output"; false; }
}

@test "the target suppressor matches its case table, both polarities" {
  # Both polarities are expensive here: a miss reds a line stating a bar, and a
  # spurious match silently EXEMPTS a published claim from B4.
  run run_case_table target --match-target
  [ "$status" -eq 0 ] || { echo "$output"; false; }
}

@test "B4's diff scan does not red the PR that adds its own case table" {
  # The over-reach this scope exists to prevent, asserted rather than assumed: every
  # subject in the comparative table is a banned ratio, and the table itself is a
  # changed file on any PR that edits it. If fixtures were in scope, this guard could
  # never be modified again without a comparator for thirteen invented claims.
  run "$GUARD" --match-shipped-surface tests/fixtures/pr-review/comparative-claim-cases.tsv
  [ "$status" -ne 0 ]
  run "$GUARD" --match-shipped-surface docs/specifications/apr-perf-gate-001.md
  [ "$status" -ne 0 ]
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

@test "every S6.3 row, the contract's owed row, and PRREV-008's seven have a fixture" {
  local n
  n=$(find "$FIX" -maxdepth 1 -type d -name 'row-*' | wc -l)
  [ "$n" -eq 22 ] || { echo "expected 22 row fixtures (14 from S6.3 + row 15 owed by the contract + rows 16-22 from PRREV-008), found $n"; false; }
  local i
  for i in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22; do
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
# One case per validation branch the twenty-two rows do not reach. Each exists because
# the mutation sweep in scripts/mutate-guard.sh found the corresponding mutant ALIVE:
# a rule the guard states and nothing tests. They are named for the branch, not for a
# spec row, because they are not spec rows.
# =============================================================================

# --- artifact presence, shape, and the offline schema gate -------------------

@test "probe findings.sarif absent                                     RED  B1" {
  local d="$WORK/probe-no-sarif"
  mkdir -p "$d"
  cp "$FIX/row-07-honest-docs-only-pmat-consulted/receipt.intoto.jsonl" \
     "$FIX/row-07-honest-docs-only-pmat-consulted/receipt.intoto.jsonl.minisig" "$d/"
  run "$GUARD" "$d"
  [ "$status" -eq 1 ]
  [[ "$output" == *"[B1]"* ]]
  [[ "$output" == *"findings.sarif is missing"* ]]
}

@test "probe receipt holding two JSON records is not JSON Lines        RED  B1" {
  # JSON Lines with two records parses as two Statements. Which one was signed, and
  # which one is the review? A file that cannot answer that is not a receipt.
  local d="$WORK/probe-two-records" src="$FIX/row-07-honest-docs-only-pmat-consulted"
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
  local d="$WORK/probe-receipt-not-json" src="$FIX/row-07-honest-docs-only-pmat-consulted"
  mkdir -p "$d"
  cp "$src/findings.sarif" "$src/receipt.intoto.jsonl.minisig" "$d/"
  printf 'this is not JSON at all' > "$d/receipt.intoto.jsonl"
  run "$GUARD" "$d"
  [ "$status" -eq 1 ]
  [[ "$output" == *"[B1]"* ]]
  [[ "$output" == *"receipt.intoto.jsonl is not parseable JSON"* ]]
}

@test "probe findings.sarif that is not parseable JSON                 RED  B1" {
  local d="$WORK/probe-sarif-not-json" src="$FIX/row-07-honest-docs-only-pmat-consulted"
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
  local d="$WORK/probe-sarif-schema" src="$FIX/row-07-honest-docs-only-pmat-consulted"
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
    run "$GUARD" "$FIX/row-07-honest-docs-only-pmat-consulted"
  [ "$status" -eq 1 ]
  [[ "$output" == *"[B1]"* ]]
  [[ "$output" == *"is absent; an unverifiable signature is not a verified one"* ]]
}

@test "probe the receipt carries no signature at all                   RED  B1" {
  local d="$WORK/probe-unsigned" src="$FIX/row-07-honest-docs-only-pmat-consulted"
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
  assert_probe predicate-type row-07-honest-docs-only-pmat-consulted B1 \
    "predicateType is 'https://slsa.dev/verification_summary/v1'" \
    '.predicateType = "https://slsa.dev/verification_summary/v1"'
}

@test "probe attestation_level claims more than L1-self                RED  B1" {
  # R1: SLSA Build L3 requires an isolated builder the tenant cannot influence. A
  # skill invoked by the authoring agent is self-attestation. A receipt that claims
  # otherwise is the exact enforcement theatre the spec rejects, so it is refused.
  assert_probe attestation-level row-07-honest-docs-only-pmat-consulted B1 \
    "attestation_level is 'SLSA-BUILD-L3'" \
    '.predicate.attestation_level = "SLSA-BUILD-L3"'
}

@test "probe head_sha is absent                                        RED  B1" {
  assert_probe head-absent row-07-honest-docs-only-pmat-consulted B1 \
    "predicate.head_sha is absent" 'del(.predicate.head_sha)'
}

@test "probe base_sha is absent                                        RED  B1" {
  assert_probe base-absent row-07-honest-docs-only-pmat-consulted B1 \
    "predicate.base_sha is absent" 'del(.predicate.base_sha)'
}

@test "probe the subject digest is not the head the predicate reviews  RED  B1" {
  # in-toto binds the attestation to subject[].digest. If the predicate reviews a
  # different commit from the one the statement is ABOUT, the signature attests to
  # a review of something else.
  assert_probe subject-digest row-07-honest-docs-only-pmat-consulted B1 \
    "the sha1 digest of subject 0 is" \
    '.subject[0].digest.sha1 = "0000000000000000000000000000000000000000"'
}

@test "probe verdict outside the four defined values                   RED  B1" {
  assert_probe verdict-outside row-07-honest-docs-only-pmat-consulted B1 \
    "verdict 'PROBABLY-FINE' is outside" '.predicate.verdict = "PROBABLY-FINE"'
}

@test "probe findings_ref.path points somewhere else                   RED  B1" {
  assert_probe findings-ref-path row-07-honest-docs-only-pmat-consulted B1 \
    "findings_ref.path is 'somewhere-else.sarif'" \
    '.predicate.findings_ref.path = "somewhere-else.sarif"'
}

@test "probe author_actor.id is absent                                 RED  B1" {
  assert_probe author-absent row-07-honest-docs-only-pmat-consulted B1 \
    "author_actor.id is absent" 'del(.predicate.author_actor.id)'
}

@test "probe reviewer_actor.id is absent                               RED  B1" {
  # S5's separation cannot be checked against an absent reviewer, and an absent
  # reviewer is indistinguishable from no review at all.
  assert_probe reviewer-absent row-07-honest-docs-only-pmat-consulted B1 \
    "reviewer_actor.id is absent" 'del(.predicate.reviewer_actor.id)'
}

# --- the diff boundary -------------------------------------------------------

@test "probe head_sha does not resolve in the repository               RED  B1" {
  assert_probe head-unresolvable row-07-honest-docs-only-pmat-consulted B1 \
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
  d=$(make_probe orphan-head row-07-honest-docs-only-pmat-consulted \
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
  assert_probe status-absent row-07-honest-docs-only-pmat-consulted B1 \
    "consultations.crux.status is absent" 'del(.predicate.consultations.crux.status)'
}

@test "probe a consultation status outside the three-state vocabulary  RED  B1" {
  # "skipped" is the word that hides the difference S3.0 exists to make visible.
  assert_probe status-invalid row-07-honest-docs-only-pmat-consulted B1 \
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

# --- every consultation's emptiness (PRREV-008 F2) ---------------------------
#
# The audit that opened PRREV-008: only cuda's trigger was recomputed from the diff and
# only mutation's emptiness was checked, so NO consultation had both. Rows 18-21 pin the
# four missing halves at the top level; these probes pin the sub-branches underneath,
# each of which is a separate `reject` site and therefore a separate mutant.

@test "probe pmat consulted with duplication_hits absent              RED  B1" {
  # S3.A calls duplication_hits "the highest-EV field in the receipt" — PERF-055 nearly
  # re-implemented ~7,200 lines across 46 files that already existed. An ABSENT key and
  # an EMPTY array are "did not look" and "looked and found nothing", which is exactly
  # the distinction S3.0 exists to make impossible to blur.
  assert_probe pmat-no-duplication row-14-complete-gpu-review B1 \
    "these S3.A outputs are absent or are not arrays: duplication_hits" \
    'del(.predicate.consultations.pmat.duplication_hits)'
}

@test "probe pmat consulted with complexity_delta not an array        RED  B1" {
  # "none" is the word a reviewer reaches for when there is nothing to report, and it
  # is indistinguishable from the field never having been produced.
  assert_probe pmat-scalar-delta row-14-complete-gpu-review B1 \
    "absent or are not arrays: complexity_delta" \
    '.predicate.consultations.pmat.complexity_delta = "none"'
}

@test "probe a cuda query whose result is outside the vocabulary      RED  B1" {
  # S3.B admits exactly two outcomes: a citation, or a NAMED query that returned
  # nothing. A third word puts "the docs said nothing" and "I did not ask" back into
  # one field, one level below the status row 18 pins.
  assert_probe cuda-query-result row-14-complete-gpu-review B1 \
    'outside { found, no-authority-found }' \
    '.predicate.consultations.cuda.queries[0].result = "inconclusive"'
}

@test "probe a cuda query recording found with no excerpt_sha256      RED  B1" {
  # A `found` with nothing to verify against is an assertion wearing the mark of a
  # citation — S1.1, applied to the consultation record rather than to the finding.
  assert_probe cuda-found-no-digest row-14-complete-gpu-review B1 \
    "records result found with no excerpt_sha256" \
    'del(.predicate.consultations.cuda.queries[0].excerpt_sha256)'
}

@test "probe crux consulted with no surfaces key at all               RED  B1" {
  assert_probe crux-no-surfaces row-14-complete-gpu-review B1 \
    "these S3.C outputs are absent or are not arrays: surfaces" \
    'del(.predicate.consultations.crux.surfaces)'
}

@test "probe crux consulted over nothing at all                       RED  B1" {
  # The vacuous-pass shape: `pv lint <FILE>` returning PASS over zero contracts, in
  # another costume. A crux run that looked at no surface and found no claim has the
  # same artifact as one that was never run.
  assert_probe crux-consulted-over-nothing row-14-complete-gpu-review B1 \
    "consulted with no surfaces and no comparative claims" \
    '.predicate.consultations.crux.surfaces = []
     | .predicate.consultations.crux.comparative_claims = []'
}

@test "probe crux_coverage outside its vocabulary                     RED  B1" {
  # S3.C: `crux_coverage: none` — no contract covers this surface — "is itself a
  # finding". A blank or an invented word is how that finding goes unwritten.
  assert_probe crux-coverage-vocab row-14-complete-gpu-review B1 \
    "crux.crux_coverage is 'partial', outside { covered, none }" \
    '.predicate.consultations.crux.crux_coverage = "partial"'
}

@test "probe gap_effect outside its vocabulary                        RED  B1" {
  assert_probe crux-gap-vocab row-14-complete-gpu-review B1 \
    "crux.gap_effect is 'unknown', outside { closes, widens, none }" \
    '.predicate.consultations.crux.gap_effect = "unknown"'
}

@test "probe mutation killed exceeds attempted                        RED  B1" {
  # guard_mutation_score is killed / attempted and S8 fixes it at one with no ratchet.
  # A score ABOVE one is not a stricter guard, it is a miscount, and the number every
  # other verdict rests on is read from these two fields.
  assert_probe mutation-score-above-one row-14-complete-gpu-review B1 \
    "killed=99 of attempted=37" \
    '.predicate.consultations.mutation.killed = 99'
}

@test "probe mutation survivors do not match the arithmetic           RED  B1" {
  # S3.D: "Surviving mutants are recorded with mutant, file, line, killed: false."
  # Seven survivors and an empty list makes the only actionable half of the record
  # unfalsifiable — a rule the guard states and nothing could check.
  assert_probe mutation-survivors-miscount row-14-complete-gpu-review B1 \
    "7 mutant/s survived, but survivors[] holds 0" \
    '.predicate.consultations.mutation.killed = 30'
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
  PATH="$pd" run "$GUARD" "$FIX/row-07-honest-docs-only-pmat-consulted"
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
  PR_REVIEW_REPO="" run "$GUARD" "$FIX/row-07-honest-docs-only-pmat-consulted"
  [ "$status" -eq 1 ]
  [[ "$output" == *"not in a git repository and PR_REVIEW_REPO is unset"* ]]
  [[ "$output" != *"ACCEPT"* ]]
}

@test "S6.1 a positive control that is ACCEPTED fails the whole run" {
  # The control set's own reason for existing. Seeded with an HONEST receipt, the
  # self-review control is accepted - and the guard must refuse to validate anything,
  # because a green verdict from a guard whose controls do not fire is a count of files.
  seed_controls "$WORK/pc3"
  cp "$FIX/row-07-honest-docs-only-pmat-consulted/receipt.intoto.jsonl" \
     "$FIX/row-07-honest-docs-only-pmat-consulted/receipt.intoto.jsonl.minisig" \
     "$FIX/row-07-honest-docs-only-pmat-consulted/findings.sarif" "$WORK/pc3/self-review/"
  PR_REVIEW_POSITIVE_CONTROL_DIR="$WORK/pc3" run "$GUARD" \
    "$FIX/row-07-honest-docs-only-pmat-consulted"
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
    "$FIX/row-07-honest-docs-only-pmat-consulted"
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
      "$FIX/row-07-honest-docs-only-pmat-consulted"
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
