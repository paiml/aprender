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

  # THE CONTROL CACHE IS SCOPED TO THIS RUN AND NO OTHER. It lives under BATS_FILE_TMPDIR
  # -- unpredictable, ours, and gone when the file finishes. It was a fixed path under
  # $TMPDIR, which on a runner this organisation shares across repositories meant any
  # process on the box could compute the key from public files and seed a passing
  # transcript, and meant one job's entry could be consumed by another job whose
  # environment the key does not describe. Unset, the guard simply has no cache.
  export PR_REVIEW_PC_CACHE_DIR="$BATS_FILE_TMPDIR/pc-cache"
  mkdir -m 700 -p "$PR_REVIEW_PC_CACHE_DIR"
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

@test "row 38 pmat unreachable, no transport probed at all             RED       [andon]" {
  # OPERATOR RULING 2026-09-01: "'pmat doesn't work' is never accepted - toyota way."
  # Row 6 used to carry exactly this receipt and was GREEN. On the box that produced its
  # "pmat MCP server: ConnectionRefused" message, `pmat --mode mcp` listed 19 tools and
  # `pmat query` loaded 84,919 functions across 10,136 files - so what had failed was one
  # transport on a hand-started, unsupervised HTTP server, not the source. S3.A makes pmat
  # unconditional and pmat is the one arm with TWO independent transports, so `unreachable`
  # is a claim about both and the probe of each must be recorded. Row 6 keeps the GREEN
  # half: an unreachable pmat that EARNS the claim is still accepted.
  assert_row row-38-pmat-unreachable-no-transport-probe RED
}

@test "row 39 pmat unreachable, only one transport probed              RED       [andon]" {
  # A transport nobody tried is not a transport that failed. The required set is a
  # WHITELIST - a receipt may probe more, never fewer, and renaming a probe to dodge the
  # requirement lands here too rather than in the good bucket (S13's forgery lesson: every
  # clause that survived was a whitelist, every clause that fell was a blacklist).
  assert_row row-39-pmat-unreachable-one-transport-only RED
}

@test "row 40 pmat unreachable but the CLI recorded no error           RED       [andon]" {
  # One reachable transport refutes `unreachable` outright, and S3.A's consultation is
  # then owed over it. An empty-string error lands here too: a probe that recorded no
  # failure SUCCEEDED, and "did not record" must not be readable as "failed".
  assert_row row-40-pmat-unreachable-but-cli-worked RED
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
  # F6, asserted in the harness's own voice rather than left to the table, because this
  # is the exact path da069a25f published to and a table can silently shrink.
  run "$GUARD" --match-shipped-surface book/src/examples/showcase-benchmark.md
  [ "$status" -eq 0 ]
  # ...and the exemption is book/, NOT examples/. Same directory name, a cargo example
  # target, still out of scope. One variable between the two lines.
  run "$GUARD" --match-shipped-surface crates/aprender-core/examples/demo.rs
  [ "$status" -ne 0 ]
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
  [ "$n" -eq 43 ] || { echo "expected 43 row fixtures (14 from S6.3 + row 15 owed by the contract + rows 16-22 from PRREV-008 + rows 23-24 from PRREV-009 + rows 25-26 from PRREV-012/F6 + rows 27-35 from PRREV-015/S3.E + rows 36-37 from PRREV-020/S3.E.4 + rows 38-40 from the pmat transport probes + rows 41-43 from PRREV-023/S4.2), found $n"; false; }
  local i
  for i in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43; do
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

@test "probe a survivors field that is a STRING is not an empty one       RED  B1" {
  # PRREV-019. `[ .survivors[]? ] | length` is 0 over a string: jq's `?` suppresses the
  # TYPE ERROR, not the value. So a receipt confessing twelve survivors satisfied the
  # arithmetic above (37 attempted, 37 killed, "0" survivors) and this guard accepted it.
  # An adversarial verifier then walked it through S13's arm script, which read the same
  # field with the same idiom.
  assert_probe mutation-survivors-not-a-list row-14-complete-gpu-review B1 \
    "survivors is a string, not a list" \
    '.predicate.consultations.mutation.survivors = "12 survived, shipping anyway"'
}

@test "probe an ABSENT survivors field is 'not recorded', never 'none'      RED  B1" {
  # The S3.0 half of the same defect: the field's absence read as an empty list.
  assert_probe mutation-survivors-absent row-14-complete-gpu-review B1 \
    "survivors is absent, not a list" \
    'del(.predicate.consultations.mutation.survivors)'
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

@test "row 23 a duplication surface that could not be searched, PASS    RED  B1" {
  assert_row row-23-duplication-surface-unsearched-verdict-pass RED B1 \
    "duplication_coverage could not search [shell, python, config, docs, other] and the verdict is PASS"
}

@test "row 25 a ratio published under book/src/examples/               RED  B4" {
  # F6, and it is row 16 with ONE directory changed. Until F6 the two verdicts diverged:
  # row 16 RED, this one ACCEPTED, because `match_shipped_surface` excluded */examples/* —
  # a cargo target-layout rule — and applied to the book that removed book/src/examples/,
  # 153 of 441 published pages, 34.7%, every one of them in SUMMARY.md. da069a25f
  # published `851.8 tok/s = 2.93x Ollama` into exactly that directory, and B4 fired ZERO
  # times over it. The publication the spec is written about, accepted by the gate.
  assert_row row-25-examples-page-publishes-a-ratio RED B4 \
    "the diff publishes a comparative claim on a user-facing surface"
  # The PATH is asserted, not merely the class: a rejection quoting some other file would
  # be this row passing for a reason that has nothing to do with F6.
  run "$GUARD" "$FIX/row-25-examples-page-publishes-a-ratio"
  [[ "$output" == *"book/src/examples/showcase-benchmark.md"* ]] || {
    echo "rejected, but not on the examples/ page — F6 is not what made this RED"
    echo "$output"; return 1; }
}

@test "row 26 the same examples/ page, ratio RECORDED               GREEN     [discrimination]" {
  # Without this, "block every book page under examples/" reads green and F6 would have
  # widened B4 into a rule with no honest exit. The exit is row 17's, one directory over.
  assert_row row-26-examples-page-ratio-recorded GREEN
}

@test "row 24 the SAME unsearched surface, verdict DEGRADED          GREEN     [discrimination]" {
  # Without this row the rule would punish the honest receipt exactly as hard as the
  # silent one - and a coverage field that costs you a PASS whatever you write in it
  # is a field that learns to stay empty.
  assert_row row-24-duplication-surface-unsearched-verdict-degraded GREEN
}

@test "probe pmat consulted with duplication_hits not an array         RED  B1" {
  # PRREV-011 merge note: PRREV-008 widened this branch from duplication_hits alone to
  # all four S3.A outputs, so the reason text now NAMES the offending field inside a
  # list. The assertion is re-anchored on "not arrays: duplication_hits" rather than
  # relaxed to B1 alone: the class on its own would also be satisfied by any of the
  # other thirty B1 branches, and an assertion that does not EXCLUDE an outcome is the
  # 0.63.0 hansei defect. The substring below is produced only when duplication_hits is
  # the FIRST offending field, which is the mutation this probe applies.
  assert_probe dup-hits-not-array row-14-complete-gpu-review B1 \
    "not arrays: duplication_hits" \
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

@test "probe duplication_coverage with no merge_base_to_main region     RED  B1" {
  # F7. `merge-base..origin/main` is the region #2781's prior art actually sat in — one
  # commit, 46 files, 11 of them #2742's perf_gate work — and before F7 the coverage map
  # had no key for it at all. An absent key is not a small omission: it is the one state
  # S3.0 forbids, because nothing distinguishes it from a searched-and-empty region.
  assert_probe dup-cov-no-mergebase row-14-complete-gpu-review B1 \
    "duplication_coverage records no verdict for: merge_base_to_main" \
    'del(.predicate.consultations.pmat.duplication_coverage.merge_base_to_main)'
}

@test "probe the merge_base_to_main region unsearched under a PASS     RED  B1" {
  # The rule rows 23/24 apply to a language surface, applied to a REF region. It needs no
  # new branch in the guard — merge_base_to_main is a coverage key like any other, so the
  # existing `none may not read PASS` rule reaches it. Asserted here rather than assumed,
  # because "it should be covered by the existing rule" is how a rule comes to cover
  # nothing.
  # row 07, not row 14: row 14's verdict is FINDINGS, and a probe built on it would be
  # ACCEPTED for a reason that has nothing to do with the rule under test. The base row
  # has to be a GREEN receipt whose verdict is actually PASS, or the polarity is fake.
  assert_probe dup-mergebase-none-pass row-07-honest-docs-only-pmat-consulted B1 \
    "duplication_coverage could not search [merge_base_to_main] and the verdict is PASS" \
    '.predicate.consultations.pmat.duplication_coverage.merge_base_to_main = "none"'
}

@test "probe the SAME unsearched region, verdict DEGRADED            GREEN     [discrimination]" {
  # Without this arm, "reject every receipt that admits a blind region" reads green, and
  # the honest receipt is punished exactly as hard as the silent one — which is how a
  # coverage field learns to stay empty.
  local d
  d=$(make_probe dup-mergebase-none-degraded row-07-honest-docs-only-pmat-consulted \
      '.predicate.consultations.pmat.duplication_coverage.merge_base_to_main = "none"
       | .predicate.verdict = "DEGRADED"')
  run "$GUARD" "$d"
  [ "$status" -eq 0 ] || { echo "$output"; false; }
  [[ "$output" == *"ACCEPT"* ]]
}

@test "probe duplication_horizon naming only two of its three regions  RED  B1" {
  # F7's other half. The horizon used to be built from the METHOD, so a region that was
  # not swept was simply ABSENT — and the pre-F7 receipt read
  # ["HEAD","refs/remotes/origin/* unmerged into origin/main"] under a PASS while a
  # 46-file region sat outside both. The horizon states which regions EXIST; the coverage
  # map states which were SEARCHED. Neither is inferable from the other.
  assert_probe dup-horizon-two-regions row-14-complete-gpu-review B1 \
    "duplication_horizon names no region for: merge_base_to_main" \
    '.predicate.consultations.pmat.duplication_horizon =
       ["head=HEAD","siblings=refs/remotes/origin/* unmerged into origin/main"]'
}

@test "probe the pre-F7 horizon spelling, verbatim                     RED  B1" {
  # The exact array every receipt in this epic carried before F7, quoted rather than
  # paraphrased: an unlabelled ["HEAD", ...] names none of the three components, so it is
  # rejected for ALL THREE. If a future edit made the labels optional, this row goes green
  # and the regression is loud.
  assert_probe dup-horizon-pre-f7 row-14-complete-gpu-review B1 \
    "duplication_horizon names no region for: head, siblings, merge_base_to_main" \
    '.predicate.consultations.pmat.duplication_horizon =
       ["HEAD","refs/remotes/origin/* unmerged into origin/main"]'
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

# land_prior_art_on_main <dir> - main advances PAST the fork, carrying the prior art.
# This is the ordinary shape F7 is about: your branch is a day behind and someone merged
# the thing you were about to write. #2781 against #2742, in miniature.
land_prior_art_on_main() {
  local d=$1
  export GIT_AUTHOR_DATE="2026-02-03T00:00:00+0000" GIT_COMMITTER_DATE="2026-02-03T00:00:00+0000"
  git -C "$d" checkout -q main
  printf '#!/usr/bin/env bash\nrender_band_receipt() { :; }\n' > "$d/scripts/landed_helper.sh"
  git -C "$d" add -A && git -C "$d" commit -q -m "M2 the prior art LANDS on main after the fork"
  git -C "$d" update-ref refs/remotes/origin/main refs/heads/main
}

@test "F7 the scan sees prior art that LANDED on main after the merge base [discrimination]" {
  local d="$WORK/scanrepo-g"
  make_scan_repo "$d"
  land_prior_art_on_main "$d"
  local head base all none
  head=$(git -C "$d" rev-parse pr)
  # git merge-base, NOT origin/main and NOT GitHub's baseRefOid. Reading the base from
  # baseRefOid is what made PRREV-007's #2781 result unreproducible outside its own
  # worktree: baseRefOid was main's TIP at merge time, which is a descendant of the fork,
  # so the blind region collapsed to nothing and the prior art looked reachable.
  base=$(git -C "$d" merge-base refs/remotes/origin/main "$head")
  [ "$base" != "$(git -C "$d" rev-parse refs/remotes/origin/main)" ] || {
    echo "the fixture is degenerate: main did not advance past the fork"; false; }

  # A. the pre-F7 horizon. The region is not swept, and — this is the whole finding —
  #    the receipt did not even NAME it, so [] read as "there is nothing like this".
  none=$("$SCAN" --repo "$d" --base "$base" --head "$head" --horizon none)
  run jq -r '[.duplication_hits[] | select(.where == "main")] | length' <<<"$none"
  [ "$output" -eq 0 ] || { echo "expected no main-region hits with --horizon none: $none"; false; }
  run jq -r '.duplication_coverage.merge_base_to_main' <<<"$none"
  [ "$output" = none ] || { echo "an unswept region must read none, got '$output': $none"; false; }
  # ...and it is NAMED even when it was not searched. That is F7's rule: the horizon says
  # which regions exist, the coverage map says which were reached.
  run jq -r '[.duplication_horizon[] | select(startswith("merge_base_to_main="))] | length' <<<"$none"
  [ "$output" -eq 1 ] || { echo "the unsearched region is absent from the horizon: $none"; false; }

  # B. the repair. SAME diff, SAME script, the horizon is the only variable.
  all=$("$SCAN" --repo "$d" --base "$base" --head "$head")
  run jq -r '[.duplication_hits[] | select(.where == "main" and .ref == "origin/main" and .path == "scripts/landed_helper.sh" and .needle == "render_band_receipt")] | length' <<<"$all"
  [ "$output" -eq 1 ] || { echo "the prior art that landed on main was not found: $all"; false; }
  run jq -r '.duplication_coverage.merge_base_to_main' <<<"$all"
  [ "$output" = lexical ]
  run jq -r '.merge_base_to_main_files' <<<"$all"
  [ "$output" -eq 1 ] || { echo "the region's denominator is wrong: $all"; false; }
}

@test "F7 the main-region sweep is SCOPED to the region, not to all of origin/main" {
  # The grep takes a rev, not a diff, so without the region filter every hit already
  # visible on HEAD would be counted a second time under a different `where` — and the
  # hits_total / symbols_searched ratio the scan reports about itself would stop being
  # judgeable. scripts/existing_helper.sh is present at the merge base, so it is a HEAD
  # hit and must NOT also appear as a main-region hit.
  local d="$WORK/scanrepo-h"
  make_scan_repo "$d"
  land_prior_art_on_main "$d"
  local head base all
  head=$(git -C "$d" rev-parse pr)
  base=$(git -C "$d" merge-base refs/remotes/origin/main "$head")
  all=$("$SCAN" --repo "$d" --base "$base" --head "$head")
  run jq -r '[.duplication_hits[] | select(.where == "main" and .path == "scripts/existing_helper.sh")] | length' <<<"$all"
  [ "$output" -eq 0 ] || { echo "a file unchanged since the merge base was double-counted: $all"; false; }
  run jq -r '[.duplication_hits[] | select(.where == "HEAD" and .path == "scripts/existing_helper.sh")] | length' <<<"$all"
  [ "$output" -eq 1 ] || { echo "the HEAD hit disappeared: $all"; false; }
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
             and (["rust","shell","python","config","docs","other","sibling_branches","merge_base_to_main"]
                  - (.duplication_coverage | keys) | length == 0)
             and (.duplication_coverage | to_entries
                  | map(.value as $v | ["semantic","lexical","none"] | index($v) != null) | all)
             and (.duplication_horizon | (type == "array") and (length > 0) and (map(type == "string") | all))
             and ((["head","siblings","merge_base_to_main"]
                   - [ .duplication_horizon[] | (capture("^(?<k>[a-z_]+)=") | .k)? ]) | length == 0)
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

# =============================================================================
# S3.E — THE FOURTH-VENDOR ARM (PRREV-015).
#
# S3.A..S3.D consult SOURCES. S3.E consults a different REVIEWING AGENT, from a
# different vendor and model family, in its own process. That is a materially stronger
# form of S5's author/reviewer separation than a second prompt of the same model: S5
# cites Huang et al. (ICLR'24) on self-preference bias and on intrinsic self-correction
# degrading reasoning, and neither result is escaped by asking one family twice.
#
# The arm is ADVISORY and every test below is about the HONESTY OF THE RECORD, never
# about what agy said. S7's admission rule admits a class to the blocking tier only
# while its measured precision on the rolling sample is >= 90%; S3.E has zero samples,
# so there is nothing for that rule to apply and the arm cannot block. What the guard
# may refuse is a receipt that MISDESCRIBES what agy did — which is every rule here.
# =============================================================================

@test "row 27 a 2.0.0 receipt written before the arm existed        GREEN     [discrimination]" {
  # THE ROW THAT KEEPS THE VERSION GATE HONEST IN BOTH DIRECTIONS. Without it,
  # "require an antigravity block on every receipt ever written" reads green — and the
  # only way to make this repository's one real receipt pass again
  # (evidence/pr-review/2795/f5fe1479.../, skill_version 2.0.0, four consultations)
  # would be to back-fill a block describing a consultation nobody performed. That is
  # the never-ran-Ollama shape with a JSON schema in front of it.
  assert_row row-27-legacy-2-0-0-receipt-has-no-arm-e GREEN
}

@test "row 28 a 2.1.0 receipt that owes the arm and omits it           RED  B1" {
  # Row 27 with ONE field changed. S3.E's trigger is unconditional, so at 2.1.0 an
  # absent block is the consultation missing, not "not applicable" — and S3.0's whole
  # subject is that an absent record and an empty one must not be the same artifact.
  assert_row row-28-arm-e-owed-and-absent RED B1 \
    "owes the S3.E antigravity consultation and consultations.antigravity is absent"
}

@test "row 29 agy UNAVAILABLE, verdict PASS                            RED  B1" {
  # S3.0 row 3 in the fifth arm. agy fails SLOWLY as readily as fast — --print-timeout
  # defaults to 5m and a repository-scale review needs more — so a timeout is
  # `unavailable`, never a run that found nothing. Those two are otherwise the same
  # artifact, which is precisely what S3.0 exists to make impossible.
  assert_row row-29-arm-e-unavailable-verdict-pass RED B1 \
    "consultations.antigravity is unreachable but the verdict is PASS"
}

@test "row 30 the SAME unavailable agy, verdict DEGRADED             GREEN     [discrimination]" {
  # Without it, "refuse every receipt whose agy did not run" reads green, and the arm
  # punishes the honest DEGRADED exactly as hard as the silent PASS — which is how an
  # unavailability field learns to stay empty. This is also the arm's intended
  # behaviour on a box with no agy installed: DEGRADED proceeds on a feature branch.
  assert_row row-30-arm-e-unavailable-verdict-degraded GREEN
}

@test "row 31 agy consulted having invoked nothing                     RED  B1" {
  # S8's fourth zero — vacuous_consultations = 0 — in the fifth arm. The same artifact
  # as mutation.attempted=0 (row 2) and cuda.queries=[] (row 18), and the same shape as
  # `pv lint <FILE>` returning PASS over zero contracts.
  assert_row row-31-arm-e-consulted-attempted-zero RED B1 \
    "antigravity.status is consulted with attempted=0"
}

@test "row 32 an agy finding claiming a BLOCKING class                 RED  B1" {
  # S7's admission rule: >= 90% measured precision on the rolling sample. S3.E has
  # ZERO samples, so the rule that governs the tier has nothing to apply and the arm
  # cannot be admitted to it.
  assert_row row-32-arm-e-finding-claims-a-blocking-class RED B1 \
    "S3.E is advisory until 30 samples exist"
  # THE REASON IS LOAD-BEARING AND WAS MEASURED, NOT ASSUMED. The first build of this
  # fixture marked the agy finding `asserted`, and it was rejected by S1's OLDER rule —
  # "an asserted claim never blocks" — two hundred lines earlier. Same class, same exit
  # code, and the row pinned NOTHING: drop S3.E's rule and it stays red on S1's. The
  # finding is `measured` now, which is also the honest mark for an agent that ran its
  # own commands, and the assertion below excludes the neighbouring branch by name.
  run "$GUARD" "$FIX/row-32-arm-e-finding-claims-a-blocking-class"
  [[ "$output" != *"an asserted claim never blocks"* ]] || {
    echo "row 32 was rejected by S1's asserted rule, not by S3.E's advisory rule:"
    echo "$output"; return 1; }
}

@test "row 33 the SAME agy finding, marked advisory                  GREEN     [discrimination]" {
  # Without it, "refuse every receipt carrying an agy finding" reads green — and the
  # arm becomes a rule whose only satisfiable behaviour is to find nothing, which is
  # the opposite of why a second vendor is being consulted at all. One token differs
  # from row 32.
  assert_row row-33-arm-e-finding-advisory GREEN
}

@test "row 34 agy declared not-triggered                               RED  B1" {
  # Row 19's rule (pmat: not-triggered) for the fifth arm, and STRICTER: pmat's
  # illegality needed a code file in the diff, S3.E's needs nothing, because there is
  # no diff shape a second opinion is not owed on. The head here is the DOCS-ONLY one,
  # which is the hardest case for that claim and therefore the right one to pin it.
  assert_row row-34-arm-e-not-triggered RED B1 \
    "consultations.antigravity is not-triggered, but S3.E's trigger is unconditional"
}

@test "probe skill_version absent                                      RED  B1" {
  # The version selects the rule set the receipt is judged by (ARM_E_MIN_VERSION), so
  # a receipt that omits it cannot be judged against any. Before S3.E nothing read the
  # field at all and it was decoration.
  assert_probe skill-version-absent row-07-honest-docs-only-pmat-consulted B1 \
    "predicate.skill_version is absent" \
    'del(.predicate.skill_version)'
}

@test "probe a 2.0.0 receipt that CARRIES an antigravity block is still checked RED B1" {
  # THE VERSION GATE MUST NOT CREATE AN UNCHECKED FIELD. It exists to spare the honest
  # historical receipt, not to open a lane where anything can be written under an old
  # version number. `arm_e_required` and `arm_e_present` are two separate facts in the
  # guard for exactly this reason: a block that is present is validated in full
  # whatever the declared version says.
  assert_probe arm-e-block-at-2-0-0 row-27-legacy-2-0-0-receipt-has-no-arm-e B1 \
    "antigravity.status is consulted with attempted=0" \
    '.predicate.consultations.antigravity = {"status":"consulted","attempted":0,"agy_version":"agy 1.1.22","binary_path":"/home/noah/.local/bin/agy","model_family":"google/antigravity","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2},"reverified_by_primary":false,"divergence":{"agreed":0,"agy_only":0,"primary_only":0,"contradicted":0},"findings":[]}'
}

@test "probe antigravity status outside its vocabulary                 RED  B1" {
  # The status vocabulary is applied over a LIST and antigravity joins it rather than
  # getting a private copy of the rule. Two implementations of one rule drift, and each
  # stays green against its own copy — D8, in the guard that implements F4.
  assert_probe arm-e-bad-status row-07-honest-docs-only-pmat-consulted B1 \
    "consultations.antigravity.status is 'ran', outside { consulted, not-triggered, unreachable }" \
    '.predicate.consultations.antigravity.status = "ran"'
}

@test "probe antigravity attempted that is not a count                 RED  B1" {
  # "once" is not a number of invocations. Without the type check the comparison below
  # it errors, the `if` reads false, and the vacuity rule is SKIPPED — a guard failing
  # open on a value its own type check admitted, which is the class this file exists
  # to remove.
  assert_probe arm-e-attempted-not-a-count row-07-honest-docs-only-pmat-consulted B1 \
    "antigravity.status is consulted but attempted is 'once'" \
    '.predicate.consultations.antigravity.attempted = "once"'
}

@test "probe antigravity with no recorded agy version                  RED  B1" {
  # This repository has had four `apr` binaries coexist, and a bare invocation resolve
  # to a 26-day-old one. `agy_version` and `binary_path` are the OUTPUT of an explicit
  # resolution, which is the opposite of a hardcoded path: it is provenance, recorded
  # per run, and it is what makes a later precision sample attributable to a build.
  assert_probe arm-e-no-version row-07-honest-docs-only-pmat-consulted B1 \
    "S3.E identity fields are absent or empty: agy_version" \
    'del(.predicate.consultations.antigravity.agy_version)'
}

@test "probe antigravity with no recorded model family                 RED  B1" {
  # `model_family` is what makes "cross-vendor" a CHECKABLE claim rather than an
  # asserted one. Without it the arm's entire justification — that this is not the same
  # model family reviewing itself, S5 and A5 — rests on nothing in the artifact.
  assert_probe arm-e-no-model-family row-07-honest-docs-only-pmat-consulted B1 \
    "S3.E identity fields are absent or empty: model_family" \
    'del(.predicate.consultations.antigravity.model_family)'
}

@test "probe antigravity usage missing a token count                   RED  B1" {
  # S8's cost_per_actionable is fed from agy's own usage block, which is real token
  # accounting rather than an estimate. Record-only is not unenforced — the same rule
  # the receipt's own `cost` block already carries.
  assert_probe arm-e-usage-incomplete row-07-honest-docs-only-pmat-consulted B1 \
    "antigravity.usage must carry numeric input_tokens, output_tokens and total_tokens (missing or non-numeric: input_tokens)" \
    'del(.predicate.consultations.antigravity.usage.input_tokens)'
}

@test "probe antigravity usage that is not an object                   RED  B1" {
  assert_probe arm-e-usage-not-object row-07-honest-docs-only-pmat-consulted B1 \
    "usage is not an object" \
    '.predicate.consultations.antigravity.usage = "20836 in, 904 out"'
}

@test "probe reverified_by_primary absent                              RED  B1" {
  # agy's `measured` claims were produced by agy's process. S1 defines `measured` as
  # "produced by a command THIS RUN executed", and for an agy finding "this run" is
  # agy's. THE RULING IS THAT THE PRIMARY DOES NOT RE-RUN THEM and the receipt says so
  # here; re-running and adjudicating would dissolve the independence the arm exists to
  # create. What is forbidden is leaving it unsaid.
  assert_probe arm-e-reverified-absent row-07-honest-docs-only-pmat-consulted B1 \
    "antigravity.reverified_by_primary must be present and boolean" \
    'del(.predicate.consultations.antigravity.reverified_by_primary)'
}

@test "probe reverified_by_primary that is a string, not a boolean     RED  B1" {
  # "no" is not false. A string here reads as an answer while committing to nothing,
  # which is the shape of every field this spec has had to close.
  assert_probe arm-e-reverified-string row-07-honest-docs-only-pmat-consulted B1 \
    "antigravity.reverified_by_primary must be present and boolean" \
    '.predicate.consultations.antigravity.reverified_by_primary = "no"'
}

@test "probe divergence with no contradicted column                    RED  B1" {
  # `contradicted` is the row that matters and the row a lazy implementation drops: agy
  # and the primary reached OPPOSITE conclusions on one subject. A receipt that cannot
  # REPRESENT that is a receipt in which the primary always wins — which is the failure
  # S5 names, one level up from the second-invocation audit.
  assert_probe arm-e-divergence-no-contradicted row-07-honest-docs-only-pmat-consulted B1 \
    "bad or missing: contradicted" \
    'del(.predicate.consultations.antigravity.divergence.contradicted)'
}

@test "probe divergence with a negative count                          RED  B1" {
  assert_probe arm-e-divergence-negative row-07-honest-docs-only-pmat-consulted B1 \
    "bad or missing: agy_only" \
    '.predicate.consultations.antigravity.divergence.agy_only = -1'
}

@test "probe divergence that is not an object                          RED  B1" {
  assert_probe arm-e-divergence-not-object row-07-honest-docs-only-pmat-consulted B1 \
    "divergence is not an object" \
    '.predicate.consultations.antigravity.divergence = "none"'
}

@test "probe the divergence ledger does not account for the findings   RED  B1" {
  # WITHOUT THE IDENTITY, `divergence` IS FOUR NUMBERS NOTHING CONSTRAINS, and
  # {0,0,0,0} beside a non-empty findings array reads as perfect agreement. Every agy
  # finding is exactly one of agreed, agy-only or contradicted; `primary_only` is
  # deliberately OUTSIDE the identity because it counts the primary's findings agy did
  # not raise, which are not in this array.
  assert_probe arm-e-ledger-unbalanced row-33-arm-e-finding-advisory B1 \
    "divergence accounts for 0 of them" \
    '.predicate.consultations.antigravity.divergence = {"agreed":0,"agy_only":0,"primary_only":0,"contradicted":0}'
}

@test "probe an agy finding at level error but precision_class advisory GREEN    [discrimination]" {
  # A finding is refused on its PRECISION CLASS, never on its content or its severity.
  # agy may report anything it likes at `advisory`, and SARIF level `error` stays legal:
  # the arm is advisory about AUTHORITY, not about how loudly it may speak. Without this
  # row, "silence the second vendor" and "do not let it block" read the same.
  local d
  d=$(make_probe arm-e-error-level-advisory row-33-arm-e-finding-advisory '.' \
      '.runs[0].results[0].level = "error"') || { echo "probe build failed"; false; }
  run "$GUARD" "$d"
  [ "$status" -eq 0 ] || { echo "expected GREEN, got exit $status:"; echo "$output"; false; }
}

@test "S3.E the not-a-second-vendor table matches, both polarities" {
  # ANY REGEX IN THIS REPOSITORY SHIPS A CASE TABLE, and this one's universe is a
  # command's OUTPUT rather than the pattern's own vocabulary — F5's lesson, where a
  # table written from the regex passed 13/13 while missing three real spellings.
  # Every id `agy models` printed is a row here - the UNION of two runs on 2026-08-31,
  # because the catalogue MOVED between them (14 ids, then 11: the gemini-3.5-flash-*
  # tier gone). An id the vendor has pruned is still an id a stale script can pass, so
  # rows are never dropped when the vendor's list shrinks (D12).
  run run_case_table arm-e-model --match-arm-e-same-family
  [ "$status" -eq 0 ] || { echo "$output"; false; }
}

@test "S3.E the two Claude ids agy actually offers are asserted by name" {
  # The table above would still pass if it silently shrank. These two are the reason
  # this rule exists at all: they are in `agy models` output on the box this was
  # measured on, so "agy is Antigravity, Antigravity is Google's" is not sufficient.
  "$GUARD" --match-arm-e-same-family 'claude-sonnet-4-6'
  "$GUARD" --match-arm-e-same-family 'claude-opus-4-6-thinking'
  # ...and the rule must NOT be a wildcard that refuses every model, which would make
  # the arm unusable and read as "correctly strict".
  run "$GUARD" --match-arm-e-same-family 'gemini-3.1-pro-high'
  [ "$status" -ne 0 ]
}

@test "row 35 agy routed to the primary reviewer's own model family    RED  B1" {
  # ROW 7 WITH ONE TOKEN CHANGED. Every other field is honest — binary resolved and
  # recorded, version recorded, usage real, ledger balanced, arm advisory — and the
  # consultation is still the same model family reviewing itself. S5 cites Huang et al.
  # (ICLR'24) for why that is worth close to nothing; A5 calls a separate grounded
  # critic the first configuration that beats single-agent, and a same-family critic is
  # not one.
  assert_row row-35-arm-e-routed-to-the-same-model-family RED B1 \
    "which is the reviewing agent's OWN model family"
}

@test "row 35's receipt is otherwise IDENTICAL to an accepted one   [single variable]" {
  # A single-variable control, in the idiom F6's book/src/examples/ pair uses. If the
  # two receipts differed anywhere else, row 35 would be evidence of nothing in
  # particular. The diff below must be exactly the model id — and `model_family` is
  # DELIBERATELY left reading google/gemini in row 35, because the whole point is that
  # a label can say cross-vendor while the mechanism did not engage.
  local a b
  a=$(jq -S 'del(.predicate.consultations.antigravity.model_id)' \
      "$FIX/row-07-honest-docs-only-pmat-consulted/receipt.intoto.jsonl")
  b=$(jq -S 'del(.predicate.consultations.antigravity.model_id)' \
      "$FIX/row-35-arm-e-routed-to-the-same-model-family/receipt.intoto.jsonl")
  [ "$a" = "$b" ] || {
    echo "row 35 differs from row 07 in more than the model id:"
    diff <(printf '%s\n' "$a") <(printf '%s\n' "$b") || true
    return 1; }
}

@test "S3.E a single argv element is capped at MAX_ARG_STRLEN, not at ARG_MAX" {
  # THE MEASUREMENT S3.E.1(b)'s RECIPE IS BUILT ON, pinned so it cannot rot into folklore.
  # The previous invocation inlined the diff with -p "$(cat …)" and died at rc 127,
  # `argument list too long`, on any non-trivial PR — #2803's merge-base diff is 144 325
  # bytes. THE OBVIOUS LIMIT IS THE WRONG ONE: `getconf ARG_MAX` reads 2 097 152 on this
  # box, so a reader who checks it concludes there is room. The real cap is Linux's
  # MAX_ARG_STRLEN — 32 pages on a SINGLE argument, raised by no ulimit and no ARG_MAX.
  #
  # BOTH POLARITIES, because a one-sided test passes on a box where everything fails.
  # The rejection is what is asserted, NOT a particular exit code: the failing exec is
  # reported 127 by zsh (where agy itself was measured) and 126 by bash. Pinning one of
  # them would make this row a test of the shell.
  local lim=131072 rc_under=0 rc_at=0
  /bin/true "$(head -c $((lim - 1)) /dev/zero | tr '\0' 'x')" 2>/dev/null || rc_under=$?
  /bin/true "$(head -c "$lim"       /dev/zero | tr '\0' 'x')" 2>/dev/null || rc_at=$?
  [ "$rc_under" -eq 0 ] || {
    echo "expected an argument of $((lim - 1)) bytes to be accepted, got rc=$rc_under"
    echo "if this box's page size differs, the recipe's file-based prompt is still correct;"
    echo "re-measure the boundary and update S3.E.1(b) rather than deleting this row."
    return 1; }
  [ "$rc_at" -ne 0 ] || {
    echo "expected an argument of $lim bytes to be REFUSED, got rc=0."
    echo "the cap this box enforces is larger than MAX_ARG_STRLEN was measured to be;"
    echo "S3.E.1(b) states 131072 as MEASURED and must be re-measured, not widened."
    return 1; }
  # And the number the spec quotes is on the wrong side of it, which is the whole point.
  [ 144325 -ge "$lim" ] || { echo "#2803's diff no longer exceeds the cap"; return 1; }
}

@test "S3.E the documented invocation carries the disposable tree and the file prompt" {
  # A SHAPE GATE OVER PROSE, and it is labelled one: it cannot prove the recipe RUNS,
  # only that the two fixes are still written down together. They are inseparable —
  # --dangerously-skip-permissions alone hands a second agent write access to the
  # working checkout, and the `git archive | tar -x` copy is what makes it safe — so a
  # future edit that keeps the flag and drops the copy is the dangerous one, and this is
  # the row that catches it.
  local skill="$REPO_ROOT/.claude/skills/pr-review/SKILL.md"
  grep -qF 'git archive "$HEAD_SHA" | tar -x -C "$REVIEW"' "$skill" || {
    echo "S3.E step 3 no longer extracts the head into a disposable directory"; return 1; }
  grep -qF -- '--dangerously-skip-permissions' "$skill" || {
    echo "S3.E step 3 no longer passes --dangerously-skip-permissions; headless print mode"
    echo "auto-denies tool permissions and returns rc 0 having reviewed nothing"; return 1; }
  # The diff must reach agy as a FILE. If a future edit inlines it back into -p, the
  # command dies at rc 127 on every non-trivial PR (row above).
  grep -qF '.pr-review-diff.patch' "$skill" || {
    echo "S3.E step 3 no longer passes the diff as a file"; return 1; }
  # $OUT is RELATIVE in S4.1. The recipe cds into the disposable tree before
  # redirecting into it, so it must be absolutised first — otherwise the transcript is
  # written inside the copy and deleted with it, silently, at rc 0. That defect was in
  # the first draft of this very recipe.
  grep -qF 'OUT=$(cd "$OUT" && pwd)' "$skill" || {
    echo "S3.E step 3 redirects to \$OUT after cd-ing away without absolutising it;"
    echo "the agy transcript would land in the disposable tree and be deleted with it"
    return 1; }
  ! grep -qE '^\s*"\$AGY" -p "\$\(git diff' "$skill" || {
    echo "S3.E step 3 inlines the diff into -p; that is rc 127 past 131072 bytes"; return 1; }
}

@test "row 36 agy exited 0 and produced nothing, recorded consulted     RED  B1" {
  # THE ROW THAT EXISTS BECAUSE THIS SKILL SHIPPED THE DEFECT. S3.E step 3's documented
  # invocation, run verbatim against a real checkout on 2026-08-31, returned rc 0 with
  # NO structured output: headless print mode cannot prompt for a tool permission, so it
  # auto-denied one, and reported `.status "SUCCESS"` over an empty `.response`. Read
  # through the exit code, that is a clean consultation; it is a review that never
  # happened. A FAILURE THAT EXITS 0 is the class this repository closed six times in
  # one session, and the fifth arm is where it costs most.
  assert_row row-36-arm-e-consulted-with-no-usable-output RED B1 \
    "output_check does not record a passing availability test"
  # THE REASON IS LOAD-BEARING, exactly as row 32's is. Every other field in this
  # receipt is honest and generous - attempted 1, real usage, a resolved binary path, a
  # Gemini model id, a balanced ledger - so if the rejection came from any neighbouring
  # branch the row would pin nothing and would stay red with the new rule deleted.
  run "$GUARD" "$FIX/row-36-arm-e-consulted-with-no-usable-output"
  [[ "$output" != *"attempted=0"* ]] || {
    echo "row 36 was rejected by the vacuity rule, not by the availability rule:"
    echo "$output"; return 1; }
  [[ "$output" != *"OWN model family"* ]] || {
    echo "row 36 was rejected by the second-vendor rule, not by the availability rule:"
    echo "$output"; return 1; }
}

@test "row 37 the SAME rc-0 run, recorded unreachable + DEGRADED      GREEN     [discrimination]" {
  # Without it the new rule reads green as "refuse any receipt whose agy exited 0" -
  # which refuses every consultation that WORKED, because agy returns rc 0 then too.
  # The variable between rows 36 and 37 is not the exit code, the duration, the usage or
  # the status agy printed: all four are identical, and both are transcripts of the same
  # measured run. It is what the receipt CLAIMS about them.
  assert_row row-37-arm-e-rc-zero-recorded-unreachable GREEN
  # And the two rows really are the same run. If a later edit made row 37 pass by
  # softening its measurements rather than by its honesty, this comparison goes red.
  local a b
  a=$(jq -r '.predicate.consultations.antigravity | [.exit_code, .agy_status, .duration_seconds] | @tsv' \
      "$FIX/row-36-arm-e-consulted-with-no-usable-output/receipt.intoto.jsonl")
  b=$(jq -r '.predicate.consultations.antigravity | [.exit_code, .agy_status, .duration_seconds] | @tsv' \
      "$FIX/row-37-arm-e-rc-zero-recorded-unreachable/receipt.intoto.jsonl")
  [ "$a" = "$b" ] || {
    echo "rows 36 and 37 must record the SAME agy run; got:"
    echo "  36: $a"; echo "  37: $b"; return 1; }
}

@test "probe an rc-0 consultation missing output_check entirely         RED  B1" {
  # The absent-field polarity. Row 36 records the test and records it FAILING; this is
  # the receipt that does not record it at all, which is the shape every pre-PRREV-020
  # receipt has and the one a reviewer reaches for first. Absent is not passing.
  assert_probe arm-e-output-check-absent row-07-honest-docs-only-pmat-consulted B1 \
    "output_check is not an object" \
    'del(.predicate.consultations.antigravity.output_check)'
}

@test "probe output_check present but schema_valid false               RED  B1" {
  # The conjunct that a reviewer is likeliest to fudge: agy answered, the JSON parsed,
  # `reviewed` is true - and the payload does not validate against the schema it was
  # asked for. S3.E.4's test is a CONJUNCTION, so one false member fails it, and a
  # rule that only read `structured_output_present` would accept this.
  assert_probe arm-e-output-check-schema-invalid row-07-honest-docs-only-pmat-consulted B1 \
    "not true: schema_valid" \
    '.predicate.consultations.antigravity.output_check.schema_valid = false'
}

@test "probe output_check booleans written as strings                  RED  B1" {
  # `"true"` is not `true`, and a receipt whose availability test is a STRING has
  # recorded a word rather than a result. Same rule as reverified_by_primary's type
  # check one branch down, for the same reason: a field that accepts any type accepts
  # a field that was never computed.
  assert_probe arm-e-output-check-stringy row-07-honest-docs-only-pmat-consulted B1 \
    "not true: reviewed" \
    '.predicate.consultations.antigravity.output_check.reviewed = "true"'
}

@test "probe antigravity with no recorded model id                    RED  B1" {
  # An unrecorded model id is not a smaller defect than a same-family one: it is the
  # same defect with the evidence removed. The rule above can only fire on a value
  # that is written down.
  assert_probe arm-e-no-model-id row-07-honest-docs-only-pmat-consulted B1 \
    "S3.E identity fields are absent or empty: model_id" \
    'del(.predicate.consultations.antigravity.model_id)'
}

@test "S6.1 the control cache is keyed by the guard's own bytes, so a mutant re-proves them" {
  # The controls prove THIS GUARD can still fail -- a property of the file, not of the
  # receipt -- so they are cached under sha256(guard)+sha256(seeds)+sha256(schemas).
  # Measured: that took one invocation from 2362ms to 1004ms and the suite from 286s to
  # 107s, which is what put `pr-review-receipt` back inside its 150-minute cap.
  #
  # THE PROPERTY EVERYTHING RESTS ON is that a CHANGED guard re-proves them. Every mutant
  # in scripts/mutate-guard.sh changes the guard, so a key that ignored the guard's bytes
  # would serve 233 mutants a transcript proved against a different file.
  local cache="$PR_REVIEW_PC_CACHE_DIR"
  rm -rf "$cache"; mkdir -m 700 -p "$cache"

  run "$GUARD" "$FIX/row-14-complete-gpu-review"          # cold: writes one entry
  [ "$status" -eq 0 ] || { echo "$output"; false; }
  local n1; n1=$(find "$cache" -type f 2>/dev/null | wc -l)
  [ "$n1" -eq 1 ] || { echo "expected 1 cache entry, found $n1"; false; }

  run "$GUARD" "$FIX/row-14-complete-gpu-review"          # warm: no new entry
  [ "$status" -eq 0 ] || { echo "$output"; false; }
  local n2; n2=$(find "$cache" -type f 2>/dev/null | wc -l)
  [ "$n2" -eq 1 ] || { echo "warm run added an entry ($n1 -> $n2); the key is not stable"; false; }

  # A one-byte change to the guard must produce a SECOND entry.
  # The copy must be told where the seeds and schemas are: the guard resolves both
  # relative to its OWN path, and a copy in a tmpdir would fail before it ever reached
  # the cache -- which is a fixture-resolution failure, not a cache-key one.
  cp "$GUARD" "$BATS_TEST_TMPDIR/mutant.sh"
  printf '# one byte of mutant\n' >> "$BATS_TEST_TMPDIR/mutant.sh"
  chmod +x "$BATS_TEST_TMPDIR/mutant.sh"
  PR_REVIEW_POSITIVE_CONTROL_DIR="$FIX/positive-control" \
  PR_REVIEW_SCHEMA_DIR="$REPO_ROOT/schemas" \
    run "$BATS_TEST_TMPDIR/mutant.sh" "$FIX/row-14-complete-gpu-review"
  local n3; n3=$(find "$cache" -type f 2>/dev/null | wc -l)
  [ "$n3" -eq 2 ] || {
    echo "a changed guard did NOT re-prove the controls: entries $n2 -> $n3, expected 2"
    echo "every mutant would reuse a transcript proved against different bytes"; false; }
}

# --- PRREV-023: the runs array is not allowed to be empty --------------------

@test "row 41 findings.sarif with an empty runs array                   RED  B1" {
  # THE VACUITY. Every result rule in the guard reads `.runs[]? | .results[]?`, so over
  # an empty runs array each of them iterates nothing and finds nothing to reject. This
  # exact receipt returned rc=0 ACCEPT before the rule existed, while still declaring
  # verdict FINDINGS and five consultations consulted, and the four positive controls
  # fired correctly in that same run -- so it was genuine vacuity, not a broken harness.
  assert_row row-41-sarif-with-no-runs RED B1 "carries no runs[]"
}

@test "row 42 a run whose tool.driver.name is outside the vocabulary    RED  B1" {
  # Checked as a WHITELIST. S13.13 found every clause that fell to the forged receipts
  # was a blacklist over a field and every clause that survived was a whitelist; the
  # difference was the direction of the test. `Pmat` is one capital, which is all it
  # takes to hide a run from `select(.tool.driver.name == "antigravity")`.
  #
  # The first draft of the rule ACCEPTED this row: `select([...] | index(.) == null)`
  # rebinds `.` to the literal array through the pipe, so it asked whether the array
  # contained itself. Caught by running the row, not by reading the line.
  assert_row row-42-run-driver-outside-vocabulary RED B1 "outside { pmat, nvidia-cuda-docs, crux, cargo-mutants, antigravity }"
}

@test "row 43 verdict FINDINGS beside zero results                      RED  B1" {
  # A verdict and the artifact it points at must not disagree about whether anything was
  # found. FINDINGS over an empty result set is the same review as PASS, which makes the
  # verdict decorative on exactly the transition S7 reads to decide blocking.
  assert_row row-43-findings-verdict-with-no-results RED B1 "carries zero results"
}

@test "row 7 is PASS with an empty result set and stays GREEN     [discrimination]" {
  # THE PARTNER FOR ROW 40, and the reason row 40's rule is about the VERDICT rather
  # than the emptiness. row-07's findings.sarif carries a run with `results: []` and a
  # PASS verdict, and it must stay GREEN: "consulted, found nothing" is S3.0 row 1, the
  # honest outcome the whole three-state encoding exists to keep available. A rule that
  # refused this would teach reviewers to invent findings.
  run jq -e '[ .runs[] | .results[]? ] | length == 0' "$FIX/row-07-honest-docs-only-pmat-consulted/findings.sarif"
  [ "$status" -eq 0 ] || { echo "row 7 no longer has an empty result set"; return 1; }
  run jq -r '.predicate.verdict' "$FIX/row-07-honest-docs-only-pmat-consulted/receipt.intoto.jsonl"
  [ "$output" = "PASS" ] || { echo "row 7 verdict is $output, expected PASS"; return 1; }
  assert_row row-07-honest-docs-only-pmat-consulted GREEN
}

# --- the cache's three fail-closed properties, each mutation-verified --------
# A cross-vendor review of the control cache named three ways it could serve a
# transcript the live run would not have produced. Each is a row here, and each was
# confirmed to go RED against the pre-fix guard before the fix was written.

@test "S6.1 a truncated cache entry is a MISS, not a hit          [fail-closed]" {
  # `[ -s ]` was the entire read-side check and it is TRUE OF A ONE-BYTE FRAGMENT. This
  # queue evicts jobs mid-run, so a torn write was reachable, and it would have been a
  # PERMANENT free pass: every later run cats the fragment and skips the controls.
  local cache="$PR_REVIEW_PC_CACHE_DIR"
  rm -rf "$cache"; mkdir -m 700 -p "$cache"
  run "$GUARD" "$FIX/row-14-complete-gpu-review"
  local k; k=$(ls "$cache")
  [ -n "$k" ] || { echo "cold run wrote no entry"; false; }

  # Tear it. The terminator carries sha256 of the body, so a prefix cannot satisfy it.
  head -c 30 "$cache/$k" > "$cache/$k.t" && mv "$cache/$k.t" "$cache/$k"
  run "$GUARD" "$FIX/row-14-complete-gpu-review"
  local fired; fired=$(printf '%s\n' "$output" | grep -c '^positive-control') || true
  [ "$fired" -eq 4 ] || {
    echo "a torn entry was served as a hit: $fired control lines, expected 4 from a re-run"
    false; }
  # ...and the re-run must have REPAIRED it, terminator and all.
  run tail -n 1 "$cache/$k"
  [[ "$output" == __PC_CACHE_END__\ * ]] || { echo "entry not restamped: $output"; false; }
}

@test "S6.1 a changed TOOL re-proves the controls, not just a changed guard" {
  # The controls do not exercise the guard's bytes, they exercise the guard running
  # THROUGH jq / check-jsonschema / minisign / git. If check-jsonschema began accepting
  # invalid documents, control 1 would stop firing while the guard's bytes -- and so a
  # guard-only key -- stayed identical. A shim earlier on PATH is the cheapest way to
  # prove the key sees the difference; it resolves to the same jq, so ONLY the resolved
  # path and file hash differ, which is the weakest form of the drift the key must catch.
  local cache="$PR_REVIEW_PC_CACHE_DIR"
  rm -rf "$cache"; mkdir -m 700 -p "$cache"
  run "$GUARD" "$FIX/row-14-complete-gpu-review"
  [ "$(find "$cache" -maxdepth 1 -type f | wc -l)" -eq 1 ]

  mkdir -p "$BATS_TEST_TMPDIR/shim"
  printf '#!/bin/sh\nexec %s "$@"\n' "$(command -v jq)" > "$BATS_TEST_TMPDIR/shim/jq"
  chmod +x "$BATS_TEST_TMPDIR/shim/jq"
  PATH="$BATS_TEST_TMPDIR/shim:$PATH" run "$GUARD" "$FIX/row-14-complete-gpu-review"
  local n; n=$(find "$cache" -maxdepth 1 -type f | wc -l)
  [ "$n" -eq 2 ] || {
    echo "a different jq reused the same key ($n entries, expected 2)"
    echo "a broken tool would be cached past, which is the disarmed-guard class"; false; }
}

@test "S6.1 a run whose controls did NOT fire writes no cache entry [fail-closed]" {
  # THE PROPERTY THE OTHER TWO REST ON. Every control is `|| exit 1`, so the write is
  # unreachable unless all four fired -- an entry can only be created by a run that
  # PROVED the guard can still fail. Verified by removing a seed: the seeded control
  # refuses, the guard exits non-zero, and the cache must stay empty.
  local cache="$PR_REVIEW_PC_CACHE_DIR"
  rm -rf "$cache"; mkdir -m 700 -p "$cache"
  cp -a "$REPO_ROOT/tests/fixtures/pr-review/positive-control" "$BATS_TEST_TMPDIR/pc"
  rm -f "$BATS_TEST_TMPDIR/pc/self-review/receipt.intoto.jsonl"

  PR_REVIEW_POSITIVE_CONTROL_DIR="$BATS_TEST_TMPDIR/pc" \
    run "$GUARD" "$FIX/row-14-complete-gpu-review"
  [ "$status" -ne 0 ] || { echo "a missing control seed did not stop the run"; false; }
  local n; n=$(find "$cache" -maxdepth 1 -type f 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] || {
    echo "a run that could not prove the controls still wrote $n cache entry(ies)"
    echo "that entry would then be served to runs that never proved anything"; false; }
}

@test "S6.1 with no cache directory supplied the controls ALWAYS run  [fail-closed]" {
  # THE SCOPE IS THE FIX, not more key material. A second review refused the shared-path
  # design on three grounds that all reduce to one: the environment is not fully hashable
  # (PYTHONPATH, LD_LIBRARY_PATH and check-jsonschema's own Python packages are outside the
  # key), so an entry written under one environment can be consumed under another -- and
  # positive controls exist precisely to detect runtime environmental drift. Caching them
  # across invocations assumes the stability they are there to test.
  #
  # Unset, therefore: no cache, controls every time. That is what a developer running the
  # guard by hand gets, and it is the safe default rather than the fast one.
  # COUNTING THE CONTROL LINES CANNOT PROVE THIS, and the first version of this row tried
  # to. A cache hit REPLAYS the transcript verbatim -- that is the point of it -- so a
  # replayed run prints the same four lines as a live one, and the row stayed GREEN under
  # the mutation that restored the shared fallback path. Caught by re-mutating rather than
  # by re-reading. The discriminator has to be PERSISTENCE: a run with no cache directory
  # must leave nothing behind for a later run to read.
  local shared="${TMPDIR:-/tmp}/pr-review-pc-cache"
  rm -rf "$shared"
  local before; before=$(find "${TMPDIR:-/tmp}" -maxdepth 2 -name '*pr-review-pc-cache*' 2>/dev/null | wc -l)

  run env -u PR_REVIEW_PC_CACHE_DIR "$GUARD" "$FIX/row-14-complete-gpu-review"
  local n1; n1=$(printf '%s\n' "$output" | grep -c '^positive-control') || true
  [ "$n1" -eq 4 ] || { echo "expected 4 controls with no cache dir, saw $n1"; false; }

  local after; after=$(find "${TMPDIR:-/tmp}" -maxdepth 2 -name '*pr-review-pc-cache*' 2>/dev/null | wc -l)
  [ "$after" -eq "$before" ] || {
    echo "a cache-less run persisted state ($before -> $after paths under TMPDIR)"
    echo "the next invocation would replay it, and the controls would stop running"; false; }

  run env -u PR_REVIEW_PC_CACHE_DIR "$GUARD" "$FIX/row-14-complete-gpu-review"
  local n2; n2=$(printf '%s\n' "$output" | grep -c '^positive-control') || true
  [ "$n2" -eq 4 ] || { echo "a second cache-less run skipped controls ($n2)"; false; }
}

@test "S6.1 the guard never CREATES a cache directory, only uses one it is given" {
  # The old path was ${TMPDIR:-/tmp}/pr-review-pc-cache and the guard mkdir -p'd it. On a
  # runner shared across repositories that is a predictable, world-writeable location whose
  # key is computable from public files, so any process on the box could seed a passing
  # transcript and the guard would skip its controls because a stranger said they fired.
  # It now writes only into a directory the caller already created.
  local shared="${TMPDIR:-/tmp}/pr-review-pc-cache"
  rm -rf "$shared"
  run env -u PR_REVIEW_PC_CACHE_DIR "$GUARD" "$FIX/row-14-complete-gpu-review"
  [ ! -e "$shared" ] || { echo "the guard created the shared path $shared"; rm -rf "$shared"; false; }

  # And a named-but-absent directory is no cache, never a newly created one.
  local ghost="$BATS_TEST_TMPDIR/never-created"
  PR_REVIEW_PC_CACHE_DIR="$ghost" run "$GUARD" "$FIX/row-14-complete-gpu-review"
  [ ! -e "$ghost" ] || { echo "the guard created $ghost instead of running uncached"; false; }
  local n; n=$(printf '%s\n' "$output" | grep -c '^positive-control') || true
  [ "$n" -eq 4 ] || { echo "expected 4 controls against an absent cache dir, saw $n"; false; }
}
