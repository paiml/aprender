#!/usr/bin/env bats
#
# PR-REVIEW-SKILL-002 v2 S13 — the autonomous-merge refusal table.
#
# ONE ROW PER REFUSAL PATH of scripts/pr_review_quorum_arm.sh, plus four that PERMIT.
# scripts/mutate_quorum_arm.sh derives one `drop` mutant per `refuse Q<n>` site in that
# script, so a site with no row here leaves a SURVIVOR: a rule the arm script states
# that a receipt could break with this table still green. That is the defect class the
# whole skill exists to remove, and S13 is where it would matter most — a refusal path
# that cannot fire is, under autonomy, a merge nobody authorised.
#
# THE CLASS AND THE REASON ARE BOTH ASSERTED. Q1 covers fifteen branches and Q2 covers
# seven; a row that trips a DIFFERENT branch of the same class still reports the class
# and still exits 1 — it passes for the wrong reason, and the mutant that dropped its
# branch lives. tests/pr-review.bats measured that effect on the receipt guard (nine of
# 119 mutants died only because the reason was asserted) and the same rule is applied
# here from the start rather than after the counter-sweep.
#
# FOUR ROWS MUST PERMIT, and three of them are DISCRIMINATION rows — they differ from a
# refusing row in ONE variable:
#
#   q-53 vs q-47   which file the post-review commit touches (evidence/ or source)
#   q-54 vs q-49   whether the delta region was swept
#   q-55 vs q-45   the mutation scope on a guard-touching diff
#
# Without them "refuse every receipt" and "evaluate every receipt" produce the same
# green, which is the over-reach the S6.3 discrimination rows already caught twice.

setup_file() {
  REPO_ROOT=$(cd -- "$BATS_TEST_DIRNAME/.." && pwd)
  export REPO_ROOT
  export ARM="$REPO_ROOT/scripts/pr_review_quorum_arm.sh"
  export FIX="$REPO_ROOT/tests/fixtures/pr-review"

  # The same purpose-built repository the S6.3 table uses, for the same reason: against
  # aprender's own history `git merge-base origin/main X` is X, so the merge-base and
  # ancestry clauses would pass vacuously.
  export FIXTURE_REPO="$BATS_FILE_TMPDIR/fixture-repo"
  "$FIX/make-fixture-repo.sh" "$FIXTURE_REPO" >/dev/null

  # A SECOND repository, identical except that origin/main carries the kill switch.
  # Built by moving the remote ref to K1 rather than by committing the switch onto
  # main: merge-base(K1, F1) is still C1, so every committed receipt stays valid under
  # it and the kill-switch row differs from the row that permits in the REPOSITORY and
  # in nothing else.
  export KILLSWITCH_REPO="$BATS_FILE_TMPDIR/killswitch-repo"
  cp -a "$FIXTURE_REPO" "$KILLSWITCH_REPO"
  git -C "$KILLSWITCH_REPO" update-ref refs/remotes/origin/main refs/heads/ksmain-pr

  # A RECORDING STUB FOR gh, AND IT IS A SAFETY DEVICE BEFORE IT IS AN ASSERTION.
  # scripts/mutate_quorum_arm.sh runs this file against a MUTATED arm script, and one of
  # the mutants disables the --explain guard. With a real gh on PATH that mutant would
  # try to arm auto-merge on a live pull request. The stub cannot: it records the call
  # and exits 1. That it is ALSO how "--explain never calls gh" is asserted is a bonus,
  # not the reason.
  export GH_STUB_DIR="$BATS_FILE_TMPDIR/gh-stub"
  mkdir -p "$GH_STUB_DIR"
  cat > "$GH_STUB_DIR/gh" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${GH_STUB_LOG:?}"
exit 1
STUB
  chmod +x "$GH_STUB_DIR/gh"
  export PR_REVIEW_GH="$GH_STUB_DIR/gh"
}

setup() {
  export PR_REVIEW_REPO="$FIXTURE_REPO"
  export PR_REVIEW_PUBKEY="$FIX/keys/pr-review-test.pub"
  WORK="$BATS_TEST_TMPDIR/work"
  mkdir -p "$WORK"
  export GH_STUB_LOG="$BATS_TEST_TMPDIR/gh-calls.log"
  : > "$GH_STUB_LOG"
}

# arm_row <fixture-dir-name> <PERMIT|REFUSE> [<class>] [<reason-substring>]
arm_row() {
  local name=$1 want=$2 class=${3:-} reason=${4:-}
  run "$ARM" --explain --pr 2783 --receipt "$FIX/$name" --context "$FIX/$name/pr-context.json"
  if [ "$want" = PERMIT ]; then
    [ "$status" -eq 0 ] || { echo "expected PERMIT (exit 0), got $status:"; echo "$output"; return 1; }
    [[ "$output" == *"PERMIT"* ]] || { echo "no PERMIT line:"; echo "$output"; return 1; }
    [[ "$output" != *"REFUSE"* ]] || { echo "PERMIT and REFUSE in one run:"; echo "$output"; return 1; }
  else
    [ "$status" -eq 1 ] || { echo "expected REFUSE (exit 1), got $status:"; echo "$output"; return 1; }
    [[ "$output" == *"REFUSE"* ]] || { echo "no REFUSE line:"; echo "$output"; return 1; }
    if [ -n "$class" ]; then
      [[ "$output" == *"[$class]"* ]] || { echo "expected class $class:"; echo "$output"; return 1; }
    fi
    if [ -n "$reason" ]; then
      [[ "$output" == *"$reason"* ]] || {
        echo "refused under $class but on the wrong branch; wanted a reason containing:"
        echo "  $reason"; echo "got:"; echo "$output"; return 1; }
    fi
  fi
}

# ---------------------------------------------------------------------------
# Q1 — the receipt is missing, unreadable, or asks for no autonomy.
# ---------------------------------------------------------------------------

@test "q-01 a receipt directory that does not exist refuses, it does not skip" {
  run "$ARM" --explain --pr 2783 --receipt "$FIX/q-01-there-is-no-such-directory" \
       --context "$FIX/q-52-permits-a-clean-quorum/pr-context.json"
  [ "$status" -eq 1 ]
  [[ "$output" == *"[Q1]"* ]]
  [[ "$output" == *"no such receipt directory"* ]]
}

@test "q-02 the receipt file is missing" {
  arm_row q-02-receipt-file-missing REFUSE Q1 "receipt.intoto.jsonl is missing"
}

@test "q-03 findings.sarif is missing" {
  arm_row q-03-sarif-missing REFUSE Q1 "findings.sarif is missing"
}

@test "q-04 an UNSIGNED receipt never arms an autonomous merge" {
  arm_row q-04-unsigned-receipt REFUSE Q1 "the receipt is unsigned"
}

@test "q-05 the receipt is not parseable JSON" {
  arm_row q-05-receipt-unparseable REFUSE Q1 "not parseable JSON"
}

@test "q-06 findings.sarif is not parseable JSON" {
  arm_row q-06-sarif-unparseable REFUSE Q1 "findings.sarif is not parseable JSON"
}

@test "q-07 the PR context is absent" {
  run "$ARM" --explain --pr 2783 --receipt "$FIX/q-07-context-missing" \
       --context "$FIX/q-07-context-missing/pr-context.json"
  [ "$status" -eq 1 ]
  [[ "$output" == *"[Q1]"* ]]
  [[ "$output" == *"no PR context at"* ]]
}

@test "q-08 the PR context is not parseable JSON" {
  arm_row q-08-context-unparseable REFUSE Q1 "PR context at"
}

@test "q-09 a receipt with NO autonomy block does not default to yes" {
  arm_row q-09-no-autonomy-block REFUSE Q1 "carries no predicate.autonomy block"
}

@test "q-10 autonomy.requested is false" {
  arm_row q-10-autonomy-not-requested REFUSE Q1 "autonomy.requested is not true"
}

@test "q-11 main_sha_at_review is absent, so the delta region cannot be named" {
  arm_row q-11-no-main-sha-at-review REFUSE Q1 "main_sha_at_review is absent"
}

@test "q-12 the quorum is empty" {
  arm_row q-12-empty-quorum REFUSE Q1 "autonomy.quorum is absent or empty"
}

@test "q-13 the delta sweep is not recorded at all" {
  arm_row q-13-no-delta-sweep REFUSE Q1 "delta_sweep is absent"
}

@test "q-14 delta_sweep.status is outside its vocabulary" {
  arm_row q-14-delta-sweep-status-outside-vocabulary REFUSE Q1 "outside { clean, dirty, not-run }"
}

# ---------------------------------------------------------------------------
# Q6 — the verdict, and the facts the verdict summarises.
# ---------------------------------------------------------------------------

@test "q-15 DEGRADED does not auto-merge (S13.3, adopted without weakening)" {
  arm_row q-15-verdict-degraded REFUSE Q6 "the verdict is 'DEGRADED', not PASS"
}

@test "q-16 an ASSERTED finding classed blocking is diagnosed as itself, not as a blocker" {
  arm_row q-16-asserted-finding-classed-blocking REFUSE Q6 "marked asserted AND classed blocking"
}

@test "q-17 a surviving blocking-class finding under a PASS verdict" {
  arm_row q-17-blocking-finding-survives REFUSE Q6 "classed blocking under a PASS verdict"
}

@test "q-18 an unmarked claim (S8 fixes unmarked_claims at 0, no ratchet)" {
  arm_row q-18-unmarked-claim REFUSE Q6 "with no grounding mark"
}

@test "q-19 a SARIF invocation that did not execute successfully" {
  arm_row q-19-tool-execution-failed REFUSE Q6 "executionSuccessful: false"
}

# ---------------------------------------------------------------------------
# Q7 — the cross-vendor reviewer's one power.
# ---------------------------------------------------------------------------

@test "q-20 the cross-vendor reviewer may not block, and MAY refuse autonomy" {
  arm_row q-20-cross-vendor-refuses-autonomy REFUSE Q7 "autonomy_effect: refuse"
}

# ---------------------------------------------------------------------------
# Q5 — separation, threefold.
# ---------------------------------------------------------------------------

@test "q-21 author_actor.id is absent, so separation cannot be checked" {
  arm_row q-21-author-actor-absent REFUSE Q5 "author_actor.id is absent"
}

@test "q-22 reviewer_actor == author_actor" {
  arm_row q-22-self-review REFUSE Q5 "reviewer_actor.id = author_actor.id"
}

@test "q-23 a single-vendor quorum raises the count and not the independence" {
  arm_row q-23-single-vendor-quorum REFUSE Q5 "distinct vendor(s)"
}

@test "q-24 the quorum has no cross_vendor member" {
  arm_row q-24-quorum-role-missing REFUSE Q5 "no member in role(s): cross_vendor"
}

@test "q-25 the AUTHOR sits in the quorum" {
  arm_row q-25-author-sits-in-the-quorum REFUSE Q5 "carry the author's own actor id"
}

@test "q-26 one member dissents, and S13.4 counts only dissent" {
  arm_row q-26-quorum-not-unanimous REFUSE Q5 "not unanimous"
}

# ---------------------------------------------------------------------------
# Q2 — a consultation did not run, or ran over nothing.
# ---------------------------------------------------------------------------

@test "q-27 pmat is unconditional, so an unreachable pmat does not arm" {
  # THE REASON HERE IS THE S3.A HALF, NOT THE STATUS HALF, AND THAT IS A MUTATION
  # RESULT. The status loop three clauses down ALSO refuses an `unreachable` pmat, and
  # its message begins with the same words - so asserting "consultations.pmat.status is
  # 'unreachable'" passed with the pmat clause DELETED, and `refuse-27-drop` SURVIVED
  # (run of 2026-08-31, 120/122). The tail of the message is what only this clause says.
  arm_row q-27-pmat-not-consulted REFUSE Q2 "S3.A makes pmat unconditional"
}

@test "q-57 pmat not-triggered is the shape ONLY the S3.A clause catches" {
  # The status loop ADMITS not-triggered for every consultation, so this receipt reaches
  # nothing else. Without this row the pmat clause is exercised only by receipts a later
  # clause would refuse anyway, which is a rule the script states and nothing tests.
  arm_row q-57-pmat-not-triggered REFUSE Q2 "S3.A makes pmat unconditional"
}

@test "q-28 not-triggered with an empty trigger_reason" {
  arm_row q-28-not-triggered-with-no-reason REFUSE Q2 "empty trigger_reason"
}

@test "q-29 an unreachable consultation is excluded here, not weakened" {
  arm_row q-29-consultation-unreachable REFUSE Q2 "only { consulted, not-triggered } arm a merge"
}

@test "q-30 a pmat run that searched zero symbols is vacuous" {
  arm_row q-30-vacuous-zero-symbols REFUSE Q2 "pmat searched 0 symbols"
}

@test "q-31 a surviving mutant does not ship on an unattended merge" {
  arm_row q-31-mutation-survivor REFUSE Q2 "surviving mutant(s)"
}

@test "q-32 a duplication surface that could not be searched" {
  arm_row q-32-duplication-surface-unsearched REFUSE Q2 "could not search [shell]"
}

@test "q-33 a horizon region recorded as none" {
  arm_row q-33-horizon-region-unswept REFUSE Q2 "records no refspec for"
}

# ---------------------------------------------------------------------------
# Q8 / Q9 / Q10 — the human surfaces, eligibility, the mechanical checks.
# ---------------------------------------------------------------------------

@test "q-34 the autonomy-hold label" {
  arm_row q-34-autonomy-hold-label REFUSE Q8 "autonomy-hold label"
}

@test "q-35 an open, undismissed CHANGES_REQUESTED review" {
  arm_row q-35-changes-requested-open REFUSE Q8 "CHANGES_REQUESTED review(s) are open"
}

@test "q-36 a pull request that does not target main" {
  arm_row q-36-not-targeting-main REFUSE Q9 "targets 'release/0.65', not main"
}

@test "q-37 workspace-test is not green" {
  arm_row q-37-workspace-test-not-green REFUSE Q10 "workspace-test check is 'failure'"
}

@test "q-38 the gate check is ABSENT under both of its two spellings" {
  arm_row q-38-gate-check-absent REFUSE Q10 "gate check is 'absent'"
}

@test "q-39 the kill switch on origin/main stops everything, and it is read from origin/main" {
  PR_REVIEW_REPO="$KILLSWITCH_REPO" run "$ARM" --explain --pr 2783 \
      --receipt "$FIX/q-39-kill-switch-on-origin-main" \
      --context "$FIX/q-39-kill-switch-on-origin-main/pr-context.json"
  [ "$status" -eq 1 ]
  [[ "$output" == *"[Q8]"* ]]
  [[ "$output" == *"kill switch"* ]]
}

@test "q-39b the SAME receipt permits against the repository without the kill switch" {
  arm_row q-39-kill-switch-on-origin-main PERMIT
}

# ---------------------------------------------------------------------------
# The repository clauses.
# ---------------------------------------------------------------------------

@test "q-40 a stale index under a PASS verdict is refused by the receipt guard, delegated" {
  arm_row q-40-stale-index-verdict-pass REFUSE Q1 "the receipt guard rejects this receipt"
}

@test "q-40b the delegated refusal carries the receipt guard's own class" {
  run "$ARM" --explain --pr 2783 --receipt "$FIX/q-40-stale-index-verdict-pass" \
       --context "$FIX/q-40-stale-index-verdict-pass/pr-context.json"
  [ "$status" -eq 1 ]
  # B6, from the ONE implementation of receipt validity. A second copy of the stale-index
  # rule here would be two implementations of one rule, each green against its own copy.
  [[ "$output" == *"[B6]"* ]] || { echo "the delegated reason lost the guard's class:"; echo "$output"; return 1; }
}

@test "q-41 the PR context records no head_sha" {
  arm_row q-41-context-has-no-head-sha REFUSE Q1 "records no head_sha"
}

@test "q-42 the PR context records no number" {
  arm_row q-42-context-has-no-number REFUSE Q1 "records no number"
}

@test "q-43 the PR head does not resolve to a commit" {
  arm_row q-43-context-head-unresolvable REFUSE Q1 "does not resolve to a commit"
}

@test "q-44 a PR that edits the merge MECHANISM does not merge itself (S13.8)" {
  arm_row q-44-edits-the-merge-mechanism REFUSE Q9 "edits the merge mechanism itself"
}

@test "q-45 a guard-touching diff with a diff-scoped mutation run" {
  arm_row q-45-guard-diff-without-a-guard-scoped-run REFUSE Q2 "touches a guard"
}

@test "q-46 the reviewed head is not an ancestor of the tip" {
  arm_row q-46-reviewed-head-not-an-ancestor REFUSE Q3 "is not an ancestor of the pull request tip"
}

@test "q-47 an unreviewed commit rides in behind the receipt (S13.3.a)" {
  arm_row q-47-unreviewed-commit-rides-in REFUSE Q3 "crates/aprender-core/src/late.rs"
}

@test "q-48 main_sha_at_review does not resolve" {
  arm_row q-48-main-sha-at-review-unresolvable REFUSE Q4 "does not resolve to a commit"
}

@test "q-49 main moved and the delta region was never swept (S13.3.b)" {
  arm_row q-49-delta-region-never-swept REFUSE Q4 "delta_sweep.status is 'not-run'"
}

@test "q-50 a clean delta sweep with no recorded needle set" {
  arm_row q-50-delta-clean-with-no-needles REFUSE Q4 "duplication_needles is absent or empty"
}

@test "q-51 the delta sweep did not replay the needle set the review searched with" {
  arm_row q-51-needles-digest-mismatch REFUSE Q4 "needles_sha256"
}

# ---------------------------------------------------------------------------
# THE ROWS THAT PERMIT. Every `flip` mutant dies here.
# ---------------------------------------------------------------------------

@test "q-52 a clean, unanimous, cross-vendor quorum PERMITS" {
  arm_row q-52-permits-a-clean-quorum PERMIT
}

@test "q-53 an evidence-only commit after the reviewed SHA PERMITS (discriminates with q-47)" {
  arm_row q-53-permits-an-evidence-only-tip PERMIT
}

@test "q-54 a swept delta region PERMITS (discriminates with q-49)" {
  arm_row q-54-permits-a-swept-delta-region PERMIT
}

@test "q-55 a guard-touching diff with a GUARD-scoped 100% run PERMITS (discriminates with q-45)" {
  arm_row q-55-permits-a-guard-scoped-run PERMIT
}

# ---------------------------------------------------------------------------
# The mechanism's own guarantees.
# ---------------------------------------------------------------------------

@test "--explain NEVER calls gh, and the stub proves it rather than the exit code" {
  arm_row q-52-permits-a-clean-quorum PERMIT
  # An exit code of 0 would also be produced by a gh that happened to succeed. The
  # empty call log is the assertion: nothing reached the network.
  [ ! -s "$GH_STUB_LOG" ] || { echo "gh was invoked under --explain:"; cat "$GH_STUB_LOG"; return 1; }
}

@test "q-56 the gate under its OTHER required spelling PERMITS" {
  arm_row q-56-permits-the-bare-gate-spelling PERMIT
}

@test "IDEMPOTENT: a PR whose auto-merge is already armed is a no-op, not an error" {
  local d="$WORK/already"
  mkdir -p "$d"
  cp "$FIX/q-52-permits-a-clean-quorum/receipt.intoto.jsonl" \
     "$FIX/q-52-permits-a-clean-quorum/receipt.intoto.jsonl.minisig" \
     "$FIX/q-52-permits-a-clean-quorum/findings.sarif" "$d/"
  jq '.auto_merge_armed = true' "$FIX/q-52-permits-a-clean-quorum/pr-context.json" > "$d/pr-context.json"
  run "$ARM" --pr 2783 --receipt "$d" --context "$d/pr-context.json"
  [ "$status" -eq 0 ]
  [[ "$output" == *"ALREADY-ARMED"* ]]
  # The arming verb, not --explain: if the script armed a second time the stub would
  # have recorded the call and exited 1.
  [ ! -s "$GH_STUB_LOG" ] || { echo "a second arm was attempted:"; cat "$GH_STUB_LOG"; return 1; }
}

@test "the positive control ABORTS the run when its seed is missing" {
  PR_REVIEW_QUORUM_CONTROL_DIR="$WORK/there-is-no-control-here" \
    run "$ARM" --explain --pr 2783 --receipt "$FIX/q-52-permits-a-clean-quorum" \
        --context "$FIX/q-52-permits-a-clean-quorum/pr-context.json"
  [ "$status" -eq 2 ]
  [[ "$output" == *"positive-control fixture is missing"* ]]
}

@test "the positive control ABORTS the run when its seed stops refusing" {
  # A control fixture edited until it PERMITS is a control that proves nothing. The
  # run must stop rather than report the verdicts it went on to compute.
  local seed="$WORK/controls/single-vendor"
  mkdir -p "$seed"
  cp "$FIX/quorum-control/single-vendor/findings.sarif" "$seed/"
  cp "$FIX/quorum-control/single-vendor/pr-context.json" "$seed/"
  jq -c '.predicate.autonomy.quorum[1].vendor = "google"' \
     "$FIX/quorum-control/single-vendor/receipt.intoto.jsonl" > "$seed/receipt.intoto.jsonl"
  cp "$FIX/quorum-control/single-vendor/receipt.intoto.jsonl.minisig" "$seed/"
  PR_REVIEW_QUORUM_CONTROL_DIR="$WORK/controls" \
    run "$ARM" --explain --pr 2783 --receipt "$FIX/q-52-permits-a-clean-quorum" \
        --context "$FIX/q-52-permits-a-clean-quorum/pr-context.json"
  [ "$status" -eq 1 ]
  [[ "$output" == *"POSITIVE CONTROL FAILED"* ]]
}

@test "the positive control ABORTS when it fires under the WRONG class" {
  # Mislabeled evidence is worse than none: a control that fires for a reason other
  # than the one it names stops being evidence that its branch is wired.
  local seed="$WORK/wrongclass/single-vendor"
  mkdir -p "$seed"
  cp "$FIX/quorum-control/single-vendor/findings.sarif" \
     "$FIX/quorum-control/single-vendor/pr-context.json" \
     "$FIX/quorum-control/single-vendor/receipt.intoto.jsonl.minisig" "$seed/"
  jq -c '.predicate.verdict = "DEGRADED"' \
     "$FIX/quorum-control/single-vendor/receipt.intoto.jsonl" > "$seed/receipt.intoto.jsonl"
  PR_REVIEW_QUORUM_CONTROL_DIR="$WORK/wrongclass" \
    run "$ARM" --explain --pr 2783 --receipt "$FIX/q-52-permits-a-clean-quorum" \
        --context "$FIX/q-52-permits-a-clean-quorum/pr-context.json"
  [ "$status" -eq 1 ]
  [[ "$output" == *"POSITIVE CONTROL MISFIRED"* ]]
  # IT MUST MISFIRE ON THE CLASS, and say which it expected and which it got. Asserting
  # only "MISFIRED" left the class comparison entirely untested: with that comparison
  # deleted this control still misfires - on the REASON - and the mutant SURVIVED the
  # whole sweep. Measured on run 1, 2026-08-31, not argued.
  [[ "$output" == *"Expected a refusal under Q5"* ]]
  [[ "$output" == *"got Q6"* ]]
  [[ "$output" != *"PERMIT"* ]]
}

@test "the positive control ABORTS when it fires on the right class but the WRONG BRANCH" {
  # Q5 has six branches. A control seeded with a SELF-REVIEW receipt is still refused
  # under Q5 - on `reviewer_actor.id = author_actor.id`, not on the vendor-distinctness
  # clause it exists to pin. The class is right and the evidence is worthless, so only
  # the REASON assertion can catch it; without this row, deleting that assertion left
  # the mutant alive (measured, run 1).
  local seed="$WORK/wrongbranch/single-vendor"
  mkdir -p "$seed"
  cp "$FIX/q-22-self-review/receipt.intoto.jsonl" \
     "$FIX/q-22-self-review/receipt.intoto.jsonl.minisig" \
     "$FIX/q-22-self-review/findings.sarif" \
     "$FIX/q-22-self-review/pr-context.json" "$seed/"
  PR_REVIEW_QUORUM_CONTROL_DIR="$WORK/wrongbranch" \
    run "$ARM" --explain --pr 2783 --receipt "$FIX/q-52-permits-a-clean-quorum" \
        --context "$FIX/q-52-permits-a-clean-quorum/pr-context.json"
  [ "$status" -eq 1 ]
  [[ "$output" == *"POSITIVE CONTROL MISFIRED"* ]]
  [[ "$output" == *"on the wrong branch"* ]]
  [[ "$output" != *"PERMIT"* ]]
}

@test "an absent tool is a REFUSAL to arm (exit 2), never a skip" {
  # A gate that cannot execute its own checks must not arm a merge. Without this row
  # the tool check was never run in its RED direction and `tools-absent-is-a-skip`
  # SURVIVED the whole sweep (run of 2026-08-31, 120/122) - the script simply never met
  # a box without jq.
  #
  # A PRIVATE PATH holding bash, git and sha256sum but NOT jq. Not `PATH=` emptied:
  # the shebang is `/usr/bin/env bash`, so env must still find bash, and a test that
  # cannot start the script proves nothing about the script.
  local bin="$WORK/thin-bin"
  mkdir -p "$bin"
  local t
  for t in bash env git sha256sum grep sed awk cat mkdir rm cp mktemp tr cut basename dirname printf; do
    local src
    src=$(command -v "$t" 2>/dev/null) || continue
    ln -sf "$src" "$bin/$t"
  done
  [ ! -e "$bin/jq" ]
  PATH="$bin" run "$ARM" --explain --pr 2783 \
      --receipt "$FIX/q-52-permits-a-clean-quorum" \
      --context "$FIX/q-52-permits-a-clean-quorum/pr-context.json"
  # 2, not 1: the BOX could not answer. A broken environment reported as a refusal is
  # the env-vs-defect confusion three guards in this repository shipped in one day.
  [ "$status" -eq 2 ] || { echo "expected exit 2 (ENV), got $status:"; echo "$output"; return 1; }
  [[ "$output" == *"cannot run:"* ]]
  [[ "$output" == *"jq"* ]]
  [[ "$output" != *"PERMIT"* ]]
}

# ---------------------------------------------------------------------------
# PRREV-019 — the nine forged receipts.
#
# An adversarial verifier holding tests/fixtures/pr-review/keys/pr-review-test-TEST-ONLY.key
# got nine bad receipts to PERMIT. Every signature was GENUINE — the key signs whatever
# bytes it is given — every document was schema-valid, and every one of the nine evaded a
# clause shaped `refuse if field == "<the one bad spelling>"`.
#
# THE ROWS BELOW COME IN PAIRS, AND THE PAIRING IS THE TEST. A vocabulary clause that
# refuses everything would satisfy any single row; it cannot satisfy a pair whose two
# members must refuse on DIFFERENT branches. q-58/q-59 carry a precision_class outside the
# vocabulary and must refuse on the vocabulary branch; q-60/q-61 carry the SAME field in a
# different case, must FOLD into the vocabulary, and must refuse on the substantive
# blocking branch. Same field, one character different, opposite branch.
# ---------------------------------------------------------------------------

@test "q-58 a precision_class outside { blocking, advisory } is not a permission" {
  arm_row q-58-precision-class-outside-the-vocabulary REFUSE Q6 "outside { blocking, advisory }"
}

@test "q-59 an ABSENT precision_class is not 'advisory' by default" {
  # The S3.0 rule one level down: 'the reviewer did not classify this' and 'the reviewer
  # classified this as harmless' must not be the same artifact.
  arm_row q-59-precision-class-absent REFUSE Q6 "outside { blocking, advisory }"
}

@test "q-60 'Blocking' FOLDS into the vocabulary and refuses as a blocker, not as a typo" {
  # The discrimination half of q-58. If this row refused on the VOCABULARY branch the
  # fold would be doing nothing, and a receipt could still hide a blocker by shouting it.
  arm_row q-60-blocking-in-a-different-case REFUSE Q6 "classed blocking under a PASS verdict"
}

@test "q-61 'BLOCKING' folds the same way" {
  arm_row q-61-blocking-shouted REFUSE Q6 "classed blocking under a PASS verdict"
}

@test "q-62 an autonomy_effect outside { refuse } is not a grant of autonomy" {
  arm_row q-62-autonomy-effect-outside-the-vocabulary REFUSE Q7 "outside { refuse }"
}

@test "q-63 'Refuse' FOLDS, and refuses as the S13.5 veto it is" {
  # The forged receipt carried an explicit cross-vendor refusal spelled with a capital R
  # and merged anyway. The pair with q-62 is what keeps the fold and the vocabulary from
  # collapsing into one clause that cannot tell a veto from a typo.
  arm_row q-63-veto-in-a-different-case REFUSE Q7 "autonomy_effect: refuse"
}

@test "q-64 a quorum vendor that is a NUMBER is distinct, and is not a vendor" {
  arm_row q-64-vendor-is-not-an-identity REFUSE Q5 "not well-formed identities"
}

@test "q-65 'anthropic ' is not a second vendor beside 'anthropic'" {
  # Rejected rather than trimmed. A receipt whose identities need normalising to be
  # legible is malformed, and repairing it silently would hide the producer that emitted
  # it — which is the defect, not the space.
  arm_row q-65-vendor-carries-trailing-whitespace REFUSE Q5 "not well-formed identities"
}

@test "q-66 the author with one trailing space is still the author" {
  arm_row q-66-author-id-shifted-by-one-space REFUSE Q5 "not well-formed identities"
}

@test "q-67 one actor listed twice under two vendor labels is ONE reviewer" {
  # |distinct vendor| >= 2 holds and the quorum is one agent. S13.1 rests on the members
  # failing DIFFERENTLY, which is a property of who ran — not of what they were labelled.
  arm_row q-67-one-actor-under-two-vendors REFUSE Q5 "distinct actor id(s)"
}

@test "q-68 a survivors field that is a STRING is not an empty survivors field" {
  # jq's `?` suppresses the TYPE ERROR, not the value: `[ "12 survived"[]? ] | length`
  # is 0. A receipt that CONFESSED twelve survivors was counted as clean by the clause
  # written to stop exactly that.
  arm_row q-68-survivors-is-not-a-list REFUSE Q2 "survivors is a string, not an array"
}

@test "q-69 an ABSENT survivors field is 'not recorded', never 'none'" {
  arm_row q-69-survivors-absent REFUSE Q2 "survivors is absent"
}

@test "q-70 an anchor that is not on main makes the delta region empty for the wrong reason" {
  # `rev-list A..origin/main` is empty when main has not moved AND when A was never on
  # main. The producer chooses A, so anchoring at the PR's own head collapses S13.3.b's
  # region to nothing and skips the sweep with delta_sweep.status 'not-run'.
  arm_row q-70-anchor-is-not-on-main REFUSE Q4 "is not an ancestor of origin/main"
}

@test "the fold is a whitelist over a literal list, and here is its case table" {
  # S13.2 ships no new regex, so this table is over `index()` membership and a literal
  # whitespace set. This repository's guard patterns have been wrong six times and a
  # must-match / must-not-match table caught every one; re-run the table rather than
  # re-reading the code.
  local defs
  defs=$(sed -n "/^JQ_DEFS='/,/^'$/p" "$ARM" | sed "1s/^JQ_DEFS='//; \$d")

  # fold: MUST map every case variant onto the vocabulary member.
  for v in blocking Blocking BLOCKING bLoCkInG; do
    run jq -rn "$defs \"$v\" | fold"
    [ "$output" = blocking ] || { echo "fold(\"$v\") = '$output', wanted 'blocking'"; return 1; }
  done
  # fold: MUST NOT map a different token, or surrounding whitespace, onto it.
  for v in critical advisory "blocking " " blocking" blockings; do
    run jq -rn "$defs \"$v\" | fold"
    [ "$output" != blocking ] || { echo "fold(\"$v\") collapsed onto 'blocking'"; return 1; }
  done
  # fold: a non-string never becomes a vocabulary member.
  for v in 1 null true '[]' '{}'; do
    run jq -rn "$defs $v | fold"
    [ "$output" != blocking ] && [ "$output" != advisory ] \
      || { echo "fold($v) = '$output' entered the vocabulary"; return 1; }
  done

  # wellformed_id: MUST accept a plain identifier.
  for v in anthropic google "agent:agy-1.1.22/session-cross-vendor" "a"; do
    run jq -rn "$defs \"$v\" | wellformed_id"
    [ "$output" = true ] || { echo "wellformed_id(\"$v\") = $output, wanted true"; return 1; }
  done
  # wellformed_id: MUST reject every shape the attack used, and the empty string.
  for v in "anthropic " " anthropic" "" "	tabbed" "trailing	"; do
    run jq -rn "$defs \"$v\" | wellformed_id"
    [ "$output" = false ] || { echo "wellformed_id(\"$v\") = $output, wanted false"; return 1; }
  done
  # wellformed_id: MUST reject a non-string, whatever it is.
  for v in 1 null true '[]' '{}'; do
    run jq -rn "$defs $v | wellformed_id"
    [ "$output" = false ] || { echo "wellformed_id($v) = $output, wanted false"; return 1; }
  done
}

@test "MECHANISM_PATHS is a literal prefix list, and its own script is on it" {
  # The falsifiable property of S13.8: the PR that introduces S13 cannot auto-merge
  # itself. Checked here as a property of the list rather than of a fixture, because
  # the fixture repository has no aprender paths in it.
  grep -qx 'scripts/pr_review_quorum_arm.sh' <<<"$(sed -n "/^MECHANISM_PATHS='/,/^'$/p" "$ARM")"
  grep -qx 'docs/specifications/PR-REVIEW-SKILL-002-v2.md' <<<"$(sed -n "/^MECHANISM_PATHS='/,/^'$/p" "$ARM")"
  grep -qx 'tests/pr-review-quorum.bats' <<<"$(sed -n "/^MECHANISM_PATHS='/,/^'$/p" "$ARM")"
}

@test "every refusal site in the arm script carries a || return 1 terminator" {
  # scripts/mutate_quorum_arm.sh derives its `flip` mutants from that terminator. A
  # site without one is a site the mutation set silently skips — a rule with no
  # mutant, which is the same defect as a rule with no fixture.
  run awk '/refuse Q[0-9]/ { n += 1; if (index($0, "|| return 1") == 0) bad += 1 }
           END { printf "%d %d\n", n + 0, bad + 0 }' "$ARM"
  [ "$status" -eq 0 ]
  local n bad
  read -r n bad <<<"$output"
  [ "$bad" -eq 0 ] || { echo "$bad refusal site(s) have no terminator"; return 1; }
  [ "$n" -gt 40 ] || { echo "only $n refusal sites; the arm script has always had more"; return 1; }
}

@test "the q-* table has as many rows as this file has row tests" {
  # The same fact from two sides, which is what stops a fixture from being added and
  # never exercised — the dark-target class this repository has hit with benches, with
  # 58 cgp tests, and with tests/pr-review.bats itself before PRREV-006 wired it.
  local n
  n=$(find "$FIX" -maxdepth 1 -type d -name 'q-*' | wc -l)
  [ "$n" -eq 69 ] || { echo "expected 69 q-* fixture directories (q-02..q-70; q-01 is a path that must NOT exist), found $n"; false; }
}
