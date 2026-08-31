#!/usr/bin/env bash
# Regenerate the committed S13 quorum fixtures - the q-* rows of tests/pr-review-quorum.bats.
#
# ONE ROW PER REFUSAL PATH OF scripts/pr_review_quorum_arm.sh, PLUS FOUR THAT PERMIT.
#
# A refusal path with no row that exercises it is a gate that cannot fire, which is this
# repository's most common defect and the one PR-REVIEW-SKILL-002 exists to remove. So
# the count here is not a taste decision: scripts/mutate_quorum_arm.sh derives one `drop`
# mutant per `refuse Q<n>` site in the arm script, and a site with no row leaves a
# SURVIVOR - a rule the script states that a receipt could break with the table still
# green. The rows are named for their subject rather than for a line number, because a
# line number goes stale the first time somebody adds a clause above it.
#
# THE FOUR THAT PERMIT ARE NOT DECORATION. "Refuse everything" and "evaluate everything"
# produce the same colour on a table made only of RED rows, and every `flip` mutant in
# the set is killed by a row that must PERMIT. Three of them are DISCRIMINATION rows:
# q-53 differs from q-47 only in WHICH file the post-review commit touches, q-54 differs
# from q-49 only in whether the delta region was swept, and q-55 differs from q-45 only
# in the mutation scope. One variable, opposite verdict.
#
# Every digest is COMPUTED here and never typed:
#   findings_ref.sha256                  = sha256(findings.sarif as written)
#   autonomy.delta_sweep.needles_sha256  = sha256(join(duplication_needles, "\n"))
# A hand-typed digest is a fixture that passes for the wrong reason the first time
# somebody edits the file it describes.
#
# Usage: build-quorum-fixtures.sh
set -euo pipefail

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
KEY="$HERE/keys/pr-review-test-TEST-ONLY.key"
PUB="$HERE/keys/pr-review-test.pub"

command -v jq       >/dev/null || { echo "need jq" >&2; exit 1; }
command -v minisign >/dev/null || { echo "need minisign" >&2; exit 1; }
[ -f "$KEY" ] || { echo "missing fixture signing key $KEY" >&2; exit 1; }

sha_of() { awk -v k="$1" '$1==k{print $2}' "$HERE/expected-shas.txt"; }
C1=$(sha_of C1)   # the merge base of every fixture PR
C3=$(sha_of C3)   # origin/main's tip in the fixture repo
F1=$(sha_of F1)   # the GPU pull request head
D1=$(sha_of D1)   # the docs head - NOT an ancestor of F1, so a tip that never was
M1=$(sha_of M1)   # the head that edits the merge MECHANISM (S13.8)
H1=$(sha_of H1)   # a guard-shaped file that is NOT a mechanism path
T1=$(sha_of T1)   # F1 + a commit under evidence/pr-review/2783/ only
T2=$(sha_of T2)   # F1 + a commit nobody reviewed
for v in C1 C3 F1 D1 M1 H1 T1 T2; do
  [ -n "${!v}" ] || { echo "expected-shas.txt has no $v" >&2; exit 1; }
done
ZERO=0000000000000000000000000000000000000000

AUTHOR='agent:claude-opus-5/session-authoring'
REVIEWER='agent:claude-opus-5/session-review'
CROSSVENDOR='agent:agy-1.1.22/session-cross-vendor'

# The needle set S3.A searched with. S13.3.b requires the delta sweep to REPLAY it
# rather than re-derive it: a sweep that re-derives its needles from the diff is a
# second implementation of S3.A's derivation, each green against its own copy, which
# is D8 in the code written to enforce D8.
NEEDLES='["fused_kernel_launch","stream_ordering_guard","cuda_stream_pool"]'
NEEDLES_SHA=$(printf '%s' "$NEEDLES" | jq -r 'join("\n")' | sha256sum | cut -d' ' -f1)

# --- the four SARIF shapes -------------------------------------------------
sarif_clean() {
  cat <<'JSON'
{ "$schema": "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [ { "tool": { "driver": { "name": "pmat" } },
              "invocations": [ { "executionSuccessful": true, "toolExecutionNotifications": [] } ],
              "results": [ { "ruleId": "complexity_delta", "level": "note",
                "message": { "text": "The launch wrapper is unchanged in cyclomatic complexity." },
                "properties": { "grounding": "measured",
                  "source": "pmat analyze complexity --format json",
                  "failure_scenario": "None: the measurement shows no delta, and a no-delta measurement is still a measurement.",
                  "precision_class": "advisory" } } ] } ] }
JSON
}
# sarif_result <result-json> [<executionSuccessful>]
sarif_result() {
  cat <<JSON
{ "\$schema": "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [ { "tool": { "driver": { "name": "pmat" } },
              "invocations": [ { "executionSuccessful": ${2:-true}, "toolExecutionNotifications": [] } ],
              "results": [ $1 ] } ] }
JSON
}

R_BLOCKING='{ "ruleId": "device_behaviour_claim", "level": "error",
  "message": { "text": "The PR asserts kernels on separate streams are implicitly ordered." },
  "properties": { "grounding": "cited",
    "source": "nvidia-cuda-docs: CUDA C++ Programming Guide, Streams",
    "excerpt": "there is no implicit synchronization between them",
    "excerpt_sha256": "@@EXCERPT_SHA@@",
    "failure_scenario": "The second kernel reads the first kernel output before it is written.",
    "precision_class": "blocking" } }'
EXCERPT_SHA=$(printf '%s' 'there is no implicit synchronization between them' | sha256sum | cut -d' ' -f1)
R_BLOCKING=${R_BLOCKING//@@EXCERPT_SHA@@/$EXCERPT_SHA}

R_ASSERTED_BLOCKING='{ "ruleId": "reviewer_judgement", "level": "error",
  "message": { "text": "This refactor reads as riskier than the diff makes it look." },
  "properties": { "grounding": "asserted",
    "rationale": "judgement, not measurement",
    "failure_scenario": "A latent path nothing exercises.",
    "precision_class": "blocking" } }'

R_UNMARKED='{ "ruleId": "an_unmarked_claim", "level": "warning",
  "message": { "text": "The kernel is faster now." },
  "properties": { "failure_scenario": "Nothing says how this is known.",
    "precision_class": "advisory" } }'

R_VETO='{ "ruleId": "cross_vendor_reservation", "level": "warning",
  "message": { "text": "The stream-ordering change deserves a human read before it merges unattended." },
  "properties": { "grounding": "asserted",
    "rationale": "the cross-vendor reviewer is advisory under S7 and may refuse autonomy under S13.5",
    "failure_scenario": "An unattended merge of a concurrency change nobody read.",
    "precision_class": "advisory",
    "autonomy_effect": "refuse" } }'

# THE FOLD / VOCABULARY TABLE. Each of these is R_BLOCKING or R_VETO with EXACTLY ONE
# character-level edit to the field the S13.2 predicate reads. That is the whole point:
# the rows they build differ from q-17 and q-20 in the SPELLING of one token and in
# nothing else, so a row that passes for a reason other than the spelling is visible.
#
#   MUST fold INTO the vocabulary and refuse on the SUBSTANTIVE branch:
#     "Blocking"  "BLOCKING"      -> Q6 "classed blocking under a PASS verdict"
#     "Refuse"                    -> Q7 "autonomy_effect: refuse"
#   MUST fall OUTSIDE the vocabulary and refuse on the VOCABULARY branch:
#     "critical"  <absent>        -> Q6 "outside { blocking, advisory }"
#     "veto"                      -> Q7 "outside { refuse }"
#
# Nine forged receipts were PERMITTED because the predicate compared these fields against
# the ONE bad spelling and read everything else as consent. A blacklist over a field no
# schema constrains is fail-open on its own complement; the table above is the whitelist
# that replaced it, and these fixtures are what stop it silently narrowing again.
R_BLOCKING_FOLDED=${R_BLOCKING/'"precision_class": "blocking"'/'"precision_class": "Blocking"'}
R_BLOCKING_SHOUTED=${R_BLOCKING/'"precision_class": "blocking"'/'"precision_class": "BLOCKING"'}
R_PC_NOVEL=${R_BLOCKING/'"precision_class": "blocking"'/'"precision_class": "critical"'}
R_PC_ABSENT=${R_BLOCKING/'"precision_class": "blocking"'/'"note": "this result declines to name a precision class"'}
R_VETO_FOLDED=${R_VETO/'"autonomy_effect": "refuse"'/'"autonomy_effect": "Refuse"'}
R_AE_NOVEL=${R_VETO/'"autonomy_effect": "refuse"'/'"autonomy_effect": "veto"'}
for v in R_BLOCKING_FOLDED R_BLOCKING_SHOUTED R_PC_NOVEL R_PC_ABSENT R_VETO_FOLDED R_AE_NOVEL; do
  # A substitution that matched nothing builds a COPY of the row it was derived from,
  # which then passes for the wrong reason. Checked rather than assumed.
  [ "${!v}" != "$R_BLOCKING" ] && [ "${!v}" != "$R_VETO" ] \
    || { echo "$v is identical to the shape it was derived from; the edit matched nothing" >&2; exit 1; }
done

# --- the base receipt ------------------------------------------------------
# A COMPLETE, VALID, PASS receipt on the GPU head, carrying the S13 autonomy block.
# Every row below is this document with ONE jq edit, so a row differs from the one that
# permits in exactly the variable it is named for.
base_receipt() {
  cat <<JSON
{
  "_type": "https://in-toto.io/Statement/v1",
  "subject": [ { "name": "git+https://github.com/paiml/aprender",
                 "digest": { "sha1": "$F1" } } ],
  "predicateType": "https://paiml.dev/attestations/pr-review/v2",
  "predicate": {
    "skill_version": "2.0.0",
    "attestation_level": "L1-self",
    "pr": 2783,
    "base_sha": "$C1",
    "head_sha": "$F1",
    "author_actor":   { "kind": "agent", "id": "$AUTHOR" },
    "reviewer_actor": { "kind": "agent", "id": "$REVIEWER" },
    "affected_crates": ["aprender-core"],
    "verdict": "PASS",
    "autonomy": {
      "requested": true,
      "main_sha_at_review": "$C3",
      "quorum": [
        { "role": "primary", "vendor": "anthropic",
          "actor": { "kind": "agent", "id": "$REVIEWER" },
          "verdict": "PASS", "refusal": null },
        { "role": "cross_vendor", "vendor": "google",
          "actor": { "kind": "agent", "id": "$CROSSVENDOR" },
          "verdict": "PASS", "refusal": null }
      ],
      "delta_sweep": { "status": "clean",
                       "region": "$C3..refs/remotes/origin/main",
                       "needles_sha256": "$NEEDLES_SHA", "hits": [] }
    },
    "consultations": {
      "pmat": { "status": "consulted", "index_commit": "$C1", "index_is_ancestor": true,
        "complexity_delta": [], "tdg_delta": [], "satd_introduced": [],
        "duplication_hits": [], "cache_hits": 0,
        "duplication_coverage": { "rust": "semantic", "shell": "lexical", "python": "lexical",
          "config": "lexical", "docs": "lexical", "other": "lexical",
          "sibling_branches": "lexical", "merge_base_to_main": "lexical" },
        "duplication_horizon": [ "head=HEAD",
          "siblings=refs/remotes/origin/* unmerged into origin/main",
          "merge_base_to_main=$C1..refs/remotes/origin/main" ],
        "duplication_needles": $NEEDLES,
        "horizon_branches_total": 0, "horizon_branches_scanned": 0, "symbols_searched": 3 },
      "cuda": { "status": "consulted", "trigger_reason": "diff touches src/cuda/kernel.cu",
        "queries": [ { "q": "stream ordering guarantees between non-default streams",
          "excerpt_sha256": "$EXCERPT_SHA", "result": "found" } ] },
      "crux": { "status": "not-triggered",
        "trigger_reason": "no CLI flag, HTTP route, MCP tool, config key or output format changed",
        "surfaces": [], "contracts": [], "comparative_claims": [] },
      "mutation": { "status": "consulted", "scope": "in-diff",
        "attempted": 37, "killed": 37, "survivors": [] }
    },
    "findings_ref": { "path": "findings.sarif", "sha256": "@@FINDINGS_SHA@@" },
    "cost": { "input_tokens": 41200, "output_tokens": 3180, "wall_seconds": 74 }
  }
}
JSON
}

base_context() {
  cat <<JSON
{ "number": 2783,
  "base_ref": "main",
  "head_sha": "$F1",
  "labels": [ "ready-to-merge" ],
  "reviews": [ { "state": "APPROVED", "dismissed": false } ],
  "checks": { "ci / gate": "success", "workspace-test": "success" },
  "auto_merge_armed": false }
JSON
}

# emit <row> <receipt-jq> <context-jq> <sarif-producer...>
emit() {
  local row=$1 rfilter=$2 cfilter=$3; shift 3
  local dir="$HERE/$row"
  case "$row" in ""|*/*|.|..) echo "refusing to build fixture '$row'" >&2; exit 1 ;; esac
  if [ -z "$dir" ] || [ "$dir" = "/" ]; then echo "refusing rm -rf on $dir" >&2; exit 1; fi
  rm -rf -- "$dir"; mkdir -p -- "$dir"
  "$@" | jq . > "$dir/findings.sarif"
  local fsha
  fsha=$(sha256sum "$dir/findings.sarif" | cut -d' ' -f1)
  base_receipt | jq -c "$rfilter" \
    | sed "s/@@FINDINGS_SHA@@/$fsha/" > "$dir/receipt.intoto.jsonl"
  base_context | jq -c "$cfilter" > "$dir/pr-context.json"
  minisign -q -S -s "$KEY" -m "$dir/receipt.intoto.jsonl" \
    -t "S13 quorum fixture $row" -c "TEST ONLY fixture signature" </dev/null
  minisign -q -V -m "$dir/receipt.intoto.jsonl" -p "$PUB" >/dev/null \
    || { echo "self-check failed: $row signature does not verify" >&2; exit 1; }
  echo "built $row"
}

ID='.'

# ===========================================================================
# Q1 - the receipt is missing, unreadable, or carries no S13 autonomy block.
# The first four are STRUCTURAL: they are built by deleting a file, because a
# fixture that describes a missing file is not a missing file.
# (q-01, "no such directory", needs no fixture at all - the bats row names a
# path that was never created, which is the only honest way to build it.)
# ===========================================================================
emit q-02-receipt-file-missing        "$ID" "$ID" sarif_clean
rm -f -- "$HERE/q-02-receipt-file-missing/receipt.intoto.jsonl" \
         "$HERE/q-02-receipt-file-missing/receipt.intoto.jsonl.minisig"

emit q-03-sarif-missing               "$ID" "$ID" sarif_clean
rm -f -- "$HERE/q-03-sarif-missing/findings.sarif"

emit q-04-unsigned-receipt            "$ID" "$ID" sarif_clean
rm -f -- "$HERE/q-04-unsigned-receipt/receipt.intoto.jsonl.minisig"

emit q-05-receipt-unparseable         "$ID" "$ID" sarif_clean
printf 'this is not JSON, and a guard that shrugs at that is not a guard\n' \
  > "$HERE/q-05-receipt-unparseable/receipt.intoto.jsonl"

emit q-06-sarif-unparseable           "$ID" "$ID" sarif_clean
printf '{ "version": "2.1.0", "runs": [ \n' > "$HERE/q-06-sarif-unparseable/findings.sarif"

emit q-07-context-missing             "$ID" "$ID" sarif_clean
rm -f -- "$HERE/q-07-context-missing/pr-context.json"

emit q-08-context-unparseable         "$ID" "$ID" sarif_clean
printf '{ "number": 2783, \n' > "$HERE/q-08-context-unparseable/pr-context.json"

emit q-09-no-autonomy-block           'del(.predicate.autonomy)' "$ID" sarif_clean
emit q-10-autonomy-not-requested      '.predicate.autonomy.requested = false' "$ID" sarif_clean
emit q-11-no-main-sha-at-review       'del(.predicate.autonomy.main_sha_at_review)' "$ID" sarif_clean
emit q-12-empty-quorum                '.predicate.autonomy.quorum = []' "$ID" sarif_clean
emit q-13-no-delta-sweep              'del(.predicate.autonomy.delta_sweep)' "$ID" sarif_clean
emit q-14-delta-sweep-status-outside-vocabulary \
                                      '.predicate.autonomy.delta_sweep.status = "probably fine"' "$ID" sarif_clean

# ===========================================================================
# Q6 - the verdict, and the facts the verdict summarises.
# ===========================================================================
emit q-15-verdict-degraded            '.predicate.verdict = "DEGRADED"' "$ID" sarif_clean
emit q-16-asserted-finding-classed-blocking "$ID" "$ID" sarif_result "$R_ASSERTED_BLOCKING"
emit q-17-blocking-finding-survives   "$ID" "$ID" sarif_result "$R_BLOCKING"
emit q-18-unmarked-claim              "$ID" "$ID" sarif_result "$R_UNMARKED"
# The results array is CLEAN here on purpose: the row has to reach the
# executionSuccessful clause, and a blocking result would refuse two clauses earlier,
# reporting a kill this row never earned.
emit q-19-tool-execution-failed       "$ID" "$ID" sarif_result \
  '{ "ruleId": "complexity_delta", "level": "note",
     "message": { "text": "No delta." },
     "properties": { "grounding": "measured", "source": "pmat analyze complexity",
       "failure_scenario": "None.", "precision_class": "advisory" } }' false

# ===========================================================================
# Q7 - the cross-vendor reviewer's veto (S13.5). It may not BLOCK and it may
# refuse the unattended merge; that is the only power S13.1 gives it.
# ===========================================================================
emit q-20-cross-vendor-refuses-autonomy "$ID" "$ID" sarif_result "$R_VETO"

# ===========================================================================
# Q5 - separation, threefold.
# ===========================================================================
emit q-21-author-actor-absent         'del(.predicate.author_actor.id)' "$ID" sarif_clean
emit q-22-self-review                 ".predicate.reviewer_actor.id = \"$AUTHOR\"" "$ID" sarif_clean
emit q-23-single-vendor-quorum        '.predicate.autonomy.quorum[1].vendor = "anthropic"' "$ID" sarif_clean
emit q-24-quorum-role-missing         '.predicate.autonomy.quorum[1].role = "primary"' "$ID" sarif_clean
emit q-25-author-sits-in-the-quorum   ".predicate.autonomy.quorum[1].actor.id = \"$AUTHOR\"" "$ID" sarif_clean
emit q-26-quorum-not-unanimous        '.predicate.autonomy.quorum[1].refusal = "the delta region was never swept"' "$ID" sarif_clean

# ===========================================================================
# Q2 - a consultation did not run, or ran over nothing.
# ===========================================================================
emit q-27-pmat-not-consulted          '.predicate.consultations.pmat = { "status": "unreachable", "trigger_reason": "pmat MCP: ConnectionRefused" }' "$ID" sarif_clean
# S3.A's trigger is UNCONDITIONAL, so `not-triggered` is never true of pmat. This is the
# one shape the status loop below cannot catch: it ADMITS not-triggered for every
# consultation, so without this row the pmat clause is only ever reached by receipts the
# loop would refuse anyway - and scripts/mutate_quorum_arm.sh reported exactly that,
# `refuse-27-drop` SURVIVED (run of 2026-08-31, 120/122).
emit q-57-pmat-not-triggered          '.predicate.consultations.pmat = { "status": "not-triggered", "trigger_reason": "docs-only, nothing worth indexing" }' "$ID" sarif_clean
emit q-28-not-triggered-with-no-reason '.predicate.consultations.crux.trigger_reason = ""' "$ID" sarif_clean
emit q-29-consultation-unreachable    '.predicate.consultations.cuda = { "status": "unreachable", "trigger_reason": "nvidia-cuda-docs MCP: ConnectionRefused" }' "$ID" sarif_clean
emit q-30-vacuous-zero-symbols        '.predicate.consultations.pmat.symbols_searched = 0' "$ID" sarif_clean
emit q-31-mutation-survivor           '.predicate.consultations.mutation = { "status": "consulted", "scope": "in-diff", "attempted": 37, "killed": 36, "survivors": [ { "mutant": "reject-31-drop", "file": "scripts/check_pr_review_receipt.sh", "line": 501, "killed": false } ] }' "$ID" sarif_clean
emit q-32-duplication-surface-unsearched '.predicate.consultations.pmat.duplication_coverage.shell = "none"' "$ID" sarif_clean
emit q-33-horizon-region-unswept      '.predicate.consultations.pmat.duplication_horizon = [ "head=HEAD", "siblings=none", "merge_base_to_main=none" ]' "$ID" sarif_clean

# ===========================================================================
# Q8 / Q9 / Q10 - the human surfaces, eligibility, and the mechanical checks.
# These are CONTEXT edits: the receipt is the one that permits, and the pull
# request around it is what changed.
# ===========================================================================
emit q-34-autonomy-hold-label         "$ID" '.labels += ["autonomy-hold"]' sarif_clean
emit q-35-changes-requested-open      "$ID" '.reviews += [ { "state": "CHANGES_REQUESTED", "dismissed": false } ]' sarif_clean
emit q-36-not-targeting-main          "$ID" '.base_ref = "release/0.65"' sarif_clean
emit q-37-workspace-test-not-green    "$ID" '.checks["workspace-test"] = "failure"' sarif_clean
emit q-38-gate-check-absent           "$ID" 'del(.checks["ci / gate"])' sarif_clean
# The kill switch is a property of origin/main, not of the receipt: the bats row points
# PR_REVIEW_REPO at a copy whose origin/main is K1. The fixture is the one that PERMITS,
# so the row differs from q-52 in the REPOSITORY and in nothing else.
emit q-39-kill-switch-on-origin-main  "$ID" "$ID" sarif_clean

# ===========================================================================
# Q1 (phase B) / Q9 / Q2 / Q3 / Q4 - the repository clauses.
# ===========================================================================
emit q-40-stale-index-verdict-pass \
  ".predicate.consultations.pmat.index_commit = \"$C3\" | .predicate.consultations.pmat.index_is_ancestor = false" \
  "$ID" sarif_clean
emit q-41-context-has-no-head-sha     "$ID" 'del(.head_sha)' sarif_clean
emit q-42-context-has-no-number       "$ID" 'del(.number)' sarif_clean
emit q-43-context-head-unresolvable   "$ID" ".head_sha = \"$ZERO\"" sarif_clean

# The mechanism head. Its diff touches scripts/check_pr_review_receipt.sh, which is on
# MECHANISM_PATHS and is also guard-shaped, so S3.D's trigger fires and the mutation
# consultation must be present and GUARD-scoped - otherwise this row would refuse under
# Q2 and prove nothing about Q9.
emit q-44-edits-the-merge-mechanism \
  ".subject[0].digest.sha1 = \"$M1\" | .predicate.head_sha = \"$M1\"
   | .predicate.consultations.cuda = { \"status\": \"not-triggered\", \"trigger_reason\": \"no changed path and no commit message matches the S3.B trigger\" }
   | .predicate.consultations.mutation = { \"status\": \"consulted\", \"scope\": \"guard\", \"attempted\": 185, \"killed\": 185, \"survivors\": [] }" \
  ".head_sha = \"$M1\"" sarif_clean

# The guard head. scripts/check_no_claim_literals.sh is guard-shaped and is NOT on
# MECHANISM_PATHS: one variable different from q-44, opposite clause.
emit q-45-guard-diff-without-a-guard-scoped-run \
  ".subject[0].digest.sha1 = \"$H1\" | .predicate.head_sha = \"$H1\"
   | .predicate.consultations.cuda = { \"status\": \"not-triggered\", \"trigger_reason\": \"no changed path and no commit message matches the S3.B trigger\" }
   | .predicate.consultations.mutation = { \"status\": \"consulted\", \"scope\": \"in-diff\", \"attempted\": 37, \"killed\": 37, \"survivors\": [] }" \
  ".head_sha = \"$H1\"" sarif_clean

emit q-46-reviewed-head-not-an-ancestor "$ID" ".head_sha = \"$D1\"" sarif_clean
emit q-47-unreviewed-commit-rides-in    "$ID" ".head_sha = \"$T2\"" sarif_clean

emit q-48-main-sha-at-review-unresolvable ".predicate.autonomy.main_sha_at_review = \"$ZERO\"" "$ID" sarif_clean
emit q-49-delta-region-never-swept \
  ".predicate.autonomy.main_sha_at_review = \"$C1\" | .predicate.autonomy.delta_sweep.status = \"not-run\"" \
  "$ID" sarif_clean
emit q-50-delta-clean-with-no-needles \
  ".predicate.autonomy.main_sha_at_review = \"$C1\" | del(.predicate.consultations.pmat.duplication_needles)" \
  "$ID" sarif_clean
emit q-51-needles-digest-mismatch \
  ".predicate.autonomy.main_sha_at_review = \"$C1\"
   | .predicate.autonomy.delta_sweep.needles_sha256 = \"$ZERO$ZERO\"" \
  "$ID" sarif_clean

# ===========================================================================
# PRREV-019 - THE FORGED-RECEIPT ROWS.
#
# An adversarial verifier holding tests/fixtures/pr-review/keys/pr-review-test-TEST-ONLY.key
# produced nine receipts that scripts/pr_review_quorum_arm.sh PERMITTED. Every signature
# was genuine and every document was schema-valid; each evaded a clause of the form
# `refuse if field == "<the one bad spelling>"`. These rows are those receipts, reduced
# to one variable each, so the clauses that now catch them cannot silently narrow back.
#
# THEY COME IN DISCRIMINATION PAIRS. q-58/q-60 carry the SAME field with two spellings
# and must refuse on DIFFERENT branches - one folds into the vocabulary and is caught by
# the substantive rule, the other does not and is caught by the vocabulary rule. A single
# row could be satisfied by a clause that refuses everything; a pair cannot.
# ===========================================================================

# Q6 - precision_class. The vocabulary is two words (S13: "blocking|advisory").
emit q-58-precision-class-outside-the-vocabulary "$ID" "$ID" sarif_result "$R_PC_NOVEL"
emit q-59-precision-class-absent                 "$ID" "$ID" sarif_result "$R_PC_ABSENT"
emit q-60-blocking-in-a-different-case           "$ID" "$ID" sarif_result "$R_BLOCKING_FOLDED"
emit q-61-blocking-shouted                       "$ID" "$ID" sarif_result "$R_BLOCKING_SHOUTED"

# Q7 - autonomy_effect. S13.5 defines exactly one value, so the field is absent or a veto.
emit q-62-autonomy-effect-outside-the-vocabulary "$ID" "$ID" sarif_result "$R_AE_NOVEL"
emit q-63-veto-in-a-different-case               "$ID" "$ID" sarif_result "$R_VETO_FOLDED"

# Q5 - the quorum members are IDENTITIES, and a string comparison reads two spellings of
# one identity as two identities. The vendors here are the NUMBERS 1 and 2 (distinct, and
# not vendors); then one vendor with a trailing space, which counted as a second vendor
# beside `anthropic`; then the AUTHOR's id with a trailing space, which walked past
# `m.actor.id != author_actor.id`.
emit q-64-vendor-is-not-an-identity \
  '.predicate.autonomy.quorum[0].vendor = 1 | .predicate.autonomy.quorum[1].vendor = 2' "$ID" sarif_clean
emit q-65-vendor-carries-trailing-whitespace \
  '.predicate.autonomy.quorum[1].vendor = "anthropic "' "$ID" sarif_clean
emit q-66-author-id-shifted-by-one-space \
  ".predicate.autonomy.quorum[1].actor.id = \"$AUTHOR \"" "$ID" sarif_clean
# ONE actor, two vendor LABELS. This satisfies |distinct vendor| >= 2 and is one
# reviewer; S13.1 rests on the members failing DIFFERENTLY, which is about who ran.
emit q-67-one-actor-under-two-vendors \
  ".predicate.autonomy.quorum[1].actor.id = \"$REVIEWER\"" "$ID" sarif_clean

# Q2 - jq's `?` suppresses the TYPE ERROR, not the value: over a string, `.survivors[]?`
# yields the empty stream and the count is 0. The first row is the receipt that CONFESSED
# twelve survivors and was counted as clean; the second is the S3.0 half - an absent
# record read as an empty one.
emit q-68-survivors-is-not-a-list \
  '.predicate.consultations.mutation.survivors = "12 survived, shipping anyway"' "$ID" sarif_clean
emit q-69-survivors-absent \
  'del(.predicate.consultations.mutation.survivors)' "$ID" sarif_clean

# Q4 - `rev-list A..origin/main` is empty when main has not moved AND when A is not on
# main at all. The producer chooses A, so anchoring at the PR's own head collapses the
# S13.3.b region to nothing and skips the sweep. F1 is this PR's head: it resolves, and
# it is not an ancestor of origin/main.
emit q-70-anchor-is-not-on-main \
  ".predicate.autonomy.main_sha_at_review = \"$F1\" | .predicate.autonomy.delta_sweep.status = \"not-run\"" \
  "$ID" sarif_clean

# ===========================================================================
# THE ROWS THAT PERMIT. Without them "refuse everything" reads green, and every
# `flip` mutant in scripts/mutate_quorum_arm.sh is killed here rather than above.
# ===========================================================================
emit q-52-permits-a-clean-quorum      "$ID" "$ID" sarif_clean
emit q-53-permits-an-evidence-only-tip "$ID" ".head_sha = \"$T1\"" sarif_clean
emit q-54-permits-a-swept-delta-region \
  ".predicate.autonomy.main_sha_at_review = \"$C1\"
   | .predicate.autonomy.delta_sweep.region = \"$C1..refs/remotes/origin/main\"" \
  "$ID" sarif_clean
# Required checks live in TWO places here: branch protection names `ci / gate`, ruleset
# 13878864 names a bare `gate`. Reading only one gives a wrong answer about what blocks
# a merge, so both spellings are accepted -- and this row is what proves the second
# spelling is not dead code. Without it, deleting the fallback leaves the table green.
emit q-56-permits-the-bare-gate-spelling \
  "$ID" '.checks = { "gate": "success", "workspace-test": "success" }' sarif_clean

emit q-55-permits-a-guard-scoped-run \
  ".subject[0].digest.sha1 = \"$H1\" | .predicate.head_sha = \"$H1\"
   | .predicate.consultations.cuda = { \"status\": \"not-triggered\", \"trigger_reason\": \"no changed path and no commit message matches the S3.B trigger\" }
   | .predicate.consultations.mutation = { \"status\": \"consulted\", \"scope\": \"guard\", \"attempted\": 185, \"killed\": 185, \"survivors\": [] }" \
  ".head_sha = \"$H1\"" sarif_clean

# ===========================================================================
# The positive-control seed. scripts/pr_review_quorum_arm.sh loads it BEFORE it
# evaluates anything real and requires a Q5 refusal on the vendor-distinctness
# branch. It is a COPY of q-23 under its own name, because a control that is an
# alias for a table row disappears the day somebody renames the row.
# ===========================================================================
CTRL="$HERE/quorum-control/single-vendor"
if [ -z "$CTRL" ] || [ "$CTRL" = "/" ]; then echo "refusing rm -rf on $CTRL" >&2; exit 1; fi
rm -rf -- "$CTRL"; mkdir -p -- "$CTRL"
cp -- "$HERE/q-23-single-vendor-quorum/receipt.intoto.jsonl" \
      "$HERE/q-23-single-vendor-quorum/receipt.intoto.jsonl.minisig" \
      "$HERE/q-23-single-vendor-quorum/findings.sarif" \
      "$HERE/q-23-single-vendor-quorum/pr-context.json" "$CTRL/"
echo "built quorum-control/single-vendor"

echo
echo "rows: $(find "$HERE" -maxdepth 1 -type d -name 'q-*' | wc -l) committed q-* directories"
