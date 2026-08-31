#!/usr/bin/env bash
# pr_review_quorum_arm.sh - evaluate S13's autonomous-merge predicate and either arm
# `gh pr merge --auto` or REFUSE with a named class.
#
# PR-REVIEW-SKILL-002 v2 S13. Every refusal names exactly one class Q1..Q10, because a
# refusal with no name is undiagnosable and an undiagnosable refusal gets routed around.
#
#   Q1   the receipt is missing, unreadable, invalid, or carries no S13 autonomy block
#   Q2   a consultation did not actually run, or ran over nothing
#   Q3   tip drift: a commit after the reviewed SHA touches something nobody reviewed
#   Q4   horizon drift: main moved and the delta region was not swept clean
#   Q5   separation: the quorum is single-vendor, incomplete, non-unanimous, or the author
#   Q6   the verdict is not PASS, or a finding blocks, or a claim carries no mark
#   Q7   the cross-vendor reviewer refused autonomy (S13.5)
#   Q8   a human surface says stop: the kill switch, the autonomy-hold label, CHANGES_REQUESTED
#   Q9   categorical: the PR does not target main, or it edits the merge mechanism itself
#   Q10  a required check is not green
#
# A REFUSAL IS NOT A BLOCK. S13 adds zero rows to S7: the blocking tier stays at six
# (B1..B6). When this script refuses, the pull request stays exactly as green as it was
# and a human merges it. That is why the classes are lettered Q and not B - so a log line
# cannot confuse the two.
#
# USAGE
#   pr_review_quorum_arm.sh --pr <N> [--receipt <dir>] [--context <file>] [--explain]
#
#   --explain      evaluate and report; NEVER call `gh pr merge`. Exit 0 permitted,
#                  1 refused. This is the shadow-mode verb of S13.11 rung 1.
#   --receipt DIR  the receipt directory. Default evidence/pr-review/<pr>/<head_sha>/.
#   --context FILE a PR-context JSON document (see below). Default: produced from `gh`.
#
# THE PR CONTEXT IS AN ARTIFACT, AND THAT IS DELIBERATE.
#
#   { "number": 2795, "base_ref": "main", "head_sha": "<sha>",
#     "labels": ["..."], "reviews": [ {"state":"CHANGES_REQUESTED","dismissed":false} ],
#     "checks": { "ci / gate": "success", "workspace-test": "success" },
#     "auto_merge_armed": false }
#
# ONE evaluator, TWO producers: `gh` in a live run, a committed file in a fixture. The
# alternative - a live path that queries GitHub and a test path that re-implements the
# predicate over stubs - is two implementations of one rule, each green against its own
# copy. That is D8, and this repository has now shipped it twice inside the artifacts
# written to prevent it. So `gh` produces the context and STOPS; nothing downstream of
# the producer knows whether it came from the network.
#
# EVALUATION ORDER: EVERY ARTIFACT-ONLY CLAUSE BEFORE EVERY REPOSITORY CLAUSE.
#
# The predicate is a conjunction, so order cannot change the verdict - only which
# refusal is REPORTED, and whether the positive control can fire. That second half is
# the reason for the rule: a control that needs the repository can misfire for an
# environmental reason and still be read as evidence that the branch works. The receipt
# guard learned this (its four controls all fire before its first `git` call) and the
# rule is written down here rather than rediscovered.
#
# ENVIRONMENT
#   PR_REVIEW_REPO        repository the SHAs resolve against (default: git toplevel)
#   PR_REVIEW_PUBKEY      minisign public key, passed through to the receipt guard
#   PR_REVIEW_SCHEMA_DIR  vendored schemas, passed through to the receipt guard
#   PR_REVIEW_GH          the gh binary (default: gh)
#   PR_REVIEW_QUORUM_CONTROL_DIR  seed directory for the positive controls
#
# There is no variable that turns a check off, and no variable that forces an arm. A
# tool this script needs and cannot find is a REFUSAL, not a skip.
#
# EXIT: 0 permitted (and armed, unless --explain); 1 refused; 2 the box cannot answer.

set -euo pipefail

PROG=${0##*/}
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# ---------------------------------------------------------------------------
# S13.8 MECHANISM_PATHS - a literal prefix list, never a regex.
#
# A pull request that edits the machinery deciding whether it may merge must not merge
# itself. This is not a hypothetical: the PR that introduces S13 touches every path
# below, so S13 refuses to auto-merge its own introduction, which is the first
# falsifiable property of the whole section (S13.8, the q-44 fixture, and the
# MECHANISM_PATHS row of tests/pr-review-quorum.bats).
#
# A LIST AND NOT A PATTERN. This repository's guard regexes have been wrong six times
# and a must-match / must-not-match table caught every one; a prefix list cannot be
# wrong in that way, because matching is `case "$p" in "$m"*)`. Adding a mechanism file
# means adding a line, which is a diff a reviewer can read.
MECHANISM_PATHS='
.claude/skills/pr-review/
.github/pr-review.pub
.github/pr-review-autonomy.disabled
.github/workflows/
contracts/pr-review-skill-v2.yaml
contracts/binding.yaml
docs/specifications/PR-REVIEW-SKILL-002-v2.md
schemas/
scripts/check_pr_review_arm4.sh
scripts/check_pr_review_counts.sh
scripts/check_pr_review_receipt.sh
scripts/check_pr_review_wiring.sh
scripts/install_pr_review_tools.sh
scripts/mutate-guard.sh
scripts/mutate_pr_review_wiring_guard.sh
scripts/mutate_quorum_arm.sh
scripts/pr_review_duplication_scan.sh
scripts/pr_review_quorum_arm.sh
tests/fixtures/pr-review/
tests/pr-review.bats
tests/pr-review-quorum.bats
'

# The commits after the reviewed SHA may touch this prefix and nothing else (S13.3.a).
# It is a CONTENT rule, not a count: S8 forbids inventing a threshold, and Arm 4's own
# argument is that the receipt-recording commit necessarily follows the commit it
# reviews and is confined to this path by construction.
EVIDENCE_PREFIX_TEMPLATE='evidence/pr-review/@@PR@@/'

REFUSAL_EXIT=1
REFUSE_CLASS=''
REFUSE_REASON=''
refuse() { REFUSE_CLASS=$1; REFUSE_REASON=$2; return 1; }

die_env() { echo "$PROG: ENV - $*" >&2; exit 2; }

sha256_stdin() { sha256sum | cut -d' ' -f1; }

# ---------------------------------------------------------------------------
# Tools. An absent tool is a REFUSAL to arm, never a skip.
# ---------------------------------------------------------------------------
MISSING_TOOLS=''
for t in git jq sha256sum; do
  command -v "$t" >/dev/null 2>&1 || MISSING_TOOLS="$MISSING_TOOLS $t"
done
if [ -n "$MISSING_TOOLS" ]; then
  echo "$PROG: ENV - cannot run:$MISSING_TOOLS not on PATH." >&2
  echo "  A gate that cannot execute its own checks must not arm a merge (S6.2)." >&2
  exit 2
fi

MODE=arm
PR_NUMBER=''
RECEIPT_DIR=''
CONTEXT_FILE=''
while [ "$#" -gt 0 ]; do
  case $1 in
    --explain)  MODE=explain; shift ;;
    --pr)       PR_NUMBER=${2:?--pr needs a number}; shift 2 ;;
    --receipt)  RECEIPT_DIR=${2:?--receipt needs a directory}; shift 2 ;;
    --context)  CONTEXT_FILE=${2:?--context needs a file}; shift 2 ;;
    -h|--help)  sed -n '2,60p' "$0"; exit 0 ;;
    *) echo "$PROG: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

REPO=${PR_REVIEW_REPO:-}
if [ -z "$REPO" ]; then
  REPO=$(git rev-parse --show-toplevel 2>/dev/null) \
    || die_env "not in a git repository and PR_REVIEW_REPO is unset"
fi
GH=${PR_REVIEW_GH:-gh}
RECEIPT_GUARD="$HERE/check_pr_review_receipt.sh"

# ===========================================================================
# PHASE A - THE ARTIFACTS. No repository access below this line, and that is
# what lets the positive control fire in any working directory.
# ===========================================================================

# phase_a <receipt-dir> <context-file>
phase_a() {
  local dir=$1 ctx=$2
  local rcpt="$dir/receipt.intoto.jsonl"
  local sarif="$dir/findings.sarif"
  local sig="$rcpt.minisig"

  # --- Q1: readability. NOT validity - the receipt guard is the authority on
  # that, and it is called in phase B. This is only what must hold before any
  # field can be read at all. A missing receipt is a REFUSAL, never a skip.
  [ -d "$dir" ]   || refuse Q1 "no such receipt directory: $dir; a missing receipt does not arm a merge" || return 1
  [ -f "$rcpt" ]  || refuse Q1 "receipt.intoto.jsonl is missing from $dir; there is nothing to evaluate, so there is nothing to arm" || return 1
  [ -f "$sarif" ] || refuse Q1 "findings.sarif is missing from $dir" || return 1
  [ -f "$sig" ]   || refuse Q1 "the receipt is unsigned - no $sig; an unsigned receipt never arms an autonomous merge (S4.3)" || return 1
  jq -e . "$rcpt"  >/dev/null 2>&1 || refuse Q1 "receipt.intoto.jsonl is not parseable JSON" || return 1
  jq -e . "$sarif" >/dev/null 2>&1 || refuse Q1 "findings.sarif is not parseable JSON" || return 1
  [ -f "$ctx" ]   || refuse Q1 "no PR context at $ctx; the predicate reads the pull request's labels, reviews and checks, and cannot assume them" || return 1
  jq -e . "$ctx"   >/dev/null 2>&1 || refuse Q1 "the PR context at $ctx is not parseable JSON" || return 1

  # --- Q1: the S13 autonomy block exists and has the shape S13.2 reads. -----
  # ABSENT IS A REFUSAL, NOT A DEFAULT. A receipt written before S13, or by a
  # reviewer that did not run the quorum, records no autonomy block; it is a
  # perfectly good receipt and it does not authorise an unattended merge. The
  # difference between "did not ask for autonomy" and "asked and qualified" must
  # be an artifact, which is S3.0 one level up.
  jq -e '.predicate.autonomy | type == "object"' "$rcpt" >/dev/null 2>&1 \
    || refuse Q1 "the receipt carries no predicate.autonomy block; a review that did not run the S13.1 quorum does not arm a merge, and its absence is not a default to yes" || return 1
  jq -e '.predicate.autonomy.requested == true' "$rcpt" >/dev/null 2>&1 \
    || refuse Q1 "predicate.autonomy.requested is not true; the reviewer did not ask for an autonomous merge" || return 1
  jq -e '.predicate.autonomy.main_sha_at_review | type == "string" and (length > 0)' "$rcpt" >/dev/null 2>&1 \
    || refuse Q1 "predicate.autonomy.main_sha_at_review is absent; without it the S13.3.b delta region cannot be named, and an unnamed region is one nobody can tell was swept" || return 1
  jq -e '.predicate.autonomy.quorum | (type == "array") and (length > 0)' "$rcpt" >/dev/null 2>&1 \
    || refuse Q1 "predicate.autonomy.quorum is absent or empty" || return 1
  jq -e '.predicate.autonomy.delta_sweep | type == "object"' "$rcpt" >/dev/null 2>&1 \
    || refuse Q1 "predicate.autonomy.delta_sweep is absent; S13.3.b requires the sweep over main_sha_at_review..origin/main to be recorded whether or not the region was empty" || return 1

  local sweep_status
  sweep_status=$(jq -r '.predicate.autonomy.delta_sweep.status // ""' "$rcpt")
  case "$sweep_status" in
    clean|dirty|not-run) ;;
    *) refuse Q1 "delta_sweep.status is '$sweep_status', outside { clean, dirty, not-run }" || return 1 ;;
  esac

  # --- Q6: the verdict, and then the facts the verdict summarises. ----------
  # Clause (5) of S13.2 makes several lines below redundant TODAY, and they are
  # written anyway: the verdict is a summary computed by the party being trusted.
  # B6 exists because `verdict: PASS` once coexisted with a stale index.
  local verdict
  verdict=$(jq -r '.predicate.verdict // ""' "$rcpt")
  [ "$verdict" = "PASS" ] \
    || refuse Q6 "the verdict is '$verdict', not PASS; DEGRADED, FINDINGS and BLOCK all mean a human reads this one (S13.3 - a quorum that proceeds when its sources were unreachable is an unreviewed merge wearing a signed receipt)" || return 1

  # THE NARROWER CLAUSE FIRST, AND THAT IS NOT A STYLE CHOICE. Every asserted-and-
  # blocking finding is ALSO a blocking finding, so behind the general clause this one
  # can never fire, and scripts/mutate_quorum_arm.sh would report a permanent survivor
  # - a rule the script states and nothing tests. Ordering the specific diagnosis ahead
  # of the general one makes both reachable and both fixturable: q-16 carries an
  # ASSERTED result classed blocking and q-17 a CITED one, both with
  # precision_class: blocking, and each must refuse on its own reason rather than on
  # the other's.
  local n_asserted_blocking
  n_asserted_blocking=$(jq -r '[ .runs[]? | .results[]?
      | select(.properties.grounding == "asserted" and .properties.precision_class == "blocking") ] | length' "$sarif")
  [ "$n_asserted_blocking" -eq 0 ] \
    || refuse Q6 "$n_asserted_blocking finding(s) are marked asserted AND classed blocking; an asserted claim never carries a blocking class (S1)" || return 1

  local n_blocking
  n_blocking=$(jq -r '[ .runs[]? | .results[]? | select(.properties.precision_class == "blocking") ] | length' "$sarif")
  [ "$n_blocking" -eq 0 ] \
    || refuse Q6 "findings.sarif carries $n_blocking finding(s) classed blocking under a PASS verdict; a surviving blocking-class finding does not arm a merge" || return 1

  local n_unmarked
  n_unmarked=$(jq -r '[ .runs[]? | .results[]?
      | select((.properties.grounding // "") as $g | (["cited","measured","asserted"] | index($g)) == null) ] | length' "$sarif")
  [ "$n_unmarked" -eq 0 ] \
    || refuse Q6 "findings.sarif carries $n_unmarked claim(s) with no grounding mark, or a mark outside { cited, measured, asserted }; S8 fixes unmarked_claims at 0 with no ratchet" || return 1

  local n_failed_runs
  n_failed_runs=$(jq -r '[ .runs[]? | .invocations[]? | select(.executionSuccessful == false) ] | length' "$sarif")
  [ "$n_failed_runs" -eq 0 ] \
    || refuse Q6 "$n_failed_runs SARIF invocation(s) record executionSuccessful: false; a tool that did not run is not a tool that found nothing (S3.0)" || return 1

  # --- Q7: the cross-vendor reviewer's veto (S13.5). ------------------------
  # S3.E's agy is ADVISORY under S7 - it may not block. S13 gives it exactly one
  # power it did not have: it may refuse AUTONOMY. That is the whole reason a
  # second vendor is in the quorum, and it is expressed as a property on an
  # advisory finding so that nothing about S7's tier changes.
  local n_veto
  n_veto=$(jq -r '[ .runs[]? | .results[]?
      | select(.properties.autonomy_effect == "refuse") ] | length' "$sarif")
  [ "$n_veto" -eq 0 ] \
    || refuse Q7 "$n_veto advisory finding(s) carry autonomy_effect: refuse; the cross-vendor reviewer may not block the PR and may refuse the unattended merge, which is the only power S13.1 gives it (S13.5)" || return 1

  # --- Q5: separation, threefold (S13.1, strengthening S5). -----------------
  local author reviewer
  author=$(jq -r '.predicate.author_actor.id // ""' "$rcpt")
  reviewer=$(jq -r '.predicate.reviewer_actor.id // ""' "$rcpt")
  [ -n "$author" ] || refuse Q5 "author_actor.id is absent; separation cannot be checked against an unnamed author" || return 1
  [ "$reviewer" != "$author" ] \
    || refuse Q5 "reviewer_actor.id = author_actor.id = '$author'; S5 already blocks this, and S13 will not arm on it either" || return 1

  local n_vendors
  n_vendors=$(jq -r '[ .predicate.autonomy.quorum[]? | .vendor // "" ] | map(select(length > 0)) | unique | length' "$rcpt")
  [ "$n_vendors" -ge 2 ] \
    || refuse Q5 "the quorum spans $n_vendors distinct vendor(s); S13.1 requires at least two, because a second reviewer buys safety only if it FAILS DIFFERENTLY - two Claudes raise the count and not the independence" || return 1

  local roles_missing
  roles_missing=$(jq -r '(["primary","cross_vendor"] - [ .predicate.autonomy.quorum[]? | .role // "" ]) | join(", ")' "$rcpt")
  [ -z "$roles_missing" ] \
    || refuse Q5 "the quorum records no member in role(s): $roles_missing; S13.1's quorum is one primary reviewer and one cross-vendor reviewer, and a missing role is a member that never voted" || return 1

  local author_in_quorum
  author_in_quorum=$(jq -r --arg a "$author" '[ .predicate.autonomy.quorum[]?
      | select((.actor.id // "") == $a) | .role // "?" ] | join(", ")' "$rcpt")
  [ -z "$author_in_quorum" ] \
    || refuse Q5 "quorum member(s) in role [$author_in_quorum] carry the author's own actor id '$author'; a quorum whose second member is the author is S5's defect with an extra row in a table" || return 1

  local dissent
  dissent=$(jq -r '[ .predicate.autonomy.quorum[]?
      | select((.verdict != "PASS") or (.refusal != null))
      | (.role // "?") + "=" + ((.verdict // "?")|tostring)
        + (if .refusal != null then " (" + ((.refusal)|tostring) + ")" else "" end) ] | join("; ")' "$rcpt")
  [ -z "$dissent" ] \
    || refuse Q5 "the quorum is not unanimous: $dissent; S13.4 counts no member's assent, only its dissent, so one refusal ends it" || return 1

  # --- Q2: every consultation actually ran (S13.2 clause 4). ---------------
  local pmat_st st k
  pmat_st=$(jq -r '.predicate.consultations.pmat.status // ""' "$rcpt")
  [ "$pmat_st" = "consulted" ] \
    || refuse Q2 "consultations.pmat.status is '$pmat_st'; S3.A makes pmat unconditional, so an autonomous merge requires it to have actually run" || return 1
  for k in pmat cuda crux mutation; do
    st=$(jq -r --arg k "$k" '.predicate.consultations[$k].status // ""' "$rcpt")
    case "$st" in
      consulted) ;;
      not-triggered)
        jq -e --arg k "$k" '(.predicate.consultations[$k].trigger_reason // "") | length > 0' "$rcpt" >/dev/null 2>&1 \
          || refuse Q2 "consultations.$k is not-triggered with an empty trigger_reason; 'the trigger did not fire' and 'I did not look' must not be the same artifact (S3.0)" || return 1
        ;;
      *) refuse Q2 "consultations.$k.status is '$st'; only { consulted, not-triggered } arm a merge, and 'unreachable' is excluded here on purpose - S13.3 refuses DEGRADED rather than weakening it" || return 1 ;;
    esac
  done

  # A consultation that asked nothing passes vacuously. S8 counts these as
  # `vacuous_consultations` and fixes the number at zero with no ratchet. The
  # receipt guard already rejects three of the four shapes; symbols_searched = 0
  # is the one it admits, because record-only is not the same as unenforced.
  local vacuous
  vacuous=$(jq -r '.predicate.consultations as $c
      | [ (if ($c.pmat.status == "consulted") and (($c.pmat.symbols_searched // 0) == 0)
             then "pmat searched 0 symbols" else empty end),
          (if ($c.cuda.status == "consulted") and (([$c.cuda.queries[]?] | length) == 0)
             then "cuda asked 0 queries" else empty end),
          (if ($c.crux.status == "consulted")
              and (([$c.crux.surfaces[]?] | length) == 0)
              and (([$c.crux.comparative_claims[]?] | length) == 0)
             then "crux looked at 0 surfaces and 0 claims" else empty end),
          (if ($c.mutation.status == "consulted") and (($c.mutation.attempted // 0) == 0)
             then "mutation attempted 0 mutants" else empty end) ] | join("; ")' "$rcpt")
  [ -z "$vacuous" ] \
    || refuse Q2 "vacuous consultation(s): $vacuous; a consultation over nothing passes vacuously, which is the shape S3.D already calls DEGRADED (S8 vacuous_consultations = 0, no ratchet)" || return 1

  local surviving
  surviving=$(jq -r 'if (.predicate.consultations.mutation.status == "consulted")
                     then ([ .predicate.consultations.mutation.survivors[]? ] | length)
                     else 0 end' "$rcpt")
  [ "$surviving" -eq 0 ] \
    || refuse Q2 "the mutation run records $surviving surviving mutant(s); a survivor is a rule the code states and nothing tests, and an unattended merge does not ship one" || return 1

  # A surface the run could not search must not sit under an unattended merge.
  # Same rule S3.0 applies to an unreachable consultation, one field down.
  local cov_none horizon_none
  cov_none=$(jq -r '[ .predicate.consultations.pmat.duplication_coverage // {} | to_entries[]
                      | select(.value == "none") | .key ] | join(", ")' "$rcpt")
  [ -z "$cov_none" ] \
    || refuse Q2 "duplication_coverage could not search [$cov_none]; an unsearched surface is DEGRADED, and S13.3 does not merge DEGRADED" || return 1
  horizon_none=$(jq -r '[ .predicate.consultations.pmat.duplication_horizon[]?
                          | select(endswith("=none")) ] | join(", ")' "$rcpt")
  [ -z "$horizon_none" ] \
    || refuse Q2 "duplication_horizon records no refspec for [$horizon_none]; a region named 'none' was not swept" || return 1

  # --- Q8: the human surfaces, artifact half. ------------------------------
  local held
  held=$(jq -r '[ .labels[]? | select(. == "autonomy-hold") ] | length' "$ctx")
  [ "$held" -eq 0 ] \
    || refuse Q8 "the pull request carries the autonomy-hold label; a human asked for this one by hand, and that is the whole point of having a label" || return 1

  local changes_requested
  changes_requested=$(jq -r '[ .reviews[]? | select(.state == "CHANGES_REQUESTED" and (.dismissed != true)) ] | length' "$ctx")
  [ "$changes_requested" -eq 0 ] \
    || refuse Q8 "$changes_requested undismissed CHANGES_REQUESTED review(s) are open on this pull request" || return 1

  # --- Q9: categorical eligibility, artifact half. -------------------------
  local base_ref
  base_ref=$(jq -r '.base_ref // ""' "$ctx")
  [ "$base_ref" = "main" ] \
    || refuse Q9 "the pull request targets '$base_ref', not main; S13 arms merges into main and nothing else, because a stacked PR gets no CI (ci.yml fires only on PRs targeting main)" || return 1

  # --- Q10: the mechanical checks the review never replaces. ---------------
  # Required checks live in TWO places here - branch protection names `ci / gate`
  # and `workspace-test`, ruleset 13878864 names a bare `gate` - so both spellings
  # of the gate are accepted and the ABSENCE of either spelling is a refusal. A
  # check the context does not mention is a check nobody saw pass.
  #
  # The wording of the refusal below is constrained by the linter, not by taste: the
  # first draft read "looked for 'ci / gate' then 'gate'", and bashrs parses the inside
  # of a double-quoted string, so it read that as a `for` loop terminated by `then` and
  # raised SC2135 - a real ERROR against a scripts/ gate that is shrink-only on the
  # error count. Two false errors here are two somebody else has to triage.
  local wt gate_conclusion
  wt=$(jq -r '.checks["workspace-test"] // "absent"' "$ctx")
  [ "$wt" = "success" ] \
    || refuse Q10 "the workspace-test check is '$wt', not success; S13 never replaces a mechanical check, it waits for one" || return 1
  gate_conclusion=$(jq -r '.checks["ci / gate"] // .checks["gate"] // "absent"' "$ctx")
  [ "$gate_conclusion" = "success" ] \
    || refuse Q10 "the gate check is '$gate_conclusion', not success. Both required spellings were read: 'ci / gate' first, a bare 'gate' second - branch protection names the one and ruleset 13878864 the other, and reading only one gives a wrong answer about what blocks a merge" || return 1

  return 0
}

# ===========================================================================
# PHASE B - THE REPOSITORY. Everything below reads git.
# ===========================================================================

# phase_b <receipt-dir> <context-file>
phase_b() {
  local dir=$1 ctx=$2
  local rcpt="$dir/receipt.intoto.jsonl"
  local out rc

  # --- Q8, repository half: THE KILL SWITCH, AND IT IS FIRST. -------------
  # READ FROM origin/main, NEVER from the PR tree: a pull request that deletes the
  # kill switch in its own branch would otherwise disable the control that exists to
  # stop it. It is the first clause of phase B and not the last, because it is the
  # operator's off switch and an off switch consulted only after five other checks
  # have passed is an off switch that costs a minute of compute to use. Nothing else
  # in phase B is cheaper or more authoritative.
  if git -C "$REPO" cat-file -e "refs/remotes/origin/main:.github/pr-review-autonomy.disabled" 2>/dev/null; then
    refuse Q8 "the autonomy kill switch .github/pr-review-autonomy.disabled is present on origin/main; it is read from origin/main and never from the PR tree, so deleting it in a branch does not turn autonomy back on" || return 1
  fi

  # --- Q1: the receipt guard is the AUTHORITY on validity. -----------------
  # Delegated, never re-implemented. Signature, schema, merge base, index
  # ancestry, the marks, B4 - all of it has exactly one implementation, and it is
  # the one 185 mutants are run against. A second copy here would be two
  # implementations of one rule, each green against its own copy, which is the
  # defect D8 names and the one this file must not add a third instance of.
  #
  # NOT `cd "$REPO"` FIRST, and that was a real defect for the length of one smoke
  # test: the guard resolves PR_REVIEW_REPO for git and everything ELSE - the vendored
  # schemas, the public key, the positive-control seeds - relative to the working
  # directory. Run from inside the fixture repository it could not find schemas/, its
  # own positive control MISFIRED, and this script reported "the receipt guard rejects
  # this receipt" about a receipt that is fine. A delegation that reports the callee's
  # broken environment as the caller's verdict is the env-vs-defect confusion three
  # guards shipped in one day here.
  set +e
  out=$(PR_REVIEW_REPO="$REPO" bash "$RECEIPT_GUARD" "$dir" 2>&1)
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    # Report the guard's REJECT lines, not its positive-control chatter: a refusal a
    # reader cannot act on is a refusal that gets routed around.
    local detail
    detail=$(awk '/^REJECT/ { p = 1 } p' <<<"$out" | tr '\n' ' ' | sed 's/  */ /g')
    [ -n "$detail" ] || detail=$(tr '\n' ' ' <<<"$out" | sed 's/  */ /g')
    refuse Q1 "the receipt guard rejects this receipt, so there is nothing valid to arm on: $(printf '%s' "$detail" | cut -c1-320)" || return 1
  fi

  local head base pr_head pr_number
  head=$(jq -r '.predicate.head_sha // ""' "$rcpt")
  base=$(jq -r '.predicate.base_sha // ""' "$rcpt")
  pr_head=$(jq -r '.head_sha // ""' "$ctx")
  pr_number=$(jq -r '.number // ""' "$ctx")
  [ -n "$pr_head" ] || refuse Q1 "the PR context records no head_sha; the review must be shown to be OF the code about to merge" || return 1
  [ -n "$pr_number" ] || refuse Q1 "the PR context records no number; the S13.3.a evidence path is derived from it" || return 1
  git -C "$REPO" cat-file -e "${pr_head}^{commit}" 2>/dev/null \
    || refuse Q1 "the PR context's head_sha $pr_head does not resolve to a commit in $REPO" || return 1

  # --- Q9: the PR does not edit the machinery that decides its own merge. --
  local changed f m hit=''
  changed=$(git -C "$REPO" diff --name-only "$base" "$head" 2>/dev/null || true)
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    while IFS= read -r m; do
      [ -n "$m" ] || continue
      case "$f" in "$m"*) hit="$f (matches $m)"; break 2 ;; esac
    done <<<"$MECHANISM_PATHS"
  done <<<"$changed"
  [ -z "$hit" ] \
    || refuse Q9 "the diff edits the merge mechanism itself: $hit; a pull request that changes the machinery deciding whether it may merge does not get to apply the new machinery to itself (S13.8)" || return 1

  # --- Q2, repository half: a guard-touching diff owes a GUARD-scoped run. -
  local touches_guard='' mscope mattempted mkilled
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    case "$f" in
      scripts/check_*.sh|*/scripts/check_*.sh|scripts/mutate*.sh|*/scripts/mutate*.sh)
        touches_guard=$f; break ;;
    esac
  done <<<"$changed"
  if [ -n "$touches_guard" ]; then
    mscope=$(jq -r '.predicate.consultations.mutation.scope // ""' "$rcpt")
    mattempted=$(jq -r '.predicate.consultations.mutation.attempted // 0' "$rcpt")
    mkilled=$(jq -r '.predicate.consultations.mutation.killed // -1' "$rcpt")
    { [ "$mscope" = "guard" ] && [ "$mattempted" -gt 0 ] && [ "$mkilled" -eq "$mattempted" ]; } \
      || refuse Q2 "the diff touches a guard ($touches_guard) but the mutation run records scope='$mscope' attempted=$mattempted killed=$mkilled; S7 class B3 requires 100% on a guard-touching PR and S8 fixes guard_mutation_score at one with no ratchet" || return 1
  fi

  # --- Q3: tip drift, by CONTENT and not by count (S13.3.a). --------------
  # check_pr_review_arm4.sh selects a receipt whose head_sha is an ANCESTOR of the
  # tip and prints commits_after_reviewed WITHOUT gating - correct for Arm 4, and
  # correct for the reason it gives: S8 sets thresholds from thirty samples and
  # never invents them. But the ancestor rule is exactly the relaxation that lets
  # an UNREVIEWED commit ride in, and under autonomy nobody reads the diff to
  # notice. So the commits between the reviewed SHA and the tip may touch the
  # evidence path and nothing else. That is not a number, so S8 is untouched.
  if [ "$head" != "$pr_head" ]; then
    git -C "$REPO" merge-base --is-ancestor "$head" "$pr_head" 2>/dev/null \
      || refuse Q3 "the reviewed head_sha $head is not an ancestor of the pull request tip $pr_head; the review is not OF the code about to merge" || return 1
    local prefix late drift=''
    prefix=${EVIDENCE_PREFIX_TEMPLATE//@@PR@@/$pr_number}
    late=$(git -C "$REPO" diff --name-only "$head" "$pr_head" 2>/dev/null || true)
    while IFS= read -r f; do
      [ -n "$f" ] || continue
      case "$f" in "$prefix"*) continue ;; esac
      drift=$f; break
    done <<<"$late"
    [ -z "$drift" ] \
      || refuse Q3 "the commits after the reviewed SHA touch $drift, which is outside $prefix**; that code was never reviewed, and under autonomy nobody reads the diff to notice (S13.3.a)" || return 1
  fi

  # --- Q4: horizon drift (S13.3.b). ---------------------------------------
  # S3.A's duplication horizon was swept AT REVIEW TIME. The queue merges at about
  # one PR an hour with max_entries_to_build: 1, so hours pass and main moves; the
  # region main_sha_at_review..origin/main is unswept and, before S13, unnamed.
  # That is D7's defect recurring in the TIME dimension instead of the graph one.
  local main_at_review region
  main_at_review=$(jq -r '.predicate.autonomy.main_sha_at_review' "$rcpt")
  git -C "$REPO" cat-file -e "${main_at_review}^{commit}" 2>/dev/null \
    || refuse Q4 "autonomy.main_sha_at_review $main_at_review does not resolve to a commit in $REPO; an unresolvable anchor makes the delta region unfalsifiable" || return 1
  # NOT A REFUSAL CLASS, and the reason is a measurement rather than a preference.
  # An unenumerable region is the BOX failing to answer, and this script's exit 2
  # exists so a broken environment is never read as a receipt defect. Written as a Q4
  # refusal first, it was UNREACHABLE: by the time control arrives here the receipt
  # guard has already resolved refs/remotes/origin/main (it computes a merge-base
  # against it), so no receipt can reach a branch that fires only when that ref is
  # missing. A permanently unkillable mutant would put a hole in a score S13.10 fixes
  # at one, so the branch stays what it always was - an environment exit.
  set +e
  region=$(git -C "$REPO" rev-list "$main_at_review..refs/remotes/origin/main" 2>/dev/null)
  region_rc=$?
  set -e
  [ "$region_rc" -eq 0 ] \
    || die_env "cannot enumerate $main_at_review..origin/main in $REPO"
  if [ -n "$region" ]; then
    local sweep_status needles n_needles recorded_digest computed_digest
    sweep_status=$(jq -r '.predicate.autonomy.delta_sweep.status' "$rcpt")
    [ "$sweep_status" = "clean" ] \
      || refuse Q4 "main advanced by $(printf '%s\n' "$region" | grep -c .) commit(s) since main_sha_at_review $main_at_review and delta_sweep.status is '$sweep_status'; the region between them was never searched for prior art, which is exactly F7's blind region moved into the time dimension (S13.3.b)" || return 1

    # THE NEEDLES ARE REPLAYED, NOT RE-DERIVED. A delta sweep that re-derives its
    # needles from the diff is a SECOND implementation of S3.A's derivation, each
    # green against its own copy - D8 exactly, in the code written to enforce D8.
    # So the receipt records the needle set and the sweep records a digest over it.
    n_needles=$(jq -r '[ .predicate.consultations.pmat.duplication_needles[]? ] | length' "$rcpt")
    [ "$n_needles" -gt 0 ] \
      || refuse Q4 "the delta sweep reports clean but pmat.duplication_needles is absent or empty; a sweep whose needle set is unrecorded cannot be shown to have replayed S3.A's, and one that re-derives them is a second implementation of the same rule (S13.3.b)" || return 1
    recorded_digest=$(jq -r '.predicate.autonomy.delta_sweep.needles_sha256 // ""' "$rcpt")
    computed_digest=$(jq -r '[ .predicate.consultations.pmat.duplication_needles[]? ] | join("\n")' "$rcpt" | sha256_stdin)
    [ "$recorded_digest" = "$computed_digest" ] \
      || refuse Q4 "delta_sweep.needles_sha256 is $recorded_digest but sha256 over pmat.duplication_needles is $computed_digest; the sweep did not replay the needle set the review searched with" || return 1
  fi

  return 0
}

# evaluate <receipt-dir> <context-file>  -> 0 permitted, 1 refused (class set)
evaluate() {
  REFUSE_CLASS=''; REFUSE_REASON=''
  phase_a "$1" "$2" || return 1
  phase_b "$1" "$2" || return 1
  return 0
}

# ===========================================================================
# S13.10 POSITIVE CONTROL, FIRST.
#
# Before evaluating anything real, the predicate is run against deliberately
# non-permitted inputs and a REFUSAL is required from each, under the class the
# control names AND on the branch it names. If a non-permitted input is permitted,
# this script's PERMIT is a count of files rather than a verdict, and the run stops
# before it can arm anything.
#
# Both controls fire in PHASE A, which is why phase A touches no repository: a
# control that needs git can misfire for an environmental reason and still be read
# as evidence. Two DEPTHS, because a control that fires at the readability branch
# stays green even if every semantic clause below it is deleted.
# ===========================================================================
QC_SEED_DIR=${PR_REVIEW_QUORUM_CONTROL_DIR:-$HERE/../tests/fixtures/pr-review/quorum-control}

run_quorum_control() {
  local name=$1 want=$2 want_reason=$3 dir=$4 ctx=$5
  REFUSE_CLASS=''; REFUSE_REASON=''
  if phase_a "$dir" "$ctx"; then
    cat >&2 <<EOF
$PROG: POSITIVE CONTROL FAILED ($name)
  A deliberately non-permitted input was PERMITTED. This run's verdicts cannot be
  trusted: a PERMIT would be a count of files, not a verdict (S6.1, S13.10).
  Refusing to evaluate anything.
EOF
    return 1
  fi
  if [ "$REFUSE_CLASS" != "$want" ]; then
    cat >&2 <<EOF
$PROG: POSITIVE CONTROL MISFIRED ($name)
  Expected a refusal under $want; got $REFUSE_CLASS:
    $REFUSE_REASON
  A control that fires for a reason other than the one it names is mislabeled
  evidence, not a control. Refusing to evaluate anything.
EOF
    return 1
  fi
  case "$REFUSE_REASON" in
    *"$want_reason"*) ;;
    *)
      cat >&2 <<EOF
$PROG: POSITIVE CONTROL MISFIRED ($name)
  Refused under the expected class $want, but on the wrong branch. Expected a reason
  containing:
    $want_reason
  got:
    $REFUSE_REASON
  Refusing to evaluate anything.
EOF
      return 1 ;;
  esac
  printf 'quorum-control  %-16s refused (%s: %s)\n' "$name" "$REFUSE_CLASS" \
    "$(printf '%s' "$REFUSE_REASON" | cut -c1-56)"
  return 0
}

# Control 1: readability depth. Synthesized inline, so a deleted fixture tree cannot
# silently take the control with it.
QC_TMP=$(mktemp -d "${TMPDIR:-/tmp}/pr-review-quorum-control.XXXXXX")
case "$QC_TMP" in
  */pr-review-quorum-control.*) ;;
  *) echo "$PROG: ENV - refusing to use scratch dir $QC_TMP" >&2; exit 2 ;;
esac
qc_cleanup() {
  cleanup_dir=${QC_TMP:-}
  case "$cleanup_dir" in
    */pr-review-quorum-control.*) ;;
    *) return 0 ;;
  esac
  if [ -z "$cleanup_dir" ] || [ "$cleanup_dir" = "/" ]; then
    return 0
  fi
  rm -rf -- "$cleanup_dir"
}
trap qc_cleanup EXIT

QC1="$QC_TMP/no-autonomy-block"; mkdir -p "$QC1"
printf '{"_type":"https://in-toto.io/Statement/v1","subject":[],"predicateType":"https://paiml.dev/attestations/pr-review/v2","predicate":{"verdict":"PASS"}}\n' \
  > "$QC1/receipt.intoto.jsonl"
printf '{"version":"2.1.0","runs":[]}\n' > "$QC1/findings.sarif"
: > "$QC1/receipt.intoto.jsonl.minisig"
printf '{"number":1,"base_ref":"main","head_sha":"0","labels":[],"reviews":[],"checks":{}}\n' \
  > "$QC_TMP/qc1-context.json"
# Control 2: semantic depth. A committed, signature-bearing, schema-valid receipt whose
# only defect is that its quorum is single-vendor. It can only fire by reaching the
# vendor-distinctness clause, which is the one clause S13.1 rests on.
seeded_quorum_control() {
  local name=$1 seed="$QC_SEED_DIR/$2" class=$3 reason=$4 d="$QC_TMP/$1"
  if [ ! -f "$seed/receipt.intoto.jsonl" ]; then
    echo "$PROG: ENV - the quorum positive-control fixture is missing at $seed" >&2
    echo "  Without it, deleting the clause it pins would leave this script green." >&2
    exit 2
  fi
  mkdir -p "$d"
  cp -- "$seed/receipt.intoto.jsonl" "$seed/findings.sarif" \
        "$seed/receipt.intoto.jsonl.minisig" "$d/" \
    || { echo "$PROG: ENV - the control fixture at $seed is incomplete" >&2; exit 2; }
  run_quorum_control "$name" "$class" "$reason" "$d" "$seed/pr-context.json"
}

# ONE ARMING LINE FOR BOTH CONTROLS, and that is a mutation-testing result rather than
# a preference. Written as two `|| exit 1` lines, the FIRST of them was an equivalent
# mutant: control 1 is synthesized inline, so its only failure mode is a phase-A defect
# - and every phase-A defect is already its own mutant in the set. No single-mutation
# test could make control 1 fail, so `|| true` on that line changed nothing and
# scripts/mutate_quorum_arm.sh reported a SURVIVOR (measured, run 1 of 2026-08-31). A
# permanently unkillable mutant puts a hole in a score S13.10 fixes at one, so the two
# controls share the arming line the seed tests can actually reach.
run_all_quorum_controls() {
  run_quorum_control no-autonomy-block Q1 "carries no predicate.autonomy block" \
    "$QC1" "$QC_TMP/qc1-context.json" || return 1
  seeded_quorum_control single-vendor single-vendor Q5 "distinct vendor" || return 1
  return 0
}
run_all_quorum_controls || exit 1

# ===========================================================================
# The real evaluation.
# ===========================================================================
[ -n "$PR_NUMBER" ] || { echo "$PROG: --pr <N> is required" >&2; exit 2; }
case "$PR_NUMBER" in ''|*[!0-9]*) echo "$PROG: --pr must be a number, got '$PR_NUMBER'" >&2; exit 2 ;; esac

# --- the context producer. `gh` runs HERE and nowhere else. ----------------
if [ -z "$CONTEXT_FILE" ]; then
  command -v "$GH" >/dev/null 2>&1 \
    || die_env "$GH is not on PATH and no --context was given; the predicate reads the pull request's labels, reviews and checks and will not assume them"
  CONTEXT_FILE="$QC_TMP/context.json"
  set +e
  gh_raw=$("$GH" pr view "$PR_NUMBER" --json number,baseRefName,headRefOid,labels,reviews,autoMergeRequest,statusCheckRollup 2>&1)
  gh_rc=$?
  set -e
  [ "$gh_rc" -eq 0 ] \
    || die_env "gh pr view $PR_NUMBER failed (exit $gh_rc): $(printf '%s' "$gh_raw" | tr '\n' ' ' | cut -c1-200)"
  printf '%s' "$gh_raw" | jq '{
      number: .number,
      base_ref: .baseRefName,
      head_sha: .headRefOid,
      labels: [ .labels[]?.name ],
      reviews: [ .reviews[]? | { state: .state, dismissed: (.state == "DISMISSED") } ],
      checks: ( [ .statusCheckRollup[]?
                  | select(.name != null)
                  | { key: .name, value: ((.conclusion // "") | ascii_downcase) } ] | from_entries ),
      auto_merge_armed: (.autoMergeRequest != null)
    }' > "$CONTEXT_FILE" \
    || die_env "could not transform the gh response into a PR context"
fi

if [ -z "$RECEIPT_DIR" ]; then
  ctx_head=$(jq -r '.head_sha // ""' "$CONTEXT_FILE" 2>/dev/null || printf '')
  [ -n "$ctx_head" ] || die_env "cannot derive a receipt directory: the context records no head_sha"
  RECEIPT_DIR="$REPO/evidence/pr-review/$PR_NUMBER/$ctx_head"
  # The receipt of the tip cannot live in a directory named after the tip - committing
  # it changes the tip. Arm 4 shipped exactly that gate-that-cannot-pass. So the tip
  # directory is tried first and the newest ANCESTOR-named directory is the fallback,
  # and Q3 is what makes the relaxation safe.
  if [ ! -d "$RECEIPT_DIR" ]; then
    for cand in "$REPO/evidence/pr-review/$PR_NUMBER"/*/; do
      [ -d "$cand" ] || continue
      cand_sha=$(basename "$cand")
      if git -C "$REPO" merge-base --is-ancestor "$cand_sha" "$ctx_head" 2>/dev/null; then
        RECEIPT_DIR=${cand%/}
      fi
    done
  fi
fi

if evaluate "$RECEIPT_DIR" "$CONTEXT_FILE"; then
  printf 'PERMIT  pr=%s  receipt=%s\n' "$PR_NUMBER" "$RECEIPT_DIR"
  printf '  every clause of the S13.2 predicate holds; the quorum is unanimous and cross-vendor.\n'
  if [ "$MODE" = explain ]; then
    printf '  --explain: NOT arming. S13.11 rung 1 is shadow mode.\n'
    exit 0
  fi
  # IDEMPOTENT. `gh pr merge --auto` is safe to repeat, but a second call is a second
  # audit-log entry and a second chance to fail for an unrelated reason, so the armed
  # state is read first and a no-op says so.
  already=$(jq -r '.auto_merge_armed // false' "$CONTEXT_FILE")
  if [ "$already" = true ]; then
    printf 'ALREADY-ARMED  pr=%s  auto-merge is already requested; nothing to do.\n' "$PR_NUMBER"
    exit 0
  fi
  command -v "$GH" >/dev/null 2>&1 || die_env "$GH is not on PATH; cannot arm"
  set +e
  arm_out=$("$GH" pr merge "$PR_NUMBER" --squash --auto 2>&1)
  arm_rc=$?
  set -e
  if [ "$arm_rc" -ne 0 ]; then
    echo "$PROG: ENV - gh pr merge --auto failed (exit $arm_rc): $arm_out" >&2
    exit 2
  fi
  printf 'ARMED  pr=%s  %s\n' "$PR_NUMBER" "$(printf '%s' "$arm_out" | tr '\n' ' ')"
  exit 0
fi

printf 'REFUSE  pr=%s  [%s]\n    %s\n' "$PR_NUMBER" "$REFUSE_CLASS" "$REFUSE_REASON"
printf '  This is a REFUSAL TO ARM, not a block. The pull request is exactly as green as\n'
printf '  it was; S13 adds zero rows to S7. A human merges it.\n'
# Named rather than literal so scripts/mutate_quorum_arm.sh can target THIS exit and no
# other: the file carries several `|| exit 1` lines and a substring mutation on a bare
# `exit 1` would match the first of them instead. A mutation entry that matches the
# wrong line reports a kill it never earned.
exit "$REFUSAL_EXIT"
