#!/usr/bin/env bash
# check_pr_review_receipt.sh - validate a grounded PR-review receipt.
#
# PR-REVIEW-SKILL-002 v2 S6. Every rejection names exactly one blocking class from
# contracts/pr-review-skill-v2.yaml (B1..B6), because a rejection with no named class
# is how a guard grows a rule nothing governs.
#
#   B1  receipt missing, schema-invalid, unsigned, or internally inconsistent
#   B2  reviewer_actor.id = author_actor.id
#   B3  guard mutation score < 1.00        (PRREV-004; not evaluated here)
#   B4  unverified_comparative_claim
#   B5  breaking API surface with no semver bump   (no consultation emits this)
#   B6  index_is_ancestor = false AND verdict = PASS
#
# USAGE
#   check_pr_review_receipt.sh <receipt-dir> [<receipt-dir> ...]
#   check_pr_review_receipt.sh --match-path <path>          predicate: S3.B path trigger
#   check_pr_review_receipt.sh --match-message <text>       predicate: S3.B message trigger
#   check_pr_review_receipt.sh --match-comparative <text>   predicate: S3.C.1 comparative claim
#
# The three --match-* forms are pure predicates over one string. They exist so the
# regexes can be driven by a must-match / must-not-match case table
# (tests/fixtures/pr-review/*-cases.tsv) rather than by reading them. This repository's
# guard patterns have been wrong six times; a table caught every one and review caught
# none. They cannot bypass any validation - they evaluate a regex and exit.
#
# ENVIRONMENT
#   PR_REVIEW_REPO     repository the receipt's SHAs are resolved against
#                      (default: the git toplevel of the working directory)
#   PR_REVIEW_PUBKEY   minisign public key (default: .github/pr-review.pub)
#   PR_REVIEW_SCHEMA_DIR  vendored schemas (default: schemas/)
#
# There is no variable that turns a check off. A tool this guard needs and cannot find
# is a REJECTION, not a skip: a gate that cannot run must not read green.
#
# EXIT: 0 every receipt accepted; 1 anything else, including a failed positive control.

set -euo pipefail

PROG=${0##*/}

# ---------------------------------------------------------------------------
# S3.B triggers and S3.C.1 comparative-claim detection.
#
# Each pattern is transcribed from the spec, NOT broadened. Where the spec's literal
# pattern misses something a human would call a CUDA change, that is recorded as a gap
# in tests/fixtures/pr-review/README.md and pinned by a must-not-match case, rather
# than silently widened here. A guard that quietly does more than its spec is as
# unreviewable as one that quietly does less.
# ---------------------------------------------------------------------------

# Paths. S3.B: crates/aprender-gpu/**, crates/aprender-serve/src/cuda/**,
# *cuda*, *ptx*, *cublas*, *fp8*, *nvrtc*. Matched case-INSENSITIVELY: the
# spec writes these as concept globs over a path, and CUDA_NOTES.md is a CUDA file.
CUDA_PATH_RE='(^crates/aprender-gpu/)|(^crates/aprender-serve/src/cuda/)|(cuda)|(ptx)|(cublas)|(fp8)|(nvrtc)'

# Messages. S3.B: sm_\d+, cu[A-Z]\w+, cuda[A-Z]\w+, or a GPU architecture name.
# Matched case-SENSITIVELY - the character classes are explicitly uppercase, so
# case-folding here would silently rewrite the spec's pattern into a different one.
# The architecture list omits Turing and Pascal on purpose: "Turing complete" and
# "PascalCase" are commoner in this repository's commit messages than the 2016/2018
# parts, and a class must hold >=90% precision to stay in the blocking tier (S7
# admission rule). Both omissions are pinned as NO-MATCH rows in
# tests/fixtures/pr-review/cuda-message-cases.tsv rather than left to be rediscovered.
CUDA_MSG_RE='(sm_[0-9]+)|(cu[A-Z][A-Za-z0-9_]+)|(cuda[A-Z][A-Za-z0-9_]+)|(Blackwell)|(Hopper)|(Ada Lovelace)|(Ampere)|(Volta)'

# S3.C.1 "N x competitor". The competitor must be NAMED: a bare "2x speedup" is a
# self-relative claim with no comparator to record, and demanding an artifact hash for
# it would be exactly the over-reach the discrimination rows exist to catch.
COMPARATIVE_RE='[0-9]+(\.[0-9]+)?[[:space:]]*(x|×)[[:space:]]+((faster[[:space:]]+than|vs\.?|over|of)[[:space:]]+)?(ollama|llama\.cpp|llama-cpp|llamacpp|vllm|pytorch|torch|sklearn|scikit-learn|unsloth|tensorrt|onnxruntime|onnx|transformers|candle|burn|ggml|tinygrad|mlx)'

match_cuda_path()    { grep -Eqi -- "$CUDA_PATH_RE"   <<<"$1"; }
match_cuda_message() { grep -Eq  -- "$CUDA_MSG_RE"    <<<"$1"; }
match_comparative()  { grep -Eqi -- "$COMPARATIVE_RE" <<<"$1"; }

case "${1-}" in
  --match-path)        match_cuda_path    "${2?--match-path needs an argument}";        exit $? ;;
  --match-message)     match_cuda_message "${2?--match-message needs an argument}";     exit $? ;;
  --match-comparative) match_comparative  "${2?--match-comparative needs an argument}"; exit $? ;;
  -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
esac

# ---------------------------------------------------------------------------
# Tools. Absent tool => rejection, never a skip.
# ---------------------------------------------------------------------------
MISSING_TOOLS=""
for t in git jq sha256sum check-jsonschema minisign; do
  command -v "$t" >/dev/null 2>&1 || MISSING_TOOLS="$MISSING_TOOLS $t"
done
if [ -n "$MISSING_TOOLS" ]; then
  echo "$PROG: FAIL - cannot run:$MISSING_TOOLS not on PATH." >&2
  echo "  A gate that cannot execute its own checks must not report green (S6.2)." >&2
  exit 1
fi

SCHEMA_DIR=${PR_REVIEW_SCHEMA_DIR:-schemas}
PUBKEY=${PR_REVIEW_PUBKEY:-.github/pr-review.pub}
REPO=${PR_REVIEW_REPO:-}
if [ -z "$REPO" ]; then
  REPO=$(git rev-parse --show-toplevel 2>/dev/null) || {
    echo "$PROG: FAIL - not in a git repository and PR_REVIEW_REPO is unset." >&2
    exit 1
  }
fi

VERDICTS='PASS FINDINGS DEGRADED BLOCK'
PREDICATE_TYPE='https://paiml.dev/attestations/pr-review/v2'

REJECT_CLASS=''
REJECT_REASON=''
reject() { REJECT_CLASS=$1; REJECT_REASON=$2; return 1; }

# sha256 of stdin, bare hex.
sha256_stdin() { sha256sum | cut -d' ' -f1; }

# -------------------------------------------------------------------------
# validate_receipt <dir>
# 0 = accept. 1 = reject, with REJECT_CLASS / REJECT_REASON set.
# Ordering is load-bearing and follows the contract's preconditions: the receipt
# parses against the vendored schema before any mark is read, and the signature
# verifies before any blocking class is evaluated.
# -------------------------------------------------------------------------
validate_receipt() {
  local dir=$1
  local rcpt="$dir/receipt.intoto.jsonl"
  local sarif="$dir/findings.sarif"
  local sig="$rcpt.minisig"

  # --- B1: presence. A missing receipt is RED, not skipped (S6.3). -----------
  [ -d "$dir" ]   || reject B1 "no such receipt directory: $dir" || return 1
  [ -f "$rcpt" ]  || reject B1 "receipt.intoto.jsonl is missing - a missing receipt is RED, not skipped, per S6.3" || return 1
  [ -f "$sarif" ] || reject B1 "findings.sarif is missing" || return 1

  # --- B1: it is JSON Lines: exactly one JSON document, on one line. ---------
  # RECORDS, not newlines. `wc -l` counts line TERMINATORS, so a perfectly valid
  # single-record file with no trailing newline reports 0 and an earlier draft of this
  # block rejected it. bashrs surfaced the dead `lines=$((lines))` that hid the bug.
  local newlines lastbyte records
  newlines=$(wc -l < "$rcpt")
  lastbyte=$(tail -c1 "$rcpt")
  records=$newlines
  if [ -n "$lastbyte" ]; then
    records=$((newlines + 1))
  fi
  if [ "$records" -ne 1 ]; then
    reject B1 "receipt.intoto.jsonl holds $records JSON record/s; the in-toto Statement must be exactly one" || return 1
  fi
  jq -e . "$rcpt"  >/dev/null 2>&1 || reject B1 "receipt.intoto.jsonl is not parseable JSON" || return 1
  jq -e . "$sarif" >/dev/null 2>&1 || reject B1 "findings.sarif is not parseable JSON" || return 1

  # --- B1: schema gate, offline, against the vendored copies (S6.2). ---------
  local sout
  if ! sout=$(check-jsonschema --schemafile "$SCHEMA_DIR/in-toto-statement-v1.json" "$rcpt" 2>&1); then
    reject B1 "receipt fails schemas/in-toto-statement-v1.json: $(printf '%s' "$sout" | tr '\n' ' ' | cut -c1-220)" || return 1
  fi
  if ! sout=$(check-jsonschema --schemafile "$SCHEMA_DIR/sarif-2.1.0.json" "$sarif" 2>&1); then
    reject B1 "findings.sarif fails schemas/sarif-2.1.0.json: $(printf '%s' "$sout" | tr '\n' ' ' | cut -c1-220)" || return 1
  fi

  # --- B1: signature, before any blocking class is evaluated (S4.3). --------
  # The signature proves the receipt came from the signing environment. It does NOT
  # prove the review was honest, and the PR comment says so (S4.3, R1: L1-self).
  [ -f "$PUBKEY" ] || reject B1 "public key $PUBKEY is absent; an unverifiable signature is not a verified one" || return 1
  [ -f "$sig" ]    || reject B1 "receipt is unsigned - no $sig" || return 1
  if ! sout=$(minisign -V -m "$rcpt" -p "$PUBKEY" 2>&1); then
    reject B1 "signature does not verify against $PUBKEY: $(printf '%s' "$sout" | tr '\n' ' ' | cut -c1-160)" || return 1
  fi

  # --- read the predicate ---------------------------------------------------
  local ptype alevel head base verdict author reviewer subj_sha
  ptype=$(jq -r '.predicateType // ""' "$rcpt")
  [ "$ptype" = "$PREDICATE_TYPE" ] || reject B1 "predicateType is '$ptype', expected '$PREDICATE_TYPE'" || return 1

  alevel=$(jq -r '.predicate.attestation_level // ""' "$rcpt")
  [ "$alevel" = "L1-self" ] || reject B1 "attestation_level is '$alevel'; a skill invoked by the authoring agent is self-attestation, and R1 requires it to say so" || return 1

  head=$(jq -r '.predicate.head_sha // ""' "$rcpt")
  base=$(jq -r '.predicate.base_sha // ""' "$rcpt")
  verdict=$(jq -r '.predicate.verdict // ""' "$rcpt")
  author=$(jq -r '.predicate.author_actor.id // ""' "$rcpt")
  reviewer=$(jq -r '.predicate.reviewer_actor.id // ""' "$rcpt")
  subj_sha=$(jq -r '.subject[0].digest.sha1 // ""' "$rcpt")

  [ -n "$head" ] || reject B1 "predicate.head_sha is absent" || return 1
  [ -n "$base" ] || reject B1 "predicate.base_sha is absent" || return 1
  [ "$subj_sha" = "$head" ] || reject B1 "the sha1 digest of subject 0 is $subj_sha but the predicate reviews head_sha $head" || return 1
  grep -qx -- "$verdict" <<<"$(tr ' ' '\n' <<<"$VERDICTS")" \
    || reject B1 "verdict '$verdict' is outside { PASS FINDINGS DEGRADED BLOCK }" || return 1

  # --- B1: the findings the receipt points at are the findings on disk. -----
  local fref_path fref_sha actual_sha
  fref_path=$(jq -r '.predicate.findings_ref.path // ""' "$rcpt")
  fref_sha=$(jq -r '.predicate.findings_ref.sha256 // ""' "$rcpt")
  [ "$fref_path" = "findings.sarif" ] || reject B1 "findings_ref.path is '$fref_path', expected 'findings.sarif'" || return 1
  actual_sha=$(sha256_stdin < "$sarif")
  [ "$fref_sha" = "$actual_sha" ] \
    || reject B1 "findings_ref.sha256 ($fref_sha) is not sha256(findings.sarif) ($actual_sha)" || return 1

  # --- B1: the cost block. Record-only is not unenforced (contract S8). ------
  jq -e '.predicate.cost | (.input_tokens|type=="number") and (.output_tokens|type=="number") and (.wall_seconds|type=="number")' \
      "$rcpt" >/dev/null 2>&1 \
    || reject B1 "predicate.cost must carry numeric input_tokens, output_tokens and wall_seconds; record-only is not unenforced" || return 1

  # --- B2: author / reviewer separation (S5). -------------------------------
  [ -n "$author" ]   || reject B1 "author_actor.id is absent"   || return 1
  [ -n "$reviewer" ] || reject B1 "reviewer_actor.id is absent" || return 1
  [ "$reviewer" != "$author" ] \
    || reject B2 "reviewer_actor.id = author_actor.id = '$author'; a self-review is not a review (S5)" || return 1

  # --- B1: the diff boundary really is the merge base (S2, row 10). ---------
  # This is the fix for diff-scope pollution by other agents' merges in the
  # parallel-worktree workflow: a floating base pulls their commits into the review.
  git -C "$REPO" cat-file -e "${head}^{commit}" 2>/dev/null \
    || reject B1 "head_sha $head does not resolve to a commit in $REPO" || return 1
  local computed_base
  computed_base=$(git -C "$REPO" merge-base refs/remotes/origin/main "$head" 2>/dev/null) \
    || reject B1 "cannot compute merge-base(origin/main, $head) in $REPO" || return 1
  [ "$base" = "$computed_base" ] \
    || reject B1 "base_sha $base is not git merge-base origin/main $head (= $computed_base); the diff scope of this review is not the merge base (S2)" || return 1

  # --- consultation statuses -----------------------------------------------
  local pmat_st cuda_st crux_st mut_st
  pmat_st=$(jq -r '.predicate.consultations.pmat.status // ""' "$rcpt")
  cuda_st=$(jq -r '.predicate.consultations.cuda.status // ""' "$rcpt")
  crux_st=$(jq -r '.predicate.consultations.crux.status // ""' "$rcpt")
  mut_st=$(jq  -r '.predicate.consultations.mutation.status // ""' "$rcpt")
  local k st
  for k in pmat cuda crux mutation; do
    st=$(jq -r --arg k "$k" '.predicate.consultations[$k].status // ""' "$rcpt")
    case "$st" in
      consulted|not-triggered|unreachable) ;;
      "") reject B1 "consultations.$k.status is absent; an omitted consultation is indistinguishable from one that found nothing (S3.0)" || return 1 ;;
      *)  reject B1 "consultations.$k.status is '$st', outside { consulted, not-triggered, unreachable }" || return 1 ;;
    esac
  done

  # --- B1: an unreachable source must not read clean (S3.0, rows 5 and 6). --
  for k in pmat cuda crux mutation; do
    st=$(jq -r --arg k "$k" '.predicate.consultations[$k].status // ""' "$rcpt")
    if [ "$st" = "unreachable" ] && [ "$verdict" = "PASS" ]; then
      reject B1 "consultations.$k is unreachable but the verdict is PASS; an unreachable source must be DEGRADED, not clean (S3.0)" || return 1
    fi
  done

  # --- B1: a consulted-but-vacuous mutation run (row 2). -------------------
  # A mutation set that matches nothing passes vacuously - the same shape as
  # `pv lint <FILE>` returning PASS over zero contracts.
  if [ "$mut_st" = "consulted" ]; then
    local attempted killed
    attempted=$(jq -r '.predicate.consultations.mutation.attempted // "null"' "$rcpt")
    killed=$(jq -r    '.predicate.consultations.mutation.killed    // "null"' "$rcpt")
    case "$attempted" in
      ''|null|*[!0-9]*) reject B1 "mutation.status is consulted but attempted is '$attempted'" || return 1 ;;
    esac
    [ "$attempted" -gt 0 ] \
      || reject B1 "mutation.status is consulted with attempted=0; a run that attempts nothing is DEGRADED, not clean (S3.D)" || return 1
    case "$killed" in ''|null|*[!0-9]*) reject B1 "mutation.killed is '$killed', not a count" || return 1 ;; esac
  fi

  # --- B1: the CUDA consultation was skipped while its trigger fired (row 1).
  if [ "$cuda_st" = "not-triggered" ]; then
    local f fired=''
    while IFS= read -r f; do
      [ -n "$f" ] || continue
      if match_cuda_path "$f"; then fired="path $f"; break; fi
    done < <(git -C "$REPO" diff --name-only "$base" "$head" 2>/dev/null || true)
    if [ -z "$fired" ]; then
      local msg
      msg=$(git -C "$REPO" log --format=%B "$base..$head" 2>/dev/null || true)
      if [ -n "$msg" ] && match_cuda_message "$msg"; then fired="commit message"; fi
    fi
    [ -z "$fired" ] \
      || reject B1 "consultations.cuda is not-triggered but its S3.B trigger fires on this diff ($fired); 'the docs said nothing' and 'I did not ask' must not be the same artifact" || return 1
  fi

  # --- B6: a stale index must not read PASS (S3.A, row 9). ------------------
  if [ "$pmat_st" = "consulted" ]; then
    local idx idx_rec idx_computed
    idx=$(jq -r '.predicate.consultations.pmat.index_commit // ""' "$rcpt")
    # NOT `// "null"`: jq's alternative operator treats `false` as absent, so
    # `false // "null"` yields the string "null" and a correctly-recorded stale index
    # would be rejected as a misreport. Fixture row 9 caught this.
    idx_rec=$(jq -r 'if (.predicate.consultations.pmat | has("index_is_ancestor"))
                     then (.predicate.consultations.pmat.index_is_ancestor | tostring)
                     else "absent" end' "$rcpt")
    [ -n "$idx" ] || reject B1 "pmat.status is consulted but index_commit is absent; an unrecorded index cannot be shown to describe this PR" || return 1
    git -C "$REPO" cat-file -e "${idx}^{commit}" 2>/dev/null \
      || reject B1 "pmat.index_commit $idx does not resolve to a commit in $REPO" || return 1
    if git -C "$REPO" merge-base --is-ancestor "$idx" "$head" 2>/dev/null; then
      idx_computed=true
    else
      idx_computed=false
    fi
    # A receipt that MISREPORTS the ancestry is invalid whatever the verdict: the
    # 66-commit drift scar is an index answering about code that is not in the PR.
    [ "$idx_rec" = "$idx_computed" ] \
      || reject B1 "index_is_ancestor is recorded as $idx_rec but merge-base --is-ancestor $idx $head says $idx_computed" || return 1
    if [ "$idx_computed" = false ] && [ "$verdict" = "PASS" ]; then
      reject B6 "index_commit $idx is not an ancestor of head $head and the verdict is PASS; a stale index answers about code that is not in this PR (S3.A)" || return 1
    fi
  fi

  # --- B1: S3.A duplication coverage is RECORDED, never merely absent. -------
  #
  # PRREV-007 F4 measured two holes in the field S3.A calls "the highest-EV field in the
  # receipt", and both have the same shape:
  #
  #   (a) pmat's semantic index is Rust-only. On #2742 - 46 files, 7,244 insertions -
  #       3,533 of those insertions (48.8%) are sh, py and yaml, and a `pmat query`
  #       cannot return any of them. `duplication_hits: []` therefore meant "the half I
  #       can see is clean" and READ as "nothing like this exists".
  #   (b) prior art on an UNMERGED SIBLING BRANCH is invisible by construction. #2781
  #       found #2742's only because #2742 had merged 17 hours earlier. Luck, not
  #       mechanism.
  #
  # S3.0 applied to this field: it must not be possible to read "searched and found
  # nothing" as identical to "could not search". So the coverage the run ACHIEVED is
  # recorded per surface, an unrecorded coverage claim is rejected, and a surface the run
  # could not reach cannot sit under a PASS - exactly the rule rows 5 and 6 already
  # apply to an unreachable consultation.
  #
  # scripts/pr_review_duplication_scan.sh produces this block. The guard does NOT re-run
  # it: a gate that recomputes a 19-second sweep on every receipt is a gate that gets
  # routed around, and the attestation is L1-self for exactly this class of field.
  if [ "$pmat_st" = "consulted" ]; then
    local cov_missing cov_bad cov_none dup_sib dup_scanned dup_total
    jq -e '.predicate.consultations.pmat.duplication_hits | type == "array"' "$rcpt" >/dev/null 2>&1 \
      || reject B1 "pmat.status is consulted but duplication_hits is not an array; S3.A requires it present even when empty" || return 1

    jq -e '.predicate.consultations.pmat.duplication_coverage | (type == "object") and (length > 0)' "$rcpt" >/dev/null 2>&1 \
      || reject B1 "pmat.status is consulted but duplication_coverage is absent; an unrecorded coverage claim cannot be told apart from a searched-and-empty one, and S3.0 forbids exactly that (F4)" || return 1

    cov_missing=$(jq -r '(["rust","shell","python","config","docs","other","sibling_branches"]
                          - (.predicate.consultations.pmat.duplication_coverage | keys)) | join(", ")' "$rcpt")
    [ -z "$cov_missing" ] \
      || reject B1 "duplication_coverage records no verdict for: $cov_missing; a surface with no entry is the silently-absent coverage F4 exists to stop" || return 1

    # `.value` is bound to $v FIRST. Inside `["..."] | index(f)`, jq evaluates f with the
    # ARRAY as input, so a bare `.value` there reads the array's non-existent .value,
    # errors, and the check silently passes over everything. The dup-cov-bad-method probe
    # caught exactly that: with the unbound form the guard ACCEPTED "shell": "yes".
    cov_bad=$(jq -r '.predicate.consultations.pmat.duplication_coverage | to_entries
                     | map(select(.value as $v | (["semantic","lexical","none"] | index($v | tostring)) == null))
                     | map(.key + "=" + (.value | tostring)) | join(", ")' "$rcpt")
    [ -z "$cov_bad" ] \
      || reject B1 "duplication_coverage holds a method outside { semantic, lexical, none }: $cov_bad" || return 1

    cov_none=$(jq -r '.predicate.consultations.pmat.duplication_coverage | to_entries
                      | map(select(.value == "none")) | map(.key) | join(", ")' "$rcpt")
    if [ -n "$cov_none" ] && [ "$verdict" = "PASS" ]; then
      reject B1 "duplication_coverage could not search [$cov_none] and the verdict is PASS; a surface that was not searched must read DEGRADED, the same rule rows 5 and 6 apply to an unreachable consultation (S3.0)" || return 1
    fi

    jq -e '.predicate.consultations.pmat.duplication_horizon
           | (type == "array") and (length > 0) and (map(type == "string") | all)' "$rcpt" >/dev/null 2>&1 \
      || reject B1 "duplication_horizon is absent or is not a non-empty array of refspecs; an unstated horizon makes 'nothing found' and 'did not look off this branch' the same artifact (F4)" || return 1

    # WHOLE numbers, and the `floor` clauses are not decoration. The two rules below
    # compare these with `[ -lt ]`, which cannot read "2.0": it errors, the `if` reads
    # false, and BOTH horizon rules are skipped - so a receipt writing
    # `scanned: 2.0, total: 40` would sail past the capped-horizon check under a PASS.
    # A guard that fails OPEN on a value its own type check admitted is the class this
    # whole file exists to remove, so the type check admits integers only.
    jq -e '.predicate.consultations.pmat
           | (.horizon_branches_total | type == "number") and (.horizon_branches_scanned | type == "number")
             and ((.horizon_branches_total | floor) == .horizon_branches_total)
             and ((.horizon_branches_scanned | floor) == .horizon_branches_scanned)
             and (.horizon_branches_scanned >= 0) and (.horizon_branches_scanned <= .horizon_branches_total)' \
       "$rcpt" >/dev/null 2>&1 \
      || reject B1 "horizon_branches_total and horizon_branches_scanned must both be whole numbers with 0 <= scanned <= total; the horizon's denominator is what makes a partial sweep visible, and a fractional count would make the shell comparisons below error and skip" || return 1

    dup_sib=$(jq -r '.predicate.consultations.pmat.duplication_coverage.sibling_branches // ""' "$rcpt")
    dup_scanned=$(jq -r '.predicate.consultations.pmat.horizon_branches_scanned' "$rcpt")
    dup_total=$(jq -r '.predicate.consultations.pmat.horizon_branches_total' "$rcpt")

    if [ "$dup_sib" != "none" ] && [ "$dup_scanned" -eq 0 ] && [ "$dup_total" -gt 0 ]; then
      reject B1 "duplication_coverage claims the sibling branches were searched ($dup_sib) with horizon_branches_scanned=0 of $dup_total; a sweep that attempted nothing passes vacuously, which S3.D already calls DEGRADED for mutation" || return 1
    fi

    if [ "$dup_scanned" -lt "$dup_total" ] && [ "$verdict" = "PASS" ]; then
      reject B1 "the horizon sweep covered $dup_scanned of $dup_total sibling branches and the verdict is PASS; the branches it skipped were not searched, and unsearched is DEGRADED (F4)" || return 1
    fi

    jq -e '.predicate.consultations.pmat.symbols_searched | type == "number"' "$rcpt" >/dev/null 2>&1 \
      || reject B1 "pmat.symbols_searched must be a number; a scan whose needle count is unrecorded cannot have its precision judged, and record-only is not unenforced" || return 1
  fi

  # --- S1 / S4.2: every claim carries a mark, and a cited mark is verified. --
  local v
  while IFS= read -r v; do
    [ -n "$v" ] || continue
    case "$v" in
      no-grounding\|*)      reject B1 "a finding carries no properties.grounding (${v#*|}); an unmarked claim is a defect in the review (S1)" || return 1 ;;
      bad-grounding\|*)     reject B1 "the grounding of a finding is outside { cited, measured, asserted } (${v#*|})" || return 1 ;;
      no-failure-scenario\|*) reject B1 "a finding has an empty failure_scenario (${v#*|}); a finding that cannot name the failure it permits is a comment (S4.2)" || return 1 ;;
      cited-no-source\|*)   reject B1 "a cited finding has an empty source (${v#*|}); a citation with no source is an assertion wearing the mark of a citation (S1.1)" || return 1 ;;
      cited-no-excerpt\|*)  reject B1 "a cited finding has an empty excerpt (${v#*|}); a recorded excerpt that is not there is the same failure as no excerpt (S1.1)" || return 1 ;;
      cited-no-digest\|*)   reject B1 "a cited finding has no excerpt_sha256 (${v#*|}); cited is verified, not just labelled (S1.1)" || return 1 ;;
      asserted-blocking\|*) reject B1 "a finding marked asserted is classed blocking (${v#*|}); an asserted claim never blocks (S1)" || return 1 ;;
    esac
  done < <(jq -r '
      [ .runs[]? | .results[]? ]
      | map((.ruleId // "<no ruleId>") as $id | .properties as $p
            | if ($p == null) or (($p.grounding // "") | tostring | length) == 0
                then "no-grounding|" + $id
              elif (["cited","measured","asserted"] | index($p.grounding|tostring)) == null
                then "bad-grounding|" + $id + " = " + ($p.grounding|tostring)
              elif (($p.failure_scenario // "") | tostring | length) == 0
                then "no-failure-scenario|" + $id
              elif ($p.grounding == "asserted") and ($p.precision_class == "blocking")
                then "asserted-blocking|" + $id
              elif $p.grounding == "cited" and (($p.source // "") | tostring | length) == 0
                then "cited-no-source|" + $id
              elif $p.grounding == "cited" and (($p.excerpt // "") | tostring | length) == 0
                then "cited-no-excerpt|" + $id
              elif $p.grounding == "cited" and (($p.excerpt_sha256 // "") | tostring | length) == 0
                then "cited-no-digest|" + $id
              else empty end)
      | .[]' "$sarif")

  # --- S1.1: excerpt_sha256 = sha256(excerpt), over the bytes AS STORED. ----
  local line id want got
  while IFS=$'\t' read -r id want got; do
    [ -n "$id" ] || continue
    got=$(printf '%s' "$got" | base64 -d | sha256_stdin)
    [ "$want" = "$got" ] \
      || reject B1 "cited finding '$id' records excerpt_sha256 $want but sha256(excerpt) is $got; a citation whose digest does not match its excerpt is not verified (S1.1)" || return 1
  done < <(jq -r '
      .runs[]? | .results[]? | select(.properties.grounding == "cited")
      | [(.ruleId // "<no ruleId>"), .properties.excerpt_sha256,
          (.properties.excerpt | @base64) ] | @tsv' "$sarif")

  # --- B4: a comparative claim carries its comparator (S3.C.1). -------------
  # The 2.93x Ollama rule: the book published that ratio from a harness that never
  # ran Ollama. Every field below is what makes the claim reproducible by a reader.
  local claim missing
  while IFS= read -r claim; do
    [ -n "$claim" ] || continue
    reject B4 "$claim" || return 1
  done < <(jq -r '
      def need($c; $n):
        [ "command","version","env_sha256","artifact_sha256","log_path" ]
        | map(. as $f | $c | getpath([$f]) as $v
              | select(($v == null) or (($v | tostring) == "") or ($v == [])) | $f)
        | if length == 0 then empty
          else "comparative claim " + $n + " is missing comparator field(s): "
               + (join(", ")) + "; absent any field the claim is unverified and blocks (S3.C.1)"
          end;
      [ .predicate.consultations.crux.comparative_claims[]? ]
      | to_entries[]
      | need(.value.comparator // {}; "#" + (.key|tostring)
             + " (" + ((.value.claim // "<unnamed>")|tostring) + ")")
      ' "$rcpt")

  # A comparative claim asserted in a FINDING must also be backed. A claim the
  # reviewer wrote into a result but never recorded in comparative_claims[] is the
  # never-ran-Ollama shape with an extra step.
  local n_recorded
  n_recorded=$(jq -r '[ .predicate.consultations.crux.comparative_claims[]? ] | length' "$rcpt")
  while IFS=$'\t' read -r id got; do
    [ -n "$id" ] || continue
    got=$(printf '%s' "$got" | base64 -d)
    if match_comparative "$got"; then
      [ "$n_recorded" -gt 0 ] || reject B4 "finding $id states a comparative claim -- $(printf '%s' "$got" | cut -c1-70) -- but consultations.crux.comparative_claims is empty; a competitor ratio with no recorded comparator is unverified, per S3.C.1" || return 1
    fi
  done < <(jq -r '.runs[]? | .results[]?
      | [(.ruleId // "<no ruleId>"), ((.message.text // "") | @base64)] | @tsv' "$sarif")

  return 0
}

# -------------------------------------------------------------------------
# S6.1 POSITIVE CONTROL, FIRST.
#
# Before validating anything real, the guard validates deliberately malformed receipts
# and REQUIRES a non-zero exit from each. If a malformed receipt passes, this guard's
# GREEN is a count of files rather than a verdict, and the run fails. Same idiom as
# dogfood.sh's `pv validate` positive control.
#
# TWO controls at DIFFERENT DEPTHS, and each asserts WHICH class fired.
#
# The depth matters: a schema-only control stays green even if every semantic branch
# below the schema gate is deleted, which is precisely the mutation PRREV-004 will try.
# So the second control is signature-valid and schema-valid, and its only defect is
# semantic (B2) - it can only fire by reaching the actor check.
#
# Asserting the CLASS matters just as much. The first draft of this guard had a
# self-review control whose signature was a stub, so it "fired" at the signature branch
# while reporting itself as the self-review control: a control that passes for a reason
# other than the one it names is mislabeled evidence, not a control. Requiring the
# expected class turns that silent mislabel into a loud failure.
# -------------------------------------------------------------------------
PC_TMP=$(mktemp -d "${TMPDIR:-/tmp}/pr-review-positive-control.XXXXXX")
case "$PC_TMP" in
  */pr-review-positive-control.*) ;;
  *) echo "$PROG: FAIL - refusing to use scratch dir $PC_TMP" >&2; exit 1 ;;
esac
cleanup() {
  # This expands into rm -rf, so it is gated twice: the path must still look like the
  # scratch directory this script created, and must be neither empty nor the root.
  cleanup_dir=${PC_TMP:-}
  case "$cleanup_dir" in
    */pr-review-positive-control.*) ;;
    *) return 0 ;;
  esac
  if [ -z "$cleanup_dir" ] || [ "$cleanup_dir" = "/" ]; then
    return 0
  fi
  rm -rf -- "$cleanup_dir"
}
trap cleanup EXIT

GUARD_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PC_SEED_DIR=${PR_REVIEW_POSITIVE_CONTROL_DIR:-$GUARD_DIR/../tests/fixtures/pr-review/positive-control}

# run_control <name> <expected-class> <expected-reason-substring> <dir>
#
# The REASON is asserted, not just the class. B1 covers a dozen branches, so a control
# that fires on a different one still reports the class it was asked for. Measured: with
# only the class asserted, deleting the in-toto schema gate left the schema control
# firing (on the signature branch) and the mutation SURVIVED. Pinning the reason kills it.
run_control() {
  local name=$1 want=$2 want_reason=$3 dir=$4
  REJECT_CLASS=''; REJECT_REASON=''
  if validate_receipt "$dir"; then
    cat >&2 <<EOF
$PROG: POSITIVE CONTROL FAILED ($name)
  A deliberately malformed receipt was ACCEPTED (exit 0). This guard's verdicts cannot
  be trusted this run: a green result would be a count of files, not a verdict (S6.1).
  Refusing to validate anything.
EOF
    return 1
  fi
  if [ "$REJECT_CLASS" != "$want" ]; then
    cat >&2 <<EOF
$PROG: POSITIVE CONTROL MISFIRED ($name)
  Expected the receipt to be rejected under $want; it was rejected under $REJECT_CLASS:
    $REJECT_REASON
  The control fired for a reason other than the one it exists to test, so it is no
  longer evidence that the $want branch works. Refusing to validate anything.
EOF
    return 1
  fi
  case "$REJECT_REASON" in
    *"$want_reason"*) ;;
    *)
      cat >&2 <<EOF
$PROG: POSITIVE CONTROL MISFIRED ($name)
  Rejected under the expected class $want, but on the wrong branch. Expected a reason
  containing:
    $want_reason
  got:
    $REJECT_REASON
  The control no longer proves the branch it names is wired. Refusing to validate.
EOF
      return 1 ;;
  esac
  printf 'positive-control  %-15s fired (%s: %s)\n' "$name" "$REJECT_CLASS" \
    "$(printf '%s' "$REJECT_REASON" | cut -c1-64)"
  return 0
}

# Control 1: schema depth. Synthesized inline - it must not depend on any committed
# artifact, so that a deleted fixture tree cannot silently take the control with it.
PC1="$PC_TMP/schema-invalid"; mkdir -p "$PC1"
printf '{"_type":"https://in-toto.io/Statement/v0.9","subject":[],"predicateType":"nope"}\n' \
  > "$PC1/receipt.intoto.jsonl"
printf '{"version":"2.1.0","runs":[]}\n' > "$PC1/findings.sarif"
: > "$PC1/receipt.intoto.jsonl.minisig"
run_control schema-invalid B1 "fails schemas/in-toto-statement-v1.json" "$PC1" || exit 1

# Control 2: semantic depth. A committed receipt that is schema-valid AND correctly
# signed, whose only defect is that the reviewer is the author. It carries its own
# public key, so it verifies whatever PR_REVIEW_PUBKEY is set to for the real run.
# seeded_control <name> <seed-subdir> <class> <reason-substring>
seeded_control() {
  local name=$1 seed="$PC_SEED_DIR/$2" class=$3 reason=$4 d="$PC_TMP/$1"
  if [ ! -f "$seed/receipt.intoto.jsonl" ]; then
    echo "$PROG: FAIL - positive-control fixture missing at $seed" >&2
    echo "  Without it, deleting the check it pins would leave this guard green (S6.1)." >&2
    echo "  Refusing to validate anything." >&2
    return 1
  fi
  mkdir -p "$d"
  cp -- "$seed/receipt.intoto.jsonl" "$seed/findings.sarif" \
        "$seed/receipt.intoto.jsonl.minisig" "$d/" || {
    echo "$PROG: FAIL - positive-control fixture at $seed is incomplete" >&2; return 1; }
  (
    PUBKEY="$PC_SEED_DIR/positive-control.pub"
    run_control "$name" "$class" "$reason" "$d"
  )
}

seeded_control self-review     self-review     B2 "reviewer_actor.id ="        || exit 1
seeded_control findings-digest findings-digest B1 "findings_ref.sha256"        || exit 1
seeded_control cost-missing    cost-missing    B1 "predicate.cost must carry"  || exit 1

# -------------------------------------------------------------------------
# The real receipts.
# -------------------------------------------------------------------------
if [ "$#" -eq 0 ]; then
  echo "$PROG: FAIL - no receipt directory given." >&2
  echo "  usage: $PROG <receipt-dir> [<receipt-dir> ...]" >&2
  echo "  A run over zero receipts is not a pass (S6.3: a missing receipt is RED)." >&2
  exit 1
fi

RC=0
for d in "$@"; do
  REJECT_CLASS=''; REJECT_REASON=''
  if validate_receipt "$d"; then
    printf 'ACCEPT  %s\n' "$d"
  else
    printf 'REJECT  %s  [%s]\n    %s\n' "$d" "$REJECT_CLASS" "$REJECT_REASON"
    RC=1
  fi
done
exit "$RC"
