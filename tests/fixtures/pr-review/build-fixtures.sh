#!/usr/bin/env bash
# Regenerate the committed pr-review fixtures.
#
# The fixtures are COMMITTED static bytes - the guard is run against exactly what is in
# git, signature included. This script exists so they can be regenerated deliberately
# when the spec or the topology changes, not so the harness can build them at test
# time. `tests/pr-review.bats` never runs it.
#
# Every digest inside a fixture is COMPUTED here rather than typed:
#   findings_ref.sha256  = sha256(findings.sarif)
#   excerpt_sha256       = sha256(excerpt bytes as stored)
# A hand-typed digest is a fixture that passes for the wrong reason the first time
# someone edits the file it describes.
#
# Usage: build-fixtures.sh            (rebuilds every row in place)
set -euo pipefail

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
KEY="$HERE/keys/pr-review-test-TEST-ONLY.key"
PUB="$HERE/keys/pr-review-test.pub"

command -v jq       >/dev/null || { echo "need jq" >&2; exit 1; }
command -v minisign >/dev/null || { echo "need minisign" >&2; exit 1; }
[ -f "$KEY" ] || { echo "missing fixture signing key $KEY" >&2; exit 1; }

# Commit SHAs of the fixture repository (tests/fixtures/pr-review/make-fixture-repo.sh).
# Read from expected-shas.txt so there is ONE place they are written down.
sha_of() { awk -v k="$1" '$1==k{print $2}' "$HERE/expected-shas.txt"; }
C1=$(sha_of C1)   # the merge base of both fixture PRs
C3=$(sha_of C3)   # main's tip: NOT an ancestor of F1, so a stale index
F1=$(sha_of F1)   # the GPU pull request head, adds src/cuda/kernel.cu
D1=$(sha_of D1)   # the docs-only pull request head
for v in C1 C3 F1 D1; do
  [ -n "${!v}" ] || { echo "expected-shas.txt has no $v" >&2; exit 1; }
done

AUTHOR='agent:claude-opus-5/session-authoring'
REVIEWER='agent:claude-opus-5/session-review'

# The one excerpt used by the cited findings. Single-line ASCII on purpose: the digest
# is taken over these bytes exactly as stored, so an editor's line-ending or wrapping
# behaviour must not be able to change it.
EXCERPT='Operations in different, non-default streams may execute concurrently; there is no implicit synchronization between them.'
EXCERPT_SHA=$(printf '%s' "$EXCERPT" | sha256sum | cut -d' ' -f1)

emit() {  # emit <row-dir> <sarif-json> <receipt-json-with-@@FINDINGS_SHA@@>
  local dir="$HERE/$1" sarif=$2 receipt=$3
  case "$1" in ""|*/*|.|..) echo "refusing to build fixture $1" >&2; exit 1 ;; esac
  if [ -z "$dir" ] || [ "$dir" = "/" ]; then
    echo "refusing rm -rf on $dir" >&2; exit 1
  fi
  rm -rf -- "$dir"; mkdir -p -- "$dir"
  printf '%s\n' "$sarif" | jq . > "$dir/findings.sarif"
  local fsha
  fsha=$(sha256sum "$dir/findings.sarif" | cut -d' ' -f1)
  printf '%s\n' "${receipt//@@FINDINGS_SHA@@/$fsha}" | jq -c . > "$dir/receipt.intoto.jsonl"
  minisign -q -S -s "$KEY" -m "$dir/receipt.intoto.jsonl" \
    -t "PRREV-003 fixture $1" -c "TEST ONLY fixture signature" </dev/null
  minisign -q -V -m "$dir/receipt.intoto.jsonl" -p "$PUB" >/dev/null \
    || { echo "self-check failed: $1 signature does not verify" >&2; exit 1; }
  echo "built $1"
}

# --- receipt builder -------------------------------------------------------
# receipt <head> <base> <verdict> <pmat-json> <cuda-json> <crux-json> <mutation-json>
#         [<author-id>] [<reviewer-id>]
receipt() {
  local head=$1 base=$2 verdict=$3 pmat=$4 cuda=$5 crux=$6 mut=$7
  local author=${8:-$AUTHOR} reviewer=${9:-$REVIEWER}
  cat <<JSON
{
  "_type": "https://in-toto.io/Statement/v1",
  "subject": [
    { "name": "git+https://github.com/paiml/aprender",
      "digest": { "sha1": "$head" } }
  ],
  "predicateType": "https://paiml.dev/attestations/pr-review/v2",
  "predicate": {
    "skill_version": "2.0.0",
    "attestation_level": "L1-self",
    "pr": 2783,
    "base_sha": "$base",
    "head_sha": "$head",
    "author_actor":   { "kind": "agent", "id": "$author" },
    "reviewer_actor": { "kind": "agent", "id": "$reviewer" },
    "affected_crates": ["aprender-core"],
    "verdict": "$verdict",
    "consultations": {
      "pmat": $pmat,
      "cuda": $cuda,
      "crux": $crux,
      "mutation": $mut
    },
    "findings_ref": { "path": "findings.sarif", "sha256": "@@FINDINGS_SHA@@" },
    "cost": { "input_tokens": 41200, "output_tokens": 3180, "wall_seconds": 74 }
  }
}
JSON
}

# S3.A duplication coverage (PRREV-009 / backtest F4). Every surface carries a method,
# because a surface with NO entry is the silently-absent coverage that made
# `duplication_hits: []` read as "nothing like this exists" on a diff that was 48.8%
# shell, python and yaml - none of which pmat's semantic index can see.
#
# total/scanned are 0/0 here and that is CORRECT rather than convenient: the fixture
# repository has one remote ref (origin/main) and no unmerged sibling, so the honest
# denominator is zero. The horizon rules that need a non-zero denominator are exercised
# by the branch probes in tests/pr-review.bats, not by these rows.
DUP_FULL='"duplication_coverage":{"rust":"semantic","shell":"lexical","python":"lexical","config":"lexical","docs":"lexical","other":"lexical","sibling_branches":"lexical"},"duplication_horizon":["HEAD","refs/remotes/origin/* unmerged into origin/main"],"horizon_branches_total":0,"horizon_branches_scanned":0,"symbols_searched":4'
# The same run with the shell surface unreachable - e.g. git grep unavailable. Rows 16
# and 17 are the same coverage under two different verdicts.
DUP_SHELL_NONE='"duplication_coverage":{"rust":"semantic","shell":"none","python":"none","config":"none","docs":"none","other":"none","sibling_branches":"lexical"},"duplication_horizon":["HEAD","refs/remotes/origin/* unmerged into origin/main"],"horizon_branches_total":0,"horizon_branches_scanned":0,"symbols_searched":4'

PMAT_OK="{\"status\":\"consulted\",\"index_commit\":\"$C1\",\"index_is_ancestor\":true,\"complexity_delta\":[],\"tdg_delta\":[],\"satd_introduced\":[],\"duplication_hits\":[],\"cache_hits\":0,$DUP_FULL}"
PMAT_STALE="{\"status\":\"consulted\",\"index_commit\":\"$C3\",\"index_is_ancestor\":false,\"complexity_delta\":[],\"tdg_delta\":[],\"satd_introduced\":[],\"duplication_hits\":[],\"cache_hits\":0,$DUP_FULL}"
PMAT_SHELL_BLIND="{\"status\":\"consulted\",\"index_commit\":\"$C1\",\"index_is_ancestor\":true,\"complexity_delta\":[],\"tdg_delta\":[],\"satd_introduced\":[],\"duplication_hits\":[],\"cache_hits\":0,$DUP_SHELL_NONE}"
PMAT_UNREACHABLE='{"status":"unreachable","trigger_reason":"pmat MCP server: ConnectionRefused"}'
PMAT_NT='{"status":"not-triggered","trigger_reason":"pmat is unconditional; not-triggered is never correct for it"}'

CUDA_OK="{\"status\":\"consulted\",\"trigger_reason\":\"diff touches src/cuda/kernel.cu\",\"queries\":[{\"q\":\"stream ordering guarantees between non-default streams\",\"excerpt_sha256\":\"$EXCERPT_SHA\",\"result\":\"found\"}]}"
CUDA_NT='{"status":"not-triggered","trigger_reason":"no changed path and no commit message matches the S3.B trigger"}'

CRUX_NT='{"status":"not-triggered","trigger_reason":"no CLI flag, HTTP route, MCP tool, config key or output format changed","surfaces":[],"contracts":[],"comparative_claims":[]}'
CRUX_OK_FULL="{\"status\":\"consulted\",\"surfaces\":[\"apr bench --gpu\"],\"contracts\":[\"CRUX-A-08\"],\"gap_effect\":\"closes\",\"crux_coverage\":\"covered\",\"comparative_claims\":[{\"claim\":\"1.21x llama.cpp on aarch64 Q4_K\",\"comparator\":{\"command\":[\"llama-cli\",\"-m\",\"qwen2.5-1.5b-q4_k_m.gguf\",\"-p\",\"2+2?\",\"-n\",\"128\",\"-ngl\",\"99\"],\"version\":\"llama.cpp b4021\",\"env_sha256\":\"9f2b1c4d5e6a7b8c9d0e1f2a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e\",\"artifact_sha256\":\"1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f809\",\"log_path\":\"evidence/bench/2026-08-30-gb10/comparator.log\"}}]}"
CRUX_OK_BARE='{"status":"consulted","surfaces":[],"contracts":[],"gap_effect":"none","crux_coverage":"covered","comparative_claims":[]}'
CRUX_BAD_COMPARATOR='{"status":"consulted","surfaces":["docs/performance.md"],"contracts":["CRUX-A-08"],"gap_effect":"none","crux_coverage":"covered","comparative_claims":[{"claim":"2.93x Ollama","comparator":{"version":"ollama 0.5.7","env_sha256":"","log_path":"evidence/bench/none.log"}}]}'

MUT_OK='{"status":"consulted","scope":"in-diff","attempted":37,"killed":37,"survivors":[]}'
MUT_VACUOUS='{"status":"consulted","scope":"in-diff","attempted":0,"killed":0,"survivors":[]}'
MUT_NT='{"status":"not-triggered","trigger_reason":"docs-only diff; S3.D row 3"}'

# --- SARIF builders --------------------------------------------------------
sarif_empty() {
  cat <<'JSON'
{ "$schema": "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [ { "tool": { "driver": { "name": "pmat" } },
              "invocations": [ { "executionSuccessful": true, "toolExecutionNotifications": [] } ],
              "results": [] } ] }
JSON
}

# sarif_one <driver> <result-json>
sarif_one() {
  cat <<JSON
{ "\$schema": "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [ { "tool": { "driver": { "name": "$1" } },
              "invocations": [ { "executionSuccessful": true, "toolExecutionNotifications": [] } ],
              "results": [ $2 ] } ] }
JSON
}

RESULT_MEASURED='{
  "ruleId": "complexity_delta", "level": "warning",
  "message": { "text": "fixture_kernel launch wrapper rose from 6 to 14 cyclomatic." },
  "properties": { "grounding": "measured",
    "source": "pmat analyze complexity --format json",
    "failure_scenario": "The launch wrapper grows a branch nothing covers, and the untested arm is the one that skips the stream sync.",
    "precision_class": "advisory" } }'

RESULT_CITED=$(cat <<JSON
{
  "ruleId": "device_behaviour_claim", "level": "error",
  "message": { "text": "The PR asserts kernels on separate streams are implicitly ordered; the documentation says the opposite." },
  "properties": { "grounding": "cited",
    "source": "nvidia-cuda-docs: CUDA C++ Programming Guide, Streams",
    "excerpt": "$EXCERPT",
    "excerpt_sha256": "$EXCERPT_SHA",
    "failure_scenario": "The second kernel reads the output of the first kernel before it is written, producing a race that only appears under load.",
    "precision_class": "blocking" } }
JSON
)

# ===========================================================================
# ROW 1 - cuda not-triggered on a diff touching src/cuda/            -> RED B1
# ===========================================================================
SARIF=$(
sarif_empty
)
RCPT=$(receipt "$F1" "$C1" PASS "$PMAT_OK" "$CUDA_NT" "$CRUX_NT" "$MUT_OK")
emit row-01-cuda-not-triggered-on-cuda-diff "$SARIF" "$RCPT"

# ===========================================================================
# ROW 2 - mutation.attempted 0 with status consulted                 -> RED B1
# ===========================================================================
SARIF=$(
sarif_empty
)
RCPT=$(receipt "$F1" "$C1" PASS "$PMAT_OK" "$CUDA_OK" "$CRUX_NT" "$MUT_VACUOUS")
emit row-02-mutation-attempted-zero "$SARIF" "$RCPT"

# ===========================================================================
# ROW 3 - cited finding with an empty excerpt                        -> RED B1
# ===========================================================================
SARIF=$(
sarif_one nvidia-cuda-docs '{
      "ruleId": "device_behaviour_claim", "level": "error",
      "message": { "text": "Stream ordering claim." },
      "properties": { "grounding": "cited",
        "source": "nvidia-cuda-docs: CUDA C++ Programming Guide, Streams",
        "excerpt": "",
        "excerpt_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "failure_scenario": "A race that only appears under load.",
        "precision_class": "blocking" } }'
)
RCPT=$(receipt "$D1" "$C1" FINDINGS "$PMAT_OK" "$CUDA_NT" "$CRUX_NT" "$MUT_NT")
emit row-03-cited-empty-excerpt "$SARIF" "$RCPT"

# ===========================================================================
# ROW 4 - comparative claim with no comparator command or hash       -> RED B4
# ===========================================================================
SARIF=$(
sarif_one crux '{
      "ruleId": "comparative_claim", "level": "error",
      "message": { "text": "docs/performance.md states apr sustains 2.93x Ollama on 1.5B Q4_K decode." },
      "properties": { "grounding": "asserted",
        "source": "docs/performance.md line 41",
        "failure_scenario": "A reader reproduces the published ratio and gets parity, because the harness never executed the comparator.",
        "precision_class": "advisory" } }'
)
RCPT=$(receipt "$D1" "$C1" FINDINGS "$PMAT_OK" "$CUDA_NT" "$CRUX_BAD_COMPARATOR" "$MUT_NT")
emit row-04-comparative-claim-no-comparator "$SARIF" "$RCPT"

# ===========================================================================
# ROW 5 - pmat unreachable, verdict PASS                             -> RED B1
# ===========================================================================
SARIF=$(
sarif_empty
)
RCPT=$(receipt "$D1" "$C1" PASS "$PMAT_UNREACHABLE" "$CUDA_NT" "$CRUX_NT" "$MUT_NT")
emit row-05-unreachable-pmat-verdict-pass "$SARIF" "$RCPT"

# ===========================================================================
# ROW 6 - pmat unreachable, verdict DEGRADED                          -> GREEN
# DISCRIMINATION CASE. Without it, "refuse every receipt" reads green.
# ===========================================================================
SARIF=$(
cat <<'JSON'
{ "$schema": "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [ { "tool": { "driver": { "name": "pmat" } },
              "invocations": [ { "executionSuccessful": false,
                "toolExecutionNotifications": [ { "level": "error",
                  "message": { "text": "pmat MCP server: ConnectionRefused. Consultation could not be performed; verdict DEGRADED." } } ] } ],
              "results": [] } ] }
JSON

)
RCPT=$(receipt "$D1" "$C1" DEGRADED "$PMAT_UNREACHABLE" "$CUDA_NT" "$CRUX_NT" "$MUT_NT")
emit row-06-unreachable-pmat-verdict-degraded "$SARIF" "$RCPT"

# ===========================================================================
# ROW 7 - honest docs-only PR, all consultations not-triggered        -> GREEN
# DISCRIMINATION CASE.
# ===========================================================================
SARIF=$(
sarif_empty
)
RCPT=$(receipt "$D1" "$C1" PASS "$PMAT_NT" "$CUDA_NT" "$CRUX_NT" "$MUT_NT")
emit row-07-honest-docs-only-all-not-triggered "$SARIF" "$RCPT"

# ===========================================================================
# ROW 8 - reviewer_actor = author_actor                              -> RED B2
# ===========================================================================
SARIF=$(
sarif_empty
)
RCPT=$(receipt "$D1" "$C1" PASS "$PMAT_OK" "$CUDA_NT" "$CRUX_NT" "$MUT_NT" \
      "$AUTHOR" "$AUTHOR")
emit row-08-self-review "$SARIF" "$RCPT"

# ===========================================================================
# ROW 9 - index_commit not an ancestor of head_sha, verdict PASS     -> RED B6
# ===========================================================================
SARIF=$(
sarif_empty
)
RCPT=$(receipt "$F1" "$C1" PASS "$PMAT_STALE" "$CUDA_OK" "$CRUX_NT" "$MUT_OK")
emit row-09-stale-index-verdict-pass "$SARIF" "$RCPT"

# ===========================================================================
# ROW 10 - base_sha is not git merge-base origin/main head_sha       -> RED B1
# base_sha names main's tip instead of the fork point, so the review's diff
# would swallow the two unrelated PRs another agent landed at C2 and C3.
# ===========================================================================
SARIF=$(
sarif_empty
)
RCPT=$(receipt "$F1" "$C3" PASS "$PMAT_OK" "$CUDA_OK" "$CRUX_NT" "$MUT_OK")
emit row-10-base-sha-not-merge-base "$SARIF" "$RCPT"

# ===========================================================================
# ROW 11 - finding with an empty failure_scenario                    -> RED B1
# ===========================================================================
SARIF=$(
sarif_one pmat '{
      "ruleId": "duplication_hits", "level": "warning",
      "message": { "text": "This fused path may already exist." },
      "properties": { "grounding": "measured",
        "source": "pmat query \"fused gate up matvec\" --limit 5",
        "failure_scenario": "",
        "precision_class": "advisory" } }'
)
RCPT=$(receipt "$D1" "$C1" FINDINGS "$PMAT_OK" "$CUDA_NT" "$CRUX_NT" "$MUT_NT")
emit row-11-empty-failure-scenario "$SARIF" "$RCPT"

# ===========================================================================
# ROW 12 - excerpt_sha256 is not sha256(excerpt)                     -> RED B1
# The excerpt and the digest are each individually well-formed. Only the
# relation between them is broken - which is the whole point of S1.1.
# ===========================================================================
SARIF=$(
R12=$(cat <<JSON
{ "ruleId": "device_behaviour_claim", "level": "error",
  "message": { "text": "Stream ordering claim." },
  "properties": { "grounding": "cited",
    "source": "nvidia-cuda-docs: CUDA C++ Programming Guide, Streams",
    "excerpt": "$EXCERPT",
    "excerpt_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    "failure_scenario": "A race that only appears under load.",
    "precision_class": "blocking" } }
JSON
)
sarif_one nvidia-cuda-docs "$R12"
)
RCPT=$(receipt "$D1" "$C1" FINDINGS "$PMAT_OK" "$CUDA_NT" "$CRUX_NT" "$MUT_NT")
emit row-12-excerpt-digest-mismatch "$SARIF" "$RCPT"

# ===========================================================================
# ROW 13 - valid receipt, invalid signature                          -> RED B1
# Built by signing DIFFERENT bytes: a real Ed25519 signature over the wrong
# payload, not a corrupted string. A truncated file would also be rejected by
# minisign's parser, which would test the parser instead of the verification.
# ===========================================================================
SARIF=$(
sarif_empty
)
RCPT=$(receipt "$D1" "$C1" PASS "$PMAT_NT" "$CUDA_NT" "$CRUX_NT" "$MUT_NT")
emit row-13-invalid-signature "$SARIF" "$RCPT"
DECOY=$(mktemp); printf 'these are not the receipt bytes\n' > "$DECOY"
minisign -q -S -s "$KEY" -m "$DECOY" -t "decoy" -c "decoy" </dev/null
mv -- "$DECOY.minisig" "$HERE/row-13-invalid-signature/receipt.intoto.jsonl.minisig"
rm -f -- "$DECOY"
if minisign -q -V -m "$HERE/row-13-invalid-signature/receipt.intoto.jsonl" -p "$PUB" >/dev/null 2>&1; then
  echo "self-check failed: the row 13 signature verifies, so it does not test row 13" >&2; exit 1
fi
echo "built row-13-invalid-signature (signature deliberately over other bytes)"

# ===========================================================================
# ROW 14 - complete receipt, GPU PR, all four consulted, findings     -> GREEN
# DISCRIMINATION CASE, and the widest one: it carries a cited finding whose
# digest matches, a measured finding, and a comparative claim with a COMPLETE
# comparator. If the guard refuses this, it refuses correct work.
# ===========================================================================
SARIF=$(
cat <<JSON
{ "\$schema": "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [
    { "tool": { "driver": { "name": "pmat" } },
      "invocations": [ { "executionSuccessful": true, "toolExecutionNotifications": [] } ],
      "results": [ $RESULT_MEASURED ] },
    { "tool": { "driver": { "name": "nvidia-cuda-docs" } },
      "invocations": [ { "executionSuccessful": true, "toolExecutionNotifications": [] } ],
      "results": [ $RESULT_CITED ] },
    { "tool": { "driver": { "name": "crux" } },
      "invocations": [ { "executionSuccessful": true, "toolExecutionNotifications": [] } ],
      "results": [ { "ruleId": "comparative_claim", "level": "note",
        "message": { "text": "Benchmark output records 1.21x llama.cpp on aarch64 Q4_K; the comparator command, version and artifact hash are recorded." },
        "properties": { "grounding": "measured",
          "source": "evidence/bench/2026-08-30-gb10/comparator.log",
          "failure_scenario": "An unreproducible ratio reaches the book and a reader measures parity instead.",
          "precision_class": "advisory" } } ] },
    { "tool": { "driver": { "name": "cargo-mutants" } },
      "invocations": [ { "executionSuccessful": true, "toolExecutionNotifications": [] } ],
      "results": [] } ] }
JSON

)
RCPT=$(receipt "$F1" "$C1" FINDINGS "$PMAT_OK" "$CUDA_OK" "$CRUX_OK_FULL" "$MUT_OK")
emit row-14-complete-gpu-review "$SARIF" "$RCPT"

# ===========================================================================
# ROW 15 - a result carrying NO properties.grounding at all          -> RED B1
#
# NOT in spec S6.3's fourteen rows. It is owed to PRREV-003 by
# contracts/pr-review-skill-v2.yaml, whose falsification test F-PRREV-001 is
# recorded LIVE-PENDING on exactly this fixture: S6.3 covers an empty source or
# excerpt (row 3), an empty failure_scenario (row 11) and a digest mismatch
# (row 12), but NOTHING covers a claim with no mark at all - which is the one
# S8 metric (`unmarked_claims = 0`) that the fourteen rows leave asserted.
# ===========================================================================
SARIF=$(
sarif_one pmat '{
      "ruleId": "duplication_hits", "level": "warning",
      "message": { "text": "This fused path may already exist elsewhere in the tree." },
      "properties": { "source": "pmat query \"fused gate up matvec\"",
        "failure_scenario": "Two implementations diverge and one keeps the pre-LAYOUT-001 column-major order.",
        "precision_class": "advisory" } }'
)
RCPT=$(receipt "$D1" "$C1" FINDINGS "$PMAT_OK" "$CUDA_NT" "$CRUX_NT" "$MUT_NT")
emit row-15-finding-with-no-grounding-mark "$SARIF" "$RCPT"

# ===========================================================================
# ROW 16 - a duplication surface that could not be searched, verdict PASS -> RED B1
#
# NOT in spec S6.3's fourteen rows. It is owed by PRREV-007's backtest finding F4:
# `duplication_hits: []` on a diff that is 48.8% shell/python/yaml meant "the half pmat
# can see is clean" and read as "nothing like this exists". S3.0's three-state rule
# applied to this field says the unsearched surface must not sit under a PASS - it is
# the same rule rows 5 and 6 apply to an unreachable consultation, one level down.
# ===========================================================================
SARIF=$(
sarif_empty
)
RCPT=$(receipt "$D1" "$C1" PASS "$PMAT_SHELL_BLIND" "$CUDA_NT" "$CRUX_NT" "$MUT_NT")
emit row-16-duplication-surface-unsearched-verdict-pass "$SARIF" "$RCPT"

# ===========================================================================
# ROW 17 - the SAME unsearched surface, verdict DEGRADED             -> GREEN
# DISCRIMINATION CASE. Without it, "reject every receipt that admits a gap" reads
# green - and the rule would punish the honest receipt exactly as hard as the
# silent one, which is how a coverage field learns to stay empty.
# ===========================================================================
SARIF=$(
cat <<'JSON'
{ "$schema": "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [ { "tool": { "driver": { "name": "pmat" } },
              "invocations": [ { "executionSuccessful": false,
                "toolExecutionNotifications": [ { "level": "error",
                  "message": { "text": "pr_review_duplication_scan.sh: git grep unavailable; the shell, python, config, docs and other surfaces were NOT searched. Verdict DEGRADED." } } ] } ],
              "results": [] } ] }
JSON

)
RCPT=$(receipt "$D1" "$C1" DEGRADED "$PMAT_SHELL_BLIND" "$CUDA_NT" "$CRUX_NT" "$MUT_NT")
emit row-17-duplication-surface-unsearched-verdict-degraded "$SARIF" "$RCPT"

# ===========================================================================
# THE POSITIVE CONTROLS (S6.1) - not rows of the S6.3 table.
#
# scripts/check_pr_review_receipt.sh validates these BEFORE it validates anything
# real, and requires each to be rejected under a specific class AND with a specific
# reason. Each is schema-valid and correctly signed, so it can only be rejected by
# REACHING the semantic branch it pins - which is what makes it evidence that the
# branch is still wired.
#
# Three of them, not one, because they are a MUTATION-KILL SET. Measured: with a
# single schema-depth control, deleting the signature check, the merge-base check,
# or any of nine others left the guard green on receipts no S6.3 row covers. The
# three branches below are the ones the fourteen rows do not reach.
#
# Their SHAs are deliberately all-zero: every class they pin is evaluated BEFORE the
# merge-base check, so the controls need no git repository to resolve anything. That
# is what lets them run identically in CI, in a worktree, and on a laptop.
# ===========================================================================
PC="$HERE/positive-control"
if [ -z "$PC" ] || [ "$PC" = "/" ]; then
  echo "refusing rm -rf on $PC" >&2; exit 1
fi
rm -rf -- "$PC"; mkdir -p -- "$PC"
cp -- "$PUB" "$PC/positive-control.pub"
Z=0000000000000000000000000000000000000000

# pc_emit <subdir> <author-id> <reviewer-id> <digest-mode> <cost-mode>
#   digest-mode: ok | wrong        cost-mode: ok | missing
pc_emit() {
  local sub=$1 author=$2 reviewer=$3 digest=$4 cost=$5 d="$PC/$1"
  mkdir -p -- "$d"
  sarif_empty | jq . > "$d/findings.sarif"
  local fsha
  fsha=$(sha256sum "$d/findings.sarif" | cut -d' ' -f1)
  [ "$digest" = wrong ] && fsha=$(printf '%064d' 0)
  local costblock='"cost": { "input_tokens": 0, "output_tokens": 0, "wall_seconds": 0 }'
  [ "$cost" = missing ] && costblock='"cost_omitted_on_purpose": true'
  jq -c . > "$d/receipt.intoto.jsonl" <<JSON
{
  "_type": "https://in-toto.io/Statement/v1",
  "subject": [ { "name": "git+https://github.com/paiml/aprender", "digest": { "sha1": "$Z" } } ],
  "predicateType": "https://paiml.dev/attestations/pr-review/v2",
  "predicate": {
    "skill_version": "2.0.0",
    "attestation_level": "L1-self",
    "pr": 0,
    "base_sha": "$Z",
    "head_sha": "$Z",
    "author_actor":   { "kind": "agent", "id": "$author" },
    "reviewer_actor": { "kind": "agent", "id": "$reviewer" },
    "affected_crates": [],
    "verdict": "PASS",
    "consultations": {
      "pmat":     { "status": "not-triggered", "trigger_reason": "positive control" },
      "cuda":     { "status": "not-triggered", "trigger_reason": "positive control" },
      "crux":     { "status": "not-triggered", "trigger_reason": "positive control" },
      "mutation": { "status": "not-triggered", "trigger_reason": "positive control" }
    },
    "findings_ref": { "path": "findings.sarif", "sha256": "$fsha" },
    $costblock
  }
}
JSON
  minisign -q -S -s "$KEY" -m "$d/receipt.intoto.jsonl" \
    -t "PRREV-003 S6.1 positive control: $sub" -c "TEST ONLY fixture signature" </dev/null
  minisign -q -V -m "$d/receipt.intoto.jsonl" -p "$PC/positive-control.pub" >/dev/null \
    || { echo "self-check failed: positive control $sub does not verify" >&2; exit 1; }
  echo "built positive-control/$sub"
}

# B2: the reviewer is the author.
pc_emit self-review     agent:positive-control agent:positive-control ok    ok
# B1: the receipt points at findings that are not the findings on disk.
pc_emit findings-digest agent:pc-author        agent:pc-reviewer      wrong ok
# B1: no cost block. Record-only is not unenforced.
pc_emit cost-missing    agent:pc-author        agent:pc-reviewer      ok    missing

echo "all fixtures rebuilt"
