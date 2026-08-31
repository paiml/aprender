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
C1=$(sha_of C1)   # the merge base of every fixture PR
C3=$(sha_of C3)   # main's tip: NOT an ancestor of F1, so a stale index
F1=$(sha_of F1)   # the GPU pull request head, adds src/cuda/kernel.cu
D1=$(sha_of D1)   # the docs-only pull request head
G1=$(sha_of G1)   # the claim head: PUBLISHES 2.93x Ollama in book/
S1=$(sha_of S1)   # the code head: adds a plain .rs file, so S3.D triggers
P1=$(sha_of P1)   # the printed head: the same ratio in a comment AND in a format!
E1=$(sha_of E1)   # the examples head: the SAME ratio, under book/src/examples/ (F6)
for v in C1 C3 F1 D1 G1 S1 P1 E1; do
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
#         [<author-id>] [<reviewer-id>] [<antigravity-json>] [<skill-version>]
#
# ARG 10 IS THE EMPTY STRING TO OMIT THE ANTIGRAVITY KEY ENTIRELY, which is what row 27
# needs: a 2.0.0 receipt written before S3.E existed, which the guard must still accept.
# Arg 11 is the declared skill version, and it is what decides whether the arm is owed.
receipt() {
  local head=$1 base=$2 verdict=$3 pmat=$4 cuda=$5 crux=$6 mut=$7
  local author=${8:-$AUTHOR} reviewer=${9:-$REVIEWER}
  local ag=${10-$AG_OK} sv=${11:-2.1.0}
  local agline=""
  [ -z "$ag" ] || agline=",
      \"antigravity\": $ag"
  cat <<JSON
{
  "_type": "https://in-toto.io/Statement/v1",
  "subject": [
    { "name": "git+https://github.com/paiml/aprender",
      "digest": { "sha1": "$head" } }
  ],
  "predicateType": "https://paiml.dev/attestations/pr-review/v2",
  "predicate": {
    "skill_version": "$sv",
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
      "mutation": $mut$agline
    },
    "findings_ref": { "path": "findings.sarif", "sha256": "@@FINDINGS_SHA@@" },
    "cost": { "input_tokens": 41200, "output_tokens": 3180, "wall_seconds": 74 }
  }
}
JSON
}

# --- S3.E, the second-vendor arm (the FIFTH consultation) -------------------------------------------
#
# THE NUMBERS BELOW WERE MEASURED, NOT INVENTED. One real print-mode invocation of the
# binary `command -v agy` resolved to on the development box, 2026-08-31:
#
#   agy -p '<prompt>' --output-format json --json-schema <schema>
#   -> rc 0, wall 13 s, duration_seconds 9.522230891, status SUCCESS,
#      usage { input_tokens 20836, output_tokens 904, thinking_tokens 752,
#              cache_read_tokens 8142, total_tokens 21740 }
#
# The usage block is what S8's cost_per_actionable is fed from, so a fixture carrying a
# made-up one would be a fixture that cannot detect a made-up one.
AGY_VERSION='agy 1.1.22'
AGY_PATH='/home/noah/.local/bin/agy'
AG_USAGE='"usage":{"input_tokens":20836,"output_tokens":904,"thinking_tokens":752,"cache_read_tokens":8142,"total_tokens":21740}'

# THE MODEL ID IS THE ARM'S CORRECTNESS PROPERTY, NOT A LABEL. `agy models` on the same
# box the same day printed fourteen ids, and TWO ARE CLAUDE - claude-sonnet-4-6 and
# claude-opus-4-6-thinking. agy is a HARNESS, not a model: run with a Claude model, or
# with --model omitted and the default landing there, S3.E is the primary reviewer's own
# family reviewing itself while every field a reader checks still reads `antigravity`.
# That is S5's self-preference case with a cross-vendor label on it, and it is this
# repository's standing rule against labelling a run by intent.
AGY_MODEL='gemini-3.1-pro-high'
AG_IDENT="\"agy_version\":\"$AGY_VERSION\",\"binary_path\":\"$AGY_PATH\",\"model_id\":\"$AGY_MODEL\",\"model_family\":\"google/gemini\""

# The ordinary shape: agy ran, agreed with the primary about everything, raised nothing.
# `divergence` is all zeros AND `findings` is empty, which the guard checks against each
# other - four zeros beside twelve findings would otherwise read as perfect agreement.
# THE AVAILABILITY TEST'S OWN RESULT, RECORDED. `exit_code: 0` is in this block and it
# is NOT what decides the three-state, because rc 0 was MEASURED to be worthless here:
# running S3.E step 3's documented invocation verbatim on 2026-08-31, a tool needed the
# `command` permission, headless print mode auto-denied it, and agy reported rc 0,
# `.status "SUCCESS"`, an empty `.response` and no `.structured_output` at all. The
# three booleans are S3.E.4's three conjuncts, and a `consulted` receipt must carry them
# all true - which is what row 36 breaks and row 37 records honestly.
AG_OUT_OK='"output_check":{"structured_output_present":true,"reviewed":true,"schema_valid":true}'

AG_OK="{\"status\":\"consulted\",\"attempted\":1,$AG_IDENT,\"exit_code\":0,\"duration_seconds\":9.52,$AG_USAGE,$AG_OUT_OK,\"reverified_by_primary\":false,\"divergence\":{\"agreed\":0,\"agy_only\":0,\"primary_only\":0,\"contradicted\":0},\"findings\":[]}"

# agy raised one thing the primary did not. The ledger says so: agy_only = 1, and the
# findings array holds exactly that one. This is the shape rows 32 and 33 both use, so
# the only variable between them is the finding's precision_class.
AG_ONE_FINDING="{\"status\":\"consulted\",\"attempted\":1,$AG_IDENT,\"exit_code\":0,\"duration_seconds\":31.7,$AG_USAGE,$AG_OUT_OK,\"reverified_by_primary\":false,\"divergence\":{\"agreed\":0,\"agy_only\":1,\"primary_only\":2,\"contradicted\":0},\"findings\":[{\"id\":\"agy-1\",\"grounding\":\"measured\",\"summary\":\"the added retry loop has no bound\"}]}"

# S3.E's `unavailable` state. agy fails SLOWLY as readily as it fails fast:
# --print-timeout defaults to 5m and a repository-scale review needs more, so the honest
# record of a timeout is unreachable - never a run that found nothing.
AG_UNREACHABLE='{"status":"unreachable","trigger_reason":"agy exceeded --print-timeout 900s with no structured output; a timeout is unavailable, not a review that found nothing"}'

# Illegal, and row 34 exists to be rejected for carrying it: S3.E is unconditional.
AG_NT_ILLEGAL='{"status":"not-triggered","trigger_reason":"docs-only diff, nothing worth a second opinion"}'

# The vacuous shape S8 fixes at zero, in the fifth arm: recorded as performed, having
# invoked nothing. Same defect as mutation.attempted=0 and cuda.queries=[].
AG_VACUOUS="{\"status\":\"consulted\",\"attempted\":0,$AG_IDENT,\"exit_code\":0,\"duration_seconds\":0,$AG_USAGE,$AG_OUT_OK,\"reverified_by_primary\":false,\"divergence\":{\"agreed\":0,\"agy_only\":0,\"primary_only\":0,\"contradicted\":0},\"findings\":[]}"

# ROW 36's SHAPE, AND IT IS A TRANSCRIPT RATHER THAN AN INVENTION. Every field here was
# taken from a real run of S3.E step 3's documented invocation on 2026-08-31: rc 0,
# `.status "SUCCESS"`, `.response ""`, `num_turns 1`, usage really spent (21237 in / 544
# out / 21781 total), duration 6.700545979 - and NO `.structured_output`. The run
# consumed tokens, exited clean and called itself a success while reviewing nothing. A
# receipt that reads `exit_code: 0` off that and writes `consulted` is the whole defect.
AG_NO_OUTPUT="{\"status\":\"consulted\",\"attempted\":1,$AG_IDENT,\"exit_code\":0,\"duration_seconds\":6.7,\"agy_status\":\"SUCCESS\",\"usage\":{\"input_tokens\":21237,\"output_tokens\":544,\"thinking_tokens\":420,\"cache_read_tokens\":8143,\"total_tokens\":21781},\"output_check\":{\"structured_output_present\":false,\"reviewed\":false,\"schema_valid\":false},\"reverified_by_primary\":false,\"divergence\":{\"agreed\":0,\"agy_only\":0,\"primary_only\":0,\"contradicted\":0},\"findings\":[]}"

# ROW 37 - THE SAME RUN, RECORDED HONESTLY. Identical measurements, identical rc 0,
# identical `.status "SUCCESS"` kept as the diagnostic S3.E.4 says it is - and
# `unreachable`, because the artifact was absent. This is not row 29 with a new
# `trigger_reason`: row 29's agy TIMED OUT, which is the failure mode a reader expects
# to look like a failure. This one exited 0 and said SUCCESS, which is the failure mode
# that looks like a clean review, and it is the one that shipped.
AG_UNREACHABLE_DENIED='{"status":"unreachable","exit_code":0,"agy_status":"SUCCESS","duration_seconds":6.7,"trigger_reason":"agy exited 0 with .status SUCCESS, an empty .response and no .structured_output: a tool required the command permission that headless print mode cannot prompt for and it was auto-denied. rc 0 is not a review, so this is unavailable, not a run that found nothing."}'

# S3.A duplication coverage (PRREV-009 / backtest F4). Every surface carries a method,
# because a surface with NO entry is the silently-absent coverage that made
# `duplication_hits: []` read as "nothing like this exists" on a diff that was 48.8%
# shell, python and yaml - none of which pmat's semantic index can see.
#
# total/scanned are 0/0 here and that is CORRECT rather than convenient: the fixture
# repository has one remote ref (origin/main) and no unmerged sibling, so the honest
# denominator is zero. The horizon rules that need a non-zero denominator are exercised
# by the branch probes in tests/pr-review.bats, not by these rows.
# F7: the horizon names all THREE regions - head, siblings, merge_base_to_main - and
# names them whether or not each was reached, because a region absent from the horizon
# cannot be told apart from a region that was searched and held nothing. Whether each was
# SEARCHED is the coverage map's job, one field down.
DUP_FULL="\"duplication_coverage\":{\"rust\":\"semantic\",\"shell\":\"lexical\",\"python\":\"lexical\",\"config\":\"lexical\",\"docs\":\"lexical\",\"other\":\"lexical\",\"sibling_branches\":\"lexical\",\"merge_base_to_main\":\"lexical\"},\"duplication_horizon\":[\"head=HEAD\",\"siblings=refs/remotes/origin/* unmerged into origin/main\",\"merge_base_to_main=$C1..refs/remotes/origin/main\"],\"horizon_branches_total\":0,\"horizon_branches_scanned\":0,\"symbols_searched\":4"
# The same run with the shell surface unreachable - e.g. git grep unavailable. Rows 16
# and 17 are the same coverage under two different verdicts.
DUP_SHELL_NONE="\"duplication_coverage\":{\"rust\":\"semantic\",\"shell\":\"none\",\"python\":\"none\",\"config\":\"none\",\"docs\":\"none\",\"other\":\"none\",\"sibling_branches\":\"lexical\",\"merge_base_to_main\":\"lexical\"},\"duplication_horizon\":[\"head=HEAD\",\"siblings=refs/remotes/origin/* unmerged into origin/main\",\"merge_base_to_main=$C1..refs/remotes/origin/main\"],\"horizon_branches_total\":0,\"horizon_branches_scanned\":0,\"symbols_searched\":4"

PMAT_OK="{\"status\":\"consulted\",\"index_commit\":\"$C1\",\"index_is_ancestor\":true,\"complexity_delta\":[],\"tdg_delta\":[],\"satd_introduced\":[],\"duplication_hits\":[],\"cache_hits\":0,$DUP_FULL}"
PMAT_STALE="{\"status\":\"consulted\",\"index_commit\":\"$C3\",\"index_is_ancestor\":false,\"complexity_delta\":[],\"tdg_delta\":[],\"satd_introduced\":[],\"duplication_hits\":[],\"cache_hits\":0,$DUP_FULL}"
PMAT_SHELL_BLIND="{\"status\":\"consulted\",\"index_commit\":\"$C1\",\"index_is_ancestor\":true,\"complexity_delta\":[],\"tdg_delta\":[],\"satd_introduced\":[],\"duplication_hits\":[],\"cache_hits\":0,$DUP_SHELL_NONE}"
PMAT_UNREACHABLE='{"status":"unreachable","trigger_reason":"pmat MCP server: ConnectionRefused"}'
# NOT a PMAT_NT. The previous revision of this file carried one, used by row 7, whose
# own trigger_reason read "pmat is unconditional; not-triggered is never correct for it"
# - a fixture that stated the rule it exempted, and the guard accepted it. S3.A's
# trigger is unconditional, so `not-triggered` is never a legal pmat status and the only
# fixture that may carry it is row 19, which exists to be REJECTED for carrying it.
PMAT_NT_ILLEGAL='{"status":"not-triggered","trigger_reason":"docs-only, nothing worth indexing"}'

CUDA_OK="{\"status\":\"consulted\",\"trigger_reason\":\"diff touches src/cuda/kernel.cu\",\"queries\":[{\"q\":\"stream ordering guarantees between non-default streams\",\"excerpt_sha256\":\"$EXCERPT_SHA\",\"result\":\"found\"}]}"
CUDA_NT='{"status":"not-triggered","trigger_reason":"no changed path and no commit message matches the S3.B trigger"}'
CUDA_NO_QUERIES='{"status":"consulted","trigger_reason":"diff touches src/cuda/kernel.cu","queries":[]}'

CRUX_NT='{"status":"not-triggered","trigger_reason":"no CLI flag, HTTP route, MCP tool, config key or output format changed","surfaces":[],"contracts":[],"comparative_claims":[]}'
CRUX_OK_FULL="{\"status\":\"consulted\",\"surfaces\":[\"apr bench --gpu\"],\"contracts\":[\"CRUX-A-08\"],\"gap_effect\":\"closes\",\"crux_coverage\":\"covered\",\"comparative_claims\":[{\"claim\":\"1.21x llama.cpp on aarch64 Q4_K\",\"comparator\":{\"command\":[\"llama-cli\",\"-m\",\"qwen2.5-1.5b-q4_k_m.gguf\",\"-p\",\"2+2?\",\"-n\",\"128\",\"-ngl\",\"99\"],\"version\":\"llama.cpp b4021\",\"env_sha256\":\"9f2b1c4d5e6a7b8c9d0e1f2a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e\",\"artifact_sha256\":\"1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f809\",\"log_path\":\"evidence/bench/2026-08-30-gb10/comparator.log\"}}]}"
# The claim head's two shapes, which differ ONLY in whether the published ratio was
# recorded. They are B4's discrimination pair, and rows 16 and 17 are the fixtures.
CRUX_SAW_SURFACE_NO_CLAIM='{"status":"consulted","surfaces":["book/src/tools/apr-cli.md"],"contracts":[],"gap_effect":"none","crux_coverage":"none","comparative_claims":[]}'
CRUX_SAW_BANNER_NO_CLAIM='{"status":"consulted","surfaces":["apr banner output"],"contracts":[],"gap_effect":"none","crux_coverage":"none","comparative_claims":[]}'
CRUX_CLAIM_RECORDED="{\"status\":\"consulted\",\"surfaces\":[\"book/src/tools/apr-cli.md\"],\"contracts\":[\"CRUX-A-08\"],\"gap_effect\":\"none\",\"crux_coverage\":\"none\",\"comparative_claims\":[{\"claim\":\"2.93x Ollama on 1.5B Q4_K decode\",\"comparator\":{\"command\":[\"ollama\",\"run\",\"qwen2.5-coder:1.5b\",\"--verbose\"],\"version\":\"ollama 0.5.7\",\"env_sha256\":\"3c1d9e0f2a4b6c8d0e1f2a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f\",\"artifact_sha256\":\"5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d\",\"log_path\":\"evidence/bench/2026-08-30-ollama/comparator.log\"}}]}"
# F6's pair. Identical to CRUX_SAW_SURFACE_NO_CLAIM / CRUX_CLAIM_RECORDED except for the
# PATH: book/src/examples/ rather than book/src/tools/. That one directory name is the
# whole of F6 - */examples/* was a Rust cargo-target exclusion applied to prose, and it
# removed 153 of the book's 441 published pages, including the one da069a25f published
# `851.8 tok/s = 2.93x Ollama` to.
CRUX_SAW_EXAMPLES_NO_CLAIM='{"status":"consulted","surfaces":["book/src/examples/showcase-benchmark.md"],"contracts":[],"gap_effect":"none","crux_coverage":"none","comparative_claims":[]}'
CRUX_EXAMPLES_CLAIM_RECORDED="{\"status\":\"consulted\",\"surfaces\":[\"book/src/examples/showcase-benchmark.md\"],\"contracts\":[\"CRUX-A-08\"],\"gap_effect\":\"none\",\"crux_coverage\":\"none\",\"comparative_claims\":[{\"claim\":\"2.93x Ollama on 1.5B Q4_K decode\",\"comparator\":{\"command\":[\"ollama\",\"run\",\"qwen2.5-coder:1.5b\",\"--verbose\"],\"version\":\"ollama 0.5.7\",\"env_sha256\":\"3c1d9e0f2a4b6c8d0e1f2a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f\",\"artifact_sha256\":\"5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d\",\"log_path\":\"evidence/bench/2026-08-30-ollama/comparator.log\"}}]}"
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
# ROW 7 - honest docs-only PR, pmat consulted, the rest not-triggered -> GREEN
# DISCRIMINATION CASE.
#
# S6.3 writes this row as "all consultations not-triggered". S3.A and S8.4 both say
# pmat's trigger is UNCONDITIONAL, so the spec contradicts itself here and the two
# normative statements win over the illustrative row. This fixture previously carried
# `pmat: not-triggered` with the trigger_reason "pmat is unconditional; not-triggered
# is never correct for it" - it BLESSED the rule it stated, and the guard accepted it.
# It now carries a consulted pmat whose four arrays are empty, which is the honest
# shape of "ran and found nothing" and keeps the row discriminating: a docs-only PR
# must still be ACCEPTED.
# ===========================================================================
SARIF=$(
sarif_empty
)
RCPT=$(receipt "$D1" "$C1" PASS "$PMAT_OK" "$CUDA_NT" "$CRUX_NT" "$MUT_NT")
emit row-07-honest-docs-only-pmat-consulted "$SARIF" "$RCPT"

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
RCPT=$(receipt "$D1" "$C1" PASS "$PMAT_OK" "$CUDA_NT" "$CRUX_NT" "$MUT_NT")
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
# ROW 16 - the diff PUBLISHES a competitor ratio, nothing records it  -> RED B4
#
# NOT one of S6.3's fourteen. B4's blocking half read only the receipt, so a head that
# publishes `2.93x Ollama` in book/ was ACCEPTED whenever the reviewer stayed silent
# about it, and REJECTED only when the reviewer wrote the ratio into a finding: the
# same diff, the verdict turning on candour. The crux consultation here is HONEST in
# every other respect - it names the surface it looked at - so nothing but the diff
# recomputation can reject this receipt.
# ===========================================================================
SARIF=$(
sarif_empty
)
RCPT=$(receipt "$G1" "$C1" PASS "$PMAT_OK" "$CUDA_NT" "$CRUX_SAW_SURFACE_NO_CLAIM" "$MUT_NT")
emit row-16-comparative-claim-only-in-the-diff "$SARIF" "$RCPT"

# ===========================================================================
# ROW 17 - the same diff, the same ratio, RECORDED with a comparator   -> GREEN
# DISCRIMINATION CASE for row 16. Without it, "reject every PR that mentions a
# competitor" reads green, and B4 would block the honest path it exists to require.
# ===========================================================================
SARIF=$(
sarif_one crux '{
      "ruleId": "comparative_claim", "level": "note",
      "message": { "text": "book/src/tools/apr-cli.md publishes 2.93x Ollama; the comparator command, version, environment and artifact hash are recorded." },
      "properties": { "grounding": "measured",
        "source": "evidence/bench/2026-08-30-ollama/comparator.log",
        "failure_scenario": "A reader reproduces the published ratio and measures parity, because the harness never executed the comparator.",
        "precision_class": "advisory" } }'
)
RCPT=$(receipt "$G1" "$C1" FINDINGS "$PMAT_OK" "$CUDA_NT" "$CRUX_CLAIM_RECORDED" "$MUT_NT")
emit row-17-comparative-claim-recorded "$SARIF" "$RCPT"

# ===========================================================================
# ROW 18 - cuda consulted, queries: []                                -> RED B1
#
# The vacuous-consultation shape the guard already rejected for mutation
# (`attempted: 0`, row 2) and did not for cuda. S8 sets vacuous_consultations = 0 as
# one of its four zeros; without this row that zero was enforced for one consultation
# out of four.
# ===========================================================================
SARIF=$(
sarif_empty
)
RCPT=$(receipt "$F1" "$C1" PASS "$PMAT_OK" "$CUDA_NO_QUERIES" "$CRUX_NT" "$MUT_OK")
emit row-18-cuda-consulted-no-queries "$SARIF" "$RCPT"

# ===========================================================================
# ROW 19 - pmat not-triggered on a diff that changes Rust source      -> RED B1
#
# S3.A: "Trigger: unconditional". Every other consultation here is honest, so this
# receipt is rejected for exactly one reason: the review skipped the consultation that
# carries duplication_hits, which S3.A calls the highest-EV field in the receipt.
# ===========================================================================
SARIF=$(
sarif_empty
)
RCPT=$(receipt "$S1" "$C1" PASS "$PMAT_NT_ILLEGAL" "$CUDA_NT" "$CRUX_NT" "$MUT_OK")
emit row-19-pmat-not-triggered-on-a-code-diff "$SARIF" "$RCPT"

# ===========================================================================
# ROW 20 - mutation not-triggered on a diff that changes Rust source  -> RED B1
#
# S3.D's rows 1 and 2 differ in whether the RESULT blocks, not in whether the
# consultation is owed. Before the trigger was recomputed, `not-triggered` was accepted
# on any diff at all: the one consultation whose emptiness WAS checked could be skipped
# outright by writing three words, which is the same hole one level up.
# ===========================================================================
SARIF=$(
sarif_empty
)
RCPT=$(receipt "$S1" "$C1" PASS "$PMAT_OK" "$CUDA_NT" "$CRUX_NT" "$MUT_NT")
emit row-20-mutation-not-triggered-on-a-code-diff "$SARIF" "$RCPT"

# ===========================================================================
# ROW 21 - crux not-triggered on a diff publishing a competitor ratio -> RED B1
#
# S3.C.1 lives under S3.C, so a comparative claim in the diff IS a crux trigger. This
# is the route B4's diff half depends on: without it a reviewer could put crux beyond
# reach of every claim rule by declaring the consultation untriggered.
# ===========================================================================
SARIF=$(
sarif_empty
)
RCPT=$(receipt "$G1" "$C1" PASS "$PMAT_OK" "$CUDA_NT" "$CRUX_NT" "$MUT_NT")
emit row-21-crux-not-triggered-on-a-claim-diff "$SARIF" "$RCPT"

# ===========================================================================
# ROW 22 - a ratio inside format!, and the SAME ratio in a plain comment -> RED B4
#
# The scope discrimination. B4's diff half must fire on the printed literal and must NOT
# fire on the comment two lines above it, which quotes the withdrawn claim in order to
# name it. The comment is FIRST in the file, so the rejection reason naming the format!
# line is the proof that the comment was skipped: if the comment fired, it would be the
# one quoted.
#
# Measured over 300 commits of origin/main, every comparative claim this repository has
# added to a plain `//` comment was a claim it was WITHDRAWING, and a block on those has
# no honest exit - S3.C.1's remedy is a comparator log, and there is no log for a number
# nobody measured.
# ===========================================================================
SARIF=$(
sarif_empty
)
RCPT=$(receipt "$P1" "$C1" PASS "$PMAT_OK" "$CUDA_NT" "$CRUX_SAW_BANNER_NO_CLAIM" "$MUT_OK")
emit row-22-printed-ratio-not-the-quoted-one "$SARIF" "$RCPT"

# ===========================================================================
# ROW 23 - a duplication surface that could not be searched, verdict PASS -> RED B1
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
emit row-23-duplication-surface-unsearched-verdict-pass "$SARIF" "$RCPT"

# ===========================================================================
# ROW 24 - the SAME unsearched surface, verdict DEGRADED             -> GREEN
# DISCRIMINATION CASE for row 23. Without it, "reject every receipt that admits a gap" reads
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
emit row-24-duplication-surface-unsearched-verdict-degraded "$SARIF" "$RCPT"

# ===========================================================================
# ROW 25 - the SAME ratio, published under book/src/examples/          -> RED B4
#
# F6. Row 16 is this fixture with one directory changed, and until F6 the two verdicts
# DIVERGED: row 16 RED, this one ACCEPTED. `match_shipped_surface` excluded */examples/*,
# a rule scoped from a cargo target layout, and applied to the book it removed
# book/src/examples/ - 153 of 441 published pages, 34.7%, all of them in SUMMARY.md.
# da069a25f published `851.8 tok/s = 2.93x Ollama` into precisely that directory and B4
# fired ZERO times on it: the exact publication S3.C.1, S9 and S11 are written about,
# accepted by the gate that names it.
#
# Without this row the case table goes green on a guard that cannot see a third of the
# book - the guard-universe defect, whose seventh instance this is.
# ===========================================================================
SARIF=$(
sarif_empty
)
RCPT=$(receipt "$E1" "$C1" PASS "$PMAT_OK" "$CUDA_NT" "$CRUX_SAW_EXAMPLES_NO_CLAIM" "$MUT_NT")
emit row-25-examples-page-publishes-a-ratio "$SARIF" "$RCPT"

# ===========================================================================
# ROW 26 - the same book/src/examples/ page, ratio RECORDED             -> GREEN
# DISCRIMINATION CASE for row 25, and it is not decorative: "block every book page under
# examples/" would read green on row 25 alone, and F6 would have widened the scope into
# a rule with no honest exit. The exit is the same one row 17 demonstrates - record the
# comparator - and this proves it is still open one directory over.
# ===========================================================================
SARIF=$(
sarif_one crux '{
      "ruleId": "comparative_claim", "level": "note",
      "message": { "text": "book/src/examples/showcase-benchmark.md publishes 2.93x Ollama; the comparator command, version, environment and artifact hash are recorded." },
      "properties": { "grounding": "measured",
        "source": "evidence/bench/2026-08-30-ollama/comparator.log",
        "failure_scenario": "A reader reproduces the published ratio and measures parity, because the harness never executed the comparator.",
        "precision_class": "advisory" } }'
)
RCPT=$(receipt "$E1" "$C1" FINDINGS "$PMAT_OK" "$CUDA_NT" "$CRUX_EXAMPLES_CLAIM_RECORDED" "$MUT_NT")
emit row-26-examples-page-ratio-recorded "$SARIF" "$RCPT"

# ===========================================================================
# S3.E - THE FOURTH-VENDOR ARM. Rows 27..34.
#
# Every row below sits on D1, the docs-only head, with pmat consulted and the other
# three honestly not-triggered - the shape row 7 already proves is ACCEPTED. So the only
# variable in each is the antigravity block or the declared skill version, and a row that
# goes red goes red for its own reason and no other.
# ===========================================================================

# sarif_antigravity <precision_class> - one agy-origin result.
# The two calls differ in ONE token. Everything else - the driver name that routes the
# result to S3.E's rule, the grounding mark, the failure scenario - is byte-identical.
#
# THE GROUNDING MARK IS `measured`, AND THAT IS NOT DECORATION. The first draft wrote
# `asserted`, and row 32 was then rejected by S1's OLDER rule - "an asserted claim never
# blocks" - which fires two hundred lines before S3.E's. The row passed, exited 1, named
# B1, and pinned NOTHING: drop S3.E's rule and row 32 stays red on S1's. A fixture that
# is rejected for a reason other than the one it exists to test is mislabeled evidence,
# which is why assert_row asserts the REASON and not only the class.
#
# `measured` is also the honest mark for this arm. agy runs as its own process with its
# own tools, so the interesting agy finding is precisely the one it MEASURED - and S1's
# asserted rule, which happens to cover the uninteresting case, covers none of them.
sarif_antigravity() {
  cat <<JSON
{ "\$schema": "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [ { "tool": { "driver": { "name": "antigravity" } },
              "invocations": [ { "executionSuccessful": true, "toolExecutionNotifications": [] } ],
              "results": [ {
                "ruleId": "agy-1", "level": "warning",
                "message": { "text": "The retry loop added in this diff has no bound; a permanently failing dependency spins forever." },
                "properties": { "grounding": "measured",
                  "source": "antigravity (agy 1.1.22), independent review",
                  "failure_scenario": "A dependency that fails every time is retried without limit, and the request never returns rather than returning an error.",
                  "precision_class": "$1" } } ] } ] }
JSON
}

# ===========================================================================
# ROW 27 - a 2.0.0 receipt, written before S3.E existed, no arm at all -> GREEN
#
# DISCRIMINATION CASE, and the one that keeps the version gate honest in BOTH
# directions. Without it, "require the antigravity block on every receipt ever written"
# reads green - and the only way to make this repository's one real receipt
# (evidence/pr-review/2795/f5fe1479.../, skill_version 2.0.0, four consultations) pass
# again would be to back-fill an antigravity block describing a consultation nobody
# performed. That is the never-ran-Ollama shape with a JSON schema in front of it, and
# S3.C.1 exists to make it impossible, not to make it convenient.
#
# It also kills the `arm-e-always-required` mutant: with the version gate forced true,
# this row is REJECTED and the row goes red.
# ===========================================================================
SARIF=$(
sarif_empty
)
RCPT=$(receipt "$D1" "$C1" PASS "$PMAT_OK" "$CUDA_NT" "$CRUX_NT" "$MUT_NT" \
      "$AUTHOR" "$REVIEWER" "" "2.0.0")
emit row-27-legacy-2-0-0-receipt-has-no-arm-e "$SARIF" "$RCPT"

# ===========================================================================
# ROW 28 - a 2.1.0 receipt that owes the arm and omits it            -> RED B1
#
# The other half of row 27, one field changed. S3.E's trigger is unconditional, so at
# 2.1.0 an absent block is not "not applicable" - it is the consultation missing, and
# S3.0's whole subject is that an absent record and an empty one must not be the same
# artifact. Kills `arm-e-never-required`.
# ===========================================================================
SARIF=$(
sarif_empty
)
RCPT=$(receipt "$D1" "$C1" PASS "$PMAT_OK" "$CUDA_NT" "$CRUX_NT" "$MUT_NT" \
      "$AUTHOR" "$REVIEWER" "" "2.1.0")
emit row-28-arm-e-owed-and-absent "$SARIF" "$RCPT"

# ===========================================================================
# ROW 29 - agy UNAVAILABLE, verdict PASS                             -> RED B1
#
# The brief's rule and S3.0's row 3, in the fifth arm: a missing or failing agy must
# surface as `unavailable` and DEGRADED, never as a silent pass. agy fails SLOWLY as
# readily as fast - --print-timeout defaults to 5m and a repository-scale review needs
# more - so the trigger_reason here is a TIMEOUT, which is the failure mode most likely
# to be mistaken for "it ran and found nothing".
#
# Rows 5 and 6 are this same rule for pmat; this row is not a copy of the check but of
# the CASE, because the guard applies one rule over a list and antigravity joins the
# list. That is what makes the `antigravity-dropped-from-the-consultation-list` mutant
# killable: with antigravity out of the list, this receipt is ACCEPTED.
# ===========================================================================
SARIF=$(
cat <<'JSON'
{ "$schema": "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [ { "tool": { "driver": { "name": "antigravity" } },
              "invocations": [ { "executionSuccessful": false,
                "toolExecutionNotifications": [ { "level": "error",
                  "message": { "text": "agy exceeded --print-timeout 900s with no structured output. Consultation could not be performed; verdict DEGRADED." } } ] } ],
              "results": [] } ] }
JSON

)
RCPT=$(receipt "$D1" "$C1" PASS "$PMAT_OK" "$CUDA_NT" "$CRUX_NT" "$MUT_NT" \
      "$AUTHOR" "$REVIEWER" "$AG_UNREACHABLE")
emit row-29-arm-e-unavailable-verdict-pass "$SARIF" "$RCPT"

# ===========================================================================
# ROW 30 - the SAME unavailable agy, verdict DEGRADED                 -> GREEN
#
# DISCRIMINATION CASE for row 29. Without it, "refuse every receipt whose agy did not
# run" reads green, and the arm would punish the honest DEGRADED exactly as hard as the
# silent PASS - which is how an unavailability field learns to stay empty. S3.E is
# advisory and DEGRADED proceeds on a feature branch (S7), so this is not merely
# tolerated: it is the arm's intended behaviour on a box where agy is not installed.
# ===========================================================================
SARIF=$(
cat <<'JSON'
{ "$schema": "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [ { "tool": { "driver": { "name": "antigravity" } },
              "invocations": [ { "executionSuccessful": false,
                "toolExecutionNotifications": [ { "level": "error",
                  "message": { "text": "agy exceeded --print-timeout 900s with no structured output. Consultation could not be performed; verdict DEGRADED." } } ] } ],
              "results": [] } ] }
JSON

)
RCPT=$(receipt "$D1" "$C1" DEGRADED "$PMAT_OK" "$CUDA_NT" "$CRUX_NT" "$MUT_NT" \
      "$AUTHOR" "$REVIEWER" "$AG_UNREACHABLE")
emit row-30-arm-e-unavailable-verdict-degraded "$SARIF" "$RCPT"

# ===========================================================================
# ROW 31 - agy consulted having invoked nothing                      -> RED B1
#
# The vacuous consultation the brief names, and S8's fourth zero applied to the fifth
# arm. `status: consulted` with `attempted: 0` is the same artifact as
# `mutation.attempted: 0` (row 2) and `cuda.queries: []` (row 18): a consultation
# recorded as performed that performed nothing, which is the shape `pv lint <FILE>`
# returning PASS over zero contracts already taught this repository to refuse.
#
# Every other field here is well-formed - version, path, model family, usage,
# divergence, the empty findings array - so nothing but the count can reject it.
# ===========================================================================
SARIF=$(
sarif_empty
)
RCPT=$(receipt "$D1" "$C1" PASS "$PMAT_OK" "$CUDA_NT" "$CRUX_NT" "$MUT_NT" \
      "$AUTHOR" "$REVIEWER" "$AG_VACUOUS")
emit row-31-arm-e-consulted-attempted-zero "$SARIF" "$RCPT"

# ===========================================================================
# ROW 32 - an agy finding claiming a BLOCKING class                  -> RED B1
#
# S7's admission rule says a class may block only while its measured precision on the
# rolling sample is >= 90%, and S8 says instrument -> 30 samples -> ratchet. S3.E has
# ZERO samples. So the arm cannot be in the blocking tier yet, and a receipt that puts
# it there is claiming an authority no measurement supports - internally inconsistent,
# B1, not a new blocking class of its own.
#
# The receipt is otherwise perfectly honest: the ledger balances (agy_only 1, one
# finding), the identity fields are recorded, the usage block is real.
# ===========================================================================
SARIF=$(
sarif_antigravity blocking
)
RCPT=$(receipt "$D1" "$C1" FINDINGS "$PMAT_OK" "$CUDA_NT" "$CRUX_NT" "$MUT_NT" \
      "$AUTHOR" "$REVIEWER" "$AG_ONE_FINDING")
emit row-32-arm-e-finding-claims-a-blocking-class "$SARIF" "$RCPT"

# ===========================================================================
# ROW 33 - the SAME agy finding, marked advisory                      -> GREEN
#
# DISCRIMINATION CASE for row 32, and it is not decorative. Without it, "refuse every
# receipt that carries an agy finding" reads green - and the arm would be a rule whose
# only satisfiable behaviour is to find nothing, which is the opposite of why a second
# vendor is being asked. One token differs from row 32.
# ===========================================================================
SARIF=$(
sarif_antigravity advisory
)
RCPT=$(receipt "$D1" "$C1" FINDINGS "$PMAT_OK" "$CUDA_NT" "$CRUX_NT" "$MUT_NT" \
      "$AUTHOR" "$REVIEWER" "$AG_ONE_FINDING")
emit row-33-arm-e-finding-advisory "$SARIF" "$RCPT"

# ===========================================================================
# ROW 34 - agy declared not-triggered                                -> RED B1
#
# S3.E's trigger is unconditional, for the same reason S3.A's is and not for a cost
# reason: a shape trigger exempts exactly the diffs where an independent reader is worth
# most - the small ones that look obvious, which is what every PR in S9's spine looked
# like to its author, all four of them carrying reviews=0 and comments=0.
#
# This is row 19's rule (pmat: not-triggered on a code diff) for the fifth arm, and it
# is stricter: pmat's illegality needed a code file in the diff, and S3.E's does not,
# because there is no diff shape a second opinion is not owed on. The head here is D1,
# the DOCS-ONLY one, which is the hardest case for that claim and therefore the right
# one to pin it with.
# ===========================================================================
SARIF=$(
sarif_empty
)
RCPT=$(receipt "$D1" "$C1" PASS "$PMAT_OK" "$CUDA_NT" "$CRUX_NT" "$MUT_NT" \
      "$AUTHOR" "$REVIEWER" "$AG_NT_ILLEGAL")
emit row-34-arm-e-not-triggered "$SARIF" "$RCPT"

# ===========================================================================
# ROW 35 - agy routed to the PRIMARY REVIEWER'S OWN MODEL FAMILY      -> RED B1
#
# THE ROW THE ARM WOULD BE WORTHLESS WITHOUT, and the one the design nearly shipped
# without, because agy's name says Antigravity and Antigravity is Google's. `agy models`
# prints fourteen ids and two of them are Claude. agy is a HARNESS, not a model.
#
# So a receipt can be honest in every other field - the binary resolved and recorded,
# the version recorded, the usage block real, the divergence ledger balanced, the arm
# marked advisory - and the consultation still be THE SAME MODEL FAMILY REVIEWING
# ITSELF. S5 cites Huang et al. (ICLR'24) for why that is worth close to nothing, and
# A5 calls a separate grounded critic the first configuration that beats single-agent;
# a same-family critic is not one.
#
# This fixture is row 7 with ONE token changed - the model id - which is what makes it
# evidence rather than illustration: nothing else about the receipt differs from a row
# the guard ACCEPTS.
#
# It is also the standing verification rule this repository writes down and keeps
# breaking: never label a run by intent, prove the mechanism ENGAGED. A harness printing
# `device: GPU` from a build with no CUDA in it produced three findings from CPU runs.
# `model_family: google/gemini` written beside `--model claude-opus-4-6-thinking` is the
# same artifact.
# ===========================================================================
AG_SAME_FAMILY="{\"status\":\"consulted\",\"attempted\":1,\"agy_version\":\"$AGY_VERSION\",\"binary_path\":\"$AGY_PATH\",\"model_id\":\"claude-opus-4-6-thinking\",\"model_family\":\"google/gemini\",\"exit_code\":0,\"duration_seconds\":9.52,$AG_USAGE,$AG_OUT_OK,\"reverified_by_primary\":false,\"divergence\":{\"agreed\":0,\"agy_only\":0,\"primary_only\":0,\"contradicted\":0},\"findings\":[]}"
SARIF=$(
sarif_empty
)
RCPT=$(receipt "$D1" "$C1" PASS "$PMAT_OK" "$CUDA_NT" "$CRUX_NT" "$MUT_NT" \
      "$AUTHOR" "$REVIEWER" "$AG_SAME_FAMILY")
emit row-35-arm-e-routed-to-the-same-model-family "$SARIF" "$RCPT"

# ===========================================================================
# ROW 36 - agy exited 0 and produced nothing, recorded `consulted`    -> RED B1
#
# THE DEFECT THIS ROW EXISTS FOR WAS SHIPPED IN THIS SKILL, not imagined for a test.
# S3.E step 3's documented invocation, run verbatim against a real checkout on
# 2026-08-31, returned rc 0 and NO structured output: headless print mode cannot prompt
# for a tool permission, so it auto-denied one and reported `.status "SUCCESS"` over an
# empty response. Read through the exit code - which is what "run it and check rc" means
# - that is a clean consultation. It is a review that never happened.
#
# A FAILURE THAT EXITS 0 IS THIS REPOSITORY'S RECURRING DEFECT: an EPIPE-inverted grep
# that PASSED a safety check on the error, a chown swallowing errors, a timeout naming
# no step, a `sed` draining its sibling's stream, a self-test satisfied by its own
# vacuity floor, five make targets claiming CI. Six in one session. The fifth arm is
# where it costs most, because a receipt that records a second vendor reviewed the diff
# is the one artifact nobody re-derives.
#
# Every other field is honest and generous - real usage, real duration, attempted 1,
# a resolved binary path, a Gemini model id, a balanced empty ledger - so nothing but
# `output_check` can reject it. That is deliberate: the row must fail on the new rule
# or it pins nothing, exactly as row 32 had to be rebuilt to stop passing on S1's.
# ===========================================================================
SARIF=$(
sarif_empty
)
RCPT=$(receipt "$D1" "$C1" PASS "$PMAT_OK" "$CUDA_NT" "$CRUX_NT" "$MUT_NT" \
      "$AUTHOR" "$REVIEWER" "$AG_NO_OUTPUT")
emit row-36-arm-e-consulted-with-no-usable-output "$SARIF" "$RCPT"

# ===========================================================================
# ROW 37 - THE SAME rc-0 RUN, recorded unreachable + DEGRADED       -> GREEN
#
# DISCRIMINATION CASE for row 36, and it carries a second job. Without it the new rule
# reads green as "refuse any receipt whose agy exited 0", which would refuse EVERY
# successful consultation - agy returns rc 0 when it works too. The variable between the
# two rows is not the exit code, the duration, the usage or the status agy printed: all
# four are identical. It is what the receipt CLAIMS about them.
#
# This is also the row that says what a reviewer on a box with a permission-denying agy
# should actually write. `unreachable` + DEGRADED is the intended behaviour, the same
# thing rows 29-30 say about a timeout and S3.0 says about pmat's ConnectionRefused, and
# an arm that punished this record as hard as row 36 would teach the field to stay empty.
# ===========================================================================
SARIF=$(
cat <<'JSON'
{ "$schema": "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [ { "tool": { "driver": { "name": "antigravity" } },
              "invocations": [ { "executionSuccessful": false,
                "toolExecutionNotifications": [ { "level": "error",
                  "message": { "text": "agy exited 0 with status SUCCESS and no structured_output: a tool permission was auto-denied in headless mode. Consultation could not be performed; verdict DEGRADED." } } ] } ],
              "results": [] } ] }
JSON

)
RCPT=$(receipt "$D1" "$C1" DEGRADED "$PMAT_OK" "$CUDA_NT" "$CRUX_NT" "$MUT_NT" \
      "$AUTHOR" "$REVIEWER" "$AG_UNREACHABLE_DENIED")
emit row-37-arm-e-rc-zero-recorded-unreachable "$SARIF" "$RCPT"

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
