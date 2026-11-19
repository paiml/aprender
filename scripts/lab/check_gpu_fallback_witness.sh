#!/usr/bin/env bash
# check_gpu_fallback_witness.sh -- LAB (nightly only; cannot block a merge or a release).
#
# On a host with a GPU, run one short generation per listed model and go red when the
# backend that answered was the CPU, or the CUDA fallback line appears. A silent CPU
# fallback (#7285: Qwen3-0.6B Q4_K_M, cosine 0.453, "falling back to CPU") is otherwise
# only a log line that no lane reads.
#
#   bash scripts/lab/check_gpu_fallback_witness.sh --apr "$APR" --models LIST [--expected FILE]
#   bash scripts/lab/check_gpu_fallback_witness.sh --self-test
#
# LIST     one GGUF path per line; blank lines and '#' lines ignored.
# EXPECTED one line per model known to fall back: "<gguf basename> <issue number>".
#          Default: evidence/models/expected-gpu-fallback.txt. An entry whose issue is
#          CLOSED is red, and so is an entry whose model now runs on the GPU
#          ("remove the entry"): the file can only be emptied by the defect being fixed.
#
# Exit: 0 every model admitted or expected-and-still-falling-back
#       1 red (fell back unexpectedly, stale expected entry)
#       2 usage error
#       3 not_measured (no GPU, missing model file, unreadable output, issue state
#         unreadable). Never green.
#
# Pin the binary: pass the path "$APR" from scripts/apr_bin.sh; a bare `apr` is refused.
# The apr invocation is `apr run MODEL -n 4 --json`; APR_RUN_EXTRA appends flags.
# This script is not wired into any workflow, PR path or guard tree.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_EXPECTED="${REPO_ROOT}/evidence/models/expected-gpu-fallback.txt"
ISSUE_REPO="${ISSUE_REPO:-paiml/aprender}"

# --- injectable seams (replaced by the self-test) ---------------------------------

gpu_probe() { nvidia-smi -L 2>/dev/null; }
file_exists() { [[ -f "$1" ]]; }
is_exec() { [[ -x "$1" ]]; }
read_text() { cat -- "$1"; }

# run_apr APR MODEL: prints stdout+stderr of a 4-token run; returns apr's status.
run_apr() {
  local apr="$1" model="$2" out rc=0
  # shellcheck disable=SC2086
  out="$("$apr" run "$model" -n 4 --json ${APR_RUN_EXTRA:-} 2>&1)" || rc=$?
  printf '%s\n' "$out"
  return "$rc"
}

# issue_state NUMBER: prints OPEN or CLOSED; non-zero when unreadable.
issue_state() {
  gh issue view "$1" -R "$ISSUE_REPO" --json state --jq .state
}

# --- pure logic -------------------------------------------------------------------

have_gpu() {
  local probe
  probe="$(gpu_probe)" || return 1
  [[ "$probe" =~ (^|$'\n')GPU\ [0-9]+: ]]
}

# classify_backend TEXT -> admitted | fell_back | unknown
classify_backend() {
  local text="$1"
  if [[ "$text" == *"GPU (CUDA) path rejected"* || "$text" == *"falling back to CPU"* ]]; then
    echo fell_back; return 0
  fi
  if [[ "$text" =~ \"fell_back\"[[:space:]]*:[[:space:]]*true ]] \
    || [[ "$text" =~ \"ran\"[[:space:]]*:[[:space:]]*\"cpu\" ]]; then
    echo fell_back; return 0
  fi
  if [[ "$text" =~ \"fell_back\"[[:space:]]*:[[:space:]]*false ]] \
    && [[ "$text" =~ \"ran\"[[:space:]]*:[[:space:]]*\"gpu\" ]]; then
    echo admitted; return 0
  fi
  echo unknown
}

# verdict BACKEND EXPECTED_ISSUE ISSUE_STATE -> GREEN | EXPECTED | RED_FELL_BACK |
#   RED_REMOVE_ENTRY | RED_ISSUE_CLOSED | NOT_MEASURED
# EXPECTED_ISSUE is "" when the model has no entry; ISSUE_STATE is OPEN|CLOSED|"".
verdict() {
  local backend="$1" issue="$2" state="$3"
  if [[ "$backend" == unknown ]]; then echo NOT_MEASURED; return 0; fi
  if [[ -z "$issue" ]]; then
    if [[ "$backend" == admitted ]]; then echo GREEN; else echo RED_FELL_BACK; fi
    return 0
  fi
  case "$state" in
    CLOSED) echo RED_ISSUE_CLOSED; return 0 ;;
    OPEN) ;;
    *) echo NOT_MEASURED; return 0 ;;
  esac
  if [[ "$backend" == admitted ]]; then echo RED_REMOVE_ENTRY; else echo EXPECTED; fi
}

# expected_issue_for EXPECTED_TEXT MODEL_BASENAME -> issue number or ""
expected_issue_for() {
  local text="$1" base="$2" name num rest
  while read -r name num rest; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    num="${num#\#}"
    if [[ "$name" == "$base" ]]; then echo "$num"; return 0; fi
  done <<<"$text"
  echo ""
}

usage() {
  echo "usage: $0 --apr APR_PATH --models LIST_FILE [--expected FILE] | --self-test" >&2
  exit 2
}

# --- main run ---------------------------------------------------------------------

run_all() {
  local apr="$1" list="$2" expected_file="$3"
  local red=0 nm=0 line base out rc backend issue state v expected_text="" list_text

  case "$apr" in
    */*) ;;
    *) echo "refused: --apr must be a pinned path, not a bare name" >&2; exit 2 ;;
  esac
  if ! is_exec "$apr"; then echo "not_measured: apr binary not executable: $apr"; exit 3; fi
  if ! file_exists "$list"; then echo "not_measured: model list missing: $list"; exit 3; fi
  if file_exists "$expected_file"; then expected_text="$(read_text "$expected_file")"; fi
  list_text="$(read_text "$list")"

  if ! have_gpu; then
    echo "not_measured: NO-GPU (nvidia-smi -L lists no GPU); this is not a pass"
    exit 3
  fi

  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "${line// }" || "$line" == \#* ]] && continue
    base="${line##*/}"
    if ! file_exists "$line"; then
      echo "NOT_MEASURED ${base}: model file missing"; nm=1; continue
    fi
    rc=0; out="$(run_apr "$apr" "$line")" || rc=$?
    backend="$(classify_backend "$out")"
    if [[ "$rc" -ne 0 && "$backend" == unknown ]]; then
      echo "NOT_MEASURED ${base}: apr exited ${rc} with no backend line"; nm=1; continue
    fi
    issue="$(expected_issue_for "$expected_text" "$base")"
    state=""
    if [[ -n "$issue" ]]; then state="$(issue_state "$issue" 2>/dev/null)" || state=""; fi
    v="$(verdict "$backend" "$issue" "$state")"
    case "$v" in
      GREEN) echo "GREEN ${base}: admitted on GPU" ;;
      EXPECTED) echo "EXPECTED ${base}: falls back, tracked by #${issue} (open)" ;;
      RED_FELL_BACK) echo "RED ${base}: fell back to CPU, no expected-fallback entry"; red=1 ;;
      RED_REMOVE_ENTRY) echo "RED ${base}: runs on GPU now; remove the entry (#${issue}) from the expected-fallback file"; red=1 ;;
      RED_ISSUE_CLOSED) echo "RED ${base}: expected-fallback entry names #${issue}, which is closed; fix or remove"; red=1 ;;
      NOT_MEASURED) echo "NOT_MEASURED ${base}: backend or issue state unreadable"; nm=1 ;;
    esac
  done <<<"$list_text"

  if [[ "$red" -eq 1 ]]; then exit 1; fi
  if [[ "$nm" -eq 1 ]]; then exit 3; fi
  exit 0
}

# --- self-test --------------------------------------------------------------------

self_test() {
  local pass=0 fail=0
  t() { # name want got
    if [[ "$2" == "$3" ]]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL $1: want '$2' got '$3'"; fi
  }
  local ADM='{"backend":{"requested":"default","ran":"gpu","fell_back":false}}'
  local FB='{"backend":{"requested":"default","ran":"cpu","fell_back":true}}'
  local MSG='warning: GPU (CUDA) path rejected, falling back to CPU: CUDA init failed: PARITY-GATE FAILED cosine 0.453'
  local MIXED="${MSG}"$'\n'"${ADM}"

  # classify_backend table
  t classify-admitted admitted "$(classify_backend "$ADM")"
  t classify-fell-back-json fell_back "$(classify_backend "$FB")"
  t classify-fell-back-ran-cpu fell_back "$(classify_backend '{"backend":{"ran":"cpu"}}')"
  t classify-message-only fell_back "$(classify_backend "$MSG")"
  t classify-message-beats-json fell_back "$(classify_backend "$MIXED")"
  t classify-spaced-json admitted "$(classify_backend '{"backend": {"ran": "gpu", "fell_back": false}}')"
  t classify-empty unknown "$(classify_backend '')"
  t classify-gpu-without-fell-back-field unknown "$(classify_backend '{"backend":{"ran":"gpu"}}')"
  t classify-garbage unknown "$(classify_backend 'Segmentation fault')"

  # verdict table: backend issue state
  t v-admitted GREEN "$(verdict admitted '' '')"
  t v-fell-back RED_FELL_BACK "$(verdict fell_back '' '')"
  t v-expected EXPECTED "$(verdict fell_back 7285 OPEN)"
  t v-expected-now-passing RED_REMOVE_ENTRY "$(verdict admitted 7285 OPEN)"
  t v-expected-issue-closed RED_ISSUE_CLOSED "$(verdict fell_back 7285 CLOSED)"
  t v-expected-closed-and-passing RED_ISSUE_CLOSED "$(verdict admitted 7285 CLOSED)"
  t v-expected-state-unreadable NOT_MEASURED "$(verdict fell_back 7285 '')"
  t v-unknown-backend NOT_MEASURED "$(verdict unknown '' '')"
  t v-unknown-with-entry NOT_MEASURED "$(verdict unknown 7285 OPEN)"

  # expected file parsing
  local EXP=$'# comment\nQwen3-0.6B-Q4_K_M.gguf 7285\nother.gguf #12\n'
  t exp-hit 7285 "$(expected_issue_for "$EXP" Qwen3-0.6B-Q4_K_M.gguf)"
  t exp-hash-prefix 12 "$(expected_issue_for "$EXP" other.gguf)"
  t exp-miss '' "$(expected_issue_for "$EXP" Qwen3-1.7B-Q4_K_M.gguf)"
  t exp-no-prefix-match '' "$(expected_issue_for "$EXP" Qwen3-0.6B)"

  # have_gpu table (probe output is the seam)
  local g
  g="$(gpu_probe_case 'GPU 0: NVIDIA GB10 (UUID: GPU-abc)')"; t gpu-present yes "$g"
  g="$(gpu_probe_case '')"; t gpu-absent-empty no "$g"
  g="$(gpu_probe_case 'No devices were found')"; t gpu-absent-msg no "$g"
  g="$(gpu_probe_case 'FAIL')"; t gpu-probe-fails no "$g"

  # end to end through run_all, in subshells because it exits. No files are made:
  # every filesystem and process touch goes through a seam.
  local apr=/pinned/pinned-binary list=/lists/models.txt exp=/lists/expected.txt
  e2e() { # name want apr_output issue_state gpu(yes|no) with_expected(yes|no)
    local name="$1" want="$2" aout="$3" st="$4" gpu="$5" wexp="$6" r=0
    (
      if [[ "$gpu" == yes ]]; then gpu_probe() { echo "GPU 0: NVIDIA GB10"; }; else gpu_probe() { echo ""; }; fi
      is_exec() { return 0; }
      file_exists() { case "$1" in /lists/expected.txt) [[ "$wexp" == yes ]] ;; *) return 0 ;; esac; }
      read_text() { case "$1" in /lists/expected.txt) echo "Qwen3-0.6B-Q4_K_M.gguf 7285" ;; *) echo "/models/Qwen3-0.6B-Q4_K_M.gguf" ;; esac; }
      run_apr() { printf '%s\n' "$aout"; }
      issue_state() { if [[ -n "$st" ]]; then echo "$st"; else return 1; fi; }
      run_all "$apr" "$list" "$exp"
    ) >/dev/null || r=$?
    t "e2e-$name" "$want" "$r"
  }
  e2e admitted-no-entry 0 "$ADM" OPEN yes no
  e2e fell-back-no-entry 1 "$FB" OPEN yes no
  e2e expected-fallback 0 "$FB" OPEN yes yes
  e2e expected-now-passing 1 "$ADM" OPEN yes yes
  e2e expected-issue-closed 1 "$FB" CLOSED yes yes
  e2e expected-state-unreadable 3 "$FB" "" yes yes
  e2e no-gpu 3 "$ADM" OPEN no no
  e2e unreadable-output 3 "Segmentation fault" OPEN yes no

  local rc=0
  ( run_all "apr" "$list" "$exp" ) >/dev/null 2>&1 || rc=$?
  t refuse-bare-apr 2 "$rc"
  rc=0
  ( is_exec() { return 1; }; run_all "$apr" "$list" "$exp" ) >/dev/null 2>&1 || rc=$?
  t apr-not-executable 3 "$rc"
  rc=0
  ( is_exec() { return 0; }; file_exists() { return 1; }; run_all "$apr" "$list" "$exp" ) >/dev/null 2>&1 || rc=$?
  t missing-list 3 "$rc"
  rc=0
  (
    is_exec() { return 0; }
    gpu_probe() { echo "GPU 0: x"; }
    file_exists() { case "$1" in /lists/models.txt) return 0 ;; *) return 1 ;; esac; }
    read_text() { echo "/models/not-there.gguf"; }
    run_all "$apr" "$list" "$exp"
  ) >/dev/null 2>&1 || rc=$?
  t missing-model-file 3 "$rc"

  echo "self-test: ${pass} passed, ${fail} failed"
  [[ "$fail" -eq 0 ]]
}

# gpu_probe_case OUTPUT: runs have_gpu against a canned probe; 'FAIL' = probe errors.
gpu_probe_case() {
  local canned="$1"
  (
    if [[ "$canned" == FAIL ]]; then gpu_probe() { return 9; }; else gpu_probe() { echo "$canned"; }; fi
    if have_gpu; then echo yes; else echo no; fi
  )
}

# --- entry ------------------------------------------------------------------------

main() {
  local apr="" list="" expected="$DEFAULT_EXPECTED"
  if [[ "${1:-}" == "--self-test" ]]; then self_test; exit $?; fi
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --apr) apr="${2:-}"; shift 2 ;;
      --models) list="${2:-}"; shift 2 ;;
      --expected) expected="${2:-}"; shift 2 ;;
      *) usage ;;
    esac
  done
  [[ -n "$apr" && -n "$list" ]] || usage
  run_all "$apr" "$list" "$expected"
}

main "$@"
