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
# LIST     one GGUF path per line; blank lines, whitespace-only lines and '#' lines (also
#          indented) are ignored. A list with no model line is not_measured, never green.
# EXPECTED one line per model known to fall back: "<gguf basename> <issue number>" (the
#          number may carry a leading '#'). Default: evidence/models/expected-gpu-fallback.txt.
#          The file must exist and be readable; a line without a numeric issue, or two lines
#          for one basename, is an error (not_measured). An entry whose issue is CLOSED is
#          red, and so is an entry whose model now runs on the GPU ("remove the entry"): the
#          file can only be emptied by the defect being fixed. An entry whose model is not in
#          LIST was not exercised, so the run is not_measured.
#
# Exit: 0 every model admitted or expected-and-still-falling-back
#       1 red (fell back unexpectedly, stale expected entry)
#       2 usage error (unknown option, option missing its value, missing --apr/--models,
#         bare apr name)
#       3 not_measured (no GPU, missing/unreadable/non-file input, no model line, malformed
#         expected file, apr exited non-zero, unreadable output, issue state unreadable,
#         expected entry not exercised). Never green.
#
# Pin the binary: pass the path "$APR" from scripts/apr_bin.sh; a bare `apr` is refused.
# The apr invocation is `apr run MODEL -n 4 --json`; APR_RUN_EXTRA appends flags, split on
# whitespace (spaces, tabs, newlines) into an array (no glob expansion, no quote processing).
# This script is not wired into any workflow, PR path or guard tree.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_EXPECTED="${REPO_ROOT}/evidence/models/expected-gpu-fallback.txt"
ISSUE_REPO="${ISSUE_REPO:-paiml/aprender}"

# --- injectable seams (replaced by the self-test) ---------------------------------

gpu_probe() { nvidia-smi -L 2>/dev/null; }
file_exists() { [[ -f "$1" && -r "$1" ]]; }
is_exec() { [[ -f "$1" && -x "$1" ]]; }
read_text() { cat -- "$1"; }

# run_apr APR MODEL: prints stdout+stderr of a 4-token run; returns apr's status.
run_apr() {
  local apr="$1" model="$2" out rc=0
  local -a extra=()
  IFS=$' \t\n' read -r -d '' -a extra <<<"${APR_RUN_EXTRA:-}" || true
  out="$("$apr" run "$model" -n 4 --json "${extra[@]+"${extra[@]}"}" 2>&1)" || rc=$?
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

# expected_problems EXPECTED_TEXT: prints one line per malformed or duplicate entry.
expected_problems() {
  local text="$1" name num rest
  local -A seen=()
  while read -r name num rest; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    if ! [[ "${num#\#}" =~ ^[0-9]+$ ]]; then
      echo "entry '${name}' has no numeric issue (got '${num}')"
    fi
    if [[ -n "${rest}" && "${rest}" != \#* ]]; then echo "entry '${name}' has trailing text (got '${rest}')"; fi
    if [[ -n "${seen[$name]:-}" ]]; then echo "duplicate entry for '${name}'"; fi
    seen[$name]=1
  done <<<"$text"
}

# expected_names EXPECTED_TEXT: prints each entry's basename.
expected_names() {
  local name num rest
  while read -r name num rest; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    echo "$name"
  done <<<"$1"
}

# trim LINE: strips leading and trailing whitespace.
trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

usage() {
  echo "usage: $0 --apr APR_PATH --models LIST_FILE [--expected FILE] | --self-test" >&2
  exit 2
}

# --- main run ---------------------------------------------------------------------

run_all() {
  local apr="$1" list="$2" expected_file="$3"
  local red=0 nm=0 measured=0 line base out rc backend issue state v expected_text list_text problems name
  local -A listed=()

  case "$apr" in
    */*) ;;
    *) echo "refused: --apr must be a pinned path, not a bare name" >&2; exit 2 ;;
  esac
  if ! is_exec "$apr"; then echo "not_measured: apr binary not an executable file: $apr"; exit 3; fi
  if ! file_exists "$list"; then echo "not_measured: model list missing, unreadable or not a file: $list"; exit 3; fi
  if ! file_exists "$expected_file"; then
    echo "not_measured: expected-fallback file missing, unreadable or not a file: $expected_file"; exit 3
  fi
  if ! list_text="$(read_text "$list")"; then echo "not_measured: cannot read $list"; exit 3; fi
  if ! expected_text="$(read_text "$expected_file")"; then echo "not_measured: cannot read $expected_file"; exit 3; fi
  problems="$(expected_problems "$expected_text")"
  if [[ -n "$problems" ]]; then
    echo "not_measured: malformed expected-fallback file ${expected_file}:"; echo "$problems"; exit 3
  fi

  if ! have_gpu; then
    echo "not_measured: NO-GPU (nvidia-smi -L lists no GPU); this is not a pass"
    exit 3
  fi

  while IFS= read -r line || [[ -n "$line" ]]; do
    line="$(trim "$line")"
    [[ -z "$line" || "$line" == \#* ]] && continue
    base="${line##*/}"
    if [[ -z "$base" ]]; then echo "NOT_MEASURED ${line}: model path has no file name (trailing slash?)"; nm=1; continue; fi
    listed[$base]=1
    if ! file_exists "$line"; then
      echo "NOT_MEASURED ${base}: model file missing"; nm=1; continue
    fi
    rc=0; out="$(run_apr "$apr" "$line")" || rc=$?
    if [[ "$rc" -ne 0 ]]; then
      echo "NOT_MEASURED ${base}: apr exited ${rc}; output is not a measurement"; nm=1; continue
    fi
    backend="$(classify_backend "$out")"
    measured=$((measured + 1))
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

  if [[ "${#listed[@]}" -eq 0 ]]; then
    echo "not_measured: no model line in ${list}; zero models run is not a pass"; exit 3
  fi
  while IFS= read -r name; do
    [[ -z "$name" ]] && continue
    if [[ -z "${listed[$name]:-}" ]]; then
      echo "NOT_MEASURED ${name}: expected-fallback entry, but the model is not in ${list}; not exercised"; nm=1
    fi
  done <<<"$(expected_names "$expected_text")"

  if [[ "$red" -eq 1 ]]; then exit 1; fi
  if [[ "$nm" -eq 1 || "$measured" -eq 0 ]]; then exit 3; fi
  exit 0
}

# --- self-test --------------------------------------------------------------------

# gpu_probe_case OUTPUT: runs have_gpu against a canned probe; 'FAIL' = probe errors.
gpu_probe_case() {
  local canned="$1"
  (
    if [[ "$canned" == FAIL ]]; then gpu_probe() { return 9; }; else gpu_probe() { echo "$canned"; }; fi
    if have_gpu; then echo yes; else echo no; fi
  )
}

self_test() {
  local pass=0 fail=0 tmp
  t() { # name want got
    if [[ "$2" == "$3" ]]; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL $1: want '$2' got '$3'"; fi
  }
  tmp="$(mktemp -d)"
  trap '[[ -n "${tmp:-}" ]] && rm -rf "${tmp:?}"' RETURN
  local ADM='{"backend":{"requested":"default","ran":"gpu","fell_back":false}}'
  local FB='{"backend":{"requested":"default","ran":"cpu","fell_back":true}}'
  local MSG='warning: GPU (CUDA) path rejected, falling back to CPU: CUDA init failed: PARITY-GATE FAILED cosine 0.453'
  local MIXED="${MSG}"$'\n'"${ADM}"

  # classify_backend table: every alternative of every || is its own row
  t classify-admitted admitted "$(classify_backend "$ADM")"
  t classify-fell-back-json fell_back "$(classify_backend "$FB")"
  t classify-fell-back-true-only fell_back "$(classify_backend '{"backend":{"fell_back":true}}')"
  t classify-fell-back-true-spaced fell_back "$(classify_backend '{"fell_back" : true}')"
  t classify-fell-back-ran-cpu fell_back "$(classify_backend '{"backend":{"ran":"cpu"}}')"
  t classify-ran-cpu-with-false fell_back "$(classify_backend '{"backend":{"ran":"cpu","fell_back":false}}')"
  t classify-message-only fell_back "$(classify_backend "$MSG")"
  t classify-rejected-only fell_back "$(classify_backend 'GPU (CUDA) path rejected: x')"
  t classify-falling-back-only fell_back "$(classify_backend 'falling back to CPU')"
  t classify-message-beats-json fell_back "$(classify_backend "$MIXED")"
  t classify-spaced-json admitted "$(classify_backend '{"backend": {"ran": "gpu", "fell_back": false}}')"
  t classify-empty unknown "$(classify_backend '')"
  t classify-gpu-without-fell-back-field unknown "$(classify_backend '{"backend":{"ran":"gpu"}}')"
  t classify-fell-back-false-without-ran unknown "$(classify_backend '{"backend":{"fell_back":false}}')"
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
  t exp-valid-no-problems '' "$(expected_problems "$EXP")"
  t exp-no-number-problem "entry 'a.gguf' has no numeric issue (got '')" "$(expected_problems 'a.gguf')"
  t exp-bare-hash-problem "entry 'a.gguf' has no numeric issue (got '#')" "$(expected_problems 'a.gguf #')"
  t exp-word-number-problem "entry 'a.gguf' has no numeric issue (got 'abc')" "$(expected_problems 'a.gguf abc')"
  t exp-duplicate-problem "duplicate entry for 'a.gguf'" "$(expected_problems $'a.gguf 1\na.gguf 2')"
  t exp-names $'Qwen3-0.6B-Q4_K_M.gguf\nother.gguf' "$(expected_names "$EXP")"

  # trim
  t trim-tab-only '' "$(trim $'\t\t')"
  t trim-both-ends '/m/a.gguf' "$(trim $'  \t/m/a.gguf \t')"

  # have_gpu table (probe output is the seam), including the line-start anchor
  t gpu-present yes "$(gpu_probe_case 'GPU 0: NVIDIA GB10 (UUID: GPU-abc)')"
  t gpu-present-second-line yes "$(gpu_probe_case $'note\nGPU 1: NVIDIA GB10')"
  t gpu-absent-empty no "$(gpu_probe_case '')"
  t gpu-absent-msg no "$(gpu_probe_case 'No devices were found')"
  t gpu-probe-fails no "$(gpu_probe_case 'FAIL')"
  t gpu-anchor-mid-line no "$(gpu_probe_case 'xGPU 0: NVIDIA')"
  t gpu-anchor-mid-line-second no "$(gpu_probe_case $'ok\nnot a GPU 0: line')"

  # real seams against real files
  : >"${tmp}/f.txt"
  mkdir "${tmp}/d"
  t seam-file-ok yes "$(if file_exists "${tmp}/f.txt"; then echo yes; else echo no; fi)"
  t seam-dir-refused no "$(if file_exists "${tmp}/d"; then echo yes; else echo no; fi)"
  t seam-missing-refused no "$(if file_exists "${tmp}/nope"; then echo yes; else echo no; fi)"
  t seam-exec-dir-refused no "$(if is_exec "${tmp}/d"; then echo yes; else echo no; fi)"

  # run_apr against a real fake binary: argv, stderr merged, status passed through, no glob
  cat >"${tmp}/fakeapr" <<'FAKE'
#!/usr/bin/env bash
for a in "$@"; do printf 'ARG[%s]\n' "$a"; done
echo "stderr-line" >&2
exit "${FAKE_RC:-0}"
FAKE
  chmod +x "${tmp}/fakeapr"
  touch "${tmp}/globbed1" "${tmp}/globbed2"
  local ra rrc=0
  ra="$(cd "$tmp" && APR_RUN_EXTRA='--x *' run_apr "${tmp}/fakeapr" /m/a.gguf)" || rrc=$?
  t run-apr-argv $'ARG[run]\nARG[/m/a.gguf]\nARG[-n]\nARG[4]\nARG[--json]\nARG[--x]\nARG[*]\nstderr-line' "$ra"
  t run-apr-rc-zero 0 "$rrc"
  ra="$(FAKE_RC=7 run_apr "${tmp}/fakeapr" /m/a.gguf)" || rrc=$?
  t run-apr-rc-passthrough 7 "$rrc"
  ra="$(run_apr "${tmp}/fakeapr" /m/a.gguf)"
  t run-apr-no-extra-argc 6 "$(grep -c . <<<"$ra")"

  # end to end through run_all, in subshells because it exits. Filesystem and process
  # touches go through seams, except the run_apr rows above.
  local apr=/pinned/pinned-binary list=/lists/models.txt exp=/lists/expected.txt
  e2e() { # name want apr_output issue_state gpu(yes|no) expected_text list_text apr_rc
    local name="$1" want="$2" aout="$3" st="$4" gpu="$5" etext="$6" ltext="$7" arc="${8:-0}" r=0
    (
      if [[ "$gpu" == yes ]]; then gpu_probe() { echo "GPU 0: NVIDIA GB10"; }; else gpu_probe() { echo ""; }; fi
      is_exec() { return 0; }
      file_exists() { return 0; }
      read_text() { case "$1" in /lists/expected.txt) printf '%s\n' "$etext" ;; *) printf '%s\n' "$ltext" ;; esac; }
      run_apr() { printf '%s\n' "$aout"; return "$arc"; }
      issue_state() { if [[ -n "$st" ]]; then echo "$st"; else return 1; fi; }
      run_all "$apr" "$list" "$exp"
    ) >/dev/null || r=$?
    t "e2e-$name" "$want" "$r"
  }
  local NONE='# none' ONE='/models/Qwen3-0.6B-Q4_K_M.gguf' QE='Qwen3-0.6B-Q4_K_M.gguf 7285'
  e2e admitted-no-entry 0 "$ADM" OPEN yes "$NONE" "$ONE"
  e2e fell-back-no-entry 1 "$FB" OPEN yes "$NONE" "$ONE"
  e2e expected-fallback 0 "$FB" OPEN yes "$QE" "$ONE"
  e2e expected-now-passing 1 "$ADM" OPEN yes "$QE" "$ONE"
  e2e expected-issue-closed 1 "$FB" CLOSED yes "$QE" "$ONE"
  e2e expected-state-unreadable 3 "$FB" "" yes "$QE" "$ONE"
  e2e no-gpu 3 "$ADM" OPEN no "$NONE" "$ONE"
  e2e unreadable-output 3 "Segmentation fault" OPEN yes "$NONE" "$ONE"
  e2e apr-nonzero-with-admitted-json 3 "$ADM" OPEN yes "$NONE" "$ONE" 1
  e2e apr-nonzero-with-fallback-expected 3 "$FB" OPEN yes "$QE" "$ONE" 137
  e2e apr-nonzero-no-backend 3 "crash" OPEN yes "$NONE" "$ONE" 1
  e2e empty-list 3 "$ADM" OPEN yes "$NONE" ""
  e2e comment-only-list 3 "$ADM" OPEN yes "$NONE" $'# todo\n   # indented\n\t'
  e2e tab-only-and-indented-comment-skipped 0 "$ADM" OPEN yes "$NONE" $'\t\n  # note\n\t# note\n/models/a.gguf'
  e2e padded-model-line 0 "$ADM" OPEN yes "$NONE" $'  \t/models/a.gguf\t '
  e2e expected-entry-not-in-list 3 "$ADM" OPEN yes "$QE" '/models/other.gguf'
  e2e expected-no-number 3 "$ADM" OPEN yes 'a.gguf' '/models/a.gguf'
  e2e expected-duplicate 3 "$FB" OPEN yes $'a.gguf 1\na.gguf 2' '/models/a.gguf'
  e2e two-models-both-red 1 "$FB" OPEN yes "$NONE" $'/models/a.gguf\n/models/b.gguf'
  e2e expected-trailing-text 3 "$FB" OPEN yes 'a.gguf 7285 junk' '/models/a.gguf'
  e2e expected-trailing-comment-ok 0 "$FB" OPEN yes 'a.gguf 7285 # why' '/models/a.gguf'
  e2e trailing-slash-model-path 3 "$ADM" OPEN yes "$NONE" '/models/dir/'
  e2e bare-slash-model-path 3 "$ADM" OPEN yes "$NONE" '/'

  # two models with DIFFERENT outcomes: the exit precedence (red beats not_measured beats
  # green) and the absence of any flag reset by a later green model
  e2m() { # name want etext "model=output[#rc]"...
    local name="$1" want="$2" etext="$3" ltext="" spec r=0
    shift 3
    for spec in "$@"; do ltext+="/models/${spec%%=*}"$'\n'; done
    (
      gpu_probe() { echo "GPU 0: NVIDIA GB10"; }
      is_exec() { return 0; }
      file_exists() { return 0; }
      read_text() { case "$1" in /lists/expected.txt) printf '%s\n' "$etext" ;; *) printf '%s\n' "$ltext" ;; esac; }
      run_apr() {
        local s o m="${2##*/}"
        for s in "${SPECS[@]}"; do
          if [[ "${s%%=*}" == "$m" ]]; then
            o="${s#*=}"
            if [[ "$o" == *'#'* ]]; then printf '%s\n' "${o%#*}"; return "${o##*#}"; fi
            printf '%s\n' "$o"; return 0
          fi
        done
      }
      issue_state() { echo OPEN; }
      SPECS=("$@")
      run_all "$apr" "$list" "$exp"
    ) >/dev/null || r=$?
    t "e2m-$name" "$want" "$r"
  }
  e2m red-then-green 1 "$NONE" "a.gguf=$FB" "b.gguf=$ADM"
  e2m green-then-red 1 "$NONE" "a.gguf=$ADM" "b.gguf=$FB"
  e2m red-then-not-measured 1 "$NONE" "a.gguf=$FB" "b.gguf=crash#1"
  e2m not-measured-then-red 1 "$NONE" "a.gguf=crash#1" "b.gguf=$FB"
  e2m not-measured-then-green 3 "$NONE" "a.gguf=crash#1" "b.gguf=$ADM"
  e2m green-then-not-measured 3 "$NONE" "a.gguf=$ADM" "b.gguf=crash#1"
  e2m both-green 0 "$NONE" "a.gguf=$ADM" "b.gguf=$ADM"
  e2m green-and-expected 0 'b.gguf 7285' "a.gguf=$ADM" "b.gguf=$FB"
  e2m red-then-expected 1 'b.gguf 7285' "a.gguf=$FB" "b.gguf=$FB"
  e2m not-measured-then-expected 3 'b.gguf 7285' "a.gguf=crash#1" "b.gguf=$FB"
  e2m red-then-expected-then-green 1 'b.gguf 7285' "a.gguf=$FB" "b.gguf=$FB" "c.gguf=$ADM"

  # the real file_exists and is_exec on real files (no seam): a directory, a missing path,
  # a file without the read or the exec bit
  local d="${tmp}/d" f="${tmp}/f" x="${tmp}/x" nr="${tmp}/nr"
  mkdir -p "$d"
  : >"$f"; : >"$x"; : >"$nr"
  chmod 644 "$f"; chmod 755 "$x"; chmod 000 "$nr"
  t seam-file-exists-file yes "$(if file_exists "$f"; then echo yes; else echo no; fi)"
  t seam-file-exists-dir no "$(if file_exists "$d"; then echo yes; else echo no; fi)"
  t seam-file-exists-missing no "$(if file_exists "${tmp}/none"; then echo yes; else echo no; fi)"
  if [[ "$(id -u)" -ne 0 ]]; then
    t seam-file-exists-unreadable no "$(if file_exists "$nr"; then echo yes; else echo no; fi)"
  fi
  t seam-is-exec-exec yes "$(if is_exec "$x"; then echo yes; else echo no; fi)"
  t seam-is-exec-plain-file no "$(if is_exec "$f"; then echo yes; else echo no; fi)"
  t seam-is-exec-dir no "$(if is_exec "$d"; then echo yes; else echo no; fi)"
  t seam-is-exec-missing no "$(if is_exec "${tmp}/none"; then echo yes; else echo no; fi)"
  t seam-read-text-roundtrip hello "$(printf 'hello\n' >"$f"; read_text "$f")"

  # APR_RUN_EXTRA splits on any whitespace, never globs
  t run-apr-extra-tab-split 2 "$(APR_RUN_EXTRA=$'--a\t--b' run_apr "${tmp}/fakeapr" /m/a.gguf | grep -c '^ARG\[--[ab]\]$')"
  t run-apr-extra-newline-split 2 "$(APR_RUN_EXTRA=$'--a\n--b' run_apr "${tmp}/fakeapr" /m/a.gguf | grep -c '^ARG\[--[ab]\]$')"
  t run-apr-extra-mixed-whitespace-split 3 "$(APR_RUN_EXTRA=$' --a\n\n\t--b \n--c\n' run_apr "${tmp}/fakeapr" /m/a.gguf | grep -c '^ARG\[--[abc]\]$')"
  t shipped-expected-valid '' "$(expected_problems "$(<"$DEFAULT_EXPECTED")")"

  # input-file refusals (seams answer per path)
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
    file_exists() { [[ "$1" != /lists/expected.txt ]]; }
    run_all "$apr" "$list" "$exp"
  ) >/dev/null 2>&1 || rc=$?
  t missing-expected-file 3 "$rc"
  rc=0
  (
    is_exec() { return 0; }
    file_exists() { return 0; }
    read_text() { return 1; }
    run_all "$apr" "$list" "$exp"
  ) >/dev/null 2>&1 || rc=$?
  t unreadable-list-text 3 "$rc"
  rc=0
  (
    is_exec() { return 0; }
    gpu_probe() { echo "GPU 0: x"; }
    file_exists() { return 0; }
    read_text() { case "$1" in /lists/expected.txt) return 1 ;; *) echo "/models/a.gguf" ;; esac; }
    run_apr() { printf '%s\n' "$ADM"; }
    run_all "$apr" "$list" "$exp"
  ) >/dev/null 2>&1 || rc=$?
  t unreadable-expected-text-only 3 "$rc"
  rc=0
  (
    is_exec() { return 0; }
    gpu_probe() { echo "GPU 0: x"; }
    file_exists() { case "$1" in /lists/models.txt | /lists/expected.txt) return 0 ;; *) return 1 ;; esac; }
    read_text() { case "$1" in /lists/expected.txt) echo "# none" ;; *) echo "/models/not-there.gguf" ;; esac; }
    run_all "$apr" "$list" "$exp"
  ) >/dev/null 2>&1 || rc=$?
  t missing-model-file 3 "$rc"

  # main(): the real entry path, argument handling and the exit code it forwards
  mc() { # name want args...
    local name="$1" want="$2" r=0
    shift 2
    ( main "$@" ) >/dev/null 2>&1 || r=$?
    t "main-$name" "$want" "$r"
  }
  mc no-args 2
  mc unknown-option 2 --bogus
  mc apr-missing-value 2 --apr
  mc models-missing-value 2 --apr /p/apr --models
  mc expected-missing-value 2 --apr /p/apr --models /l/m --expected
  mc no-models 2 --apr /p/apr
  mc self-test-extra-arg 2 --self-test --bogus
  mc no-apr 2 --models /l/m
  mc empty-apr-value 2 --apr '' --models /l/m
  rc=0
  (
    gpu_probe() { echo "GPU 0: x"; }
    is_exec() { return 0; }
    file_exists() { return 0; }
    read_text() { case "$1" in /l/e) echo "# none" ;; *) echo "/m/a.gguf" ;; esac; }
    run_apr() { echo "$ADM"; }
    main --apr /p/apr --models /l/m --expected /l/e
  ) >/dev/null 2>&1 || rc=$?
  t main-forwards-green 0 "$rc"
  rc=0
  (
    gpu_probe() { echo "GPU 0: x"; }
    is_exec() { return 0; }
    file_exists() { return 0; }
    read_text() { case "$1" in /l/e) echo "# none" ;; *) echo "/m/a.gguf" ;; esac; }
    run_apr() { echo "$FB"; }
    main --apr /p/apr --models /l/m --expected /l/e
  ) >/dev/null 2>&1 || rc=$?
  t main-forwards-red 1 "$rc"
  rc=0
  (
    gpu_probe() { echo "GPU 0: x"; }
    is_exec() { return 0; }
    file_exists() { return 0; }
    read_text() { case "$1" in /l/e | "$DEFAULT_EXPECTED") echo "# none" ;; *) echo "/m/a.gguf" ;; esac; }
    run_apr() { echo "$ADM"; }
    main --apr /p/apr --models /l/m
  ) >/dev/null 2>&1 || rc=$?
  t main-default-expected-used 0 "$rc"
  t default-expected-path-exists yes "$(if [[ -f "$DEFAULT_EXPECTED" ]]; then echo yes; else echo no; fi)"

  echo "self-test: ${pass} passed, ${fail} failed"
  [[ "$fail" -eq 0 ]]
}

# --- entry ------------------------------------------------------------------------

main() {
  local apr="" list="" expected="$DEFAULT_EXPECTED"
  if [[ "${1:-}" == "--self-test" ]]; then [[ $# -eq 1 ]] || usage; self_test; exit $?; fi
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --apr | --models | --expected)
        [[ $# -ge 2 ]] || usage
        case "$1" in
          --apr) apr="$2" ;;
          --models) list="$2" ;;
          --expected) expected="$2" ;;
        esac
        shift 2 ;;
      *) usage ;;
    esac
  done
  [[ -n "$apr" && -n "$list" ]] || usage
  run_all "$apr" "$list" "$expected"
}

main "$@"
