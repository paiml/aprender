#!/usr/bin/env bash
# check_ladder_output_judged.sh — a verb that RAN is not a verb that WORKED (#3921).
#
# WHY. `chat` and `code` were judged by rc, `serve` by http status, and OUTPUT only
# by `apr qa`'s golden leg. So three of the operator's four verbs were covered for
# whether they ran, never for whether they worked. Measured on BOTH required hosts:
#
#   qwen2.5-coder-1.5b-instruct-q4k.apr, `apr chat --gpu`
#     rc=0  ran="gpu"  fell_back=false
#     "ürnópez zombie.ERRópez zombieópez zombie zombie zombie…"
#   the same model on CPU: "The capital of France is Paris."
#
# and the row was GREEN.
#
# WHAT THIS CHECKS, IN THREE LAYERS.
#
#   1. THE DETECTOR MIRRORS ITS SOURCE. The four signals are
#      `output_verification.rs`'s, in its order, with its thresholds: non-ASCII
#      saturation >60%, a 4+ byte fragment repeated 3x, U+FFFD density >1/32,
#      dominant character >=90%. Two judges that disagree about what "degenerate"
#      means on the same completion are worse than one judge, so the table below
#      pins the boundaries rather than trusting that they were copied.
#
#   3. THE RIGHT TEXT REACHES IT (#3925). Layer 1 proves the detector is right
#      about a completion and says nothing about what string is handed to it. That
#      is where this gate failed: `apr chat` frames its session in `════` rules,
#      the ladder captures with 2>&1, and the SEPARATOR tripped the repeated-
#      fragment signal on a reply that read "The capital of France is Paris."
#
#   2. THE VERDICT USES IT. A row whose verb ran with rc=0 and bad output must be
#      RED, and must SAY so — separately from the rc, because "ran and produced
#      garbage" is a different fact from "did not run" and a reader who sees rc=0
#      stops looking.
#
# Both the detector and the row builder are LIFTED FROM model_ladder.sh and run, so
# this tests the shipped code rather than a copy of its rules.
#
# Exit: 0 all cases as expected · 1 a case landed wrong · 2 could not check.
#       --self-test: 0 when each planted mutation turns this RED, 1 otherwise.
set -euo pipefail

SCRIPT="scripts/model_ladder.sh"
SELF_TEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --script) [ $# -ge 2 ] || { echo "--script needs a value" >&2; exit 2; }; SCRIPT="$2"; shift 2 ;;
    --self-test) SELF_TEST=1; shift ;;
    -h|--help) awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
    *) echo "check_ladder_output_judged: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

# The detector, lifted from the script. Anti-vacuity: it must contain all four
# signals, or this check has stopped testing what it claims to.
load_detector() {
  local src="$1" body
  [ -f "$src" ] || { echo "  cannot read $src" >&2; return 2; }
  body=$(awk '/^gibberish_reason\(\) \{/{f=1} f{print} f && /^\}$/{exit}' "$src")
  for sig in non_ascii repeated fffd dominant; do
    grep -q "def $sig" <<< "$body" || {
      echo "  the extracted detector is missing signal '$sig' — it no longer mirrors output_verification.rs" >&2
      return 2
    }
  done
  printf '%s' "$body"
}

judge() { # judge <src> <text> -> prints the reason, or nothing
  local src="$1" det
  det=$(load_detector "$src") || return 2
  printf '%s' "$2" | bash -c "$det; gibberish_reason" 2>/dev/null || true
}

# name|text|expect  (expect: bad = must be flagged, clean = must not be)
detector_cases() {
cat <<'CASES'
the-measured-artifact|ürnópez zombie.ERRópez zombieópez zombie zombie zombie zombie|bad
the-correct-answer|The capital of France is Paris.|clean
dominant-char-at-90|!!!!!!!!!!|bad
short-repeat-below-threshold|abab|clean
ascii-prose-is-clean|Paris is the capital and largest city of France today.|clean
CASES
}

run_detector_table() { # -> 0 all as expected
  local src="$1" rc=0 name text want got
  while IFS='|' read -r name text want; do
    [ -n "$name" ] || continue
    got=$(judge "$src" "$text") || return 2
    if [ -n "$got" ]; then got=bad; else got=clean; fi
    if [ "$got" = "$want" ]; then
      printf '  ok    %-30s %s\n' "$name" "$got"
    else
      printf '  FAIL  %-30s %s, expected %s\n' "$name" "$got" "$want"; rc=1
    fi
  done < <(detector_cases)
  return $rc
}

# ── layer 2: the VERDICT ─────────────────────────────────────────────────────
extract_builder() {
  local src="$1" body
  body=$(awk '/^[[:space:]]*row=\$\(python3/{f=1; next} f && /^PY$/{exit} f' "$src")
  grep -q 'green' <<< "$body" || { echo "  the extracted row builder does not compute green" >&2; return 2; }
  printf '%s' "$body"
}

# A row green on EVERY axis except that `chat` ran with rc=0 and bad output — the
# live gx10/lambda case.
row_chat_garbage() {
cat <<'JSON'
{"cuda":{"ran":true,"fallback":false,"escaped_special":false,"rc":0,
 "verbs":{"run":{"ran":true,"rc":0},
          "chat":{"ran":true,"rc":0,"output_bad":"gibberish (fragment ' zombie' repeats 3+ times)"},
          "code":{"ran":true,"rc":0},
          "serve":{"probed":true,"teardown":"clean","routes":{
            "/v1/completions|stream=false":{"http":200,"output_bad":null}}}}}}
JSON
}

QA_OK='{"capability_match":{"passed":true,"skipped":false,"message":"ok"},
        "golden_output":{"passed":true,"skipped":false,"message":"ok"},
        "gates":{},"gates_failed":[],"gates_reported":2}'

green_of() { # green_of <src> <be-json> -> true|false
  local src="$1" builder out
  builder=$(extract_builder "$src") || return 2
  out=$(printf '%s' "$builder" | python3 - "rid" "$QA_OK" "$2" 0 1 "deadbeef" "m.gguf" 1 2>/dev/null) || return 2
  python3 -c 'import json,sys; print(str(json.loads(sys.stdin.read())["green"]).lower())' <<< "$out"
}

check_verdict() { # -> 0 ok
  local src="$1" g
  g=$(green_of "$src" "$(row_chat_garbage)") || return 2
  if [ "$g" = "false" ]; then
    printf '  ok    %-30s green=false\n' "chat-rc0-bad-output"
    return 0
  fi
  printf '  FAIL  %-30s green=%s — a verb that ran and produced garbage is recorded as working\n' \
    "chat-rc0-bad-output" "$g"
  return 1
}

# ── layer 3: WHAT TEXT IS JUDGED (#3925) ─────────────────────────────────────
# Layer 1 proves the detector is right about a completion. It says nothing about
# what string reaches it, and that is where this gate actually failed: `apr chat`
# frames its session in `════` rules, the ladder captures with 2>&1, and the
# separator tripped the repeated-fragment signal on a run whose reply was "The
# capital of France is Paris." Five required rungs on gx10, seven on lambda, both
# backends, all previously green. A sound predicate pointed at the wrong input.
#
# The fix's own hazard is the opposite one, so it is pinned here too: an extractor
# that silently yields "" gives the detector nothing to flag and turns every row
# green. Absent, truncated and UNBOUNDED replies are each their own red.
load_judge() {
  local src="$1" fn body out=""
  [ -f "$src" ] || { echo "  cannot read $src" >&2; return 2; }
  for fn in gibberish_reason assistant_reply judge_reply; do
    body=$(awk -v F="^$fn\\\\(\\\\) \\\\{" '$0 ~ F {f=1} f{print} f && /^\}$/{exit}' "$src")
    [ -n "$body" ] || { echo "  $src defines no $fn() -- the judging layer is gone" >&2; return 2; }
    out="$out$body
"
  done
  printf '%s' "$out"
}

judge_capture() { # judge_capture <src> <capture-text> -> the reason, or nothing
  local src="$1" tmp rc
  tmp=$(mktemp) || return 2
  load_judge "$src" > "$tmp" || { rm -f "$tmp"; return 2; }
  printf 'text=$(cat); judge_reply "$text" chat\n' >> "$tmp"
  printf '%s' "$2" | bash "$tmp"; rc=$?
  rm -f "$tmp"
  return $rc
}

# The captures are VERBATIM from gx10 at 057f9a3a2, not invented shapes.
cap_correct_answer() { cat <<'T'
=== Model Chat (GGUF Format) ===

════════════════════════════════════════════════════════════
Loading model...
You: [BOS-FALLBACK] No tokenizer.ggml.bos_token_id in GGUF
[6.2s, ~1 tok/s]
Assistant: The capital of France is Paris.

You: Goodbye!
{"backend":{"requested":"cpu","ran":"cpu","fell_back":false}}
T
}
cap_code_json() { cat <<'T'
Launched apr serve on port 19745 (pid 2085384)
apr serve ready (2.0s)
{"type":"result","subtype":"success","status":"ok","result":"ok","session_id":"00000000-0006-7000-5c1e-5c1e26a9ee1b","num_turns":1}
T
}
cap_chrome_gibberish() { cat <<'T'
════════════════════════════════════════════════════════════
Assistant: ürnópez zombie.ERRópez zombieópez zombie zombie zombie zombie
You: Goodbye!
{"backend":{"requested":"cuda","ran":"cuda","fell_back":false}}
T
}
# aprender-3e's case: the reply itself contains `You:`. Terminating on the next
# `You:` cuts it and judges a FRAGMENT -- which can pass or fail for reasons that
# are not about the reply. The garbage lives AFTER the marker, so an early cut
# reads clean and this case goes red.
cap_reply_with_you() { cat <<'T'
════════════════════════════════════════════════════════════
Assistant: In a transcript the user turn is written
You: like this, and zombie zombie zombie zombie follows.
{"backend":{"requested":"cpu","ran":"cpu","fell_back":false}}
T
}
cap_no_envelope() { cat <<'T'
════════════════════════════════════════════════════════════
Assistant: The capital of France is Paris.
T
}
cap_no_reply() { cat <<'T'
Loading model...
{"backend":{"requested":"cpu","ran":"cpu","fell_back":false}}
T
}

# name|capture-fn|expect   (clean = no reason · bad = a verdict about the reply ·
# red = a refusal to judge, which must never be silent)
extraction_cases() {
cat <<'CASES'
chrome-wrapped-correct-answer|cap_correct_answer|clean
code-json-envelope-not-uuid|cap_code_json|clean
chrome-does-not-mask-gibberish|cap_chrome_gibberish|bad
reply-containing-You-judged-whole|cap_reply_with_you|bad
truncated-capture-no-envelope|cap_no_envelope|red
envelope-but-no-reply|cap_no_reply|red
CASES
}

run_extraction_table() { # -> 0 all as expected
  local src="$1" rc=0 name fn want got cls
  while IFS='|' read -r name fn want; do
    [ -n "$name" ] || continue
    got=$(judge_capture "$src" "$($fn)") || return 2
    if [ -z "$got" ]; then cls=clean
    elif case "$got" in "could not"*) true ;; *) false ;; esac; then cls=red
    else cls=bad; fi
    if [ "$cls" = "$want" ]; then
      printf '  ok    %-34s %s\n' "$name" "$cls"
    else
      printf '  FAIL  %-34s %s, expected %s  [%s]\n' "$name" "$cls" "$want" "${got:0:70}"; rc=1
    fi
  done < <(extraction_cases)
  return $rc
}

# ── layer 4: EVERY serve wire shape reaches the detector (#3957 F4c) ─────────────
# The probe read each body with one `json.load`. Streaming bodies are SSE (/v1/*) or
# NDJSON (/api/chat), so the load failed, the text was "" and the route read CLEAN:
# measured before the fix on the three bodies below, all three `output_bad=''`.
# The table pins both halves: the text is extracted from every shape, and a body
# that yields no text is a reason ("nothing measured"), never clean.
load_serve_judge() {
  local src="$1" fn body out=""
  [ -f "$src" ] || { echo "  cannot read $src" >&2; return 2; }
  for fn in gibberish_reason serve_reply_text serve_route_bad; do
    body=$(awk -v F="^$fn\\\\(\\\\) \\\\{" '$0 ~ F {f=1} f{print} f && /^\}$/{exit}' "$src")
    [ -n "$body" ] || { echo "  $src defines no $fn() -- the serve body judge is gone" >&2; return 2; }
    out="$out$body
"
  done
  printf '%s' "$out"
}

judge_body() { # judge_body <src> <body-text> -> the reason, or nothing
  local src="$1" tmp bodyf rc
  tmp=$(mktemp) || return 2; bodyf=$(mktemp) || { rm -f "$tmp"; return 2; }
  load_serve_judge "$src" > "$tmp" || { rm -f "$tmp" "$bodyf"; return 2; }
  printf '%s' "$2" > "$bodyf"
  printf 'serve_route_bad "$1" /probe\n' >> "$tmp"
  bash "$tmp" "$bodyf"; rc=$?
  rm -f "$tmp" "$bodyf"
  return $rc
}

# name|printf-format of the body|expect  (clean · bad = a verdict about the text ·
# empty = "nothing measured" · truncated = a stream with no terminal event; neither is ever clean)
body_cases() {
cat <<'CASES'
json-nonstream-correct|{"choices":[{"message":{"role":"assistant","content":"The capital of France is Paris."}}]}|clean
sse-chat-deltas-correct|data: {"choices":[{"delta":{"content":"The capital"}}]}\n\ndata: {"choices":[{"delta":{"content":" of France is Paris."}}]}\n\ndata: [DONE]\n\n|clean
sse-completions-text-garbage|data: {"choices":[{"text":"zombie zombie "}]}\n\ndata: {"choices":[{"text":"zombie zombie zombie"}]}\n\ndata: [DONE]\n\n|bad
sse-zero-deltas|data: [DONE]\n\n|empty
sse-role-only-delta|data: {"choices":[{"delta":{"role":"assistant"}}]}\n\ndata: [DONE]\n\n|empty
ndjson-api-chat-garbage|{"message":{"role":"assistant","content":"zombie zombie "},"done":false}\n{"message":{"role":"assistant","content":"zombie zombie zombie"},"done":false}\n{"done":true}\n|bad
ndjson-api-chat-correct|{"message":{"role":"assistant","content":"Paris."},"done":false}\n{"done":true}\n|clean
empty-body|\n|empty
sse-truncated-no-done|data: {"choices":[{"delta":{"content":"The capital of France is Paris."}}]}\n\n|truncated
ndjson-truncated-no-done|{"message":{"role":"assistant","content":"Paris."},"done":false}\n|truncated
CASES
}

run_body_table() { # -> 0 all as expected
  local src="$1" rc=0 name fmt want got cls body
  while IFS='|' read -r name fmt want; do
    [ -n "$name" ] || continue
    # shellcheck disable=SC2059  # the format IS the fixture: it carries the \n escapes
    body=$(printf "$fmt"; printf x); body=${body%x}
    got=$(judge_body "$src" "$body") || return 2
    if [ -z "$got" ]; then cls=clean
    elif case "$got" in "nothing measured"*) true ;; *) false ;; esac; then cls=empty
    elif case "$got" in "truncated stream"*) true ;; *) false ;; esac; then cls=truncated
    else cls=bad; fi
    if [ "$cls" = "$want" ]; then
      printf '  ok    %-34s %s\n' "$name" "$cls"
    else
      printf '  FAIL  %-34s %s, expected %s  [%s]\n' "$name" "$cls" "$want" "${got:0:70}"; rc=1
    fi
  done < <(body_cases)
  return $rc
}

if [ "$SELF_TEST" = 1 ]; then
  [ -f "$SCRIPT" ] || { echo "cannot read $SCRIPT" >&2; exit 2; }
  m1=$(mktemp); m2=$(mktemp); m3=$(mktemp); m4=$(mktemp); m5=$(mktemp); m6=$(mktemp)
  trap 'rm -f "$m1" "$m2" "$m3" "$m4" "$m5" "$m6"' EXIT

  echo "self-test: the shipped script"
  run_detector_table "$SCRIPT" > /dev/null && check_verdict "$SCRIPT" > /dev/null \
    && run_extraction_table "$SCRIPT" > /dev/null && run_body_table "$SCRIPT" > /dev/null \
    || { echo "SELF-TEST FAILED: the shipped script is already red" >&2; exit 1; }
  echo "  GREEN (expected)"

  # Mutant 1: the detector never flags anything — the state that shipped, where
  # nothing judged output at all.
  sed 's/^for sig in (non_ascii, repeated, fffd, dominant):/for sig in ():/' "$SCRIPT" > "$m1"
  cmp -s "$SCRIPT" "$m1" && { echo "SELF-TEST INCONCLUSIVE: mutant 1 changed nothing" >&2; exit 1; }
  echo "self-test: mutant 1 (detector flags nothing)"
  if run_detector_table "$m1" > /dev/null 2>&1; then
    echo "SELF-TEST FAILED: mutant 1 passed the detector table" >&2; exit 1
  fi
  echo "  RED (expected)"

  # Mutant 2: the detector still works, the VERDICT ignores it — the subtler state,
  # and the one #3901 taught us to plant separately.
  sed '/if x.get("output_bad"):/,+1d' "$SCRIPT" > "$m2"
  cmp -s "$SCRIPT" "$m2" && { echo "SELF-TEST INCONCLUSIVE: mutant 2 changed nothing" >&2; exit 1; }
  echo "self-test: mutant 2 (verdict ignores output_bad)"
  if check_verdict "$m2" > /dev/null 2>&1; then
    echo "SELF-TEST FAILED: mutant 2 still reddened the row — the verdict is not reading output_bad" >&2
    exit 1
  fi
  echo "  RED (expected)"

  # Mutant 3: the extractor cannot find a reply and says nothing about it. This is
  # the hazard the FIX introduces, not the one it removes: a silent "" leaves the
  # detector nothing to flag, and every row goes green.
  sed -e 's|.*could not find the backend envelope.*|    3) printf "" ;;|' \
      -e 's|.*could not locate the %s reply.*|    *) printf "" ;;|' "$SCRIPT" > "$m3"
  cmp -s "$SCRIPT" "$m3" && { echo "SELF-TEST INCONCLUSIVE: mutant 3 changed nothing" >&2; exit 1; }
  echo "self-test: mutant 3 (an unlocatable reply passes silently)"
  if run_extraction_table "$m3" > /dev/null 2>&1; then
    echo "SELF-TEST FAILED: mutant 3 passed -- a reply that was never found reads as clean" >&2
    exit 1
  fi
  echo "  RED (expected)"

  # Mutant 4: terminate the reply at the next `You:` instead of at the backend
  # envelope -- the PARTIAL extraction aprender-3e caught in review. A reply that
  # contains `You:` is then cut, and a fragment is judged in its place.
  sed 's|return line.strip() == "You: Goodbye!"|return line.lstrip().startswith("You:")|' "$SCRIPT" > "$m4"
  cmp -s "$SCRIPT" "$m4" && { echo "SELF-TEST INCONCLUSIVE: mutant 4 changed nothing" >&2; exit 1; }
  echo "self-test: mutant 4 (reply cut at the next You:)"
  if run_extraction_table "$m4" > /dev/null 2>&1; then
    echo "SELF-TEST FAILED: mutant 4 passed -- a truncated reply is judged as if whole" >&2
    exit 1
  fi
  echo "  RED (expected)"

  # Mutant 5 (#3957 F4c): the stream shapes are not parsed -- the state that shipped,
  # where every stream=true body was read with one json.load.
  sed 's/^elif any(ln.startswith("data:") for ln in raw.splitlines()):/elif False:/' "$SCRIPT" > "$m5"
  cmp -s "$SCRIPT" "$m5" && { echo "SELF-TEST INCONCLUSIVE: mutant 5 changed nothing" >&2; exit 1; }
  echo "self-test: mutant 5 (SSE bodies not parsed)"
  if run_body_table "$m5" > /dev/null 2>&1; then
    echo "SELF-TEST FAILED: mutant 5 passed -- an SSE body is judged without its text" >&2; exit 1
  fi
  echo "  RED (expected)"

  # Mutant 6 (#3957 F4c): a body that yields no text passes silently.
  sed 's/^  if \[ -z "${text\/\/\[\[:space:\]\]\/}" \]; then$/  if false; then/' "$SCRIPT" > "$m6"
  cmp -s "$SCRIPT" "$m6" && { echo "SELF-TEST INCONCLUSIVE: mutant 6 changed nothing" >&2; exit 1; }
  echo "self-test: mutant 6 (empty stream text reads clean)"
  if run_body_table "$m6" > /dev/null 2>&1; then
    echo "SELF-TEST FAILED: mutant 6 passed -- a stream that carried nothing reads as clean" >&2; exit 1
  fi
  echo "  RED (expected)"

  # Mutant 7 (#3957 Q6): a stream with no terminal event is judged on its prefix.
  m7=$(mktemp)
  sed 's/^  if \[ "$term" = open \]; then$/  if false; then/' "$SCRIPT" > "$m7"
  cmp -s "$SCRIPT" "$m7" && { rm -f "$m7"; echo "SELF-TEST INCONCLUSIVE: mutant 7 changed nothing" >&2; exit 1; }
  echo "self-test: mutant 7 (truncated stream judged on its prefix)"
  if run_body_table "$m7" > /dev/null 2>&1; then
    rm -f "$m7"; echo "SELF-TEST FAILED: mutant 7 passed -- a stream that never finished reads as an answer" >&2; exit 1
  fi
  rm -f "$m7"
  echo "  RED (expected)"

  echo "self-test: PASS — red when the detector stops flagging, when the verdict stops reading it, when an unlocatable reply passes silently, and when the reply is cut short"
  exit 0
fi

echo "ladder output judging: a verb that ran is not a verb that worked ($SCRIPT)"
rc=0
run_detector_table "$SCRIPT" || rc=$?
[ "$rc" = 2 ] && exit 2
check_verdict "$SCRIPT" || rc=1
run_extraction_table "$SCRIPT" || rc=$?
[ "$rc" = 2 ] && exit 2
run_body_table "$SCRIPT" || rc=$?
[ "$rc" = 2 ] && exit 2
if [ "$rc" = 0 ]; then
  echo "OK: the detector mirrors output_verification.rs, and the verdict reads it"
  exit 0
fi
echo "FAIL: a verb's output is not judged (#3921)"
exit 1
