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

# #3928: the same row with `run` as the offender. Planted separately from the chat
# fixture because "the detector stopped flagging" and "the verdict stopped reading it"
# are different failures, and #3901 was the fix that only closed one of them.
row_run_garbage() {
cat <<'JSON'
{"cuda":{"ran":true,"fallback":false,"escaped_special":false,"rc":0,
 "verbs":{"run":{"ran":true,"rc":0,"output_bad":"gibberish (fragment ' zombie' repeats 3+ times)"},
          "chat":{"ran":true,"rc":0},
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
  local rc=0
  for case in chat run; do
    if [ "$case" = chat ]; then g=$(green_of "$src" "$(row_chat_garbage)") || return 2
    else g=$(green_of "$src" "$(row_run_garbage)") || return 2; fi
    if [ "$g" = "false" ]; then
      printf '  ok    %-30s green=false\n' "$case-rc0-bad-output"
    else
      printf '  FAIL  %-30s green=%s -- a verb that ran and produced garbage is recorded as working\n' \
        "$case-rc0-bad-output" "$g"; rc=1
    fi
  done
  return $rc
}

# A red row must SAY why (#3932 re-review). The verdict reads `run` since #3928; the
# explanation builder did not, so a row red only on run output printed `unknown` --
# #3901/#3902's defect, a third time. Same extraction as check_ladder_serve_verdict.sh.
extract_why() {
  local src="$1" body
  body=$(awk '/why=.*python3 -c/{f=1; next} f && /^print\(/{print; exit} f' "$src")
  grep -q 'w.append' <<< "$body" || { echo "  the extracted why builder does not append reasons" >&2; return 2; }
  printf '%s' "$body" | sed "s/')\$//"
}

RED_ONLY_ON_RUN_OUTPUT='{"capability_match":{"passed":true,"skipped":false,"message":"ok"},
 "golden_output":{"passed":true,"skipped":false,"message":"ok"},
 "backends":'"$(row_run_garbage)"'}'

check_run_reason() { # -> 0 the run-only red names run, 1 it does not
  local src="$1" why got
  why=$(extract_why "$src") || return 2
  got=$(printf '%s' "$RED_ONLY_ON_RUN_OUTPUT" | python3 -c "$why" 2>/dev/null)
  case "$got" in
    *'verb `run` RAN (rc=0) and produced bad output'*)
      printf '  ok    %-30s %s\n' "reason:run-output" "${got:0:90}" ;;
    *) printf '  FAIL  %-30s a row red only on run output says: %s\n' "reason:run-output" "${got:-empty}"; return 1 ;;
  esac
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

# #3928: `apr run --verbose`. VERBATIM from gx10 -- the chatter is the point: kernel
# counts, VRAM figures and a list of hex pointers, none of which is the model speaking.
cap_run_verbose() { cat <<'T'
[GH-480] Patched 2 backward branch(es) for sm_121 JIT workaround
[trueno#243] Manual graph: 591 kernels. first_args=Some(["0xe326a87db200", "0xe326a87e2e00", "0xe326b87a4000"]), last_args=Some(["0xe326e0320000", "0xe32427c00000"])
Backend: GPU (NVIDIA GB10, 122502 MB VRAM)
[DEBUG] generated token ids: [151668, 271, 785, 6722, 315, 9625, 374, 12095, 13]

Output:
The capital of France is Paris.

Completed in 5.08s (cached)
T
}
cap_run_degenerate() { cat <<'T'
Backend: GPU (NVIDIA GB10, 122502 MB VRAM)

Output:
zombie zombie zombie zombie

Completed in 5.08s (cached)
T
}
# `Output:` opened and never closed -- the process was cut off mid-reply, so whatever
# is there is a fragment. A refusal to judge, never a clean reply.
cap_run_unterminated() { cat <<'T'
Backend: GPU (NVIDIA GB10, 122502 MB VRAM)

Output:
The capital of France is Paris.
T
}

# name|capture-fn|expect   (clean = no reason · bad = a verdict about the reply ·
# red:<what> = a refusal to judge, which must never be silent, NAMING what is missing:
# envelope · reply · unclosed. The first cut of #3928 swapped exit codes 3 and 4, so
# a capture WITH an envelope was reported as missing one; a bare `red` could not see it.)
extraction_cases() {
cat <<'CASES'
chrome-wrapped-correct-answer|cap_correct_answer|clean
code-json-envelope-not-uuid|cap_code_json|clean
chrome-does-not-mask-gibberish|cap_chrome_gibberish|bad
reply-containing-You-judged-whole|cap_reply_with_you|bad
truncated-capture-no-envelope|cap_no_envelope|red:envelope
envelope-but-no-reply|cap_no_reply|red:reply
run-verbose-chatter-not-judged|cap_run_verbose|clean
run-degenerate-reply-in-block|cap_run_degenerate|bad
run-output-never-closed|cap_run_unterminated|red:unclosed
CASES
}

run_extraction_table() { # -> 0 all as expected
  local src="$1" rc=0 name fn want got cls
  while IFS='|' read -r name fn want; do
    [ -n "$name" ] || continue
    got=$(judge_capture "$src" "$($fn)") || return 2
    if [ -z "$got" ]; then cls=clean
    else
      case "$got" in
        "could not find the backend envelope"*) cls=red:envelope ;;
        "could not locate the"*)                cls=red:reply ;;
        "could not find the closing"*)          cls=red:unclosed ;;
        "could not"*)                           cls=red:unnamed ;;
        *)                                      cls=bad ;;
      esac
    fi
    if [ "$cls" = "$want" ]; then
      printf '  ok    %-34s %s\n' "$name" "$cls"
    else
      printf '  FAIL  %-34s %s, expected %s  [%s]\n' "$name" "$cls" "$want" "${got:0:70}"; rc=1
    fi
  done < <(extraction_cases)
  return $rc
}

if [ "$SELF_TEST" = 1 ]; then
  [ -f "$SCRIPT" ] || { echo "cannot read $SCRIPT" >&2; exit 2; }
  m1=$(mktemp); m2=$(mktemp); m3=$(mktemp); m4=$(mktemp); m5=$(mktemp); m6=$(mktemp); m7=$(mktemp); m8=$(mktemp)
  trap 'rm -f "$m1" "$m2" "$m3" "$m4" "$m5" "$m6" "$m7" "$m8"' EXIT

  echo "self-test: the shipped script"
  run_detector_table "$SCRIPT" > /dev/null && check_verdict "$SCRIPT" > /dev/null \
    && check_run_reason "$SCRIPT" > /dev/null && run_extraction_table "$SCRIPT" > /dev/null \
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

  # Mutant 5: the extractor is bypassed and the raw capture is judged again -- the
  # exact state #3925 fixed, and so the one regression this file exists to catch.
  # aprender-3e ran it by hand while folding #3926 and it was caught; naming it here
  # means the next person does not have to re-derive that, and a refactor that
  # reintroduces raw judging fails rather than being noticed in review.
  sed 's#| assistant_reply); rc=$?#| cat); rc=$?#' "$SCRIPT" > "$m5"
  cmp -s "$SCRIPT" "$m5" && { echo "SELF-TEST INCONCLUSIVE: mutant 5 changed nothing" >&2; exit 1; }
  echo "self-test: mutant 5 (extractor bypassed, the transcript judged raw again)"
  if run_extraction_table "$m5" > /dev/null 2>&1; then
    echo "SELF-TEST FAILED: mutant 5 passed -- the chrome is being judged and nothing noticed" >&2
    exit 1
  fi
  echo "  RED (expected)"

  # Mutant 6 (#3928): when `Completed in` never arrives, take the rest of the capture
  # instead of refusing. A cut-off run then reads as a clean reply -- the silent-pass
  # hazard again, on the path added for `run`.
  sed -e 's|^elif oi >= 0 and ci >= 0:|elif oi >= 0:|' \
      -e 's|^    body = lines\[oi + 1:ci\]|    body = lines[oi + 1:]|' "$SCRIPT" > "$m6"
  cmp -s "$SCRIPT" "$m6" && { echo "SELF-TEST INCONCLUSIVE: mutant 6 changed nothing" >&2; exit 1; }
  echo "self-test: mutant 6 (an unterminated run reply is taken as whole)"
  if run_extraction_table "$m6" > /dev/null 2>&1; then
    echo "SELF-TEST FAILED: mutant 6 passed -- a cut-off run reply reads as clean" >&2
    exit 1
  fi
  echo "  RED (expected)"

  # Mutant 7 (#3928): `run` drops out of the green expression while chat and code stay.
  # Mutant 2 removes the output_bad check for EVERY verb at once, so it cannot tell
  # whether `run` was ever wired in -- which is exactly how #3886 folded serve and left
  # chat and code unread in the same structure. One verb, one mutant.
  sed 's|and verb_ok(v, "run") and verb_ok(v, "chat")|and verb_ok(v, "chat")|' "$SCRIPT" > "$m7"
  cmp -s "$SCRIPT" "$m7" && { echo "SELF-TEST INCONCLUSIVE: mutant 7 changed nothing" >&2; exit 1; }
  echo "self-test: mutant 7 (green stops reading the run verb)"
  if check_verdict "$m7" > /dev/null 2>&1; then
    echo "SELF-TEST FAILED: mutant 7 passed -- run garbage no longer reddens the row" >&2
    exit 1
  fi
  echo "  RED (expected)"

  # Mutant 8 (#3932 re-review): the explanation builder stops naming `run`. The row
  # stays red (mutant 7's check still passes) and says `unknown` -- the verdict moved
  # and the explanation did not.
  sed 's|for _vn in ("run", "chat", "code"):|for _vn in ("chat", "code"):|' "$SCRIPT" > "$m8"
  cmp -s "$SCRIPT" "$m8" && { echo "SELF-TEST INCONCLUSIVE: mutant 8 changed nothing" >&2; exit 1; }
  echo "self-test: mutant 8 (the red reason stops naming the run verb)"
  if check_run_reason "$m8" > /dev/null 2>&1; then
    echo "SELF-TEST FAILED: mutant 8 passed -- a row red only on run output explains nothing" >&2
    exit 1
  fi
  echo "  RED (expected)"

  echo "self-test: PASS — red when the detector stops flagging, when the verdict stops reading it, when an unlocatable reply passes silently, when the reply is cut short, when the extractor is bypassed entirely, when an unterminated run reply is taken as whole, when green stops reading the run verb, and when the red reason stops naming it"
  exit 0
fi

echo "ladder output judging: a verb that ran is not a verb that worked ($SCRIPT)"
rc=0
run_detector_table "$SCRIPT" || rc=$?
[ "$rc" = 2 ] && exit 2
check_verdict "$SCRIPT" || rc=1
check_run_reason "$SCRIPT" || rc=1
run_extraction_table "$SCRIPT" || rc=$?
[ "$rc" = 2 ] && exit 2
if [ "$rc" = 0 ]; then
  echo "OK: the detector mirrors output_verification.rs, and the verdict reads it"
  exit 0
fi
echo "FAIL: a verb's output is not judged (#3921)"
exit 1
