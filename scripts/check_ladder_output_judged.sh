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
# WHAT THIS CHECKS, IN TWO LAYERS.
#
#   1. THE DETECTOR MIRRORS ITS SOURCE. The four signals are
#      `output_verification.rs`'s, in its order, with its thresholds: non-ASCII
#      saturation >60%, a 4+ byte fragment repeated 3x, U+FFFD density >1/32,
#      dominant character >=90%. Two judges that disagree about what "degenerate"
#      means on the same completion are worse than one judge, so the table below
#      pins the boundaries rather than trusting that they were copied.
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

if [ "$SELF_TEST" = 1 ]; then
  [ -f "$SCRIPT" ] || { echo "cannot read $SCRIPT" >&2; exit 2; }
  m1=$(mktemp); m2=$(mktemp); trap 'rm -f "$m1" "$m2"' EXIT

  echo "self-test: the shipped script"
  run_detector_table "$SCRIPT" > /dev/null && check_verdict "$SCRIPT" > /dev/null \
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

  echo "self-test: PASS — red when the detector stops flagging AND when the verdict stops reading it"
  exit 0
fi

echo "ladder output judging: a verb that ran is not a verb that worked ($SCRIPT)"
rc=0
run_detector_table "$SCRIPT" || rc=$?
[ "$rc" = 2 ] && exit 2
check_verdict "$SCRIPT" || rc=1
if [ "$rc" = 0 ]; then
  echo "OK: the detector mirrors output_verification.rs, and the verdict reads it"
  exit 0
fi
echo "FAIL: a verb's output is not judged (#3921)"
exit 1
