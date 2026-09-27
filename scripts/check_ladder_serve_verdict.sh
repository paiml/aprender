#!/usr/bin/env bash
# check_ladder_serve_verdict.sh — a ladder row's `green` must account for the SERVE
# probe, in both directions (#3886).
#
# WHY. `ladder_serve_probe` computes a correct verdict (`[ "$code" = 200 ] || rc=1`)
# and returns it. The caller assigned it to `serve_rc` at two sites and read it at
# none — `grep -n 'serve_rc' scripts/model_ladder.sh` returned exactly two lines, both
# assignments. So every route could fail and the row stayed green:
#
#   lambda @ 87d9d5484   qwen35-0.8b-q4km   green=TRUE   qa_rc=0
#       /v1/completions  cpu+cuda, stream and no-stream  ->  503 503 503 503
#
# That is how #3874 survived to be found by a human reading route data out of a GREEN
# row instead of by this gate, and gx10's `fp16.apr` is green today with six HTTP 500s
# (#3885). A gate that computes the right answer and discards it is a fourth vacuity
# shape, distinct from one that cannot fail, one that cannot pass, and one nothing
# reaches.
#
# WHAT THIS CHECKS, AND WHY IT IS NOT A GREP. It extracts the ROW BUILDER from
# model_ladder.sh and runs it — the shipped code, on crafted rows — then asserts the
# `green` it prints. A rewrite that computes the same verdict differently passes; one
# that stops consulting serve fails. Assert the value, not the flag.
#
# BOTH DIRECTIONS ARE REQUIRED. A condition like this can be written always-false as
# easily as always-true, and an always-false one would reclassify every row in every
# committed receipt. So a healthy row must stay GREEN and each failing shape must go
# RED, and `--self-test` plants the regression to prove the table discriminates.
#
# Exit: 0 all cases as expected · 1 a case landed wrong · 2 could not check.
#       --self-test: 0 when the planted mutation turns this RED, 1 otherwise.
set -euo pipefail

SCRIPT="scripts/model_ladder.sh"
SELF_TEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --script) [ $# -ge 2 ] || { echo "--script needs a value" >&2; exit 2; }; SCRIPT="$2"; shift 2 ;;
    --self-test) SELF_TEST=1; shift ;;
    -h|--help) awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
    *) echo "check_ladder_serve_verdict: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

# The row builder, lifted from the script. Anti-vacuity: it must be non-trivial and
# must actually compute `green`, or this check has silently stopped testing anything.
extract_builder() {
  local src="$1" body
  [ -f "$src" ] || { echo "  cannot read $src" >&2; return 2; }
  # Anchored at the start of the assignment. An unanchored `row=$(python3` also matches
  # `qa_row=$(python3` sixty lines earlier — a substring collision that silently
  # extracts the WRONG heredoc, which is what it did on the first run here. (Same shape
  # as `"qwen35".contains("qwen3")` in #3883: a containment test standing in for an
  # identity test.) The `grep -q green` below is what caught it.
  body=$(awk '/^[[:space:]]*row=\$\(python3/{f=1; next} f && /^PY$/{exit} f' "$src")
  if ! grep -q 'green' <<< "$body"; then
    echo "  the extracted row builder does not mention 'green' — this check no longer knows what it is running" >&2
    return 2
  fi
  printf '%s' "$body"
}

# A backend whose serve probe is healthy, and the knobs each case turns.
be_json() { # be_json <probed> <teardown> <http-of-first-route> [chat-rc]
  # #3902: the chat rc is a knob because `chat` and `code` joined the verdict. A row
  # green on every other axis with `chat rc=3` is the live gx10 case.
  local _chat_rc="${4:-0}" _chat_ran=true
  [ "$_chat_rc" = "0" ] || _chat_ran=false
  cat <<JSON
{"cuda":{"ran":true,"fallback":false,"escaped_special":false,"rc":0,
  "verbs":{"run":{"ran":true,"rc":0},"chat":{"ran":$_chat_ran,"rc":$_chat_rc},
           "code":{"ran":true,"rc":0,"backend":"inherited-from-spawned-serve"},
           "serve":{"probed":$1,"teardown":"$2","routes":{
             "/api/chat|stream=false":{"http":$3,"ok":true},
             "/v1/completions|stream=false":{"http":200,"ok":true}}}}}}
JSON
}

QA_OK='{"capability_match":{"passed":true,"skipped":false,"message":"ok"},
        "golden_output":{"passed":true,"skipped":false,"message":"3 golden test cases passed"},
        "gates":{},"gates_failed":[],"gates_reported":2}'

run_row() { # run_row <src> <be-json> [qa-json] -> prints true|false (the row's green)
  local src="$1" be="$2" qa="${3:-$QA_OK}" builder out
  builder=$(extract_builder "$src") || return 2
  out=$(printf '%s' "$builder" \
        | python3 - "rid" "$qa" "$be" 0 1 "deadbeef" "m.gguf" 1 "qwen2" '{"verdict": "fits"}' 2>/dev/null) || {
    echo "  the row builder errored on this input" >&2; return 2; }
  python3 -c 'import json,sys; print(str(json.loads(sys.stdin.read())["green"]).lower())' <<< "$out"
}

# The WHY builder, lifted separately (#3901). A row can be RED with its only cause
# unexplained: #3886 folded serve into `green` and left this builder unchanged, so a
# row red solely on serve printed `unknown`. The verdict moved; the explanation did
# not. Measured live on gx10's fp16 `.apr` row.
extract_why() {
  local src="$1" body
  body=$(awk '/why=.*python3 -c/{f=1; next} f && /^print\(/{print; exit} f' "$src")
  if ! grep -q 'w.append' <<< "$body"; then
    echo "  the extracted why builder does not append reasons — this check no longer knows what it runs" >&2
    return 2
  fi
  # The final line carries the shell's closing `')` after the python. Strip it, or
  # the extracted program ends in an unterminated string literal.
  printf '%s' "$body" | sed "s/')\$//"
}

# A row that is green on EVERY other axis and red only because a serve route 500s.
# Each reason fixture is green on EVERY axis but one, so the reason it produces
# isolates a single cause. The first version omitted `chat`/`code` entirely and the
# reason came back naming three causes — which would have passed a substring check
# while proving nothing about which clause fired.
_verbs_ok='"run":{"ran":true,"rc":0},"chat":{"ran":true,"rc":0},"code":{"ran":true,"rc":0},'
RED_ONLY_ON_SERVE='{"capability_match":{"passed":true,"skipped":false,"message":"ok"},
 "golden_output":{"passed":true,"skipped":false,"message":"ok"},
 "backends":{"cuda":{"ran":true,"fallback":false,"escaped_special":false,"rc":0,
   "verbs":{'"$_verbs_ok"'"serve":{"probed":true,"teardown":"clean","routes":{
     "/v1/completions|stream=false":{"http":500},"/api/chat|stream=false":{"http":200}}}}}}}'

# #3902: green everywhere except `chat`, with serve healthy.
RED_ONLY_ON_CHAT='{"capability_match":{"passed":true,"skipped":false,"message":"ok"},
 "golden_output":{"passed":true,"skipped":false,"message":"ok"},
 "backends":{"cuda":{"ran":true,"fallback":false,"escaped_special":false,"rc":0,
   "verbs":{"run":{"ran":true,"rc":0},"chat":{"ran":false,"rc":3},"code":{"ran":true,"rc":0},
     "serve":{"probed":true,"teardown":"clean","routes":{
       "/v1/completions|stream=false":{"http":200}}}}}}}'

why_of() { # why_of <src> -> the reason line that row would print
  # The builder already reads its row from STDIN, so it just gets piped. (The first
  # version of this helper rebuilt it through `exec` and string-splitting, which is
  # more machinery than the thing under test.)
  local src="$1" fixture="${2:-$RED_ONLY_ON_SERVE}" why
  why=$(extract_why "$src") || return 2
  printf '%s' "$fixture" | python3 -c "$why" 2>/dev/null
}

# A row red ONLY on serve must EXPLAIN itself. `unknown` is the failure this
# guards: the verdict says red and the operator is told nothing.
check_reason() { # check_reason <src> -> 0 explained, 1 not
  local src="$1" got chat
  # #3902: a chat-only red must name the VERB, not fall through to `unknown`.
  chat=$(why_of "$src" "$RED_ONLY_ON_CHAT") || return 2
  case "$chat" in
    *'verb `chat` did not run'*) printf '  ok    %-16s %s\n' "reason:chat" "$chat" ;;
    *) printf '  FAIL  %-16s chat-only red says: %s\n' "reason:chat" "${chat:-empty}"; return 1 ;;
  esac
  got=$(why_of "$src") || return 2
  case "$got" in
    *"serve routes non-200"*)
      printf '  ok    %-16s %s\n' "reason:serve" "$got"; return 0 ;;
    ""|unknown)
      printf '  FAIL  %-16s reason is %s — the row is red and says nothing\n' "reason" "${got:-empty}"; return 1 ;;
    *)
      printf '  FAIL  %-16s reason does not name serve: %s\n' "reason" "$got"; return 1 ;;
  esac
}

# case: <name> <probed> <teardown> <http> <expected green>
CASES='healthy|true|clean|200|0|true
route-503|true|clean|503|0|false
route-500|true|clean|500|0|false
not-probed|false|clean|200|0|false
teardown-failed|true|failed|200|0|false
teardown-undetermined|true|undetermined|200|0|false
chat-rc3|true|clean|200|3|false'

# #3965: a SKIPPED golden gate in the OLD encoding (passed:true, skipped:true, which is
# what apr wrote before #3965 and what existing receipts still carry) must not make a
# healthy row green. Every other axis of this row is healthy, so skipped is the only
# thing that can turn it red.
QA_GOLDEN_SKIPPED='{"capability_match":{"passed":true,"skipped":false,"message":"ok"},
        "golden_output":{"passed":true,"skipped":true,"message":"Skipped: no engine"},
        "gates":{},"gates_failed":[],"gates_reported":2}'

golden_skip_case() { # <src> -> 0 when a skipped golden gate keeps the row red
  local got
  got=$(run_row "$1" "$(be_json true clean 200 0)" "$QA_GOLDEN_SKIPPED") || return 2
  if [ "$got" = "false" ]; then
    printf '  ok    %-16s green=%s\n' "golden-skipped" "$got"; return 0
  fi
  printf '  FAIL  %-16s green=%s, expected false — a skipped golden gate counted as a pass (#3965)\n' "golden-skipped" "$got"; return 1
}

run_table() { # run_table <src> -> 0 all as expected, 1 otherwise
  local src="$1" rc=0 name probed td http want got
  while IFS='|' read -r name probed td http chatrc want; do
    [ -n "$name" ] || continue
    got=$(run_row "$src" "$(be_json "$probed" "$td" "$http" "$chatrc")") || return 2
    if [ "$got" = "$want" ]; then
      printf '  ok    %-16s green=%s\n' "$name" "$got"
    else
      printf '  FAIL  %-16s green=%s, expected %s\n' "$name" "$got" "$want"
      rc=1
    fi
  done <<< "$CASES"
  return $rc
}

if [ "$SELF_TEST" = 1 ]; then
  [ -f "$SCRIPT" ] || { echo "cannot read $SCRIPT" >&2; exit 2; }
  mutant=$(mktemp); trap 'rm -f "$mutant"' EXIT
  # The regression this exists to catch: green stops consulting serve.
  sed 's/ and serve_ok(v)//' "$SCRIPT" > "$mutant"
  if cmp -s "$SCRIPT" "$mutant"; then
    echo 'SELF-TEST INCONCLUSIVE: the mutation changed nothing — green does not consult serve_ok in' "$SCRIPT" >&2
    exit 1
  fi
  echo "self-test: the shipped script"
  if run_table "$SCRIPT"; then echo "  GREEN (expected)"; else
    echo "SELF-TEST FAILED: the shipped script is already red" >&2; exit 1; fi
  echo "self-test: the mutant (green no longer consults serve)"
  if run_table "$mutant" > /dev/null 2>&1; then
    echo "SELF-TEST FAILED: the mutant passed the table — this check does not discriminate" >&2; exit 1
  else
    echo "  RED (expected)"
  fi
  # Mutant C (#3901): the verdict keeps serve, the EXPLANATION loses it — which is
  # exactly the state #3886 shipped and gx10 measured as `[FAIL] … unknown`.
  mutant_c=$(mktemp); trap 'rm -f "$mutant" "$mutant_c"' EXIT
  sed '/serve routes non-200/d' "$SCRIPT" > "$mutant_c"
  if cmp -s "$SCRIPT" "$mutant_c"; then
    echo "SELF-TEST INCONCLUSIVE: mutant C changed nothing" >&2; exit 1
  fi
  echo "self-test: the shipped script explains a serve-only red"
  check_reason "$SCRIPT" > /dev/null || { echo "SELF-TEST FAILED: shipped script does not explain it" >&2; exit 1; }
  echo "  GREEN (expected)"
  echo "self-test: mutant C (verdict keeps serve, explanation drops it)"
  if check_reason "$mutant_c" > /dev/null 2>&1; then
    echo "SELF-TEST FAILED: mutant C still explained the red — a row could go red saying nothing" >&2; exit 1
  else
    echo "  RED (expected)"
  fi

  # Mutant D (#3902): the verdict stops consulting the other two verbs — the state
  # that shipped in #3886, where serve was folded in and its two siblings were not.
  mutant_d=$(mktemp); trap 'rm -f "$mutant" "$mutant_c" "$mutant_d" "$mutant_e"' EXIT
  sed 's/ and verb_ok(v, "chat") and verb_ok(v, "code")//' "$SCRIPT" > "$mutant_d"
  if cmp -s "$SCRIPT" "$mutant_d"; then
    echo "SELF-TEST INCONCLUSIVE: mutant D changed nothing" >&2; exit 1
  fi
  echo "self-test: mutant D (green no longer consults chat/code)"
  if run_table "$mutant_d" > /dev/null 2>&1; then
    echo "SELF-TEST FAILED: mutant D passed the table — a chat-only failure would stay green" >&2; exit 1
  else
    echo "  RED (expected)"
  fi

  # Mutant E (#3902): the verdict keeps the verbs, the EXPLANATION drops them —
  # #3901's shape applied to the new axis, planted deliberately this time.
  mutant_e=$(mktemp)
  sed '/verb `%s` did not run/d' "$SCRIPT" > "$mutant_e"
  if cmp -s "$SCRIPT" "$mutant_e"; then
    echo "SELF-TEST INCONCLUSIVE: mutant E changed nothing" >&2; exit 1
  fi
  echo "self-test: mutant E (verdict keeps chat/code, explanation drops them)"
  if check_reason "$mutant_e" > /dev/null 2>&1; then
    echo "SELF-TEST FAILED: mutant E still explained a chat-only red" >&2; exit 1
  else
    echo "  RED (expected)"
  fi

  echo "self-test: PASS — red on all four plantings: verdict/explanation x serve/verbs"
  exit 0
fi

echo "ladder serve verdict: a row's green must account for the serve probe ($SCRIPT)"
if run_table "$SCRIPT" && check_reason "$SCRIPT" && golden_skip_case "$SCRIPT"; then
  echo "OK: a healthy serve keeps a row green; a non-200, an unprobed serve and a failed teardown each redden it AND say why"
  exit 0
else
  rc=$?
  [ "$rc" = 2 ] && exit 2
  echo "FAIL: the row's green does not account for the serve probe (#3886)"
  exit 1
fi
