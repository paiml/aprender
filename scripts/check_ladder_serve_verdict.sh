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
be_json() { # be_json <probed> <teardown> <http-of-first-route>
  cat <<JSON
{"cuda":{"ran":true,"fallback":false,"escaped_special":false,"rc":0,
  "verbs":{"run":{"ran":true,"rc":0},"chat":{"ran":true,"rc":0},
           "code":{"ran":true,"rc":0,"backend":"inherited-from-spawned-serve"},
           "serve":{"probed":$1,"teardown":"$2","routes":{
             "/api/chat|stream=false":{"http":$3,"ok":true},
             "/v1/completions|stream=false":{"http":200,"ok":true}}}}}}
JSON
}

QA_OK='{"capability_match":{"passed":true,"skipped":false,"message":"ok"},
        "golden_output":{"passed":true,"skipped":false,"message":"3 golden test cases passed"},
        "gates":{},"gates_failed":[],"gates_reported":2}'

run_row() { # run_row <src> <be-json> -> prints true|false (the row's green)
  local src="$1" be="$2" builder out
  builder=$(extract_builder "$src") || return 2
  out=$(printf '%s' "$builder" \
        | python3 - "rid" "$QA_OK" "$be" 0 1 "deadbeef" "m.gguf" 1 2>/dev/null) || {
    echo "  the row builder errored on this input" >&2; return 2; }
  python3 -c 'import json,sys; print(str(json.loads(sys.stdin.read())["green"]).lower())' <<< "$out"
}

# case: <name> <probed> <teardown> <http> <expected green>
CASES='healthy|true|clean|200|true
route-503|true|clean|503|false
route-500|true|clean|500|false
not-probed|false|clean|200|false
teardown-failed|true|failed|200|false'

run_table() { # run_table <src> -> 0 all as expected, 1 otherwise
  local src="$1" rc=0 name probed td http want got
  while IFS='|' read -r name probed td http want; do
    [ -n "$name" ] || continue
    got=$(run_row "$src" "$(be_json "$probed" "$td" "$http")") || return 2
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
  echo "self-test: PASS — the table turns red when green stops reading serve"
  exit 0
fi

echo "ladder serve verdict: a row's green must account for the serve probe ($SCRIPT)"
if run_table "$SCRIPT"; then
  echo "OK: a healthy serve keeps a row green; a non-200, an unprobed serve and a failed teardown each redden it"
  exit 0
else
  rc=$?
  [ "$rc" = 2 ] && exit 2
  echo "FAIL: the row's green does not account for the serve probe (#3886)"
  exit 1
fi
