#!/usr/bin/env bash
# failfast_gate.sh -- did a GATE section fail in this fat job's results? (#5738, refs #5394)
# The merge_group fail-fast step cancels the run only when a section that `ci / gate` reads is RED.
# A section outside the gate (vendored-schemas, pr-review-*, ...) stays reported but never cancels:
# cancelling on it would make a non-gate red block the queue, a gate added without a C324 rule-7 yes.
# GATE_SECTIONS must equal the `X86:<section>` pairs of the gate job's loop in ci.yml;
# scripts/check_ci_failfast.sh checks that they agree.
#   exit 0: a gate section is not success/skipped (cancel)   exit 1: every gate section is fine
#   exit 2: results unreadable (not measured; never a cancel and never a pass)
#   bash scripts/ci/failfast_gate.sh <results.json> | --self-test
set -uo pipefail
GATE_SECTIONS="guard-tree guard-cargo sov.gate"

failed_gates() {
  local f=$1 s r bad=""
  [ -s "$f" ] && jq -e . "$f" > /dev/null 2>&1 || return 2
  for s in $GATE_SECTIONS; do
    r=$(jq -r --arg s "$s" '.[$s].result // "missing"' "$f") || return 2
    case "$r" in success|skipped) ;; *) bad="$bad $s=$r" ;; esac
  done
  [ -z "$bad" ] && return 1
  printf 'gate section(s) not green:%s\n' "$bad"
  return 0
}

if [ "${1:-}" = "--self-test" ]; then
  t=$(mktemp -d); trap 'rm -rf "${t:?}"' EXIT
  ok='"guard-tree":{"result":"success"},"guard-cargo":{"result":"success"},"sov.gate":{"result":"success"}'
  put() { printf '{%s}' "$1" > "$t/$2.json"; }
  put "$ok,\"vendored-schemas\":{\"result\":\"failure\"}" nongate
  put "$ok" green
  put '"guard-tree":{"result":"success"},"guard-cargo":{"result":"failure"},"sov.gate":{"result":"success"}' cargo
  put '"guard-tree":{"result":"success"},"guard-cargo":{"result":"success"},"sov.gate":{"result":"cancelled"}' sov
  put '"guard-tree":{"result":"success"},"sov.gate":{"result":"success"}' missing
  printf 'not json' > "$t/bad.json"
  rc() { failed_gates "$t/$1.json" > /dev/null; echo $?; }
  [ "$(rc nongate)" = 1 ] || { echo "FAIL: a non-gate red must not cancel"; exit 1; }
  [ "$(rc green)" = 1 ] || { echo "FAIL: all gates green must not cancel"; exit 1; }
  [ "$(rc cargo)" = 0 ] || { echo "FAIL: guard-cargo red must cancel"; exit 1; }
  [ "$(rc sov)" = 0 ] || { echo "FAIL: sov.gate cancelled/red must cancel"; exit 1; }
  [ "$(rc missing)" = 0 ] || { echo "FAIL: a gate section absent from the results is not green"; exit 1; }
  [ "$(rc bad)" = 2 ] || { echo "FAIL: unreadable results must be not-measured (2)"; exit 1; }
  [ "$(failed_gates "$t/nofile.json" > /dev/null; echo $?)" = 2 ] || { echo "FAIL: missing results file must be 2"; exit 1; }
  echo "ok: failfast_gate case table"; exit 0
fi

failed_gates "${1:?usage: failfast_gate.sh <results.json>|--self-test}"
