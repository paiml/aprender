#!/usr/bin/env bash
# check_serve_probe_evidence.sh — a failed serve probe must carry its own evidence (#3943).
#
# WHY. `WORK` is a `mktemp -d` under `trap _rm_work EXIT`. A `why` that names a serve
# log by PATH therefore names a file the run deletes before anyone reads the receipt.
# Measured on gx10: qwen35-27b-q4km's cpu serve probe "exited rather than returned",
# and the one artifact that could say why was already gone. The probe's exit status,
# which distinguishes a signal (137/143) from lock_timeout (2) from anything else, was
# captured and then overwritten with 1 before it was recorded.
#
# WHAT THIS CHECKS. The helper and both `probed:false` emitters are LIFTED from
# model_ladder.sh and run, not re-described:
#
#   1. serve_log_tail returns the last non-empty lines of a real log, and nothing
#      for an empty or missing one.
#   2. Each emitter produces VALID JSON on hostile input — a log line carrying quotes,
#      a backslash, a tab and non-ASCII. That is the input that would have broken the
#      old hand-quoted printf and made the whole row unparseable (#3847's shape).
#   3. The health-timeout `why` carries the LAST LOG LINE, and the refusal carries
#      the EXIT STATUS and the probe's stderr — so the receipt answers "what happened"
#      without the deleted directory.
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
    *) echo "check_serve_probe_evidence: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

TMPD=$(mktemp -d) || { echo "cannot create a temp dir" >&2; exit 2; }
# Only ever remove what mktemp made: never an empty value, never "/", never outside a
# temp root. Same rule as model_ladder.sh's _rm_work (SEC011).
_rm_tmpd() {
  case "${TMPD:-}" in
    /tmp/?*|/var/folders/?*) [ -d "$TMPD" ] && rm -rf -- "$TMPD" ;;
    *) : ;;
  esac
}
trap _rm_tmpd EXIT

# A serve log with the lines that break naive JSON quoting, then a blank line, then the
# line that must come out as "last".
HOSTILE="$TMPD/serve.log"
printf '%s\n' \
  'Loading GGUF model (mmap)...' \
  'path "C:\models\x" said: "bad" 	tab' \
  'Qwen3.5 — hybrid ✓ résumé' \
  '' \
  'thread main panicked at "cuda alloc": out of memory' > "$HOSTILE"
LAST='thread main panicked at "cuda alloc": out of memory'
: > "$TMPD/empty.log"
printf '%s\n' 'error: lock /tmp/apr-gpu.lock not free after 1800s' > "$TMPD/probe.err"

lift() { # lift <src> <name> -> the snippet or function body on stdout
  python3 - "$1" "$2" <<'PY'
import re, sys
src = open(sys.argv[1], encoding="utf-8").read()
which = sys.argv[2]
if which == "serve_log_tail":
    m = re.search(r"^serve_log_tail\(\) \{.*?^\}$", src, re.S | re.M)
elif which == "health":
    m = re.search(r"python3 -c '\n(import json, sys\ntail = sys\.argv\[2\].*?)\n' \"\$waited\"", src, re.S)
elif which == "refusal":
    m = re.search(r"python3 -c '\n(import json, sys\nprint\(json\.dumps\(\{\"probed\": False,\n\s+\"why\": \"the serve probe.*?)\n' \"\$b\"", src, re.S)
else:
    sys.exit(2)
if not m:
    sys.exit(1)
print(m.group(1) if m.groups() else m.group(0))
PY
}

run_cases() { # -> 0 all as expected
  local src="$1" rc=0 fn health refusal out got
  fn=$(lift "$src" serve_log_tail) || { echo "  FAIL  serve_log_tail is not defined — the evidence helper is gone"; return 1; }
  health=$(lift "$src" health)     || { echo "  FAIL  the health-timeout emitter no longer builds its JSON in python"; return 1; }
  refusal=$(lift "$src" refusal)   || { echo "  FAIL  the exited-rather-than-returned emitter no longer builds its JSON in python"; return 1; }

  # 1. the helper
  got=$(bash -c "$fn"$'\n''serve_log_tail "$1" 1' _ "$HOSTILE")
  if [ "$got" = "$LAST" ]; then echo "  ok    tail-returns-the-last-nonempty-line"
  else echo "  FAIL  tail-returns-the-last-nonempty-line   got [$got]"; rc=1; fi
  got=$(bash -c "$fn"$'\n''serve_log_tail "$1"' _ "$TMPD/empty.log")
  if [ -z "$got" ]; then echo "  ok    tail-of-an-empty-log-is-nothing"
  else echo "  FAIL  tail-of-an-empty-log-is-nothing   got [$got]"; rc=1; fi
  got=$(bash -c "$fn"$'\n''serve_log_tail "$1"' _ "$TMPD/does-not-exist.log")
  if [ -z "$got" ]; then echo "  ok    tail-of-a-missing-log-is-nothing"
  else echo "  FAIL  tail-of-a-missing-log-is-nothing   got [$got]"; rc=1; fi

  # 2+3. the health-timeout emitter, on hostile input
  local tail_txt; tail_txt=$(bash -c "$fn"$'\n''serve_log_tail "$1"' _ "$HOSTILE")
  out=$(python3 -c "$health" 90 "$tail_txt" clean 2>&1) || true
  if python3 -c '
import json, sys
d = json.loads(sys.argv[1]); last = sys.argv[2]
assert d["probed"] is False
assert last in d["why"], "why lacks the last log line"
assert d["log_tail"] and last in d["log_tail"]
' "$out" "$LAST" 2>/dev/null; then echo "  ok    health-timeout-json-carries-the-last-line"
  else echo "  FAIL  health-timeout-json-carries-the-last-line   got [${out:0:160}]"; rc=1; fi

  # #3943 merge: the emitter also names WHAT ended the wait, and the bounds, when told.
  out=$(python3 -c "$health" 34 "$tail_txt" escalated stalled 90 900 2>&1) || true
  if python3 -c 'import json,sys; d=json.loads(sys.argv[1]); w=d["why"]; assert "stalled after 34s" in w and "stall window 90s" in w and "ceiling 900s" in w and d["teardown"] == "escalated"' "$out" 2>/dev/null
  then echo "  ok    health-timeout-names-the-wait-verdict"
  else echo "  FAIL  health-timeout-names-the-wait-verdict   got [${out:0:160}]"; rc=1; fi

  out=$(python3 -c "$health" 90 "" clean 2>&1) || true
  if python3 -c 'import json,sys; d=json.loads(sys.argv[1]); assert d["log_tail"] is None and "empty" in d["why"]' "$out" 2>/dev/null
  then echo "  ok    health-timeout-with-no-log-says-so"
  else echo "  FAIL  health-timeout-with-no-log-says-so   got [${out:0:160}]"; rc=1; fi

  # 2+3. the refusal emitter: exit status + stderr + log, on hostile input
  local err_txt; err_txt=$(bash -c "$fn"$'\n''serve_log_tail "$1" 6' _ "$TMPD/probe.err")
  out=$(python3 -c "$refusal" cpu 137 "$err_txt" "$tail_txt" 2>&1) || true
  if python3 -c '
import json, sys
d = json.loads(sys.argv[1])
assert d["probed"] is False
assert d["probe_exit"] == 137 and "137" in d["why"]
assert "lock" in (d["probe_stderr_tail"] or "")
assert sys.argv[2] in (d["log_tail"] or "")
' "$out" "$LAST" 2>/dev/null; then echo "  ok    refusal-carries-status-stderr-and-log"
  else echo "  FAIL  refusal-carries-status-stderr-and-log   got [${out:0:160}]"; rc=1; fi
  return $rc
}

if [ "$SELF_TEST" = 1 ]; then
  [ -f "$SCRIPT" ] || { echo "cannot read $SCRIPT" >&2; exit 2; }
  m1="$TMPD/m1"; m2="$TMPD/m2"; m3="$TMPD/m3"

  echo "self-test: the shipped script"
  run_cases "$SCRIPT" > /dev/null || { echo "SELF-TEST FAILED: the shipped script is already red" >&2; exit 1; }
  echo "  GREEN (expected)"

  # Mutant 1: the helper returns nothing — the state before this fix, where the
  # receipt held a path to a deleted file and no content.
  sed '/^serve_log_tail() {/,/^}/{s/^  grep -v .*$/  return 0/}' "$SCRIPT" > "$m1"
  cmp -s "$SCRIPT" "$m1" && { echo "SELF-TEST INCONCLUSIVE: mutant 1 changed nothing" >&2; exit 1; }
  echo "self-test: mutant 1 (the log tail is dropped)"
  if run_cases "$m1" > /dev/null 2>&1; then echo "SELF-TEST FAILED: mutant 1 passed" >&2; exit 1; fi
  echo "  RED (expected)"

  # Mutant 2: the exit status is not recorded — it was overwritten with 1 before.
  sed 's/"probe_exit": int(sys.argv\[2\]),/"probe_exit": 1,/' "$SCRIPT" > "$m2"
  cmp -s "$SCRIPT" "$m2" && { echo "SELF-TEST INCONCLUSIVE: mutant 2 changed nothing" >&2; exit 1; }
  echo "self-test: mutant 2 (the probe's exit status is discarded)"
  if run_cases "$m2" > /dev/null 2>&1; then echo "SELF-TEST FAILED: mutant 2 passed" >&2; exit 1; fi
  echo "  RED (expected)"

  # Mutant 3: the last line is read from the wrong end — a tail that silently reports
  # the FIRST line would look like evidence and point at the start of a healthy load.
  sed 's/last = tail.splitlines()\[-1\]/last = tail.splitlines()[0]/' "$SCRIPT" > "$m3"
  cmp -s "$SCRIPT" "$m3" && { echo "SELF-TEST INCONCLUSIVE: mutant 3 changed nothing" >&2; exit 1; }
  echo "self-test: mutant 3 (the why quotes the FIRST log line)"
  if run_cases "$m3" > /dev/null 2>&1; then echo "SELF-TEST FAILED: mutant 3 passed" >&2; exit 1; fi
  echo "  RED (expected)"

  echo "self-test: PASS — red when the tail is dropped, when the exit status is discarded, and when the wrong line is quoted"
  exit 0
fi

echo "serve probe evidence: a failed probe must carry what happened, not a path to a deleted file ($SCRIPT)"
if run_cases "$SCRIPT"; then
  echo "OK: both probed:false emitters carry their own evidence as valid JSON"
  exit 0
fi
echo "FAIL: a failed serve probe would record no usable evidence (#3943)"
exit 1
