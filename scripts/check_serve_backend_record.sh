#!/usr/bin/env bash
# check_serve_backend_record.sh — the ladder's serve probe must be able to record
# `used_gpu: false`, not only `true` (#3894).
#
# WHY THIS EXACT CONTROL. `http: 200` proves serve ANSWERED; it never proved serve
# answered on the accelerator. #3889 is the measured case where those differ: an
# `.apr` whose CUDA attempt declined, whose wgpu attempt declined
# ("Unsupported quantization type 30 for WGPU dequant"), and whose generation ran on
# CPU — while serve returned 200 on all six routes and the ladder recorded them in a
# `cuda` cell. `apr run` refuses that (rc=14, `registry::after_generation`); serve had
# no equivalent and still has none. This field is the reporting half.
#
# So the property that matters is NOT "the recorder works". It is:
#
#     A RECORDER THAT CANNOT REPORT `false` IS THE DEFECT, NOT THE FIX.
#
# A recorder hardwired to `true` would satisfy every test written around a working
# GPU model, make every cell look like CUDA service, and be strictly worse than
# recording nothing — because it would launder an unverified claim into the receipt.
# The planted mutant here is exactly that hardwiring.
#
# WHAT IT RUNS. The extractor is lifted out of `model_ladder.sh` and executed against
# crafted response bodies, so this tests the shipped code rather than a copy. Three
# states, because `null` is a real answer and not a failure: an arm that does not
# measure its backend, and every streaming body (SSE is not one JSON object), record
# `null` — "this cell does not establish CUDA" rather than "this cell ran on CPU".
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
    *) echo "check_serve_backend_record: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

# The extractor, lifted from the probe. Anti-vacuity: it must exist exactly once and
# must actually mention the field, or this check has stopped testing anything.
extract_reader() {
  local src="$1" body
  [ -f "$src" ] || { echo "  cannot read $src" >&2; return 2; }
  body=$(awk "/ug=\\\$\(python3 -c '/{f=1; next} f && /^' \"\\\$bodyf\"/{exit} f" "$src")
  if ! grep -q 'used_gpu' <<< "$body"; then
    echo "  the extracted reader does not mention used_gpu — this check no longer knows what it runs" >&2
    return 2
  fi
  printf '%s' "$body"
}

read_one() { # read_one <src> <body-json> -> prints true|false|null
  local src="$1" tmp reader out
  reader=$(extract_reader "$src") || return 2
  tmp=$(mktemp); printf '%s' "$2" > "$tmp"
  out=$(printf '%s' "$reader" | python3 - "$tmp" 2>/dev/null) || { rm -f "$tmp"; return 2; }
  rm -f "$tmp"
  printf '%s' "$out"
}

GPU_BODY='{"id":"cmpl-x","choices":[{"text":"hi"}],"used_gpu":true}'
CPU_BODY='{"id":"cmpl-x","choices":[{"text":"hi"}],"used_gpu":false}'
ABSENT='{"id":"cmpl-x","choices":[{"text":"hi"}]}'
SSE='data: {"id":"cmpl-x"}

data: [DONE]'

run_table() { # run_table <src> -> 0 all as expected
  local src="$1" rc=0 name want got
  while IFS='|' read -r name want; do
    [ -n "$name" ] || continue
    case "$name" in
      gpu)      got=$(read_one "$src" "$GPU_BODY") ;;
      cpu)      got=$(read_one "$src" "$CPU_BODY") ;;
      absent)   got=$(read_one "$src" "$ABSENT") ;;
      streamed) got=$(read_one "$src" "$SSE") ;;
      *) echo "  unknown case $name" >&2; return 2 ;;
    esac || return 2
    if [ "$got" = "$want" ]; then
      printf '  ok    %-10s used_gpu=%s\n' "$name" "$got"
    else
      printf '  FAIL  %-10s used_gpu=%s, expected %s\n' "$name" "$got" "$want"
      rc=1
    fi
  done <<'CASES'
gpu|true
cpu|false
absent|null
streamed|null
CASES
  return $rc
}

if [ "$SELF_TEST" = 1 ]; then
  [ -f "$SCRIPT" ] || { echo "cannot read $SCRIPT" >&2; exit 2; }
  mutant=$(mktemp); trap 'rm -f "$mutant"' EXIT
  # THE mutation: a recorder that can only say "true". It passes any test built
  # around a working GPU model and launders an unverified claim into every receipt.
  sed 's|print("null" if v is None else ("true" if v else "false"))|print("true")|' \
      "$SCRIPT" > "$mutant"
  if cmp -s "$SCRIPT" "$mutant"; then
    echo 'SELF-TEST INCONCLUSIVE: the mutation changed nothing — the reader is not where this expects' >&2
    exit 1
  fi
  echo "self-test: the shipped script"
  if run_table "$SCRIPT"; then echo "  GREEN (expected)"; else
    echo "SELF-TEST FAILED: the shipped script is already red" >&2; exit 1; fi
  echo "self-test: the mutant (a recorder hardwired to true)"
  if run_table "$mutant" > /dev/null 2>&1; then
    echo "SELF-TEST FAILED: the mutant passed — this check cannot tell a recorder from a rubber stamp" >&2
    exit 1
  else
    echo "  RED (expected)"
  fi
  echo "self-test: PASS — red when the recorder loses its ability to say false"
  exit 0
fi

echo "serve backend record: the probe must be able to report false, not only true ($SCRIPT)"
if run_table "$SCRIPT"; then
  echo "OK: true, false and null are each recorded from the response the probe received"
  exit 0
else
  rc=$?
  [ "$rc" = 2 ] && exit 2
  echo "FAIL: the serve probe does not record the backend faithfully (#3894)"
  exit 1
fi
