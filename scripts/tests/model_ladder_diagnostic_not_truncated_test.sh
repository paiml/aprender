#!/usr/bin/env bash
# #3872 — the diagnostic must survive into the RECEIPT and into the producer's
# printed reason, not just into the judge's display.
#
# Four hard slices cut model diagnostics: the receipt's [:200], the producer's
# two [:70] `why` lines, and the judge's two [:60] display lines. The judge half
# is covered by the self-test case `red-diagnostic-survives-the-display-3872`.
# THIS covers the producer half, end to end, through the documented seams
# (DOGFOOD_ALLOW_UNPINNED + $APR, MODEL_LADDER_ROOT, MODEL_LADDER_INVENTORY_DIRS,
# MODEL_LADDER_GPU_LOCK, --out).
#
# ONE assertion, on a phrase that sits at char 208 of a message the code really
# emits (`unclosed_think_reason`, golden_output.rs) — beyond ALL FOUR old
# slices. Not "the message is longer than N": the phrase is the classification
# and the instruction, which is what the slices were destroying.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PHRASE='check that the prompt'
MSG="golden_output_thinking_on: think block unclosed within 2048 tokens (the model was still reasoning at the budget; 8000 chars generated, no answer was reached). This is not an empty answer — raise nothing, and ${PHRASE} is the one production sends for this architecture."

T=$(mktemp -d) || exit 2
# SEC011: validate before `rm -rf`. An empty or root $T must never reach it —
# the same guard scripts/dogfood.sh's `_rm_worklog` uses.
cleanup() {
  case "${T:-}" in
    /tmp/?*|/var/folders/?*|/mnt/?*) [ -d "$T" ] && rm -rf -- "$T" ;;
    *) return 0 ;;
  esac
}
trap cleanup EXIT
mkdir -p "$T/models" "$T/out"
: > "$T/models/fixture-q4_k_m.gguf"

# A stub apr: every verb the producer calls, with a qa report whose
# golden_output carries the long real diagnostic.
cat > "$T/apr" <<STUB
#!/usr/bin/env bash
case "\$1" in
  --version) echo "apr 0.69.1 (testsha0)" ;;
  qa) python3 -c 'import json,sys;print(json.dumps({"gates":[
        {"name":"capability_match","passed":True,"skipped":False,"message":"ok"},
        {"name":"golden_output","passed":False,"skipped":False,"message":sys.argv[1]}]}))' "\$MSG_ENV" ;;
  *) [ "\${2:-}" = "--help" ] && echo "--json --max-tokens --port --no-gpu --prompt --model" ; exit 0 ;;
esac
exit 0
STUB
chmod +x "$T/apr"
export MSG_ENV="$MSG"

out=$(cd "$ROOT" && DOGFOOD_ALLOW_UNPINNED=1 APR="$T/apr" \
      MODEL_LADDER_ROOT="$ROOT" \
      MODEL_LADDER_INVENTORY_DIRS="$T/models" \
      MODEL_LADDER_GPU_LOCK="$T/lock" MODEL_LADDER_LOCK_WAIT=10 \
      timeout 600 bash scripts/model_ladder.sh --host lambda --out "$T/out" 2>&1)
rc=$?

fail=0
receipt=$(ls "$T/out"/*.json 2>/dev/null | head -1)
if [ -z "$receipt" ]; then
  printf 'ENV   no receipt written (rc=%s) — the fixture could not exercise the producer\n' "$rc"
  printf '%s\n' "$out" | tail -5
  exit 2
fi
if grep -qF -- "$PHRASE" "$receipt"; then
  printf 'ok    receipt carries the whole diagnostic\n'
else
  printf 'FAIL  receipt TRUNCATED — it does not contain: %s\n' "$PHRASE" >&2
  printf '      receipt message: %s\n' "$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print([g.get("message") for r in d.get("rungs",[])+d.get("inventory",[]) if isinstance(r,dict) for g in [r.get("golden_output") or {}]][:1])' "$receipt")" >&2
  fail=1
fi
if printf '%s' "$out" | grep -qF -- "$PHRASE"; then
  printf 'ok    the producer printed reason carries it too\n'
else
  printf 'FAIL  producer reason TRUNCATED (the [:70] why line)\n' >&2
  fail=1
fi
[ "$fail" -eq 0 ] && printf 'PASS  #3872 the diagnostic survives into the receipt and the printed reason\n'
exit "$fail"
