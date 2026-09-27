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
# ONE assertion, on a phrase that sits at char 285 of a message the code really
# emits (`unclosed_think_reason`, golden_output.rs) — beyond ALL FOUR old
# slices. Not "the message is longer than N": the phrase is the classification
# and the instruction, which is what the slices were destroying.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PHRASE='thinking-budgets-v1.yaml'

# THE FIXTURE MUST TRACK THE SOURCE (#3907).
#
# $MSG below is a LITERAL. That is deliberate — the harness needs a message long enough
# to cross the old slices without running a model — but a self-supplied fixture cannot
# fail when the real message regresses, which is the one event this guard exists for.
# It happened: this file pinned the pre-#3907 wording and asserted in its own header that
# the phrase sat "at char 208 of a message the code really emits". It passed anyway, green
# against a message that no longer shipped. The fixture had become the thing it was
# guarding — an artifact outliving what it describes, still passing.
#
# So the literal is now tied to the emitter: $PHRASE must still occur in the source that
# builds the message. If someone rewrites `unclosed_think_reason` and drops it, this fails
# HERE, naming the file, instead of going green over a fixture nothing produces.
EMITTER="$ROOT/crates/apr-cli/src/commands/output_verification.rs"
if [ ! -f "$EMITTER" ]; then
  echo "FAIL fixture-tracks-source: $EMITTER not found — the fixture cannot be checked against its emitter"
  exit 1
fi
if ! grep -qF -- "$PHRASE" "$EMITTER"; then
  echo "FAIL fixture-tracks-source: this test's \$PHRASE ('$PHRASE') no longer occurs in"
  echo "     $EMITTER."
  echo "     The fixture below is a literal, so it would keep passing against a message the"
  echo "     code no longer emits. Update \$MSG and \$PHRASE to the real message, or this"
  echo "     guard is testing its own string (#3907)."
  exit 1
fi
MSG="golden_output_thinking_on: think block unclosed within 2048 tokens (the model was still reasoning at the budget; 8000 chars generated, no answer was reached). This is not an empty answer. Before treating it as a model defect, check whether 2048 is MEASURED for this model in contracts/${PHRASE} or inherited from \`default\` — the default's basis is one 8B model (#3907)."

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
# Here-string, NOT a pipe. `producer | grep -q` under pipefail takes the
# PRODUCER's status, and `grep -q` closes the pipe on its first match -- so a
# MATCH can make the producer die and the `if` go false BECAUSE the check
# succeeded. Measured at ~10,000 matching lines when the match is EARLY (#3864),
# and this harness's `$out` is a whole ladder run's output. check_no_pipe_into_grep_q.sh
# refuses the pipe form outright: do not raise its ceiling.
if grep -qF -- "$PHRASE" <<< "$out"; then
  printf 'ok    the producer printed reason carries it too\n'
else
  printf 'FAIL  producer reason TRUNCATED (the [:70] why line)\n' >&2
  fail=1
fi
[ "$fail" -eq 0 ] && printf 'PASS  #3872 the diagnostic survives into the receipt and the printed reason\n'
exit "$fail"
