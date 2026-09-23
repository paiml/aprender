#!/usr/bin/env bash
# check_ladder_order_verdicts.sh — the ladder's cell ORDER must never change a cell's VERDICT (#4039).
#
# WHY. #4039 made model_ladder.sh measure its cells in one planned order: CRUX-certified models first, then
# largest first. Order is a schedule. If any cell can leave state behind that the NEXT cell reads, the order
# picks which model takes the damage, and a reorder turns one model's green into another's RED. #4055 is the
# live shape: a serve orphan outlived its cell and stalled or broke the next one. A reorder test whose cells
# share nothing cannot see that. Checking that the order is a permutation (scripts/lib/ladder_order.py) is
# necessary and not sufficient.
#
# WHAT THIS CHECKS, AND WHY IT IS NOT A GREP. It runs the SHIPPED model_ladder.sh end to end, twice, against
# a temp root (MODEL_LADDER_ROOT) with a two-rung ladder, one inventory-only model and a fake `apr`, in
# OPPOSITE orders. It picks the
# order with --certification, the knob #4039 added. Then it compares every cell's row, normalized for scratch
# paths, across the two runs.
#   clean   the fake leaves nothing behind           -> rows identical in both orders   (must be GREEN)
#   leaky   the fake's `run` on one model leaves a detached "VRAM hog" orphan that outlives its cell, and any
#           later `run` fails while it lives (the #4055 shape) -> the rows differ with order (must be RED:
#           this is the positive control that the comparison can see order-dependence at all)
# Both orders must also measure the same SET of cells in the order the plan asked for, or the comparison
# compares nothing.
#
# Usage: bash scripts/check_ladder_order_verdicts.sh [--self-test]
# Exit: 0 clean is order-independent AND leaky is caught · 1 a case landed wrong · 2 could not check.
#       --self-test: 0 when a planted defect in the comparison (it ignores the rows) turns this RED.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
SELF_TEST=0
[ "${1:-}" = "--self-test" ] && SELF_TEST=1
SCRIPT="scripts/model_ladder.sh"
ORDER="scripts/lib/ladder_order.py"
for f in "$SCRIPT" "$ORDER"; do [ -f "$f" ] || { echo "cannot check: $f missing" >&2; exit 2; }; done
command -v flock > /dev/null && command -v choom > /dev/null || { echo "cannot check: flock/choom missing" >&2; exit 2; }
python3 -c 'import yaml' 2> /dev/null || { echo "cannot check: python3 yaml missing" >&2; exit 2; }
# The planner's own case table first: order, totality, and the must-RED permutation guard.
st=$(python3 "$ORDER" --selftest 2>&1); st_rc=$?
if [ "$st_rc" -ne 0 ]; then echo "FAIL: $ORDER --selftest (rc $st_rc):"; printf '%s\n' "$st" | grep -v '^ok'; exit 1; fi
echo "ok: $ORDER --selftest ($(printf '%s\n' "$st" | tail -1))"

TMP=$(mktemp -d) || exit 2
HOGS="$TMP/hogs"
cleanup() {
  if [ -d "$HOGS" ]; then
    for f in "$HOGS"/*.pid; do [ -f "$f" ] && kill -9 "$(cat "$f")" 2> /dev/null; done
  fi
  rm -rf "$TMP"
}
trap cleanup EXIT
mkdir -p "$HOGS" "$TMP/models" "$TMP/inv"

# ---- the temp root: the shipped script and planner, a two-rung ladder, a minimal crate for VERSION
ROOT="$TMP/root"
mkdir -p "$ROOT/scripts/lib" "$ROOT/contracts" "$ROOT/src" "$ROOT/crates/aprender-serve/src/api"
cp "$SCRIPT" "$ROOT/scripts/model_ladder.sh"
cp "$ORDER" "$ROOT/scripts/lib/ladder_order.py"
printf '[package]\nname = "ladder-order-fixture"\nversion = "0.0.1"\nedition = "2021"\n' > "$ROOT/Cargo.toml"
: > "$ROOT/src/lib.rs"
printf '"/v1/chat/completions"\n' > "$ROOT/crates/aprender-serve/src/api/router.rs"
head -c 4096 /dev/zero > "$TMP/models/leaky-big.gguf"     # the larger file: first by size
head -c 2048 /dev/zero > "$TMP/models/victim-small.gguf"
head -c 3000 /dev/zero > "$TMP/inv/middle-inv.gguf"   # an INVENTORY-only model, sized between the two rungs
LEAKY_SHA=$(sha256sum "$TMP/models/leaky-big.gguf" | cut -d' ' -f1)
VICTIM_SHA=$(sha256sum "$TMP/models/victim-small.gguf" | cut -d' ' -f1)
[ "$LEAKY_SHA" != "$VICTIM_SHA" ] || { head -c 2047 /dev/zero > "$TMP/models/victim-small.gguf"; printf 'v' >> "$TMP/models/victim-small.gguf"; VICTIM_SHA=$(sha256sum "$TMP/models/victim-small.gguf" | cut -d' ' -f1); }
cat > "$ROOT/contracts/model-capability-ladder-v1.yaml" <<EOF
ladder:
  serve_health: {stall_s: 2, ceiling_s: 5}
  inventory: {dirs: ["$TMP/inv"], patterns: ["*-inv.gguf"], backends: [cpu]}
  rungs:
    - {id: leaky, gguf: leaky-big.gguf, sha256: $LEAKY_SHA, backends: [cpu], required: true}
    - {id: victim, gguf: victim-small.gguf, sha256: $VICTIM_SHA, backends: [cpu], required: true}
EOF
printf '{"admitted_by_sha":{"%s":["p"]}}' "$LEAKY_SHA" > "$TMP/cert-leaky-first.json"
printf '{"admitted_by_sha":{"%s":["p"]}}' "$VICTIM_SHA" > "$TMP/cert-victim-first.json"

# ---- the fake apr. Deterministic on every verb; serve refuses at once so no port is involved.
cat > "$TMP/apr" <<'PY'
#!/usr/bin/env python3
import json, os, subprocess, sys
a = sys.argv[1:]
hogs = os.environ["FAKE_HOGS"]
if a[:1] == ["--version"]:
    print("apr 0.0.1-fake"); sys.exit(0)
if len(a) >= 2 and a[-1] == "--help":
    print("--gpu --no-gpu --json --verbose"); sys.exit(0)
verb = a[0] if a else ""
model = next((x for x in a if x.endswith(".gguf")), "")
def hog_alive():
    for f in os.listdir(hogs):
        try:
            os.kill(int(open(os.path.join(hogs, f)).read()), 0); return True
        except (OSError, ValueError):
            pass
    return False
if verb == "qa":
    print(json.dumps({"gates": [{"name": "capability_match", "passed": True, "message": "ok"},
                                {"name": "golden_output", "passed": True, "message": "ok"}]})); sys.exit(0)
if verb == "inspect":
    print(json.dumps({"architecture": "qwen2"})); sys.exit(0)
if verb in ("code", "chat"):
    print(json.dumps({"content": "The capital of France is Paris."})); sys.exit(0)
if verb == "run":
    if hog_alive():
        print("CUDA error: out of memory (a previous cell's orphan holds the device)", file=sys.stderr); sys.exit(1)
    print("The capital of France is Paris.")
    if os.environ.get("FAKE_LEAK") == "1" and "leaky" in os.path.basename(model):
        # the #4055 shape: a detached orphan outlives this cell, outside any process tree the ladder tracks
        p = subprocess.Popen(["sleep", "300"], start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        open(os.path.join(hogs, "%d.pid" % p.pid), "w").write(str(p.pid))
    sys.exit(0)
if verb == "serve":
    print("fake serve: refusing to bind (deterministic)"); sys.exit(1)
print("fake apr: unhandled " + " ".join(a), file=sys.stderr); sys.exit(3)
PY
chmod +x "$TMP/apr"

# ---- one ladder run -> {"order": [ids, as measured], "rows": {id: normalized row}}
ladder_run() { # <leak 0|1> <certification> <tag>
  local out="$TMP/out-$3" log="$TMP/log-$3" rc
  for f in "$HOGS"/*.pid; do [ -f "$f" ] && { kill -9 "$(cat "$f")" 2> /dev/null; rm -f "$f"; }; done
  FAKE_LEAK="$1" FAKE_HOGS="$HOGS" DOGFOOD_ALLOW_UNPINNED=1 APR="$TMP/apr" APR_MODELS_DIR="$TMP/models" \
    MODEL_LADDER_ROOT="$ROOT" MODEL_LADDER_GPU_LOCK="$TMP/gpu.lock" MODEL_LADDER_LOCK_WAIT=5 \
    MODEL_LADDER_INVENTORY_DIRS="$TMP/inv" \
    timeout 300 bash "$ROOT/scripts/model_ladder.sh" --host lambda --out "$out" --certification "$2" > "$log" 2>&1
  rc=$?
  if [ "$rc" -eq 2 ] || [ ! -f "$out/lambda.json" ]; then
    echo "cannot check: the ladder declined (rc $rc); tail of its log:" >&2; tail -5 "$log" >&2; return 2
  fi
  python3 - "$out/lambda.json" "$log" <<'PY'
import json, re, sys
R = json.load(open(sys.argv[1]))
rows = R.get("rows") or R.get("rungs") or []
def norm(v):
    if isinstance(v, dict):
        return {k: norm(x) for k, x in sorted(v.items()) if not re.search(r"(^t_|_s$|_ms$|waited|log_tail|stderr_tail|engine_log|path)", k)}
    if isinstance(v, list):
        return [norm(x) for x in v]
    if isinstance(v, str):
        # scratch paths and wall-clock waits vary run to run; neither is a verdict
        return re.sub(r"\b\d+(\.\d+)?s\b", "<N>s", re.sub(r"/tmp/[^\s\"']+", "<tmp>", v))
    return v
by = {r.get("id"): norm(r) for r in rows}
inv = sorted(norm(x).get("file") for x in (R.get("inventory") or []))
print(json.dumps({"order": [r.get("id") for r in rows], "rows": by, "inventory": inv}, sort_keys=True))
PY
}

compare() { # <A json> <B json> -> "same" | "differ: <ids>"
  python3 - "$1" "$2" <<'PY'
import json, sys
a, b = json.loads(sys.argv[1]), json.loads(sys.argv[2])
if sorted(a["rows"]) != sorted(b["rows"]):
    print("different cell sets: %s vs %s" % (sorted(a["rows"]), sorted(b["rows"]))); sys.exit(0)
d = [k for k in sorted(a["rows"]) if a["rows"][k] != b["rows"][k]]
print("same" if not d else "differ: " + ",".join(d))
PY
}
if [ "$SELF_TEST" = 1 ]; then
  compare() { echo same; }   # the planted defect: a comparison that never looks at the rows
fi

fails=0
case_line() { printf '  %-4s %-58s %s\n' "$1" "$2" "$3"; [ "$1" = ok ] || fails=$((fails + 1)); }

echo "ladder cell order must not change a cell's verdict (#4039) -- the shipped $SCRIPT, run twice per case"
for leak in 0 1; do
  name=$([ "$leak" = 0 ] && echo clean || echo leaky)
  A=$(ladder_run "$leak" "$TMP/cert-leaky-first.json" "$name-a") || exit 2
  B=$(ladder_run "$leak" "$TMP/cert-victim-first.json" "$name-b") || exit 2
  oa=$(python3 -c 'import json,sys; print(",".join(json.loads(sys.argv[1])["order"]))' "$A")
  ob=$(python3 -c 'import json,sys; print(",".join(json.loads(sys.argv[1])["order"]))' "$B")
  # leaky-first: leaky (certified), then largest first -- the inventory model (3000 B) before victim (2048 B).
  # victim-first: victim (certified), then leaky (4096 B), then the inventory model.
  if [ "$oa" = "leaky,inv:middle-inv.gguf,victim" ] && [ "$ob" = "victim,leaky,inv:middle-inv.gguf" ]; then
    case_line ok "$name: the two runs measured in opposite orders" "$oa | $ob"
  else
    case_line FAIL "$name: the runs were not in the orders asked for" "$oa | $ob"
  fi
  ia=$(python3 -c 'import json,sys; print(",".join(json.loads(sys.argv[1])["inventory"]))' "$A")
  ib=$(python3 -c 'import json,sys; print(",".join(json.loads(sys.argv[1])["inventory"]))' "$B")
  if [ "$ia" = "middle-inv.gguf" ] && [ "$ib" = "middle-inv.gguf" ]; then
    case_line ok "$name: the held inventory model is recorded in both receipts" "$ia | $ib"
  else
    case_line FAIL "$name: the inventory record is missing or wrong" "${ia:-none} | ${ib:-none}"
  fi
  verdict=$(compare "$A" "$B")
  if [ "$leak" = 0 ]; then
    [ "$verdict" = same ] && case_line ok "clean: every cell's row is the same in both orders" "$verdict" \
                          || case_line FAIL "clean: a cell's row depends on the order" "$verdict"
  else
    [ "$verdict" != same ] && case_line ok "leaky (positive control): the leak is seen as order-dependence" "$verdict" \
                           || case_line FAIL "leaky (positive control): an order-dependent leak went unseen" "$verdict"
  fi
done

if [ "$SELF_TEST" = 1 ]; then
  if [ "$fails" -gt 0 ]; then echo "self-test: PASS -- a comparison that ignores the rows turns this RED"; exit 0; fi
  echo "self-test: FAILED -- the planted blind comparison still passed" >&2; exit 1
fi
[ "$fails" -eq 0 ] && { echo "OK: cell order changes no verdict, and an order-dependent leak would be caught"; exit 0; }
echo "FAIL: $fails case(s) -- see above"; exit 1
