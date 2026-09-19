#!/usr/bin/env bash
# model_ladder.sh — measure every rung of contracts/model-capability-ladder-v1.yaml
# on THIS host and write the per-host receipt T-2 reads.
#
# WHY. 0.68.1 shipped with dense Qwen3 producing garbage on CUDA on both fleet
# GPUs; the per-inference parity guard fell back to CPU so `apr run` looked fine,
# `apr qa` was the only surface that said "gibberish", and no release step ran
# `apr qa` on a Qwen3 (EPIC #3477). The unit of evidence was "a model ran". The
# unit here is (architecture, backend, silicon), one rung at a time.
#
# Usage:  bash scripts/model_ladder.sh [--host <id>] [--out <dir>] [--dry-run]
#   --host  ladder host id (default: derived from `hostname`, see host_id)
#   --out   receipt dir (default: evidence/dogfood/models/<version>)
#
# Exit: 0 every present required rung green · 1 any rung red or a required rung
# absent · 2 decline (nothing executed, binary unpinned, ladder unreadable).
# The receipt is written on 0 and 1, never on 2: a receipt that measured nothing
# is not evidence.
#
# Every command's status is captured as `out=$(cmd); rc=$?` — never through a
# pipe (#2336, #2360).
set -uo pipefail

LADDER="contracts/model-capability-ladder-v1.yaml"
HOST_ID=""
OUT_DIR=""
DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --host) [ $# -ge 2 ] || { echo "model_ladder: --host needs a value" >&2; exit 2; }; HOST_ID="$2"; shift 2 ;;
    --out)  [ $# -ge 2 ] || { echo "model_ladder: --out needs a value" >&2; exit 2; };  OUT_DIR="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "model_ladder: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)" || exit 2
[ -f "$LADDER" ] || { echo "decline: $LADDER not found" >&2; exit 2; }

# Step 0 — pin the binary. A diagnostic against the wrong apr is worse than none.
if [ "${DOGFOOD_ALLOW_UNPINNED:-0}" = "1" ] && [ -n "${APR:-}" ]; then
  : # published-crate mode (apr-dogfood G13): caller pinned $APR deliberately
else
  # shellcheck disable=SC1091
  . scripts/apr_bin.sh || { echo "decline: scripts/apr_bin.sh could not pin a HEAD-built apr" >&2; exit 2; }
fi
[ -x "${APR:-}" ] || { echo "decline: \$APR is not executable: '${APR:-}'" >&2; exit 2; }

host_id() {
  if [ -n "$HOST_ID" ]; then printf '%s' "$HOST_ID"; return; fi
  local h; h=$(hostname 2>/dev/null || echo unknown)
  case "$h" in
    gx10*) printf 'gx10' ;;
    *[Ll]ambda*) printf 'lambda' ;;
    *) printf '%s' "$h" ;;
  esac
}
HOST=$(host_id)
SHA=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
VERSION=$(cargo metadata --no-deps --offline --format-version 1 2>/dev/null | python3 -c '
import json, os, sys
m = json.load(sys.stdin)
root = os.path.realpath("Cargo.toml")
for p in m.get("packages", []):
    if os.path.realpath(p["manifest_path"]) == root:
        print(p["version"]); sys.exit(0)
sys.exit(1)' 2>/dev/null) || VERSION=""
[ -n "$VERSION" ] || { echo "decline: root crate version unresolved" >&2; exit 2; }
[ -n "$OUT_DIR" ] || OUT_DIR="evidence/dogfood/models/$VERSION"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)

# Rungs, one per line: id|file|sha256|backends(csv)|required
RUNGS=$(python3 - "$LADDER" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
for r in d["ladder"]["rungs"]:
    print("|".join([r["id"], r["gguf"], r["sha256"], ",".join(r["backends"]), "1" if r.get("required") else "0", ",".join(r.get("hosts") or [])]))
PY
) || { echo "decline: ladder unreadable" >&2; exit 2; }
[ -n "$RUNGS" ] || { echo "decline: ladder has no rungs" >&2; exit 2; }

find_model() { # find_model <basename> — first match in the fleet's model dirs
  local f="$1" d
  for d in "${APR_MODELS_DIR:-}" "$HOME/models" "$HOME/.apr/models" "$HOME/.cache/apr/models"; do
    [ -n "$d" ] && [ -f "$d/$f" ] && { printf '%s' "$d/$f"; return 0; }
  done
  return 1
}

WORK=$(mktemp -d)
# The delete is guarded (SEC011): only a path under a temp root is removed.
_rm_work() {
  local v="${WORK:-}"
  case "$v" in
    /tmp/?*|/var/folders/?*|/mnt/?*) if [ -n "$v" ] && [ "$v" != "/" ]; then rm -rf -- "$v" || :; fi ;;
    *) return 0 ;;
  esac
}
trap _rm_work EXIT
ROWS="$WORK/rows.jsonl"; : > "$ROWS"
EXECUTED=0; RED=0
printf -- '--- model capability ladder on %s (%s, cc %s) apr=%s sha=%s version=%s ---\n' \
  "$HOST" "${GPU_NAME:-no-gpu}" "${GPU_CC:-?}" "$APR" "$SHA" "$VERSION"

while IFS='|' read -r -t 5 rid rfile rsha rbackends rreq rhosts; do
  [ -n "$rid" ] || continue
  # A rung that lists hosts: is a claim only on those hosts (a 122B file fits gx10's unified memory and no
  # 24 GB card). Elsewhere it is neither absent nor passed: recorded as not listed, never counted RED.
  if [ -n "$rhosts" ] && ! grep -qE "(^|,)${HOST}(,|$)" <<< "$rhosts"; then
    printf '  [N/A   ] %-22s not listed for %s (hosts: %s)\n' "$rid" "$HOST" "$rhosts"
    printf '{"id":"%s","present":false,"required":false,"not_listed_host":true}\n' "$rid" >> "$ROWS"
    continue
  fi
  path=$(find_model "$rfile") || {
    printf '  [ABSENT] %-22s %s not in any model dir\n' "$rid" "$rfile"
    printf '{"id":"%s","present":false,"required":%s}\n' "$rid" "$([ "$rreq" = 1 ] && echo true || echo false)" >> "$ROWS"
    [ "$rreq" = 1 ] && RED=$((RED + 1))
    continue
  }
  got=$(sha256sum "$path" | cut -d' ' -f1)
  if [ "$got" != "$rsha" ]; then
    printf '  [FAIL  ] %-22s sha256 mismatch (%s… vs ladder %s…) — a different file is a different measurement\n' "$rid" "${got:0:12}" "${rsha:0:12}"
    printf '{"id":"%s","present":true,"sha_ok":false,"sha256":"%s","required":%s}\n' "$rid" "$got" "$([ "$rreq" = 1 ] && echo true || echo false)" >> "$ROWS"
    RED=$((RED + 1)); continue
  fi
  if [ "$DRY" = 1 ]; then printf '  [DRY   ] %-22s %s\n' "$rid" "$path"; continue; fi

  # 1. apr qa --json: capability_match + golden_output are the two gates that name
  #    a wrong model; the perf gates are skipped here (they pass on garbage).
  qa_json="$WORK/$rid.qa.json"
  # A rung that claims only the CPU is not asked whether the GPU can run it:
  # capability_match is a GPU-capability gate, and "the GPU backend has no SSM
  # kernels (#3090)" is a true statement about a claim the rung does not make.
  # The judge accepts a SKIPPED capability_match only when cuda is not claimed.
  cap_flag=""
  case ",$rbackends," in *,cuda,*|*,gpu,*) ;; *) cap_flag="--skip-capability" ;; esac
  # shellcheck disable=SC2086
  "$APR" qa "$path" --json --offline --skip-throughput --skip-ollama --skip-gpu-speedup \
      --skip-ptx-parity --skip-gpu-state --skip-format-parity $cap_flag > "$qa_json" 2> "$WORK/$rid.qa.err"; qa_rc=$?
  qa_row=$(python3 - "$qa_json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print(json.dumps({"parse_error": str(e)})); sys.exit(0)
g = {x["name"]: x for x in d.get("gates", [])}
def gate(n):
    x = g.get(n)
    if x is None: return {"passed": False, "skipped": True, "message": "gate absent from report"}
    # `apr qa --json` marks a skipped gate passed:true. Skipped is not passed.
    return {"passed": bool(x.get("passed")) and not x.get("skipped") and not str(x.get("message","")).startswith("Skipped"),
            "skipped": bool(x.get("skipped")) or str(x.get("message","")).startswith("Skipped"),
            "message": str(x.get("message",""))[:200]}
print(json.dumps({"capability_match": gate("capability_match"), "golden_output": gate("golden_output")}))
PY
)
  # 2. per claimed backend: did the run actually stay on that backend?
  be_json="{"
  first=1
  IFS=',' read -r -a bes <<< "$rbackends"
  for b in "${bes[@]}"; do
    case "$b" in cpu) flag="--no-gpu" ;; cuda|gpu) flag="--gpu" ;; *) flag="" ;; esac
    run_out=$("$APR" run "$path" --prompt "What is the capital of France? Answer briefly." --max-tokens 16 $flag 2>&1); run_rc=$?
    fb=false; ran=true
    if grep -qE 'falling back to CPU|path rejected, attempting fallback|runs on the CPU; the GPU backend' <<< "$run_out"; then fb=true; fi
    if [ "$b" != cpu ] && [ -z "$GPU_NAME" ]; then ran=false; fi
    [ $run_rc -eq 0 ] || ran=false
    [ $first = 1 ] || be_json="$be_json,"; first=0
    be_json="$be_json\"$b\":{\"ran\":$ran,\"fallback\":$fb,\"rc\":$run_rc}"
  done
  be_json="$be_json}"
  # The receipt carries the MEASURED file hash beside sha_ok (ONT-4c1): a resolver that joins the ladder
  # contract to this receipt compares two measurements (contract rung.sha256 == receipt rung.sha256) and can
  # name a mismatch, instead of trusting the receipt's own claim that it checked. Same reason the ONT ledger
  # carries merged_sha, not merged: true.
  row=$(python3 - "$rid" "$qa_row" "$be_json" "$qa_rc" "$rreq" "$got" <<'PY'
import json, sys
rid, qa, be, qa_rc, req, sha = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3]), int(sys.argv[4]), sys.argv[5] == "1", sys.argv[6]
cap = qa.get("capability_match", {})
# `passed` is already normalised (skipped ⇒ passed=False) by the gate() reader above, but the
# judge must not depend on that: a skipped gate counts only when no GPU backend is claimed.
cap_ok = (cap.get("passed", False) and not cap.get("skipped", False)) \
         or (cap.get("skipped", False) and not ({"cuda", "gpu"} & set(be)))
green = cap_ok and qa.get("golden_output", {}).get("passed", False) \
        and all(v["ran"] and not v["fallback"] for v in be.values())
print(json.dumps({"id": rid, "present": True, "sha_ok": True, "sha256": sha, "required": req, "qa_rc": qa_rc,
                  "capability_match": qa.get("capability_match"), "golden_output": qa.get("golden_output"),
                  "backends": be, "green": green}))
PY
)
  printf '%s\n' "$row" >> "$ROWS"
  EXECUTED=$((EXECUTED + 1))
  if grep -q '"green": true' <<< "$row"; then
    printf '  [ OK   ] %-22s qa cap+golden pass, backends %s honoured\n' "$rid" "$rbackends"
  else
    RED=$((RED + 1))
    why=$(printf '%s' "$row" | python3 -c '
import json,sys; r=json.load(sys.stdin); w=[]
cap=r["capability_match"]; claims_gpu=bool({"cuda","gpu"} & set(r["backends"]))
if not (cap["passed"] and not cap["skipped"]) and not (cap["skipped"] and not claims_gpu): w.append("capability_match: "+("SKIPPED on a GPU-claiming rung: " if cap["skipped"] else "")+cap["message"][:70])
if not r["golden_output"]["passed"]: w.append("golden_output: "+r["golden_output"]["message"][:70])
for b,v in r["backends"].items():
    if v["fallback"]: w.append(b+": FELL BACK (claimed backend did not run)")
    elif not v["ran"]: w.append(b+": did not run (rc=%s)"%v["rc"])
print("; ".join(w) or "unknown")')
    printf '  [FAIL  ] %-22s %s\n' "$rid" "$why"
  fi
done <<EOF2
$RUNGS
EOF2

[ "$DRY" = 1 ] && exit 0
if [ "$EXECUTED" -eq 0 ]; then
  echo "decline: no rung executed on $HOST — nothing was measured, no receipt written" >&2
  exit 2
fi
mkdir -p "$OUT_DIR"
# The receipt names the binary by what it SAYS it is (`apr --version`, which
# carries the built-from sha), never by its path: a path is machine-specific
# (check_no_shipped_machine_paths) and says nothing about what was run.
APR_VERSION=$("$APR" --version 2>/dev/null | head -1)
python3 - "$ROWS" "$OUT_DIR/$HOST.json" "$HOST" "$VERSION" "$SHA" "${GPU_NAME:-}" "${GPU_CC:-}" "$EXECUTED" "$RED" "$APR_VERSION" <<'PY'
import json, sys, datetime, platform
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
out = {"schema": "apr-model-ladder-receipt/v1", "host": sys.argv[3], "version": sys.argv[4], "sha": sys.argv[5],
       "isa": platform.machine(), "gpu": sys.argv[6] or None, "cc": sys.argv[7] or None,
       "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
       "apr_version": sys.argv[10], "executed": int(sys.argv[8]), "red": int(sys.argv[9]), "rungs": rows}
json.dump(out, open(sys.argv[2], "w"), indent=2); open(sys.argv[2], "a").write("\n")
PY
printf 'receipt: %s/%s.json (executed=%s red=%s)\n' "$OUT_DIR" "$HOST" "$EXECUTED" "$RED"
[ "$RED" -eq 0 ] && exit 0
exit 1
