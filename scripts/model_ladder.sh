#!/usr/bin/env bash
# model_ladder.sh — measure every rung of contracts/model-capability-ladder-v1.yaml AND
# every Q4_K model this host HOLDS, and write the per-host receipt the release reads.
#
# WHY. 0.68.1 shipped with dense Qwen3 producing garbage on CUDA on both fleet
# GPUs; the per-inference parity guard fell back to CPU so `apr run` looked fine,
# `apr qa` was the only surface that said "gibberish", and no release step ran
# `apr qa` on a Qwen3 (EPIC #3477). The unit of evidence was "a model ran". The
# unit here is (architecture, backend, silicon), one model at a time.
#
# THE UNIVERSE IS THE INVENTORY (#3712, operator 2026-09-21: "you must ensure all
# models Q4_K CUDA work; the end", and publishing with "most working" is a "p0 tire
# fire"). A hand-picked ladder let a red model through by leaving it off the list, or
# by marking it `required: false`. This run measures ladder ∪ inventory, where the
# inventory is every file on THIS host matching the contract's `inventory.patterns`
# (case-insensitive, depth 1) under its `inventory.dirs`. An inventory model that is
# not already a rung is measured on the inventory's backends (cuda). The receipt
# (schema v2) carries the measured inventory, so check_model_ladder.sh can name any
# model this host holds that the run did not prove.
#
# Usage:  bash scripts/model_ladder.sh [--host <id>] [--out <dir>] [--dry-run]
#   --host  ladder host id (default: derived from `hostname`, see host_id)
#   --out   receipt dir (default: evidence/dogfood/models/<version>); writes <dir>/<host>.json
#
# Exit: 0 every present required rung AND every inventory model green · 1 any red,
# or a required rung absent · 2 decline (nothing executed, binary unpinned, ladder
# or inventory unreadable). The receipt is written on 0 and 1, never on 2: a
# receipt that measured nothing is not evidence.
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
    -h|--help) awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
    *) echo "model_ladder: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

# The root is derived from this file's path, never from `git rev-parse` (aprender#3581) or the caller's cwd.
cd "$(dirname "$0")/.." || exit 2
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
# apr_sha: the full 40-hex HEAD, which scripts/apr_bin.sh proved the binary was built from. The
# release-readiness shape (#3715) compares it to the release commit by exact equality.
APR_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)
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

# Rungs, one per line: id|file|sha256|backends(csv)|required|hosts(csv)
RUNGS=$(python3 - "$LADDER" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
for r in d["ladder"]["rungs"]:
    print("|".join([r["id"], r["gguf"], r["sha256"], ",".join(r["backends"]), "1" if r.get("required") else "0", ",".join(r.get("hosts") or [])]))
PY
) || { echo "decline: ladder unreadable" >&2; exit 2; }
[ -n "$RUNGS" ] || { echo "decline: ladder has no rungs" >&2; exit 2; }
LADDER_FILES=$(cut -d'|' -f2 <<< "$RUNGS")   # the files the rungs name; section 2 skips re-measuring them

# The inventory spec (#3712). MODEL_LADDER_INVENTORY_DIRS (colon-separated) is a test seam only.
INV_SPEC=$(python3 - "$LADDER" <<'PY'
import os, sys, yaml
inv = yaml.safe_load(open(sys.argv[1]))["ladder"].get("inventory") or {}
env = os.environ.get("MODEL_LADDER_INVENTORY_DIRS")
dirs = env.split(":") if env else [os.path.expanduser(d) for d in (inv.get("dirs") or [])]
if not dirs or not inv.get("patterns") or not inv.get("backends"):
    sys.exit(1)
print(":".join(dirs)); print(",".join(inv["patterns"])); print(",".join(inv["backends"]))
PY
) || { echo "decline: the ladder declares no inventory (dirs, patterns, backends) -- the universe cannot be measured" >&2; exit 2; }
INV_DIRS=$(sed -n 1p <<< "$INV_SPEC"); INV_PATTERNS=$(sed -n 2p <<< "$INV_SPEC"); INV_BACKENDS=$(sed -n 3p <<< "$INV_SPEC")
# INVENTORY, one per line: file|path. Measured on THIS host, never listed.
INVENTORY=$(python3 - "$INV_DIRS" "$INV_PATTERNS" <<'PY'
import fnmatch, os, sys
dirs, pats = sys.argv[1].split(":"), [p.lower() for p in sys.argv[2].split(",")]
seen = {}
for d in dirs:
    if not os.path.isdir(d):
        continue
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        if os.path.isfile(p) and f not in seen and any(fnmatch.fnmatch(f.lower(), pat) for pat in pats):
            seen[f] = p
for f, p in seen.items():
    print(f + "|" + p)
PY
) || { echo "decline: the inventory scan failed" >&2; exit 2; }

find_model() { # find_model <basename> — the inventory's copy first, then the fleet's other model dirs
  local f="$1" d p
  p=$(awk -F'|' -v f="$f" '$1 == f { print $2; exit }' <<< "$INVENTORY")
  [ -n "$p" ] && [ -f "$p" ] && { printf '%s' "$p"; return 0; }
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
INV_ROWS="$WORK/inventory.jsonl"; : > "$INV_ROWS"
EXECUTED=0; RED=0
printf -- '--- model capability ladder on %s (%s, cc %s) apr=%s sha=%s version=%s ---\n' \
  "$HOST" "${GPU_NAME:-no-gpu}" "${GPU_CC:-?}" "$APR" "$SHA" "$VERSION"
printf '    inventory: %s model(s) matching %s under %s\n' "$(grep -c . <<< "$INVENTORY")" "$INV_PATTERNS" "$INV_DIRS"

# measure <id> <file> <path> <sha> <backends(csv)> <required 0|1> <inventory_only 0|1>
# The per-model checks, one function so a ladder rung and an inventory model cannot drift:
#   1. apr qa --json: capability_match + golden_output (the two gates that name a wrong model)
#   2. per claimed backend: `apr run` rc 0 and no fallback line (did it stay on that backend?)
measure() {
  local rid=$1 rfile=$2 path=$3 got=$4 rbackends=$5 rreq=$6 rinv=$7
  local qa_json="$WORK/${rid//[^A-Za-z0-9._-]/_}.qa.json" cap_flag="" qa_rc qa_row be_json first b flag run_out run_rc fb ran row why
  # A rung that claims only the CPU is not asked whether the GPU can run it: capability_match is a
  # GPU-capability gate. The judge accepts a SKIPPED capability_match only when cuda is not claimed,
  # and check_model_ladder.sh refuses any Q4_K rung that does not claim cuda (#3712).
  case ",$rbackends," in *,cuda,*|*,gpu,*) ;; *) cap_flag="--skip-capability" ;; esac
  # shellcheck disable=SC2086
  "$APR" qa "$path" --json --offline --skip-throughput --skip-ollama --skip-gpu-speedup \
      --skip-ptx-parity --skip-gpu-state --skip-format-parity $cap_flag > "$qa_json" 2> "$qa_json.err"; qa_rc=$?
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
  be_json="{"; first=1
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
  # The receipt carries the MEASURED file hash (ONT-4c1): a resolver joining the ladder contract to
  # this receipt compares two measurements instead of trusting the receipt's own claim that it checked.
  row=$(python3 - "$rid" "$qa_row" "$be_json" "$qa_rc" "$rreq" "$got" "$rfile" "$rinv" <<'PY'
import json, sys
rid, qa, be, qa_rc = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3]), int(sys.argv[4])
req, sha, rfile, inv_only = sys.argv[5] == "1", sys.argv[6], sys.argv[7], sys.argv[8] == "1"
cap = qa.get("capability_match", {})
# `passed` is already normalised (skipped => passed=False) by the gate() reader above, but the
# judge must not depend on that: a skipped gate counts only when no GPU backend is claimed.
cap_ok = (cap.get("passed", False) and not cap.get("skipped", False)) \
         or (cap.get("skipped", False) and not ({"cuda", "gpu"} & set(be)))
green = cap_ok and qa.get("golden_output", {}).get("passed", False) \
        and all(v["ran"] and not v["fallback"] for v in be.values())
print(json.dumps({"id": rid, "file": rfile, "inventory_only": inv_only, "present": True, "sha_ok": True,
                  "sha256": sha, "required": req, "qa_rc": qa_rc, "capability_match": qa.get("capability_match"),
                  "golden_output": qa.get("golden_output"), "backends": be, "green": green}))
PY
)
  printf '%s\n' "$row" >> "$ROWS"
  EXECUTED=$((EXECUTED + 1))
  if grep -q '"green": true' <<< "$row"; then
    printf '  [ OK   ] %-30s qa cap+golden pass, backends %s honoured\n' "$rid" "$rbackends"
  else
    RED=$((RED + 1))
    why=$(printf '%s' "$row" | python3 -c '
import json,sys; r=json.load(sys.stdin); w=[]
cap=r["capability_match"] or {}; claims_gpu=bool({"cuda","gpu"} & set(r["backends"]))
if not (cap.get("passed") and not cap.get("skipped")) and not (cap.get("skipped") and not claims_gpu): w.append("capability_match: "+("SKIPPED on a GPU-claiming model: " if cap.get("skipped") else "")+str(cap.get("message",""))[:70])
gold=r["golden_output"] or {}
if not gold.get("passed"): w.append("golden_output: "+("SKIPPED: " if gold.get("skipped") else "")+str(gold.get("message",""))[:70])
for b,v in r["backends"].items():
    if v["fallback"]: w.append(b+": FELL BACK (claimed backend did not run)")
    elif not v["ran"]: w.append(b+": did not run (rc=%s)"%v["rc"])
print("; ".join(w) or "unknown")')
    printf '  [FAIL  ] %-30s %s\n' "$rid" "$why"
  fi
}

# ---- 1. the ladder's rungs
while IFS='|' read -r -t 5 rid rfile rsha rbackends rreq rhosts; do
  [ -n "$rid" ] || continue
  # A rung that lists hosts: is a claim only on those hosts (a 122B file fits gx10's unified memory and no
  # 24 GB card). Elsewhere it is neither absent nor passed: recorded as not listed, never counted RED.
  # An inventory model is never "not listed": if this host holds it, section 2 measures it.
  if [ -n "$rhosts" ] && ! grep -qE "(^|,)${HOST}(,|$)" <<< "$rhosts"; then
    printf '  [N/A   ] %-30s not listed for %s (hosts: %s)\n' "$rid" "$HOST" "$rhosts"
    printf '{"id":"%s","file":"%s","present":false,"required":false,"not_listed_host":true}\n' "$rid" "$rfile" >> "$ROWS"
    continue
  fi
  path=$(find_model "$rfile") || {
    printf '  [ABSENT] %-30s %s not in any model dir\n' "$rid" "$rfile"
    printf '{"id":"%s","file":"%s","present":false,"required":%s}\n' "$rid" "$rfile" "$([ "$rreq" = 1 ] && echo true || echo false)" >> "$ROWS"
    [ "$rreq" = 1 ] && RED=$((RED + 1))
    continue
  }
  got=$(sha256sum "$path" | cut -d' ' -f1)
  if [ "$got" != "$rsha" ]; then
    printf '  [FAIL  ] %-30s sha256 mismatch (%s… vs ladder %s…) — a different file is a different measurement\n' "$rid" "${got:0:12}" "${rsha:0:12}"
    printf '{"id":"%s","file":"%s","present":true,"sha_ok":false,"sha256":"%s","required":%s}\n' "$rid" "$rfile" "$got" "$([ "$rreq" = 1 ] && echo true || echo false)" >> "$ROWS"
    RED=$((RED + 1)); continue
  fi
  if [ "$DRY" = 1 ]; then printf '  [DRY   ] %-30s %s\n' "$rid" "$path"; continue; fi
  measure "$rid" "$rfile" "$path" "$got" "$rbackends" "$rreq" 0
done <<EOF2
$RUNGS
EOF2

# ---- 2. every inventory model: recorded in the receipt, and measured unless a rung already did
while IFS='|' read -r -t 5 ifile ipath; do
  [ -n "$ifile" ] || continue
  isha=$(sha256sum "$ipath" | cut -d' ' -f1); ibytes=$(stat -c %s "$ipath" 2>/dev/null || echo 0)
  printf '{"file":"%s","sha256":"%s","bytes":%s}\n' "$ifile" "$isha" "$ibytes" >> "$INV_ROWS"
  if grep -qxF -- "$ifile" <<< "$LADDER_FILES"; then continue; fi   # a rung measured it above
  if [ "$DRY" = 1 ]; then printf '  [DRY   ] %-30s %s (inventory, not a rung)\n' "inv:$ifile" "$ipath"; continue; fi
  measure "inv:$ifile" "$ifile" "$ipath" "$isha" "$INV_BACKENDS" 1 1
done <<EOF3
$INVENTORY
EOF3

[ "$DRY" = 1 ] && exit 0
if [ "$EXECUTED" -eq 0 ]; then
  echo "decline: nothing executed on $HOST — nothing was measured, no receipt written" >&2
  exit 2
fi
mkdir -p "$OUT_DIR"
# The receipt names the binary by what it SAYS it is (`apr --version`, which
# carries the built-from sha), never by its path: a path is machine-specific
# (check_no_shipped_machine_paths) and says nothing about what was run.
APR_VERSION=$("$APR" --version 2>/dev/null | head -1)
python3 - "$ROWS" "$OUT_DIR/$HOST.json" "$HOST" "$VERSION" "$SHA" "${GPU_NAME:-}" "${GPU_CC:-}" "$EXECUTED" "$RED" "$APR_VERSION" "$INV_ROWS" "$INV_DIRS" "$INV_PATTERNS" "$APR_SHA" <<'PY'
import json, sys, datetime, platform
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
inv = [json.loads(l) for l in open(sys.argv[11]) if l.strip()]
out = {"schema": "apr-model-ladder-receipt/v2", "host": sys.argv[3], "version": sys.argv[4], "sha": sys.argv[5], "apr_sha": sys.argv[14],
       "isa": platform.machine(), "gpu": sys.argv[6] or None, "cc": sys.argv[7] or None,
       "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
       "apr_version": sys.argv[10], "executed": int(sys.argv[8]), "red": int(sys.argv[9]),
       "inventory": inv, "inventory_dirs": sys.argv[12].split(":"), "inventory_patterns": sys.argv[13].split(","),
       "rungs": rows}
json.dump(out, open(sys.argv[2], "w"), indent=2); open(sys.argv[2], "a").write("\n")
PY
printf 'receipt: %s/%s.json (executed=%s red=%s inventory=%s)\n' "$OUT_DIR" "$HOST" "$EXECUTED" "$RED" "$(grep -c . "$INV_ROWS")"
[ "$RED" -eq 0 ] && exit 0
exit 1
