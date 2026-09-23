#!/usr/bin/env bash
# gpu_shape_conformance.sh <host-label> — run the census-driven device harness (#3968) on THIS host's GPU
# and write evidence/gpu-shape-conformance/<host-label>.json.
#
# The test binary is built OUTSIDE the GPU lock and only the run takes it (gpu-q). The receipt is bound to
# the census it tested (sha256 of each census file), the commit, and the device, so
# check_gpu_shape_conformance.sh can refuse a receipt that no longer describes the tree.
#
# Exit: the harness's own (0 every held shape conforms and every negative control went RED) · 2 usage/ENV.
set -uo pipefail

HOST_LABEL="${1:-}"
[ -n "$HOST_LABEL" ] || { echo "usage: $0 <host-label>   (lambda | gx10)" >&2; exit 2; }
ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 2
cd "$ROOT" || exit 2
command -v gpu-q >/dev/null 2>&1 || { echo "gpu_shape_conformance: gpu-q is not on PATH — the run must hold the GPU lock" >&2; exit 2; }
OUT="$ROOT/evidence/gpu-shape-conformance/$HOST_LABEL.json"
mkdir -p "$(dirname "$OUT")" || exit 2
RAW=$(mktemp) || exit 2
trap 'rm -f "$RAW"' EXIT

bin=$(cargo test -p aprender-serve --features cuda --lib --no-run --message-format=json 2>/dev/null \
  | python3 -c 'import json,sys
for l in sys.stdin:
    try: m=json.loads(l)
    except ValueError: continue
    if m.get("reason")=="compiler-artifact" and m.get("profile",{}).get("test") and m.get("executable"): print(m["executable"])' | tail -1)
[ -x "$bin" ] || { echo "gpu_shape_conformance: the aprender-serve test binary did not build" >&2; exit 2; }

before=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 | tr '\n' ';')
CONFORMANCE_RECEIPT="$RAW" gpu-q --prio "${GPUQ_PRIO:-5}" -- "$bin" --ignored \
  shape_conformance_tests::every_whitelisted_held_shape_conforms_on_the_device --nocapture
rc=$?
after=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 | tr '\n' ';')
[ -s "$RAW" ] || { echo "gpu_shape_conformance: the harness wrote no receipt (rc $rc)" >&2; exit 2; }

python3 - "$RAW" "$OUT" "$HOST_LABEL" "$(git rev-parse HEAD)" "$before" "$after" "$rc" <<'PY'
import glob, hashlib, json, subprocess, sys
raw, out, host, sha, before, after, rc = sys.argv[1:8]
doc = json.load(open(raw))
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
doc.update({
    "host": host, "commit": sha, "gpu": gpu, "harness_rc": int(rc),
    "census": {p.split("/")[-1]: hashlib.sha256(open(p, "rb").read()).hexdigest()
               for p in sorted(glob.glob("evidence/gpu-shape-census/*.json"))},
    "gpu_apps_before": before, "gpu_apps_after": after,
})
json.dump(doc, open(out, "w"), indent=1)
open(out, "a").write("\n")
print(f"gpu_shape_conformance: {out} ({len(doc['rows'])} rows, failed {doc['failed']}, blind {doc['blind_types']})")
PY
exit "$rc"
