#!/usr/bin/env bash
# #3522 OXIDE-001 O-1: run the gated-RMSNorm pilot on THIS host and write
# evidence/kernels/gdn_gated_rmsnorm/<host>.json (schema apr-kernel-receipt/v1).
#
# The harness (src/main.rs) measures parity, timing and registers and prints one
# `RECEIPT {...}` line per variant. This wrapper adds only facts the binary
# cannot see: host, repo sha, cuda-oxide rev, ptxas and driver versions, and
# any foreign GPU process sharing the device during the run. It fails if the
# harness fails or prints no receipt, so it cannot write an empty file.
#
# Usage (from this directory, GPU host, cuda-oxide toolchain installed):
#   PATH=/usr/local/cuda-13.3/bin:$PATH flock /tmp/apr-gpu.lock ./receipt.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(git -C "$here" rev-parse --show-toplevel)"
host="$(hostname -s)"
out_dir="$repo/evidence/kernels/gdn_gated_rmsnorm"
log="$(mktemp)"
trap 'rm -f "$log"' EXIT

# Pinned rev from Cargo.toml, so the receipt names the compiler that built it.
oxide_rev="$(grep -m1 -o 'rev = "[0-9a-f]*"' "$here/Cargo.toml" | cut -d'"' -f2)"
sha="$(git -C "$repo" rev-parse HEAD)"
dirty="$(git -C "$repo" status --porcelain -- "$here" "$repo/crates/aprender-gpu/src/kernels/gdn" | wc -l)"
ptxas_version="$(ptxas --version 2>/dev/null | tail -1 || echo unknown)"
driver="$(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
foreign="$(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';' || true)"

cd "$here" || exit 1
rc=0
cargo oxide run >"$log" 2>&1 || rc=$?
if [ "$rc" -ne 0 ]; then
    tail -n 40 "$log" >&2
    echo "receipt.sh: harness exited $rc; no receipt written" >&2
    exit "$rc"
fi
receipts="$(grep '^RECEIPT ' "$log" | sed 's/^RECEIPT //')"
if [[ -z "$receipts" ]]; then
    echo "receipt.sh: harness printed no RECEIPT line" >&2
    exit 3
fi

mkdir -p "$out_dir"
HOST="$host" SHA="$sha" DIRTY="$dirty" OXIDE_REV="$oxide_rev" PTXAS="$ptxas_version" \
DRIVER="$driver" FOREIGN="$foreign" RECEIPTS="$receipts" python3 - >"$out_dir/$host.json" <<'EOF'
import json, os
rows = [json.loads(line) for line in os.environ["RECEIPTS"].splitlines() if line.strip()]
for r in rows:
    r.update(host=os.environ["HOST"], sha=os.environ["SHA"],
             tree_dirty_paths=int(os.environ["DIRTY"]),
             cuda_oxide_rev=os.environ["OXIDE_REV"],
             ptxas_version=os.environ["PTXAS"], driver=os.environ["DRIVER"],
             foreign_gpu_procs=os.environ["FOREIGN"])
print(json.dumps({"schema": "apr-kernel-receipt/v1", "kernel": "gdn_gated_rmsnorm",
                  "issue": 3522, "receipts": rows}, indent=2))
EOF
grep -E 'parity|GO|DONE' "$log"
echo "wrote $out_dir/$host.json"
