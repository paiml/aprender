#!/usr/bin/env bash
# #3522 OXIDE-001 port: run the row-batched per-head L2 norm port on THIS host and write
# evidence/kernels/gdn_l2_norm_rows/<host>.json (schema apr-kernel-receipt/v1).
#
# The harness (src/main.rs) measures parity, timing and registers and prints one
# `RECEIPT {...}` line per variant. This wrapper adds only facts the binary
# cannot see: host, repo sha, cuda-oxide rev, ptxas and driver versions, and
# any foreign GPU process sharing the device during the run. It fails if the
# harness prints no receipt, so it cannot write an empty file, and it exits
# with the harness's status after writing, so a failing gate never reads green.
#
# Usage (from this directory, GPU host, cuda-oxide toolchain installed):
#   PATH=/usr/local/cuda-13.3/bin:$PATH flock /tmp/apr-gpu.lock ./receipt.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(git -C "$here" rev-parse --show-toplevel)"
host="$(hostname -s)"
out_dir="$repo/evidence/kernels/gdn_l2_norm_rows"
log="$(mktemp)"
trap 'rm -f "$log"' EXIT

# Pinned rev from Cargo.toml, so the receipt names the compiler that built it.
oxide_rev="$(grep -m1 -o 'rev = "[0-9a-f]*"' "$here/Cargo.toml" | cut -d'"' -f2)"
sha="$(git -C "$repo" rev-parse HEAD)"
dirty="$(git -C "$repo" status --porcelain -- "$here" "$repo/crates/aprender-gpu/src/kernels/gdn" | wc -l)"
ptxas_version="$(ptxas --version 2>/dev/null | tail -1 || echo unknown)"
driver="$(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
# A compute-app row whose pid is not in /proc is a context the driver kept after its
# process died (lambda, 2026-09-25: pid 3650689, 936 MiB, `[Not Found]`, for hours).
# Nothing is left to launch work on it, so it cannot skew timing; it is recorded as
# `phantom_gpu_ctx` instead of failing every receipt until a GPU reset. Host /proc
# shows container processes too, so a live process in a container stays foreign.
# Anything that does not parse as a numeric pid stays foreign (fail closed).
foreign=""
phantom=""
while IFS= read -r row; do
    [[ -n "$row" ]] || continue
    pid="${row%%,*}"
    if [[ "$pid" =~ ^[0-9]+$ ]] && [[ ! -e "/proc/$pid" ]]; then
        phantom+="$row;"
    else
        foreign+="$row;"
    fi
done < <(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null || true)

cd "$here" || exit 1
rc=0
cargo oxide run >"$log" 2>&1 || rc=$?
receipts="$(grep '^RECEIPT ' "$log" | sed 's/^RECEIPT //' || true)"
if [[ -z "$receipts" ]]; then
    tail -n 40 "$log" >&2
    echo "receipt.sh: harness exited $rc and printed no RECEIPT line; nothing written" >&2
    exit "$(( rc == 0 ? 3 : rc ))"
fi

mkdir -p "$out_dir"
HOST="$host" SHA="$sha" DIRTY="$dirty" OXIDE_REV="$oxide_rev" PTXAS="$ptxas_version" \
DRIVER="$driver" FOREIGN="$foreign" PHANTOM="$phantom" RECEIPTS="$receipts" python3 - >"$out_dir/$host.json" <<'EOF'
import json, os
rows = [json.loads(line) for line in os.environ["RECEIPTS"].splitlines() if line.strip()]
for r in rows:
    r.update(host=os.environ["HOST"], sha=os.environ["SHA"],
             tree_dirty_paths=int(os.environ["DIRTY"]),
             cuda_oxide_rev=os.environ["OXIDE_REV"],
             ptxas_version=os.environ["PTXAS"], driver=os.environ["DRIVER"],
             foreign_gpu_procs=os.environ["FOREIGN"],
             phantom_gpu_ctx=os.environ["PHANTOM"])
print(json.dumps({"schema": "apr-kernel-receipt/v1", "kernel": "gdn_l2_norm_rows",
                  "issue": 3522, "receipts": rows}, indent=2))
EOF
grep -E 'parity|GO|DONE|FAILED' "$log"
echo "wrote $out_dir/$host.json"
# A failing gate still writes its receipt (the file carries "pass":false), then
# fails the run: 1 = parity, 4 = timing NO-GO.
if [ "$rc" -ne 0 ]; then
    echo "receipt.sh: harness exited $rc (1 parity / 4 timing); receipt records the failure" >&2
    exit "$rc"
fi
