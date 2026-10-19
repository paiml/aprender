#!/usr/bin/env bash
# toolchain_id.sh - one sha256 hex naming the toolchain a gate ran on (PMAT-5737, FLOW-009 RC-1 `toolchain`).
#
#   toolchain_id.sh [repo-root]
#
# Prints sha256 over, byte for byte, in this order:
#   <contents of <repo-root>/rust-toolchain.toml, or nothing when absent>
#   "--rustc--\n" + the full output of `rustc -vV`
#   "--image--\n" + "image=none\n"
# The gate container/image digest term is the literal `image=none`: lambda's `make gate` runs
# UNCONTAINERISED. A containerised gate MUST hash its image digest here instead (e.g. `image=sha256:<digest>`),
# and a receipt signed under one image term never verifies under another. infra's script
# (fleet/receipts/receipt.sh toolchain) hashes the same three terms in the same order.
# No rustc: prints nothing on stdout, a reason on stderr, exit 1.
set -uo pipefail

root=${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
vv=$(rustc -vV 2>/dev/null) || { echo "toolchain_id: rustc -vV failed" >&2; exit 1; }
[ -n "$vv" ] || { echo "toolchain_id: rustc -vV printed nothing" >&2; exit 1; }
{
    if [ -f "$root/rust-toolchain.toml" ]; then cat "$root/rust-toolchain.toml"; fi
    printf -- '--rustc--\n%s\n' "$vv"
    printf -- '--image--\nimage=none\n'
} | sha256sum | cut -d' ' -f1
