#!/usr/bin/env bash
# toolchain_id.sh - one sha256 hex naming the Rust toolchain a gate ran on (PMAT-5737, receipt format v1).
#
#   toolchain_id.sh [repo-root]
#
# Prints sha256 over, byte for byte:
#   <contents of <repo-root>/rust-toolchain.toml, or nothing when it is absent> "\n"
#   the `commit-hash:` and `release:` lines of `rustc -vV`, in the order rustc prints them
# Deliberately NOT included: cargo, nextest, image digests, rustc's date/host/LLVM lines. The same pins give
# the same answer on lambda-labs and inside CI. The infra variant of this script (receipt format v1) must
# hash the same two rustc lines in the same order; aprender has no forjar or bashrs term.
# No rustc, or a rustc that prints neither line: prints nothing on stdout, a reason on stderr, exit 1.
set -uo pipefail

root=${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
vv=$(rustc -vV 2>/dev/null) || { echo "toolchain_id: rustc -vV failed" >&2; exit 1; }
lines=$(printf '%s\n' "$vv" | grep -E '^(commit-hash|release):')
[ "$(printf '%s\n' "$lines" | grep -c .)" -eq 2 ] || { echo "toolchain_id: rustc -vV has no release/commit-hash pair" >&2; exit 1; }
{
    if [ -f "$root/rust-toolchain.toml" ]; then cat "$root/rust-toolchain.toml"; fi
    printf '\n%s\n' "$lines"
} | sha256sum | cut -d' ' -f1
