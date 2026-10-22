#!/usr/bin/env bash
# manifest_id.sh - one sha256 hex naming the gate definition and its test list (PMAT-5737, FLOW-009 RC-1 `manifest`).
#
#   manifest_id.sh [repo-root]
#
# Prints sha256 over, byte for byte, in this order:
#   for each file in GATE_FILES (fixed order): "--file <path>\n" + its bytes (a missing file hashes as "--missing\n")
#   "--tests--\n" + the TEST LIST
# GATE_FILES: Makefile (holds `gate:`), scripts/gate_touched_crates.sh, scripts/guard_tree.sh,
#   scripts/ci_test_tier.sh, scripts/ci_receipt_verify.sh, scripts/ci/toolchain_id.sh, scripts/ci/manifest_id.sh.
# TEST LIST: the workspace's test targets, `cargo metadata --no-deps --locked --offline --format-version 1`
#   reduced by jq to a sorted `[{name, targets:[{name, kind}]}]` -- cheap (no build) and it changes whenever a
#   crate or a test/bench/example target is added, removed or renamed. It does NOT name individual #[test] fns
#   (that needs a build); a change to which tests exist INSIDE a target is not caught here, only by the gate.
# Any gate-definition or test-target change invalidates every outstanding receipt, by design.
# A missing cargo/jq or a metadata failure: prints nothing on stdout, a reason on stderr, exit 1.
set -uo pipefail

root=${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
GATE_FILES=(Makefile scripts/gate_touched_crates.sh scripts/guard_tree.sh scripts/ci_test_tier.sh scripts/ci_receipt_verify.sh scripts/ci/toolchain_id.sh scripts/ci/manifest_id.sh)
command -v jq >/dev/null 2>&1 || { echo "manifest_id: jq missing" >&2; exit 1; }
tests=$(cd "$root" && cargo metadata --no-deps --locked --offline --format-version 1 2>/dev/null \
    | jq -S -c '[.packages[] | {name, targets: [.targets[] | {name, kind}] | sort_by(.name, (.kind | join(",")))}] | sort_by(.name)') \
    || { echo "manifest_id: cargo metadata failed" >&2; exit 1; }
[ -n "$tests" ] && [ "$tests" != null ] || { echo "manifest_id: empty test list" >&2; exit 1; }
{
    for f in "${GATE_FILES[@]}"; do
        printf -- '--file %s\n' "$f"
        if [ -f "$root/$f" ]; then cat "$root/$f"; else printf -- '--missing\n'; fi
    done
    printf -- '--tests--\n%s\n' "$tests"
} | sha256sum | cut -d' ' -f1
