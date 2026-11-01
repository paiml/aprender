#!/usr/bin/env bash
# check_promote_nightly.sh -- guard_tree's handle on promote_nightly.sh (N26, PMAT-5306): run its case table.
# Every refusal path (off-main sha, existing tag, missing/red/not_measured receipt, missing certification,
# version mismatch, dry-run never tags) is a row. The mutants (one dropped check each, ~2 min) stay on demand:
#   bash scripts/release/falsify_promote_nightly.sh --mutants
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
exec bash "$ROOT/scripts/release/falsify_promote_nightly.sh"
