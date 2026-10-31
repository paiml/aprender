#!/usr/bin/env bash
# check_tag_publish_andon.sh -- guard_tree's handle on tag_publish_andon.sh (PMAT-5306): run its case table (a few seconds).
# A tag not on crates.io after the window pages once; a present version, a young tag, an old-era tag and an unmeasured read page
# nobody; the autopilot close gate refuses without crates.io. The mutants (one dropped check each) stay on demand:
#   bash scripts/release/falsify_tag_publish_andon.sh --mutants
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
exec bash "$ROOT/scripts/release/falsify_tag_publish_andon.sh"
