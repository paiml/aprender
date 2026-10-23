#!/usr/bin/env bash
# roadmap_trim.sh -- collapse a `pmat work add`/`work complete` roadmap.yaml re-serialisation
# back to the base's bytes, one command (PMAT-980, G-6, #2874; ported to Rust in #5902).
#
#     bash scripts/roadmap_trim.sh [--base <ref>] [--file <path>]
#
# is exactly `pv-roadmap diff trim --base <ref> --file <path> --write`. --base defaults to
# `git merge-base origin/main HEAD` and --file to docs/roadmaps/roadmap.yaml.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/roadmap_bin.sh
. "$ROOT/scripts/lib/roadmap_bin.sh" || exit 2
exec "$ROADMAP_BIN" trim "$@"
