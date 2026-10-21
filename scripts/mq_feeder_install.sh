#!/usr/bin/env bash
# mq_feeder_install.sh -- PMAT-5843. Install the feeder AND the arm script it calls into the path the user units run
# (%h/.local/libexec/mq-feeder). The feeder resolves arm_pr_automerge.sh next to itself, so both must be there.
# Run from a checkout of origin/main; re-run after either script changes (forjar converges the same files on the hosts).
# Usage: mq_feeder_install.sh [DEST]
set -euo pipefail
src="$(cd "$(dirname "$0")" && pwd)"
dest="${1:-$HOME/.local/libexec/mq-feeder}"
mkdir -p "$dest"
for s in mq_feeder.sh arm_pr_automerge.sh; do
  install -m 0755 "$src/$s" "$dest/$s"
done
sha256sum "$dest/mq_feeder.sh" "$dest/arm_pr_automerge.sh" > "$dest/INSTALLED.sha256"
echo "installed to $dest"
