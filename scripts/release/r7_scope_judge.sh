#!/usr/bin/env bash
# r7_scope_judge.sh -- PUBLISH_PREFLIGHT_LADDER_JUDGE for the 0.69.1 release ONLY (operator emergency scope).
# The cut's own check_publish_preflight.sh calls its R7 judge as `bash <judge> --version <v>` from ROOT.
# This hands that call to main's judge under the RECORDED scope, bound to the cut and the canonical CRUX dir.
set -uo pipefail
: "${R7_SCOPE_TREE:?main checkout carrying the emergency_scopes entry}" "${R7_CUT:?the cut sha}" "${R7_CRUX:?the CRUX dir}"
[ "${1:-}" = "--version" ] && [ -n "${2:-}" ] || { echo "FAIL  r7_scope_judge: called as [$*], want --version <v>"; exit 2; }
[ "$(git rev-parse HEAD)" = "$R7_CUT" ] || { echo "FAIL  r7_scope_judge: ROOT HEAD $(git rev-parse HEAD) is not the cut $R7_CUT"; exit 1; }
cd "$R7_SCOPE_TREE" || exit 2
CRUX_CERT="$R7_SCOPE_TREE/evidence/crux/$2/prompt-certification.json" \
  exec bash scripts/check_model_ladder.sh --version "$2" --scope crux-smoke --cut-commit "$R7_CUT" --crux "$R7_CRUX"
