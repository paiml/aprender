#!/usr/bin/env bash
# check_crux_nightly_smoke.sh -- the nightly CRUX smoke judge still refuses what it must (#4702)
#
#   bash scripts/check_crux_nightly_smoke.sh
#
# scripts/release/crux_nightly_smoke.sh is what release day reads instead of running CRUX:
# the night's full-lane receipts for H, judged by the smoke rule. This guard runs its case
# table (fixtures under scripts/release/crux_nightly_cases/) and its mutants, so a change
# that lets a missing host, a missing smoke cell, a RED cell, a missing control or a
# receipt from another binary through goes red here, before any release reads it.
# Cargo-free, so guard-tree runs it on every PR.
#
# EXIT 0 the judge's self-test passes · 1 it does not.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
JUDGE="$ROOT/scripts/release/crux_nightly_smoke.sh"
if [ ! -f "$JUDGE" ]; then
    echo "FAIL  $JUDGE is absent -- release day would have no nightly smoke judge"
    exit 1
fi
if bash "$JUDGE" --self-test; then
    echo "check_crux_nightly_smoke: PASS"
    exit 0
fi
echo "check_crux_nightly_smoke: FAIL"
exit 1
