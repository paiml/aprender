#!/usr/bin/env bash
# L0-1 accept.sh — re-runs every A_i in one call (I5)
set -uo pipefail; cd "$(dirname "$0")/../.."; rc=0
run() { printf '== %s\n' "$*"; "$@"; local r=$?; printf 'rc=%s\n' "$r"; [ "$r" = 0 ] || rc=1; }
run bash scripts/derive_model_manifest.sh --self-test
run bash scripts/derive_model_manifest.sh --check
run bash scripts/check_model_parity.sh --self-test
# the horizon record on THIS host (needs an apr built from HEAD with cuda and the models under $APR_MODELS_DIR)
if [ -n "${APR_MODELS_DIR:-}" ] || [ -d "$HOME/models" ]; then run bash scripts/check_model_parity.sh --manifest; else printf '== check_model_parity.sh --manifest: UNMEASURED on this host (no models dir)\nrc=1\n'; rc=1; fi
exit "$rc"
