#!/usr/bin/env bash
# L0-1a accept.sh — re-runs every A_i in one call (I5). GREEN on a GPU-less host by judging the
# recorded runs; the live manifest (`check_model_parity.sh --manifest`) is the fleet-verify leg on
# lambda and gx10 (card acceptance vi), never faked here.
set -uo pipefail; cd "$(dirname "$0")/../.."; rc=0
run() { printf '== %s\n' "$*"; "$@"; local r=$?; printf 'rc=%s\n' "$r"; [ "$r" = 0 ] || rc=1; }
expect_fail() { printf '== (must FAIL) %s\n' "$*"; if "$@"; then printf 'rc=0 (wanted non-zero)\n'; rc=1; else printf 'rc=%s (as required)\n' "$?"; fi; }
CARGO=/home/noah/.cargo/bin/cargo; export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/mnt/nvme-raid0/agent-wt/target-l01}"
run bash scripts/derive_model_manifest.sh --self-test
run bash scripts/derive_model_manifest.sh --check
run bash scripts/check_model_parity.sh --self-test
run bash scripts/check_model_parity.sh --judge evidence/parity/l0-1/lambda/qwen2.5-coder-7b-instruct-q4_k_m.json --model qwen2.5-coder-7b-instruct
expect_fail bash scripts/check_model_parity.sh --judge evidence/parity/l0-1/lambda/qwen2.5-coder-1.5b-instruct-q4_k_m.json --model qwen2.5-coder-1.5b-instruct
expect_fail bash scripts/check_model_parity.sh --judge tests/fixtures/parity/defective/one-position-at-0.5.json --model qwen2.5-coder-7b-instruct
run env "$CARGO" test -p apr-cli --test reg15_admission
run env "$CARGO" test -p apr-cli --lib sentinel_tests
run env "$CARGO" test -p aprender-serve --lib parity_report_carries
. scripts/pv_bin.sh >/dev/null 2>&1 && run "$PV" validate contracts/apr-gpu-cpu-parity-v1.yaml
if [ -d "${APR_MODELS_DIR:-$HOME/models}" ] && [ -n "${APR_BIN_FOR_C14:-}" ]; then run bash scripts/check_model_parity.sh --manifest --apr "$APR_BIN_FOR_C14"; else printf '== check_model_parity.sh --manifest: not on this host (no models dir / no cuda apr) — the fleet-verify leg on lambda and gx10\n'; fi
exit "$rc"
