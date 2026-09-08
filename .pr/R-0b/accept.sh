#!/usr/bin/env bash
# R-0b acceptance — every A_i in one call (I5).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT" || exit 2
CT="${CARGO_TARGET_DIR:-}"
red=0; n=0
leg() { n=$((n+1)); if "$@" >/tmp/r0b-leg.$n 2>&1; then echo "ok    A$n  $*"; else echo "FAIL  A$n  $* (rc=$?)"; tail -4 /tmp/r0b-leg.$n; red=1; fi; }
# A1: zero cfg!(feature=cuda|wgpu) backend reads in apr-cli outside registry.rs
leg bash scripts/check_backend_registry.sh --static
# A2: the resolution case table — a forced accelerator never downgrades to cpu
leg env ${CT:+CARGO_TARGET_DIR=$CT} cargo test -p apr-cli --test backend_refusal_case_table
# A3: the static guard's own case table (both polarities)
leg bash scripts/check_backend_registry.sh --self-test
# A4: the registry-resolution unit tests (Request classification, selected line)
leg env ${CT:+CARGO_TARGET_DIR=$CT} cargo test -p apr-cli --lib registry::tests
# A6 (A3 in the plan): GET /v1/effective-config carries the startup resolution (REG-12)
leg env ${CT:+CARGO_TARGET_DIR=$CT} cargo test -p aprender-serve --lib effective_config_route_pp2
# A5: the contract validates through the pin
leg bash -c '. scripts/pv_bin.sh >/dev/null 2>&1 && "$PV" validate contracts/apr-backend-registry-v1.yaml'
echo "$n legs"; [ "$red" = 0 ]
