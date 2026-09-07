#!/usr/bin/env bash
# R-0b acceptance — every A_i in one call (I5). RED until P2 lands.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT" || exit 2
red=0; n=0
leg() { n=$((n+1)); if "$@" >/tmp/r0b-leg.$n 2>&1; then echo "ok    A$n  $*"; else echo "FAIL  A$n  $* (rc=$?)"; tail -3 /tmp/r0b-leg.$n; red=1; fi; }
leg bash -c '[ "$(git grep -c "cfg!(any(feature = \"cuda\"" -- crates/apr-cli/src ":!crates/apr-cli/src/backend.rs" ":!crates/apr-cli/src/**/tests*" | awk -F: "{s+=\$2} END{print s+0}")" = 0 ]'
leg env ${CARGO_TARGET_DIR:+CARGO_TARGET_DIR=$CARGO_TARGET_DIR} cargo test -p apr-cli --test backend_refusal_case_table
leg bash scripts/check_backend_registry.sh --static
leg bash -c '. scripts/pv_bin.sh >/dev/null 2>&1 && "$PV" validate contracts/apr-backend-registry-v1.yaml'
echo "$n legs"; [ "$red" = 0 ]
