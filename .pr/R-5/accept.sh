#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT" || exit 2
red=0; n=0
leg() { n=$((n+1)); if "$@" >/tmp/r5-leg.$n 2>&1; then echo "ok    A$n  $*"; else echo "FAIL  A$n  $* (rc=$?)"; tail -3 /tmp/r5-leg.$n; red=1; fi; }
leg bash -c 'bash scripts/publish_cascade.sh --help 2>&1 | grep -q -- --dry-run'
leg bash scripts/publish_cascade.sh --self-test
leg bash -c '. scripts/pv_bin.sh >/dev/null 2>&1 && "$PV" validate contracts/apr-publish-cascade-v1.yaml'
leg bash -c 'bash scripts/publish_cascade.sh --list | head -1 | grep -q .'
leg bash -c 'bashrs lint scripts/publish_cascade.sh 2>&1 | grep -qE "0 error"'
echo "$n legs"; [ "$red" = 0 ]
