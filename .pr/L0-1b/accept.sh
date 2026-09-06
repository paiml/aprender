#!/usr/bin/env bash
# L0-1b acceptance — every A_i re-run in one call (I5). Hardware legs need the
# release-host binary (`. scripts/apr_bin.sh` or APR=<cuda build>) and the two
# manifest models under /home/noah/models.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 2
red=0; n=0
leg() { n=$((n+1)); if "$@" >/tmp/l01b-leg.$n 2>&1; then echo "ok    A$n  $*"; else echo "FAIL  A$n  $* (rc=$?)"; tail -5 /tmp/l01b-leg.$n; red=1; fi; }
CT="${CARGO_TARGET_DIR:-}"
leg env ${CT:+CARGO_TARGET_DIR=$CT} cargo test -p apr-cli --lib parity_per_op_table
leg env ${CT:+CARGO_TARGET_DIR=$CT} cargo test -p aprender-serve --lib per_op_tap
leg bash -c '. scripts/pv_bin.sh >/dev/null 2>&1 && "$PV" validate contracts/apr-parity-per-op-v1.yaml'
leg test -s docs/audits/l0-1b-arms.md
if [ -n "${APR:-}" ] && [ -x "$APR" ]; then
    eval "$(grep '^PROMPT=' scripts/check_model_parity.sh)"
    OUT=$(mktemp -d "${TMPDIR:-/tmp}/l01b-accept.XXXXXX")
    leg bash -c "\"$APR\" parity /home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --prompt \"$PROMPT\" --per-op --out $OUT/15 --json > $OUT/15.json && python3 -c \"import json,sys; d=json.load(open('$OUT/15.json')); f=d['first_divergence']; assert d['positions']>=64, d['positions']; assert f and f['layer']==26, f; lm=[r for r in d['rows'] if r['stage']=='lm_head'][0]; assert abs(lm['min_cosine']-0.9508)<1e-3, lm; print('1.5B names', f)\""
    leg bash -c "\"$APR\" parity /home/noah/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf --prompt \"$PROMPT\" --per-op --out $OUT/7 --json > $OUT/7.json && python3 -c \"import json; d=json.load(open('$OUT/7.json')); assert d['positions']>=64; assert d['first_divergence'] is None, d['first_divergence']; print('7B clean')\""
    rm -rf "$OUT"
else
    echo "SKIP  A5/A6 hardware legs: set APR=<cuda-built apr> on a CUDA host (not a pass)"; red=1
fi
echo "$((n-red>0?n:0)) legs run"; [ "$red" = 0 ]
