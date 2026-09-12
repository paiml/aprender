#!/usr/bin/env bash
# L0-1b acceptance — every A_i re-run in one call (I5). Hardware legs need the
# release-host binary (`. scripts/apr_bin.sh` or APR=<cuda build>) and the two
# manifest models under $APR_MODELS_DIR (default $HOME/models).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 2
red=0; n=0
MODELS_DIR="${APR_MODELS_DIR:-$HOME/models}"   # never a machine path: the guard counts one, and another host has another dir
leg() { n=$((n+1)); if "$@" >/tmp/l01b-leg.$n 2>&1; then echo "ok    A$n  $*"; else echo "FAIL  A$n  $* (rc=$?)"; tail -5 /tmp/l01b-leg.$n; red=1; fi; }
CT="${CARGO_TARGET_DIR:-}"
leg env ${CT:+CARGO_TARGET_DIR=$CT} cargo test -p apr-cli --lib parity_per_op_table
leg env ${CT:+CARGO_TARGET_DIR=$CT} cargo test -p aprender-serve --lib per_op_tap
leg bash -c '. scripts/pv_bin.sh >/dev/null 2>&1 && "$PV" validate contracts/apr-parity-per-op-v1.yaml'
leg test -s docs/audits/l0-1b-arms.md
if [ -n "${APR:-}" ] && [ -x "$APR" ]; then
    PROMPT=$(sed -n "s/^PROMPT='\(.*\)'\$/\1/p" scripts/check_model_parity.sh)   # the corpus prompt, read as text (bashrs SEC001)
    OUT=$(mktemp -d "${TMPDIR:-/tmp}/l01b-accept.XXXXXX")
    # A5 — after step 2 the 1.5B names NO op and its lm_head row clears the threshold; the RED twin is
    # the pre-fix binary (71a25c2bb: post_ffn_residual layer 26, lm_head 0.950827), recorded in the receipt.
    leg bash -c "\"$APR\" parity $MODELS_DIR/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --prompt \"$PROMPT\" --per-op --out $OUT/15 --json > $OUT/15.json && python3 -c \"import json; d=json.load(open('$OUT/15.json')); assert d['positions']>=64, d['positions']; assert d['first_divergence'] is None, d['first_divergence']; lm=[r for r in d['rows'] if r['stage']=='lm_head'][0]; assert lm['min_cosine']>=0.98, lm; print('1.5B: no diverging op; lm_head', lm['min_cosine'])\""
    leg bash -c "\"$APR\" parity $MODELS_DIR/qwen2.5-coder-7b-instruct-q4_k_m.gguf --prompt \"$PROMPT\" --per-op --out $OUT/7 --json > $OUT/7.json && python3 -c \"import json; d=json.load(open('$OUT/7.json')); assert d['positions']>=64; assert d['first_divergence'] is None, d['first_divergence']; print('7B clean')\""
    [ -n "${OUT:-}" ] && [ "$OUT" != / ] && [ -d "$OUT" ] && rm -rf "$OUT"   # validated before rm -rf (bashrs SEC011)
else
    echo "SKIP  A5/A6 hardware legs: set APR=<cuda-built apr> on a CUDA host (not a pass)"; red=1
fi
echo "$((n-red>0?n:0)) legs run"; [ "$red" = 0 ]
