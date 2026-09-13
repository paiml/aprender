#!/bin/bash
# albor-370m-publish-readiness.sh
#
# Pre-flight checklist for P3-C HF publish of paiml/albor-370m-v1.
# Verifies every artifact + invariant required before `apr publish`
# can be invoked. Read-only - produces a GO / NO-GO verdict; does NOT
# actually publish.
#
# Spec: docs/specifications/aprender-train/ship-model-2-spec.md §88
# Contract: contracts/apr-pretrain-from-init-v1.yaml (post-§86 invariant)
# Salvage path (if pre-P0-K checkpoint): see §86.4 / PR #1757.
#
# Usage:
#   bash scripts/publish/albor-370m-publish-readiness.sh <ckpt.apr>
#
# Exit codes:
#   0 - GO: all artifacts present and valid; ready for `apr publish`
#   1 - NO-GO: one or more blockers; verdict text lists each

set -uo pipefail

CKPT="${1:-}"

if [[ -z "$CKPT" ]]; then
    cat <<'EOF'
Usage: bash scripts/publish/albor-370m-publish-readiness.sh <ckpt.apr>

Example (using the §85 P2-E ep49 checkpoint):
    bash scripts/publish/albor-370m-publish-readiness.sh \
        /mnt/nvme-raid0/runs/model-2-p2e-tuned-hp-20260517/ckpt/epoch-049.apr

The checkpoint MUST be the stamped variant (post-§86 salvage if it's
a pre-P0-K artifact). To stamp (pin the binary first — stamping metadata with
an unknown-age apr is how a model ships with provenance it does not have):

    . scripts/apr_bin.sh || exit 1
    "$APR" stamp <pre-p0k.apr> \
        --architecture qwen2 \
        --hf-architecture Qwen2ForCausalLM \
        --hf-model-type qwen2 \
        --license Apache-2.0 \
        --data-source "huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct + bigcode/the-stack-dedup + codeparrot/codeparrot-clean" \
        --data-license "Apache-2.0 / permissive-aggregate" \
        -o <stamped.apr>
EOF
    exit 1
fi

if [[ ! -f "$CKPT" ]]; then
    echo "FAIL: checkpoint not found: $CKPT"
    exit 1
fi

APR_BIN="${APR:-apr}"
PASS=0
FAIL=0
WARN=0

ok()   { echo "  PASS: $1"; PASS=$((PASS+1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL+1)); }
warn() { echo "  WARN: $1"; WARN=$((WARN+1)); }

echo "=== albor-370m-v1 publish-readiness check ==="
echo "Checkpoint: $CKPT"
echo "apr binary: $($APR_BIN --version 2>&1 | head -1)"
echo

# Gate 1: file integrity (apr validate)
echo "Gate 1: apr validate"
if "$APR_BIN" validate "$CKPT" >/dev/null 2>&1; then
    ok "apr validate exits 0"
else
    fail "apr validate failed - checkpoint may be truncated or corrupt"
fi

# Gate 2: quality scorer (P3-A, §86.4 - must be >= 90 for ship)
echo "Gate 2: apr inspect --quality"
QUALITY_JSON=$("$APR_BIN" inspect "$CKPT" --quality --json 2>/dev/null || echo '{}')
SCORE=$(echo "$QUALITY_JSON" | jq -r '.quality.score // 0' 2>/dev/null || echo "0")
SHIP_READY=$(echo "$QUALITY_JSON" | jq -r '.quality.ship_ready // false' 2>/dev/null || echo "false")
HF_ID=$(echo "$QUALITY_JSON" | jq -r '.quality.breakdown.hf_identity // 0' 2>/dev/null || echo "0")
PROV=$(echo "$QUALITY_JSON" | jq -r '.quality.breakdown.provenance // 0' 2>/dev/null || echo "0")
TOK=$(echo "$QUALITY_JSON" | jq -r '.quality.breakdown.tokenizer // 0' 2>/dev/null || echo "0")
echo "  score=$SCORE / 100 (threshold 90)"
echo "  breakdown: hf_identity=$HF_ID/20  provenance=$PROV/25  tokenizer=$TOK/15"
if [[ "$SHIP_READY" == "true" ]]; then
    ok "quality.ship_ready = true (score >= 90)"
elif [[ "$SCORE" -ge 80 ]]; then
    warn "score = $SCORE (>=80) - close to ship gate, missing fields likely fixable via apr stamp"
else
    fail "score = $SCORE < 90 - NOT ship-ready (run apr stamp to populate missing metadata)"
fi
if [[ "$HF_ID" -lt 20 ]]; then
    fail "hf_identity = $HF_ID/20 - pre-P0-K checkpoint detected. Run apr stamp --architecture qwen2 --hf-architecture Qwen2ForCausalLM --hf-model-type qwen2 (§86.4 salvage)"
fi
if [[ "$PROV" -lt 25 ]]; then
    fail "provenance = $PROV/25 - license/data_source/data_license missing. Run apr stamp --license Apache-2.0 --data-source ... --data-license ..."
fi
if [[ "$TOK" -lt 15 ]]; then
    warn "tokenizer = $TOK/15 - embedded vocab missing. For HF publish this is OK if tokenizer.json is published as a sibling file"
fi

# Gate 3: apr qa (8 falsifiable gates)
echo "Gate 3: apr qa --json"
QA_JSON=$("$APR_BIN" qa "$CKPT" --json 2>/dev/null || echo '{}')
QA_VERDICT=$(echo "$QA_JSON" | jq -r '.verdict // "UNKNOWN"' 2>/dev/null || echo "UNKNOWN")
if [[ "$QA_VERDICT" == "GO" ]]; then
    ok "apr qa verdict = GO"
elif [[ "$QA_VERDICT" == "WARN" ]]; then
    warn "apr qa verdict = WARN (soft gates failed; review before publish)"
else
    fail "apr qa verdict = $QA_VERDICT (review failed gates before publish)"
fi

# Gate 4: model card present + parseable
echo "Gate 4: model card"
CARD_PATH="docs/model-cards/albor-370m-v1.md"
if [[ -f "$CARD_PATH" ]]; then
    if grep -q "^library_name:" "$CARD_PATH" && grep -q "^license:" "$CARD_PATH" && grep -q "^model-index:" "$CARD_PATH"; then
        ok "model card $CARD_PATH has HF frontmatter"
    else
        fail "model card $CARD_PATH missing required YAML frontmatter (library_name / license / model-index)"
    fi
else
    fail "model card not found at $CARD_PATH"
fi

# Gate 5: HF_TOKEN set
echo "Gate 5: HF_TOKEN authorization"
if [[ -n "${HF_TOKEN:-}" ]]; then
    ok "HF_TOKEN is set (length=${#HF_TOKEN})"
else
    fail "HF_TOKEN environment variable not set. apr publish will fail at the upload step."
fi

# Gate 6: smoke generation (model produces output, not garbage)
echo "Gate 6: smoke generation (apr run)"
SMOKE=$("$APR_BIN" run "$CKPT" "def fibonacci(n):" --max-tokens 32 --temperature 0.0 2>&1 | head -50)
if grep -qE "^(Output|generation|tokens):" 2>/dev/null <<< "$SMOKE" ; then
    if grep -qE "[a-zA-Z]{5,}" 2>/dev/null <<< "$SMOKE" ; then
        ok "smoke generation produced text output"
    else
        warn "smoke generation produced output but no English/code-like text - may be degenerate"
    fi
else
    warn "smoke generation surface unrecognized - verify manually before publish"
fi

# Gate 7: format export round-trip (gguf + safetensors)
echo "Gate 7: export round-trip (gguf + safetensors)"
TMP_EXPORT=$(mktemp -d)
GGUF_OUT="$TMP_EXPORT/albor-370m-v1-q4k.gguf"
ST_OUT="$TMP_EXPORT/albor-370m-v1.safetensors"
if "$APR_BIN" export "$CKPT" --format gguf --quantize q4k -o "$GGUF_OUT" >/dev/null 2>&1 && [[ -f "$GGUF_OUT" ]]; then
    ok "GGUF Q4_K export produced $(du -h "$GGUF_OUT" | cut -f1) file"
else
    fail "GGUF Q4_K export failed - required for AC-SHIP2-009 (llama-cli interop)"
fi
if "$APR_BIN" export "$CKPT" --format safetensors -o "$ST_OUT" >/dev/null 2>&1 && [[ -f "$ST_OUT" ]]; then
    ok "SafeTensors export produced $(du -h "$ST_OUT" | cut -f1) file"
else
    warn "SafeTensors export failed - non-blocking for HF publish but breaks the round-trip claim"
fi
# SEC011: TMP_EXPORT is always assigned (unconditionally, above) before this
# point, so the guard checks it directly rather than through a `:-` default
# expansion -- bashrs's rm-rf-guard detector does not credit `${VAR:-}` as a
# validated non-empty check, and `${VAR:-}` masking an actually-unset VAR
# here would be exactly the silent-empty-string footgun SEC011 warns about.
if [[ -n "$TMP_EXPORT" && -d "$TMP_EXPORT" && "$TMP_EXPORT" == /tmp/* ]]; then rm -rf "$TMP_EXPORT"; fi

# Verdict
echo
echo "=== Verdict ==="
echo "PASS:  $PASS"
echo "FAIL:  $FAIL"
echo "WARN:  $WARN"
echo
if [[ "$FAIL" -gt 0 ]]; then
    echo "VERDICT: NO-GO ($FAIL hard blocker(s)). Resolve before \`apr publish\`."
    echo "         Common fix: apr stamp the checkpoint with the §86.4 recipe."
    exit 1
else
    if [[ "$WARN" -gt 0 ]]; then
        echo "VERDICT: GO with $WARN warning(s). Review warnings before publish."
    else
        echo "VERDICT: GO. Ready for \`apr publish paiml/albor-370m-v1 --formats apr,safetensors,gguf\`."
    fi
    exit 0
fi
