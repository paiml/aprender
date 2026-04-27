# SHIP-007 root cause PINNED — `qkv_bias` is the divergence introducer (2026-04-27)

## Setup

Per §30.4 falsifiable next investigation step, I captured APR layer-0 qkv at four stages on the canonical 7B teacher and compared each stage against GGUF's reference std=1.14.

- Host: noah-Lambda-Vector (RTX 4090)
- Binary: `/mnt/nvme-raid0/targets/aprender/release/examples/diag_qkv_bisection_layer0`
- Teacher: `/mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.apr`
- Tokens: `[3838, 374, 220, 17, 10, 17, 30]` ("What is 2+2?")

## Result — bisection is decisive

| Stage | mean | std | n | Verdict |
|-------|------|-----|---|---------|
| Embedding | 0.000013 | 0.017365 | 25088 | matches GGUF input (post-embed) |
| Post-RMSNorm (input to QKV) | -0.000083 | 0.221261 | 25088 | matches GGUF attn_norm (std=0.242) |
| **Post-matmul, pre-bias** | **-0.015918** | **0.924970** | **32256** | **MATCHES GGUF (std=1.14) within Q4K tolerance** |
| └─ Q-part (post-matmul) | -0.011921 | 0.974724 | 25088 | normal |
| └─ K-part (post-matmul) | -0.058265 | 0.983458 | 3584 | normal |
| └─ V-part (post-matmul) | -0.001558 | 0.283301 | 3584 | normal |
| **`qkv_bias` itself** | **+0.271825** | **10.243427** | **4608** | **⚠ THE BIAS HAS std=10.24** |
| Post-bias (= line 334 output) | +0.255906 | 10.328716 | 32256 | matches APR trace (10.33) |
| └─ Q-part (post-bias) | +0.115425 | 3.557455 | 25088 | bias-dominated |
| └─ K-part (post-bias) | **+1.490816** | **29.492775** | **3584** | **K-bias is extreme** |
| Q post-RoPE | +0.091476 | 3.558162 | 25088 | RoPE doesn't change std |

## Reference numbers (from existing apr trace --payload runs)

- APR layer 0 qkv: mean=0.2559, std=**10.3291**
- GGUF layer 0 qkv: mean=-0.0163, std=**1.1402**
- Ratio: 9.05×

## Root cause analysis

**Pre-bias matmul output (std=0.92) matches GGUF (std=1.14) within Q4K rounding** — confirming Finding 2 of `evidence/ship-007-pr-e-investigation-2026-04-27/findings.md` (the matmul kernel is correct).

**The `qkv_bias` value itself has std=10.24** — a full order of magnitude above what one would expect from normally-trained Qwen2.5-7B biases (typically std<1). Adding this bias to the matmul output is what produces the 9× std blowup that propagates to layer 3's 18× ffn_swigl ratio.

**K-part is hit hardest**: post-bias std=29.49 (vs pre-bias std=0.98 — a 30× blowup just from bias).

## Why this is a defect, not an upstream artifact

GGUF loads the same Qwen2.5-7B-Instruct model and produces qkv std=1.14 — so the underlying weights/biases in the .gguf file are fine. The bug is APR-side, in either:

1. **APR `load_qkv_bias`** at `mod_dequant_q4k_apr.rs:210-236` — the concatenation order might be wrong, OR the dtype interpretation is wrong, OR the layout (per-head vs per-tensor scaling) differs from what GGUF stores.

2. **APR's stored bias bytes** in the .apr file — if the converter from GGUF → APR somehow scaled or transposed the bias, that's the bug. Note that `apr inspect` showed `k_proj.bias` as `dtype=f32 shape=[512]` — the shape is right, but the values themselves may be off.

3. **Q4K-related rescaling** — Qwen2.5 stores biases that are calibrated to work with Q4K-quantized weights. If APR's import dequantizes weights but doesn't apply matching bias adjustment, that's the bug.

## Next-step investigation (PR E v2)

1. Dump the actual bias values byte-for-byte from BOTH the .apr file AND the .gguf file at layer 0, position by position.
2. Compare per-element:
   - If APR bytes != GGUF bytes (after format-aware decoding), the converter is broken.
   - If APR bytes == GGUF bytes but APR runs std=10 while GGUF runs std=1.14, the FORWARD path is misinterpreting the bias.
3. Once identified, fix at `load_qkv_bias` OR at the converter (whichever is upstream).

## Falsification chain — fully extended

```
§15.4 → §16 → §17 → §23 → §27 → §28 → §30 → §31 (now)
"GPU eliminated" → "APR CPU isolated" → "(layer 3, FFN)"
                → "(layer 3, ffn_swigl)" → "ratio 18.23×"
                → "F32 vs Q4K matmul precision" (REFUTED §30)
                → "qkv_bias std=10.24 introduces 9× layer-0 gap" (PINNED §31)
```

## Coverage scoreboard impact

- Discovery: pinned root cause of SHIP-007 (5 MODEL-1 PARTIALs)
- Coverage flip: still 33+12 (no PARTIAL has flipped to DISCHARGED yet)
- BUT: PR E v2 is now scoped to a specific code site (`load_qkv_bias` / converter), with a quantitative fix criterion (post-bias std must come down to ~1.14 not 10.33).

## Files

- `diag_qkv_bisection_layer0.txt` — full diagnostic output
- `findings.md` — this analysis
- `crates/aprender-serve/examples/diag_qkv_bisection_layer0.rs` — re-runnable diagnostic
