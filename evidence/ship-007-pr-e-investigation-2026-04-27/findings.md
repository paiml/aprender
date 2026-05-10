## SHIP-007 PR E investigation — §28 hypothesis is incomplete (2026-04-27)

### Setup
- Host: noah-Lambda-Vector (RTX 4090)
- Binary: `/mnt/nvme-raid0/targets/aprender/release/apr` rebuilt at HEAD of main + §27/§28/§29 spec branch
- Teacher: `/mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.apr` (8.04 GB, 339 tensors, 196 Q4K + 143 F32)

### Empirical findings (refute §28 narrow hypothesis)

**Finding 1 — `q4k_layers` IS populated for the canonical 7B teacher** (see `check_q4k_population.txt`):
- All 28 layers have Q/K/V/O + gate/up/down Q4K bytes (e.g. layer 0 Q=7,225,344 bytes, K=V=1,032,192 bytes, gate=up=down=38,191,104 bytes).
- The hypothesis in §28.4 ("APR currently stores weights as Vec<f32> (dequantized)") is FALSE for this teacher.

**Finding 2 — APR's F32 fused qkv weight is NUMERICALLY EQUIVALENT to Q4K dispatch within Q4K tolerance** (see `diag_apr_qkv_layer0.txt`):
- Path A (F32 fused qkv via `helpers::f32_matmul`): Q-out mean=-0.003912, std=0.260898
- Path B (Q4K bytes via `fused_q4k_parallel_matvec`): Q-out mean=-0.003899, std=0.260868
- max |diff| = 0.005294 ; RMS diff = 0.000673 — both well within ±5% Q4K tolerance.
- This means **the F32 fused qkv weight in `layer.qkv_weight` is correctly dequantized + concatenated**.

**Finding 3 — `fused_q4k_parallel_matvec_into` ALREADY uses the GGUF Q8K-acts path** (see `parallel_k.rs:285-306`):
- The kernel internally does `quantize_activations_q8k_into` → `fused_q4k_q8k_parallel_matvec_into`.
- §28.4's claim that APR "uses dequantized weights with F32 matmul" while GGUF "uses Q4K-aware fused" is incomplete: APR's Q4K dispatch path uses the SAME inner kernel as GGUF.

### What's still unexplained

The trace evidence (`evidence/ship-007-apr-vs-gguf-2026-04-27/{apr,gguf}-trace.txt`) shows:
- **APR layer 0 qkv: mean=0.2559, std=10.3291**
- **GGUF layer 0 qkv: mean=-0.0163, std=1.1402**

That's a **9× std divergence at layer 0** — propagating to layer 3's 18.23× ffn_swigl ratio.

But:
- The static F32 vs Q4K test on the same weight + synthetic input shows zero divergence.
- The fused qkv weight is statistically reasonable (std≈0.022, range [-0.61, 0.60]).

So either:
- (a) The trace's "qkv" stat measures something post-bias-add and post-RoPE that diverges (RoPE could rotate inputs into a high-std basis at certain positions/frequencies).
- (b) The 7-token prompt produces embeddings that hit a different code path than my synthetic input (e.g., Q8K activation quantization edge case).
- (c) An upstream operation (RMSNorm precision, embedding lookup) differs between APR and GGUF in a way that the trace doesn't decompose.
- (d) `qkv_bias` is being applied on APR but not GGUF (or with different values).

### Implication for PR E

The §28 fix (replace `helpers::f32_matmul` with Q4K-fused dispatch in `AprTransformer::matmul`) **would not change runtime behavior** for the FFN gate/up/down/attn_output paths in `apr_swiglu_ffn` because those already dispatch Q4K via `seq_matmul_q4k` → `fused_q4k_parallel_matvec_into` (which IS the Q8K-acts kernel).

The only `helpers::f32_matmul` call sites in APR's forward path are:
- Line 331: fused QKV matmul (uses `layer.qkv_weight` F32)
- Line 458: LM head matmul (uses `self.lm_head_weight` F32)
- Fallback paths in `apr_swiglu_ffn` / `apr_attn_output_projection` when q4k_layer is None (NOT this teacher)

Of these, switching QKV to Q4K dispatch (per `q4k_layers[i].attn_q/k/v_weight`) is the only candidate that touches the divergence-source layer. But Finding 2 shows the F32 fused qkv weight IS numerically equivalent to Q4K dispatch, so this swap should produce <0.5% std change — NOT enough to close a 9× gap.

### Recommended next step

The §28 spec section overcommitted to a specific code site before sufficient empirical narrowing. The next bisection step is:
1. **Bisect the trace's "qkv" stat**: capture the qkv tensor BEFORE bias-add and BEFORE RoPE. Compare APR vs GGUF.
2. **Bisect post-bias**: capture qkv AFTER bias-add but BEFORE RoPE. Compare.
3. **Bisect post-RoPE**: capture full output. Compare.

This will identify whether the 9× divergence is from the matmul (already shown equivalent), from `qkv_bias`, or from RoPE.

Both APR's `forward()` (pmat-260.rs:331-388) and GGUF's `forward_traced` capture activation stats at specific points; the contracts of "what counts as `qkv`" may differ between the two implementations.

### Files

- `check_q4k_population.txt` — proves q4k_layers fully populated for canonical 7B teacher
- `diag_apr_qkv_layer0.txt` — proves F32 fused qkv = Q4K dispatch within tolerance
- `crates/aprender-serve/examples/check_q4k_population.rs` — diagnostic source
- `crates/aprender-serve/examples/diag_apr_qkv_layer0.rs` — diagnostic source
