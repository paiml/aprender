# Qwen2-0.5B-Instruct gibberish bisection — 2026-05-04

## Summary

`apr run qwen2.5-coder-0.5b-instruct.apr` produces gibberish (CJK/Polish byte fragments). Root cause empirically pinned via `apr diff` to **LAYOUT-001/002 contract violation** in safetensors→APR FFN tensor import.

## Trustworthy facts (verified, not guessed)

1. **Model weights are correct.** `apr trace` on the same model converted to GGUF produces coherent top-5 logits: `<|im_end|>` (10.54), ` The` (10.47), ` For` (10.18), ` Each` (10.15). See `gguf-trace-coherent-logits.json`.

2. **Bug class:** `apr diff <0.5b.apr> <0.5b.gguf> --values --limit 5` outputs:
   ```
   [TRANSPOSED] model.layers.0.mlp.down_proj.weight
     TRANSPOSED shapes: [896, 4864] vs [4864, 896]
     dist: 100.0% ident, 0.0% small, 0.0% med, 0.0% large
   [TRANSPOSED] model.layers.0.mlp.gate_proj.weight
     TRANSPOSED shapes: [4864, 896] vs [896, 4864]
   [TRANSPOSED] model.layers.0.mlp.up_proj.weight
     TRANSPOSED shapes: [4864, 896] vs [896, 4864]
   [IDENTICAL] model.layers.0.post_attention_layernorm
   DIAGNOSIS: Values identical, shapes transposed (format layout diff)
   ```

3. **Why 7B works, 0.5B fails:**
   - 7B Qwen2.5-Coder was GGUF-imported → APR file inherits GGUF FFN layout → kernel-compatible
   - 0.5B Qwen2.5-Coder was safetensors-imported → APR file preserves HF SafeTensors `[out, in]` layout → kernel-incompatible

## Falsified hypotheses (do not re-investigate)

| # | Hypothesis | Falsification |
|---|-----------|---------------|
| 1 | Tied-embedding shape orientation `[vocab, hidden]` vs `[hidden, vocab]` | Both runtime paths handle correctly |
| 2 | dtype string case mismatch `"f16"` vs `"F16"` | Writer emits uppercase per `tensor_index_impl.rs:188` |
| 3 | `dtype_to_qtype("F16")` falls through to F32 | Returns 1 correctly per `mapped_apr_model.rs:212` |
| 4 | `fused_matmul` doesn't support qtype=1 | Has explicit F16 branch at `matmul_fused.rs:116-121` |

## Five Whys

1. **Why does 0.5B `apr run` produce gibberish?** FFN matmul reads weights in wrong orientation → garbage output propagated through 24 layers.
2. **Why wrong orientation?** APR FFN tensors stored in HF axis-0=`out_dim` convention; GGUF/kernel uses opposite.
3. **Why APR has HF layout?** Safetensors→APR import path preserved HF shape labels without transposing to canonical APR row-major form.
4. **Why no transpose?** The import code path lacks the LAYOUT-001/002 transpose step that GGUF→APR import has.
5. **Why undetected?** No round-trip falsification gate compares post-import APR shapes against GGUF reference. `apr diff` finds it instantly but isn't run automatically post-import.

## Methodology lesson

This investigation burned ~15 turns on falsified hypotheses (shape orientation, dtype case, F16 dispatch) before running `apr diff`. CLAUDE.md mandates "use apr tools first" — internalize as: when investigating any model output defect, **run `apr diff` and `apr qa --verbose` before any code reading**.

## Next-session scope (NOT in this PR)

Fix is in `crates/aprender-core/src/format/converter/` — safetensors→APR import for `mlp.{down,gate,up}_proj.weight` needs to call layout transpose to match GGUF/kernel convention. Bounded scope: ~50 LOC + drift-prevention test using `apr diff` + a contract gate.

## Related contracts

- `tied-embeddings-v1` (existing — phase `transpose_embed`)
- `tensor-layout-v1` (existing — LAYOUT-001/002 SOURCE OF TRUTH)
- This finding suggests adding **TENSOR-LAYOUT-FFN-001**: post-import shape labels for `mlp.{down,gate,up}_proj.weight` must match GGUF reference, falsifiable via `apr diff` round-trip.
