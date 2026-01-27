# PMAT-114: Five-Whys Analysis - APR CUDA Output Quality

## Status: TWO ROOT CAUSES FIXED - CALIBRATION REMAINING

## Problem Statement
APR CUDA (F32) produces garbage output while GGUF (Q4K) produces correct output for the same prompt.

**Test Case**: `"2+2="` with ChatML template
- **GGUF Output**: "2+2 equals 4." (correct)
- **APR Output (before fixes)**: " <|fim_suffix|>AK" (garbage)
- **APR Output (after fixes)**: "5" (close but wrong by 0.52 logits)

## Five-Whys Analysis

### Q1: Why garbage output?
**Answer**: The logit distribution from forward_cuda is incorrect.
- Token 0 should have "The" (785) as top token
- Instead has space (220) as top token
- Evidence: APR_TRACE_LOGITS shows different top-5 tokens

### Q2: Why different logit distribution?
**Answer**: QKV projection produces very different K values.
- APR K mean: -0.059579 (pre-fix)
- GGUF K mean: 5.757745
- Difference: ~100x magnitude, wrong sign

### Q3: Where does forward_cuda diverge from GGUF?
**Answer**: QKV bias was NOT being applied in APR path.
- GGUF code applies K bias with mean=5.697892 after QKV projection
- APR was computing K without bias
- K BEFORE bias: mean≈0.06 (matches APR pre-fix)
- K AFTER bias: mean≈5.76 (matches GGUF)

### Q4: Why was APR missing the bias?
**Answer**: APR converter (converter.rs) was NOT fusing QKV bias tensors.
- Only QKV weights were fused (PMAT-101)
- QKV biases were silently dropped during import
- SafeTensors has `q_proj.bias`, `k_proj.bias`, `v_proj.bias`
- These were never written to the APR file

### Q5: ROOT CAUSE #1
**Answer**: Missing QKV bias fusion in APR SafeTensors import pipeline.
- Location: `aprender/src/format/converter.rs::write_apr_file()`
- The PMAT-101 fix added weight fusion but not bias fusion
- Qwen2 models use `attention_bias=True` by default
- **FIX IMPLEMENTED**: converter.rs now fuses Q/K/V biases

### Q6: Why was output "22" after QKV bias fix?
**Answer**: rope_type was 0 (NORM style) but Qwen2.5 requires rope_type=2 (NEOX style)
- APR_TRACE_CONFIG showed: `rope_type=0 (raw=None)`
- NEOX style rotates first half with second half of embedding dimension
- NORM style rotates adjacent pairs
- Wrong rotation style = wrong position encoding = wrong output

### Q7: ROOT CAUSE #2
**Answer**: rope_type was not being written to APR metadata during import.
- Location: `aprender/src/format/v2.rs::AprV2Metadata` - missing field
- Location: `aprender/src/format/converter.rs` - not extracting/writing rope_type
- **FIX IMPLEMENTED**:
  - Added rope_type field to AprV2Metadata
  - Added rope_type inference in load_model_config_from_json (qwen2 → type 2)
  - Added rope_type to metadata writing path

## Fix Implementation

### 1. v2.rs (aprender) - AprV2Metadata
```rust
/// RoPE type: 0=NORM (adjacent pairs), 2=NEOX (split halves)
/// CORRECTNESS-011: Qwen2.5 models require rope_type=2 (NEOX style)
#[serde(default)]
pub rope_type: Option<u32>,
```

### 2. converter.rs (aprender) - QKV Bias Fusion + rope_type
```rust
// PMAT-114: Also fuse Q, K, V biases if present (Qwen2 has attention bias)
let q_bias_name = format!("model.layers.{layer_idx}.self_attn.q_proj.bias");
// ... fuse biases into qkv_proj.bias

// PMAT-114: Infer rope_type from architecture
let rope_type = match architecture.as_deref() {
    Some("qwen2") | Some("qwen2.5") | Some("qwen") => Some(2), // NEOX style
    _ => Some(0), // Default to NORM style
};
```

### 3. cuda.rs (realizar) - QKV Bias Application
```rust
// PMAT-114: Apply QKV bias if present (Qwen2 has attention bias)
if let Ok(qkv_bias) = self.model.get_tensor_f32(&fused_bias_name) {
    // Unfuse and apply bias to Q, K, V
}
```

## Verification

**Before fixes:**
```
[APR CONFIG] rope_type=0 (raw=None)  // WRONG
[PMAT-114] L0 after QKV: K mean=-0.059579  // WRONG - no bias
APR Output: " <|fim_suffix|>AK" (garbage)
```

**After fixes:**
```
[APR CONFIG] rope_type=2 (raw=Some(2))  // CORRECT - NEOX style
[PMAT-114] L0 after QKV: K mean=5.638313  // CORRECT - bias applied
APR Output: "5"  // Close but wrong by 0.52 logits
```

**Logit analysis (after fixes):**
```
Top 5 tokens: [(20="5", 18.14), (19="4", 17.62), (18="3", 17.56), ...]
Correct answer "4" is rank 2 with delta 0.52 logits
```

## Files Modified
1. `aprender/src/format/v2.rs` - Added rope_type to AprV2Metadata
2. `aprender/src/format/converter.rs` - QKV bias fusion + rope_type inference
3. `aprender/src/format/gguf.rs` - Added rope_type to GgufModelConfig
4. `realizar/src/apr/cuda.rs` - QKV bias application during forward pass

## Remaining Work
- **CALIBRATION**: Output is "5" (logit 18.14) instead of "4" (logit 17.62)
- Delta is only 0.52 logits - model is very close
- Possible causes:
  1. F32 vs Q4K precision differences
  2. Other subtle numerical differences in forward pass
  3. Tokenizer mismatch (using Qwen2.5-Coder tokenizer for Qwen2 model)

## Tooling Gap Identified
- `apr rosetta` doesn't have inference comparison between formats
- `apr qa` only supports GGUF format, not APR
- **TODO**: Add `apr rosetta compare-inference` for debugging parity issues
