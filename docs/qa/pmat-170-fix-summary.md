# PMAT-170: Q4_K GPU Explosion Fix

**Date:** 2026-01-29
**Status:** FIXED
**Severity:** P0 (Showcase-blocking)

## Problem

APR models with `--gpu` flag produced garbage output ("veisveisveisveisveis") due to
hidden state explosion through transformer layers.

**Evidence (before fix):**
```
[PMAT-114] After layer 0: mean=-0.116661, max=11.210302
[PMAT-114] After layer 1: mean=-0.459027, max=35.231682
[PMAT-114] After layer 27: mean=-8475.701172, max=124856.054688  ← EXPLOSION
```

## Root Cause

The `dequantize_q4_k` function in `src/apr/mod.rs` had incorrect element ordering
that didn't match the working `fused_q4k_parallel_matvec` kernel.

**Bug:** Elements were interleaved (L0, H0, L1, H1, ...)
**Fix:** Elements must be sequential (L0, L1, ..., L31, H0, H1, ..., H31)

Additionally, the scale extraction logic was incorrect:
- Bug: Same scale for both low and high nibbles
- Fix: Different scales (is for low nibbles, is+1 for high nibbles)

## Files Changed

1. **`src/apr/mod.rs`** - Fixed `dequantize_q4_k` and `dequantize_q6_k`:
   - Changed element ordering from interleaved to sequential (4 chunks of 64)
   - Added `extract_scale_min_q4k` helper matching `fused_k.rs`
   - Fixed Q6_K layout to match `fused_q6k_dot`

2. **`src/quantize/fused_k.rs`** - Added regression test:
   - `test_q4k_layout_consistency_pmat170`: Verifies dequantize matches fused kernel

## Verification

**Test Results:**
```
test quantize::fused_k::tests::test_q4k_layout_consistency_pmat170 ... ok
test result: ok. 489 passed; 0 failed (Q4K tests)
```

**GPU Hidden State (after fix):**
```
input L2: 0.8780
QKV L2: 1248.4465
attn_output L2: 10.5989
block output L2: 11.8829
final hidden L2: 82.0405  ← STABLE (not 124856)
```

## Technical Details

### Before (BROKEN)
```rust
// Interleaved: L0, H0, L1, H1, ... (WRONG)
for j in 0..8 {
    let scale = d * f32::from(scales[j]);
    for l in 0..16 {
        let q_byte = qs[j * 16 + l];
        result.push((q_byte & 0x0F) as f32 * scale);
        result.push((q_byte >> 4) as f32 * scale);
    }
}
```

### After (FIXED)
```rust
// Sequential: L0..L31, H0..H31 per 64-element chunk
for j in (0..256).step_by(64) {
    let q = &qs[j / 2..j / 2 + 32];
    let is = j / 32;
    let (sc1, m1) = extract_scale_min_q4k(&scales, is);     // Low nibbles
    let (sc2, m2) = extract_scale_min_q4k(&scales, is + 1); // High nibbles

    // First: ALL 32 low nibbles
    for &byte in q {
        result.push(d * sc1 * (byte & 0x0F) as f32 - dmin * m1);
    }
    // Then: ALL 32 high nibbles
    for &byte in q {
        result.push(d * sc2 * (byte >> 4) as f32 - dmin * m2);
    }
}
```

## Remaining Issue

The APR model outputs appear as empty/newlines, suggesting a separate issue with
tokenizer decoding or model conversion. This is a different bug from #170.

- GGUF + GPU: Works correctly ("Hello! I'm just a computer program...")
- GGUF + CPU: Works correctly
- APR + CPU: Empty output (possible conversion issue)
- APR + GPU: Empty output (but no longer explodes - #170 is FIXED)

## References

- LAYOUT-001 specification in CLAUDE.md
- `fused_q4k_dot` reference implementation in `quantize/fused_k.rs`
- Iron Lotus QA Protocol Round 4 Report
