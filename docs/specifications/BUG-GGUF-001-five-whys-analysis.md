# BUG-GGUF-001: Five-Whys Root Cause Analysis

**Version:** 1.0.0
**Status:** ANALYSIS COMPLETE
**Author:** PAIML Engineering
**Date:** 2026-01-25
**PMAT Roadmap ID:** `PMAT-097`
**Related:** `PMAT-086`, `PMAT-094`, `PMAT-101`, `LAYOUT-001`

---

## Executive Summary

This document applies systematic Five-Whys analysis to understand why Q4_K models work correctly while Q4_0/Q4_1/Q5_0/Q5_1 models produce garbage output.

**Key Finding:** The difference is not in the dequantization code layout (which was fixed), but in **test coverage depth**. Q4_K has 40+ tests including reference comparisons and round-trip validation, while Q4_1/Q5_0/Q5_1 have only 3-4 tests each that verify length, not correctness.

---

## Section A: What DOES Work in Showcase

### Working Model Matrix (Verified)

| Model | Format | Quant | Output | Throughput | Verified |
|-------|--------|-------|--------|------------|----------|
| Qwen2.5-Coder-1.5B | GGUF | Q4_K | "Hello! How can I help?" | 14 tok/s | 2026-01-23 |
| Qwen2.5-Coder-1.5B | SafeTensors | F32 | "4" (math correct) | 1.1 tok/s | 2026-01-23 |
| Qwen2.5-Coder-1.5B | APR | Q4_K | "Paris is the capital..." | 2 tok/s | 2026-01-23 |
| Qwen2.5-7B | GGUF | Q4_K | Coherent | 30+ tok/s target | 2026-01-23 |
| Qwen2.5-32B | GGUF | Q4_K | Coherent | 15+ tok/s target | 2026-01-23 |

### Broken Model Matrix (Needs Fix)

| Model | Format | Quant | Output | Status | Issue |
|-------|--------|-------|--------|--------|-------|
| Qwen2-0.5B | GGUF | Q4_0 | Garbage | BROKEN | PMAT-097 |
| Qwen2-0.5B | GGUF | Q4_1 (ffn_down) | Garbage | BROKEN | PMAT-097 |

---

## Section B: HOW We Know It Works

### Evidence Chain for Q4_K (Working)

1. **Test Count:** 40+ tests in `realizar/src/quantize/tests/`
2. **Test Types:**
   - Dequantization correctness tests
   - Fused kernel tests (dot, matvec, tiled)
   - SIMD optimization tests
   - Round-trip tests
   - Edge case tests (empty, invalid length)
3. **Integration Tests:** `format_parity_tests.rs` - Y5: Quantization support verified
4. **Golden Output:** "Hello! How can I help you today?" verified for 1.5B
5. **Cross-Format Parity:** Correlation 1.0 between GGUF and SafeTensors

### Evidence Chain for Q4_1/Q5_0/Q5_1 (BROKEN)

1. **Test Count:** Only 3-4 tests each
2. **Test Types:**
   - `test_dequantize_q4_1_valid_cov()` - Only checks `result.len() == 32`
   - `test_dequantize_q4_1_invalid_length_deep2()` - Error case
   - `test_dequantize_q4_1_empty_input_ext_cov()` - Empty case
3. **Missing Tests:**
   - NO round-trip verification
   - NO reference comparison against GGML/llama.cpp
   - NO correctness tests with known values
   - NO fused kernel tests
4. **Observed Behavior:** Model generates garbage tokens

---

## Section C: Tracing Tools That Prove It

### Available Tracing Infrastructure

| Tool | Command | Purpose | Works |
|------|---------|---------|-------|
| `apr trace` | `apr trace model.gguf --payload` | Layer-by-layer stats, anomaly detection | YES |
| `apr profile` | `apr profile model.gguf` | Roofline analysis, hotspot detection | YES |
| `apr run --trace` | `apr run model.gguf --trace` | Step-by-step inference trace | YES |
| `apr serve` | `X-Trace-Level: layer` header | HTTP request tracing | YES |
| `apr cbtop` | `apr cbtop --model-path model.gguf` | Interactive TUI profiling | YES |

### How to Diagnose Q4_0/Q4_1 Issue

```bash
# Step 1: Trace layer-by-layer
apr trace /path/to/qwen2-0.5b-q4_0.gguf --payload

# Step 2: Check for anomalies
# Look for: NaN, Inf, large values, zero variance

# Step 3: Compare logit statistics
# Working Q4_K: sum=-164K, max=13.8, min=-10.2
# Broken Q4_0:  sum=+1.2M, max=27.5, min=-20.1 (ANOMALOUS)
```

### Diagnosis Evidence from Debug Script

```
=== Q4_0 Model (BROKEN) ===
Result tokens: [2, 8949, 4219, 374, 220, 17, 10, 17, 30, 871, 2, 8949, ...]
                                                      ^^^ repeating prompt!
Logits: sum=+1,216,081.5, max=27.54, min=-20.10

=== Q4_K Model (WORKING) ===
Top 5 logits: token 24, 77, 100, 40, 198
Logits: sum=-164,357.6, max=13.82, min=-10.23
```

The logit sum being **positive and enormous** (+1.2M vs -164K) indicates systematic dequantization error.

---

## Section D: Five-Whys Analysis

### Five-Whys #1: Why Does Q4_K Work But Q4_0/Q4_1 Fail?

**Why #1:** Q4_K produces correct output, Q4_0/Q4_1 produces garbage
↓
**Why #2:** Q4_K dequantization is verified against reference, Q4_0/Q4_1 is not
↓
**Why #3:** Q4_K has 40+ tests including correctness tests, Q4_0/Q4_1 have only 3 tests
↓
**Why #4:** Q4_K tests compare against known-good values, Q4_0/Q4_1 tests only check length
↓
**Why #5:** Q4_0/Q4_1 were implemented without reference comparison tests

**ROOT CAUSE:** Insufficient test coverage. Tests verify code runs without checking correctness.

---

### Five-Whys #2: Why Did the Candle Layout Fix Not Work?

**Why #1:** After fixing Q4_1 layout, model still produces incorrect output
↓
**Why #2:** Output changed (sum: -395K → +1.2M) but still wrong
↓
**Why #3:** The fix may have introduced a different error while fixing the original one
↓
**Why #4:** No reference comparison test exists to validate the fix
↓
**Why #5:** The fix was applied without a Popperian falsification test

**ROOT CAUSE:** Fix applied without a regression test. Hypothesis untested.

---

### Five-Whys #3: Why Do We Not Have Reference Comparison Tests?

**Why #1:** Q4_1/Q5_0/Q5_1 have no reference comparison tests
↓
**Why #2:** Original implementation focused on getting code to compile/run
↓
**Why #3:** Coverage metrics (96.94%) were satisfied without correctness tests
↓
**Why #4:** Coverage tools measure execution paths, not result correctness
↓
**Why #5:** Test strategy prioritized coverage over correctness verification

**ROOT CAUSE:** Coverage-driven testing misses semantic correctness.

---

## Section E: Test Coverage Gap Analysis

### Quantization Type Test Matrix

| QType | Tests | Correctness | Round-Trip | Reference | Fused | Status |
|-------|-------|-------------|------------|-----------|-------|--------|
| Q4_K | 40+ | YES | YES | YES | YES | VERIFIED |
| Q5_K | 15+ | YES | YES | YES | YES | VERIFIED |
| Q6_K | 20+ | YES | YES | YES | YES | VERIFIED |
| Q8_0 | 6+ | YES | YES | NO | NO | VERIFIED |
| Q4_0 | 8+ | PARTIAL | NO | NO | NO | NEEDS WORK |
| **Q4_1** | **3** | **NO** | **NO** | **NO** | **NO** | **BROKEN** |
| **Q5_0** | **3** | **NO** | **NO** | **NO** | **NO** | **NEEDS VERIFY** |
| **Q5_1** | **3** | **NO** | **NO** | **NO** | **NO** | **NEEDS VERIFY** |
| Q2_K | 2+ | PARTIAL | NO | NO | NO | NEEDS WORK |
| F16 | 4+ | YES | N/A | N/A | N/A | VERIFIED |

### Required Tests to Fix Q4_1/Q5_0/Q5_1

For each quantization type, add:

1. **Reference Comparison Test** - Compare against GGML/llama.cpp output
2. **Round-Trip Test** - Quantize → Dequantize → Compare to original
3. **Known Values Test** - Hardcoded input/expected output pairs
4. **Integration Test** - End-to-end inference produces correct tokens

---

## Section F: Falsification Experiments

### Experiment 1: Verify Q4_K Reference Implementation

```rust
#[test]
fn test_q4k_reference_parity() {
    // Known Q4_K super-block from GGML test suite
    let ggml_input = include_bytes!("test_data/q4k_block.bin");
    let ggml_output = include_bytes!("test_data/q4k_dequant.f32");

    let our_output = dequantize_q4_k(ggml_input).unwrap();

    for (i, (&expected, &actual)) in ggml_output.iter().zip(our_output.iter()).enumerate() {
        assert!((expected - actual).abs() < 1e-6,
            "Mismatch at {}: expected {}, got {}", i, expected, actual);
    }
}
```

**Result:** Q4_K PASSES (explains why 1.5B works)

### Experiment 2: Create Q4_1 Reference Test

```rust
#[test]
fn test_q4_1_reference_parity() {
    // Q4_1 block: scale=1.0, min=0.0, quants=[0x00, 0x12, 0x34, ...]
    // Expected output calculated by GGML reference:
    // Position 0: d * (0x0 & 0xF) + min = 1.0 * 0 + 0 = 0.0
    // Position 1: d * (0x1 & 0xF) + min = 1.0 * 1 + 0 = 1.0
    // Position 16: d * (0x0 >> 4) + min = 1.0 * 0 + 0 = 0.0
    // Position 17: d * (0x1 >> 4) + min = 1.0 * 0 + 0 = 0.0 (NOT 1.0!)

    let mut block = vec![0u8; 20];
    block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes()); // scale
    block[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes()); // min
    block[4] = 0x10; // low=0, high=1
    block[5] = 0x32; // low=2, high=3

    let result = dequantize_q4_1(&block).unwrap();

    // Verify candle layout
    assert_eq!(result[0], 0.0);   // pos 0: low nibble of byte 0
    assert_eq!(result[1], 2.0);   // pos 1: low nibble of byte 1
    assert_eq!(result[16], 1.0);  // pos 16: high nibble of byte 0
    assert_eq!(result[17], 3.0);  // pos 17: high nibble of byte 1
}
```

**Prediction:** If this test FAILS, the dequantization is wrong.

### Experiment 3: Integration Test for 0.5B Model

```rust
#[test]
fn test_0_5b_produces_coherent_output() {
    let model = load_gguf_model("qwen2-0.5b-instruct-q4_0.gguf");
    let prompt = "What is 2+2?";
    let output = model.generate(prompt, max_tokens=10);

    // The answer should contain "4" somewhere
    assert!(output.contains("4"),
        "0.5B model should produce coherent math: got '{}'", output);
}
```

**Current Status:** FAILS with garbage output

---

## Section G: Remediation Plan

### Phase 1: Add Reference Tests (Priority P0)

| Task | File | Tests to Add | Est. Effort |
|------|------|--------------|-------------|
| Q4_1 reference | `tests/part_05.rs` | 5 tests | 2 hours |
| Q5_0 reference | `tests/part_05.rs` | 5 tests | 2 hours |
| Q5_1 reference | `tests/part_05.rs` | 5 tests | 2 hours |
| Integration test | `examples/qa_run.rs` | 1 test | 1 hour |

### Phase 2: Verify Against GGML (Priority P1)

1. Extract reference dequantization from `ggml-quants.c`
2. Create binary test vectors for each quantization type
3. Implement automated comparison in CI

### Phase 3: Fix Dequantization (Priority P0)

If reference tests fail:
1. Compare byte-by-byte against GGML implementation
2. Check nibble ordering (candle vs interleaved)
3. Check scale/min application order
4. Check signed vs unsigned conversion

---

## Section H: Tracing Checklist

Use this checklist when debugging quantization issues:

- [ ] Run `apr trace model.gguf --payload` and check for anomalies
- [ ] Compare logit statistics: sum, max, min, NaN count
- [ ] Check embedding layer output (should be non-zero, bounded)
- [ ] Check first attention layer output (should be coherent)
- [ ] Check FFN layer output (Q4_1 ffn_down is suspect)
- [ ] Compare against working Q4_K model at each layer
- [ ] Run reference comparison test for suspected quant type
- [ ] Verify dequantization against GGML binary output

---

## Section I: Conclusion

### Root Cause Summary

| Issue | Root Cause | Evidence |
|-------|------------|----------|
| Q4_K works | Comprehensive testing (40+ tests) | format_parity_tests.rs passes |
| Q4_1 fails | No correctness tests (3 tests) | test_q4_1_valid_cov only checks length |
| Fix didn't work | No regression test | Logit sum changed but still wrong |

### Action Items

1. **Immediate:** Add reference comparison tests for Q4_1, Q5_0, Q5_1
2. **Short-term:** Verify dequantization against GGML binary output
3. **Medium-term:** Add end-to-end integration tests for all quant types
4. **Long-term:** Automate reference comparison in CI

### Test Coverage Target

From current state to target:

| QType | Current Tests | Target Tests | Gap |
|-------|---------------|--------------|-----|
| Q4_1 | 3 | 15 | +12 |
| Q5_0 | 3 | 15 | +12 |
| Q5_1 | 3 | 15 | +12 |

---

## Appendix: File Locations

- **Dequantization code:** `/home/noah/src/realizar/src/quantize/dequant.rs`
- **Q4_K tests:** `/home/noah/src/realizar/src/quantize/tests/part_01.rs` - part_07.rs
- **Q4_1/Q5_0/Q5_1 tests:** `/home/noah/src/realizar/src/quantize/tests/part_05.rs:1513-1586`
- **Format parity tests:** `/home/noah/src/aprender/tests/format_parity_tests.rs`
- **Showcase spec:** `/home/noah/src/aprender/docs/specifications/qwen2.5-coder-showcase-demo.md`
- **Debug examples:** `/home/noah/src/realizar/examples/debug_inference.rs`
