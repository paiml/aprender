# Falsification Prompt: Tensor Layout Contract (LAYOUT-CONTRACT-001)

**Date:** 2026-02-04
**Spec Version:** 9.21.0
**Target:** `contracts/tensor-layout-v1.yaml` and `src/format/layout_contract.rs`

## Popperian Falsification Protocol

The goal is to **disprove** the claims made by the tensor layout contract. If we cannot falsify them, they are corroborated (not proven).

---

## Claim F-LAYOUT-CONTRACT-001: All 2D Weights Are Transposed

**Hypothesis:** For all tensors in the contract with `transpose: true`, the APR shape equals the swapped GGUF shape.

**Falsification Test:**
```bash
cargo test --lib format::layout_contract::tests::test_f_layout_contract_001_all_2d_transposed
```

**Expected Outcome:** PASS (hypothesis corroborated)
**Falsification Criteria:** Any tensor with `transpose: true` that has `apr_shape == gguf_shape` (not swapped)

---

## Claim F-LAYOUT-CONTRACT-002: lm_head Shape Matches Kernel Expectation

**Hypothesis:** The lm_head tensor in APR format has shape `[vocab_size, hidden_dim]`.

**Falsification Test:**
```bash
cargo test --lib format::layout_contract::tests::test_f_layout_contract_002_lm_head_shape
```

**Expected Outcome:** PASS (hypothesis corroborated)
**Falsification Criteria:**
- `lm_head.apr_shape[0] != vocab_size` OR
- `lm_head.apr_shape[1] != hidden_dim`

---

## Claim F-LAYOUT-CONTRACT-003: 1D Tensors Unchanged

**Hypothesis:** For all tensors with `transpose: false`, the APR shape equals the GGUF shape.

**Falsification Test:**
```bash
cargo test --lib format::layout_contract::tests::test_f_layout_contract_003_1d_unchanged
```

**Expected Outcome:** PASS (hypothesis corroborated)
**Falsification Criteria:** Any 1D tensor (layernorms) with `apr_shape != gguf_shape`

---

## Claim F-LAYOUT-CONTRACT-004: Byte Size Matches Kernel Expectation

**Hypothesis:** Quantized tensor byte size = `out_dim * ceil(in_dim / QK_K) * block_bytes`

**Falsification Test:**
```bash
cargo test --lib format::layout_contract::tests::test_f_layout_contract_004_byte_size
```

**Expected Outcome:** PASS (hypothesis corroborated)
**Falsification Criteria:** Calculated bytes != expected bytes for Q4K or Q6K

---

## Claim F-LAYOUT-CONTRACT-005: Pattern Matching Works for All Layers

**Hypothesis:** Tensor names with layer numbers (e.g., `blk.15.attn_q.weight`) correctly match contract patterns.

**Falsification Test:**
```bash
cargo test --lib format::layout_contract::tests::test_pattern_matching
```

**Expected Outcome:** PASS (hypothesis corroborated)
**Falsification Criteria:** Any layer-numbered tensor that doesn't match `blk.{n}.*.weight` pattern

---

## Claim F-LAYOUT-CONTRACT-006: Global CONTRACT Static Is Thread-Safe

**Hypothesis:** The `CONTRACT` static can be accessed from multiple threads without data races.

**Falsification Test:**
```bash
cargo test --lib format::layout_contract::tests::test_global_contract
```

**Expected Outcome:** PASS (hypothesis corroborated)
**Falsification Criteria:** Data race or panic when accessing CONTRACT concurrently

---

## End-to-End Falsification: GH-202 Regression

**Hypothesis:** APR converted from GGUF produces coherent output, not garbage `[PAD]` tokens.

**Falsification Test:**
```bash
# Convert GGUF to APR
apr import model.gguf -o model.apr

# Run inference
apr run model.apr --prompt "2+2=" --max-tokens 10

# Expected: Contains "4"
# Falsified if: Contains "[PAD" or gibberish
```

**Expected Outcome:** Output contains "4" (hypothesis corroborated)
**Falsification Criteria:** Output contains `[PAD`, `olumbia`, or non-ASCII garbage

---

## Summary Test Command

Run all falsification tests:
```bash
cargo test --lib format::layout_contract 2>&1 | grep -E "(ok|FAILED|test result)"
```

Expected: `test result: ok. 10 passed; 0 failed`
