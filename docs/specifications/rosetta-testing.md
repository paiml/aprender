# SQLite-Style Conversion Test Harness

**Status:** Implemented / Verified
**Refs:** GH-196, PMAT-197
**Code:** `src/format/test_factory.rs`, `src/format/converter/tests/core.rs`

## Theoretical Foundation

This testing strategy is grounded in the epistemological principles of **Critical Rationalism** and the operational excellence of the **Toyota Production System (TPS)**.

### 1. Popperian Falsification
We reject the notion of "verifying" the converter is correct. Instead, we adopt the stance of **falsification** [1]. Every test case is a severe attempt to refute the hypothesis: *"The conversion pipeline preserves tensor data and metadata fidelity across formats."*
- **Conjecture:** The `ConversionTestHarness` generates rigorous conjectures (e.g., "Round-tripping Llama-style tensors preserves bitwise integrity").
- **Refutation:** The `verify()` and `round_trip_safetensors()` methods actively seek discrepancies (refutations). A test passes only if it survives this rigorous scrutiny, corroborating (but not proving) the pipeline's reliability.

> *"In so far as a scientific statement speaks about reality, it must be falsifiable: and in so far as it is not falsifiable, it does not speak about reality."* — Karl Popper

### 2. The Toyota Way (Jidoka & Standard Work)
The harness implements key TPS principles [2] to ensure built-in quality (*Jidoka*) and consistency:
- **Jidoka (Autonomation):** The harness automatically detects abnormalities (e.g., shape mismatches, epsilon deviations) and stops the process immediately with detailed diagnostics, preventing defects from passing downstream.
- **Standardized Work:** By replacing ad-hoc `/tmp/` paths and manual BTreeMap construction with `PygmyConfig` and `TempDir`, we establish a stable, repeatable standard for creating tests. This reduces variability (Mura) and overburden (Muri) on developers.
- **Genchi Genbutsu (Go and See):** The harness does not assume a write was successful; it reads the data back from disk (`verify()`) to observe the actual facts.

### References
1. Popper, K. (1959). *The Logic of Scientific Discovery*. Basic Books.
2. Liker, J. K. (2004). *The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer*. McGraw-Hill.

## Problem

The conversion pipeline (SafeTensors <-> APR <-> GGUF) is error-prone (GH-196 had 4 defects). The legacy tests in `core.rs` suffered from:
1. **Ad-hoc Data:** Manual BTreeMap construction instead of using the robust `PygmyConfig`.
2. **Fragile Paths:** Hardcoded `/tmp/` paths causing race conditions and littering.
3. **Blind Writes:** No read-back verification to ensure data integrity.
4. **Missing Round-Trips:** SafeTensors->APR->SafeTensors never tested end-to-end.

## Solution: The ConversionTestHarness

We implemented a **SQLite-style Test Harness** in `src/format/test_factory.rs` that provides a RAII-managed environment for rigorous conversion testing.

### Reference API (Standard Work)

The `ConversionTestHarness` provides a fluent API for testing. Developers **must** use this harness for all conversion tests.

```rust
use crate::format::test_factory::harness::ConversionTestHarness;
use crate::format::test_factory::PygmyConfig;

// ONE-LINER (Preferred for regression tests)
ConversionTestHarness::assert_import_ok(PygmyConfig::llama_style());
ConversionTestHarness::assert_roundtrip_ok(PygmyConfig::default());

// FLUENT API (Preferred for edge cases)
let h = ConversionTestHarness::new()
    .with_safetensors(PygmyConfig::llama_style()) // 1. Setup Input
    .import_to_apr(ImportOptions::default());     // 2. Exercise SUT

// 3. Verify (Jidoka) - Panics with detailed mismatch info
h.verify_apr().assert_passed(); 
```

### Key Components

| Component | Responsibility | Falsification Check |
|-----------|----------------|---------------------|
| `TempDir` | RAII cleanup of test artifacts. | **F-STR-04:** Deletion during test triggers IO error. |
| `PygmyConfig` | Generates deterministic, valid tensor data. | **F-STR-02:** Handles 0-tensor configs gracefully. |
| `verify_apr()` | Reads disk artifact, compares data with `ToleranceConfig`. | **F-JID-01:** Detects single-byte corruption. |
| `verify_safetensors()` | Verifies export fidelity (round-trip). | **F-REG-03:** Input vs Output binary identity (approx). |

## Implementation Details

### 1. Harness Module (`src/format/test_factory.rs`)
The `harness` module (~200 lines) implements the `ConversionTestHarness` struct.
- **Tolerance:** Default F32=1e-6, F16=1e-3, Q8=0.1.
- **Verification:** Explicitly checks (1) Tensor Existence, (2) Shape Equality, (3) Data values within tolerance.
- **Safety:** Uses `unwrap()` only on test setup; SUT errors are returned or asserted with context.

### 2. Core Tests Rewrite (`src/format/converter/tests/core.rs`)
We replaced the legacy `/tmp/` tests with:
- **`tests_conversion`:** Uses harness for standard flow, keeps manual `BTreeMap` only for negative testing (NaN, invalid LayerNorm).
- **`tests_gh196_roundtrip`:** 8 new regression tests covering the GH-196 defects (Auto-Arch, Strict Mode, Round-Trip).

## Falsification Protocol (QA Matrix)

To ensure the harness remains reliable, any changes to `test_factory.rs` must pass this Falsification Matrix.

| ID | Test | Expectation |
|----|------|-------------|
| **F-HAR-01** | Manually corrupt output `.apr` byte | `verify()` PANICS with `DataMismatch` |
| **F-HAR-02** | Set tolerance to `1e-9` (too strict) | `verify()` FAILS on float arithmetic noise |
| **F-HAR-03** | Use `--strict` on `embedding_only` config | Import FAILS (Unverified Architecture) |
| **F-HAR-04** | Use `PygmyConfig` with 0 tensors | Harness handles gracefully (no crash) |
| **F-REG-01** | Round-trip Llama-style tensors | `verify_safetensors()` PASSES |

## Verification Commands

```bash
# 1. Run Harness Unit Tests (Self-Verification)
cargo test --lib -- test_factory::harness

# 2. Run Converter Regression Tests (The actual workload)
cargo test --lib -- converter::tests::core

# 3. Run GH-196 Specific Regression
cargo test --lib -- gh196
```