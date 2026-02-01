# SQLite-Style Conversion Test Harness

**Status:** Approved
**Refs:** GH-196, PMAT-197

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

The conversion pipeline (SafeTensors <-> APR <-> GGUF) is error-prone (GH-196 had 4 defects). The existing converter tests in `core.rs` have these problems:

1. **Don't use pygmy factory** -- manually construct adhoc BTreeMap tensors
2. **Hardcoded `/tmp/` paths** -- `"/tmp/test_valid_input.safetensors"` etc.
3. **Manual cleanup** -- `fs::remove_file(input).ok()` (silently fails)
4. **No read-back verification** -- writes APR but never reads it back to check tensor data
5. **No round-trip testing** -- SafeTensors->APR->SafeTensors never tested end-to-end

Meanwhile, `test_factory.rs` (1,519 lines) has excellent pygmy builders that only 4 test files use.

## Solution

Add a `ConversionTestHarness` to `test_factory.rs` and migrate `core.rs` tests to use it.

## Changes

### 1. Extend `src/format/test_factory.rs` -- add harness module (~200 lines)

Add `#[cfg(test)] pub(crate) mod harness` after the existing builders. Contains:

- **`ConversionTestHarness`** struct with `TempDir` (RAII cleanup, no manual deletion)
- **Setup methods**: `with_safetensors(config)`, `with_apr(config)` -- write pygmy bytes to tempdir
- **Exercise methods**: `import_to_apr(options)`, `export_to_safetensors(options)` -- run real pipeline
- **Verify methods**: `verify()` -- read back output, compare tensor data with tolerance
- **Round-trip**: `round_trip_safetensors()` -- SafeTensors->APR->SafeTensors end-to-end
- **Convenience**: `assert_import_ok(config)`, `assert_roundtrip_ok(config)` -- one-liners

Key implementation details:
- Read APR back via `AprV2Reader::from_bytes(&fs::read(path)?)` -> `get_tensor(name).shape` + `get_tensor_as_f32(name)`
- Read SafeTensors back via `MappedSafeTensors::open(path)` -> `tensor_names()` + `get_tensor(name)` + `get_metadata(name).shape`
- `ToleranceConfig`: F32=1e-6, F16=1e-3, Q8=0.1, Q4=0.5
- `VerificationResult` with `assert_passed()` that panics with detailed mismatch info

### 2. Rewrite `src/format/converter/tests/core.rs` -- `tests_conversion` module

Replace 6 existing tests that use `/tmp/` paths:
- `test_convert_valid_safetensors` -> use harness with `PygmyConfig::llama_style()`
- `test_convert_invalid_layernorm_fails_strict` -> keep manual tensor (intentionally bad data)
- `test_convert_invalid_layernorm_force_succeeds` -> keep manual tensor (intentionally bad data)
- `test_convert_nan_fails` -> keep manual tensor (intentionally bad data)
- `test_convert_nonexistent_file` -> keep as-is (tests error path)
- `test_name_mapping_whisper` -> use harness, verify tensor names in output

For the manual tensor tests that need bad data: convert `/tmp/` paths to `TempDir`.

### 3. Add GH-196 regression tests -- new `tests_gh196_roundtrip` module in `core.rs`

~8 new tests covering the exact paths that broke:

```
test_gh196_auto_arch_import          -- Auto architecture infers from tensor names
test_gh196_strict_rejects_unverified -- --strict blocks Auto architecture
test_gh196_default_permissive        -- default (non-strict) allows Auto
test_gh196_tensor_data_preserved     -- imported tensors match source bit-for-bit
test_gh196_full_roundtrip_default    -- SafeTensors->APR->SafeTensors round-trip
test_gh196_full_roundtrip_llama      -- round-trip with PygmyConfig::llama_style()
test_gh196_full_roundtrip_minimal    -- round-trip with PygmyConfig::minimal()
test_gh196_metadata_architecture     -- architecture field preserved in APR metadata
```

## Files Modified

| File | Change |
|------|--------|
| `src/format/test_factory.rs` | Add `harness` module (~200 lines) |
| `src/format/converter/tests/core.rs` | Rewrite `tests_conversion`, add `tests_gh196_roundtrip` |

## Verification

```bash
cargo test --lib -- test_factory::harness     # Harness self-tests
cargo test --lib -- converter::tests::core    # Rewritten converter tests
cargo test --lib -- gh196                     # GH-196 regression tests
cargo clippy -- -D warnings                  # Lint clean
```
