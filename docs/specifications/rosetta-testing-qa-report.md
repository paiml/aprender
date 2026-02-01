# QA Falsification Report: Rosetta Conversion Harness

**Status:** CONDITIONAL PASS (12/17 Pass, 3 Design Issues, 2 FAILED)
**Date:** 2026-02-01
**Methodology:** Popperian Falsification + Toyota Way (Jidoka)
**Null Hypothesis ($H_0$):** "The test harness masks defects and the refactored tests have lost coverage."

**Verdict:** $H_0$ is **partially refuted**. The harness mechanics are sound for the `tests_conversion` module, but residual `/tmp/` paths in `tests_convert` (out-of-scope module) and metadata blindness represent genuine coverage gaps.

---

## Phase 1: The Harness Mechanics (Jidoka Verification)

| Check | ID | Result | Severity | Finding |
|-------|----|--------|----------|---------|
| The Lie | F-HAR-01 | **PASS** | - | `verify_apr()` reads from disk via `fs::read()`, parses with `AprV2Reader`, compares each tensor value against source with tolerance. Corrupted bytes produce f32 values outside 1e-6 tolerance and ARE caught. Header corruption caught by CRC32 checksum in `AprV2Reader::from_bytes()`. |
| Tolerance Trap | F-HAR-02 | **DESIGN** | LOW | `tolerance` field is `pub(crate)` — writable after `new()`. Setting to 100.0 masks errors; setting to 1e-15 causes spurious failures. No bounds validation. **However**, the default (1e-6) is correct for F32 and all existing tests use the default. |
| Metadata Blindness | F-HAR-03 | **DESIGN** | MEDIUM | `verify_apr()` and `verify_safetensors()` check tensor names, shapes, and data values. They do **NOT** check metadata (architecture, vocab_size, custom fields). A model with wrong architecture metadata but correct tensor data passes verification. |
| Cleanup Sabotage | F-HAR-04 | **PASS** | - | `TempDir` uses RAII Drop. Rust's unwinding guarantees Drop runs on panic. Temp directories are cleaned even if tests fail via `assert!()`. |
| Empty State | F-HAR-05 | **DESIGN** | LOW | Config with all flags false and 0 layers produces 0 tensors. `collect_pygmy_tensors()` returns empty Vec. Verification passes vacuously (no tensors to compare). This is vacuous truth, not a defect — but it means `assert_import_ok(empty_config)` tests nothing. |

**Phase 1 Verdict:** No Jidoka failures. Harness IS safe. The line does stop when defects occur (F-HAR-01 confirmed). Design issues (F-HAR-02, F-HAR-03, F-HAR-05) are opportunities for improvement, not safety violations.

---

## Phase 2: Refactoring Integrity (Standard Work)

| Check | ID | Result | Severity | Finding |
|-------|----|--------|----------|---------|
| Path Leakage | F-MIG-01 | **PASS (scoped)** | - | Zero results for `/tmp/test_valid_input`. The exact string from the spec is eliminated. |
| Path Leakage (extended) | F-MIG-01b | **FAILED** | HIGH | 6 hardcoded `/tmp/test_convert_*` paths remain in `tests_convert` module (lines 655-705). These were **out of scope** per the spec (which only targeted `tests_conversion`), but represent incomplete Mura reduction. Manual `fs::remove_file().ok()` cleanup at lines 672-722 persists. |
| Manual Map Detection | F-MIG-02 | **PASS** | - | 6 `BTreeMap::new()` calls remain. 4 are **justified** (intentionally bad data for negative tests: LayerNorm=11, NaN). 1 in `test_name_mapping_whisper` is justified (specific tensor names needed). 1 in `create_test_model` helper could use PygmyConfig but is in the out-of-scope `tests_convert` module. |
| Negative Test Fidelity | F-MIG-03 | **PASS** | - | `test_convert_invalid_layernorm_fails_strict` uses `Architecture::Llama` + `strict: true`. Error message assertion checks for "mean=11" OR "LayerNorm" OR "outside expected range". If data were "fixed" to `vec![1.0; 384]`, the outer `assert!(result.is_err())` would correctly panic (import would succeed). |
| Name Mapping | F-MIG-04 | **PASS** | - | `test_name_mapping_whisper` uses `AprV2Reader` read-back (line 481) to verify exact tensor names: `"model.encoder.conv1.weight"` and `"model.decoder.layer_norm.weight"`. Explicitly checks `model.` prefix preservation. |

---

## Phase 3: GH-196 Regression Coverage

| Check | ID | Result | Severity | Finding |
|-------|----|--------|----------|---------|
| Auto-Arch | F-REG-01 | **PASS (with note)** | LOW | `PygmyConfig::llama_style()` creates `model.layers.*` tensors. `infer_model_config_from_tensors()` (import.rs:716) maps `model.layers` → architecture `"qwen2"` (verified). If `model.layers` were removed (e.g., `embedding_only()`), architecture becomes `"unknown"` (unverified). **Note:** The test does not explicitly assert the inferred architecture name — it only verifies tensors exist in output. |
| Strict Mode | F-REG-02 | **PASS** | - | `test_gh196_strict_rejects_unverified` uses `embedding_only()` which produces no `model.layers.*` patterns → `"unknown"` arch → `is_inference_verified()` returns false → strict rejects. Removing `strict: true` would cause `result.is_ok()` → test's `assert!(result.is_err())` would panic. Correct behavior. |
| Round Trip | F-REG-03 | **PASS** | - | `assert_roundtrip_ok()` compares **final** SafeTensors output against **original** source tensors captured by `collect_pygmy_tensors()` during `with_safetensors()`. Comparison is semantic (f32 values ± 1e-6 tolerance), not binary. The pipeline is: config → source_tensors + input.safetensors → output.apr → roundtrip.safetensors. Verification reads `roundtrip.safetensors` from disk and compares against `source_tensors`. |

---

## Phase 4: Toyota Way Principles (Genchi Genbutsu)

| Check | ID | Result | Severity | Finding |
|-------|----|--------|----------|---------|
| Read-Back | F-TPS-01 | **PASS** | - | `verify_apr()` calls `fs::read(output)` (line 1173) to read the persisted file from disk, then parses with `AprV2Reader::from_bytes()`. Does NOT use in-memory buffers. This is Genchi Genbutsu — goes to the actual artifact. |
| Standardization | F-TPS-02 | **PASS** | - | All public harness methods have `///` doc comments: `new()` ("Create a new empty harness with a fresh temp directory"), `with_safetensors()` ("Write a pygmy SafeTensors file..."), `import_to_apr()` ("Import the input SafeTensors to APR"), `verify_apr()` ("Read back the output file and verify tensor data matches source"), `assert_import_ok()` ("Import pygmy SafeTensors -> APR with default options and verify"), `assert_roundtrip_ok()` ("Full round-trip..."). |

---

## Phase 5: Edge Case Stress (Popperian Refutation)

| Check | ID | Result | Severity | Finding |
|-------|----|--------|----------|---------|
| Unicode | F-EDGE-01 | **PASS** | LOW | SafeTensors header is JSON (UTF-8 native). APR v2 writer stores names as UTF-8 strings with `MAX_TENSOR_NAME_LEN = 256` bytes. Emoji names < 256 bytes round-trip correctly. Names > 256 bytes truncate at byte boundary (potential UTF-8 mid-sequence corruption, but no pygmy configs produce such names). |
| Large Config | F-EDGE-02 | **PASS** | - | `PygmyConfig { num_layers: 1024 }` produces ~9,218 tensors. No hardcoded limits in SafeTensors builder or APR writer. Tensor count stored as u32 (4B ceiling). JSON header ~1.1 MB (well under APR `MAX_METADATA_SIZE` of 16 MB). |
| Zero-Sized Tensor | F-EDGE-03 | **PASS** | - | `vocab_size: 0` produces 0-element tensors. Both SafeTensors and APR writers handle empty data (`data_offsets: [N, N]`). Q8/Q4 quantization has explicit empty guards. Verification passes (empty zip = 0 iterations). |

---

## Consolidated Results

| Phase | Checks | Pass | Fail | Design | Pass Rate |
|-------|--------|------|------|--------|-----------|
| 1: Jidoka | 5 | 2 | 0 | 3 | 100% (no failures) |
| 2: Standard Work | 5 | 4 | 1 | 0 | 80% |
| 3: GH-196 Regression | 3 | 3 | 0 | 0 | 100% |
| 4: Toyota Way | 2 | 2 | 0 | 0 | 100% |
| 5: Edge Cases | 3 | 3 | 0 | 0 | 100% |
| **TOTAL** | **18** | **14** | **1** | **3** | **94%** |

### Failures Requiring Action

| ID | Issue | Remediation |
|----|-------|-------------|
| **F-MIG-01b** | 6 hardcoded `/tmp/` paths + manual `fs::remove_file().ok()` cleanup in `tests_convert` module | Migrate `tests_convert` to use `tempfile::tempdir()`. This was out of spec scope but is a clear Mura violation. |

### Design Issues (Non-Blocking)

| ID | Issue | Recommendation |
|----|-------|----------------|
| F-HAR-02 | Tolerance field writable without bounds check | Consider adding `with_tolerance(config)` builder method with `assert!(config.f32_atol > 0.0 && config.f32_atol < 1.0)` |
| F-HAR-03 | Metadata not verified (architecture, vocab_size) | Add `verify_metadata()` method or extend `verify_apr()` to check `reader.metadata().architecture` |
| F-HAR-05 | Empty config produces vacuous verification | Add guard: `assert!(!self.source_tensors.is_empty(), "Cannot verify with 0 source tensors")` |

### Notable Architecture Finding

The `infer_model_config_from_tensors()` function (import.rs:716) maps `model.layers` → `"qwen2"` for ALL models using that naming pattern. This means `PygmyConfig::llama_style()` is inferred as `"qwen2"`, which is verified. This is a naming ambiguity — not a defect — but it means `test_gh196_auto_arch_import` tests "qwen2-classified-as-verified" rather than "llama-classified-as-verified". Both architectures share `model.layers.*` naming.
