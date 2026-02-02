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

---

## Phase 6: Round-Trip Conversion Integrity (PMAT-ROSETTA-002)

**Date:** 2026-02-02
**Status:** FAILED — All 6 conversion paths produce incorrect output for real models
**Severity:** P0 — RELEASE BLOCKING

### Discovery Context

During Qwen2.5-Coder-1.5B-Instruct pipeline testing (Section 30.5 of showcase spec),
the self-converted GGUF crashed at inference:

```
error: Model architecture not supported for GPU-resident path
       (requires separate Q/K/V, SwiGLU, RMSNorm)
```

### Root Cause: PMAT-101 QKV Fusion Without Corresponding Unfusion

**Location:** `src/format/converter/write.rs:72-136`

The APR writer (PMAT-101) pre-fuses separate Q/K/V into a single `qkv_proj` tensor
for realizar inference performance. This is a **lossy, irreversible transformation**
performed during every SafeTensors/GGUF → APR conversion:

```
SafeTensors: q_proj [1536,1536] + k_proj [256,1536] + v_proj [256,1536] = 3 tensors
APR:         qkv_proj [2048,1536]                                       = 1 tensor
```

No code exists to split QKV back into separate Q/K/V during export.

### Falsification Checks

| Check | ID | Result | Severity | Finding |
|-------|----|--------|----------|---------|
| QKV Fusion Detection | F-RT-01 | **FAILED** | P0 | SafeTensors (338 tensors) → APR (227 tensors): 111 tensors lost to QKV fusion across 28 layers. No test catches this because PygmyConfig doesn't trigger fusion. |
| APR→GGUF Tensor Names | F-RT-02 | **FAILED** | P0 | Exported GGUF contains `attn_qkv.weight` instead of standard `attn_q.weight` + `attn_k.weight` + `attn_v.weight`. Both llama.cpp and realizar reject merged QKV. |
| APR→SafeTensors Names | F-RT-03 | **FAILED** | P0 | Exported SafeTensors contains `qkv_proj.weight` instead of `q_proj.weight` + `k_proj.weight` + `v_proj.weight`. HuggingFace transformers cannot load this. |
| Multi-Hop Idempotency | F-RT-04 | **FAILED** | P0 | Chain `ST→APR→GGUF→APR→ST` produces 227 tensors instead of 338. Damage is permanent — QKV fusion cannot be reversed without dimension metadata. |
| PygmyConfig Trigger | F-RT-05 | **FAILED** | HIGH | PygmyConfig (hidden=4, layers=1) does NOT trigger PMAT-101 fusion code path. All existing round-trip tests pass vacuously — they never exercise the fusion/unfusion path that real models hit. |

### Impact on Conversion Matrix

| Path | Status | Failure |
|------|--------|---------|
| SafeTensors → APR | **LOSSY** | Q/K/V fused into QKV |
| APR → SafeTensors | **BROKEN** | Outputs `qkv_proj` (non-standard) |
| APR → GGUF | **BROKEN** | Outputs `attn_qkv` (non-standard) |
| GGUF → APR | **LOSSY** | Separate Q/K/V fused into QKV |
| SafeTensors → GGUF (via APR) | **BROKEN** | Inherits ST→APR + APR→GGUF |
| GGUF → SafeTensors (via APR) | **BROKEN** | Inherits GGUF→APR + APR→ST |

**0 of 6 paths produce correct output for GQA models (Qwen2, LLaMA 3, Mistral, etc.).**

### Remediation Required

1. **Immediate (P0):** Add QKV splitting to `export_to_gguf()` and SafeTensors export
   using APR metadata (`num_heads`, `num_kv_heads`, `hidden_size`) to compute split dimensions.
2. **Test Coverage:** Add `PygmyConfig::qwen2_gqa()` that triggers PMAT-101 fusion,
   then test round-trip with tensor NAME equality (not just data fidelity).
3. **Architecture Decision:** Evaluate whether PMAT-101 fusion belongs in APR writer
   or should be deferred to realizar load-time (Option C: store separate, fuse at runtime).

### Updated Consolidated Results

| Phase | Checks | Pass | Fail | Design | Pass Rate |
|-------|--------|------|------|--------|-----------|
| 1: Jidoka | 5 | 2 | 0 | 3 | 100% |
| 2: Standard Work | 5 | 4 | 1 | 0 | 80% |
| 3: GH-196 Regression | 3 | 3 | 0 | 0 | 100% |
| 4: Toyota Way | 2 | 2 | 0 | 0 | 100% |
| 5: Edge Cases | 3 | 3 | 0 | 0 | 100% |
| **6: Round-Trip (NEW)** | **5** | **0** | **5** | **0** | **0%** |
| **TOTAL** | **23** | **14** | **6** | **3** | **61%** |

**Previous Status:** CONDITIONAL PASS (94%)
**Updated Status:** FAILED (61%) — RELEASE BLOCKED by Phase 6

---

## Phase 7: Format-Blind Loading (GH-192)

**Date:** 2026-02-02
**Related Issue:** [GH-192](https://github.com/paiml/aprender/issues/192) — APR 500x
slower than GGUF (0.5 vs 270 tok/s)
**Status:** FAILED — No pre-load inspection, cannot handle multiple model sizes

### Connection to Round-Trip Breakage

GH-192 and the QKV fusion trap (Phase 6) share the same architectural root cause:
**the pipeline does not inspect model structure before committing to a code path.**

| Symptom | Root Cause |
|---------|-----------|
| PMAT-101 fuses Q/K/V blindly | No inspection of whether export path can unfuse |
| APR inference 500x slower than GGUF | No inspection to select GPU kernels, quant type, attention structure |
| Cannot load different model sizes | No inspection to detect hidden_size, num_heads, num_kv_heads before allocation |
| Stale config.json in pacha cache (GH-198) | No per-model inspection; flat cache shares one config across all models |

### Falsification Checks

| Check | ID | Result | Severity | Finding |
|-------|----|--------|----------|---------|
| Pre-Load Inspection | F-GH192-01 | **FAILED** | P1 | `realizar::Model::load_safetensors()` reads tensors without first calling `FormatType::from_magic()` or `RosettaStone::inspect()`. It hardcodes architecture assumptions. |
| Model Size Switching | F-GH192-02 | **FAILED** | P1 | Loading a 0.5B model (MHA, 14 heads, hidden=896) then a 1.5B model (GQA, 12 heads, 2 KV heads, hidden=1536) reuses stale config. The pacha cache stores one `config.json` for all models — loading a different size requires manual cache invalidation. |
| GPU Kernel Selection | F-GH192-03 | **FAILED** | P1 | APR loader uses generic F32 path regardless of tensor quantization type. GGUF loader inspects `general.file_type` metadata to select Q4_K/Q6_K dequant kernels. This metadata-driven dispatch is missing from APR path. |

### Proposed Fix: Inspect-Before-Load

The Rosetta Stone module already has per-format inspection (`inspect()` →
`InspectionReport`). The fix is to make inspection **mandatory** before load:

```
1. detect_format(path) → FormatType
2. inspect(path)       → ModelManifest { arch, hidden, heads, kv_heads, quant, ... }
3. select_loader(manifest) → specialized loader with pre-allocated buffers
4. load(path, loader)  → model ready for inference
```

The same `ModelManifest` provides metadata for:
- QKV splitting during export (fixes Phase 6)
- GPU kernel selection (fixes GH-192 throughput)
- Correct buffer pre-allocation per model size

### Updated Final Consolidated Results

| Phase | Checks | Pass | Fail | Design | Pass Rate |
|-------|--------|------|------|--------|-----------|
| 1: Jidoka | 5 | 2 | 0 | 3 | 100% |
| 2: Standard Work | 5 | 4 | 1 | 0 | 80% |
| 3: GH-196 Regression | 3 | 3 | 0 | 0 | 100% |
| 4: Toyota Way | 2 | 2 | 0 | 0 | 100% |
| 5: Edge Cases | 3 | 3 | 0 | 0 | 100% |
| 6: Round-Trip (PMAT-ROSETTA-002) | 5 | 0 | 5 | 0 | 0% |
| **7: Format-Blind Loading (GH-192)** | **3** | **0** | **3** | **0** | **0%** |
| **TOTAL** | **26** | **14** | **9** | **3** | **54%** |

**Final Status:** FAILED (54%) — RELEASE BLOCKED by Phase 6 + Phase 7

---

## Phase 8: Full Pipeline Certification — GH-199 Evidence

**Date:** 2026-02-02
**Related Issue:** [GH-199](https://github.com/paiml/aprender/issues/199) — APR 1.5B:
dequantize `lm_head.weight` fails, inference 8x slower than GGUF, GPU output garbage
**Status:** FAILED — 3 P0/P1 bugs, MQS 283/1000 (BLOCKED)
**Model:** Qwen/Qwen2.5-Coder-1.5B-Instruct (28 layers, hidden=1536, GQA 12/2kv)

### Context

GH-199 provides the empirical certification data that confirms Phases 6 and 7 in
production. The `apr-qa certify --subprocess` tool ran 32 falsification scenarios
against all three formats (GGUF, APR, SafeTensors) on both backends (CPU, GPU).

### Falsification Checks

| Check | ID | Result | Severity | Finding |
|-------|----|--------|----------|---------|
| Q6K Dequantization | F-GH199-01 | **FAILED** | P0 | `apr convert model.apr -o model.safetensors` crashes on `lm_head.weight` (Q6K dtype=14, 191MB). APR→SafeTensors round-trip is completely broken for quantized models. The reverse conversion path does not exist for Q6K tensors. |
| APR Multi-Threading | F-GH199-02 | **FAILED** | P1 | GGUF achieves 358% CPU utilization (multi-threaded), APR stuck at 99% (single-threaded). GGUF: 16.0 tok/s, APR: 1.9 tok/s. The APR loader does not inspect model metadata to configure thread pool size. |
| APR GPU Correctness | F-GH199-03 | **FAILED** | P1 | GGUF GPU: "2 + 2 equals 4." APR GPU: "2T". Model pre-caches 5596 MB on GPU (197 quantized + 112 F32 tensors) but produces garbage. Consistent with LAYOUT-001 kernel mismatch or incorrect Q4K/Q6K dequant during GPU matmul. |
| APR GPU vs CPU Perf | F-GH199-04 | **FAILED** | P1 | APR GPU (0.5 tok/s) is **slower** than APR CPU (1.9 tok/s). GPU codepath adds overhead without benefit — the kernels selected are wrong for APR's tensor layout. |
| Conversion Round-Trip | F-GH199-05 | **FAILED** | P0 | 12 of 12 F-CONV-* scenarios failed. APR→SafeTensors crashes (dequant). APR→GGUF produces non-standard tensor names (Phase 6 QKV fusion). SafeTensors→APR→SafeTensors loses tensor identity. |

### Throughput Evidence (12 passing inference tests from GH-199)

```
Gate           Format         Backend   tok/s   Duration
──────────────────────────────────────────────────────────
F-A1-001       gguf           cpu         4.5     4948ms
F-A1-001       safetensors    cpu         0.4    14282ms
F-A2-001       gguf           gpu         4.3     5186ms
F-A2-001       safetensors    gpu         0.4    14093ms
F-A3-001       gguf           cpu         4.5     4997ms
F-A3-001       safetensors    cpu         0.4    13544ms
F-A4-001       gguf           gpu         5.2     4296ms
F-A4-001       safetensors    gpu         0.5    12392ms
F-A5-001       gguf           cpu         5.2     4316ms
F-A5-001       safetensors    cpu         0.5    12420ms
F-A6-001       gguf           gpu         5.4     4190ms
F-A6-001       safetensors    gpu         0.5    12297ms
```

Note: APR format not in this table because `apr-qa` could not create APR at test time
(conversion failure). The 1.9/0.5 tok/s figures come from manual `apr run` testing
documented in GH-199.

### Connection to Prior Phases

| GH-199 Bug | Prior Phase | Shared Root Cause |
|------------|-------------|-------------------|
| 199-A: Q6K dequant crash | Phase 6 (QKV Fusion) | Import transforms are one-way — no reverse path for either QKV unfusion or Q6K dequantization |
| 199-B: APR 8x slower | Phase 7 (GH-192) | No pre-load inspection → single-threaded generic path instead of metadata-driven thread pool |
| 199-C: GPU garbage | Phase 7 (GH-192) | No pre-load inspection → wrong GPU kernel for tensor layout and quantization type |

### Certification Record

```csv
model_id,mqs_score,grade,status,g1,g2,g3,g4,tps_gguf_cpu,tps_gguf_gpu,tps_apr_cpu,tps_apr_gpu
Qwen/Qwen2.5-Coder-1.5B-Instruct,283,F,BLOCKED,true,true,false,true,16.0,118.7,1.9,0.5
```

### Final Consolidated Results (All Phases)

| Phase | Checks | Pass | Fail | Design | Pass Rate |
|-------|--------|------|------|--------|-----------|
| 1: Jidoka | 5 | 2 | 0 | 3 | 100% |
| 2: Standard Work | 5 | 4 | 1 | 0 | 80% |
| 3: GH-196 Regression | 3 | 3 | 0 | 0 | 100% |
| 4: Toyota Way | 2 | 2 | 0 | 0 | 100% |
| 5: Edge Cases | 3 | 3 | 0 | 0 | 100% |
| 6: Round-Trip (PMAT-ROSETTA-002) | 5 | 0 | 5 | 0 | 0% |
| 7: Format-Blind Loading (GH-192) | 3 | 0 | 3 | 0 | 0% |
| **8: Full Certification (GH-199)** | **5** | **0** | **5** | **0** | **0%** |
| **TOTAL** | **31** | **14** | **14** | **3** | **45%** |

**Final Status:** FAILED (45%) — RELEASE BLOCKED by Phases 6 + 7 + 8

---

## Phase 9: Architectural Analysis — Industry Parity Assessment (PMAT-ROSETTA-003)

**Date:** 2026-02-02
**Status:** ANALYSIS COMPLETE — Architecture decision required before implementation
**Methodology:** Comparative industry survey + Toyota Way + Popperian falsification
**Academic Citations:** 30 peer-reviewed sources (see rosetta-testing.md References)

### Context

Following the Phase 6-8 failures, a deep architectural analysis was conducted
comparing the APR conversion pipeline against industry-leading systems: llama.cpp,
vLLM, TensorRT-LLM, ONNX, and MLIR. The analysis was grounded in Toyota
Production System principles (Ohno 1988, Liker 2004, Shingo 1986) and Popperian
falsification methodology (Popper 1959, Mayo 2018, Claessen & Hughes 2000).

### Key Finding: APR is the Only System That Fuses QKV Without Unfusion Metadata

| System | QKV Storage | Fusion Point | Reversible? |
|--------|-------------|-------------|-------------|
| llama.cpp | Separate Q/K/V | Never fused in storage | N/A |
| vLLM | Separate Q/K/V | Runtime (QKVParallelLinear) | Yes |
| TensorRT-LLM | Fused qkv.weight | Storage (with full metadata) | Yes |
| ONNX | Operator-defined | Runtime | Yes |
| **APR (current)** | **Fused qkv_proj** | **Import time (PMAT-101)** | **No** |

### TPS Principle Violations Identified

| Principle | Violation | Citation |
|-----------|-----------|---------|
| **Jidoka** | PMAT-101 performs lossy QKV fusion silently — no alarm, no halt | Ohno (1988), Danovaro et al. (2008) |
| **Heijunka** | Canonical form (APR) introduces irreversible transformation instead of eliminating representation anomalies | Codd (1970), Liker (2004) |
| **Genchi Genbutsu** | No pre-load inspection — pipeline commits to code path without inspecting actual artifact | Staats et al. (2011) |
| **Poka-Yoke** | No type-level prevention of lossy conversion — `convert()` and `convert_lossy()` use same API | Shingo (1986) |
| **Standard Work** | No documented conversion protocol — each path has ad-hoc validation | Ohno (1988) |
| **Muda** | Dequant→requant round-trip for Q4K when raw byte preservation available | Foidl et al. (2024) |

### Falsification Severity Assessment

| Severity Criterion (Mayo 2018) | Current Level | Required Level |
|-------------------------------|---------------|----------------|
| Input adversariness | LOW (PygmyConfig only) | HIGH (NaN, Inf, denormals, GQA configs) |
| Path coverage | LOW (no multi-hop) | HIGH (all 720 permutations) |
| Comparison granularity | LOW (statistics only) | HIGH (per-value comparison) |
| Cross-format differential | NONE | HIGH (same model through all formats) |
| Mutation score on converter code | UNKNOWN | Target 85%+ |

### Recommended Architecture: Option C (Store Separate, Fuse at Runtime)

Aligned with vLLM and llama.cpp consensus. Full details in rosetta-testing.md
Section "Part IV: Recommended Architecture."

### Assessment Checks

| Check | ID | Result | Finding |
|-------|----|--------|---------|
| Industry parity | F-ARCH-01 | **FAILED** | APR is the only system that performs irreversible QKV fusion during import without storing unfusion metadata |
| TPS compliance | F-ARCH-02 | **FAILED** | 6 of 6 TPS principles violated (Jidoka, Heijunka, Genchi Genbutsu, Poka-Yoke, Standard Work, Muda) |
| Test severity | F-ARCH-03 | **FAILED** | Current tests do not constitute "severe tests" per Mayo (2018) — PygmyConfig does not exercise critical code paths |
| Canonical form correctness | F-ARCH-04 | **FAILED** | APR canonical form violates Codd's normalization theory — introduces representation anomaly (QKV fusion) rather than eliminating them |
| Conversion protocol | F-ARCH-05 | **FAILED** | No standardized conversion protocol — each path has different validation, transformation, and verification steps |

### Final Consolidated Results (All Phases)

| Phase | Checks | Pass | Fail | Design | Pass Rate |
|-------|--------|------|------|--------|-----------|
| 1: Jidoka | 5 | 2 | 0 | 3 | 100% |
| 2: Standard Work | 5 | 4 | 1 | 0 | 80% |
| 3: GH-196 Regression | 3 | 3 | 0 | 0 | 100% |
| 4: Toyota Way | 2 | 2 | 0 | 0 | 100% |
| 5: Edge Cases | 3 | 3 | 0 | 0 | 100% |
| 6: Round-Trip (PMAT-ROSETTA-002) | 5 | 0 | 5 | 0 | 0% |
| 7: Format-Blind Loading (GH-192) | 3 | 0 | 3 | 0 | 0% |
| 8: Full Certification (GH-199) | 5 | 0 | 5 | 0 | 0% |
| **9: Architecture Analysis (NEW)** | **5** | **0** | **5** | **0** | **0%** |
| **TOTAL** | **36** | **14** | **19** | **3** | **39%** |

**Final Status:** FAILED (39%) — RELEASE BLOCKED by Phases 6 + 7 + 8 + 9
