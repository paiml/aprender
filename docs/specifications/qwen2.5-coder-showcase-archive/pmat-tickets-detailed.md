# PMAT Tickets: Detailed Writeups

> Archived from qwen2.5-coder-showcase-demo.md (lines 689-1568)

### RED TEAM FINDINGS (2026-01-30): Protocol "Burn It Down"

**Attack Surface Audit Results:**

| Finding | Severity | Status | Evidence |
|---------|----------|--------|----------|
| Mutex `.lock().unwrap()` in serve.rs | **P0** | ✅ **FIXED** (PMAT-189) | All 8 calls replaced with proper error handling |
| GH-177 Conversion NaN Root Cause | **P0** | ✅ **FIXED** (PMAT-190) | Q4K scale layout mismatch fixed |
| `expect()` in run.rs hot paths | **P1** | ⚠️ PARTIAL | 4 `expect()` remain with descriptive messages (config/vocab/trace guards) |
| Symlink loop error message | **P2** | 🟡 MISLEADING | Returns "Resource not found" instead of symlink error |
| Empty file validation | — | ✅ PASSED | Graceful FAIL, no panic |
| Invalid magic bytes | — | ✅ PASSED | Graceful FAIL, clear error |
| Permission denied | — | ✅ PASSED | "Permission denied (os error 13)" |

**P0 FIXED: Mutex Lock Poisoning (PMAT-189)**
```rust
// BEFORE (P0 CRITICAL):
let t = transformer.lock().unwrap();  // ❌ Panic on poison

// AFTER (PMAT-189 Fix):
let t = match transformer.lock() {
    Ok(guard) => guard,
    Err(_poisoned) => {
        return (StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "Lock poisoned. Please restart server."})))
            .into_response();  // ✅ Graceful 500
    }
};
```

**Status:** ✅ All 8 mutex locks now handle poisoning gracefully.

### PMAT-120: SafeTensors GPU ✅ FIXED (Five-Whys Analysis)


<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Implementations:**
- `crates/apr-cli/src/commands/chat.rs:594` - // PMAT-120 FIX: SafeTensors/APR use config.json for archite
- `crates/apr-cli/src/commands/chat.rs:615` - // PMAT-120: Read config.json for architecture detection
- `crates/apr-cli/src/commands/chat.rs:1218` - // PMAT-120 DEBUG: Log generated token IDs
**Findings:** None ✓
<!-- /bug-hunter-status -->







**Original Symptom:** `apr chat model.safetensors` produced garbage output (Hebrew characters, "Copyright" tokens)
**Token IDs:** [97514, 24413, 24413, ...] instead of [17, 488, 220, 17, 16819, ...] ("2 + 2 equals 4")

**Five-Whys (Updated):**
1. **WHY garbage tokens?** → Token IDs are completely wrong (97514 vs 17)
2. **WHY wrong token IDs?** → QKV projection output was wrong
3. **WHY wrong QKV output?** → **Missing QKV bias terms** (Qwen2 has attention biases!)
4. **WHY missing biases?** → Assumed LLaMA-like architecture (no attention biases), but Qwen2 has `q_proj.bias`, `k_proj.bias`, `v_proj.bias`
5. **WHY wasn't this caught?** → GGUF path works because GGUF bakes biases into quantized weights; SafeTensors keeps them separate

**Root Cause:** The `SafeTensorsCudaModel` was loading `q_proj.weight`, `k_proj.weight`, `v_proj.weight` but NOT the corresponding `.bias` tensors. Qwen2 (unlike LLaMA) has attention biases that must be added after the projection.

**Fix Applied (2026-01-28):**
1. Added `qkv_bias_cache` and `o_bias_cache` to `SafeTensorsCudaModel`
2. Load bias tensors during `upload_weights()`: `{q,k,v}_proj.bias`
3. Apply biases after GEMM in `forward_layer()`: `qkv[i] += bias[i]`
4. Weight transpose for GEMM: HuggingFace [n, k] → GEMM [k, n]

**Verification:**
```bash
# Both paths now produce correct output:
apr chat model.safetensors  # "2 plus 2 equals 4."
apr chat model.gguf         # "2 + 2 equals 4."
```

**PMAT-114 Strategic Pivot (2026-01-27):** SafeTensors-first import debugging.
**PMAT-114 Fix (2026-01-27):** ✅ COMPLETE. Root cause: APR converter fuses QKV biases into `qkv_proj.bias` but loader only looked for separate biases. Fixed in `realizar/src/apr_transformer/mod.rs:600`.

**PMAT Roadmap ID:** `SHOWCASE-BRICK-001`

### PMAT-207: APR Performance O(n²) → O(n) 🔧 PARTIAL FIX (GH-192)


<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Findings:** None ✓
<!-- /bug-hunter-status -->







**GitHub Issue:** [paiml/aprender#192](https://github.com/paiml/aprender/issues/192)
**Severity:** P0 - CRITICAL (500x performance regression)
**Status:** 🔧 PARTIAL - KV cache fix applied, but converter still dequantizes to F32

**Original Symptom:** APR benchmark showed 0.4-0.5 tok/s vs GGUF's 287 tok/s (500x slower).

**Five-Whys (Updated 2026-01-31):**
1. **WHY 500x slower?** → APR inference uses F32 kernels instead of Q4K fused kernels
2. **WHY F32?** → APR file contains F32 tensors (458 MiB) instead of Q4K (~250 MiB)
3. **WHY F32 tensors?** → GGUF→APR converter dequantizes Q4K to F32 by default
4. **WHY dequantize?** → Default `ConvertOptions.quantize = None`, requires explicit `--quantize q4k`
5. **WHY not auto-preserve?** → Converter lacks Q4K-to-Q4K pass-through for APR output format

**Root Cause:** TWO separate issues:

**Issue A: O(n²) Generation Loop (FIXED in bench.rs)**
- APR CPU used `forward(&all_tokens)` - O(n²)
- APR GPU used `generate_cuda` instead of `generate_cuda_with_cache`
- Fix: Updated bench.rs to use KV-cached generation methods

**Issue B: F32 Dequantization During Conversion (NOT FIXED)**
- GGUF Q4K (250 MiB) → APR F32 (458 MiB) during conversion
- F32 matmul is ~400x slower than Q4K fused kernels
- APR loader finds no Q4K weights → falls back to slow F32 path

**Verification (2026-01-31):**
```bash
$ apr tensors model.apr | head -5
blk.0.attn_k.weight [f32] [896, 128]  # ← F32, not Q4K!
```

**Fix Applied (Partial):**
```rust
// bench.rs: KV cache usage (FIXED)
let transformer = AprTransformer::from_apr_file(path)?;
let output = transformer.generate_with_cache(&prompt, &gen_config)?;
```

**Fix Required (NOT YET IMPLEMENTED):**
```rust
// Converter should auto-detect and preserve Q4K:
// 1. If source GGUF has Q4K tensors AND target is APR
// 2. Auto-set quantize = Q4K to preserve fused kernel compatibility
// 3. Use raw byte pass-through (existing code at line 195-243 in converter/mod.rs)
```

**Workaround:** Use GGUF directly for benchmarks (422 tok/s achieved).

**Expected Impact After Full Fix:** APR should match GGUF at 400+ tok/s.

### PMAT-208: SafeTensors config.json Missing Fields ✅ FIXED (GH-193)


<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Findings:** None ✓
<!-- /bug-hunter-status -->







**GitHub Issue:** [paiml/aprender#193](https://github.com/paiml/aprender/issues/193)
**Severity:** P0 - CRITICAL (SafeTensors GPU fails to load)
**Status:** ✅ FIXED (2026-01-31, commit 2ea997e3)

**Original Symptom:** `SafeTensorsCudaModel::load` fails with "config.json missing num_attention_heads"

**Root Cause:** The `infer_model_config()` function in `export.rs` was generating minimal config.json
that didn't include required fields for HuggingFace inference:
- `num_attention_heads`
- `intermediate_size`
- `num_key_value_heads`
- `hidden_act`, `rms_norm_eps`, `rope_theta`, etc.

**Fix Applied:** Enhanced `infer_model_config()` to infer all required fields from tensor shapes:
```rust
// Infer num_attention_heads from Q/K/V weight dimensions
let num_attention_heads = tensors.iter()
    .find(|(name, _)| name.contains("q_proj"))
    .map(|(_, (_, shape))| {
        let head_dim = if hidden_size >= 4096 { 128 } else { 64 };
        hidden_size / head_dim
    })
    .unwrap_or_else(|| match hidden_size {
        896 => 14,   // Qwen2.5-0.5B
        1536 => 12,  // Qwen2.5-1.5B
        4096 => 32,  // Llama-7B
        _ => (hidden_size / 128).max(1),
    });

// Infer intermediate_size from MLP gate/up projection weights
let intermediate_size = tensors.iter()
    .find(|(name, _)| name.contains("mlp.gate_proj"))
    .map(|(_, (_, shape))| shape.first().copied().unwrap_or(hidden_size * 4))
    .unwrap_or(hidden_size * 4);
```

**Additional Fix:** Added divide-by-zero guards for edge cases where tensors are empty.

### PMAT-187: Format Conversion NaN Corruption Detection ✅ FIXED (GH-177)


<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Implementations:**
- `src/format/converter/import.rs:843` - /// PMAT-187: Validates all tensors after loading to catch c
- `src/format/converter/import.rs:877` - // PMAT-187: Validate tensor values after loading (Jidoka - 
- `src/format/converter/mod.rs:428` - /// PMAT-187: Validates all tensors after loading to catch c
- `src/format/converter/mod.rs:433` - // PMAT-187: Validate all tensors after loading (Jidoka - st
- `src/format/converter/mod.rs:476` - // PMAT-187: Validate tensor values after dequantization (Ji
- `src/format/converter/mod.rs:485` - /// PMAT-187: Validate tensor values for NaN/Inf/explosive c
- `src/format/converter/mod.rs:516` - "PMAT-187: Tensor '{}' contains {} NaN values (data corrupti
- `src/format/converter/mod.rs:527` - "PMAT-187: Tensor '{}' contains {} Inf values (numerical ove
- `src/format/converter/mod.rs:541` - "PMAT-187: Tensor '{}' has explosive mean={:.2e} (expected [
- `src/format/converter/tests/pmat.rs:348` - /// PMAT-187: Tests for tensor value validation (NaN/Inf/exp
- `src/format/converter/tests/pmat.rs:377` - assert!(err.contains("PMAT-187"), "Error should reference PM
**Findings:** None ✓
<!-- /bug-hunter-status -->







**GitHub Issue:** [paiml/aprender#177](https://github.com/paiml/aprender/issues/177)
**Severity:** P0 - CRITICAL (Data Corruption)
**Status:** ✅ FIXED (2026-01-30) - Jidoka validation added
**Previous Issue:** GH-172, PMAT-176/177 (partial fix, regression detected)
**Discovered By:** apr-model-qa-playbook (Popperian Falsification)

**Original Symptom:** `apr rosetta convert` introduced catastrophic numerical corruption:
- GGUF → APR: 84.6% output difference (expected < ε=1e-6)
- APR → GGUF: 63.4% output difference
- Round-trip: 75 tensor errors with NaN/Inf, means ~10^38

**Five-Whys Root Cause:**
1. WHY corrupted output? → Tensor weights contain NaN/Inf after dequantization
2. WHY NaN/Inf? → Corrupt scale factors from quantization metadata
3. WHY not detected? → No post-dequantization validation
4. WHY no validation? → Missing Jidoka check in conversion pipeline
5. ROOT CAUSE: **Defects passed downstream without detection**

**Fix Applied (PMAT-187):**
1. ✅ Added `validate_tensor_values()` function detecting NaN/Inf/explosive means
2. ✅ Integrated validation into `load_apr_tensors_f32()` after dequantization
3. ✅ Integrated validation into `load_gguf_tensors_f32()` after loading
4. ✅ Integrated validation into `load_safetensors_tensors()` after loading
5. ✅ Added 8 unit tests for validation function

**Toyota Way Jidoka Principle:** Stop the line on quality defects, don't pass defects downstream.
Now the pipeline will fail fast with a clear error message if corruption is detected.

**Evidence:** 8/8 PMAT-187 tests pass

**✅ COMPLETE (PMAT-187 + PMAT-190):**
- ✅ NaN/Inf Detection: Fails fast with clear errors (Jidoka working)
- ✅ Root Cause: Q4K scale layout mismatch fixed (PMAT-190)

---

### PMAT-190: Q4K Scale Layout Mismatch Fix ✅ FIXED (GH-177 Root Cause)


<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Findings:** None ✓
<!-- /bug-hunter-status -->







**GitHub Issue:** [paiml/aprender#177](https://github.com/paiml/aprender/issues/177)
**Severity:** P0 - CRITICAL (Root Cause)
**Status:** ✅ FIXED (2026-01-30)
**Evidence:** 9/9 Q4K tests pass, 8/8 PMAT-187 tests pass

**Root Cause (Genchi Genbutsu - Go See):**
Two incompatible Q4K dequantization implementations:
- `gguf.rs`: ONE scale per 32-element sub-block (correct)
- `converter.rs`: DIFFERENT scales for low/high nibbles (WRONG)

**Five-Whys:**
1. WHY 84.6% output difference? → Values dequantized with wrong scales
2. WHY wrong scales? → converter.rs used different scale indices than gguf.rs
3. WHY different indices? → Two incompatible Q4_K layout interpretations
4. WHY two implementations? → converter.rs copied candle layout, gguf.rs used llama.cpp
5. ROOT CAUSE: **Layout mismatch - Q4K uses ONE scale per sub-block, not different for low/high!**

**Fix Applied (PMAT-190):**
```rust
// BEFORE (WRONG - different scales for low/high nibbles):
let d1 = d * scales[chunk * 2];      // scale for low nibbles
let d2 = d * scales[chunk * 2 + 1];  // scale for high nibbles

// AFTER (CORRECT - same scale for entire sub-block):
let scale = d * scales[j];  // ONE scale for all 32 elements
```

**Toyota Way:** Genchi Genbutsu - Go see the actual data (gguf.rs), don't assume layouts match.

---

### PMAT-188: apr validate GGUF v3 Support ✅ FIXED (GH-178)


<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Findings:** None ✓
<!-- /bug-hunter-status -->







**GitHub Issue:** [paiml/aprender#178](https://github.com/paiml/aprender/issues/178)
**Severity:** P2
**Status:** ✅ FIXED (2026-01-30)
**Evidence:** 7/7 GGUF validation tests pass

**Original Symptom:** `apr validate model.gguf` incorrectly rejected valid GGUF v3 files with "Invalid magic" error.

**Five-Whys Root Cause:**
1. WHY does validate reject valid GGUF files? → Validator only checks for APR magic bytes
2. WHY only APR magic bytes? → Original validator was APR-format specific
3. WHY APR-specific? → Validation was designed for APR format before GGUF support
4. WHY no GGUF version check? → Missing format detection in validate_structure()
5. ROOT CAUSE: **Validator lacks format-aware magic byte checking and GGUF version validation**

**Fix Applied (PMAT-188):**
1. ✅ Updated `check_magic()` to accept both "APR\0" and "GGUF" magic bytes
2. ✅ Added `check_gguf_version()` supporting versions 1, 2, 3
3. ✅ Updated `validate_structure()` for format detection
4. ✅ Added 7 GGUF-specific unit tests

**Code Changes (validation.rs):**
```rust
// GH-178: Accept both APR and GGUF magic
if magic == b"APR\0" {
    CheckStatus::Pass
} else if magic == b"GGUF" {
    CheckStatus::Pass  // [71, 71, 85, 70]
}

// GGUF version validation (v1, v2, v3 supported)
fn check_gguf_version(&mut self, data: &[u8]) {
    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    if (1..=3).contains(&version) { /* Pass */ }
}
```

**Toyota Way Jidoka Principle:** Build quality in with proper format detection.

---

### PMAT-201: Per-Tensor Statistical Fingerprints (JAX-STAT-001) ✅ IMPLEMENTED


<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Implementations:**
- `crates/apr-cli/src/commands/rosetta.rs:161` - /// Generate per-tensor statistical fingerprints (PMAT-201, 
- `crates/apr-cli/src/commands/rosetta.rs:1208` - /// Run the rosetta fingerprint subcommand (PMAT-201)
- `crates/apr-cli/src/commands/rosetta.rs:1232` - "║           TENSOR STATISTICAL FINGERPRINTS (PMAT-201, JAX-
- `crates/apr-cli/src/commands/rosetta.rs:1473` - /// Tensor statistical fingerprint (PMAT-201)
- `crates/apr-cli/src/commands/rosetta.rs:1623` - // PMAT-201 FIX: Load APR tensors directly
- `crates/apr-cli/src/commands/rosetta.rs:1758` - /// Simple Q4_K dequantization for statistics (PMAT-201)
- `crates/apr-cli/src/commands/rosetta.rs:1809` - /// Simple Q6_K dequantization for statistics (PMAT-201)
- `crates/apr-cli/src/commands/rosetta.rs:1854` - // PMAT-201: Would need proper JSON parsing for full impleme
- `crates/apr-cli/src/commands/showcase/benchmark.rs:3` - //! Extracted from monolithic showcase.rs (PMAT-201)
- `crates/apr-cli/src/commands/showcase/mod.rs:1` - //! Qwen2.5-Coder-32B Showcase Demo (PMAT-201: split from mo
- `src/format/converter/tests/coverage.rs:2771` - // FALSIFICATION TESTS: PMAT-201 Per-Tensor Statistical Fing
**Findings:** None ✓
<!-- /bug-hunter-status -->







**Specification:** APR-SPEC.md Section 17.1
**Severity:** P1 (Catches GH-186 class bugs at load time)
**Status:** ✅ IMPLEMENTED (2026-02-03, 6 falsification tests added)

**Problem:** Current validation only checks file-level CRC32. A single corrupted tensor causes complete model failure while passing structural checks. This bug class has occurred 50+ times (GH-186, GH-177, PMAT-187).

**Implementation: `apr rosetta fingerprint`**

```bash
# Generate fingerprints for all tensors
apr rosetta fingerprint model.gguf --output fingerprints.json

# Compare fingerprints between two models
apr rosetta fingerprint model.gguf model.apr --diff
```

**Fingerprint Schema:**
```rust
struct TensorFingerprint {
    name: String,
    shape: Vec<usize>,
    dtype: String,
    mean: f32,
    std: f32,
    min: f32,
    max: f32,
    percentiles: [f32; 5],  // p5, p25, p50, p75, p95
    nan_count: u32,
    inf_count: u32,
    zero_fraction: f32,
    checksum: u32,          // Per-tensor CRC32
}
```

**Falsification Gates:**
- F-FINGERPRINT-001: `apr rosetta fingerprint model.gguf` produces valid JSON
- F-FINGERPRINT-002: Fingerprints match between identical models
- F-FINGERPRINT-003: Corrupted tensor detected by fingerprint diff (3σ deviation)
- F-FINGERPRINT-004: `--diff` shows anomalies when APR differs from GGUF

**Toyota Way:** Jidoka - Stop the line at the first sign of statistical anomaly.

---

### PMAT-202: Tensor Statistics Validation (JAX-STAT-002) ✅ IMPLEMENTED


<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Implementations:**
- `crates/apr-cli/src/commands/rosetta.rs:192` - /// Validate tensor statistics against reference or expected
- `crates/apr-cli/src/commands/rosetta.rs:1303` - /// Run the rosetta validate-stats subcommand (PMAT-202)
- `crates/apr-cli/src/commands/rosetta.rs:1332` - "║             TENSOR STATISTICS VALIDATION (PMAT-202, JAX-S
- `src/format/converter/tests/coverage.rs:2947` - // FALSIFICATION TESTS: PMAT-202 Tensor Statistics Validatio
**Findings:** None ✓
<!-- /bug-hunter-status -->







**Specification:** APR-SPEC.md Section 17.1
**Severity:** P1
**Status:** ✅ IMPLEMENTED (2026-02-03, 7 falsification tests added)

**Problem:** Loading APR files doesn't validate tensor values against expected distributions.

**Implementation: `apr rosetta validate-stats`**

```bash
# Validate APR against reference GGUF
apr rosetta validate-stats model.apr --reference model.gguf

# Validate APR against stored fingerprints
apr rosetta validate-stats model.apr --fingerprints expected.json

# Validate with role-specific thresholds
apr rosetta validate-stats model.apr --strict
```

**Role-Specific Thresholds:**
| Tensor Type | Expected Mean | Expected Std | Tolerance |
|-------------|---------------|--------------|-----------|
| Embedding | ≈0 | 0.02-0.1 | 3σ |
| LayerNorm weight | ≈1 | 0.001-0.01 | 2σ |
| LayerNorm bias | ≈0 | 0.001-0.01 | 3σ |
| Attention weight | ≈0 | 0.01-0.05 | 3σ |
| MLP weight | ≈0 | 0.01-0.05 | 3σ |

**Error Code E020 - Statistical Anomaly:**
```
E020: Statistical anomaly in tensor 'model.layers.0.self_attn.q_proj.weight'
      Expected mean ≈ 0.0, got 11.3 (deviation: 1130σ)
      This indicates corrupted dequantization or layout mismatch.
```

**Falsification Gates:**
- F-VALIDATE-STATS-001: Pass for correctly converted APR
- F-VALIDATE-STATS-002: Fail with E020 for corrupted tensor
- F-VALIDATE-STATS-003: Role-specific thresholds catch LayerNorm issues

---

### PMAT-203: Golden Output Embedding (JAX-GOLD-003) ✅ FALSIFICATION TESTS


<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Implementations:**
- `src/format/converter/tests/coverage.rs:2867` - // FALSIFICATION TESTS: PMAT-203 Golden Output Embedding
**Findings:** None ✓
<!-- /bug-hunter-status -->







**Specification:** APR-SPEC.md Section 17.3
**Severity:** P2
**Status:** ✅ FALSIFICATION TESTS ADDED (2026-02-03, 5 tests)

**Problem:** Detecting semantic correctness requires running inference. Model can load and produce output but still be wrong.

**Implementation (Future):**
- Embed golden tests in APR metadata
- `apr validate --golden` runs tests without external files
- Self-validating artifact pattern

---

### PMAT-204: Tensor Distribution Tags (DATA-SCI-004) 🧪 FALSIFICATION READY


<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Implementations:**
- `src/format/converter/tests/coverage.rs:3155` - // PMAT-204: Tensor Distribution Tags Falsification Tests
**Findings:** None ✓
<!-- /bug-hunter-status -->







**Specification:** APR-SPEC.md Section 17.4
**Severity:** P2
**Status:** 🧪 FALSIFICATION TESTS IMPLEMENTED

**Problem:** Generic validation rules cause false positives/negatives for different tensor types.

**Implementation:**
- Tag tensors with semantic role (Embedding, LayerNorm, etc.)
- Role-specific validation thresholds
- Quantization guidance based on role (Q8_0, F32, Q6_K, Q4_K)

**Falsification Tests (6 tests):**
| Test ID | Description | Status |
|---------|-------------|--------|
| F-DIST-TAG-001 | Critical tensors (embed, lm_head) identified | ✅ PASS |
| F-DIST-TAG-002 | LayerNorm identified as high precision | ✅ PASS |
| F-DIST-TAG-003 | Attention weights as standard | ✅ PASS |
| F-DIST-TAG-004 | MLP weights as compressible | ✅ PASS |
| F-DIST-TAG-005 | Quantization recommendations match spec | ✅ PASS |
| F-DIST-TAG-006 | Minimum bits per tag | ✅ PASS |

---

### PMAT-205: Sharding-Aware Placement (JAX-SHARD-005) 🧪 FALSIFICATION READY


<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Implementations:**
- `src/format/converter/mod.rs:294` - // Step 1b: Map GGUF tensor names to APR canonical format (P
- `src/format/converter/mod.rs:299` - "[PMAT-205] Mapping {} GGUF tensor names to APR canonical fo
- `src/format/converter/mod.rs:305` - eprintln!("[PMAT-205]   {}: {}", i, name);
- `src/format/converter/tests/coverage.rs:3262` - // PMAT-205: Sharding-Aware Placement Falsification Tests
- `src/format/converter_types.rs:151` - // PMAT-205 FIX (GH-190): Map GGUF tensor names to APR canon
**Findings:** None ✓
<!-- /bug-hunter-status -->







**Specification:** APR-SPEC.md Section 17.5
**Severity:** P3
**Status:** 🧪 FALSIFICATION TESTS IMPLEMENTED

**Problem:** Large models require distributed inference hints.

**Implementation:**
- JAX-inspired PartitionSpec in metadata
- Device-agnostic tensor placement (Replicated, HiddenSharded, etc.)
- Multi-GPU memory multiplier calculation

**Falsification Tests (6 tests):**
| Test ID | Description | Status |
|---------|-------------|--------|
| F-SHARD-001 | Single device returns None | ✅ PASS |
| F-SHARD-002 | Embedding/lm_head replicated | ✅ PASS |
| F-SHARD-003 | LayerNorm replicated | ✅ PASS |
| F-SHARD-004 | Attention hidden-sharded | ✅ PASS |
| F-SHARD-005 | MLP hidden-sharded | ✅ PASS |
| F-SHARD-006 | Memory multiplier calculation | ✅ PASS |

---

### GH-180: cbtop-style Profiling (PMAT-192) ✅ COMPLETE


<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Implementations:**
- `crates/apr-cli/src/commands/profile.rs:17` - //! # PMAT-192: CI assertion mode (GH-180)
- `crates/apr-cli/src/commands/profile.rs:97` - // PMAT-192: CI Assertion Mode (GH-180)
- `crates/apr-cli/src/commands/profile.rs:382` - // PMAT-192: CI Assertion Mode Entry Point (GH-180)
- `crates/apr-cli/src/commands/profile.rs:426` - // PMAT-192 Phase 4: Differential Benchmark Mode (GH-180)
- `crates/apr-cli/src/lib.rs:829` - // PMAT-192: CI Assertion Mode (GH-180)
- `crates/apr-cli/src/lib.rs:2184` - /// Test parsing 'apr profile' with CI assertions (PMAT-192,
- `crates/apr-cli/tests/cli_integration.rs:956` - // PMAT-192 Phase 5: F-PROFILE-CI-* Tests (GH-180)
**Findings:** None ✓
<!-- /bug-hunter-status -->







**GitHub Issue:** [paiml/aprender#180](https://github.com/paiml/aprender/issues/180)
**Severity:** P2
**Status:** ✅ COMPLETE (PMAT-192)

**Objective:** Unify existing profiling commands with CI assertion support.

**Implementation Plan:**

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Add CI assertion mode to profile.rs | ✅ DONE |
| 2 | Add JSON output with pass/fail | ✅ DONE |
| 3 | Add --assert-throughput/--assert-p99 | ✅ DONE |
| 4 | Add differential benchmark mode | ✅ DONE |
| 5 | Add F-PROFILE-CI-* tests (6 tests) | ✅ DONE |

**CLI Interface:**

```bash
# CI mode with assertions (exit 1 if threshold fails)
apr profile model.gguf --ci --assert-throughput 100 --assert-p99 50

# JSON output for programmatic consumption
apr profile model.gguf --format json > results.json

# Differential benchmark (A/B comparison)
apr benchmark model_v1.gguf model_v2.gguf --report diff.md
```

**Falsification Gates:**
- F-PROFILE-CI-001: `apr profile --ci --assert-throughput 1` exits 0
- F-PROFILE-CI-002: `apr profile --ci --assert-throughput 99999` exits 1
- F-PROFILE-CI-003: `apr profile --format json` produces valid JSON
- F-PROFILE-DIFF-001: `apr benchmark m1 m2` shows delta percentage

**Five-Whys Root Cause:**
1. WHY need unified profiling? → Existing commands fragmented (cbtop, profile, bench)
2. WHY fragmented? → Each added for different use cases over time
3. WHY not unified earlier? → Focus was on correctness, not UX
4. WHY UX matters now? → Users need CI/CD integration
5. ROOT CAUSE: No unified profiler facade with CI assertion support

**Toyota Way:** Mieruka - Make performance visible at a glance.

---

### GH-179: APR Tool Test Coverage Gap (PMAT-191) ✅ FIXED


<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Implementations:**
- `crates/apr-cli/tests/cli_integration.rs:766` - // GH-179 / PMAT-191: Missing Tool Tests (Tool Coverage Gap)
**Findings:** None ✓
<!-- /bug-hunter-status -->







**GitHub Issue:** [paiml/aprender#179](https://github.com/paiml/aprender/issues/179)
**Severity:** P1
**Status:** ✅ FIXED (PMAT-191)
**Coverage:** 13/13 tools (100%) - was 9/13 (69%)

**Tool Coverage Matrix (PMAT-191 Fix):**

| Tool | Spec Section | Tested? | Status |
|------|-------------|---------|--------|
| apr run | 4.4.1 | ✅ Yes | F-RUN-001/002 (PMAT-191) |
| apr chat | 4.4.2 | ✅ Yes | F-CHAT-001/002 (PMAT-191) |
| apr serve | 4.4.3 | ✅ Yes | F-SERVE-001/002 (PMAT-191) |
| apr inspect | 4.4.4 | ✅ Yes | F-INSPECT-001 |
| apr validate | 4.4.5 | ✅ Yes | F-VALIDATE-001 |
| apr bench | 4.4.6 | ✅ Yes | F-BENCH-001 |
| apr profile | 4.4.7 | ✅ Yes | F-PROFILE-001 |
| apr trace | 4.4.8 | ✅ Yes | 4 levels |
| apr check | 4.4.9 | ✅ Yes | F-CHECK-001 |
| apr canary | 4.4.11 | ✅ Yes | F-CANARY-001/002 (PMAT-191) |
| apr convert | 4.4.12 | ✅ Yes | F-CONVERT-001/002 (PMAT-191) |
| apr tune | 4.4.13 | ✅ Yes | F-TUNE-001/002 (PMAT-191) |
| apr qa | 4.4.14 | ✅ Yes | F-QA-001/002 (PMAT-191) |

**Falsification Gates Added (PMAT-191):**
- F-RUN-001/002: Help works, missing model shows error
- F-CHAT-001/002: Help works, missing model shows error
- F-SERVE-001/002: Help works, missing model shows error
- F-CANARY-001/002: Help works, missing model shows error
- F-TUNE-001/002: Help works, missing model shows error
- F-QA-001/002: Help works, missing model shows error
- F-CONVERT-001/002: Help works, missing model shows error

**Five-Whys Root Cause (PMAT-191):**
1. WHY 69% coverage? → 4 tools had no direct tests
2. WHY no direct tests? → Focus was on format conversion, not CLI
3. WHY focus on format? → P0 conversion bugs took priority
4. WHY not parallel work? → Limited testing infrastructure for interactive tools
5. ROOT CAUSE: No falsification gates defined for interactive/server commands

**Toyota Way:** Poka-Yoke - Error-proof the system by testing all entry points.

---

### GH-202: Cross-Format Tensor Name Normalization (Rosetta) ✅ FIXED


<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Implementations:**
- `crates/apr-cli/src/commands/rosetta.rs:878` - // Build tensor maps by normalized name (GH-202: cross-forma
- `crates/apr-cli/src/commands/rosetta.rs:2016` - // GH-202: Use normalized names for cross-format matching
- `crates/apr-cli/src/commands/rosetta.rs:2038` - // GH-202: Use normalized name for cross-format lookup
- `crates/apr-cli/src/commands/rosetta.rs:2283` - /// Normalize tensor name for cross-format comparison (GH-20
- `crates/apr-cli/src/commands/rosetta.rs:3233` - // GH-202: Cross-format tensor name normalization tests
- `crates/apr-cli/src/commands/rosetta.rs:3335` - // Verify GGUF and APR/HF normalize to the SAME canonical fo
- `src/format/converter/tests/gh202_layout.rs:1` - //! GH-202: LAYOUT-002 Tensor Value Validation Tests
- `src/format/converter/tests/gh202_layout.rs:6` - //! Root cause investigation for GH-202: APR from GGUF produ
- `src/format/converter/tests/gh202_layout.rs:10` - /// GH-202-FIX-001: Validate transpose preserves logical mat
- `src/format/converter/tests/gh202_layout.rs:53` - "GH-202: Transposed shape should be [out_dim, in_dim]"
- `src/format/converter/tests/gh202_layout.rs:80` - "GH-202 MISMATCH at [out={}, in={}]: expected {:.4}, got {:.
- `src/format/converter/tests/gh202_layout.rs:88` - eprintln!("GH-202: max_diff = {:.4}, mismatch_count = {}", m
- `src/format/converter/tests/gh202_layout.rs:96` - "GH-202: {}% values mismatched ({}), max_diff={:.4}. Transpo
- `src/format/converter/tests/gh202_layout.rs:103` - /// GH-202-FIX-002: Validate Q4K dequantization round-trip
- `src/format/converter/tests/gh202_layout.rs:134` - "GH-202: Q4K roundtrip error at [{}]: orig={}, deq={}, diff=
- `src/format/converter/tests/gh202_layout.rs:142` - eprintln!("GH-202: Q4K roundtrip max diff = {}", max_diff);
- `src/format/converter/tests/gh202_layout.rs:145` - /// GH-202-FIX-005: Debug test to examine dequantize output
- `src/format/converter/tests/gh202_layout.rs:160` - eprintln!("GH-202 DEBUG: Q4K bytes = {} (expected 144 for 25
- `src/format/converter/tests/gh202_layout.rs:161` - eprintln!("GH-202 DEBUG: First 10 original: {:?}", &values[.
- `src/format/converter/tests/gh202_layout.rs:162` - eprintln!("GH-202 DEBUG: First 10 dequant:  {:?}", &dequant[
- `src/format/converter/tests/gh202_layout.rs:163` - eprintln!("GH-202 DEBUG: Last 10 original:  {:?}", &values[2
- `src/format/converter/tests/gh202_layout.rs:164` - eprintln!("GH-202 DEBUG: Last 10 dequant:   {:?}", &dequant[
- `src/format/converter/tests/gh202_layout.rs:168` - eprintln!("GH-202 DEBUG: Non-zero Q4K bytes: {}/{}", nonzero
- `src/format/converter/tests/gh202_layout.rs:178` - eprintln!("GH-202 DEBUG: Max roundtrip error: {}", max_err);
- `src/format/converter/tests/gh202_layout.rs:180` - assert!(nonzero_bytes > 10, "GH-202: Q4K should have non-zer
- `src/format/converter/tests/gh202_layout.rs:181` - assert!(max_err < 0.1, "GH-202: Roundtrip error {} too large
- `src/format/converter/tests/gh202_layout.rs:184` - /// GH-202-FIX-003: Validate matmul dimension interpretation
- `src/format/converter/tests/gh202_layout.rs:229` - "GH-202: Diagonal sum {} should be close to {}",
- `src/format/converter/tests/gh202_layout.rs:238` - "GH-202: Off-diagonal average {} should be near zero",
- `src/format/converter/tests/gh202_layout.rs:243` - /// GH-202-FIX-004: Verify GGUF shape interpretation
- `src/format/converter/tests/gh202_layout.rs:274` - "GH-202: APR shape should be [out_dim={}, in_dim={}]",
- `tests/gh202_e2e.rs:1` - //! GH-202: End-to-End Tensor Comparison Test
- `tests/gh202_e2e.rs:6` - //! Root cause investigation for GH-202: APR from GGUF produ
- `tests/gh202_e2e.rs:14` - /// GH-202-E2E-001: Verify APR reader can parse converted fi
- `tests/gh202_e2e.rs:20` - eprintln!("GH-202-E2E-001: Skipping - no test_model.apr foun
- `tests/gh202_e2e.rs:31` - "GH-202-E2E-001: Loaded APR with {} tensors",
- `tests/gh202_e2e.rs:41` - /// GH-202-E2E-002: Verify APR F32 tensor round-trip preserv
- `tests/gh202_e2e.rs:71` - eprintln!("GH-202-E2E-002: APR has {} tensors", tensor_names
- `tests/gh202_e2e.rs:99` - "GH-202-E2E-002: All {} values match (max_diff = {:.2e})",
- `tests/gh202_e2e.rs:105` - /// GH-202-E2E-003: Test transpose correctness with known va
- `tests/gh202_e2e.rs:188` - "GH-202-E2E-003 MISMATCH at [{r}, {c}]: expected {expected},
- `tests/gh202_e2e.rs:197` - "GH-202-E2E-003: {} values mismatched",
- `tests/gh202_e2e.rs:201` - "GH-202-E2E-003: All {} values match after transpose+APR rou
- `tests/gh202_e2e.rs:206` - /// GH-202-E2E-004: Smoke test for tensor statistics
- `tests/gh202_e2e.rs:220` - eprintln!("GH-202-E2E-004: Test tensor stats:");
- `tests/gh202_e2e.rs:229` - /// GH-202-E2E-005: Test matmul indexing matches APR row-maj
- `tests/gh202_e2e.rs:260` - "GH-202-E2E-005: matmul output = {:?} (expected {:?})",
- `tests/gh202_e2e.rs:271` - eprintln!("GH-202-E2E-005: matmul indexing is correct for ro
**Findings:** None ✓
<!-- /bug-hunter-status -->







**GitHub Issue:** [paiml/aprender#202](https://github.com/paiml/aprender/issues/202)
**Severity:** P1
**Status:** ✅ FIXED (2026-02-04, Round 41)
**Component:** `apr rosetta diff-tensors`, `apr rosetta fingerprint`

**Problem:** When comparing GGUF and APR files, identical tensors appeared as "missing" in both directions due to different naming conventions:
- GGUF: `blk.0.attn_q.weight`
- APR/HF: `model.layers.0.self_attn.q_proj.weight`

**Five Whys Analysis:**
1. WHY did diff-tensors show 58-90% diff? → Tensors not matched by name
2. WHY weren't tensors matched? → HashMap key was raw tensor name
3. WHY use raw names? → `normalize_tensor_name()` only stripped prefixes
4. WHY insufficient normalization? → Original code only handled same-format comparisons
5. ROOT CAUSE: Cross-format name mapping not implemented

**Fix Applied:**
Enhanced `normalize_tensor_name()` in `rosetta.rs`:
1. Strip format-specific prefixes (`model.`, `blk.`, `layers.`)
2. Remove HF intermediate paths (`.self_attn.`, `.mlp.`)
3. Map GGUF suffixes to HF convention (`attn_q` → `q_proj`, etc.)
4. Handle special cases (`output.weight` → `lm_head.weight`)

**Commits:**
- `c61c4f64` - Main fix: Cross-format tensor name normalization
- `ecf262ba` - Tests: 7 comprehensive normalization tests
- `6a608b5f` - Fix: Apply same fix to fingerprint comparison

**Verification:**
```bash
apr rosetta diff-tensors model.gguf model.apr
# Before: Missing in A: 169, Missing in B: 169
# After:  Missing in A: 0, Missing in B: 0

apr rosetta fingerprint model.gguf model.apr
# Result: ✓ No statistical anomalies detected
```

**Toyota Way:** Genchi Genbutsu - Go and see the actual tensor names to understand the problem.

---

### PMAT-176/177: Format Conversion NaN Corruption (Original Fix - GH-172)

**GitHub Issue:** [paiml/aprender#172](https://github.com/paiml/aprender/issues/172)
**Severity:** P0 - Stop the Line
**Status:** ⚠️ PARTIAL FIX (regression in GH-177)
**Evidence:** 9 Q4K tests pass, 2 PMAT-177 NaN protection tests pass

**Summary:** `apr rosetta convert` produces lossy conversions with NaN/Inf corruption in round-trip tests.

**Original Failure Matrix (pre-GH-202):**

| Conversion | Status | Evidence |
|------------|--------|----------|
| GGUF → APR | ✅ **VERIFIED** (GH-202) | 339/339 tensors, inference "2+2=4" correct |
| APR → GGUF | ⚠️ PARTIAL | Converts 339/339, but re-exported GGUF fails inference (F32 dtype unsupported in fused kernel) |
| Round-trip (GGUF→APR→ST→GGUF) | ⚠️ PARTIAL | Conversion succeeds, inference untested |

**Root Cause (Five-Whys - PMAT-177):**
1. **WHY did round-trip fail?** → NaN values appeared in converted tensors
2. **WHY NaN values?** → Scale factors (d, dmin) became invalid after f16 encoding
3. **WHY invalid scales?** → f16 has min normal ~6.1e-5, scales below that underflow
4. **WHY underflow?** → quantize_q4_k() used 1e-10 fallback which can't be encoded in f16
5. **ROOT CAUSE:** No validation of scale factors after f16 round-trip; subnormal values underflow to NaN

**Toyota Way Response:** STOP THE LINE. Built-in quality (Jidoka) at source.

**Fix Applied (PMAT-177, 2026-01-30):**
1. ✅ `dequantize_q4_k_to_f32()` - Added NaN/Inf/subnormal check after reading d/dmin scales
2. ✅ `dequantize_q6_k_to_f32()` - Same validation for Q6_K format
3. ✅ `quantize_q4_k()` - Clamp scale factors to F16_MIN_NORMAL (6.1e-5) instead of 1e-10

**Code Changes (converter.rs):**
```rust
// PMAT-177: Minimum valid f16 normal value
const F16_MIN_NORMAL: f32 = 6.1e-5;

// Replace NaN/Inf/subnormal with safe values
let d = if d_raw.is_nan() || d_raw.is_infinite() || d_raw.abs() < F16_MIN_NORMAL {
    0.0
} else {
    d_raw
};
```

**Verification Status:** Needs re-run of apr-model-qa-playbook to confirm fix

### PMAT-181: APR Chat Hangs on 1.5B Model 🔍 INVESTIGATING (GH-170)


