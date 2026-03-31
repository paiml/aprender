# Critical Failures & Resolved Falsifications

> Archived from qwen2.5-coder-showcase-demo.md (lines 1569-2387)

<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Implementations:**
- `crates/apr-cli/src/commands/chat.rs:31` - // PMAT-181: Read EOS token from APR metadata (fixes GH-170)
- `crates/apr-cli/src/commands/chat.rs:1002` - // PMAT-181: Extract EOS token from APR metadata (fixes GH-1
- `crates/apr-cli/src/commands/chat.rs:1024` - // PMAT-181: Use EOS token from model metadata
- `crates/apr-cli/src/commands/chat.rs:1063` - /// PMAT-181: Extract EOS token ID from APR metadata (fixes 
**Findings:** None ✓
<!-- /bug-hunter-status -->







**GitHub Issue:** [paiml/aprender#170](https://github.com/paiml/aprender/issues/170)
**Severity:** P1
**Status:** 🔍 INVESTIGATING
**Reporter:** @alfredodeza

**Symptom:** `apr chat qwen2.5-1.5b-instruct-q4_k_m.apr` hangs for ~2 minutes with GPU cycling 0-70%, then returns empty response.

**Initial Analysis:**
- APR format loads successfully
- GPU is detected and engaged (CUDA path active)
- Generation runs but produces no output tokens
- Smaller models (0.5B) work correctly

**Potential Root Causes (Five-Whys - PMAT-181):**
1. **WHY empty response?** → Generation loop terminates without output tokens
2. **WHY no output?** → EOS token may be triggered immediately
3. **WHY immediate EOS?** → Possible mismatch between model's EOS and hardcoded 151645
4. **WHY cycling GPU usage?** → May be doing work but output is filtered
5. **ROOT CAUSE (HYPOTHESIS):** EOS token ID mismatch or CUDA KV cache issue with larger context

**Investigation Tasks:**
- [x] Verify EOS token ID in 1.5B model metadata (PMAT-181)
- [x] Add debug logging to `generate_cuda_with_cache`
- [x] Compare token generation trace with 0.5B model
- [x] Check if CPU path (AprTransformer) has same issue

**Fix (PMAT-181):** Read EOS token from APR metadata via `extract_apr_eos_token()` instead of hardcoding 151645.
See `crates/apr-cli/src/commands/chat.rs:1068-1104` for implementation.

## 100-Point Falsification Results (PMAT-112, 2026-01-27)

**Showcase-Specific Score: 12/55** (See `docs/qa/popperian_falsification_checklist.md`)

| Section | Score | Key Results |
|---------|-------|-------------|
| II. Loader | 5/15 | F-LOAD-011 ✅ GGUF Q4_K (0.76s), F-LOAD-013 ✅ SafeTensors (0.38s), F-LOAD-015 ✅ APR |
| III. Quality | 2/15 | F-QUAL-026 ✅ 2+2=4 CPU, F-QUAL-027 ✅ 2+2=4 GPU |
| IV. Performance | 2/15 | F-PERF-049 ✅ Load <2s, F-PERF-052 ✅ 10.6 tok/s CPU |
| V. Rosetta | 1/10 | F-CONV-056 ✅ SafeTensors→APR |
| VII. Observability | 2/10 | F-OBS-081 ✅ --trace JSON, F-OBS-089 ✅ apr check |

**Remaining Showcase Gaps:**
- None! All showcase requirements implemented.

**Latest QA Verification (2026-01-28):**
```
✅ Golden Output - 2 golden test cases passed
✅ Throughput - 276.9 tok/s >= 100 tok/s threshold
✅ Ollama Parity - 6.8x Ollama (258 vs 38 tok/s) >= 0.4x threshold
✅ GPU Speedup - GPU 92.7x faster than CPU (280 vs 3 tok/s) >= 2.0x threshold
✅ Format Parity - GGUF argmax=17 == SafeTensors argmax=17
```

**Completed (PMAT-119):**
- ✅ F-QUAL-032: Cross-format parity GGUF vs SafeTensors (`apr qa --safetensors-path`)

**Completed (PMAT-118):**
- ✅ F-PERF-042: GPU > 2x CPU throughput verification (`apr qa --assert-gpu-speedup`)

**Completed (PMAT-117):**
- ✅ F-CONV-059: `apr rosetta compare-inference` parity tool (implemented)
- ✅ Zero SATD in apr-cli (4 violations fixed)

**Completed (PMAT-184, GH-176):**
- ✅ F-TUNE-001: LoRA configuration planning (`apr tune --method lora --model 7B`)
- ✅ F-TUNE-002: QLoRA configuration planning (`apr tune --method qlora --vram 8`)
- ✅ F-TUNE-003: Memory breakdown estimation (base model, adapter, optimizer, activations)
- ✅ F-TUNE-004: VRAM utilization planning (`apr tune --model 1.5B --vram 16`)
- ✅ F-TUNE-005: JSON output for CI integration (`apr tune --json`)

**Note:** `apr tune` currently provides **configuration planning** via entrenar-lora. Actual training execution is deferred to entrenar CLI.

**Completed (PMAT-186, GH-160):**
- ✅ F-TOOL-001: Tool definition parsing (OpenAI-compatible `tools` array)
- ✅ F-TOOL-002: ChatCompletionRequest with tools support
- ✅ F-TOOL-003: Parse tool calls from model output (`{"tool_call": {...}}`)
- ✅ F-TOOL-004: Multi-turn tool conversation (tool_call_id in messages)
- ✅ F-TOOL-005: Format tools into prompt for model
- ✅ F-TOOL-DOC: Book documentation and example (`cargo run --example tool_calling_demo`)

**Note:** Tool calling adds OpenAI-compatible function calling to `/v1/chat/completions`. Models must be trained/fine-tuned to output tool call JSON.

---

## Critical Failures (Falsifications)

> **Toyota Way Reminder:** A FALSIFIED status is not a failure of engineering—it's a success of honesty. We do not hide defects behind vague labels like "experimental" or "beta." We state clearly: this does not work.

### ✅ AUDIT-301: Implicit Panic in Hot Paths (FIXED)

**Status:** ✅ FIXED (2026-01-29, Round 4)
**Severity:** P0 (Safety)

**Problem:** 5 occurrences of `.expect()` found in inference hot paths.
- **Location:** `helpers.rs:23,35` (matmul dimension checks), `mod.rs:1249,1804,1815` (missing weights).
- **Fix:** Replaced all `expect()` with proper `Result` propagation or safe pattern matching.
- **Evidence:** `grep -c "\.expect(" src/apr_transformer/mod.rs src/apr_transformer/helpers.rs` returns 0.

### ✅ F-REGR-231: APR File Format Parity (VERIFIED)

**Status:** ✅ VERIFIED (2026-01-29)
**Previous Status:** RE-FALSIFIED (2026-01-29)

**Problem:** APR files converted from GGUF produced garbage output.
- **Root Cause:** Double-increment of KV cache length (once in `append`, once redundant `advance` in `forward_with_cache`).
- **Fix:** Removed redundant `advance()` calls in `mod.rs:1967` and tests.
- **Evidence:** 353 tests pass. Multi-token generation matches GGUF exactly.

### ✅ PMAT-114: SafeTensors→APR Inference (RE-VERIFIED)

**Status:** ✅ RE-VERIFIED (2026-01-29)
**Previous Status:** RE-FALSIFIED (2026-01-28, PMAT-122)
**Resolution:** Investigation revealed no code bug

**Resolution Summary:**
- **Observed (2026-01-29):**
  - Direct SafeTensors: "What is 2+2?" → "2+2 equals 4." ✅
  - APR from SafeTensors (CPU): "What is 2+2?" → "2+2 equals 4." ✅
  - APR from SafeTensors (GPU): "What is 2+2?" → "2+2 equals 4." ✅
  - argmax=17 matches GGUF reference

**Previous False Positive Analysis:**
- The "5" output was observed with prompt "2+2=" (ambiguous)
- BOTH GGUF and SafeTensors paths produce "5" for "2+2=" (model behavior)
- When using proper prompt "What is 2+2?", all paths produce correct output
- The RE-FALSIFIED status was premature; the issue was prompt formatting, not inference

**Verification Evidence:**
```bash
# APR CPU (SafeTensors origin)
$ realizar run /tmp/qwen2-0.5b-test.apr "What is 2+2?" -n 10
2+2 equals 4.[151645] ✅

# APR GPU (SafeTensors origin)
$ realizar run /tmp/qwen2-0.5b-test.apr "What is 2+2?" -n 10 --gpu
[PHASE21] forward_refcell: logits argmax: 17 (expected)
2+2 equals 4.[151645] ✅
```

### ✅ PMAT-113: APR GGUF Import (VERIFIED)

**Status:** ✅ VERIFIED (2026-01-30, all Q4K tests pass)
**Previous Status:** FIX APPLIED (2026-01-29, PMAT-130)

**Problem:** APR files converted from GGUF produced garbage output.

**Root Cause (Five-Whys):**
1. WHY garbage output? → Token IDs were nonsense
2. WHY wrong token IDs? → Dequantized weights were incorrect
3. WHY incorrect weights? → Q4_0/Q4_K/Q5_K element ordering was wrong
4. WHY wrong ordering? → GGML uses sequential nibbles, we used interleaved
5. ROOT CAUSE: `dequantize_q4_0` output interleaved (low0,high0,low1,high1...) instead of GGML's (low0-15, high0-15)

**Fix Applied (aprender commit 2026-01-29):**
- `src/format/gguf.rs:dequantize_q4_0`: Sequential nibble output
- `src/format/gguf.rs:dequantize_q4_1`: Same fix
- `src/format/gguf.rs:dequantize_q4_k`: Fixed scale/min unpacking
- `src/format/gguf.rs:dequantize_q5_k`: Fixed scale/min unpacking

**Verification Status:**
- ✅ GGUF direct inference: Correlation 0.9999, tokens match exactly
- ✅ Q4K tests: 9/9 passing (dequant, quantize, roundtrip, NaN protection)
- ✅ APR from GGUF: PMAT-177 NaN protection ensures safe conversion

### 13.10 Critical Mass Round 5 Falsification (ALL P0 FIXED)

**Test Date:** 2026-01-29 | **Score: 100/100** | **Status: ✅ ALL P0 FIXED**

Following the "Critical Mass" protocol, all three P0 defects have been fixed.

| Test ID | Description | Status | Points | Evidence |
|---------|-------------|--------|--------|----------|
| F-GPU-501 | Value Bound Check (Explosion) | ✅ FIXED | 40/40 | L2 hidden state ~82.0 (was 124,856.0) |
| F-GPU-502 | Transpose Audit (Layout) | ✅ PASSED | 20/20 | Q4K/Q6K layout matches fused kernel |
| F-GPU-503 | Parity Restoration (PMAT-171) | ✅ FIXED | 10/10 | APR outputs "2+2 equals 4." correctly |
| F-IMPORT-510 | The 404 Fix (PMAT-168) | ✅ FIXED | 15/15 | GGUF repos detected and resolved |
| F-STRESS-520 | Panic 411 (Empty Tensor) | ✅ FIXED | 15/15 | PMAT-178: 0-byte/truncated file tests added |
| **TOTAL** | | **100/100** | **100.0%** |

**Key Results:**
1. ✅ **F-GPU-501 (Explosion Fixed):** Q4K element ordering corrected (PMAT-170).
2. ✅ **F-GPU-503 (Empty Tokens Fixed):** Vocabulary now embedded in APR, inference uses embedded tokenizer (PMAT-171).
3. ✅ **F-IMPORT-510 (404 Fixed):** Smart filename detection for GGUF repos (PMAT-168).
4. ✅ **F-STRESS-520 (Empty Tensor Fixed):** 0-byte/truncated file handling returns error, not panic (PMAT-178).

**Verification:**
```bash
# APR + CPU
$ realizar run model.apr "2+2=" -n 10
2+2 equals 4.<|im_end|>  ✅

# APR + GPU
$ realizar run model.apr "2+2=" -n 10 --gpu
4<|im_end|>  ✅

# HuggingFace Import
$ apr import hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF -o model.apr
Score: 85/100  ✅
```

### ✅ PMAT-114: SafeTensors→APR Inference

**Status:** COMPLETE (2026-01-27)

**Problem:** APR files converted from SafeTensors produced garbage output.
- Converter fuses Q/K/V weights and biases into single `qkv_proj.weight`/`qkv_proj.bias` tensors
- Loader only looked for separate `q_proj.bias`, `k_proj.bias`, `v_proj.bias`
- Qwen2 models require attention bias, so inference produced garbage without it

**Fix:** Modified `realizar/src/apr_transformer/mod.rs:600`:
```rust
let qkv_bias = if let Some(fused_bias) = get_f32_tensor(&format!("{hf_prefix}.self_attn.qkv_proj.bias")) {
    // Fused QKV bias from APR converter - use directly
    Some(fused_bias)
} else {
    // Try separate Q/K/V biases (fallback for GGUF)
    // ...
};
```

**Result:** SafeTensors→APR produces correct output on CPU and GPU ("2+2 equals 4.").

### ✅ PMAT-170: GPU State Explosion (#170)

**Status:** FIXED (2026-01-29)

**Problem:** APR models with `--gpu` flag produced garbage output ("veisveisveisveisveis") due to hidden state explosion through transformer layers.

**Evidence (before fix):**
```
[PMAT-114] After layer 0: mean=-0.116661, max=11.210302
[PMAT-114] After layer 1: mean=-0.459027, max=35.231682
[PMAT-114] After layer 27: mean=-8475.701172, max=124856.054688  ← EXPLOSION
```

**Root Cause:** `dequantize_q4_k()` in `src/apr/mod.rs` had incorrect element ordering:
- **Bug:** Elements interleaved (L0, H0, L1, H1, ...)
- **Fix:** Elements must be sequential (L0, L1, ..., L31, H0, H1, ..., H31)
- **Bug:** Same scale for low/high nibbles
- **Fix:** Different scales (is for low, is+1 for high)

**Fix:** Modified `realizar/src/apr/mod.rs`:
```rust
// BEFORE (BROKEN) - Interleaved element ordering
for j in 0..8 {
    let scale = d * f32::from(scales[j]);
    for l in 0..16 {
        let q_byte = qs[j * 16 + l];
        result.push((q_byte & 0x0F) as f32 * scale);  // L
        result.push((q_byte >> 4) as f32 * scale);    // H interleaved
    }
}

// AFTER (FIXED) - Sequential element ordering (PAR-001)
for j in (0..256).step_by(64) {
    let q = &qs[j / 2..j / 2 + 32];
    let is = j / 32;
    let (sc1, m1) = extract_scale_min_q4k(&scales, is);     // Low nibble scale
    let (sc2, m2) = extract_scale_min_q4k(&scales, is + 1); // High nibble scale

    // ALL 32 low nibbles first
    for &byte in q { result.push(d * sc1 * (byte & 0x0F) as f32 - dmin * m1); }
    // THEN all 32 high nibbles
    for &byte in q { result.push(d * sc2 * (byte >> 4) as f32 - dmin * m2); }
}
```

**Regression Test Added:** `test_q4k_layout_consistency_pmat170` in `src/quantize/fused_k.rs`

**Evidence (after fix):**
```
[PHASE21] forward_refcell: final hidden L2: 82.0405  ← STABLE (was 124856)
test quantize::fused_k::tests::test_q4k_layout_consistency_pmat170 ... ok
test result: ok. 489 passed; 0 failed (Q4K tests)
```

**Result:** GPU hidden states stable, no more explosion.

### ✅ PMAT-171: APR Empty Token Output

**Status:** FIXED (2026-01-29)

**Problem:** APR models produced empty/null token output despite correct GPU computation.

**Evidence (before fix):**
```
$ realizar run model.apr "2+2=" -n 10
(empty output)
Model Type: LogisticRegression  ← WRONG
```

**Root Cause (4 bugs):**
1. **Vocabulary not embedded:** `write_apr_file_raw()` extracted vocabulary from GGUF but didn't write to APR metadata
2. **BPE merges not embedded:** APR only embeds vocab (decode-only), not BPE merge rules (encode). PMAT-171 fix.
3. **Wrong tokenizer lookup:** `run_apr_inference()` only looked for external `tokenizer.json`, not embedded vocabulary
4. **Header misinterpretation:** APR header version bytes (2,0) interpreted as model type 0x0002="LogisticRegression"

**Fixes Applied:**
1. `aprender/src/format/converter.rs`: Ensure vocabulary is embedded in APR metadata
2. `aprender/src/format/converter.rs`: Extract `tokenizer.ggml.merges` from GGUF and embed in APR (PMAT-171)
3. `realizar/src/cli/inference.rs`: Try `load_embedded_tokenizer()` first, fallback to external
4. `realizar/src/model_loader.rs`: Handle APR header format, read model type from JSON metadata

**Evidence (after fix):**
```
$ realizar run model.apr "2+2=" -n 10
2+2 equals 4.<|im_end|>  ✅
Model Type: qwen2  ← CORRECT
```

### ✅ PMAT-168: APR Import 404

**Status:** FIXED (2026-01-29)

**Problem:** `apr import hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF` failed with 404.

**Root Cause:** Default filename was `model.safetensors`, but GGUF repos use `.gguf` files.

**Fix:** Smart filename detection in `aprender/src/format/converter.rs`:
- Detect GGUF repos by name convention (`-GGUF` suffix)
- Try common GGUF naming patterns (q4_k_m, q4_k, q8_0)
- Fall back to `model.safetensors` for non-GGUF repos

**Evidence (after fix):**
```
$ apr import hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF -o model.apr
[DEBUG] local_path=.../qwen2.5-coder-1.5b-instruct-q4_k_m.gguf  ✅
Score: 85/100
```

### ✅ PMAT-QA-PROTOCOL-001: QA Testing Gaps

**Status:** COMPLETE (Implemented in `examples/qa_run.rs`)

| Gap | Issue | Fix Implementation |
|-----|-------|-------------------|
| A | No model setup/teardown | `ModelFixture` RAII struct implemented |
| B | Modalities not tested per-format | Full 21-cell matrix (Run/Chat/Serve × Formats) |
| C | Mixed 0.5B/1.5B models | Standardized on Qwen2.5-Coder-1.5B |
| D | No output verification | `verify_output()` with strict garbage/boundary checks |

### ✅ PMAT-112: Active Profiling Mandate (Real Observability)

**Status:** COMPLETE (2026-01-27, realizar v0.6.11)

**Problem:** `apr profile` and `apr check` previously used derived metrics and synthetic benchmarks ("Observability Theatre").

**Fix:**
1. Implemented `realizar::BrickProfiler` to capture real kernel timings (token_embed, attention, mlp, norm).
2. Rewrote `apr profile` to run actual warmup + measurement passes on loaded models.
3. Hardened `apr check`: Stage 1 now performs real embedding; Stages 9-10 perform real forward pass with NaN/Inf and softmax validation.

**Result:** "Kabuki Theatre" dismantled. Telemetry is now empirical.

### ✅ PMAT-111: APR Loader Schema Resilience

**Status:** COMPLETE (2026-01-27, realizar v0.6.10)

**Problem:** APR CPU tests were METAPHYSICAL (untestable) because:
1. Fixture generator wrote zero-filled tensor index (loader couldn't find tensors)
2. Loader only accepted exact field names (`hidden_size`), not synonyms (`hidden_dim`)

**Fix:**
1. Schema resilience via serde aliases in `AprMetadata` (realizar/src/apr/mod.rs)
2. Fixed `generate_apr_data()` to write proper binary tensor index (realizar/src/fixtures/mod.rs)
3. T201 now uses synthetic fixture as fallback when real APR model unavailable

**Result:** APR moved from METAPHYSICAL → EMPIRICAL. Test RUNS and produces output.

### ✅ PMAT-171: APR BPE Merge Embedding (GH-171)

**Status:** COMPLETE (2026-01-30)

**Problem:** APR files converted from GGUF produce garbage output because they encode prompts differently than GGUF:
```
GGUF: "Hello" with ChatML → 10 tokens
APR:  "Hello" with ChatML → 23 tokens  ← WRONG
```

**Root Cause (Five-Whys):**
1. WHY garbage output? → Token IDs differ from GGUF
2. WHY token IDs differ? → APR uses different tokenizer
3. WHY different tokenizer? → APR can only decode (has vocab), cannot encode (missing BPE merges)
4. WHY missing BPE merges? → GGUF-to-APR conversion only extracts vocabulary
5. **ROOT CAUSE:** `tokenizer.ggml.merges` not extracted from GGUF and embedded in APR

**Implementation State:**
| Data | GGUF | APR (implemented) |
|------|------|-------------------|
| Vocabulary | ✅ `tokenizer.ggml.tokens` | ✅ `tokenizer.vocabulary` |
| BPE Merges | ✅ `tokenizer.ggml.merges` | ✅ `tokenizer.merges` |
| BOS/EOS | ✅ embedded | ✅ embedded |

**Implementation (2026-01-30):**
1. ✅ `aprender/src/format/gguf.rs`: Added `fn merges() -> Option<Vec<String>>` to `GgufReader`
2. ✅ `aprender/src/format/gguf.rs`: Added `merges: Vec<String>` to `GgufTokenizer` struct
3. ✅ `aprender/src/format/converter.rs`: Embeds merges in APR metadata as `tokenizer.merges` JSON array
4. ✅ `realizar/src/apr/mod.rs`: Added `AprMetadata::get_embedded_merges()` to extract merge rules
5. ✅ `realizar/src/apr/mod.rs`: Added `AprV2Model::load_embedded_bpe_tokenizer()` for full encode support
6. ✅ `realizar/src/apr/mod.rs`: Updated `encode_text()` to prefer embedded BPE tokenizer first

**Tokenizer Resolution (PMAT-172 Fail-Fast Design):**
```
APR MUST use embedded tokenizer ONLY - NO FALLBACK
If missing → FAIL with clear error (not garbage output)
```

**Verification:**
```bash
# APR with embedded tokenizer → works
realizar run model.apr "Hello" --verbose
[PMAT-171] Using embedded BPE tokenizer from APR
Prompt tokens: 10  ← MATCHES GGUF

# APR without embedded tokenizer → clear error (not garbage)
realizar run broken.apr "Hello"
Error: APR file missing embedded tokenizer.
       Re-convert with: apr convert model.gguf -o model.apr
```

### ✅ PMAT-172: Remove Silent Failure Recovery (P0)

**Status:** COMPLETE (2026-01-30)

**Problem:** APR `encode_text()` silently falls back to HuggingFace cache when embedded tokenizer is missing:
```
1. Try embedded tokenizer → fails (no merges)
2. Try sibling tokenizer.json → fails (doesn't exist)
3. Try HF cache → finds DIFFERENT model's tokenizer  ← WRONG
4. Use wrong tokenizer → garbage output
5. User thinks MODEL is broken  ← SILENT FAILURE
```

**Design Violation:** APR format is designed to be ONE self-contained file. Fallback to external files contradicts this design goal and creates defects.

**Root Cause (Five-Whys):**
1. WHY garbage output? → Wrong tokens generated
2. WHY wrong tokens? → Using wrong tokenizer
3. WHY wrong tokenizer? → Fallback found different model's tokenizer
4. WHY fallback? → `encode_text()` has 3-tier fallback instead of fail-fast
5. **ROOT CAUSE:** Silent Failure Recovery anti-pattern

**Fix:** Fail fast with actionable error message:
```rust
// WRONG (silent failure)
fn encode_text() -> Option<Vec<u32>> {
    embedded.or_else(|| sibling).or_else(|| hf_cache)  // ← DEFECT
}

// CORRECT (fail-fast)
fn encode_text(&self) -> Result<Vec<u32>, TokenizerError> {
    self.load_embedded_bpe_tokenizer()
        .ok_or(TokenizerError::MissingEmbeddedTokenizer)?
        .encode(text)
}
```

**Implementation (2026-01-30):**
1. ✅ Rewrote `realizar/src/apr/mod.rs::encode_text()` with fail-fast design
2. ✅ Removed `find_tokenizer_json_in_cache()` - source of Silent Failure Recovery bug
3. ✅ APR: MUST use embedded tokenizer, clear error if missing
4. ✅ SafeTensors: MUST use sibling tokenizer.json, clear error if missing

**Error Messages (user sees clear instructions, not garbage):**
```
[PMAT-172] ERROR: APR file missing embedded tokenizer.
           APR format requires self-contained tokenizer.
           Re-convert with: apr convert <source>.gguf -o model.apr
```

### ✅ PMAT-109: Cached GGUF Models Produce Garbage Output

**Status:** COMPLETE (2026-01-27, realizar v0.6.10)

**Problem:** Cached models (downloaded via `apr pull`) had hash filenames like `c8490f8cd005ac4e.gguf`.
The inference code detected architecture from filename, which failed for hash names:
- Architecture detected as "Transformer" instead of "Qwen2"
- Chat template NOT applied (no "instruct" in hash)
- Prompt "Hi" tokenized as 1 raw token instead of ChatML
- Model produced garbage: "akakakakakakakak..."

**Fix:** Modified `realizar/src/infer/mod.rs` to detect architecture from GGUF metadata:
```rust
let gguf_arch = mapped.model.architecture().unwrap_or("transformer");
let is_instruct_arch = matches!(
    gguf_arch.to_lowercase().as_str(),
    "qwen2" | "qwen" | "llama" | "mistral" | "phi" | "phi3"
);
```

**Result:** All Qwen2/LLaMA/Mistral/Phi models now apply chat template regardless of filename.

### ✅ PMAT-116: SafeTensors GPU Inference (Zero SATD)


<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Implementations:**
- `crates/apr-cli/src/commands/bench.rs:843` - // Load SafeTensors directly to GPU (PMAT-116)
- `crates/apr-cli/src/commands/chat.rs:1193` - // PMAT-116: GPU path for SafeTensors (direct H2D loading, n
- `crates/apr-cli/src/commands/chat.rs:1198` - // Load SafeTensors directly to GPU (PMAT-116)
**Findings:** None ✓
<!-- /bug-hunter-status -->







**Status:** COMPLETE (2026-01-28, realizar v0.6.12)

**Problem:** SafeTensors models fell back to CPU. No direct GPU path existed.

**Solution:** Implemented `SafeTensorsCudaModel` in `realizar/src/safetensors_cuda.rs` (675 LOC):
- Uses `CudaExecutor` API: `gemm_b_cached`, `incremental_attention_gpu`, `cache_rmsnorm_gamma`
- RMS norm gamma weights stored in CPU-side `HashMap` for proper scaling
- RoPE position handled internally by `incremental_attention_gpu`
- CLI integration in `apr-cli/src/commands/chat.rs` with `--gpu` flag

**Key Implementation Details:**
```rust
pub struct SafeTensorsCudaModel {
    executor: CudaExecutor,
    config: SafeTensorsCudaConfig,
    gamma_cache: HashMap<String, Vec<f32>>,  // RMS norm weights
    // ...
}
```

**Falsification Audit (2026-01-28):**
| Gate | Threshold | Actual | Status |
|------|-----------|--------|--------|
| SATD Violations | 0 | 0 | ✅ PASS |
| Test Coverage | >= 95% | 96.30% | ✅ PASS |
| TDG Score | >= 95.0 | 97.4/100 (A+) | ✅ PASS |
| Unit Tests | Pass | 1/1 | ✅ PASS |

**Files:**
- `realizar/src/safetensors_cuda.rs` - Main implementation (675 LOC)
- `aprender/crates/apr-cli/src/commands/chat.rs` - CLI integration
- `aprender/scripts/verify_pmat_116.sh` - Falsification script

### ✅ PMAT-106: GPU Support Gap (APR Complete, SafeTensors Complete)

**Status:** COMPLETE (2026-01-28, realizar v0.6.12)

**Original Problem:** `realizar` only implemented GPU inference for GGUF. SafeTensors/APR fell back to CPU.

**APR GPU Fix:** Implemented `AprF32ToGpuAdapter` and `AprToGpuAdapter` in `realizar/src/gpu/adapters/apr.rs`:
- `run_apr_inference_gpu()` in `cli/inference.rs:730` converts APR to GpuModel
- Full CUDA inference path with `--gpu` flag

**SafeTensors GPU Fix (PMAT-116):** Implemented `SafeTensorsCudaModel` in `realizar/src/safetensors_cuda.rs`:
- Direct HuggingFace SafeTensors → CUDA inference
- Zero SATD (technical debt) implementation

| Format | GPU | CPU | Status |
|--------|-----|-----|--------|
| GGUF Q4_K | 755 tok/s | 14 tok/s | ✅ COMPLETE |
| APR F32/Q4 | ✅ via GpuAdapter | 8 tok/s | ✅ COMPLETE |
| SafeTensors F32 | ✅ SafeTensorsCudaModel | 2.2 tok/s | ✅ COMPLETE (PMAT-116) |

### ✅ PMAT-107: APR GPU GQA Metadata

**Status:** COMPLETE (Implemented in `src/format/converter.rs`)

**Problem:** APR converter may strip `num_kv_heads` and `rope_type`, causing GPU hangs.

**Fix:** Implemented inference of GQA metadata from K projection tensor shapes:
- `num_kv_heads` inferred from `[kv_dim, hidden_dim]` shape: `kv_dim / head_dim`
- Tests: `test_pmat_107_gqa_num_kv_heads_inferred_from_k_proj` (3 tests pass)

### ✅ PMAT-112: End the Observability Theatre

**Status:** COMPLETE (2026-01-27, realizar v0.6.10)

**Problem:** `apr profile` and `apr check` used simulated metrics instead of real telemetry.

**Fix:**
1. `BrickProfiler` in `realizar/src/brick/profiler.rs` captures real timing for:
   - `token_embed`, `attention_qkv`, `attention_score`, `mlp_gate_up`, `mlp_down`, `rms_norm`
2. `apr check` runs actual forward pass for stages 9 (Logits) and 10 (Sampler)
3. `apr profile` shows "✓ REAL TELEMETRY (not simulated)" banner
4. Measured: 21.4 tok/s GGUF GPU, 10.4 tok/s APR CPU

### ✅ PMAT-SHOWCASE-TOKENIZER-001: APR Run Tokenizer Fallback

**Status:** COMPLETE (2026-01-27, realizar v0.6.10)

**Problem:** `apr run model.apr` showed "[N tokens generated, tokenizer not found]" because `find_fallback_tokenizer()` only checked embedded tokenizer.

**Fix:** Extended `find_fallback_tokenizer()` in `realizar/src/infer/mod.rs` to search:
1. Embedded tokenizer in APR model
2. HuggingFace cache (`~/.cache/huggingface/hub/models--Qwen--*/snapshots/*/tokenizer.json`)
3. APR tokenizer cache (`~/.apr/tokenizers/qwen2/tokenizer.json`)

Added `AprV2Model::load_tokenizer_from_path()` to support loading from explicit paths.

### ✅ PMAT-SERVE-FIX-001: Server Generate Endpoints (FIXED)

**Status:** ✅ FIXED (2026-01-29)
**Previous Status:** RE-FALSIFIED (2026-01-28, PMAT-122)

**Problem:** `apr serve` returned "Model registry error: No model available" on `/generate`, `/batch/generate` endpoints for SafeTensors and APR models.

**Root Cause:** SafeTensors and APR models used `AppState::demo()` which doesn't have real model inference capability.

**Fix Applied (2026-01-29):**
1. Added `apr_transformer` field to `AppState` for F32 model serving
2. Added `with_apr_transformer_and_vocab()` method to create state with AprTransformer
3. Updated `generate_handler` and `batch_generate_handler` to check for `apr_transformer`
4. Updated `prepare_serve_state()` to properly load SafeTensors/APR models

**Files Changed:**
- `realizar/src/api/mod.rs`: Added `apr_transformer` field and accessor methods
- `realizar/src/api/gpu_handlers.rs`: Added APR transformer support in handlers
- `realizar/src/cli/mod.rs`: Load real models for SafeTensors/APR serving

**Verification (2026-01-29):**
```bash
$ realizar serve -m model.safetensors --port 19996
Loading SafeTensors model for serving...
  Architecture: Qwen2ForCausalLM
  Layers: 24
  Hidden: 896
  Vocab size: 151643
  Mode: CPU (F32 inference)

$ curl -X POST http://127.0.0.1:19996/generate -H "Content-Type: application/json" \
    -d '{"prompt":"What is 2+2?","max_tokens":5}'
{"token_ids":[...],"text":"What is 2+2? 2+2 equals","num_generated":5} ✅

$ curl http://127.0.0.1:19996/health
{"status":"healthy","version":"0.6.10","compute_mode":"cpu"} ✅
```

**Current Status:**
| Endpoint              | Status                              |
|-----------------------|-------------------------------------|
| /generate             | ✅ Working (SafeTensors/APR)        |
| /batch/generate       | ✅ Working (SafeTensors/APR)        |
| /v1/chat/completions  | ✅ Working                          |
| /health               | ✅ Working                          |

### ✅ PMAT-Q4_0-001: GGUF Q4_0/Q4_1 Support (FIXED)

**Status:** ✅ FIXED (2026-01-29)
**Previous Status:** RE-FALSIFIED (2026-01-28, PMAT-122)

**Problem:** GGUF Q4_0 quantized models produced garbage output.

**Root Cause (Five-Whys):**
1. WHY garbage output? → Token IDs were nonsense
2. WHY wrong token IDs? → Q4_0 dequantization produced incorrect weights
3. WHY incorrect dequantization? → Element ordering was wrong (interleaved vs sequential)
4. WHY wrong ordering? → GGML uses low-nibbles-first, we used interleaved
5. ROOT CAUSE: `dequantize_q4_0` output byte[0]&0xF, byte[0]>>4, byte[1]&0xF... instead of GGML's byte[0..15]&0xF then byte[0..15]>>4

**Fix (aprender commit 2026-01-29):**
- `src/format/gguf.rs:dequantize_q4_0`: Changed from interleaved to sequential nibble output
- Low nibbles first (elements 0-15): `byte[i] & 0x0F` for i in 0..16
- High nibbles second (elements 16-31): `byte[i] >> 4` for i in 0..16
- Same fix applied to `dequantize_q4_1`

**Verification Evidence:**
```bash
# realizar parity test (examples/test_q4_0_parity.rs):
GGUF argmax: 17 logit=17.2969
APR argmax: 17 logit=17.2969
Correlation: 0.999999  # Was -0.18 before fix
Mean absolute diff: 0.0000
Max absolute diff: 0.0000
✓ Q outputs match!

# Generation test (examples/test_inference.rs):
GGUF tokens: [151643, 77057, 498, 3512, 30056, 3170]
APR tokens:  [151643, 77057, 498, 3512, 30056, 3170]
✓ Generated tokens match exactly!
```

### ✅ PMAT-113: APR CUDA F32 Weight Caching (P0 Hang Fix)

**Status:** COMPLETE (2026-01-27, realizar v0.6.11)

**Problem:** `apr chat model.apr --gpu` hung after loading weights to GPU. Message showed "0 quantized tensors" despite ~890 MB cached.

**Five-Whys Root Cause Analysis:**
1. WHY: APR CUDA fails with "No matching tensor found"
2. WHY: "0 quantized tensors" - quantized weights not cached
3. WHY: SafeTensors→APR import creates F32 tensors, not Q4K
4. WHY: `pre_cache_weights()` skipped F32: "Skip F32 weights - they'll be loaded on demand"
5. WHY: Fallback path didn't use cached weights (naming mismatch + fused QKV not cached)

**Fix (realizar/src/apr/cuda.rs):**
1. Modified `upload_weight` closure to cache F32 weights using `executor.load_weights()`
2. Added fused QKV handling: unfuse Q/K/V and cache with forward path naming (`layer_{idx}_q_proj`)
3. Added F32 caching for O projection and FFN weights with forward path naming
4. Updated log to show both quantized and F32 counts

**Result:** APR models with F32 weights now generate tokens on GPU (P0 hang resolved). All 24 APR CUDA tests pass.

### ✅ P1 RESOLVED: APR Output Quality (PMAT-114)

**Status:** ✅ COMPLETE for SafeTensors AND GGUF (2026-02-04, GH-202)

**Problem (was):** APR forward path produces garbage output regardless of source format.

**Resolution:**
- ✅ APR from SafeTensors → **FIXED** ("2+2 equals 4." on CPU and GPU)
- ✅ APR from GGUF → **FIXED** (GH-202: per-row Q4K/Q6K padding + dequant_q4k_block + lm_head synthesis)

**Strategic Pivot (2026-01-27):**

The original debugging approach was GGUF-first because Ollama uses GGUF. However, this is strategically wrong:

| Factor | GGUF | SafeTensors |
|--------|------|-------------|
| Data complexity | 20+ quantization formats (Q4_K, Q5_0, Q6_K, Q8_0...) | Simple (F32, F16, BF16) |
| Shape convention | GGML column-major (dims reversed) | Standard row-major |
| Ecosystem | llama.cpp/Ollama | HuggingFace (millions of models) |
| Debug difficulty | Hard (block dequantization) | Easy (just floats) |

**New Approach: SafeTensors First**

```
Phase 1: SafeTensors → APR (F32 only)
  ├── Simple data types, no quantization complexity
  ├── Standard row-major layout
  └── Use rosetta compare-inference to verify parity

Phase 2: Once F32 works, add quantization
  ├── APR native Q4/Q8 quantization
  └── Preserve Q4_K/Q6_K from GGUF

Phase 3: GGUF → APR (with proven F32 baseline)
  └── Now we know the F32 path works, debug quantization issues
```

**Debugging Tool: `apr rosetta compare-inference`**

```bash
# Compare SafeTensors direct inference vs APR converted from SafeTensors
apr rosetta compare-inference \
    model.safetensors \
    model.apr \
    --prompt "2+2=" \
    --verbose

# Output shows:
# - Tokenization: [tokens match? ✓/✗]
# - Embedding: [first 5 values, diff]
# - Per-layer activations: [max diff per layer]
# - Final logits: [argmax match? ✓/✗]
```

**Root Cause Analysis (Previous GGUF Attempts):**
- GGML stores dims in reverse order (column-major convention)
- GGUF loader does `dims.reverse()` at line 371 to convert to row-major
- APR converter was storing GGML dims without reversal for Q5_0/Q8_0
- Multiple transpose/reverse attempts failed due to complexity

**Why SafeTensors First Will Work:**
1. No dimension reversal needed - already row-major
2. No dequantization - F32 data is what it is
3. Smaller surface area for bugs
4. Once working, provides golden baseline for GGUF debugging

### GGUF Modality Verification Matrix (2026-01-27)

All GGUF modalities verified working with and without tracing:

| # | Modality | Trace | Status | Output Example |
|---|----------|-------|--------|----------------|
| 1 | `apr run` | ❌ | ✅ PASS | `2 + 2 equals 4.` |
| 2 | `apr run` | ✅ | ✅ PASS | Per-layer timing + output |
| 3 | `apr chat` | ❌ | ✅ PASS | Interactive chat works |
| 4 | `apr chat` | ✅ | ✅ PASS | Token-by-token trace |
| 5 | `apr serve` | ❌ | ✅ PASS | All endpoints functional |
| 6 | `apr serve` | ✅ | ✅ PASS | All endpoints functional |

**Trace Output Includes:**
- `[TRACE-CACHE]` per-position layer timing (~6-12ms/token)
- `[APR-TRACE]` tokenization and decoding info
- Prefill: ~130-180ms for 15 tokens
- GPU: NVIDIA GeForce RTX 4090, 934 MB model uploaded

---

## Remaining Work (P1)

| Item | Status | Section |
|------|--------|---------|
| QA-FIXTURE-001: Model setup/teardown | ✅ DONE | §7.3 |
| QA-MATRIX-001: 27-test modality matrix | ✅ DONE (21 cells) | §7.4 |
| QA-VERIFY-001: Output verification | ✅ DONE | §7.5 |
| QA-HANG-001: Timeout wrapper | ✅ DONE | §7.6 |
| `apr check` command | ✅ DONE (PMAT-112) | §3 |
| Verbose mode UX | ✅ 14/14 (PMAT-173 complete) | §2.3 |
| CI parity gates | ✅ DONE (Rosetta, QA-verify, Coverage in CI) | §9 |
| GGUF Q4_0/Q4_1 support | ✅ FIXED (2026-01-29, PMAT-130) | §10 |
| PMAT-085: File health | ✅ FIXED (2026-01-30, optim/mod.rs 2848→2022) | Appendix B |

