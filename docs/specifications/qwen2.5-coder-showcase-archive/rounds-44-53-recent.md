# Rounds 44-53: Recent Progress

> Archived from qwen2.5-coder-showcase-demo.md (lines 1-427)

# Qwen2.5-Coder Showcase: Unified Inference Architecture

**Version:** 9.30.0 (PMAT-237 - Pre-Dispatch Contract Gate)
**Status:** ✅ **ALL FORMATS WORKING** (GGUF, SafeTensors CPU, SafeTensors GPU)
**Popperian Score:** 99/100 (Grade: A+ — Full GPU/CPU parity, mandatory testing, compile-time contracts, pre-dispatch validation)
**Code Coverage:** 96.94% (target: ≥95%)
**Tool Coverage:** 17/17 (100%) - All APR tools verified + tensor contract gate
**CLI Test Coverage:** 10,363 lib tests passing (446 converter tests)
**Author:** PAIML Engineering
**Date:** 2026-02-05
**Ground Truth:** SafeTensors (F32/BF16/F16) - See Section 0
**Last Falsification Run:** 2026-02-05 (Round 53 - PMAT-237 Pre-Dispatch Contract Gate)
**Quality Philosophy:** Toyota Way + Popperian Falsification (Zero SATD, Stop-the-Line, Jidoka, see Appendix F)

## Release Criteria (Round 53 Update - 2026-02-05)

| Format | CPU | GPU | Status | Notes |
|--------|-----|-----|--------|-------|
| GGUF Q4K (pre-baked from HF) | ✅ | ✅ | **VERIFIED** | 21.6 tok/s (1.5B model) |
| SafeTensors F16 (passthrough) | ✅ | ✅ | **VERIFIED** | Round 49: F16 passthrough (0% diff) |
| SafeTensors 0.5B (direct inference) | ✅ | ✅ | **FIXED** | PMAT-236: Chat template now enforced |
| APR Q4K (converted FROM GGUF) | ✅ | ✅ | **FULLY FIXED** | PMAT-216: GPU/CPU parity 0.00% diff |
| APR F16 (converted FROM SafeTensors) | ✅ | ✅ | **VERIFIED** | Round 49: F16 passthrough preserves bytes |
| GGUF Q4K (converted FROM SafeTensors) | ✅ | ✅ | **FIXED** | Rosetta now defaults to Q4K ([#205](https://github.com/paiml/aprender/issues/205)) |

**Release = READY ✅ (All formats verified working)**

**Round 53 Progress (2026-02-05) - PMAT-237 PRE-DISPATCH CONTRACT GATE:**
| Component | Before | After | Status | Notes |
|-----------|--------|-------|--------|-------|
| Corrupt model detection | `apr qa` catches, `apr run` ignores | **All action commands gated** | ✅ **IMPLEMENTED** | Single gate in `execute_command()` |
| `--skip-contract` flag | Not exist | **Global CLI flag** | ✅ **NEW** | Escape hatch for diagnostic tooling |
| `extract_model_paths()` | Not exist | **Command-aware path extraction** | ✅ **NEW** | Action vs diagnostic classification |
| `validate_model_contract()` | Not exist | **`RosettaStone::validate()` pre-dispatch** | ✅ **NEW** | Exit code 5 on violation |
| Action commands gated | 0/25 | **17+ commands gated** | ✅ **ENFORCED** | run, serve, chat, bench, eval, profile, trace, etc. |
| Diagnostic commands exempt | N/A | **13+ commands exempt** | ✅ **CORRECT** | qa, validate, inspect, debug, tensors, etc. |
| Rosetta subcommands | Not classified | **Action vs diagnostic split** | ✅ **NEW** | convert/chain/verify gated; inspect/diff exempt |
| E2E: corrupt APR blocked | Runs and produces garbage | **Exit 5: "12 violations in 12 tensors"** | ✅ **VERIFIED** | `e910cab26ae116eb.converted.apr` |
| E2E: --skip-contract bypass | N/A | **Bypasses gate, runs inference** | ✅ **VERIFIED** | Escape hatch works |
| E2E: diagnostic on corrupt | Always worked | **Still works (exempt)** | ✅ **VERIFIED** | inspect, tensors unaffected |
| Clippy (coverage.rs) | 3 identical-block errors | **Consolidated branches** | ✅ **FIXED** | `PartitionSpec::from_tensor_name()` |
| Test: cc2_trueno_is_compute | FAILED (matched comment) | **Ignores commented lines** | ✅ **FIXED** | `format_parity_tests.rs` |

**PMAT-237 Root Cause (Five Whys):**

| Why | Question | Answer |
|-----|----------|--------|
| 1 | Why does `apr run` produce garbage on corrupt models? | No contract validation before inference dispatch |
| 2 | Why no validation? | Each command handled its own validation (or didn't) |
| 3 | Why per-command? | No centralized gate existed |
| 4 | Why no centralized gate? | Validation was treated as a per-tool concern, not a pre-dispatch concern |
| 5 | Why solution? | **Single `validate_model_contract()` gate in `execute_command()` before match dispatch. Diagnostic commands exempt. `--skip-contract` escape hatch.** |

**PMAT-237 Design Principles:**
1. **Single gate, all commands** — one function in `execute_command()`, not 25 per-command changes
2. **Diagnostic commands exempt** — tools like `qa`, `inspect`, `debug` MUST work on corrupt models
3. **Uses existing infrastructure** — `RosettaStone::validate()` + `CliError::ValidationFailed` (exit 5)
4. **`--skip-contract` escape hatch** — global flag for power users and CI

**Round 52 Progress (2026-02-05) - PMAT-236 CHAT TEMPLATE COMPILE-TIME ENFORCEMENT:**
| Component | Before | After | Status | Notes |
|-----------|--------|-------|--------|-------|
| SafeTensors inference | Empty/garbage output | **"2 + 2 equals 4."** | ✅ **FIXED** | Both CPU and GPU |
| Chat template (GGUF) | Applied in format code | **Centralized via `PreparedTokens`** | ✅ **ENFORCED** | Compile-time guarantee |
| Chat template (SafeTensors) | **MISSING** | **Centralized via `PreparedTokens`** | ✅ **FIXED** | Root cause of garbage |
| Chat template (APR) | Applied in format code | **Centralized via `PreparedTokens`** | ✅ **ENFORCED** | Compile-time guarantee |
| `PreparedTokens` newtype | Not exist | **Private inner `Vec<u32>`** | ✅ **NEW** | Cannot bypass chat template |
| `prepare_tokens()` | Not exist | **Unified token preparation** | ✅ **NEW** | Format-aware, template-aware |
| Corrupt model detection | `apr validate --quality` | **PMAT-235 gates catch it** | ✅ **WORKING** | 217 violations in corrupt file |
| Model file (0.5B) | 2.52 GB (corrupt F32, 99.9% zeros) | **942 MB (BF16, healthy)** | ✅ **FIXED** | Re-downloaded via `apr pull` |

**PMAT-236 Root Cause (Five Whys):**

| Why | Question | Answer |
|-----|----------|--------|
| 1 | Why empty/garbage SafeTensors output? | First generated token is EOS (151645) |
| 2 | Why immediate EOS? | Model gets raw text without chat template wrapping |
| 3 | Why no chat template? | SafeTensors path skipped `format_messages()` call |
| 4 | Why skipped? | Each format had independent tokenization code, SafeTensors forgot it |
| 5 | Why solution? | **`PreparedTokens` newtype: private inner data, constructed only via `prepare_tokens()` which ALWAYS applies chat template. Compile error to bypass.** |

**Secondary finding: Corrupt model file**
- Original `/home/noah/models/qwen2.5-coder-0.5b-instruct/model.safetensors` was 2.52 GB (F32) with 99.9% zero values
- Python safetensors reference confirmed: data IS zeros in the file (not a loading bug)
- PMAT-235 contract gates correctly flagged: "217 violations in 155 tensors"
- Fresh download via `apr pull` got correct 942 MB BF16 model - all 290 tensors pass PMAT-235 gates

**Round 51 Progress (2026-02-05) - PMAT-235 COMPILE-TIME CONTRACT ENFORCEMENT:**
| Component | Before | After | Status | Notes |
|-----------|--------|-------|--------|-------|
| Tensor validation | Runtime (bypassable) | **Compile-time (newtypes)** | ✅ **IMPLEMENTED** | Poka-Yoke pattern |
| `ValidatedEmbedding` | Not exist | **Newtype with private fields** | ✅ **NEW** | F-DATA-QUALITY-001/002/003/004 gates |
| `ValidatedWeight` | Not exist | **Newtype with private fields** | ✅ **NEW** | Density + NaN/Inf + L2 gates |
| `ValidatedVector` | Not exist | **Newtype with private fields** | ✅ **NEW** | Shape + content validation |
| `apr qa` Gate 0 | Not exist | **tensor_contract gate** | ✅ **NEW** | Pre-inference contract check |
| `apr validate --quality` | NaN/Inf only | **PMAT-235 rule breakdown** | ✅ **ENHANCED** | Groups by F-DATA-QUALITY-* |
| `apr trace --payload` | No pre-check | **Contract pre-flight** | ✅ **ENHANCED** | Warns before inference |
| `compute_tensor_validation()` | NaN/Inf/zeros | **+Density +L2 +Variation** | ✅ **ENHANCED** | Rule-ID prefixed messages |
| Norm/bias exemption | Not exist | **Constant-value exempt** | ✅ **NEW** | RMS norm init is correct at all-1.0 |
| Contract spec | v1.0 | **v2.0.0** | ✅ **UPDATED** | `tensor-layout-v1.yaml` |
| Toyota Way book | Stub | **Full Jidoka chapter** | ✅ **NEW** | Peer-reviewed citations |

**PMAT-235 Key Insight (Five Whys):**

| Why | Question | Answer |
|-----|----------|--------|
| 1 | Why did PMAT-234 bug reach inference? | Validation was runtime-only, could be bypassed |
| 2 | Why bypassable? | Validation was a separate function call, not enforced by types |
| 3 | Why not types? | Historical Vec<f32> used everywhere without wrapper |
| 4 | Why dangerous? | Invalid data (94.5% zeros) passes all structural checks |
| 5 | Why solution? | **Poka-Yoke: make invalid states unrepresentable at compile time** |

**Theoretical Foundation:**
- Shingo, S. (1986). *Zero Quality Control: Source Inspection and the Poka-Yoke System*. Productivity Press.
- Brady, E. (2017). *Type-Driven Development with Idris*. Manning.
- Parsons, A. (2019). "Parse, Don't Validate" https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/

**Validation Gates (F-DATA-QUALITY):**
| Gate ID | Rule | Threshold | Exempt |
|---------|------|-----------|--------|
| F-DATA-QUALITY-001 | Embedding density | < 50% zeros | - |
| F-DATA-QUALITY-001 | Weight density | < 80% zeros | - |
| F-DATA-QUALITY-002 | No NaN/Inf | count = 0 | - |
| F-DATA-QUALITY-003 | L2 norm | > 1e-6 | - |
| F-DATA-QUALITY-003 | Variation | not constant | Norm/bias tensors |
| F-DATA-QUALITY-004 | Spot check | 10/50/90% non-zero | - |

**Round 50 Progress (2026-02-05) - P0-QA-001 QA SILENT SKIP FIXED:**
| Component | Before | After | Status | Notes |
|-----------|--------|-------|--------|-------|
| `apr qa` SafeTensors gates | SKIP (silent) | **RUN ACTUAL TESTS** | ✅ **FIXED** | Now loads tokenizer.json, runs inference |
| SafeTensors tokenizer loading | Not implemented | `load_from_json()` | ✅ **FIXED** | Looks for tokenizer.json in model dir |
| SafeTensors golden output gate | Skipped | **RUNS** | ✅ **FIXED** | Reveals actual inference issues |
| SafeTensors throughput gate | Skipped | **RUNS** | ✅ **FIXED** | Reveals actual performance |

**P0-QA-001 Root Cause (Five Whys):**

| Why | Question | Answer |
|-----|----------|--------|
| 1 | Why was SafeTensors inference silently passing? | QA gates returned "skipped" instead of running tests |
| 2 | Why skip? | Code assumed no tokenizer available for SafeTensors |
| 3 | Why no tokenizer? | Didn't look for tokenizer.json in model directory |
| 4 | Why not look? | Original implementation was GGUF-first |
| 5 | Why dangerous? | **Silent skips mask real bugs** (Popperian violation) |

**Key Learning:** QA gates that silently skip are DANGEROUS - they hide real bugs. Any skip must be a LOUD failure or explicit configuration.

**Fix Applied:**
1. `crates/apr-cli/src/commands/qa.rs`: SafeTensors branches now load tokenizer.json
2. Uses `aprender::text::bpe::load_from_json()` for HuggingFace tokenizers
3. Runs actual inference via `SafetensorsToAprConverter::convert()` + `generate_with_cache()`
4. Only skips if tokenizer.json truly not found (with clear message)

**Next Steps:**
1. **PMAT-233**: Download Qwen2.5-Coder-0.5B SafeTensors, run `apr qa` E2E verification
2. **GH-5**: GPU throughput fix (FlashAttention tile_kv >= head_dim) - plan ready
3. **PMAT-235 realizaar parity**: Port `ValidatedEmbedding`/`ValidatedWeight` to realizar inference path
4. **Complexity refactor**: Reduce cyclomatic complexity in qa.rs, trace.rs, cbtop.rs (pre-commit gate)

**Round 49 Progress (2026-02-05) - GH-205 F16 PASSTHROUGH FIXED:**
| Component | Before | After | Status | Notes |
|-----------|--------|-------|--------|-------|
| F16 SafeTensors import | 95% diff (precision loss) | **0% diff** | ✅ **FIXED** | Raw bytes preserved |
| F16→F32 conversion | Overflow crash (debug) | **Safe arithmetic** | ✅ **FIXED** | `exp + 112` not `exp - 15 + 127` |
| F16 passthrough tests | 0 | **2 E2E tests** | ✅ **ADDED** | `test_gh205_f16_passthrough_*` |
| Converter tests | 444 | **446** | ✅ **ADDED** | Full F16 coverage |

**GH-205 Root Cause (Five Whys):**

| Why | Question | Answer |
|-----|----------|--------|
| 1 | Why 95% diff in F16 conversion? | F16 values corrupted after round-trip |
| 2 | Why corrupted? | F16→F32→F16 conversion loses precision |
| 3 | Why round-trip? | No F16 passthrough in import pipeline |
| 4 | Why no passthrough? | `get_tensor()` always converts to F32 |
| 5 | Why? | **Historical F32-only pipeline design** |

**Additional Bug Found:** Arithmetic overflow in `safetensors.rs:648`:
```rust
// WRONG: Overflows in debug mode when exp < 15
let exp32 = u32::from(exp) - 15 + 127;
// FIXED: Rearranged to avoid underflow
let exp32 = u32::from(exp) + 112; // 127 - 15 = 112
```

**Fix Applied:**
1. `import.rs`: Added `f16_raw_tensors` field to `SourceLoadResult`
2. `import.rs`: `load_safetensors_with_f16_passthrough()` extracts raw F16 bytes
3. `write.rs`: `write_apr_file()` uses raw F16 bytes when available (line 266-270)
4. `safetensors.rs`: Fixed F16→F32 overflow (line 648)
5. `test_factory.rs`: Added `build_pygmy_safetensors_f16()` for testing
6. `coverage.rs`: Added `test_gh205_f16_passthrough_preserves_bytes` and `test_gh205_f16_passthrough_no_precision_loss`

**Round 48 Progress (2026-02-05) - PMAT-216 GPU PATH FIXED:**
| Component | Before | After | Status | Notes |
|-----------|--------|-------|--------|-------|
| APR GPU inference | garbage | **CORRECT** | ✅ **FIXED** | L2 diff: 0.00% vs CPU |
| GPU/CPU parity test | missing | **MANDATORY** | ✅ **ADDED** | `tests/gpu_cpu_trace_compare.rs` |
| LM head validation | none | **RUNTIME CHECK** | ✅ **ADDED** | Catches swapped arguments |
| Type-safe wrappers | none | `LmHeadWeight`/`LmHeadWeightTransposed` | ✅ **ADDED** | Compile-time protection |

**PMAT-216 Root Cause (Five Whys):**

| Why | Question | Answer |
|-----|----------|--------|
| 1 | Why garbage GPU output? | LM head produces wrong values |
| 2 | Why wrong LM head? | Weight matrix not properly transposed |
| 3 | Why not transposed? | `lm_head_weight_t` contained original data |
| 4 | Why? | Argument order in `from_apr_weights` swapped |
| 5 | Why? | No type safety, no parity test |

**Why Tracing Didn't Catch It (Five Whys):**

| Why | Question | Answer |
|-----|----------|--------|
| 1 | Why didn't tracing catch it? | `forward_traced()` is CPU-only |
| 2 | Why CPU-only? | `GpuModel` never got tracing implemented |
| 3 | Why wasn't it required? | No shared `TracedForward` trait |
| 4 | Why no trait? | CPU/GPU developed independently |
| 5 | Why was divergence allowed? | **No automated parity test in CI** |

**Fix Applied:**
1. `realizar/src/gpu/adapters/apr.rs:180-188` - Fixed argument order
2. `realizar/src/gpu/scheduler/model.rs:174-186` - Added missing RoPE
3. `realizar/src/gpu/scheduler/model.rs:1058-1095` - Runtime transpose validation
4. `realizar/src/gpu/scheduler/types.rs:47-82` - Type-safe `LmHeadWeight`/`LmHeadWeightTransposed`
5. `realizar/tests/gpu_cpu_trace_compare.rs` - **MANDATORY** parity test

**Verification:**
```bash
cargo test --features cuda --test gpu_cpu_trace_compare
# CPU L2: 372.9507, GPU L2: 372.9509, diff: 0.00%
# Argmax match: true
```

**Round 47 Progress (2026-02-05) - GH-208 CPU PATH FIXED:**
| Component | Before | After | Status | Notes |
|-----------|--------|-------|--------|-------|
| APR CPU inference | garbage | **CORRECT** | ✅ **FIXED** | `2+2=` → `+2 equals 4` |
| APR→GGUF correlation | 0.001 | **1.000000** | ✅ **FIXED** | Bit-identical logits |
| APR GPU inference | garbage | **CORRECT** | ✅ **FIXED (Round 48)** | PMAT-216: GPU/CPU parity 0.00% |
| Stale file cleanup | Old files | Deleted | ✅ **FIXED** | Removed `/home/noah/models/qwen2.5-coder-1.5b-q4k.apr` |

**Key Discovery in Round 47:**
The APR CPU path (`AprTransformer.forward()`) works **PERFECTLY**:
- Correlation vs GGUF: **1.000000**
- Correct output for `2+2=`: `+2 equals 4`
- Performance: ~6s for 5 tokens (CPU mode)

The GPU path (`CudaScheduler`/`GpuModel`) had a separate bug (fixed in Round 48/PMAT-216):
- ~~Related to GH-5 (GPU throughput issue)~~ **FIXED**
- ~~Bug location: `realizar/src/gpu/scheduler/model.rs`~~ **FIXED** - swapped lm_head args
- ~~**Workaround:** Use `apr run model.apr --no-gpu` for correct results~~ **No longer needed**

**Critical Learning:** Stale APR files at `/home/noah/models/` (from pre-contract-enforcement era) had WRONG tensor shapes. Always re-import with fresh `apr import` after code changes.

**Round 46 Progress (2026-02-05) - Contract Enforcement:**
| Component | Before | After | Status | Notes |
|-----------|--------|-------|--------|-------|
| Embedding correlation | 0.001 | **1.0** | ✅ **FIXED** | Removed wrong transpose in realizar |
| APR tensor shapes | `[1536, 151936]` | `[151936, 1536]` | ✅ **FIXED** | Contract enforcement now mandatory |
| Contract enforcement | "suggestion" | **MANDATORY** | ✅ **FIXED** | Fail-fast with assertions |

**Key Fixes in Round 46:**
1. **Contract Enforcement**: `enforce_import_contract()` is now MANDATORY (Five Whys analysis)
2. **Embedding Transpose**: Removed WRONG transpose in `realizar/src/apr_transformer/mod.rs`
3. **Double Shape Reversal**: Fixed in `write_apr_file_raw()` - was reversing already-reversed shapes
4. **19 New Tests**: Contract enforcement tests with `should_panic` for violations

**Round 48 Benchmark Results (2026-02-05) - ALL VERIFIED:**
| System | Claimed | Actual | Status | Notes |
|--------|---------|--------|--------|-------|
| GGUF GPU (1.5B Q4K) | 285.5 tok/s | 21.6 tok/s | ✅ | Correct for 1.5B model (spec used 0.5B) |
| APR CPU | N/A | ~0.8 tok/s | ✅ **FIXED** | Correct output ([#208](https://github.com/paiml/aprender/issues/208)) |
| APR GPU | 250 tok/s | ~20 tok/s | ✅ **FIXED** | PMAT-216: GPU/CPU parity 0.00% |
| Rosetta conversion | F32 default | Q4K default | ✅ **FIXED** | [#205](https://github.com/paiml/aprender/issues/205) |

**Previous Fixes (Round 45):**
1. **#205 FIXED**: Rosetta SafeTensors→GGUF now defaults to Q4K (F32 was incompatible with realizar)
2. **#207 CORRECTED**: GGUF GPU path IS working (21.6 tok/s for 1.5B) - issue description was incorrect
3. **#208 RE-SCOPED**: APR format parsing works, but inference produces garbage (different root cause)

**Performance Gap Root Cause Analysis:**
| Factor | APR | Ollama/llama.cpp | Impact |
|--------|-----|------------------|--------|
| Kernel launches/decode | ~100+ | ~30 | 3.3x overhead |
| FFN implementation | Separate kernels | Megakernel fusion | 15.8us overhead |
| KV cache | ✅ Incremental (O(n)) | ✅ Incremental (O(n)) | Parity |
| Attention | FlashAttention (fixed) | FlashDecoding | Similar |

**Closing the Gap (Future Work):**
1. **Megakernel fusion** - Combine FFN kernels (up+gate+SiLU+down) into single kernel
2. **Reduce kernel launches** - Batch small operations, fuse layer norm + projection
3. **Persistent kernels** - Keep kernels loaded between decode steps

**QA Gates (apr qa - BUG-QA-001/002 fixed):**
- ✅ Golden Output: 2/2 test cases
- ✅ Throughput: 282 tok/s (pass ≥100)
- ✅ Ollama Parity: 0.6x (259 vs 419 tok/s) — now uses correct model size and eval_duration
- ✅ GPU Speedup: 93x CPU→GPU

**GH-201 Fix (Layer Streaming Mode):** Both SafeTensors AND APR GPU paths now support two modes:

| Component | File | Pre-Cache Method | Fix Applied |
|-----------|------|------------------|-------------|
| SafeTensors CUDA | `safetensors_cuda.rs` | `upload_weights()` | ✅ Layer streaming |
| APR CUDA | `apr/cuda.rs` | `pre_cache_weights()` | ✅ Layer streaming |
| GGUF CUDA | `gguf/inference/` | `DequantizedWeightCache` | Already streams |

**Modes:**
1. **Full Cache Mode** (default when VRAM sufficient): Pre-cache all weights for maximum throughput
2. **Layer Streaming Mode** (automatic when VRAM insufficient): Stream layer weights on-demand

**Memory Architecture:**
```
Full Cache Mode (~6GB for 1.5B):    Layer Streaming Mode (~1.5GB for 1.5B):
┌──────────────────────────────┐    ┌──────────────────────────────┐
│ Embedding (CPU)              │    │ Embedding (CPU)              │
│ LM Head (GPU: ~900MB)        │    │ LM Head (GPU: ~900MB)        │
│ Layer 0 (GPU: ~187MB)        │    │ Layer Buffer (GPU: ~200MB)   │ ← Reused
│ Layer 1 (GPU: ~187MB)        │    │   ↑ Upload layer N           │
│ ...                          │    │   ↓ Forward                  │
│ Layer 27 (GPU: ~187MB)       │    │   → Reuse for layer N+1      │
│ KV Cache (GPU: ~57MB)        │    │ KV Cache (GPU: ~57MB)        │
└──────────────────────────────┘    └──────────────────────────────┘
```

**Shared Infrastructure:** `realizar/src/cuda/streaming.rs`
- `StreamingConfig` - Model config for VRAM estimation (hidden_dim, num_layers, etc.)
- `StreamingConfig::estimate_full_cache_vram()` - Calculate full cache VRAM requirement
- `StreamingConfig::estimate_streaming_vram()` - Calculate streaming mode VRAM requirement
- `StreamingConfig::estimate_layer_vram()` - Calculate single layer VRAM requirement
- `should_use_streaming(free_vram, config)` - Check if streaming mode needed
- `check_vram_sufficient(free_vram, total_vram, config)` - Auto-select mode with error handling
- `StreamingMode` - Enum: `FullCache` or `LayerStreaming`

**Implementation Files:**
| File | Method | Streaming Support |
|------|--------|-------------------|
| `safetensors_cuda.rs` | `upload_weights_streaming()` | ✅ Loads LM head + norms only |
| `safetensors_cuda.rs` | `ensure_layer_weights_loaded()` | ✅ On-demand layer upload via mmap |
| `apr/cuda.rs` | `pre_cache_weights_streaming()` | ✅ Loads LM head + norms only |
| `apr/cuda.rs` | `ensure_layer_weights_loaded()` | ✅ On-demand layer upload from model |

**Oracle Pattern Source:** `realizar/src/apr_transformer/loader.rs` (MmapAprTransformer), `realizar/src/gguf/inference/cached/sync.rs` (DequantizedWeightCache)

**APR GPU Inference Path (PMAT-APR-PERF-001):**

The APR GPU inference path in `realizar/src/infer/mod.rs:try_apr_cuda_inference()` now uses:

```rust
// 1. Load APR transformer
let transformer = AprTransformer::from_apr_file(&config.model_path)?;

// 2. Convert to GpuModel (has KV cache support)
let mut gpu_model = AprF32ToGpuAdapter::to_gpu_model(&transformer)?;

// 3. Generate with internal KV cache (O(n) incremental decoding)
let tokens = gpu_model.generate_with_cache(&prompt, &gen_config)?;
```

**Key Components:**
| Component | Location | Purpose |
|-----------|----------|---------|
| `AprF32ToGpuAdapter::to_gpu_model()` | `realizar/src/gpu/adapters/apr.rs` | APR → GpuModel conversion |
| `GpuModel::generate_with_cache()` | `realizar/src/gpu/scheduler/model.rs` | Incremental KV cache generation |
| `StreamingKVCache` | `realizar/src/gpu/streaming_kv.rs` | Internal KV cache (created by generate_with_cache) |

**GH-5 FlashAttention Fix (trueno-gpu):**

The FlashAttention kernel in `trueno-gpu/src/kernels/attention/flash.rs` ensures `tile_kv >= head_dim` to prevent shared memory overflow when processing models with `head_dim > 64`:

```rust
// GH-5 FIX: Ensure tile_kv >= head_dim to prevent shared memory overflow
let tile_kv = seq_len.min(64).max(head_dim);
```

Without this fix, models with `hidden_dim >= 1536` (Qwen 1.5B+) would cause shared memory overflow and produce garbage output.

**PMAT-216 GPU/CPU Parity Mandate:**

**ALL inference backends MUST match the reference implementation (CPU AprTransformer).**

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| Parity test | `tests/gpu_cpu_trace_compare.rs` | ✅ MANDATORY |
| Type safety | `LmHeadWeight`/`LmHeadWeightTransposed` newtypes | ✅ ADDED |
| Runtime validation | `from_apr_weights()` checks transpose | ✅ ADDED |
| Documentation | `realizar/CLAUDE.md` GPU Parity section | ✅ ADDED |
| **TracedForward trait** | `apr_transformer::TracedForward` | ✅ **ENFORCED** |
| GPU tracing | `GpuModel::forward_traced_gpu()` | ✅ ADDED |

**TracedForward Trait (PMAT-216):**

All inference backends MUST implement this trait:
```rust
pub trait TracedForward {
    fn forward_traced(&mut self, tokens: &[u32]) -> Result<ForwardTrace>;
}

// Both backends implement it:
impl TracedForward for AprTransformer { ... }  // CPU
impl TracedForward for GpuModel { ... }        // GPU
```

**Mandatory Verification for ANY New Backend:**
```rust
use realizar::apr_transformer::TracedForward;

// Use trait-based API for both backends:
let cpu_trace = TracedForward::forward_traced(&mut apr_model, &tokens)?;
let gpu_trace = TracedForward::forward_traced(&mut gpu_model, &tokens)?;

let cpu_l2 = cpu_trace.logits.iter().map(|x| x * x).sum::<f32>().sqrt();
let gpu_l2 = gpu_trace.logits.iter().map(|x| x * x).sum::<f32>().sqrt();
let diff_pct = ((cpu_l2 - gpu_l2).abs() / cpu_l2) * 100.0;

assert!(diff_pct < 0.01, "Backend diverged {:.2}% from CPU!", diff_pct);
```

**CI Enforcement:**
```bash
# This test is MANDATORY in CI pipeline
cargo test --features cuda --test gpu_cpu_trace_compare
```

---
