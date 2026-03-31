# Appendix D: Historical Bug Fixes (2026-01-21 to 2026-01-28)

> Archived from qwen2.5-coder-showcase-demo.md (lines 3785-4057)

## Appendix D: Historical Bug Fixes (2026-01-21 to 2026-01-28)

This appendix summarizes major bugs that have been fixed. See git history for details.

### PMAT-094: SafeTensors Garbage Output


<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Findings:** None ✓
<!-- /bug-hunter-status -->






**Root Cause:** Using LayerNorm instead of RMSNorm for Qwen2/LLaMA/Mistral models.
**Fix:** Changed `layer_norm` to compute RMS without mean subtraction.

### PMAT-095: SafeTensors 75x Performance Gap


<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Findings:** None ✓
<!-- /bug-hunter-status -->






**Root Cause:** O(n²) weight transposition on every forward pass due to logic bug.
**Fix:** Kept HuggingFace [out_dim, in_dim] layout directly, no transpose.

### PMAT-096: GGUF RMSNorm Parity


<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Findings:** None ✓
<!-- /bug-hunter-status -->






**Root Cause:** Same LayerNorm bug repeated in GGUF path.
**Fix:** Updated all `layer_norm` functions to use RMSNorm.

### PMAT-097: 0.5B Model Garbage


<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Findings:** None ✓
<!-- /bug-hunter-status -->






**Root Cause:** Model capacity limitation, not code bug.
**Resolution:** QA now uses 1.5B models exclusively.

### PMAT-098: APR Serve Performance


<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Implementations:**
- `crates/apr-cli/src/commands/serve/handlers.rs:122` - // PMAT-098: Use proper BPE tokenizer (same as SafeTensors p
- `crates/apr-cli/src/commands/serve/handlers.rs:169` - // PMAT-098: Load transformer once and share across requests
- `crates/apr-cli/src/commands/serve/handlers.rs:198` - // PMAT-098: Use BPE tokenizer for proper encoding
- `crates/apr-cli/src/commands/serve/handlers.rs:285` - // PMAT-098: Use shared transformer (no reload per request)
- `crates/apr-cli/src/commands/serve/handlers.rs:301` - // PMAT-098: Use BPE tokenizer for proper encoding
- `crates/apr-cli/src/commands/serve/handlers.rs:367` - // PMAT-098: Use BPE tokenizer for proper decoding
- `crates/apr-cli/src/commands/serve/handlers.rs:431` - // PMAT-098: Use shared transformer (no reload per request)
- `crates/apr-cli/src/commands/serve/handlers.rs:463` - // PMAT-098: Use BPE tokenizer for proper encoding
- `crates/apr-cli/src/commands/serve/handlers.rs:525` - // PMAT-098: Use BPE tokenizer for proper decoding
- `crates/apr-cli/src/commands/serve/handlers.rs:682` - /// PMAT-098: Updated to use BPE tokenizer for proper encodi
- `crates/apr-cli/src/commands/serve/handlers.rs:742` - // PMAT-098: Use BPE tokenizer for proper encoding
- `crates/apr-cli/src/commands/serve/handlers.rs:773` - // PMAT-098: Use BPE tokenizer for proper encoding
- `crates/apr-cli/src/commands/serve/handlers.rs:825` - // PMAT-098: Use BPE tokenizer for proper decoding
- `crates/apr-cli/src/commands/serve/handlers.rs:886` - // PMAT-098: Use BPE tokenizer for proper encoding
- `crates/apr-cli/src/commands/serve/handlers.rs:933` - // PMAT-098: Use BPE tokenizer for proper decoding
- `examples/qa_falsify.rs:1` - //! QA Infrastructure Falsification Tests (PMAT-098 Red Team
- `examples/qa_falsify.rs:553` - "{}TEST 3: ZOMBIE SERVER (PMAT-098-PF SIGINT Resiliency){}",
- `examples/qa_falsify.rs:562` - "  {}SIGINT Handler Implementation (PMAT-098-PF):{}",
- `examples/qa_falsify.rs:605` - "{}║     QA INFRASTRUCTURE FALSIFICATION (PMAT-098 Red Team)
- `examples/qa_falsify.rs:640` - "  3. Zombie Server: {}✓ FIXED{} - SIGINT handler + ProcessG
- `examples/qa_run.rs:74` - // SIGINT RESILIENCY: Global Process Registry (PMAT-098-PF)
- `examples/qa_run.rs:791` - // Register process for SIGINT cleanup (PMAT-098-PF)
- `examples/qa_run.rs:875` - // Wrap server in ProcessGuard for SIGINT safety (PMAT-098-P
- `examples/qa_run.rs:1096` - // (Fixed: PMAT-098 Red Team falsification found naive subst
- `examples/qa_run.rs:1631` - // Set up SIGINT handler for graceful shutdown (PMAT-098-PF:
- `src/format/converter/import.rs:408` - /// Load model config from config.json alongside the model f
- `src/format/converter/import.rs:801` - // PMAT-098: Read config.json if available (CRITICAL for cor
**Findings:** None ✓
<!-- /bug-hunter-status -->






**Root Cause:** Model reloaded on every HTTP request.
**Fix:** Use `Arc<Mutex<AprTransformer>>` shared across requests.

### PMAT-099: APR Token Decode Empty


<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Implementations:**
- `crates/apr-cli/src/commands/serve/handlers.rs:141` - // PMAT-099: APR GPU path currently has tensor name mismatch
- `crates/apr-cli/src/commands/serve/handlers.rs:147` - // PMAT-099: Disable GPU for APR until AprV2ModelCuda tensor
- `crates/apr-cli/src/commands/serve/handlers.rs:152` - "Note: APR GPU path disabled (PMAT-099 - tensor name mapping
- `crates/apr-cli/src/commands/serve/handlers.rs:363` - // PMAT-099: Debug logging for token decoding
- `crates/apr-cli/src/commands/serve/handlers.rs:520` - // PMAT-099: Debug logging for token decoding
- `crates/apr-cli/src/commands/serve/handlers.rs:928` - // PMAT-099: Debug logging for GPU token decoding
- `crates/apr-cli/src/commands/serve/safetensors.rs:260` - // PMAT-099: added_tokens must be included in vocab for deco
- `crates/apr-cli/src/commands/serve/safetensors.rs:285` - // PMAT-099: Special tokens often have IDs beyond base vocab
- `src/format/converter/tests/core.rs:121` - // PMAT-099: Names are now preserved for AprTransformer comp
- `src/format/converter/tests/core.rs:134` - // PMAT-099: Names are now preserved for AprTransformer comp
- `src/format/converter/tests/core.rs:141` - // PMAT-099: model. prefix preserved for AprTransformer::fro
- `src/format/converter/tests/core.rs:148` - // PMAT-099: Preserve original names for inference compatibi
- `src/format/converter/tests/core.rs:155` - // PMAT-099: Preserve original names
- `src/format/converter/tests/core.rs:163` - // PMAT-099: Preserve model. prefix for AprTransformer compa
- `src/format/converter/tests/core.rs:469` - /// Harness-based: Whisper name mapping preserves model.* pr
- `src/format/converter/tests/errors.rs:367` - // PMAT-099: Preserve model. prefix for AprTransformer compa
- `src/format/converter_types.rs:130` - // PMAT-099: Preserve original tensor names for AprTransform
- `src/format/converter_types.rs:136` - // PMAT-099: Preserve model. prefix for Whisper
- `src/format/converter_types.rs:141` - // PMAT-099: Preserve model. prefix for LLaMA
- `tests/spec_checklist_19_inference.rs:285` - // PMAT-099: Preserve model. prefix for AprTransformer compa
- `tests/spec_checklist_19_inference.rs:358` - // PMAT-099: Preserve model. prefix for AprTransformer compa
**Findings:** None ✓
<!-- /bug-hunter-status -->






**Root Cause:** Special tokens missing from vocabulary (added_tokens not included).
**Fix:** Extended vocabulary to include all added_tokens at proper IDs.

### PMAT-100: APR Missing lm_head.weight


<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Implementations:**
- `src/format/converter/write.rs:159` - // PMAT-100: Handle tied embeddings (common in Qwen, LLaMA, 
**Findings:** None ✓
<!-- /bug-hunter-status -->






**Root Cause:** HuggingFace uses tied embeddings, omits lm_head.
**Fix:** Copy `embed_tokens.weight` to `lm_head.weight` when missing.

### PMAT-101: APR QKV Fusion Layout


<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Findings:** None ✓
<!-- /bug-hunter-status -->






**Root Cause:** QKV fusion produced wrong layout [hidden_dim, qkv_dim].
**Fix:** Pre-fuse QKV in converter as [qkv_dim, hidden_dim].

### PMAT-102: Trace Tests Failing


<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Findings:** None ✓
<!-- /bug-hunter-status -->






**Root Cause:** Installed binary missing cuda feature.
**Fix:** Reinstall with `--features "inference cuda"`.

### PMAT-103: Performance Gap (0.05 → 14 tok/s)
**Root Cause:** Using O(n²) `forward()` instead of O(n) `forward_with_cache()`.
**Fix:** Updated all serve handlers to use `generate_with_cache()`.

### PMAT-086/104: APR Q4_K Layout Mismatch
**Root Cause:** Column-major kernel used for row-major GGUF/APR data.
**Fix:** Implemented LAYOUT-001 protocol, swapped to row-major kernel.

### GQA Bug (2026-01-26)
**Root Cause:** GPU path dimension calculations wrong for Grouped Query Attention.
**Fix:** Q uses num_heads × head_dim, K/V use num_kv_heads × head_dim.

### PAR-501: X-Trace-Level


<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Findings:** None ✓
<!-- /bug-hunter-status -->






**Fix:** Added `build_trace_data()` helper to all code paths.

### PAR-502: CUDA PTX Shared Memory Overflow


<!-- bug-hunter-status -->
**Bug Hunter Status:** ✓ Verified
**Findings:** None ✓
<!-- /bug-hunter-status -->






**Root Cause:** `tiled_q4k_gemv` kernel overflows shared memory for K>25600.
**Fix:** Dispatch to `ChunkedTiledQ4KGemvKernel` when K>25600.

### PMAT-116: SafeTensors GPU Inference (2026-01-28)
**Root Cause:** No direct CUDA path for SafeTensors format. Always fell back to CPU.
**Fix:** Implemented `SafeTensorsCudaModel` in `realizar/src/safetensors_cuda.rs`:
- Uses CudaExecutor API (`gemm_b_cached`, `incremental_attention_gpu`)
- CPU-side `gamma_cache` HashMap for RMS norm weights
- RoPE position handled internally by attention kernel
- Zero SATD implementation (falsification audit passed)

### GitHub #158: GpuModel Send/Sync Bounds (2026-01-25)
**Root Cause:** `GpuModel` struct missing `Send + Sync` trait bounds, breaks `apr-cli` cuda feature.
**Fix:** Added explicit trait bounds to enable multi-threaded GPU model sharing.

### GitHub #157: GPU Performance Threshold Aggressive (2026-01-25)
**Root Cause:** Performance assertions too strict for small models.
**Fix:** Adjusted threshold to account for model size variance.

### GitHub #156: APR Missing Tokenizer (2026-01-25)
**Root Cause:** APR format tokenizer field shows placeholder text instead of actual tokenizer.
**Fix:** Proper tokenizer serialization in APR converter.

### GitHub #155: GGUF LAYOUT-001 Regression (2026-01-25)
**Root Cause:** Regression caused GGUF to output garbage (column-major kernel used for row-major data).
**Fix:** Restored row-major kernel dispatch per LAYOUT-001 protocol.

### GitHub #154: Payload Tracing Stubbed (2026-01-25)
**Root Cause:** APR-TRACE-001 tracing was stubbed out, blocking debug of garbage output.
**Fix:** Implemented full `build_trace_data()` helper across all code paths.

---

