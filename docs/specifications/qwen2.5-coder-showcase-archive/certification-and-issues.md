# Certification Results & GitHub Issues

> Archived from qwen2.5-coder-showcase-demo.md (lines 429-604)

## Certification Results (Round 39)

**Qwen2.5-Coder-0.5B-Instruct:**
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Tests Passed | 19/32 | 32/32 | ❌ **BLOCKED** |
| Pass Rate | 59.4% | 100% | ❌ |
| MQS Score | 415/1000 | 800/1000 | ❌ |
| Grade | F | A | ❌ |

**Streaming Tests (realizar):**
| Test | Status |
|------|--------|
| `test_full_cache_vram_qwen2_1_5b` | ✅ PASS |
| `test_streaming_vram_much_smaller` | ✅ PASS |
| `test_streaming_vram_includes_lm_head_and_kv` | ✅ PASS |
| `test_layer_vram_estimate` | ✅ PASS |
| `test_should_use_streaming_small_vram` | ✅ PASS |
| `test_should_use_streaming_large_vram` | ✅ PASS |
| `test_check_vram_sufficient_full_cache` | ✅ PASS |
| `test_check_vram_sufficient_streaming` | ✅ PASS |
| `test_check_vram_insufficient` | ✅ PASS |
| `test_streaming_mode_description` | ✅ PASS |

**GH-201 Implementation Status:** ✅ COMPLETE (10/10 streaming tests pass)

**PMAT SATD Analysis:**
| Metric | Before | After | Status |
|--------|--------|-------|--------|
| Real SATD | 1 (PMAT-XXX) | 0 | ✅ Fixed → PMAT-230 |
| False Positives | 5 | 5 | ⚠️ Tracked in [pmat#144](https://github.com/paiml/paiml-mcp-agent-toolkit/issues/144) |

**False Positive Categories (pmat#144):**
- Section headers with `===` separators
- Documentation describing phone formats (`XXX-XXX-XXXX`)
- Comments mentioning security topics ("XSS/Injection mitigation")
- Mathematical notation (`s^T × temp`)

**Certification Failures Root Cause (13/32 failures):**

All 13 failures were conversion tests (`F-CONV-*`) with the same error:
```
Invalid model file extension: '.'. Expected one of: gguf, safetensors, apr, bin
```

**Root Cause:** `ConversionTest::execute()` in `apr-model-qa-playbook` received directory paths
but passed them directly to `apr run` without resolving to format-specific files.

**Fix Applied:** Added `resolve_format_path()` method to `conversion.rs` that:
1. Handles file mode (direct path with extension check)
2. Handles directory mode (looks in `<dir>/<format>/model.<ext>` or any matching file)

**PR:** `apr-model-qa-playbook` - conversion.rs path resolution fix

---

## GitHub Issues Status (Toyota Way: Transparency)

**Summary:** 🛑 **METHODOLOGY BLOCKER** - Round 23: All code P0s resolved, but QA used pre-baked GGUF models. Retest required with self-converted models only.

| Issue | Title | Severity | Status | PMAT |
|-------|-------|----------|--------|------|
| [#201](https://github.com/paiml/aprender/issues/201) | **SafeTensors/APR GPU OOM: pre-caches 6GB upfront** | **P1** | ✅ **FIXED** | GH-201 |
| [#198](https://github.com/paiml/aprender/issues/198) | **apr pull: SafeTensors missing tokenizer.json, config.json** | **P0** | ✅ **FIXED** | PMAT-195 |
| [#197](https://github.com/paiml/aprender/issues/197) | **SafeTensors inference garbage: layer misdetection** | **P0** | ✅ **FIXED** | GH-197 |
| [#196](https://github.com/paiml/aprender/issues/196) | **Conversion pipeline: 4 defects blocking MVP** | **P0** | ✅ **FIXED** | PMAT-197 |
| **FIXED** | **GGUF→APR Shape Convention (GGML layout)** | **P0** | ✅ **FIXED** | PMAT-222 |
| **FIXED** | **Quantized GEMM Dispatch (CUDA)** | **P0** | ✅ **FIXED** | PMAT-222 |
| **FIXED** | **F32 Weight Transpose (SafeTensors)** | **P0** | ✅ **FIXED** | PMAT-222 |
| [#194](https://github.com/paiml/aprender/issues/194) | **Conversion: --preserve-q4k fails** | **P0** | ✅ **FIXED** | PMAT-210 |
| [#192](https://github.com/paiml/aprender/issues/192) | **APR Import Drops Tensors** | **P0** | ✅ **FIXED** | PMAT-209 |

**Benchmark Results (2026-02-01 - Round 17):**
| Format | Throughput | Output Quality | Notes |
|--------|------------|----------------|-------|
| ~~GGUF Q4K~~ | ~~266.4 tok/s~~ | ~~✅ Correct~~ | ❌ **INVALIDATED** — pre-baked HF GGUF |
| SafeTensors | 19.4 tok/s | ✅ Correct | SafeTensors F32 baseline |
| APR (from ST) | **19.4 tok/s** | ✅ Correct | Identical to ST source |
| ~~APR (from GGUF)~~ | ~~265.8 tok/s~~ | ~~✅ Correct~~ | ❌ **INVALIDATED** — source was pre-baked |

**⚠️ Round 15 AND Round 22 Comparison INVALID:** Both rounds used pre-baked GGUF (Q4_K_M) from HuggingFace instead of self-converted GGUF. This violates Section 0 methodology. See **Section 30** for full audit.

**Correct Approach (Section 0, enforced from Round 23):**
1. `apr pull` SafeTensors from HuggingFace (ground truth)
2. Convert SafeTensors → APR (`apr import`)
3. Convert SafeTensors → GGUF (`apr export --format gguf`)
4. Run inference on all three — must match
5. **NO pre-baked GGUF from HuggingFace. EVER.**

**Previously Fixed Issues:**
| Issue | Description | Priority | Status | PMAT |
|-------|-------------|----------|--------|------|
| BUG-QA-001 | apr qa compared 0.5B APR vs 1.5B Ollama (unfair) | P1 | ✅ FIXED | 2026-02-03 |
| BUG-QA-002 | apr qa used wall clock time instead of eval_duration | P0 | ✅ FIXED | 2026-02-03 |
| BUG-SHOWCASE-001 | APR inference used wrong loader (binary vs JSON format) | P0 | ✅ FIXED | 2026-02-03 |
| BUG-SHOWCASE-002 | APR inference hardcoded to 32b model path | P1 | ✅ FIXED | 2026-02-03 |
| BUG-EXPORT-001 | Export infer_model_config confused hidden_size with vocab_size | P1 | ✅ FIXED | 2026-02-04 |
| BUG-LINT-001 | Lint flagged valid GGUF tensor names (blk.N.) as non-standard | P2 | ✅ FIXED | 2026-02-04 |
| BUG-TRACE-001 | Trace command showed Parameters: 0 instead of actual count | P2 | ✅ FIXED | 2026-02-04 |
| BUG-DEBUG-001 | Debug command showed INVALID/CORRUPTED for valid GGUF files | P2 | ✅ FIXED | 2026-02-04 |
| BUG-PROBAR-001 | Probar showed "Format: Unknown" for GGUF files | P3 | ✅ FIXED | 2026-02-04 |
| BUG-RUN-001 | Benchmark token count uses word approximation instead of actual count | P1 | ✅ FIXED | PMAT-203 |
| BUG-EXPORT-002 | Export to GGUF doesn't transpose data (LAYOUT-002 violation on export) | P0 | ✅ FIXED | 2026-02-04 |
| BUG-CONV-001 | Legacy quant (Q4_0/Q4_1/Q5_0/Q8_0) and F32/F16 import doesn't transpose | P0 | ✅ FIXED | 2026-02-04 |
| BUG-PATH-001 | "No file extension found" unhelpful when directory passed | P2 | ✅ FIXED | 2026-02-04 |
| BUG-MERGE-001 | Merge command missing weight validation (count, negative, NaN) | P1 | ✅ FIXED | 2026-02-04 |
| BUG-MERGE-002 | Merge --weights silently ignored for non-weighted strategies | P2 | ✅ FIXED | 2026-02-04 |
| BUG-VALIDATE-001 | Validate --min-score accepts values > 100 | P2 | ✅ FIXED | 2026-02-04 |
| BUG-IMPORT-001 | Import --preserve-q4k silently ignored without inference feature | P2 | ✅ FIXED | 2026-02-04 |
| BUG-TOK-001 | LlamaTokenizer byte tokens >= 128 decoded as wrong Unicode chars | P1 | ✅ FIXED | 2026-02-04 |
| BUG-EXPORT-003 | Export report tensor_count stale after unfuse/remove operations | P2 | ✅ FIXED | 2026-02-04 |
| BUG-TRACE-002 | Trace error message missing GGUF from valid formats list | P3 | ✅ FIXED | 2026-02-04 |
| BUG-INSPECT-001 | Inspect shows "Legacy APR format" for GGUF files (misleading) | P3 | ✅ FIXED | 2026-02-04 |
| BUG-MERGE-006 | calculate_merge_weights accepts NaN/Inf (NaN <= 0 is false) | P1 | ✅ FIXED | 2026-02-04 |
| BUG-TRACE-003 | APR trace hardcodes total_params=0 (BUG-TRACE-001 fix incomplete) | P2 | ✅ FIXED | 2026-02-04 |
| BUG-GGUF-001 | GGUF reader allocates Vec without validating count (OOM attack vector) | P0 | ✅ FIXED | 2026-02-04 |
| BUG-LAYOUT-003 | GGUF→APR error paths bypass LAYOUT-002 transpose (corrupt output) | P0 | ✅ FIXED | 2026-02-04 |
| BUG-GGUF-002 | GGUF reader shape.iter().product() integer overflow (security) | P0 | ✅ FIXED | 2026-02-04 |
| GH-202 | diff-tensors/fingerprint cross-format tensor name mismatch | P1 | ✅ FIXED | 2026-02-04 |
| BUG-TOK-002 | Tokenizer not found for Pacha cache layout ({hash}.tokenizer.json) | P0 | ✅ FIXED | 2026-02-04 |
| BUG-APR-GPU-001 | APR GPU inference used wrong API (3 args vs 2, wrong field names) | P1 | ✅ FIXED | 2026-02-05 |
| GH-5 | FlashAttention shared memory overflow when tile_kv < head_dim | P0 | ✅ FIXED | 2026-02-05 |
| BUG-F16-001 | F16 SafeTensors→APR 95% diff (F16→F32→F16 precision loss) | P0 | ✅ FIXED | 2026-02-05 |
| BUG-F16-002 | F16→F32 conversion overflow (`exp - 15` underflows when exp < 15) | P1 | ✅ FIXED | 2026-02-05 |
| GH-191 | APR dtype byte mapping mismatch | P0 | ✅ FIXED | PMAT-223 |
| GH-190 | GGUF→APR tensor name mismatch | P0 | ✅ FIXED | PMAT-205 |
| GH-189 | APR chat special tokens not atomic | P0 | ✅ FIXED | PMAT-206 |
| [#188](https://github.com/paiml/aprender/issues/188) | Rosetta differential tracing | P1 | ✅ FIXED | PMAT-200 |
| [#186](https://github.com/paiml/aprender/issues/186) | APR Q4_K PAD token garbage | P0 | ✅ FIXED | PMAT-196 |
| [#185](https://github.com/paiml/aprender/issues/185) | APR missing embedded tokenizer | P0 | ✅ FIXED | PMAT-195 |

**Last Updated:** 2026-02-05 (Round 44 - PMAT-APR-PERF-001: APR GPU KV cache integration)

**Round 44 Summary (2026-02-05):**
- Fixed APR GPU inference path in `realizar/src/infer/mod.rs` to use `GpuModel.generate_with_cache()`
- KV cache now managed internally by `generate_with_cache()` for incremental O(n) decoding
- Fixed API mismatches: `StreamingKVCache` import path, `context_length` field, `GpuGenerateConfig` fields
- GH-5 FlashAttention fix verified in `trueno-gpu/src/kernels/attention/flash.rs` (tile_kv >= head_dim)
- PMAT-232: External tokenizer support documented (weights-only GGUF import requires `--tokenizer`)
- Filed [paiml-mcp-agent-toolkit#150](https://github.com/paiml/paiml-mcp-agent-toolkit/issues/150): pmat query struct/type search
- All 10,333 aprender tests pass, 29/29 realizar `generate_with_cache` tests pass

| Fix | File | Description |
|-----|------|-------------|
| PMAT-APR-PERF-001 | `realizar/src/infer/mod.rs` | Use `GpuModel.generate_with_cache()` with internal KV cache |
| GH-5 | `trueno-gpu/src/kernels/attention/flash.rs` | `tile_kv >= head_dim` prevents shared memory overflow |
| PMAT-232 | `aprender/src/format/converter/import.rs` | External tokenizer via `--tokenizer` for weights-only GGUF |

**Round 43 Summary (2026-02-04):**
- BUG-TOK-002 Fix: APR tokenizer path resolution for Pacha cache

**Round 42 Summary (2026-02-04):**
- Implemented `batuta bug-hunter` subcommand with 5 hunting modes (FDV, SBEST, LLIFT, FourFuzz, COTTONTAIL)
- Added 10 new checklist items (BH-01 to BH-10) to popperian-falsification-checklist.md
- PMAT work cleanup: 18 tickets verified and marked done
- Remaining in-progress: 10 items (mostly performance optimization)

| Mode | Pattern | Description |
|------|---------|-------------|
| falsify | FDV | Mutation-based invariant falsification |
| hunt | SBEST | SBFL from stack traces/coverage |
| analyze | LLIFT | LLM-augmented static analysis |
| fuzz | FourFuzz | Targeted unsafe Rust fuzzing |
| deep-hunt | COTTONTAIL | Hybrid concolic + SBFL |
| ensemble | — | Run all modes combined |

**Previous:** Round 41 - GH-202 cross-format tensor name normalization in rosetta diff-tensors/fingerprint

**APR Format Note:** Two APR variants exist:
1. **realizar JSON-APR** - JSON tensor index, used by `GgufToAprConverter` for showcase
2. **aprender APR v2** - Binary tensor index, used by rosetta/format tools

The showcase pipeline uses realizar's JSON-APR format for GGUF→APR conversion. Rosetta inspect expects APR v2 binary format. Cross-format tools should detect and handle both.

---

