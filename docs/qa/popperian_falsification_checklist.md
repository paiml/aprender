# 100-Point Popperian Falsification Checklist
## For the Qwen2.5-Coder Showcase & Unified Inference Architecture

**"Our method of research is not to defend them, in order to prove us right, but, on the contrary, to try to overthrow them." — Karl Popper, *The Logic of Scientific Discovery***

This checklist is NOT designed to confirm that the software works. It is designed to aggressively attempt to prove that it *does not* work. If the software survives these 100 severe tests, it is considered **corroborated** (not verified).

---

## Latest Run Summary

| Section | Score | Status |
|---------|-------|--------|
| I. Metaphysical Baseline | 8/10 | ⚠️ Binary size exceeds limit |
| II. Loader Gauntlet | 5/15 | ⚠️ Partial testing |
| III. Output Quality | 2/15 | ⚠️ Core 2+2 tests pass |
| IV. Performance | 2/15 | ⚠️ Partial testing |
| V. Rosetta Conversion | 1/10 | ⚠️ Partial testing |
| VI. Jidoka & Safety | 0/15 | ❌ cargo deny fails |
| VII. Observability | 1/10 | ⚠️ Trace works |
| VIII. T-Series | 0/10 | ⏳ Not tested |
| **TOTAL** | **19/100** | ❌ **FALSIFIED** |

**Last Updated:** 2026-01-27 (PMAT-112)
**Verdict:** Release criteria NOT met. Continue testing.

---

### I. The Metaphysical Baseline (Existence & Hygiene) [10 Points]
*Tests if the artifacts exist and are observable.*

**Run Date:** 2026-01-27 | **Score: 8/10**

- [x] **F-META-001**: `apr --version` returns a valid semantic version string (not empty, not error). ✅ `apr 0.2.12`
- [x] **F-META-002**: `apr --help` returns valid help text with all subcommands (`run`, `chat`, `serve`, `check`, `convert`). ✅
- [x] **F-META-003**: `apr run --help` shows model path argument and generation flags. ✅
- [ ] **F-META-004**: Binary size is within expected bounds (< 50MB release, < 200MB debug). ❌ **290MB** (debug symbols)
- [x] **F-META-005**: Execution on a clean environment (no `.env`, no cache) fails gracefully with clear instructions, not a panic. ✅
- [x] **F-META-006**: `apr run` with a non-existent file path returns "File not found" error code (not panic). ✅ Exit 3
- [x] **F-META-007**: `apr run` with a directory path instead of file returns appropriate error. ✅ "Is a directory"
- [x] **F-META-008**: `apr run` with a 0-byte file returns "Invalid format" error. ✅ "File too small for format detection"
- [ ] **F-META-009**: Application handles SIGINT (Ctrl+C) during loading phase without zombie processes. ⏳ Not tested
- [ ] **F-META-010**: Application handles SIGINT (Ctrl+C) during inference phase without zombie processes. ⏳ Not tested

### II. Input Format Falsification (The Loader Gauntlet) [15 Points]
*Tests the robustness of the "Universal Loader" hypothesis.*

**Run Date:** 2026-01-27 | **Score: 5/15**

- [x] **F-LOAD-011**: Load **GGUF (Q4_K_M)**: Must succeed (Qwen2.5-1.5B). ✅ 0.76s, 1117MB
- [ ] **F-LOAD-012**: Load **GGUF (Q8_0)**: Must succeed or gracefully decline. ⏳ Not tested
- [x] **F-LOAD-013**: Load **SafeTensors (F32)**: Must succeed. ✅ 0.38s, 988MB
- [ ] **F-LOAD-014**: Load **SafeTensors (BF16)**: Must succeed or explicit error "BF16 not yet supported" (no silent garbage). ⏳ Not tested
- [x] **F-LOAD-015**: Load **APR (Native)**: Must succeed (converted from SafeTensors). ✅ Loads, but see F-QUAL
- [ ] **F-LOAD-016**: Load **APR (Corrupt)**: Header valid, body truncated -> Immediate error (no hang). ⏳ Not tested
- [ ] **F-LOAD-017**: Load **Unknown Architecture**: Load a BERT/SDXL model -> Error "Unsupported architecture", not random inference. ⏳ Not tested
- [x] **F-LOAD-018**: **Architecture Detection**: Qwen2.5 GGUF identified as "Qwen2" (not "Transformer"). ✅ Via apr check
- [ ] **F-LOAD-019**: **Metadata Extraction**: Correct `hidden_size`, `num_layers`, `num_heads` reported in `--verbose` mode. ⏳ Not tested
- [x] **F-LOAD-020**: **Tokenizer Fallback**: Model without embedded tokenizer finds `tokenizer.json` in local cache. ✅ APR finds HF cache
- [ ] **F-LOAD-021**: **Missing Tokenizer**: No embedded, no local -> Error "Tokenizer not found" (no crash). ⏳ Not tested
- [ ] **F-LOAD-022**: **Hash Filenames**: Load `c8490f8...gguf` -> Correctly detects Qwen2 arch and applies chat template (PMAT-109). ⏳ Not tested
- [ ] **F-LOAD-023**: **Tensor Layout**: Row-major tensors loaded into row-major memory (log verification). ⏳ Not tested
- [ ] **F-LOAD-024**: **GQA Metadata**: `num_kv_heads` correctly inferred from tensor shapes if missing in metadata (PMAT-107). ⏳ Not tested
- [ ] **F-LOAD-025**: **Schema Aliases**: Loader accepts `n_embd` synonym for `hidden_size` (PMAT-111). ⏳ Not tested

### III. The Output Quality Invariants (Strict Verification) [15 Points]
*Tests if the output is semantically valid. "2+2=4".*

**Run Date:** 2026-01-27 | **Score: 2/15**

- [x] **F-QUAL-026**: **Deterministic Arithmetic**: `apr run ... "2+2="` -> Output contains "4". ✅ GGUF CPU: "4"
- [x] **F-QUAL-027**: **Deterministic Arithmetic (GPU)**: GPU run "2+2=" -> Output contains "4". ✅ GGUF GPU: "4"
- [ ] **F-QUAL-028**: **No Garbage (Start)**: Output does not start with `` or random unicode. ⏳ Not tested
- [ ] **F-QUAL-029**: **No Garbage (Loop)**: Output does not enter infinite repetition loop ("and and and and"). ⏳ Not tested
- [ ] **F-QUAL-030**: **No Garbage (Tokens)**: Output does not contain raw token IDs (e.g., "token151643"). ⏳ Not tested
- [ ] **F-QUAL-031**: **Stop Tokens**: Generation stops at `<|im_end|>` or equivalent (doesn't spew user prompt).
- [ ] **F-QUAL-032**: **Argmax Parity (CPU)**: GGUF vs SafeTensors for same prompt -> Same Argmax token at pos 0 (T100 vs T200).
- [ ] **F-QUAL-033**: **Argmax Parity (GPU)**: GGUF CPU vs GGUF GPU -> Same Argmax token at pos 0.
- [ ] **F-QUAL-034**: **Chat Template**: `apr chat` correctly formats `<|im_start|>user...` (Model doesn't reply "I am not a chat model").
- [ ] **F-QUAL-035**: **System Prompt**: `--system "You are a pirate"` -> Model replies in pirate voice.
- [ ] **F-QUAL-036**: **Temperature 0**: Two consecutive runs with `-t 0` produce bit-identical text output.
- [ ] **F-QUAL-037**: **Context Window**: Input > 4096 tokens -> Error "Context limit exceeded" or proper truncation (no silent failure).
- [ ] **F-QUAL-038**: **Empty Prompt**: `apr run ... ""` -> Handles gracefully (generates or exits, no panic).
- [ ] **F-QUAL-039**: **Whitespace Prompt**: `apr run ... "   "` -> Handles gracefully.
- [ ] **F-QUAL-040**: **Special Tokens**: Input containing `<|im_end|>` is sanitized or handled per policy.

### IV. Performance & Resource Falsification [15 Points]
*Tests the "Efficient Inference" hypothesis.*

- [ ] **F-PERF-041**: **KV Cache O(n)**: Generation speed for token 100 vs token 1000 is roughly constant (not O(n²) slowdown).
- [ ] **F-PERF-042**: **GPU Acceleration**: GGUF GPU tok/s > GGUF CPU tok/s (Must be > 2x to pass).
- [ ] **F-PERF-043**: **Pre-fill Speed**: Prompt processing is faster than generation (batched vs serial).
- [ ] **F-PERF-044**: **Memory Leak (Run)**: RAM usage stable during long generation (1000 tokens).
- [ ] **F-PERF-045**: **Memory Leak (Server)**: RAM usage stable after 1000 requests.
- [ ] **F-PERF-046**: **VRAM Limit**: Attempting to load model > VRAM -> Falls back to CPU or errors gracefully (no CUDA OOM crash).
- [ ] **F-PERF-047**: **CPU Usage**: `apr run` uses all cores (or specified `--threads`).
- [ ] **F-PERF-048**: **Idle Usage**: `apr serve` uses near-zero CPU when idle.
- [ ] **F-PERF-049**: **Model Loading Time**: < 10s for 1.5B model on SSD.
- [ ] **F-PERF-050**: **First Token Latency**: < 1s for short prompt (warm start).
- [ ] **F-PERF-051**: **Concurrency**: Server handles 2 simultaneous requests (queueing or batching) without crashing.
- [ ] **F-PERF-052**: **Throughput**: 1.5B Q4_K on CPU AVX2 > 10 tok/s.
- [ ] **F-PERF-053**: **Throughput**: 1.5B Q4_K on GPU RTX 4090 > 50 tok/s.
- [ ] **F-PERF-054**: **Oversubscription**: Running with threads > physical cores does not degrade performance > 50%.
- [ ] **F-PERF-055**: **Hang Detection**: Any operation taking > 60s without output is killed and flagged "HANG".

### V. Rosetta Conversion & Interop [10 Points]
*Tests the "Universal Translator" hypothesis.*

- [ ] **F-CONV-056**: **SafeTensors -> APR**: Conversion succeeds.
- [ ] **F-CONV-057**: **APR -> SafeTensors**: Conversion succeeds.
- [ ] **F-CONV-058**: **Round Trip**: SafeTensors -> APR -> SafeTensors -> Checksum/Size matches (approx).
- [ ] **F-CONV-059**: **Inference Parity**: `apr rosetta compare-inference ST APR` -> "MATCH" (PMAT-114).
- [ ] **F-CONV-060**: **GGUF -> APR**: Attempted (Current status: FALSIFIED/Garbage is acceptable if documented, Panic is NOT).
- [ ] **F-CONV-061**: **Metadata Preservation**: Converted model retains architecture/tokenizer info.
- [ ] **F-CONV-062**: **Quantization Preservation**: F32 in -> F32 out (unless quant flag used).
- [ ] **F-CONV-063**: **File Size**: APR file size roughly equivalent to source tensor data size.
- [ ] **F-CONV-064**: **Overwrite Protection**: Converter refuses to overwrite existing file without `--force`.
- [ ] **F-CONV-065**: **Partial Convert**: Interrupting conversion deletes partial file.

### VI. Jidoka & Safety (The Andon Cord) [15 Points]
*Tests the "Stop on Defect" hypothesis.*

- [ ] **F-SAFE-066**: **NaN Detection**: Injecting NaN into weights -> Inference Halts (Panic/Error), does not output garbage.
- [ ] **F-SAFE-067**: **Inf Detection**: Intermediate activation overflow -> Inference Halts.
- [ ] **F-SAFE-068**: **Softmax Norm**: Sum of probs != 1.0 ± epsilon -> Warning/Error.
- [ ] **F-SAFE-069**: **Vocab Bounds**: Token ID >= vocab_size -> Error (no out-of-bounds read).
- [ ] **F-SAFE-070**: **Embedding Bounds**: Embedding lookup with invalid index -> Error.
- [ ] **F-SAFE-071**: **Dimension Mismatch**: Matrix mult with wrong shapes -> Explicit panic "Shape mismatch", not segfault.
- [ ] **F-SAFE-072**: **Unsafe Code**: Minimal `unsafe` blocks audit (grep `unsafe`).
- [ ] **F-SAFE-073**: **Sandboxing**: `apr run` does not write outside CWD or `/tmp`.
- [ ] **F-SAFE-074**: **Network**: `apr run` (offline mode) makes NO network requests.
- [ ] **F-SAFE-075**: **API Security**: `apr serve` binds to localhost by default (not 0.0.0.0).
- [ ] **F-SAFE-076**: **Input Sanitization**: Server payload > 10MB -> 413 Payload Too Large.
- [ ] **F-SAFE-077**: **Trace Safety**: `--trace` does not log environment variables or secrets.
- [ ] **F-SAFE-078**: **Path Traversal**: `apr run ../../../etc/passwd` -> Blocked or handled as file (no exposure).
- [ ] **F-SAFE-079**: **Dependency Audit**: `cargo deny check` passes.
- [ ] **F-SAFE-080**: **Panic Handler**: Custom panic hook logs to stderr/file before exit.

### VII. Observability & Tracing (The Microscope) [10 Points]
*Tests the "No Black Box" hypothesis.*

- [ ] **F-OBS-081**: **Trace Flag**: `--trace` generates JSON output.
- [ ] **F-OBS-082**: **Trace Schema**: JSON output matches defined schema (Layers, Timings, Tokens).
- [ ] **F-OBS-083**: **Real Profiling**: `apr profile` shows non-zero timings for `attention` (BrickProfiler).
- [ ] **F-OBS-084**: **Timing Sum**: Sum of layer timings <= Total wall time.
- [ ] **F-OBS-085**: **Token Stream**: Trace includes stream of generated tokens.
- [ ] **F-OBS-086**: **Verbose Mode**: `--verbose` logs model config, backend used (CPU/GPU), and thread count.
- [ ] **F-OBS-087**: **Server Logs**: Requests logged with Method, Path, Status, Latency.
- [ ] **F-OBS-088**: **Error Logs**: Stack traces hidden from API responses, logged to stderr.
- [ ] **F-OBS-089**: **Check Command**: `apr check` runs Stages 1-10 successfully on valid model.
- [ ] **F-OBS-090**: **Check Failure**: `apr check` on broken model fails at specific stage (e.g., Tokenizer).

### VIII. The Severe Testing "T-Series" (The Final Boss) [10 Points]
*Methodological requirements from Appendix D.*

- [ ] **F-METH-091**: **T100 (Real GGUF)**: Pass.
- [ ] **F-METH-092**: **T200 (Real ST)**: Pass.
- [ ] **F-METH-093**: **No Fixtures**: Final Verification done with REAL models, not mock objects.
- [ ] **F-METH-094**: **Cross-Platform**: Tests run on Linux (Current).
- [ ] **F-METH-095**: **CI Gate**: CI fails if `cargo test` fails.
- [ ] **F-METH-096**: **Clean State**: Tests do not rely on state from previous tests.
- [ ] **F-METH-097**: **Falsifiable Statements**: Every bug report includes a reproduction case.
- [ ] **F-METH-098**: **Root Cause Analysis**: Every fix includes a "5 Whys" (as seen in Spec).
- [ ] **F-METH-099**: **Regression**: Old bugs (PMAT-094, PMAT-103) added to regression suite.
- [ ] **F-METH-100**: **The 2+2=5 Test**: If the model answers "5", the test MUST fail (Semantics matter).

---
**Score Calculation:**
- **Pass**: 1 Point
- **Fail**: 0 Points (Blocks Release)
- **Total**: / 100

**Falsification Threshold**: < 100/100 means the hypothesis "The software is release-ready" is **FALSIFIED**.
