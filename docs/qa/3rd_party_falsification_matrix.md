# 3rd Party QA Falsification Matrix: Qwen2.5-Coder Showcase

**Target Specification:** `docs/specifications/qwen2.5-coder-showcase-demo.md` (v9.5.1)
**Date:** 2026-02-03
**Auditor:** 3rd Party QA (Simulated)
**Philosophy:** Popperian Falsification. We do not look for proof that the system works; we actively search for evidence that the system is broken or that the specification is lying.

> "A theory that is not falsifiable is not scientific." — Karl Popper

## 1. Release Criteria Falsification (Section 0 & Release Table)

**Claim:** The release status is "PARTIAL" with specific pass/fail states for different formats.

| ID | Claim to Falsify | Falsification Test | Falsification Criteria (FAIL if...) |
|----|-------------------|-------------------|-------------------------------------|
| **F-REL-001** | **SafeTensors 1.5B (HF) passes CPU/GPU.** | 1. `apr pull hf://Qwen/Qwen2.5-Coder-1.5B-Instruct --output st_1.5b.safetensors`<br>2. `apr run st_1.5b.safetensors "2+2="`<br>3. `apr run st_1.5b.safetensors "2+2=" --gpu` | Output is garbage, empty, crashes, or math is wrong (e.g. "2+2=5"). |
| **F-REL-002** | **SafeTensors 0.5B (HF) passes CPU/GPU.** | 1. `apr pull hf://Qwen/Qwen2.5-Coder-0.5B-Instruct --output st_0.5b.safetensors`<br>2. `apr run st_0.5b.safetensors "2+2="`<br>3. `apr run st_0.5b.safetensors "2+2=" --gpu` | Output is garbage, empty, crashes, or math is wrong. |
| **F-REL-003** | **APR F32 (from ST) is PARTIAL (First token correct).** | 1. `apr import st_1.5b.safetensors -o apr_f32.apr`<br>2. `apr run apr_f32.apr "2+2="` | **EITHER**: <br>A) It works perfectly (Claim "PARTIAL" is false).<br>B) First token is WRONG (Claim "First token correct" is false). |
| **F-REL-004** | **GGUF F32 (from ST) is CRASH.** | 1. `apr export st_1.5b.safetensors --format gguf -o gguf_f32.gguf`<br>2. `apr run gguf_f32.gguf` | It runs successfully (Claim "CRASH" is false). |
| **F-REL-005** | **Pre-baked GGUF is BANNED.** | 1. Attempt `apr pull hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF`<br>2. Check logs/warnings. | System allows usage without explicit warning/error about methodology violation (Claim "BANNED" is unenforceable). |

## 2. P0 Defect Fix Falsification (GitHub Issues Section)

**Claim:** All listed P0 issues are "FIXED".

| ID | Claim to Falsify | Falsification Test | Falsification Criteria (FAIL if...) |
|----|-------------------|-------------------|-------------------------------------|
| **F-FIX-198** | **apr pull fetches tokenizer.json (PMAT-195).** | 1. `rm -rf ~/.cache/apr`<br>2. `apr pull hf://Qwen/Qwen2.5-Coder-0.5B-Instruct`<br>3. `ls ~/.cache/apr/models/.../tokenizer.json` | `tokenizer.json` or `config.json` is missing. |
| **F-FIX-197** | **SafeTensors Inference Garbage (PMAT-197).** | `apr chat st_0.5b.safetensors` with prompt "What is 2+2?" | Output contains garbage tokens, mixed languages, or copyright headers instead of clear text. |
| **F-FIX-194** | **Conversion --preserve-q4k works (PMAT-210).** | 1. Get GGUF Q4K (if available/allowed for testing).<br>2. `apr import model.gguf --preserve-q4k`<br>3. `apr tensors model.apr` | Tensors are F32 instead of Q4_K (Quantization lost). |
| **F-FIX-192** | **APR Import Drops Tensors (PMAT-209).** | 1. `apr import st_0.5b.safetensors`<br>2. Compare tensor count vs original `apr inspect`. | Tensor count in APR < Tensor count in SafeTensors. |
| **F-FIX-189** | **Chat Special Tokens atomic (PMAT-206).** | `apr run st_0.5b.safetensors "Hello"` | Output reveals raw special tokens split into pieces or mishandled chat template tags. |
| **F-FIX-187** | **NaN Corruption Detection (PMAT-187).** | 1. Create a dummy ST file with NaN weights (using python script).<br>2. `apr import broken.safetensors` | Conversion SUCCEEDS silently (Validation failed to stop the line). |

## 3. Performance & Parity Falsification (Section "Benchmark Results")

**Claim:** SafeTensors Throughput ~19.4 tok/s (CPU) and GPU > 2x CPU.

| ID | Claim to Falsify | Falsification Test | Falsification Criteria (FAIL if...) |
|----|-------------------|-------------------|-------------------------------------|
| **F-PERF-001** | **SafeTensors CPU Baseline ~19 tok/s.** | `apr bench st_0.5b.safetensors --device cpu` | Throughput is significantly lower (< 10 tok/s) on reference hardware (Claim is exaggerated). |
| **F-PERF-002** | **GPU Speedup > 2.0x (PMAT-118).** | 1. `apr bench st_0.5b.safetensors --device cpu`<br>2. `apr bench st_0.5b.safetensors --device gpu` | GPU throughput <= 2.0 * CPU throughput (Claim "GPU > 2x" is false). |
| **F-PERF-003** | **APR F32 matches SafeTensors F32 speed.** | Compare `apr bench` results of ST vs APR(F32). | APR is > 20% slower than SafeTensors (Claim "Identical to ST source" is false). |

## 4. Methodology & Tooling Falsification (Sections 0, 1, 2)

**Claim:** All 13 APR tools are covered (100% coverage) and comply with "Toyota Way".

| ID | Claim to Falsify | Falsification Test | Falsification Criteria (FAIL if...) |
|----|-------------------|-------------------|-------------------------------------|
| **F-TOOL-001** | **Tool Existence (PMAT-191).** | Run `--help` for: `run`, `chat`, `serve`, `inspect`, `validate`, `bench`, `profile`, `trace`, `check`, `canary`, `convert`, `tune`, `qa`. | Any command returns "unknown subcommand" or panics. |
| **F-METH-001** | **SafeTensors Ground Truth (Section 0.3).** | 1. `apr run st.safetensors "2+2="`<br>2. `apr run apr.apr "2+2="`<br>3. Verify equality. | Output differs between formats (Ground Truth violation). |
| **F-METH-002** | **Verbose Mode UX (F-UX-027 to 040).** | `apr run st_0.5b.safetensors --verbose` | Missing ANY of: Source path, File size, Arch, Layers, Vocab, Load time, Backend, Hidden size, Threads, Quant type, Context length. |

## 5. Critical Failure Handling (Section "Critical Failures")

**Claim:** Panic 411 (Empty Tensor) and other critical failures are handled gracefully.

| ID | Claim to Falsify | Falsification Test | Falsification Criteria (FAIL if...) |
|----|-------------------|-------------------|-------------------------------------|
| **F-CRIT-001** | **Empty Tensor Panic (F-STRESS-520).** | 1. Create a 0-byte file `empty.safetensors`.<br>2. `apr run empty.safetensors` | Process PANICS (stack trace) instead of printing error message. |
| **F-CRIT-002** | **Missing Tokenizer (PMAT-172).** | 1. Move `tokenizer.json` away from model.<br>2. `apr run model.safetensors` | Process PANICS or generates garbage. (Should fail fast with specific error). |
| **F-CRIT-003** | **Lock Poisoning (PMAT-189).** | *Requires simulated crash inside lock*. (Mock test) | Server hangs indefinitely or panics on subsequent requests after a handler failure. |

## 6. Execution Instructions

1.  **Setup:**
    ```bash
    mkdir -p QA_ARTIFACTS
    export RUST_BACKTRACE=1
    ```
2.  **Run Falsification:**
    Execute tests in order. Stop immediately if any **P0 Claim** (Section 1 & 2) is falsified.
3.  **Reporting:**
    Mark specification as **FALSIFIED** if *any* single test fails criteria.
    Update `docs/specifications/qwen2.5-coder-showcase-demo.md` status to "❌ FALSIFIED" if failures are found.

---
**End of Matrix**
