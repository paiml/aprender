# QA Showcase Methodology: Same-Model Comparison Protocol

**Version:** 1.1.0
**Status:** IMPLEMENTED
**Date:** 2026-01-26
**Refs:** PMAT-SHOWCASE-METHODOLOGY-001

---

## Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| CPU/GPU Performance Thresholds | ✅ DONE | qa_chat (3.0/5.0), qa_run (5.0/5.0) |
| GGUF [TRACE-CACHE] Tracing | ✅ DONE | CPU and GPU paths |
| `--class` flag for qa_run | ✅ DONE | quantized, full-precision, all |
| qa_chat tests | ✅ PASS | 20/20 points |
| qa_run CPU GGUF | ✅ PASS | 15/15 points |
| qa_run GPU GGUF | ✅ PASS | 15/15 points |
| qa_serve tests | ✅ PASS | 35/35 points |
| qa_verify tests | ✅ PASS | 20/20 points |
| `--trace-level` CLI flag | ✅ DONE | Added to run, chat, serve |
| `--profile` CLI flag | ✅ DONE | Added to run, chat, serve |
| `--trace-level layer` output | ✅ DONE | Per-layer timing table in run command |
| `--profile` Roofline output | ✅ DONE | Memory/compute analysis in run command |
| `--trace-output` for GGUF | ✅ DONE | JSON output to file |
| Ollama parity | ✅ DONE | Automated comparison via `--with-ollama` |

---

## 1. FATAL DEFECT: Comparing Different Models

**Previous (WRONG) approach:**
```
GGUF:       Qwen2.5-Coder-1.5B-Instruct Q4_K_M (~1.1GB, quantized)
SafeTensors: Qwen2.5-Coder-1.5B-Instruct F32   (~3.0GB, full precision)
APR:        Converted from SafeTensors F32     (~3.0GB, full precision)
```

**Problem:** Comparing Q4_K_M (4.5 bits/weight) vs F32 (32 bits/weight) is **INVALID**.
- Different memory footprint (7x difference)
- Different compute characteristics
- Different accuracy profiles
- **NOT the same model**

---

## 2. CORRECT Methodology: Single Canonical Model

### 2.1 Canonical Source: GGUF Q4_K_M

```
CANONICAL MODEL: Qwen2.5-Coder-1.5B-Instruct-GGUF/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
```

**Why GGUF as canonical:**
- Pre-quantized by model authors (Qwen team)
- Industry-standard quantization (llama.cpp compatible)
- Same weights used by Ollama (our groundtruth)
- Smallest file size for distribution

### 2.2 Conversion Pipeline

```
                    ┌─────────────────────────────────────────┐
                    │  GGUF Q4_K_M (Canonical Source)         │
                    │  qwen2.5-coder-1.5b-instruct-q4_k_m.gguf│
                    └────────────────┬────────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │   GGUF Q4_K_M   │   │   APR Q4_K_M    │   │  Ollama Q4_K_M  │
    │   (original)    │   │   (converted)   │   │  (groundtruth)  │
    └─────────────────┘   └─────────────────┘   └─────────────────┘
              │                      │                      │
              ▼                      ▼                      ▼
         apr run               apr run                ollama run
         (realizar)            (realizar)             (reference)
```

### 2.3 SafeTensors: Separate Comparison Class

SafeTensors is F32/F16, cannot be directly compared to quantized formats:

```
┌─────────────────────────────────────────────────────────────┐
│  CLASS A: Quantized Inference (Q4_K_M, ~1.1GB)              │
│  - GGUF Q4_K_M                                               │
│  - APR Q4_K_M (converted from GGUF)                         │
│  - Ollama Q4_K_M (groundtruth)                              │
│  - EXPECTED: ~8-15 tok/s CPU, ~100+ tok/s GPU               │
├─────────────────────────────────────────────────────────────┤
│  CLASS B: Full Precision Inference (F32, ~3.0GB)            │
│  - SafeTensors F32                                           │
│  - APR F32 (converted from SafeTensors)                     │
│  - EXPECTED: ~1-3 tok/s CPU, ~10-30 tok/s GPU               │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Test Matrix (Revised)

### 3.1 Class A: Quantized Comparison (Primary)

| Cell | Backend | Format | Model Source | Points |
|------|---------|--------|--------------|--------|
| A1 | CPU | GGUF Q4_K | HF GGUF | 15 |
| A2 | CPU | APR Q4_K | Converted from A1 | 15 |
| A3 | GPU | GGUF Q4_K | HF GGUF | 15 |
| A4 | GPU | APR Q4_K | Converted from A3 | 15 |

**Groundtruth:** `ollama run qwen2.5-coder:1.5b-instruct-q4_K_M`

### 3.2 Class B: Full Precision Comparison (Secondary)

| Cell | Backend | Format | Model Source | Points |
|------|---------|--------|--------------|--------|
| B1 | CPU | SafeTensors F32 | HF SafeTensors | 10 |
| B2 | CPU | APR F32 | Converted from B1 | 10 |
| B3 | GPU | SafeTensors F32 | HF SafeTensors | 10 |
| B4 | GPU | APR F32 | Converted from B3 | 10 |

**Note:** No Ollama groundtruth for F32 (Ollama only uses quantized).

### 3.3 Total Points

- Class A (Quantized): 60 points
- Class B (Full Precision): 40 points
- **Total: 100 points**

---

## 4. Tracing Requirements (ALL MUST WORK)

**CRITICAL:** Tracing MUST work for ALL three inference modalities: `run`, `chat`, `serve`.

### 4.0 GGUF Tracing (PMAT-TRACE-GGUF-001)

**Status:** ✅ IMPLEMENTED (2026-01-26)

| Path | `[TRACE-CACHE]` Output | Status |
|------|------------------------|--------|
| SafeTensors/APR | ✅ Has tracing in `realizar/src/apr_transformer/mod.rs` | Works |
| GGUF CPU | ✅ Tracing in `realizar/src/gguf/inference/generation.rs` | **IMPLEMENTED** |
| GGUF GPU | ✅ Tracing in `realizar/src/gguf/cuda/generation.rs` | **IMPLEMENTED** |

**Example Output:**
```bash
apr run $MODEL --prompt "2+2=" --max-tokens 5 --trace
# Output:
# [TRACE-CACHE] GGUF model (GPU): 28 layers, hidden_dim=1536, vocab=151936
# [TRACE-CACHE] Prefill: 12 tokens, max_gen=5
# [TRACE-CACHE] Prefill complete: 12 tokens in 112.08ms
# [TRACE-CACHE] pos=11: 28 layers took 6.95ms
# [TRACE-CACHE] pos=12: 28 layers took 6.83ms
```

**Changes Made:**
- `realizar/src/gguf/inference/generation.rs`: Added trace output to `generate_with_cache()`
- `realizar/src/gguf/cuda/generation.rs`: Added trace output to `generate_gpu_resident()`
- `apr-cli/src/commands/run.rs`: Removed outdated "not implemented" warnings

### 4.1 Canonical Model for All Tests

```bash
MODEL="hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
```

### 4.2 Tracing Modes

| Mode | Flag | Description | Output |
|------|------|-------------|--------|
| **None** | (default) | No tracing | Clean output only |
| **Trace** | `--trace` | Step-by-step timing | [TRACE-CACHE] messages |
| **Layer** | `--trace-level layer` | Per-layer breakdown | Layer timing table |
| **Profile** | `--profile` | Roofline analysis | Memory/compute bounds |
| **Payload** | `--trace-payload` | Tensor value inspection | Activation stats |

### 4.3 RUN Command Tracing (MUST ALL WORK)

```bash
# No tracing (clean output)
apr run $MODEL --prompt "What is 2+2?" --max-tokens 10

# Basic trace
apr run $MODEL --prompt "What is 2+2?" --max-tokens 10 --trace

# Layer-level trace
apr run $MODEL --prompt "What is 2+2?" --max-tokens 10 --trace --trace-level layer

# Full profile
apr run $MODEL --prompt "What is 2+2?" --max-tokens 10 --profile

# Payload trace (inspect activations)
apr run $MODEL --prompt "What is 2+2?" --max-tokens 10 --trace-payload

# Combined: trace + layer + output JSON
apr run $MODEL --prompt "What is 2+2?" --max-tokens 10 --trace --trace-level layer --trace-output trace.json
```

### 4.4 CHAT Command Tracing (MUST ALL WORK)

```bash
# No tracing (interactive chat)
apr chat $MODEL

# Basic trace (shows timing per response)
apr chat $MODEL --trace

# Layer-level trace
apr chat $MODEL --trace --trace-level layer

# Profile mode
apr chat $MODEL --profile
```

### 4.5 SERVE Command Tracing (MUST ALL WORK)

```bash
# No tracing (production mode)
apr serve $MODEL --port 8080

# Basic trace (logs per-request timing)
apr serve $MODEL --port 8080 --trace

# Layer-level trace (detailed per-request)
apr serve $MODEL --port 8080 --trace --trace-level layer

# Profile mode (adds X-Profile headers)
apr serve $MODEL --port 8080 --profile
```

**HTTP Header Support (SERVE):**
- `X-Trace-Level: layer` - Request per-layer timing in response
- `X-Profile: true` - Request roofline analysis in response

### 4.6 Tracing QA Matrix (45 points)

| Command | Mode | Test | Points |
|---------|------|------|--------|
| **RUN** | None | Clean output, correct answer | 3 |
| **RUN** | `--trace` | Shows [TRACE-CACHE] pos=N | 3 |
| **RUN** | `--trace-level layer` | Shows layer timing | 3 |
| **RUN** | `--profile` | Shows roofline analysis | 3 |
| **RUN** | `--trace-output` | Writes valid JSON | 3 |
| **CHAT** | None | Interactive works | 3 |
| **CHAT** | `--trace` | Shows timing per turn | 3 |
| **CHAT** | `--trace-level layer` | Shows layer breakdown | 3 |
| **CHAT** | `--profile` | Shows roofline | 3 |
| **SERVE** | None | HTTP 200 on /health | 3 |
| **SERVE** | `--trace` | Logs request timing | 3 |
| **SERVE** | `--trace-level layer` | Logs layer timing | 3 |
| **SERVE** | `--profile` | X-Profile header works | 3 |
| **SERVE** | X-Trace-Level header | Returns trace in response | 3 |
| **ALL** | GPU + trace | Tracing works on GPU path | 3 |

**Total Tracing Points: 45**

### 4.7 Expected Trace Output Format

**RUN `--trace` output:**
```
[TRACE-CACHE] Layer 0: QKV projection using Q4_K (fused)
[TRACE-CACHE] Layer 0: attn_output using Q4_K SIMD
[TRACE-CACHE] pos=0: 28 layers took 45.2ms
[TRACE-CACHE] pos=1: 28 layers took 44.8ms
...
```

**RUN `--trace-level layer` output:**
```
Layer Timing (28 layers × N tokens):
  Layer  | Attn (ms) | FFN (ms) | Norm (ms) | Total (ms)
  -------|-----------|----------|-----------|------------
  0      | 12.3      | 18.4     | 0.8       | 31.5
  1      | 11.9      | 17.2     | 0.7       | 29.8
  ...
```

**RUN `--profile` output:**
```
Roofline Analysis:
  Compute Bound: 45% of layers
  Memory Bound:  55% of layers
  Bottleneck:    Memory bandwidth (DRAM)
  Recommendation: Use quantized model for better cache utilization
```

**SERVE trace headers:**
```
X-Trace-Time-Ms: 234.5
X-Trace-Tokens: 15
X-Trace-Tok-Per-Sec: 63.9
X-Trace-Layers: 28
X-Trace-Avg-Layer-Ms: 8.4
```

---

## 5. Ollama Parity Test

### 5.1 Groundtruth Definition

Ollama is the groundtruth for quantized inference:

```bash
# Install Ollama model (same Q4_K_M)
ollama pull qwen2.5-coder:1.5b-instruct-q4_K_M

# Run Ollama
ollama run qwen2.5-coder:1.5b-instruct-q4_K_M "What is 2+2? Answer with just the number."
```

### 5.2 Parity Requirements

| Metric | Requirement | Measurement |
|--------|-------------|-------------|
| **Correctness** | Output matches Ollama | Exact or semantic match |
| **Performance** | Within 2x of Ollama | tok/s comparison |
| **Memory** | Within 1.5x of Ollama | Peak RSS |

### 5.3 Ollama Comparison Script

```bash
# Run Ollama benchmark
ollama run qwen2.5-coder:1.5b-instruct-q4_K_M "What is 2+2?" --verbose 2>&1 | grep "eval rate"

# Run APR benchmark
apr run hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
    --prompt "What is 2+2?" --trace 2>&1 | grep "tok/s"

# Compare
```

---

## 6. Conversion Commands

### 6.1 GGUF → APR Q4_K

```bash
# Download GGUF
apr download hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf

# Convert to APR (preserving Q4_K quantization)
apr import ~/.cache/apr/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
    --output qwen2.5-coder-1.5b-q4k.apr \
    --preserve-quantization
```

### 6.2 SafeTensors → APR F32

```bash
# Download SafeTensors
apr download hf://Qwen/Qwen2.5-Coder-1.5B-Instruct

# Convert to APR F32
apr import ~/.cache/apr/Qwen2.5-Coder-1.5B-Instruct/ \
    --output qwen2.5-coder-1.5b-f32.apr
```

---

## 7. QA Run Command Updates

### 7.1 New `--class` Flag

```bash
# Class A: Quantized (GGUF vs APR Q4_K)
cargo run --example qa_run -- --class quantized --matrix

# Class B: Full Precision (SafeTensors vs APR F32)
cargo run --example qa_run -- --class full-precision --matrix

# Both classes
cargo run --example qa_run -- --class all --matrix
```

### 7.2 New `--with-ollama` Flag

```bash
# Compare against Ollama groundtruth
cargo run --example qa_run -- --class quantized --with-ollama
```

### 7.3 Tracing Tests

```bash
# Run all tracing tests
cargo run --example qa_run -- --tracing-tests

# Run specific trace level
cargo run --example qa_run -- --backend cpu --format gguf --trace-level layer
```

---

## 8. Implementation Checklist

- [ ] Update `qa_run.rs` with `--class` flag
- [ ] Add GGUF → APR Q4_K conversion in QA setup
- [ ] Add Ollama comparison tests
- [ ] Implement all tracing mode tests
- [ ] Update `default_model_for_format()` to use same canonical source
- [ ] Add pre-conversion step before matrix tests
- [ ] Update spec with actual benchmark results

---

## 9. Success Criteria

| Criterion | Threshold |
|-----------|-----------|
| Class A (Quantized) all pass | 60/60 |
| Class B (Full Precision) all pass | 40/40 |
| Tracing tests all pass | 45/45 |
| Ollama parity (performance) | Within 2x |
| Ollama parity (correctness) | Exact match |

**Total: 145 points + Ollama parity**

---

## 10. Falsification Requirements (NOISY-GUARD + F-MODEL-COMPLETE)

These are hard requirements that MUST be satisfied. Violating them is a test failure.

### 10.1 NOISY-GUARD: Silent by Default

| Requirement | Test | Violation |
|-------------|------|-----------|
| **F-NOISY-001** | `apr chat model.gguf` produces ZERO `[DEBUG]` or `[TRACE]` lines | Fail if any debug output without `--trace` |
| **F-NOISY-002** | `apr run model.gguf --prompt "Hi"` has clean output | Fail if trace messages appear |
| **F-NOISY-003** | `apr serve model.gguf` logs only startup and requests | Fail if debug spam in logs |
| **F-NOISY-004** | `--trace` flag MUST enable trace output | Fail if `--trace` produces no trace |
| **F-NOISY-005** | `REALIZE_TRACE=1` env var enables traces | Fail if env var ignored |

**Test Command:**
```bash
# MUST produce clean output (no [DEBUG], [TRACE], [TRACE-CACHE])
apr run $MODEL --prompt "Hi" --max-tokens 5 2>&1 | grep -E '^\[' && echo "FAIL: Debug output detected"

# MUST produce trace output
apr run $MODEL --prompt "Hi" --max-tokens 5 --trace 2>&1 | grep -q '\[TRACE' || echo "FAIL: No trace with --trace"
```

### 10.2 F-GPU-134: GPU by Default

| Requirement | Test | Violation |
|-------------|------|-----------|
| **F-GPU-001** | GGUF chat uses GPU when CUDA available | Fail if CPU used without `--no-gpu` |
| **F-GPU-002** | APR chat uses GPU when CUDA available | Fail if CPU used without `--no-gpu` |
| **F-GPU-003** | `--no-gpu` forces CPU path | Fail if GPU used with `--no-gpu` |
| **F-GPU-004** | `--gpu` forces GPU (error if unavailable) | Fail if silently falls back to CPU |
| **F-GPU-005** | GPU info message printed on GPU use | `[GGUF CUDA: ...]` or `[APR CUDA: ...]` |

**Test Command:**
```bash
# MUST show GPU info (if CUDA available)
apr chat $MODEL --max-tokens 5 2>&1 | grep -q 'CUDA' || echo "FAIL: Not using GPU"

# MUST NOT show GPU info with --no-gpu
apr chat $MODEL --max-tokens 5 --no-gpu 2>&1 | grep -q 'CUDA' && echo "FAIL: GPU used with --no-gpu"
```

### 10.3 F-MODEL-COMPLETE: No Silent Fallbacks

| Requirement | Test | Violation |
|-------------|------|-----------|
| **F-COMPLETE-001** | Missing tokenizer is FATAL error | Fail if "using byte fallback" appears |
| **F-COMPLETE-002** | Failed tokenizer load is FATAL error | Fail if warning and continues |
| **F-COMPLETE-003** | Missing config.json is FATAL for SafeTensors | Fail if defaults used silently |
| **F-COMPLETE-004** | Error message explains WHY model broken | Fail if generic "load failed" |
| **F-COMPLETE-005** | Error suggests FIX (improper conversion) | Fail if no remediation guidance |

**Test Command:**
```bash
# MUST fail with clear error for incomplete model
apr chat incomplete_model.apr 2>&1 | grep -q 'incomplete' || echo "FAIL: No incomplete error"
apr chat incomplete_model.apr 2>&1 | grep -q 'tokenizer' || echo "FAIL: Doesn't mention tokenizer"
```

### 10.4 Falsification Test Matrix (15 points)

| Test | Requirement | Points |
|------|-------------|--------|
| Clean output (no debug) | F-NOISY-001 | 3 |
| Trace with `--trace` | F-NOISY-004 | 3 |
| GPU by default | F-GPU-001 | 3 |
| `--no-gpu` works | F-GPU-003 | 3 |
| Missing tokenizer = error | F-COMPLETE-001 | 3 |

**Total Falsification Points: 15**

---

## 11. Quick Verification Commands

Run these commands to verify the showcase works:

```bash
# Set canonical model
MODEL="hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"

# 1. RUN - basic (expect: "4")
apr run $MODEL --prompt "What is 2+2? Answer with just the number." --max-tokens 10

# 2. RUN - with trace (expect: [TRACE-CACHE] messages)
apr run $MODEL --prompt "What is 2+2?" --max-tokens 10 --trace

# 3. CHAT - basic (expect: interactive prompt)
apr chat $MODEL

# 4. CHAT - with trace (expect: timing per response)
apr chat $MODEL --trace

# 5. SERVE - basic (expect: server starts on port 8080)
apr serve $MODEL --port 8080 &
curl http://localhost:8080/health
kill %1

# 6. SERVE - with trace (expect: trace logging)
apr serve $MODEL --port 8080 --trace &
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":10}'
kill %1

# 7. Ollama comparison (groundtruth)
ollama run qwen2.5-coder:1.5b-instruct-q4_K_M "What is 2+2? Answer with just the number."
```

---

## 12. References

- PMAT-SHOWCASE-NEXT
- PMAT-APR-TOK-001
- APR-TRACE-001
- LAYOUT-001
