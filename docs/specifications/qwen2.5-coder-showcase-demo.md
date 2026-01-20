# Qwen2.5-Coder Showcase: Unified Inference Architecture

**Version:** 7.6.0
**Status:** GGUF READY — SafeTensors/APR Gaps Remain (100% GGUF, 71% Overall)
**Author:** PAIML Engineering
**Date:** 2026-01-20
**PMAT Roadmap ID:** `SHOWCASE-BRICK-001`
**Issue:** `APR-REALIZE-001`

---

## Executive Summary: The Unified Inference Hypothesis

This specification defines the **Unified Inference Architecture** for the aprender ecosystem. In accordance with **Popperian epistemology**, we frame this architecture not as a verified truth, but as a **falsifiable hypothesis**:

> **Hypothesis ($H_1$):** A strict separation of concerns between `aprender` (library), `realizar` (runtime), and `apr-cli` (interface) yields a system that is performant (≥2x Ollama), consistent (Ollama parity), and scientifically verifiable (10-stage pipeline), without introducing significant overhead.

**Null Hypothesis ($H_0$):** The abstraction overhead of the unified architecture degrades performance below targets or introduces integration bugs that prevent parity, necessitating a return to monolithic design.

We accept $H_1$ strictly provisionally, only as long as it survives the **300-point Falsification Checklist** defined herein.

## Blocking Issues (P0) — Format Support Gaps

The following issues currently **falsify** full readiness (Current GGUF Score: 25/25, Overall: 25/35):

1.  🛑 **PAR-301 (SafeTensors Gap):** `realizar` lacks inference support for SafeTensors format.
    *   *Impact:* Falsifies "Unified Architecture" claim; currently GGUF-only.
2.  🛑 **PAR-302 (APR Format Gap):** APR format loading fails due to missing `config.json` support.
    *   *Impact:* Falsifies "Native Format" support ($H_1$ requires `aprender` integration).

## Resolved Issues — ✅ VERIFIED

3.  ✅ **PAR-303 (0.5B Coherency) FIXED (2026-01-20):** All GGUF model sizes now produce coherent output.
    *   *Verification:* `qa-serve.sh --all-models` passes 95/95 tests across 0.5B, 1B, 1.5B, 7B, 32B.
    *   *Evidence:* Multi-model QA suite validates OpenAI-compatible API for all sizes.

## Known Regressions (PAR-201) — ✅ REFUTED

Previous falsification attempts (QA) successfully refuted the release candidate v0.2.2. The following regressions have since been addressed and the fixes verified:

1.  ✅ **F-GPU-134b FIXED**: `force_cpu` logic corrected. Refutation: `apr chat` now correctly utilizes GPU by default.
2.  ✅ **F-CLI-013b/014b VERIFIED**: Feature flags `--gpu`/`--no-gpu` empirically verified.
3.  ✅ **F-PIPE-166b FIXED**: BPE artifacts (`Ġ`, `Ċ`) eliminated from output stream.
4.  ✅ **F-UX-40 FIXED**: "Noisy" debug output successfully confined to `--verbose`.

---

## 1. Architecture Overview

### 1.1 Component Responsibility Matrix

| Responsibility | aprender | realizar | apr-cli | trueno |
|---------------|----------|----------|---------|--------|
| **Model Training** | ✅ Primary | ❌ Never | ❌ Never | Compute |
| **Autograd/Backprop** | ✅ Primary | ❌ Never | ❌ Never | ❌ |
| **.apr Format R/W** | ✅ Primary | Read-only | ❌ | ❌ |
| **GGUF Loading** | ❌ Never | ✅ Primary | ❌ | ❌ |
| **SafeTensors Loading** | ❌ Never | ✅ Primary | ❌ | ❌ |
| **Model Inference** | ❌ **FORBIDDEN** | ✅ Primary | Delegates | Kernels |
| **KV Cache** | ❌ Never | ✅ Primary | ❌ | Storage |
| **GPU Dispatch** | ❌ Never | ✅ Primary | ❌ | CUDA PTX |
| **HTTP Server** | ❌ Never | ✅ Primary | Calls | ❌ |
| **CLI Interface** | ❌ Never | Has own | ✅ Primary | ❌ |
| **Model Registry** | ❌ Never | ❌ | ✅ (via pacha) | ❌ |
| **10-Stage Pipeline** | ❌ | ✅ Primary | Displays | ❌ |
| **Inference Tracing** | ❌ | ✅ Primary | `--trace` flag | ❌ |
| **Ollama-style UX** | ❌ | ✅ (presentar) | Inherits | ❌ |

### 1.2 Data Flow

```
User Request
     │
     ▼
┌─────────────┐
│   apr-cli   │  ← Model resolution, caching, UX
│  (apr run)  │
└─────┬───────┘
      │ delegates
      ▼
┌─────────────┐
│  realizar   │  ← Inference engine, tracing, GPU/CPU
│  (library)  │
└─────┬───────┘
      │ uses
      ▼
┌─────────────┐
│   trueno    │  ← SIMD kernels, CUDA PTX
│  (compute)  │
└─────────────┘
```

### 1.3 Peer-Reviewed Citations & Theoretical Basis

1.  **Popper, K. (1959).** *The Logic of Scientific Discovery*. Hutchinson.
    -   Foundation of our "Falsification Protocol": We do not prove software works; we fail to prove it breaks.
2.  **Vaswani, A., et al. (2017).** "Attention Is All You Need." *NeurIPS*.
3.  **Liker, J. K. (2004).** *The Toyota Way*. McGraw-Hill.
    -   *Jidoka*: Automated stopping on abnormality (see Section VII).
4.  **Gregg, B. (2020).** *Systems Performance*. Addison-Wesley.
    -   Observability vs. Monitoring (see Section V).
5.  **Dao, T., et al. (2022).** "FlashAttention." *NeurIPS*.
6.  **Little, J. D. C. (1961).** "A Proof for the Queuing Formula: L = λW". *Operations Research*.
    -   Theoretical basis for batching throughput calculations.

### 1.4 Falsification Methodology

To ensure scientific rigor, we classify falsification events (bugs/failures) by severity:

*   **Level 1 (Cosmetic):** Output formatting, help text, typos. Does not refute $H_1$, but requires correction.
*   **Level 2 (Functional):** Feature fails to execute as described (e.g., flag ignored). Requires code fix.
*   **Level 3 (Structural):** Feature works but implementation violates architecture (e.g., CLI doing inference). **Refutes the Design ($H_1$).** Requires refactor.
*   **Level 4 (Existential):** Performance targets physically impossible or core premise invalid. **Refutes the Project Goals.** Requires strategic pivot.

---

## 2. CLI Interface

### 2.1 Commands (apr-cli → realizar delegation)

```bash
# Run inference (delegates to realizar)
apr run model.gguf "What is 2+2?" --max-tokens 32

# With verbose output
apr run model.gguf "prompt" --verbose

# With tracing (AWS Step Functions parity)
apr run model.gguf "prompt" --trace --trace-output trace.json

# GPU acceleration
apr run model.gguf "prompt" --gpu

# Interactive chat (delegates to realizar)
apr chat model.gguf --system "You are helpful."

# HTTP server (delegates to realizar serve)
apr serve model.gguf --port 8080

# 10-stage pipeline verification
apr check model.gguf
```

### 2.2 Output Modes

**Default (Ollama-style):**
```
⠋ (spinner while loading)
The answer is 4.
```

**Verbose (`--verbose`):**
```
Loading model: model.gguf
  Source: local file
  Format: GGUF v3
  Tensors: 339
Backend: CPU (AVX2 + SIMD)
Model loaded in 1446.07ms
Architecture: qwen2, Hidden: 1536, Layers: 28
...
Performance: 25.3 tok/s
```

**Trace (`--trace`):**
```json
{
  "execution_arn": "arn:apr:execution:local:uuid",
  "events": [
    {"type": "TaskStateEntered", "name": "TOKENIZE", "input": "What is 2+2?"},
    {"type": "TaskStateExited", "name": "TOKENIZE", "output": [3, 1025, 8234]},
    ...
  ]
}
```

---

## 3. 10-Stage Pipeline Verification

### 3.1 Pipeline Stages

```
┌─────┬─────────────────────┬──────────┬────────────────────────┬──────┐
│  #  │      Component      │ Softmax? │          ELI5          │ Done │
├─────┼─────────────────────┼──────────┼────────────────────────┼──────┤
│ 1   │ Tokenizer           │ -        │ Words → numbers        │ ✅   │
├─────┼─────────────────────┼──────────┼────────────────────────┼──────┤
│ 2   │ Embedding           │ -        │ Numbers → vectors      │ ✅   │
├─────┼─────────────────────┼──────────┼────────────────────────┼──────┤
│ 3   │ Positional Encoding │ -        │ "You are word #3"      │ ✅   │
├─────┼─────────────────────┼──────────┼────────────────────────┼──────┤
│ 4   │ Q/K/V Projection    │ -        │ Make 3 question copies │ ✅   │
├─────┼─────────────────────┼──────────┼────────────────────────┼──────┤
│ 5   │ Attention Scores    │ ✓        │ "Who to look at?"      │ ✅   │
├─────┼─────────────────────┼──────────┼────────────────────────┼──────┤
│ 6   │ Feed-Forward (MLP)  │ -        │ "Think about it"       │ ✅   │
├─────┼─────────────────────┼──────────┼────────────────────────┼──────┤
│ 7   │ Layer Norm          │ -        │ Keep numbers stable    │ ✅   │
├─────┼─────────────────────┼──────────┼────────────────────────┼──────┤
│ 8   │ LM Head             │ -        │ Vector → vocab scores  │ ✅   │
├─────┼─────────────────────┼──────────┼────────────────────────┼──────┤
│ 9   │ Logits → Probs      │ ✓        │ Scores → percentages   │ ✅   │
├─────┼─────────────────────┼──────────┼────────────────────────┼──────┤
│ 10  │ Sampler/Decode      │ -        │ Pick word, return      │ ✅   │
└─────┴─────────────────────┴──────────┴────────────────────────┴──────┘
```

### 3.2 Verification Logic (apr check)

The `apr check` command performs **automated falsification** of the following invariants:

| Stage | Invariant ($H$) | Falsification Test (Rejection Criteria) |
|-------|-----------|-------------------|
| 1. Tokenizer | encode(decode(x)) = x | `encode(decode(x)) != x` |
| 2. Embedding | ‖v‖ > 0, no NaN | `Any(isnan(v)) OR norm(v) == 0` |
| 3. RoPE | θ = 10000, rotation applied | `cos/sin tables all zero` |
| 4. QKV | Output variance > 0 | `std(output) < epsilon` (Collapsed) |
| 5. Attention | Entropy > 0.1 | `entropy(attn) < 0.1` (Degenerate) |
| 6. FFN | SwiGLU non-linear | `output == input` (Identity/Bypass) |
| 7. LayerNorm | std(output) ≈ 1.0 | `abs(std(out) - 1.0) > 0.1` |
| 8. LM Head | shape = [vocab_size] | `dim(out) != vocab_size` |
| 9. Softmax | Σprobs = 1.0 ± 1e-5 | `abs(sum(p) - 1.0) > 1e-5` |
| 10. Sampler | Deterministic at temp=0 | `run(s, t=0) != run(s, t=0)` |

---

## 4. Modality Matrix

### 4.1 Model Size Coverage

The showcase validates across **five model sizes** to ensure architecture detection and inference correctness scales properly:

| Model | HuggingFace Path | Size | Layers | Hidden | Use Case |
|-------|------------------|------|--------|--------|----------|
| **0.5B** | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` | ~400MB | 24 | 896 | Edge/Mobile, Fast CI |
| **1B** | `Qwen/Qwen2.5-Coder-1B-Instruct-GGUF` | ~700MB | 24 | 1024 | Lightweight Development |
| **1.5B** | `Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF` | ~1GB | 28 | 1536 | Development, Primary QA |
| **7B** | `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF` | ~4GB | 32 | 3584 | Production, Perf Testing |
| **32B** | `Qwen/Qwen2.5-Coder-32B-Instruct-GGUF` | ~18GB | 64 | 5120 | Large-scale, High-memory |

**Architecture Detection Requirement:**
All model sizes MUST be detected as `Qwen2` architecture (not generic `Transformer`).
See: realizar#39 for 0.5B detection bug.

### 4.2 Modality Matrix (Per Model Size)

**Status Legend:** ✅ Verified | ❌ Broken/Missing | 🚧 Work in Progress

| Modality | 0.5B GGUF | 1B GGUF | 1.5B GGUF | 7B GGUF | 32B GGUF | APR | SafeTensors |
|----------|-----------|---------|-----------|---------|----------|-----|-------------|
| **apr run** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ (PAR-302) | ❌ (PAR-301) |
| **apr chat** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **apr serve** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **apr check** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **--trace** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Architecture** | Qwen2 | Qwen2 | Qwen2 | Qwen2 | Qwen2 | N/A | N/A |

**GGUF Score:** 25/25 (100%) — **PASSED** ✅
**Overall Score:** 25/35 (71%) — APR/SafeTensors formats pending

**QA Validation (2026-01-20):**
```
qa-serve.sh --all-models: 95/95 tests PASSED
├── 0.5B: 19/19 ✅
├── 1B:   19/19 ✅
├── 1.5B: 19/19 ✅
├── 7B:   19/19 ✅
└── 32B:  19/19 ✅
```

### 4.3 Performance Targets (Per Model Size)

These targets act as **falsifiable predictions**. If the system consistently fails to meet them on reference hardware (RTX 3090/4090, Modern AVX2 CPU), the optimization hypothesis is falsified.

**0.5B Model (Edge/Mobile):**
| Backend | Minimum | Target | Notes |
|---------|---------|--------|-------|
| CPU | 20 tok/s | 50 tok/s | Fast iteration |
| GPU | 200 tok/s | 500 tok/s | Realtime |

**1B Model (Lightweight Development):**
| Backend | Minimum | Target | Notes |
|---------|---------|--------|-------|
| CPU | 15 tok/s | 40 tok/s | Quick testing |
| GPU | 150 tok/s | 400 tok/s | Responsive |

**1.5B Model (Development):**
| Backend | Minimum | Target | Ollama Parity |
|---------|---------|--------|---------------|
| CPU | 10 tok/s | 25 tok/s | 1.0x |
| GPU Single | 100 tok/s | 300 tok/s | 2.0x |
| GPU Batch | 500 tok/s | 800 tok/s | 3.0x |

**7B Model (Production):**
| Backend | Minimum | Target | Notes |
|---------|---------|--------|-------|
| CPU | 2 tok/s | 8 tok/s | Memory-bound |
| GPU | 50 tok/s | 150 tok/s | VRAM: 6GB+ |
| GPU Batch | 200 tok/s | 400 tok/s | Batch size 4+ |

**32B Model (Large-scale):**
| Backend | Minimum | Target | Notes |
|---------|---------|--------|-------|
| CPU | 1 tok/s | 3 tok/s | Memory-bound, 32GB+ RAM |
| GPU | 25 tok/s | 80 tok/s | VRAM: 24GB+ (A100/H100) |
| GPU Batch | 100 tok/s | 250 tok/s | Multi-GPU recommended |

---

## 5. Implementation: apr-cli → realizar

### 5.1 Current State (GGUF WORKING, Formats Pending)

**GGUF Path (✅ VERIFIED):**
```rust
// apr-cli delegates to realizar for GGUF inference
// Validated via qa-serve.sh: 95/95 tests across 5 model sizes
// OpenAI-compatible API: /v1/chat/completions working
// Streaming (SSE): Working with [DONE] termination
// Tracing: X-Trace-Level header supported (brick/step/layer)
```

**APR/SafeTensors Path (❌ PENDING):**
```rust
// apr-cli/src/commands/run.rs - 1600 lines of DUPLICATED inference code
fn run_gguf_model(...) {
    // Duplicates realizar's inference logic for non-GGUF formats
    // No spinner, verbose by default
    // Separate code path from realizar
}
```

### 5.2 Target State (UNIFIED)

```rust
// apr-cli/src/commands/run.rs - ~100 lines, delegates to realizar
pub async fn execute(args: RunArgs) -> Result<()> {
    // 1. Resolve model (local, hf://, pacha://)
    let model_path = resolve_model(&args.source).await?;

    // 2. Build realizar config
    let config = realizar::InferenceConfig {
        model_path,
        prompt: args.prompt,
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        verbose: args.verbose,
        trace: args.trace.map(|t| realizar::TraceConfig {
            enabled: true,
            steps: t.steps,
            output: t.output,
        }),
        gpu: !args.no_gpu,
    };

    // 3. Run inference via realizar (has spinner, clean output)
    let result = realizar::run_inference(config).await?;

    // 4. Output already handled by realizar
    Ok(())
}
```

### 5.3 realizar Public API

```rust
// realizar/src/lib.rs - Public inference API

/// Run model inference with Ollama-style UX
pub async fn run_inference(config: InferenceConfig) -> Result<InferenceResult>;

/// Start interactive chat session
pub async fn run_chat(config: ChatConfig) -> Result<()>;

/// Start HTTP inference server
pub async fn run_serve(config: ServeConfig) -> Result<()>;

/// Run 10-stage pipeline verification
pub fn check_model(path: &Path) -> Result<CheckResult>;

/// Configuration for inference
pub struct InferenceConfig {
    pub model_path: PathBuf,
    pub prompt: Option<String>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub verbose: bool,
    pub trace: Option<TraceConfig>,
    pub gpu: bool,
}

/// Trace configuration (AWS Step Functions parity)
pub struct TraceConfig {
    pub enabled: bool,
    pub steps: Option<Vec<TraceStep>>,
    pub output: Option<PathBuf>,
    pub verbose: bool,
}

/// Inference result
pub struct InferenceResult {
    pub text: String,
    pub tokens_generated: usize,
    pub duration_ms: f64,
    pub tokens_per_second: f64,
    pub trace: Option<TraceOutput>,
}
```

---

## 6. 300-Point Popperian Falsification Checklist

**Protocol:** A single check failure constitutes a successful falsification of the release candidate. All boxes must be checked to accept the hypothesis $H_1$.

### Section I: CLI & UX (40 Points)

#### I-A: Basic Commands (20 pts)
- [ ] **F-CLI-001**: `apr run model.gguf "prompt"` executes without panic
- [ ] **F-CLI-002**: `apr run` without model shows usage help
- [ ] **F-CLI-003**: `apr run nonexistent.gguf` shows "file not found"
- [ ] **F-CLI-004**: `apr chat model.gguf` enters interactive mode
- [ ] **F-CLI-005**: `apr serve model.gguf` starts HTTP server
- [ ] **F-CLI-006**: `apr check model.gguf` runs 10-stage verification
- [ ] **F-CLI-007**: `--help` shows all options
- [ ] **F-CLI-008**: `--version` shows version
- [ ] **F-CLI-009**: `-v/--verbose` enables verbose output
- [ ] **F-CLI-010**: `-q/--quiet` suppresses non-error output
- [ ] **F-CLI-011**: `--max-tokens N` limits generation
- [ ] **F-CLI-012**: `--temperature T` affects sampling
- [ ] **F-CLI-013**: `--gpu` forces GPU path
- [ ] **F-CLI-013b**: `apr chat` has `--gpu` flag (consistency)
- [ ] **F-CLI-014**: `--no-gpu` forces CPU path
- [ ] **F-CLI-014b**: `apr chat` has `--no-gpu` flag (consistency)
- [ ] **F-CLI-015**: `--json` outputs JSON format
- [ ] **F-CLI-016**: `--trace` enables tracing
- [ ] **F-CLI-017**: `--trace-output FILE` saves trace
- [ ] **F-CLI-018**: `--trace-verbose` shows tensor values
- [ ] **F-CLI-019**: Ctrl+C gracefully terminates
- [ ] **F-CLI-020**: Exit code 0 on success, non-zero on failure

#### I-B: Ollama-Style UX (Normal vs. Noisy) (20 pts)

**Normal Mode (Default):** *Zero noise. Only the spinner and the final response.*
- [ ] **F-UX-021**: Spinner shows during model loading (no --verbose)
- [ ] **F-UX-022**: Spinner clears before output
- [ ] **F-UX-023**: Clean output shows ONLY response text
- [ ] **F-UX-024**: **NOISY-GUARD**: No debug tags like `[PAR-*]`, `[BIAS-FIX]`, or `[DEBUG]` in output
- [ ] **F-UX-025**: **NOISY-GUARD**: No internal timing logs (e.g., "Layer 1 took 5ms") in output
- [ ] **F-UX-026**: **NOISY-GUARD**: No backend initialization noise (e.g., "CUDA device 0 initialized")

**Noisy Mode (--verbose):** *Complete transparency. All metadata and internal state.*
- [ ] **F-UX-027**: Verbose mode shows loading details (source, format, tensors)
- [ ] **F-UX-028**: Verbose mode shows architecture info (hidden size, layers, heads)
- [ ] **F-UX-029**: Verbose mode shows prompt token count
- [ ] **F-UX-030**: Verbose mode shows performance stats (tok/s, total duration)
- [ ] **F-UX-031**: Verbose mode shows backend dispatch info (AVX/CUDA/SIMD)
- [ ] **F-UX-032**: Chat mode shows prompt indicator (`>>>`)
- [ ] **F-UX-033**: Chat mode supports `/exit` and `/clear`
- [ ] **F-UX-034**: Server mode shows endpoint URLs and stopping instructions
- [ ] **F-UX-035**: Colors work on TTY, disabled on non-TTY
- [ ] **F-UX-036**: UTF-8 and emoji render correctly without mojibake
- [ ] **F-UX-037**: Error messages are user-friendly (no raw Rust panics)
- [ ] **F-UX-038**: Progress bar shown for downloads/long operations
- [ ] **F-UX-039**: Output streaming works (text appears as generated)
- [ ] **F-UX-040**: Final stats summary suppressed unless `--verbose` is used

### Section II: Model Format Parity (50 Points)

#### II-A: GGUF Support (20 pts)
- [ ] **F-GGUF-041**: Load Q4_K_M quantization
- [ ] **F-GGUF-042**: Load Q4_0 quantization
- [ ] **F-GGUF-043**: Load Q5_K_M quantization
- [ ] **F-GGUF-044**: Load Q6_K quantization
- [ ] **F-GGUF-045**: Load Q8_0 quantization
- [ ] **F-GGUF-046**: Load F16 weights
- [ ] **F-GGUF-047**: Load F32 weights
- [ ] **F-GGUF-048**: Read GGUF metadata
- [ ] **F-GGUF-049**: Use GGUF tokenizer
- [ ] **F-GGUF-050**: Handle BOS/EOS tokens
- [ ] **F-GGUF-051**: Support chat templates (ChatML, LLaMA)
- [ ] **F-GGUF-052**: Memory-map large files
- [ ] **F-GGUF-053**: Detect architecture (qwen2, llama, etc.)
- [ ] **F-GGUF-054**: Handle vocab size mismatch gracefully
- [ ] **F-GGUF-055**: Support GQA (grouped query attention)
- [ ] **F-GGUF-056**: Support RoPE scaling
- [ ] **F-GGUF-057**: Validate tensor shapes
- [ ] **F-GGUF-058**: Error on corrupted file
- [ ] **F-GGUF-059**: Error on unsupported architecture
- [ ] **F-GGUF-060**: Same output as llama.cpp (deterministic)

#### II-B: APR Support (15 pts)
- [ ] **F-APR-061**: Load APR v2 format
- [ ] **F-APR-062**: Load INT4 quantized tensors
- [ ] **F-APR-063**: Load INT8 quantized tensors
- [ ] **F-APR-064**: Load F16 tensors
- [ ] **F-APR-065**: Load F32 tensors
- [ ] **F-APR-066**: Read APR metadata
- [ ] **F-APR-067**: Handle compression (LZ4, ZSTD)
- [ ] **F-APR-068**: Auto-dequantize to F32
- [ ] **F-APR-069**: Tensor name mapping works
- [ ] **F-APR-070**: Error on corrupted bundle
- [ ] **F-APR-071**: Error on version mismatch
- [ ] **F-APR-072**: Support streaming read
- [ ] **F-APR-073**: Validate checksums
- [ ] **F-APR-074**: Same output as GGUF (same model)
- [ ] **F-APR-075**: APR → GGUF round-trip preserves accuracy

#### II-C: SafeTensors Support (15 pts)
- [ ] **F-ST-076**: Load .safetensors files
- [ ] **F-ST-077**: Load F16 tensors
- [ ] **F-ST-078**: Load F32 tensors
- [ ] **F-ST-079**: Load BF16 tensors
- [ ] **F-ST-080**: Read metadata JSON
- [ ] **F-ST-081**: Memory-map for zero-copy
- [ ] **F-ST-082**: Handle config.json for architecture
- [ ] **F-ST-083**: Handle tokenizer.json
- [ ] **F-ST-084**: Handle tokenizer_config.json
- [ ] **F-ST-085**: Support HuggingFace model layout
- [ ] **F-ST-086**: Support sharded models (model-00001-of-00002)
- [ ] **F-ST-087**: Error on missing tensors
- [ ] **F-ST-088**: Error on shape mismatch
- [ ] **F-ST-089**: Same output as transformers library
- [ ] **F-ST-090**: Support nested model directories

### Section III: Backend Parity (50 Points)

#### III-A: CPU Backend (25 pts)
- [ ] **F-CPU-091**: AVX2 SIMD acceleration works
- [ ] **F-CPU-092**: AVX-512 SIMD acceleration works (if available)
- [ ] **F-CPU-093**: NEON SIMD works (ARM)
- [ ] **F-CPU-094**: Scalar fallback works (no SIMD)
- [ ] **F-CPU-095**: Multi-threaded inference
- [ ] **F-CPU-096**: Thread count configurable
- [ ] **F-CPU-097**: Memory-efficient (< 2x model size)
- [ ] **F-CPU-098**: ≥ 10 tok/s on Qwen 1.5B Q4_K_M
- [ ] **F-CPU-099**: ≥ 25 tok/s target
- [ ] **F-CPU-100**: No memory leaks (valgrind clean)
- [ ] **F-CPU-101**: Deterministic output (same seed)
- [ ] **F-CPU-102**: KV cache works correctly
- [ ] **F-CPU-103**: Prefill phase optimized
- [ ] **F-CPU-104**: Decode phase optimized
- [ ] **F-CPU-105**: Handles long contexts (>2K tokens)
- [ ] **F-CPU-106**: Handles batch size 1
- [ ] **F-CPU-107**: Graceful OOM handling
- [ ] **F-CPU-108**: Works on Linux x86_64
- [ ] **F-CPU-109**: Works on macOS ARM64
- [ ] **F-CPU-110**: Works on Windows x86_64
- [ ] **F-CPU-111**: Q4_K dequantization correct
- [ ] **F-CPU-112**: Q6_K dequantization correct
- [ ] **F-CPU-113**: F16→F32 conversion correct
- [ ] **F-CPU-114**: RMSNorm numerically stable
- [ ] **F-CPU-115**: Softmax numerically stable

#### III-B: GPU Backend (25 pts)
- [ ] **F-GPU-116**: CUDA acceleration works
- [ ] **F-GPU-117**: Supports CUDA compute 7.0+ (V100+)
- [ ] **F-GPU-118**: Supports CUDA compute 8.0+ (A100+)
- [ ] **F-GPU-119**: Supports CUDA compute 8.9+ (RTX 4090)
- [ ] **F-GPU-120**: ≥ 100 tok/s single stream
- [ ] **F-GPU-121**: ≥ 300 tok/s target single
- [ ] **F-GPU-122**: ≥ 500 tok/s batched
- [ ] **F-GPU-123**: ≥ 800 tok/s target batched
- [ ] **F-GPU-124**: 2x Ollama parity achieved
- [ ] **F-GPU-125**: GPU memory usage < model size + 20%
- [ ] **F-GPU-126**: No CUDA memory leaks
- [ ] **F-GPU-127**: Graceful OOM handling
- [ ] **F-GPU-128**: Multi-GPU support (future)
- [ ] **F-GPU-129**: GPU-resident KV cache
- [ ] **F-GPU-130**: Fused dequant+matmul kernels
- [ ] **F-GPU-131**: FlashAttention-style attention
- [ ] **F-GPU-132**: Deterministic output (same seed)
- [ ] **F-GPU-133**: CPU↔GPU same output (within tolerance)
- [ ] **F-GPU-134**: --gpu flag forces GPU path
- [ ] **F-GPU-134b**: Default to GPU if available (no hardcoded force_cpu)
- [ ] **F-GPU-135**: Fallback to CPU if no GPU
- [ ] **F-GPU-136**: Clear error if CUDA unavailable
- [ ] **F-GPU-137**: nvidia-smi shows expected VRAM
- [ ] **F-GPU-138**: Works with CUDA 11.x
- [ ] **F-GPU-139**: Works with CUDA 12.x
- [ ] **F-GPU-140**: PTX kernels compile at runtime

### Section IV: Correctness (50 Points)

#### IV-A: Math Correctness (25 pts)
- [ ] **F-MATH-141**: 2+2=4 test passes
- [ ] **F-MATH-142**: Basic arithmetic correct
- [ ] **F-MATH-143**: Code generation produces valid syntax
- [ ] **F-MATH-144**: Python code executes correctly
- [ ] **F-MATH-145**: Function definitions correct
- [ ] **F-MATH-146**: UTF-8 Chinese output correct
- [ ] **F-MATH-147**: No mojibake in multilingual output
- [ ] **F-MATH-148**: Temperature=0 is deterministic
- [ ] **F-MATH-149**: Temperature=1 produces variety
- [ ] **F-MATH-150**: Top-k sampling works
- [ ] **F-MATH-151**: Top-p (nucleus) sampling works
- [ ] **F-MATH-152**: Repetition penalty works
- [ ] **F-MATH-153**: EOS token stops generation
- [ ] **F-MATH-154**: Max tokens limit respected
- [ ] **F-MATH-155**: Empty prompt handled
- [ ] **F-MATH-156**: Whitespace-only prompt handled
- [ ] **F-MATH-157**: Very long prompt handled
- [ ] **F-MATH-158**: Special characters handled
- [ ] **F-MATH-159**: Embedding vectors non-zero
- [ ] **F-MATH-160**: Attention weights sum to 1
- [ ] **F-MATH-161**: Softmax output sums to 1
- [ ] **F-MATH-162**: No NaN in forward pass
- [ ] **F-MATH-163**: No Inf in forward pass
- [ ] **F-MATH-164**: LayerNorm output normalized
- [ ] **F-MATH-165**: RoPE rotation applied correctly

#### IV-B: Pipeline Correctness (25 pts)
- [ ] **F-PIPE-166**: Tokenizer encode/decode round-trip
- [ ] **F-PIPE-166b**: No tokenizer artifacts (e.g. 'Ġ', '!!!')
- [ ] **F-PIPE-167**: BOS token prepended
- [ ] **F-PIPE-168**: EOS token recognized
- [ ] **F-PIPE-169**: Chat template applied correctly
- [ ] **F-PIPE-170**: System prompt works
- [ ] **F-PIPE-171**: Multi-turn conversation works
- [ ] **F-PIPE-172**: KV cache populated during prefill
- [ ] **F-PIPE-173**: KV cache used during decode
- [ ] **F-PIPE-174**: KV cache cleared between requests
- [ ] **F-PIPE-175**: Attention mask correct
- [ ] **F-PIPE-176**: Causal masking enforced
- [ ] **F-PIPE-177**: Position IDs correct
- [ ] **F-PIPE-178**: Layer output shapes correct
- [ ] **F-PIPE-179**: Final logits shape = vocab_size
- [ ] **F-PIPE-180**: Sampling respects temperature
- [ ] **F-PIPE-181**: Token decoded correctly
- [ ] **F-PIPE-182**: Streaming output works
- [ ] **F-PIPE-183**: Batch inference produces correct output
- [ ] **F-PIPE-184**: Different prompts give different outputs
- [ ] **F-PIPE-185**: Same prompt+seed gives same output
- [ ] **F-PIPE-186**: Context window respected
- [ ] **F-PIPE-187**: Truncation warning shown
- [ ] **F-PIPE-188**: Generation stops at max_tokens
- [ ] **F-PIPE-189**: Generation stops at EOS
- [ ] **F-PIPE-190**: Full pipeline matches llama.cpp output

### Section V: Tracing & Observability (40 Points)

This section enforces the strict separation between **Runtime Observation** (seeing what happens) and **Static Verification** (proving it works).

#### V-A: Runtime Tracing (The Flight Recorder) (20 pts)
*Capture dynamic execution paths during live inference. Analogous to a black box flight recorder.*

- [ ] **F-TRACE-191**: --trace produces JSON output
- [ ] **F-TRACE-192**: JSON is valid (parseable by jq)
- [ ] **F-TRACE-193**: TaskStateEntered events present
- [ ] **F-TRACE-194**: TaskStateExited events present
- [ ] **F-TRACE-195**: Events have timestamps (ISO 8601)
- [ ] **F-TRACE-196**: Events have unique IDs
- [ ] **F-TRACE-197**: Exit events link to entry events
- [ ] **F-TRACE-198**: Input/output captured for each step
- [ ] **F-TRACE-199**: Tensor stats (min/max/mean) included
- [ ] **F-TRACE-200**: Schema version field present
- [ ] **F-TRACE-201**: Model metadata in trace
- [ ] **F-TRACE-202**: Run config in trace
- [ ] **F-TRACE-203**: --trace-output FILE works
- [ ] **F-TRACE-204**: --trace-verbose shows tensor values
- [ ] **F-TRACE-205**: Trace doesn't alter inference result
- [ ] **F-TRACE-206**: Trace overhead < 50%
- [ ] **F-TRACE-207**: NaN/Inf flagged in stats
- [ ] **F-TRACE-208**: Large arrays truncated
- [ ] **F-TRACE-209**: Python json.load() compatible
- [ ] **F-TRACE-210**: AWS Step Functions schema parity

#### V-B: Static Verification (The Pre-Flight Checklist) (20 pts)
*Diagnostic integrity check of model components. Does NOT generate text. Analogous to a pilot's pre-flight check.*

- [ ] **F-CHECK-211**: apr check runs without crash
- [ ] **F-CHECK-212**: Stage 1 (Tokenizer) verified
- [ ] **F-CHECK-213**: Stage 2 (Embedding) verified
- [ ] **F-CHECK-214**: Stage 3 (RoPE) verified
- [ ] **F-CHECK-215**: Stage 4 (QKV) verified
- [ ] **F-CHECK-216**: Stage 5 (Attention) verified
- [ ] **F-CHECK-217**: Stage 6 (FFN) verified
- [ ] **F-CHECK-218**: Stage 7 (LayerNorm) verified
- [ ] **F-CHECK-219**: Stage 8 (LM Head) verified
- [ ] **F-CHECK-220**: Stage 9 (Softmax) verified
- [ ] **F-CHECK-221**: Stage 10 (Sampler) verified
- [ ] **F-CHECK-222**: 10/10 STAGES PASSED message
- [ ] **F-CHECK-223**: Failed stage shows error
- [ ] **F-CHECK-224**: ELI5 descriptions shown
- [ ] **F-CHECK-225**: Table format renders correctly
- [ ] **F-CHECK-226**: Works for GGUF models
- [ ] **F-CHECK-227**: Works for APR models
- [ ] **F-CHECK-228**: Works for SafeTensors models
- [ ] **F-CHECK-229**: GPU path verified
- [ ] **F-CHECK-230**: CPU path verified

### Section VI: Server (HTTP API) (30 Points)

- [ ] **F-SERVE-231**: apr serve starts server
- [ ] **F-SERVE-232**: GET /health returns healthy
- [ ] **F-SERVE-233**: GET /metrics returns Prometheus format
- [ ] **F-SERVE-234**: POST /generate works
- [ ] **F-SERVE-235**: POST /v1/completions (OpenAI compat)
- [ ] **F-SERVE-236**: POST /v1/chat/completions (OpenAI compat)
- [ ] **F-SERVE-237**: Streaming (SSE) works
- [ ] **F-SERVE-238**: Batch inference works
- [ ] **F-SERVE-239**: Concurrent requests handled
- [ ] **F-SERVE-240**: Request timeout configurable
- [ ] **F-SERVE-241**: Max tokens enforced
- [ ] **F-SERVE-242**: Temperature parameter works
- [ ] **F-SERVE-243**: Stop sequences work
- [ ] **F-SERVE-244**: Error responses proper JSON
- [ ] **F-SERVE-245**: CORS headers configurable
- [ ] **F-SERVE-246**: Port configurable (--port)
- [ ] **F-SERVE-247**: Host configurable (--host)
- [ ] **F-SERVE-248**: Graceful shutdown on SIGINT
- [ ] **F-SERVE-249**: Model info endpoint (/model)
- [ ] **F-SERVE-250**: apr_inference_count metric increments
- [ ] **F-SERVE-251**: apr_tokens_generated metric increments
- [ ] **F-SERVE-252**: apr_inference_duration_seconds histogram
- [ ] **F-SERVE-253**: Memory doesn't grow unbounded
- [ ] **F-SERVE-254**: Handles malformed JSON gracefully
- [ ] **F-SERVE-255**: Handles missing fields gracefully
- [ ] **F-SERVE-256**: Works with curl
- [ ] **F-SERVE-257**: Works with httpie
- [ ] **F-SERVE-258**: Works with Python requests
- [ ] **F-SERVE-259**: Load test (100 concurrent) passes
- [ ] **F-SERVE-260**: Tracing works in serve mode

### Section VII: Jidoka (Error Detection) (20 Points)

- [ ] **F-JID-261**: Vocab size mismatch detected
- [ ] **F-JID-262**: Embedding dimension mismatch detected
- [ ] **F-JID-263**: Attention head count mismatch detected
- [ ] **F-JID-264**: Softmax overflow detected
- [ ] **F-JID-265**: Invalid UTF-8 sequence detected
- [ ] **F-JID-266**: Temperature < 0 warning
- [ ] **F-JID-267**: Temperature > 2 warning
- [ ] **F-JID-268**: Top-p out of range warning
- [ ] **F-JID-269**: High perplexity spike detected
- [ ] **F-JID-270**: Repeated token loop detected
- [ ] **F-JID-271**: Premature EOS detected
- [ ] **F-JID-272**: OOV token hint provided
- [ ] **F-JID-273**: Shape mismatch hint provided
- [ ] **F-JID-274**: NaN logits hint provided
- [ ] **F-JID-275**: CPU fallback logged
- [ ] **F-JID-276**: CUDA OOM logged
- [ ] **F-JID-277**: File not found logged
- [ ] **F-JID-278**: Invalid format logged
- [ ] **F-JID-279**: Network error logged (HF download)
- [ ] **F-JID-280**: Summary of errors/warnings at end

### Section VIII: Integration & Ecosystem (20 Points)

- [ ] **F-INT-281**: apr-cli uses realizar for inference
- [ ] **F-INT-282**: realizar uses trueno for compute
- [ ] **F-INT-283**: presentar-terminal spinner works
- [ ] **F-INT-284**: pacha model registry works
- [ ] **F-INT-285**: hf:// URLs resolve
- [ ] **F-INT-286**: pacha:// URLs resolve
- [ ] **F-INT-287**: Local paths work
- [ ] **F-INT-288**: HTTP URLs work
- [ ] **F-INT-289**: Model caching works (~/.apr/cache)
- [ ] **F-INT-290**: --offline mode works
- [ ] **F-INT-291**: cargo install apr-cli works
- [ ] **F-INT-292**: No duplicate code between apr-cli and realizar
- [ ] **F-INT-293**: Shared chat template logic
- [ ] **F-INT-294**: Shared tokenizer logic
- [ ] **F-INT-295**: Shared quantization logic
- [ ] **F-INT-296**: Version compatibility checked
- [ ] **F-INT-297**: Dependency versions aligned
- [ ] **F-INT-298**: CI/CD passes for all crates
- [ ] **F-INT-299**: Documentation complete
- [ ] **F-INT-300**: Examples work out of box

---

## 7. Implementation Roadmap

### Phase 1: Architecture Refactor (Week 1)
1. Add public inference API to realizar
2. Add presentar-terminal spinner (✅ DONE)
3. Refactor apr-cli run.rs to delegate to realizar
4. Refactor apr-cli chat.rs to delegate to realizar

### Phase 2: Tracing Integration (Week 2)
1. Move InferenceTracer to realizar
2. Implement AWS Step Functions schema
3. Add --trace flag handling in realized
4. Implement 10-stage apr check

### Phase 3: Parity Testing (Week 3)
1. Run 300-point falsification checklist
2. Fix all failing tests
3. Performance benchmarking
4. Documentation

### Phase 4: Release (Week 4)
1. Publish realizar 0.7.0
2. Publish apr-cli 0.3.0
3. Update showcase demo
4. Final QA sign-off

---

## 8. Definition of Done

1. `scripts/showcase-qa.sh` exits 0
2. 300-point falsification: ≥ 290 pass
3. All modalities (run/chat/serve × formats × backends) work
4. GPU ≥ 2x Ollama throughput
5. apr-cli has no duplicated inference code
6. Ollama-style UX (spinner, clean output)
7. Tracing works across all paths

---

## 9. Falsification Criteria & Pivot Strategy

We define "Success" not as a working feature, but as the **failure to falsify the hypothesis**.

### Falsification Triggers (Refutation of Release)
If ANY of the following occur, the Release Candidate is **REJECTED**:
*   **F-CRIT-001**: Any modality (run/chat/serve) fails to execute on reference hardware (Level 2).
*   **F-CRIT-002**: `apr-cli` is found to contain independent inference logic (Level 3).
*   **F-CRIT-003**: GPU throughput is consistently < 1.0x Ollama (Level 4).
*   **F-CRIT-004**: Falsification Score < 290/300.
*   **F-CRIT-005**: `apr check` passes but model generates garbage (Invalid Falsification Test).
*   **F-CRIT-301**: SafeTensors support missing (PAR-301). 🛑 BLOCKING
*   **F-CRIT-302**: APR format support missing (PAR-302). 🛑 BLOCKING
*   ~~**F-CRIT-303**: 0.5B model coherency failure (PAR-303).~~ ✅ RESOLVED (2026-01-20)

### Pivot Strategy (In case of Level 4 Failure)
If the Unified Architecture is falsified at Level 4 (Structural/Performance limits):
1.  **Stop** all feature work.
2.  **Revert** to `trueno` direct binding (bypass `realizar` middleware) for critical paths.
3.  **Document** the overhead cost of the abstraction layer.
4.  **Issue** Post-Mortem: "Why Rust Abstractions Failed to Scale".

---

## Appendix D: Implementation Breakdown

The Sovereign AI Stack is composed of modular crates working in concert.

| Component | Repository Path | Role | Key Technologies |
|-----------|-----------------|------|------------------|
| **aprender** | `src/` (root) | ML Library | AutoDiff, Algorithms, .apr Format |
| **realizar** | `../realizar` | Inference Engine | KV Cache, HTTP Server, Scheduler |
| **trueno** | `../trueno` | Compute Kernels | AVX-512, CUDA PTX, Tensor Ops |
| **apr-cli** | `crates/apr-cli` | User Interface | CLI/TUI, Model Management |
| **renacer** | `../renacer` | Profiling | Syscall Tracing, GPU Metrics |
| **pacha** | `../pacha` | Registry | Model Versioning, Deduplication |

**Note:** `realizar` and `trueno` are integrated as local path dependencies during development (see `crates/apr-cli/Cargo.toml`) but are published as separate crates.

---

## Appendix E: ML Tuning Taxonomy

Optimization in the aprender ecosystem follows a strict 5-level hierarchy.

### Level 1: Kernel Tuning (Compute)
**Scope:** `trueno`
- SIMD instruction selection (AVX2 vs AVX-512).
- CUDA PTX kernel optimization (coalesced access, shared memory).
- Register blocking and tiling for matmul.
- **Goal:** Maximize FLOPS/IOPS on specific hardware.

### Level 2: System Tuning (Runtime)
**Scope:** `realizar`, `trueno-zram`
- Memory management (paging, allocation strategies).
- I/O throughput (batching, prefetching).
- Thread scheduling and affinity.
- KV cache compression (ZRAM).
- **Goal:** Minimize latency and maximize throughput/utilization.

### Level 3: Model Tuning (Representation)
**Scope:** `aprender` (format), `realizar` (inference)
- Quantization (Q4_K, Q8_0, INT8).
- Pruning (magnitude-based, structured).
- Distillation (teacher-student).
- **Goal:** Reduce model size and compute requirements without retraining.

### Level 4: Hyperparameter Tuning (Training)
**Scope:** `aprender` (AutoML)
- Grid Search, Random Search, Bayesian Optimization.
- **Status:** **Out of Scope** for runtime inference (pre-training only).

### Level 5: Learned Auto-Tuning (Compiler-in-the-Loop)
**Scope:** `aprender-citl`, `trueno`
- Dynamic kernel selection based on input shapes.
- JIT compilation of fused kernels.
- Automated exploration of tuning parameters (tile sizes, unroll factors).
- **Goal:** Self-optimizing runtime that adapts to workload.
