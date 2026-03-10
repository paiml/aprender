# apr - APR Model Operations CLI

The `apr` command-line tool provides inspection, debugging, validation, and comparison capabilities for `.apr` model files. It follows Toyota Way principles for quality and visibility.

## Installation

```bash
cargo install --path crates/apr-cli
```

Or build from the workspace:

```bash
cargo build --release -p apr-cli
```

The binary will be available at `target/release/apr`.

## Commands Overview

| Command | Description | Toyota Way Principle |
|---------|-------------|---------------------|
| `run` | Run model directly (auto-download, cache, execute) | Just-in-Time Production |
| `serve plan` | Pre-flight capacity planning (VRAM, throughput, contracts) | Poka-Yoke (Error Prevention) |
| `serve run` | Start inference server with GPU acceleration | Just-in-Time Production |
| `chat` | Interactive chat with language models | Genchi Genbutsu (Go and See) |
| `inspect` | View model metadata and structure | Genchi Genbutsu (Go and See) |
| `debug` | Debug output with optional drama mode | Visualization |
| `validate` | Validate integrity with quality scoring | Jidoka (Built-in Quality) |
| `diff` | Compare two models | Kaizen (Continuous Improvement) |
| `tensors` | List tensor names, shapes, and statistics | Genchi Genbutsu (Go to the Source) |
| `trace` | Layer-by-layer analysis with anomaly detection | Visualization |
| `lint` | Check for best practices and conventions | Jidoka (Built-in Quality) |
| `probar` | Export for visual regression testing | Standardization |
| `import` | Import from HuggingFace, local files, or URLs | Automation |
| `export` | Export to SafeTensors, GGUF formats | Automation |
| `pull` | Download and cache model (Ollama-style UX) | Automation |
| `list` | List cached models | Visibility |
| `rm` | Remove model from cache | Standardization |
| `convert` | Quantization (int8, int4, fp16) and optimization | Kaizen |
| `merge` | Merge models (average, weighted strategies) | Kaizen |
| `tree` | Model architecture tree view | Visualization |
| `hex` | Hex dump tensor data | Genchi Genbutsu |
| `flow` | Data flow visualization | Visualization |
| `bench` | Benchmark throughput (spec H12: >= 10 tok/s) | Measurement |
| `eval` | Evaluate model perplexity (spec H13: PPL <= 20) | Measurement |
| `profile` | Deep profiling with Roofline analysis | Genchi Genbutsu |
| `qa` | Falsifiable QA checklist for model releases | Jidoka |
| `qualify` | Cross-subcommand smoke test (does every tool handle this model?) | Jidoka |
| `showcase` | Qwen2.5-Coder showcase demo | Standardization |
| `check` | Model self-test: 10-stage pipeline integrity | Jidoka |
| `publish` | Publish model to HuggingFace Hub | Automation |
| `cbtop` | ComputeBrick pipeline monitor | Visualization |
| `compare-hf` | Compare local model (APR/GGUF/SafeTensors) against HuggingFace | Jidoka |
| `explain` | Explain errors, architecture, and tensors | Knowledge Sharing |
| `tui` | Interactive terminal UI | Visualization |
| `canary` | Regression testing via tensor statistics | Jidoka |
| `finetune` | Fine-tune model with LoRA (classification, test-gen) | Kaizen |
| `tune` | Hyperparameter search for fine-tuning | Kaizen |

## Serve Command

The `serve` command has two subcommands: `plan` (pre-flight capacity check) and
`run` (start the server).

### Serve Run

Start an OpenAI-compatible inference server with optional GPU acceleration.

```bash
# Basic server (CPU)
apr serve run model.gguf --port 8080

# GPU-accelerated server
apr serve run model.gguf --port 8080 --gpu

# Batched GPU mode (2.9x faster than Ollama)
apr serve run model.gguf --port 8080 --gpu --batch
```

### Performance

| Mode | Model | Throughput | vs Ollama | Memory |
|------|-------|------------|-----------|--------|
| GPU (APR Q4K, GH-88) | Qwen 1.5B | **240 tok/s** | — | 1.5 GB |
| GPU (batched M=16) | Qwen 1.5B | ~850 tok/s | 2.9x | 1.9 GB |
| GPU (single GGUF) | Qwen 7B | ~68 tok/s | 0.2x | 5.5 GB |
| CPU (baseline) | Qwen 1.5B | ~18 tok/s | 0.05x | 1.1 GB |
| Ollama | Qwen 1.5B | ~333 tok/s | 1.0x | - |

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |
| `/v1/chat/completions` | POST | OpenAI-compatible chat |
| `/v1/completions` | POST | OpenAI-compatible completions |
| `/generate` | POST | Native generation endpoint |

### Example Request

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

### Tracing Headers

Use the `X-Trace-Level` header for performance debugging:

```bash
# Token-level timing
curl -H "X-Trace-Level: brick" http://localhost:8080/v1/chat/completions ...

# Layer-level timing
curl -H "X-Trace-Level: layer" http://localhost:8080/v1/chat/completions ...
```

### Serve Plan (Pre-flight Capacity Planning)

Before downloading or launching a model, `apr serve plan` computes VRAM budget,
throughput estimates, and contract checks. Header-only — no weights loaded.

```bash
# Plan from a local file
apr serve plan model.gguf --gpu

# Plan from a HuggingFace repo (fetches only ~2KB config.json)
apr serve plan hf://Qwen/Qwen2.5-Coder-1.5B-Instruct --gpu

# Bare org/repo also works (auto-detected as HuggingFace)
apr serve plan microsoft/phi-2 --gpu --quant Q4_K_M

# JSON output for tooling
apr serve plan hf://mistralai/Mistral-7B-Instruct-v0.3 --gpu --format json
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--gpu` | Detect GPU via nvidia-smi for VRAM budget |
| `--quant <Q>` | Quantization override for HF models (e.g., Q4_K_M, Q6_K, F16) |
| `--batch-size <N>` | Target batch size for throughput estimation (default: 1) |
| `--seq-len <N>` | Sequence length for KV cache estimation (default: 4096) |
| `--format <F>` | Output format: text, json, yaml (default: text) |

**Contracts verified:**
- `BUDGET-001`: Total VRAM fits within 95% safety margin
- `BUDGET-002`: Model weights loadable contiguous
- `BUDGET-003`: KV cache fits at batch=1
- `BUDGET-004`: Target batch size achievable (when batch > 1)

**Verdict:** READY, WARNINGS, or BLOCKED.

### Tool Calling (GH-160)

The server supports OpenAI-compatible tool calling, allowing models to invoke external functions.

**Define tools in your request:**

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the current weather for a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string", "description": "City name"},
              "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
          }
        }
      }
    ],
    "max_tokens": 100
  }'
```

**Response with tool call:**

```json
{
  "id": "chatcmpl-abc123",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_xyz789",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"Tokyo\", \"unit\": \"celsius\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

**Multi-turn with tool result:**

After executing the tool, send the result back:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "What is the weather in Tokyo?"},
      {"role": "assistant", "content": null, "tool_calls": [{"id": "call_xyz789", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"Tokyo\"}"}}]},
      {"role": "tool", "tool_call_id": "call_xyz789", "content": "{\"temperature\": 22, \"condition\": \"sunny\"}"}
    ],
    "max_tokens": 100
  }'
```

The model will then generate a response incorporating the tool result.

**Tool choice control:**

```json
{
  "tool_choice": "auto"
}
```

Options: `"auto"` (default), `"none"` (disable tools), or `{"type": "function", "function": {"name": "specific_tool"}}`.

**Example code:** See `cargo run --example tool_calling_demo` for a complete Rust example.

## Chat Command

Interactive chat with language models (supports GGUF, APR, SafeTensors).

```bash
# Interactive chat (GPU by default)
apr chat model.gguf

# Force CPU inference
apr chat model.gguf --no-gpu

# Adjust generation parameters
apr chat model.gguf --temperature 0.7 --top-p 0.9 --max-tokens 512
```

## Inspect Command

View model metadata, structure, and flags without loading the full payload.

```bash
# Basic inspection
apr inspect model.apr

# JSON output for automation
apr inspect model.apr --json

# Show vocabulary details
apr inspect model.apr --vocab

# Show filter/security details
apr inspect model.apr --filters

# Show weight statistics
apr inspect model.apr --weights
```

### Example Output

```
=== model.apr ===

  Type: LinearRegression
  Version: 1.0
  Size: 2.5 KiB
  Compressed: 1.2 KiB (ratio: 2.08x)
  Flags: COMPRESSED | SIGNED
  Created: 2025-01-15T10:30:00Z
  Framework: aprender 0.18.2
  Name: Boston Housing Predictor
  Description: Linear regression model for house price prediction
```

## Debug Command

Simple debugging with optional theatrical "drama" mode.

```bash
# Basic debug output
apr debug model.apr

# Drama mode - theatrical output (inspired by whisper.apr)
apr debug model.apr --drama

# Hex dump of file bytes
apr debug model.apr --hex

# Extract ASCII strings
apr debug model.apr --strings

# Limit output lines
apr debug model.apr --hex --limit 512
```

### Drama Mode Output

```
====[ DRAMA: model.apr ]====

ACT I: THE HEADER
  Scene 1: Magic bytes... APRN (applause!)
  Scene 2: Version check... 1.0 (standing ovation!)
  Scene 3: Model type... LinearRegression (the protagonist!)

ACT II: THE METADATA
  Scene 1: File size... 2.5 KiB
  Scene 2: Flags... COMPRESSED | SIGNED

ACT III: THE VERDICT
  CURTAIN CALL: Model is READY!

====[ END DRAMA ]====
```

## Validate Command

Validate model integrity with optional 100-point quality assessment.

```bash
# Basic validation
apr validate model.apr

# With 100-point quality scoring
apr validate model.apr --quality

# Strict mode (fail on warnings)
apr validate model.apr --strict
```

### Quality Assessment Output

```
Validating model.apr...

[PASS] Header complete (32 bytes)
[PASS] Magic bytes: APRN
[PASS] Version: 1.0 (supported)
[PASS] Digital signature present
[PASS] Metadata readable

Result: VALID (with 0 warnings)

=== 100-Point Quality Assessment ===

Structure: 25/25
  - Header valid:        5/5
  - Metadata complete:   5/5
  - Checksum valid:      5/5
  - Magic valid:         5/5
  - Version supported:   5/5

Security: 25/25
  - No pickle code:      5/5
  - No eval/exec:        5/5
  - Signed:              5/5
  - Safe format:         5/5
  - Safe tensors:        5/5

Weights: 25/25
  - No NaN values:       5/5
  - No Inf values:       5/5
  - Reasonable range:    5/5
  - Low sparsity:        5/5
  - Healthy distribution: 5/5

Metadata: 25/25
  - Training info:       5/5
  - Hyperparameters:     5/5
  - Metrics recorded:    5/5
  - Provenance:          5/5
  - Description:         5/5

TOTAL: 100/100 (EXCELLENT)
```

## Diff Command

Compare two models to identify differences.

```bash
# Compare models
apr diff model1.apr model2.apr

# JSON output
apr diff model1.apr model2.apr --json

# Show weight-level differences
apr diff model1.apr model2.apr --weights
```

### Example Output

```
Comparing model1.apr vs model2.apr

DIFF: 3 differences found:

  version: 1.0 → 1.1
  model_name: old-model → new-model
  payload_size: 1024 → 2048
```

## Tensors Command

List tensor names, shapes, and statistics from APR model files. Useful for debugging model structure and identifying issues.

```bash
# List all tensors
apr tensors model.apr

# Show statistics (mean, std, min, max)
apr tensors model.apr --stats

# Filter by name pattern
apr tensors model.apr --filter encoder

# Limit output
apr tensors model.apr --limit 10

# JSON output
apr tensors model.apr --json
```

### Example Output

```
=== Tensors: model.apr ===

  Total tensors: 4
  Total size: 79.7 MiB

  encoder.conv1.weight [f32] [384, 80, 3]
    Size: 360.0 KiB
  encoder.conv1.bias [f32] [384]
    Size: 1.5 KiB
  decoder.embed_tokens.weight [f32] [51865, 384]
    Size: 76.0 MiB
  audio.mel_filterbank [f32] [80, 201]
    Size: 62.8 KiB
```

### With Statistics

```bash
apr tensors model.apr --stats
```

```
=== Tensors: model.apr ===

  encoder.conv1.weight [f32] [384, 80, 3]
    Size: 360.0 KiB
    Stats: mean=0.0012, std=0.0534
    Range: [-0.1823, 0.1756]
```

## Trace Command

Layer-by-layer analysis with anomaly detection. Useful for debugging model behavior and identifying numerical issues.

```bash
# Basic layer trace
apr trace model.apr

# Verbose with per-layer statistics
apr trace model.apr --verbose

# Filter by layer name pattern
apr trace model.apr --layer encoder

# Compare with reference model
apr trace model.apr --reference baseline.apr

# JSON output for automation
apr trace model.apr --json

# Payload tracing through model
apr trace model.apr --payload

# Diff mode with reference
apr trace model.apr --diff --reference old.apr
```

### Example Output

```
=== Layer Trace: model.apr ===

  Format: APR v1.0
  Layers: 6
  Parameters: 39680000

Layer Breakdown:
  embedding
  transformer_block_0 [0]
  transformer_block_1 [1]
  transformer_block_2 [2]
  transformer_block_3 [3]
  final_layer_norm
```

### Verbose Output

```bash
apr trace model.apr --verbose
```

```
=== Layer Trace: model.apr ===

Layer Breakdown:
  embedding
  transformer_block_0 [0]
    weights: 768000 params, mean=0.0012, std=0.0534, L2=45.2
    output:  mean=0.0001, std=0.9832, range=[-2.34, 2.45]
  transformer_block_1 [1]
    weights: 768000 params, mean=0.0008, std=0.0521, L2=44.8
```

### Anomaly Detection

The trace command automatically detects numerical issues:

```
⚠ 2 anomalies detected:
  - transformer_block_2: 5/1024 NaN values
  - transformer_block_3: large values (max_abs=156.7)
```

## Probar Command

Export layer-by-layer data for visual regression testing with the probar framework.

```bash
# Basic export (JSON + PNG)
apr probar model.apr -o ./probar-export

# JSON only
apr probar model.apr -o ./probar-export --format json

# PNG histograms only
apr probar model.apr -o ./probar-export --format png

# Compare with golden reference
apr probar model.apr -o ./probar-export --golden ./golden-ref

# Filter specific layers
apr probar model.apr -o ./probar-export --layer encoder
```

### Example Output

```
=== Probar Export Complete ===

  Source: model.apr
  Output: ./probar-export
  Format: APR v1.0
  Layers: 4

Golden reference comparison generated

Generated files:
  - ./probar-export/manifest.json
  - ./probar-export/layer_000_block_0.pgm
  - ./probar-export/layer_000_block_0.meta.json
  - ./probar-export/layer_001_block_1.pgm
  - ./probar-export/layer_001_block_1.meta.json

Integration with probar:
  1. Copy output to probar test fixtures
  2. Use VisualRegressionTester to compare snapshots
  3. Run: probar test --visual-diff
```

### Manifest Format

The generated `manifest.json` contains:

```json
{
  "source_model": "model.apr",
  "timestamp": "2025-01-15T12:00:00Z",
  "format": "APR v1.0",
  "layers": [
    {
      "name": "block_0",
      "index": 0,
      "histogram": [100, 100, ...],
      "mean": 0.0,
      "std": 1.0,
      "min": -3.0,
      "max": 3.0
    }
  ],
  "golden_reference": null
}
```

## Import Command

Import models from HuggingFace, local files, or URLs into APR format.

```bash
# Import from HuggingFace
apr import hf://openai/whisper-tiny -o whisper.apr

# Import with specific architecture
apr import hf://meta-llama/Llama-2-7b -o llama.apr --arch llama

# Import from local safetensors file
apr import ./model.safetensors -o converted.apr

# Import with quantization
apr import hf://org/repo -o model.apr --quantize int8

# Force import (skip validation)
apr import ./model.bin -o model.apr --force
```

### Supported Sources

| Source Type | Format | Example |
|-------------|--------|---------|
| HuggingFace | `hf://org/repo` | `hf://openai/whisper-tiny` |
| Local File | Path | `./model.safetensors` |
| URL | HTTP(S) | `https://example.com/model.bin` |

### Architectures

| Architecture | Flag | Auto-Detection |
|--------------|------|----------------|
| Whisper | `--arch whisper` | ✓ |
| LLaMA | `--arch llama` | ✓ |
| BERT | `--arch bert` | ✓ |
| Auto | `--arch auto` (default) | ✓ |

### Quantization Options

| Option | Description |
|--------|-------------|
| `--quantize int8` | 8-bit integer quantization |
| `--quantize int4` | 4-bit integer quantization |
| `--quantize fp16` | 16-bit floating point |

### Example Output

```
=== APR Import Pipeline ===

Source: hf:// (HuggingFace)
  Organization: openai
  Repository: whisper-tiny
Output: whisper.apr

Architecture: Whisper
Validation: Strict

Importing...

=== Validation Report ===
Score: 98/100 (Grade: A+)

✓ Import successful
```

## Explain Command

Get explanations for error codes, tensor names, and model architectures. Supports all formats (APR, GGUF, SafeTensors) via RosettaStone.

```bash
# Explain an error code
apr explain E002

# Explain a specific tensor (by naming convention)
apr explain --tensor encoder.conv1.weight

# Explain a tensor from an actual model file (with shape, dtype, role)
apr explain --tensor encoder.conv1.weight --file model.safetensors

# Explain model architecture from file
apr explain --file model.apr
```

### Error Code Explanations

```bash
apr explain E002
```

```
Explain error code: E002
**E002: Corrupted Data**
The payload checksum does not match the header.
- **Common Causes**: Interrupted download, bit rot, disk error.
- **Troubleshooting**:
  1. Run `apr validate --checksum` to verify.
  2. Check source file integrity (MD5/SHA256).
```

### Tensor Explanations

When a `--file` is provided, the tensor is looked up in the actual model via RosettaStone:

```bash
apr explain --tensor conv1 --file whisper-tiny.safetensors
```

```
Explain tensor: conv1

**model.encoder.conv1.weight**
- **Shape**: [384, 80, 3]
- **DType**: F32
- **Role**: First convolutional layer (feature extraction)

**model.encoder.conv1.bias**
- **Shape**: [384]
- **DType**: F32
- **Role**: First convolutional layer (feature extraction)
```

Fuzzy matching finds all tensors containing the search term. If no match is found, similar tensor names are suggested.

Without `--file`, explains the tensor role by naming convention:

```bash
apr explain --tensor q_proj
```

```
Explain tensor: q_proj
- **Role**: Query projection in attention mechanism
```

### Architecture Explanations

Uses RosettaStone to inspect the actual model file and detect architecture:

```bash
apr explain --file whisper-tiny.safetensors
```

```
Explain model architecture: whisper-tiny.safetensors
- **Format**: SafeTensors
- **Tensors**: 99
- **Architecture**: Encoder-Decoder Transformer
- **Examples**: Whisper, T5, BART
- **Layers**: 4
```

### Kernel Explainability

Explain which kernel pipeline a model family uses, what architectural constraints drive the selection, and what proof exists for correctness.

```bash
# Explain kernel pipeline for a model family
apr explain --kernel llama

# JSON output for tooling integration
apr explain --kernel qwen2 --json

# Resolve kernel from a config.json file
apr explain --kernel /path/to/config.json

# Resolve kernel from a HuggingFace repo ID
apr explain --kernel Qwen/Qwen2.5-Coder-0.5B-Instruct

# Include proof status for each kernel contract
apr explain --kernel gemma --proof-status

# Verbose output with config.json field mapping
apr explain --kernel /path/to/config.json --verbose
```

#### Kernel Equivalence Classes

Models are grouped into kernel equivalence classes (A-F) based on five architectural constraints:

| Class | Constraints | Families |
|-------|------------|----------|
| **A** | GQA + RMSNorm + SiLU + SwiGLU + RoPE | llama, qwen2, qwen3, mistral, deepseek, ... |
| **B** | MHA + LayerNorm + GELU + RoPE/Absolute | gpt2, bert, whisper, rwkv7 |
| **C** | MQA + LayerNorm + GELU + ALiBi | bloom, falcon-40b |
| **D** | Mixed LayerNorm + SiLU | phi, moonshine |
| **E** | MoE variants | mixtral, qwen-moe |
| **F** | RMSNorm + GELU + GatedMlp + RoPE | gemma, codegemma |

#### Resolution Chain

The `--kernel` flag accepts multiple input types, resolved in order:

1. **Family name** (e.g., `llama`) — direct lookup
2. **File path** to `config.json` — extracts `model_type` field
3. **HuggingFace repo ID** (e.g., `Qwen/Qwen2.5-Coder-0.5B-Instruct`) — checks local cache for config.json
4. **Architecture string** (e.g., `Qwen2ForCausalLM`) — maps to family

#### Example Output

```
$ apr explain --kernel qwen2
Kernel Explainability: qwen2

Family:          qwen2
Kernel Class:    A
Description:     GQA + RMSNorm + SiLU + SwiGLU + RoPE

Architectural Constraints:
  attention_type:      GQA
  norm_type:           RMSNorm
  activation:          SiLU
  mlp_type:            SwiGLU
  positional_encoding: RoPE

Kernel Ops Pipeline:
  1. RmsNorm         — pre-attention normalization
  2. RotaryEmbedding — positional encoding via rotation
  3. GroupedAttention — grouped query attention (fewer KV heads)
  4. SiluActivation  — SiLU/swish gating
  5. SwigluMlp       — SwiGLU feed-forward
  6. RmsNorm         — post-attention normalization
  7. LinearProjection — output projection

Equivalence Class Members:
  llama, codellama, qwen2, qwen3, qwen3_5, mistral, deepseek, ...

LAYOUT-002: Row-major tensors required
```

#### JSON Output

```bash
apr explain --kernel llama --json
```

```json
{
  "family": "llama",
  "kernel_class": "A",
  "description": "GQA + RMSNorm + SiLU + SwiGLU + RoPE",
  "constraints": {
    "attention_type": "GQA",
    "norm_type": "RMSNorm",
    "activation": "SiLU",
    "mlp_type": "SwiGLU",
    "positional_encoding": "RoPE"
  },
  "kernel_ops": ["RmsNorm", "RotaryEmbedding", "GroupedAttention", ...],
  "equivalence_class": ["llama", "codellama", "qwen2", ...],
  "proof_summary": {
    "proven": 3,
    "tested": 2,
    "documented": 1,
    "unknown": 0
  }
}
```

## Pull Command

Download and cache models from HuggingFace with Ollama-style UX.

```bash
# Download model to local cache
apr pull hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF

# Download to specific directory
apr pull hf://openai/whisper-tiny -o ./models/

# Download specific file from repo
apr pull hf://TheBloke/Llama-2-7B-GGUF --file llama-2-7b.Q4_K_M.gguf
```

### Example Output

```
Downloading: Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf
Progress: [████████████████████] 100% (1.2 GB)
Cached to: ~/.cache/apr/models/qwen2.5-coder-1.5b-q4_k_m.gguf
```

## List Command

List all cached models.

```bash
# List cached models
apr list

# List with sizes
apr list --size

# JSON output
apr list --json
```

### Example Output

```
Cached Models:
  qwen2.5-coder-1.5b-q4_k_m.gguf  1.2 GB  2025-01-20
  whisper-tiny.apr                39 MB   2025-01-18
  llama-2-7b.Q4_K_M.gguf         3.8 GB  2025-01-15

Total: 3 models, 5.04 GB
```

## Rm Command

Remove models from cache.

```bash
# Remove specific model
apr rm qwen2.5-coder-1.5b-q4_k_m.gguf

# Remove all cached models
apr rm --all

# Dry run (show what would be deleted)
apr rm --all --dry-run
```

## Cbtop Command

Interactive ComputeBrick pipeline monitor (similar to htop for GPU/CPU inference).

```bash
# Start monitor
apr cbtop

# Monitor specific model
apr cbtop --model model.gguf

# Set refresh rate
apr cbtop --refresh 500  # 500ms
```

### Example Output

```
┌─ ComputeBrick Pipeline Monitor ─────────────────────────┐
│ Model: qwen2.5-coder-1.5b-q4_k_m.gguf                   │
│ Backend: GPU (CUDA)                                      │
├──────────────────────────────────────────────────────────┤
│ Throughput: 125.3 tok/s                                  │
│ Latency:    8.0 ms/tok                                   │
│ Memory:     1.2 GB / 8.0 GB                              │
│ Utilization: ████████████░░░░░░░░ 60%                    │
├──────────────────────────────────────────────────────────┤
│ Layer Timing:                                            │
│   attention:  4.2 ms (52%)                               │
│   ffn:        2.8 ms (35%)                               │
│   other:      1.0 ms (13%)                               │
└──────────────────────────────────────────────────────────┘
```

## Compare-hf Command

Compare a local model against HuggingFace source for validation. Supports APR, GGUF, and SafeTensors formats via automatic format detection.

```bash
# Compare local model against HF source (any format)
apr compare-hf model.apr --hf openai/whisper-tiny
apr compare-hf model.gguf --hf openai/whisper-tiny
apr compare-hf model.safetensors --hf openai/whisper-tiny

# Filter to specific tensor
apr compare-hf model.apr --hf openai/whisper-tiny --tensor conv1

# Custom threshold for floating point comparison
apr compare-hf model.apr --hf openai/whisper-tiny --threshold 1e-5

# JSON output
apr compare-hf model.apr --hf openai/whisper-tiny --json
```

### Example Output

```
Loading local model: model.apr (Apr)
Downloading HF model: openai/whisper-tiny
Found 99 tensors in HF model

======================================================================
HuggingFace vs APR Weight Comparison
======================================================================

Total tensors compared: 99
Passed threshold (< 1e-06): 99

Worst tensor: encoder.conv1.weight (diff=0.000000)

All tensors match within threshold!
```

## Hex Command

Hex dump tensor data for low-level debugging.

```bash
# Hex dump first 256 bytes
apr hex model.apr --limit 256

# Hex dump specific tensor
apr hex model.apr --tensor encoder.conv1.weight --limit 128

# Show ASCII alongside hex
apr hex model.apr --ascii
```

### Example Output

```
=== Hex Dump: model.apr ===

00000000: 4150 524e 0100 0000 0200 0000 4c69 6e65  APRN........Line
00000010: 6172 5265 6772 6573 7369 6f6e 0000 0000  arRegression....
00000020: 0000 0000 0000 0000 0000 0000 0000 0000  ................
00000030: 0a00 0000 0000 0000 0000 0000 0000 0000  ................
```

## Tree Command

Display model architecture as a tree view.

```bash
# Show architecture tree
apr tree model.gguf

# Show with tensor shapes
apr tree model.gguf --shapes

# Show with parameter counts
apr tree model.gguf --params
```

### Example Output

```
model.gguf (1.5B parameters)
├── token_embd [51865, 384]
├── encoder
│   ├── conv1 [384, 80, 3]
│   ├── conv2 [384, 384, 3]
│   └── blocks (4 layers)
│       ├── block.0
│       │   ├── attn [384, 384] × 4
│       │   └── mlp [384, 1536, 384]
│       └── ...
├── decoder
│   ├── embed_tokens [51865, 384]
│   └── blocks (4 layers)
└── lm_head [51865, 384]
```

## Flow Command

Visualize data flow through the model. Supports APR, GGUF, and SafeTensors formats via RosettaStone.

```bash
# Show data flow diagram
apr flow model.safetensors

# Filter to specific layer
apr flow model.gguf --layer 0

# Filter by component
apr flow model.apr --component attention

# JSON output (structured tensor groups and architecture)
apr flow model.safetensors --json

# Verbose output with tensor shapes
apr flow model.apr --verbose
```

### Example Output

```
=== Data Flow: whisper-tiny.safetensors ===

Architecture: Encoder-Decoder Transformer

Embedding:
  model.decoder.embed_tokens.weight [51865, 384] F32
  model.decoder.embed_positions.weight [448, 384] F32

Encoder Layers (4):
  Layer 0: self_attn (q_proj, k_proj, v_proj, out_proj) + mlp (fc1, fc2) + layer_norm (x2)
  Layer 1: ...
  ...

Decoder Layers (4):
  Layer 0: self_attn + encoder_attn + mlp + layer_norm (x3)
  ...

Output:
  proj_out.weight [51865, 384] F32
```

### JSON Output

```bash
apr flow model.safetensors --json
```

```json
{
  "file": "model.safetensors",
  "format": "SafeTensors",
  "architecture": "Encoder-Decoder Transformer",
  "total_tensors": 99,
  "groups": {
    "embedding": ["model.decoder.embed_tokens.weight", "..."],
    "encoder": ["model.encoder.layers.0.self_attn.q_proj.weight", "..."],
    "decoder": ["model.decoder.layers.0.self_attn.q_proj.weight", "..."],
    "output": ["proj_out.weight"]
  },
  "encoder_layers": 4,
  "decoder_layers": 4
}
```

## Bench Command

Benchmark model throughput (spec H12: >= 10 tok/s).

```bash
# Real GPU benchmark (recommended)
apr bench model.apr --fast

# CPU benchmark
apr bench model.gguf

# Specify iterations
apr bench model.apr --fast --iterations 10

# Benchmark with specific prompt
apr bench model.apr --fast --prompt "Hello, world!"

# JSON output for CI
apr bench model.apr --fast --json

# Brick-level analytical budgets (GH-90: these are theoretical, not measured)
apr bench model.apr --brick qkv
apr bench model.apr --brick layer
```

### Example Output (GPU)

```
=== APR Benchmark ===

  Model: qwen2.5-coder-1.5b-q4k.apr
  Warmup iterations: 3
  Measurement iterations: 5
  Max tokens: 32

Using realizar inference engine
Model ready in 1.42s (339 tensors, GPU device 0, fused Q4K kernels)

=== Results ===

Throughput: 240.0 tok/s (PASS: >= 10 tok/s)

  Time to first token: 4ms
  Mean iteration time: 0.13s
  Performance Grade: A+ (Excellent)
```

### Brick Benchmarks

Brick benchmarks (`--brick`) report **analytical budget estimates**, not measured execution time (GH-90). Only `rms_norm` has a real `run()` implementation. All other bricks (qkv, rope, attn, ffn, o_proj, layer) report their theoretical FLOP/bandwidth budget.

Use `apr bench --fast` for real measured GPU throughput.

## Eval Command

Evaluate model quality. Supports two modes: **perplexity** (language models) and **classification** (fine-tuned classifiers).

### Perplexity Evaluation (default)

```bash
# Evaluate perplexity
apr eval model.gguf

# Evaluate on specific dataset
apr eval model.gguf --dataset wikitext-2

# Limit context length
apr eval model.gguf --context 512

# JSON output
apr eval model.gguf --json
```

#### Example Output (Perplexity)

```
=== Evaluation: model.gguf ===

Dataset: wikitext-2
Tokens: 10000
Context: 2048

Results:
  Perplexity: 8.45
  Bits per byte: 2.31
  Cross-entropy: 2.13

Spec H13 (PPL <= 20): ✓ PASS
```

### Classification Evaluation (--task classify)

Evaluate a fine-tuned classifier checkpoint against a JSONL test set. Computes 13 metrics
with bootstrap confidence intervals and optional HuggingFace model card generation.

```bash
# Text report (sklearn-style)
apr eval /path/to/checkpoint/ --task classify \
    --data test.jsonl --model-size 0.5B --num-classes 5

# JSON output
apr eval /path/to/checkpoint/ --task classify \
    --data test.jsonl --model-size 0.5B --num-classes 5 --json

# Generate HuggingFace model card
apr eval /path/to/checkpoint/ --task classify \
    --data test.jsonl --model-size 0.5B --num-classes 5 --generate-card
```

| Flag | Description |
|------|-------------|
| `--task classify` | Switch to classification evaluation mode |
| `--data FILE` | JSONL test set (`{"input":"...","label":N}`) |
| `--model-size SIZE` | Base model size hint: `0.5B`, `tiny` |
| `--num-classes N` | Number of output classes (default: 5) |
| `--generate-card` | Write HuggingFace README.md to checkpoint directory |
| `--json` | Machine-readable JSON output |

#### Metrics Computed

- **Accuracy & Agreement**: accuracy, top-2 accuracy, Cohen's kappa, MCC (with 95% bootstrap CIs)
- **Per-Class**: precision, recall, F1, support for each class
- **Proper Scoring Rules**: Brier score, log loss
- **Calibration**: ECE, mean confidence, confidence gap
- **Baselines**: random (1/K), majority-class, lift

#### Example Output (Classification)

```
=== Classification Report ===

                precision    recall  f1-score   support
              safe    0.8022    0.5840    0.6759       125
     needs-quoting    0.5000    0.0526    0.0952        38
 non-deterministic    0.5423    0.7624    0.6337       101
    non-idempotent    0.5389    0.8333    0.6545       108
            unsafe    0.7188    0.5391    0.6161       128
 ----------------------------------------------------------
         macro avg    0.6204    0.5543    0.5351       500
      weighted avg    0.6329    0.6220    0.6033       500

Accuracy: 62.20% [57.80%, 66.80%]
Cohen's kappa: 0.5124 (moderate)
MCC: 0.5241 [0.4701, 0.5793]
Macro F1: 0.5351 [0.4901, 0.5791]

Brier Score: 0.6077 (lower is better)
Log Loss: 1.8209 (lower is better)

Baselines: random=20.0%, majority=25.6%, model=62.2% (2.4x lift over majority)

Top confused pairs:
  safe → non-deterministic: 28
  unsafe → non-idempotent: 24
  safe → non-idempotent: 22
```

## Profile Command

Deep profiling with Roofline analysis.

```bash
# Run profiler
apr profile model.gguf

# Profile specific layers
apr profile model.gguf --layer attention

# Generate roofline plot data
apr profile model.gguf --roofline

# Output as JSON
apr profile model.gguf --json
```

### Example Output

```
=== Profile: model.gguf ===

Roofline Analysis:
  Peak Compute: 2.5 TFLOPS
  Peak Memory BW: 200 GB/s
  Arithmetic Intensity: 12.5 FLOPS/byte

Layer Breakdown:
  Layer              Time (ms)   Memory   Compute   Bound
  ─────────────────────────────────────────────────────────
  token_embd         0.5         128 MB   0.1 TF    Memory
  attention          4.2         256 MB   0.8 TF    Compute
  ffn                2.8         512 MB   1.2 TF    Compute
  lm_head            0.8         384 MB   0.4 TF    Memory

Bottleneck: Attention layer (compute-bound)
Recommendation: Increase batch size for better GPU utilization
```

## QA Command

Falsifiable QA checklist for model releases.

```bash
# Run full QA checklist
apr qa model.gguf

# Specify throughput threshold
apr qa model.gguf --assert-tps 100

# Require Ollama speedup
apr qa model.gguf --assert-speedup 2.0

# Skip Ollama comparison
apr qa model.gguf --skip-ollama

# JSON output for CI
apr qa model.gguf --json
```

### Example Output

```
=== QA Checklist: model.gguf ===

[1/10] Format Validation
  ✓ Valid GGUF header
  ✓ All tensors readable
  ✓ No NaN/Inf values

[2/10] Golden Output Test
  ✓ Prompt: "Hello" → "Hello! How can I help you today?"
  ✓ Output matches expected (cosine sim: 0.98)

[3/10] Throughput Test
  ✓ 125.3 tok/s (threshold: 10 tok/s)

[4/10] Perplexity Test
  ✓ PPL: 8.45 (threshold: 20.0)

[5/10] Ollama Parity
  ✓ 2.93x Ollama throughput

...

Result: 10/10 PASS
```

## Qualify Command

Cross-subcommand smoke test: runs every diagnostic CLI tool against a model to verify no crashes. Fills the gap between `apr qa` (inference quality gates) and unit tests (isolated logic).

```bash
# Smoke test all 11 diagnostic tools on a model
apr qualify model.gguf

# Standard tier (smoke + contract audit via pv)
apr qualify model.gguf --tier standard

# Full tier (standard + playbook check via apr-qa)
apr qualify model.gguf --tier full

# JSON output for CI
apr qualify model.gguf --json

# Skip slow gates
apr qualify model.gguf --skip validate,validate_quality

# Show subcommand output
apr qualify model.gguf --verbose
```

### Tiers

| Tier | Gates | Description |
|------|-------|-------------|
| `smoke` (default) | 11 | In-process: inspect, validate, validate --quality, tensors, lint, debug, tree, hex, flow, explain, check |
| `standard` | 12 | Smoke + contract audit via `pv` |
| `full` | 13 | Standard + playbook check via `apr-qa` |

### Example Output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Qualify
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Model: model.gguf
  Tier: smoke

  ✓ PASS Inspect (1.3s)
  ✓ PASS Validate (1.3m)
  ✓ PASS Validate (quality) (1.4m)
  ✓ PASS Tensors (1.3s)
  ✓ PASS Lint (688ms)
  ✓ PASS Debug (1.3s)
  ✓ PASS Tree (1.3s)
  ✓ PASS Hex (674ms)
  ✓ PASS Flow (1.9s)
  ✓ PASS Explain (1.3s)
  ✓ PASS Check (pipeline) (3.9s)

  ✓ ALL GATES PASSED
  Total Duration: 2.9m
```

## Showcase Command

Qwen2.5-Coder showcase demo for performance demonstration.

```bash
# Run showcase demo
apr showcase model.gguf

# Specify warmup and iterations
apr showcase model.gguf --warmup 3 --iterations 10

# GPU mode
apr showcase model.gguf --gpu

# Batched GPU mode
apr showcase model.gguf --gpu --batch
```

### Example Output

```
╔════════════════════════════════════════════════════════════╗
║           APR Showcase: Qwen2.5-Coder Performance          ║
╚════════════════════════════════════════════════════════════╝

Model: qwen2.5-coder-1.5b-q4_k_m.gguf
Backend: GPU (CUDA)
Mode: Batched (M=16)

Benchmark Results:
  ┌────────────────┬────────────┬───────────┐
  │ Metric         │ Value      │ vs Ollama │
  ├────────────────┼────────────┼───────────┤
  │ Throughput     │ 851.8 t/s  │ 2.93x     │
  │ Time to First  │ 45 ms      │ 0.8x      │
  │ Memory         │ 1.9 GB     │ 1.2x      │
  └────────────────┴────────────┴───────────┘

✓ Showcase PASSED: 2.93x Ollama performance achieved
```

## Check Command

Model self-test: 10-stage pipeline integrity check (APR-TRACE-001).

```bash
# Run full check
apr check model.gguf

# Verbose output
apr check model.gguf --verbose

# JSON output
apr check model.gguf --json
```

### Example Output

```
=== Model Self-Test: model.gguf ===

Stage 1: Format Validation
  ✓ GGUF magic bytes valid
  ✓ Version: 3
  ✓ Tensor count: 145

Stage 2: Tensor Integrity
  ✓ All tensors readable
  ✓ Shapes consistent
  ✓ No NaN/Inf values

Stage 3: Tokenizer Check
  ✓ Vocabulary size: 151936
  ✓ Special tokens present
  ✓ BPE merges valid

Stage 4: Embedding Test
  ✓ Token embedding produces valid vectors
  ✓ L2 norm in expected range

Stage 5: Attention Test
  ✓ Self-attention computes correctly
  ✓ KV cache initialized

Stage 6: FFN Test
  ✓ Feed-forward produces valid output
  ✓ Activation function working

Stage 7: Layer Norm Test
  ✓ RMSNorm produces normalized output
  ✓ Epsilon handling correct

Stage 8: LM Head Test
  ✓ Logits in valid range
  ✓ Vocabulary mapping correct

Stage 9: Generation Test
  ✓ Can generate 10 tokens
  ✓ Output is coherent text

Stage 10: Performance Test
  ✓ Throughput: 125 tok/s (> 10 tok/s)

Result: 10/10 PASS
```

## Publish Command

Publish model to HuggingFace Hub (APR-PUB-001).

```bash
# Publish model directory
apr publish ./model-dir/ org/model-name

# Dry run (show what would be uploaded)
apr publish ./model-dir/ org/model-name --dry-run

# Specify license and tags
apr publish ./model-dir/ org/model-name --license mit --tags rust,ml

# Custom commit message
apr publish ./model-dir/ org/model-name --message "v1.0.0 release"
```

### Example Output

```
=== Publishing to HuggingFace Hub ===

Repository: org/model-name
Files to upload:
  - model.gguf (1.2 GB)
  - config.json (2 KB)
  - tokenizer.json (500 KB)

Generating README.md with model card...

Uploading...
  [████████████████████] 100% model.gguf
  [████████████████████] 100% config.json
  [████████████████████] 100% tokenizer.json
  [████████████████████] 100% README.md

✓ Published to https://huggingface.co/org/model-name
```

## Finetune Command

Fine-tune a model with LoRA adapters for classification or test generation tasks.

```bash
# Classification fine-tuning (shell safety classifier)
apr finetune --task classify --model-size 0.5B \
    ./models/qwen2.5-coder-0.5b \
    --data corpus.jsonl \
    --epochs 3 \
    --learning-rate 0.0001 \
    --num-classes 5 \
    -o ./ssc-checkpoints/

# Test generation fine-tuning
apr finetune --task test-gen --model-size 0.5B \
    ./models/qwen2.5-coder-0.5b \
    --data tests.jsonl \
    --epochs 5 \
    -o ./test-gen-checkpoints/
```

### Corpus Format

Classification JSONL with `input` (text) and `label` (integer). The shell preamble (shebang, `set -euf`, `trap` cleanup) is automatically stripped during export so the model sees only safety-relevant commands:

```json
{"input": "echo \"hello\"\n", "label": 0}
{"input": "eval \"$x\"\n", "label": 4}
```

Export from bashrs: `cargo run -p bashrs --release --example fast_classify_export /tmp/ssc-corpus.jsonl`

> **Auto-class-balancing**: When training with `apr finetune --task classify`, entrenar auto-detects class imbalance (ratio >2:1) and applies sqrt-inverse weights. No manual `--class-weights` flag is needed for typical corpora.

### Distributed Training (Multi-Node)

Train across multiple machines using TCP gradient AllReduce. One node acts as the
coordinator (manages epochs, averages gradients), others are workers (compute
forward/backward, send gradients).

```bash
# On coordinator (e.g., intel machine)
apr finetune --task classify --model-size 4B \
    ./models/qwen3-4b \
    --data train.jsonl \
    --num-classes 2 \
    --role coordinator \
    --bind 0.0.0.0:9000 \
    --expect-workers 1 \
    -o ./ssc-distributed/

# On worker (e.g., lambda machine)
apr finetune --task classify --model-size 4B \
    ./models/qwen3-4b \
    --data train.jsonl \
    --num-classes 2 \
    --role worker \
    --coordinator intel:9000 \
    -o ./ssc-distributed/
```

**Flags**:
- `--role coordinator|worker` — Node role in distributed training
- `--bind ADDR` — Address for coordinator to listen on (default: `0.0.0.0:9000`)
- `--coordinator ADDR` — Coordinator address for worker to connect to
- `--expect-workers N` — Number of workers the coordinator waits for

**Design**: LoRA gradients (~5-22MB) are averaged on CPU via AllReduce, making
the system backend-agnostic (CUDA workers and wgpu workers can coexist).
See `docs/specifications/distributed-training-spec.md` for the full spec.

## Tune Command

Automatic hyperparameter search for fine-tuning (SPEC-TUNE-2026-001). Searches over 9 parameters using TPE, Grid, or Random strategies.

```bash
# Scout: fast 1-epoch sweep to find good HP region
apr tune --task classify --budget 5 --scout --data corpus.jsonl --json

# Full: multi-epoch search with ASHA early stopping
apr tune --task classify --budget 10 --data corpus.jsonl \
    --strategy tpe --scheduler asha

# Grid search
apr tune --task classify --budget 27 --data corpus.jsonl \
    --strategy grid --scheduler none
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--task` | Task type: `classify` | Required |
| `--budget` | Number of trials | 10 |
| `--strategy` | Search strategy: `tpe`, `grid`, `random` | `tpe` |
| `--scheduler` | Early stopping: `asha`, `median`, `none` | `asha` |
| `--scout` | Scout mode (1 epoch per trial, no scheduling) | false |
| `--data` | Path to JSONL corpus | Required |
| `--json` | Output result as JSON | false |
| `--seed` | Random seed for reproducibility | 42 |

### Search Space (Classification)

| Parameter | Domain | Range |
|-----------|--------|-------|
| `learning_rate` | Continuous (log) | 5e-6 .. 5e-4 |
| `lora_rank` | Discrete | 4 .. 64 (step 4) |
| `lora_alpha_ratio` | Continuous | 0.5 .. 2.0 |
| `batch_size` | Categorical | {8, 16, 32, 64, 128} |
| `warmup_fraction` | Continuous | 0.01 .. 0.2 |
| `gradient_clip_norm` | Continuous | 0.5 .. 5.0 |
| `class_weights` | Categorical | {uniform, inverse_freq, sqrt_inverse} |
| `target_modules` | Categorical | {qv, qkv, all_linear} |
| `lr_min_ratio` | Continuous (log) | 0.001 .. 0.1 |

### Example Output

```
{
  "strategy": "tpe",
  "mode": "scout",
  "budget": 3,
  "trials": [
    {"id": 0, "val_loss": 1.5823, "val_accuracy": 0.333, ...},
    {"id": 1, "val_loss": 1.5987, "val_accuracy": 0.267, ...},
    {"id": 2, "val_loss": 1.6094, "val_accuracy": 0.200, ...}
  ],
  "best_trial_id": 0,
  "total_time_ms": 1430
}
```

### Workflow

1. **Export corpus** from bashrs: `cargo run -p bashrs --release --example fast_classify_export /tmp/ssc-corpus.jsonl`
2. **Scout** to find good HP region: `apr tune --task classify --budget 5 --scout --data /tmp/ssc-corpus.jsonl --json`
3. **Full run** with best strategy: `apr tune --task classify --budget 20 --data /tmp/ssc-corpus.jsonl --strategy tpe --scheduler asha`
4. **Fine-tune** with discovered HPs: `apr finetune --task classify --model-size 0.5B ...`

See also: [entrenar classify_tune_demo example](https://github.com/paiml/entrenar/blob/main/examples/classify_tune_demo.rs) for the programmatic API.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 3 | File not found / Not a file |
| 4 | Invalid APR format |
| 5 | Validation failed |
| 7 | I/O error |

## Integration with CI/CD

Use `apr validate --strict` in CI pipelines to ensure model quality:

```yaml
# GitHub Actions example
- name: Validate Model
  run: apr validate models/production.apr --quality --strict
```

## Toyota Way Principles in apr-cli

1. **Genchi Genbutsu (Go and See)**: `apr inspect` lets you see the actual model data, not abstractions
2. **Genchi Genbutsu (Go to the Source)**: `apr tensors` reveals the actual tensor structure and statistics
3. **Jidoka (Built-in Quality)**: `apr validate` stops on quality issues with clear feedback
4. **Visualization**: `apr debug --drama` makes problems visible and understandable
5. **Kaizen (Continuous Improvement)**: `apr diff` enables comparing models for improvement
6. **Visualization**: `apr trace` makes layer-by-layer behavior visible with anomaly detection
7. **Standardization**: `apr probar` creates repeatable visual regression tests
8. **Automation**: `apr import` automates model conversion with inline validation
9. **Knowledge Sharing**: `apr explain` documents errors, tensors, and architectures

## See Also

- [APR Model Format Specification](../examples/model-format.md)
- [APR Model Inspection](../examples/apr-inspection.md)
- [APR 100-Point Quality Scoring](../examples/apr-scoring.md)
