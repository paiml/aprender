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
| `serve` | Start inference server with GPU acceleration | Just-in-Time Production |
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
| `showcase` | Qwen2.5-Coder showcase demo | Standardization |
| `check` | Model self-test: 10-stage pipeline integrity | Jidoka |
| `publish` | Publish model to HuggingFace Hub | Automation |
| `cbtop` | ComputeBrick pipeline monitor | Visualization |
| `compare-hf` | Compare local model (APR/GGUF/SafeTensors) against HuggingFace | Jidoka |
| `explain` | Explain errors, architecture, and tensors | Knowledge Sharing |
| `tui` | Interactive terminal UI | Visualization |
| `canary` | Regression testing via tensor statistics | Jidoka |

## Serve Command

Start an OpenAI-compatible inference server with optional GPU acceleration.

```bash
# Basic server (CPU)
apr serve model.gguf --port 8080

# GPU-accelerated server
apr serve model.gguf --port 8080 --gpu

# Batched GPU mode (2.9x faster than Ollama)
apr serve model.gguf --port 8080 --gpu --batch
```

### Performance

| Mode | Throughput | vs Ollama | Memory |
|------|------------|-----------|--------|
| CPU (baseline) | ~15 tok/s | 0.05x | 1.1 GB |
| GPU (single) | ~83 tok/s | 0.25x | 1.5 GB |
| GPU (batched) | ~850 tok/s | 2.9x | 1.9 GB |
| Ollama | ~333 tok/s | 1.0x | - |

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
# Run benchmark
apr bench model.gguf

# Specify iterations
apr bench model.gguf --iterations 100

# Benchmark with specific prompt
apr bench model.gguf --prompt "Hello, world!"

# JSON output for CI
apr bench model.gguf --json
```

### Example Output

```
=== Benchmark: model.gguf ===

Configuration:
  Iterations: 50
  Warmup: 5
  Prompt: "Hello, how are you?"

Results:
  Throughput: 125.3 tok/s
  Latency (p50): 8.0 ms
  Latency (p99): 12.3 ms
  Memory Peak: 1.2 GB

Spec H12 (>= 10 tok/s): ✓ PASS
```

## Eval Command

Evaluate model perplexity (spec H13: PPL <= 20).

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

### Example Output

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
