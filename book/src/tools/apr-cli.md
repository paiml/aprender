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
| `serve` | Start inference server with GPU acceleration | Just-in-Time Production |
| `chat` | Interactive chat with language models | Genchi Genbutsu (Go and See) |
| `inspect` | View model metadata and structure | Genchi Genbutsu (Go and See) |
| `debug` | Debug output with optional drama mode | Visualization |
| `validate` | Validate integrity with quality scoring | Jidoka (Built-in Quality) |
| `diff` | Compare two models | Kaizen (Continuous Improvement) |
| `tensors` | List tensor names, shapes, and statistics | Genchi Genbutsu (Go to the Source) |
| `trace` | Layer-by-layer analysis with anomaly detection | Visualization |
| `probar` | Export for visual regression testing | Standardization |
| `import` | Import from HuggingFace, local files, or URLs | Automation |
| `explain` | Explain errors, architecture, and tensors | Knowledge Sharing |

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

Get explanations for error codes, tensor names, and model architectures.

```bash
# Explain an error code
apr explain E002

# Explain a specific tensor
apr explain --tensor encoder.conv1.weight

# Explain model architecture
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

```bash
apr explain --tensor encoder.conv1.weight
```

```
**encoder.conv1.weight**
- **Role**: Initial feature extraction (Audio -> Latent)
- **Shape**: [384, 80, 3] (Filters, Input Channels, Kernel Size)
- **Stats**: Mean 0.002, Std 0.04 (Healthy)
```

### Architecture Explanations

```bash
apr explain --file whisper.apr
```

```
Explain model architecture: whisper.apr
This is a **Whisper (Tiny)** model.
- **Purpose**: Automatic Speech Recognition (ASR)
- **Architecture**: Encoder-Decoder Transformer
- **Input**: 80-channel Mel spectrograms
- **Output**: Text tokens (multilingual)
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
