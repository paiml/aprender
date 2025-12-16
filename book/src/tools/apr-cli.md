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
| `inspect` | View model metadata and structure | Genchi Genbutsu (Go and See) |
| `debug` | Debug output with optional drama mode | Visualization |
| `validate` | Validate integrity with quality scoring | Jidoka (Built-in Quality) |
| `diff` | Compare two models | Kaizen (Continuous Improvement) |
| `tensors` | List tensor names, shapes, and statistics | Genchi Genbutsu (Go to the Source) |

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

## See Also

- [APR Model Format Specification](../examples/model-format.md)
- [APR Model Inspection](../examples/apr-inspection.md)
- [APR 100-Point Quality Scoring](../examples/apr-scoring.md)
