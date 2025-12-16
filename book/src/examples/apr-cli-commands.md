# Case Study: APR CLI Commands Demo

This case study demonstrates creating test models and using all 9 working apr-cli commands for model inspection, validation, and debugging.

## The Problem

APR model files need comprehensive tooling for:

| Need | Traditional Approach | Problem |
|------|---------------------|---------|
| Inspection | Custom scripts | No standardization |
| Validation | Manual checksums | Incomplete coverage |
| Debugging | Hex editors | Not ML-aware |
| Comparison | File diff | Ignores structure |

## The Solution: apr-cli

The `apr` CLI provides 9 working commands for complete model lifecycle management:

```bash
# Build the CLI
cargo build -p apr-cli

# Inspect model metadata
./target/debug/apr inspect model.apr --json

# Validate integrity (100-point QA)
./target/debug/apr validate model.apr --quality

# Debug tensor data
./target/debug/apr debug model.apr --drama
```

## Complete Example

Run: `cargo run --example apr_cli_commands`

```rust,ignore
{{#include ../../../examples/apr_cli_commands.rs}}
```

## The 9 Working Commands

### 1. INSPECT - View Model Metadata

```bash
apr inspect model.apr              # Basic info
apr inspect model.apr --json       # JSON output
apr inspect model.apr --weights    # Include tensor info
```

Shows model type, framework, hyperparameters, and training info.

### 2. VALIDATE - Check Model Integrity

```bash
apr validate model.apr             # Basic validation
apr validate model.apr --quality   # 100-point QA checklist
apr validate model.apr --strict    # Strict mode
```

Runs the 100-point quality assessment with grades A+ to F.

### 3. DEBUG - Debug Output

```bash
apr debug model.apr                # Standard debug
apr debug model.apr --drama        # Detailed drama mode
apr debug model.apr --hex --limit 64  # Hex dump
```

Provides detailed tensor inspection for debugging.

### 4. TENSORS - List Tensor Info

```bash
apr tensors model.apr              # List all tensors
apr tensors model.apr --stats      # Include statistics
apr tensors model.apr --json       # JSON output
```

Lists tensor names, shapes, dtypes, and statistics.

### 5. TRACE - Layer-by-Layer Analysis

```bash
apr trace model.apr                # Basic trace
apr trace model.apr --verbose      # Detailed trace
apr trace model.apr --json         # JSON output
```

Analyzes model layer by layer for debugging inference.

### 6. DIFF - Compare Two Models

```bash
apr diff model_v1.apr model_v2.apr       # Compare models
apr diff model_v1.apr model_v2.apr --json  # JSON output
```

Shows metadata and tensor differences between model versions.

### 7. PROBAR - Visual Regression Testing

```bash
apr probar model.apr -o probar_output         # Create probar suite
apr probar model.apr -o output --format json  # JSON format
```

Exports model data for visual regression testing.

### 8. IMPORT - Import External Models

```bash
apr import external.safetensors -o imported.apr
apr import hf://org/repo -o model.apr --arch whisper
```

Imports from SafeTensors, HuggingFace Hub, and other formats.

### 9. EXPLAIN - Get Explanations

```bash
apr explain E002                           # Explain error code
apr explain --tensor encoder.conv1.weight  # Explain tensor name
apr explain --file model.apr               # Analyze file
```

Provides context-aware explanations for errors and tensor patterns.

## Stub Commands (Team Finishing)

Six additional commands are being implemented:

| Command | Purpose | Status |
|---------|---------|--------|
| convert | Quantization/optimization | Stub |
| export | Export to safetensors/gguf | Stub |
| merge | Model merging | Stub |
| lint | Best practices check | Stub |
| tui | Interactive terminal UI | Stub |
| canary | Regression testing | Implemented |

## Example Output

Running the example creates demo models:

```
=== APR CLI Commands Demo ===

--- Part 1: Creating Demo Model ---
  Adding tensors...
  Model type: Linear Regression
  Tensors: 4
  Size: 1690 bytes
Created: /tmp/apr_cli_demo/demo_model.apr

--- Part 2: Creating Second Model (for diff) ---
  Model type: Linear Regression v2
  Tensors: 4
  Size: 1707 bytes
Created: /tmp/apr_cli_demo/demo_model_v2.apr
```

Then use the CLI:

```bash
$ apr validate /tmp/apr_cli_demo/demo_model.apr --quality

  1. Magic Number Check                 [PASS]
  2. Version Check                      [PASS]
  ...
 25. Tensor Checksum                    [PASS]

Result: VALID (100/100 points)

=== 100-Point Quality Assessment ===

A. Format & Structural Integrity       25/25 [████████████████████]
B. Tensor Physics & Statistics         25/25 [████████████████████]
C. Tooling & Operations                25/25 [████████████████████]
D. Conversion & Interoperability       25/25 [████████████████████]

TOTAL: 100/100 (Grade: A+)
```

## Use Cases

### CI/CD Model Validation

```bash
# In CI pipeline
apr validate model.apr --strict --min-score 90
if [ $? -ne 0 ]; then
    echo "Model validation failed"
    exit 1
fi
```

### Model Version Comparison

```bash
# Compare before/after optimization
apr diff original.apr quantized.apr --json | jq '.tensor_changes'
```

### Debugging Inference Issues

```bash
# Layer-by-layer trace
apr trace model.apr --verbose | grep -i "nan\|inf"
```

## Benefits

| Benefit | Description |
|---------|-------------|
| Standardized | Consistent CLI for all APR models |
| Comprehensive | 9 commands cover full lifecycle |
| Scriptable | JSON output for automation |
| Debuggable | Deep inspection with drama mode |
| Validatable | 100-point QA with grades |

## Related Resources

- [Case Study: APR with JSON Metadata](./apr-with-metadata.md)
- [The .apr Format: A Five Whys Deep Dive](./apr-format-deep-dive.md)
- [APR Loading Modes](./apr-loading-modes.md)
