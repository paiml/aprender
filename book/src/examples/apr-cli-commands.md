# Case Study: APR CLI Commands Demo

This case study demonstrates creating test models and using all 15 apr-cli commands for model inspection, validation, transformation, and testing.

## The Problem

APR model files need comprehensive tooling for:

| Need | Traditional Approach | Problem |
|------|---------------------|---------|
| Inspection | Custom scripts | No standardization |
| Validation | Manual checksums | Incomplete coverage |
| Transformation | Framework-specific | Lock-in |
| Regression | Manual testing | Error-prone |

## The Solution: apr-cli

The `apr` CLI provides 15 commands for complete model lifecycle management:

```bash
# Build the CLI
cargo build -p apr-cli

# Inspect model metadata
./target/debug/apr inspect model.apr --json

# Validate integrity (100-point QA)
./target/debug/apr validate model.apr --quality

# Quantize model
./target/debug/apr convert model.apr --quantize int8 -o model-int8.apr
```

## Complete Example

Run: `cargo run --example apr_cli_commands`

```rust,ignore
{{#include ../../../examples/apr_cli_commands.rs}}
```

## All 15 Commands

### Model Inspection

#### 1. INSPECT - View Model Metadata

```bash
apr inspect model.apr              # Basic info
apr inspect model.apr --json       # JSON output
apr inspect model.apr --weights    # Include tensor info
```

Shows model type, framework, hyperparameters, and training info.

#### 2. TENSORS - List Tensor Info

```bash
apr tensors model.apr              # List all tensors
apr tensors model.apr --stats      # Include statistics
apr tensors model.apr --json       # JSON output
```

Lists tensor names, shapes, dtypes, and statistics.

#### 3. TRACE - Layer-by-Layer Analysis

```bash
apr trace model.apr                # Basic trace
apr trace model.apr --verbose      # Detailed trace
apr trace model.apr --json         # JSON output
```

Analyzes model layer by layer for debugging inference.

#### 4. DEBUG - Debug Output

```bash
apr debug model.apr                # Standard debug
apr debug model.apr --drama        # Detailed drama mode
apr debug model.apr --hex --limit 64  # Hex dump
```

Provides detailed tensor inspection for debugging.

### Quality & Validation

#### 5. VALIDATE - Check Model Integrity

```bash
apr validate model.apr             # Basic validation
apr validate model.apr --quality   # 100-point QA checklist
apr validate model.apr --strict    # Strict mode
```

Runs the 100-point quality assessment with grades A+ to F.

#### 6. LINT - Best Practices Check

```bash
apr lint model.apr                 # Check best practices
```

Static analysis for naming conventions, metadata completeness, and efficiency.

Checks:
- Standard tensor naming patterns (layer.0.weight, not l0_w)
- Required metadata (author, license, provenance)
- Tensor alignment (64-byte boundaries)
- Compression for large tensors (>1MB)

#### 7. DIFF - Compare Two Models

```bash
apr diff model_v1.apr model_v2.apr       # Compare models
apr diff model_v1.apr model_v2.apr --json  # JSON output
```

Shows metadata and tensor differences between model versions.

### Model Transformation

#### 8. CONVERT - Quantization/Optimization

```bash
apr convert model.apr --quantize int8 -o model-int8.apr
apr convert model.apr --quantize int4 -o model-int4.apr
apr convert model.apr --quantize fp16 -o model-fp16.apr
```

Applies quantization for reduced model size and faster inference.

| Quantization | Size Reduction | Accuracy Impact |
|--------------|----------------|-----------------|
| fp16 | 50% | Minimal |
| int8 | 75% | Small |
| int4 | 87.5% | Moderate |

#### 9. EXPORT - Export to Other Formats

```bash
apr export model.apr --format safetensors -o model.safetensors
apr export model.apr --format gguf -o model.gguf
```

Exports APR models to other ecosystems:
- **SafeTensors** - HuggingFace ecosystem
- **GGUF** - llama.cpp / local inference

#### 10. MERGE - Merge Models

```bash
apr merge model1.apr model2.apr --strategy average -o merged.apr
apr merge model1.apr model2.apr --strategy weighted -o merged.apr
```

Combines multiple models using different strategies:
- **average** - Simple tensor averaging
- **weighted** - Weighted combination

### Import & Interop

#### 11. IMPORT - Import External Models

```bash
apr import external.safetensors -o imported.apr
apr import hf://org/repo -o model.apr --arch whisper
```

Imports from SafeTensors, HuggingFace Hub, and other formats.

### Testing & Regression

#### 12. CANARY - Regression Testing

```bash
# Create canary from original model
apr canary create model.apr --input ref.wav --output canary.json

# Check optimized model against canary
apr canary check model-optimized.apr --canary canary.json
```

Captures tensor statistics for regression testing after transformations (quantization, pruning).

Canary data includes:
- Tensor shapes and counts
- Mean, std, min, max for each tensor
- Drift tolerance checking

#### 13. PROBAR - Visual Regression Testing

```bash
apr probar model.apr -o probar_output         # Create probar suite
apr probar model.apr -o output --format json  # JSON format
```

Exports model data for visual regression testing.

### Help & Documentation

#### 14. EXPLAIN - Get Explanations

```bash
apr explain E002                           # Explain error code
apr explain --tensor encoder.conv1.weight  # Explain tensor name
apr explain --file model.apr               # Analyze file
```

Provides context-aware explanations for errors and tensor patterns.

### Interactive

#### 15. TUI - Interactive Terminal UI

```bash
apr tui model.apr                          # Launch interactive UI
```

Interactive terminal interface for model exploration (coming soon).

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

## Use Cases

### CI/CD Model Validation

```bash
# In CI pipeline
apr validate model.apr --strict --min-score 90 && apr lint model.apr
if [ $? -ne 0 ]; then
    echo "Model validation failed"
    exit 1
fi
```

### Model Optimization Pipeline

```bash
# Quantize for production
apr convert model.apr --quantize int8 -o model-int8.apr

# Verify no regression
apr canary create model.apr --input test.wav --output canary.json
apr canary check model-int8.apr --canary canary.json

# Export for deployment
apr export model-int8.apr --format gguf -o model.gguf
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

# Drama mode for detailed analysis
apr debug model.apr --drama
```

## Benefits

| Benefit | Description |
|---------|-------------|
| Standardized | Consistent CLI for all APR models |
| Comprehensive | 15 commands cover full lifecycle |
| Scriptable | JSON output for automation |
| Debuggable | Deep inspection with drama mode |
| Validatable | 100-point QA with grades |
| Transformable | Quantization and format conversion |
| Testable | Canary regression testing |

## Related Resources

- [Case Study: APR with JSON Metadata](./apr-with-metadata.md)
- [The .apr Format: A Five Whys Deep Dive](./apr-format-deep-dive.md)
- [APR Loading Modes](./apr-loading-modes.md)
- [apr (APR Model Operations CLI)](../tools/apr-cli.md)
