# Case Study: APR CLI Commands Demo

This case study demonstrates creating test models and using all 26 apr-cli commands for model inspection, validation, transformation, testing, and inference.

## The Problem

APR model files need comprehensive tooling for:

| Need | Traditional Approach | Problem |
|------|---------------------|---------|
| Inspection | Custom scripts | No standardization |
| Validation | Manual checksums | Incomplete coverage |
| Transformation | Framework-specific | Lock-in |
| Regression | Manual testing | Error-prone |

## The Solution: apr-cli

The `apr` CLI provides 26 commands for complete model lifecycle management:

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

## All 26 Commands

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
apr explain E002                                            # Explain error code
apr explain --tensor encoder.conv1.weight                   # Explain tensor by convention
apr explain --tensor conv1 --file model.safetensors         # Look up in actual model
apr explain --file model.apr                                # Analyze architecture
```

Provides context-aware explanations for errors, tensors, and model architectures. When `--file` is provided with `--tensor`, looks up the tensor in the actual model via RosettaStone (supports APR, GGUF, SafeTensors).

### Interactive

#### 15. TUI - Interactive Terminal UI

```bash
apr tui model.apr                          # Launch interactive UI
```

Interactive terminal interface for model exploration with four tabs:

| Tab | Key | Description |
|-----|-----|-------------|
| Overview | `1` | Model metadata, hyperparameters, training info |
| Tensors | `2` | Tensor list with shapes, dtypes, sizes |
| Stats | `3` | Tensor statistics (mean, std, min, max, zeros, NaNs) |
| Help | `?` | Keyboard shortcuts and navigation help |

**Keyboard Navigation:**
- `1`, `2`, `3`, `?` - Switch tabs directly
- `Tab` / `Shift+Tab` - Cycle through tabs
- `j` / `↓` - Next item in list
- `k` / `↑` - Previous item in list
- `q` / `Esc` - Quit

### Inference (requires `--features inference`)

Build with inference support:

```bash
cargo build -p apr-cli --features inference
```

#### 16. RUN - Run Model Inference

```bash
apr run model.apr --input "[1.0, 2.0]"       # JSON array input
apr run model.apr --input "1.0,2.0"          # CSV input
apr run model.apr --input "[1.0, 2.0]" --json  # JSON output
```

Runs inference on APR, SafeTensors, or GGUF models:

| Format | Inference Type |
|--------|----------------|
| APR (.apr) | Full ML inference via realizar |
| SafeTensors (.safetensors) | Tensor inspection |
| GGUF (.gguf) | Model inspection (mmap) |

**Input Formats:**
- JSON array: `"[1.0, 2.0, 3.0]"`
- CSV: `"1.0,2.0,3.0"`

#### 17. SERVE - Start Inference Server

```bash
apr serve model.apr --port 8080              # Start on port 8080
apr serve model.apr --host 0.0.0.0 --port 3000  # Bind to all interfaces
```

Starts a REST API server for model inference:

**APR Models (full inference):**
```bash
# Health check
curl http://localhost:8080/health

# Run inference
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [1.0, 2.0]}'
```

**Server Features:**
- `/health` - Health check endpoint
- `/predict` - Inference endpoint (APR models)
- `/model` - Model info endpoint (GGUF/SafeTensors)
- `/tensors` - Tensor listing (SafeTensors)
- Graceful shutdown via Ctrl+C

### Chat & Comparison

#### 18. CHAT - Interactive Chat (LLM models)

```bash
apr chat model.gguf                                          # Interactive chat
apr chat model.gguf --system "You are a helpful assistant"   # Custom system prompt
```

#### 19. FLOW - Visualize Data Flow

```bash
apr flow model.safetensors            # Show data flow
apr flow model.gguf --json            # JSON output (architecture, groups)
apr flow model.apr --verbose           # Verbose with shapes
```

Detects architecture (Encoder-Decoder, Decoder-Only, Encoder-Only) and groups tensors by layer. Supports APR, GGUF, and SafeTensors.

#### 20. COMPARE-HF - Compare Against HuggingFace Source

```bash
apr compare-hf model.apr --hf openai/whisper-tiny              # APR format
apr compare-hf model.gguf --hf openai/whisper-tiny             # GGUF format
apr compare-hf model.safetensors --hf openai/whisper-tiny      # SafeTensors format
apr compare-hf model.apr --hf openai/whisper-tiny --json       # JSON output
```

Auto-detects local model format. Compares tensor-by-tensor against HuggingFace source.

### HuggingFace Hub

#### 21. PUBLISH - Push to HuggingFace Hub

```bash
apr publish model_dir/ org/model-name --dry-run
```

#### 22. PULL - Download Model

```bash
apr pull hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF -o ./models/
```

### Benchmarking & QA

#### 23. QA - Falsifiable QA Checklist

```bash
apr qa model.gguf                     # Run 8-gate QA checklist
apr qa model.gguf --json              # JSON output
```

#### 24. SHOWCASE - Performance Benchmark

```bash
apr showcase model.gguf --warmup 3 --iterations 10
```

#### 25. PROFILE - Deep Performance Profiling

```bash
apr profile model.gguf --roofline
```

#### 26. BENCH - Run Benchmarks

```bash
apr bench model.gguf --iterations 100
```

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
| Comprehensive | 26 commands cover full lifecycle |
| Scriptable | JSON output for automation |
| Debuggable | Deep inspection with drama mode |
| Validatable | 100-point QA with grades |
| Transformable | Quantization and format conversion |
| Testable | Canary regression testing |
| Inference | Run predictions and serve REST APIs |

## Related Resources

- [Case Study: APR with JSON Metadata](./apr-with-metadata.md)
- [The .apr Format: A Five Whys Deep Dive](./apr-format-deep-dive.md)
- [APR Loading Modes](./apr-loading-modes.md)
- [apr (APR Model Operations CLI)](../tools/apr-cli.md)
