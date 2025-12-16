# APR-OPS-001: Model Operations Specification

**Version**: 1.0.0
**Status**: Draft
**Authors**: Aprender Team
**Created**: 2025-12-16
**Toyota Way Principle**: Genchi Genbutsu (Go and See)

---

## Abstract

This specification defines a comprehensive CLI and programmatic interface for `.apr` model file operations, enabling developers to inspect, debug, trace, export, convert, import, and merge machine learning models. The design follows Toyota Way principles of *visualization* (make problems visible) and *genchi genbutsu* (go to the source to understand).

The primary goal is **simple debugging**: a developer should be able to quickly answer "what's in this model?" with a single command, similar to `hexdump` or `file` for binary inspection.

---

## 1. Design Principles

### 1.1 Toyota Way Alignment

| Principle | Application |
|-----------|-------------|
| **Genchi Genbutsu** | Go and see the actual model data, not abstractions |
| **Visualization** | Make model internals visible for debugging |
| **Jidoka** | Stop on quality issues (corrupted models, NaN weights) |
| **Kaizen** | Continuous improvement via diff and merge operations |
| **Standardization** | Consistent CLI interface across all operations |

### 1.2 PMAT Quality Gates

All operations must pass PMAT quality gates:

```toml
# .pmat-gates.toml additions
[apr-ops]
test_coverage = 95.0
mutation_score = 85.0
max_cyclomatic = 10
satd_count = 0
```

### 1.3 Academic Foundations

This specification draws from established research in model serialization, debugging, and MLOps:

1. **Sculley, D., et al.** (2015). "Hidden Technical Debt in Machine Learning Systems." *NeurIPS 2015*. Google. - Establishes need for model versioning and inspection [1]

2. **Amershi, S., et al.** (2019). "Software Engineering for Machine Learning: A Case Study." *ICSE 2019*. Microsoft. - Documents debugging challenges in ML pipelines [2]

3. **Vartak, M., et al.** (2016). "ModelDB: A System for Machine Learning Model Management." *HILDA Workshop, SIGMOD 2016*. MIT CSAIL. - Defines model CRUD operations [3]

4. **Baylor, D., et al.** (2017). "TFX: A TensorFlow-Based Production-Scale Machine Learning Platform." *KDD 2017*. Google. - Production model lifecycle patterns [4]

5. **Ratner, A., et al.** (2019). "MLflow: A Platform for the Machine Learning Lifecycle." *OpML 2019*. Databricks. - Model registry and artifact management [5]

---

## 2. CLI Interface: `apr`

### 2.1 Command Overview

```
apr - APR Model Operations Tool

USAGE:
    apr <COMMAND> [OPTIONS]

COMMANDS:
    inspect     Inspect model metadata, vocab, and structure
    debug       Simple debugging output ("drama" mode)
    trace       Trace model operations with renacer
    export      Export model to other formats
    convert     Convert between model types
    import      Import from external formats
    merge       Merge multiple models
    diff        Compare two models
    validate    Validate model integrity

OPTIONS:
    -h, --help       Print help
    -V, --version    Print version
    -v, --verbose    Verbose output
    -q, --quiet      Quiet mode (errors only)
    --json           Output as JSON
```

---

## 3. CRUD Operations

### 3.1 Create (Training → Save)

Models are created via training and saved with `aprender::format::save()`:

```rust
use aprender::format::{save, SaveOptions, ModelType};

let options = SaveOptions::default()
    .with_name("my-classifier")
    .with_description("Iris species classifier")
    .with_hyperparameter("n_estimators", 100)
    .with_metric("accuracy", 0.95)
    .with_custom("vocab_size", vocab.len())
    .with_custom("feature_names", &feature_names);

save(&model, ModelType::RandomForest, "model.apr", options)?;
```

### 3.2 Read (Load/Inspect)

```bash
# Quick inspection without loading payload
apr inspect model.apr

# Full load
apr inspect model.apr --full
```

### 3.3 Update (Incremental/Fine-tune)

```bash
# Update model with new data
apr update model.apr --data new_samples.csv --output updated.apr

# Fine-tune with learning rate
apr update model.apr --data finetune.csv --lr 0.001
```

### 3.4 Delete (with audit trail)

```bash
# Soft delete (moves to .apr.deleted with timestamp)
apr delete model.apr

# Hard delete (permanent, requires confirmation)
apr delete model.apr --hard --yes
```

---

## 4. Inspect Command

### 4.1 Basic Inspection

```bash
$ apr inspect whisper.apr

=== whisper.apr ===
Type:        NeuralCustom (Whisper ASR)
Version:     1.0
Size:        1.5 GB (compressed: 890 MB)
Created:     2024-12-15T10:30:00Z
Framework:   aprender 0.18.2

Parameters:  39,000,000
Vocab Size:  51,865
Languages:   99

Flags:       COMPRESSED | SIGNED
Checksum:    0xA1B2C3D4 (valid)
```

### 4.2 Vocabulary Inspection

```bash
$ apr inspect whisper.apr --vocab

=== Vocabulary ===
Size:        51,865 tokens
Type:        BPE (Byte-Pair Encoding)

Special Tokens:
  <|endoftext|>     50256
  <|startoftranscript|>  50257
  <|translate|>     50358
  <|transcribe|>    50359
  <|notimestamps|>  50362

Sample Tokens (first 10):
  0: "the"
  1: "a"
  2: "is"
  3: "to"
  4: "of"
  ...
```

### 4.3 Filter/Security Inspection

```bash
$ apr inspect shell.apr --filters

=== Security Filters ===
Type:        Sensitive Command Filter
Patterns:    45 active

Categories:
  Credentials:    12 patterns (password=, secret=, ...)
  Cloud Auth:     8 patterns (aws configure, gcloud auth, ...)
  Database:       6 patterns (mysql -p, psql -W, ...)
  Keys:           5 patterns (ssh-keygen, gpg --gen-key, ...)
  Environment:    14 patterns (AWS_SECRET, GITHUB_TOKEN, ...)

Custom Filters:
  (none defined)
```

### 4.4 Programmatic API

```rust
use aprender::format::inspect;
use aprender::inspect::{InspectionResult, InspectOptions};

// Quick inspection
let info = inspect("model.apr")?;
println!("Vocab size: {:?}", info.metadata.metrics.get("vocab_size"));

// Full inspection with weights
let options = InspectOptions::default()
    .with_weights(true)
    .with_vocab(true)
    .with_quality_score(true);

let result = aprender::inspect::inspect_full("model.apr", options)?;
println!("Quality: {}/100", result.quality_score.unwrap_or(0));
```

---

## 5. Debug Command ("Drama" Mode)

Simple, human-readable debugging inspired by `xxd`, `file`, and `strings`.

### 5.1 Quick Debug

```bash
$ apr debug whisper.apr

whisper.apr: APR v1.0 NeuralCustom (1.5GB, zstd)
  magic: APRN (valid)
  params: 39M
  vocab: 51865 tokens
  langs: 99
  flags: compressed, signed
  health: OK
```

### 5.2 Drama Mode (Verbose Debug)

```bash
$ apr debug whisper.apr --drama

====[ DRAMA: whisper.apr ]====

ACT I: THE HEADER
  Scene 1: Magic bytes... APRN (applause!)
  Scene 2: Version check... 1.0 (standing ovation!)
  Scene 3: Model type... NeuralCustom (the protagonist!)

ACT II: THE METADATA
  Scene 1: Created... 2024-12-15T10:30:00Z
  Scene 2: Parameters... 39,000,000 (a cast of millions!)
  Scene 3: Vocabulary... 51,865 tokens (they all have lines!)

ACT III: THE WEIGHTS
  Scene 1: encoder.conv1.weight [384, 80, 3]
    - Min: -0.42, Max: 0.38, Mean: 0.001
    - No NaN! No Inf! (relief!)
  Scene 2: decoder.embed_tokens.weight [51865, 384]
    - 19.9M parameters (the ensemble!)
    - Healthy distribution (bell curve!)

ACT IV: THE VERDICT
  Checksum: 0xA1B2C3D4 (VALID!)
  Signature: Ed25519 (VERIFIED!)

  CURTAIN CALL: Model is PRODUCTION READY!

====[ END DRAMA ]====
```

### 5.3 Hex Dump Mode

```bash
$ apr debug model.apr --hex --limit 256

00000000: 4150 524e 0100 0001 0000 0040 0000 0800  APRN.......@....
00000010: 0000 2000 0100 0000 0000 0000 0000 0000  .. .............
00000020: 8292 a763 7265 6174 6564 5f61 74b4 3230  ...created_at.20
00000030: 3234 2d31 322d 3135 5431 303a 3330 3a30  24-12-15T10:30:0
...
```

### 5.4 Strings Mode

```bash
$ apr debug model.apr --strings

Extracted strings (min length 4):
  APRN
  created_at
  2024-12-15T10:30:00Z
  aprender
  0.18.2
  encoder.conv1.weight
  decoder.embed_tokens.weight
  ...
```

---

## 6. Trace Command (Renacer Integration)

Integration with `renacer` for production tracing and profiling.

### 6.1 Basic Trace

```bash
$ apr trace model.apr --input sample.wav --output trace.json

Tracing inference on sample.wav...

Layer                          Time (ms)   Memory (MB)   FLOPs
─────────────────────────────────────────────────────────────
encoder.conv1                      12.3         45.2      1.2G
encoder.conv2                       8.7         32.1      0.8G
encoder.positional                  0.1          0.5      0.0G
decoder.embed_tokens                2.1         76.8      0.1G
decoder.attention.0                15.4         12.3      2.1G
...
─────────────────────────────────────────────────────────────
TOTAL                             142.5        312.4     15.2G

Trace saved to: trace.json
Flamegraph: trace.svg
```

### 6.2 Renacer Chaos Testing

```bash
$ apr trace model.apr --chaos latency --duration 60s

Running chaos test: latency injection...

Baseline:     45ms p50, 89ms p99
With chaos:   78ms p50, 234ms p99

Resilience score: 72/100 (ACCEPTABLE)
Recommendations:
  - Add timeout handling for >200ms latency
  - Consider async inference for high-latency scenarios
```

### 6.3 Programmatic Tracing

```rust
use aprender::trace::{Tracer, TraceOptions};
use renacer::ChaosConfig;

let tracer = Tracer::new("model.apr")?;
let options = TraceOptions::default()
    .with_memory_tracking(true)
    .with_flops_counting(true)
    .with_chaos(ChaosConfig::latency(50..200));

let trace = tracer.run(&input, options)?;
trace.save_flamegraph("trace.svg")?;
```

---

## 7. Export Command

### 7.1 Supported Export Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| ONNX | `.onnx` | Cross-framework inference |
| SafeTensors | `.safetensors` | HuggingFace ecosystem |
| GGUF | `.gguf` | llama.cpp / local inference |
| TorchScript | `.pt` | PyTorch deployment |
| TFLite | `.tflite` | Mobile/edge deployment |
| JSON | `.json` | Metadata only |
| MessagePack | `.msgpack` | Compact metadata |

### 7.2 Export Examples

```bash
# Export to ONNX
apr export model.apr --format onnx --output model.onnx

# Export to SafeTensors (HuggingFace)
apr export model.apr --format safetensors --output model.safetensors

# Export to GGUF with quantization
apr export model.apr --format gguf --quantize q4_0 --output model.gguf

# Export metadata only
apr export model.apr --format json --metadata-only --output model.json
```

### 7.3 Vocabulary Export

```bash
# Export vocabulary to JSON
apr export model.apr --vocab --format json --output vocab.json

# Export vocabulary to text (one token per line)
apr export model.apr --vocab --format text --output vocab.txt

# Export with token IDs
apr export model.apr --vocab --format csv --output vocab.csv
```

---

## 8. Convert Command

### 8.1 Model Type Conversion

```bash
# Convert RandomForest to ONNX-compatible format
apr convert model.apr --to onnx-compatible --output model_onnx.apr

# Quantize model (reduce precision)
apr convert model.apr --quantize q8_0 --output model_q8.apr

# Prune model (remove low-magnitude weights)
apr convert model.apr --prune 0.1 --output model_pruned.apr
```

### 8.2 Precision Conversion

```bash
# FP32 to FP16
apr convert model.apr --precision fp16 --output model_fp16.apr

# FP32 to BF16 (brain float)
apr convert model.apr --precision bf16 --output model_bf16.apr

# Quantize to INT8
apr convert model.apr --precision int8 --calibration calibration.csv
```

---

## 9. Import Command

### 9.1 Supported Import Formats

| Source | Command |
|--------|---------|
| PyTorch | `apr import model.pt --from pytorch` |
| TensorFlow | `apr import model.pb --from tensorflow` |
| ONNX | `apr import model.onnx --from onnx` |
| SafeTensors | `apr import model.safetensors --from safetensors` |
| Scikit-learn | `apr import model.pkl --from sklearn` |
| HuggingFace | `apr import hf://org/model --from huggingface` |

### 9.2 Import Examples

```bash
# Import from HuggingFace Hub
apr import hf://openai/whisper-tiny --output whisper.apr

# Import from PyTorch checkpoint
apr import checkpoint.pt --from pytorch --model-type NeuralCustom --output model.apr

# Import from scikit-learn pickle (with safety scan)
apr import classifier.pkl --from sklearn --scan-pickle --output classifier.apr
```

---

## 10. Merge Command

### 10.1 Model Merging Strategies

| Strategy | Description |
|----------|-------------|
| `average` | Average weights (ensemble) |
| `weighted` | Weighted average by performance |
| `ties` | TIES merging (trim, elect, sign) |
| `dare` | DARE merging (drop and rescale) |
| `slerp` | Spherical linear interpolation |

### 10.2 Merge Examples

```bash
# Simple average merge
apr merge model1.apr model2.apr --strategy average --output merged.apr

# Weighted merge (by validation accuracy)
apr merge model1.apr model2.apr model3.apr \
    --strategy weighted \
    --weights 0.5,0.3,0.2 \
    --output merged.apr

# TIES merge (for LLM fine-tunes)
apr merge base.apr finetune1.apr finetune2.apr \
    --strategy ties \
    --density 0.5 \
    --output merged.apr
```

### 10.3 Vocabulary Merge

```bash
# Merge vocabularies (union)
apr merge model1.apr model2.apr --merge-vocab union --output merged.apr

# Merge vocabularies (intersection only)
apr merge model1.apr model2.apr --merge-vocab intersection --output merged.apr
```

---

## 11. Diff Command

### 11.1 Model Comparison

```bash
$ apr diff model_v1.apr model_v2.apr

=== Model Diff: model_v1.apr vs model_v2.apr ===

Similarity: 94.2%

Header Changes:
  version: 1.0 -> 1.1

Metadata Changes:
  + description: "Updated with more training data"
  ~ n_samples: 10000 -> 15000
  ~ accuracy: 0.92 -> 0.95

Hyperparameter Changes:
  ~ max_depth: 10 -> 12
  + min_samples_leaf: 5

Weight Changes:
  Layers modified: 3/12
  Max delta: 0.0234
  Mean delta: 0.0012
  L2 distance: 1.234

Vocab Changes:
  Added: 42 tokens
  Removed: 3 tokens
  Total: 51865 -> 51904
```

### 11.2 Programmatic Diff

```rust
use aprender::diff::{diff_models, DiffOptions};

let diff = diff_models("v1.apr", "v2.apr", DiffOptions::default())?;

println!("Similarity: {:.1}%", diff.similarity * 100.0);
println!("Weight L2 distance: {:.4}", diff.weight_diff.l2_distance);

for change in diff.metadata_changes {
    println!("  {}: {} -> {}", change.key, change.old, change.new);
}
```

---

## 12. Validate Command

### 12.1 Integrity Validation

```bash
$ apr validate model.apr

Validating model.apr...

[PASS] Magic bytes: APRN
[PASS] Header checksum: 0xA1B2C3D4
[PASS] Payload decompression
[PASS] Weight integrity (no NaN/Inf)
[PASS] Metadata schema
[WARN] No digital signature
[PASS] Vocab consistency

Result: VALID (1 warning)
```

### 12.2 Quality Scoring

```bash
$ apr validate model.apr --quality

=== 100-Point Quality Assessment ===

Structure (25 pts):     24/25
  - Header valid:        5/5
  - Metadata complete:   4/5  (-1: missing description)
  - Checksum valid:      5/5
  - Compression:         5/5
  - Version supported:   5/5

Security (25 pts):      20/25
  - No pickle code:      5/5
  - No eval/exec:        5/5
  - Signed:              0/5  (-5: unsigned)
  - Encrypted:           5/5
  - Safe tensors:        5/5

Weights (25 pts):       25/25
  - No NaN values:       5/5
  - No Inf values:       5/5
  - Reasonable range:    5/5
  - Low sparsity:        5/5
  - Healthy distribution: 5/5

Metadata (25 pts):      22/25
  - Training info:       5/5
  - Hyperparameters:     5/5
  - Metrics recorded:    5/5
  - Provenance:          4/5  (-1: no commit hash)
  - License:             3/5  (-2: license unclear)

TOTAL: 91/100 (EXCELLENT)
```

---

## 13. Visualization

### 13.1 Architecture Visualization

```bash
# Generate architecture diagram
apr visualize model.apr --arch --output arch.svg

# Generate weight heatmaps
apr visualize model.apr --weights --output weights.png

# Generate vocab embeddings (t-SNE)
apr visualize model.apr --vocab-embeddings --output vocab_tsne.html
```

### 13.2 Training Visualization

```bash
# Loss curves (if training metadata present)
apr visualize model.apr --training --output training.html

# Feature importance
apr visualize model.apr --importance --output importance.svg
```

---

## 14. Configuration

### 14.1 Config File

```toml
# ~/.config/apr/config.toml

[defaults]
output_format = "text"  # text, json, yaml
color = true
verbose = false

[inspect]
show_vocab = true
show_weights_summary = true
max_tokens_display = 20

[debug]
drama_mode = false
hex_limit = 256

[trace]
enable_memory = true
enable_flops = true
chaos_enabled = false

[export]
default_format = "safetensors"
compression = "zstd"

[validate]
strict = true
require_signature = false
require_license = false
```

---

## 15. Error Handling

### 15.1 Error Categories

| Code | Category | Description |
|------|----------|-------------|
| E001 | FORMAT | Invalid file format |
| E002 | CORRUPT | Corrupted data |
| E003 | VERSION | Unsupported version |
| E004 | CHECKSUM | Checksum mismatch |
| E005 | DECRYPT | Decryption failed |
| E006 | SIGNATURE | Signature invalid |
| E007 | IO | File I/O error |
| E008 | MEMORY | Out of memory |

### 15.2 Error Output

```bash
$ apr inspect corrupted.apr

ERROR [E002]: Model file corrupted

Details:
  File: corrupted.apr
  Location: payload section (offset 0x1A40)
  Expected checksum: 0xA1B2C3D4
  Actual checksum: 0xDEADBEEF

Suggestions:
  1. Re-download the model file
  2. Check for incomplete transfers
  3. Verify disk integrity

Documentation: https://docs.aprender.dev/errors/E002
```

---

## 16. Implementation Roadmap

### Phase 1: Core Operations (Sprint 1-2)
- [ ] `apr inspect` with basic metadata
- [ ] `apr debug` with drama mode
- [ ] `apr validate` with quality scoring

### Phase 2: Export/Import (Sprint 3-4)
- [ ] `apr export` to ONNX, SafeTensors, GGUF
- [ ] `apr import` from HuggingFace, PyTorch
- [ ] `apr convert` for quantization

### Phase 3: Advanced (Sprint 5-6)
- [ ] `apr trace` with renacer integration
- [ ] `apr merge` with TIES/DARE strategies
- [ ] `apr diff` with semantic comparison
- [ ] `apr visualize` with SVG/HTML output

---

## 17. References

[1] Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., ... & Dennison, D. (2015). Hidden technical debt in machine learning systems. *Advances in Neural Information Processing Systems*, 28.

[2] Amershi, S., Begel, A., Bird, C., DeLine, R., Gall, H., Kamar, E., ... & Zimmermann, T. (2019). Software engineering for machine learning: A case study. In *2019 IEEE/ACM 41st International Conference on Software Engineering: Software Engineering in Practice (ICSE-SEIP)* (pp. 291-300). IEEE.

[3] Vartak, M., Subramanyam, H., Lee, W. E., Viswanathan, S., Huber, S., Madden, S., & Zaharia, M. (2016). ModelDB: a system for machine learning model management. In *Proceedings of the Workshop on Human-In-the-Loop Data Analytics* (pp. 1-3).

[4] Baylor, D., Breck, E., Cheng, H. T., Fiedel, N., Foo, C. Y., Haque, Z., ... & Zinkevich, M. (2017). TFX: A tensorflow-based production-scale machine learning platform. In *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 1387-1395).

[5] Zaharia, M., Chen, A., Davidson, A., Ghodsi, A., Hong, S. A., Konwinski, A., ... & Zumar, C. (2018). Accelerating the machine learning lifecycle with MLflow. *IEEE Data Eng. Bull.*, 41(4), 39-45.

---

## 18. Falsification QA Checklist (25 Points)

Per Popper's falsifiability principle, each claim must be testable and falsifiable.

### Header & Format (5 points)

| # | Claim | Test | Falsification |
|---|-------|------|---------------|
| 1 | Magic bytes are always "APRN" | `assert_eq!(&bytes[0..4], b"APRN")` | Find valid .apr without APRN magic |
| 2 | Version (1,0) is always supported | Load any v1.0 file successfully | Find v1.0 file that fails to load |
| 3 | Header is exactly 32 bytes | `assert_eq!(HEADER_SIZE, 32)` | Find valid .apr with different header size |
| 4 | Checksum detects single-bit errors | Flip one bit, verify checksum fails | Find bit flip that passes checksum |
| 5 | Compressed size ≤ uncompressed size | `assert!(compressed <= uncompressed)` | Find file where compressed > uncompressed |

### Inspection (5 points)

| # | Claim | Test | Falsification |
|---|-------|------|---------------|
| 6 | Inspect never loads full payload | Memory stays < header + metadata | Find inspect that loads full weights |
| 7 | Vocab size matches actual tokens | `assert_eq!(reported, actual.len())` | Find vocab size mismatch |
| 8 | All metadata fields are optional | Save with empty metadata | Find required metadata field |
| 9 | JSON output is valid JSON | Parse with serde_json | Find JSON output that fails parsing |
| 10 | Inspect completes in < 100ms for any size | Benchmark on 10GB model | Find model where inspect > 100ms |

### Debug/Drama Mode (5 points)

| # | Claim | Test | Falsification |
|---|-------|------|---------------|
| 11 | Drama mode produces valid UTF-8 | `String::from_utf8(output)` | Find drama output with invalid UTF-8 |
| 12 | Hex dump shows exact bytes | Compare with `xxd` output | Find hex mismatch |
| 13 | Strings extracts all ASCII ≥4 chars | Compare with `strings` command | Find missed string |
| 14 | Debug never modifies the file | Compare checksums before/after | Find file modified by debug |
| 15 | NaN/Inf detection has zero false negatives | Inject known NaN, verify detection | Find undetected NaN/Inf |

### Export/Import (5 points)

| # | Claim | Test | Falsification |
|---|-------|------|---------------|
| 16 | Export→Import roundtrip preserves weights | `assert!((w1 - w2).abs() < 1e-6)` | Find weight drift > 1e-6 |
| 17 | ONNX export produces valid ONNX | Validate with onnx.checker | Find invalid ONNX output |
| 18 | SafeTensors export is pickle-free | Scan for pickle opcodes | Find pickle in safetensors |
| 19 | Import rejects malicious pickles | Test with known exploit | Find accepted malicious pickle |
| 20 | Quantization reduces file size | `assert!(q8_size < fp32_size)` | Find quantization that increases size |

### Merge/Diff (5 points)

| # | Claim | Test | Falsification |
|---|-------|------|---------------|
| 21 | Diff of identical models shows 100% similarity | `assert_eq!(diff.similarity, 1.0)` | Find identical models with < 100% |
| 22 | Average merge is commutative | `merge(a,b) == merge(b,a)` | Find non-commutative merge |
| 23 | Merge preserves model type | Type unchanged after merge | Find type change after merge |
| 24 | Vocab merge union contains both vocabs | All tokens from both present | Find missing token in union |
| 25 | Weighted merge with [1.0, 0.0] equals first model | Compare weights exactly | Find difference with weight=1.0 |

---

## 19. PMAT Integration

### 19.1 Quality Gate Configuration

```toml
# .pmat-gates.toml
[apr-ops]
# Coverage
test_coverage_minimum = 95.0
test_coverage_target = 98.0

# Complexity
max_cyclomatic_complexity = 10
max_cognitive_complexity = 15

# Technical Debt
satd_maximum = 0
tdg_minimum_grade = "A"

# Mutation Testing
mutation_score_minimum = 85.0

# Performance
max_inspect_latency_ms = 100
max_debug_latency_ms = 500
```

### 19.2 CI Integration

```yaml
# .github/workflows/apr-ops.yml
name: APR-OPS Quality Gates

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: PMAT Quality Gates
        run: |
          pmat quality-gates --config .pmat-gates.toml

      - name: Falsification Tests
        run: |
          cargo test --test apr_ops_falsification -- --test-threads=1

      - name: Mutation Testing
        run: |
          cargo mutants --package apr-ops --timeout 300
```

---

## Appendix A: Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | File not found |
| 4 | Format error |
| 5 | Validation failed |
| 6 | Signature invalid |
| 7 | Decryption failed |

---

## Appendix B: Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `APR_CONFIG` | Config file path | `~/.config/apr/config.toml` |
| `APR_CACHE` | Cache directory | `~/.cache/apr` |
| `APR_LOG_LEVEL` | Log level | `info` |
| `APR_COLOR` | Enable colors | `auto` |
| `APR_NO_DRAMA` | Disable drama mode | `false` |

---

*Document generated following Toyota Way principles and PMAT quality standards.*
