---
title: "ALB-093: Direct SafeTensors to Q4K APR Streaming Quantization Pipeline"
issue: GH-434
status: Complete
created: 2026-03-08
---

# ALB-093: Direct SafeTensors to Q4K APR Streaming Pipeline

**GitHub Issue**: [#434](https://github.com/paiml/aprender/issues/434)
**Status**: Complete
**Contract**: `contracts/safetensors-to-q4k-v1.yaml` (albor repo)

## Summary

The `apr quantize` command supports direct streaming quantization from sharded
HuggingFace SafeTensors models to Q4K APR format. The pipeline uses single-piece
flow: one tensor at a time flows through load, validate, quantize, write. No
intermediate files are created. Peak memory is bounded by the largest single
tensor, not total model size.

## Motivation (Five Whys)

1. **Why can't we quantize the teacher model?** OOM during `apr quantize`.
2. **Why OOM?** `load_model_tensors()` reads the entire model + dequants to f32 (~170 GB).
3. **Why full load?** The quantize path was designed for small models with no streaming support.
4. **Why create a 57 GB intermediate APR?** The two-step pipeline required: import to APR, then APR to Q4K.
5. **Why two steps?** No direct SafeTensors to Q4K path existed.

The root cause is a missing single-piece flow path from SafeTensors shards
directly to Q4K APR output.

## Usage

```bash
# Basic: quantize sharded SafeTensors model to Q4K APR
apr quantize /path/to/safetensors/model/ --scheme q4k -o output.apr

# Also accepts the index file directly
apr quantize /path/to/model/model.safetensors.index.json --scheme q4k -o output.apr

# Plan mode: estimate output size and memory without executing
apr quantize /path/to/model/ --scheme q4k --plan

# JSON output for CI integration
apr quantize /path/to/model/ --scheme q4k -o output.apr --json
```

The streaming path activates automatically when:
- `--scheme q4k` is specified, AND
- The input is a directory containing `model.safetensors.index.json` (or the index file itself)

For non-sharded SafeTensors or APR inputs, the existing `apr_convert` path is used.

## Architecture

### Data Flow

```
SafeTensors Shards (mmap)
    |
    v
[Shard 1] --mmap--> tensor_1 --dequant--> f32 --validate--> Q4K --stream write--> |
                     tensor_2 --dequant--> f32 --validate--> Q4K --stream write--> |
                     ...                                                           |
           (mmap dropped -- OS reclaims virtual address space)                     |
    |                                                                              |
    v                                                                              |
[Shard 2] --mmap--> tensor_N --dequant--> f32 --validate--> Q4K --stream write--> |
                     ...                                                           |
    |                                                                              |
    v                                                                              v
[Shard K]                                                              Q4K APR output file
```

### Key Design Decisions

**Single-piece flow (Toyota Way: Heijunka)**. One tensor is in memory at a time.
The f32 dequantized buffer is freed before the next tensor is loaded. Memory
pressure is constant, with no spike-and-release pattern.

**Jidoka (stop-the-line)**. Every tensor is validated for NaN and Inf values
before quantization. A single corrupt value halts the pipeline with a diagnostic
message identifying the tensor name and element index.

**Pull system**. Tensor precision is determined by its role, not by a blanket
policy:

| Tensor Role | Criteria | Output Precision |
|-------------|----------|-----------------|
| Weight matrices | 2D shape, >= 256 elements | Q4K (4.5 bits/weight) |
| Norm weights | Name contains `norm` | F32 (precision-critical) |
| Embeddings | Name contains `embed` | F32 (lookup table) |
| Bias vectors | Name contains `bias` | F32 (precision-critical) |
| Scale parameters | Name contains `scale` | F32 (normalization) |
| Small tensors | < 256 elements | F32 (not worth quantizing) |
| 1D tensors | 1D shape | F32 (vectors, not weight matrices) |

### Memory Bounds

Peak memory = 2 * sizeof(largest_tensor_in_f32):
- One copy for the dequantized f32 data
- One copy for the Q4K output bytes

For Qwen3-Coder-30B-A3B-Instruct, the largest tensor is ~400M elements
(1.6 GB in f32). Peak RSS during quantization: ~3.8 GB. This compares to
~170 GB if the entire model were loaded monolithically.

### Shard Discovery

The pipeline locates shards via `model.safetensors.index.json`, which maps
tensor names to shard filenames. Accepts either:
- A directory containing the index file
- A direct path to the index file itself

Each shard is opened via mmap (`MappedSafeTensors`), processed, then unmapped.
Only one shard's virtual address space is active at a time.

### Metadata Propagation

Model architecture metadata (`config.json`) is read from the model directory
and propagated into the APR output file. Fields include: architecture,
hidden_size, num_layers, num_heads, num_kv_heads, vocab_size,
intermediate_size, max_position_embeddings, rope_theta, rope_type, rms_norm_eps.

## Results

### Qwen3-Coder-30B-A3B-Instruct (57 GB, 16 shards, 18,867 tensors)

| Metric | Value |
|--------|-------|
| Input size | 57 GB (16 SafeTensors shards) |
| Output size | 17 GB (single Q4K APR file) |
| Compression ratio | 3.3x |
| Wall-clock time | ~4 minutes (NVMe storage) |
| Peak RSS | ~3.8 GB |
| Monolithic peak RSS | ~170 GB (impossible on consumer hardware) |
| Tensors quantized | Weight matrices (2D, >= 256 elements) |
| Tensors kept F32 | Norms, embeddings, biases, scale, small, 1D |

## Implementation

| File | Purpose |
|------|---------|
| `src/format/converter/streaming_quantize.rs` | Core streaming pipeline |
| `crates/apr-cli/src/commands/quantize.rs` | CLI integration and routing |

### Entry Point

`streaming_quantize_q4k(model_dir, output) -> Result<StreamingQuantizeReport>`

Called by the `apr quantize` CLI when the input is a SafeTensors shard directory
and the scheme is Q4K. The function returns a `StreamingQuantizeReport` with
counts of total/quantized/skipped tensors, input/output sizes, and peak tensor
bytes.

### Output Format

The output is an APR v2 file written via `AprV2StreamingWriter`. Tensors are
written incrementally (no full-model buffer). The writer finalizes by writing
the header, metadata, tensor index, and data sections.

## Contract

The pipeline is governed by `contracts/safetensors-to-q4k-v1.yaml` in the
albor repo. The contract specifies:
- Memory bound: peak RSS < 2 * max_single_tensor_bytes + overhead
- Correctness: no NaN or Inf in output tensors
- Completeness: every tensor in the shard index appears in the output
- Precision routing: norm/embed/bias/scale tensors preserved at F32

## Testing

```bash
# Unit tests for tensor routing and validation
cargo test --lib -- streaming_quantize

# Integration test (requires SafeTensors model directory)
apr quantize /path/to/sharded-model/ --scheme q4k -o /tmp/test-q4k.apr
apr inspect /tmp/test-q4k.apr
```

## References

- Toyota Production System: Monden, Y. (1983). *Toyota Production System*. Industrial Engineering and Management Press.
- Design by Contract: [`docs/design-by-contract.md`](../design-by-contract.md)
- APR Format Spec: [`docs/specifications/APR-SPEC.md`](APR-SPEC.md)
- GH-243 (original `apr quantize`): [`docs/specifications/243-spec.md`](243-spec.md)
