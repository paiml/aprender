# Quantize Component Specification

**Parent**: [apr-spec.md](../apr-spec.md) §8
**Status**: Active
**CLI**: `apr quantize`
**Implementation**: `crates/apr-cli/src/commands/quantize.rs`

---

## 1. Overview

Weight quantization reduces model precision for smaller file size and faster
inference. APR supports symmetric integer quantization and Q4_K super-block
format for fused kernel compatibility.

## 2. CLI Interface

```
apr quantize <FILE> \
  --scheme <int8|int4|fp16|q4k> \
  --output <PATH> \
  [--format <apr|gguf|safetensors>] \
  [--batch <scheme1,scheme2>] \
  [--plan] \
  [--force]
```

## 3. Schemes

| Scheme | Bits | Block Size | Description |
|--------|------|------------|-------------|
| `fp16` | 16 | — | Half-precision float |
| `int8` | 8 | — | Symmetric 8-bit integer |
| `int4` | 4 | — | 4-bit integer |
| `q4k` | 4 | 256 | Q4_K super-block (fused kernel compatible) |

### Q4_K Super-Block Layout

Each super-block quantizes 256 elements:
- Scale: f16 (2 bytes)
- Min: f16 (2 bytes)
- Quantized data: 128 bytes (4 bits per element)
- Per-block scales: 12 bytes
- Total: 144 bytes per 256 elements

## 4. Features

- **Batch quantization**: Multiple schemes in one pass
- **Output format selection**: APR, GGUF, SafeTensors
- **Plan mode**: Estimate compression ratio
- **Streaming quantization**: GH-434 (in progress for >57GB models)

---

## Provable Contracts

### Contract: `quantization-v1.yaml`

```yaml
metadata:
  description: "Weight quantization — int8, int4, fp16, Q4K"
  references:
    - "Dettmers et al. (2022) LLM.int8()"
    - "GGML Q4_K format specification"
  depends_on:
    - "tensor-shape-flow-v1"

equations:
  symmetric_quantize:
    formula: "q = round(x / scale), scale = max(|x|) / (2^(b-1) - 1)"
    invariants:
      - "q ∈ [-2^(b-1), 2^(b-1)-1]"
      - "dequant(q) ≈ x within quantization error"
      - "scale > 0 for non-zero input"

  q4k_block:
    formula: "block = {scale, min, quants[256]} where quants[i] = round((x[i]-min)/scale * 15)"
    invariants:
      - "quants[i] ∈ [0, 15] for Q4"
      - "block_size == 144 bytes for 256 elements"
      - "dequant recovers values within tolerance"

  compression_ratio:
    formula: "ratio = original_size / quantized_size"
    invariants:
      - "fp16: ratio ≈ 2.0"
      - "int8: ratio ≈ 4.0"
      - "int4/q4k: ratio ≈ 7-8x"

proof_obligations:
  - type: bound
    property: "Quantized value range"
    formal: "q ∈ [-2^(b-1), 2^(b-1)-1] for b-bit quantization"
  - type: bound
    property: "Dequantization error"
    formal: "|dequant(quant(x)) - x| <= scale / 2"
  - type: invariant
    property: "Q4K block size"
    formal: "block_bytes == ceil(dim/256) * 144"
  - type: invariant
    property: "Shape preservation"
    formal: "dequant(quant(tensor)).shape == tensor.shape"

falsification_tests:
  - id: FALSIFY-QUANT-001
    rule: "Range bounds"
    prediction: "all quantized values within [-128, 127] for int8"
    if_fails: "Clamping missing or scale computation wrong"
  - id: FALSIFY-QUANT-002
    rule: "Dequant error bound"
    prediction: "max |dequant(quant(x)) - x| <= scale/2"
    if_fails: "Rounding or scale computation incorrect"
  - id: FALSIFY-QUANT-003
    rule: "Q4K block size"
    prediction: "output size == ceil(dim/256) * 144 * num_rows"
    if_fails: "Per-row padding or block size calculation wrong"
  - id: FALSIFY-QUANT-004
    rule: "Zero preservation"
    prediction: "quant(0.0) dequantizes to 0.0"
    if_fails: "Zero point offset or min subtraction bug"
  - id: FALSIFY-QUANT-005
    rule: "Shape preservation"
    prediction: "dequant output has same shape as original"
    if_fails: "Reshape during quantization corrupts dimensions"

kani_harnesses:
  - id: KANI-QUANT-001
    obligation: "Range bounds"
    property: "int8 values in [-128, 127]"
    bound: 256
    strategy: exhaustive
```

### Binding Requirements

```yaml
  - contract: quantization-v1.yaml
    equation: symmetric_quantize
    module_path: "aprender::quantize"
    function: quantize_symmetric
    status: implemented

  - contract: quantization-v1.yaml
    equation: q4k_block
    module_path: "realizar::convert"
    function: quantize_q4k
    status: implemented
```
