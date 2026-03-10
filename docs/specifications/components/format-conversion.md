# Export / Import Component Specification

**Parent**: [apr-spec.md](../apr-spec.md) §14
**Status**: Active
**CLI**: `apr export`, `apr import`, `apr convert`
**Implementation**: `crates/apr-cli/src/commands/export.rs`, `import.rs`, `convert.rs`

---

## 1. Overview

Convert between model formats. Import from HuggingFace, GGUF, SafeTensors.
Export to GGUF, SafeTensors, MLX, ONNX, OpenVINO, CoreML.

## 2. Import

```
apr import <SOURCE> --arch <auto|llama|qwen2|...> --output <PATH>
```

### Supported Sources
- `hf://org/repo` — HuggingFace Hub
- Local SafeTensors files
- Local GGUF files (with `--preserve-q4k` for fused kernels)
- URLs

### Architecture Auto-Detection
14+ architectures: llama, qwen2, qwen3, gpt2, bert, phi, gemma, falcon,
mamba, t5, starcoder, gpt-neox, opt, whisper.

## 3. Export

```
apr export <FILE> --format <safetensors|gguf|mlx|onnx|openvino|coreml>
```

### Batch Export
```bash
apr export model.apr --batch gguf,mlx,safetensors -o exports/
```

## 4. Convert

```
apr convert <FILE> --quantize <int8|int4|q4k> --compress <lz4|zstd> -o output.apr
```

## 5. Features

- Provenance chain enforcement (`--enforce-provenance`)
- Q4K preservation for GGUF import
- Plan mode for all operations
- External tokenizer support for weights-only GGUF

---

## Provable Contracts

### Contract: `format-conversion-v1.yaml`

```yaml
metadata:
  description: "Format conversion — import/export with fidelity guarantees"
  depends_on:
    - "apr-format-v2"
    - "tensor-shape-flow-v1"

equations:
  import_roundtrip:
    formula: "export(import(source), source_format) ≈ source"
    invariants:
      - "Tensor values within tolerance after roundtrip"
      - "Metadata preserved (arch, vocab_size, etc.)"
      - "Quantized imports: dequant error bounded"

  layout_transpose:
    formula: "apr_tensor = transpose(gguf_tensor) when gguf is col-major"
    invariants:
      - "LAYOUT-002: APR is always row-major"
      - "transpose(transpose(T)) == T"
      - "Shape dimensions swapped: (m,n) → (n,m)"

  provenance_chain:
    formula: "provenance(apr) traces back to exactly one source format"
    invariants:
      - "enforce_provenance rejects pre-baked GGUF"
      - "Only SafeTensors allowed as original source"

proof_obligations:
  - type: equivalence
    property: "Tensor value fidelity"
    formal: "|imported[i] - original[i]| < ε for F32 tensors"
    tolerance: 1.0e-6
  - type: invariant
    property: "Row-major guarantee"
    formal: "all imported APR tensors are row-major"
  - type: invariant
    property: "Transpose involution"
    formal: "transpose(transpose(T)) == T"
  - type: invariant
    property: "Metadata preservation"
    formal: "import preserves arch, vocab_size, hidden_size"

falsification_tests:
  - id: FALSIFY-CONV-001
    rule: "F32 roundtrip fidelity"
    prediction: "import(safetensors) → export(safetensors) bitwise identical"
    if_fails: "Precision loss in conversion pipeline"
  - id: FALSIFY-CONV-002
    rule: "GGUF transpose"
    prediction: "GGUF col-major tensor transposed to row-major in APR"
    if_fails: "LAYOUT-002 violation: col-major leak"
  - id: FALSIFY-CONV-003
    rule: "Provenance enforcement"
    prediction: "enforce-provenance rejects GGUF source"
    if_fails: "Provenance check missing"
  - id: FALSIFY-CONV-004
    rule: "Metadata roundtrip"
    prediction: "arch field identical after import→export"
    if_fails: "Metadata dropped or renamed during conversion"
  - id: FALSIFY-CONV-005
    rule: "Unknown arch rejection"
    prediction: "unsupported architecture → clear error message"
    if_fails: "Silent fallback to wrong architecture"

kani_harnesses:
  - id: KANI-CONV-001
    obligation: "Transpose involution"
    property: "transpose(transpose(M)) == M for small matrices"
    bound: 8
    strategy: exhaustive
```

### Binding Requirements

```yaml
  - contract: format-conversion-v1.yaml
    equation: layout_transpose
    module_path: "aprender::format"
    function: transpose_gguf_to_rowmajor
    status: implemented

  - contract: format-conversion-v1.yaml
    equation: import_roundtrip
    module_path: "aprender::format"
    function: import_safetensors
    status: implemented
```
