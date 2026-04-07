---
title: GGUF Export Pipeline via Aprender Delegation
issue: ENT-GGUF-EXPORT
status: In Progress
created: 2026-02-14T16:31:41.228873161+00:00
updated: 2026-02-14T16:32:00.000000000+00:00
---

# GGUF Export Pipeline via Aprender Delegation

**Ticket ID**: ENT-GGUF-EXPORT
**Status**: Complete

## Summary

Entrenar delegates all GGUF v3 binary serialization to `aprender::format::gguf::export_tensors_to_gguf()`.
Entrenar retains ownership of the quantization layer (`GgufQuantization`, Q4_0/Q8_0 encoding) and the
high-level export pipeline (`quantize_and_export()`). This spec defines the falsifiable invariants that
must hold across the export path.

## Falsifiable Claims

### C-001: GGUF Magic and Version

Every file produced by `Exporter::export_gguf()` starts with bytes `b"GGUF"` followed by version `3u32` LE.
This holds for all three quantization modes (None/Q4_0/Q8_0) and all metadata combinations.

**Disconfirming evidence**: Any exported `.gguf` file where bytes 0..4 != `b"GGUF"` or bytes 4..8 != `3u32 LE`.

### C-002: Tensor Count Header Matches Actual Tensors

The GGUF header field `tensor_count` (bytes 8..16, u64 LE) equals the number of tensor info entries
that can be parsed from the file, which equals `ModelWeights.tensors.len()`.

**Disconfirming evidence**: `verify_gguf(data).tensor_count != weights.tensors.len()`.

### C-003: Metadata Count Header Matches Actual Metadata

The GGUF header field `metadata_count` (bytes 16..24, u64 LE) equals the number of metadata KV pairs
that can be parsed. With `include_metadata=true`, the count equals:
`1 (param_count) + [arch].is_some() + [name].is_some() + [hidden_size].is_some() + [num_layers].is_some()`.
With `include_metadata=false`, the count is 0.

**Disconfirming evidence**: Any exported file where parsed metadata count differs from the formula.

### C-004: Alphabetical Tensor Ordering

Tensors in the exported GGUF file appear in lexicographic (alphabetical) order by name, regardless of
insertion order into `ModelWeights`. This ensures deterministic output.

**Disconfirming evidence**: Any pair of adjacent tensors where `tensors[i].name > tensors[i+1].name`.

### C-005: Deterministic Output

Exporting identical `ModelWeights` with identical `Exporter` configuration produces byte-identical
GGUF files across multiple invocations.

**Disconfirming evidence**: Two exports of same input producing different file bytes.

### C-006: Quantization Size Ordering

For the same tensor data, file sizes obey: `size(Q4_0) < size(Q8_0) < size(F32)`.
This must hold for any tensor with >= 32 elements.

**Disconfirming evidence**: Any tensor count >= 32 where the size ordering is violated.

### C-007: Q4_0 Block Encoding Size

Q4_0 encoding produces exactly `ceil(n_elements / 32) * 18` bytes
(2-byte f16 scale + 16 bytes packed 4-bit data per block).

**Disconfirming evidence**: `quantize_to_gguf_bytes(data, Q4_0).0.len() != ceil(data.len() / 32) * 18`.

### C-008: Q8_0 Block Encoding Size

Q8_0 encoding produces exactly `ceil(n_elements / 32) * 34` bytes (2-byte f16 scale + 32 bytes i8 data per block).

**Disconfirming evidence**: `quantize_to_gguf_bytes(data, Q8_0).0.len() != ceil(data.len() / 32) * 34`.

### C-009: F32 Data Integrity

F32 (unquantized) tensor data survives the full pipeline bit-for-bit. Every f32 value written can be
recovered from the GGUF file at the correct offset with identical IEEE 754 bits.

**Disconfirming evidence**: Any f32 value where `original.to_bits() != recovered.to_bits()` (excluding NaN).

### C-010: Empty Data Handling

`quantize_to_gguf_bytes(&[], mode)` returns `(vec!(), dtype)` for all three modes. No panic, no allocation.

**Disconfirming evidence**: Empty input producing non-empty output or panicking.

### C-011: PyTorch Format Rejection

`Exporter::export(weights, ExportFormat::PyTorch, filename)` always returns `Err(FetchError::PickleSecurityRisk)`.

**Disconfirming evidence**: PyTorch export succeeding or returning a different error variant.

### C-012: ExportFormat::from_path Detection

Format detection from file extension follows these rules:
- `.gguf` → GGUF
- `.safetensors` → SafeTensors
- `.apr.json` or `.apr` → APR
- `.pt` or `.bin` → PyTorch
- Other extensions → None

**Disconfirming evidence**: Any extension returning the wrong format or None when it should match.

### C-013: File Size Monotonicity

For the same quantization mode, adding more tensors always increases file size.

**Disconfirming evidence**: `export(n+1 tensors).size <= export(n tensors).size`.

### C-014: export_auto Format Detection

`Exporter::export_auto(weights, filename)` uses `ExportFormat::from_path` to detect format,
falling back to `default_format` when extension is unrecognized.

**Disconfirming evidence**: `export_auto("model.gguf")` not producing GGUF format output.

### C-015: README Generation

`quantize_and_export()` generates a README containing:
- YAML frontmatter with tags (entrenar, gguf, quantized)
- Model name from metadata
- Architecture from metadata
- Quantization mode name
- File size in human-readable format
- Tensor count

**Disconfirming evidence**: Any of these fields missing from the generated README.

### C-016: Metadata Value Types

All metadata values written via the export pipeline must use the correct GGUF value types:
architecture → String, param_count → Uint64, hidden_size → Uint32, num_layers → Uint32, name → String.

**Disconfirming evidence**: Any metadata KV pair where the stored GGUF type tag does not match the expected type.

## Acceptance Criteria

- [x] AC-01: All exported GGUF files start with magic `b"GGUF"` and version 3
- [x] AC-02: Tensor count in header matches actual tensor info entries
- [x] AC-03: Metadata count in header matches actual metadata KV pairs
- [x] AC-04: Tensors appear in alphabetical order by name
- [x] AC-05: Identical inputs produce byte-identical outputs (determinism)
- [x] AC-06: Q4_0 files are smaller than Q8_0 files which are smaller than F32 files
- [x] AC-07: F32 tensor data survives roundtrip bit-for-bit
- [x] AC-08: PyTorch format export is rejected with `PickleSecurityRisk`
- [x] AC-09: `export_auto()` detects format from file extension
- [x] AC-10: All 143+ export tests pass with zero failures
- [x] AC-11: Zero clippy warnings on export module
- [x] AC-12: Stress test validates 100+ tensors in a single file

## Architecture

### Module Structure

| Module | Responsibility |
|--------|---------------|
| `gguf_writer.rs` | `GgufQuantization` enum, `quantize_to_gguf_bytes()`, Q4_0/Q8_0 block encoding |
| `exporter.rs` | `Exporter` builder, delegates GGUF writing to `aprender::format::gguf` |
| `pipeline.rs` | `quantize_and_export()` high-level pipeline, README generation |
| `gguf_verify.rs` | GGUF file parser/validator for testing |
| `format.rs` | `ExportFormat` enum with extension detection |
| `weights.rs` | `ModelWeights` container with metadata |

### Data Flow

```
ModelWeights → quantize_to_gguf_bytes() → GgufTensor[] → export_tensors_to_gguf() → .gguf file
                                                          ↑ (aprender crate)
```

### Quantization Bridge

```rust
/// Quantize f32 data and return raw GGUF bytes + dtype
pub fn quantize_to_gguf_bytes(data: &[f32], quant: GgufQuantization) -> (Vec<u8>, GgmlType) {
    match quant {
        GgufQuantization::None => {
            let bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
            (bytes, GgmlType::F32)
        }
        GgufQuantization::Q4_0 => {
            let quantized = Q4_0::quantize(data);
            (encode_q4_0_blocks(&quantized), GgmlType::Q4_0)
        }
        GgufQuantization::Q8_0 => {
            let quantized = Q8_0::quantize(data);
            (encode_q8_0_blocks(&quantized), GgmlType::Q8_0)
        }
    }
}
```

### GGUF Export via Aprender

```rust
fn export_gguf(&self, weights: &ModelWeights, path: &Path) -> Result<ExportResult> {
    use aprender::format::gguf::{export_tensors_to_gguf, GgufTensor, GgufValue};

    let mut metadata: Vec<(String, GgufValue)> = Vec::new();
    if self.include_metadata {
        if let Some(arch) = &weights.metadata.architecture {
            metadata.push(("general.architecture".into(), GgufValue::String(arch.clone())));
        }
        // ... additional metadata
    }

    let mut tensor_names: Vec<&String> = weights.tensors.keys().collect();
    tensor_names.sort(); // Deterministic alphabetical order

    let tensors: Vec<GgufTensor> = tensor_names.iter().map(|name| {
        let (bytes, dtype) = quantize_to_gguf_bytes(data, self.gguf_quantization);
        GgufTensor { name: (*name).clone(), shape, dtype, data: bytes }
    }).collect();

    let mut file = std::fs::File::create(path)?;
    export_tensors_to_gguf(&mut file, &tensors, &metadata)?;
    Ok(ExportResult { path, format: GGUF, size_bytes, num_tensors })
}
```

### Exporter Builder Pattern

```rust
let exporter = Exporter::new()
    .output_dir("/output")
    .gguf_quantization(GgufQuantization::Q4_0)
    .include_metadata(true);
let result = exporter.export(&weights, ExportFormat::GGUF, "model.gguf")?;
```

### Pipeline API

```rust
let result = quantize_and_export(&weights, GgufQuantization::Q8_0, output_dir, "model.gguf")?;
assert!(result.export.size_bytes > 0);
assert!(result.readme.is_some());
```

## Testing Strategy

### Test Requirements

- Unit tests must verify each quantization mode (None, Q4_0, Q8_0) produces correct block sizes
- Integration tests must validate the full export pipeline from ModelWeights to .gguf file
- Property tests should require at least 50 cases per invariant with proptest
- Stress tests must require validation of 100+ tensors in a single GGUF file
- Regression tests should require that PyTorch format is always rejected with PickleSecurityRisk

### Test Coverage: 143 tests across 5 files

- **gguf_writer.rs**: 27 tests (block encoding, edge cases, property tests)
- **gguf_verify.rs**: 47 tests (parser, roundtrip, stress, metadata combos, property tests)
- **exporter.rs**: 16 tests (builder, format detection, rejection, regression)
- **pipeline.rs**: 28 tests (full pipeline, size ordering, README, data integrity)
- **format.rs**: 12 tests (from_path, extension, is_safe, Display)

### Property-Based Tests (proptest)

- Arbitrary f32 tensor roundtrip (1-6 tensors, 1-64 elements each)
- Metadata count always matches (0-8 entries)
- Q4_0/Q8_0 dtype preservation across arbitrary sizes
- GGUF header validity (0-5 tensors, 0-5 metadata)
- Alphabetical sort stability (20 tensors, 1000 seeds)
- F32 byte preservation (1-128 elements)
- Q4_0/Q8_0 byte size invariant (1-512 elements)

## Success Criteria

- All 15 falsifiable claims have corresponding test evidence
- 143+ tests pass with zero failures
- Zero clippy warnings
- All property tests run 50-100 cases each
- Stress test covers 100+ tensors

## Academic Citations

| Citation | Relevance |
|----------|-----------|
| **Dettmers et al. (2023)** "QLoRA: Efficient Finetuning of Quantized LLMs" [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) | Q4_0 and Q8_0 quantization block formats |
| **Frantar et al. (2022)** "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) | Block-wise quantization theory |
| **Popper (1959)** "The Logic of Scientific Discovery" ISBN: 0415278449 | Falsifiability criterion for spec verification |
| **Popper (1963)** "Conjectures and Refutations" ISBN: 0415285941 | Hypothesis testing methodology |
| **IEEE 754-2019** "IEEE Standard for Floating-Point Arithmetic" [IEEE](https://doi.org/10.1109/IEEESTD.2019.8766229) | F32/F16 bit-level data integrity requirements |
| **Lin et al. (2024)** "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) | Weight quantization compression theory |
| **Jia & Harman (2011)** "An Analysis and Survey of the Development of Mutation Testing" [IEEE TSE](https://ieeexplore.ieee.org/document/5487526) | Mutation testing as falsification oracle |
| **Gerganov (2023)** "GGML GGUF Format Specification" [GitHub](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) | Canonical GGUF v3 binary format specification |
| **Gerganov (2023)** "llama.cpp GGUF Implementation" [GitHub](https://github.com/ggerganov/llama.cpp) | Reference GGUF implementation for compatibility |
