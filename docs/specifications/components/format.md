# APR Binary Format Specification

**Parent**: [apr-spec.md](../apr-spec.md) §2
**Status**: Active
**Implementation**: `aprender::format`, `aprender::native`

---

## 1. Overview

APR v2 is a zero-copy binary model format with LZ4/ZSTD tensor compression,
64-byte aligned tensor data, and streaming support for WASM environments.

## 2. Binary Layout

```
┌──────────────────────────────────────┐
│ Header (32 bytes)                    │
│   Magic: APR\x02                    │
│   Version: u16                      │
│   Feature flags: u32                │
│   Metadata offset: u64              │
│   Metadata size: u64                │
│   Reserved: 6 bytes                 │
├──────────────────────────────────────┤
│ Metadata (JSON, padded to 64B)      │
│   arch, vocab_size, hidden_size,    │
│   num_layers, num_heads, etc.       │
├──────────────────────────────────────┤
│ Tensor Index (binary)               │
│   Per tensor:                       │
│     name_len: u16                   │
│     name: [u8; name_len]            │
│     dtype: u8                       │
│     ndim: u8                        │
│     dims: [u64; ndim]               │
│     offset: u64                     │
│     size: u64                       │
├──────────────────────────────────────┤
│ Tensor Data (64-byte aligned)       │
│   Raw or compressed (LZ4/ZSTD)      │
├──────────────────────────────────────┤
│ Footer (16 bytes)                   │
│   Checksum, total size              │
└──────────────────────────────────────┘
```

## 3. Feature Flags

| Bit | Flag | Description |
|-----|------|-------------|
| 0 | `COMPRESSED_LZ4` | Tensor data uses LZ4 compression |
| 1 | `COMPRESSED_ZSTD` | Tensor data uses ZSTD compression |
| 2 | `SHARDED` | Multi-file model |
| 3 | `QUANTIZED` | Contains quantized tensors |

## 4. Sharding

Models >2GB split across multiple files: `model-00001-of-00004.apr`.
Each shard is self-contained with its own header and tensor index.

## 5. WASM Considerations

- Streaming tensor loading (no full file in memory)
- No mmap (WASM lacks mmap support)
- LZ4 decompression in WASM via trueno

## 6. Row-Major Mandate (LAYOUT-002)

APR is exclusively row-major. GGUF column-major data is transposed at import.
There is ONE WAY ONLY.

---

## Provable Contracts

### Contract: `apr-format-v2.yaml`

```yaml
metadata:
  description: "APR v2 binary format — header, index, data integrity"
  depends_on:
    - "tensor-shape-flow-v1"

equations:
  header_magic:
    formula: "bytes[0..4] == [0x41, 0x50, 0x52, 0x02]"
    invariants:
      - "Magic bytes identify APR v2 format"
      - "Version field is u16 little-endian"

  alignment:
    formula: "tensor_data_offset % 64 == 0"
    invariants:
      - "All tensor data starts at 64-byte boundary"
      - "Metadata padded to 64-byte alignment"

  checksum:
    formula: "footer.checksum == hash(header + metadata + index + data)"
    invariants:
      - "Any byte modification detected"
      - "Checksum covers all sections"

  roundtrip:
    formula: "from_apr_bytes(to_apr_bytes(model)) == model"
    invariants:
      - "No data loss through serialization"
      - "Tensor values bitwise identical"

proof_obligations:
  - type: invariant
    property: "64-byte alignment"
    formal: "tensor_data_offset % 64 == 0"
  - type: invariant
    property: "Roundtrip fidelity"
    formal: "deserialize(serialize(model)).tensors == model.tensors"
  - type: invariant
    property: "Magic validation"
    formal: "invalid magic → parse error (not garbage output)"
  - type: completeness
    property: "All tensors indexed"
    formal: "len(index) == len(data_segments)"

falsification_tests:
  - id: FALSIFY-FMT-001
    rule: "Alignment"
    prediction: "tensor data offset is 64-byte aligned for all models"
    if_fails: "Padding computation incorrect"
  - id: FALSIFY-FMT-002
    rule: "Roundtrip"
    prediction: "serialize → deserialize produces bitwise identical tensors"
    if_fails: "Serialization drops or corrupts tensor data"
  - id: FALSIFY-FMT-003
    rule: "Magic rejection"
    prediction: "file with wrong magic bytes returns Err, not panic"
    if_fails: "Missing magic validation"
  - id: FALSIFY-FMT-004
    rule: "Checksum integrity"
    prediction: "flipping any byte in file → checksum mismatch error"
    if_fails: "Checksum covers incomplete range"

kani_harnesses:
  - id: KANI-FMT-001
    obligation: "Alignment"
    property: "padding to 64-byte boundary is correct"
    bound: 256
    strategy: exhaustive
```

### Binding Requirements

```yaml
  - contract: apr-format-v2.yaml
    equation: roundtrip
    module_path: "aprender::format"
    function: "to_apr_bytes / from_apr_bytes"
    status: implemented
```
