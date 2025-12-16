# APR-FORMAT-002: APR v2 Format Specification

**Status:** Draft
**Author:** Claude Code
**Created:** 2025-12-16
**GitHub Issue:** https://github.com/paiml/aprender/issues/119
**Version:** 2.0.0-draft

## Abstract

APR v2 extends the APR format to support web-scale models (10B+ parameters) while maintaining the WASM-first design philosophy. This specification adds tensor alignment for zero-copy mmap, LZ4 streaming compression, and multi-file sharding.

## Design Principles

1. **WASM-first**: Must work in `wasm32-unknown-unknown` without Emscripten
2. **Progressive enhancement**: Features degrade gracefully (mmap → heap, compression → raw)
3. **Backward compatibility**: APR1 files remain readable
4. **Zero-copy where possible**: Alignment enables direct tensor access
5. **Streaming**: Support chunked loading for large models

## Format Overview

```
┌─────────────────────────────────────────────────────────────┐
│ Header (32 bytes, aligned)                                  │
├─────────────────────────────────────────────────────────────┤
│ Metadata Section (JSON, variable length)                    │
├─────────────────────────────────────────────────────────────┤
│ Tensor Index (binary, variable length)                      │
├─────────────────────────────────────────────────────────────┤
│ [Padding to 64-byte alignment]                              │
├─────────────────────────────────────────────────────────────┤
│ Tensor Data Section (aligned tensors)                       │
│   ├── Tensor 0 (64-byte aligned)                           │
│   ├── Tensor 1 (64-byte aligned)                           │
│   └── ...                                                   │
├─────────────────────────────────────────────────────────────┤
│ Footer (16 bytes)                                           │
└─────────────────────────────────────────────────────────────┘
```

## Header (32 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | magic | `APR2` (0x41505232) |
| 4 | 2 | version_major | Format major version (2) |
| 6 | 2 | version_minor | Format minor version (0) |
| 8 | 4 | flags | Feature flags (see below) |
| 12 | 4 | metadata_offset | Offset to metadata section |
| 16 | 4 | metadata_size | Size of metadata section |
| 20 | 4 | index_offset | Offset to tensor index |
| 24 | 4 | index_size | Size of tensor index |
| 28 | 4 | data_offset | Offset to tensor data section |

### Feature Flags

```rust
bitflags! {
    pub struct AprFlags: u32 {
        const COMPRESSED     = 0b0000_0001;  // LZ4 compression enabled
        const ALIGNED_64     = 0b0000_0010;  // 64-byte tensor alignment
        const ALIGNED_32     = 0b0000_0100;  // 32-byte tensor alignment (GGUF compat)
        const SHARDED        = 0b0000_1000;  // Multi-file model
        const ENCRYPTED      = 0b0001_0000;  // AES-256-GCM encryption
        const SIGNED         = 0b0010_0000;  // Ed25519 signature present
        const QUANTIZED      = 0b0100_0000;  // Contains quantized tensors
        const STREAMING      = 0b1000_0000;  // Streaming-optimized layout
    }
}
```

## Metadata Section

JSON object containing model configuration and auxiliary data.

### Required Keys

```json
{
  "apr_version": "2.0.0",
  "model_type": "whisper",
  "architecture": {
    "n_vocab": 51865,
    "n_audio_ctx": 1500,
    "n_text_ctx": 448,
    "n_mels": 80,
    "n_audio_layer": 4,
    "n_text_layer": 4,
    "n_audio_head": 6,
    "n_text_head": 6,
    "n_audio_state": 384,
    "n_text_state": 384
  }
}
```

### Optional Keys

```json
{
  "vocab": ["<|endoftext|>", "<|startoftranscript|>", ...],
  "mel_filterbank": [0.0, 0.0, ...],
  "mel_filterbank_shape": [80, 201],
  "tokenizer_config": { ... },
  "model_card": { ... },
  "quantization": {
    "method": "Q8_0",
    "bits_per_weight": 8.5
  }
}
```

## Tensor Index (Binary)

Replaces JSON tensor index for efficiency.

### Index Header (8 bytes)

| Offset | Size | Field |
|--------|------|-------|
| 0 | 4 | tensor_count |
| 4 | 4 | reserved |

### Tensor Entry (variable, ~40+ bytes each)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 2 | name_len | Length of tensor name |
| 2 | name_len | name | UTF-8 tensor name |
| +0 | 1 | dtype | Data type enum |
| +1 | 1 | n_dims | Number of dimensions (1-8) |
| +2 | 8×n_dims | dims | Dimension sizes (u64 each) |
| +n | 8 | offset | Byte offset in data section |
| +n+8 | 8 | size | Compressed size (or raw size) |
| +n+16 | 8 | raw_size | Uncompressed size (0 if not compressed) |
| +n+24 | 4 | flags | Per-tensor flags |

### Data Type Enum

```rust
#[repr(u8)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    I8 = 3,
    I16 = 4,
    I32 = 5,
    I64 = 6,
    U8 = 7,
    Q8_0 = 16,   // GGUF-compatible block quantization
    Q4_0 = 17,
    Q4_1 = 18,
    Q5_0 = 19,
    Q5_1 = 20,
}
```

## Tensor Data Section

Tensors are stored contiguously with alignment padding.

### Alignment

- Default: 64-byte alignment (cache-line optimal)
- GGUF-compatible: 32-byte alignment
- Alignment specified in flags

### Compression (Optional)

Per-tensor LZ4 block compression:

```
┌──────────────────────────────────────┐
│ Block Header (4 bytes)               │
│   - compressed_size: u32             │
├──────────────────────────────────────┤
│ LZ4 Compressed Data                  │
│   (64KB max uncompressed block)      │
├──────────────────────────────────────┤
│ ... more blocks ...                  │
└──────────────────────────────────────┘
```

Block size: 64KB uncompressed (streaming-friendly for WASM)

## Footer (16 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | crc32 | CRC32 of all preceding bytes |
| 4 | 4 | magic_end | `2RPA` (reverse magic) |
| 8 | 8 | file_size | Total file size for validation |

## Sharding (Multi-File)

For models > 2GB, use manifest + shard files.

### Manifest File (model.apr)

```json
{
  "apr_version": "2.0.0",
  "sharded": true,
  "shard_count": 4,
  "shards": [
    {"file": "model-00001-of-00004.apr", "size": 2147483648, "crc32": "..."},
    {"file": "model-00002-of-00004.apr", "size": 2147483648, "crc32": "..."},
    {"file": "model-00003-of-00004.apr", "size": 2147483648, "crc32": "..."},
    {"file": "model-00004-of-00004.apr", "size": 1073741824, "crc32": "..."}
  ],
  "tensor_shard_map": {
    "encoder.conv1.weight": 0,
    "encoder.conv2.weight": 0,
    "decoder.token_embedding.weight": 1,
    ...
  }
}
```

### Shard Files

Each shard is a valid APR2 file containing a subset of tensors.

## WASM Considerations

### Memory Constraints

- WASM32: 4GB address space limit
- Browser: Typically 2GB practical limit
- Solution: Streaming decompression, demand paging

### Streaming Load

```rust
pub trait StreamingLoader {
    /// Load metadata only (small, immediate)
    fn load_metadata(&mut self) -> Result<AprMetadata>;

    /// Load tensor index (moderate size)
    fn load_index(&mut self) -> Result<Vec<TensorDescriptor>>;

    /// Load single tensor on demand
    fn load_tensor(&mut self, name: &str) -> Result<Tensor>;

    /// Prefetch tensors for upcoming inference
    fn prefetch(&mut self, names: &[&str]);
}
```

### Zero-Copy (Native Only)

```rust
#[cfg(not(target_arch = "wasm32"))]
impl AprReader {
    /// Get tensor slice without copying (mmap)
    pub fn tensor_slice(&self, name: &str) -> Result<&[u8]>;
}

#[cfg(target_arch = "wasm32")]
impl AprReader {
    /// Load tensor into owned buffer (heap)
    pub fn tensor_slice(&self, name: &str) -> Result<Vec<u8>>;
}
```

## Conversion

### From GGUF

```bash
aprender convert model.gguf model.apr --format apr2
```

### From SafeTensors

```bash
aprender convert model.safetensors model.apr --format apr2 --metadata config.json
```

### APR1 to APR2

```bash
aprender upgrade model-v1.apr model-v2.apr
```

## Comparison

| Feature | APR1 | APR2 | GGUF | SafeTensors |
|---------|------|------|------|-------------|
| WASM-first | Yes | Yes | No | Yes |
| Tensor alignment | No | Yes (64B) | Yes (32B) | Yes |
| Compression | No | LZ4 | No | No |
| Quantization | Metadata | Native | Native | No |
| Sharding | No | Yes | No | Yes |
| Streaming | No | Yes | No | No |
| JSON metadata | Yes | Yes | Typed KV | JSON |
| CRC32 | Yes | Yes | No | No |

## Implementation Phases

### Phase 1: Alignment (v2.0)
- 64-byte tensor alignment
- Binary tensor index
- Backward-compatible reader

### Phase 2: Compression (v2.1)
- LZ4 block compression
- Per-tensor compression flag
- Streaming decompression

### Phase 3: Sharding (v2.2)
- Manifest file format
- Multi-file loader
- Tensor-level demand loading

## References

- APR v1: `src/serialization/apr.rs`
- GGUF: `src/format/gguf.rs`
- Bundle system: `src/bundle/`
- SafeTensors: `src/serialization/safetensors.rs`
- LZ4 frame format: https://github.com/lz4/lz4/blob/dev/doc/lz4_Frame_format.md
