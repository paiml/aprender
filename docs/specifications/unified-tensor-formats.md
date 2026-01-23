# Unified Tensor Format Specification

**Version**: 1.0.0
**Status**: Draft
**Created**: 2026-01-23
**Scope**: aprender (write/inspect), realizar (read/infer)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Format Comparison Matrix](#2-format-comparison-matrix)
3. [SafeTensors Format](#3-safetensors-format)
4. [GGUF Format](#4-gguf-format)
5. [APR Format](#5-apr-format)
6. [Unified Rust Traits](#6-unified-rust-traits)
7. [Data Type Mappings](#7-data-type-mappings)
8. [Dimension Conventions](#8-dimension-conventions)
9. [Stack Responsibility Matrix](#9-stack-responsibility-matrix)
10. [CLI Operations](#10-cli-operations)
11. [Performance Characteristics](#11-performance-characteristics)
12. [Migration Guide](#12-migration-guide)

---

## 1. Overview

The Sovereign AI Stack supports three tensor serialization formats with distinct use cases:

| Format | Primary Use | Origin | Ecosystem |
|--------|-------------|--------|-----------|
| **SafeTensors** | HuggingFace models, training checkpoints | HuggingFace | PyTorch, Transformers, Diffusers |
| **GGUF** | Quantized inference, edge deployment | llama.cpp | Ollama, llama.cpp, koboldcpp |
| **APR** | WASM-first, cross-platform deployment | Aprender | aprender, realizar, trueno |

### 1.1 Design Goals

1. **Unified API**: Single trait interface across all formats
2. **Zero-copy loading**: Memory-mapped tensor access where possible
3. **Format preservation**: Round-trip conversion without data loss
4. **Quantization support**: Native handling of Q4K, Q6K, Q8_0, etc.
5. **WASM compatibility**: All formats loadable in browser/WASI

---

## 2. Format Comparison Matrix

| Feature | SafeTensors | GGUF | APR |
|---------|-------------|------|-----|
| **Header Format** | JSON (8-byte len prefix) | Binary KV pairs | Binary + JSON metadata |
| **Tensor Index** | JSON embedded | Binary tensor info | Binary tensor index |
| **Data Layout** | Row-major (C-order) | Column-major (Fortran) | Row-major (C-order) |
| **Alignment** | 8-byte | 32-byte | 64-byte (configurable) |
| **Compression** | None | None | LZ4 (optional) |
| **Quantization** | Limited (I8, I4 via custom) | Extensive (Q4K, Q5K, Q6K, Q8_0) | Mirrors GGUF types |
| **Max File Size** | ~100MB header limit | No limit | No limit |
| **Sharding** | Native (index.json) | Single file | Native (manifest) |
| **Streaming** | Yes (lazy loading) | Yes (mmap) | Yes (chunked) |
| **WASM Support** | Yes | Yes | Yes (primary target) |

---

## 3. SafeTensors Format

### 3.1 Binary Layout

```
┌─────────────────────────────────────────────────┐
│ Header Length (8 bytes, u64 LE)                 │
├─────────────────────────────────────────────────┤
│ JSON Header (variable, padded to 8-byte align)  │
│ {                                               │
│   "tensor_name": {                              │
│     "dtype": "F32",                             │
│     "shape": [1024, 768],                       │
│     "data_offsets": [0, 3145728]                │
│   },                                            │
│   "__metadata__": { "format": "pt" }            │
│ }                                               │
├─────────────────────────────────────────────────┤
│ Tensor Data (contiguous, row-major)             │
│   tensor_0 bytes [offset_start..offset_end]     │
│   tensor_1 bytes [offset_start..offset_end]     │
│   ...                                           │
└─────────────────────────────────────────────────┘
```

### 3.2 Supported Data Types

```rust
pub enum SafeTensorsDtype {
    BOOL,      // 1 byte per element
    U8,        // unsigned 8-bit
    I8,        // signed 8-bit
    I16,       // signed 16-bit
    U16,       // unsigned 16-bit
    F16,       // IEEE 754 half-precision
    BF16,      // Brain floating point
    I32,       // signed 32-bit
    U32,       // unsigned 32-bit
    F32,       // IEEE 754 single-precision
    F64,       // IEEE 754 double-precision
    I64,       // signed 64-bit
    U64,       // unsigned 64-bit
    // Extended types (v0.5+)
    F8_E5M2,   // FP8 (5 exponent, 2 mantissa)
    F8_E4M3,   // FP8 (4 exponent, 3 mantissa)
}
```

### 3.3 Rust Implementation

```rust
// aprender: src/serialization/safetensors.rs
pub fn save_safetensors<P: AsRef<Path>>(
    path: P,
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    metadata: Option<HashMap<String, String>>,
) -> Result<(), SafeTensorsError>;

// realizar: src/safetensors.rs
pub struct SafeTensorsFile<'a> {
    header: SafeTensorsHeader,
    data: &'a [u8],  // mmap'd or owned
}

impl<'a> SafeTensorsFile<'a> {
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, SafeTensorsError>;
    pub fn from_mmap(path: &Path) -> Result<Self, SafeTensorsError>;
    pub fn tensor(&self, name: &str) -> Option<TensorView<'a>>;
    pub fn tensor_names(&self) -> impl Iterator<Item = &str>;
}
```

---

## 4. GGUF Format

### 4.1 Binary Layout

```
┌─────────────────────────────────────────────────┐
│ Magic (4 bytes): "GGUF" (0x46554747)            │
│ Version (4 bytes): u32 (currently 3)            │
│ Tensor Count (8 bytes): u64                     │
│ Metadata KV Count (8 bytes): u64                │
├─────────────────────────────────────────────────┤
│ Metadata Key-Value Pairs                        │
│   key_len: u64, key: [u8], type: u32, value     │
│   ...                                           │
├─────────────────────────────────────────────────┤
│ Tensor Info Array                               │
│   name_len: u64, name: [u8]                     │
│   n_dims: u32, dims: [u64; n_dims]              │
│   type: u32, offset: u64                        │
│   ...                                           │
├─────────────────────────────────────────────────┤
│ [Padding to 32-byte alignment]                  │
├─────────────────────────────────────────────────┤
│ Tensor Data (32-byte aligned, COLUMN-MAJOR)     │
│   tensor_0 bytes                                │
│   tensor_1 bytes                                │
│   ...                                           │
└─────────────────────────────────────────────────┘
```

### 4.2 Quantization Types

| Type | ID | Bits/Weight | Block Size | Description |
|------|-----|-------------|------------|-------------|
| F32 | 0 | 32 | 1 | IEEE 754 float |
| F16 | 1 | 16 | 1 | IEEE 754 half |
| Q4_0 | 2 | 4.5 | 32 | 4-bit quantized, 1 scale |
| Q4_1 | 3 | 5.0 | 32 | 4-bit + min/max |
| Q5_0 | 6 | 5.5 | 32 | 5-bit quantized |
| Q5_1 | 7 | 6.0 | 32 | 5-bit + min/max |
| Q8_0 | 8 | 8.5 | 32 | 8-bit quantized |
| Q8_1 | 9 | 9.0 | 32 | 8-bit + min/max |
| Q2_K | 10 | 2.5625 | 256 | 2-bit k-quant |
| Q3_K | 11 | 3.4375 | 256 | 3-bit k-quant |
| **Q4_K** | 12 | 4.5 | 256 | 4-bit k-quant (144 bytes/block) |
| Q5_K | 13 | 5.5 | 256 | 5-bit k-quant |
| **Q6_K** | 14 | 6.5625 | 256 | 6-bit k-quant (210 bytes/block) |
| Q8_K | 15 | 8.0 | 256 | 8-bit k-quant |
| IQ2_XXS | 16 | 2.0625 | 256 | i-quant 2-bit |
| IQ2_XS | 17 | 2.3125 | 256 | i-quant 2-bit |
| IQ3_XXS | 18 | 3.0625 | 256 | i-quant 3-bit |
| BF16 | 30 | 16 | 1 | Brain float 16 |

### 4.3 Dimension Convention (CRITICAL)

**GGUF uses GGML's column-major convention:**

```
dims[0] = ne0 = number of columns = OUTPUT dimension
dims[1] = ne1 = number of rows = INPUT dimension
```

For a weight matrix W where y = Wx:
- `dims = [output_dim, input_dim]`
- Data stored column-by-column

**Example**: FFN gate weight mapping input (1536) to intermediate (8960):
- GGUF dims: `[8960, 1536]`
- Data layout: 1536 columns of 8960 elements each
- Q4K blocks: organized per-column (8960 elements → 35 super-blocks per column)

### 4.4 Rust Implementation

```rust
// realizar: src/gguf/mod.rs
pub struct MappedGGUFModel<'a> {
    header: GGUFHeader,
    metadata: HashMap<String, GGUFValue>,
    tensor_infos: Vec<TensorInfo>,
    data: &'a [u8],
}

impl<'a> MappedGGUFModel<'a> {
    pub fn from_path(path: &Path) -> Result<Self, GGUFError>;
    pub fn get_tensor_f32(&self, name: &str) -> Result<Vec<f32>, GGUFError>;
    pub fn get_tensor_raw(&self, name: &str) -> Result<&[u8], GGUFError>;
    pub fn dequantize(&self, name: &str) -> Result<Vec<f32>, GGUFError>;
}

// Quantized tensor operations
pub fn matmul_q4k_f32_colmajor(
    q4k_data: &[u8],
    input: &[f32],
    ne0: usize,  // output dimension
    ne1: usize,  // input dimension
) -> Vec<f32>;
```

---

## 5. APR Format

### 5.1 Binary Layout

```
┌─────────────────────────────────────────────────┐
│ Header (32 bytes)                               │
│   magic: "APR2" (4 bytes)                       │
│   version: u16 major, u16 minor                 │
│   flags: u32 (feature flags)                    │
│   metadata_offset: u32                          │
│   metadata_size: u32                            │
│   index_offset: u32                             │
│   index_size: u32                               │
│   data_offset: u32                              │
├─────────────────────────────────────────────────┤
│ Metadata Section (JSON)                         │
│ {                                               │
│   "apr_version": "2.0.0",                       │
│   "model_type": "transformer_lm",               │
│   "architecture": { ... },                      │
│   "source_format": "gguf",                      │
│   "quantization": { "method": "Q4_K" }          │
│ }                                               │
├─────────────────────────────────────────────────┤
│ Tensor Index (binary)                           │
│   tensor_count: u32                             │
│   entries: [TensorEntry; count]                 │
├─────────────────────────────────────────────────┤
│ [Padding to 64-byte alignment]                  │
├─────────────────────────────────────────────────┤
│ Tensor Data (64-byte aligned, ROW-MAJOR)        │
│   tensor_0 bytes                                │
│   tensor_1 bytes                                │
│   ...                                           │
├─────────────────────────────────────────────────┤
│ Footer (16 bytes)                               │
│   crc32: u32                                    │
│   magic_end: "2RPA" (4 bytes)                   │
│   file_size: u64                                │
└─────────────────────────────────────────────────┘
```

### 5.2 APR Data Types

```rust
#[repr(u8)]
pub enum AprDtype {
    // Standard types
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    I8 = 3,
    I16 = 4,
    I32 = 5,
    I64 = 6,
    U8 = 7,

    // Quantized types (GGUF-compatible)
    Q4_K = 8,   // 4-bit k-quant, 144 bytes/256 elements
    Q6_K = 9,   // 6-bit k-quant, 210 bytes/256 elements
    Q8_0 = 10,  // 8-bit quantized
    Q4_0 = 11,  // 4-bit legacy
    Q5_K = 12,  // 5-bit k-quant
    Q2_K = 13,  // 2-bit k-quant
    Q3_K = 14,  // 3-bit k-quant
}
```

### 5.3 Feature Flags

```rust
bitflags! {
    pub struct AprFlags: u32 {
        const COMPRESSED      = 0x0001;  // LZ4 compression
        const ALIGNED_64      = 0x0002;  // 64-byte alignment
        const ALIGNED_32      = 0x0004;  // 32-byte alignment (GGUF compat)
        const SHARDED         = 0x0008;  // Multi-file model
        const ENCRYPTED       = 0x0010;  // AES-256-GCM
        const SIGNED          = 0x0020;  // Ed25519 signature
        const QUANTIZED       = 0x0040;  // Contains quantized tensors
        const GGUF_LAYOUT     = 0x0080;  // Column-major data layout
        const SAFETENSORS_SRC = 0x0100;  // Converted from SafeTensors
        const GGUF_SRC        = 0x0200;  // Converted from GGUF
    }
}
```

### 5.4 Rust Implementation

```rust
// aprender: src/format/mod.rs
pub fn write_apr<P: AsRef<Path>>(
    path: P,
    tensors: &[(String, AprTensor)],
    metadata: AprMetadata,
) -> Result<(), AprError>;

// realizar: src/apr_transformer/mod.rs
pub struct AprTransformer {
    pub config: AprTransformerConfig,
    pub token_embedding: Vec<f32>,
    pub layers: Vec<AprTransformerLayer>,
    pub q4k_layers: Option<Vec<Q4KLayerWeights>>,
    // ...
}

impl AprTransformer {
    pub fn from_apr_file(path: &Path) -> Result<Self, AprError>;
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32>;
    pub fn forward_with_cache(&mut self, token: u32, pos: usize) -> Vec<f32>;
}
```

---

## 6. Unified Rust Traits

### 6.1 Core Traits

```rust
/// A view into a tensor (borrowed or owned)
pub trait TensorView {
    fn name(&self) -> &str;
    fn dtype(&self) -> Dtype;
    fn shape(&self) -> &[usize];
    fn data_bytes(&self) -> &[u8];
    fn data_f32(&self) -> Cow<'_, [f32]>;  // Dequantize if needed
    fn is_quantized(&self) -> bool;
}

/// A tensor file that can be read
pub trait TensorReader {
    type Error: std::error::Error;
    type Tensor<'a>: TensorView where Self: 'a;

    fn from_path(path: &Path) -> Result<Self, Self::Error> where Self: Sized;
    fn from_bytes(bytes: &[u8]) -> Result<Self, Self::Error> where Self: Sized;

    fn tensor(&self, name: &str) -> Option<Self::Tensor<'_>>;
    fn tensor_names(&self) -> impl Iterator<Item = &str>;
    fn metadata(&self) -> &HashMap<String, String>;
}

/// A tensor file that can be written
pub trait TensorWriter {
    type Error: std::error::Error;

    fn new() -> Self;
    fn add_tensor(&mut self, name: &str, data: &[f32], shape: &[usize]) -> Result<(), Self::Error>;
    fn add_tensor_quantized(&mut self, name: &str, data: &[u8], dtype: Dtype, shape: &[usize]) -> Result<(), Self::Error>;
    fn set_metadata(&mut self, key: &str, value: &str);
    fn write_to_path(&self, path: &Path) -> Result<(), Self::Error>;
    fn write_to_bytes(&self) -> Result<Vec<u8>, Self::Error>;
}

/// Format-agnostic model loader
pub trait ModelLoader: TensorReader {
    type Model;

    fn load_model(&self) -> Result<Self::Model, Self::Error>;
    fn architecture(&self) -> Option<&str>;
    fn vocab_size(&self) -> Option<usize>;
    fn hidden_dim(&self) -> Option<usize>;
}
```

### 6.2 Format Implementations

```rust
// SafeTensors implementation
impl TensorReader for SafeTensorsFile<'_> { ... }
impl TensorWriter for SafeTensorsBuilder { ... }

// GGUF implementation
impl TensorReader for MappedGGUFModel<'_> { ... }
impl TensorWriter for GGUFBuilder { ... }  // Future

// APR implementation
impl TensorReader for AprFile<'_> { ... }
impl TensorWriter for AprBuilder { ... }
```

### 6.3 Format Detection

```rust
pub enum TensorFormat {
    SafeTensors,
    GGUF,
    APR,
    Unknown,
}

pub fn detect_format(path: &Path) -> TensorFormat {
    let mut file = File::open(path).ok()?;
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic).ok()?;

    match &magic[0..4] {
        b"GGUF" => TensorFormat::GGUF,
        b"APR2" | b"APR1" => TensorFormat::APR,
        _ => {
            // SafeTensors: first 8 bytes are JSON length (u64 LE)
            // Typically small value, followed by '{'
            let len = u64::from_le_bytes(magic);
            if len < 100_000_000 {
                TensorFormat::SafeTensors
            } else {
                TensorFormat::Unknown
            }
        }
    }
}

pub fn open_any(path: &Path) -> Result<Box<dyn TensorReader>, FormatError> {
    match detect_format(path) {
        TensorFormat::SafeTensors => Ok(Box::new(SafeTensorsFile::from_path(path)?)),
        TensorFormat::GGUF => Ok(Box::new(MappedGGUFModel::from_path(path)?)),
        TensorFormat::APR => Ok(Box::new(AprFile::from_path(path)?)),
        TensorFormat::Unknown => Err(FormatError::UnknownFormat),
    }
}
```

---

## 7. Data Type Mappings

### 7.1 Cross-Format Type Mapping

| SafeTensors | GGUF | APR | Bytes | Notes |
|-------------|------|-----|-------|-------|
| F32 | GGML_TYPE_F32 | F32 | 4 | Direct copy |
| F16 | GGML_TYPE_F16 | F16 | 2 | Direct copy |
| BF16 | GGML_TYPE_BF16 | BF16 | 2 | Direct copy |
| I8 | - | I8 | 1 | Direct copy |
| I32 | - | I32 | 4 | Direct copy |
| - | GGML_TYPE_Q4_K | Q4_K | 144/256 | Column-major blocks |
| - | GGML_TYPE_Q6_K | Q6_K | 210/256 | Column-major blocks |
| - | GGML_TYPE_Q8_0 | Q8_0 | 34/32 | Row-major blocks |

### 7.2 Conversion Rules

```rust
pub fn convert_dtype(src: Dtype, dst_format: TensorFormat) -> Result<Dtype, ConversionError> {
    match (src, dst_format) {
        // SafeTensors doesn't support k-quants natively
        (Dtype::Q4_K | Dtype::Q6_K, TensorFormat::SafeTensors) => {
            Ok(Dtype::F32)  // Must dequantize
        }
        // GGUF supports all types
        (dtype, TensorFormat::GGUF) => Ok(dtype),
        // APR supports all types
        (dtype, TensorFormat::APR) => Ok(dtype),
        _ => Ok(src),
    }
}
```

---

## 8. Dimension Conventions

### 8.1 Layout Comparison

| Format | Matrix Storage | Dimension Order | Example: W[out, in] |
|--------|---------------|-----------------|---------------------|
| SafeTensors | Row-major | [out_dim, in_dim] | shape=[1536, 768] |
| GGUF | Column-major | [out_dim, in_dim] | dims=[1536, 768]* |
| APR | Row-major | [out_dim, in_dim] | dims=[1536, 768] |

*GGUF stores dims as [ne0, ne1] where ne0=cols, ne1=rows in GGML convention.

### 8.2 Conversion Between Layouts

```rust
/// Convert GGUF column-major tensor to row-major
pub fn gguf_to_rowmajor(data: &[f32], dims: &[usize]) -> Vec<f32> {
    let (ne0, ne1) = (dims[0], dims[1]);  // GGUF: ne0=out_dim, ne1=in_dim
    let mut result = vec![0.0; ne0 * ne1];

    // GGUF data: column by column (ne0 elements per column, ne1 columns)
    // Row-major: row by row (ne1 elements per row, ne0 rows)
    for col in 0..ne1 {
        for row in 0..ne0 {
            let src_idx = col * ne0 + row;  // Column-major
            let dst_idx = row * ne1 + col;  // Row-major
            result[dst_idx] = data[src_idx];
        }
    }
    result
}

/// For quantized GGUF, preserve column-major and use specialized kernels
pub fn matmul_q4k_colmajor(weights: &[u8], input: &[f32], ne0: usize, ne1: usize) -> Vec<f32>;
```

---

## 9. Stack Responsibility Matrix

| Operation | aprender | realizar | trueno |
|-----------|----------|----------|--------|
| **SafeTensors Write** | `src/serialization/safetensors.rs` | - | - |
| **SafeTensors Read** | `src/inspect/safetensors.rs` | `src/safetensors.rs` | - |
| **SafeTensors Infer** | - | `src/safetensors_infer.rs` | SIMD kernels |
| **GGUF Read** | `apr inspect` (via realizar) | `src/gguf/mod.rs` | - |
| **GGUF Infer** | - | `src/gguf/mod.rs` | Q4K/Q6K kernels |
| **GGUF Write** | `apr export --format gguf` | `src/convert/mod.rs` | - |
| **APR Read** | `apr inspect` | `src/apr_transformer/mod.rs` | - |
| **APR Write** | `src/format/mod.rs` | `src/convert/mod.rs` | - |
| **APR Infer** | - | `src/apr_transformer/mod.rs` | SIMD/Q4K kernels |
| **Quantization** | - | `src/gguf/dequant.rs` | `backends/q4k.rs` |

---

## 10. CLI Operations

### 10.1 Universal Commands

```bash
# Inspect any format
apr inspect model.safetensors
apr inspect model.gguf
apr inspect model.apr

# Convert between formats
apr convert model.safetensors -o model.apr
apr convert model.gguf -o model.apr --preserve-quantization
apr convert model.apr -o model.safetensors --dequantize

# Export with quantization
apr export model.safetensors --format gguf --quantize q4_k_m -o model.gguf
apr export model.apr --format safetensors -o model.safetensors

# Run inference (auto-detect format)
apr run model.safetensors --prompt "Hello"
apr run model.gguf --prompt "Hello"
apr run model.apr --prompt "Hello"
```

### 10.2 Format-Specific Options

```bash
# SafeTensors
apr convert model.safetensors -o model.apr \
    --source-format safetensors \
    --bf16-to-f32  # Convert BF16 to F32

# GGUF
apr convert model.gguf -o model.apr \
    --preserve-quantization \  # Keep Q4K/Q6K as-is
    --preserve-layout          # Keep column-major (for fused kernels)

# APR
apr convert model.apr -o model.gguf \
    --quantize q4_k_m \        # Quantize during export
    --align 32                 # Use GGUF alignment
```

---

## 11. Performance Characteristics

### 11.1 Load Times (1.5B parameter model)

| Format | Cold Load | Warm Load (mmap) | Memory |
|--------|-----------|------------------|--------|
| SafeTensors (F32) | 2.1s | 0.3s | 6.0 GB |
| SafeTensors (F16) | 1.8s | 0.2s | 3.0 GB |
| GGUF (Q4_K_M) | 1.2s | 0.15s | 0.9 GB |
| APR (Q4_K) | 1.3s | 0.15s | 0.9 GB |

### 11.2 Inference Throughput

| Format | CPU (tok/s) | GPU (tok/s) | Notes |
|--------|-------------|-------------|-------|
| SafeTensors (F32) | 15 | 180 | No quantization |
| GGUF (Q4_K_M) | 35 | 755 | Fused kernels |
| APR (Q4_K) | 30+ | 500+ | Target parity |

---

## 12. Migration Guide

### 12.1 From SafeTensors to APR

```rust
use aprender::format::convert_safetensors_to_apr;

let result = convert_safetensors_to_apr(
    "model.safetensors",
    "model.apr",
    ConvertOptions {
        quantize: Some(QuantizationType::Q4_K),
        preserve_bf16: false,
        ..Default::default()
    }
)?;
```

### 12.2 From GGUF to APR

```rust
use realizar::convert::GgufToAprConverter;

let converter = GgufToAprConverter::new();
converter.convert(
    "model.gguf",
    "model.apr",
    GgufToAprOptions {
        preserve_quantization: true,  // Keep Q4K/Q6K bytes
        preserve_layout: true,        // Keep column-major for fused kernels
        ..Default::default()
    }
)?;
```

### 12.3 From APR to SafeTensors

```rust
use aprender::format::export_to_safetensors;

export_to_safetensors(
    "model.apr",
    "model.safetensors",
    ExportOptions {
        dequantize: true,  // Required: SafeTensors doesn't support Q4K
        dtype: Dtype::F16, // Optional: convert to F16
        ..Default::default()
    }
)?;
```

---

## Appendix A: Q4_K Super-Block Format

```
Q4_K Super-Block (144 bytes for 256 elements):
┌─────────────────────────────────────────────────┐
│ d: f16 (2 bytes) - main scale                   │
│ dmin: f16 (2 bytes) - min scale                 │
│ scales: [u8; 12] - sub-block scales (6-bit)     │
│ qs: [u8; 128] - quantized values (4-bit pairs)  │
└─────────────────────────────────────────────────┘

GGML Column-Major Layout:
- Data organized per-column (not per-row)
- Each column has ceil(ne0 / 256) super-blocks
- Total blocks = ceil(ne0 / 256) × ne1
```

## Appendix B: Q6_K Super-Block Format

```
Q6_K Super-Block (210 bytes for 256 elements):
┌─────────────────────────────────────────────────┐
│ ql: [u8; 128] - low 4 bits of quantized values  │
│ qh: [u8; 64] - high 2 bits of quantized values  │
│ scales: [i8; 16] - sub-block scales             │
│ d: f16 (2 bytes) - main scale                   │
└─────────────────────────────────────────────────┘
```

---

## References

1. [SafeTensors Specification](https://github.com/huggingface/safetensors)
2. [GGUF Format Documentation](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
3. [llama.cpp Quantization](https://github.com/ggerganov/llama.cpp/blob/master/ggml-quants.c)
4. [APR Specification](./APR-SPEC.md)
