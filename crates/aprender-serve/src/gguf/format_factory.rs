//! Rosetta Format Factory - Synthetic Model Files for All Formats
//!
//! Following aprender's Rosetta Stone spec, this module provides builders
//! for synthesizing valid model files in ALL supported formats:
//!
//! | Format | Builder | Magic |
//! |--------|---------|-------|
//! | GGUF | `GGUFBuilder` | "GGUF" |
//! | SafeTensors | `SafetensorsBuilder` | JSON header |
//! | APR | `AprBuilder` | "APR\0" |
//!
//! # Conversion Matrix (6 Direct Paths)
//!
//! ```text
//!     GGUF ←──────→ APR ←──────→ SafeTensors
//!       ↑                              ↑
//!       └──────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use realizar::gguf::format_factory::{GGUFBuilder, SafetensorsBuilder, AprBuilder};
//!
//! // Create a minimal model in each format
//! let gguf_data = GGUFBuilder::minimal_llama(100, 64);
//! let st_data = SafetensorsBuilder::minimal_model(100, 64);
//! let apr_data = AprBuilder::minimal_model(100, 64);
//! ```

use serde::Serialize;
use std::collections::BTreeMap;

// Re-export GGUFBuilder from test_factory
pub use super::test_factory::{
    build_minimal_llama_gguf, create_f32_embedding_data, create_f32_norm_weights, create_q4_0_data,
    create_q8_0_data, GGUFBuilder,
};

// =============================================================================
// SafeTensors Builder
// =============================================================================

/// SafeTensors tensor metadata
#[derive(Debug, Clone, Serialize)]
struct SafetensorsTensorMeta {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// Builder for creating valid SafeTensors files in memory
///
/// SafeTensors format:
/// - 8 bytes: JSON header length (little-endian u64)
/// - N bytes: JSON header with tensor metadata
/// - Tensor data (contiguous, aligned)
pub struct SafetensorsBuilder {
    tensors: Vec<(String, String, Vec<usize>, Vec<u8>)>, // name, dtype, shape, data
}

impl Default for SafetensorsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SafetensorsBuilder {
    /// Create a new SafeTensors builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
        }
    }

    /// Add an F32 tensor
    #[must_use]
    pub fn add_f32_tensor(mut self, name: &str, shape: &[usize], data: &[f32]) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.tensors
            .push((name.to_string(), "F32".to_string(), shape.to_vec(), bytes));
        self
    }

    /// Add an F16 tensor
    #[must_use]
    pub fn add_f16_tensor(mut self, name: &str, shape: &[usize], data: &[u8]) -> Self {
        self.tensors.push((
            name.to_string(),
            "F16".to_string(),
            shape.to_vec(),
            data.to_vec(),
        ));
        self
    }

    /// Add a BF16 tensor
    #[must_use]
    pub fn add_bf16_tensor(mut self, name: &str, shape: &[usize], data: &[u8]) -> Self {
        self.tensors.push((
            name.to_string(),
            "BF16".to_string(),
            shape.to_vec(),
            data.to_vec(),
        ));
        self
    }

    /// Build the SafeTensors file as a byte vector
    #[must_use]
    pub fn build(self) -> Vec<u8> {
        // Calculate offsets and build metadata
        let mut metadata: BTreeMap<String, SafetensorsTensorMeta> = BTreeMap::new();
        let mut current_offset = 0usize;

        for (name, dtype, shape, data) in &self.tensors {
            let end_offset = current_offset + data.len();
            metadata.insert(
                name.clone(),
                SafetensorsTensorMeta {
                    dtype: dtype.clone(),
                    shape: shape.clone(),
                    data_offsets: [current_offset, end_offset],
                },
            );
            current_offset = end_offset;
        }

        // Serialize metadata to JSON
        let json = serde_json::to_string(&metadata).expect("JSON serialization");
        let json_bytes = json.as_bytes();

        // Build final file
        let mut data = Vec::new();

        // Header: JSON length as u64
        data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());

        // JSON metadata
        data.extend_from_slice(json_bytes);

        // Tensor data
        for (_, _, _, tensor_data) in &self.tensors {
            data.extend_from_slice(tensor_data);
        }

        data
    }

    /// Build a minimal SafeTensors model for testing
    #[must_use]
    pub fn minimal_model(vocab_size: usize, hidden_dim: usize) -> Vec<u8> {
        let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
        let norm_data = create_f32_norm_weights(hidden_dim);

        Self::new()
            .add_f32_tensor(
                "model.embed_tokens.weight",
                &[vocab_size, hidden_dim],
                &embed_data,
            )
            .add_f32_tensor("model.norm.weight", &[hidden_dim], &norm_data)
            .build()
    }
}

// =============================================================================
// APR Builder
// =============================================================================

/// APR v2 format constants
const APR_MAGIC: &[u8; 4] = b"APR\0";
const APR_VERSION_MAJOR: u8 = 2;
const APR_VERSION_MINOR: u8 = 0;
const APR_HEADER_SIZE: usize = 64;
const APR_ALIGNMENT: usize = 64;

/// Builder for creating valid APR v2 files in memory
///
/// APR v2 format (64-byte header):
/// - 4 bytes: Magic "APR\0"
/// - 2 bytes: Version (major.minor)
/// - 2 bytes: Flags
/// - 4 bytes: Tensor count
/// - 8 bytes: Metadata offset
/// - 4 bytes: Metadata size
/// - 8 bytes: Tensor index offset
/// - 8 bytes: Data offset
/// - 4 bytes: Checksum
/// - 20 bytes: Reserved
pub struct AprBuilder {
    metadata: BTreeMap<String, serde_json::Value>,
    tensors: Vec<(String, Vec<usize>, u32, Vec<u8>)>, // name, shape, dtype, data
}

impl Default for AprBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// APR dtype codes
pub const APR_DTYPE_F32: u32 = 0;
pub const APR_DTYPE_F16: u32 = 1;
pub const APR_DTYPE_Q4_0: u32 = 2;
pub const APR_DTYPE_Q8_0: u32 = 8;

impl AprBuilder {
    /// Create a new APR builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: BTreeMap::new(),
            tensors: Vec::new(),
        }
    }

    /// Set architecture metadata
    #[must_use]
    pub fn architecture(mut self, arch: &str) -> Self {
        self.metadata
            .insert("architecture".to_string(), serde_json::json!(arch));
        self
    }

    /// Set hidden dimension metadata
    #[must_use]
    pub fn hidden_dim(mut self, dim: usize) -> Self {
        self.metadata
            .insert("hidden_dim".to_string(), serde_json::json!(dim));
        self
    }

    /// Set number of layers metadata
    #[must_use]
    pub fn num_layers(mut self, count: usize) -> Self {
        self.metadata
            .insert("num_layers".to_string(), serde_json::json!(count));
        self
    }

    /// Add an F32 tensor
    #[must_use]
    pub fn add_f32_tensor(mut self, name: &str, shape: &[usize], data: &[f32]) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.tensors
            .push((name.to_string(), shape.to_vec(), APR_DTYPE_F32, bytes));
        self
    }

    /// Add a Q4_0 tensor
    #[must_use]
    pub fn add_q4_0_tensor(mut self, name: &str, shape: &[usize], data: &[u8]) -> Self {
        self.tensors.push((
            name.to_string(),
            shape.to_vec(),
            APR_DTYPE_Q4_0,
            data.to_vec(),
        ));
        self
    }

    /// Add a Q8_0 tensor
    #[must_use]
    pub fn add_q8_0_tensor(mut self, name: &str, shape: &[usize], data: &[u8]) -> Self {
        self.tensors.push((
            name.to_string(),
            shape.to_vec(),
            APR_DTYPE_Q8_0,
            data.to_vec(),
        ));
        self
    }

    /// Build the APR v2 file as a byte vector
    #[must_use]
    pub fn build(self) -> Vec<u8> {
        let mut data = Vec::new();

        // Serialize metadata to JSON
        let json = serde_json::to_string(&self.metadata).expect("JSON serialization");
        let json_bytes = json.as_bytes();

        // Pad JSON to 64-byte boundary
        let json_padded_len = json_bytes.len().div_ceil(APR_ALIGNMENT) * APR_ALIGNMENT;

        // Build tensor index
        let mut tensor_index = Vec::new();
        let mut tensor_data_offset = 0u64;

        for (name, shape, dtype, tensor_bytes) in &self.tensors {
            // Tensor index entry: name_len(4) + name + ndims(4) + dims + dtype(4) + offset(8) + size(8)
            let name_bytes = name.as_bytes();
            tensor_index.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            tensor_index.extend_from_slice(name_bytes);
            tensor_index.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for dim in shape {
                tensor_index.extend_from_slice(&(*dim as u64).to_le_bytes());
            }
            tensor_index.extend_from_slice(&dtype.to_le_bytes());
            tensor_index.extend_from_slice(&tensor_data_offset.to_le_bytes());
            tensor_index.extend_from_slice(&(tensor_bytes.len() as u64).to_le_bytes());

            // Align tensor data to 64 bytes
            let aligned_size = tensor_bytes.len().div_ceil(APR_ALIGNMENT) * APR_ALIGNMENT;
            tensor_data_offset += aligned_size as u64;
        }

        // Pad tensor index to 64-byte boundary
        let index_padded_len = tensor_index.len().div_ceil(APR_ALIGNMENT) * APR_ALIGNMENT;

        // Calculate offsets
        let metadata_offset = APR_HEADER_SIZE as u64;
        let tensor_index_offset = metadata_offset + json_padded_len as u64;
        let data_offset = tensor_index_offset + index_padded_len as u64;

        // Write header (64 bytes)
        data.extend_from_slice(APR_MAGIC);
        data.push(APR_VERSION_MAJOR);
        data.push(APR_VERSION_MINOR);
        data.extend_from_slice(&0u16.to_le_bytes()); // flags
        data.extend_from_slice(&(self.tensors.len() as u32).to_le_bytes());
        data.extend_from_slice(&metadata_offset.to_le_bytes());
        data.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        data.extend_from_slice(&tensor_index_offset.to_le_bytes());
        data.extend_from_slice(&data_offset.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes()); // checksum (placeholder)
        data.extend([0u8; 20]); // reserved

        assert_eq!(data.len(), APR_HEADER_SIZE);

        // Write metadata (padded)
        data.extend_from_slice(json_bytes);
        data.resize(APR_HEADER_SIZE + json_padded_len, 0);

        // Write tensor index (padded)
        data.extend_from_slice(&tensor_index);
        data.resize(APR_HEADER_SIZE + json_padded_len + index_padded_len, 0);

        // Write tensor data (each aligned to 64 bytes)
        for (_, _, _, tensor_bytes) in &self.tensors {
            let start = data.len();
            data.extend_from_slice(tensor_bytes);
            let aligned_end = start + tensor_bytes.len().div_ceil(APR_ALIGNMENT) * APR_ALIGNMENT;
            data.resize(aligned_end, 0);
        }

        data
    }

    /// Build a minimal APR model for testing
    #[must_use]
    pub fn minimal_model(vocab_size: usize, hidden_dim: usize) -> Vec<u8> {
        let embed_data = create_f32_embedding_data(vocab_size, hidden_dim);
        let norm_data = create_f32_norm_weights(hidden_dim);

        Self::new()
            .architecture("llama")
            .hidden_dim(hidden_dim)
            .num_layers(1)
            .add_f32_tensor("token_embd.weight", &[vocab_size, hidden_dim], &embed_data)
            .add_f32_tensor("output_norm.weight", &[hidden_dim], &norm_data)
            .build()
    }
}

// =============================================================================
// Format Detection
// =============================================================================

/// Detected model format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatType {
    /// GGUF format (llama.cpp)
    Gguf,
    /// SafeTensors format (HuggingFace)
    SafeTensors,
    /// APR format (Aprender)
    Apr,
    /// Unknown format
    Unknown,
}

impl FormatType {
    /// Detect format from magic bytes (Genchi Genbutsu)
    #[must_use]
    pub fn from_magic(data: &[u8]) -> Self {
        if data.len() < 8 {
            return Self::Unknown;
        }

        // GGUF: "GGUF" magic
        if data.get(0..4).expect("len >= 8 checked above") == b"GGUF" {
            return Self::Gguf;
        }

        // APR: "APR\0" magic
        let magic4 = data.get(0..4).expect("len >= 8 checked above");
        if magic4 == b"APR\0" || magic4 == b"APR2" {
            return Self::Apr;
        }

        // SafeTensors: u64 header length followed by '{"'
        if data.len() >= 10 {
            let header_len = u64::from_le_bytes(
                data.get(0..8)
                    .expect("len >= 10 checked above")
                    .try_into()
                    .unwrap_or([0; 8]),
            );
            if header_len < 100_000_000
                && data.get(8..10).expect("len >= 10 checked above") == b"{\""
            {
                return Self::SafeTensors;
            }
        }

        Self::Unknown
    }
}

include!("format_factory_safetensors_builder.rs");
