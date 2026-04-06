//! GGUF Test Factory - Synthesizes valid GGUF files in memory
//!
//! This module provides `GGUFBuilder` for creating valid GGUF v3 files
//! without needing real model files. Essential for testing transformer
//! loading code that requires properly formatted binary data.
//!
//! # Example
//!
//! ```ignore
//! let data = GGUFBuilder::new()
//!     .architecture("llama")
//!     .hidden_dim(64)
//!     .num_layers(1)
//!     .add_f32_tensor("token_embd.weight", &[100, 64], &embedding_data)
//!     .add_q4_k_tensor("blk.0.attn_q.weight", &[64, 64], &q4k_data)
//!     .build();
//!
//! let model = GGUFModel::from_bytes(&data)?;
//! ```

use super::types::{
    GGUF_ALIGNMENT, GGUF_MAGIC, GGUF_TYPE_F16, GGUF_TYPE_F32, GGUF_TYPE_Q2_K, GGUF_TYPE_Q4_0,
    GGUF_TYPE_Q4_1, GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_0, GGUF_TYPE_Q5_1, GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K,
    GGUF_TYPE_Q8_0, GGUF_VERSION_V3,
};

/// Builder for creating valid GGUF v3 files in memory
pub struct GGUFBuilder {
    /// Metadata key-value pairs (key, type, value_bytes)
    metadata: Vec<(String, u32, Vec<u8>)>,
    /// Tensor info entries (name, dims, qtype, data)
    tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)>,
}

impl Default for GGUFBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GGUFBuilder {
    /// Create a new GGUF builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: Vec::new(),
            tensors: Vec::new(),
        }
    }

    // =========================================================================
    // Metadata Helpers
    // =========================================================================

    /// Add a string metadata value
    #[must_use]
    pub fn add_string(mut self, key: &str, value: &str) -> Self {
        let mut bytes = Vec::new();
        // String: u64 length + UTF-8 bytes
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
        self.metadata.push((key.to_string(), 8, bytes)); // type 8 = string
        self
    }

    /// Add a u32 metadata value
    #[must_use]
    pub fn add_u32(mut self, key: &str, value: u32) -> Self {
        self.metadata
            .push((key.to_string(), 4, value.to_le_bytes().to_vec())); // type 4 = u32
        self
    }

    /// Add a f32 metadata value
    #[must_use]
    pub fn add_f32(mut self, key: &str, value: f32) -> Self {
        self.metadata
            .push((key.to_string(), 6, value.to_le_bytes().to_vec())); // type 6 = f32
        self
    }

    /// Set architecture (shorthand for general.architecture)
    #[must_use]
    pub fn architecture(self, arch: &str) -> Self {
        self.add_string(super::keys::GENERAL_ARCHITECTURE, arch)
    }

    /// Set hidden dimension (embedding length)
    #[must_use]
    pub fn hidden_dim(self, arch: &str, dim: u32) -> Self {
        self.add_u32(
            &super::keys::arch_key(arch, super::keys::EMBEDDING_LENGTH),
            dim,
        )
    }

    /// Set number of layers (block count)
    #[must_use]
    pub fn num_layers(self, arch: &str, count: u32) -> Self {
        self.add_u32(
            &super::keys::arch_key(arch, super::keys::BLOCK_COUNT),
            count,
        )
    }

    /// Set number of attention heads
    #[must_use]
    pub fn num_heads(self, arch: &str, count: u32) -> Self {
        self.add_u32(
            &super::keys::arch_key(arch, super::keys::ATTENTION_HEAD_COUNT),
            count,
        )
    }

    /// Set number of KV heads (for GQA)
    #[must_use]
    pub fn num_kv_heads(self, arch: &str, count: u32) -> Self {
        self.add_u32(
            &super::keys::arch_key(arch, super::keys::ATTENTION_HEAD_COUNT_KV),
            count,
        )
    }

    /// Set context length
    #[must_use]
    pub fn context_length(self, arch: &str, len: u32) -> Self {
        self.add_u32(
            &super::keys::arch_key(arch, super::keys::CONTEXT_LENGTH),
            len,
        )
    }

    /// Set RoPE frequency base
    #[must_use]
    pub fn rope_freq_base(self, arch: &str, base: f32) -> Self {
        self.add_f32(
            &super::keys::arch_key(arch, super::keys::ROPE_FREQ_BASE),
            base,
        )
    }

    /// Set RMS epsilon
    #[must_use]
    pub fn rms_epsilon(self, arch: &str, eps: f32) -> Self {
        self.add_f32(
            &super::keys::arch_key(arch, super::keys::ATTENTION_LAYER_NORM_RMS_EPSILON),
            eps,
        )
    }

    /// Set feed-forward hidden dimension
    #[must_use]
    pub fn ffn_hidden_dim(self, arch: &str, dim: u32) -> Self {
        self.add_u32(
            &super::keys::arch_key(arch, super::keys::FEED_FORWARD_LENGTH),
            dim,
        )
    }

    /// Set vocab size (for completeness)
    #[must_use]
    pub fn vocab_size(self, _arch: &str, size: u32) -> Self {
        // Vocab size is typically inferred from token_embd.weight shape
        // But we can store it in metadata if needed
        self.add_u32(super::keys::TOKENIZER_TOKENS_SIZE, size)
    }

    // =========================================================================
    // Tensor Helpers
    // =========================================================================

    /// Add an F32 tensor
    #[must_use]
    pub fn add_f32_tensor(mut self, name: &str, dims: &[u64], data: &[f32]) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.tensors
            .push((name.to_string(), dims.to_vec(), GGUF_TYPE_F32, bytes));
        self
    }

    /// Add a Q4_0 tensor (18 bytes per 32 elements)
    #[must_use]
    pub fn add_q4_0_tensor(mut self, name: &str, dims: &[u64], data: &[u8]) -> Self {
        self.tensors.push((
            name.to_string(),
            dims.to_vec(),
            GGUF_TYPE_Q4_0,
            data.to_vec(),
        ));
        self
    }

    /// Add a Q8_0 tensor (34 bytes per 32 elements)
    #[must_use]
    pub fn add_q8_0_tensor(mut self, name: &str, dims: &[u64], data: &[u8]) -> Self {
        self.tensors.push((
            name.to_string(),
            dims.to_vec(),
            GGUF_TYPE_Q8_0,
            data.to_vec(),
        ));
        self
    }

    /// Add a Q4_K tensor (144 bytes per 256 elements)
    #[must_use]
    pub fn add_q4_k_tensor(mut self, name: &str, dims: &[u64], data: &[u8]) -> Self {
        self.tensors.push((
            name.to_string(),
            dims.to_vec(),
            GGUF_TYPE_Q4_K,
            data.to_vec(),
        ));
        self
    }

    /// Add a Q5_K tensor (176 bytes per 256 elements)
    #[must_use]
    pub fn add_q5_k_tensor(mut self, name: &str, dims: &[u64], data: &[u8]) -> Self {
        self.tensors.push((
            name.to_string(),
            dims.to_vec(),
            GGUF_TYPE_Q5_K,
            data.to_vec(),
        ));
        self
    }

    /// Add a Q6_K tensor (210 bytes per 256 elements)
    #[must_use]
    pub fn add_q6_k_tensor(mut self, name: &str, dims: &[u64], data: &[u8]) -> Self {
        self.tensors.push((
            name.to_string(),
            dims.to_vec(),
            GGUF_TYPE_Q6_K,
            data.to_vec(),
        ));
        self
    }

    /// Add a Q2_K tensor (84 bytes per 256 elements)
    #[must_use]
    pub fn add_q2_k_tensor(mut self, name: &str, dims: &[u64], data: &[u8]) -> Self {
        self.tensors.push((
            name.to_string(),
            dims.to_vec(),
            GGUF_TYPE_Q2_K,
            data.to_vec(),
        ));
        self
    }

    /// Add an F16 tensor (2 bytes per element)
    #[must_use]
    pub fn add_f16_tensor(mut self, name: &str, dims: &[u64], data: &[u8]) -> Self {
        self.tensors.push((
            name.to_string(),
            dims.to_vec(),
            GGUF_TYPE_F16,
            data.to_vec(),
        ));
        self
    }

    /// Add a Q4_1 tensor (20 bytes per 32 elements)
    #[must_use]
    pub fn add_q4_1_tensor(mut self, name: &str, dims: &[u64], data: &[u8]) -> Self {
        self.tensors.push((
            name.to_string(),
            dims.to_vec(),
            GGUF_TYPE_Q4_1,
            data.to_vec(),
        ));
        self
    }

    /// Add a Q5_0 tensor (22 bytes per 32 elements)
    #[must_use]
    pub fn add_q5_0_tensor(mut self, name: &str, dims: &[u64], data: &[u8]) -> Self {
        self.tensors.push((
            name.to_string(),
            dims.to_vec(),
            GGUF_TYPE_Q5_0,
            data.to_vec(),
        ));
        self
    }

    /// Add a Q5_1 tensor (24 bytes per 32 elements)
    #[must_use]
    pub fn add_q5_1_tensor(mut self, name: &str, dims: &[u64], data: &[u8]) -> Self {
        self.tensors.push((
            name.to_string(),
            dims.to_vec(),
            GGUF_TYPE_Q5_1,
            data.to_vec(),
        ));
        self
    }

    // =========================================================================
    // Additional Metadata Helpers
    // =========================================================================

    /// Add a u8 metadata value (type 0)
    #[must_use]
    pub fn add_u8(mut self, key: &str, value: u8) -> Self {
        self.metadata.push((key.to_string(), 0, vec![value]));
        self
    }

    /// Add an i8 metadata value (type 1)
    #[must_use]
    pub fn add_i8(mut self, key: &str, value: i8) -> Self {
        self.metadata.push((key.to_string(), 1, vec![value as u8]));
        self
    }

    /// Add a u16 metadata value (type 2)
    #[must_use]
    pub fn add_u16(mut self, key: &str, value: u16) -> Self {
        self.metadata
            .push((key.to_string(), 2, value.to_le_bytes().to_vec()));
        self
    }

    /// Add an i16 metadata value (type 3)
    #[must_use]
    pub fn add_i16(mut self, key: &str, value: i16) -> Self {
        self.metadata
            .push((key.to_string(), 3, value.to_le_bytes().to_vec()));
        self
    }

    /// Add an i32 metadata value (type 5)
    #[must_use]
    pub fn add_i32(mut self, key: &str, value: i32) -> Self {
        self.metadata
            .push((key.to_string(), 5, value.to_le_bytes().to_vec()));
        self
    }

    /// Add a bool metadata value (type 7)
    #[must_use]
    pub fn add_bool(mut self, key: &str, value: bool) -> Self {
        self.metadata
            .push((key.to_string(), 7, vec![u8::from(value)]));
        self
    }

    /// Add a u64 metadata value (type 10)
    #[must_use]
    pub fn add_u64(mut self, key: &str, value: u64) -> Self {
        self.metadata
            .push((key.to_string(), 10, value.to_le_bytes().to_vec()));
        self
    }

    /// Add an i64 metadata value (type 11)
    #[must_use]
    pub fn add_i64(mut self, key: &str, value: i64) -> Self {
        self.metadata
            .push((key.to_string(), 11, value.to_le_bytes().to_vec()));
        self
    }

    /// Add an f64 metadata value (type 12)
    #[must_use]
    pub fn add_f64(mut self, key: &str, value: f64) -> Self {
        self.metadata
            .push((key.to_string(), 12, value.to_le_bytes().to_vec()));
        self
    }

    /// Add a string array metadata value (type 9, element_type 8)
    #[must_use]
    pub fn add_string_array(mut self, key: &str, values: &[&str]) -> Self {
        let mut bytes = Vec::new();
        // Array header: element_type (u32) + array_len (u64)
        bytes.extend_from_slice(&8u32.to_le_bytes()); // element type = string
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        // Elements: each is u64 length + UTF-8 bytes
        for &val in values {
            bytes.extend_from_slice(&(val.len() as u64).to_le_bytes());
            bytes.extend_from_slice(val.as_bytes());
        }
        self.metadata.push((key.to_string(), 9, bytes));
        self
    }

    // =========================================================================
    // Build
    // =========================================================================

    /// Build the GGUF file as a byte vector
    #[must_use]
    pub fn build(self) -> Vec<u8> {
        let mut data = Vec::new();

        // Header
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
        data.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        data.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());

        // Metadata
        for (key, value_type, value_bytes) in &self.metadata {
            // Key string: u64 length + UTF-8 bytes
            data.extend_from_slice(&(key.len() as u64).to_le_bytes());
            data.extend_from_slice(key.as_bytes());
            // Value type
            data.extend_from_slice(&value_type.to_le_bytes());
            // Value bytes
            data.extend_from_slice(value_bytes);
        }

        // Tensor info
        let mut tensor_data_offset = 0u64;
        for (name, dims, qtype, tensor_bytes) in &self.tensors {
            // Name string
            data.extend_from_slice(&(name.len() as u64).to_le_bytes());
            data.extend_from_slice(name.as_bytes());

            // n_dims
            data.extend_from_slice(&(dims.len() as u32).to_le_bytes());

            // Dimensions (reversed for GGML order)
            for dim in dims.iter().rev() {
                data.extend_from_slice(&dim.to_le_bytes());
            }

            // Quantization type
            data.extend_from_slice(&qtype.to_le_bytes());

            // Offset (relative to tensor data start)
            data.extend_from_slice(&tensor_data_offset.to_le_bytes());

            tensor_data_offset += tensor_bytes.len() as u64;
        }

        // Align to GGUF_ALIGNMENT (32 bytes)
        let current_len = data.len();
        let aligned = current_len.div_ceil(GGUF_ALIGNMENT) * GGUF_ALIGNMENT;
        data.resize(aligned, 0);

        // Tensor data
        for (_, _, _, tensor_bytes) in &self.tensors {
            data.extend_from_slice(tensor_bytes);
        }

        data
    }
}

include!("test_factory_create.rs");
include!("test_factory_build_minimal.rs");
include!("test_factory_gguf_builder.rs");
