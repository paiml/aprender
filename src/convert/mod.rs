//! GGUF to APR Transformer Converter
//!
//! Converts GGUF models to APR Transformer format for fair comparison.
//! All weights are dequantized to F32 for WASM compatibility.
//!
//! ## Example
//!
//! ```rust,ignore
//! use realizar::convert::GgufToAprConverter;
//!
//! let gguf_data = std::fs::read("model.gguf")?;
//! let apr_transformer = GgufToAprConverter::convert(&gguf_data)?;
//!
//! // Save to APR format
//! let apr_bytes = apr_transformer.to_apr_bytes()?;
//! std::fs::write("model.apr_transformer", apr_bytes)?;
//! ```

use crate::apr::{AprHeader, TensorEntry, ALIGNMENT, HEADER_SIZE, MAGIC};
use crate::apr_transformer::{AprTransformer, AprTransformerConfig, AprTransformerLayer};
use crate::error::{RealizarError, Result};
use crate::gguf::{GGUFModel, GGUFTransformer};

// ============================================================================
// CRC32 (IEEE polynomial, matches aprender's format/v2/mod.rs)
// ============================================================================

/// CRC32 checksum (IEEE polynomial 0xEDB88320)
/// Used for APR v2 header checksum verification
fn crc32(data: &[u8]) -> u32 {
    const TABLE: [u32; 256] = {
        let mut table = [0u32; 256];
        let mut i = 0;
        while i < 256 {
            let mut crc = i as u32;
            let mut j = 0;
            while j < 8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xEDB8_8320;
                } else {
                    crc >>= 1;
                }
                j += 1;
            }
            table[i] = crc;
            i += 1;
        }
        table
    };

    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        let idx = ((crc ^ u32::from(byte)) & 0xFF) as usize;
        crc = TABLE[idx] ^ (crc >> 8);
    }
    !crc
}

/// Compute APR v2 header checksum
/// Checksum is CRC32 over bytes [0..40] + [44..64], excluding checksum field at [40..44]
fn compute_apr_header_checksum(header: &[u8]) -> u32 {
    let mut data = Vec::with_capacity(60);
    data.extend_from_slice(header.get(0..40).expect("APR header requires 64 bytes"));
    data.extend_from_slice(header.get(44..64).expect("APR header requires 64 bytes"));
    crc32(&data)
}

/// GGUF to APR Transformer converter
///
/// Converts GGUF models with quantized weights to APR format with F32 weights.
/// This enables fair comparison between GGUF and APR serving performance.
pub struct GgufToAprConverter;

impl GgufToAprConverter {
    /// Convert GGUF file bytes to APR Transformer
    ///
    /// # Arguments
    ///
    /// * `gguf_data` - Raw GGUF file bytes
    ///
    /// # Returns
    ///
    /// `AprTransformer` with dequantized F32 weights
    ///
    /// # Errors
    ///
    /// Returns error if GGUF parsing or conversion fails
    pub fn convert(gguf_data: &[u8]) -> Result<AprTransformer> {
        // Parse GGUF model
        let gguf_model = GGUFModel::from_bytes(gguf_data)?;

        // Load transformer weights (dequantizes to F32)
        let gguf_transformer = GGUFTransformer::from_gguf(&gguf_model, gguf_data)?;

        // Convert to APR format
        Ok(Self::from_gguf_transformer(&gguf_transformer))
    }

    /// Convert from existing `GGUFTransformer` to `AprTransformer`
    ///
    /// # Arguments
    ///
    /// * `gguf` - Loaded GGUF transformer with dequantized weights
    ///
    /// # Returns
    ///
    /// `AprTransformer` with the same weights
    pub fn from_gguf_transformer(gguf: &GGUFTransformer) -> AprTransformer {
        let config = AprTransformerConfig {
            architecture: gguf.config.architecture.clone(),
            hidden_dim: gguf.config.hidden_dim,
            num_layers: gguf.config.num_layers,
            num_heads: gguf.config.num_heads,
            num_kv_heads: gguf.config.num_kv_heads,
            vocab_size: gguf.config.vocab_size,
            intermediate_dim: gguf.config.intermediate_dim,
            context_length: gguf.config.context_length,
            rope_theta: gguf.config.rope_theta,
            eps: gguf.config.eps,
            eos_token_id: gguf.config.eos_token_id,
            ..Default::default()
        };

        let layers = gguf
            .layers
            .iter()
            .map(|l| AprTransformerLayer {
                attn_norm_weight: l.attn_norm_weight.clone(),
                attn_norm_bias: l.attn_norm_bias.clone(),
                qkv_weight: l.qkv_weight.clone(),
                qkv_bias: l.qkv_bias.clone(),
                attn_output_weight: l.attn_output_weight.clone(),
                attn_output_bias: l.attn_output_bias.clone(),
                ffn_gate_weight: l.ffn_gate_weight.clone(),
                ffn_gate_bias: l.ffn_gate_bias.clone(),
                ffn_up_weight: l.ffn_up_weight.clone(),
                ffn_up_bias: l.ffn_up_bias.clone(),
                ffn_down_weight: l.ffn_down_weight.clone(),
                ffn_down_bias: l.ffn_down_bias.clone(),
                ffn_norm_weight: l.ffn_norm_weight.clone(),
                ffn_norm_bias: l.ffn_norm_bias.clone(),
                attn_q_norm_weight: l.attn_q_norm_weight.clone(),
                attn_k_norm_weight: l.attn_k_norm_weight.clone(),
                linear_attn_z_weight: None,
                linear_attn_b_weight: None,
                linear_attn_a_weight: None,
                linear_attn_conv1d_weight: None,
                linear_attn_a_log: None,
                linear_attn_dt_bias: None,
                linear_attn_norm_weight: None,
                moe_gate_weight: None,
                moe_expert_gate_up: None,
                moe_expert_down: None,
                moe_shared_gate: None,
                moe_shared_up: None,
                moe_shared_down: None,
                moe_shared_expert_gate_weight: None,
            })
            .collect();

        AprTransformer {
            config,
            token_embedding: gguf.token_embedding.clone(),
            layers,
            output_norm_weight: gguf.output_norm_weight.clone(),
            output_norm_bias: gguf.output_norm_bias.clone(),
            lm_head_weight: gguf.lm_head_weight.clone(),
            lm_head_bias: gguf.lm_head_bias.clone(),
            q4k_layers: None,
            lm_head_weight_q6k: None,
            lm_head_weight_q4k: None,
        }
    }

    /// Convert APR Transformer to serialized APR v2 bytes
    ///
    /// Creates a valid .apr v2 file with:
    /// - APR v2 header (64 bytes)
    /// - JSON metadata (padded to 64-byte boundary)
    /// - Tensor index (JSON array)
    /// - Tensor data (each 64-byte aligned)
    ///
    /// # Arguments
    ///
    /// * `transformer` - APR Transformer to serialize
    ///
    /// # Returns
    ///
    /// Raw bytes in APR v2 format
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails
    // serde_json::json!() uses infallible unwrap
    #[allow(clippy::disallowed_methods)]
    #[allow(clippy::cast_possible_truncation)]
    pub fn to_apr_bytes(transformer: &AprTransformer) -> Result<Vec<u8>> {
        // Serialize metadata
        let metadata = serde_json::json!({
            "model_type": "transformer_lm",
            "architecture": transformer.config.architecture,
            "hidden_size": transformer.config.hidden_dim,
            "num_layers": transformer.config.num_layers,
            "num_heads": transformer.config.num_heads,
            "num_kv_heads": transformer.config.num_kv_heads,
            "vocab_size": transformer.config.vocab_size,
            "intermediate_dim": transformer.config.intermediate_dim,
            "context_length": transformer.config.context_length,
            "rope_theta": transformer.config.rope_theta,
            "eps": transformer.config.eps,
        });
        let metadata_bytes =
            serde_json::to_vec(&metadata).map_err(|e| RealizarError::FormatError {
                reason: format!("Failed to serialize metadata: {e}"),
            })?;

        // Pad metadata to 64-byte boundary
        let metadata_padded_len = metadata_bytes.len().div_ceil(ALIGNMENT) * ALIGNMENT;

        // Serialize weights as single tensor (JSON payload for now)
        let payload_bytes =
            serde_json::to_vec(transformer).map_err(|e| RealizarError::FormatError {
                reason: format!("Failed to serialize weights: {e}"),
            })?;

        // Create tensor index with single entry for the full payload
        let tensor_entries = vec![TensorEntry {
            name: "weights".to_string(),
            dtype: "json".to_string(),
            shape: vec![payload_bytes.len()],
            offset: 0,
            size: payload_bytes.len() as u64,
        }];
        let tensor_index_bytes =
            serde_json::to_vec(&tensor_entries).map_err(|e| RealizarError::FormatError {
                reason: format!("Failed to serialize tensor index: {e}"),
            })?;

        // Calculate offsets
        let metadata_offset = HEADER_SIZE as u64;
        let tensor_index_offset = metadata_offset + metadata_padded_len as u64;
        let data_offset = tensor_index_offset + tensor_index_bytes.len() as u64;

        // Build APR v2 header (64 bytes)
        let mut header = vec![0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(&MAGIC);
        header[4] = 2; // version major
        header[5] = 0; // version minor
        header[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags
        header[8..12].copy_from_slice(&1u32.to_le_bytes()); // tensor_count
        header[12..20].copy_from_slice(&metadata_offset.to_le_bytes());
        header[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        header[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
        header[32..40].copy_from_slice(&data_offset.to_le_bytes());
        // Compute and set checksum (CRC32 over header excluding checksum field)
        let checksum = compute_apr_header_checksum(&header);
        header[40..44].copy_from_slice(&checksum.to_le_bytes());
        // bytes 44-63 reserved

        // Combine all parts
        let total_size =
            HEADER_SIZE + metadata_padded_len + tensor_index_bytes.len() + payload_bytes.len();
        let mut result = Vec::with_capacity(total_size);
        result.extend_from_slice(&header);
        result.extend_from_slice(&metadata_bytes);
        result.resize(HEADER_SIZE + metadata_padded_len, 0); // pad metadata
        result.extend_from_slice(&tensor_index_bytes);
        result.extend_from_slice(&payload_bytes);

        Ok(result)
    }

    /// Load APR Transformer from APR v2 bytes
    ///
    /// # Arguments
    ///
    /// * `data` - Raw APR v2 file bytes
    ///
    /// # Returns
    ///
    /// Loaded `AprTransformer`
    ///
    /// # Errors
    ///
    /// Returns error if parsing fails
    pub fn from_apr_bytes(data: &[u8]) -> Result<AprTransformer> {
        // Parse header
        let header = AprHeader::from_bytes(data)?;

        // Get tensor index to find the weights tensor
        let index_start = header.tensor_index_offset as usize;
        let index_end = header.data_offset as usize;

        if data.len() < index_end {
            return Err(RealizarError::FormatError {
                reason: format!(
                    "APR file truncated: expected {} bytes for tensor index, got {}",
                    index_end,
                    data.len()
                ),
            });
        }

        let tensor_entries: Vec<TensorEntry> =
            serde_json::from_slice(&data[index_start..index_end]).map_err(|e| {
                RealizarError::FormatError {
                    reason: format!("Failed to parse tensor index: {e}"),
                }
            })?;

        // Find the weights tensor
        let weights_entry = tensor_entries
            .iter()
            .find(|e| e.name == "weights")
            .ok_or_else(|| RealizarError::FormatError {
                reason: "No 'weights' tensor found in APR file".to_string(),
            })?;

        // Extract weights data
        let data_start = header.data_offset as usize + weights_entry.offset as usize;
        let data_end = data_start + weights_entry.size as usize;

        if data.len() < data_end {
            return Err(RealizarError::FormatError {
                reason: format!(
                    "APR file truncated: expected {} bytes for tensor data, got {}",
                    data_end,
                    data.len()
                ),
            });
        }

        let payload_bytes = &data[data_start..data_end];

        // Deserialize transformer
        let transformer: AprTransformer =
            serde_json::from_slice(payload_bytes).map_err(|e| RealizarError::FormatError {
                reason: format!("Failed to deserialize transformer: {e}"),
            })?;

        Ok(transformer)
    }

    /// Get conversion statistics
    ///
    /// # Arguments
    ///
    /// * `transformer` - APR Transformer to analyze
    ///
    /// # Returns
    ///
    /// Statistics about the conversion
    pub fn stats(transformer: &AprTransformer) -> ConversionStats {
        let params = transformer.num_parameters();
        let memory_bytes = transformer.memory_size();

        ConversionStats {
            total_parameters: params,
            memory_bytes_f32: memory_bytes,
            num_layers: transformer.config.num_layers,
            hidden_dim: transformer.config.hidden_dim,
            vocab_size: transformer.config.vocab_size,
            architecture: transformer.config.architecture.clone(),
        }
    }
}

/// Statistics about a converted model
#[derive(Debug, Clone)]
pub struct ConversionStats {
    /// Total number of parameters
    pub total_parameters: usize,
    /// Memory size in bytes (F32)
    pub memory_bytes_f32: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Model architecture name
    pub architecture: String,
}

impl ConversionStats {
    /// Memory size in MB
    #[must_use]
    pub fn memory_mb(&self) -> f64 {
        self.memory_bytes_f32 as f64 / (1024.0 * 1024.0)
    }

    /// Memory size in GB
    #[must_use]
    pub fn memory_gb(&self) -> f64 {
        self.memory_bytes_f32 as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Parameters in millions
    #[must_use]
    pub fn parameters_m(&self) -> f64 {
        self.total_parameters as f64 / 1_000_000.0
    }

    /// Parameters in billions
    #[must_use]
    pub fn parameters_b(&self) -> f64 {
        self.total_parameters as f64 / 1_000_000_000.0
    }
}

// =============================================================================
// Q4K APR Converter (preserves GGUF quantization for GPU inference)
// =============================================================================

/// Raw tensor with preserved quantization
#[derive(Debug, Clone)]
pub struct RawTensor {
    /// Tensor name
    pub name: String,
    /// Raw bytes (Q4K super-blocks or F32/F16 data)
    pub data: Vec<u8>,
    /// Tensor shape (logical elements, not bytes)
    pub shape: Vec<usize>,
    /// GGML dtype: 0=F32, 1=F16, 6=Q4_K, 7=Q5_K, 8=Q8_0, 12=Q6_K
    pub dtype: u32,
}

/// GGUF to APR Q4K converter (preserves quantization)
///
/// Unlike `GgufToAprConverter` which dequantizes to F32, this converter
/// preserves Q4K/Q6K quantization for GPU inference with batched GEMV.
///
/// This is essential for achieving 2X Ollama performance.
pub struct GgufToAprQ4KConverter;

include!("q4k_converter_helpers.rs");
include!("mod_03.rs");
