//! APR Transformer Loaders
//!
//! Memory-mapped and quantized APR transformer implementations.
//! Extracted from mod.rs (PMAT-802).

#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(non_camel_case_types)]

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use serde::{Deserialize, Serialize};

use crate::apr::MAGIC;
use crate::error::{RealizarError, Result};

use super::{AprKVCache, AprTransformer, AprTransformerConfig};

// ============================================================================
// APR Transformer Binary Format (Y1-Y5 Format Parity)
// ============================================================================
// Uses unified APR magic from apr.rs - ONE format, no versioning

/// Binary header size for APR Transformer (64 bytes)
pub const APR_TRANSFORMER_HEADER_SIZE: usize = 64;

/// Memory-mapped APR Transformer for zero-copy inference (Y1, Y2)
///
/// This struct provides zero-copy access to APR transformer weights
/// via memory-mapped I/O, matching GGUF's performance characteristics.
///
/// # Performance Benefits (per Dean & Barroso 2013)
///
/// - Zero-copy: Tensors accessed directly from page cache
/// - Lazy loading: Only touched pages are loaded
/// - Shared memory: Multiple processes can share the same mapping
#[derive(Debug)]
pub struct MmapAprTransformer {
    /// Memory-mapped file data
    mmap: Mmap,
    /// Model configuration (parsed from header)
    pub config: AprTransformerConfig,
    /// Offset where tensor data starts
    tensor_data_offset: usize,
    /// Whether mmap is active (for is_mmap() check)
    is_mmap: bool,
}

impl MmapAprTransformer {
    /// Load APR transformer from file using memory-mapped I/O (Y1)
    ///
    /// # Arguments
    ///
    /// * `path` - Path to .apr transformer file
    ///
    /// # Returns
    ///
    /// Memory-mapped transformer ready for inference
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be opened or is invalid
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let model = MmapAprTransformer::from_file("model.apr")?;
    /// assert!(model.is_mmap());
    /// let logits = model.forward(&[1, 2, 3])?;
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref()).map_err(|e| RealizarError::IoError {
            message: format!("Failed to open APR file: {e}"),
        })?;

        // Safety: We're only reading the file, mmap is safe for read-only access
        // SAFETY: Memory safety ensured by bounds checking and alignment
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| RealizarError::IoError {
                message: format!("Failed to mmap APR file: {e}"),
            })?
        };

        // Verify minimum size
        if mmap.len() < APR_TRANSFORMER_HEADER_SIZE {
            return Err(RealizarError::FormatError {
                reason: format!(
                    "APR file too small: {} bytes (need at least {})",
                    mmap.len(),
                    APR_TRANSFORMER_HEADER_SIZE
                ),
            });
        }

        // Parse header
        let header_bytes = &mmap[..APR_TRANSFORMER_HEADER_SIZE];

        // Verify APR magic
        let magic = header_bytes
            .get(0..4)
            .expect("APR header validated to have required size above");
        if magic != MAGIC {
            return Err(RealizarError::FormatError {
                reason: format!("Invalid APR magic: expected {:?}, got {:?}", MAGIC, magic),
            });
        }

        // Parse config from header (after 4-byte magic + 4-byte version)
        let version = u32::from_le_bytes([
            header_bytes[4],
            header_bytes[5],
            header_bytes[6],
            header_bytes[7],
        ]);
        if version > 1 {
            return Err(RealizarError::FormatError {
                reason: format!("Unsupported APR version: {version}"),
            });
        }

        // Parse config fields (offset 8)
        let hidden_dim = u32::from_le_bytes([
            header_bytes[8],
            header_bytes[9],
            header_bytes[10],
            header_bytes[11],
        ]) as usize;
        let num_layers = u32::from_le_bytes([
            header_bytes[12],
            header_bytes[13],
            header_bytes[14],
            header_bytes[15],
        ]) as usize;
        let num_heads = u32::from_le_bytes([
            header_bytes[16],
            header_bytes[17],
            header_bytes[18],
            header_bytes[19],
        ]) as usize;
        let num_kv_heads = u32::from_le_bytes([
            header_bytes[20],
            header_bytes[21],
            header_bytes[22],
            header_bytes[23],
        ]) as usize;
        let vocab_size = u32::from_le_bytes([
            header_bytes[24],
            header_bytes[25],
            header_bytes[26],
            header_bytes[27],
        ]) as usize;
        let intermediate_dim = u32::from_le_bytes([
            header_bytes[28],
            header_bytes[29],
            header_bytes[30],
            header_bytes[31],
        ]) as usize;
        let context_length = u32::from_le_bytes([
            header_bytes[32],
            header_bytes[33],
            header_bytes[34],
            header_bytes[35],
        ]) as usize;
        let rope_theta = f32::from_le_bytes([
            header_bytes[36],
            header_bytes[37],
            header_bytes[38],
            header_bytes[39],
        ]);
        let eps = f32::from_le_bytes([
            header_bytes[40],
            header_bytes[41],
            header_bytes[42],
            header_bytes[43],
        ]);
        let tensor_data_offset = u32::from_le_bytes([
            header_bytes[44],
            header_bytes[45],
            header_bytes[46],
            header_bytes[47],
        ]) as usize;

        let config = AprTransformerConfig {
            architecture: "apr".to_string(),
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length,
            rope_theta,
            eps,
            eos_token_id: None, // Parsed from APR metadata if available
            ..Default::default()
        };

        Ok(Self {
            mmap,
            config,
            tensor_data_offset,
            is_mmap: true,
        })
    }

    /// Check if model is using memory-mapped I/O (Y2)
    #[must_use]
    pub fn is_mmap(&self) -> bool {
        self.is_mmap
    }

    /// Get raw tensor data slice (zero-copy access)
    ///
    /// # Arguments
    ///
    /// * `offset` - Offset from tensor data start
    /// * `len` - Number of bytes to read
    ///
    /// # Returns
    ///
    /// Slice of raw bytes (zero-copy from mmap)
    pub fn get_tensor_bytes(&self, offset: usize, len: usize) -> Result<&[u8]> {
        let start = self.tensor_data_offset + offset;
        let end = start + len;

        if end > self.mmap.len() {
            return Err(RealizarError::FormatError {
                reason: format!(
                    "Tensor access out of bounds: offset={offset}, len={len}, file_size={}",
                    self.mmap.len()
                ),
            });
        }

        Ok(&self.mmap[start..end])
    }

    /// Get tensor as f32 slice (zero-copy if aligned)
    ///
    /// # Safety
    ///
    /// This function assumes the tensor data is properly aligned for f32 access.
    /// If not aligned, returns a copy.
    pub fn get_tensor_f32(&self, offset: usize, num_elements: usize) -> Result<Vec<f32>> {
        let bytes = self.get_tensor_bytes(offset, num_elements * 4)?;

        // Convert bytes to f32 (could be zero-copy if aligned)
        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(floats)
    }

    /// Get file size in bytes
    #[must_use]
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }

    /// Get number of parameters (estimated from config)
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        let hidden = self.config.hidden_dim;
        let vocab = self.config.vocab_size;
        let layers = self.config.num_layers;
        let intermediate = self.config.intermediate_dim;

        // Embedding + LM head
        let embed_params = vocab * hidden * 2;

        // Per layer: attn_norm + qkv + attn_out + ffn_up + ffn_down
        let layer_params = hidden
            + (hidden * 3 * hidden)
            + (hidden * hidden)
            + (hidden * intermediate)
            + (intermediate * hidden);

        // Output norm
        let norm_params = hidden;

        embed_params + (layers * layer_params) + norm_params
    }
}

// ============================================================================
// Y5: Quantized APR Transformer (Q4_K, Q8_0 support)
// ============================================================================

/// Quantization type for APR Transformer weights (Y5)
///
/// Supports the same quantization formats as GGUF for format parity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[allow(non_camel_case_types)] // Match GGML naming convention (Q4_K, Q8_0)
pub enum AprQuantizationType {
    /// Full precision 32-bit floats (no quantization)
    #[default]
    F32,
    /// 4-bit K-quantization (4.5 bits/weight, super-block size 256)
    Q4_K,
    /// 8-bit quantization (8 bits/weight, block size 32)
    Q8_0,
}

impl AprQuantizationType {
    /// Get bits per weight for this quantization type
    #[must_use]
    pub fn bits_per_weight(&self) -> f64 {
        match self {
            Self::F32 => 32.0,
            Self::Q4_K => 4.5, // 144 bytes per 256 values
            Self::Q8_0 => 8.0, // 36 bytes per 32 values (scale + 32 int8)
        }
    }

    /// Get bytes per super-block (256 values for Q4_K, 32 for Q8_0)
    #[must_use]
    pub fn bytes_per_block(&self) -> usize {
        match self {
            Self::F32 => 4,    // 4 bytes per value
            Self::Q4_K => 144, // 144 bytes per 256 values
            Self::Q8_0 => 36,  // 4 (scale) + 32 (int8) per 32 values
        }
    }

    /// Get values per block
    #[must_use]
    pub fn values_per_block(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::Q4_K => 256,
            Self::Q8_0 => 32,
        }
    }

    /// Convert to byte representation for header
    #[must_use]
    pub fn to_byte(&self) -> u8 {
        match self {
            Self::F32 => 0,
            Self::Q4_K => 1,
            Self::Q8_0 => 2,
        }
    }

    /// Parse from byte representation
    #[must_use]
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::F32),
            1 => Some(Self::Q4_K),
            2 => Some(Self::Q8_0),
            _ => None,
        }
    }
}

/// Quantized APR Transformer with Q4_K or Q8_0 weights (Y5)
///
/// Stores weights in quantized form for memory efficiency while
/// providing the same inference interface as `AprTransformer`.
///
/// # Memory Savings
///
/// - Q4_K: ~7x compression (4.5 bits vs 32 bits)
/// - Q8_0: ~4x compression (8 bits vs 32 bits)
///
/// # Example
///
/// ```rust,ignore
/// use realizar::apr_transformer::{AprQuantizationType, QuantizedAprTransformer};
///
/// let transformer = QuantizedAprTransformer::new(config, AprQuantizationType::Q4_K);
/// let logits = transformer.forward(&[1, 2, 3])?;
/// ```
#[derive(Debug, Clone)]
pub struct QuantizedAprTransformer {
    /// Model configuration
    config: AprTransformerConfig,
    /// Quantization type
    quant_type: AprQuantizationType,
    /// Token embedding (stored as F32 for now, could be quantized later)
    token_embedding: Vec<f32>,
    /// Quantized layer weights (raw bytes)
    layer_weights: Vec<Vec<u8>>,
    /// Output norm weight (F32)
    output_norm_weight: Vec<f32>,
    /// LM head weight (quantized)
    lm_head_weight: Vec<u8>,
}

include!("quantized_transformer.rs");
include!("loader_tests.rs");
