//! Safetensors parser
//!
//! Pure Rust implementation of Safetensors format reader.
//! Used by `HuggingFace` for safe, zero-copy tensor storage.
//!
//! Format specification: <https://github.com/huggingface/safetensors>
//!
//! ## Format Overview
//!
//! ```text
//! Safetensors := HEADER METADATA TENSOR_DATA
//!
//! HEADER := {
//!   metadata_len: u64 (little-endian)
//! }
//!
//! METADATA := JSON {
//!   "tensor_name": {
//!     "dtype": "F32" | "F16" | "I32" | ...,
//!     "shape": [dim1, dim2, ...],
//!     "data_offsets": [start, end]
//!   },
//!   ...
//! }
//! ```

use std::{
    collections::HashMap,
    io::{Cursor, Read},
};

use serde::{Deserialize, Serialize};

use crate::error::{RealizarError, Result};
use crate::inference::simd_bf16_to_f32;

/// GAP-UX-002: Find sibling companion file with hash-prefix fallback.
///
/// Companion files (config.json, tokenizer.json) may be stored with:
/// 1. Hash prefix: `{stem}.{filename}` (e.g., `d71534cb.config.json`) - PREFERRED
/// 2. Plain name: `{filename}` (e.g., `config.json`) - FALLBACK for backwards compatibility
///
/// # Arguments
///
/// * `model_path` - Path to the model file (e.g., `/cache/d71534cb.safetensors`)
/// * `companion_name` - Name of companion file (e.g., `config.json`)
///
/// # Returns
///
/// Path to the companion file if found, None otherwise
pub fn find_sibling_file(
    model_path: &std::path::Path,
    companion_name: &str,
) -> Option<std::path::PathBuf> {
    let parent = model_path.parent()?;
    let filename = model_path.file_name()?.to_str()?;

    // GH-213: For sharded index.json paths (e.g., "model.safetensors.index.json"),
    // skip the hash-prefix logic and go straight to plain name lookup.
    // For normal model files, use the hash-prefix strategy.
    if !filename.ends_with(".index.json") {
        let stem = model_path.file_stem()?.to_str()?;

        // GAP-UX-002: Try hash-prefixed first (e.g., "d71534cb.config.json")
        let prefixed = parent.join(format!("{stem}.{companion_name}"));
        if prefixed.exists() {
            return Some(prefixed);
        }
    }

    // Fallback: Try plain name for backwards compatibility
    // This is the primary path for sharded models where config.json/tokenizer.json
    // are siblings in the same directory from `apr pull`.
    let plain = parent.join(companion_name);
    if plain.exists() {
        return Some(plain);
    }

    None
}

/// Safetensors data type
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub enum SafetensorsDtype {
    /// 32-bit float
    F32,
    /// 16-bit float
    F16,
    /// Brain float 16
    BF16,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 8-bit unsigned integer
    U8,
    /// Boolean
    Bool,
}

impl SafetensorsDtype {
    /// Size in bytes of a single element of this dtype.
    ///
    /// Matches the official `safetensors` library element widths.
    #[must_use]
    pub fn size_in_bytes(&self) -> usize {
        match self {
            SafetensorsDtype::F32 | SafetensorsDtype::I32 => 4,
            SafetensorsDtype::F16 | SafetensorsDtype::BF16 => 2,
            SafetensorsDtype::I64 => 8,
            SafetensorsDtype::U8 | SafetensorsDtype::Bool => 1,
        }
    }
}

/// JSON tensor metadata (internal)
#[derive(Debug, Deserialize)]
struct TensorMetadata {
    dtype: SafetensorsDtype,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// Tensor metadata
#[derive(Debug, Clone, PartialEq)]
pub struct SafetensorsTensorInfo {
    /// Tensor name
    pub name: String,
    /// Data type
    pub dtype: SafetensorsDtype,
    /// Shape (dimensions)
    pub shape: Vec<usize>,
    /// Data offsets in file [start, end)
    pub data_offsets: [usize; 2],
}

impl SafetensorsTensorInfo {
    /// Number of bytes occupied by this tensor's data, per `data_offsets`.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.data_offsets[1].saturating_sub(self.data_offsets[0])
    }

    /// Validate that the declared `data_offsets` byte length agrees with the
    /// declared `shape` and `dtype` element size.
    ///
    /// The safetensors spec requires `byte_len == product(shape) * dtype_size`.
    /// The official `HuggingFace` `safetensors` library rejects any file that
    /// violates this; aprender's raw reader historically derived the element
    /// count from the byte length and silently ignored the declared shape, so a
    /// crafted or corrupt file whose byte length contradicts its shape would
    /// load with a wrong-sized tensor (garbage inference / integrity hole).
    /// This check makes the reader fail closed to reach parity with the
    /// reference implementation (defense-in-depth, per-tensor integrity).
    ///
    /// # Errors
    ///
    /// Returns an error if `shape` overflows `usize`, or if the byte length
    /// does not equal `product(shape) * dtype_size`.
    pub fn validate_shape_matches_bytes(&self) -> Result<()> {
        let dtype_size = self.dtype.size_in_bytes();

        let elem_count = self
            .shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| RealizarError::UnsupportedOperation {
                operation: "validate_shape_matches_bytes".to_string(),
                reason: format!(
                    "Tensor '{}' shape {:?} overflows usize",
                    self.name, self.shape
                ),
            })?;

        let expected = elem_count.checked_mul(dtype_size).ok_or_else(|| {
            RealizarError::UnsupportedOperation {
                operation: "validate_shape_matches_bytes".to_string(),
                reason: format!(
                    "Tensor '{}' byte size (shape {:?} * {dtype_size}) overflows usize",
                    self.name, self.shape
                ),
            }
        })?;

        let actual = self.byte_len();
        if actual != expected {
            return Err(RealizarError::UnsupportedOperation {
                operation: "validate_shape_matches_bytes".to_string(),
                reason: format!(
                    "Tensor '{}' byte length {actual} contradicts declared shape {:?} \
                     ({:?}, {dtype_size} bytes/elem => expected {expected} bytes)",
                    self.name, self.shape, self.dtype
                ),
            });
        }

        Ok(())
    }
}

/// Safetensors model container
#[derive(Debug, Clone)]
pub struct SafetensorsModel {
    /// Tensor metadata
    pub tensors: HashMap<String, SafetensorsTensorInfo>,
    /// Raw tensor data (not parsed yet)
    pub data: Vec<u8>,
}

include!("safetensors_parser.rs");
include!("mapped_model.rs");
include!("shard.rs");
