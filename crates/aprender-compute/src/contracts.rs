//! GH-279: Kernel-Level Contracts for the Sovereign AI Stack
//!
//! Defines invariants that MUST hold for ANY data entering trueno compute kernels.
//! Consumers (realizar, aprender) validate data against these contracts BEFORE
//! calling any kernel. Violating a contract is a hard error, not a silent default.
//!
//! # Contract Hierarchy
//!
//! ```text
//! aprender (import) ──► enforce_architecture_completeness()  [tensor names]
//!                        │
//! realizar (load)   ──► contract_gate::validate_model_load()  [architecture]
//!                        │
//! trueno (kernel)   ──► contracts::validate_weight_buffer()    [bytes & layout]
//! ```
//!
//! This module is the bottom layer — raw buffer and layout validation.
//! If these fail, the kernel WILL produce garbage or crash.

// Re-export trueno-quant constants as the canonical source of truth for block sizes.
// These constants are also used directly by trueno kernels (tiling, brick, etc.).
pub use trueno_quant::{
    Q4_K_BLOCK_BYTES, Q4_K_BLOCK_SIZE, Q5_K_BLOCK_BYTES, Q5_K_BLOCK_SIZE, Q6_K_BLOCK_BYTES,
    Q6_K_BLOCK_SIZE,
};

// ============================================================================
// Layout Contract (LAYOUT-001/002)
// ============================================================================

/// Tensor layout used by all trueno kernels.
///
/// The entire stack (APR format, realizar inference, trueno kernels) uses
/// row-major layout EXCLUSIVELY. Column-major data from GGUF is transposed
/// at the import boundary in aprender.
///
/// Kernel contract: `weight[row * cols + col]` is the access pattern.
/// Passing column-major data produces GARBAGE — there is no runtime flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorLayout {
    /// Row-major: shape [rows, cols], stride [cols, 1]
    /// This is the ONLY layout trueno kernels accept.
    RowMajor,
}

/// The stack-wide tensor layout. All kernels assume this.
pub const STACK_LAYOUT: TensorLayout = TensorLayout::RowMajor;

// ============================================================================
// Quantization Format Descriptors
// ============================================================================

/// Quantization format descriptor with block geometry.
///
/// Each format defines a fixed relationship between element count and byte size.
/// Kernels use these to compute buffer sizes and validate inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantFormat {
    /// Human-readable name (e.g., "Q4_K")
    pub name: &'static str,
    /// Elements per quantization block
    pub block_size: usize,
    /// Bytes per quantization block
    pub block_bytes: usize,
    /// GGML type ID (for GGUF interop)
    pub ggml_type_id: u32,
}

/// Q4_K super-block format: 256 elements, 144 bytes (4.5 bits/weight)
pub const Q4_K: QuantFormat = QuantFormat {
    name: "Q4_K",
    block_size: Q4_K_BLOCK_SIZE,
    block_bytes: Q4_K_BLOCK_BYTES,
    ggml_type_id: 12,
};

/// Q5_K super-block format: 256 elements, 176 bytes (5.5 bits/weight)
pub const Q5_K: QuantFormat = QuantFormat {
    name: "Q5_K",
    block_size: Q5_K_BLOCK_SIZE,
    block_bytes: Q5_K_BLOCK_BYTES,
    ggml_type_id: 13,
};

/// Q6_K super-block format: 256 elements, 210 bytes (6.5 bits/weight)
pub const Q6_K: QuantFormat = QuantFormat {
    name: "Q6_K",
    block_size: Q6_K_BLOCK_SIZE,
    block_bytes: Q6_K_BLOCK_BYTES,
    ggml_type_id: 14,
};

/// Q8_0 block format: 32 elements, 34 bytes
pub const Q8_0: QuantFormat =
    QuantFormat { name: "Q8_0", block_size: 32, block_bytes: 34, ggml_type_id: 8 };

/// Q5_0 block format: 32 elements, 22 bytes
pub const Q5_0: QuantFormat =
    QuantFormat { name: "Q5_0", block_size: 32, block_bytes: 22, ggml_type_id: 6 };

/// Q4_0 block format: 32 elements, 18 bytes
pub const Q4_0: QuantFormat =
    QuantFormat { name: "Q4_0", block_size: 32, block_bytes: 18, ggml_type_id: 2 };

/// Q4_1 block format: 32 elements, 20 bytes
pub const Q4_1: QuantFormat =
    QuantFormat { name: "Q4_1", block_size: 32, block_bytes: 20, ggml_type_id: 3 };

/// All supported quantization formats, ordered by GGML type ID.
pub const ALL_FORMATS: &[QuantFormat] = &[Q4_0, Q4_1, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K];

/// Lookup a quantization format by GGML type ID.
#[must_use]
pub fn format_by_ggml_type(type_id: u32) -> Option<&'static QuantFormat> {
    ALL_FORMATS.iter().find(|f| f.ggml_type_id == type_id)
}

// ============================================================================
// Weight Buffer Validation
// ============================================================================

/// Error returned when a weight buffer fails contract validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightBufferError {
    /// Which weight (e.g., "blk.0.attn_q.weight")
    pub weight_name: String,
    /// What went wrong
    pub reason: String,
}

impl std::fmt::Display for WeightBufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Kernel contract violation for '{}': {}", self.weight_name, self.reason)
    }
}

impl std::error::Error for WeightBufferError {}

impl QuantFormat {
    /// Compute the expected byte size for a weight matrix [rows, cols].
    ///
    /// The matrix is stored as `rows` independent row vectors, each quantized
    /// into ceil(cols / block_size) blocks of `block_bytes` bytes.
    ///
    /// # Row-Major Contract
    ///
    /// For GEMV `y = W·x` where W is [out_dim, in_dim]:
    /// - `rows` = out_dim (number of output features)
    /// - `cols` = in_dim (number of input features, quantized along this axis)
    #[must_use]
    pub const fn expected_bytes(&self, rows: usize, cols: usize) -> usize {
        let blocks_per_row = (cols + self.block_size - 1) / self.block_size;
        rows * blocks_per_row * self.block_bytes
    }

    /// Validate that a weight buffer has the correct size for [rows, cols].
    ///
    /// # Errors
    ///
    /// Returns `WeightBufferError` if `actual_bytes` does not match the expected
    /// size for the given dimensions and quantization format.
    pub fn validate_buffer(
        &self,
        weight_name: &str,
        actual_bytes: usize,
        rows: usize,
        cols: usize,
    ) -> Result<(), WeightBufferError> {
        let expected = self.expected_bytes(rows, cols);
        if actual_bytes != expected {
            return Err(WeightBufferError {
                weight_name: weight_name.to_string(),
                reason: format!(
                    "{} buffer size mismatch: got {} bytes, expected {} bytes \
                     for [{}, {}] ({} blocks/row * {} bytes/block * {} rows)",
                    self.name,
                    actual_bytes,
                    expected,
                    rows,
                    cols,
                    (cols + self.block_size - 1) / self.block_size,
                    self.block_bytes,
                    rows,
                ),
            });
        }
        Ok(())
    }
}

/// Validate a weight buffer against a known GGML type ID.
///
/// This is the primary entry point for realizar and aprender to validate
/// quantized weight buffers before passing them to trueno kernels.
///
/// # Arguments
///
/// * `weight_name` - Human-readable name (for error messages)
/// * `ggml_type` - GGML quantization type ID
/// * `actual_bytes` - Actual buffer size in bytes
/// * `rows` - Number of rows (out_dim for GEMV)
/// * `cols` - Number of columns (in_dim for GEMV)
///
/// # Errors
///
/// Returns `WeightBufferError` if:
/// - The GGML type ID is unknown
/// - The buffer size doesn't match expected dimensions
pub fn validate_weight_buffer(
    weight_name: &str,
    ggml_type: u32,
    actual_bytes: usize,
    rows: usize,
    cols: usize,
) -> Result<(), WeightBufferError> {
    let format = format_by_ggml_type(ggml_type).ok_or_else(|| WeightBufferError {
        weight_name: weight_name.to_string(),
        reason: format!("Unknown GGML quantization type ID: {ggml_type}"),
    })?;
    format.validate_buffer(weight_name, actual_bytes, rows, cols)
}

/// Validate that an F32 weight buffer has correct element count.
///
/// For unquantized (F32) weights, the buffer must have exactly `rows * cols * 4` bytes
/// (or `rows * cols` elements).
///
/// # Errors
///
/// Returns `WeightBufferError` if the element count doesn't match.
pub fn validate_f32_buffer(
    weight_name: &str,
    actual_elements: usize,
    rows: usize,
    cols: usize,
) -> Result<(), WeightBufferError> {
    let expected = rows * cols;
    if actual_elements != expected {
        return Err(WeightBufferError {
            weight_name: weight_name.to_string(),
            reason: format!(
                "F32 element count mismatch: got {actual_elements}, expected {expected} \
                 for [{rows}, {cols}]"
            ),
        });
    }
    Ok(())
}

// ============================================================================
// Matmul Shape Contract
// ============================================================================

/// Validate GEMV shape invariants for row-major layout.
///
/// For `y = W · x` with row-major W[out_dim, in_dim]:
/// - `weight_rows` MUST equal `out_dim`
/// - `weight_cols` MUST equal `in_dim`
/// - `input_len` MUST equal `in_dim`
/// - `output_len` MUST equal `out_dim`
///
/// # Errors
///
/// Returns `WeightBufferError` describing the shape mismatch.
pub fn validate_gemv_shapes(
    weight_name: &str,
    weight_rows: usize,
    weight_cols: usize,
    input_len: usize,
    output_len: usize,
) -> Result<(), WeightBufferError> {
    if weight_cols != input_len {
        return Err(WeightBufferError {
            weight_name: weight_name.to_string(),
            reason: format!(
                "GEMV input dimension mismatch: weight has {weight_cols} cols \
                 but input has {input_len} elements"
            ),
        });
    }
    if weight_rows != output_len {
        return Err(WeightBufferError {
            weight_name: weight_name.to_string(),
            reason: format!(
                "GEMV output dimension mismatch: weight has {weight_rows} rows \
                 but output has {output_len} elements"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4k_expected_bytes() {
        // 256 elements per block, 144 bytes per block
        // For a [4096, 4096] weight: 4096 * (4096/256) * 144 = 4096 * 16 * 144 = 9_437_184
        assert_eq!(Q4_K.expected_bytes(4096, 4096), 9_437_184);
    }

    #[test]
    fn test_q6k_expected_bytes() {
        // 256 elements per block, 210 bytes per block
        assert_eq!(Q6_K.expected_bytes(4096, 4096), 4096 * 16 * 210);
    }

    #[test]
    fn test_q8_0_expected_bytes() {
        // 32 elements per block, 34 bytes per block
        // For [4096, 4096]: 4096 * (4096/32) * 34 = 4096 * 128 * 34
        assert_eq!(Q8_0.expected_bytes(4096, 4096), 4096 * 128 * 34);
    }

    #[test]
    fn test_validate_buffer_ok() {
        let bytes = Q4_K.expected_bytes(4096, 4096);
        assert!(Q4_K.validate_buffer("test.weight", bytes, 4096, 4096).is_ok());
    }

    #[test]
    fn test_validate_buffer_wrong_size() {
        let err = Q4_K.validate_buffer("test.weight", 1000, 4096, 4096).unwrap_err();
        assert!(err.reason.contains("buffer size mismatch"));
    }

    #[test]
    fn test_validate_weight_buffer_unknown_type() {
        let err = validate_weight_buffer("test.weight", 99, 1000, 4096, 4096).unwrap_err();
        assert!(err.reason.contains("Unknown GGML"));
    }

    #[test]
    fn test_validate_f32_buffer_ok() {
        assert!(validate_f32_buffer("test.weight", 4096 * 4096, 4096, 4096).is_ok());
    }

    #[test]
    fn test_validate_f32_buffer_mismatch() {
        let err = validate_f32_buffer("test.weight", 100, 4096, 4096).unwrap_err();
        assert!(err.reason.contains("element count mismatch"));
    }

    #[test]
    fn test_validate_gemv_shapes_ok() {
        assert!(validate_gemv_shapes("test", 4096, 4096, 4096, 4096).is_ok());
    }

    #[test]
    fn test_validate_gemv_shapes_input_mismatch() {
        let err = validate_gemv_shapes("test", 4096, 4096, 2048, 4096).unwrap_err();
        assert!(err.reason.contains("input dimension mismatch"));
    }

    #[test]
    fn test_validate_gemv_shapes_output_mismatch() {
        let err = validate_gemv_shapes("test", 4096, 4096, 4096, 2048).unwrap_err();
        assert!(err.reason.contains("output dimension mismatch"));
    }

    #[test]
    fn test_format_lookup_all_types() {
        assert_eq!(format_by_ggml_type(2).unwrap().name, "Q4_0");
        assert_eq!(format_by_ggml_type(3).unwrap().name, "Q4_1");
        assert_eq!(format_by_ggml_type(6).unwrap().name, "Q5_0");
        assert_eq!(format_by_ggml_type(8).unwrap().name, "Q8_0");
        assert_eq!(format_by_ggml_type(12).unwrap().name, "Q4_K");
        assert_eq!(format_by_ggml_type(13).unwrap().name, "Q5_K");
        assert_eq!(format_by_ggml_type(14).unwrap().name, "Q6_K");
        assert!(format_by_ggml_type(99).is_none());
    }

    #[test]
    fn test_stack_layout_is_row_major() {
        assert_eq!(STACK_LAYOUT, TensorLayout::RowMajor);
    }

    #[test]
    fn test_non_aligned_cols() {
        // 100 cols doesn't divide evenly into 256-element blocks
        // ceil(100/256) = 1 block per row
        assert_eq!(Q4_K.expected_bytes(10, 100), 10 * 144);
        // 300 cols: ceil(300/256) = 2 blocks per row
        assert_eq!(Q4_K.expected_bytes(10, 300), 10 * 2 * 144);
    }
}
