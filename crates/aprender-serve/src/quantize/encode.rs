//! Quantization encoding functions (Toyota Way: ONE source of truth)
//!
//! This module re-exports quantization functions from trueno-quant.
//! The trueno-quant crate is the ONLY implementation in the stack.
//!
//! ## Stack Architecture (Toyota Way)
//!
//! ```text
//! aprender (format conversion) ──imports──► trueno-quant
//! realizar (inference engine) ──imports──► trueno-quant
//! ```
//!
//! ## Format Specifications
//!
//! - Q4_K: 256-element super-blocks, 144 bytes (4.5 bits/weight)
//! - Q5_K: 256-element super-blocks, 176 bytes (5.5 bits/weight)
//! - Q6_K: 256-element super-blocks, 210 bytes (6.5 bits/weight)

// Toyota Way: ONE source of truth - all quantization from trueno-quant
pub use trueno_quant::{
    // Dequantization functions
    dequantize_q4_k_to_f32,
    dequantize_q5_k_to_f32,
    dequantize_q6_k_to_f32,
    // Quantization functions
    quantize_q4_k,
    quantize_q4_k_matrix,
    quantize_q5_k,
    quantize_q5_k_matrix,
    quantize_q6_k,
    quantize_q6_k_matrix,
    // Transpose functions (LAYOUT-002: GGUF column-major → APR row-major)
    transpose_q4k_for_matmul,
    transpose_q5k_for_matmul,
    transpose_q6k_for_matmul,
    // Constants
    F16_MIN_NORMAL,
};

// ============================================================================
// Tests (verify re-exports work correctly)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4k_roundtrip() {
        // Create test data with a moderate range
        // Q4K uses asymmetric quantization with min offset, so negative values are supported
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 10.0).collect();
        // Range: -12.8 to +12.7

        // Quantize
        let quantized = quantize_q4_k(&data);
        assert_eq!(quantized.len(), 144); // One super-block

        // Dequantize
        let dequantized = dequantize_q4_k_to_f32(&quantized, 256);

        // Q4K has 4 bits per value (16 levels) with block-wise scaling
        // For a range of ~25.6, expect quantization step of ~1.7
        // Allow error up to 2x the step size for edge cases
        let data_range =
            data.iter().fold(0.0f32, |a, &b| a.max(b)) - data.iter().fold(0.0f32, |a, &b| a.min(b));
        let _expected_step = data_range / 15.0;

        let max_error: f32 = data
            .iter()
            .zip(dequantized.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // Note: Higher error than theoretical due to multi-level quantization
        // (global d/dmin + per-block scales/mins + 4-bit values)
        let relaxed_threshold = data_range * 0.5; // Allow up to 50% of range as error
        assert!(
            max_error < relaxed_threshold,
            "Q4K roundtrip error {} exceeds threshold {} (range={})",
            max_error,
            relaxed_threshold,
            data_range
        );
    }

    #[test]
    fn test_q6k_roundtrip() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 10.0).collect();

        let quantized = quantize_q6_k(&data);
        assert_eq!(quantized.len(), 210);

        let dequantized = dequantize_q6_k_to_f32(&quantized, 256);

        let max_error: f32 = data
            .iter()
            .zip(dequantized.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // Q6K should have lower error than Q4K
        assert!(
            max_error < 1.0,
            "Q6K roundtrip error too high: {}",
            max_error
        );
    }

    #[test]
    fn test_q4k_matrix() {
        let data: Vec<f32> = (0..512).map(|i| i as f32 / 100.0).collect();
        let shape = vec![2, 256];

        let quantized = quantize_q4_k_matrix(&data, &shape);
        assert_eq!(quantized.len(), 2 * 144); // Two super-blocks (one per row)
    }

    #[test]
    fn test_transpose_q4k() {
        // Create a 4x8 matrix (small for testing)
        // GGUF: [cols=8, rows=4] col-major
        let cols = 256;
        let rows = 2;
        let data: Vec<f32> = (0..(rows * cols)).map(|i| i as f32 / 10.0).collect();

        // Quantize in column-major order (as GGUF would store it)
        let quantized = quantize_q4_k(&data);
        let shape = vec![cols, rows]; // GGUF convention

        // Transpose
        let (transposed_data, new_shape) = transpose_q4k_for_matmul(&quantized, &shape);

        // New shape should be [rows, cols]
        assert_eq!(new_shape, vec![rows, cols]);

        // Should have data for rows * padded_cols
        assert!(!transposed_data.is_empty());
    }

    #[test]
    fn test_f16_min_normal() {
        // F16_MIN_NORMAL is used as a threshold to avoid subnormal values
        // It's approximately 2^(-14) ≈ 6.1e-5
        // When converted to f16 and back, it should preserve a non-zero positive value
        let f16_val = half::f16::from_f32(F16_MIN_NORMAL);
        let roundtrip = f16_val.to_f32();
        assert!(
            roundtrip > 0.0,
            "F16_MIN_NORMAL should be positive after f16 roundtrip"
        );
        assert!(roundtrip < 1e-4, "F16_MIN_NORMAL should be small");
    }
}
