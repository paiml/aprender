//! Transpose functions (LAYOUT-002: GGUF column-major -> APR row-major)

use crate::dequantize::{dequantize_q4_k_to_f32, dequantize_q5_k_to_f32, dequantize_q6_k_to_f32};
use crate::quantize::{quantize_q4_k_matrix, quantize_q6_k_matrix};

/// Transpose Q4K tensor from GGUF column-major to APR row-major layout
///
/// GGUF stores weights as [cols, rows] in column-major order.
/// APR requires [rows, cols] in row-major order.
/// This function dequantizes, transposes, and re-quantizes.
#[must_use]
pub fn transpose_q4k_for_matmul(data: &[u8], shape: &[usize]) -> (Vec<u8>, Vec<usize>) {
    if shape.len() != 2 {
        return (data.to_vec(), shape.to_vec());
    }

    let cols = shape[0];
    let rows = shape[1];
    let num_elements = rows * cols;

    let f32_data = dequantize_q4_k_to_f32(data, num_elements);

    let mut transposed = vec![0.0f32; num_elements];
    for r in 0..rows {
        for c in 0..cols {
            transposed[r * cols + c] = f32_data[c * rows + r];
        }
    }

    let new_shape = vec![rows, cols];
    let quantized = quantize_q4_k_matrix(&transposed, &new_shape);

    (quantized, new_shape)
}

/// Transpose Q5K tensor from GGUF column-major to APR row-major layout
#[must_use]
pub fn transpose_q5k_for_matmul(data: &[u8], shape: &[usize]) -> (Vec<u8>, Vec<usize>) {
    if shape.len() != 2 {
        return (data.to_vec(), shape.to_vec());
    }

    let cols = shape[0];
    let rows = shape[1];
    let num_elements = rows * cols;

    let f32_data = dequantize_q5_k_to_f32(data, num_elements);

    let mut transposed = vec![0.0f32; num_elements];
    for r in 0..rows {
        for c in 0..cols {
            transposed[r * cols + c] = f32_data[c * rows + r];
        }
    }

    // Note: APR doesn't have native Q5K, convert to Q6K for better precision
    let new_shape = vec![rows, cols];
    let quantized = quantize_q6_k_matrix(&transposed, &new_shape);

    (quantized, new_shape)
}

/// Transpose Q6K tensor from GGUF column-major to APR row-major layout
#[must_use]
pub fn transpose_q6k_for_matmul(data: &[u8], shape: &[usize]) -> (Vec<u8>, Vec<usize>) {
    if shape.len() != 2 {
        return (data.to_vec(), shape.to_vec());
    }

    let cols = shape[0];
    let rows = shape[1];
    let num_elements = rows * cols;

    let f32_data = dequantize_q6_k_to_f32(data, num_elements);

    let mut transposed = vec![0.0f32; num_elements];
    for r in 0..rows {
        for c in 0..cols {
            transposed[r * cols + c] = f32_data[c * rows + r];
        }
    }

    let new_shape = vec![rows, cols];
    let quantized = quantize_q6_k_matrix(&transposed, &new_shape);

    (quantized, new_shape)
}
