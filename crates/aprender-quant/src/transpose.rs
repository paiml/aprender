//! Transpose functions (LAYOUT-002: GGUF column-major -> APR row-major)

use crate::dequantize::{dequantize_q4_k_to_f32, dequantize_q5_k_to_f32, dequantize_q6_k_to_f32};
use crate::quantize::{quantize_q4_k_matrix, quantize_q6_k_matrix};

/// Row-major transpose reindex (LAYOUT-002): `src` is `[cols, rows]` column-major
/// (GGUF), the result is `[rows, cols]` row-major (APR), `out[r * cols + c] = src[c * rows + r]`.
///
/// Written output-indexed so it is one expression per slot `k = r * cols + c`
/// (`r = k / cols`, `c = k % cols`). It is modelled by `transposeRowMajor` in
/// `ProvableContracts/Theorems/TensorLayout/IndexAlgebra.lean`, which transcribes this
/// body; `transpose_row_major_witness_tests.rs` checks it against that model's golden table.
///
/// # Panics
/// If `src.len() < rows * cols`.
#[must_use]
pub fn transpose_row_major<T: Copy>(src: &[T], rows: usize, cols: usize) -> Vec<T> {
    (0..rows * cols)
        .map(|k| src[(k % cols) * rows + k / cols])
        .collect()
}

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

    let transposed = transpose_row_major(&f32_data, rows, cols);

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

    let transposed = transpose_row_major(&f32_data, rows, cols);

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

    let transposed = transpose_row_major(&f32_data, rows, cols);

    let new_shape = vec![rows, cols];
    let quantized = quantize_q6_k_matrix(&transposed, &new_shape);

    (quantized, new_shape)
}

#[cfg(test)]
#[path = "transpose_row_major_witness_tests.rs"]
mod transpose_row_major_witness_tests;
