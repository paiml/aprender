
/// SIMD-accelerated BF16 matmul
///
/// Computes matrix-vector product with BF16 weights.
/// Uses batch conversion to minimize conversion overhead.
///
/// # Arguments
///
/// * `input` - F32 input vector
/// * `weight_bf16` - BF16 weight matrix (raw bytes, row-major)
/// * `in_dim` - Input dimension
/// * `out_dim` - Output dimension
///
/// # Returns
///
/// F32 output vector
///
/// # Performance
///
/// This function batch-converts BF16 rows to F32 in tiles to amortize
/// conversion overhead and improve cache utilization. The tile size is
/// chosen to fit in L2 cache (~256KB per tile).
#[must_use]
pub fn simd_bf16_matmul(
    input: &[f32],
    weight_bf16: &[u8],
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    // Batch conversion: convert all BF16 weights to F32 once
    // This is more efficient than row-by-row conversion because:
    // 1. Amortizes function call overhead
    // 2. Better SIMD utilization (longer vectors)
    // 3. Memory prefetching works better
    //
    // Trade-off: Uses 2x memory (BF16 + F32), but much faster
    let weight_f32 = simd_bf16_to_f32(weight_bf16);

    // Now use optimized F32 matmul
    simd_matmul(input, &weight_f32, in_dim, out_dim)
}

/// SIMD-accelerated BF16 matmul with streaming conversion
///
/// This variant uses row-by-row conversion for lower memory usage
/// at the cost of performance. Use for very large matrices that
/// don't fit in memory when fully converted.
///
/// # Arguments
///
/// * `input` - F32 input vector
/// * `weight_bf16` - BF16 weight matrix (raw bytes, row-major)
/// * `in_dim` - Input dimension
/// * `out_dim` - Output dimension
///
/// # Returns
///
/// F32 output vector
#[must_use]
pub(super) fn simd_bf16_matmul_streaming(
    input: &[f32],
    weight_bf16: &[u8],
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    const TILE_SIZE: usize = 64;

    let mut output = vec![0.0f32; out_dim];
    let input_vec = Vector::from_slice(input);

    for tile_start in (0..out_dim).step_by(TILE_SIZE) {
        let tile_end = (tile_start + TILE_SIZE).min(out_dim);

        for row in tile_start..tile_end {
            let row_byte_start = row * in_dim * 2;
            let row_byte_end = row_byte_start + in_dim * 2;
            let row_bf16 = &weight_bf16[row_byte_start..row_byte_end];

            // Convert row to F32
            let row_f32 = simd_bf16_to_f32(row_bf16);
            let row_vec = Vector::from_slice(&row_f32);

            output[row] = input_vec.dot(&row_vec).expect("dot product failed");
        }
    }

    output
}
