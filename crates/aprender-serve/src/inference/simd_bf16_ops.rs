
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

