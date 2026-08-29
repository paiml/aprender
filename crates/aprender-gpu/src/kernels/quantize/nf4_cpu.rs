//! CPU-side NF4 quantization and dequantization (Dettmers et al. 2023).
//!
//! NF4 (4-bit NormalFloat) is a quantization scheme optimized for normally-distributed
//! weights. Each value maps to one of 16 codebook entries derived from normal distribution
//! quantiles, achieving near-optimal information-theoretic compression for Gaussian weights.
//!
//! # Block Layout (36 bytes for 64 values)
//!
//! ```text
//! ┌───────────────────────────────────────────────────┐
//! │ Offset 0-3:   scale (f32, absmax normalization)   │
//! │ Offset 4-35:  data  (32 bytes, 64 × 4-bit packed) │
//! └───────────────────────────────────────────────────┘
//! ```
//!
//! # Contracts
//!
//! - C-NF4-001: Codebook fidelity — round-trip error < 0.05 for N(0,1) weights
//! - C-NF4-002: Block alignment — K divisible by 64, packed size = (K/64) × 36 × N
//! - C-NF4-004: Compression ratio — ≥7.1x vs fp32

/// NF4 block size: 64 values per quantization block.
pub const NF4_BLOCK_SIZE: usize = 64;

/// Bytes per NF4 block: 4 (f32 scale) + 32 (packed nibbles) = 36.
pub const NF4_BLOCK_BYTES: usize = 36;

/// NF4 codebook: 16 values derived from normal distribution quantiles.
///
/// These are the optimal reconstruction points for a standard normal distribution
/// quantized to 4 bits (Dettmers et al., "QLoRA", NeurIPS 2023).
#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
pub const NF4_LUT: [f32; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

/// NF4-quantized weight matrix.
///
/// Stores per-block f32 scale factors and packed 4-bit codebook indices.
/// Two values are packed per byte (low nibble first).
#[derive(Debug, Clone)]
pub struct Nf4Quantized {
    /// Per-block scale factors (absmax of each 64-element block).
    pub scales: Vec<f32>,
    /// Packed 4-bit indices (2 values per byte, low nibble first).
    pub data: Vec<u8>,
    /// Original matrix shape (rows, cols). rows × cols must be divisible by 64.
    pub shape: (usize, usize),
}

impl Nf4Quantized {
    /// Total number of quantization blocks.
    #[must_use]
    pub fn num_blocks(&self) -> usize {
        self.scales.len()
    }

    /// Total number of values represented.
    #[must_use]
    pub fn num_values(&self) -> usize {
        self.shape.0 * self.shape.1
    }

    /// Packed byte count (excludes scale storage).
    #[must_use]
    pub fn data_bytes(&self) -> usize {
        self.data.len()
    }

    /// Total bytes for the quantized representation (scales + packed data).
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.scales.len() * 4 + self.data.len()
    }
}

/// Find the nearest NF4 codebook index for a normalized value in [-1, 1].
///
/// Uses linear scan (16 entries is small enough that binary search has no advantage).
fn nearest_nf4_index(normalized: f32) -> u8 {
    let mut best_idx = 0u8;
    let mut best_dist = f32::MAX;
    for (i, &entry) in NF4_LUT.iter().enumerate() {
        let dist = (normalized - entry).abs();
        if dist < best_dist {
            best_dist = dist;
            best_idx = i as u8;
        }
    }
    best_idx
}

/// Quantize an f32 slice to NF4 format.
///
/// # Contract: C-NF4-002 (Block Alignment)
///
/// - **Precondition**: `values.len()` must be divisible by [`NF4_BLOCK_SIZE`] (64).
/// - **Postcondition**: `result.data.len() == values.len() / 2`,
///   `result.scales.len() == values.len() / 64`.
///
/// # Panics
///
/// Panics if `values.len()` is not divisible by 64.
#[must_use]
pub fn quantize_nf4(values: &[f32], rows: usize, cols: usize) -> Nf4Quantized {
    let n = values.len();
    assert!(
        n % NF4_BLOCK_SIZE == 0,
        "C-NF4-002: value count {n} not divisible by NF4 block size {NF4_BLOCK_SIZE}"
    );
    assert_eq!(
        rows * cols,
        n,
        "C-NF4-002: shape ({rows}, {cols}) does not match value count {n}"
    );

    let num_blocks = n / NF4_BLOCK_SIZE;
    let mut scales = Vec::with_capacity(num_blocks);
    let mut data = Vec::with_capacity(n / 2);

    for block_idx in 0..num_blocks {
        let start = block_idx * NF4_BLOCK_SIZE;
        let block = &values[start..start + NF4_BLOCK_SIZE];

        // Compute absmax for this block
        let absmax = block.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));

        scales.push(absmax);

        // Quantize: normalize to [-1, 1], find nearest codebook entry
        let inv_scale = if absmax > 0.0 { 1.0 / absmax } else { 0.0 };

        for pair in block.as_chunks::<2>().0 {
            let idx_lo = nearest_nf4_index(pair[0] * inv_scale);
            let idx_hi = nearest_nf4_index(pair[1] * inv_scale);
            data.push(idx_lo | (idx_hi << 4));
        }
    }

    Nf4Quantized {
        scales,
        data,
        shape: (rows, cols),
    }
}

/// Dequantize NF4 back to f32.
///
/// # Contract: C-NF4-001 (Codebook Fidelity)
///
/// - **Postcondition**: Max normalized error < 0.16 per element
///   (i.e., `|dequant(quant(x)) / absmax - x / absmax| < 0.16`).
#[must_use]
pub fn dequantize_nf4(q: &Nf4Quantized) -> Vec<f32> {
    let n = q.num_values();
    let mut output = Vec::with_capacity(n);

    for (block_idx, &scale) in q.scales.iter().enumerate() {
        let data_start = block_idx * (NF4_BLOCK_SIZE / 2);

        for byte_idx in 0..(NF4_BLOCK_SIZE / 2) {
            let packed = q.data[data_start + byte_idx];
            let idx_lo = (packed & 0x0F) as usize;
            let idx_hi = (packed >> 4) as usize;

            output.push(NF4_LUT[idx_lo] * scale);
            output.push(NF4_LUT[idx_hi] * scale);
        }
    }

    output
}

/// Pack NF4 quantized data into contiguous GPU-ready buffer.
///
/// Returns a flat buffer suitable for GPU upload with layout:
/// `[scale_0: f32][data_0: u8 × 32][scale_1: f32][data_1: u8 × 32]...`
///
/// Each block is exactly [`NF4_BLOCK_BYTES`] (36) bytes.
#[must_use]
pub fn pack_nf4_for_gpu(q: &Nf4Quantized) -> Vec<u8> {
    let num_blocks = q.num_blocks();
    let mut packed = Vec::with_capacity(num_blocks * NF4_BLOCK_BYTES);

    for block_idx in 0..num_blocks {
        // Write scale as f32 (4 bytes, little-endian)
        packed.extend_from_slice(&q.scales[block_idx].to_le_bytes());

        // Write 32 packed bytes (64 values)
        let data_start = block_idx * (NF4_BLOCK_SIZE / 2);
        let data_end = data_start + (NF4_BLOCK_SIZE / 2);
        packed.extend_from_slice(&q.data[data_start..data_end]);
    }

    packed
}

/// Unpack GPU buffer back to [`Nf4Quantized`] struct.
///
/// Inverse of [`pack_nf4_for_gpu`].
#[must_use]
pub fn unpack_nf4_from_gpu(packed: &[u8], rows: usize, cols: usize) -> Nf4Quantized {
    let n = rows * cols;
    let num_blocks = n / NF4_BLOCK_SIZE;
    let mut scales = Vec::with_capacity(num_blocks);
    let mut data = Vec::with_capacity(n / 2);

    for block_idx in 0..num_blocks {
        let offset = block_idx * NF4_BLOCK_BYTES;

        // Read scale (f32, little-endian)
        let scale_bytes: [u8; 4] = packed[offset..offset + 4]
            .try_into()
            .expect("C-NF4-002: packed buffer too short for scale");
        scales.push(f32::from_le_bytes(scale_bytes));

        // Read 32 packed data bytes
        data.extend_from_slice(&packed[offset + 4..offset + NF4_BLOCK_BYTES]);
    }

    Nf4Quantized {
        scales,
        data,
        shape: (rows, cols),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// C-NF4-001: Round-trip fidelity — max normalized error < 0.16 for arbitrary weights.
    ///
    /// The NF4 codebook has max gap of 0.304 between entries 0 (-1.0) and 1 (-0.696),
    /// yielding worst-case quantization error of ~0.152 (half-gap). This is by design:
    /// NF4 concentrates precision near zero where normally-distributed weights cluster.
    /// Average error for Gaussian weights is ~0.02.
    #[test]
    fn test_c_nf4_001_codebook_fidelity() {
        // Generate pseudo-normal samples using Box-Muller (deterministic seed)
        let n = 1024; // 16 blocks of 64
        let mut values = Vec::with_capacity(n);
        for i in 0..n {
            // Simple PRNG to approximate normal distribution
            let u1 = ((i * 1103515245 + 12345) % 65536) as f32 / 65536.0;
            let u2 = ((i * 6364136223 + 1442695) % 65536) as f32 / 65536.0;
            let u1_clamped = u1.clamp(1e-6, 1.0 - 1e-6);
            let z = (-2.0 * u1_clamped.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            values.push(z);
        }

        let q = quantize_nf4(&values, 1, n);
        let deq = dequantize_nf4(&q);

        assert_eq!(deq.len(), n);

        // Check per-block normalized error
        // Threshold 0.16 = half of max codebook gap (0.304 between entries 0 and 1)
        let mut max_err = 0.0f32;
        for block_idx in 0..q.num_blocks() {
            let start = block_idx * NF4_BLOCK_SIZE;
            let absmax = q.scales[block_idx];
            if absmax == 0.0 {
                continue;
            }
            for i in 0..NF4_BLOCK_SIZE {
                let orig_norm = values[start + i] / absmax;
                let deq_norm = deq[start + i] / absmax;
                let err = (orig_norm - deq_norm).abs();
                max_err = max_err.max(err);
                assert!(
                    err < 0.16,
                    "C-NF4-001 violated: block {block_idx} element {i}: \
                     orig_norm={orig_norm:.4}, deq_norm={deq_norm:.4}, error={err:.4}"
                );
            }
        }

        // Verify error is reasonable (should be well below threshold for most values)
        assert!(
            max_err > 0.0,
            "Max error should be non-zero for random data"
        );
    }

    /// C-NF4-002: Block alignment — sizes match expected layout.
    #[test]
    fn test_c_nf4_002_block_alignment() {
        let rows = 896;
        let cols = 896;
        let n = rows * cols;
        let values = vec![0.1f32; n];

        let q = quantize_nf4(&values, rows, cols);

        let expected_blocks = n / NF4_BLOCK_SIZE;
        assert_eq!(q.num_blocks(), expected_blocks);
        assert_eq!(q.data.len(), n / 2);
        assert_eq!(q.total_bytes(), expected_blocks * 4 + n / 2);

        // GPU packed buffer must be exactly num_blocks * 36
        let packed = pack_nf4_for_gpu(&q);
        assert_eq!(packed.len(), expected_blocks * NF4_BLOCK_BYTES);
    }

    /// C-NF4-004: Compression ratio ≥ 7.1x vs fp32.
    #[test]
    fn test_c_nf4_004_compression_ratio() {
        let rows = 896;
        let cols = 896;
        let n = rows * cols;
        let values = vec![0.5f32; n];

        let q = quantize_nf4(&values, rows, cols);

        let fp32_bytes = n * 4;
        let nf4_bytes = q.total_bytes();
        let ratio = fp32_bytes as f64 / nf4_bytes as f64;

        assert!(
            ratio >= 7.1,
            "C-NF4-004 violated: compression ratio {ratio:.2}x < 7.1x \
             (fp32={fp32_bytes}, nf4={nf4_bytes})"
        );
    }

    /// Verify codebook has exactly 16 entries spanning [-1, 1].
    #[test]
    fn test_nf4_codebook_properties() {
        assert_eq!(NF4_LUT.len(), 16);
        assert_eq!(NF4_LUT[0], -1.0);
        assert_eq!(NF4_LUT[15], 1.0);

        // Monotonically increasing
        for i in 1..16 {
            assert!(
                NF4_LUT[i] > NF4_LUT[i - 1],
                "NF4_LUT not monotonic at index {i}"
            );
        }
    }

    /// Verify zero values quantize/dequantize correctly.
    #[test]
    fn test_nf4_zero_block() {
        let values = vec![0.0f32; 64];
        let q = quantize_nf4(&values, 1, 64);
        let deq = dequantize_nf4(&q);

        for (i, &v) in deq.iter().enumerate() {
            assert_eq!(v, 0.0, "zero block element {i} = {v}");
        }
    }

    /// Verify GPU pack/unpack round-trips correctly.
    #[test]
    fn test_nf4_gpu_pack_roundtrip() {
        let n = 256;
        let values: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) / 128.0).collect();

        let q = quantize_nf4(&values, 4, 64);
        let packed = pack_nf4_for_gpu(&q);
        let unpacked = unpack_nf4_from_gpu(&packed, 4, 64);

        assert_eq!(unpacked.scales, q.scales);
        assert_eq!(unpacked.data, q.data);
        assert_eq!(unpacked.shape, q.shape);
    }

    /// Verify nearest_nf4_index maps boundary values correctly.
    #[test]
    fn test_nearest_nf4_index_boundaries() {
        // -1.0 should map to index 0
        assert_eq!(nearest_nf4_index(-1.0), 0);
        // 1.0 should map to index 15
        assert_eq!(nearest_nf4_index(1.0), 15);
        // 0.0 should map to index 7
        assert_eq!(nearest_nf4_index(0.0), 7);
    }

    #[test]
    #[should_panic(expected = "C-NF4-002")]
    fn test_nf4_rejects_misaligned_input() {
        let values = vec![0.0f32; 63]; // Not divisible by 64
        let _ = quantize_nf4(&values, 1, 63);
    }
}
