//! Golden parity, scalar vs optimized, NaN/Inf, and determinism tests.

use super::super::*;

/// Golden Test: Fused kernel must match dequant+matmul within e = 1e-3
/// This is the core falsification test from Section 12.4 of the spec.
#[test]
fn test_fused_q4k_golden_parity() {
    // Create synthetic Q4K data (one super-block = 256 elements)
    let in_dim = 256;
    let out_dim = 4;
    let num_blocks = 1;

    // Build Q4K test data
    let mut q4k_data = Vec::with_capacity(out_dim * num_blocks * SUPER_BLOCK_BYTES);

    for row in 0..out_dim {
        // d = 0.1, dmin = 0.05 (as f16)
        let d: u16 = 0x2E66; // ~0.1 in f16
        let dmin: u16 = 0x2A66; // ~0.05 in f16
        q4k_data.extend_from_slice(&d.to_le_bytes());
        q4k_data.extend_from_slice(&dmin.to_le_bytes());

        // Scales and mins (all set to 1 for simplicity)
        let scales_packed = [0x01u8; 12];
        q4k_data.extend_from_slice(&scales_packed);

        // Quantized values: pattern based on row
        let mut qs = [0u8; 128];
        for (i, q) in qs.iter_mut().enumerate() {
            // Low nibble: (row + i) % 16, High nibble: (row + i + 1) % 16
            let low = ((row + i) % 16) as u8;
            let high = ((row + i + 1) % 16) as u8;
            *q = low | (high << 4);
        }
        q4k_data.extend_from_slice(&qs);
    }

    // Create input vector
    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01).collect();

    // Compute using fused kernel
    let fused_output = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);

    // Compute reference: dequant then matmul
    let mut reference_output = vec![0.0f32; out_dim];
    for row in 0..out_dim {
        let row_start = row * SUPER_BLOCK_BYTES;
        let row_q4k = &q4k_data[row_start..row_start + SUPER_BLOCK_BYTES];
        let f32_weights = dequantize_q4k_to_f32(row_q4k, in_dim);

        let mut sum = 0.0f32;
        for (w, x) in f32_weights.iter().zip(input.iter()) {
            sum += w * x;
        }
        reference_output[row] = sum;
    }

    // Golden parity check: |fused - reference| < 1e-3
    for (i, (fused, reference)) in fused_output.iter().zip(reference_output.iter()).enumerate() {
        let diff = (fused - reference).abs();
        assert!(
            diff < 1e-3,
            "Row {}: Fused kernel divergence: {} vs {} (d={})",
            i,
            fused,
            reference,
            diff
        );
    }
}

/// Test scalar implementation matches optimized version
#[test]
fn test_scalar_vs_optimized_parity() {
    let in_dim = 256;
    let out_dim = 2;

    // Build simple Q4K test data
    let mut q4k_data = Vec::new();
    for _ in 0..out_dim {
        q4k_data.extend_from_slice(&[0x00, 0x3C]); // d ~ 1.0
        q4k_data.extend_from_slice(&[0x00, 0x00]); // dmin = 0
        q4k_data.extend_from_slice(&[0x01u8; 12]); // scales
        q4k_data.extend_from_slice(&[0x55u8; 128]); // qs = 5 | (5 << 4)
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.001).collect();

    let scalar_output = matmul_q4k_f32_scalar(&q4k_data, &input, out_dim, in_dim);
    let optimized_output = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);

    for (i, (s, o)) in scalar_output.iter().zip(optimized_output.iter()).enumerate() {
        let diff = (s - o).abs();
        // Allow small FP differences from mul_add vs separate multiply-add
        assert!(
            diff < 1e-4,
            "Row {}: Scalar vs optimized divergence: {} vs {} (d={})",
            i,
            s,
            o,
            diff
        );
    }
}

/// Test that output contains no NaN or Inf
#[test]
fn test_no_nan_inf() {
    let in_dim = 256;
    let out_dim = 4;

    let mut q4k_data = Vec::new();
    for _ in 0..out_dim {
        q4k_data.extend_from_slice(&[0x00, 0x3C]); // d ~ 1.0
        q4k_data.extend_from_slice(&[0x00, 0x38]); // dmin ~ 0.5
        q4k_data.extend_from_slice(&[0x3Fu8; 12]); // max scales
        q4k_data.extend_from_slice(&[0xFFu8; 128]); // max qs
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01).collect();
    let output = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);

    for (i, &val) in output.iter().enumerate() {
        assert!(val.is_finite(), "Row {}: Output is not finite: {}", i, val);
    }
}

/// Test determinism: same input produces same output
#[test]
fn test_determinism() {
    let in_dim = 256;
    let out_dim = 2;

    let mut q4k_data = Vec::new();
    for _ in 0..out_dim {
        q4k_data.extend_from_slice(&[0x66, 0x2E]); // d
        q4k_data.extend_from_slice(&[0x66, 0x2A]); // dmin
        q4k_data.extend_from_slice(&[0x15u8; 12]);
        q4k_data.extend_from_slice(&[0xABu8; 128]);
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.005).collect();

    let output1 = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);
    let output2 = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);

    for (i, (a, b)) in output1.iter().zip(output2.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "Row {}: Non-deterministic output: {} vs {}", i, a, b);
    }
}

/// FALSIFIER C-AVX512-Q4K-003: the AVX-512 Q4_K GEMV must not read the
/// activation buffer out of bounds for a non-256-aligned `in_dim`, and must
/// match the scalar reference exactly.
///
/// The pre-fix kernel wrapped the high-nibble loads (activation positions
/// `[base+32, base+64)`) in the low-nibble guard `base + 32 <= in_dim`, so for
/// any `in_dim` that is `32/96/160/224 (mod 256)` the guard passed but the
/// `_mm512_loadu_ps` over-read up to 128 bytes past the buffer (and it also
/// silently dropped the valid high-nibble tail).
///
/// RED-on-bug: the activation buffer has a non-zero SENTINEL tail in the
/// allocation past `in_dim` (len() stays == in_dim for the dispatch assert), so
/// the pre-fix over-read injects the sentinel → divergence from scalar. The fix
/// uses masked loads that read only `[0, in_dim)` and zero the rest → exact
/// parity. On non-AVX-512 hosts the dispatch routes to AVX2/scalar (already
/// correct), so this self-gates to a no-op parity check there.
#[test]
#[cfg(target_arch = "x86_64")]
fn falsify_avx512_q4k_003_non_aligned_in_dim_no_oob() {
    // All < 256 (single padded super-block, supported by the scalar reference)
    // and each ≡ 32/96/160/224 (mod 256) or a mid-chunk tail — the cases where
    // the pre-fix high-nibble load over-reads. 256 is the aligned control.
    for &in_dim in &[32usize, 48, 96, 160, 224, 256] {
        let out_dim = 3;
        let num_blocks = in_dim.div_ceil(SUPER_BLOCK_SIZE);
        let mut q4k_data = Vec::with_capacity(out_dim * num_blocks * SUPER_BLOCK_BYTES);
        for row in 0..out_dim {
            for _ in 0..num_blocks {
                q4k_data.extend_from_slice(&[0x00, 0x3C]); // d ~ 1.0 (f16)
                q4k_data.extend_from_slice(&[0x00, 0x38]); // dmin ~ 0.5 (f16)
                q4k_data.extend_from_slice(&[0x21u8; 12]); // packed scales/mins
                let mut qs = [0u8; 128];
                for (i, q) in qs.iter_mut().enumerate() {
                    let low = ((row + i + 3) % 16) as u8;
                    let high = ((row * 2 + i + 7) % 16) as u8;
                    *q = low | (high << 4);
                }
                q4k_data.extend_from_slice(&qs);
            }
        }

        // Activation buffer: capacity is a full multiple of 256 (>= in_dim) with
        // a NON-ZERO sentinel tail past in_dim still inside the allocation; then
        // len() is truncated to in_dim so the dispatch assert holds. A pre-fix
        // high-nibble over-read past in_dim reads the sentinel and diverges.
        let cap = num_blocks * SUPER_BLOCK_SIZE;
        let mut input = vec![1.0e6_f32; cap];
        for (i, v) in input.iter_mut().enumerate().take(in_dim) {
            *v = ((i % 17) as f32) * 0.05 - 0.3;
        }
        input.truncate(in_dim);

        let simd = matmul_q4k_f32_dispatch(&q4k_data, &input, out_dim, in_dim);
        let scalar = matmul_q4k_f32_scalar(&q4k_data, &input, out_dim, in_dim);

        for (i, (a, s)) in simd.iter().zip(scalar.iter()).enumerate() {
            assert!(
                (a - s).abs() < 1e-2,
                "in_dim={in_dim} row={i}: SIMD {a} vs scalar {s} (diff {}) — \
                 non-256-aligned in_dim must not over-read or drop the high-nibble tail",
                (a - s).abs()
            );
        }
    }
}
