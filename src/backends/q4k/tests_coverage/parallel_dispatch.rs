//! Coverage tests for `matmul_q4k_f32_parallel` — the threaded dispatch path
//! triggered when `out_dim * in_dim >= 8_000_000`.
//!
//! These tests exercise:
//! - The parallel path entry via `matmul_q4k_f32_dispatch`
//! - Multi-thread chunk splitting with even and uneven out_dim
//! - Parity between parallel and scalar results
//! - Edge cases: single output row, prime out_dim, just-at-threshold

use super::super::*;

/// Helper: build deterministic Q4K row-major test data.
///
/// Each super-block has:
///   d (2 bytes, f16) | dmin (2 bytes, f16) | scales (12 bytes) | qs (128 bytes)
/// = 144 bytes per super-block.
fn build_q4k_test_data(out_dim: usize, in_dim: usize) -> Vec<u8> {
    let num_blocks_per_row = (in_dim + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let row_bytes = num_blocks_per_row * SUPER_BLOCK_BYTES;
    let total_bytes = out_dim * row_bytes;
    let mut data = vec![0u8; total_bytes];

    for row in 0..out_dim {
        for sb in 0..num_blocks_per_row {
            let offset = row * row_bytes + sb * SUPER_BLOCK_BYTES;
            // d = 0.5 as f16 (0x3800)
            data[offset] = 0x00;
            data[offset + 1] = 0x38;
            // dmin = 0.25 as f16 (0x3400)
            data[offset + 2] = 0x00;
            data[offset + 3] = 0x34;
            // scales: vary by row and super-block for diversity
            for i in 0..12 {
                data[offset + 4 + i] = ((row + sb + i + 1) & 0x3F) as u8;
            }
            // qs: varying nibble patterns
            for i in 0..128 {
                let low = ((row + sb + i) % 16) as u8;
                let high = ((row + sb + i + 5) % 16) as u8;
                data[offset + 16 + i] = low | (high << 4);
            }
        }
    }
    data
}

/// Helper: build a deterministic input vector.
fn build_input(in_dim: usize) -> Vec<f32> {
    (0..in_dim).map(|i| (i as f32 * 0.00137).sin()).collect()
}

// ============================================================================
// Parallel dispatch path tests
// ============================================================================

/// Core test: parallel dispatch produces results matching scalar.
///
/// Uses out_dim=4096, in_dim=2048 => total_work = 8,388,608 (>= 8M threshold).
/// This directly exercises the `matmul_q4k_f32_parallel` function on x86_64.
#[test]
fn test_q4k_parallel_dispatch_matches_scalar() {
    let out_dim = 4096;
    let in_dim = 2048; // 8 super-blocks per row
    let total_work = out_dim * in_dim;
    assert!(total_work >= 8_000_000, "Must trigger parallel path");
    assert_eq!(in_dim % SUPER_BLOCK_SIZE, 0);

    let q4k_data = build_q4k_test_data(out_dim, in_dim);
    let input = build_input(in_dim);

    let scalar = matmul_q4k_f32_scalar(&q4k_data, &input, out_dim, in_dim);
    let dispatch = matmul_q4k_f32_dispatch(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(scalar.len(), out_dim);
    assert_eq!(dispatch.len(), out_dim);

    // Compare every 64th row for speed, plus first and last
    let check_indices: Vec<usize> =
        (0..out_dim).step_by(64).chain(std::iter::once(out_dim - 1)).collect();

    for &i in &check_indices {
        let diff = (scalar[i] - dispatch[i]).abs();
        let tol = scalar[i].abs() * 1e-4 + 1e-4;
        assert!(
            diff < tol,
            "Row {}: scalar={}, dispatch={}, diff={}",
            i,
            scalar[i],
            dispatch[i],
            diff
        );
    }
}

/// Uneven chunk splitting: out_dim not divisible by typical thread counts.
///
/// Uses a prime out_dim (4099) so no thread count evenly divides it.
/// The last thread's chunk will be smaller, exercising remainder handling.
#[test]
fn test_q4k_parallel_dispatch_prime_outdim() {
    let out_dim = 4099; // prime
    let in_dim = 2048;
    let total_work = out_dim * in_dim;
    assert!(total_work >= 8_000_000);
    assert_eq!(in_dim % SUPER_BLOCK_SIZE, 0);

    let q4k_data = build_q4k_test_data(out_dim, in_dim);
    let input = build_input(in_dim);

    let scalar = matmul_q4k_f32_scalar(&q4k_data, &input, out_dim, in_dim);
    let dispatch = matmul_q4k_f32_dispatch(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(dispatch.len(), out_dim);

    // Check first, middle, and last rows
    for &i in &[0, out_dim / 2, out_dim - 1] {
        let diff = (scalar[i] - dispatch[i]).abs();
        let tol = scalar[i].abs() * 1e-4 + 1e-4;
        assert!(
            diff < tol,
            "Row {}: scalar={}, dispatch={}, diff={}",
            i,
            scalar[i],
            dispatch[i],
            diff
        );
    }
}

/// Small out_dim but very large in_dim to trigger parallel path.
///
/// out_dim=2, in_dim=4194304 (16384 super-blocks) => total_work = 8,388,608.
/// With only 2 output rows, each thread gets at most 1 row, exercising
/// the chunk_size=1 path.
#[test]
fn test_q4k_parallel_dispatch_few_rows_large_indim() {
    let out_dim = 2;
    let in_dim = 4_194_304; // 16384 super-blocks, 2 * 4M = 8M
    let total_work = out_dim * in_dim;
    assert!(total_work >= 8_000_000);
    assert_eq!(in_dim % SUPER_BLOCK_SIZE, 0);

    let q4k_data = build_q4k_test_data(out_dim, in_dim);
    let input = build_input(in_dim);

    let scalar = matmul_q4k_f32_scalar(&q4k_data, &input, out_dim, in_dim);
    let dispatch = matmul_q4k_f32_dispatch(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(dispatch.len(), out_dim);
    for i in 0..out_dim {
        let diff = (scalar[i] - dispatch[i]).abs();
        let tol = scalar[i].abs() * 1e-4 + 1e-4;
        assert!(
            diff < tol,
            "Row {}: scalar={}, dispatch={}, diff={}",
            i,
            scalar[i],
            dispatch[i],
            diff
        );
    }
}

/// Just at threshold: total_work = 8_000_000 exactly.
///
/// out_dim=31250, in_dim=256 => 31250 * 256 = 8,000,000.
#[test]
fn test_q4k_parallel_dispatch_exact_threshold() {
    let out_dim = 31_250;
    let in_dim = 256; // 1 super-block per row
    let total_work = out_dim * in_dim;
    assert_eq!(total_work, 8_000_000);

    let q4k_data = build_q4k_test_data(out_dim, in_dim);
    let input = build_input(in_dim);

    let scalar = matmul_q4k_f32_scalar(&q4k_data, &input, out_dim, in_dim);
    let dispatch = matmul_q4k_f32_dispatch(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(dispatch.len(), out_dim);

    // Spot-check rows
    for &i in &[0, 100, 10_000, 31_249] {
        let diff = (scalar[i] - dispatch[i]).abs();
        let tol = scalar[i].abs() * 1e-4 + 1e-4;
        assert!(
            diff < tol,
            "Row {}: scalar={}, dispatch={}, diff={}",
            i,
            scalar[i],
            dispatch[i],
            diff
        );
    }
}

/// Just below threshold: total_work = 7_999_999 should NOT use parallel.
///
/// Verifies that dispatch still produces correct results for the non-parallel path.
/// This ensures the threshold boundary is tested from both sides.
#[test]
fn test_q4k_dispatch_just_below_threshold() {
    // 31249 * 256 = 7,999,744 < 8M (close to threshold but below)
    let out_dim = 31_249;
    let in_dim = 256;
    let total_work = out_dim * in_dim;
    assert!(total_work < 8_000_000);

    let q4k_data = build_q4k_test_data(out_dim, in_dim);
    let input = build_input(in_dim);

    let scalar = matmul_q4k_f32_scalar(&q4k_data, &input, out_dim, in_dim);
    let dispatch = matmul_q4k_f32_dispatch(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(dispatch.len(), out_dim);

    for &i in &[0, 100, 15_000, 31_248] {
        let diff = (scalar[i] - dispatch[i]).abs();
        let tol = scalar[i].abs() * 1e-4 + 1e-4;
        assert!(
            diff < tol,
            "Row {}: scalar={}, dispatch={}, diff={}",
            i,
            scalar[i],
            dispatch[i],
            diff
        );
    }
}

/// Single output row with massive in_dim: tests chunk_size > out_dim scenario.
///
/// out_dim=1, in_dim=8M => total_work=8M. Only 1 chunk exists, assigned to 1 thread.
#[test]
fn test_q4k_parallel_dispatch_single_row() {
    let out_dim = 1;
    let in_dim = 8_388_608; // 32768 super-blocks => total_work = 8,388,608
    let total_work = out_dim * in_dim;
    assert!(total_work >= 8_000_000);
    assert_eq!(in_dim % SUPER_BLOCK_SIZE, 0);

    let q4k_data = build_q4k_test_data(out_dim, in_dim);
    let input = build_input(in_dim);

    let scalar = matmul_q4k_f32_scalar(&q4k_data, &input, out_dim, in_dim);
    let dispatch = matmul_q4k_f32_dispatch(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(dispatch.len(), 1);
    let diff = (scalar[0] - dispatch[0]).abs();
    // Tolerance accounts for FP32 reduction order differences between scalar,
    // AVX2 (8-wide), and AVX-512 (16-wide) accumulation paths.
    // Contract: avx512-q4k-v1.yaml allows wider tolerance for different SIMD widths.
    let tol = scalar[0].abs() * 2e-4 + 1e-4;
    assert!(diff < tol, "scalar={}, dispatch={}, diff={}", scalar[0], dispatch[0], diff);
}

/// All-zero input: parallel path should produce all-zero output regardless.
#[test]
fn test_q4k_parallel_dispatch_zero_input() {
    let out_dim = 4096;
    let in_dim = 2048;
    assert!(out_dim * in_dim >= 8_000_000);

    let q4k_data = build_q4k_test_data(out_dim, in_dim);
    let input = vec![0.0f32; in_dim];

    let dispatch = matmul_q4k_f32_dispatch(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(dispatch.len(), out_dim);
    // With zero input, result should be the negative dmin contribution only
    // (since dmin * mins * 0.0 = 0 and d * scales * q * 0.0 = 0)
    // Actually: sum += (d1*q_val - dm1) * input[idx], so with input=0, sum=0
    for (i, &val) in dispatch.iter().enumerate() {
        assert_eq!(val, 0.0, "Row {}: expected 0.0 with zero input, got {}", i, val);
    }
}
