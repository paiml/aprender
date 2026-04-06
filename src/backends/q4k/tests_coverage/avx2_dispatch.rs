use super::super::*;

// ================================================================
// Q4K AVX2 GEMV coverage - dispatch with various sizes
// ================================================================

/// Test dispatch with multiple super-blocks, exercising AVX2 inner loops
#[test]
fn test_q4k_dispatch_multi_superblock() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    // 4 super-blocks = 1024 elements
    let in_dim = 1024;
    let out_dim = 8;
    let num_blocks = in_dim / SUPER_BLOCK_SIZE;

    let mut q4k_data = Vec::new();
    for row in 0..out_dim {
        for sb in 0..num_blocks {
            q4k_data.extend_from_slice(&[0x66, 0x2E]); // d
            q4k_data.extend_from_slice(&[0x66, 0x2A]); // dmin
            let sv = ((row + sb) as u8 + 1) | (((row + sb) as u8 + 2) << 4);
            q4k_data.extend_from_slice(&[sv; 12]);
            for i in 0..128 {
                let low = ((row + sb + i) % 16) as u8;
                let high = ((row + sb + i + 3) % 16) as u8;
                q4k_data.push(low | (high << 4));
            }
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.001 - 0.5).collect();

    let scalar = matmul_q4k_f32_scalar(&q4k_data, &input, out_dim, in_dim);
    let dispatch = matmul_q4k_f32_dispatch(&q4k_data, &input, out_dim, in_dim);

    for (i, (s, d)) in scalar.iter().zip(dispatch.iter()).enumerate() {
        let diff = (s - d).abs();
        let rel_diff = if s.abs() > 1e-6 { diff / s.abs() } else { diff };
        assert!(
            rel_diff < 1e-4 || diff < 1e-4,
            "Row {}: scalar={}, dispatch={}, diff={}, rel={}",
            i,
            s,
            d,
            diff,
            rel_diff
        );
    }
}

/// Test dispatch with 3 super-blocks (768 elements), exercising more inner loop iterations
#[test]
fn test_q4k_dispatch_three_superblocks() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    let in_dim = 768; // 3 super-blocks
    let out_dim = 2;
    let num_blocks = in_dim / SUPER_BLOCK_SIZE;

    let mut q4k_data = Vec::new();
    for row in 0..out_dim {
        for sb in 0..num_blocks {
            q4k_data.extend_from_slice(&[0x00, 0x3C]); // d ~ 1.0
            q4k_data.extend_from_slice(&[0x00, 0x38]); // dmin ~ 0.5
            let sv = ((row + sb) as u8 + 1) & 0x3F;
            q4k_data.extend_from_slice(&[sv; 12]);
            q4k_data.extend_from_slice(&[0x77u8; 128]);
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.005).collect();

    let scalar = matmul_q4k_f32_scalar(&q4k_data, &input, out_dim, in_dim);
    let dispatch = matmul_q4k_f32_dispatch(&q4k_data, &input, out_dim, in_dim);

    for (i, (s, d)) in scalar.iter().zip(dispatch.iter()).enumerate() {
        let diff = (s - d).abs();
        let rel_diff = if s.abs() > 1e-6 { diff / s.abs() } else { diff };
        assert!(
            rel_diff < 1e-4 || diff < 1e-4,
            "Row {}: scalar={}, dispatch={}, diff={}",
            i,
            s,
            d,
            diff
        );
    }
}

/// Test dispatch with negative input values (exercises subtract branches)
#[test]
fn test_q4k_dispatch_negative_inputs() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    let in_dim = 256;
    let out_dim = 4;

    let mut q4k_data = Vec::new();
    for row in 0..out_dim {
        q4k_data.extend_from_slice(&[0x66, 0x2E]); // d ~ 0.1
        q4k_data.extend_from_slice(&[0x66, 0x2A]); // dmin ~ 0.05
        q4k_data.extend_from_slice(&[((row + 1) as u8); 12]);
        for i in 0..128 {
            q4k_data.push(((i * 3 + row) % 256) as u8);
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| -1.0 + (i as f32) * 0.008).collect();

    let scalar = matmul_q4k_f32_scalar(&q4k_data, &input, out_dim, in_dim);
    let dispatch = matmul_q4k_f32_dispatch(&q4k_data, &input, out_dim, in_dim);

    for (i, (s, d)) in scalar.iter().zip(dispatch.iter()).enumerate() {
        let diff = (s - d).abs();
        let rel_diff = if s.abs() > 1e-6 { diff / s.abs() } else { diff };
        assert!(
            rel_diff < 1e-4 || diff < 1e-4,
            "Row {}: scalar={}, dispatch={}, diff={}",
            i,
            s,
            d,
            diff
        );
    }
}
