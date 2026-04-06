
#[test]
fn test_fused_q4k_tiled_matvec_large() {
    // RED: Test with larger dimensions to exercise tiling
    use crate::quantize::fused_q4k_tiled_matvec;

    // 128 output dimensions, 512 input dimensions (2 super-blocks per row)
    let in_dim = 512;
    let out_dim = 128;
    let bytes_per_row = 2 * 144; // 2 super-blocks × 144 bytes

    // Create weight data
    let mut weight_data = Vec::with_capacity(out_dim * bytes_per_row);
    for row in 0..out_dim {
        for sb in 0..2 {
            let d = 1.0 + (row as f32) * 0.01 + (sb as f32) * 0.001;
            weight_data.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
            weight_data.extend_from_slice(&half::f16::from_f32(0.0).to_bits().to_le_bytes());
            for i in 0..12 {
                weight_data.push(((row * 3 + sb * 5 + i) % 64) as u8);
            }
            for i in 0..128 {
                weight_data.push(((row * 7 + sb * 11 + i) % 256) as u8);
            }
        }
    }

    // Activations
    let activations: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.005).cos()).collect();

    // Reference
    let mut reference = Vec::with_capacity(out_dim);
    for row in 0..out_dim {
        let row_start = row * bytes_per_row;
        let row_data = &weight_data[row_start..row_start + bytes_per_row];
        let dot = fused_q4k_dot_simd(row_data, &activations).expect("test");
        reference.push(dot);
    }

    // Tiled with default tile size (64)
    let tiled =
        fused_q4k_tiled_matvec(&weight_data, &activations, in_dim, out_dim, None).expect("test");

    assert_eq!(tiled.len(), out_dim);
    for i in 0..out_dim {
        assert_ulp_eq(
            tiled[i],
            reference[i],
            8,
            &format!("tiled_matvec_large output {}", i),
        );
    }
}

#[test]
fn test_fused_q4k_tiled_matvec_custom_tile_size() {
    // RED: Test that different tile sizes produce same results
    use crate::quantize::fused_q4k_tiled_matvec;

    let in_dim = 256;
    let out_dim = 100;

    // Create weight data
    let mut weight_data = Vec::with_capacity(out_dim * 144);
    for row in 0..out_dim {
        let d = 1.0 + (row as f32) * 0.02;
        weight_data.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        weight_data.extend_from_slice(&half::f16::from_f32(0.1).to_bits().to_le_bytes());
        for i in 0..12 {
            weight_data.push(((row + i) % 64) as u8);
        }
        for i in 0..128 {
            weight_data.push(((row * 2 + i) % 256) as u8);
        }
    }

    let activations: Vec<f32> = (0..in_dim).map(|i| i as f32 * 0.01).collect();

    // Test with different tile sizes
    let tile_sizes = [1, 8, 16, 32, 64, 100, 128];
    let reference =
        fused_q4k_tiled_matvec(&weight_data, &activations, in_dim, out_dim, Some(1)).expect("test");

    for &tile_size in &tile_sizes[1..] {
        let result =
            fused_q4k_tiled_matvec(&weight_data, &activations, in_dim, out_dim, Some(tile_size))
                .expect("test");
        assert_eq!(result.len(), out_dim);
        for i in 0..out_dim {
            assert_ulp_eq(
                result[i],
                reference[i],
                4,
                &format!("tile_size={} output {}", tile_size, i),
            );
        }
    }
}

#[test]
fn test_fused_q4k_tiled_matvec_error_handling() {
    // RED: Test error cases
    use crate::quantize::fused_q4k_tiled_matvec;

    // Weight data too small
    let small_data = vec![0u8; 100];
    let activations = vec![0.0f32; 256];
    assert!(fused_q4k_tiled_matvec(&small_data, &activations, 256, 4, None).is_err());

    // Activation length mismatch
    let weight_data = vec![0u8; 4 * 144];
    let bad_activations = vec![0.0f32; 128];
    assert!(fused_q4k_tiled_matvec(&weight_data, &bad_activations, 256, 4, None).is_err());
}

// -------------------------------------------------------------------------
// Phase 2: Parallel Matrix-Vector Multiplication Tests
// -------------------------------------------------------------------------

#[test]
fn test_fused_q4k_parallel_matvec_basic() {
    // RED: Test parallel matvec produces same results as sequential
    use crate::quantize::fused_q4k_parallel_matvec;

    let in_dim = 256;
    let out_dim = 64;

    // Create weight data
    let mut weight_data = Vec::with_capacity(out_dim * 144);
    for row in 0..out_dim {
        let d = 0.5 + (row as f32) * 0.01;
        weight_data.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        weight_data.extend_from_slice(&half::f16::from_f32(0.05).to_bits().to_le_bytes());
        for i in 0..12 {
            weight_data.push(((row * 7 + i) % 64) as u8);
        }
        for i in 0..128 {
            weight_data.push(((row * 13 + i) % 256) as u8);
        }
    }

    let activations: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.01).sin()).collect();

    // Reference: sequential computation
    let mut reference = Vec::with_capacity(out_dim);
    for row in 0..out_dim {
        let row_start = row * 144;
        let row_data = &weight_data[row_start..row_start + 144];
        let dot = fused_q4k_dot_simd(row_data, &activations).expect("test");
        reference.push(dot);
    }

    // Parallel result (uses Q8K activation quantization internally — Refs realizar#96)
    let parallel =
        fused_q4k_parallel_matvec(&weight_data, &activations, in_dim, out_dim).expect("test");

    assert_eq!(parallel.len(), out_dim);
    // Contract: cpu-q4k-activation-quant-v1.yaml — Q8K error < 0.1% relative to f32
    for i in 0..out_dim {
        let rel_err = if reference[i].abs() > 1e-6 {
            (parallel[i] - reference[i]).abs() / reference[i].abs()
        } else {
            (parallel[i] - reference[i]).abs()
        };
        assert!(
            rel_err < 0.002,
            "parallel_matvec output {}: actual={}, expected={}, rel_err={:.4}% > 0.2%",
            i, parallel[i], reference[i], rel_err * 100.0
        );
    }
}

#[test]
fn test_fused_q4k_parallel_matvec_large() {
    // RED: Test with larger dimensions typical of real models
    use crate::quantize::fused_q4k_parallel_matvec;

    let in_dim = 512;
    let out_dim = 256;
    let bytes_per_row = 2 * 144; // 2 super-blocks × 144 bytes

    // Create weight data
    let mut weight_data = Vec::with_capacity(out_dim * bytes_per_row);
    for row in 0..out_dim {
        for sb in 0..2 {
            let d = 1.0 + (row as f32) * 0.005 + (sb as f32) * 0.001;
            weight_data.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
            weight_data.extend_from_slice(&half::f16::from_f32(0.0).to_bits().to_le_bytes());
            for i in 0..12 {
                weight_data.push(((row * 3 + sb * 5 + i) % 64) as u8);
            }
            for i in 0..128 {
                weight_data.push(((row * 7 + sb * 11 + i) % 256) as u8);
            }
        }
    }

    let activations: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.003).cos()).collect();

    // Reference (f32 dot product path)
    let mut reference = Vec::with_capacity(out_dim);
    for row in 0..out_dim {
        let row_start = row * bytes_per_row;
        let row_data = &weight_data[row_start..row_start + bytes_per_row];
        let dot = fused_q4k_dot_simd(row_data, &activations).expect("test");
        reference.push(dot);
    }

    // Parallel result (uses Q8K activation quantization internally — Refs realizar#96)
    let parallel =
        fused_q4k_parallel_matvec(&weight_data, &activations, in_dim, out_dim).expect("test");

    assert_eq!(parallel.len(), out_dim);
    // Contract: cpu-q4k-activation-quant-v1.yaml — Q8K error < 0.1% relative to f32
    for i in 0..out_dim {
        let rel_err = if reference[i].abs() > 1e-6 {
            (parallel[i] - reference[i]).abs() / reference[i].abs()
        } else {
            (parallel[i] - reference[i]).abs()
        };
        assert!(
            rel_err < 0.002,
            "parallel_matvec_large output {}: actual={}, expected={}, rel_err={:.4}% > 0.2%",
            i, parallel[i], reference[i], rel_err * 100.0
        );
    }
}

#[test]
fn test_fused_q5k_parallel_matvec_basic() {
    // RED: Test Q5_K parallel matvec
    use crate::quantize::fused_q5k_parallel_matvec;

    let in_dim = 256;
    let out_dim = 32;
    let bytes_per_row = 176;

    // Create weight data
    let mut weight_data = Vec::with_capacity(out_dim * bytes_per_row);
    for row in 0..out_dim {
        let d = 0.5 + (row as f32) * 0.02;
        weight_data.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        weight_data.extend_from_slice(&half::f16::from_f32(0.05).to_bits().to_le_bytes());
        // scales (12 bytes)
        for i in 0..12 {
            weight_data.push(((row * 5 + i) % 64) as u8);
        }
        // qh (32 bytes)
        for i in 0..32 {
            weight_data.push(((row * 3 + i) % 256) as u8);
        }
        // qs (128 bytes)
        for i in 0..128 {
            weight_data.push(((row * 11 + i) % 256) as u8);
        }
    }

    let activations: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.01).sin()).collect();

    // Reference
    let mut reference = Vec::with_capacity(out_dim);
    for row in 0..out_dim {
        let row_start = row * bytes_per_row;
        let row_data = &weight_data[row_start..row_start + bytes_per_row];
        let dot = fused_q5k_dot_simd(row_data, &activations).expect("test");
        reference.push(dot);
    }

    // Parallel result
    let parallel =
        fused_q5k_parallel_matvec(&weight_data, &activations, in_dim, out_dim).expect("test");

    assert_eq!(parallel.len(), out_dim);
    for i in 0..out_dim {
        assert_ulp_eq(
            parallel[i],
            reference[i],
            4,
            &format!("q5k_parallel output {}", i),
        );
    }
}

#[test]
fn test_fused_q6k_parallel_matvec_basic() {
    // RED: Test Q6_K parallel matvec
    use crate::quantize::fused_q6k_parallel_matvec;

    let in_dim = 256;
    let out_dim = 32;
    let bytes_per_row = 210;

    // Create weight data (Q6_K layout: ql + qh + scales + d)
    let mut weight_data = Vec::with_capacity(out_dim * bytes_per_row);
    for row in 0..out_dim {
        // ql: 128 bytes
        for i in 0..128 {
            weight_data.push(((row * 7 + i) % 256) as u8);
        }
        // qh: 64 bytes
        for i in 0..64 {
            weight_data.push(((row * 3 + i) % 256) as u8);
        }
        // scales: 16 bytes (i8)
        for i in 0..16 {
            weight_data.push(((row + i) % 128) as u8);
        }
        // d: 2 bytes (f16)
        let d = 0.5 + (row as f32) * 0.02;
        weight_data.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
    }

    let activations: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.01).sin()).collect();

    // Reference
    let mut reference = Vec::with_capacity(out_dim);
    for row in 0..out_dim {
        let row_start = row * bytes_per_row;
        let row_data = &weight_data[row_start..row_start + bytes_per_row];
        let dot = fused_q6k_dot_simd(row_data, &activations).expect("test");
        reference.push(dot);
    }

    // Parallel result
    let parallel =
        fused_q6k_parallel_matvec(&weight_data, &activations, in_dim, out_dim).expect("test");

    assert_eq!(parallel.len(), out_dim);
    for i in 0..out_dim {
        assert_ulp_eq(
            parallel[i],
            reference[i],
            4,
            &format!("q6k_parallel output {}", i),
        );
    }
}

#[test]
fn test_fused_parallel_matvec_error_handling() {
    // RED: Test error cases for parallel matvec
    use crate::quantize::fused_q4k_parallel_matvec;

    // Weight data too small
    let small_data = vec![0u8; 100];
    let activations = vec![0.0f32; 256];
    assert!(fused_q4k_parallel_matvec(&small_data, &activations, 256, 4).is_err());

    // Activation length mismatch
    let weight_data = vec![0u8; 4 * 144];
    let bad_activations = vec![0.0f32; 128];
    assert!(fused_q4k_parallel_matvec(&weight_data, &bad_activations, 256, 4).is_err());
}
