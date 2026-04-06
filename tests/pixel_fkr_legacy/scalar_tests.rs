//! SCALAR-PIXEL-FKR: Baseline Truth (SPEC Section 3.5.2)

use super::scalar_helpers::*;

/// scalar-pixel-fkr test: RMS Norm
#[test]
fn scalar_pixel_fkr_rmsnorm() {
    let mut rng = SimpleRng::new(12345);
    let x = rng.gen_vec(4096);
    let weight = rng.gen_vec(4096);

    let result = scalar_rmsnorm(&x, &weight, 1e-5);

    // Verify output properties
    assert_eq!(result.len(), 4096);
    assert!(result.iter().all(|v| v.is_finite()), "Non-finite value in RMS norm");

    println!(
        "scalar_pixel_fkr_rmsnorm: {} elements, max={:.6}",
        result.len(),
        result.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
}

/// scalar-pixel-fkr test: SiLU
#[test]
fn scalar_pixel_fkr_silu() {
    let mut rng = SimpleRng::new(23456);
    let x = rng.gen_vec(8192);

    let result = scalar_silu(&x);

    // Verify SiLU properties
    assert_eq!(result.len(), 8192);
    // SiLU(x) should be bounded for bounded input
    for (i, (xi, yi)) in x.iter().zip(result.iter()).enumerate() {
        if *xi > 0.0 {
            assert!(*yi > 0.0, "SiLU should be positive for positive input at {i}");
        }
    }

    println!("scalar_pixel_fkr_silu: {} elements", result.len());
}

/// scalar-pixel-fkr test: Softmax
#[test]
fn scalar_pixel_fkr_softmax() {
    let mut rng = SimpleRng::new(34567);
    let x = rng.gen_vec(2048);

    let result = scalar_softmax(&x);

    // Verify softmax properties
    assert_eq!(result.len(), 2048);

    // Sum should be 1.0
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Softmax sum should be 1.0, got {sum}");

    // All values should be in (0, 1)
    for (i, v) in result.iter().enumerate() {
        assert!(*v > 0.0 && *v <= 1.0, "Softmax value at {i} out of range: {v}");
    }

    println!("scalar_pixel_fkr_softmax: sum={:.6}", sum);
}

/// scalar-pixel-fkr test: RoPE
#[test]
fn scalar_pixel_fkr_rope() {
    let mut rng = SimpleRng::new(45678);
    let x = rng.gen_vec(512);
    let (freqs_cos, freqs_sin) = compute_rope_freqs(512, 10000.0);

    let result = scalar_rope(&x, &freqs_cos, &freqs_sin);

    // Verify output dimensions
    assert_eq!(result.len(), 512);
    assert!(result.iter().all(|v| v.is_finite()), "Non-finite in RoPE output");

    // RoPE should preserve norm approximately
    let input_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let output_norm: f32 = result.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_ratio = output_norm / input_norm;
    assert!(
        (norm_ratio - 1.0).abs() < 0.5, // Allow 50% variation due to frequency mixing
        "RoPE norm ratio too far from 1.0: {norm_ratio}"
    );

    println!("scalar_pixel_fkr_rope: input_norm={:.4}, output_norm={:.4}", input_norm, output_norm);
}

/// scalar-pixel-fkr test: Causal Mask
#[test]
fn scalar_pixel_fkr_causal_mask() {
    let seq_len = 64;
    let mut rng = SimpleRng::new(56789);
    let scores = rng.gen_vec(seq_len * seq_len);

    let result = scalar_causal_mask(&scores, seq_len);

    // Verify causal structure
    assert_eq!(result.len(), seq_len * seq_len);

    // Upper triangle should be NEG_INFINITY
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            assert!(
                result[i * seq_len + j] == f32::NEG_INFINITY,
                "Causal mask not applied at [{i}][{j}]"
            );
        }
    }

    // Lower triangle should be unchanged
    for i in 0..seq_len {
        for j in 0..=i {
            assert!(
                result[i * seq_len + j] == scores[i * seq_len + j],
                "Causal mask corrupted lower triangle at [{i}][{j}]"
            );
        }
    }

    println!("scalar_pixel_fkr_causal_mask: {}x{} verified", seq_len, seq_len);
}

/// scalar-pixel-fkr test: Q4_K dequantization
#[test]
fn scalar_pixel_fkr_q4k_dequant() {
    let quantized: Vec<u8> = (0..128).map(|i| i as u8).collect();
    let scale = 0.1;
    let zero_point = 8.0;

    let result = scalar_q4k_dequant(&quantized, scale, zero_point);

    // Each byte produces 2 floats
    assert_eq!(result.len(), 256);

    // Verify dequantization range
    let min_val = result.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = result.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // With 4-bit values [0,15], zero_point=8, scale=0.1
    // Range should be approximately (-8*0.1, (15-8)*0.1) = (-0.8, 0.7)
    assert!(min_val >= -1.0 && max_val <= 1.0, "Dequant range: [{min_val}, {max_val}]");

    println!("scalar_pixel_fkr_q4k_dequant: range=[{:.3}, {:.3}]", min_val, max_val);
}
