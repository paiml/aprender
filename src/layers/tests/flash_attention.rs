
// Flash Attention v2 tests

#[test]
fn test_flash_attention_v2_matches_standard() {
    // Flash Attention v2 with SIMD should match standard attention
    let attn = Attention::new(8).expect("test");

    let q_data = vec![1.0, 2.0, 0.5, 1.5, 2.5, 0.3, 1.2, 0.8];
    let k_data = vec![0.5, 1.0, 1.5, 0.8, 0.3, 2.0, 0.9, 1.1];
    let v_data = vec![2.0, 1.0, 0.5, 3.0, 1.5, 0.7, 2.5, 0.9];

    let q = Tensor::from_vec(vec![1, 8], q_data).expect("test");
    let k = Tensor::from_vec(vec![1, 8], k_data).expect("test");
    let v = Tensor::from_vec(vec![1, 8], v_data).expect("test");

    let standard = attn.forward(&q, &k, &v).expect("test");
    let v2 = attn.flash_forward_v2(&q, &k, &v, 1).expect("test");

    assert_eq!(standard.shape(), v2.shape());
    for i in 0..standard.data().len() {
        assert!(
            (standard.data()[i] - v2.data()[i]).abs() < 1e-4,
            "Mismatch at {}: {} vs {}",
            i,
            standard.data()[i],
            v2.data()[i]
        );
    }
}

#[test]
fn test_flash_attention_v2_multi_position() {
    let attn = Attention::new(4).expect("test");

    #[rustfmt::skip]
    let q_data = vec![
        1.0, 0.5, 0.3, 1.2,
        0.5, 1.0, 0.8, 0.4,
        0.3, 0.8, 1.0, 0.6,
    ];
    #[rustfmt::skip]
    let k_data = vec![
        1.0, 0.5, 0.3, 1.2,
        0.5, 1.0, 0.8, 0.4,
        0.3, 0.8, 1.0, 0.6,
    ];
    #[rustfmt::skip]
    let v_data = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
    ];

    let q = Tensor::from_vec(vec![3, 4], q_data).expect("test");
    let k = Tensor::from_vec(vec![3, 4], k_data).expect("test");
    let v = Tensor::from_vec(vec![3, 4], v_data).expect("test");

    let standard = attn.forward(&q, &k, &v).expect("test");

    for block_size in [1, 2, 3, 4] {
        let v2 = attn.flash_forward_v2(&q, &k, &v, block_size).expect("test");
        assert_eq!(standard.shape(), v2.shape());
        for i in 0..standard.data().len() {
            assert!(
                (standard.data()[i] - v2.data()[i]).abs() < 1e-4,
                "Block size {}, mismatch at {}: {} vs {}",
                block_size,
                i,
                standard.data()[i],
                v2.data()[i]
            );
        }
    }
}

#[test]
fn test_flash_attention_v2_zero_block_size_error() {
    let attn = Attention::new(4).expect("test");
    let q = Tensor::from_vec(vec![1, 4], vec![1.0; 4]).expect("test");
    let k = Tensor::from_vec(vec![1, 4], vec![1.0; 4]).expect("test");
    let v = Tensor::from_vec(vec![1, 4], vec![1.0; 4]).expect("test");

    let result = attn.flash_forward_v2(&q, &k, &v, 0);
    assert!(result.is_err());
}

#[test]
fn test_flash_attention_v2_large_sequence() {
    let attn = Attention::new(8).expect("test");

    let mut q_data = Vec::new();
    let mut k_data = Vec::new();
    let mut v_data = Vec::new();

    for i in 0..32 {
        for j in 0..8 {
            #[allow(clippy::cast_precision_loss)]
            {
                q_data.push((i * 8 + j) as f32 * 0.05);
                k_data.push((i * 8 + j) as f32 * 0.03);
                v_data.push((i * 8 + j) as f32 * 0.1);
            }
        }
    }

    let q = Tensor::from_vec(vec![32, 8], q_data).expect("test");
    let k = Tensor::from_vec(vec![32, 8], k_data).expect("test");
    let v = Tensor::from_vec(vec![32, 8], v_data).expect("test");

    let standard = attn.forward(&q, &k, &v).expect("test");
    let v2 = attn.flash_forward_v2(&q, &k, &v, 8).expect("test");

    assert_eq!(standard.shape(), v2.shape());
    for i in 0..standard.data().len() {
        assert!(
            (standard.data()[i] - v2.data()[i]).abs() < 1e-3,
            "Mismatch at {}: {} vs {}",
            i,
            standard.data()[i],
            v2.data()[i]
        );
    }
}

// Parallel Flash Attention tests

#[test]
fn test_flash_attention_parallel_matches_standard() {
    let attn = Attention::new(8).expect("test");

    let q_data = vec![1.0, 2.0, 0.5, 1.5, 2.5, 0.3, 1.2, 0.8];
    let k_data = vec![0.5, 1.0, 1.5, 0.8, 0.3, 2.0, 0.9, 1.1];
    let v_data = vec![2.0, 1.0, 0.5, 3.0, 1.5, 0.7, 2.5, 0.9];

    let q = Tensor::from_vec(vec![1, 8], q_data).expect("test");
    let k = Tensor::from_vec(vec![1, 8], k_data).expect("test");
    let v = Tensor::from_vec(vec![1, 8], v_data).expect("test");

    let standard = attn.forward(&q, &k, &v).expect("test");
    let parallel = attn.flash_forward_parallel(&q, &k, &v, 1).expect("test");

    assert_eq!(standard.shape(), parallel.shape());
    for i in 0..standard.data().len() {
        assert!(
            (standard.data()[i] - parallel.data()[i]).abs() < 1e-4,
            "Mismatch at {}: {} vs {}",
            i,
            standard.data()[i],
            parallel.data()[i]
        );
    }
}

#[test]
fn test_flash_attention_parallel_multi_position() {
    let attn = Attention::new(4).expect("test");

    #[rustfmt::skip]
    let q_data = vec![
        1.0, 0.5, 0.3, 1.2,
        0.5, 1.0, 0.8, 0.4,
        0.3, 0.8, 1.0, 0.6,
        0.7, 0.2, 0.9, 0.5,
    ];
    let k_data = q_data.clone();
    #[rustfmt::skip]
    let v_data = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];

    let q = Tensor::from_vec(vec![4, 4], q_data).expect("test");
    let k = Tensor::from_vec(vec![4, 4], k_data).expect("test");
    let v = Tensor::from_vec(vec![4, 4], v_data).expect("test");

    let standard = attn.forward(&q, &k, &v).expect("test");

    for block_size in [1, 2, 4] {
        let parallel = attn
            .flash_forward_parallel(&q, &k, &v, block_size)
            .expect("test");
        assert_eq!(standard.shape(), parallel.shape());
        for i in 0..standard.data().len() {
            assert!(
                (standard.data()[i] - parallel.data()[i]).abs() < 1e-4,
                "Block size {}, mismatch at {}: {} vs {}",
                block_size,
                i,
                standard.data()[i],
                parallel.data()[i]
            );
        }
    }
}

#[test]
fn test_flash_attention_parallel_zero_block_size_error() {
    let attn = Attention::new(4).expect("test");
    let q = Tensor::from_vec(vec![1, 4], vec![1.0; 4]).expect("test");
    let k = Tensor::from_vec(vec![1, 4], vec![1.0; 4]).expect("test");
    let v = Tensor::from_vec(vec![1, 4], vec![1.0; 4]).expect("test");

    let result = attn.flash_forward_parallel(&q, &k, &v, 0);
    assert!(result.is_err());
}

#[test]
fn test_flash_attention_parallel_large_sequence() {
    let attn = Attention::new(16).expect("test");

    let mut q_data = Vec::new();
    let mut k_data = Vec::new();
    let mut v_data = Vec::new();

    for i in 0..64 {
        for j in 0..16 {
            #[allow(clippy::cast_precision_loss)]
            {
                q_data.push((i * 16 + j) as f32 * 0.02);
                k_data.push((i * 16 + j) as f32 * 0.015);
                v_data.push((i * 16 + j) as f32 * 0.05);
            }
        }
    }

    let q = Tensor::from_vec(vec![64, 16], q_data).expect("test");
    let k = Tensor::from_vec(vec![64, 16], k_data).expect("test");
    let v = Tensor::from_vec(vec![64, 16], v_data).expect("test");

    let standard = attn.forward(&q, &k, &v).expect("test");
    let parallel = attn.flash_forward_parallel(&q, &k, &v, 16).expect("test");

    assert_eq!(standard.shape(), parallel.shape());
    for i in 0..standard.data().len() {
        assert!(
            (standard.data()[i] - parallel.data()[i]).abs() < 1e-3,
            "Mismatch at {}: {} vs {}",
            i,
            standard.data()[i],
            parallel.data()[i]
        );
    }
}

#[test]
fn test_flash_attention_v2_vs_parallel_consistency() {
    // Both v2 and parallel should produce same results
    let attn = Attention::new(8).expect("test");

    let mut q_data = Vec::new();
    let mut k_data = Vec::new();
    let mut v_data = Vec::new();

    for i in 0..16 {
        for j in 0..8 {
            #[allow(clippy::cast_precision_loss)]
            {
                q_data.push((i * 8 + j) as f32 * 0.1);
                k_data.push((i * 8 + j) as f32 * 0.08);
                v_data.push((i * 8 + j) as f32 * 0.15);
            }
        }
    }

    let q = Tensor::from_vec(vec![16, 8], q_data).expect("test");
    let k = Tensor::from_vec(vec![16, 8], k_data).expect("test");
    let v = Tensor::from_vec(vec![16, 8], v_data).expect("test");

    let v2 = attn.flash_forward_v2(&q, &k, &v, 4).expect("test");
    let parallel = attn.flash_forward_parallel(&q, &k, &v, 4).expect("test");

    assert_eq!(v2.shape(), parallel.shape());
    for i in 0..v2.data().len() {
        assert!(
            (v2.data()[i] - parallel.data()[i]).abs() < 1e-5,
            "Mismatch at {}: v2={} vs parallel={}",
            i,
            v2.data()[i],
            parallel.data()[i]
        );
    }
}

// RoPE (Rotary Position Embeddings) tests

#[test]
fn test_rope_creation() {
    let rope = RoPE::new(64, 10000.0).expect("test");
    assert_eq!(rope.dim(), 64);
    assert!((rope.base() - 10000.0).abs() < 1e-6);
    assert_eq!(rope.inv_freq().len(), 32); // dim/2
}

#[test]
fn test_rope_with_default_base() {
    let rope = RoPE::with_default_base(128).expect("test");
    assert_eq!(rope.dim(), 128);
    assert!((rope.base() - 10000.0).abs() < 1e-6);
}

#[test]
fn test_rope_zero_dim_error() {
    let result = RoPE::new(0, 10000.0);
    assert!(result.is_err());
}

#[test]
fn test_rope_odd_dim_error() {
    let result = RoPE::new(63, 10000.0);
    assert!(result.is_err());
}

#[test]
fn test_rope_forward_shape() {
    let rope = RoPE::with_default_base(4).expect("test");
    let input = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");

    let output = rope.forward(&input, 0).expect("test");
    assert_eq!(output.shape(), &[2, 4]);
    assert_eq!(output.data().len(), 8);
}

#[test]
fn test_rope_position_zero_identity() {
    // At position 0, rotation angles are 0, so cos=1, sin=0
    // This should return input unchanged
    let rope = RoPE::with_default_base(4).expect("test");
    let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");

    let output = rope.forward(&input, 0).expect("test");

    // At position 0, angles are 0, so cos(0)=1, sin(0)=0
    // y0 = x0 * 1 - x1 * 0 = x0
    // y1 = x0 * 0 + x1 * 1 = x1
    for i in 0..4 {
        assert!(
            (output.data()[i] - input.data()[i]).abs() < 1e-6,
            "Position 0 should be identity: expected {}, got {}",
            input.data()[i],
            output.data()[i]
        );
    }
}

#[test]
fn test_rope_preserves_norm() {
    // Rotation should preserve vector norm
    let rope = RoPE::with_default_base(4).expect("test");
    let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");

    let output = rope.forward(&input, 100).expect("test");

    // Compute L2 norm of input pairs and output pairs
    // Each pair should have the same norm after rotation
    let in_norm_0 = (input.data()[0].powi(2) + input.data()[1].powi(2)).sqrt();
    let in_norm_1 = (input.data()[2].powi(2) + input.data()[3].powi(2)).sqrt();
    let out_norm_0 = (output.data()[0].powi(2) + output.data()[1].powi(2)).sqrt();
    let out_norm_1 = (output.data()[2].powi(2) + output.data()[3].powi(2)).sqrt();

    assert!(
        (in_norm_0 - out_norm_0).abs() < 1e-5,
        "Pair 0 norm should be preserved"
    );
    assert!(
        (in_norm_1 - out_norm_1).abs() < 1e-5,
        "Pair 1 norm should be preserved"
    );
}

#[test]
fn test_rope_different_positions() {
    let rope = RoPE::with_default_base(4).expect("test");
    let input = Tensor::from_vec(vec![4], vec![1.0, 0.0, 1.0, 0.0]).expect("test");

    let out_pos_zero = rope.forward(&input, 0).expect("test");
    let out_pos_ten = rope.forward(&input, 10).expect("test");
    let out_pos_hundred = rope.forward(&input, 100).expect("test");

    // Different positions should give different outputs
    assert!(
        (out_pos_zero.data()[0] - out_pos_ten.data()[0]).abs() > 1e-6
            || (out_pos_zero.data()[1] - out_pos_ten.data()[1]).abs() > 1e-6
    );
    assert!(
        (out_pos_ten.data()[0] - out_pos_hundred.data()[0]).abs() > 1e-6
            || (out_pos_ten.data()[1] - out_pos_hundred.data()[1]).abs() > 1e-6
    );
}

#[test]
fn test_rope_dimension_mismatch_error() {
    let rope = RoPE::with_default_base(4).expect("test");
    let input = Tensor::from_vec(vec![6], vec![1.0; 6]).expect("test");

    let result = rope.forward(&input, 0);
    assert!(result.is_err());
}

#[test]
fn test_rope_batched() {
    // Test with batched input [batch, dim]
    let rope = RoPE::with_default_base(4).expect("test");
    let input = Tensor::from_vec(vec![3, 4], vec![1.0; 12]).expect("test");

    let output = rope.forward(&input, 5).expect("test");
    assert_eq!(output.shape(), &[3, 4]);

    // All vectors in batch should have same rotation applied
    // (since same position)
    for batch in 0..3 {
        for i in 0..4 {
            let expected = output.data()[i]; // First vector
            let actual = output.data()[batch * 4 + i];
            assert!(
                (expected - actual).abs() < 1e-6,
                "All batch elements should have same rotation"
            );
        }
    }
}

#[test]
fn test_rope_inv_freq_computation() {
    // Test that inverse frequencies are computed correctly
    let rope = RoPE::new(4, 10000.0).expect("test");
    let inv_freq = rope.inv_freq();

    // For dim=4, we have 2 pairs
    // inv_freq[0] = 10000^(-2*0/4) = 10000^0 = 1.0
    // inv_freq[1] = 10000^(-2*1/4) = 10000^(-0.5) = 0.01
    assert!((inv_freq[0] - 1.0).abs() < 1e-6);
    assert!((inv_freq[1] - 0.01).abs() < 1e-6);
}

// ScaledRoPE (NTK, YaRN, Linear, Dynamic NTK) tests

#[test]
fn test_scaled_rope_no_scaling() {
    // ScaledRoPE with None scaling should behave like regular RoPE
    let scaled = ScaledRoPE::new(64, 10000.0, RopeScalingType::None).expect("test");
    assert_eq!(scaled.dim(), 64);
    assert!((scaled.original_base() - 10000.0).abs() < 1e-6);
    assert!((scaled.scaled_base() - 10000.0).abs() < 1e-6);
    assert!((scaled.mscale() - 1.0).abs() < 1e-6);
    assert!((scaled.context_length_multiplier() - 1.0).abs() < 1e-6);
}
