
#[test]
fn test_softmax_preserves_shape() {
    let input = Tensor::from_vec(vec![2, 3, 4], vec![1.0; 24]).expect("test");
    let output = softmax(&input).expect("test");

    assert_eq!(output.shape(), &[2, 3, 4]);
}

// GELU activation tests

#[test]
fn test_gelu_zero() {
    let input = Tensor::from_vec(vec![1], vec![0.0]).expect("test");
    let output = gelu(&input).expect("test");
    // GELU(0) = 0
    assert!((output.data()[0] - 0.0).abs() < 1e-6);
}

#[test]
fn test_gelu_positive() {
    let input = Tensor::from_vec(vec![1], vec![1.0]).expect("test");
    let output = gelu(&input).expect("test");
    // GELU(1) ≈ 0.841 (approximately x for large positive x)
    assert!(output.data()[0] > 0.8);
    assert!(output.data()[0] < 0.9);
}

#[test]
fn test_gelu_negative() {
    let input = Tensor::from_vec(vec![1], vec![-1.0]).expect("test");
    let output = gelu(&input).expect("test");
    // GELU(-1) is small negative (smooth near zero)
    assert!(output.data()[0] < 0.0);
    assert!(output.data()[0] > -0.2);
}

#[test]
fn test_gelu_batched() {
    let input = Tensor::from_vec(vec![2, 3], vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]).expect("test");
    let output = gelu(&input).expect("test");

    assert_eq!(output.shape(), &[2, 3]);
    assert_eq!(output.data().len(), 6);

    // GELU(0) = 0
    assert!((output.data()[2] - 0.0).abs() < 1e-6);
    // Positive values should be positive
    assert!(output.data()[3] > 0.0);
    assert!(output.data()[4] > 0.0);
    assert!(output.data()[5] > 0.0);
}

#[test]
fn test_gelu_preserves_shape() {
    // Test that GELU preserves tensor shape
    let input = Tensor::from_vec(vec![2, 3, 4], vec![0.5; 24]).expect("test");
    let output = gelu(&input).expect("test");
    assert_eq!(output.shape(), &[2, 3, 4]);
    assert_eq!(output.data().len(), 24);
}

// FeedForward (FFN) tests

#[test]
fn test_ffn_creation() {
    let ffn = FeedForward::new(512, 2048).expect("test");
    assert_eq!(ffn.hidden_dim(), 512);
    assert_eq!(ffn.intermediate_dim(), 2048);
}

#[test]
fn test_ffn_zero_dimensions_error() {
    let result = FeedForward::new(0, 2048);
    assert!(result.is_err());

    let result = FeedForward::new(512, 0);
    assert!(result.is_err());
}

#[test]
fn test_ffn_forward_shape() {
    // Test that FFN preserves hidden_dim
    let ffn = FeedForward::new(4, 16).expect("test"); // Small sizes for testing
    let input = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).expect("test");
    let output = ffn.forward(&input).expect("test");

    // Output should have same shape as input
    assert_eq!(output.shape(), &[2, 4]);
}

#[test]
fn test_ffn_forward_computation() {
    // Test FFN with known weights
    let mut ffn = FeedForward::new(2, 4).expect("test");

    // Set fc1 weights to identity-like (for simplicity)
    for i in 0..8 {
        ffn.fc1_mut().weight_mut()[i] = 0.1;
    }
    for i in 0..4 {
        ffn.fc1_mut().bias_mut()[i] = 0.0;
    }

    // Set fc2 weights
    for i in 0..8 {
        ffn.fc2_mut().weight_mut()[i] = 0.1;
    }
    for i in 0..2 {
        ffn.fc2_mut().bias_mut()[i] = 0.0;
    }

    // Input: [1.0, 2.0]
    let input = Tensor::from_vec(vec![2], vec![1.0, 2.0]).expect("test");
    let output = ffn.forward(&input).expect("test");

    // Output should be valid (not NaN, not Inf)
    assert_eq!(output.shape(), &[2]);
    assert!(output.data()[0].is_finite());
    assert!(output.data()[1].is_finite());
}

#[test]
fn test_ffn_batched() {
    let ffn = FeedForward::new(3, 12).expect("test");

    // Batched input: [2, 3]
    let input = Tensor::from_vec(vec![2, 3], vec![0.5; 6]).expect("test");
    let output = ffn.forward(&input).expect("test");

    // Output shape should match input
    assert_eq!(output.shape(), &[2, 3]);
    assert_eq!(output.data().len(), 6);
}

#[test]
fn test_ffn_weight_access() {
    let mut ffn = FeedForward::new(2, 4).expect("test");

    // Modify fc1 weights
    ffn.fc1_mut().weight_mut()[0] = 42.0;
    assert!((ffn.fc1_mut().weight_mut()[0] - 42.0).abs() < 1e-6);

    // Modify fc2 bias
    ffn.fc2_mut().bias_mut()[0] = 7.0;
    assert!((ffn.fc2_mut().bias_mut()[0] - 7.0).abs() < 1e-6);
}

// Attention tests

#[test]
fn test_attention_creation() {
    let attn = Attention::new(64).expect("test");
    assert_eq!(attn.head_dim(), 64);
    // scale = 1 / sqrt(64) = 1/8 = 0.125
    assert!((attn.scale() - 0.125).abs() < 1e-6);
}

#[test]
fn test_attention_zero_head_dim_error() {
    let result = Attention::new(0);
    assert!(result.is_err());
}

#[test]
fn test_attention_forward_shape() {
    let attn = Attention::new(4).expect("test");

    // Q, K, V all have shape [3, 4] (seq_len=3, head_dim=4)
    let q = Tensor::from_vec(vec![3, 4], vec![0.1; 12]).expect("test");
    let k = Tensor::from_vec(vec![3, 4], vec![0.2; 12]).expect("test");
    let v = Tensor::from_vec(vec![3, 4], vec![0.3; 12]).expect("test");

    let output = attn.forward(&q, &k, &v).expect("test");

    // Output should have shape [3, 4]
    assert_eq!(output.shape(), &[3, 4]);
    assert_eq!(output.data().len(), 12);
}

#[test]
fn test_attention_forward_computation() {
    let attn = Attention::new(2).expect("test");

    // Simple 2x2 case for manual verification
    // Q = [[1, 0], [0, 1]]
    // K = [[1, 0], [0, 1]]
    // V = [[1, 2], [3, 4]]
    let q = Tensor::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).expect("test");
    let k = Tensor::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).expect("test");
    let v = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("test");

    let output = attn.forward(&q, &k, &v).expect("test");

    // Output should be valid (not NaN, not Inf)
    assert_eq!(output.shape(), &[2, 2]);
    for &val in output.data() {
        assert!(val.is_finite());
    }

    // First row of Q=[1,0] has dot products: with K[0]=[1,0] -> 1, with K[1]=[0,1] -> 0
    // After scaling and softmax, should attend more to first position
    // So output[0] should be closer to V[0]=[1,2] than V[1]=[3,4]
    assert!(output.data()[0] < 2.0); // Closer to 1 than 3
    assert!(output.data()[1] < 3.0); // Closer to 2 than 4
}

#[test]
fn test_attention_shape_mismatch_error() {
    let attn = Attention::new(4).expect("test");

    // Q has head_dim=4, K has head_dim=3
    let q = Tensor::from_vec(vec![2, 4], vec![0.1; 8]).expect("test");
    let k = Tensor::from_vec(vec![2, 3], vec![0.2; 6]).expect("test");
    let v = Tensor::from_vec(vec![2, 4], vec![0.3; 8]).expect("test");

    let result = attn.forward(&q, &k, &v);
    assert!(result.is_err());
}

#[test]
fn test_attention_kv_seq_len_mismatch_error() {
    let attn = Attention::new(4).expect("test");

    // K has seq_len=3, V has seq_len=2
    let q = Tensor::from_vec(vec![2, 4], vec![0.1; 8]).expect("test");
    let k = Tensor::from_vec(vec![3, 4], vec![0.2; 12]).expect("test");
    let v = Tensor::from_vec(vec![2, 4], vec![0.3; 8]).expect("test");

    let result = attn.forward(&q, &k, &v);
    assert!(result.is_err());
}

#[test]
fn test_attention_softmax_weights_sum() {
    // Verify that attention is using softmax correctly
    // by checking output is weighted combination of values
    let attn = Attention::new(3).expect("test");

    // All equal Q and K means uniform attention
    let q = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test");
    let k = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).expect("test");
    // V = [[1, 2, 3], [4, 5, 6]]
    let v = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test");

    let output = attn.forward(&q, &k, &v).expect("test");

    // With uniform attention, output should be average of V rows
    // Each output row should be close to [2.5, 3.5, 4.5]
    let expected = [2.5, 3.5, 4.5];
    for row in 0..2 {
        for (col, &exp) in expected.iter().enumerate() {
            let actual = output.data()[row * 3 + col];
            assert!(
                (actual - exp).abs() < 0.01,
                "row={row}, col={col}: expected {exp}, got {actual}",
            );
        }
    }
}

#[test]
fn test_attention_single_position() {
    // Test with single position (seq_len=1)
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 0.0]).expect("test");
    let k = Tensor::from_vec(vec![4], vec![1.0, 0.0, 0.0, 0.0]).expect("test");
    let v = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");

    let output = attn.forward(&q, &k, &v).expect("test");

    // With single position, output should equal V
    assert_eq!(output.shape(), &[1, 4]);
    for i in 0..4 {
        assert!((output.data()[i] - v.data()[i]).abs() < 1e-6);
    }
}

// Flash Attention tests

#[test]
fn test_flash_attention_matches_standard() {
    // Flash Attention should produce same output as standard attention
    let attn = Attention::new(8).expect("test");

    // Create test data
    let q_data = vec![1.0, 2.0, 0.5, 1.5, 2.5, 0.3, 1.2, 0.8];
    let k_data = vec![0.5, 1.0, 1.5, 0.8, 0.3, 2.0, 0.9, 1.1];
    let v_data = vec![2.0, 1.0, 0.5, 3.0, 1.5, 0.7, 2.5, 0.9];

    let q = Tensor::from_vec(vec![1, 8], q_data.clone()).expect("test");
    let k = Tensor::from_vec(vec![1, 8], k_data.clone()).expect("test");
    let v = Tensor::from_vec(vec![1, 8], v_data.clone()).expect("test");

    // Standard attention
    let standard_output = attn.forward(&q, &k, &v).expect("test");

    // Flash attention with block_size=1 (should be identical)
    let flash_output = attn.flash_forward(&q, &k, &v, 1).expect("test");

    // Results should match
    assert_eq!(standard_output.shape(), flash_output.shape());
    for i in 0..standard_output.data().len() {
        assert!(
            (standard_output.data()[i] - flash_output.data()[i]).abs() < 1e-5,
            "Mismatch at index {}: {} vs {}",
            i,
            standard_output.data()[i],
            flash_output.data()[i]
        );
    }
}

#[test]
fn test_flash_attention_multi_position() {
    // Test Flash Attention with multiple positions
    let attn = Attention::new(4).expect("test");

    #[rustfmt::skip]
    let q_data = vec![
        1.0, 0.0, 0.0, 1.0,  // pos 0
        0.0, 1.0, 1.0, 0.0,  // pos 1
        1.0, 1.0, 0.0, 0.0,  // pos 2
    ];
    #[rustfmt::skip]
    let k_data = vec![
        1.0, 0.0, 0.0, 1.0,  // pos 0
        0.0, 1.0, 1.0, 0.0,  // pos 1
        1.0, 1.0, 0.0, 0.0,  // pos 2
    ];
    #[rustfmt::skip]
    let v_data = vec![
        1.0, 2.0, 3.0, 4.0,  // pos 0
        5.0, 6.0, 7.0, 8.0,  // pos 1
        9.0, 10.0, 11.0, 12.0,  // pos 2
    ];

    let q = Tensor::from_vec(vec![3, 4], q_data).expect("test");
    let k = Tensor::from_vec(vec![3, 4], k_data).expect("test");
    let v = Tensor::from_vec(vec![3, 4], v_data).expect("test");

    // Standard attention
    let standard_output = attn.forward(&q, &k, &v).expect("test");

    // Flash attention with different block sizes
    for block_size in [1, 2, 3, 4] {
        let flash_output = attn.flash_forward(&q, &k, &v, block_size).expect("test");

        assert_eq!(standard_output.shape(), flash_output.shape());
        for i in 0..standard_output.data().len() {
            assert!(
                (standard_output.data()[i] - flash_output.data()[i]).abs() < 1e-4,
                "Block size {}, mismatch at index {}: {} vs {}",
                block_size,
                i,
                standard_output.data()[i],
                flash_output.data()[i]
            );
        }
    }
}

#[test]
fn test_flash_attention_zero_block_size_error() {
    let attn = Attention::new(4).expect("test");

    let q = Tensor::from_vec(vec![1, 4], vec![1.0, 0.0, 0.0, 0.0]).expect("test");
    let k = Tensor::from_vec(vec![1, 4], vec![1.0, 0.0, 0.0, 0.0]).expect("test");
    let v = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).expect("test");

    let result = attn.flash_forward(&q, &k, &v, 0);
    assert!(result.is_err());
}

#[test]
fn test_flash_attention_large_sequence() {
    // Test with larger sequence to verify block-wise computation
    let attn = Attention::new(8).expect("test");

    // Create larger test data (seq_len=16)
    let mut q_data = Vec::new();
    let mut k_data = Vec::new();
    let mut v_data = Vec::new();

    for i in 0..16 {
        for j in 0..8 {
            #[allow(clippy::cast_precision_loss)]
            {
                q_data.push((i * 8 + j) as f32 * 0.1);
                k_data.push((i * 8 + j) as f32 * 0.05);
                v_data.push((i * 8 + j) as f32 * 0.2);
            }
        }
    }

    let q = Tensor::from_vec(vec![16, 8], q_data).expect("test");
    let k = Tensor::from_vec(vec![16, 8], k_data).expect("test");
    let v = Tensor::from_vec(vec![16, 8], v_data).expect("test");

    // Standard attention
    let standard_output = attn.forward(&q, &k, &v).expect("test");

    // Flash attention with block_size=4
    let flash_output = attn.flash_forward(&q, &k, &v, 4).expect("test");

    assert_eq!(standard_output.shape(), flash_output.shape());
    for i in 0..standard_output.data().len() {
        assert!(
            (standard_output.data()[i] - flash_output.data()[i]).abs() < 1e-3,
            "Mismatch at index {}: {} vs {}",
            i,
            standard_output.data()[i],
            flash_output.data()[i]
        );
    }
}

#[test]
fn test_flash_attention_shape_errors() {
    let attn = Attention::new(4).expect("test");

    let q =
        Tensor::from_vec(vec![2, 4], vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).expect("test");
    let k =
        Tensor::from_vec(vec![2, 4], vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).expect("test");
    let v_wrong = Tensor::from_vec(
        vec![3, 4],
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    )
    .expect("test");

    // K/V sequence length mismatch
    let result = attn.flash_forward(&q, &k, &v_wrong, 2);
    assert!(result.is_err());
}
