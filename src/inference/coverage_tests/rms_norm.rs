
#[test]
fn test_rms_norm_large_input() {
    let input = vec![1e6, 1e6, 1e6];
    let weight = vec![1.0; 3];
    let output = simd_rms_norm(&input, &weight, 1e-5);

    // RMS = sqrt(mean(x^2)) = sqrt(1e12) = 1e6
    // So output = input / rms = 1
    for &v in &output {
        assert!((v - 1.0).abs() < 1e-5);
    }
}

#[test]
fn test_rms_norm_mixed_signs() {
    let input = vec![3.0, -4.0];
    let weight = vec![1.0, 1.0];
    let output = simd_rms_norm(&input, &weight, 1e-5);

    // RMS = sqrt((9 + 16) / 2) = sqrt(12.5) ~ 3.54
    let rms = (12.5_f32).sqrt();

    assert!((output[0] - 3.0 / rms).abs() < 1e-5);
    assert!((output[1] - (-4.0 / rms)).abs() < 1e-5);
}

#[test]
fn test_rms_norm_asymmetric_weight() {
    let input = vec![1.0, 2.0, 3.0];
    let weight = vec![0.0, 1.0, 2.0];
    let output = simd_rms_norm(&input, &weight, 1e-5);

    // First element should be 0 (weight is 0)
    assert!((output[0]).abs() < 1e-5);

    // Other elements scaled appropriately
    let rms = ((1.0 + 4.0 + 9.0) / 3.0_f32).sqrt();
    assert!((output[1] - 2.0 / rms * 1.0).abs() < 1e-5);
    assert!((output[2] - 3.0 / rms * 2.0).abs() < 1e-5);
}

// ============================================================================
// RoPE Edge Cases
// ============================================================================

#[test]
fn test_rope_single_head() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    apply_rope(&mut x, 4, 1, 0, 10000.0);

    // At position 0, no rotation
    assert!((x[0] - 1.0).abs() < 1e-5);
    assert!((x[1] - 2.0).abs() < 1e-5);
    assert!((x[2] - 3.0).abs() < 1e-5);
    assert!((x[3] - 4.0).abs() < 1e-5);
}

#[test]
fn test_rope_position_one() {
    let original = vec![1.0, 0.0, 1.0, 0.0]; // 4 hidden, 1 head
    let mut x = original.clone();
    apply_rope(&mut x, 4, 1, 1, 10000.0);

    // At position 1, some rotation should occur
    // All values should be finite
    for v in &x {
        assert!(v.is_finite(), "RoPE output should be finite");
    }

    // At position > 0, at least some rotation should happen
    // (at position 0, no rotation occurs)
    let changed = x
        .iter()
        .zip(original.iter())
        .any(|(a, b)| (a - b).abs() > 1e-10);
    assert!(changed, "RoPE at position 1 should rotate the input");
}

#[test]
fn test_rope_very_large_position() {
    let mut x = vec![1.0; 8];
    apply_rope(&mut x, 8, 2, 10000, 10000.0);

    // Results should still be finite
    for &v in &x {
        assert!(v.is_finite());
    }
}

#[test]
fn test_rope_small_theta() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    apply_rope(&mut x, 4, 1, 1, 100.0);

    // With smaller theta, rotations are faster
    // Magnitudes should still be preserved
    let orig_mag0 = (1.0_f32.powi(2) + 3.0_f32.powi(2)).sqrt();
    let orig_mag1 = (2.0_f32.powi(2) + 4.0_f32.powi(2)).sqrt();

    let new_mag0 = (x[0] * x[0] + x[2] * x[2]).sqrt();
    let new_mag1 = (x[1] * x[1] + x[3] * x[3]).sqrt();

    assert!((new_mag0 - orig_mag0).abs() < 1e-4);
    assert!((new_mag1 - orig_mag1).abs() < 1e-4);
}

#[test]
fn test_rope_many_heads() {
    let hidden_dim = 128;
    let num_heads = 32;
    let mut x: Vec<f32> = (0..hidden_dim).map(|i| i as f32 * 0.1).collect();
    let original = x.clone();

    apply_rope(&mut x, hidden_dim, num_heads, 5, 10000.0);

    // Length should be preserved
    assert_eq!(x.len(), hidden_dim);

    // Values should change (at non-zero position)
    assert!(x != original);

    // All values should be finite
    for &v in &x {
        assert!(v.is_finite());
    }
}

#[test]
fn test_rope_position_sequence() {
    let hidden_dim = 8;
    let num_heads = 2;
    let mut results = Vec::new();

    // Apply RoPE at multiple positions
    for pos in 0..5 {
        let mut x = vec![1.0; hidden_dim];
        apply_rope(&mut x, hidden_dim, num_heads, pos, 10000.0);
        results.push(x);
    }

    // Position 0 should be unchanged
    for &v in &results[0] {
        assert!((v - 1.0).abs() < 1e-5);
    }

    // Subsequent positions should differ
    for i in 1..5 {
        assert!(results[i] != results[0]);
    }
}

// ============================================================================
// Q4KWeight Tests
// ============================================================================

#[test]
fn test_q4k_weight_compression_ratio() {
    let in_dim = 256;
    let out_dim = 4;
    let bytes_per_row = 144;
    let data = vec![0u8; out_dim * bytes_per_row];

    let weight = Q4KWeight::new(data, in_dim, out_dim).expect("valid weight");

    // F32 would need: 256 * 4 * 4 = 4096 bytes
    // Q4_K uses: 4 * 144 = 576 bytes
    let ratio = weight.compression_ratio();
    assert!(ratio > 7.0, "Expected >7x compression, got {}", ratio);
}

#[test]
fn test_q4k_weight_invalid_data_size() {
    let data = vec![0u8; 100]; // Wrong size
    let result = Q4KWeight::new(data, 256, 1);
    assert!(result.is_err());
}

#[test]
fn test_q4k_weight_memory_stats_consistency() {
    let in_dim: usize = 512;
    let out_dim: usize = 2;
    let blocks_per_row = in_dim.div_ceil(256);
    let bytes_per_row = blocks_per_row * 144;
    let data = vec![0u8; out_dim * bytes_per_row];

    let weight = Q4KWeight::new(data, in_dim, out_dim).expect("valid weight");

    assert_eq!(weight.memory_bytes(), out_dim * bytes_per_row);
    assert_eq!(weight.f32_equivalent_bytes(), in_dim * out_dim * 4);
    assert!(weight.compression_ratio() > 1.0);
}

#[test]
fn test_q4k_weight_clone() {
    let in_dim = 256;
    let out_dim = 1;
    let bytes_per_row = 144;
    let data = vec![42u8; out_dim * bytes_per_row];

    let weight = Q4KWeight::new(data, in_dim, out_dim).expect("valid weight");
    let cloned = weight.clone();

    assert_eq!(weight.in_dim, cloned.in_dim);
    assert_eq!(weight.out_dim, cloned.out_dim);
    assert_eq!(weight.data, cloned.data);
}

#[test]
fn test_q4k_weight_matvec_dimension_mismatch() {
    let in_dim = 256;
    let out_dim = 1;
    let bytes_per_row = 144;
    let data = vec![0u8; out_dim * bytes_per_row];

    let weight = Q4KWeight::new(data, in_dim, out_dim).expect("valid weight");

    // Wrong input dimension
    let wrong_input = vec![1.0; in_dim + 1];
    let result = weight.matvec(&wrong_input);
    assert!(result.is_err());
}

// ============================================================================
// Integration: Full Attention Pipeline
// ============================================================================

#[test]
fn test_full_attention_pipeline() {
    let num_layers = 2;
    let hidden_dim = 8;
    let num_heads = 2;
    let max_seq_len = 10;

    let mut cache = KVCache::new(num_layers, hidden_dim, max_seq_len);

    // Simulate processing 5 tokens
    for pos in 0..5 {
        // Create Q, K, V for this position
        let q = vec![0.1 * (pos + 1) as f32; hidden_dim];
        let k = vec![0.2 * (pos + 1) as f32; hidden_dim];
        let v = vec![0.3 * (pos + 1) as f32; hidden_dim];

        // Apply RoPE to Q and K
        let mut q_rope = q.clone();
        let mut k_rope = k.clone();
        apply_rope(&mut q_rope, hidden_dim, num_heads, pos, 10000.0);
        apply_rope(&mut k_rope, hidden_dim, num_heads, pos, 10000.0);

        // Compute attention using cached KV
        for layer in 0..num_layers {
            let output = attention_with_cache(
                &q_rope,
                cache.get_k(layer),
                cache.get_v(layer),
                &k_rope,
                &v,
                num_heads,
            );

            assert_eq!(output.len(), hidden_dim);
            for &val in &output {
                assert!(val.is_finite());
            }

            // Store new KV
            cache.store(layer, &k_rope, &v);
        }
        cache.advance();
    }

    assert_eq!(cache.len(), 5);
}

#[test]
fn test_attention_with_normalization() {
    let hidden_dim = 8;
    let num_heads = 2;

    // Create input and normalize
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let weight = vec![1.0; hidden_dim];

    let normalized = simd_rms_norm(&input, &weight, 1e-5);

    // Apply RoPE
    let mut q = normalized.clone();
    apply_rope(&mut q, hidden_dim, num_heads, 0, 10000.0);

    // Compute attention (no history)
    let k = normalized.clone();
    let v = normalized.clone();

    let output = attention_with_cache(&q, &[], &[], &k, &v, num_heads);

    assert_eq!(output.len(), hidden_dim);
    // Output should equal v when no history and uniform attention
    for (out, v_val) in output.iter().zip(v.iter()) {
        assert!((out - v_val).abs() < 1e-5);
    }
}
