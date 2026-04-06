// ============================================================================
// GQA (FALSIFY-GQ-001 through FALSIFY-GQ-006)
// ============================================================================

proptest! {
    /// FALSIFY-GQ-001
    /// Contract: gqa-kernel-v1.yaml
    /// Prediction: output bounded by V range (convex combination)
    /// Failure: output outside V value range
    #[test]
    fn falsify_gq_001_weight_normalization(
        seq_len in 2usize..=3,
        d_k in 2usize..=3,
        d_v in 2usize..=3,
        q_vals in proptest::collection::vec(-3.0f32..3.0, 1..=12),
        k_vals in proptest::collection::vec(-3.0f32..3.0, 1..=6),
        v_vals in proptest::collection::vec(-3.0f32..3.0, 1..=6),
    ) {
        let num_heads = 2usize;
        let num_kv_heads = 1usize;

        let q: Vec<f32> = q_vals.iter().copied().cycle().take(num_heads * seq_len * d_k).collect();
        let k: Vec<f32> = k_vals.iter().copied().cycle().take(num_kv_heads * seq_len * d_k).collect();
        let v: Vec<f32> = v_vals.iter().copied().cycle().take(num_kv_heads * seq_len * d_v).collect();
        let mut output = vec![0.0f32; num_heads * seq_len * d_v];

        gqa_scalar(&q, &k, &v, seq_len, d_k, d_v, num_heads, num_kv_heads, &mut output);

        let v_min = v.iter().copied().fold(f32::INFINITY, f32::min);
        let v_max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        for (idx, &val) in output.iter().enumerate() {
            prop_assert!(
                val >= v_min - 1e-5 && val <= v_max + 1e-5,
                "FALSIFY-GQ-001: output[{idx}] = {val} outside V range [{v_min}, {v_max}]"
            );
        }
    }

    /// FALSIFY-GQ-002
    /// Contract: gqa-kernel-v1.yaml
    /// Prediction: with num_heads=2, num_kv_heads=1, both heads use same KV but different Q
    /// Failure: KV broadcasting broken (heads produce identical output despite different Q)
    #[allow(clippy::cast_precision_loss)]
    #[test]
    fn falsify_gq_002_kv_head_broadcasting(
        seq_len in 2usize..=3,
        d_k in 2usize..=3,
        d_v in 2usize..=3,
    ) {
        let num_heads = 2usize;
        let num_kv_heads = 1usize;
        let head_stride = seq_len * d_v;

        // Generate Q with distinctly different values per head
        let q: Vec<f32> = (0..num_heads * seq_len * d_k)
            .map(|i| if i < seq_len * d_k { (i as f32) * 0.3 } else { (i as f32) * 0.7 + 1.0 })
            .collect();
        let k: Vec<f32> = (0..num_kv_heads * seq_len * d_k)
            .map(|i| (i as f32) * 0.2)
            .collect();
        let v: Vec<f32> = (0..num_kv_heads * seq_len * d_v)
            .map(|i| (i as f32) * 0.5 - 1.0)
            .collect();
        let mut output = vec![0.0f32; num_heads * seq_len * d_v];

        gqa_scalar(&q, &k, &v, seq_len, d_k, d_v, num_heads, num_kv_heads, &mut output);

        // Both heads should produce finite output
        common::assert_all_finite(&output);

        // Both heads use same KV but different Q, so outputs should generally differ
        let head0 = &output[0..head_stride];
        let head1 = &output[head_stride..2 * head_stride];
        let diff = common::l2_distance(head0, head1);
        prop_assert!(
            diff > 1e-7,
            "FALSIFY-GQ-002: heads produced identical output despite different Q, diff = {diff}"
        );
    }

    /// FALSIFY-GQ-003
    /// Contract: gqa-kernel-v1.yaml
    /// Prediction: when num_heads == num_kv_heads, GQA equals standard attention per head
    /// Failure: MHA equivalence broken
    #[allow(clippy::cast_precision_loss)]
    #[test]
    fn falsify_gq_003_mha_equivalence(
        seq_len in 2usize..=3,
        d_k in 2usize..=3,
        d_v in 2usize..=3,
    ) {
        let num_heads = 1usize;
        let num_kv_heads = 1usize;

        let q: Vec<f32> = (0..num_heads * seq_len * d_k)
            .map(|i| (i as f32) * 0.15)
            .collect();
        let k: Vec<f32> = (0..num_kv_heads * seq_len * d_k)
            .map(|i| (i as f32) * 0.2)
            .collect();
        let v: Vec<f32> = (0..num_kv_heads * seq_len * d_v)
            .map(|i| (i as f32) * 0.25)
            .collect();

        // GQA with num_heads == num_kv_heads == 1
        let mut gqa_out = vec![0.0f32; num_heads * seq_len * d_v];
        gqa_scalar(&q, &k, &v, seq_len, d_k, d_v, num_heads, num_kv_heads, &mut gqa_out);

        // Standard attention: q is seq_len x d_k, k is seq_len x d_k, v is seq_len x d_v
        let mut att_out = vec![0.0f32; seq_len * d_v];
        attention_scalar(&q, &k, &v, seq_len, seq_len, d_k, d_v, &mut att_out);

        let dist = common::l2_distance(&gqa_out, &att_out);
        prop_assert!(
            dist < 1e-5,
            "FALSIFY-GQ-003: GQA != standard attention, L2 distance = {dist}"
        );
    }

    /// FALSIFY-GQ-004
    /// Contract: gqa-kernel-v1.yaml
    /// Prediction: output values within per-column V range (convexity)
    /// Failure: convex combination property violated
    #[test]
    fn falsify_gq_004_convexity(
        seq_len in 2usize..=3,
        d_k in 2usize..=3,
        d_v in 2usize..=3,
        q_vals in proptest::collection::vec(-4.0f32..4.0, 1..=18),
        k_vals in proptest::collection::vec(-4.0f32..4.0, 1..=6),
        v_vals in proptest::collection::vec(-4.0f32..4.0, 1..=6),
    ) {
        let num_heads = 2usize;
        let num_kv_heads = 2usize;

        let q: Vec<f32> = q_vals.iter().copied().cycle().take(num_heads * seq_len * d_k).collect();
        let k: Vec<f32> = k_vals.iter().copied().cycle().take(num_kv_heads * seq_len * d_k).collect();
        let v: Vec<f32> = v_vals.iter().copied().cycle().take(num_kv_heads * seq_len * d_v).collect();
        let mut output = vec![0.0f32; num_heads * seq_len * d_v];

        gqa_scalar(&q, &k, &v, seq_len, d_k, d_v, num_heads, num_kv_heads, &mut output);

        // For each head, output is convex combination of V rows from the mapped KV head
        let heads_per_kv = num_heads / num_kv_heads;
        for h in 0..num_heads {
            let kv_head = h / heads_per_kv;
            let v_start = kv_head * seq_len * d_v;
            let v_slice = &v[v_start..v_start + seq_len * d_v];

            for j in 0..d_v {
                let v_col_min = (0..seq_len).map(|r| v_slice[r * d_v + j]).fold(f32::INFINITY, f32::min);
                let v_col_max = (0..seq_len).map(|r| v_slice[r * d_v + j]).fold(f32::NEG_INFINITY, f32::max);
                for i in 0..seq_len {
                    let val = output[h * seq_len * d_v + i * d_v + j];
                    prop_assert!(
                        val >= v_col_min - 1e-5 && val <= v_col_max + 1e-5,
                        "FALSIFY-GQ-004: head {h} output[{i},{j}] = {val} outside V column range [{v_col_min}, {v_col_max}]"
                    );
                }
            }
        }
    }

    /// FALSIFY-GQ-005
    /// Contract: gqa-kernel-v1.yaml
    /// Prediction: AVX2 output matches scalar within 8 ULP
    /// Failure: SIMD divergence exceeds tolerance
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn falsify_gq_005_simd_equivalence(
        seq_len in 2usize..=3,
        d_k in 2usize..=3,
        d_v in 2usize..=3,
        q_vals in proptest::collection::vec(-3.0f32..3.0, 1..=12),
        k_vals in proptest::collection::vec(-3.0f32..3.0, 1..=6),
        v_vals in proptest::collection::vec(-3.0f32..3.0, 1..=6),
    ) {
        if !is_x86_feature_detected!("avx2") {
            return Ok(());
        }
        let num_heads = 2usize;
        let num_kv_heads = 1usize;

        let q: Vec<f32> = q_vals.iter().copied().cycle().take(num_heads * seq_len * d_k).collect();
        let k: Vec<f32> = k_vals.iter().copied().cycle().take(num_kv_heads * seq_len * d_k).collect();
        let v: Vec<f32> = v_vals.iter().copied().cycle().take(num_kv_heads * seq_len * d_v).collect();

        let mut scalar_out = vec![0.0f32; num_heads * seq_len * d_v];
        let mut avx2_out = vec![0.0f32; num_heads * seq_len * d_v];

        gqa_scalar(&q, &k, &v, seq_len, d_k, d_v, num_heads, num_kv_heads, &mut scalar_out);
        unsafe { gqa_avx2(&q, &k, &v, seq_len, d_k, d_v, num_heads, num_kv_heads, &mut avx2_out) };

        let ulp = common::max_ulp_distance(&scalar_out, &avx2_out);
        prop_assert!(
            ulp <= 8,
            "FALSIFY-GQ-005: SIMD divergence = {ulp} ULP (max 8)"
        );
    }
}

/// FALSIFY-GQ-006
/// Contract: gqa-kernel-v1.yaml
/// Prediction: `num_heads` must be divisible by `num_kv_heads`; valid combos work
/// Failure: valid head divisibility combinations rejected
#[test]
fn falsify_gq_006_head_divisibility() {
    let valid_combos: &[(usize, usize)] = &[(2, 1), (4, 2), (4, 1), (1, 1), (6, 3)];
    let seq_len = 2;
    let d_k = 2;
    let d_v = 2;

    for &(num_heads, num_kv_heads) in valid_combos {
        let q = vec![0.5f32; num_heads * seq_len * d_k];
        let k = vec![0.5f32; num_kv_heads * seq_len * d_k];
        let v = vec![1.0f32; num_kv_heads * seq_len * d_v];
        let mut output = vec![0.0f32; num_heads * seq_len * d_v];

        // Should not panic
        gqa_scalar(
            &q,
            &k,
            &v,
            seq_len,
            d_k,
            d_v,
            num_heads,
            num_kv_heads,
            &mut output,
        );

        common::assert_all_finite(&output);

        // V is all 1.0 and attention weights sum to 1 per row,
        // so each output element should be 1.0
        for (idx, &val) in output.iter().enumerate() {
            assert!(
                (val - 1.0).abs() < 1e-5,
                "FALSIFY-GQ-006: output[{idx}] = {val} (expected 1.0) for heads={num_heads}, kv_heads={num_kv_heads}"
            );
        }
    }
}

// ============================================================================
// Flash Attention (FALSIFY-FA-001 through FALSIFY-FA-004)
// ============================================================================

proptest! {
    /// FALSIFY-FA-001
    /// Contract: flash-attention-v1.yaml
    /// Prediction: flash_attention matches standard attention for small inputs
    /// Failure: online softmax produces different result than standard softmax
    #[test]
    fn falsify_fa_001_matches_standard(
        n in 2usize..=4,
        d in 2usize..=4,
        q_vals in proptest::collection::vec(-3.0f32..3.0, 1..=16),
        k_vals in proptest::collection::vec(-3.0f32..3.0, 1..=16),
        v_vals in proptest::collection::vec(-3.0f32..3.0, 1..=16),
    ) {
        let q: Vec<f32> = q_vals.iter().copied().cycle().take(n * d).collect();
        let k: Vec<f32> = k_vals.iter().copied().cycle().take(n * d).collect();
        let v: Vec<f32> = v_vals.iter().copied().cycle().take(n * d).collect();

        // Flash attention with tile_size = n (single tile)
        let mut flash_out = vec![0.0f32; n * d];
        flash_attention_scalar(&q, &k, &v, n, d, n, &mut flash_out);

        // Standard attention (self-attention: n queries, n keys)
        let mut expected = vec![0.0f32; n * d];
        attention_scalar(&q, &k, &v, n, n, d, d, &mut expected);

        let dist = common::l2_distance(&flash_out, &expected);
        prop_assert!(
            dist < 1e-4,
            "FALSIFY-FA-001: flash vs standard attention L2 distance = {dist}"
        );
    }

    /// FALSIFY-FA-002
    /// Contract: flash-attention-v1.yaml
    /// Prediction: output is finite for moderate inputs (online softmax does not overflow)
    /// Failure: NaN or Inf in output
    #[test]
    fn falsify_fa_002_online_softmax(
        n in 2usize..=6,
        d in 2usize..=4,
        tile_size in 1usize..=4,
        q_vals in proptest::collection::vec(-5.0f32..5.0, 1..=24),
        k_vals in proptest::collection::vec(-5.0f32..5.0, 1..=24),
        v_vals in proptest::collection::vec(-5.0f32..5.0, 1..=24),
    ) {
        let q: Vec<f32> = q_vals.iter().copied().cycle().take(n * d).collect();
        let k: Vec<f32> = k_vals.iter().copied().cycle().take(n * d).collect();
        let v: Vec<f32> = v_vals.iter().copied().cycle().take(n * d).collect();
        let mut output = vec![0.0f32; n * d];

        flash_attention_scalar(&q, &k, &v, n, d, tile_size, &mut output);

        for (idx, &val) in output.iter().enumerate() {
            prop_assert!(
                val.is_finite(),
                "FALSIFY-FA-002: output[{idx}] = {val} is not finite (tile_size={tile_size})"
            );
        }
    }

    /// FALSIFY-FA-003
    /// Contract: flash-attention-v1.yaml
    /// Prediction: tile_size < n produces same output as single-tile (tile_size = n)
    /// Failure: tiled computation diverges from single-tile computation
    #[test]
    fn falsify_fa_003_tile_coverage(
        d in 2usize..=4,
        q_vals in proptest::collection::vec(-3.0f32..3.0, 1..=32),
        k_vals in proptest::collection::vec(-3.0f32..3.0, 1..=32),
        v_vals in proptest::collection::vec(-3.0f32..3.0, 1..=32),
    ) {
        let n = 8usize;
        let tile_small = 4usize;
        let tile_full = n;

        let q: Vec<f32> = q_vals.iter().copied().cycle().take(n * d).collect();
        let k: Vec<f32> = k_vals.iter().copied().cycle().take(n * d).collect();
        let v: Vec<f32> = v_vals.iter().copied().cycle().take(n * d).collect();

        let mut out_small = vec![0.0f32; n * d];
        let mut out_full = vec![0.0f32; n * d];

        flash_attention_scalar(&q, &k, &v, n, d, tile_small, &mut out_small);
        flash_attention_scalar(&q, &k, &v, n, d, tile_full, &mut out_full);

        let dist = common::l2_distance(&out_small, &out_full);
        prop_assert!(
            dist < 1e-4,
            "FALSIFY-FA-003: tile_size={tile_small} vs tile_size={tile_full} L2 distance = {dist}"
        );
    }

    /// FALSIFY-FA-004
    /// Contract: flash-attention-v1.yaml
    /// Prediction: output bounded by V range (convex combination of V rows)
    /// Failure: output outside V value range
    #[test]
    fn falsify_fa_004_output_conservation(
        n in 4usize..=8,
        d in 2usize..=4,
        tile_size in 2usize..=4,
        q_vals in proptest::collection::vec(-5.0f32..5.0, 1..=32),
        k_vals in proptest::collection::vec(-5.0f32..5.0, 1..=32),
        v_vals in proptest::collection::vec(-5.0f32..5.0, 1..=32),
    ) {
        let q: Vec<f32> = q_vals.iter().copied().cycle().take(n * d).collect();
        let k: Vec<f32> = k_vals.iter().copied().cycle().take(n * d).collect();
        let v: Vec<f32> = v_vals.iter().copied().cycle().take(n * d).collect();
        let mut output = vec![0.0f32; n * d];

        flash_attention_scalar(&q, &k, &v, n, d, tile_size, &mut output);

        // Per-column convexity: output in each column bounded by V column range
        for j in 0..d {
            let v_col_min = (0..n).map(|r| v[r * d + j]).fold(f32::INFINITY, f32::min);
            let v_col_max = (0..n).map(|r| v[r * d + j]).fold(f32::NEG_INFINITY, f32::max);
            for i in 0..n {
                let val = output[i * d + j];
                prop_assert!(
                    val >= v_col_min - 1e-5 && val <= v_col_max + 1e-5,
                    "FALSIFY-FA-004: output[{i},{j}] = {val} outside V column range [{v_col_min}, {v_col_max}]"
                );
            }
        }
    }
}
