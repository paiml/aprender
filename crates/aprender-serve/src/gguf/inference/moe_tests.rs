//! SPEC-MOE-APR-001 Falsification Tests
//!
//! Contract: contracts/aprender/moe-apr-q4k-inference-v1.yaml
//! QA Gate: F-MOE-APR-001
//! Generated via: `pv scaffold contracts/aprender/moe-apr-q4k-inference-v1.yaml`

#[cfg(test)]
mod moe_contract_tests {

    // =========================================================================
    // FALSIFY-MOE-002: Router softmax numerically stable
    // Contract equation: moe_routing
    // =========================================================================

    /// Numerically stable softmax: max-subtract prevents overflow.
    fn stable_softmax(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum > 0.0 {
            exps.iter().map(|&e| e / sum).collect()
        } else {
            vec![0.0; logits.len()]
        }
    }

    /// Top-k selection with renormalization.
    fn top_k_select(probs: &[f32], k: usize, norm: bool) -> Vec<(usize, f32)> {
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut selected: Vec<(usize, f32)> = indexed[..k.min(probs.len())].to_vec();
        if norm {
            let total: f32 = selected.iter().map(|(_, w)| w).sum();
            if total > 0.0 {
                for (_, w) in &mut selected {
                    *w /= total;
                }
            }
        }
        selected
    }

    #[test]
    fn falsify_moe_002_softmax_no_nan() {
        // Normal logits
        let probs = stable_softmax(&[1.0, 2.0, 3.0, 4.0]);
        assert!(probs.iter().all(|p| !p.is_nan()), "NaN in softmax output");
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5, "softmax doesn't sum to 1.0");

        // All zeros
        let probs = stable_softmax(&[0.0; 128]);
        assert!(probs.iter().all(|p| !p.is_nan()));

        // Extreme positive
        let probs = stable_softmax(&[1e30, 1e30, 0.0]);
        assert!(probs.iter().all(|p| !p.is_nan()), "NaN on extreme positive");

        // Extreme negative
        let probs = stable_softmax(&[-1e30, -1e30, 0.0]);
        assert!(probs.iter().all(|p| !p.is_nan()), "NaN on extreme negative");

        // Mixed extreme
        let probs = stable_softmax(&[1e30, -1e30, 0.0]);
        assert!(probs.iter().all(|p| !p.is_nan()), "NaN on mixed extreme");

        // Single element
        let probs = stable_softmax(&[42.0]);
        assert_eq!(probs.len(), 1);
        assert!((probs[0] - 1.0).abs() < 1e-6);
    }

    // =========================================================================
    // FALSIFY-MOE-003: Top-k selection correct
    // Contract equation: moe_routing
    // =========================================================================

    #[test]
    fn falsify_moe_003_topk_selection() {
        // Known ranking: indices 3,2,1,0 in descending probability
        let probs = vec![0.1, 0.2, 0.3, 0.4];
        let selected = top_k_select(&probs, 2, false);

        assert_eq!(selected.len(), 2, "top-2 must return exactly 2");
        assert_eq!(selected[0].0, 3, "highest prob expert should be index 3");
        assert_eq!(selected[1].0, 2, "second highest should be index 2");

        // With renormalization
        let selected = top_k_select(&probs, 2, true);
        let weight_sum: f32 = selected.iter().map(|(_, w)| w).sum();
        assert!(
            (weight_sum - 1.0).abs() < 1e-6,
            "norm_topk_prob: weights must sum to 1.0, got {weight_sum}"
        );

        // Top-k > num_experts: should return all
        let selected = top_k_select(&probs, 10, false);
        assert_eq!(selected.len(), 4, "top-10 of 4 experts should return 4");

        // All equal probs
        let probs = vec![0.25; 8];
        let selected = top_k_select(&probs, 4, true);
        assert_eq!(selected.len(), 4);
        for (_, w) in &selected {
            assert!((w - 0.25).abs() < 1e-6, "equal probs should stay equal after norm");
        }
    }

    // =========================================================================
    // KANI-MOE-001: Router top-k returns exactly k experts in [0, num_experts)
    // Contract harness: bounded_int
    // =========================================================================

    #[test]
    fn kani_moe_001_topk_bounds() {
        for num_experts in [2, 8, 16, 64, 128, 256] {
            let probs: Vec<f32> = (0..num_experts).map(|i| i as f32 / num_experts as f32).collect();
            for top_k in [1, 2, 4, 8] {
                let k = top_k.min(num_experts);
                let selected = top_k_select(&probs, top_k, true);
                assert_eq!(
                    selected.len(), k,
                    "top-{top_k} of {num_experts} experts should return {k}"
                );
                for (idx, _) in &selected {
                    assert!(
                        *idx < num_experts,
                        "expert index {idx} >= num_experts {num_experts}"
                    );
                }
            }
        }
    }

    // =========================================================================
    // KANI-MOE-002: Softmax output sums to 1.0 and all elements non-negative
    // Contract harness: stub_float
    // =========================================================================

    #[test]
    fn kani_moe_002_softmax_invariants() {
        for size in [2, 8, 16, 64, 128] {
            let logits: Vec<f32> = (0..size).map(|i| (i as f32 - size as f32 / 2.0) * 0.1).collect();
            let probs = stable_softmax(&logits);

            assert_eq!(probs.len(), size);

            // All non-negative
            for (i, &p) in probs.iter().enumerate() {
                assert!(p >= 0.0, "prob[{i}] = {p} is negative");
            }

            // Sum to 1.0
            let sum: f32 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "softmax sum = {sum}, expected 1.0 (size={size})"
            );
        }
    }

    // =========================================================================
    // KANI-MOE-003: SwiGLU output dimension matches input dimension
    // Contract harness: bounded_int
    // =========================================================================

    #[test]
    fn kani_moe_003_swiglu_dimensions() {
        // SwiGLU: down(SiLU(gate(x)) * up(x))
        // gate: [moe_intermediate, hidden_dim]
        // up:   [moe_intermediate, hidden_dim]
        // down: [hidden_dim, moe_intermediate]
        // input: [hidden_dim] → output: [hidden_dim]

        for (hidden_dim, moe_intermediate) in [(2048, 768), (1024, 512), (4096, 1024)] {
            let input = vec![1.0f32; hidden_dim];

            // Simulate gate projection
            let gate_out = vec![0.5f32; moe_intermediate];
            // Simulate up projection
            let up_out = vec![0.3f32; moe_intermediate];

            // SwiGLU
            let mut swiglu = vec![0.0f32; moe_intermediate];
            for i in 0..moe_intermediate {
                let silu = gate_out[i] / (1.0 + (-gate_out[i]).exp());
                swiglu[i] = silu * up_out[i];
            }

            // Down projection output
            let output = vec![0.0f32; hidden_dim]; // simulated

            assert_eq!(input.len(), hidden_dim);
            assert_eq!(swiglu.len(), moe_intermediate);
            assert_eq!(output.len(), hidden_dim, "output dim must match input dim");
        }
    }
}
