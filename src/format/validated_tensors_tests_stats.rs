
    #[test]
    fn tensor_stats_compute_mixed_nan_inf_valid() {
        let stats = TensorStats::compute(&[1.0, f32::NAN, f32::INFINITY, 2.0, f32::NEG_INFINITY]);
        assert_eq!(stats.len, 5);
        assert_eq!(stats.nan_count, 1);
        assert_eq!(stats.inf_count, 2);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 2.0);
    }

    #[test]
    fn tensor_stats_zero_pct_empty() {
        let stats = TensorStats::compute(&[]);
        assert_eq!(stats.zero_pct(), 0.0);
    }

    #[test]
    fn tensor_stats_zero_pct_with_zeros() {
        let stats = TensorStats::compute(&[0.0, 0.0, 1.0, 2.0]);
        // 2 out of 4 are near-zero = 50%
        assert!((stats.zero_pct() - 50.0).abs() < 0.01);
    }

    // ================================================================
    // ContractValidationError Display and Error trait
    // ================================================================

    #[test]
    fn contract_validation_error_display_format() {
        let err = ContractValidationError {
            tensor_name: "embedding".to_string(),
            rule_id: "F-DATA-QUALITY-001".to_string(),
            message: "DENSITY FAILURE: 94.5% zeros".to_string(),
        };
        let display = format!("{err}");
        assert_eq!(
            display,
            "[F-DATA-QUALITY-001] Tensor 'embedding': DENSITY FAILURE: 94.5% zeros"
        );
    }

    #[test]
    fn contract_validation_error_implements_std_error() {
        let err = ContractValidationError {
            tensor_name: "weight".to_string(),
            rule_id: "F-LAYOUT-CONTRACT-001".to_string(),
            message: "Shape mismatch".to_string(),
        };
        // Verify it implements std::error::Error
        let std_err: &dyn std::error::Error = &err;
        let display_via_error = format!("{std_err}");
        assert!(display_via_error.contains("Shape mismatch"));
        // source() should return None (no wrapped error)
        assert!(std_err.source().is_none());
    }

    #[test]
    fn contract_validation_error_clone() {
        let err = ContractValidationError {
            tensor_name: "test".to_string(),
            rule_id: "F-001".to_string(),
            message: "fail".to_string(),
        };
        let cloned = err.clone();
        assert_eq!(cloned.tensor_name, err.tensor_name);
        assert_eq!(cloned.rule_id, err.rule_id);
        assert_eq!(cloned.message, err.message);
    }

    #[test]
    fn contract_validation_error_debug() {
        let err = ContractValidationError {
            tensor_name: "test".to_string(),
            rule_id: "F-001".to_string(),
            message: "fail".to_string(),
        };
        let debug_str = format!("{err:?}");
        assert!(debug_str.contains("ContractValidationError"));
        assert!(debug_str.contains("test"));
    }

    // ================================================================
    // FALSIFY: §2.1.1 Embedding Contract — Five-Whys Gap Analysis
    //
    // These tests target gaps identified during systematic falsification:
    //
    // GAP-E1: vocab_size varies across model sizes (151936 vs 152064)
    //         → Code that hardcodes one value silently corrupts the other
    // GAP-E2: OOB token_id behavior diverges between aprender/realizar
    //         → aprender skips, realizar zero-fills — neither errors
    // GAP-E3: Zero-dimension embeddings (vocab=0 or hidden=0)
    //         → Should be rejected, not silently accepted as empty
    // GAP-E4: Embedding shape contract is per-family, not universal
    //         → 152064 (7B+) vs 151936 (≤3B) for Qwen2 alone
    //
    // Root cause (Five Whys):
    //   Why 1: Why could embedding lookup still OOB? → token_id ≥ vocab_size
    //   Why 2: Why isn't this caught? → ValidatedEmbedding validates DATA, not ACCESS
    //   Why 3: Why no access validation? → Lookup happens in forward(), not in constructor
    //   Why 4: Why not in constructor? → Constructor validates the tensor, not the tokenizer
    //   Why 5: Why not validate tokenizer↔embedding agreement? → No cross-domain contract
    //
    // Popper, K. (1959): "These tests attempt to falsify the claim that
    // ValidatedEmbedding prevents ALL embedding-related garbage output."
    // ================================================================

    /// FALSIFY-E1: Zero vocab_size must be rejected
    ///
    /// If vocab_size=0, the embedding has 0 elements. TensorStats::compute
    /// on empty data returns zeroed stats. The density gate checks
    /// `zero_pct() > 50.0`, but zero_pct() on empty data returns 0.0.
    /// This means an empty embedding would PASS all statistical gates.
    #[test]
    fn falsify_e1_zero_vocab_size_rejected() {
        let data: Vec<f32> = vec![];
        let result = ValidatedEmbedding::new(data, 0, 64);
        // vocab_size=0 produces expected_len=0, data.len()=0, so shape check passes.
        // But an empty embedding is semantically useless — no token can be looked up.
        // Current behavior: L2 norm = 0.0 < 1e-6 → rejected by Gate 5.
        assert!(result.is_err(),
            "FALSIFY-E1: vocab_size=0 must be rejected — empty embedding cannot serve inference");
    }

    /// FALSIFY-E1b: Zero hidden_dim must be rejected
    #[test]
    fn falsify_e1b_zero_hidden_dim_rejected() {
        let data: Vec<f32> = vec![];
        let result = ValidatedEmbedding::new(data, 100, 0);
        assert!(result.is_err(),
            "FALSIFY-E1b: hidden_dim=0 must be rejected — zero-width embeddings are meaningless");
    }

    /// FALSIFY-E2: Qwen2 vocab_size varies by model size — contract must handle both
    ///
    /// Qwen2 ≤3B: vocab_size=151936 (from contracts/model-families/qwen2.yaml)
    /// Qwen2 7B+: vocab_size=152064
    ///
    /// If code assumes one value for all sizes, 128 tokens are either:
    /// - Missing (OOB access) for ≤3B model with 7B tokenizer
    /// - Wasted (unused rows) for 7B model with ≤3B tokenizer
    #[test]
    fn falsify_e2_qwen2_vocab_size_divergence_acknowledged() {
        let small_vocab = 151_936_usize; // Qwen2 0.5B-3B
        let large_vocab = 152_064_usize; // Qwen2 7B-32B

        // The contract must NOT assume these are equal
        assert_ne!(small_vocab, large_vocab,
            "FALSIFY-E2: Qwen2 vocab sizes differ between model sizes — this is by design");

        // Both must be valid embedding sizes (shape check)
        let hidden = 64; // small hidden for test speed
        let small_data: Vec<f32> = (0..small_vocab * hidden)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        let large_data: Vec<f32> = (0..large_vocab * hidden)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();

        assert!(ValidatedEmbedding::new(small_data.clone(), small_vocab, hidden).is_ok(),
            "FALSIFY-E2: 151936 embedding must be valid");
        assert!(ValidatedEmbedding::new(large_data.clone(), large_vocab, hidden).is_ok(),
            "FALSIFY-E2: 152064 embedding must be valid");

        // Using wrong vocab_size for the data MUST fail
        assert!(ValidatedEmbedding::new(small_data, large_vocab, hidden).is_err(),
            "FALSIFY-E2: 151936 elements with vocab=152064 must fail shape check");
        assert!(ValidatedEmbedding::new(large_data, small_vocab, hidden).is_err(),
            "FALSIFY-E2: 152064 elements with vocab=151936 must fail shape check");
    }

    /// FALSIFY-E3: enforce_embedding_contract panics on mismatched dimensions
    ///
    /// The runtime assert in layout_contract_enforce.rs must fire when
    /// embedding length doesn't match vocab*hidden. This is the last line
    /// of defense before garbage inference.
    #[test]
    #[should_panic(expected = "CONTRACT VIOLATION")]
    fn falsify_e3_enforce_embedding_contract_catches_mismatch() {
        use crate::format::layout_contract::enforce_embedding_contract;
        // Simulate loading a 7B embedding (152064) but telling the system it's 0.5B (151936)
        let actual_len = 152_064 * 896;
        enforce_embedding_contract(actual_len, 151_936, 896);
    }

    /// FALSIFY-E4: Spot check percentiles cover the critical offset bug region
    ///
    /// PMAT-234 had 94.5% leading zeros. The spot check at 10% catches this.
    /// But what about a model where ONLY the last 10% is corrupted?
    /// The 90% spot check should catch that.
    #[test]
    fn falsify_e4_spot_check_catches_trailing_corruption() {
        let vocab_size = 100;
        let hidden_dim = 64;
        let mut data: Vec<f32> = (0..vocab_size * hidden_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();

        // Zero out token at 90% of vocab (token 90)
        let token_90_start = 90 * hidden_dim;
        for v in &mut data[token_90_start..token_90_start + hidden_dim] {
            *v = 0.0;
        }

        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_err(),
            "FALSIFY-E4: Zero token at 90% of vocab must be caught by spot check");
        assert_eq!(result.unwrap_err().rule_id, "F-DATA-QUALITY-004",
            "FALSIFY-E4: Should be caught by spot check rule, not density rule");
    }

    /// FALSIFY-E5: Embedding with exactly 50% zeros is borderline — must be deterministic
    ///
    /// The contract says ">50% zeros" fails. Exactly 50.0% should PASS the density gate.
    /// We scatter zeros uniformly (every other element) to avoid triggering the
    /// spot check gate which checks specific token rows at 10/50/90%.
    #[test]
    fn falsify_e5_exactly_50_pct_zeros_accepted() {
        let vocab_size = 100;
        let hidden_dim = 64;
        let total = vocab_size * hidden_dim;

        // Start with good data, then zero out every other element.
        // This gives exactly 50% zeros distributed uniformly, so no
        // single token row is all-zero (each row has ~32 zeros and ~32 non-zeros).
        let mut data: Vec<f32> = (0..total)
            .map(|i| (i as f32 * 0.01).sin() * 0.1 + 0.05)
            .collect();
        for i in (0..total).step_by(2) {
            data[i] = 0.0;
        }

        // Verify we actually have ~50% zeros (within 1% due to the abs<1e-10 threshold)
        let zero_count = data.iter().filter(|v| v.abs() < 1e-10).count();
        let zero_pct = 100.0 * zero_count as f64 / total as f64;
        assert!((zero_pct - 50.0).abs() < 1.0,
            "Test setup error: expected ~50% zeros, got {zero_pct:.1}%");

        // Must pass: 50.0% is NOT > 50.0%
        let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
        assert!(result.is_ok(),
            "FALSIFY-E5: Exactly 50% zeros should PASS (threshold is >50%, not >=50%): {:?}",
            result.err());
    }

    // =========================================================================
    // FALSIFY-L: §2.1.2 LM Head Contract — Five-Whys Gap Analysis
    //
    // Contract: tensor-layout-v1.yaml §tensors.lm_head
    //   apr_shape: "[vocab, hidden]"
    //   kernel: "matmul_q*k_rowmajor(W, x, vocab_size, hidden_dim)"
    //   validation: "shape[0] == vocab_size AND shape[1] == hidden_dim"
    //   critical: "true"
    //   note: "GH-202 root cause - wrong shape caused [PAD] garbage output"
    //
    // Five-Whys:
    //   Why 1: lm_head matmul could produce wrong logits
    //   Why 2: Weight shape [hidden, vocab] instead of [vocab, hidden]
    //   Why 3: enforce_matmul_contract not called at every matmul site
    //   Why 4: Contract enforcement is opt-in, not mandatory-by-type
    //   Why 5: ValidatedWeight exists but GGUF path doesn't use it
    //
    // Popper (1959): "These tests attempt to falsify the claim that
    // the lm_head contract prevents ALL garbage logit output."
    // =========================================================================

    /// FALSIFY-L1: enforce_matmul_contract catches swapped lm_head dims
    ///
    /// GH-202 root cause: lm_head [hidden, vocab] instead of [vocab, hidden].
    /// enforce_matmul_contract MUST panic on this exact scenario.
    #[test]
    #[should_panic(expected = "CONTRACT VIOLATION")]
    fn falsify_l1_matmul_contract_catches_gh202_root_cause() {
        use crate::format::layout_contract::enforce_matmul_contract;
        // GH-202 scenario: shape is [hidden=896, vocab=151936] (wrong order)
        enforce_matmul_contract("lm_head.weight", &[896, 151_936], 151_936, 896);
    }

    /// FALSIFY-L2: enforce_matmul_contract accepts correct lm_head shape
    #[test]
    fn falsify_l2_matmul_contract_accepts_correct_shape() {
        use crate::format::layout_contract::enforce_matmul_contract;
        // Correct: [vocab=151936, hidden=896]
        enforce_matmul_contract("lm_head.weight", &[151_936, 896], 151_936, 896);
    }

    /// FALSIFY-L3: enforce_matmul_contract rejects 1D lm_head
    ///
    /// A flattened weight vector [vocab*hidden] should not pass the 2D check.
    #[test]
    #[should_panic(expected = "CONTRACT VIOLATION")]
    fn falsify_l3_matmul_contract_rejects_1d_weight() {
        use crate::format::layout_contract::enforce_matmul_contract;
        // 1D shape — should panic
        enforce_matmul_contract("lm_head.weight", &[151_936 * 896], 151_936, 896);
    }

    /// FALSIFY-L4: ValidatedWeight rejects lm_head with wrong shape
    ///
    /// ValidatedWeight::new() is the Poka-Yoke gate for weight tensors.
    /// It must reject data whose length != out_dim * in_dim.
    #[test]
    fn falsify_l4_validated_weight_rejects_wrong_shape() {
        // Data for [vocab=100, hidden=64] = 6400 elements
        // But claim it's [vocab=152064, hidden=896] — shape mismatch
        let data: Vec<f32> = (0..6400).map(|i| (i as f32 * 0.01).sin() * 0.1).collect();
        let result = ValidatedWeight::new(data, 152_064, 896, "lm_head.weight");
        assert!(result.is_err(),
            "FALSIFY-L4: ValidatedWeight must reject data whose length != out_dim * in_dim");
        assert_eq!(result.unwrap_err().rule_id, "F-LAYOUT-CONTRACT-001",
            "FALSIFY-L4: Should cite shape mismatch rule");
    }

    /// FALSIFY-L5: ValidatedWeight rejects all-NaN lm_head
    ///
    /// If lm_head is all NaN, every logit would be NaN → argmax returns 0 → [PAD].
    /// ValidatedWeight MUST catch this.
    #[test]
    fn falsify_l5_validated_weight_rejects_nan_lm_head() {
        let data = vec![f32::NAN; 100 * 64];
        let result = ValidatedWeight::new(data, 100, 64, "lm_head.weight");
        assert!(result.is_err(),
            "FALSIFY-L5: All-NaN lm_head must be rejected — produces [PAD] garbage");
    }

    /// FALSIFY-L6: Qwen2 vocab_size divergence affects lm_head too
    ///
    /// Same as FALSIFY-E2 but for lm_head: wrong vocab_size → matmul OOB.
    /// ValidatedWeight must reject when data length doesn't match declared dims.
    #[test]
    fn falsify_l6_qwen2_vocab_divergence_in_lm_head() {
        let small_vocab = 151_936_usize;
        let large_vocab = 152_064_usize;
        let hidden = 64;

        // Data for small vocab
        let small_data: Vec<f32> = (0..small_vocab * hidden)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();

        // Correct vocab → ok
        assert!(ValidatedWeight::new(small_data.clone(), small_vocab, hidden, "lm_head").is_ok(),
            "FALSIFY-L6: 151936 lm_head must be valid with correct vocab_size");

        // Wrong vocab → must fail shape check
        assert!(ValidatedWeight::new(small_data, large_vocab, hidden, "lm_head").is_err(),
            "FALSIFY-L6: 151936 elements with vocab=152064 must fail shape check");
    }

    /// FALSIFY-L7: enforce_load_contract catches swapped lm_head at APR load time
    ///
    /// enforce_load_contract is the last line of defense when loading from APR.
    /// lm_head is the ONLY critical tensor — its shape MUST be checked.
    #[test]
    fn falsify_l7_load_contract_rejects_swapped_lm_head() {
        use crate::format::layout_contract::enforce_load_contract;

        // Correct shape → ok
        let ok = enforce_load_contract("lm_head.weight", &[151_936, 896], 151_936, 896);
        assert!(ok.is_ok(), "FALSIFY-L7: Valid lm_head shape must pass load contract");

        // Swapped shape → must fail (GH-202 regression)
        let err = enforce_load_contract("lm_head.weight", &[896, 151_936], 151_936, 896);
        assert!(err.is_err(),
            "FALSIFY-L7: Swapped lm_head shape must fail load contract — GH-202 regression guard");
    }

    /// FALSIFY-L8: Q6K/Q4K byte size calculation matches contract formula
    ///
    /// tensor-layout-v1.yaml F-LAYOUT-CONTRACT-004:
    ///   Q6K: 210 bytes per 256-element super-block
    ///   Q4K: 144 bytes per 256-element super-block
    /// If these calculations are wrong, dequantized matmul reads past buffer → UB.
    #[test]
    fn falsify_l8_quantized_byte_size_for_lm_head() {
        use crate::format::layout_contract::LayoutContract;

        // Qwen2 0.5B: vocab=151936, hidden=896
        let q6k_bytes = LayoutContract::calculate_q6k_bytes(151_936, 896);
        let q4k_bytes = LayoutContract::calculate_q4k_bytes(151_936, 896);

        // ceil(896/256) = 4 super-blocks per row
        let sb_per_row = (896 + 255) / 256; // = 4
        assert_eq!(sb_per_row, 4);

        let expected_q6k = 151_936 * sb_per_row * 210;
        let expected_q4k = 151_936 * sb_per_row * 144;

        assert_eq!(q6k_bytes, expected_q6k,
            "FALSIFY-L8: Q6K byte size for lm_head must be vocab * ceil(hidden/256) * 210");
        assert_eq!(q4k_bytes, expected_q4k,
            "FALSIFY-L8: Q4K byte size for lm_head must be vocab * ceil(hidden/256) * 144");

        // Qwen2 7B: vocab=152064, hidden=3584
        let q6k_7b = LayoutContract::calculate_q6k_bytes(152_064, 3584);
        let sb_7b = (3584 + 255) / 256; // = 14
        assert_eq!(sb_7b, 14);
        assert_eq!(q6k_7b, 152_064 * 14 * 210,
            "FALSIFY-L8: Q6K byte size for 7B lm_head must match formula");
    }

    // =========================================================================
    // FALSIFY-A: §2.1.3 Attention Projections — Five-Whys Gap Analysis
    //
    // Contract: tensor-layout-v1.yaml §tensors.q_proj/k_proj/v_proj/o_proj
    //   q_proj: [num_heads*head_dim, hidden]
    //   k_proj: [num_kv_heads*head_dim, hidden]  ← different for GQA!
    //   v_proj: [num_kv_heads*head_dim, hidden]  ← different for GQA!
    //   o_proj: [hidden, num_heads*head_dim]      ← reversed from q_proj!
    //
    // Five-Whys:
    //   Why 1: Attention could produce wrong results
    //   Why 2: GQA dimensions (kv_heads vs heads) confused
    //   Why 3: ValidatedWeight has no semantic context (Q vs K vs V)
    //   Why 4: Generic out_dim/in_dim — doesn't know it's attention
    //   Why 5: No per-tensor-role type system exists
    //
    // Popper (1959): "These tests attempt to falsify the claim that
    // the attention projection contracts prevent incorrect matmul shapes."
    // =========================================================================

    /// FALSIFY-A1: ValidatedWeight accepts correct GQA K/V dimensions
    ///
    /// For Qwen2 0.5B: num_heads=14, num_kv_heads=2, head_dim=64, hidden=896.
    /// K/V shape: [128, 896] (2 kv_heads × 64 head_dim = 128 out_dim).
    /// Q shape: [896, 896] (14 heads × 64 head_dim = 896 out_dim).
    #[test]
    fn falsify_a1_validated_weight_accepts_gqa_k_v_shapes() {
        let hidden = 896_usize;
        let num_heads = 14_usize;
        let num_kv_heads = 2_usize;
        let head_dim = 64_usize;

        let q_dim = num_heads * head_dim;    // 896
        let kv_dim = num_kv_heads * head_dim; // 128

        // Q: [q_dim, hidden] = [896, 896]
        let q_data: Vec<f32> = (0..q_dim * hidden)
            .map(|i| (i as f32 * 0.001).sin() * 0.1).collect();
        assert!(ValidatedWeight::new(q_data, q_dim, hidden, "q_proj").is_ok(),
            "FALSIFY-A1: q_proj [896, 896] must be valid for Qwen2 0.5B");

        // K: [kv_dim, hidden] = [128, 896]
        let k_data: Vec<f32> = (0..kv_dim * hidden)
            .map(|i| (i as f32 * 0.001).sin() * 0.1).collect();
        assert!(ValidatedWeight::new(k_data, kv_dim, hidden, "k_proj").is_ok(),
            "FALSIFY-A1: k_proj [128, 896] must be valid for Qwen2 0.5B GQA");

        // V: [kv_dim, hidden] = [128, 896]
        let v_data: Vec<f32> = (0..kv_dim * hidden)
            .map(|i| (i as f32 * 0.001).sin() * 0.1).collect();
        assert!(ValidatedWeight::new(v_data, kv_dim, hidden, "v_proj").is_ok(),
            "FALSIFY-A1: v_proj [128, 896] must be valid for Qwen2 0.5B GQA");
    }

    /// FALSIFY-A2: ValidatedWeight rejects K/V with wrong GQA dimensions
    ///
    /// If someone passes num_heads instead of num_kv_heads for K/V,
    /// data length won't match declared dims → shape check fires.
    #[test]
    fn falsify_a2_validated_weight_rejects_wrong_gqa_dims() {
        let hidden = 896_usize;
        let num_kv_heads = 2_usize;
        let head_dim = 64_usize;
        let kv_dim = num_kv_heads * head_dim; // 128 — correct

        // Data for correct K shape [128, 896]
        let k_data: Vec<f32> = (0..kv_dim * hidden)
            .map(|i| (i as f32 * 0.001).sin() * 0.1).collect();

        // Declare wrong out_dim (896 instead of 128) — shape mismatch
        let result = ValidatedWeight::new(k_data, 896, hidden, "k_proj");
        assert!(result.is_err(),
            "FALSIFY-A2: K data with 128*896 elements must fail when out_dim=896 declared");
        assert_eq!(result.unwrap_err().rule_id, "F-LAYOUT-CONTRACT-001");
    }

    /// FALSIFY-A3: o_proj has reversed dimensions from q_proj
    ///
    /// Contract: q_proj is [heads*head_dim, hidden], o_proj is [hidden, heads*head_dim].
    /// If you accidentally use q_proj order for o_proj, matmul is wrong.
    #[test]
    fn falsify_a3_o_proj_reversed_dims_from_q_proj() {
        use crate::format::layout_contract::enforce_matmul_contract;

        let hidden = 896_usize;
        let q_dim = 896_usize; // num_heads * head_dim

        // q_proj: out=q_dim, in=hidden → [q_dim, hidden]
        enforce_matmul_contract("q_proj.weight", &[q_dim, hidden], q_dim, hidden);

        // o_proj: out=hidden, in=q_dim → [hidden, q_dim] (REVERSED)
        enforce_matmul_contract("o_proj.weight", &[hidden, q_dim], hidden, q_dim);
    }

    /// FALSIFY-A3b: For MHA (q_dim == hidden), o_proj shape is indistinguishable
    ///
    /// This documents a contract limitation: when num_heads * head_dim == hidden_dim
    /// (true for all standard MHA), o_proj and q_proj have identical shapes.
    /// The dimension reversal in the contract only matters for non-standard architectures.
    #[test]
    fn falsify_a3b_mha_o_proj_shape_equals_q_proj() {
        use crate::format::layout_contract::enforce_matmul_contract;

        let hidden = 896_usize;
        let q_dim = 896_usize; // num_heads * head_dim == hidden for MHA

        // For MHA: o_proj [hidden, q_dim] == [896, 896] == q_proj [q_dim, hidden]
        // Both pass because out_dim == in_dim == 896
        enforce_matmul_contract("o_proj.weight", &[hidden, q_dim], hidden, q_dim);
        enforce_matmul_contract("q_proj.weight", &[q_dim, hidden], q_dim, hidden);

        // Document: this is NOT a bug — MHA o_proj and q_proj are genuinely the same shape.
        // The contract's dimension reversal only manifests in non-standard architectures
        // where num_heads * head_dim != hidden_dim.
        assert_eq!(q_dim, hidden,
            "FALSIFY-A3b: For standard MHA, q_dim == hidden — shapes are symmetric");
    }

    /// FALSIFY-A4: enforce_import_contract handles per-layer attention names
    ///
    /// Attention projections use blk.{n}.attn_*.weight naming.
    /// The contract must correctly pattern-match these.
    #[test]
    fn falsify_a4_import_contract_handles_layer_patterns() {
        use crate::format::layout_contract::enforce_import_contract;

        let hidden = 896_usize;
        let q_dim = 896_usize;
        let kv_dim = 128_usize;

        // q_proj: GGUF [hidden, q_dim] → APR [q_dim, hidden]
        let (shape, needs_t) = enforce_import_contract(
            "blk.0.attn_q.weight", &[hidden, q_dim], 151_936, hidden);
        assert_eq!(shape, vec![q_dim, hidden],
            "FALSIFY-A4: q_proj shape must be [q_dim, hidden] after import");
        assert!(!needs_t);

        // k_proj: GGUF [hidden, kv_dim] → APR [kv_dim, hidden]
        let (shape, needs_t) = enforce_import_contract(
            "blk.0.attn_k.weight", &[hidden, kv_dim], 151_936, hidden);
        assert_eq!(shape, vec![kv_dim, hidden],
            "FALSIFY-A4: k_proj shape must be [kv_dim, hidden] after import");
        assert!(!needs_t);

        // o_proj: GGUF [q_dim, hidden] → APR [hidden, q_dim]
        let (shape, needs_t) = enforce_import_contract(
            "blk.0.attn_output.weight", &[q_dim, hidden], 151_936, hidden);
        assert_eq!(shape, vec![hidden, q_dim],
            "FALSIFY-A4: o_proj shape must be [hidden, q_dim] after import");
        assert!(!needs_t);

        // Layer 27 — verify pattern matching works for high indices
        let (shape, _) = enforce_import_contract(
            "blk.27.attn_v.weight", &[hidden, kv_dim], 151_936, hidden);
        assert_eq!(shape, vec![kv_dim, hidden],
            "FALSIFY-A4: v_proj layer 27 must match pattern");
    }

    /// FALSIFY-A5: ValidatedWeight rejects all-zero attention projection
    ///
    /// An all-zero Q projection would make all attention scores identical,
    /// producing uniform attention → garbage output.
    #[test]
    fn falsify_a5_validated_weight_rejects_zero_attention_weight() {
        let hidden = 64_usize;
        let q_dim = 64_usize;
        let data = vec![0.0f32; q_dim * hidden];

        let result = ValidatedWeight::new(data, q_dim, hidden, "q_proj.weight");
        assert!(result.is_err(),
            "FALSIFY-A5: All-zero q_proj must be rejected — produces uniform attention");
    }

    /// FALSIFY-A6: ValidatedWeight rejects NaN in K projection
    ///
    /// NaN in K → NaN attention scores → NaN softmax → NaN output.
    #[test]
    fn falsify_a6_validated_weight_rejects_nan_k_proj() {
        let hidden = 64_usize;
        let kv_dim = 16_usize;
        let mut data: Vec<f32> = (0..kv_dim * hidden)
            .map(|i| (i as f32 * 0.001).sin() * 0.1).collect();
        data[42] = f32::NAN;

        let result = ValidatedWeight::new(data, kv_dim, hidden, "k_proj.weight");
        assert!(result.is_err(),
            "FALSIFY-A6: NaN in k_proj must be rejected — produces NaN attention");
    }

    // =========================================================================
    // FALSIFY-F: §2.1.4 FFN Projections (gate/up/down) — Five-Whys
    //
    // Contract: tensor-layout-v1.yaml §tensors.gate_proj/up_proj/down_proj
    //   gate_proj: [intermediate, hidden] — SwiGLU gate
    //   up_proj:   [intermediate, hidden] — SwiGLU up
    //   down_proj: [hidden, intermediate] — reversed from gate/up!
    //
    // Five-Whys:
    //   Why 1: SwiGLU could produce wrong FFN output
    //   Why 2: gate/up same shape but different roles — swap is silent
    //   Why 3: down_proj reversed dims from gate/up (like o_proj vs q_proj)
    //   Why 4: intermediate_dim varies by architecture (not universal)
    //   Why 5: No test verifying down_proj dim reversal is enforced
    // =========================================================================

    /// FALSIFY-F1: gate/up/down shapes accepted by ValidatedWeight
    #[test]
    fn falsify_f1_ffn_shapes_accepted() {
        let hidden = 896_usize;
        let intermediate = 4864_usize; // Qwen2 0.5B

        // gate: [intermediate, hidden]
        let gate_data: Vec<f32> = (0..intermediate * hidden)
            .map(|i| (i as f32 * 0.001).sin() * 0.1).collect();
        assert!(ValidatedWeight::new(gate_data, intermediate, hidden, "gate_proj").is_ok(),
            "FALSIFY-F1: gate_proj [4864, 896] must be valid");

        // up: [intermediate, hidden] — same shape as gate
        let up_data: Vec<f32> = (0..intermediate * hidden)
            .map(|i| (i as f32 * 0.002).sin() * 0.1).collect();
        assert!(ValidatedWeight::new(up_data, intermediate, hidden, "up_proj").is_ok(),
            "FALSIFY-F1: up_proj [4864, 896] must be valid");

        // down: [hidden, intermediate] — REVERSED
        let down_data: Vec<f32> = (0..hidden * intermediate)
            .map(|i| (i as f32 * 0.003).sin() * 0.1).collect();
        assert!(ValidatedWeight::new(down_data, hidden, intermediate, "down_proj").is_ok(),
            "FALSIFY-F1: down_proj [896, 4864] must be valid");
    }

    /// FALSIFY-F2: enforce_matmul_contract validates down_proj reversed dims
    #[test]
    fn falsify_f2_down_proj_reversed_from_gate_up() {
        use crate::format::layout_contract::enforce_matmul_contract;
        let hidden = 896_usize;
        let intermediate = 4864_usize;

        // gate/up: out=intermediate, in=hidden
        enforce_matmul_contract("gate_proj", &[intermediate, hidden], intermediate, hidden);
        enforce_matmul_contract("up_proj", &[intermediate, hidden], intermediate, hidden);

        // down: out=hidden, in=intermediate (REVERSED)
        enforce_matmul_contract("down_proj", &[hidden, intermediate], hidden, intermediate);
    }

    /// FALSIFY-F3: enforce_matmul_contract panics on wrong down_proj dims
    #[test]
    #[should_panic(expected = "CONTRACT VIOLATION")]
    fn falsify_f3_down_proj_with_gate_dims_panics() {
        use crate::format::layout_contract::enforce_matmul_contract;
        // WRONG: down_proj with gate_proj's [intermediate, hidden] but expected [hidden, intermediate]
        enforce_matmul_contract("down_proj", &[4864, 896], 896, 4864);
    }

    /// FALSIFY-F4: enforce_import_contract handles FFN layer patterns
    #[test]
    fn falsify_f4_import_contract_ffn_patterns() {
        use crate::format::layout_contract::enforce_import_contract;
        let hidden = 896_usize;
        let intermediate = 4864_usize;

        // gate: GGUF [hidden, intermediate] → APR [intermediate, hidden]
        let (shape, _) = enforce_import_contract(
            "blk.0.ffn_gate.weight", &[hidden, intermediate], 151_936, hidden);
        assert_eq!(shape, vec![intermediate, hidden]);

        // up: GGUF [hidden, intermediate] → APR [intermediate, hidden]
        let (shape, _) = enforce_import_contract(
            "blk.0.ffn_up.weight", &[hidden, intermediate], 151_936, hidden);
        assert_eq!(shape, vec![intermediate, hidden]);

        // down: GGUF [intermediate, hidden] → APR [hidden, intermediate]
        let (shape, _) = enforce_import_contract(
            "blk.0.ffn_down.weight", &[intermediate, hidden], 151_936, hidden);
        assert_eq!(shape, vec![hidden, intermediate]);

        // Layer 23 — high index pattern
        let (shape, _) = enforce_import_contract(
            "blk.23.ffn_gate.weight", &[hidden, intermediate], 151_936, hidden);
        assert_eq!(shape, vec![intermediate, hidden]);
    }

    // =========================================================================
    // FALSIFY-N: §2.1.5-6 Layer Norms (attn_norm, ffn_norm, output_norm)
    //
    // Contract: tensor-layout-v1.yaml §tensors.input_layernorm etc.
    //   All norms: shape=[hidden], transpose=false, kernel=element-wise multiply
    //
    // Five-Whys:
    //   Why 1: Zero norm weight → zero output after normalization
    //   Why 2: ValidatedVector doesn't check for near-zero norms
    //   Why 3: 1D tensors need length validation (hidden_dim match)
    //   Why 4: Norm import must NOT transpose (1D identity)
    //   Why 5: No test verifying norm vector length == hidden_dim
    // =========================================================================

    /// FALSIFY-N1: ValidatedVector accepts correct norm weight
    #[test]
    fn falsify_n1_validated_vector_accepts_correct_norm() {
        let hidden = 896_usize;
        let data = vec![1.0f32; hidden]; // RMSNorm typically init to 1.0
        let result = ValidatedVector::new(data, hidden, "attn_norm");
        assert!(result.is_ok(), "FALSIFY-N1: Norm weight of ones must be valid: {:?}", result.err());
    }

    /// FALSIFY-N2: ValidatedVector rejects wrong-length norm
    #[test]
    fn falsify_n2_validated_vector_rejects_wrong_length() {
        let data = vec![1.0f32; 100];
        let result = ValidatedVector::new(data, 896, "attn_norm");
        assert!(result.is_err(),
            "FALSIFY-N2: Norm weight length 100 != hidden_dim 896 must fail");
    }

    /// FALSIFY-N3: ValidatedVector rejects NaN in norm
    #[test]
    fn falsify_n3_validated_vector_rejects_nan_norm() {
        let mut data = vec![1.0f32; 64];
        data[10] = f32::NAN;
        let result = ValidatedVector::new(data, 64, "ffn_norm");
        assert!(result.is_err(),
            "FALSIFY-N3: NaN in norm weight must be rejected");
    }

    /// FALSIFY-N4: ValidatedVector rejects Inf in norm
    #[test]
    fn falsify_n4_validated_vector_rejects_inf_norm() {
        let mut data = vec![1.0f32; 64];
        data[5] = f32::INFINITY;
        let result = ValidatedVector::new(data, 64, "output_norm");
        assert!(result.is_err(),
            "FALSIFY-N4: Inf in norm weight must be rejected");
    }

    /// FALSIFY-N5: enforce_import_contract does NOT transpose 1D norms
    #[test]
    fn falsify_n5_1d_norms_not_transposed() {
        use crate::format::layout_contract::enforce_import_contract;
        let hidden = 896_usize;

        for name in &["blk.0.attn_norm.weight", "blk.0.ffn_norm.weight", "output_norm.weight"] {
            let (shape, needs_t) = enforce_import_contract(name, &[hidden], 151_936, hidden);
            assert_eq!(shape, vec![hidden],
                "FALSIFY-N5: 1D norm '{name}' must pass through unchanged");
            assert!(!needs_t,
                "FALSIFY-N5: 1D norm '{name}' must NOT need data transpose");
        }
    }

    /// FALSIFY-N6: Zero-length norm ACCEPTED — documents PMAT-332 gap
    ///
    /// Five-Whys:
    /// 1. Why does ValidatedVector accept zero-length? → data.len()==expected_len (0==0)
    /// 2. Why no zero-length guard? → Gate 1 only checks length mismatch, not minimum
    /// 3. Why no minimum length? → ValidatedVector was designed for general 1D tensors
    /// 4. Why not add a minimum? → Would need to decide policy (>0? >=hidden?)
    /// 5. Why does this matter? → Zero-length norm produces NaN in LayerNorm division
    #[test]
    fn falsify_n6_zero_length_norm_accepted_gap() {
        // GAP: ValidatedVector::new(vec![], 0, name) returns Ok — no minimum length gate
        let result = ValidatedVector::new(vec![], 0, "attn_norm");
        assert!(result.is_ok(),
            "FALSIFY-N6: Documents PMAT-332 — zero-length norm NOT rejected by ValidatedVector");
    }
