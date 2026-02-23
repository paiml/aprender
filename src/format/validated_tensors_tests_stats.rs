
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
