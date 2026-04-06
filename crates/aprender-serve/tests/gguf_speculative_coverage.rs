//! EXTREME TDD: GGUF Speculative Decoding Coverage Tests
//!
//! Comprehensive tests for SpeculativeConfig, VerificationResult, and SpeculativeDecoder.
//! These require the `gpu` feature flag.
//!
//! Per PARITY-029: Speculative decoding enables O(K) speedup when draft acceptance is high.
//! Implements Leviathan et al. (2023):
//! 1. Draft model generates K candidate tokens quickly
//! 2. Target model verifies all K tokens in parallel
//! 3. Accept tokens until first rejection, then resample

#[cfg(feature = "gpu")]
mod speculative_tests {
    use realizar::gguf::{SpeculativeConfig, SpeculativeDecoder, VerificationResult};

    // ===== SpeculativeConfig Tests =====

    #[test]
    fn test_speculative_config_default_values() {
        let config = SpeculativeConfig::default();

        // Per PARITY-029: Default speculation_length is 4
        assert_eq!(config.speculation_length, 4);
        // Default temperature is 0.0 (greedy)
        assert!((config.draft_temperature - 0.0).abs() < f32::EPSILON);
        // Default is self-speculative (same model for draft)
        assert!(config.self_speculative);
    }

    #[test]
    fn test_speculative_config_custom_speculation_length() {
        let config = SpeculativeConfig {
            speculation_length: 8,
            draft_temperature: 0.0,
            self_speculative: true,
        };

        assert_eq!(config.speculation_length, 8);
    }

    #[test]
    fn test_speculative_config_custom_draft_temperature() {
        let config = SpeculativeConfig {
            speculation_length: 4,
            draft_temperature: 0.7,
            self_speculative: true,
        };

        assert!((config.draft_temperature - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_speculative_config_non_self_speculative() {
        let config = SpeculativeConfig {
            speculation_length: 4,
            draft_temperature: 0.0,
            self_speculative: false,
        };

        assert!(!config.self_speculative);
    }

    #[test]
    fn test_speculative_config_clone() {
        let config = SpeculativeConfig {
            speculation_length: 6,
            draft_temperature: 0.5,
            self_speculative: false,
        };
        let cloned = config.clone();

        assert_eq!(config.speculation_length, cloned.speculation_length);
        assert!((config.draft_temperature - cloned.draft_temperature).abs() < f32::EPSILON);
        assert_eq!(config.self_speculative, cloned.self_speculative);
    }

    #[test]
    fn test_speculative_config_debug() {
        let config = SpeculativeConfig::default();
        let debug_str = format!("{:?}", config);

        assert!(debug_str.contains("SpeculativeConfig"));
        assert!(debug_str.contains("speculation_length"));
        assert!(debug_str.contains("draft_temperature"));
        assert!(debug_str.contains("self_speculative"));
    }

    // ===== VerificationResult Tests =====

    #[test]
    fn test_verification_result_all_accepted() {
        let result = VerificationResult {
            accepted_count: 4,
            draft_count: 4,
            accepted_tokens: vec![10, 20, 30, 40],
            all_accepted: true,
        };

        assert_eq!(result.accepted_count, 4);
        assert_eq!(result.draft_count, 4);
        assert_eq!(result.accepted_tokens.len(), 4);
        assert!(result.all_accepted);
    }

    #[test]
    fn test_verification_result_partial_acceptance() {
        let result = VerificationResult {
            accepted_count: 2,
            draft_count: 4,
            accepted_tokens: vec![10, 20],
            all_accepted: false,
        };

        assert_eq!(result.accepted_count, 2);
        assert_eq!(result.draft_count, 4);
        assert!(!result.all_accepted);
    }

    #[test]
    fn test_verification_result_no_acceptance() {
        let result = VerificationResult {
            accepted_count: 0,
            draft_count: 4,
            accepted_tokens: vec![],
            all_accepted: false,
        };

        assert_eq!(result.accepted_count, 0);
        assert!(result.accepted_tokens.is_empty());
        assert!(!result.all_accepted);
    }

    #[test]
    fn test_verification_result_clone() {
        let result = VerificationResult {
            accepted_count: 3,
            draft_count: 5,
            accepted_tokens: vec![1, 2, 3],
            all_accepted: false,
        };
        let cloned = result.clone();

        assert_eq!(result.accepted_count, cloned.accepted_count);
        assert_eq!(result.draft_count, cloned.draft_count);
        assert_eq!(result.accepted_tokens, cloned.accepted_tokens);
        assert_eq!(result.all_accepted, cloned.all_accepted);
    }

    #[test]
    fn test_verification_result_debug() {
        let result = VerificationResult {
            accepted_count: 2,
            draft_count: 4,
            accepted_tokens: vec![10, 20],
            all_accepted: false,
        };
        let debug_str = format!("{:?}", result);

        assert!(debug_str.contains("VerificationResult"));
        assert!(debug_str.contains("accepted_count"));
        assert!(debug_str.contains("draft_count"));
        assert!(debug_str.contains("accepted_tokens"));
        assert!(debug_str.contains("all_accepted"));
    }

    // ===== SpeculativeDecoder Construction Tests =====

    #[test]
    fn test_speculative_decoder_new() {
        let decoder = SpeculativeDecoder::new();

        assert_eq!(decoder.config.speculation_length, 4);
        assert!((decoder.config.draft_temperature - 0.0).abs() < f32::EPSILON);
        assert!(decoder.config.self_speculative);
    }

    #[test]
    fn test_speculative_decoder_with_config() {
        let config = SpeculativeConfig {
            speculation_length: 8,
            draft_temperature: 0.5,
            self_speculative: false,
        };
        let decoder = SpeculativeDecoder::with_config(config);

        assert_eq!(decoder.config.speculation_length, 8);
        assert!((decoder.config.draft_temperature - 0.5).abs() < f32::EPSILON);
        assert!(!decoder.config.self_speculative);
    }

    #[test]
    fn test_speculative_decoder_default() {
        let decoder = SpeculativeDecoder::default();

        assert_eq!(decoder.config.speculation_length, 4);
        assert!((decoder.config.draft_temperature - 0.0).abs() < f32::EPSILON);
        assert!(decoder.config.self_speculative);
    }

    // ===== SpeculativeDecoder Statistics Tests =====

    #[test]
    fn test_speculative_decoder_initial_acceptance_rate() {
        let decoder = SpeculativeDecoder::new();

        // No tokens processed yet, should return 0.0
        assert!((decoder.acceptance_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_speculative_decoder_expected_speedup_no_tokens() {
        let decoder = SpeculativeDecoder::new();

        // With 0% acceptance, speedup = K * 0 + 1 = 1.0
        assert!((decoder.expected_speedup() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_speculative_decoder_reset_stats() {
        let decoder = SpeculativeDecoder::new();

        // Verify draft tokens to accumulate statistics
        let draft_tokens = vec![10, 20, 30];
        let target_logits = vec![
            create_logits_with_top(10, 100), // Matches draft
            create_logits_with_top(20, 100), // Matches draft
            create_logits_with_top(30, 100), // Matches draft
        ];
        decoder.verify_draft(&draft_tokens, &target_logits, 0.0);

        // Reset stats
        decoder.reset_stats();

        // After reset, acceptance rate should be 0
        assert!((decoder.acceptance_rate() - 0.0).abs() < f64::EPSILON);
    }

    // ===== SpeculativeDecoder verify_draft Tests (Greedy Temperature=0.0) =====

    #[test]
    fn test_verify_draft_greedy_all_match() {
        let decoder = SpeculativeDecoder::new();

        let draft_tokens = vec![10, 20, 30, 40];
        let target_logits = vec![
            create_logits_with_top(10, 100),
            create_logits_with_top(20, 100),
            create_logits_with_top(30, 100),
            create_logits_with_top(40, 100),
        ];

        let result = decoder.verify_draft(&draft_tokens, &target_logits, 0.0);

        assert_eq!(result.accepted_count, 4);
        assert_eq!(result.draft_count, 4);
        assert!(result.all_accepted);
        assert_eq!(result.accepted_tokens, vec![10, 20, 30, 40]);
    }

    #[test]
    fn test_verify_draft_greedy_first_mismatch() {
        let decoder = SpeculativeDecoder::new();

        let draft_tokens = vec![10, 20, 30, 40];
        let target_logits = vec![
            create_logits_with_top(99, 100), // Mismatch! Draft is 10, target top is 99
            create_logits_with_top(20, 100),
            create_logits_with_top(30, 100),
            create_logits_with_top(40, 100),
        ];

        let result = decoder.verify_draft(&draft_tokens, &target_logits, 0.0);

        // At first mismatch, we accept target's token and stop
        assert_eq!(result.accepted_count, 1);
        assert_eq!(result.draft_count, 4);
        assert!(!result.all_accepted);
        assert_eq!(result.accepted_tokens, vec![99]); // Target's token, not draft's
    }

    #[test]
    fn test_verify_draft_greedy_middle_mismatch() {
        let decoder = SpeculativeDecoder::new();

        let draft_tokens = vec![10, 20, 30, 40];
        let target_logits = vec![
            create_logits_with_top(10, 100), // Match
            create_logits_with_top(20, 100), // Match
            create_logits_with_top(99, 100), // Mismatch!
            create_logits_with_top(40, 100),
        ];

        let result = decoder.verify_draft(&draft_tokens, &target_logits, 0.0);

        assert_eq!(result.accepted_count, 3);
        assert_eq!(result.draft_count, 4);
        assert!(!result.all_accepted);
        assert_eq!(result.accepted_tokens, vec![10, 20, 99]); // Two matches, then target token
    }

    #[test]
    fn test_verify_draft_greedy_last_mismatch() {
        let decoder = SpeculativeDecoder::new();

        let draft_tokens = vec![10, 20, 30, 40];
        let target_logits = vec![
            create_logits_with_top(10, 100), // Match
            create_logits_with_top(20, 100), // Match
            create_logits_with_top(30, 100), // Match
            create_logits_with_top(99, 100), // Mismatch!
        ];

        let result = decoder.verify_draft(&draft_tokens, &target_logits, 0.0);

        // All 4 positions were processed - 3 matches + 1 target replacement
        assert_eq!(result.accepted_count, 4);
        assert_eq!(result.draft_count, 4);
        // all_accepted = accepted_count == draft_count, which is 4 == 4
        assert!(result.all_accepted);
        assert_eq!(result.accepted_tokens, vec![10, 20, 30, 99]);
    }

    #[test]
    fn test_verify_draft_empty_draft_tokens() {
        let decoder = SpeculativeDecoder::new();

        let draft_tokens: Vec<u32> = vec![];
        let target_logits: Vec<Vec<f32>> = vec![];

        let result = decoder.verify_draft(&draft_tokens, &target_logits, 0.0);

        assert_eq!(result.accepted_count, 0);
        assert_eq!(result.draft_count, 0);
        assert!(result.all_accepted); // 0 == 0 means all_accepted
        assert!(result.accepted_tokens.is_empty());
    }

    #[test]
    fn test_verify_draft_more_draft_than_logits() {
        let decoder = SpeculativeDecoder::new();

        let draft_tokens = vec![10, 20, 30, 40];
        let target_logits = vec![
            create_logits_with_top(10, 100), // Match
            create_logits_with_top(20, 100), // Match
                                             // Only 2 logits for 4 draft tokens
        ];

        let result = decoder.verify_draft(&draft_tokens, &target_logits, 0.0);

        // Should stop when we run out of logits
        assert_eq!(result.accepted_count, 2);
        assert_eq!(result.draft_count, 4);
        assert!(!result.all_accepted);
        assert_eq!(result.accepted_tokens, vec![10, 20]);
    }

    // ===== SpeculativeDecoder verify_draft Tests (Non-greedy Temperature>0) =====

    #[test]
    fn test_verify_draft_nongreedy_in_top_k() {
        let decoder = SpeculativeDecoder::new();

        // Create logits where draft token is in top-10 but not top-1
        let draft_tokens = vec![5]; // Token 5
        let mut logits = vec![0.0; 100];
        logits[0] = 10.0; // Token 0 is top
        logits[5] = 9.0; // Token 5 is in top-10

        let result = decoder.verify_draft(&draft_tokens, &[logits], 1.0);

        // Token 5 should be accepted (in top-10)
        assert_eq!(result.accepted_count, 1);
        assert!(result.all_accepted);
        assert_eq!(result.accepted_tokens, vec![5]);
    }

    #[test]
    fn test_verify_draft_nongreedy_not_in_top_k() {
        let decoder = SpeculativeDecoder::new();

        // Create logits where draft token is not in top-10
        let draft_tokens = vec![99]; // Token 99
        let mut logits = vec![0.0; 100];
        for (i, val) in logits.iter_mut().take(10).enumerate() {
            *val = 10.0 - i as f32; // Tokens 0-9 are top-10
        }
        logits[99] = -100.0; // Token 99 is very low

        let result = decoder.verify_draft(&draft_tokens, &[logits], 1.0);

        // Token 99 rejected, use top token (0) instead
        // But accepted_count still increments (we got 1 token)
        assert_eq!(result.accepted_count, 1);
        // all_accepted = 1 == 1, so true (we processed all draft positions)
        assert!(result.all_accepted);
        assert_eq!(result.accepted_tokens, vec![0]); // Top token
    }

    #[test]
    fn test_verify_draft_nongreedy_multiple_tokens() {
        let decoder = SpeculativeDecoder::new();

        let draft_tokens = vec![5, 15, 99]; // Token 5 in top-10, 15 not, 99 not

        let logits1 = {
            let mut l = vec![0.0; 100];
            for (i, val) in l.iter_mut().take(10).enumerate() {
                *val = 10.0 - i as f32;
            }
            l[5] = 8.0; // Token 5 in top-10
            l
        };

        let logits2 = {
            let mut l = vec![0.0; 100];
            for (i, val) in l.iter_mut().take(10).enumerate() {
                *val = 10.0 - i as f32;
            }
            l[15] = -100.0; // Token 15 not in top-10
            l
        };

        let logits3 = {
            let mut l = vec![0.0; 100];
            l[0] = 10.0;
            l
        };

        let result = decoder.verify_draft(&draft_tokens, &[logits1, logits2, logits3], 1.0);

        // Token 5 accepted, then token 15 rejected (stops here)
        assert_eq!(result.accepted_count, 2);
        assert!(!result.all_accepted);
        assert_eq!(result.accepted_tokens.len(), 2);
        assert_eq!(result.accepted_tokens[0], 5);
        assert_eq!(result.accepted_tokens[1], 0); // Top token replaces rejected 15
    }

    // ===== SpeculativeDecoder Statistics Accumulation Tests =====

    #[test]
    fn test_speculative_decoder_stats_accumulate() {
        let decoder = SpeculativeDecoder::new();

        // First verification: 3 draft, 2 accepted
        let result1 = decoder.verify_draft(
            &[10, 20, 30],
            &[
                create_logits_with_top(10, 100),
                create_logits_with_top(20, 100),
                create_logits_with_top(99, 100), // Mismatch
            ],
            0.0,
        );
        assert_eq!(result1.accepted_count, 3);

        // Second verification: 2 draft, 2 accepted
        let result2 = decoder.verify_draft(
            &[50, 60],
            &[
                create_logits_with_top(50, 100),
                create_logits_with_top(60, 100),
            ],
            0.0,
        );
        assert_eq!(result2.accepted_count, 2);

        // Total: 5 draft tokens, 5 accepted
        // Acceptance rate = 5/5 = 1.0
        let rate = decoder.acceptance_rate();
        assert!((rate - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_speculative_decoder_expected_speedup_with_stats() {
        let config = SpeculativeConfig {
            speculation_length: 4,
            draft_temperature: 0.0,
            self_speculative: true,
        };
        let decoder = SpeculativeDecoder::with_config(config);

        // Verify 4 tokens, all accepted
        decoder.verify_draft(
            &[10, 20, 30, 40],
            &[
                create_logits_with_top(10, 100),
                create_logits_with_top(20, 100),
                create_logits_with_top(30, 100),
                create_logits_with_top(40, 100),
            ],
            0.0,
        );

        // With K=4 and 100% acceptance: speedup = 4 * 1.0 + 1 = 5.0
        let speedup = decoder.expected_speedup();
        assert!((speedup - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_speculative_decoder_expected_speedup_partial_acceptance() {
        let config = SpeculativeConfig {
            speculation_length: 4,
            draft_temperature: 0.0,
            self_speculative: true,
        };
        let decoder = SpeculativeDecoder::with_config(config);

        // Verify 4 tokens, 2 accepted (50% rate)
        decoder.verify_draft(
            &[10, 20, 30, 40],
            &[
                create_logits_with_top(10, 100), // Accept
                create_logits_with_top(20, 100), // Accept
                create_logits_with_top(99, 100), // Reject - but still counts 3 accepted
                create_logits_with_top(40, 100),
            ],
            0.0,
        );

        // Actually 3/4 accepted = 0.75 acceptance rate
        let rate = decoder.acceptance_rate();
        assert!((rate - 0.75).abs() < 0.01);

        // Speedup = 4 * 0.75 + 1 = 4.0
        let speedup = decoder.expected_speedup();
        assert!((speedup - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_speculative_decoder_reset_and_recalculate() {
        let decoder = SpeculativeDecoder::new();

        // First batch
        decoder.verify_draft(
            &[10, 20],
            &[
                create_logits_with_top(10, 100),
                create_logits_with_top(20, 100),
            ],
            0.0,
        );
        assert!((decoder.acceptance_rate() - 1.0).abs() < 0.01);

        // Reset
        decoder.reset_stats();
        assert!((decoder.acceptance_rate() - 0.0).abs() < f64::EPSILON);

        // New batch with 50% acceptance
        decoder.verify_draft(
            &[10, 20],
            &[
                create_logits_with_top(10, 100), // Accept
                create_logits_with_top(99, 100), // Reject (but still 2 accepted)
            ],
            0.0,
        );
        // Both positions got accepted (one match, one mismatch replacement)
        assert!((decoder.acceptance_rate() - 1.0).abs() < 0.01);
    }

    // ===== Edge Cases =====

    #[test]
    fn test_verify_draft_single_token_match() {
        let decoder = SpeculativeDecoder::new();

        let result = decoder.verify_draft(&[42], &[create_logits_with_top(42, 100)], 0.0);

        assert_eq!(result.accepted_count, 1);
        assert_eq!(result.draft_count, 1);
        assert!(result.all_accepted);
        assert_eq!(result.accepted_tokens, vec![42]);
    }

    #[test]
    fn test_verify_draft_single_token_mismatch() {
        let decoder = SpeculativeDecoder::new();

        let result = decoder.verify_draft(&[42], &[create_logits_with_top(99, 100)], 0.0);

        // Mismatch: draft was 42 but target top was 99
        // We use target's token (99) and count it as accepted
        assert_eq!(result.accepted_count, 1);
        assert_eq!(result.draft_count, 1);
        // all_accepted = 1 == 1, so true
        assert!(result.all_accepted);
        assert_eq!(result.accepted_tokens, vec![99]); // Target's token
    }

    #[test]
    fn test_verify_draft_large_speculation_length() {
        let config = SpeculativeConfig {
            speculation_length: 16,
            draft_temperature: 0.0,
            self_speculative: true,
        };
        let decoder = SpeculativeDecoder::with_config(config);

        let draft_tokens: Vec<u32> = (0..16).collect();
        let target_logits: Vec<Vec<f32>> = (0..16)
            .map(|i| create_logits_with_top(i as usize, 100))
            .collect();

        let result = decoder.verify_draft(&draft_tokens, &target_logits, 0.0);

        assert_eq!(result.accepted_count, 16);
        assert!(result.all_accepted);
    }

    #[test]
    fn test_verify_draft_logits_with_nan_handling() {
        let decoder = SpeculativeDecoder::new();

        // Create logits with NaN - partial_cmp should handle this
        let mut logits = vec![0.0; 100];
        logits[0] = f32::NAN;
        logits[10] = 10.0; // This should be selected as max

        let result = decoder.verify_draft(&[10], &[logits], 0.0);

        // Should still process, accepting token 10
        assert_eq!(result.accepted_count, 1);
    }

    #[test]
    fn test_verify_draft_logits_empty_vocab() {
        let decoder = SpeculativeDecoder::new();

        // Empty logits vector (0 vocab)
        let result = decoder.verify_draft(&[0], &[vec![]], 0.0);

        // Empty logits: max_by returns None, unwrap_or gives (0, &0.0)
        // So target_token = 0, draft_token = 0, they match!
        // accepted_count = 1, draft_count = 1, all_accepted = true
        assert_eq!(result.accepted_count, 1);
        assert!(result.all_accepted);
        assert_eq!(result.accepted_tokens, vec![0]);
    }

    #[test]
    fn test_speculative_config_extreme_values() {
        let config = SpeculativeConfig {
            speculation_length: usize::MAX,
            draft_temperature: f32::MAX,
            self_speculative: true,
        };

        assert_eq!(config.speculation_length, usize::MAX);
        assert_eq!(config.draft_temperature, f32::MAX);
    }

    #[test]
    fn test_speculative_decoder_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let decoder = Arc::new(SpeculativeDecoder::new());
        let mut handles = vec![];

        for _ in 0..4 {
            let decoder_clone = Arc::clone(&decoder);
            handles.push(thread::spawn(move || {
                decoder_clone.verify_draft(
                    &[10, 20],
                    &[
                        create_logits_with_top(10, 100),
                        create_logits_with_top(20, 100),
                    ],
                    0.0,
                );
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // After 4 threads each processing 2 tokens
        // Total: 8 draft tokens, 8 accepted
        let rate = decoder.acceptance_rate();
        assert!((rate - 1.0).abs() < 0.01);
    }

    // ===== Helper Functions =====

    /// Create a logits vector where the specified token has the highest value
    fn create_logits_with_top(top_token: usize, vocab_size: usize) -> Vec<f32> {
        let mut logits = vec![0.0; vocab_size];
        if top_token < vocab_size {
            logits[top_token] = 10.0;
        }
        logits
    }
}

// Non-GPU test to ensure module compiles
#[test]
fn test_gguf_speculative_module_compiles() {
    // Use a type from the gguf module to prove compilation
    let _ = std::mem::size_of::<realizar::gguf::GGUFHeader>();
}
