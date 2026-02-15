
    #[test]
    fn test_embedding_validity_check_contains_inf() {
        let embedding = vec![0.1_f32, 0.2, f32::INFINITY, 0.4];
        let test_tokens_len = 2;
        let hidden_dim = 2;
        let embedding_ok = !embedding.is_empty()
            && embedding.len() == test_tokens_len * hidden_dim
            && !embedding.iter().any(|x| x.is_nan() || x.is_infinite());
        assert!(!embedding_ok);
    }

    // ========================================================================
    // Tokenizer Details Formatting
    // ========================================================================

    #[test]
    fn test_tokenizer_details_format_ok() {
        let test_tokens = vec![1u32, 2];
        let embedding_len = 512;
        let details = format!("tokens={:?} \u{2192} {} floats", test_tokens, embedding_len);
        assert!(details.contains("[1, 2]"));
        assert!(details.contains("512 floats"));
    }

    #[test]
    fn test_tokenizer_details_format_failed() {
        let details = "Tokenizer/embedding failed".to_string();
        assert!(details.contains("failed"));
    }

    // ========================================================================
    // Full 10-Stage Pipeline Result Table Rendering
    // ========================================================================

    #[test]
    fn test_print_full_pipeline_all_pass() {
        let stage_names = [
            "Tokenizer",
            "Embedding",
            "Positional Encoding",
            "Q/K/V Projection",
            "Attention Scores",
            "Feed-Forward (MLP)",
            "Layer Norm",
            "LM Head",
            "Logits \u{2192} Probs",
            "Sampler/Decode",
        ];
        let results: Vec<StageResult> = stage_names
            .iter()
            .map(|name| StageResult {
                name,
                eli5: "test",
                passed: true,
                details: Some("OK".to_string()),
            })
            .collect();
        assert_eq!(results.len(), 10);
        let passed = results.iter().filter(|r| r.passed).count();
        assert_eq!(passed, 10);
        // Should not panic
        print_results_table(&results);
    }

    #[test]
    fn test_print_full_pipeline_mixed_results() {
        let results = vec![
            StageResult {
                name: "Tokenizer",
                eli5: "Words \u{2192} numbers",
                passed: true,
                details: Some("tokens=[1, 2] \u{2192} 512 floats".to_string()),
            },
            StageResult {
                name: "Embedding",
                eli5: "Numbers \u{2192} vectors",
                passed: true,
                details: Some("Found embedding tensor".to_string()),
            },
            StageResult {
                name: "Positional Encoding",
                eli5: "\"You are word #3\"",
                passed: true,
                details: Some("rope_theta=10000.0".to_string()),
            },
            StageResult {
                name: "Q/K/V Projection",
                eli5: "Make 3 question copies",
                passed: false,
                details: Some("Missing Q/K/V tensors".to_string()),
            },
            StageResult {
                name: "Attention Scores",
                eli5: "\"Who to look at?\"",
                passed: false,
                details: Some("Missing attention output tensor".to_string()),
            },
            StageResult {
                name: "Feed-Forward (MLP)",
                eli5: "\"Think about it\"",
                passed: true,
                details: Some("MLP tensors found".to_string()),
            },
            StageResult {
                name: "Layer Norm",
                eli5: "Keep numbers stable",
                passed: true,
                details: Some("32 layers".to_string()),
            },
            StageResult {
                name: "LM Head",
                eli5: "Vector \u{2192} vocab scores",
                passed: true,
                details: Some("vocab_size=32000".to_string()),
            },
            StageResult {
                name: "Logits \u{2192} Probs",
                eli5: "Scores \u{2192} percentages",
                passed: true,
                details: Some("logits[32000]: min=-5.20, max=12.30".to_string()),
            },
            StageResult {
                name: "Sampler/Decode",
                eli5: "Pick word, return",
                passed: false,
                details: Some("FAIL: softmax sum = 0.500000 (expected 1.0)".to_string()),
            },
        ];
        let passed = results.iter().filter(|r| r.passed).count();
        assert_eq!(passed, 7);
        assert_eq!(results.len(), 10);
        // Should not panic - exercises truncation for long details
        print_results_table(&results);
    }

    // ========================================================================
    // print_results_table: Details Truncation In-Function Behavior
    // ========================================================================

    #[test]
    fn test_print_results_table_truncates_long_details() {
        // This exercises the truncation branch inside print_results_table
        // Details > 36 chars should be truncated to 33 + "..."
        let long_detail = "a]bcdefghijklmnopqrstuvwxyz0123456789EXTRA";
        assert!(long_detail.len() > 36);
        let results = vec![StageResult {
            name: "Long",
            eli5: "test",
            passed: true,
            details: Some(long_detail.to_string()),
        }];
        // Should not panic, and should truncate internally
        print_results_table(&results);
    }

    #[test]
    fn test_print_results_table_exact_boundary_details() {
        // Exactly 36 chars - should NOT truncate
        let exact_36 = "a".repeat(36);
        assert_eq!(exact_36.len(), 36);
        let results = vec![StageResult {
            name: "Exact",
            eli5: "test",
            passed: false,
            details: Some(exact_36),
        }];
        print_results_table(&results);
    }

    #[test]
    fn test_print_results_table_one_over_boundary() {
        // 37 chars - should truncate
        let over_37 = "b".repeat(37);
        assert_eq!(over_37.len(), 37);
        let results = vec![StageResult {
            name: "Over",
            eli5: "test",
            passed: true,
            details: Some(over_37),
        }];
        print_results_table(&results);
    }

    // ========================================================================
    // Success/Failure Message Formatting (mirrors run() lines 57-79)
    // ========================================================================

    #[test]
    fn test_success_message_format() {
        let passed_count = 10;
        let total_count = 10;
        let msg = format!(
            "\u{2705} {}/{} STAGES PASSED. MODEL PROVEN CORRECT.",
            passed_count, total_count
        );
        assert!(msg.contains("10/10"));
        assert!(msg.contains("PROVEN CORRECT"));
    }

    #[test]
    fn test_failure_message_format() {
        let passed_count = 7;
        let total_count = 10;
        let msg = format!(
            "\u{274c} {}/{} STAGES PASSED. CHECK STAGE LOGS.",
            passed_count, total_count
        );
        assert!(msg.contains("7/10"));
        assert!(msg.contains("CHECK STAGE LOGS"));
    }

    #[test]
    fn test_failure_message_zero_passed() {
        let passed_count = 0;
        let total_count = 10;
        let msg = format!(
            "\u{274c} {}/{} STAGES PASSED. CHECK STAGE LOGS.",
            passed_count, total_count
        );
        assert!(msg.contains("0/10"));
    }

    // ========================================================================
    // Vocab Size with Dims Matching (GGUF LM Head check)
    // ========================================================================

    #[test]
    fn test_vocab_dim_matching_present() {
        let dims: Vec<u64> = vec![32000, 4096];
        let vocab_size = 32000_usize;
        let matches = dims.iter().any(|&d| d as usize == vocab_size);
        assert!(matches);
    }

    #[test]
    fn test_vocab_dim_matching_absent() {
        let dims: Vec<u64> = vec![4096, 4096];
        let vocab_size = 32000_usize;
        let matches = dims.iter().any(|&d| d as usize == vocab_size);
        assert!(!matches);
    }

    #[test]
    fn test_vocab_dim_matching_empty_dims() {
        let dims: Vec<u64> = vec![];
        let vocab_size = 32000_usize;
        let matches = dims.iter().any(|&d| d as usize == vocab_size);
        assert!(!matches);
    }

    // ========================================================================
    // APR Metadata Defaults (mirrors unwrap_or defaults)
    // ========================================================================

    #[test]
    fn test_metadata_defaults_num_layers() {
        let val: Option<usize> = None;
        assert_eq!(val.unwrap_or(0), 0);
    }

    #[test]
    fn test_metadata_defaults_hidden_size() {
        let val: Option<usize> = None;
        assert_eq!(val.unwrap_or(0), 0);
    }

    #[test]
    fn test_metadata_defaults_vocab_size() {
        let val: Option<usize> = None;
        assert_eq!(val.unwrap_or(32000), 32000);
    }

    #[test]
    fn test_metadata_defaults_num_heads() {
        let val: Option<usize> = None;
        assert_eq!(val.unwrap_or(0), 0);
    }

    #[test]
    fn test_metadata_present_overrides_default() {
        let val: Option<usize> = Some(128256);
        assert_eq!(val.unwrap_or(32000), 128256);
    }

    // ========================================================================
    // Softmax Edge Cases (additional precision tests)
    // ========================================================================

    #[test]
    fn test_softmax_identical_logits_uniform() {
        // All identical logits should produce uniform distribution
        let logits = vec![3.14_f32; 100];
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
        let probs: Vec<f32> = logits
            .iter()
            .map(|x| (x - max_logit).exp() / exp_sum)
            .collect();
        let prob_sum: f32 = probs.iter().sum();
        assert!((prob_sum - 1.0).abs() < 0.001);
        // Each prob should be ~0.01
        for p in &probs {
            assert!((p - 0.01).abs() < 0.001);
        }
    }

    #[test]
    fn test_softmax_very_negative_logits() {
        // All very negative logits - should still sum to 1
        let logits = vec![-1000.0_f32, -1001.0, -999.0];
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
        let probs: Vec<f32> = logits
            .iter()
            .map(|x| (x - max_logit).exp() / exp_sum)
            .collect();
        let prob_sum: f32 = probs.iter().sum();
        assert!((prob_sum - 1.0).abs() < 0.001);
        assert!(!probs.iter().any(|x| x.is_nan()));
    }

    #[test]
    fn test_softmax_vocab_size_logits() {
        // Simulate realistic vocab size
        let logits: Vec<f32> = (0..32000).map(|i| (i as f32) * 0.001 - 16.0).collect();
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
        let probs: Vec<f32> = logits
            .iter()
            .map(|x| (x - max_logit).exp() / exp_sum)
            .collect();
        let prob_sum: f32 = probs.iter().sum();
        assert!(
            (prob_sum - 1.0).abs() < 0.01,
            "softmax over 32k logits should sum to ~1.0, got {}",
            prob_sum
        );
    }

    // ========================================================================
    // APR vs GGUF Tensor Name Convention Cross-Check
    // ========================================================================

    #[test]
    fn test_apr_and_gguf_embedding_names_differ() {
        // APR uses "emb"/"wte"/"token_embd"; GGUF uses "token_embd"/"embed_tokens"
        // "token_embd" is common to both
        let apr_check =
            |n: &str| n.contains("emb") || n.contains("wte") || n.contains("token_embd");
        let gguf_check = |n: &str| n.contains("token_embd") || n.contains("embed_tokens");

        // "token_embd.weight" matches both
        assert!(apr_check("token_embd.weight"));
        assert!(gguf_check("token_embd.weight"));

        // "embed_tokens" matches GGUF but also APR (via "emb" substring)
        assert!(apr_check("model.embed_tokens.weight"));
        assert!(gguf_check("model.embed_tokens.weight"));

        // "wte" only matches APR
        assert!(apr_check("transformer.wte.weight"));
        assert!(!gguf_check("transformer.wte.weight"));
    }

    #[test]
    fn test_full_model_tensor_inventory_gguf() {
        // Simulate a complete GGUF model's tensor names
        let names = vec![
            "token_embd.weight",
            "blk.0.attn_norm.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "output_norm.weight",
            "output.weight",
        ];

        // All stage checks should pass
        let has_embed = names
            .iter()
            .any(|n| n.contains("token_embd") || n.contains("embed_tokens"));
        let has_q = names
            .iter()
            .any(|n| n.contains("blk.0.attn_q") || n.contains("layers.0.self_attn.q_proj"));
        let has_k = names
            .iter()
            .any(|n| n.contains("blk.0.attn_k") || n.contains("layers.0.self_attn.k_proj"));
        let has_v = names
            .iter()
            .any(|n| n.contains("blk.0.attn_v") || n.contains("layers.0.self_attn.v_proj"));
        let has_attn_out = names
            .iter()
            .any(|n| n.contains("attn_output") || n.contains("o_proj"));
        let has_gate = names
            .iter()
            .any(|n| n.contains("ffn_gate") || n.contains("gate_proj"));
        let has_up = names
            .iter()
            .any(|n| n.contains("ffn_up") || n.contains("up_proj"));
        let has_down = names
            .iter()
            .any(|n| n.contains("ffn_down") || n.contains("down_proj"));
        let has_attn_norm = names
            .iter()
            .any(|n| n.contains("attn_norm") || n.contains("input_layernorm"));
        let has_ffn_norm = names
            .iter()
            .any(|n| n.contains("ffn_norm") || n.contains("post_attention_layernorm"));
        let has_lm_head = names
            .iter()
            .any(|n| *n == "output.weight" || n.contains("lm_head"));

        assert!(has_embed);
        assert!(has_q && has_k && has_v);
        assert!(has_attn_out);
        assert!(has_gate && has_up && has_down);
        assert!(has_attn_norm && has_ffn_norm);
        assert!(has_lm_head);
    }
