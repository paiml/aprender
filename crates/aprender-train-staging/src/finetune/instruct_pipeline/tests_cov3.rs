//! Coverage (cov3) tests for the instruct pipeline.

use super::*;
use std::path::Path;

// ── cov3: additional coverage tests ─────────────────────────────

#[test]
fn test_cov3_compute_causal_lm_loss_gradient_shape_matches_input() {
    for (seq_len, vocab_size) in [(1, 3), (4, 10), (2, 5)] {
        let logits = vec![0.1f32; seq_len * vocab_size];
        let full_ids: Vec<u32> = (0..seq_len as u32).collect();
        let loss_end = seq_len.saturating_sub(1);
        let (_, grad) =
            InstructPipeline::compute_causal_lm_loss(&logits, &full_ids, 0, loss_end, vocab_size);
        assert_eq!(grad.len(), seq_len * vocab_size);
    }
}

#[test]
fn test_cov3_compute_causal_lm_loss_gradient_sums_to_zero_per_row() {
    let vocab_size = 5;
    let seq_len = 4;
    let logits: Vec<f32> = (0..seq_len * vocab_size).map(|i| (i as f32) * 0.1).collect();
    let full_ids = vec![0u32, 2, 1, 3];
    let (_, grad) = InstructPipeline::compute_causal_lm_loss(&logits, &full_ids, 0, 3, vocab_size);

    for pos in 0..3 {
        let row_start = pos * vocab_size;
        let row_sum: f32 = grad[row_start..row_start + vocab_size].iter().sum();
        assert!(row_sum.abs() < 1e-5, "Gradient row {pos} sums to {row_sum}, expected ~0");
    }
}

#[test]
fn test_cov3_compute_causal_lm_loss_uniform_logits_high_loss() {
    let vocab_size = 10;
    let seq_len = 3;
    let logits = vec![0.0f32; seq_len * vocab_size];
    let full_ids = vec![0u32, 5, 3];
    let (loss, _) = InstructPipeline::compute_causal_lm_loss(&logits, &full_ids, 0, 2, vocab_size);
    let expected = (vocab_size as f32).ln();
    assert!(
        (loss - expected).abs() < 0.01,
        "Uniform logits loss {loss} should be ~ln({vocab_size})={expected}"
    );
}

#[test]
fn test_cov3_compute_causal_lm_loss_single_token_range() {
    let vocab_size = 5;
    let logits = vec![0.0f32; 10];
    logits[..vocab_size].iter().for_each(|_| {});
    let full_ids = vec![0u32, 2];
    let (loss, grad) =
        InstructPipeline::compute_causal_lm_loss(&logits, &full_ids, 0, 1, vocab_size);
    assert!(loss > 0.0);
    let row1_sum: f32 = grad[vocab_size..].iter().map(|v| v.abs()).sum();
    assert!(row1_sum < 1e-10, "Second row should be all zeros");
}

#[test]
fn test_cov3_compute_causal_lm_loss_loss_start_equals_loss_end() {
    let vocab_size = 5;
    let logits = vec![1.0f32; 15];
    let full_ids = vec![0u32, 1, 2];
    let (loss, _) = InstructPipeline::compute_causal_lm_loss(&logits, &full_ids, 1, 1, vocab_size);
    assert_eq!(loss, 0.0);
}

#[test]
fn test_cov3_compute_causal_lm_loss_multiple_targets_out_of_vocab() {
    let vocab_size = 3;
    let logits = vec![1.0f32; 9];
    let full_ids = vec![10u32, 20, 30];
    let (loss, grad) =
        InstructPipeline::compute_causal_lm_loss(&logits, &full_ids, 0, 2, vocab_size);
    assert_eq!(loss, 0.0);
    assert!(grad.iter().all(|&v| v == 0.0));
}

#[test]
fn test_cov3_train_step_empty_prompt_empty_response() {
    let model_config = TransformerConfig::tiny();
    let instruct_config =
        InstructConfig { lora_rank: 4, max_seq_len: 64, ..InstructConfig::default() };
    let mut pipeline = InstructPipeline::new(&model_config, instruct_config);
    let result = pipeline.train_step(&[], &[]);
    assert_eq!(result.loss, 0.0);
    assert_eq!(result.num_response_tokens, 0);
    assert_eq!(result.perplexity, 1.0);
}

#[test]
fn test_cov3_train_step_empty_prompt_nonempty_response() {
    let model_config = TransformerConfig::tiny();
    let instruct_config =
        InstructConfig { lora_rank: 4, max_seq_len: 64, ..InstructConfig::default() };
    let mut pipeline = InstructPipeline::new(&model_config, instruct_config);
    let result = pipeline.train_step(&[], &[5, 6, 7]);
    assert!(result.loss >= 0.0);
}

#[test]
fn test_cov3_train_step_prompt_exceeds_max_seq_len() {
    let model_config = TransformerConfig::tiny();
    let instruct_config =
        InstructConfig { lora_rank: 4, max_seq_len: 8, ..InstructConfig::default() };
    let mut pipeline = InstructPipeline::new(&model_config, instruct_config);
    let prompt_ids: Vec<u32> = (0..10).collect();
    let response_ids: Vec<u32> = (10..15).collect();
    let result = pipeline.train_step(&prompt_ids, &response_ids);
    assert!(result.loss >= 0.0);
}

#[test]
fn test_cov3_train_step_single_token_total() {
    let model_config = TransformerConfig::tiny();
    let instruct_config =
        InstructConfig { lora_rank: 4, max_seq_len: 64, ..InstructConfig::default() };
    let mut pipeline = InstructPipeline::new(&model_config, instruct_config);
    let result = pipeline.train_step(&[0], &[]);
    assert_eq!(result.loss, 0.0);
    assert_eq!(result.num_response_tokens, 0);
    assert_eq!(result.perplexity, 1.0);
}

#[test]
fn test_cov3_evaluate_single_token_response() {
    let model_config = TransformerConfig::tiny();
    let instruct_config =
        InstructConfig { lora_rank: 4, max_seq_len: 64, ..InstructConfig::default() };
    let pipeline = InstructPipeline::new(&model_config, instruct_config);
    let prompts = vec![vec![0u32, 1, 2]];
    let responses = vec![vec![3u32]];
    let result = pipeline.evaluate(&prompts, &responses);
    assert_eq!(result.total_response_tokens, 1);
    assert!(result.avg_loss >= 0.0);
}

#[test]
fn test_cov3_evaluate_prompt_exceeds_max_seq_len() {
    let model_config = TransformerConfig::tiny();
    let instruct_config =
        InstructConfig { lora_rank: 4, max_seq_len: 5, ..InstructConfig::default() };
    let pipeline = InstructPipeline::new(&model_config, instruct_config);
    let prompts = vec![vec![0u32; 8]];
    let responses = vec![vec![1u32; 3]];
    let result = pipeline.evaluate(&prompts, &responses);
    assert!(result.avg_loss >= 0.0);
}

#[test]
fn test_cov3_evaluate_single_token_full_sequence() {
    let model_config = TransformerConfig::tiny();
    let instruct_config =
        InstructConfig { lora_rank: 4, max_seq_len: 64, ..InstructConfig::default() };
    let pipeline = InstructPipeline::new(&model_config, instruct_config);
    let prompts = vec![vec![0u32]];
    let responses = vec![vec![1u32]];
    let result = pipeline.evaluate(&prompts, &responses);
    assert_eq!(result.total_response_tokens, 1);
}

#[test]
fn test_cov3_sample_token_negative_temperature() {
    let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
    let token = sample_token(&logits, -1.0, 0);
    assert_eq!(token, 3);
}

#[test]
fn test_cov3_sample_token_top_k_larger_than_vocab() {
    let logits = vec![0.1, 0.9, 0.3];
    let token = sample_token(&logits, 0.0, 100);
    assert_eq!(token, 1);
}

#[test]
fn test_cov3_sample_token_top_k_zero_with_temperature() {
    let logits = vec![0.1, 0.5, 0.3, 10.0, 0.2];
    let mut count_3 = 0;
    for _ in 0..50 {
        let token = sample_token(&logits, 0.5, 0);
        if token == 3 {
            count_3 += 1;
        }
    }
    assert!(count_3 > 30, "Token 3 with logit=10.0 should dominate, got {count_3}/50");
}

#[test]
fn test_cov3_sample_token_equal_logits() {
    let logits = vec![1.0; 5];
    let mut seen = std::collections::HashSet::new();
    for _ in 0..200 {
        let token = sample_token(&logits, 1.0, 0);
        seen.insert(token);
    }
    assert!(seen.len() >= 2, "Equal logits should produce diverse tokens");
}

#[test]
fn test_cov3_sample_token_two_elements_temperature() {
    let logits = vec![100.0, 0.0];
    for _ in 0..20 {
        let token = sample_token(&logits, 0.1, 0);
        assert_eq!(token, 0, "Very low temperature should always pick highest");
    }
}

#[test]
fn test_cov3_generate_config_debug() {
    let config = GenerateConfig::default();
    let debug = format!("{config:?}");
    assert!(debug.contains("GenerateConfig"));
    assert!(debug.contains("max_new_tokens"));
}

#[test]
fn test_cov3_generate_config_clone() {
    let config = GenerateConfig {
        max_new_tokens: 100,
        temperature: 0.5,
        top_k: 10,
        stop_tokens: vec![42, 99],
    };
    let cloned = config.clone();
    assert_eq!(cloned.max_new_tokens, 100);
    assert!((cloned.temperature - 0.5).abs() < f32::EPSILON);
    assert_eq!(cloned.top_k, 10);
    assert_eq!(cloned.stop_tokens, vec![42, 99]);
}

#[test]
fn test_cov3_instruct_config_debug() {
    let config = InstructConfig::default();
    let debug = format!("{config:?}");
    assert!(debug.contains("InstructConfig"));
    assert!(debug.contains("lora_rank"));
    assert!(debug.contains("learning_rate"));
}

#[test]
fn test_cov3_from_pretrained_missing_dir() {
    let model_config = TransformerConfig::tiny();
    let instruct_config = InstructConfig::default();
    let result = InstructPipeline::from_pretrained(
        Path::new("/tmp/nonexistent_model_dir_xyz_abc"),
        &model_config,
        instruct_config,
    );
    assert!(result.is_err());
}

#[test]
fn test_cov3_from_pretrained_no_tokenizer() {
    let dir = std::env::temp_dir().join("entrenar_cov3_no_tokenizer");
    let _ = std::fs::create_dir_all(&dir);
    let model_config = TransformerConfig::tiny();
    let instruct_config = InstructConfig::default();
    let result = InstructPipeline::from_pretrained(&dir, &model_config, instruct_config);
    assert!(result.is_err());
    let err_msg = format!("{}", result.err().expect("expected Err"));
    assert!(
        err_msg.contains("tokenizer") || err_msg.contains("safetensors") || err_msg.contains("No"),
        "Error should mention missing tokenizer or model files: {err_msg}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[should_panic(expected = "Pre-condition violated")]
fn test_cov3_from_apr_missing_file() {
    // The #[requires(apr_path.exists())] contract on from_apr triggers a
    // debug_assert! panic when the file does not exist (debug builds only).
    let model_config = TransformerConfig::tiny();
    let instruct_config = InstructConfig::default();
    let _result = InstructPipeline::from_apr(
        Path::new("/tmp/nonexistent_model.apr"),
        &model_config,
        instruct_config,
    );
}

#[test]
fn test_cov3_tokenize_empty_string() {
    let model_config = TransformerConfig::tiny();
    let pipeline = InstructPipeline::new(&model_config, InstructConfig::default());
    let tokens = pipeline.tokenize("");
    assert!(tokens.is_empty());
}

#[test]
fn test_cov3_tokenize_ascii_chars() {
    let model_config = TransformerConfig::tiny();
    let pipeline = InstructPipeline::new(&model_config, InstructConfig::default());
    let tokens = pipeline.tokenize("hello");
    assert_eq!(tokens, vec![104, 101, 108, 108, 111]);
}

#[test]
fn test_cov3_tokenize_unicode_multibyte() {
    let model_config = TransformerConfig::tiny();
    let pipeline = InstructPipeline::new(&model_config, InstructConfig::default());
    let tokens = pipeline.tokenize("\u{65e5}");
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0], 0xE6);
    assert_eq!(tokens[1], 0x97);
    assert_eq!(tokens[2], 0xA5);
}

#[test]
fn test_cov3_build_lora_layers_rank_and_alpha() {
    let model_config = TransformerConfig::tiny();
    let model = crate::transformer::Transformer::new(&model_config);
    let instruct_config =
        InstructConfig { lora_rank: 16, lora_alpha: 32.0, ..InstructConfig::default() };
    let layers = InstructPipeline::build_lora_layers(&model, &model_config, &instruct_config);
    for layer in &layers {
        assert_eq!(layer.rank(), 16);
    }
}

#[test]
fn test_cov3_build_lora_layers_q_v_alternating() {
    let model_config = TransformerConfig::tiny();
    let model = crate::transformer::Transformer::new(&model_config);
    let instruct_config = InstructConfig { lora_rank: 4, ..InstructConfig::default() };
    let layers = InstructPipeline::build_lora_layers(&model, &model_config, &instruct_config);
    let head_dim = model_config.hidden_size / model_config.num_attention_heads;
    let q_dim = model_config.num_attention_heads * head_dim;
    let v_dim = model_config.num_kv_heads * head_dim;
    for (i, layer) in layers.iter().enumerate() {
        if i % 2 == 0 {
            assert_eq!(layer.d_out(), q_dim, "Even layer {i} should be Q projection");
        } else {
            assert_eq!(layer.d_out(), v_dim, "Odd layer {i} should be V projection");
        }
        assert_eq!(layer.d_in(), model_config.hidden_size);
    }
}

#[test]
fn test_cov3_num_trainable_parameters_zero_rank() {
    let model_config = TransformerConfig::tiny();
    let instruct_config = InstructConfig { lora_rank: 0, ..InstructConfig::default() };
    let pipeline = InstructPipeline::new(&model_config, instruct_config);
    assert_eq!(pipeline.num_trainable_parameters(), 0);
}

#[test]
fn test_cov3_summary_no_nf4() {
    let model_config = TransformerConfig::tiny();
    let instruct_config = InstructConfig {
        lora_rank: 4,
        lora_alpha: 8.0,
        quantize_nf4: false,
        ..InstructConfig::default()
    };
    let pipeline = InstructPipeline::new(&model_config, instruct_config);
    let summary = pipeline.summary();
    assert!(summary.contains("4 LoRA layers"));
    assert!(summary.contains("rank=4"));
    assert!(summary.contains("alpha=8.0"));
    assert!(!summary.contains("NF4"));
}

#[test]
fn test_cov3_summary_with_nf4() {
    let model_config = TransformerConfig::tiny();
    let instruct_config =
        InstructConfig { lora_rank: 4, quantize_nf4: true, ..InstructConfig::default() };
    let pipeline = InstructPipeline::new(&model_config, instruct_config);
    let summary = pipeline.summary();
    assert!(summary.contains("NF4 QLoRA"));
}

#[test]
fn test_cov3_set_learning_rate_zero() {
    let model_config = TransformerConfig::tiny();
    let mut pipeline = InstructPipeline::new(&model_config, InstructConfig::default());
    pipeline.set_learning_rate(0.0);
    assert_eq!(pipeline.learning_rate(), 0.0);
}

#[test]
fn test_cov3_set_learning_rate_very_small() {
    let model_config = TransformerConfig::tiny();
    let mut pipeline = InstructPipeline::new(&model_config, InstructConfig::default());
    pipeline.set_learning_rate(1e-10);
    assert!((pipeline.learning_rate() - 1e-10).abs() < 1e-12);
}

#[test]
fn test_cov3_set_model_path_overwrite() {
    let model_config = TransformerConfig::tiny();
    let mut pipeline = InstructPipeline::new(&model_config, InstructConfig::default());
    pipeline.set_model_path(Path::new("/first/path"));
    pipeline.set_model_path(Path::new("/second/path"));
    assert_eq!(pipeline.model_dir.as_ref().unwrap().to_str().unwrap(), "/second/path");
}
