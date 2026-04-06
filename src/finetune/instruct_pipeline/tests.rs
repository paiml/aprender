//! Tests for the instruct pipeline.

use super::*;

#[test]
fn test_instruct_pipeline_new() {
    let model_config = TransformerConfig::tiny();
    let instruct_config = InstructConfig { lora_rank: 4, ..InstructConfig::default() };
    let pipeline = InstructPipeline::new(&model_config, instruct_config);
    // 2 layers * 2 (Q+V) = 4 LoRA layers for tiny config
    assert_eq!(pipeline.lora_layers.len(), 4);
}

#[test]
fn test_instruct_train_step() {
    let model_config = TransformerConfig::tiny();
    let instruct_config =
        InstructConfig { lora_rank: 4, max_seq_len: 64, ..InstructConfig::default() };
    let mut pipeline = InstructPipeline::new(&model_config, instruct_config);

    let prompt_ids: Vec<u32> = (0..10).collect();
    let response_ids: Vec<u32> = (10..20).collect();

    let result = pipeline.train_step(&prompt_ids, &response_ids);
    assert!(result.loss >= 0.0);
    assert_eq!(result.num_response_tokens, 10);
    assert!(result.perplexity >= 1.0);
}

#[test]
fn test_instruct_evaluate() {
    let model_config = TransformerConfig::tiny();
    let instruct_config =
        InstructConfig { lora_rank: 4, max_seq_len: 64, ..InstructConfig::default() };
    let pipeline = InstructPipeline::new(&model_config, instruct_config);

    let prompts = vec![vec![0u32, 1, 2, 3, 4]];
    let responses = vec![vec![5u32, 6, 7, 8, 9]];

    let result = pipeline.evaluate(&prompts, &responses);
    assert!(result.avg_loss >= 0.0);
    assert_eq!(result.total_response_tokens, 5);
}

#[test]
fn test_empty_response_noop() {
    let model_config = TransformerConfig::tiny();
    let instruct_config =
        InstructConfig { lora_rank: 4, max_seq_len: 64, ..InstructConfig::default() };
    let mut pipeline = InstructPipeline::new(&model_config, instruct_config);

    let result = pipeline.train_step(&[0, 1, 2], &[]);
    assert_eq!(result.loss, 0.0);
    assert_eq!(result.num_response_tokens, 0);
}

#[test]
fn test_generate_config_default() {
    let config = GenerateConfig::default();
    assert_eq!(config.max_new_tokens, 256);
    assert!((config.temperature - 0.7).abs() < f32::EPSILON);
    assert_eq!(config.top_k, 50);
    assert!(config.stop_tokens.is_empty());
}

#[test]
fn test_generate_config_greedy() {
    let config = GenerateConfig::greedy(128);
    assert_eq!(config.max_new_tokens, 128);
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_k, 0);
}

#[test]
fn test_sample_token_greedy() {
    let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
    let token = sample_token(&logits, 0.0, 0);
    assert_eq!(token, 3); // index of 0.9 (highest)
}

#[test]
fn test_sample_token_greedy_top_k_1() {
    let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
    let token = sample_token(&logits, 1.0, 1);
    assert_eq!(token, 3); // top-1 is always argmax
}

#[test]
fn test_sample_token_temperature_sampling() {
    // With very high temperature, distribution flattens — token 3 shouldn't always win
    let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
    let mut seen = std::collections::HashSet::new();
    for _ in 0..100 {
        let token = sample_token(&logits, 10.0, 0);
        seen.insert(token);
    }
    // With temp=10.0, we should see multiple different tokens
    assert!(seen.len() > 1, "Expected diversity with high temperature");
}

#[test]
fn test_generate_no_tokenizer_errors() {
    let model_config = TransformerConfig::tiny();
    let instruct_config = InstructConfig { lora_rank: 4, ..InstructConfig::default() };
    let pipeline = InstructPipeline::new(&model_config, instruct_config);
    // Pipeline created with new() has no tokenizer
    let result = pipeline.generate("hello", &GenerateConfig::greedy(10));
    assert!(result.is_err());
}

#[test]
fn test_simple_random_range() {
    for _ in 0..1000 {
        let r = simple_random();
        assert!((0.0..1.0).contains(&r), "Random value {r} out of [0, 1) range");
    }
}

#[test]
fn test_instruct_config_default() {
    let config = InstructConfig::default();
    assert_eq!(config.lora_rank, 16);
    assert!((config.lora_alpha - 32.0).abs() < f32::EPSILON);
    assert!((config.learning_rate - 2e-4).abs() < f32::EPSILON);
    assert_eq!(config.epochs, 3);
    assert_eq!(config.max_seq_len, 512);
    assert_eq!(config.gradient_clip_norm, Some(1.0));
    assert!(!config.quantize_nf4);
}

#[test]
fn test_instruct_step_result_fields() {
    let result = InstructStepResult { loss: 2.5, num_response_tokens: 10, perplexity: 12.18 };
    assert!((result.loss - 2.5).abs() < f32::EPSILON);
    assert_eq!(result.num_response_tokens, 10);
    assert!((result.perplexity - 12.18).abs() < 0.01);
}

#[test]
fn test_instruct_batch_result_fields() {
    let result = InstructBatchResult {
        avg_loss: 1.5,
        total_response_tokens: 100,
        perplexity: 4.48,
        grad_norm: 0.5,
    };
    assert!((result.avg_loss - 1.5).abs() < f32::EPSILON);
    assert_eq!(result.total_response_tokens, 100);
    assert!((result.perplexity - 4.48).abs() < 0.01);
    assert!((result.grad_norm - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_tokenize_byte_fallback_no_tokenizer() {
    let model_config = TransformerConfig::tiny();
    let instruct_config = InstructConfig { lora_rank: 4, ..InstructConfig::default() };
    let pipeline = InstructPipeline::new(&model_config, instruct_config);
    assert!(!pipeline.has_tokenizer());
    let tokens = pipeline.tokenize("AB");
    // Byte-level fallback: 'A' = 65, 'B' = 66
    assert_eq!(tokens, vec![65, 66]);
}

#[test]
fn test_tokenize_byte_fallback_utf8() {
    let model_config = TransformerConfig::tiny();
    let instruct_config = InstructConfig { lora_rank: 4, ..InstructConfig::default() };
    let pipeline = InstructPipeline::new(&model_config, instruct_config);
    let tokens = pipeline.tokenize("\u{00e9}"); // 'e' = 2 UTF-8 bytes: 0xC3, 0xA9
    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens[0], 0xC3);
    assert_eq!(tokens[1], 0xA9);
}

#[test]
fn test_num_trainable_parameters() {
    let model_config = TransformerConfig::tiny();
    let instruct_config = InstructConfig { lora_rank: 4, ..InstructConfig::default() };
    let pipeline = InstructPipeline::new(&model_config, instruct_config);
    let count = pipeline.num_trainable_parameters();
    // 4 LoRA layers * 2 * rank(4) * 1 = 32
    assert_eq!(count, 32);
}

#[test]
fn test_set_and_get_learning_rate() {
    let model_config = TransformerConfig::tiny();
    let instruct_config =
        InstructConfig { lora_rank: 4, learning_rate: 1e-3, ..InstructConfig::default() };
    let mut pipeline = InstructPipeline::new(&model_config, instruct_config);
    assert!((pipeline.learning_rate() - 1e-3).abs() < 1e-6);
    pipeline.set_learning_rate(5e-4);
    assert!((pipeline.learning_rate() - 5e-4).abs() < 1e-6);
}

#[test]
fn test_set_model_path() {
    let model_config = TransformerConfig::tiny();
    let instruct_config = InstructConfig { lora_rank: 4, ..InstructConfig::default() };
    let mut pipeline = InstructPipeline::new(&model_config, instruct_config);
    assert!(pipeline.model_dir.is_none());
    pipeline.set_model_path(std::path::Path::new("/tmp/test-model"));
    assert_eq!(
        pipeline.model_dir.as_ref().map(|p| p.to_str().unwrap_or("")),
        Some("/tmp/test-model")
    );
}

#[test]
fn test_is_cuda_false_without_gpu() {
    let model_config = TransformerConfig::tiny();
    let instruct_config = InstructConfig { lora_rank: 4, ..InstructConfig::default() };
    let pipeline = InstructPipeline::new(&model_config, instruct_config);
    assert!(!pipeline.is_cuda());
}

#[test]
fn test_gpu_name_none_without_gpu() {
    let model_config = TransformerConfig::tiny();
    let instruct_config = InstructConfig { lora_rank: 4, ..InstructConfig::default() };
    let pipeline = InstructPipeline::new(&model_config, instruct_config);
    assert!(pipeline.gpu_name().is_none());
}

#[test]
fn test_gpu_total_memory_none_without_gpu() {
    let model_config = TransformerConfig::tiny();
    let instruct_config = InstructConfig { lora_rank: 4, ..InstructConfig::default() };
    let pipeline = InstructPipeline::new(&model_config, instruct_config);
    assert!(pipeline.gpu_total_memory().is_none());
}

#[test]
fn test_summary_format() {
    let model_config = TransformerConfig::tiny();
    let instruct_config =
        InstructConfig { lora_rank: 8, lora_alpha: 16.0, ..InstructConfig::default() };
    let pipeline = InstructPipeline::new(&model_config, instruct_config);
    let summary = pipeline.summary();
    assert!(summary.contains("InstructPipeline"));
    assert!(summary.contains("rank=8"));
    assert!(summary.contains("alpha=16.0"));
    assert!(!summary.contains("NF4 QLoRA"));
}

#[test]
fn test_summary_nf4_label() {
    let model_config = TransformerConfig::tiny();
    let instruct_config =
        InstructConfig { lora_rank: 8, quantize_nf4: true, ..InstructConfig::default() };
    let pipeline = InstructPipeline::new(&model_config, instruct_config);
    let summary = pipeline.summary();
    assert!(summary.contains("NF4 QLoRA"));
}

#[test]
fn test_tokenizer_none_for_new_pipeline() {
    let model_config = TransformerConfig::tiny();
    let instruct_config = InstructConfig::default();
    let pipeline = InstructPipeline::new(&model_config, instruct_config);
    assert!(pipeline.tokenizer().is_none());
}

#[test]
fn test_train_step_truncation() {
    let model_config = TransformerConfig::tiny();
    let instruct_config =
        InstructConfig { lora_rank: 4, max_seq_len: 16, ..InstructConfig::default() };
    let mut pipeline = InstructPipeline::new(&model_config, instruct_config);

    let prompt_ids: Vec<u32> = (0..10).collect();
    let response_ids: Vec<u32> = (10..30).collect(); // total = 30 > 16

    let result = pipeline.train_step(&prompt_ids, &response_ids);
    assert!(result.loss >= 0.0);
}

#[test]
fn test_train_step_short_sequence() {
    let model_config = TransformerConfig::tiny();
    let instruct_config =
        InstructConfig { lora_rank: 4, max_seq_len: 64, ..InstructConfig::default() };
    let mut pipeline = InstructPipeline::new(&model_config, instruct_config);

    let result = pipeline.train_step(&[0], &[1]);
    assert!(result.loss >= 0.0);
    assert_eq!(result.num_response_tokens, 1);
}

#[test]
fn test_evaluate_empty_batch() {
    let model_config = TransformerConfig::tiny();
    let instruct_config =
        InstructConfig { lora_rank: 4, max_seq_len: 64, ..InstructConfig::default() };
    let pipeline = InstructPipeline::new(&model_config, instruct_config);

    let result = pipeline.evaluate(&[], &[]);
    assert_eq!(result.avg_loss, 0.0);
    assert_eq!(result.total_response_tokens, 0);
    assert_eq!(result.grad_norm, 0.0);
}

#[test]
fn test_evaluate_skips_empty_responses() {
    let model_config = TransformerConfig::tiny();
    let instruct_config =
        InstructConfig { lora_rank: 4, max_seq_len: 64, ..InstructConfig::default() };
    let pipeline = InstructPipeline::new(&model_config, instruct_config);

    let prompts = vec![vec![0u32, 1, 2], vec![3u32, 4, 5]];
    let responses = vec![vec![], vec![6u32, 7, 8]];

    let result = pipeline.evaluate(&prompts, &responses);
    assert_eq!(result.total_response_tokens, 3);
}

#[test]
fn test_evaluate_truncation() {
    let model_config = TransformerConfig::tiny();
    let instruct_config =
        InstructConfig { lora_rank: 4, max_seq_len: 10, ..InstructConfig::default() };
    let pipeline = InstructPipeline::new(&model_config, instruct_config);

    let prompts = vec![vec![0u32; 8]];
    let responses = vec![vec![1u32; 8]]; // total 16, will be truncated to 10

    let result = pipeline.evaluate(&prompts, &responses);
    assert!(result.avg_loss >= 0.0);
}

#[test]
fn test_compute_causal_lm_loss_basic() {
    let vocab_size = 5;
    let seq_len = 3;
    let mut logits = vec![0.0f32; seq_len * vocab_size];
    logits[1] = 10.0;
    logits[vocab_size + 2] = 10.0;

    let full_ids = vec![0u32, 1, 2];
    let (loss, grad) =
        InstructPipeline::compute_causal_lm_loss(&logits, &full_ids, 0, 2, vocab_size);
    assert!(loss >= 0.0);
    assert!(loss < 1.0, "Loss should be low when logits match targets, got {loss}");
    assert_eq!(grad.len(), seq_len * vocab_size);
}

#[test]
fn test_compute_causal_lm_loss_empty_range() {
    let vocab_size = 5;
    let logits = vec![0.0f32; 15];
    let full_ids = vec![0u32, 1, 2];
    let (loss, grad) =
        InstructPipeline::compute_causal_lm_loss(&logits, &full_ids, 2, 2, vocab_size);
    assert_eq!(loss, 0.0);
    assert!(grad.iter().all(|&v| v == 0.0));
}

#[test]
fn test_compute_causal_lm_loss_target_out_of_vocab() {
    let vocab_size = 5;
    let logits = vec![0.0f32; 10];
    let full_ids = vec![0u32, 100]; // target 100 >= vocab_size 5
    let (loss, _grad) =
        InstructPipeline::compute_causal_lm_loss(&logits, &full_ids, 0, 1, vocab_size);
    assert_eq!(loss, 0.0);
}

#[test]
fn test_build_lora_layers_count() {
    let model_config = TransformerConfig::tiny();
    let model = crate::transformer::Transformer::new(&model_config);
    let instruct_config = InstructConfig { lora_rank: 8, ..InstructConfig::default() };
    let layers = InstructPipeline::build_lora_layers(&model, &model_config, &instruct_config);
    assert_eq!(layers.len(), model_config.num_hidden_layers * 2);
}

#[test]
fn test_sample_token_with_top_k() {
    let logits = vec![0.0, 0.0, 0.0, 100.0, 99.0];
    for _ in 0..50 {
        let token = sample_token(&logits, 1.0, 2);
        assert!(token == 3 || token == 4, "Expected token 3 or 4 with top_k=2, got {token}");
    }
}

#[test]
fn test_sample_token_empty_logits() {
    let logits: Vec<f32> = vec![];
    let token = sample_token(&logits, 0.0, 0);
    assert_eq!(token, 0);
}

#[test]
fn test_sample_token_single_element() {
    let logits = vec![1.0];
    let token = sample_token(&logits, 0.0, 0);
    assert_eq!(token, 0);
}

#[test]
fn test_train_step_with_no_gradient_clip() {
    let model_config = TransformerConfig::tiny();
    let instruct_config = InstructConfig {
        lora_rank: 4,
        max_seq_len: 64,
        gradient_clip_norm: None,
        ..InstructConfig::default()
    };
    let mut pipeline = InstructPipeline::new(&model_config, instruct_config);

    let prompt_ids: Vec<u32> = (0..5).collect();
    let response_ids: Vec<u32> = (5..10).collect();

    let result = pipeline.train_step(&prompt_ids, &response_ids);
    assert!(result.loss >= 0.0);
}

#[test]
fn test_generate_chat_no_tokenizer_errors() {
    let model_config = TransformerConfig::tiny();
    let instruct_config = InstructConfig { lora_rank: 4, ..InstructConfig::default() };
    let pipeline = InstructPipeline::new(&model_config, instruct_config);
    let result = pipeline.generate_chat("system", "hello", &GenerateConfig::greedy(5));
    assert!(result.is_err());
}

#[test]
fn test_perplexity_clamped() {
    let model_config = TransformerConfig::tiny();
    let instruct_config =
        InstructConfig { lora_rank: 4, max_seq_len: 64, ..InstructConfig::default() };
    let mut pipeline = InstructPipeline::new(&model_config, instruct_config);

    let result = pipeline.train_step(&[0, 1, 2], &[3, 4, 5]);
    assert!(result.perplexity <= 1e6);
}

#[test]
fn test_evaluate_multiple_samples() {
    let model_config = TransformerConfig::tiny();
    let instruct_config =
        InstructConfig { lora_rank: 4, max_seq_len: 64, ..InstructConfig::default() };
    let pipeline = InstructPipeline::new(&model_config, instruct_config);

    let prompts = vec![vec![0u32, 1, 2], vec![3u32, 4, 5], vec![6u32, 7, 8]];
    let responses = vec![vec![10u32, 11], vec![12u32, 13, 14], vec![15u32]];

    let result = pipeline.evaluate(&prompts, &responses);
    assert_eq!(result.total_response_tokens, 6);
    assert!(result.avg_loss >= 0.0);
    assert!(result.perplexity >= 1.0);
}

#[test]
fn test_instruct_config_clone() {
    let config = InstructConfig {
        lora_rank: 32,
        lora_alpha: 64.0,
        learning_rate: 1e-5,
        epochs: 5,
        max_seq_len: 1024,
        gradient_clip_norm: Some(2.0),
        quantize_nf4: true,
    };
    let cloned = config.clone();
    assert_eq!(cloned.lora_rank, 32);
    assert!((cloned.lora_alpha - 64.0).abs() < f32::EPSILON);
    assert_eq!(cloned.epochs, 5);
    assert_eq!(cloned.max_seq_len, 1024);
    assert_eq!(cloned.gradient_clip_norm, Some(2.0));
    assert!(cloned.quantize_nf4);
}

#[test]
fn test_step_result_clone_and_debug() {
    let result = InstructStepResult { loss: 1.5, num_response_tokens: 42, perplexity: 4.48 };
    let cloned = result.clone();
    assert_eq!(cloned.num_response_tokens, 42);
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("InstructStepResult"));
}

#[test]
fn test_batch_result_clone_and_debug() {
    let result = InstructBatchResult {
        avg_loss: 0.5,
        total_response_tokens: 200,
        perplexity: 1.65,
        grad_norm: 0.8,
    };
    let cloned = result.clone();
    assert_eq!(cloned.total_response_tokens, 200);
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("InstructBatchResult"));
}
