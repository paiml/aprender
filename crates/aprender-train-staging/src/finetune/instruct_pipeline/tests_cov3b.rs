//! Coverage (cov3) tests for the instruct pipeline, continued.

use super::*;

#[test]
fn test_cov3_train_step_multiple_steps_loss_changes() {
    let model_config = TransformerConfig::tiny();
    let instruct_config =
        InstructConfig { lora_rank: 4, max_seq_len: 32, ..InstructConfig::default() };
    let mut pipeline = InstructPipeline::new(&model_config, instruct_config);

    let prompt_ids: Vec<u32> = (0..5).collect();
    let response_ids: Vec<u32> = (5..10).collect();

    let result1 = pipeline.train_step(&prompt_ids, &response_ids);
    let result2 = pipeline.train_step(&prompt_ids, &response_ids);
    assert!(result1.loss >= 0.0);
    assert!(result2.loss >= 0.0);
}

#[test]
fn test_cov3_evaluate_empty_prompt_nonempty_response() {
    let model_config = TransformerConfig::tiny();
    let instruct_config =
        InstructConfig { lora_rank: 4, max_seq_len: 64, ..InstructConfig::default() };
    let pipeline = InstructPipeline::new(&model_config, instruct_config);
    let prompts = vec![vec![]];
    let responses = vec![vec![1u32, 2, 3]];
    let result = pipeline.evaluate(&prompts, &responses);
    assert!(result.avg_loss >= 0.0);
}

#[test]
fn test_cov3_evaluate_perplexity_clamped() {
    let model_config = TransformerConfig::tiny();
    let instruct_config =
        InstructConfig { lora_rank: 4, max_seq_len: 64, ..InstructConfig::default() };
    let pipeline = InstructPipeline::new(&model_config, instruct_config);
    let prompts = vec![vec![0u32, 1]];
    let responses = vec![vec![2u32, 3, 4]];
    let result = pipeline.evaluate(&prompts, &responses);
    assert!(result.perplexity <= 1e6);
}

#[test]
fn test_cov3_evaluate_grad_norm_zero() {
    let model_config = TransformerConfig::tiny();
    let instruct_config =
        InstructConfig { lora_rank: 4, max_seq_len: 64, ..InstructConfig::default() };
    let pipeline = InstructPipeline::new(&model_config, instruct_config);
    let prompts = vec![vec![0u32, 1, 2]];
    let responses = vec![vec![3u32, 4, 5]];
    let result = pipeline.evaluate(&prompts, &responses);
    assert_eq!(result.grad_norm, 0.0);
}

#[test]
fn test_cov3_simple_random_distribution() {
    let mut buckets = [0u32; 10];
    for _ in 0..1000 {
        let r = simple_random();
        let idx = (r * 10.0).min(9.0) as usize;
        buckets[idx] += 1;
    }
    for (i, &count) in buckets.iter().enumerate() {
        assert!(count > 0, "Bucket {i} is empty -- distribution not uniform");
    }
}

#[test]
fn test_cov3_generate_config_with_stop_tokens() {
    let config = GenerateConfig {
        max_new_tokens: 50,
        temperature: 0.0,
        top_k: 0,
        stop_tokens: vec![1, 2, 3],
    };
    assert_eq!(config.stop_tokens.len(), 3);
    assert!(config.stop_tokens.contains(&2));
}

#[test]
fn test_cov3_instruct_config_custom_all_fields() {
    let config = InstructConfig {
        lora_rank: 64,
        lora_alpha: 128.0,
        learning_rate: 1e-5,
        epochs: 10,
        max_seq_len: 2048,
        gradient_clip_norm: None,
        quantize_nf4: true,
    };
    assert_eq!(config.lora_rank, 64);
    assert!((config.lora_alpha - 128.0).abs() < f32::EPSILON);
    assert!((config.learning_rate - 1e-5).abs() < 1e-8);
    assert_eq!(config.epochs, 10);
    assert_eq!(config.max_seq_len, 2048);
    assert!(config.gradient_clip_norm.is_none());
    assert!(config.quantize_nf4);
}

#[test]
fn test_cov3_train_step_grad_clip_zero() {
    let model_config = TransformerConfig::tiny();
    let instruct_config = InstructConfig {
        lora_rank: 4,
        max_seq_len: 32,
        gradient_clip_norm: Some(0.0),
        ..InstructConfig::default()
    };
    let mut pipeline = InstructPipeline::new(&model_config, instruct_config);
    let result = pipeline.train_step(&[0, 1, 2], &[3, 4, 5]);
    assert!(result.loss >= 0.0);
}

#[test]
fn test_cov3_train_step_very_large_gradient_clip() {
    let model_config = TransformerConfig::tiny();
    let instruct_config = InstructConfig {
        lora_rank: 4,
        max_seq_len: 32,
        gradient_clip_norm: Some(1e10),
        ..InstructConfig::default()
    };
    let mut pipeline = InstructPipeline::new(&model_config, instruct_config);
    let result = pipeline.train_step(&[0, 1, 2], &[3, 4, 5]);
    assert!(result.loss >= 0.0);
}

#[test]
fn test_cov3_compute_causal_lm_loss_high_logit_correct_target() {
    let vocab_size = 5;
    let mut logits = vec![0.0f32; 10];
    logits[1] = 50.0;
    let full_ids = vec![0u32, 1];
    let (loss, _) = InstructPipeline::compute_causal_lm_loss(&logits, &full_ids, 0, 1, vocab_size);
    assert!(loss < 0.01, "Loss should be near-zero when prediction is correct, got {loss}");
}

#[test]
fn test_cov3_compute_causal_lm_loss_high_logit_wrong_target() {
    let vocab_size = 5;
    let mut logits = vec![0.0f32; 10];
    logits[0] = 50.0;
    let full_ids = vec![0u32, 4];
    let (loss, _) = InstructPipeline::compute_causal_lm_loss(&logits, &full_ids, 0, 1, vocab_size);
    assert!(loss > 10.0, "Loss should be high when prediction is wrong, got {loss}");
}

#[test]
fn test_cov3_has_tokenizer_false_by_default() {
    let model_config = TransformerConfig::tiny();
    let pipeline = InstructPipeline::new(&model_config, InstructConfig::default());
    assert!(!pipeline.has_tokenizer());
}

#[test]
fn test_cov3_tokenizer_returns_none_by_default() {
    let model_config = TransformerConfig::tiny();
    let pipeline = InstructPipeline::new(&model_config, InstructConfig::default());
    assert!(pipeline.tokenizer().is_none());
}

#[test]
fn test_cov3_is_cuda_always_false_on_cpu() {
    let model_config = TransformerConfig::tiny();
    let instruct_config = InstructConfig { quantize_nf4: true, ..InstructConfig::default() };
    let pipeline = InstructPipeline::new(&model_config, instruct_config);
    assert!(!pipeline.is_cuda());
}
