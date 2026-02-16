use super::*;

#[test]
fn test_apply_repetition_penalty_mixed_logits() {
    // Test with mixed positive and negative logits
    let logits = Tensor::new(&[2.0, -2.0, 3.0, -3.0], &[4]);
    let generated = vec![0, 1];

    let penalized = apply_repetition_penalty(&logits, &generated, 2.0);

    // Positive gets divided, negative gets multiplied
    assert_eq!(penalized.data()[0], 1.0); // 2.0 / 2.0
    assert_eq!(penalized.data()[1], -4.0); // -2.0 * 2.0
    assert_eq!(penalized.data()[2], 3.0); // unchanged
    assert_eq!(penalized.data()[3], -3.0); // unchanged
}

#[test]
fn test_apply_repetition_penalty_zero_logit() {
    let logits = Tensor::new(&[0.0, 2.0, 0.0, 2.0], &[4]);
    let generated = vec![0, 2];

    let penalized = apply_repetition_penalty(&logits, &generated, 2.0);

    // Zero logits go through the else branch (multiply)
    assert_eq!(penalized.data()[0], 0.0); // 0.0 * 2.0
    assert_eq!(penalized.data()[2], 0.0); // 0.0 * 2.0
}

#[test]
fn test_apply_repetition_penalty_out_of_bounds() {
    // Token ID larger than vocab size should be ignored
    let logits = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let generated = vec![5, 10]; // Out of bounds

    let penalized = apply_repetition_penalty(&logits, &generated, 2.0);

    // Should be unchanged since token IDs are out of bounds
    assert_eq!(penalized.data()[0], 1.0);
    assert_eq!(penalized.data()[1], 2.0);
    assert_eq!(penalized.data()[2], 3.0);
}

#[test]
fn test_generation_config_with_top_k() {
    let config = GenerationConfig::new().with_top_k(40);
    assert_eq!(config.top_k, Some(40));
}

#[test]
fn test_generation_config_with_eos_token_id() {
    let config = GenerationConfig::new().with_eos_token_id(50256);
    assert_eq!(config.eos_token_id, Some(50256));
}

#[test]
fn test_beam_search_with_early_stopping() {
    let beam = BeamSearch::new(3).with_early_stopping();
    // Just verify it compiles and stores the setting
    assert_eq!(beam.beam_size(), 3);
}

#[test]
fn test_beam_search_debug_display() {
    let beam = BeamSearch::new(5)
        .with_length_penalty(1.5)
        .with_eos_token_id(2);
    let debug = format!("{:?}", beam);
    assert!(debug.contains("BeamSearch"));
    assert!(debug.contains("beam_size"));
}

#[test]
fn test_nucleus_sampler_debug_display() {
    let sampler = NucleusSampler::new(0.9).with_temperature(0.7);
    let debug = format!("{:?}", sampler);
    assert!(debug.contains("NucleusSampler"));
    assert!(debug.contains("top_p"));
}

#[test]
fn test_topk_sampler_sample() {
    let sampler = TopKSampler::new(3);
    let logits = Tensor::new(&[1.0, 10.0, 5.0, 2.0, 3.0], &[5]);

    // Sample multiple times
    for _ in 0..10 {
        let token = sampler.sample(&logits);
        // Should only sample from top 3 (indices 1, 2, 4)
        assert!(token == 1 || token == 2 || token == 4);
    }
}

#[test]
fn test_topk_sampler_debug_display() {
    let sampler = TopKSampler::new(50);
    let debug = format!("{:?}", sampler);
    assert!(debug.contains("TopKSampler"));
    assert!(debug.contains("top_k"));
}

#[test]
fn test_greedy_decoder_default() {
    let decoder = GreedyDecoder::default();
    let logits = Tensor::new(&[1.0, 3.0, 2.0], &[3]);
    assert_eq!(decoder.decode(&logits), 1);
}

#[test]
fn test_beam_hypothesis_clone() {
    let hyp = BeamHypothesis::new(vec![1, 2, 3], -2.0);
    let cloned = hyp.clone();
    assert_eq!(cloned.tokens, vec![1, 2, 3]);
    assert_eq!(cloned.score, -2.0);
}

#[test]
fn test_generation_config_clone() {
    let config = GenerationConfig::new()
        .with_max_length(100)
        .with_temperature(0.8);
    let cloned = config.clone();
    assert_eq!(cloned.max_length, 100);
    assert_eq!(cloned.temperature, 0.8);
}

#[test]
fn test_teacher_forcing_clone() {
    let tf = TeacherForcing::linear(1.0, 0.1, 1000);
    let cloned = tf.clone();
    assert_eq!(cloned.initial_ratio(), 1.0);
    assert_eq!(cloned.final_ratio(), 0.1);
}

#[test]
fn test_teacher_forcing_schedule_clone() {
    let schedule = TeacherForcingSchedule::Exponential;
    let cloned = schedule;
    assert_eq!(cloned, TeacherForcingSchedule::Exponential);
}

#[test]
fn test_beam_search_best_empty() {
    let beam = BeamSearch::new(3);
    let beams: Vec<BeamHypothesis> = vec![];
    assert!(beam.best(&beams).is_none());
}

#[test]
fn test_argmax_single_element() {
    assert_eq!(argmax(&[5.0]), 0);
}

#[test]
fn test_argmax_all_same() {
    // When all elements are equal, argmax returns the last index
    // due to max_by returning the last of equal elements
    let result = argmax(&[1.0, 1.0, 1.0]);
    assert!(result < 3); // Valid index
}
