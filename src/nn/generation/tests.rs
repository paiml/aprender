use super::*;

// ========================================================================
// Generation Config Tests
// ========================================================================

#[test]
fn test_generation_config_default() {
    let config = GenerationConfig::default();
    assert_eq!(config.max_length, 50);
    assert_eq!(config.temperature, 1.0);
    assert_eq!(config.num_beams, 1);
}

#[test]
fn test_generation_config_builder() {
    let config = GenerationConfig::new()
        .with_max_length(100)
        .with_temperature(0.7)
        .with_top_p(0.9)
        .with_num_beams(5);

    assert_eq!(config.max_length, 100);
    assert_eq!(config.temperature, 0.7);
    assert_eq!(config.top_p, Some(0.9));
    assert_eq!(config.num_beams, 5);
}

// ========================================================================
// Beam Search Tests
// ========================================================================

#[test]
fn test_beam_search_init() {
    let beam = BeamSearch::new(5);
    let beams = beam.init(0);

    assert_eq!(beams.len(), 1);
    assert_eq!(beams[0].tokens, vec![0]);
    assert_eq!(beams[0].score, 0.0);
}

#[test]
fn test_beam_search_step() {
    let beam = BeamSearch::new(3);
    let beams = beam.init(0);

    // Log probabilities for 5 tokens
    let log_probs = Tensor::new(&[-0.5, -1.0, -1.5, -2.0, -2.5], &[5]);

    let new_beams = beam.step(&log_probs, &beams);

    // Should keep top 3 beams
    assert_eq!(new_beams.len(), 3);

    // Best beam should have lowest negative log prob (first token)
    assert_eq!(new_beams[0].tokens, vec![0, 0]);
    assert!((new_beams[0].score - (-0.5)).abs() < 1e-6);
}

#[test]
fn test_beam_search_eos() {
    let beam = BeamSearch::new(3).with_eos_token_id(2);
    let beams = beam.init(0);

    let log_probs = Tensor::new(&[-1.0, -2.0, -0.5], &[3]); // EOS has highest prob

    let new_beams = beam.step(&log_probs, &beams);

    // Beam that selected EOS should be marked done
    let eos_beam = new_beams.iter().find(|b| b.tokens.last() == Some(&2));
    assert!(eos_beam.is_some());
    assert!(eos_beam.unwrap().is_done);
}

#[test]
fn test_beam_search_best() {
    let beam = BeamSearch::new(3);

    let beams = vec![
        BeamHypothesis::new(vec![0, 1, 2], -3.0),
        BeamHypothesis::new(vec![0, 2], -1.5),
        BeamHypothesis::new(vec![0, 3, 4, 5], -4.0),
    ];

    let best = beam.best(&beams).unwrap();
    // With length_penalty=1.0, normalized scores are:
    // -3.0/3 = -1.0, -1.5/2 = -0.75, -4.0/4 = -1.0
    // Best is -0.75 (second beam)
    assert_eq!(best.tokens, vec![0, 2]);
}

#[test]
fn test_beam_search_length_penalty() {
    let beam = BeamSearch::new(3).with_length_penalty(2.0);

    let beams = vec![
        BeamHypothesis::new(vec![0, 1, 2], -3.0),
        BeamHypothesis::new(vec![0, 2], -1.5),
    ];

    // With length_penalty=2.0:
    // -3.0/3^2 = -0.33, -1.5/2^2 = -0.375
    // Best is -0.33 (first beam, longer)
    let best = beam.best(&beams).unwrap();
    assert_eq!(best.tokens, vec![0, 1, 2]);
}

#[test]
fn test_beam_search_all_done() {
    let beam = BeamSearch::new(2);

    let mut beams = vec![
        BeamHypothesis::new(vec![0, 1], -1.0),
        BeamHypothesis::new(vec![0, 2], -2.0),
    ];

    assert!(!beam.all_done(&beams));

    beams[0].is_done = true;
    beams[1].is_done = true;

    assert!(beam.all_done(&beams));
}

#[test]
fn test_beam_search_getters() {
    let beam = BeamSearch::new(5).with_length_penalty(1.5);
    assert_eq!(beam.beam_size(), 5);
    assert_eq!(beam.length_penalty(), 1.5);
}

// ========================================================================
// Nucleus Sampler Tests
// ========================================================================

#[test]
fn test_nucleus_sampler_creation() {
    let sampler = NucleusSampler::new(0.95);
    assert_eq!(sampler.top_p(), 0.95);
    assert_eq!(sampler.temperature(), 1.0);
}

#[test]
fn test_nucleus_sampler_filter() {
    // Create logits where one token dominates
    let logits = Tensor::new(&[10.0, 1.0, 1.0, 1.0, 1.0], &[5]);

    let sampler = NucleusSampler::new(0.9);
    let filtered = sampler.filter(&logits);

    // The dominant token should be kept
    assert!(filtered.data()[0] > f32::NEG_INFINITY);
}

#[test]
fn test_nucleus_sampler_with_temperature() {
    let sampler = NucleusSampler::new(0.9).with_temperature(0.5);
    assert_eq!(sampler.temperature(), 0.5);
}

#[test]
fn test_nucleus_sampler_min_tokens() {
    let sampler = NucleusSampler::new(0.1).with_min_tokens_to_keep(3);

    let logits = Tensor::new(&[10.0, 1.0, 0.5, 0.1, 0.05], &[5]);
    let filtered = sampler.filter(&logits);

    // Should keep at least 3 tokens even with low top_p
    let valid_count = filtered
        .data()
        .iter()
        .filter(|&&x| x > f32::NEG_INFINITY)
        .count();
    assert!(valid_count >= 3);
}

#[test]
fn test_nucleus_sampler_sample_returns_valid() {
    let sampler = NucleusSampler::new(0.95);
    let logits = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]);

    // Sample multiple times and check all are valid
    for _ in 0..10 {
        let token = sampler.sample(&logits);
        assert!(token < 5);
    }
}

#[test]
#[should_panic(expected = "top_p must be in (0.0, 1.0]")]
fn test_nucleus_sampler_invalid_p() {
    let _sampler = NucleusSampler::new(0.0);
}

// ========================================================================
// Top-K Sampler Tests
// ========================================================================

#[test]
fn test_topk_sampler_creation() {
    let sampler = TopKSampler::new(50);
    assert_eq!(sampler.top_k(), 50);
}

#[test]
fn test_topk_sampler_filter() {
    let logits = Tensor::new(&[1.0, 5.0, 2.0, 4.0, 3.0], &[5]);

    let sampler = TopKSampler::new(3);
    let filtered = sampler.filter(&logits);

    // Should keep indices 1, 3, 4 (values 5, 4, 3)
    let valid_count = filtered
        .data()
        .iter()
        .filter(|&&x| x > f32::NEG_INFINITY)
        .count();
    assert_eq!(valid_count, 3);

    // Top tokens should be kept
    assert!(filtered.data()[1] > f32::NEG_INFINITY); // 5.0
    assert!(filtered.data()[3] > f32::NEG_INFINITY); // 4.0
    assert!(filtered.data()[4] > f32::NEG_INFINITY); // 3.0

    // Others should be -inf
    assert!(filtered.data()[0] == f32::NEG_INFINITY);
    assert!(filtered.data()[2] == f32::NEG_INFINITY);
}

#[test]
fn test_topk_sampler_k_larger_than_vocab() {
    let logits = Tensor::new(&[1.0, 2.0, 3.0], &[3]);

    let sampler = TopKSampler::new(10); // k > vocab_size
    let filtered = sampler.filter(&logits);

    // Should keep all tokens
    let valid_count = filtered
        .data()
        .iter()
        .filter(|&&x| x > f32::NEG_INFINITY)
        .count();
    assert_eq!(valid_count, 3);
}

#[test]
fn test_topk_sampler_with_temperature() {
    let sampler = TopKSampler::new(50).with_temperature(0.7);

    let logits = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let filtered = sampler.filter(&logits);

    // Values should be scaled by temperature
    assert!((filtered.data()[2] - 3.0 / 0.7).abs() < 1e-5);
}

// ========================================================================
// Greedy Decoder Tests
// ========================================================================

#[test]
fn test_greedy_decoder() {
    let decoder = GreedyDecoder::new();
    let logits = Tensor::new(&[1.0, 5.0, 2.0, 4.0, 3.0], &[5]);

    let token = decoder.decode(&logits);
    assert_eq!(token, 1); // Index of max value (5.0)
}

// ========================================================================
// Helper Function Tests
// ========================================================================

#[test]
fn test_apply_repetition_penalty() {
    let logits = Tensor::new(&[2.0, 2.0, 2.0, 2.0], &[4]);
    let generated = vec![0, 2];

    let penalized = apply_repetition_penalty(&logits, &generated, 2.0);

    // Tokens 0 and 2 should be penalized
    assert_eq!(penalized.data()[0], 1.0); // 2.0 / 2.0
    assert_eq!(penalized.data()[1], 2.0); // unchanged
    assert_eq!(penalized.data()[2], 1.0); // 2.0 / 2.0
    assert_eq!(penalized.data()[3], 2.0); // unchanged
}

#[test]
fn test_apply_temperature() {
    let logits = Tensor::new(&[1.0, 2.0, 3.0], &[3]);

    let scaled = apply_temperature(&logits, 2.0);

    assert!((scaled.data()[0] - 0.5).abs() < 1e-6);
    assert!((scaled.data()[1] - 1.0).abs() < 1e-6);
    assert!((scaled.data()[2] - 1.5).abs() < 1e-6);
}

#[test]
fn test_argmax() {
    assert_eq!(argmax(&[1.0, 5.0, 2.0, 4.0]), 1);
    assert_eq!(argmax(&[5.0, 4.0, 3.0, 2.0]), 0);
    assert_eq!(argmax(&[-1.0, -2.0, -0.5]), 2);
}

#[test]
fn test_sample_from_logits_returns_valid() {
    let logits = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]);

    for _ in 0..20 {
        let token = sample_from_logits(&logits);
        assert!(token < 5);
    }
}

#[test]
fn test_beam_hypothesis_normalized_score() {
    let hyp = BeamHypothesis::new(vec![0, 1, 2, 3], -4.0);

    // length_penalty = 1.0: -4.0 / 4 = -1.0
    assert!((hyp.normalized_score(1.0) - (-1.0)).abs() < 1e-6);

    // length_penalty = 2.0: -4.0 / 4^2 = -0.25
    assert!((hyp.normalized_score(2.0) - (-0.25)).abs() < 1e-6);
}

// ========================================================================
// Teacher Forcing Tests
// ========================================================================

#[test]
fn test_teacher_forcing_constant() {
    let tf = TeacherForcing::constant(0.8);

    assert_eq!(tf.schedule(), TeacherForcingSchedule::Constant);
    assert!((tf.get_ratio(0) - 0.8).abs() < 1e-6);
    assert!((tf.get_ratio(100) - 0.8).abs() < 1e-6);
    assert!((tf.get_ratio(1000) - 0.8).abs() < 1e-6);
}

#[test]
fn test_teacher_forcing_linear() {
    let tf = TeacherForcing::linear(1.0, 0.0, 100);

    assert_eq!(tf.schedule(), TeacherForcingSchedule::Linear);
    assert!((tf.get_ratio(0) - 1.0).abs() < 1e-6);
    assert!((tf.get_ratio(50) - 0.5).abs() < 1e-6);
    assert!((tf.get_ratio(100) - 0.0).abs() < 1e-6);
    assert!((tf.get_ratio(200) - 0.0).abs() < 1e-6); // After num_steps
}

#[test]
fn test_teacher_forcing_exponential() {
    let tf = TeacherForcing::exponential(1.0, 0.01, 100);

    assert_eq!(tf.schedule(), TeacherForcingSchedule::Exponential);
    assert!((tf.get_ratio(0) - 1.0).abs() < 1e-6);
    assert!(tf.get_ratio(50) < 1.0 && tf.get_ratio(50) > 0.01);
    assert!((tf.get_ratio(100) - 0.01).abs() < 0.01);
}

#[test]
fn test_teacher_forcing_inverse_sqrt() {
    let tf = TeacherForcing::inverse_sqrt(1.0, 100);

    assert_eq!(tf.schedule(), TeacherForcingSchedule::InverseSquareRoot);
    assert!((tf.get_ratio(0) - 1.0).abs() < 1e-6); // 1/sqrt(1) = 1
    assert!((tf.get_ratio(3) - 0.5).abs() < 1e-6); // 1/sqrt(4) = 0.5
    assert!((tf.get_ratio(8) - (1.0 / 3.0)).abs() < 1e-6); // 1/sqrt(9) = 1/3
}

#[test]
fn test_teacher_forcing_getters() {
    let tf = TeacherForcing::linear(0.9, 0.1, 500);

    assert!((tf.initial_ratio() - 0.9).abs() < 1e-6);
    assert!((tf.final_ratio() - 0.1).abs() < 1e-6);
}

#[test]
fn test_teacher_forcing_should_use_teacher_probabilistic() {
    let tf = TeacherForcing::constant(0.5);

    let mut true_count = 0;
    let trials = 1000;

    for _ in 0..trials {
        if tf.should_use_teacher(0) {
            true_count += 1;
        }
    }

    // Should be roughly 50% (allow 10% tolerance)
    let ratio = true_count as f32 / trials as f32;
    assert!(ratio > 0.4 && ratio < 0.6, "Ratio was {ratio}");
}

#[test]
fn test_teacher_forcing_linear_decreasing() {
    let tf = TeacherForcing::linear(1.0, 0.2, 100);

    let mut prev_ratio = tf.get_ratio(0);
    for step in (10..=100).step_by(10) {
        let ratio = tf.get_ratio(step);
        assert!(ratio <= prev_ratio, "Should be monotonically decreasing");
        prev_ratio = ratio;
    }
}

#[test]
fn test_teacher_forcing_schedule_equality() {
    assert_eq!(
        TeacherForcingSchedule::Linear,
        TeacherForcingSchedule::Linear
    );
    assert_ne!(
        TeacherForcingSchedule::Linear,
        TeacherForcingSchedule::Exponential
    );
}
