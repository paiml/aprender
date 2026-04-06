use super::*;
use crate::autograd::Tensor;

#[test]
fn test_classification_head_shape() {
    let head = ClassificationHead::new(128, 5);
    assert_eq!(head.num_classes(), 5);
    assert_eq!(head.hidden_size(), 128);
    assert_eq!(head.num_parameters(), 128 * 5 + 5);
}

#[test]
fn test_classification_head_forward() {
    let head = ClassificationHead::new(64, 5);
    // Simulate hidden states: 3 tokens, hidden_size=64
    let hidden = Tensor::from_vec(vec![0.1f32; 3 * 64], false);
    let logits = head.forward(&hidden, 3);
    assert_eq!(logits.len(), 5, "F-CLASS-001: must produce 5 logits");
}

#[test]
fn test_classification_head_parameters() {
    let mut head = ClassificationHead::new(64, 5);
    assert_eq!(head.parameters().len(), 2); // weight + bias
    assert_eq!(head.parameters_mut().len(), 2);
}

#[test]
fn test_cross_entropy_loss_finite() {
    let logits = Tensor::from_vec(vec![1.0, 2.0, -1.0, 0.5, 3.0], false);
    let loss = cross_entropy_loss(&logits, 2, 5);
    let loss_val = loss.data()[0];
    assert!(loss_val.is_finite(), "F-CLASS-005: loss must be finite");
    assert!(loss_val > 0.0, "Cross-entropy loss must be positive");
}

#[test]
fn test_cross_entropy_loss_correct_class() {
    // If logit for target class is much larger, loss should be small
    let logits = Tensor::from_vec(vec![-100.0, -100.0, 100.0, -100.0, -100.0], false);
    let loss = cross_entropy_loss(&logits, 2, 5);
    let loss_val = loss.data()[0];
    assert!(loss_val < 0.01, "Loss for correct high-confidence should be ~0");
}

#[test]
fn test_cross_entropy_loss_wrong_class() {
    // If logit for target class is much smaller, loss should be large
    let logits = Tensor::from_vec(vec![100.0, -100.0, -100.0, -100.0, -100.0], false);
    let loss = cross_entropy_loss(&logits, 2, 5);
    let loss_val = loss.data()[0];
    assert!(loss_val > 1.0, "Loss for wrong class should be large");
}

#[test]
fn test_mean_pool() {
    let head = ClassificationHead::new(4, 2);
    // 2 tokens, hidden_size=4: [[1,2,3,4], [5,6,7,8]] → mean = [3,4,5,6]
    let hidden = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], false);
    let pooled = head.mean_pool(&hidden, 2);
    let data = pooled.data();
    let slice = data.as_slice().expect("contiguous");
    assert_eq!(slice.len(), 4);
    assert!((slice[0] - 3.0).abs() < 1e-6);
    assert!((slice[1] - 4.0).abs() < 1e-6);
    assert!((slice[2] - 5.0).abs() < 1e-6);
    assert!((slice[3] - 6.0).abs() < 1e-6);
}

#[test]
fn test_corpus_stats_empty() {
    let stats = corpus_stats(&[], 5);
    assert_eq!(stats.total, 0);
    assert_eq!(stats.class_counts, vec![0; 5]);
}

#[test]
fn test_corpus_stats_distribution() {
    let samples = vec![
        SafetySample { input: "echo hello".into(), label: 0 },
        SafetySample { input: "echo $HOME".into(), label: 1 },
        SafetySample { input: "echo $RANDOM".into(), label: 2 },
        SafetySample { input: "mkdir /tmp/x".into(), label: 3 },
        SafetySample { input: "eval $x".into(), label: 4 },
        SafetySample { input: "ls".into(), label: 0 },
    ];
    let stats = corpus_stats(&samples, 5);
    assert_eq!(stats.total, 6);
    assert_eq!(stats.class_counts, vec![2, 1, 1, 1, 1]);
}

#[test]
#[should_panic(expected = "F-CLASS-001")]
fn test_cross_entropy_wrong_logit_count() {
    let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
    let _ = cross_entropy_loss(&logits, 0, 5);
}

#[test]
#[should_panic(expected = "F-CLASS-002")]
fn test_cross_entropy_label_out_of_range() {
    let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], false);
    let _ = cross_entropy_loss(&logits, 5, 5);
}

// ── Multi-label tests ──────────────────────────────────────────

#[test]
fn test_multi_label_from_single_label() {
    let single = SafetySample { input: "echo $RANDOM".into(), label: 2 };
    let multi = MultiLabelSafetySample::from_single_label(&single, 5);
    assert_eq!(multi.labels, vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    assert_eq!(multi.active_classes(), vec![2]);
}

#[test]
fn test_multi_label_active_classes() {
    let sample = MultiLabelSafetySample {
        input: "echo $RANDOM $HOME".into(),
        labels: vec![0.0, 1.0, 1.0, 0.0, 0.0],
    };
    assert_eq!(sample.active_classes(), vec![1, 2]);
}

#[test]
fn test_multi_label_no_active_classes() {
    let sample =
        MultiLabelSafetySample { input: String::new(), labels: vec![0.0, 0.0, 0.0, 0.0, 0.0] };
    assert!(sample.active_classes().is_empty());
}

#[test]
fn test_multi_label_all_active() {
    let sample = MultiLabelSafetySample {
        input: "eval $RANDOM; mkdir /x".into(),
        labels: vec![1.0, 1.0, 1.0, 1.0, 1.0],
    };
    assert_eq!(sample.active_classes(), vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_bce_with_logits_loss_basic() {
    let logits = Tensor::from_vec(vec![2.0, -1.0, 0.5, -2.0, 3.0], false);
    let targets = [1.0, 0.0, 1.0, 0.0, 0.0];
    let loss = bce_with_logits_loss(&logits, &targets, 5);
    let loss_val = loss.data()[0];
    assert!(loss_val.is_finite(), "F-CLASS-005: loss must be finite");
    assert!(loss_val > 0.0, "BCE loss must be positive");
}

#[test]
fn test_bce_with_logits_loss_perfect() {
    let logits = Tensor::from_vec(vec![100.0, -100.0, 100.0, -100.0, -100.0], false);
    let targets = [1.0, 0.0, 1.0, 0.0, 0.0];
    let loss = bce_with_logits_loss(&logits, &targets, 5);
    assert!(loss.data()[0] < 0.01, "Perfect prediction should have near-zero loss");
}

#[test]
#[should_panic(expected = "F-CLASS-001")]
fn test_bce_logit_shape_mismatch() {
    let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
    let _ = bce_with_logits_loss(&logits, &[1.0, 0.0, 1.0, 0.0, 0.0], 5);
}

#[test]
#[should_panic(expected = "F-CLASS-001")]
fn test_bce_target_shape_mismatch() {
    let logits = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], false);
    let _ = bce_with_logits_loss(&logits, &[1.0, 0.0, 1.0], 5);
}

// ── Class weight tests (SPEC-TUNE-2026-001) ─────────────────────

#[test]
fn test_class_weights_uniform() {
    let stats =
        SafetyCorpusStats { total: 100, class_counts: vec![60, 10, 10, 10, 10], avg_input_len: 50 };
    let weights = compute_class_weights(&stats, ClassWeightStrategy::Uniform, 5);
    assert_eq!(weights.len(), 5);
    for &w in &weights {
        assert!((w - 1.0).abs() < 1e-5, "Uniform weights should all be 1.0, got {w}");
    }
    let sum: f32 = weights.iter().sum();
    assert!((sum - 5.0).abs() < 1e-5, "F-TUNE-005: weights must sum to num_classes");
}

#[test]
fn test_class_weights_inverse_freq() {
    let stats =
        SafetyCorpusStats { total: 100, class_counts: vec![60, 10, 10, 10, 10], avg_input_len: 50 };
    let weights = compute_class_weights(&stats, ClassWeightStrategy::InverseFreq, 5);
    assert_eq!(weights.len(), 5);
    // Majority class (60 samples) should have lowest weight
    assert!(weights[0] < weights[1], "Majority class weight should be smallest");
    // Minority classes should have equal weights
    assert!(
        (weights[1] - weights[2]).abs() < 1e-5,
        "Equal-sized classes should have equal weights"
    );
    let sum: f32 = weights.iter().sum();
    assert!((sum - 5.0).abs() < 1e-5, "F-TUNE-005: weights must sum to num_classes");
}

#[test]
fn test_class_weights_sqrt_inverse() {
    let stats =
        SafetyCorpusStats { total: 100, class_counts: vec![60, 10, 10, 10, 10], avg_input_len: 50 };
    let weights = compute_class_weights(&stats, ClassWeightStrategy::SqrtInverse, 5);
    assert_eq!(weights.len(), 5);
    // Majority class should have lowest weight (but less extreme than inverse_freq)
    assert!(weights[0] < weights[1], "Majority class should have lowest weight");
    let sum: f32 = weights.iter().sum();
    assert!((sum - 5.0).abs() < 1e-5, "F-TUNE-005: weights must sum to num_classes");

    // sqrt_inverse should be less extreme than inverse_freq
    let inv_weights = compute_class_weights(&stats, ClassWeightStrategy::InverseFreq, 5);
    let inv_ratio = inv_weights[1] / inv_weights[0];
    let sqrt_ratio = weights[1] / weights[0];
    assert!(sqrt_ratio < inv_ratio, "sqrt_inverse should be less extreme than inverse_freq");
}

#[test]
fn test_class_weights_strategy_parse() {
    assert_eq!("uniform".parse::<ClassWeightStrategy>().ok(), Some(ClassWeightStrategy::Uniform));
    assert_eq!(
        "inverse_freq".parse::<ClassWeightStrategy>().ok(),
        Some(ClassWeightStrategy::InverseFreq)
    );
    assert_eq!(
        "sqrt_inverse".parse::<ClassWeightStrategy>().ok(),
        Some(ClassWeightStrategy::SqrtInverse)
    );
    assert!("invalid".parse::<ClassWeightStrategy>().is_err());
}

#[test]
fn test_class_weights_strategy_aliases() {
    assert_eq!(
        "inverse".parse::<ClassWeightStrategy>().ok(),
        Some(ClassWeightStrategy::InverseFreq)
    );
    assert_eq!("sqrt".parse::<ClassWeightStrategy>().ok(), Some(ClassWeightStrategy::SqrtInverse));
}

#[test]
fn test_class_weights_strategy_display() {
    assert_eq!(format!("{}", ClassWeightStrategy::Uniform), "uniform");
    assert_eq!(format!("{}", ClassWeightStrategy::InverseFreq), "inverse_freq");
    assert_eq!(format!("{}", ClassWeightStrategy::SqrtInverse), "sqrt_inverse");
}

#[test]
fn test_class_weights_all_equal_classes() {
    let stats =
        SafetyCorpusStats { total: 50, class_counts: vec![10, 10, 10, 10, 10], avg_input_len: 30 };
    let inv_weights = compute_class_weights(&stats, ClassWeightStrategy::InverseFreq, 5);
    // All equal → all weights should be 1.0
    for &w in &inv_weights {
        assert!((w - 1.0).abs() < 1e-5, "Equal classes should all get weight 1.0, got {w}");
    }
}

// ── Corpus I/O tests ───────────────────────────────────────────

#[test]
fn test_load_safety_corpus_valid() {
    use std::io::Write;
    let dir = std::env::temp_dir().join("entrenar_test_corpus");
    std::fs::create_dir_all(&dir).expect("valid");
    let path = dir.join("valid.jsonl");
    {
        let mut f = std::fs::File::create(&path).expect("valid");
        writeln!(f, r#"{{"input":"echo hello","label":0}}"#).expect("valid");
        writeln!(f, r#"{{"input":"eval $x","label":4}}"#).expect("valid");
        writeln!(f).expect("valid"); // empty line should be skipped
        writeln!(f, r#"{{"input":"ls","label":1}}"#).expect("valid");
    }
    let samples = load_safety_corpus(&path, 5).expect("valid");
    assert_eq!(samples.len(), 3);
    assert_eq!(samples[0].label, 0);
    assert_eq!(samples[1].label, 4);
    assert_eq!(samples[2].label, 1);
    std::fs::remove_file(&path).expect("valid");
}

#[test]
fn test_load_safety_corpus_label_out_of_range() {
    use std::io::Write;
    let path = std::env::temp_dir().join("entrenar_test_corpus_oor.jsonl");
    {
        let mut f = std::fs::File::create(&path).expect("valid");
        writeln!(f, r#"{{"input":"bad","label":10}}"#).expect("valid");
    }
    let result = load_safety_corpus(&path, 5);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("F-CLASS-002"), "Expected F-CLASS-002 error, got: {err}");
    std::fs::remove_file(&path).expect("valid");
}

#[test]
fn test_load_safety_corpus_file_not_found() {
    let result = load_safety_corpus(std::path::Path::new("/tmp/nonexistent_xyz_abc.jsonl"), 5);
    assert!(result.is_err());
}

#[test]
fn test_load_safety_corpus_invalid_json() {
    use std::io::Write;
    let path = std::env::temp_dir().join("entrenar_test_corpus_badjson.jsonl");
    {
        let mut f = std::fs::File::create(&path).expect("valid");
        writeln!(f, "not json at all").expect("valid");
    }
    let result = load_safety_corpus(&path, 5);
    assert!(result.is_err());
    std::fs::remove_file(&path).expect("valid");
}

#[test]
fn test_load_multi_label_corpus_single_label_format() {
    use std::io::Write;
    let path = std::env::temp_dir().join("entrenar_test_ml_single.jsonl");
    {
        let mut f = std::fs::File::create(&path).expect("valid");
        writeln!(f, r#"{{"input":"echo hi","label":2}}"#).expect("valid");
    }
    let samples = load_multi_label_corpus(&path, 5).expect("valid");
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].labels, vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    std::fs::remove_file(&path).expect("valid");
}

#[test]
fn test_load_multi_label_corpus_multi_label_format() {
    use std::io::Write;
    let path = std::env::temp_dir().join("entrenar_test_ml_multi.jsonl");
    {
        let mut f = std::fs::File::create(&path).expect("valid");
        writeln!(f, r#"{{"input":"eval $RANDOM","labels":[0.0,1.0,1.0,0.0,0.0]}}"#).expect("valid");
    }
    let samples = load_multi_label_corpus(&path, 5).expect("valid");
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].active_classes(), vec![1, 2]);
    std::fs::remove_file(&path).expect("valid");
}

#[test]
fn test_load_multi_label_corpus_wrong_label_length() {
    use std::io::Write;
    let path = std::env::temp_dir().join("entrenar_test_ml_wronglen.jsonl");
    {
        let mut f = std::fs::File::create(&path).expect("valid");
        writeln!(f, r#"{{"input":"x","labels":[1.0,0.0]}}"#).expect("valid");
    }
    let result = load_multi_label_corpus(&path, 5);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("F-CLASS-001"), "Expected F-CLASS-001 error, got: {err}");
    std::fs::remove_file(&path).expect("valid");
}

#[test]
fn test_load_multi_label_corpus_single_label_out_of_range() {
    use std::io::Write;
    let path = std::env::temp_dir().join("entrenar_test_ml_oor.jsonl");
    {
        let mut f = std::fs::File::create(&path).expect("valid");
        writeln!(f, r#"{{"input":"bad","label":99}}"#).expect("valid");
    }
    let result = load_multi_label_corpus(&path, 5);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("F-CLASS-002"), "Expected F-CLASS-002 error, got: {err}");
    std::fs::remove_file(&path).expect("valid");
}

#[test]
fn test_load_multi_label_corpus_invalid_format() {
    use std::io::Write;
    let path = std::env::temp_dir().join("entrenar_test_ml_badfmt.jsonl");
    {
        let mut f = std::fs::File::create(&path).expect("valid");
        writeln!(f, r#"{{"foo":"bar"}}"#).expect("valid");
    }
    let result = load_multi_label_corpus(&path, 5);
    assert!(result.is_err());
    std::fs::remove_file(&path).expect("valid");
}

#[test]
fn test_load_multi_label_corpus_file_not_found() {
    let result = load_multi_label_corpus(std::path::Path::new("/tmp/nonexistent_ml_xyz.jsonl"), 5);
    assert!(result.is_err());
}

#[test]
fn test_safety_sample_input_ids() {
    let sample = SafetySample { input: "AB".into(), label: 0 };
    let ids = sample.input_ids();
    assert_eq!(ids, vec![65, 66]); // 'A'=65, 'B'=66
}

#[test]
fn test_classification_head_num_parameters() {
    let head = ClassificationHead::new(256, 10);
    assert_eq!(head.num_parameters(), 256 * 10 + 10);
}

#[test]
fn test_cross_entropy_uniform_logits() {
    // Uniform logits → loss should be -log(1/K) = log(K)
    let logits = Tensor::from_vec(vec![0.0; 5], false);
    let loss = cross_entropy_loss(&logits, 0, 5);
    let expected = (5.0f32).ln();
    assert!((loss.data()[0] - expected).abs() < 1e-4);
}

#[test]
fn test_bce_with_logits_loss_all_zeros() {
    let logits = Tensor::from_vec(vec![0.0; 5], false);
    let targets = [0.0, 0.0, 0.0, 0.0, 0.0];
    let loss = bce_with_logits_loss(&logits, &targets, 5);
    // BCE(0, 0) = max(0,0) - 0*0 + log(1+exp(0)) = 0 + log(2) ≈ 0.693
    let expected = 2.0f32.ln();
    assert!((loss.data()[0] - expected).abs() < 1e-4);
}

// =========================================================================
// ENC-007: Pooling strategy tests (CLS, LastToken, Mean)
// =========================================================================

#[test]
fn enc_007_cls_pool_extracts_first_token() {
    let head = ClassificationHead::new(4, 2);
    // 3 tokens, hidden_size=4
    // Position 0: [1,2,3,4], Position 1: [5,6,7,8], Position 2: [9,10,11,12]
    let hidden = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        false,
    );
    let pooled = head.cls_pool(&hidden);
    assert_eq!(pooled.len(), 4);
    let data = pooled.data();
    assert_eq!(data[0], 1.0);
    assert_eq!(data[1], 2.0);
    assert_eq!(data[2], 3.0);
    assert_eq!(data[3], 4.0);
}

#[test]
fn enc_007_last_token_pool_extracts_last() {
    let head = ClassificationHead::new(4, 2);
    let hidden = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        false,
    );
    let pooled = head.last_token_pool(&hidden, 3);
    assert_eq!(pooled.len(), 4);
    let data = pooled.data();
    assert_eq!(data[0], 9.0);
    assert_eq!(data[1], 10.0);
    assert_eq!(data[2], 11.0);
    assert_eq!(data[3], 12.0);
}

#[test]
fn enc_007_pooling_strategy_from_architecture() {
    use crate::transformer::ModelArchitecture;
    assert_eq!(
        PoolingStrategy::from_architecture(ModelArchitecture::Encoder),
        PoolingStrategy::Cls
    );
    assert_eq!(
        PoolingStrategy::from_architecture(ModelArchitecture::Decoder),
        PoolingStrategy::Mean
    );
}

#[test]
fn enc_007_forward_with_cls_pooling() {
    let head = ClassificationHead::new(8, 2);
    let hidden = Tensor::from_vec(vec![0.1f32; 3 * 8], false);
    let logits = head.forward_with_pooling(&hidden, 3, PoolingStrategy::Cls);
    assert_eq!(logits.len(), 2);
    assert!(logits.data().iter().all(|v| v.is_finite()));
}

#[test]
fn enc_007_forward_with_last_token_pooling() {
    let head = ClassificationHead::new(8, 2);
    let hidden = Tensor::from_vec(vec![0.1f32; 3 * 8], false);
    let logits = head.forward_with_pooling(&hidden, 3, PoolingStrategy::LastToken);
    assert_eq!(logits.len(), 2);
    assert!(logits.data().iter().all(|v| v.is_finite()));
}

#[test]
fn enc_007_pool_dispatch() {
    let head = ClassificationHead::new(4, 2);
    let hidden = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], false);

    let cls = head.pool(&hidden, 2, PoolingStrategy::Cls);
    let last = head.pool(&hidden, 2, PoolingStrategy::LastToken);
    let mean = head.pool(&hidden, 2, PoolingStrategy::Mean);

    // CLS = first token [1,2,3,4]
    assert_eq!(cls.data()[0], 1.0);
    // Last = second token [5,6,7,8]
    assert_eq!(last.data()[0], 5.0);
    // Mean = average [(1+5)/2, (2+6)/2, (3+7)/2, (4+8)/2] = [3,4,5,6]
    assert!((mean.data()[0] - 3.0).abs() < 1e-5);
}

// ── Coverage improvement tests ───────────────────────────────────

#[test]
#[should_panic(expected = "F-CLASS-004: hidden_size must be > 0")]
fn test_classification_head_zero_hidden_size() {
    let _ = ClassificationHead::new(0, 5);
}

#[test]
#[should_panic(expected = "F-CLASS-004: num_classes must be >= 2")]
fn test_classification_head_one_class() {
    let _ = ClassificationHead::new(64, 1);
}

#[test]
#[should_panic(expected = "F-CLASS-004: num_classes must be >= 2")]
fn test_classification_head_zero_classes() {
    let _ = ClassificationHead::new(64, 0);
}

#[test]
fn test_classification_head_large_dimensions() {
    let head = ClassificationHead::new(1024, 100);
    assert_eq!(head.hidden_size(), 1024);
    assert_eq!(head.num_classes(), 100);
    assert_eq!(head.num_parameters(), 1024 * 100 + 100);
}

#[test]
fn test_classification_head_minimum_valid() {
    let head = ClassificationHead::new(1, 2);
    assert_eq!(head.hidden_size(), 1);
    assert_eq!(head.num_classes(), 2);
    assert_eq!(head.num_parameters(), 2 + 2);
}

#[test]
fn test_classification_head_forward_single_token() {
    let head = ClassificationHead::new(8, 3);
    let hidden = Tensor::from_vec(vec![0.5f32; 8], false);
    let logits = head.forward(&hidden, 1);
    assert_eq!(logits.len(), 3);
    assert!(logits.data().iter().all(|v| v.is_finite()));
}

#[test]
fn test_classification_head_forward_requires_grad() {
    let head = ClassificationHead::new(4, 2);
    let hidden = Tensor::from_vec(vec![0.1f32; 4 * 2], true);
    let logits = head.forward(&hidden, 2);
    assert_eq!(logits.len(), 2);
    // Forward should propagate requires_grad from input
    assert!(logits.requires_grad());
}

#[test]
fn test_classification_head_forward_no_grad_input() {
    // Even with no-grad input, the head's weight/bias have grad enabled,
    // so we just verify the forward produces valid finite output
    let head = ClassificationHead::new(4, 2);
    let hidden = Tensor::from_vec(vec![0.1f32; 4 * 2], false);
    let logits = head.forward(&hidden, 2);
    assert_eq!(logits.len(), 2);
    assert!(logits.data().iter().all(|v| v.is_finite()));
}

#[test]
fn test_mean_pool_single_token() {
    let head = ClassificationHead::new(4, 2);
    let hidden = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], false);
    let pooled = head.mean_pool(&hidden, 1);
    let data = pooled.data();
    let slice = data.as_slice().expect("contiguous");
    assert_eq!(slice.len(), 4);
    assert!((slice[0] - 1.0).abs() < 1e-6);
    assert!((slice[1] - 2.0).abs() < 1e-6);
    assert!((slice[2] - 3.0).abs() < 1e-6);
    assert!((slice[3] - 4.0).abs() < 1e-6);
}

#[test]
fn test_cls_pool_preserves_grad() {
    let head = ClassificationHead::new(4, 2);
    let hidden = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], true);
    let pooled = head.cls_pool(&hidden);
    assert!(pooled.requires_grad());
}

#[test]
fn test_last_token_pool_single_token() {
    let head = ClassificationHead::new(4, 2);
    let hidden = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], false);
    let pooled = head.last_token_pool(&hidden, 1);
    let data = pooled.data();
    assert_eq!(data[0], 10.0);
    assert_eq!(data[1], 20.0);
}

#[test]
fn test_forward_with_pooling_mean() {
    let head = ClassificationHead::new(4, 2);
    let hidden = Tensor::from_vec(vec![0.1f32; 4 * 3], false);
    let logits_mean = head.forward_with_pooling(&hidden, 3, PoolingStrategy::Mean);
    let logits_default = head.forward(&hidden, 3);
    // Mean pooling via forward_with_pooling should match default forward
    let mean_data = logits_mean.data();
    let default_data = logits_default.data();
    for i in 0..2 {
        assert!(
            (mean_data[i] - default_data[i]).abs() < 1e-5,
            "Mean pooling forward should match default forward"
        );
    }
}

#[test]
fn test_tokenized_sample_basic() {
    let sample = TokenizedSample { token_ids: vec![1, 2, 3], label: 0 };
    assert_eq!(sample.token_ids.len(), 3);
    assert_eq!(sample.label, 0);
}

#[test]
fn test_tokenized_sample_clone() {
    let sample = TokenizedSample { token_ids: vec![10, 20, 30], label: 2 };
    let cloned = sample.clone();
    assert_eq!(cloned.token_ids, vec![10, 20, 30]);
    assert_eq!(cloned.label, 2);
}

#[test]
fn test_safety_sample_input_ids_utf8() {
    // Multi-byte UTF-8 character: 'ñ' is 0xC3 0xB1
    let sample = SafetySample { input: "ñ".into(), label: 0 };
    let ids = sample.input_ids();
    assert_eq!(ids, vec![0xC3, 0xB1]);
}

#[test]
fn test_safety_sample_input_ids_empty() {
    let sample = SafetySample { input: String::new(), label: 0 };
    let ids = sample.input_ids();
    assert!(ids.is_empty());
}

#[test]
fn test_multi_label_from_single_label_out_of_range() {
    // Label >= num_classes: should produce all-zeros
    let single = SafetySample { input: "bad".into(), label: 10 };
    let multi = MultiLabelSafetySample::from_single_label(&single, 5);
    assert_eq!(multi.labels, vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    assert!(multi.active_classes().is_empty());
}

#[test]
fn test_multi_label_from_single_label_boundary() {
    // Label at last valid index
    let single = SafetySample { input: "ok".into(), label: 4 };
    let multi = MultiLabelSafetySample::from_single_label(&single, 5);
    assert_eq!(multi.labels, vec![0.0, 0.0, 0.0, 0.0, 1.0]);
    assert_eq!(multi.active_classes(), vec![4]);
}

#[test]
fn test_multi_label_from_single_label_first_class() {
    let single = SafetySample { input: "ok".into(), label: 0 };
    let multi = MultiLabelSafetySample::from_single_label(&single, 3);
    assert_eq!(multi.labels, vec![1.0, 0.0, 0.0]);
    assert_eq!(multi.active_classes(), vec![0]);
}

#[test]
fn test_multi_label_active_classes_threshold() {
    // Values at exactly 0.5 should NOT be active (> 0.5 required)
    let sample =
        MultiLabelSafetySample { input: "test".into(), labels: vec![0.5, 0.51, 0.49, 1.0, 0.0] };
    assert_eq!(sample.active_classes(), vec![1, 3]);
}

#[test]
fn test_corpus_stats_avg_input_len() {
    let samples = vec![
        SafetySample { input: "ab".into(), label: 0 }, // len 2
        SafetySample { input: "abcd".into(), label: 0 }, // len 4
    ];
    let stats = corpus_stats(&samples, 2);
    assert_eq!(stats.total, 2);
    assert_eq!(stats.avg_input_len, 3); // (2+4)/2 = 3
}

#[test]
fn test_corpus_stats_out_of_range_labels() {
    // Labels out of range should not be counted
    let samples = vec![
        SafetySample { input: "a".into(), label: 0 },
        SafetySample { input: "b".into(), label: 10 }, // out of range
    ];
    let stats = corpus_stats(&samples, 5);
    assert_eq!(stats.total, 2);
    assert_eq!(stats.class_counts[0], 1);
    assert_eq!(stats.class_counts.iter().sum::<usize>(), 1);
}

#[test]
fn test_class_weight_strategy_case_insensitive() {
    assert_eq!("UNIFORM".parse::<ClassWeightStrategy>().ok(), Some(ClassWeightStrategy::Uniform));
    assert_eq!(
        "Inverse_Freq".parse::<ClassWeightStrategy>().ok(),
        Some(ClassWeightStrategy::InverseFreq)
    );
    assert_eq!(
        "SQRT_INVERSE".parse::<ClassWeightStrategy>().ok(),
        Some(ClassWeightStrategy::SqrtInverse)
    );
    assert_eq!("SQRT".parse::<ClassWeightStrategy>().ok(), Some(ClassWeightStrategy::SqrtInverse));
    assert_eq!(
        "INVERSE".parse::<ClassWeightStrategy>().ok(),
        Some(ClassWeightStrategy::InverseFreq)
    );
}

#[test]
fn test_class_weight_strategy_parse_error_message() {
    let err = "bogus".parse::<ClassWeightStrategy>().unwrap_err();
    assert!(err.contains("Unknown class weight strategy"));
    assert!(err.contains("bogus"));
}

#[test]
fn test_compute_class_weights_with_zero_count() {
    // Zero-count class should be treated as 1 (max(1) path)
    let stats = SafetyCorpusStats { total: 100, class_counts: vec![50, 50, 0], avg_input_len: 10 };
    let weights = compute_class_weights(&stats, ClassWeightStrategy::InverseFreq, 3);
    assert_eq!(weights.len(), 3);
    // All weights should be finite and positive
    for &w in &weights {
        assert!(w > 0.0, "Weight should be positive, got {w}");
        assert!(w.is_finite(), "Weight should be finite, got {w}");
    }
    let sum: f32 = weights.iter().sum();
    assert!((sum - 3.0).abs() < 1e-4, "F-TUNE-005: weights must sum to num_classes");
}

#[test]
fn test_compute_class_weights_sqrt_with_zero_count() {
    let stats = SafetyCorpusStats { total: 100, class_counts: vec![0, 100, 0], avg_input_len: 10 };
    let weights = compute_class_weights(&stats, ClassWeightStrategy::SqrtInverse, 3);
    for &w in &weights {
        assert!(w > 0.0 && w.is_finite());
    }
    let sum: f32 = weights.iter().sum();
    assert!((sum - 3.0).abs() < 1e-4);
}

#[test]
#[should_panic(expected = "F-TUNE-005")]
fn test_compute_class_weights_mismatched_classes() {
    let stats = SafetyCorpusStats { total: 10, class_counts: vec![5, 5], avg_input_len: 10 };
    let _ = compute_class_weights(&stats, ClassWeightStrategy::Uniform, 3);
}

#[test]
fn test_bce_with_logits_loss_inf_input() {
    // Inf logits should trigger the finite fallback (loss = 100.0)
    let logits = Tensor::from_vec(vec![f32::INFINITY, f32::NEG_INFINITY, 0.0], false);
    let targets = [1.0, 0.0, 0.5];
    let loss = bce_with_logits_loss(&logits, &targets, 3);
    let val = loss.data()[0];
    assert!(val.is_finite(), "F-CLASS-005: loss must be finite even with Inf input");
}

#[test]
fn test_bce_with_logits_loss_nan_input() {
    let logits = Tensor::from_vec(vec![f32::NAN, 0.0, 0.0], false);
    let targets = [0.0, 0.0, 0.0];
    let loss = bce_with_logits_loss(&logits, &targets, 3);
    let val = loss.data()[0];
    // NaN in logits should trigger the finite check → clamp to 100.0
    assert!(val.is_finite(), "F-CLASS-005: loss must be finite even with NaN input");
}

#[test]
fn test_cross_entropy_loss_all_same_logits() {
    // All same logits → loss = log(K)
    let logits = Tensor::from_vec(vec![5.0, 5.0, 5.0], false);
    let loss = cross_entropy_loss(&logits, 1, 3);
    let expected = (3.0f32).ln();
    assert!((loss.data()[0] - expected).abs() < 1e-4);
}

#[test]
fn test_cross_entropy_loss_requires_grad() {
    let logits = Tensor::from_vec(vec![1.0, 2.0], true);
    let loss = cross_entropy_loss(&logits, 0, 2);
    assert!(loss.requires_grad());
}

#[test]
fn test_cross_entropy_loss_two_classes() {
    let logits = Tensor::from_vec(vec![0.0, 0.0], false);
    let loss = cross_entropy_loss(&logits, 0, 2);
    let expected = (2.0f32).ln();
    assert!((loss.data()[0] - expected).abs() < 1e-4);
}

#[test]
fn test_bce_all_ones_targets() {
    let logits = Tensor::from_vec(vec![100.0, 100.0], false);
    let targets = [1.0, 1.0];
    let loss = bce_with_logits_loss(&logits, &targets, 2);
    assert!(loss.data()[0] < 0.01, "Perfect all-ones predictions should have near-zero loss");
}

#[test]
fn test_load_multi_label_corpus_empty_lines() {
    use std::io::Write;
    let path = std::env::temp_dir().join("entrenar_test_ml_empty_lines.jsonl");
    {
        let mut f = std::fs::File::create(&path).expect("create file");
        writeln!(f).expect("write");
        writeln!(f, "  ").expect("write");
        writeln!(f, r#"{{"input":"echo hi","label":0}}"#).expect("write");
        writeln!(f).expect("write");
    }
    let samples = load_multi_label_corpus(&path, 5).expect("valid");
    assert_eq!(samples.len(), 1);
    std::fs::remove_file(&path).expect("cleanup");
}

#[test]
fn test_pooling_strategy_debug() {
    let s = format!("{:?}", PoolingStrategy::Mean);
    assert_eq!(s, "Mean");
    let s = format!("{:?}", PoolingStrategy::Cls);
    assert_eq!(s, "Cls");
    let s = format!("{:?}", PoolingStrategy::LastToken);
    assert_eq!(s, "LastToken");
}

#[test]
fn test_pooling_strategy_clone_eq() {
    let a = PoolingStrategy::Mean;
    let b = a;
    assert_eq!(a, b);
    let c = PoolingStrategy::Cls;
    assert_ne!(a, c);
}

#[test]
fn test_safety_sample_deserialize() {
    let json = r#"{"input":"echo $HOME","label":1}"#;
    let sample: SafetySample = serde_json::from_str(json).expect("deserialize");
    assert_eq!(sample.input, "echo $HOME");
    assert_eq!(sample.label, 1);
}

#[test]
fn test_multi_label_safety_sample_deserialize() {
    let json = r#"{"input":"eval $x","labels":[0.0,1.0,0.0]}"#;
    let sample: MultiLabelSafetySample = serde_json::from_str(json).expect("deserialize");
    assert_eq!(sample.input, "eval $x");
    assert_eq!(sample.labels, vec![0.0, 1.0, 0.0]);
}

#[test]
fn test_safety_corpus_stats_debug() {
    let stats = SafetyCorpusStats { total: 10, class_counts: vec![5, 5], avg_input_len: 20 };
    let debug = format!("{stats:?}");
    assert!(debug.contains("SafetyCorpusStats"));
    assert!(debug.contains("10"));
}

#[test]
fn test_safety_corpus_stats_clone() {
    let stats = SafetyCorpusStats { total: 10, class_counts: vec![5, 5], avg_input_len: 20 };
    let cloned = stats.clone();
    assert_eq!(cloned.total, 10);
    assert_eq!(cloned.class_counts, vec![5, 5]);
    assert_eq!(cloned.avg_input_len, 20);
}
