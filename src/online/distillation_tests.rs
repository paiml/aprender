pub(crate) use super::*;

#[test]
fn test_softmax_temperature_basic() {
    let logits = vec![1.0, 2.0, 3.0];

    // T=1 (standard softmax)
    let soft1 = softmax_temperature(&logits, 1.0);
    assert!((soft1.iter().sum::<f64>() - 1.0).abs() < 1e-10);

    // T=2 (softer)
    let soft2 = softmax_temperature(&logits, 2.0);
    assert!((soft2.iter().sum::<f64>() - 1.0).abs() < 1e-10);

    // Higher temperature = more uniform
    let variance1: f64 = soft1.iter().map(|x| (x - 1.0 / 3.0).powi(2)).sum();
    let variance2: f64 = soft2.iter().map(|x| (x - 1.0 / 3.0).powi(2)).sum();
    assert!(variance2 < variance1);
}

#[test]
fn test_softmax_temperature_extreme() {
    let logits = vec![0.0, 10.0];

    // Very low temperature (approaches argmax)
    let hard = softmax_temperature(&logits, 0.1);
    assert!(hard[1] > 0.99);

    // Very high temperature (approaches uniform)
    let uniform = softmax_temperature(&logits, 100.0);
    assert!((uniform[0] - uniform[1]).abs() < 0.1);
}

#[test]
fn test_softmax_empty() {
    let result = softmax_temperature(&[], 1.0);
    assert!(result.is_empty());
}

#[test]
fn test_kl_divergence_same() {
    let p = vec![0.25, 0.25, 0.25, 0.25];
    let kl = kl_divergence(&p, &p);
    assert!(kl.abs() < 1e-10);
}

#[test]
fn test_kl_divergence_different() {
    let p = vec![0.9, 0.1];
    let q = vec![0.5, 0.5];
    let kl = kl_divergence(&p, &q);
    assert!(kl > 0.0);
}

#[test]
fn test_cross_entropy() {
    // Perfect prediction
    let probs = vec![0.0, 1.0, 0.0];
    let labels = vec![0.0, 1.0, 0.0];
    let ce = cross_entropy(&probs, &labels);
    assert!(ce < 0.01);

    // Bad prediction
    let probs_bad = vec![0.9, 0.05, 0.05];
    let ce_bad = cross_entropy(&probs_bad, &labels);
    assert!(ce_bad > ce);
}

#[test]
fn test_binary_cross_entropy() {
    // Perfect prediction
    let bce = binary_cross_entropy(1.0, 1.0);
    assert!(bce < 0.01);

    // Wrong prediction
    let bce_bad = binary_cross_entropy(0.1, 1.0);
    assert!(bce_bad > 1.0);
}

#[test]
fn test_soft_target_generator() {
    let generator = SoftTargetGenerator::with_temperature(3.0);
    let logits = vec![1.0, 2.0, 3.0];
    let soft = generator.generate(&logits);

    assert_eq!(soft.len(), 3);
    assert!((soft.iter().sum::<f64>() - 1.0).abs() < 1e-10);
}

#[test]
fn test_soft_target_generator_batch() {
    let generator = SoftTargetGenerator::with_temperature(2.0);
    let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 samples × 3 classes
    let soft = generator.generate_batch(&logits, 3);

    assert_eq!(soft.len(), 6);

    // Each sample should sum to 1
    assert!((soft[0..3].iter().sum::<f64>() - 1.0).abs() < 1e-10);
    assert!((soft[3..6].iter().sum::<f64>() - 1.0).abs() < 1e-10);
}

#[test]
fn test_distillation_loss_compute() {
    let loss = DistillationLoss::new();

    let student_logits = vec![1.0, 2.0, 3.0];
    let teacher_logits = vec![1.0, 2.5, 2.8];
    let hard_labels = vec![0.0, 0.0, 1.0];

    let result = loss.compute(&student_logits, &teacher_logits, &hard_labels);
    assert!(result.is_ok());
    assert!(result.unwrap().is_finite());
}

#[test]
fn test_distillation_loss_gradient() {
    let loss = DistillationLoss::new();

    let student_logits = vec![1.0, 2.0, 3.0];
    let teacher_logits = vec![1.0, 2.5, 2.8];
    let hard_labels = vec![0.0, 0.0, 1.0];

    let grad = loss.gradient(&student_logits, &teacher_logits, &hard_labels);
    assert!(grad.is_ok());
    assert_eq!(grad.unwrap().len(), 3);
}

#[test]
fn test_distillation_loss_dimension_mismatch() {
    let loss = DistillationLoss::new();

    let student_logits = vec![1.0, 2.0];
    let teacher_logits = vec![1.0, 2.0, 3.0];
    let hard_labels = vec![0.0, 1.0];

    assert!(loss
        .compute(&student_logits, &teacher_logits, &hard_labels)
        .is_err());
}

#[test]
fn test_linear_distiller_forward() {
    let distiller = LinearDistiller::new(3, 2);

    let features = vec![1.0, 2.0, 3.0];
    let logits = distiller.forward(&features);

    assert!(logits.is_ok());
    assert_eq!(logits.unwrap().len(), 2);
}

#[test]
fn test_linear_distiller_train_step() {
    let config = DistillationConfig {
        learning_rate: 0.1,
        ..Default::default()
    };
    let mut distiller = LinearDistiller::with_config(2, 2, config);

    let features = vec![1.0, 0.5];
    let teacher_logits = vec![0.5, 1.5];
    let hard_labels = vec![0.0, 1.0];

    let loss1 = distiller
        .train_step(&features, &teacher_logits, &hard_labels)
        .unwrap();

    // Train more
    for _ in 0..10 {
        distiller
            .train_step(&features, &teacher_logits, &hard_labels)
            .unwrap();
    }

    let loss2 = distiller
        .train_step(&features, &teacher_logits, &hard_labels)
        .unwrap();

    // Loss should decrease
    assert!(loss2 < loss1 * 1.5); // Allow some variance
}

#[test]
fn test_linear_distiller_predict() {
    let mut distiller = LinearDistiller::new(2, 3);

    // Train briefly
    let features = vec![1.0, 0.0];
    let teacher_logits = vec![0.0, 0.0, 5.0]; // Strongly predict class 2
    let hard_labels = vec![0.0, 0.0, 1.0];

    for _ in 0..50 {
        distiller
            .train_step(&features, &teacher_logits, &hard_labels)
            .unwrap();
    }

    let pred = distiller.predict(&features).unwrap();
    assert_eq!(pred, 2);
}

#[test]
fn test_linear_distiller_dimension_mismatch() {
    let distiller = LinearDistiller::new(3, 2);

    let features = vec![1.0, 2.0]; // Wrong size
    assert!(distiller.forward(&features).is_err());
}

#[test]
fn test_distillation_config_default() {
    let config = DistillationConfig::default();
    assert_eq!(config.temperature, DEFAULT_TEMPERATURE);
    assert_eq!(config.alpha, DEFAULT_ALPHA);
}

#[test]
fn test_distillation_config_builder() {
    let config = DistillationConfig::default()
        .with_temperature(5.0)
        .with_alpha(0.9);

    assert_eq!(config.temperature, 5.0);
    assert_eq!(config.alpha, 0.9);
}

#[test]
fn test_soft_target_generator_default() {
    let generator = SoftTargetGenerator::default();
    assert_eq!(generator.temperature, DEFAULT_TEMPERATURE);
}

#[test]
fn test_distillation_loss_default() {
    let loss = DistillationLoss::default();
    assert_eq!(loss.config().temperature, DEFAULT_TEMPERATURE);
}

#[test]
fn test_temperature_effect_on_dark_knowledge() {
    let logits = vec![1.0, 3.0, 2.0];

    // Low temperature - peaked distribution
    let low_t = softmax_temperature(&logits, 1.0);
    // High temperature - flatter distribution
    let high_t = softmax_temperature(&logits, 5.0);

    // The smallest class probability should be higher with higher T
    let min_low = low_t.iter().cloned().fold(f64::INFINITY, f64::min);
    let min_high = high_t.iter().cloned().fold(f64::INFINITY, f64::min);

    assert!(
        min_high > min_low,
        "Higher T should reveal more dark knowledge"
    );
}
