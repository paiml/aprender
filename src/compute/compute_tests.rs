pub(crate) use super::*;

#[test]
fn test_select_backend_small() {
    let category = select_backend(100, false);
    assert_eq!(category, BackendCategory::SimdOnly);
}

#[test]
fn test_select_backend_medium() {
    let category = select_backend(10_000, false);
    assert_eq!(category, BackendCategory::SimdParallel);
}

#[test]
fn test_select_backend_large_no_gpu() {
    let category = select_backend(1_000_000, false);
    assert_eq!(category, BackendCategory::SimdParallel);
}

#[test]
fn test_select_backend_large_with_gpu() {
    let category = select_backend(1_000_000, true);
    assert_eq!(category, BackendCategory::Gpu);
}

#[test]
fn test_should_use_gpu() {
    assert!(!should_use_gpu(50_000));
    assert!(should_use_gpu(100_000));
    assert!(should_use_gpu(1_000_000));
}

#[test]
fn test_should_use_parallel() {
    assert!(!should_use_parallel(500));
    assert!(should_use_parallel(1_000));
    assert!(should_use_parallel(10_000));
}

#[test]
fn test_training_guard_clean_gradients() {
    let guard = TrainingGuard::new("test");
    let gradients = vec![0.1, 0.2, 0.3, -0.1];
    assert!(guard.check_gradients(&gradients).is_ok());
}

#[test]
fn test_training_guard_nan_gradients() {
    let guard = TrainingGuard::new("test");
    let gradients = vec![0.1, f32::NAN, 0.3];
    assert!(guard.check_gradients(&gradients).is_err());
}

#[test]
fn test_training_guard_inf_gradients() {
    let guard = TrainingGuard::new("test");
    let gradients = vec![0.1, f32::INFINITY, 0.3];
    assert!(guard.check_gradients(&gradients).is_err());
}

#[test]
fn test_training_guard_loss_nan() {
    let guard = TrainingGuard::new("test");
    assert!(guard.check_loss(f32::NAN).is_err());
}

#[test]
fn test_training_guard_loss_inf() {
    let guard = TrainingGuard::new("test");
    assert!(guard.check_loss(f32::INFINITY).is_err());
}

#[test]
fn test_training_guard_loss_valid() {
    let guard = TrainingGuard::new("test");
    assert!(guard.check_loss(0.5).is_ok());
}

#[test]
fn test_divergence_guard_within_tolerance() {
    let guard = DivergenceGuard::new(0.01, "test");
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.001, 2.002, 3.003];
    assert!(guard.check(&a, &b).is_ok());
}

#[test]
fn test_divergence_guard_exceeds_tolerance() {
    let guard = DivergenceGuard::new(0.001, "test");
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.1, 2.0, 3.0];
    assert!(guard.check(&a, &b).is_err());
}

#[test]
fn test_experiment_seed_deterministic() {
    let seed1 = ExperimentSeed::from_master(42);
    let seed2 = ExperimentSeed::from_master(42);
    assert_eq!(seed1.master, seed2.master);
    assert_eq!(seed1.data_shuffle, seed2.data_shuffle);
    assert_eq!(seed1.weight_init, seed2.weight_init);
    assert_eq!(seed1.dropout, seed2.dropout);
}

#[test]
fn test_experiment_seed_different_masters() {
    let seed1 = ExperimentSeed::from_master(42);
    let seed2 = ExperimentSeed::from_master(123);
    assert_ne!(seed1.data_shuffle, seed2.data_shuffle);
}

// =========================================================================
// Additional coverage tests
// =========================================================================

#[test]
fn test_training_guard_check_weights() {
    let guard = TrainingGuard::new("test_weights");
    let weights = vec![1.0, 2.0, 3.0, -1.0, 0.5];
    assert!(guard.check_weights(&weights).is_ok());
}

#[test]
fn test_training_guard_check_weights_nan() {
    let guard = TrainingGuard::new("test_weights");
    let weights = vec![1.0, f32::NAN, 3.0];
    let result = guard.check_weights(&weights);
    assert!(result.is_err());
}

#[test]
fn test_training_guard_check_weights_inf() {
    let guard = TrainingGuard::new("test_weights");
    let weights = vec![1.0, f32::NEG_INFINITY, 3.0];
    let result = guard.check_weights(&weights);
    assert!(result.is_err());
}

#[test]
fn test_training_guard_check_f64_valid() {
    let guard = TrainingGuard::new("test_f64");
    let values = vec![1.0f64, 2.0, 3.0, -0.5];
    assert!(guard.check_f64(&values, "activations").is_ok());
}

#[test]
fn test_training_guard_check_f64_nan() {
    let guard = TrainingGuard::new("test_f64");
    let values = vec![1.0f64, f64::NAN, 3.0];
    let result = guard.check_f64(&values, "activations");
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("NaN"));
}

#[test]
fn test_training_guard_check_f64_inf() {
    let guard = TrainingGuard::new("test_f64");
    let values = vec![1.0f64, f64::INFINITY, 3.0];
    let result = guard.check_f64(&values, "activations");
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("Inf"));
}

#[test]
fn test_training_guard_check_f64_neg_inf() {
    let guard = TrainingGuard::new("test_f64");
    let values = vec![1.0f64, f64::NEG_INFINITY, 3.0];
    let result = guard.check_f64(&values, "outputs");
    assert!(result.is_err());
}

#[test]
fn test_training_guard_debug() {
    let guard = TrainingGuard::new("debug_test");
    let debug_str = format!("{:?}", guard);
    assert!(debug_str.contains("TrainingGuard"));
}

#[test]
fn test_training_guard_clone() {
    let guard = TrainingGuard::new("clone_test");
    let cloned = guard.clone();
    assert_eq!(cloned.context, guard.context);
}

#[test]
fn test_training_guard_loss_neg_inf() {
    let guard = TrainingGuard::new("test");
    assert!(guard.check_loss(f32::NEG_INFINITY).is_err());
}

#[test]
fn test_divergence_guard_default_tolerance() {
    let guard = DivergenceGuard::default_tolerance("default_test");
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.000001, 2.000001, 3.000001];
    assert!(guard.check(&a, &b).is_ok());
}

#[test]
fn test_divergence_guard_debug() {
    let guard = DivergenceGuard::new(0.01, "debug_test");
    let debug_str = format!("{:?}", guard);
    assert!(debug_str.contains("DivergenceGuard"));
}

#[test]
fn test_divergence_guard_clone() {
    let guard = DivergenceGuard::new(0.01, "clone_test");
    let _cloned = guard.clone();
    // Just verify clone works
}

#[test]
fn test_experiment_seed_default() {
    let seed = ExperimentSeed::default();
    assert_eq!(seed.master, 42);
}

#[test]
fn test_experiment_seed_new_const() {
    let seed = ExperimentSeed::new(100, 200, 300, 400);
    assert_eq!(seed.master, 100);
    assert_eq!(seed.data_shuffle, 200);
    assert_eq!(seed.weight_init, 300);
    assert_eq!(seed.dropout, 400);
}

#[test]
fn test_experiment_seed_debug() {
    let seed = ExperimentSeed::from_master(42);
    let debug_str = format!("{:?}", seed);
    assert!(debug_str.contains("ExperimentSeed"));
    assert!(debug_str.contains("master"));
}

#[test]
fn test_experiment_seed_clone() {
    let seed = ExperimentSeed::from_master(42);
    let cloned = seed; // Copy
    assert_eq!(cloned.master, seed.master);
}

#[test]
fn test_experiment_seed_copy() {
    let seed = ExperimentSeed::from_master(99);
    let copied: ExperimentSeed = seed;
    assert_eq!(copied.master, 99);
    assert_eq!(seed.master, 99); // Original still accessible
}

#[test]
fn test_select_backend_boundary_small() {
    // Just under parallel threshold
    let cat = select_backend(999, false);
    assert_eq!(cat, BackendCategory::SimdOnly);

    // At parallel threshold
    let cat = select_backend(1000, false);
    assert_eq!(cat, BackendCategory::SimdParallel);
}

#[test]
fn test_select_backend_boundary_large() {
    // Just under GPU threshold
    let cat = select_backend(99_999, true);
    assert_eq!(cat, BackendCategory::SimdParallel);

    // At GPU threshold
    let cat = select_backend(100_000, true);
    assert_eq!(cat, BackendCategory::Gpu);
}

#[test]
fn test_should_use_gpu_boundary() {
    assert!(!should_use_gpu(99_999));
    assert!(should_use_gpu(100_000));
}

#[test]
fn test_should_use_parallel_boundary() {
    assert!(!should_use_parallel(999));
    assert!(should_use_parallel(1000));
}

#[test]
fn test_training_guard_empty_gradients() {
    let guard = TrainingGuard::new("empty");
    let gradients: Vec<f32> = vec![];
    assert!(guard.check_gradients(&gradients).is_ok());
}

#[test]
fn test_training_guard_empty_f64() {
    let guard = TrainingGuard::new("empty_f64");
    let values: Vec<f64> = vec![];
    assert!(guard.check_f64(&values, "empty").is_ok());
}

#[test]
fn test_divergence_guard_empty_arrays() {
    let guard = DivergenceGuard::new(0.01, "empty");
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    assert!(guard.check(&a, &b).is_ok());
}

#[test]
fn test_training_guard_error_messages() {
    let guard = TrainingGuard::new("error_context");

    let nan_result = guard.check_loss(f32::NAN);
    assert!(nan_result.is_err());
    let err_msg = nan_result.unwrap_err().to_string();
    assert!(err_msg.contains("error_context"));
    assert!(err_msg.contains("NaN"));
}
