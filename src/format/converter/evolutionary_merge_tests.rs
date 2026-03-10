use super::*;

fn make_model(tensors: Vec<(&str, Vec<f32>, Vec<usize>)>) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    tensors
        .into_iter()
        .map(|(name, data, shape)| (name.to_string(), (data, shape)))
        .collect()
}

fn two_simple_models() -> Vec<BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
    vec![
        make_model(vec![
            ("weight", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
            ("bias", vec![0.1, 0.2], vec![2]),
        ]),
        make_model(vec![
            ("weight", vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]),
            ("bias", vec![0.3, 0.4], vec![2]),
        ]),
    ]
}

// ============================================================================
// Config tests
// ============================================================================

#[test]
fn test_default_config() {
    let cfg = EvolutionaryMergeConfig::default();
    assert_eq!(cfg.num_models, 2);
    assert_eq!(cfg.max_evaluations, 100);
    assert!((cfg.sigma - 0.3).abs() < 1e-6);
    assert_eq!(cfg.seed, 42);
    assert!(!cfg.optimize_density);
    assert!(!cfg.optimize_drop_rate);
}

#[test]
fn test_param_dim_basic() {
    let cfg = EvolutionaryMergeConfig {
        num_models: 3,
        ..Default::default()
    };
    assert_eq!(param_dim(&cfg), 3);
}

#[test]
fn test_param_dim_with_density() {
    let cfg = EvolutionaryMergeConfig {
        num_models: 2,
        optimize_density: true,
        ..Default::default()
    };
    assert_eq!(param_dim(&cfg), 3);
}

#[test]
fn test_param_dim_with_both() {
    let cfg = EvolutionaryMergeConfig {
        num_models: 3,
        optimize_density: true,
        optimize_drop_rate: true,
        ..Default::default()
    };
    assert_eq!(param_dim(&cfg), 5);
}

// ============================================================================
// Decode / softmax tests
// ============================================================================

#[test]
fn test_softmax_normalize_equal() {
    let w = softmax_normalize(&[0.0, 0.0, 0.0]);
    assert_eq!(w.len(), 3);
    let sum: f32 = w.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    // Equal inputs => equal weights
    assert!((w[0] - w[1]).abs() < 1e-5);
}

#[test]
fn test_softmax_normalize_dominant() {
    let w = softmax_normalize(&[10.0, 0.0, 0.0]);
    assert!(w[0] > 0.9); // First weight dominates
    let sum: f32 = w.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_decode_params_basic() {
    let cfg = EvolutionaryMergeConfig {
        num_models: 2,
        ..Default::default()
    };
    let params = [0.0, 0.0];
    let (weights, density, drop_rate) = decode_params(&params, &cfg);
    assert_eq!(weights.len(), 2);
    assert!((weights[0] - 0.5).abs() < 1e-5);
    assert!((density - 0.2).abs() < 1e-5); // default
    assert!((drop_rate - 0.9).abs() < 1e-5); // default
}

#[test]
fn test_decode_params_with_density() {
    let cfg = EvolutionaryMergeConfig {
        num_models: 2,
        optimize_density: true,
        ..Default::default()
    };
    let params = [0.0, 0.0, 0.0]; // sigmoid(0) = 0.5
    let (weights, density, _) = decode_params(&params, &cfg);
    assert_eq!(weights.len(), 2);
    assert!((density - 0.5).abs() < 1e-5);
}

// ============================================================================
// In-memory merge tests
// ============================================================================

#[test]
fn test_merge_weighted() {
    let models = two_simple_models();
    let weights = vec![0.3_f32, 0.7];
    let merged = merge_tensors_in_memory(&models, &weights, MergeStrategy::Weighted);

    let (w, shape) = &merged["weight"];
    assert_eq!(shape, &[2, 2]);
    // w[0] = 1.0*0.3 + 5.0*0.7 = 3.8
    assert!((w[0] - 3.8).abs() < 1e-5);
}

#[test]
fn test_merge_average() {
    let models = two_simple_models();
    let weights = vec![0.5_f32, 0.5];
    let merged = merge_tensors_in_memory(&models, &weights, MergeStrategy::Average);

    let (w, _) = &merged["weight"];
    // w[0] = (1.0 + 5.0) / 2 = 3.0
    assert!((w[0] - 3.0).abs() < 1e-5);
}

#[test]
fn test_merge_slerp() {
    let models = two_simple_models();
    let weights = vec![0.5_f32, 0.5]; // t = weights[1] = 0.5
    let merged = merge_tensors_in_memory(&models, &weights, MergeStrategy::Slerp);

    let (w, shape) = &merged["weight"];
    assert_eq!(shape, &[2, 2]);
    // SLERP result should be between model A and model B values
    assert!(w[0] > 1.0 && w[0] < 5.0);
}

#[test]
fn test_merge_preserves_all_tensors() {
    let models = two_simple_models();
    let weights = vec![0.5_f32, 0.5];
    let merged = merge_tensors_in_memory(&models, &weights, MergeStrategy::Average);
    assert!(merged.contains_key("weight"));
    assert!(merged.contains_key("bias"));
}

// ============================================================================
// Build merge options
// ============================================================================

#[test]
fn test_build_merge_options() {
    let cfg = EvolutionaryMergeConfig::default();
    let opts = build_merge_options(&cfg, vec![0.4, 0.6], 0.3, 0.8);
    assert_eq!(opts.strategy, MergeStrategy::Weighted);
    assert_eq!(opts.weights, Some(vec![0.4, 0.6]));
    assert!((opts.density - 0.3).abs() < 1e-5);
    assert!((opts.drop_rate - 0.8).abs() < 1e-5);
}

// ============================================================================
// Evolutionary optimization
// ============================================================================

#[test]
fn test_evolutionary_merge_runs() {
    let models = two_simple_models();
    let config = EvolutionaryMergeConfig {
        num_models: 2,
        max_evaluations: 50,
        ..Default::default()
    };

    // Objective: minimize L2 norm of merged weight tensor
    let result = evolutionary_merge(&models, &config, |merged| {
        let (w, _) = &merged["weight"];
        w.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt()
    });

    assert_eq!(result.weights.len(), 2);
    let sum: f32 = result.weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4);
    assert!(result.evaluations > 0);
    assert!(result.best_score.is_finite());
}

#[test]
fn test_evolutionary_merge_deterministic() {
    let models = two_simple_models();
    let config = EvolutionaryMergeConfig {
        num_models: 2,
        max_evaluations: 30,
        seed: 123,
        ..Default::default()
    };

    let obj = |merged: &BTreeMap<String, (Vec<f32>, Vec<usize>)>| -> f64 {
        let (w, _) = &merged["weight"];
        w.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>()
    };

    let r1 = evolutionary_merge(&models, &config, &obj);
    let r2 = evolutionary_merge(&models, &config, &obj);
    assert!((r1.best_score - r2.best_score).abs() < 1e-10);
}

#[test]
fn test_evolutionary_merge_finds_good_weights() {
    let models = two_simple_models();
    let config = EvolutionaryMergeConfig {
        num_models: 2,
        max_evaluations: 80,
        ..Default::default()
    };

    // Target: merged weight should be close to [2, 3, 4, 5]
    let target = vec![2.0_f32, 3.0, 4.0, 5.0];
    let result = evolutionary_merge(&models, &config, |merged| {
        let (w, _) = &merged["weight"];
        w.iter()
            .zip(target.iter())
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            .sqrt()
    });

    // Should find weights that produce something close to target
    assert!(result.best_score < 2.0, "CMA-ES should optimize toward target");
}

// ============================================================================
// Falsification tests (Popperian)
// ============================================================================

/// FALSIFY-EVOL-MERGE-001: Weights always sum to 1.0
#[test]
fn falsify_evol_merge_001_weights_sum_to_one() {
    let models = two_simple_models();
    let config = EvolutionaryMergeConfig {
        num_models: 2,
        max_evaluations: 30,
        ..Default::default()
    };

    let result = evolutionary_merge(&models, &config, |merged| {
        let (w, _) = &merged["weight"];
        w.iter().map(|&x| x as f64).sum::<f64>().abs()
    });

    let sum: f32 = result.weights.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "Weights must sum to 1.0, got {}",
        sum
    );
}

/// FALSIFY-EVOL-MERGE-002: CMA-ES improves over random initialization
#[test]
fn falsify_evol_merge_002_improves_over_random() {
    let models = two_simple_models();
    let config = EvolutionaryMergeConfig {
        num_models: 2,
        max_evaluations: 50,
        ..Default::default()
    };

    // Use equal weights as baseline
    let equal = merge_tensors_in_memory(&models, &[0.5, 0.5], MergeStrategy::Weighted);
    let baseline_score = {
        let (w, _) = &equal["weight"];
        w.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>()
    };

    let result = evolutionary_merge(&models, &config, |merged| {
        let (w, _) = &merged["weight"];
        w.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>()
    });

    assert!(
        result.best_score <= baseline_score + 1e-6,
        "CMA-ES should be at least as good as equal weights"
    );
}

/// FALSIFY-EVOL-MERGE-003: Result produces valid MergeOptions
#[test]
fn falsify_evol_merge_003_valid_merge_options() {
    let models = two_simple_models();
    let config = EvolutionaryMergeConfig {
        num_models: 2,
        max_evaluations: 20,
        ..Default::default()
    };

    let result = evolutionary_merge(&models, &config, |merged| {
        let (w, _) = &merged["weight"];
        w.iter().map(|&x| x as f64).sum::<f64>()
    });

    let opts = &result.merge_options;
    assert_eq!(opts.strategy, MergeStrategy::Weighted);
    assert!(opts.weights.is_some());
    assert_eq!(opts.weights.as_ref().unwrap().len(), 2);
}
