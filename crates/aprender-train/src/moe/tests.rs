//! Tests for the Mixture of Experts module

use super::*;
use ndarray::Array2;
use router::{capacity_limit, expert_load_fractions, softmax_rows, RouterConfig};

// ---------------------------------------------------------------------------
// MoeConfig defaults
// ---------------------------------------------------------------------------

#[test]
fn test_moe_config_defaults() {
    let cfg = MoeConfig::default();
    assert_eq!(cfg.num_experts, 8);
    assert_eq!(cfg.top_k, 2);
    assert!((cfg.capacity_factor - 1.25).abs() < 1e-6);
    assert!((cfg.noise_std - 0.0).abs() < 1e-6);
    assert_eq!(cfg.input_dim, 64);
    assert_eq!(cfg.hidden_dim, 128);
}

// ---------------------------------------------------------------------------
// Expert forward pass
// ---------------------------------------------------------------------------

#[test]
fn test_expert_output_shape() {
    let expert = Expert::new(16, 32);
    let input = ndarray::Array1::from(vec![1.0; 16]);
    let output = expert.forward(&input);
    assert_eq!(output.len(), 16, "Expert output dim must match input dim");
}

#[test]
fn test_expert_batch_forward_shape() {
    let expert = Expert::new(16, 32);
    let input = Array2::ones((8, 16));
    let output = expert.forward_batch(&input);
    assert_eq!(output.nrows(), 8);
    assert_eq!(output.ncols(), 16);
}

#[test]
fn test_expert_relu_activation() {
    // Expert with all-negative hidden pre-activations should produce outputs
    // that are influenced only by biases (since ReLU zeros out negatives).
    let mut expert = Expert::new(4, 8);
    // Set W1 so that input @ W1 + b1 is all negative
    expert.w1 = Array2::from_elem((4, 8), -10.0);
    expert.b1 = ndarray::Array1::from(vec![-1.0; 8]);
    // After ReLU, hidden = 0, so output = 0 @ W2 + b2 = b2
    expert.b2 = ndarray::Array1::from(vec![42.0; 4]);

    let input = ndarray::Array1::ones(4);
    let output = expert.forward(&input);
    for &v in &output {
        assert!((v - 42.0).abs() < 1e-5, "Output should equal b2 when ReLU zeros hidden layer");
    }
}

#[test]
fn test_expert_deterministic() {
    let expert = Expert::new(8, 16);
    let input = ndarray::Array1::from(vec![0.5; 8]);
    let out1 = expert.forward(&input);
    let out2 = expert.forward(&input);
    assert_eq!(out1, out2, "Expert forward must be deterministic");
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

#[test]
fn test_softmax_rows_sum_to_one() {
    let logits = Array2::from_shape_fn((4, 8), |(i, j)| (i * 8 + j) as f32 * 0.1);
    let probs = softmax_rows(&logits);
    for row in probs.rows() {
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Softmax row sum should be 1.0, got {sum}");
    }
}

#[test]
fn test_softmax_rows_non_negative() {
    let logits = Array2::from_shape_fn((4, 8), |(i, j)| -((i * 8 + j) as f32));
    let probs = softmax_rows(&logits);
    for &v in &probs {
        assert!(v >= 0.0, "Softmax values must be non-negative");
    }
}

#[test]
fn test_softmax_max_gets_highest_prob() {
    let mut logits = Array2::zeros((1, 4));
    logits[[0, 2]] = 10.0; // Expert 2 has much higher logit
    let probs = softmax_rows(&logits);
    let row = probs.row(0);
    let max_idx = row
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("operation should succeed"))
        .expect("operation should succeed")
        .0;
    assert_eq!(max_idx, 2, "Highest logit should get highest probability");
}

// ---------------------------------------------------------------------------
// Capacity limit computation
// ---------------------------------------------------------------------------

#[test]
fn test_capacity_limit_basic() {
    // 8 tokens, top_k=2, 4 experts, factor=1.0
    // raw = 1.0 * 8 * 2 / 4 = 4.0 -> capacity = 4
    let cap = capacity_limit(8, 2, 4, 1.0);
    assert_eq!(cap, 4);
}

#[test]
fn test_capacity_limit_rounds_up() {
    // 7 tokens, top_k=2, 4 experts, factor=1.0
    // raw = 1.0 * 7 * 2 / 4 = 3.5 -> ceil = 4
    let cap = capacity_limit(7, 2, 4, 1.0);
    assert_eq!(cap, 4);
}

#[test]
fn test_capacity_limit_minimum_one() {
    // Very small capacity factor, but capacity should be at least 1
    let cap = capacity_limit(1, 1, 100, 0.01);
    assert!(cap >= 1, "Capacity must be at least 1");
}

#[test]
fn test_capacity_limit_with_factor() {
    // 8 tokens, top_k=1, 4 experts, factor=1.5
    // raw = 1.5 * 8 * 1 / 4 = 3.0 -> capacity = 3
    let cap = capacity_limit(8, 1, 4, 1.5);
    assert_eq!(cap, 3);
}

// ---------------------------------------------------------------------------
// TopKRouter
// ---------------------------------------------------------------------------

#[test]
fn test_top_k_router_selects_k_experts() {
    let config = RouterConfig {
        input_dim: 16,
        num_experts: 8,
        top_k: 2,
        capacity_factor: 2.0, // generous capacity so no dropping
    };
    let router = TopKRouter::new(&config);
    let input = Array2::from_shape_fn((4, 16), |(i, j)| (i * 16 + j) as f32 * 0.01);
    let result = router.route(&input);

    assert_eq!(result.expert_indices.len(), 4, "One assignment per token");
    for indices in &result.expert_indices {
        assert_eq!(indices.len(), 2, "Each token must be routed to top_k=2 experts");
    }
}

#[test]
fn test_top_k_router_weights_sum_to_one() {
    let config = RouterConfig { input_dim: 16, num_experts: 4, top_k: 2, capacity_factor: 2.0 };
    let router = TopKRouter::new(&config);
    let input = Array2::ones((8, 16));
    let result = router.route(&input);

    for weights in &result.expert_weights {
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "Routing weights must sum to 1.0, got {sum}");
    }
}

#[test]
fn test_top_k_router_deterministic() {
    let config = RouterConfig { input_dim: 8, num_experts: 4, top_k: 2, capacity_factor: 2.0 };
    let router = TopKRouter::new(&config);
    let input = Array2::from_shape_fn((4, 8), |(i, j)| (i + j) as f32 * 0.1);

    let r1 = router.route(&input);
    let r2 = router.route(&input);

    assert_eq!(
        r1.expert_indices, r2.expert_indices,
        "Deterministic router must give same assignments"
    );
}

#[test]
fn test_top_k_router_routing_probs_shape() {
    let config = RouterConfig { input_dim: 8, num_experts: 6, top_k: 2, capacity_factor: 2.0 };
    let router = TopKRouter::new(&config);
    let input = Array2::ones((5, 8));
    let result = router.route(&input);

    assert_eq!(result.routing_probs.nrows(), 5);
    assert_eq!(result.routing_probs.ncols(), 6);
}

// ---------------------------------------------------------------------------
// Capacity enforcement
// ---------------------------------------------------------------------------

#[test]
fn test_capacity_enforcement_limits_expert_usage() {
    // 8 tokens, 4 experts, top_k=1, capacity_factor=1.0
    // capacity = ceil(1.0 * 8 * 1 / 4) = 2
    // So each expert can handle at most 2 tokens.
    // Use varied input so tokens naturally want different experts.
    let config = RouterConfig { input_dim: 8, num_experts: 4, top_k: 1, capacity_factor: 1.0 };
    let router = TopKRouter::new(&config);
    let input = Array2::from_shape_fn((8, 8), |(i, j)| ((i * 8 + j) as f32 * 1.23).sin());
    let result = router.route(&input);

    // Count tokens assigned to each expert (only counting non-zero-weight slots)
    let mut counts = vec![0usize; 4];
    for (token_idx, indices) in result.expert_indices.iter().enumerate() {
        for (k, &expert_idx) in indices.iter().enumerate() {
            if result.expert_weights[token_idx][k] > 0.0 {
                counts[expert_idx] += 1;
            }
        }
    }

    let capacity = capacity_limit(8, 1, 4, 1.0);
    for (expert_id, &count) in counts.iter().enumerate() {
        assert!(
            count <= capacity,
            "Expert {expert_id} got {count} tokens (non-zero weight), but capacity is {capacity}"
        );
    }
}

// ---------------------------------------------------------------------------
// NoisyTopKRouter
// ---------------------------------------------------------------------------

#[test]
fn test_noisy_router_returns_valid_results() {
    let config = RouterConfig { input_dim: 8, num_experts: 4, top_k: 2, capacity_factor: 2.0 };
    let router = NoisyTopKRouter::new(&config, 0.1);
    let input = Array2::ones((4, 8));
    let result = router.route(&input);

    assert_eq!(result.expert_indices.len(), 4);
    for indices in &result.expert_indices {
        assert_eq!(indices.len(), 2);
        for &idx in indices {
            assert!(idx < 4, "Expert index must be < num_experts");
        }
    }
}

#[test]
fn test_noisy_router_weights_sum_to_one() {
    let config = RouterConfig { input_dim: 8, num_experts: 4, top_k: 2, capacity_factor: 2.0 };
    let router = NoisyTopKRouter::new(&config, 0.5);
    let input = Array2::from_shape_fn((10, 8), |(i, j)| (i + j) as f32);
    let result = router.route(&input);

    for weights in &result.expert_weights {
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "Noisy router weights must sum to 1.0, got {sum}");
    }
}

// ---------------------------------------------------------------------------
// Expert load fractions
// ---------------------------------------------------------------------------

#[test]
fn test_expert_load_fractions_uniform() {
    // Uniform distribution: each expert gets equal probability
    let batch = 4;
    let experts = 4;
    let probs = Array2::from_elem((batch, experts), 1.0 / experts as f32);
    let fractions = expert_load_fractions(&probs);

    for &f in &fractions {
        assert!((f - 0.25).abs() < 1e-5, "Uniform probs should give equal load fractions");
    }
}

#[test]
fn test_expert_load_fractions_skewed() {
    // All probability on expert 0
    let mut probs = Array2::zeros((4, 4));
    for i in 0..4 {
        probs[[i, 0]] = 1.0;
    }
    let fractions = expert_load_fractions(&probs);
    assert!((fractions[0] - 1.0).abs() < 1e-5, "Expert 0 should have all the load");
    for i in 1..4 {
        assert!(fractions[i].abs() < 1e-5, "Expert {i} should have zero load");
    }
}

#[test]
fn test_expert_load_fractions_empty_batch() {
    let probs = Array2::zeros((0, 4));
    let fractions = expert_load_fractions(&probs);
    assert_eq!(fractions.len(), 4);
    for &f in &fractions {
        assert!((f - 0.0).abs() < 1e-6);
    }
}

// ---------------------------------------------------------------------------
// MoeLayer forward
// ---------------------------------------------------------------------------

#[test]
fn test_moe_layer_forward_output_shape() {
    let config = MoeConfig {
        num_experts: 4,
        top_k: 2,
        capacity_factor: 2.0,
        noise_std: 0.0,
        input_dim: 8,
        hidden_dim: 16,
    };
    let layer = MoeLayer::new(config);
    let input = Array2::from_shape_fn((6, 8), |(i, j)| (i + j) as f32 * 0.1);
    let (output, routing) = layer.forward(&input);

    assert_eq!(output.nrows(), 6, "Batch size preserved");
    assert_eq!(output.ncols(), 8, "Output dim matches input dim");
    assert_eq!(routing.expert_indices.len(), 6);
}

#[test]
fn test_moe_layer_forward_nonzero_output() {
    let config = MoeConfig {
        num_experts: 4,
        top_k: 2,
        capacity_factor: 2.0,
        noise_std: 0.0,
        input_dim: 8,
        hidden_dim: 16,
    };
    let layer = MoeLayer::new(config);
    let input = Array2::ones((4, 8));
    let (output, _) = layer.forward(&input);

    // At least some outputs should be non-zero (experts have non-trivial weights)
    let any_nonzero = output.iter().any(|&v| v.abs() > 1e-10);
    assert!(any_nonzero, "MoE output should not be all zeros");
}

#[test]
fn test_moe_layer_forward_deterministic() {
    let config = MoeConfig {
        num_experts: 4,
        top_k: 2,
        capacity_factor: 2.0,
        noise_std: 0.0, // deterministic
        input_dim: 8,
        hidden_dim: 16,
    };
    let layer = MoeLayer::new(config);
    let input = Array2::from_shape_fn((4, 8), |(i, j)| (i * 8 + j) as f32 * 0.05);

    let (out1, _) = layer.forward(&input);
    let (out2, _) = layer.forward(&input);

    assert_eq!(out1, out2, "Deterministic MoE should produce identical outputs");
}

#[test]
fn test_moe_layer_uses_multiple_experts() {
    let config = MoeConfig {
        num_experts: 4,
        top_k: 2,
        capacity_factor: 2.0,
        noise_std: 0.0,
        input_dim: 8,
        hidden_dim: 16,
    };
    let layer = MoeLayer::new(config);
    // Use varied input so different tokens route to different experts
    let input = Array2::from_shape_fn((16, 8), |(i, j)| ((i * 8 + j) as f32 * 1.23).sin());
    let (_, routing) = layer.forward(&input);

    // Collect all unique experts used
    let mut used_experts: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for indices in &routing.expert_indices {
        for &idx in indices {
            used_experts.insert(idx);
        }
    }

    assert!(
        used_experts.len() > 1,
        "With varied inputs, multiple experts should be used; got {used_experts:?}"
    );
}

// ---------------------------------------------------------------------------
// Balance loss
// ---------------------------------------------------------------------------

#[test]
fn test_balance_loss_uniform_near_one() {
    // With perfectly uniform routing, balance_loss should be close to 1.0
    let config = MoeConfig {
        num_experts: 4,
        top_k: 1,
        capacity_factor: 2.0,
        noise_std: 0.0,
        input_dim: 4,
        hidden_dim: 8,
    };
    let layer = MoeLayer::new(config);

    // Construct a routing result where every expert gets exactly 1/4 of tokens
    // and uniform routing probs
    let num_experts = 4;
    let batch = 8;
    let routing = RoutingResult {
        expert_indices: (0..batch).map(|i| vec![i % num_experts]).collect(),
        expert_weights: (0..batch).map(|_| vec![1.0]).collect(),
        routing_probs: Array2::from_elem((batch, num_experts), 0.25),
    };

    let loss = layer.balance_loss(&routing);
    // For perfectly uniform: f_i = 0.25, P_i = 0.25, N * sum(f_i * P_i) = 4 * 4 * 0.0625 = 1.0
    assert!(
        (loss - 1.0).abs() < 1e-4,
        "Perfectly balanced routing should give loss ~1.0, got {loss}"
    );
}

#[test]
fn test_balance_loss_skewed_exceeds_one() {
    let config = MoeConfig {
        num_experts: 4,
        top_k: 1,
        capacity_factor: 2.0,
        noise_std: 0.0,
        input_dim: 4,
        hidden_dim: 8,
    };
    let layer = MoeLayer::new(config);

    // All tokens routed to expert 0, with all probability mass on expert 0
    let batch = 8;
    let mut probs = Array2::zeros((batch, 4));
    for i in 0..batch {
        probs[[i, 0]] = 1.0;
    }

    let routing = RoutingResult {
        expert_indices: (0..batch).map(|_| vec![0]).collect(),
        expert_weights: (0..batch).map(|_| vec![1.0]).collect(),
        routing_probs: probs,
    };

    let loss = layer.balance_loss(&routing);
    // f = [1, 0, 0, 0], P = [1, 0, 0, 0], loss = 4 * (1*1) = 4.0
    assert!(loss > 1.0, "Skewed routing should produce loss > 1.0, got {loss}");
    assert!(
        (loss - 4.0).abs() < 1e-4,
        "All-on-one routing with 4 experts should give loss = 4.0, got {loss}"
    );
}

#[test]
fn test_balance_loss_empty_batch() {
    let config = MoeConfig {
        num_experts: 4,
        top_k: 1,
        capacity_factor: 2.0,
        noise_std: 0.0,
        input_dim: 4,
        hidden_dim: 8,
    };
    let layer = MoeLayer::new(config);

    let routing = RoutingResult {
        expert_indices: vec![],
        expert_weights: vec![],
        routing_probs: Array2::zeros((0, 4)),
    };

    let loss = layer.balance_loss(&routing);
    assert!((loss - 0.0).abs() < 1e-6, "Empty batch should give zero loss, got {loss}");
}

#[test]
fn test_balance_loss_from_real_forward() {
    let config = MoeConfig {
        num_experts: 4,
        top_k: 2,
        capacity_factor: 2.0,
        noise_std: 0.0,
        input_dim: 8,
        hidden_dim: 16,
    };
    let layer = MoeLayer::new(config);
    let input = Array2::from_shape_fn((16, 8), |(i, j)| ((i * 8 + j) as f32 * 0.77).sin());
    let (_, routing) = layer.forward(&input);

    let loss = layer.balance_loss(&routing);
    assert!(loss > 0.0, "Balance loss should be positive");
    assert!(loss.is_finite(), "Balance loss should be finite");
}

// ---------------------------------------------------------------------------
// MoeLayer with noisy router
// ---------------------------------------------------------------------------

#[test]
fn test_moe_layer_noisy_router() {
    let config = MoeConfig {
        num_experts: 4,
        top_k: 2,
        capacity_factor: 2.0,
        noise_std: 1.0, // high noise
        input_dim: 8,
        hidden_dim: 16,
    };
    let layer = MoeLayer::new(config);
    let input = Array2::ones((4, 8));
    let (output, routing) = layer.forward(&input);

    assert_eq!(output.nrows(), 4);
    assert_eq!(output.ncols(), 8);
    assert_eq!(routing.expert_indices.len(), 4);
}

// ---------------------------------------------------------------------------
// Router enum dispatch
// ---------------------------------------------------------------------------

#[test]
fn test_router_enum_deterministic() {
    let config = RouterConfig { input_dim: 8, num_experts: 4, top_k: 2, capacity_factor: 2.0 };
    let router = Router::Deterministic(TopKRouter::new(&config));
    let input = Array2::ones((4, 8));
    let result = router.route(&input);
    assert_eq!(result.expert_indices.len(), 4);
}

#[test]
fn test_router_enum_noisy() {
    let config = RouterConfig { input_dim: 8, num_experts: 4, top_k: 2, capacity_factor: 2.0 };
    let router = Router::Noisy(NoisyTopKRouter::new(&config, 0.5));
    let input = Array2::ones((4, 8));
    let result = router.route(&input);
    assert_eq!(result.expert_indices.len(), 4);
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_single_expert() {
    let config = MoeConfig {
        num_experts: 1,
        top_k: 1,
        capacity_factor: 2.0,
        noise_std: 0.0,
        input_dim: 4,
        hidden_dim: 8,
    };
    let layer = MoeLayer::new(config);
    let input = Array2::ones((3, 4));
    let (output, routing) = layer.forward(&input);

    assert_eq!(output.nrows(), 3);
    // With a single expert, all tokens must go to expert 0
    for indices in &routing.expert_indices {
        assert_eq!(indices[0], 0);
    }
}

#[test]
fn test_top_k_equals_num_experts() {
    // When top_k == num_experts, every expert is used for every token
    let config = MoeConfig {
        num_experts: 3,
        top_k: 3,
        capacity_factor: 3.0,
        noise_std: 0.0,
        input_dim: 4,
        hidden_dim: 8,
    };
    let layer = MoeLayer::new(config);
    let input = Array2::from_shape_fn((2, 4), |(i, j)| (i + j) as f32);
    let (output, routing) = layer.forward(&input);

    assert_eq!(output.nrows(), 2);
    for indices in &routing.expert_indices {
        assert_eq!(indices.len(), 3, "All 3 experts should be selected");
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 3, "All experts should be distinct");
    }
}

#[test]
fn test_single_token_batch() {
    let config = MoeConfig {
        num_experts: 4,
        top_k: 2,
        capacity_factor: 2.0,
        noise_std: 0.0,
        input_dim: 8,
        hidden_dim: 16,
    };
    let layer = MoeLayer::new(config);
    let input = Array2::ones((1, 8));
    let (output, _) = layer.forward(&input);
    assert_eq!(output.nrows(), 1);
    assert_eq!(output.ncols(), 8);
}
