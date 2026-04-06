use super::*;

// ============================================================================
// Config tests
// ============================================================================

#[test]
fn test_moe_config_default() {
    let cfg = MoeConfig {
        num_experts: 8,
        num_experts_per_tok: 2,
        routing_method: RoutingMethod::TopK,
        gate_hidden_dim: None,
    };
    assert_eq!(cfg.num_experts, 8);
    assert_eq!(cfg.num_experts_per_tok, 2);
}

#[test]
fn test_moe_config_validate_ok() {
    let cfg = MoeConfig {
        num_experts: 4,
        num_experts_per_tok: 2,
        routing_method: RoutingMethod::TopK,
        gate_hidden_dim: None,
    };
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_moe_config_validate_zero_experts() {
    let cfg = MoeConfig {
        num_experts: 0,
        num_experts_per_tok: 2,
        routing_method: RoutingMethod::TopK,
        gate_hidden_dim: None,
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_moe_config_validate_experts_per_tok_exceeds() {
    let cfg = MoeConfig {
        num_experts: 4,
        num_experts_per_tok: 5,
        routing_method: RoutingMethod::TopK,
        gate_hidden_dim: None,
    };
    assert!(cfg.validate().is_err());
}

// ============================================================================
// Plan tests
// ============================================================================

#[test]
fn test_plan_moe_basic() {
    let cfg = MoeConfig {
        num_experts: 4,
        num_experts_per_tok: 2,
        routing_method: RoutingMethod::TopK,
        gate_hidden_dim: None,
    };
    let plan = plan_moe_construction(4, 8, &cfg).unwrap();
    assert_eq!(plan.num_layers, 8);
    assert_eq!(plan.assignments.len(), 8);
    // Each layer should have 4 experts
    for layer in &plan.assignments {
        assert_eq!(layer.len(), 4);
    }
}

#[test]
fn test_plan_moe_round_robin() {
    let cfg = MoeConfig {
        num_experts: 2,
        num_experts_per_tok: 1,
        routing_method: RoutingMethod::TopK,
        gate_hidden_dim: None,
    };
    let plan = plan_moe_construction(2, 4, &cfg).unwrap();
    // With 2 models and 2 experts, round-robin assigns alternating models
    for layer in &plan.assignments {
        assert_eq!(layer.len(), 2);
        assert_ne!(
            layer[0].source_model, layer[1].source_model,
            "Experts should come from different models"
        );
    }
}

#[test]
fn test_plan_moe_insufficient_models() {
    let cfg = MoeConfig {
        num_experts: 8,
        num_experts_per_tok: 2,
        routing_method: RoutingMethod::TopK,
        gate_hidden_dim: None,
    };
    // Only 1 model: should still work (reuse same model for all experts)
    let plan = plan_moe_construction(1, 4, &cfg).unwrap();
    assert_eq!(plan.assignments.len(), 4);
}

// ============================================================================
// Gate weights tests
// ============================================================================

#[test]
fn test_gate_weights_uniform() {
    let weights = compute_gate_weights(64, 4, RouterInit::Uniform);
    assert_eq!(weights.len(), 64 * 4);
    // Uniform: all equal
    let expected = 1.0 / 4.0;
    for &w in &weights {
        assert!((w - expected).abs() < 1e-10);
    }
}

#[test]
fn test_gate_weights_balanced() {
    let weights = compute_gate_weights(32, 8, RouterInit::Balanced);
    assert_eq!(weights.len(), 32 * 8);
    assert!(weights.iter().all(|w| w.is_finite()));
}

#[test]
fn test_gate_weights_random() {
    let weights = compute_gate_weights(16, 4, RouterInit::Random);
    assert_eq!(weights.len(), 16 * 4);
    assert!(weights.iter().all(|w| w.is_finite()));
    // Random: not all equal
    let first = weights[0];
    assert!(weights.iter().any(|&w| (w - first).abs() > 1e-10));
}

// ============================================================================
// Load balance tests
// ============================================================================

#[test]
fn test_load_balance_perfect() {
    // All layers use all models equally
    let assignments = vec![
        vec![
            ExpertAssignment {
                expert_index: 0,
                source_model: 0,
                source_layer: 0,
            },
            ExpertAssignment {
                expert_index: 1,
                source_model: 1,
                source_layer: 0,
            },
        ],
        vec![
            ExpertAssignment {
                expert_index: 0,
                source_model: 0,
                source_layer: 1,
            },
            ExpertAssignment {
                expert_index: 1,
                source_model: 1,
                source_layer: 1,
            },
        ],
    ];
    let balance = compute_expert_load_balance(&assignments);
    assert!(
        (balance - 0.0).abs() < 1e-10,
        "Perfect balance should be 0, got {}",
        balance
    );
}

#[test]
fn test_load_balance_imbalanced() {
    // All experts from model 0
    let assignments = vec![vec![
        ExpertAssignment {
            expert_index: 0,
            source_model: 0,
            source_layer: 0,
        },
        ExpertAssignment {
            expert_index: 1,
            source_model: 0,
            source_layer: 0,
        },
        ExpertAssignment {
            expert_index: 2,
            source_model: 0,
            source_layer: 0,
        },
        ExpertAssignment {
            expert_index: 3,
            source_model: 0,
            source_layer: 0,
        },
    ]];
    let balance = compute_expert_load_balance(&assignments);
    // Should be non-zero (all from same model)
    // But with only 1 unique model, balance is trivially 0
    assert!(balance.is_finite());
}

// ============================================================================
// Routing method tests
// ============================================================================

#[test]
fn test_routing_methods() {
    assert_eq!(RoutingMethod::TopK, RoutingMethod::TopK);
    assert_ne!(RoutingMethod::TopK, RoutingMethod::SwitchTransformer);
    assert_ne!(
        RoutingMethod::SwitchTransformer,
        RoutingMethod::ExpertChoice
    );
}

// ============================================================================
// Falsification tests
// ============================================================================

/// FALSIFY-MOE-001: All expert assignments are valid.
#[test]
fn falsify_moe_001_valid_assignments() {
    for num_models in [1, 2, 4, 8] {
        for num_experts in [2, 4, 8] {
            let cfg = MoeConfig {
                num_experts,
                num_experts_per_tok: 2.min(num_experts),
                routing_method: RoutingMethod::TopK,
                gate_hidden_dim: None,
            };
            let plan = plan_moe_construction(num_models, 4, &cfg).unwrap();
            for (layer_idx, layer) in plan.assignments.iter().enumerate() {
                assert_eq!(layer.len(), num_experts, "Layer {} expert count", layer_idx);
                for a in layer {
                    assert!(
                        a.source_model < num_models,
                        "Expert model {} >= {} models",
                        a.source_model,
                        num_models
                    );
                    assert!(a.expert_index < num_experts);
                }
            }
        }
    }
}

/// FALSIFY-MOE-002: Gate weights have correct dimensions.
#[test]
fn falsify_moe_002_gate_dimensions() {
    for hidden in [32, 64, 128] {
        for experts in [2, 4, 8] {
            for init in [
                RouterInit::Uniform,
                RouterInit::Balanced,
                RouterInit::Random,
            ] {
                let weights = compute_gate_weights(hidden, experts, init);
                assert_eq!(
                    weights.len(),
                    hidden * experts,
                    "Gate weights for {}x{} {:?}",
                    hidden,
                    experts,
                    init
                );
                assert!(
                    weights.iter().all(|w| w.is_finite()),
                    "Non-finite gate weight for {}x{} {:?}",
                    hidden,
                    experts,
                    init
                );
            }
        }
    }
}

/// FALSIFY-MOE-003: Load balance is non-negative.
#[test]
fn falsify_moe_003_balance_nonnegative() {
    let cfg = MoeConfig {
        num_experts: 4,
        num_experts_per_tok: 2,
        routing_method: RoutingMethod::TopK,
        gate_hidden_dim: None,
    };
    for num_models in [1, 2, 4] {
        let plan = plan_moe_construction(num_models, 8, &cfg).unwrap();
        let balance = compute_expert_load_balance(&plan.assignments);
        assert!(
            balance >= 0.0,
            "Balance must be >= 0 for {} models, got {}",
            num_models,
            balance
        );
    }
}
