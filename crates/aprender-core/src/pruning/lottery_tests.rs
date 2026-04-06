pub(crate) use super::*;
pub(crate) use crate::nn::Module;

// Mock module for testing
pub(super) struct MockModule {
    weights: Tensor,
}

impl MockModule {
    fn new(data: &[f32], shape: &[usize]) -> Self {
        Self {
            weights: Tensor::new(data, shape),
        }
    }
}

impl Module for MockModule {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.clone()
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weights]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weights]
    }
}

// ==========================================================================
// FALSIFICATION: RewindStrategy
// ==========================================================================
#[test]
fn test_rewind_strategy_default() {
    let strategy = RewindStrategy::default();
    assert_eq!(strategy, RewindStrategy::Init);
}

#[test]
fn test_rewind_strategy_early() {
    let strategy = RewindStrategy::Early { iteration: 500 };
    if let RewindStrategy::Early { iteration } = strategy {
        assert_eq!(iteration, 500);
    } else {
        panic!("Expected Early variant");
    }
}

#[test]
fn test_rewind_strategy_late() {
    let strategy = RewindStrategy::Late { fraction: 0.1 };
    if let RewindStrategy::Late { fraction } = strategy {
        assert!((fraction - 0.1).abs() < 1e-6);
    } else {
        panic!("Expected Late variant");
    }
}

// ==========================================================================
// FALSIFICATION: LotteryTicketConfig
// ==========================================================================
#[test]
fn test_config_new() {
    let config = LotteryTicketConfig::new(0.9, 10);

    assert!((config.target_sparsity - 0.9).abs() < 1e-6);
    assert_eq!(config.pruning_rounds, 10);
    assert_eq!(config.rewind_strategy, RewindStrategy::Init);
    assert!(config.global_pruning);

    // Check per-round rate: (1 - 0.9)^(1/10) ≈ 0.794
    // So prune_rate_per_round ≈ 1 - 0.794 ≈ 0.206
    let expected = 1.0 - 0.1_f32.powf(0.1);
    assert!((config.prune_rate_per_round - expected).abs() < 1e-5);
}

#[test]
fn test_config_clamps_sparsity() {
    let config = LotteryTicketConfig::new(1.5, 10);
    assert!(config.target_sparsity <= 0.99);

    let config = LotteryTicketConfig::new(-0.5, 10);
    assert!(config.target_sparsity >= 0.0);
}

#[test]
fn test_config_min_rounds() {
    let config = LotteryTicketConfig::new(0.9, 0);
    assert_eq!(config.pruning_rounds, 1);
}

#[test]
fn test_config_with_rewind_strategy() {
    let config = LotteryTicketConfig::new(0.9, 10)
        .with_rewind_strategy(RewindStrategy::Early { iteration: 100 });

    assert!(matches!(
        config.rewind_strategy,
        RewindStrategy::Early { iteration: 100 }
    ));
}

#[test]
fn test_config_with_global_pruning() {
    let config = LotteryTicketConfig::new(0.9, 10).with_global_pruning(false);

    assert!(!config.global_pruning);
}

#[test]
fn test_config_default() {
    let config = LotteryTicketConfig::default();

    assert!((config.target_sparsity - 0.9).abs() < 1e-6);
    assert_eq!(config.pruning_rounds, 10);
}

// ==========================================================================
// FALSIFICATION: WinningTicket
// ==========================================================================
#[test]
fn test_winning_ticket_compression_ratio() {
    let mask_tensor = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[4]);
    let mask = SparsityMask::new(mask_tensor, SparsityPattern::Unstructured).unwrap();

    let ticket = WinningTicket {
        mask,
        initial_weights: vec![1.0, 2.0, 3.0, 4.0],
        shape: vec![4],
        sparsity: 0.5,
        remaining_parameters: 2,
        total_parameters: 4,
        sparsity_history: vec![0.25, 0.5],
    };

    // 4 / 2 = 2x compression
    assert!((ticket.compression_ratio() - 2.0).abs() < 1e-6);
}

#[test]
fn test_winning_ticket_density() {
    let mask_tensor = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[4]);
    let mask = SparsityMask::new(mask_tensor, SparsityPattern::Unstructured).unwrap();

    let ticket = WinningTicket {
        mask,
        initial_weights: vec![1.0, 2.0, 3.0, 4.0],
        shape: vec![4],
        sparsity: 0.5,
        remaining_parameters: 2,
        total_parameters: 4,
        sparsity_history: vec![0.5],
    };

    assert!((ticket.density() - 0.5).abs() < 1e-6);
}

#[test]
fn test_winning_ticket_compression_ratio_zero_remaining() {
    let mask_tensor = Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[4]);
    let mask = SparsityMask::new(mask_tensor, SparsityPattern::Unstructured).unwrap();

    let ticket = WinningTicket {
        mask,
        initial_weights: vec![1.0, 2.0, 3.0, 4.0],
        shape: vec![4],
        sparsity: 1.0,
        remaining_parameters: 0,
        total_parameters: 4,
        sparsity_history: vec![1.0],
    };

    assert!(ticket.compression_ratio().is_infinite());
}

// ==========================================================================
// FALSIFICATION: Builder pattern
// ==========================================================================
#[test]
fn test_builder_default() {
    let pruner = LotteryTicketPruner::builder().build();

    assert!((pruner.config().target_sparsity - 0.9).abs() < 1e-6);
    assert_eq!(pruner.config().pruning_rounds, 10);
}

#[test]
fn test_builder_with_target_sparsity() {
    let pruner = LotteryTicketPruner::builder().target_sparsity(0.8).build();

    assert!((pruner.config().target_sparsity - 0.8).abs() < 1e-6);
}

#[test]
fn test_builder_with_pruning_rounds() {
    let pruner = LotteryTicketPruner::builder().pruning_rounds(5).build();

    assert_eq!(pruner.config().pruning_rounds, 5);
}

#[test]
fn test_builder_with_rewind_strategy() {
    let pruner = LotteryTicketPruner::builder()
        .rewind_strategy(RewindStrategy::None)
        .build();

    assert_eq!(pruner.config().rewind_strategy, RewindStrategy::None);
}

#[test]
fn test_builder_full_config() {
    let pruner = LotteryTicketPruner::builder()
        .target_sparsity(0.95)
        .pruning_rounds(20)
        .rewind_strategy(RewindStrategy::Late { fraction: 0.05 })
        .global_pruning(false)
        .build();

    assert!((pruner.config().target_sparsity - 0.95).abs() < 1e-6);
    assert_eq!(pruner.config().pruning_rounds, 20);
    assert!(matches!(
        pruner.config().rewind_strategy,
        RewindStrategy::Late { .. }
    ));
    assert!(!pruner.config().global_pruning);
}

// ==========================================================================
// FALSIFICATION: LotteryTicketPruner construction
// ==========================================================================
#[test]
fn test_pruner_new() {
    let pruner = LotteryTicketPruner::new();
    assert_eq!(pruner.name(), "lottery_ticket_pruner");
}

#[test]
fn test_pruner_default() {
    let pruner = LotteryTicketPruner::default();
    assert!((pruner.config().target_sparsity - 0.9).abs() < 1e-6);
}

#[test]
fn test_pruner_with_config() {
    let config = LotteryTicketConfig::new(0.5, 5);
    let pruner = LotteryTicketPruner::with_config(config);

    assert!((pruner.config().target_sparsity - 0.5).abs() < 1e-6);
    assert_eq!(pruner.config().pruning_rounds, 5);
}

// ==========================================================================
// FALSIFICATION: find_ticket
// ==========================================================================
#[test]
fn test_find_ticket_basic() {
    let module = MockModule::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[8]);
    let pruner = LotteryTicketPruner::builder()
        .target_sparsity(0.5)
        .pruning_rounds(2)
        .build();

    let ticket = pruner.find_ticket(&module).unwrap();

    // Should achieve approximately 50% sparsity
    assert!(ticket.sparsity > 0.4 && ticket.sparsity < 0.6);
    assert_eq!(ticket.total_parameters, 8);
    assert!(ticket.remaining_parameters > 0);
    assert_eq!(ticket.sparsity_history.len(), 2);
}

#[test]
fn test_find_ticket_preserves_initial_weights() {
    let initial_data = [1.0, 2.0, 3.0, 4.0];
    let module = MockModule::new(&initial_data, &[4]);
    let pruner = LotteryTicketPruner::new();

    let ticket = pruner.find_ticket(&module).unwrap();

    // Initial weights should be preserved
    assert_eq!(ticket.initial_weights, initial_data);
}

#[test]
fn test_find_ticket_empty_module_fails() {
    struct EmptyModule;
    impl Module for EmptyModule {
        fn forward(&self, input: &Tensor) -> Tensor {
            input.clone()
        }
        fn parameters(&self) -> Vec<&Tensor> {
            vec![]
        }
        fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
            vec![]
        }
    }

    let module = EmptyModule;
    let pruner = LotteryTicketPruner::new();

    let result = pruner.find_ticket(&module);
    assert!(result.is_err());
}

#[test]
fn test_find_ticket_high_sparsity() {
    let module = MockModule::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[10]);
    let pruner = LotteryTicketPruner::builder()
        .target_sparsity(0.9)
        .pruning_rounds(5)
        .build();

    let ticket = pruner.find_ticket(&module).unwrap();

    // Should achieve approximately 90% sparsity
    assert!(ticket.sparsity > 0.85);
    // Should have ~1 parameter remaining
    assert!(ticket.remaining_parameters >= 1);
}

// ==========================================================================
// FALSIFICATION: apply_ticket
// ==========================================================================
#[test]
fn test_apply_ticket_zeros_weights() {
    let mut module = MockModule::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let pruner = LotteryTicketPruner::builder()
        .target_sparsity(0.5)
        .pruning_rounds(1)
        .rewind_strategy(RewindStrategy::None)
        .build();

    let ticket = pruner.find_ticket(&module).unwrap();
    let result = pruner.apply_ticket(&mut module, &ticket).unwrap();

    // Check that some weights are now zero
    let zeros = module.weights.data().iter().filter(|&&v| v == 0.0).count();
    assert!(zeros > 0);
    assert!(result.achieved_sparsity > 0.0);
}

#[test]
fn test_apply_ticket_with_rewinding() {
    let initial_data = [10.0, 20.0, 30.0, 40.0];
    let mut module = MockModule::new(&initial_data, &[4]);

    // Modify weights to simulate training
    for w in module.weights.data_mut().iter_mut() {
        *w *= 2.0;
    }

    let pruner = LotteryTicketPruner::builder()
        .target_sparsity(0.5)
        .pruning_rounds(1)
        .rewind_strategy(RewindStrategy::Init)
        .build();

    // Create ticket with original weights
    let mask_tensor = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[4]);
    let mask = SparsityMask::new(mask_tensor, SparsityPattern::Unstructured).unwrap();
    let ticket = WinningTicket {
        mask,
        initial_weights: initial_data.to_vec(),
        shape: vec![4],
        sparsity: 0.5,
        remaining_parameters: 2,
        total_parameters: 4,
        sparsity_history: vec![0.5],
    };

    pruner.apply_ticket(&mut module, &ticket).unwrap();

    // Check that remaining weights are rewound to initial values
    let data = module.weights.data();
    assert!((data[0] - 10.0).abs() < 1e-6); // Kept, rewound
    assert_eq!(data[1], 0.0); // Pruned
    assert!((data[2] - 30.0).abs() < 1e-6); // Kept, rewound
    assert_eq!(data[3], 0.0); // Pruned
}

// ==========================================================================
// FALSIFICATION: Pruner trait implementation
// ==========================================================================
#[test]
fn test_pruner_trait_generate_mask() {
    let pruner = LotteryTicketPruner::new();
    let scores =
        ImportanceScores::new(Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]), "test".to_string());

    let mask = pruner
        .generate_mask(&scores, 0.5, SparsityPattern::Unstructured)
        .unwrap();

    assert!((mask.sparsity() - 0.5).abs() < 1e-6);
}

#[test]
fn test_pruner_trait_rejects_structured_patterns() {
    let pruner = LotteryTicketPruner::new();
    let scores =
        ImportanceScores::new(Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]), "test".to_string());

    let result = pruner.generate_mask(&scores, 0.5, SparsityPattern::NM { n: 2, m: 4 });
    assert!(result.is_err());
}

#[test]
fn test_pruner_trait_apply_mask() {
    let pruner = LotteryTicketPruner::new();
    let mut module = MockModule::new(&[1.0, 2.0, 3.0, 4.0], &[4]);

    let mask_tensor = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[4]);
    let mask = SparsityMask::new(mask_tensor, SparsityPattern::Unstructured).unwrap();

    let result = pruner.apply_mask(&mut module, &mask).unwrap();

    assert_eq!(result.parameters_pruned, 2);
    assert_eq!(result.total_parameters, 4);
}

#[test]
fn test_pruner_trait_importance() {
    let pruner = LotteryTicketPruner::new();
    assert!(!pruner.importance().requires_calibration());
}

#[test]
fn test_pruner_trait_name() {
    let pruner = LotteryTicketPruner::new();
    assert_eq!(pruner.name(), "lottery_ticket_pruner");
}

// ==========================================================================
// FALSIFICATION: Iterative pruning convergence
// ==========================================================================
#[test]
fn test_iterative_pruning_converges() {
    let module = MockModule::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[10]);

    let pruner = LotteryTicketPruner::builder()
        .target_sparsity(0.9)
        .pruning_rounds(10)
        .build();

    let ticket = pruner.find_ticket(&module).unwrap();

    // Sparsity should increase monotonically
    for i in 1..ticket.sparsity_history.len() {
        assert!(
            ticket.sparsity_history[i] >= ticket.sparsity_history[i - 1],
            "Sparsity should increase monotonically"
        );
    }

    // Final sparsity should be close to target
    assert!(
        (ticket.sparsity - 0.9).abs() < 0.1,
        "Final sparsity {} should be close to target 0.9",
        ticket.sparsity
    );
}

// ==========================================================================
// FALSIFICATION: Single round equivalence
// ==========================================================================
#[test]
fn test_single_round_equals_one_shot() {
    let module = MockModule::new(&[1.0, 2.0, 3.0, 4.0], &[4]);

    let pruner = LotteryTicketPruner::builder()
        .target_sparsity(0.5)
        .pruning_rounds(1)
        .build();

    let ticket = pruner.find_ticket(&module).unwrap();

    // With 1 round, should get exactly target sparsity
    assert!((ticket.sparsity - 0.5).abs() < 1e-6);
}
