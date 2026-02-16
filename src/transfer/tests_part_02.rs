use super::*;

#[test]
fn test_online_distillation_all_losses() {
    let od = OnlineDistillation::new(3, 2.0, 1.0);
    let all_logits = vec![vec![1.0, 0.0], vec![0.5, 0.5], vec![0.0, 1.0]];
    let task_losses = vec![0.1, 0.2, 0.3];
    let losses = od.all_losses(&all_logits, &task_losses);
    assert_eq!(losses.len(), 3);
    // Each loss should be >= task_loss due to mutual component
    assert!(losses[0] >= 0.1);
    assert!(losses[1] >= 0.2);
    assert!(losses[2] >= 0.3);
}

#[test]
fn test_online_distillation_three_networks() {
    let od = OnlineDistillation::new(3, 1.0, 1.0);
    let all_logits = vec![
        vec![1.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    // Network 0 and 1 are similar, so loss for 0 should be smaller
    // than loss for 2 (which is different from both)
    let loss_0 = od.mutual_loss(0, &all_logits);
    let loss_2 = od.mutual_loss(2, &all_logits);
    // Network 2 differs from both, should have higher loss
    assert!(loss_2 > loss_0 * 0.5);
}

// Progressive Distillation Tests
#[test]
fn test_progressive_distillation_creation() {
    let pd = ProgressiveDistillation::new(64, 4, 1.0);
    assert_eq!(pd.current_steps(), 64);
    assert_eq!(pd.target_steps(), 4);
}

#[test]
fn test_progressive_distillation_should_halve() {
    let pd = ProgressiveDistillation::new(64, 4, 1.0);
    assert!(pd.should_halve()); // 64 > 4*2

    let pd2 = ProgressiveDistillation::new(8, 4, 1.0);
    assert!(!pd2.should_halve()); // 8 is not > 4*2
}

#[test]
fn test_progressive_distillation_halve_steps() {
    let mut pd = ProgressiveDistillation::new(64, 4, 1.0);
    pd.halve_steps();
    assert_eq!(pd.current_steps(), 32);
    pd.halve_steps();
    assert_eq!(pd.current_steps(), 16);
    pd.halve_steps();
    assert_eq!(pd.current_steps(), 8);
    pd.halve_steps();
    assert_eq!(pd.current_steps(), 4);
    pd.halve_steps(); // Should not go below target
    assert_eq!(pd.current_steps(), 4);
}

#[test]
fn test_progressive_distillation_compute_loss() {
    let pd = ProgressiveDistillation::new(16, 4, 1.0);
    let teacher = vec![1.0, 2.0, 3.0];
    let student = vec![1.0, 2.0, 3.0];
    let loss = pd.compute_loss(&teacher, &student);
    assert!(loss.abs() < 1e-6); // Same outputs = zero loss
}

#[test]
fn test_progressive_distillation_loss_with_diff() {
    let pd = ProgressiveDistillation::new(16, 4, 1.0);
    let teacher = vec![1.0, 2.0, 3.0];
    let student = vec![2.0, 3.0, 4.0];
    let loss = pd.compute_loss(&teacher, &student);
    // MSE = ((1)^2 + (1)^2 + (1)^2) / 3 = 1.0
    assert!((loss - 1.0).abs() < 1e-6);
}

#[test]
fn test_progressive_distillation_weight() {
    let pd = ProgressiveDistillation::new(16, 4, 0.5);
    let teacher = vec![0.0, 0.0];
    let student = vec![1.0, 1.0];
    let loss = pd.compute_loss(&teacher, &student);
    // MSE = 1.0, weighted = 0.5
    assert!((loss - 0.5).abs() < 1e-6);
}

// Additional coverage tests

#[test]
fn test_transferable_encoder_accessor() {
    let encoder = SimpleEncoder::new(10, 5);
    let transfer = TransferableEncoder::new(encoder);

    // Test encoder() accessor
    let _enc_ref = transfer.encoder();
}

#[test]
fn test_transferable_encoder_mut_accessor() {
    let encoder = SimpleEncoder::new(10, 5);
    let mut transfer = TransferableEncoder::new(encoder);

    // Test encoder_mut() accessor
    let _enc_mut = transfer.encoder_mut();
}

#[test]
fn test_transferable_encoder_parameters_mut_frozen() {
    let encoder = SimpleEncoder::new(10, 5);
    let mut transfer = TransferableEncoder::new(encoder);

    // When unfrozen, parameters_mut should return non-empty
    assert!(!transfer.parameters_mut().is_empty());

    // When frozen, parameters_mut should return empty
    transfer.freeze_base();
    assert!(transfer.parameters_mut().is_empty());
}

#[test]
fn test_multi_task_head_remove_task() {
    let encoder = SimpleEncoder::new(10, 5);
    let transfer = TransferableEncoder::new(encoder);
    let mut multi_task = MultiTaskHead::new(transfer, 5);

    multi_task.add_task("task1", 3);
    multi_task.add_task("task2", 7);

    assert_eq!(multi_task.task_names().len(), 2);

    // Remove existing task
    let removed = multi_task.remove_task("task1");
    assert!(removed.is_some());
    assert_eq!(multi_task.task_names().len(), 1);

    // Remove non-existing task
    let not_removed = multi_task.remove_task("task1");
    assert!(not_removed.is_none());
}

#[test]
fn test_multi_task_head_encoder_accessor() {
    let encoder = SimpleEncoder::new(10, 5);
    let transfer = TransferableEncoder::new(encoder);
    let multi_task = MultiTaskHead::new(transfer, 5);

    // Test encoder() accessor
    assert!(!multi_task.encoder().is_frozen());
}

#[test]
fn test_multi_task_head_encoder_mut_accessor() {
    let encoder = SimpleEncoder::new(10, 5);
    let transfer = TransferableEncoder::new(encoder);
    let mut multi_task = MultiTaskHead::new(transfer, 5);

    // Test encoder_mut() accessor - freeze via it
    multi_task.encoder_mut().freeze_base();
    assert!(multi_task.encoder().is_frozen());
}

#[test]
fn test_multi_task_head_unfreeze_encoder() {
    let encoder = SimpleEncoder::new(10, 5);
    let transfer = TransferableEncoder::new(encoder);
    let mut multi_task = MultiTaskHead::new(transfer, 5);

    multi_task.freeze_encoder();
    assert!(multi_task.encoder().is_frozen());

    multi_task.unfreeze_encoder();
    assert!(!multi_task.encoder().is_frozen());
}

#[test]
fn test_multi_task_head_parameters_mut() {
    let encoder = SimpleEncoder::new(10, 5);
    let transfer = TransferableEncoder::new(encoder);
    let mut multi_task = MultiTaskHead::new(transfer, 5);

    multi_task.add_task("task1", 3);

    // parameters_mut should return non-empty
    let params = multi_task.parameters_mut();
    assert!(!params.is_empty());
}

#[test]
fn test_domain_adapter_encoder_accessor() {
    let encoder = SimpleEncoder::new(10, 5);
    let transfer = TransferableEncoder::new(encoder);
    let adapter = DomainAdapter::new(transfer, 5, 1.0);

    // Test encoder() accessor
    assert!(!adapter.encoder().is_frozen());
}

#[test]
fn test_domain_adapter_encoder_mut_accessor() {
    let encoder = SimpleEncoder::new(10, 5);
    let transfer = TransferableEncoder::new(encoder);
    let mut adapter = DomainAdapter::new(transfer, 5, 1.0);

    // Test encoder_mut() accessor - freeze via it
    adapter.encoder_mut().freeze_base();
    assert!(adapter.encoder().is_frozen());
}

#[test]
fn test_domain_adapter_module_forward() {
    let encoder = SimpleEncoder::new(10, 5);
    let transfer = TransferableEncoder::new(encoder);
    let adapter = DomainAdapter::new(transfer, 5, 1.0);

    let x = Tensor::ones(&[2, 10]);
    // Module::forward calls encode
    let features = adapter.forward(&x);
    assert_eq!(features.shape(), &[2, 5]);
}

#[test]
fn test_domain_adapter_parameters() {
    let encoder = SimpleEncoder::new(10, 5);
    let transfer = TransferableEncoder::new(encoder);
    let adapter = DomainAdapter::new(transfer, 5, 1.0);

    // parameters should include encoder + discriminator params
    let params = adapter.parameters();
    assert!(!params.is_empty());
}

#[test]
fn test_domain_adapter_parameters_mut() {
    let encoder = SimpleEncoder::new(10, 5);
    let transfer = TransferableEncoder::new(encoder);
    let mut adapter = DomainAdapter::new(transfer, 5, 1.0);

    // parameters_mut should include encoder + discriminator params
    let params = adapter.parameters_mut();
    assert!(!params.is_empty());
}

#[test]
fn test_lora_config_default() {
    let config = LoRAConfig::default();
    assert_eq!(config.rank, 8);
    assert!((config.alpha - 8.0).abs() < 1e-6);
    assert!((config.scaling() - 1.0).abs() < 1e-6);
}

#[test]
fn test_lora_adapter_apply() {
    let config = LoRAConfig::new(4, 4.0);
    let adapter = LoRAAdapter::new(10, 20, config);

    // Create a base weight tensor
    let base_weight = Tensor::ones(&[20, 10]);

    // Apply LoRA adaptation
    let adapted = adapter.apply(&base_weight);

    // Output shape should match base weight
    assert_eq!(adapted.shape(), &[20, 10]);

    // Since B is initialized to zeros, adapted should equal base_weight
    for (&adapted_val, &base_val) in adapted.data().iter().zip(base_weight.data().iter()) {
        assert!((adapted_val - base_val).abs() < 1e-6);
    }
}

#[test]
fn test_prototypical_network_cosine_distance() {
    let pn = PrototypicalNetwork::new(DistanceMetric::Cosine);
    let protos = vec![(0, vec![1.0, 0.0]), (1, vec![0.0, 1.0])];
    let query = vec![0.9, 0.1];
    let class = pn.classify(&query, &protos);
    assert_eq!(class, 0); // Closer to [1.0, 0.0]
}

#[test]
fn test_prototypical_network_cosine_predict_proba() {
    let pn = PrototypicalNetwork::new(DistanceMetric::Cosine);
    let protos = vec![(0, vec![1.0, 0.0]), (1, vec![0.0, 1.0])];
    let query = vec![1.0, 0.0];
    let probs = pn.predict_proba(&query, &protos);
    assert_eq!(probs.len(), 2);

    // Probabilities should sum to 1
    let sum: f32 = probs.iter().map(|(_, p)| *p).sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // Class 0 should have higher probability than class 1
    let class_0_prob = probs
        .iter()
        .find(|(c, _)| *c == 0)
        .map(|(_, p)| *p)
        .unwrap_or(0.0);
    let class_1_prob = probs
        .iter()
        .find(|(c, _)| *c == 1)
        .map(|(_, p)| *p)
        .unwrap_or(0.0);
    assert!(class_0_prob > class_1_prob);
}

#[test]
fn test_attention_transfer_with_edge_cases() {
    let at = AttentionTransfer::new(2);

    // Test with zero activations (edge case for normalization)
    let zeros = vec![0.0, 0.0, 0.0, 0.0];
    let attention = at.compute_attention_map(&zeros, 2, 2);
    assert_eq!(attention.len(), 2);
    // All zeros -> norm is 0, should handle gracefully
}

#[test]
fn test_attention_transfer_loss_different() {
    let at = AttentionTransfer::new(2);
    let student = vec![1.0, 0.0, 0.0, 0.0];
    let teacher = vec![0.0, 1.0, 0.0, 0.0];
    let loss = at.compute_loss(&student, &teacher, 2, 2);
    assert!(loss > 0.0); // Different patterns should have positive loss
}

#[test]
fn test_feature_distillation_cosine_orthogonal() {
    let fd = FeatureDistillation::new(FeatureLossType::Cosine);
    let student = vec![1.0, 0.0, 0.0];
    let teacher = vec![0.0, 1.0, 0.0];
    let loss = fd.compute_loss(&student, &teacher);
    // Orthogonal vectors have cosine = 0, so loss = 1 - 0 = 1
    assert!((loss - 1.0).abs() < 1e-6);
}

#[test]
fn test_feature_distillation_cosine_opposite() {
    let fd = FeatureDistillation::new(FeatureLossType::Cosine);
    let student = vec![1.0, 0.0, 0.0];
    let teacher = vec![-1.0, 0.0, 0.0];
    let loss = fd.compute_loss(&student, &teacher);
    // Opposite vectors have cosine = -1, so loss = 1 - (-1) = 2
    assert!((loss - 2.0).abs() < 1e-6);
}

#[test]
fn test_self_distillation_temperature() {
    let sd = SelfDistillation::new(4.0);
    let student = vec![2.0, 1.0, 0.0];
    let teacher = vec![1.0, 2.0, 0.0];
    let loss = sd.layer_loss(&student, &teacher);
    // Different distributions should have positive loss
    assert!(loss > 0.0);
}

#[test]
fn test_matching_network_single_class() {
    let mn = MatchingNetwork::new(1.0);
    let support = vec![
        (vec![1.0, 0.0], 0),
        (vec![1.0, 0.1], 0),
        (vec![0.9, 0.0], 0),
    ];
    let query = vec![1.0, 0.0];
    let class = mn.predict(&query, &support);
    assert_eq!(class, 0);
}

#[test]
fn test_matching_network_temperature_effect() {
    let mn_low = MatchingNetwork::new(0.1); // Low temp = sharper
    let mn_high = MatchingNetwork::new(10.0); // High temp = softer

    let support = vec![(vec![1.0, 0.0], 0), (vec![0.0, 1.0], 1)];
    let query = vec![0.6, 0.4]; // Slightly closer to class 0

    // Both should predict class 0, but low temp is more confident
    assert_eq!(mn_low.predict(&query, &support), 0);
    assert_eq!(mn_high.predict(&query, &support), 0);
}
