//! Tests for transfer module.

pub(crate) use super::super::*;

pub(crate) use crate::nn::Linear;

// Simple test encoder for testing
pub(super) struct SimpleEncoder {
    linear: Linear,
    training: bool,
}

impl SimpleEncoder {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            linear: Linear::new(input_dim, output_dim),
            training: true,
        }
    }
}

impl Module for SimpleEncoder {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.linear.forward(input)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.linear.parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.linear.parameters_mut()
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn training(&self) -> bool {
        self.training
    }
}

#[test]
fn test_transferable_encoder_basic() {
    let encoder = SimpleEncoder::new(10, 5);
    let transfer = TransferableEncoder::new(encoder);

    let x = Tensor::ones(&[2, 10]);
    let y = transfer.forward(&x);

    assert_eq!(y.shape(), &[2, 5]);
}

#[test]
fn test_transferable_encoder_freeze_unfreeze() {
    let encoder = SimpleEncoder::new(10, 5);
    let mut transfer = TransferableEncoder::new(encoder);

    // Initially not frozen
    assert!(!transfer.is_frozen());
    assert!(!transfer.parameters().is_empty());

    // Freeze
    transfer.freeze_base();
    assert!(transfer.is_frozen());
    assert!(transfer.parameters().is_empty());

    // Unfreeze
    transfer.unfreeze_base();
    assert!(!transfer.is_frozen());
    assert!(!transfer.parameters().is_empty());
}

#[test]
fn test_transferable_encoder_get_features() {
    let encoder = SimpleEncoder::new(10, 5);
    let transfer = TransferableEncoder::new(encoder);

    let x = Tensor::ones(&[2, 10]);
    let features = transfer.get_features(&x);

    assert_eq!(features.shape(), &[2, 5]);
}

#[test]
fn test_multi_task_head_basic() {
    let encoder = SimpleEncoder::new(10, 5);
    let transfer = TransferableEncoder::new(encoder);
    let mut multi_task = MultiTaskHead::new(transfer, 5);

    multi_task.add_task("task1", 3);
    multi_task.add_task("task2", 7);

    let x = Tensor::ones(&[2, 10]);

    let out1 = multi_task.forward_full("task1", &x);
    let out2 = multi_task.forward_full("task2", &x);

    assert_eq!(out1.shape(), &[2, 3]);
    assert_eq!(out2.shape(), &[2, 7]);
}

#[test]
fn test_multi_task_head_shared_features() {
    let encoder = SimpleEncoder::new(10, 5);
    let transfer = TransferableEncoder::new(encoder);
    let mut multi_task = MultiTaskHead::new(transfer, 5);

    multi_task.add_task("task1", 3);

    let x = Tensor::ones(&[2, 10]);
    let features = multi_task.forward_shared(&x);
    let output = multi_task.forward_task("task1", &features);

    assert_eq!(features.shape(), &[2, 5]);
    assert_eq!(output.shape(), &[2, 3]);
}

#[test]
fn test_multi_task_head_freeze_encoder() {
    let encoder = SimpleEncoder::new(10, 5);
    let transfer = TransferableEncoder::new(encoder);
    let mut multi_task = MultiTaskHead::new(transfer, 5);

    multi_task.add_task("task1", 3);

    // Before freeze: encoder params included
    let params_before = multi_task.parameters().len();

    // Freeze encoder
    multi_task.freeze_encoder();

    // After freeze: only head params
    let params_after = multi_task.parameters().len();

    assert!(params_after < params_before);
}

#[test]
fn test_multi_task_head_task_names() {
    let encoder = SimpleEncoder::new(10, 5);
    let transfer = TransferableEncoder::new(encoder);
    let mut multi_task = MultiTaskHead::new(transfer, 5);

    multi_task.add_task("classification", 10);
    multi_task.add_task("regression", 1);

    let names = multi_task.task_names();
    assert_eq!(names.len(), 2);
    assert!(names.contains(&&"classification".to_string()));
    assert!(names.contains(&&"regression".to_string()));
}

#[test]
fn test_domain_adapter_basic() {
    let encoder = SimpleEncoder::new(10, 5);
    let transfer = TransferableEncoder::new(encoder);
    let adapter = DomainAdapter::new(transfer, 5, 1.0);

    let x = Tensor::ones(&[2, 10]);
    let features = adapter.encode(&x);
    let domain_pred = adapter.discriminate(&features);

    assert_eq!(features.shape(), &[2, 5]);
    assert_eq!(domain_pred.shape(), &[2, 1]);
}

#[test]
fn test_domain_adapter_reversal_scale() {
    let encoder = SimpleEncoder::new(10, 5);
    let transfer = TransferableEncoder::new(encoder);
    let mut adapter = DomainAdapter::new(transfer, 5, 1.0);

    assert!((adapter.reversal_scale() - 1.0).abs() < 1e-6);

    adapter.set_reversal_scale(0.5);
    assert!((adapter.reversal_scale() - 0.5).abs() < 1e-6);
}

#[test]
fn test_lora_config() {
    let config = LoRAConfig::new(8, 16.0);

    assert_eq!(config.rank, 8);
    assert!((config.alpha - 16.0).abs() < 1e-6);
    assert!((config.scaling() - 2.0).abs() < 1e-6); // 16/8 = 2
}

#[test]
fn test_lora_config_with_modules() {
    let config = LoRAConfig::new(4, 4.0)
        .with_target_modules(vec!["attn".to_string(), "mlp".to_string()])
        .with_dropout(0.1);

    assert_eq!(config.target_modules.len(), 2);
    assert!((config.dropout - 0.1).abs() < 1e-6);
}

#[test]
fn test_lora_adapter_creation() {
    let config = LoRAConfig::new(4, 4.0);
    let adapter = LoRAAdapter::new(10, 20, config);

    assert_eq!(adapter.lora_a.shape(), &[4, 10]); // rank x input_dim
    assert_eq!(adapter.lora_b.shape(), &[20, 4]); // output_dim x rank
}

#[test]
fn test_lora_adapter_delta_weight() {
    let config = LoRAConfig::new(4, 4.0);
    let adapter = LoRAAdapter::new(10, 20, config);

    let delta = adapter.delta_weight();

    // Delta should be output_dim x input_dim
    assert_eq!(delta.shape(), &[20, 10]);
}

#[test]
fn test_lora_adapter_initial_zero_delta() {
    // B is initialized to zeros, so BA should be ~zero initially
    let config = LoRAConfig::new(4, 4.0);
    let adapter = LoRAAdapter::new(10, 20, config);

    let delta = adapter.delta_weight();

    // All values should be zero (B starts as zeros)
    for &v in delta.data() {
        assert!(v.abs() < 1e-6, "Delta should be zero initially");
    }
}

#[test]
fn test_transfer_encoder_train_eval() {
    let encoder = SimpleEncoder::new(10, 5);
    let mut transfer = TransferableEncoder::new(encoder);

    assert!(transfer.training());

    transfer.eval();
    assert!(!transfer.training());

    transfer.train();
    assert!(transfer.training());
}

// Knowledge Distillation Tests
#[test]
fn test_knowledge_distillation_creation() {
    let kd = KnowledgeDistillation::new(4.0, 0.7);
    assert!((kd.temperature() - 4.0).abs() < 1e-6);
    assert!((kd.alpha() - 0.7).abs() < 1e-6);
}

#[test]
fn test_distillation_loss_same_logits() {
    let kd = KnowledgeDistillation::new(2.0, 0.5);
    let logits = vec![2.0, 1.0, 0.0];
    let loss = kd.distillation_loss(&logits, &logits);
    assert!(loss.abs() < 0.01, "Same logits should have ~zero loss");
}

#[test]
fn test_distillation_loss_different_logits() {
    let kd = KnowledgeDistillation::new(2.0, 0.5);
    let student = vec![2.0, 0.0, 0.0];
    let teacher = vec![0.0, 2.0, 0.0];
    let loss = kd.distillation_loss(&student, &teacher);
    assert!(loss > 0.0, "Different logits should have positive loss");
}

#[test]
fn test_combined_loss() {
    let kd = KnowledgeDistillation::new(2.0, 0.6);
    let student = vec![1.0, 1.0, 1.0];
    let teacher = vec![1.0, 1.0, 1.0];
    let task_loss = 0.5;

    let combined = kd.combined_loss(&student, &teacher, task_loss);
    // With same logits, distill loss ~0, so combined ~= (1-0.6) * 0.5 = 0.2
    assert!((combined - 0.2).abs() < 0.1);
}

// Feature Distillation Tests
#[test]
fn test_feature_distillation_mse() {
    let fd = FeatureDistillation::new(FeatureLossType::MSE);
    let student = vec![1.0, 2.0, 3.0];
    let teacher = vec![1.0, 2.0, 3.0];
    let loss = fd.compute_loss(&student, &teacher);
    assert!(loss.abs() < 1e-6);
}

#[test]
fn test_feature_distillation_mae() {
    let fd = FeatureDistillation::new(FeatureLossType::MAE);
    let student = vec![1.0, 2.0, 3.0];
    let teacher = vec![2.0, 3.0, 4.0];
    let loss = fd.compute_loss(&student, &teacher);
    assert!((loss - 1.0).abs() < 1e-6); // Average diff is 1
}

#[test]
fn test_feature_distillation_cosine() {
    let fd = FeatureDistillation::new(FeatureLossType::Cosine);
    let student = vec![1.0, 0.0, 0.0];
    let teacher = vec![1.0, 0.0, 0.0];
    let loss = fd.compute_loss(&student, &teacher);
    assert!(loss.abs() < 1e-6); // Same direction = 0 loss
}

// Attention Transfer Tests
#[test]
fn test_attention_transfer_creation() {
    let at = AttentionTransfer::new(2);
    let activations = vec![1.0, 2.0, 3.0, 4.0]; // 2 channels, 2 spatial
    let attention = at.compute_attention_map(&activations, 2, 2);
    assert_eq!(attention.len(), 2);
}

#[test]
fn test_attention_transfer_loss() {
    let at = AttentionTransfer::new(2);
    let student = vec![1.0, 2.0, 3.0, 4.0];
    let teacher = vec![1.0, 2.0, 3.0, 4.0];
    let loss = at.compute_loss(&student, &teacher, 2, 2);
    assert!(loss < 1e-6);
}

// Self-Distillation Tests
#[test]
fn test_self_distillation_creation() {
    let sd = SelfDistillation::new(3.0)
        .add_layer_pair(3, 1)
        .add_layer_pair(4, 2);
    assert_eq!(sd.layer_pairs().len(), 2);
}

#[test]
fn test_self_distillation_layer_loss() {
    let sd = SelfDistillation::new(2.0);
    let student = vec![1.0, 0.0, 0.0];
    let teacher = vec![1.0, 0.0, 0.0];
    let loss = sd.layer_loss(&student, &teacher);
    assert!(loss.abs() < 0.01);
}

// Meta-Learning Tests
#[test]
fn test_prototypical_network_creation() {
    let pn = PrototypicalNetwork::new(DistanceMetric::Euclidean);
    let pn2 = PrototypicalNetwork::default();
    assert_eq!(pn.distance, pn2.distance);
}

#[test]
fn test_prototypical_compute_prototypes() {
    let pn = PrototypicalNetwork::new(DistanceMetric::Euclidean);
    let support = vec![
        (vec![1.0, 0.0], 0),
        (vec![1.0, 0.0], 0),
        (vec![0.0, 1.0], 1),
    ];
    let protos = pn.compute_prototypes(&support);
    assert_eq!(protos.len(), 2);
}

#[test]
fn test_prototypical_classify() {
    let pn = PrototypicalNetwork::new(DistanceMetric::Euclidean);
    let protos = vec![(0, vec![1.0, 0.0]), (1, vec![0.0, 1.0])];
    let query = vec![0.9, 0.1];
    let class = pn.classify(&query, &protos);
    assert_eq!(class, 0);
}

#[test]
fn test_prototypical_predict_proba() {
    let pn = PrototypicalNetwork::new(DistanceMetric::Euclidean);
    let protos = vec![(0, vec![1.0, 0.0]), (1, vec![0.0, 1.0])];
    let query = vec![1.0, 0.0];
    let probs = pn.predict_proba(&query, &protos);
    assert_eq!(probs.len(), 2);
    let sum: f32 = probs.iter().map(|(_, p)| *p).sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_matching_network_predict() {
    let mn = MatchingNetwork::new(1.0);
    let support = vec![
        (vec![1.0, 0.0], 0),
        (vec![0.9, 0.1], 0),
        (vec![0.0, 1.0], 1),
    ];
    let query = vec![0.95, 0.05];
    let class = mn.predict(&query, &support);
    assert_eq!(class, 0);
}

// Online Distillation Tests
#[test]
fn test_online_distillation_creation() {
    let od = OnlineDistillation::new(3, 2.0, 0.5);
    assert_eq!(od.num_networks(), 3);
    assert!((od.temperature() - 2.0).abs() < 1e-6);
    assert!((od.mutual_weight() - 0.5).abs() < 1e-6);
}

#[test]
fn test_online_distillation_same_logits() {
    let od = OnlineDistillation::new(2, 2.0, 1.0);
    let all_logits = vec![vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]];
    let loss = od.mutual_loss(0, &all_logits);
    assert!(loss.abs() < 0.01, "Same logits should have ~zero loss");
}

#[test]
fn test_online_distillation_different_logits() {
    let od = OnlineDistillation::new(2, 2.0, 1.0);
    let all_logits = vec![vec![2.0, 0.0, 0.0], vec![0.0, 2.0, 0.0]];
    let loss = od.mutual_loss(0, &all_logits);
    assert!(loss > 0.0, "Different logits should have positive loss");
}

#[test]
fn test_online_distillation_combined_loss() {
    let od = OnlineDistillation::new(2, 2.0, 0.5);
    let all_logits = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
    let task_loss = 0.4;
    let combined = od.combined_loss(0, &all_logits, task_loss);
    // Same logits = ~zero mutual loss, so combined ~= task_loss
    assert!((combined - 0.4).abs() < 0.1);
}

#[path = "tests_part_02.rs"]

mod tests_part_02;
