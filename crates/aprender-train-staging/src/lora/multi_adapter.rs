//! Multi-adapter training (ENT-LoRA-013)
//!
//! Supports N adapters sharing one frozen base model, each with independent
//! optimizer states. This enables:
//! - Multi-task fine-tuning (one adapter per task)
//! - A/B testing different LoRA configurations
//! - Adapter composition (combine adapters at inference)
//!
//! Architecture:
//! - One shared base model (frozen, loaded once)
//! - N LoRA adapter sets, each with own A/B matrices
//! - Independent forward passes per adapter
//! - Independent optimizer states per adapter

use crate::lora::LoRALayer;
use crate::Tensor;

/// Named adapter wrapping a set of LoRA layers
#[derive(Clone)]
pub struct NamedAdapter {
    /// Human-readable name (e.g., "task_a", "safety_classifier")
    pub name: String,
    /// LoRA layers for this adapter
    pub layers: Vec<LoRALayer>,
    /// Whether this adapter is active (receives gradients)
    pub active: bool,
}

impl NamedAdapter {
    /// Create a new named adapter
    pub fn new(name: impl Into<String>, layers: Vec<LoRALayer>) -> Self {
        Self { name: name.into(), layers, active: true }
    }

    /// Get trainable parameters across all layers
    pub fn trainable_params(&mut self) -> Vec<&mut Tensor> {
        self.layers.iter_mut().flat_map(|l| l.trainable_params()).collect()
    }

    /// Total trainable parameter count
    pub fn param_count(&self) -> usize {
        self.layers.iter().map(|l| l.lora_a().len() + l.lora_b().len()).sum()
    }

    /// Merge all layers for inference
    pub fn merge_all(&mut self) {
        for layer in &mut self.layers {
            layer.merge();
        }
    }

    /// Unmerge all layers (return to training mode)
    pub fn unmerge_all(&mut self) {
        for layer in &mut self.layers {
            layer.unmerge();
        }
    }
}

/// Multi-adapter manager
///
/// Manages multiple LoRA adapters sharing a frozen base model.
/// Each adapter can be independently trained, activated/deactivated,
/// and merged.
pub struct MultiAdapterManager {
    /// Named adapters indexed by position
    adapters: Vec<NamedAdapter>,
}

impl MultiAdapterManager {
    /// Create an empty multi-adapter manager
    pub fn new() -> Self {
        Self { adapters: Vec::new() }
    }

    /// Add an adapter, returns its index
    pub fn add_adapter(&mut self, adapter: NamedAdapter) -> usize {
        let idx = self.adapters.len();
        self.adapters.push(adapter);
        idx
    }

    /// Get adapter by index
    pub fn get(&self, idx: usize) -> Option<&NamedAdapter> {
        self.adapters.get(idx)
    }

    /// Get mutable adapter by index
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut NamedAdapter> {
        self.adapters.get_mut(idx)
    }

    /// Find adapter by name
    pub fn find_by_name(&self, name: &str) -> Option<(usize, &NamedAdapter)> {
        self.adapters.iter().enumerate().find(|(_, a)| a.name == name)
    }

    /// Number of adapters
    pub fn len(&self) -> usize {
        self.adapters.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }

    /// Get all active adapters
    pub fn active_adapters(&self) -> Vec<(usize, &NamedAdapter)> {
        self.adapters.iter().enumerate().filter(|(_, a)| a.active).collect()
    }

    /// Set adapter active/inactive
    pub fn set_active(&mut self, idx: usize, active: bool) {
        if let Some(adapter) = self.adapters.get_mut(idx) {
            adapter.active = active;
        }
    }

    /// Total trainable parameters across all active adapters
    pub fn total_trainable_params(&self) -> usize {
        self.adapters.iter().filter(|a| a.active).map(NamedAdapter::param_count).sum()
    }

    /// Summary of all adapters
    pub fn summary(&self) -> String {
        let mut lines = vec![format!("Multi-adapter manager: {} adapters", self.adapters.len())];
        for (i, adapter) in self.adapters.iter().enumerate() {
            let status = if adapter.active { "ACTIVE" } else { "INACTIVE" };
            lines.push(format!(
                "  [{}] {} — {} params, {} layers, {}",
                i,
                adapter.name,
                adapter.param_count(),
                adapter.layers.len(),
                status,
            ));
        }
        lines.join("\n")
    }

    /// Remove adapter by index (returns the removed adapter)
    pub fn remove(&mut self, idx: usize) -> Option<NamedAdapter> {
        if idx < self.adapters.len() {
            Some(self.adapters.remove(idx))
        } else {
            None
        }
    }

    /// Iterator over all adapters
    pub fn iter(&self) -> impl Iterator<Item = &NamedAdapter> {
        self.adapters.iter()
    }

    /// Mutable iterator
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut NamedAdapter> {
        self.adapters.iter_mut()
    }
}

impl Default for MultiAdapterManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::lora::LoRALayer;
    use proptest::prelude::*;

    fn make_lora_layer(d_out: usize, d_in: usize, rank: usize) -> LoRALayer {
        let base = Tensor::from_vec(vec![0.5; d_out * d_in], false);
        LoRALayer::new(base, d_out, d_in, rank, 4.0)
    }

    #[test]
    fn test_ent_lora_013_multi_adapter_creation() {
        let mut mgr = MultiAdapterManager::new();
        assert!(mgr.is_empty());
        assert_eq!(mgr.len(), 0);

        let adapter = NamedAdapter::new("task_a", vec![make_lora_layer(4, 4, 2)]);
        let idx = mgr.add_adapter(adapter);
        assert_eq!(idx, 0);
        assert_eq!(mgr.len(), 1);
        assert!(!mgr.is_empty());
    }

    #[test]
    fn test_ent_lora_013_multiple_adapters() {
        let mut mgr = MultiAdapterManager::new();
        mgr.add_adapter(NamedAdapter::new("safety", vec![make_lora_layer(4, 4, 2)]));
        mgr.add_adapter(NamedAdapter::new("style", vec![make_lora_layer(4, 4, 4)]));

        assert_eq!(mgr.len(), 2);
        assert_eq!(mgr.get(0).unwrap().name, "safety");
        assert_eq!(mgr.get(1).unwrap().name, "style");
    }

    #[test]
    fn test_ent_lora_013_find_by_name() {
        let mut mgr = MultiAdapterManager::new();
        mgr.add_adapter(NamedAdapter::new("alpha", vec![make_lora_layer(4, 4, 2)]));
        mgr.add_adapter(NamedAdapter::new("beta", vec![make_lora_layer(4, 4, 2)]));

        let (idx, adapter) = mgr.find_by_name("beta").unwrap();
        assert_eq!(idx, 1);
        assert_eq!(adapter.name, "beta");
        assert!(mgr.find_by_name("gamma").is_none());
    }

    #[test]
    fn test_ent_lora_013_active_inactive() {
        let mut mgr = MultiAdapterManager::new();
        mgr.add_adapter(NamedAdapter::new("a", vec![make_lora_layer(4, 4, 2)]));
        mgr.add_adapter(NamedAdapter::new("b", vec![make_lora_layer(4, 4, 2)]));

        assert_eq!(mgr.active_adapters().len(), 2);

        mgr.set_active(0, false);
        assert_eq!(mgr.active_adapters().len(), 1);
        assert_eq!(mgr.active_adapters()[0].1.name, "b");
    }

    #[test]
    fn test_ent_lora_013_param_count() {
        let adapter = NamedAdapter::new(
            "test",
            vec![
                make_lora_layer(8, 4, 2), // A: 2*4=8, B: 8*2=16 = 24
                make_lora_layer(4, 8, 2), // A: 2*8=16, B: 4*2=8 = 24
            ],
        );
        assert_eq!(adapter.param_count(), 48);
    }

    #[test]
    fn test_ent_lora_013_total_trainable_params() {
        let mut mgr = MultiAdapterManager::new();
        mgr.add_adapter(NamedAdapter::new("a", vec![make_lora_layer(4, 4, 2)]));
        mgr.add_adapter(NamedAdapter::new("b", vec![make_lora_layer(4, 4, 2)]));

        let total = mgr.total_trainable_params();
        assert!(total > 0);

        mgr.set_active(0, false);
        let reduced = mgr.total_trainable_params();
        assert!(reduced < total);
    }

    #[test]
    fn test_ent_lora_013_summary() {
        let mut mgr = MultiAdapterManager::new();
        mgr.add_adapter(NamedAdapter::new("task_a", vec![make_lora_layer(4, 4, 2)]));
        let summary = mgr.summary();
        assert!(summary.contains("task_a"));
        assert!(summary.contains("ACTIVE"));
    }

    #[test]
    fn test_ent_lora_013_remove_adapter() {
        let mut mgr = MultiAdapterManager::new();
        mgr.add_adapter(NamedAdapter::new("a", vec![]));
        mgr.add_adapter(NamedAdapter::new("b", vec![]));

        let removed = mgr.remove(0).unwrap();
        assert_eq!(removed.name, "a");
        assert_eq!(mgr.len(), 1);
        assert_eq!(mgr.get(0).unwrap().name, "b");
    }

    #[test]
    fn test_ent_lora_013_trainable_params_mut() {
        let mut adapter = NamedAdapter::new("test", vec![make_lora_layer(4, 4, 2)]);
        let params = adapter.trainable_params();
        // Each LoRA layer has 2 params: A and B
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_ent_lora_013_merge_unmerge() {
        let mut adapter = NamedAdapter::new("test", vec![make_lora_layer(4, 4, 2)]);
        assert!(!adapter.layers[0].is_merged());

        adapter.merge_all();
        assert!(adapter.layers[0].is_merged());

        adapter.unmerge_all();
        assert!(!adapter.layers[0].is_merged());
    }

    #[test]
    fn test_ent_lora_013_default() {
        let mgr = MultiAdapterManager::default();
        assert!(mgr.is_empty());
    }

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(30))]

        #[test]
        fn prop_multi_adapter_param_count_additive(
            n_adapters in 1usize..5,
            d in 4usize..8,
            rank in 1usize..3,
        ) {
            let mut mgr = MultiAdapterManager::new();
            let mut expected = 0usize;
            for i in 0..n_adapters {
                let adapter = NamedAdapter::new(format!("a{i}"), vec![make_lora_layer(d, d, rank)]);
                expected += adapter.param_count();
                mgr.add_adapter(adapter);
            }
            prop_assert_eq!(mgr.total_trainable_params(), expected);
        }
    }
}
