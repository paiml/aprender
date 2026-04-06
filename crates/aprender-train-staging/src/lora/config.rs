//! LoRA configuration for target module selection
//!
//! Allows selective application of LoRA adapters to specific modules/layers,
//! commonly used for transformer attention projections (q/k/v/o_proj).

use std::collections::HashSet;

/// Configuration for LoRA adapter targeting
#[derive(Clone, Debug)]
pub struct LoRAConfig {
    /// LoRA rank
    pub rank: usize,
    /// LoRA alpha (scaling parameter)
    pub alpha: f32,
    /// Target module names (e.g., "q_proj", "k_proj", "v_proj", "o_proj")
    pub target_modules: HashSet<String>,
    /// Layer indices to apply LoRA to (None = all layers)
    pub layers: Option<Vec<usize>>,
    /// Whether to apply LoRA to all linear layers
    pub all_linear: bool,
}

impl LoRAConfig {
    /// Create a new LoRA configuration
    ///
    /// # Arguments
    /// * `rank` - LoRA rank (typically 4, 8, 16, 32, or 64)
    /// * `alpha` - LoRA alpha scaling parameter (often same as rank)
    pub fn new(rank: usize, alpha: f32) -> Self {
        Self { rank, alpha, target_modules: HashSet::new(), layers: None, all_linear: false }
    }

    /// Target specific modules by name
    ///
    /// # Example
    /// ```ignore
    /// config.target_modules(&["q_proj", "v_proj"]);
    /// ```
    pub fn target_modules(mut self, modules: &[&str]) -> Self {
        self.target_modules = modules.iter().map(ToString::to_string).collect();
        self
    }

    /// Target attention projection modules (q, k, v, o)
    pub fn target_attention_projections(mut self) -> Self {
        self.target_modules = vec![
            "q_proj".to_string(),
            "k_proj".to_string(),
            "v_proj".to_string(),
            "o_proj".to_string(),
        ]
        .into_iter()
        .collect();
        self
    }

    /// Target query and value projections only (common for efficient fine-tuning)
    pub fn target_qv_projections(mut self) -> Self {
        self.target_modules =
            vec!["q_proj".to_string(), "v_proj".to_string()].into_iter().collect();
        self
    }

    /// Target all attention projections except output (q, k, v only)
    pub fn target_qkv_projections(mut self) -> Self {
        self.target_modules =
            vec!["q_proj".to_string(), "k_proj".to_string(), "v_proj".to_string()]
                .into_iter()
                .collect();
        self
    }

    /// Target specific layer indices
    ///
    /// # Example
    /// ```ignore
    /// config.target_layers(&[0, 1, 2]); // Only first 3 layers
    /// ```
    pub fn target_layers(mut self, layer_indices: &[usize]) -> Self {
        self.layers = Some(layer_indices.to_vec());
        self
    }

    /// Apply LoRA to all linear layers
    pub fn all_linear_layers(mut self) -> Self {
        self.all_linear = true;
        self
    }

    /// Expand shorthand target module names to concrete module lists (ENT-LoRA-005)
    ///
    /// Supports:
    /// - `"all_linear"` → all projection modules (q/k/v/o/gate/up/down)
    /// - `"attention"` → q/k/v/o projections
    /// - `"qv"` → q/v projections (default, original paper)
    /// - `"mlp"` → gate/up/down projections
    /// - Explicit module names passed through unchanged
    pub fn expand_shorthand(modules: &[String]) -> Vec<String> {
        if modules.len() == 1 {
            match modules[0].as_str() {
                "all_linear" => {
                    return vec![
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ]
                    .into_iter()
                    .map(String::from)
                    .collect()
                }
                "attention" => {
                    return vec!["q_proj", "k_proj", "v_proj", "o_proj"]
                        .into_iter()
                        .map(String::from)
                        .collect()
                }
                "qv" => return vec!["q_proj", "v_proj"].into_iter().map(String::from).collect(),
                "mlp" => {
                    return vec!["gate_proj", "up_proj", "down_proj"]
                        .into_iter()
                        .map(String::from)
                        .collect()
                }
                _ => {}
            }
        }
        modules.to_vec()
    }

    /// Check if a module should have LoRA applied
    ///
    /// # Arguments
    /// * `module_name` - Name of the module (e.g., "q_proj", "k_proj")
    /// * `layer_idx` - Layer index (if applicable)
    pub fn should_apply(&self, module_name: &str, layer_idx: Option<usize>) -> bool {
        // Check layer index filter
        if let Some(layers) = &self.layers {
            if let Some(idx) = layer_idx {
                if !layers.contains(&idx) {
                    return false;
                }
            }
        }

        // Check module name filter
        if self.all_linear {
            // Apply to all linear layers (assuming module names ending in "proj" or "linear")
            module_name.ends_with("proj") || module_name.ends_with("linear")
        } else {
            self.target_modules.contains(module_name)
        }
    }

    /// Get the number of target modules
    pub fn num_target_modules(&self) -> usize {
        self.target_modules.len()
    }

    /// Check if targeting all linear layers
    pub fn is_all_linear(&self) -> bool {
        self.all_linear
    }

    /// Get target module names
    pub fn get_target_modules(&self) -> Vec<&str> {
        self.target_modules.iter().map(String::as_str).collect()
    }
}

impl Default for LoRAConfig {
    /// Default configuration: rank=8, alpha=8, target q_proj and v_proj
    fn default() -> Self {
        Self::new(8, 8.0).target_qv_projections()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // ========================================================================
    // PROPERTY TESTS - Configuration correctness validation
    // ========================================================================

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(200))]

        /// should_apply should be consistent with target_modules set
        #[test]
        fn prop_should_apply_consistent_with_modules(
            rank in 1usize..64,
            alpha in 1.0f32..64.0,
            include_q in proptest::bool::ANY,
            include_k in proptest::bool::ANY,
            include_v in proptest::bool::ANY,
            include_o in proptest::bool::ANY,
        ) {
            let mut modules = vec![];
            if include_q { modules.push("q_proj"); }
            if include_k { modules.push("k_proj"); }
            if include_v { modules.push("v_proj"); }
            if include_o { modules.push("o_proj"); }

            let config = LoRAConfig::new(rank, alpha).target_modules(&modules);

            // should_apply must match what was set
            prop_assert_eq!(config.should_apply("q_proj", None), include_q);
            prop_assert_eq!(config.should_apply("k_proj", None), include_k);
            prop_assert_eq!(config.should_apply("v_proj", None), include_v);
            prop_assert_eq!(config.should_apply("o_proj", None), include_o);
            prop_assert_eq!(config.num_target_modules(), modules.len());
        }

        /// Layer filtering should respect layer indices
        #[test]
        fn prop_layer_filtering_respects_indices(
            layers in prop::collection::vec(0usize..32, 1..8),
            test_layer in 0usize..32,
        ) {
            let config = LoRAConfig::new(8, 8.0)
                .target_modules(&["q_proj"])
                .target_layers(&layers);

            // should_apply for a layer should match layer list membership
            let in_list = layers.contains(&test_layer);
            prop_assert_eq!(config.should_apply("q_proj", Some(test_layer)), in_list);
        }

        /// all_linear mode should match any *proj or *linear suffix
        #[test]
        fn prop_all_linear_matches_suffixes(
            prefix in "[a-z]{1,8}",
        ) {
            let config = LoRAConfig::new(8, 8.0).all_linear_layers();

            // Should match *_proj and *_linear
            let proj_name = format!("{prefix}_proj");
            let linear_name = format!("{prefix}_linear");
            let other_name = format!("{prefix}_norm");

            prop_assert!(config.should_apply(&proj_name, None));
            prop_assert!(config.should_apply(&linear_name, None));
            prop_assert!(!config.should_apply(&other_name, None));
        }

        /// Config parameters should be preserved after builder chain
        #[test]
        fn prop_config_params_preserved(
            rank in 1usize..128,
            alpha in 0.1f32..128.0,
        ) {
            let config = LoRAConfig::new(rank, alpha)
                .target_attention_projections()
                .target_layers(&[0, 1, 2]);

            prop_assert_eq!(config.rank, rank);
            prop_assert!((config.alpha - alpha).abs() < 1e-6);
            prop_assert_eq!(config.num_target_modules(), 4);
        }

        /// None layer index should bypass layer filtering
        #[test]
        fn prop_none_layer_bypasses_filter(
            layers in prop::collection::vec(0usize..16, 1..4),
        ) {
            let config = LoRAConfig::new(8, 8.0)
                .target_modules(&["q_proj"])
                .target_layers(&layers);

            // None layer index should always pass layer check
            prop_assert!(config.should_apply("q_proj", None));
        }
    }

    // ========================================================================
    // UNIT TESTS
    // ========================================================================

    #[test]
    fn test_lora_config_creation() {
        let config = LoRAConfig::new(16, 16.0);
        assert_eq!(config.rank, 16);
        assert_eq!(config.alpha, 16.0);
        assert_eq!(config.num_target_modules(), 0);
        assert!(!config.is_all_linear());
    }

    #[test]
    fn test_target_modules() {
        let config = LoRAConfig::new(8, 8.0).target_modules(&["q_proj", "k_proj"]);

        assert!(config.should_apply("q_proj", None));
        assert!(config.should_apply("k_proj", None));
        assert!(!config.should_apply("v_proj", None));
        assert!(!config.should_apply("o_proj", None));
        assert_eq!(config.num_target_modules(), 2);
    }

    #[test]
    fn test_target_attention_projections() {
        let config = LoRAConfig::new(8, 8.0).target_attention_projections();

        assert!(config.should_apply("q_proj", None));
        assert!(config.should_apply("k_proj", None));
        assert!(config.should_apply("v_proj", None));
        assert!(config.should_apply("o_proj", None));
        assert!(!config.should_apply("mlp_proj", None));
        assert_eq!(config.num_target_modules(), 4);
    }

    #[test]
    fn test_target_qv_projections() {
        let config = LoRAConfig::new(8, 8.0).target_qv_projections();

        assert!(config.should_apply("q_proj", None));
        assert!(config.should_apply("v_proj", None));
        assert!(!config.should_apply("k_proj", None));
        assert!(!config.should_apply("o_proj", None));
        assert_eq!(config.num_target_modules(), 2);
    }

    #[test]
    fn test_target_qkv_projections() {
        let config = LoRAConfig::new(8, 8.0).target_qkv_projections();

        assert!(config.should_apply("q_proj", None));
        assert!(config.should_apply("k_proj", None));
        assert!(config.should_apply("v_proj", None));
        assert!(!config.should_apply("o_proj", None));
        assert_eq!(config.num_target_modules(), 3);
    }

    #[test]
    fn test_target_layers() {
        let config = LoRAConfig::new(8, 8.0).target_modules(&["q_proj"]).target_layers(&[0, 2, 4]);

        // Layer 0 - should apply
        assert!(config.should_apply("q_proj", Some(0)));
        // Layer 1 - should not apply (not in target layers)
        assert!(!config.should_apply("q_proj", Some(1)));
        // Layer 2 - should apply
        assert!(config.should_apply("q_proj", Some(2)));
        // Layer 3 - should not apply
        assert!(!config.should_apply("q_proj", Some(3)));
        // Layer 4 - should apply
        assert!(config.should_apply("q_proj", Some(4)));
    }

    #[test]
    fn test_all_linear_layers() {
        let config = LoRAConfig::new(8, 8.0).all_linear_layers();

        assert!(config.is_all_linear());
        assert!(config.should_apply("q_proj", None));
        assert!(config.should_apply("k_proj", None));
        assert!(config.should_apply("mlp_proj", None));
        assert!(config.should_apply("fc_linear", None));
        assert!(!config.should_apply("layer_norm", None));
    }

    #[test]
    fn test_default_config() {
        let config = LoRAConfig::default();

        assert_eq!(config.rank, 8);
        assert_eq!(config.alpha, 8.0);
        assert!(config.should_apply("q_proj", None));
        assert!(config.should_apply("v_proj", None));
        assert!(!config.should_apply("k_proj", None));
        assert_eq!(config.num_target_modules(), 2);
    }

    #[test]
    fn test_layer_filtering_with_modules() {
        let config = LoRAConfig::new(4, 4.0).target_attention_projections().target_layers(&[1, 3]);

        // Layer 0 - wrong layer
        assert!(!config.should_apply("q_proj", Some(0)));
        // Layer 1 - correct layer and module
        assert!(config.should_apply("q_proj", Some(1)));
        assert!(config.should_apply("v_proj", Some(1)));
        // Layer 2 - wrong layer
        assert!(!config.should_apply("q_proj", Some(2)));
        // Layer 3 - correct layer and module
        assert!(config.should_apply("k_proj", Some(3)));
        assert!(config.should_apply("o_proj", Some(3)));
    }

    // ========================================================================
    // ENT-LoRA-005: Shorthand expansion tests
    // ========================================================================

    #[test]
    fn test_ent_lora_005_expand_all_linear() {
        let expanded = LoRAConfig::expand_shorthand(&["all_linear".to_string()]);
        assert_eq!(expanded.len(), 7);
        assert!(expanded.contains(&"q_proj".to_string()));
        assert!(expanded.contains(&"k_proj".to_string()));
        assert!(expanded.contains(&"v_proj".to_string()));
        assert!(expanded.contains(&"o_proj".to_string()));
        assert!(expanded.contains(&"gate_proj".to_string()));
        assert!(expanded.contains(&"up_proj".to_string()));
        assert!(expanded.contains(&"down_proj".to_string()));
    }

    #[test]
    fn test_ent_lora_005_expand_attention() {
        let expanded = LoRAConfig::expand_shorthand(&["attention".to_string()]);
        assert_eq!(expanded.len(), 4);
        assert!(expanded.contains(&"q_proj".to_string()));
        assert!(expanded.contains(&"k_proj".to_string()));
        assert!(expanded.contains(&"v_proj".to_string()));
        assert!(expanded.contains(&"o_proj".to_string()));
    }

    #[test]
    fn test_ent_lora_005_expand_qv() {
        let expanded = LoRAConfig::expand_shorthand(&["qv".to_string()]);
        assert_eq!(expanded.len(), 2);
        assert!(expanded.contains(&"q_proj".to_string()));
        assert!(expanded.contains(&"v_proj".to_string()));
    }

    #[test]
    fn test_ent_lora_005_expand_mlp() {
        let expanded = LoRAConfig::expand_shorthand(&["mlp".to_string()]);
        assert_eq!(expanded.len(), 3);
        assert!(expanded.contains(&"gate_proj".to_string()));
        assert!(expanded.contains(&"up_proj".to_string()));
        assert!(expanded.contains(&"down_proj".to_string()));
    }

    #[test]
    fn test_ent_lora_005_expand_explicit_passthrough() {
        let explicit = vec!["q_proj".to_string(), "v_proj".to_string()];
        let expanded = LoRAConfig::expand_shorthand(&explicit);
        assert_eq!(expanded, explicit);
    }

    #[test]
    fn test_ent_lora_005_expand_unknown_single() {
        let modules = vec!["custom_proj".to_string()];
        let expanded = LoRAConfig::expand_shorthand(&modules);
        assert_eq!(expanded, modules);
    }

    #[test]
    fn test_get_target_modules() {
        let config = LoRAConfig::new(8, 8.0).target_modules(&["q_proj", "v_proj"]);

        let mut modules = config.get_target_modules();
        modules.sort_unstable(); // HashSet order is not guaranteed

        assert_eq!(modules.len(), 2);
        assert!(modules.contains(&"q_proj"));
        assert!(modules.contains(&"v_proj"));
    }
}
