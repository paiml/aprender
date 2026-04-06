//! HuggingFace Ecosystem Tree Visualization
//!
//! Provides hierarchical views of:
//! - HuggingFace ecosystem components
//! - PAIML-HuggingFace integration mapping

use serde::{Deserialize, Serialize};
use std::fmt;

// ============================================================================
// HF-TREE-001: Core Types
// ============================================================================

/// HuggingFace ecosystem category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfCategory {
    pub name: String,
    pub components: Vec<HfComponent>,
}

impl HfCategory {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), components: Vec::new() }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, component: HfComponent) -> Self {
        self.components.push(component);
        self
    }
}

/// A component in the HuggingFace ecosystem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfComponent {
    pub name: String,
    pub description: String,
    pub url: Option<String>,
}

impl HfComponent {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self { name: name.into(), description: description.into(), url: None }
    }

    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }
}

/// The complete HuggingFace ecosystem tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfTree {
    pub name: String,
    pub categories: Vec<HfCategory>,
}

impl HfTree {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), categories: Vec::new() }
    }

    pub fn add_category(mut self, category: HfCategory) -> Self {
        self.categories.push(category);
        self
    }

    pub fn total_components(&self) -> usize {
        self.categories.iter().map(|c| c.components.len()).sum()
    }
}

// ============================================================================
// HF-TREE-002: Integration Types
// ============================================================================

/// Integration type between PAIML and HuggingFace
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntegrationType {
    /// PAIML component is compatible with HF format/API
    Compatible,
    /// PAIML provides a native Rust alternative
    Alternative,
    /// PAIML orchestrates/wraps HF functionality
    Orchestrates,
    /// Uses HF library directly
    Uses,
    /// Not interoperable
    Incompatible,
}

impl fmt::Display for IntegrationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compatible => write!(f, "✓ COMPATIBLE"),
            Self::Alternative => write!(f, "⚡ ALTERNATIVE"),
            Self::Orchestrates => write!(f, "🔄 ORCHESTRATES"),
            Self::Uses => write!(f, "📦 USES"),
            Self::Incompatible => write!(f, "✗ INCOMPATIBLE"),
        }
    }
}

/// A mapping between PAIML and HuggingFace components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationMapping {
    pub paiml_component: String,
    pub hf_equivalent: String,
    pub integration_type: IntegrationType,
    pub notes: Option<String>,
}

impl IntegrationMapping {
    pub fn new(
        paiml: impl Into<String>,
        hf: impl Into<String>,
        integration: IntegrationType,
    ) -> Self {
        Self {
            paiml_component: paiml.into(),
            hf_equivalent: hf.into(),
            integration_type: integration,
            notes: None,
        }
    }

    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }
}

/// Integration category for grouping mappings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationCategory {
    pub name: String,
    pub mappings: Vec<IntegrationMapping>,
}

impl IntegrationCategory {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), mappings: Vec::new() }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, mapping: IntegrationMapping) -> Self {
        self.mappings.push(mapping);
        self
    }
}

/// The complete PAIML-HuggingFace integration tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationTree {
    pub categories: Vec<IntegrationCategory>,
}

impl IntegrationTree {
    pub fn new() -> Self {
        Self { categories: Vec::new() }
    }

    pub fn add_category(mut self, category: IntegrationCategory) -> Self {
        self.categories.push(category);
        self
    }

    pub fn total_mappings(&self) -> usize {
        self.categories.iter().map(|c| c.mappings.len()).sum()
    }

    pub fn compatible_count(&self) -> usize {
        self.categories
            .iter()
            .flat_map(|c| &c.mappings)
            .filter(|m| m.integration_type == IntegrationType::Compatible)
            .count()
    }

    pub fn alternative_count(&self) -> usize {
        self.categories
            .iter()
            .flat_map(|c| &c.mappings)
            .filter(|m| m.integration_type == IntegrationType::Alternative)
            .count()
    }
}

impl Default for IntegrationTree {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// HF-TREE-003: Formatters
// ============================================================================

/// Format HF tree as ASCII
pub fn format_hf_tree_ascii(tree: &HfTree) -> String {
    let mut output = format!("{} ({} categories)\n", tree.name, tree.categories.len());

    for (cat_idx, category) in tree.categories.iter().enumerate() {
        let is_last_cat = cat_idx == tree.categories.len() - 1;
        let cat_prefix = if is_last_cat { "└── " } else { "├── " };
        output.push_str(&format!("{}{}\n", cat_prefix, category.name));

        for (comp_idx, comp) in category.components.iter().enumerate() {
            let is_last_comp = comp_idx == category.components.len() - 1;
            let comp_prefix = if is_last_cat { "    " } else { "│   " };
            let comp_branch = if is_last_comp { "└── " } else { "├── " };
            output.push_str(&format!(
                "{}{}{:<20} ({})\n",
                comp_prefix, comp_branch, comp.name, comp.description
            ));
        }
    }

    output
}

/// Format integration tree as ASCII table
pub fn format_integration_tree_ascii(tree: &IntegrationTree) -> String {
    let mut output = String::from("PAIML ↔ HuggingFace Integration Map\n");
    output.push_str(&"═".repeat(65));
    output.push('\n');
    output.push('\n');

    output.push_str(&format!("┌{:─<20}┬{:─<20}┬{:─<22}┐\n", "", "", ""));
    output.push_str(&format!(
        "│ {:^18} │ {:^18} │ {:^20} │\n",
        "PAIML Component", "HF Equivalent", "Integration Type"
    ));
    output.push_str(&format!("├{:─<20}┼{:─<20}┼{:─<22}┤\n", "", "", ""));

    for category in &tree.categories {
        output.push_str(&format!(
            "│ {:^18} │ {:^18} │ {:^20} │\n",
            category.name.to_uppercase(),
            "",
            ""
        ));
        output.push_str(&format!("├{:─<20}┼{:─<20}┼{:─<22}┤\n", "", "", ""));

        for mapping in &category.mappings {
            output.push_str(&format!(
                "│ {:<18} │ {:<18} │ {:<20} │\n",
                mapping.paiml_component,
                mapping.hf_equivalent,
                format!("{}", mapping.integration_type)
            ));
        }

        output.push_str(&format!("├{:─<20}┼{:─<20}┼{:─<22}┤\n", "", "", ""));
    }

    output.push_str(&format!("└{:─<20}┴{:─<20}┴{:─<22}┘\n", "", "", ""));

    output.push_str("\nLegend:\n");
    output.push_str("  ✓ COMPATIBLE  - Interoperates with HF format/API\n");
    output.push_str("  ⚡ ALTERNATIVE - PAIML native replacement (pure Rust)\n");
    output.push_str("  🔄 ORCHESTRATES - PAIML wraps/orchestrates HF\n");
    output.push_str("  📦 USES        - PAIML uses HF library directly\n");

    output.push_str(&format!(
        "\nSummary: {} compatible, {} alternatives, {} total mappings\n",
        tree.compatible_count(),
        tree.alternative_count(),
        tree.total_mappings()
    ));

    output
}

/// Format HF tree as JSON
pub fn format_hf_tree_json(tree: &HfTree) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(tree)
}

/// Format integration tree as JSON
pub fn format_integration_tree_json(tree: &IntegrationTree) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(tree)
}

// ============================================================================
// HF-TREE-004: Builders
// ============================================================================

/// Build the HuggingFace ecosystem tree
pub fn build_hf_tree() -> HfTree {
    HfTree::new("HuggingFace Ecosystem")
        .add_category(
            HfCategory::new("hub")
                .add(HfComponent::new("models", "700K+ models"))
                .add(HfComponent::new("datasets", "100K+ datasets"))
                .add(HfComponent::new("spaces", "300K+ spaces"))
                .add(HfComponent::new("papers", "Research papers")),
        )
        .add_category(
            HfCategory::new("libraries")
                .add(HfComponent::new("transformers", "Model architectures"))
                .add(HfComponent::new("diffusers", "Diffusion models"))
                .add(HfComponent::new("accelerate", "Distributed training"))
                .add(HfComponent::new("peft", "Parameter-efficient fine-tuning"))
                .add(HfComponent::new("trl", "Reinforcement learning"))
                .add(HfComponent::new("optimum", "Hardware optimization"))
                .add(HfComponent::new("datasets", "Dataset loading"))
                .add(HfComponent::new("tokenizers", "Fast tokenization"))
                .add(HfComponent::new("safetensors", "Safe serialization"))
                .add(HfComponent::new("huggingface_hub", "Hub API client")),
        )
        .add_category(
            HfCategory::new("inference")
                .add(HfComponent::new("inference-api", "Serverless inference"))
                .add(HfComponent::new("inference-endpoints", "Dedicated endpoints"))
                .add(HfComponent::new("text-generation-inference", "TGI server")),
        )
        .add_category(
            HfCategory::new("training")
                .add(HfComponent::new("autotrain", "AutoML training"))
                .add(HfComponent::new("trainer", "Training loops"))
                .add(HfComponent::new("evaluate", "Metrics & evaluation")),
        )
        .add_category(
            HfCategory::new("formats")
                .add(HfComponent::new("safetensors", "Safe tensor format"))
                .add(HfComponent::new("gguf", "Quantized format"))
                .add(HfComponent::new("onnx", "Cross-platform"))
                .add(HfComponent::new("pytorch", "Native PyTorch")),
        )
        .add_category(
            HfCategory::new("tasks")
                .add(HfComponent::new("text-generation", "LLM generation"))
                .add(HfComponent::new("text-classification", "Classification"))
                .add(HfComponent::new("question-answering", "QA"))
                .add(HfComponent::new("translation", "Translation"))
                .add(HfComponent::new("summarization", "Summarization"))
                .add(HfComponent::new("image-classification", "Vision"))
                .add(HfComponent::new("text-to-image", "Diffusion"))
                .add(HfComponent::new("speech-recognition", "ASR")),
        )
}

/// Build the PAIML-HuggingFace integration tree
pub fn build_integration_tree() -> IntegrationTree {
    IntegrationTree::new()
        .add_category(
            IntegrationCategory::new("formats")
                .add(
                    IntegrationMapping::new(".apr", "pickle/.joblib", IntegrationType::Alternative)
                        .with_notes("Safe sklearn model format"),
                )
                .add(
                    IntegrationMapping::new(".apr", "safetensors", IntegrationType::Alternative)
                        .with_notes("Safe tensor serialization"),
                )
                .add(
                    IntegrationMapping::new(".apr", "gguf", IntegrationType::Alternative)
                        .with_notes("Quantized model format"),
                )
                .add(
                    IntegrationMapping::new("realizar/gguf", "gguf", IntegrationType::Compatible)
                        .with_notes("Import GGUF models"),
                )
                .add(
                    IntegrationMapping::new(
                        "realizar/safetensors",
                        "safetensors",
                        IntegrationType::Compatible,
                    )
                    .with_notes("Import SafeTensors"),
                ),
        )
        .add_category(
            IntegrationCategory::new("hub_access")
                .add(
                    IntegrationMapping::new(
                        "aprender/hf_hub",
                        "huggingface_hub",
                        IntegrationType::Uses,
                    )
                    .with_notes("hf-hub crate"),
                )
                .add(
                    IntegrationMapping::new(
                        "batuta/hf",
                        "huggingface_hub",
                        IntegrationType::Orchestrates,
                    )
                    .with_notes("CLI orchestration"),
                ),
        )
        .add_category(
            IntegrationCategory::new("registry")
                .add(
                    IntegrationMapping::new(
                        "pacha",
                        "HF Hub registry",
                        IntegrationType::Alternative,
                    )
                    .with_notes("Model/data/recipe registry"),
                )
                .add(
                    IntegrationMapping::new("pacha", "MLflow/W&B", IntegrationType::Alternative)
                        .with_notes("Full lineage tracking"),
                ),
        )
        .add_category(
            IntegrationCategory::new("inference")
                .add(
                    IntegrationMapping::new(
                        "realizar",
                        "transformers",
                        IntegrationType::Alternative,
                    )
                    .with_notes("Pure Rust LLM inference"),
                )
                .add(
                    IntegrationMapping::new("realizar", "TGI", IntegrationType::Alternative)
                        .with_notes("Native server"),
                )
                .add(
                    IntegrationMapping::new(
                        "realizar/moe",
                        "optimum",
                        IntegrationType::Alternative,
                    )
                    .with_notes("MoE backend selection"),
                ),
        )
        .add_category(
            IntegrationCategory::new("classical_ml")
                .add(
                    IntegrationMapping::new("aprender", "sklearn", IntegrationType::Alternative)
                        .with_notes("Pure Rust ML algorithms"),
                )
                .add(
                    IntegrationMapping::new(
                        "aprender",
                        "xgboost/lightgbm",
                        IntegrationType::Alternative,
                    )
                    .with_notes("Gradient boosting"),
                ),
        )
        .add_category(
            IntegrationCategory::new("deep_learning")
                .add(
                    IntegrationMapping::new(
                        "entrenar",
                        "PyTorch training",
                        IntegrationType::Alternative,
                    )
                    .with_notes("DL training loops"),
                )
                .add(
                    IntegrationMapping::new("alimentar", "datasets", IntegrationType::Alternative)
                        .with_notes("Data loading/streaming"),
                ),
        )
        .add_category(
            IntegrationCategory::new("data_formats")
                .add(
                    IntegrationMapping::new(".ald", "parquet/arrow", IntegrationType::Alternative)
                        .with_notes("Secure dataset format"),
                )
                .add(
                    IntegrationMapping::new(".ald", "json/csv", IntegrationType::Alternative)
                        .with_notes("+encryption/signing/licensing"),
                ),
        )
        .add_category(
            IntegrationCategory::new("compute")
                .add(
                    IntegrationMapping::new(
                        "trueno",
                        "NumPy/PyTorch tensors",
                        IntegrationType::Alternative,
                    )
                    .with_notes("SIMD tensor ops"),
                )
                .add(
                    IntegrationMapping::new("repartir", "accelerate", IntegrationType::Alternative)
                        .with_notes("Distributed compute"),
                ),
        )
        .add_category(
            IntegrationCategory::new("tokenization")
                .add(
                    IntegrationMapping::new(
                        "realizar/tokenizer",
                        "tokenizers",
                        IntegrationType::Compatible,
                    )
                    .with_notes("Load HF tokenizers"),
                )
                .add(
                    IntegrationMapping::new(
                        "trueno-rag",
                        "tokenizers",
                        IntegrationType::Compatible,
                    )
                    .with_notes("RAG tokenization"),
                ),
        )
        .add_category(
            IntegrationCategory::new("apps")
                .add(
                    IntegrationMapping::new("presentar", "gradio", IntegrationType::Alternative)
                        .with_notes("Rust app framework"),
                )
                .add(
                    IntegrationMapping::new(
                        "trueno-viz",
                        "visualization",
                        IntegrationType::Alternative,
                    )
                    .with_notes("GPU rendering"),
                ),
        )
        .add_category(
            IntegrationCategory::new("quality").add(
                IntegrationMapping::new("certeza", "evaluate", IntegrationType::Alternative)
                    .with_notes("Rust metrics"),
            ),
        )
        .add_category(
            IntegrationCategory::new("mcp_tooling")
                .add(
                    IntegrationMapping::new(
                        "pforge",
                        "LangChain Tools",
                        IntegrationType::Alternative,
                    )
                    .with_notes("Declarative MCP servers"),
                )
                .add(
                    IntegrationMapping::new(
                        "pmat",
                        "code analysis tools",
                        IntegrationType::Alternative,
                    )
                    .with_notes("AI context generation"),
                )
                .add(
                    IntegrationMapping::new("pmcp", "mcp-sdk", IntegrationType::Alternative)
                        .with_notes("Rust MCP runtime"),
                ),
        )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    // ========================================================================
    // HF-TREE-001: HfComponent Tests
    // ========================================================================

    #[test]
    fn test_HF_TREE_001_hf_component_new() {
        let comp = HfComponent::new("transformers", "Model architectures");
        assert_eq!(comp.name, "transformers");
        assert_eq!(comp.description, "Model architectures");
        assert!(comp.url.is_none());
    }

    #[test]
    fn test_HF_TREE_001_hf_component_with_url() {
        let comp = HfComponent::new("transformers", "Models")
            .with_url("https://huggingface.co/docs/transformers");
        assert!(comp.url.is_some());
    }

    #[test]
    fn test_HF_TREE_001_hf_category_new() {
        let cat = HfCategory::new("libraries");
        assert_eq!(cat.name, "libraries");
        assert!(cat.components.is_empty());
    }

    #[test]
    fn test_HF_TREE_001_hf_category_add() {
        let cat = HfCategory::new("libraries").add(HfComponent::new("transformers", "Models"));
        assert_eq!(cat.components.len(), 1);
    }

    // ========================================================================
    // HF-TREE-002: HfTree Tests
    // ========================================================================

    #[test]
    fn test_HF_TREE_002_hf_tree_new() {
        let tree = HfTree::new("Test");
        assert_eq!(tree.name, "Test");
        assert!(tree.categories.is_empty());
    }

    #[test]
    fn test_HF_TREE_002_hf_tree_add_category() {
        let tree = HfTree::new("Test").add_category(HfCategory::new("hub"));
        assert_eq!(tree.categories.len(), 1);
    }

    #[test]
    fn test_HF_TREE_002_hf_tree_total_components() {
        let tree = HfTree::new("Test").add_category(
            HfCategory::new("hub")
                .add(HfComponent::new("models", "Models"))
                .add(HfComponent::new("datasets", "Datasets")),
        );
        assert_eq!(tree.total_components(), 2);
    }

    // ========================================================================
    // HF-TREE-003: IntegrationType Tests
    // ========================================================================

    #[test]
    fn test_HF_TREE_003_integration_type_display() {
        assert_eq!(format!("{}", IntegrationType::Compatible), "✓ COMPATIBLE");
        assert_eq!(format!("{}", IntegrationType::Alternative), "⚡ ALTERNATIVE");
        assert_eq!(format!("{}", IntegrationType::Orchestrates), "🔄 ORCHESTRATES");
        assert_eq!(format!("{}", IntegrationType::Uses), "📦 USES");
    }

    #[test]
    fn test_HF_TREE_003_integration_mapping_new() {
        let mapping =
            IntegrationMapping::new("realizar", "transformers", IntegrationType::Alternative);
        assert_eq!(mapping.paiml_component, "realizar");
        assert_eq!(mapping.hf_equivalent, "transformers");
    }

    #[test]
    fn test_HF_TREE_003_integration_mapping_with_notes() {
        let mapping =
            IntegrationMapping::new("a", "b", IntegrationType::Compatible).with_notes("Test notes");
        assert_eq!(mapping.notes, Some("Test notes".to_string()));
    }

    // ========================================================================
    // HF-TREE-004: IntegrationTree Tests
    // ========================================================================

    #[test]
    fn test_HF_TREE_004_integration_tree_new() {
        let tree = IntegrationTree::new();
        assert!(tree.categories.is_empty());
    }

    #[test]
    fn test_HF_TREE_004_integration_tree_counts() {
        let tree = IntegrationTree::new().add_category(
            IntegrationCategory::new("test")
                .add(IntegrationMapping::new("a", "b", IntegrationType::Compatible))
                .add(IntegrationMapping::new("c", "d", IntegrationType::Alternative)),
        );
        assert_eq!(tree.total_mappings(), 2);
        assert_eq!(tree.compatible_count(), 1);
        assert_eq!(tree.alternative_count(), 1);
    }

    // ========================================================================
    // HF-TREE-005: Formatter Tests
    // ========================================================================

    #[test]
    fn test_HF_TREE_005_format_hf_tree_ascii() {
        let tree = HfTree::new("Test")
            .add_category(HfCategory::new("hub").add(HfComponent::new("models", "700K+ models")));
        let output = format_hf_tree_ascii(&tree);
        assert!(output.contains("Test"));
        assert!(output.contains("hub"));
        assert!(output.contains("models"));
    }

    #[test]
    fn test_HF_TREE_005_format_integration_tree_ascii() {
        let tree = IntegrationTree::new().add_category(
            IntegrationCategory::new("formats").add(IntegrationMapping::new(
                "a",
                "b",
                IntegrationType::Compatible,
            )),
        );
        let output = format_integration_tree_ascii(&tree);
        assert!(output.contains("PAIML"));
        assert!(output.contains("HuggingFace"));
        assert!(output.contains("Legend"));
    }

    #[test]
    fn test_HF_TREE_005_format_hf_tree_json() {
        let tree = HfTree::new("Test");
        let json = format_hf_tree_json(&tree).expect("unexpected failure");
        assert!(json.contains("\"name\": \"Test\""));
    }

    #[test]
    fn test_HF_TREE_005_format_integration_tree_json() {
        let tree = IntegrationTree::new();
        let json = format_integration_tree_json(&tree).expect("unexpected failure");
        assert!(json.contains("categories"));
    }

    // ========================================================================
    // HF-TREE-006: Builder Tests
    // ========================================================================

    #[test]
    fn test_HF_TREE_006_build_hf_tree() {
        let tree = build_hf_tree();
        assert_eq!(tree.name, "HuggingFace Ecosystem");
        assert!(!tree.categories.is_empty());
        assert!(tree.total_components() > 20);
    }

    #[test]
    fn test_HF_TREE_006_build_hf_tree_has_hub() {
        let tree = build_hf_tree();
        let hub = tree.categories.iter().find(|c| c.name == "hub");
        assert!(hub.is_some());
        assert!(hub.expect("unexpected failure").components.iter().any(|c| c.name == "models"));
    }

    #[test]
    fn test_HF_TREE_006_build_hf_tree_has_libraries() {
        let tree = build_hf_tree();
        let libs = tree.categories.iter().find(|c| c.name == "libraries");
        assert!(libs.is_some());
        assert!(libs
            .expect("unexpected failure")
            .components
            .iter()
            .any(|c| c.name == "transformers"));
    }

    #[test]
    fn test_HF_TREE_006_build_integration_tree() {
        let tree = build_integration_tree();
        assert!(!tree.categories.is_empty());
        assert!(tree.total_mappings() > 10);
    }

    #[test]
    fn test_HF_TREE_006_build_integration_tree_has_formats() {
        let tree = build_integration_tree();
        let formats = tree.categories.iter().find(|c| c.name == "formats");
        assert!(formats.is_some());
    }

    #[test]
    fn test_HF_TREE_006_build_integration_tree_has_compatible() {
        let tree = build_integration_tree();
        assert!(tree.compatible_count() > 0);
    }

    #[test]
    fn test_HF_TREE_006_build_integration_tree_has_alternatives() {
        let tree = build_integration_tree();
        assert!(tree.alternative_count() > 0);
    }

    // ========================================================================
    // HF-TREE-007: Integration Tests
    // ========================================================================

    #[test]
    fn test_HF_TREE_007_full_hf_tree_output() {
        let tree = build_hf_tree();
        let output = format_hf_tree_ascii(&tree);
        assert!(output.contains("HuggingFace Ecosystem"));
        assert!(output.contains("hub"));
        assert!(output.contains("libraries"));
        assert!(output.contains("transformers"));
    }

    #[test]
    fn test_HF_TREE_007_full_integration_tree_output() {
        let tree = build_integration_tree();
        let output = format_integration_tree_ascii(&tree);
        assert!(output.contains("PAIML"));
        assert!(output.contains("realizar"));
        assert!(output.contains("COMPATIBLE"));
        assert!(output.contains("ALTERNATIVE"));
    }

    #[test]
    fn test_HF_TREE_007_json_roundtrip() {
        let tree = build_hf_tree();
        let json = format_hf_tree_json(&tree).expect("unexpected failure");
        let parsed: HfTree = serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(parsed.name, tree.name);
    }
}
