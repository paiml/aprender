//! Data Platforms Ecosystem Tree Visualization
//!
//! Provides hierarchical views of:
//! - Enterprise data platform ecosystems
//! - PAIML-Platform integration mappings

use serde::{Deserialize, Serialize};
use std::fmt;

// ============================================================================
// DATA-TREE-001: Core Types
// ============================================================================

/// Supported data platforms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Platform {
    Databricks,
    Snowflake,
    Aws,
    HuggingFace,
    Paiml,
}

impl Platform {
    /// Get display name
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Databricks => "DATABRICKS",
            Self::Snowflake => "SNOWFLAKE",
            Self::Aws => "AWS",
            Self::HuggingFace => "HUGGINGFACE",
            Self::Paiml => "PAIML",
        }
    }

    /// Get all platforms
    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![Self::Databricks, Self::Snowflake, Self::Aws, Self::HuggingFace]
    }
}

impl fmt::Display for Platform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Platform category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformCategory {
    pub name: String,
    pub components: Vec<PlatformComponent>,
}

impl PlatformCategory {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), components: Vec::new() }
    }

    pub fn with_component(mut self, component: PlatformComponent) -> Self {
        self.components.push(component);
        self
    }
}

/// A component within a platform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformComponent {
    pub name: String,
    pub description: String,
    pub sub_components: Vec<String>,
}

impl PlatformComponent {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self { name: name.into(), description: description.into(), sub_components: Vec::new() }
    }

    pub fn with_sub(mut self, sub: impl Into<String>) -> Self {
        self.sub_components.push(sub.into());
        self
    }
}

/// Complete platform tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPlatformTree {
    pub platform: Platform,
    pub categories: Vec<PlatformCategory>,
}

impl DataPlatformTree {
    pub fn new(platform: Platform) -> Self {
        Self { platform, categories: Vec::new() }
    }

    pub fn add_category(mut self, category: PlatformCategory) -> Self {
        self.categories.push(category);
        self
    }

    pub fn total_components(&self) -> usize {
        self.categories.iter().map(|c| c.components.len()).sum()
    }
}

// ============================================================================
// DATA-TREE-002: Integration Types
// ============================================================================

/// Integration type between PAIML and data platforms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntegrationType {
    /// PAIML is compatible with platform format
    Compatible,
    /// PAIML provides a native Rust alternative
    Alternative,
    /// PAIML orchestrates platform workflows
    Orchestrates,
    /// PAIML uses platform directly
    Uses,
    /// PAIML transpiles platform code
    Transpiles,
}

impl IntegrationType {
    /// Get short code for display
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Compatible => "CMP",
            Self::Alternative => "ALT",
            Self::Orchestrates => "ORC",
            Self::Uses => "USE",
            Self::Transpiles => "TRN",
        }
    }

    /// Get description
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::Compatible => "Compatible format/protocol",
            Self::Alternative => "PAIML native alternative",
            Self::Orchestrates => "PAIML orchestrates",
            Self::Uses => "PAIML uses directly",
            Self::Transpiles => "PAIML transpiles to Rust",
        }
    }
}

impl fmt::Display for IntegrationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", self.code())
    }
}

/// Integration mapping between platform and PAIML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationMapping {
    pub platform_component: String,
    pub paiml_component: String,
    pub integration_type: IntegrationType,
    pub category: String,
}

impl IntegrationMapping {
    pub fn new(
        platform: impl Into<String>,
        paiml: impl Into<String>,
        integration_type: IntegrationType,
        category: impl Into<String>,
    ) -> Self {
        Self {
            platform_component: platform.into(),
            paiml_component: paiml.into(),
            integration_type,
            category: category.into(),
        }
    }
}

// ============================================================================
// DATA-TREE-003: Tree Builders
// ============================================================================

/// Build the Databricks platform tree
#[must_use]
pub fn build_databricks_tree() -> DataPlatformTree {
    DataPlatformTree::new(Platform::Databricks)
        .add_category(
            PlatformCategory::new("Unity Catalog").with_component(
                PlatformComponent::new("Unity Catalog", "Unified governance for data and AI")
                    .with_sub("Schemas")
                    .with_sub("Tables")
                    .with_sub("Views"),
            ),
        )
        .add_category(
            PlatformCategory::new("Delta Lake").with_component(
                PlatformComponent::new("Delta Lake", "ACID transactions on data lakes")
                    .with_sub("Parquet storage")
                    .with_sub("Transaction log")
                    .with_sub("Time travel"),
            ),
        )
        .add_category(
            PlatformCategory::new("MLflow").with_component(
                PlatformComponent::new("MLflow", "ML lifecycle management")
                    .with_sub("Experiment tracking")
                    .with_sub("Model registry")
                    .with_sub("Model serving"),
            ),
        )
        .add_category(
            PlatformCategory::new("Spark").with_component(
                PlatformComponent::new("Spark", "Distributed compute engine")
                    .with_sub("DataFrames")
                    .with_sub("Structured Streaming")
                    .with_sub("MLlib"),
            ),
        )
}

/// Build the Snowflake platform tree
#[must_use]
pub fn build_snowflake_tree() -> DataPlatformTree {
    DataPlatformTree::new(Platform::Snowflake)
        .add_category(
            PlatformCategory::new("Virtual Warehouse").with_component(
                PlatformComponent::new("Virtual Warehouse", "Elastic compute clusters")
                    .with_sub("Compute clusters")
                    .with_sub("Result cache")
                    .with_sub("Auto-scaling"),
            ),
        )
        .add_category(
            PlatformCategory::new("Iceberg Tables").with_component(
                PlatformComponent::new("Iceberg Tables", "Open table format support")
                    .with_sub("Open format")
                    .with_sub("Schema evolution")
                    .with_sub("Partition pruning"),
            ),
        )
        .add_category(
            PlatformCategory::new("Snowpark").with_component(
                PlatformComponent::new("Snowpark", "Developer experience for data")
                    .with_sub("Python UDFs")
                    .with_sub("Java/Scala UDFs")
                    .with_sub("ML functions"),
            ),
        )
        .add_category(
            PlatformCategory::new("Data Sharing").with_component(
                PlatformComponent::new("Data Sharing", "Secure data exchange")
                    .with_sub("Secure shares")
                    .with_sub("Reader accounts")
                    .with_sub("Marketplace"),
            ),
        )
}

/// Build the AWS platform tree
#[must_use]
pub fn build_aws_tree() -> DataPlatformTree {
    DataPlatformTree::new(Platform::Aws)
        .add_category(
            PlatformCategory::new("Storage")
                .with_component(
                    PlatformComponent::new("S3", "Object storage")
                        .with_sub("Objects")
                        .with_sub("Versioning")
                        .with_sub("Lifecycle"),
                )
                .with_component(
                    PlatformComponent::new("Glue Catalog", "Metadata catalog")
                        .with_sub("Databases")
                        .with_sub("Tables")
                        .with_sub("Crawlers"),
                )
                .with_component(PlatformComponent::new("Lake Formation", "Data lake management")),
        )
        .add_category(
            PlatformCategory::new("Compute")
                .with_component(PlatformComponent::new("EMR", "Managed Spark/Hadoop"))
                .with_component(PlatformComponent::new("Lambda", "Serverless functions"))
                .with_component(PlatformComponent::new("ECS/EKS", "Container orchestration")),
        )
        .add_category(
            PlatformCategory::new("ML")
                .with_component(
                    PlatformComponent::new("SageMaker", "ML platform")
                        .with_sub("Training")
                        .with_sub("Endpoints")
                        .with_sub("Pipelines"),
                )
                .with_component(
                    PlatformComponent::new("Bedrock", "Foundation models")
                        .with_sub("Foundation models")
                        .with_sub("Fine-tuning")
                        .with_sub("Agents"),
                )
                .with_component(PlatformComponent::new("Comprehend", "NLP service")),
        )
        .add_category(
            PlatformCategory::new("Analytics")
                .with_component(PlatformComponent::new("Athena", "Serverless SQL"))
                .with_component(PlatformComponent::new("Redshift", "Data warehouse"))
                .with_component(PlatformComponent::new("QuickSight", "BI dashboards")),
        )
}

/// Build the HuggingFace platform tree
#[must_use]
pub fn build_huggingface_tree() -> DataPlatformTree {
    DataPlatformTree::new(Platform::HuggingFace)
        .add_category(
            PlatformCategory::new("Hub").with_component(
                PlatformComponent::new("Hub", "Model and dataset repository")
                    .with_sub("Models (500K+)")
                    .with_sub("Datasets (100K+)")
                    .with_sub("Spaces (200K+)"),
            ),
        )
        .add_category(
            PlatformCategory::new("Libraries")
                .with_component(PlatformComponent::new("Transformers", "Model library"))
                .with_component(PlatformComponent::new("Datasets", "Dataset library"))
                .with_component(PlatformComponent::new("Tokenizers", "Fast tokenization"))
                .with_component(PlatformComponent::new("Accelerate", "Distributed training")),
        )
        .add_category(
            PlatformCategory::new("Formats")
                .with_component(PlatformComponent::new("SafeTensors", "Safe model format"))
                .with_component(PlatformComponent::new("GGUF", "Quantized models"))
                .with_component(PlatformComponent::new("ONNX", "Interop format")),
        )
}

// ============================================================================
// DATA-TREE-004: Integration Mappings
// ============================================================================

/// Build all integration mappings
#[must_use]
pub fn build_integration_mappings() -> Vec<IntegrationMapping> {
    vec![
        // Storage & Catalogs
        IntegrationMapping::new(
            "Delta Lake",
            "Alimentar (.ald)",
            IntegrationType::Alternative,
            "STORAGE & CATALOGS",
        ),
        IntegrationMapping::new(
            "Iceberg Tables",
            "Alimentar (.ald)",
            IntegrationType::Compatible,
            "STORAGE & CATALOGS",
        ),
        IntegrationMapping::new(
            "S3",
            "Alimentar (sync)",
            IntegrationType::Compatible,
            "STORAGE & CATALOGS",
        ),
        IntegrationMapping::new(
            "Unity Catalog",
            "Pacha Registry",
            IntegrationType::Alternative,
            "STORAGE & CATALOGS",
        ),
        IntegrationMapping::new(
            "Glue Catalog",
            "Pacha Registry",
            IntegrationType::Alternative,
            "STORAGE & CATALOGS",
        ),
        IntegrationMapping::new(
            "HuggingFace Hub",
            "Pacha Registry",
            IntegrationType::Alternative,
            "STORAGE & CATALOGS",
        ),
        // Compute & Processing
        IntegrationMapping::new(
            "Spark DataFrames",
            "Trueno",
            IntegrationType::Alternative,
            "COMPUTE & PROCESSING",
        ),
        IntegrationMapping::new(
            "Snowpark",
            "Trueno",
            IntegrationType::Alternative,
            "COMPUTE & PROCESSING",
        ),
        IntegrationMapping::new(
            "EMR",
            "Trueno",
            IntegrationType::Alternative,
            "COMPUTE & PROCESSING",
        ),
        IntegrationMapping::new(
            "Snowpark Python",
            "Depyler → Rust",
            IntegrationType::Transpiles,
            "COMPUTE & PROCESSING",
        ),
        IntegrationMapping::new(
            "Lambda Python",
            "Depyler → Rust",
            IntegrationType::Transpiles,
            "COMPUTE & PROCESSING",
        ),
        IntegrationMapping::new(
            "Neptune/GraphQL",
            "Trueno-Graph",
            IntegrationType::Alternative,
            "COMPUTE & PROCESSING",
        ),
        // ML Training
        IntegrationMapping::new("MLlib", "Aprender", IntegrationType::Alternative, "ML TRAINING"),
        IntegrationMapping::new(
            "Snowpark ML",
            "Aprender",
            IntegrationType::Alternative,
            "ML TRAINING",
        ),
        IntegrationMapping::new(
            "SageMaker Training",
            "Entrenar",
            IntegrationType::Alternative,
            "ML TRAINING",
        ),
        IntegrationMapping::new(
            "MLflow Tracking",
            "Entrenar",
            IntegrationType::Alternative,
            "ML TRAINING",
        ),
        IntegrationMapping::new(
            "SageMaker Experiments",
            "Entrenar",
            IntegrationType::Alternative,
            "ML TRAINING",
        ),
        IntegrationMapping::new("W&B", "Entrenar", IntegrationType::Uses, "ML TRAINING"),
        // Model Serving
        IntegrationMapping::new(
            "MLflow Serving",
            "Realizar",
            IntegrationType::Alternative,
            "MODEL SERVING",
        ),
        IntegrationMapping::new(
            "SageMaker Endpoints",
            "Realizar",
            IntegrationType::Alternative,
            "MODEL SERVING",
        ),
        IntegrationMapping::new(
            "Bedrock",
            "Realizar + serve",
            IntegrationType::Alternative,
            "MODEL SERVING",
        ),
        IntegrationMapping::new("GGUF models", "Realizar", IntegrationType::Uses, "MODEL SERVING"),
        IntegrationMapping::new(
            "HF Transformers",
            "Realizar (via GGUF)",
            IntegrationType::Compatible,
            "MODEL SERVING",
        ),
        // Orchestration
        IntegrationMapping::new(
            "Databricks Workflows",
            "Batuta",
            IntegrationType::Orchestrates,
            "ORCHESTRATION",
        ),
        IntegrationMapping::new(
            "Snowflake Tasks",
            "Batuta",
            IntegrationType::Orchestrates,
            "ORCHESTRATION",
        ),
        IntegrationMapping::new(
            "Step Functions",
            "Batuta",
            IntegrationType::Orchestrates,
            "ORCHESTRATION",
        ),
        IntegrationMapping::new(
            "Airflow/Prefect",
            "Batuta",
            IntegrationType::Orchestrates,
            "ORCHESTRATION",
        ),
    ]
}

// ============================================================================
// DATA-TREE-005: Display Formatters
// ============================================================================

/// Format platform tree for display
pub fn format_platform_tree(tree: &DataPlatformTree) -> String {
    let mut output = String::new();
    output.push_str(&format!("{}\n", tree.platform.name()));

    for (i, category) in tree.categories.iter().enumerate() {
        let is_last_category = i == tree.categories.len() - 1;
        let prefix = if is_last_category { "└── " } else { "├── " };
        let child_prefix = if is_last_category { "    " } else { "│   " };

        output.push_str(&format!("{}{}\n", prefix, category.name));

        for (j, component) in category.components.iter().enumerate() {
            let is_last_component = j == category.components.len() - 1;
            let comp_prefix = if is_last_component { "└── " } else { "├── " };
            let sub_prefix = if is_last_component { "    " } else { "│   " };

            output.push_str(&format!("{}{}{}\n", child_prefix, comp_prefix, component.name));

            for (k, sub) in component.sub_components.iter().enumerate() {
                let is_last_sub = k == component.sub_components.len() - 1;
                let sub_comp_prefix = if is_last_sub { "└── " } else { "├── " };
                output
                    .push_str(&format!("{}{}{}{}", child_prefix, sub_prefix, sub_comp_prefix, sub));
                output.push('\n');
            }
        }
    }

    output
}

/// Format all platforms tree
pub fn format_all_platforms() -> String {
    let mut output = String::new();
    output.push_str("DATA PLATFORMS ECOSYSTEM\n");
    output.push_str("========================\n\n");

    let trees = vec![
        build_databricks_tree(),
        build_snowflake_tree(),
        build_aws_tree(),
        build_huggingface_tree(),
    ];

    let total_components: usize = trees.iter().map(|t| t.total_components()).sum();
    let total_categories: usize = trees.iter().map(|t| t.categories.len()).sum();

    for tree in &trees {
        output.push_str(&format_platform_tree(tree));
        output.push('\n');
    }

    output.push_str(&format!(
        "Summary: {} platforms, {} categories, {} components\n",
        trees.len(),
        total_categories,
        total_components
    ));

    output
}

/// Format integration mappings
pub fn format_integration_mappings() -> String {
    let mut output = String::new();
    output.push_str("PAIML ↔ DATA PLATFORMS INTEGRATION\n");
    output.push_str("==================================\n\n");

    let mappings = build_integration_mappings();

    // Group by category
    let mut current_category = String::new();
    for mapping in &mappings {
        if mapping.category != current_category {
            if !current_category.is_empty() {
                output.push('\n');
            }
            output.push_str(&format!("{}\n", mapping.category));
            current_category = mapping.category.clone();
        }

        output.push_str(&format!(
            "├── {} {} ←→ {}\n",
            mapping.integration_type, mapping.paiml_component, mapping.platform_component
        ));
    }

    // Count by type
    let compatible =
        mappings.iter().filter(|m| m.integration_type == IntegrationType::Compatible).count();
    let alternative =
        mappings.iter().filter(|m| m.integration_type == IntegrationType::Alternative).count();
    let uses = mappings.iter().filter(|m| m.integration_type == IntegrationType::Uses).count();
    let transpiles =
        mappings.iter().filter(|m| m.integration_type == IntegrationType::Transpiles).count();
    let orchestrates =
        mappings.iter().filter(|m| m.integration_type == IntegrationType::Orchestrates).count();

    output.push_str("\nLegend: [CMP]=Compatible [ALT]=Alternative [USE]=Uses [TRN]=Transpiles [ORC]=Orchestrates\n\n");
    output.push_str(&format!(
        "Summary: {} compatible, {} alternatives, {} uses, {} transpiles, {} orchestrates\n",
        compatible, alternative, uses, transpiles, orchestrates
    ));
    output.push_str(&format!("         Total: {} integration points\n", mappings.len()));

    output
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    // ========================================================================
    // DATA-TREE-001: Platform Tests
    // ========================================================================

    #[test]
    fn test_DATA_TREE_001_platform_names() {
        assert_eq!(Platform::Databricks.name(), "DATABRICKS");
        assert_eq!(Platform::Snowflake.name(), "SNOWFLAKE");
        assert_eq!(Platform::Aws.name(), "AWS");
        assert_eq!(Platform::HuggingFace.name(), "HUGGINGFACE");
    }

    #[test]
    fn test_DATA_TREE_001_platform_all() {
        let all = Platform::all();
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn test_DATA_TREE_001_platform_display() {
        assert_eq!(format!("{}", Platform::Databricks), "DATABRICKS");
    }

    // ========================================================================
    // DATA-TREE-002: Integration Type Tests
    // ========================================================================

    #[test]
    fn test_DATA_TREE_002_integration_codes() {
        assert_eq!(IntegrationType::Compatible.code(), "CMP");
        assert_eq!(IntegrationType::Alternative.code(), "ALT");
        assert_eq!(IntegrationType::Orchestrates.code(), "ORC");
        assert_eq!(IntegrationType::Uses.code(), "USE");
        assert_eq!(IntegrationType::Transpiles.code(), "TRN");
    }

    #[test]
    fn test_DATA_TREE_002_integration_display() {
        assert_eq!(format!("{}", IntegrationType::Compatible), "[CMP]");
    }

    // ========================================================================
    // DATA-TREE-003: Tree Builder Tests
    // ========================================================================

    #[test]
    fn test_DATA_TREE_003_databricks_tree() {
        let tree = build_databricks_tree();
        assert_eq!(tree.platform, Platform::Databricks);
        assert!(!tree.categories.is_empty());
        assert!(tree.total_components() > 0);
    }

    #[test]
    fn test_DATA_TREE_003_snowflake_tree() {
        let tree = build_snowflake_tree();
        assert_eq!(tree.platform, Platform::Snowflake);
        assert!(!tree.categories.is_empty());
    }

    #[test]
    fn test_DATA_TREE_003_aws_tree() {
        let tree = build_aws_tree();
        assert_eq!(tree.platform, Platform::Aws);
        assert!(tree.categories.len() >= 4); // Storage, Compute, ML, Analytics
    }

    #[test]
    fn test_DATA_TREE_003_huggingface_tree() {
        let tree = build_huggingface_tree();
        assert_eq!(tree.platform, Platform::HuggingFace);
        assert!(!tree.categories.is_empty());
    }

    // ========================================================================
    // DATA-TREE-004: Integration Mapping Tests
    // ========================================================================

    #[test]
    fn test_DATA_TREE_004_mappings_exist() {
        let mappings = build_integration_mappings();
        assert!(!mappings.is_empty());
        assert!(mappings.len() >= 25); // At least 25 integration points
    }

    #[test]
    fn test_DATA_TREE_004_mapping_categories() {
        let mappings = build_integration_mappings();
        let categories: std::collections::HashSet<_> =
            mappings.iter().map(|m| &m.category).collect();
        assert!(categories.contains(&"STORAGE & CATALOGS".to_string()));
        assert!(categories.contains(&"ML TRAINING".to_string()));
        assert!(categories.contains(&"MODEL SERVING".to_string()));
    }

    #[test]
    fn test_DATA_TREE_004_has_all_types() {
        let mappings = build_integration_mappings();
        let types: std::collections::HashSet<_> =
            mappings.iter().map(|m| m.integration_type).collect();
        assert!(types.contains(&IntegrationType::Compatible));
        assert!(types.contains(&IntegrationType::Alternative));
        assert!(types.contains(&IntegrationType::Uses));
        assert!(types.contains(&IntegrationType::Transpiles));
        assert!(types.contains(&IntegrationType::Orchestrates));
    }

    // ========================================================================
    // DATA-TREE-005: Formatter Tests
    // ========================================================================

    #[test]
    fn test_DATA_TREE_005_format_platform_tree() {
        let tree = build_databricks_tree();
        let output = format_platform_tree(&tree);
        assert!(output.contains("DATABRICKS"));
        assert!(output.contains("Delta Lake"));
        assert!(output.contains("MLflow"));
    }

    #[test]
    fn test_DATA_TREE_005_format_all_platforms() {
        let output = format_all_platforms();
        assert!(output.contains("DATA PLATFORMS ECOSYSTEM"));
        assert!(output.contains("DATABRICKS"));
        assert!(output.contains("SNOWFLAKE"));
        assert!(output.contains("AWS"));
        assert!(output.contains("HUGGINGFACE"));
        assert!(output.contains("Summary:"));
    }

    #[test]
    fn test_DATA_TREE_005_format_integration_mappings() {
        let output = format_integration_mappings();
        assert!(output.contains("PAIML"));
        assert!(output.contains("STORAGE & CATALOGS"));
        assert!(output.contains("Alimentar"));
        assert!(output.contains("Legend:"));
        assert!(output.contains("Total:"));
    }

    // ========================================================================
    // DATA-TREE-006: Component Tests
    // ========================================================================

    #[test]
    fn test_DATA_TREE_006_platform_category() {
        let category = PlatformCategory::new("Test Category")
            .with_component(PlatformComponent::new("Component1", "Description 1"));
        assert_eq!(category.name, "Test Category");
        assert_eq!(category.components.len(), 1);
    }

    #[test]
    fn test_DATA_TREE_006_platform_component() {
        let component =
            PlatformComponent::new("Test", "Description").with_sub("Sub1").with_sub("Sub2");
        assert_eq!(component.name, "Test");
        assert_eq!(component.sub_components.len(), 2);
    }

    #[test]
    fn test_DATA_TREE_006_integration_mapping() {
        let mapping =
            IntegrationMapping::new("Source", "Target", IntegrationType::Compatible, "Category");
        assert_eq!(mapping.platform_component, "Source");
        assert_eq!(mapping.paiml_component, "Target");
    }

    // ========================================================================
    // Additional Coverage Tests
    // ========================================================================

    #[test]
    fn test_integration_type_descriptions() {
        assert!(!IntegrationType::Compatible.description().is_empty());
        assert!(!IntegrationType::Alternative.description().is_empty());
        assert!(!IntegrationType::Orchestrates.description().is_empty());
        assert!(!IntegrationType::Uses.description().is_empty());
        assert!(!IntegrationType::Transpiles.description().is_empty());
    }

    #[test]
    fn test_platform_paiml() {
        assert_eq!(Platform::Paiml.name(), "PAIML");
        assert_eq!(format!("{}", Platform::Paiml), "PAIML");
    }

    #[test]
    fn test_data_platform_tree_empty() {
        let tree = DataPlatformTree::new(Platform::Paiml);
        assert_eq!(tree.platform, Platform::Paiml);
        assert!(tree.categories.is_empty());
        assert_eq!(tree.total_components(), 0);
    }

    #[test]
    fn test_data_platform_tree_multiple_categories() {
        let tree = DataPlatformTree::new(Platform::Aws)
            .add_category(
                PlatformCategory::new("Cat1")
                    .with_component(PlatformComponent::new("C1", "D1"))
                    .with_component(PlatformComponent::new("C2", "D2")),
            )
            .add_category(
                PlatformCategory::new("Cat2").with_component(PlatformComponent::new("C3", "D3")),
            );
        assert_eq!(tree.categories.len(), 2);
        assert_eq!(tree.total_components(), 3);
    }

    #[test]
    fn test_platform_equality() {
        assert_eq!(Platform::Databricks, Platform::Databricks);
        assert_ne!(Platform::Databricks, Platform::Snowflake);
    }

    #[test]
    fn test_integration_type_equality() {
        assert_eq!(IntegrationType::Compatible, IntegrationType::Compatible);
        assert_ne!(IntegrationType::Compatible, IntegrationType::Alternative);
    }

    #[test]
    fn test_platform_category_empty() {
        let cat = PlatformCategory::new("Empty");
        assert!(cat.components.is_empty());
    }

    #[test]
    fn test_platform_component_no_subs() {
        let comp = PlatformComponent::new("NoSubs", "Description");
        assert!(comp.sub_components.is_empty());
    }

    #[test]
    fn test_format_platform_tree_with_subs() {
        let tree = DataPlatformTree::new(Platform::Paiml).add_category(
            PlatformCategory::new("Category").with_component(
                PlatformComponent::new("Component", "Desc").with_sub("Sub1").with_sub("Sub2"),
            ),
        );
        let output = format_platform_tree(&tree);
        assert!(output.contains("PAIML"));
        assert!(output.contains("Sub1"));
        assert!(output.contains("Sub2"));
    }

    #[test]
    fn test_integration_mapping_fields() {
        let mapping =
            IntegrationMapping::new("Platform", "PAIML", IntegrationType::Orchestrates, "Category");
        assert_eq!(mapping.integration_type, IntegrationType::Orchestrates);
        assert_eq!(mapping.category, "Category");
    }

    #[test]
    fn test_all_platforms_count() {
        let platforms = Platform::all();
        // all() returns 4 platforms (not including PAIML which is internal)
        assert_eq!(platforms.len(), 4);
        assert!(!platforms.contains(&Platform::Paiml));
    }

    #[test]
    fn test_tree_with_multiple_components_per_category() {
        let tree = DataPlatformTree::new(Platform::Aws).add_category(
            PlatformCategory::new("MultiComp")
                .with_component(PlatformComponent::new("A", "DA"))
                .with_component(PlatformComponent::new("B", "DB"))
                .with_component(PlatformComponent::new("C", "DC")),
        );
        assert_eq!(tree.total_components(), 3);
    }
}
