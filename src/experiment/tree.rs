//! Experiment Tracking Frameworks Tree Visualization
//!
//! Displays comparison tree of Python experiment tracking frameworks
//! (MLflow, Weights & Biases, Neptune, etc.) and their PAIML Rust replacements.
//!
//! Core principle: Python experiment tracking is replaced by sovereign Rust alternatives.
//! No Python runtime dependencies permitted in production.

use serde::{Deserialize, Serialize};

/// Experiment tracking framework being compared
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExperimentFramework {
    /// MLflow - Open source experiment tracking
    MLflow,
    /// Weights & Biases - Commercial experiment tracking
    WandB,
    /// Neptune.ai - Commercial ML metadata store
    Neptune,
    /// Comet ML - Commercial experiment tracking
    CometML,
    /// Sacred - Academic experiment tracking
    Sacred,
    /// DVC - Data Version Control with experiment tracking
    Dvc,
}

impl ExperimentFramework {
    /// Get the display name
    pub fn name(&self) -> &'static str {
        match self {
            ExperimentFramework::MLflow => "MLflow",
            ExperimentFramework::WandB => "Weights & Biases",
            ExperimentFramework::Neptune => "Neptune.ai",
            ExperimentFramework::CometML => "Comet ML",
            ExperimentFramework::Sacred => "Sacred",
            ExperimentFramework::Dvc => "DVC",
        }
    }

    /// Get the PAIML replacement
    pub fn replacement(&self) -> &'static str {
        match self {
            ExperimentFramework::MLflow => "Entrenar + Batuta",
            ExperimentFramework::WandB => "Entrenar + Trueno-Viz",
            ExperimentFramework::Neptune => "Entrenar",
            ExperimentFramework::CometML => "Entrenar + Presentar",
            ExperimentFramework::Sacred => "Entrenar",
            ExperimentFramework::Dvc => "Batuta + Trueno-DB",
        }
    }

    /// Get all frameworks
    pub fn all() -> Vec<ExperimentFramework> {
        vec![
            ExperimentFramework::MLflow,
            ExperimentFramework::WandB,
            ExperimentFramework::Neptune,
            ExperimentFramework::CometML,
            ExperimentFramework::Sacred,
            ExperimentFramework::Dvc,
        ]
    }
}

/// Integration type - all are replacements (Python eliminated)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntegrationType {
    /// PAIML component fully replaces Python equivalent
    Replaces,
}

impl IntegrationType {
    /// Get the display code
    pub fn code(&self) -> &'static str {
        match self {
            IntegrationType::Replaces => "REP",
        }
    }
}

/// A component within a framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkComponent {
    /// Component name
    pub name: String,
    /// Description
    pub description: String,
    /// PAIML replacement
    pub replacement: String,
    /// Sub-components
    pub sub_components: Vec<String>,
}

impl FrameworkComponent {
    fn new(name: &str, description: &str, replacement: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            replacement: replacement.to_string(),
            sub_components: Vec::new(),
        }
    }

    fn with_subs(name: &str, description: &str, replacement: &str, subs: Vec<&str>) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            replacement: replacement.to_string(),
            sub_components: subs.into_iter().map(String::from).collect(),
        }
    }
}

/// A category of components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkCategory {
    /// Category name
    pub name: String,
    /// Components in this category
    pub components: Vec<FrameworkComponent>,
}

/// Integration mapping between Python and Rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationMapping {
    /// PAIML component
    pub paiml_component: String,
    /// Python component being replaced
    pub python_component: String,
    /// Integration type
    pub integration_type: IntegrationType,
    /// Category
    pub category: String,
}

impl IntegrationMapping {
    fn rep(paiml: &str, python: &str, category: &str) -> Self {
        Self {
            paiml_component: paiml.to_string(),
            python_component: python.to_string(),
            integration_type: IntegrationType::Replaces,
            category: category.to_string(),
        }
    }
}

/// Experiment tracking tree structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentTree {
    /// Framework name
    pub framework: String,
    /// PAIML replacement
    pub replacement: String,
    /// Categories
    pub categories: Vec<FrameworkCategory>,
}

/// Build MLflow tree
pub fn build_mlflow_tree() -> ExperimentTree {
    ExperimentTree {
        framework: "MLflow".to_string(),
        replacement: "Entrenar + Batuta".to_string(),
        categories: vec![
            FrameworkCategory {
                name: "Experiment Tracking".to_string(),
                components: vec![
                    FrameworkComponent::with_subs(
                        "mlflow.start_run()",
                        "Run lifecycle management",
                        "Entrenar::ExperimentRun::new()",
                        vec!["log_param", "log_metric", "log_artifact"],
                    ),
                    FrameworkComponent::new(
                        "mlflow.log_params()",
                        "Hyperparameter logging",
                        "ExperimentRun::log_param()",
                    ),
                    FrameworkComponent::new(
                        "mlflow.log_metrics()",
                        "Metric logging",
                        "ExperimentRun::log_metric()",
                    ),
                    FrameworkComponent::new(
                        "mlflow.set_tags()",
                        "Run tagging",
                        "ExperimentRun::tags",
                    ),
                ],
            },
            FrameworkCategory {
                name: "Model Registry".to_string(),
                components: vec![
                    FrameworkComponent::with_subs(
                        "mlflow.register_model()",
                        "Model versioning",
                        "SovereignDistribution",
                        vec!["stage_transitions", "model_versions"],
                    ),
                    FrameworkComponent::new(
                        "mlflow.pyfunc.log_model()",
                        "Model artifact storage",
                        "SovereignArtifact",
                    ),
                ],
            },
            FrameworkCategory {
                name: "Artifact Storage".to_string(),
                components: vec![
                    FrameworkComponent::new(
                        "mlflow.log_artifact()",
                        "File storage",
                        "SovereignArtifact",
                    ),
                    FrameworkComponent::new(
                        "S3/Azure/GCS backends",
                        "Cloud storage",
                        "Batuta deploy (sovereign)",
                    ),
                ],
            },
            FrameworkCategory {
                name: "Search & Query".to_string(),
                components: vec![
                    FrameworkComponent::new(
                        "mlflow.search_runs()",
                        "Run search",
                        "ExperimentStorage::list_runs()",
                    ),
                    FrameworkComponent::new(
                        "mlflow.search_experiments()",
                        "Experiment search",
                        "ExperimentStorage trait",
                    ),
                ],
            },
            FrameworkCategory {
                name: "GenAI / LLM".to_string(),
                components: vec![
                    FrameworkComponent::with_subs(
                        "mlflow.tracing",
                        "LLM trace capture",
                        "Realizar::trace()",
                        vec!["OpenAI", "LangChain", "LlamaIndex"],
                    ),
                    FrameworkComponent::new(
                        "mlflow.evaluate()",
                        "LLM evaluation",
                        "Entrenar::evaluate()",
                    ),
                    FrameworkComponent::new(
                        "Prompt Registry",
                        "Prompt versioning",
                        "Realizar::PromptTemplate",
                    ),
                ],
            },
            FrameworkCategory {
                name: "Deployment".to_string(),
                components: vec![
                    FrameworkComponent::new(
                        "mlflow models serve",
                        "Model serving",
                        "Batuta serve (GGUF)",
                    ),
                    FrameworkComponent::new(
                        "MLflow Gateway",
                        "LLM gateway",
                        "Batuta serve + SpilloverRouter",
                    ),
                ],
            },
        ],
    }
}

/// Build Weights & Biases tree
pub fn build_wandb_tree() -> ExperimentTree {
    ExperimentTree {
        framework: "Weights & Biases".to_string(),
        replacement: "Entrenar + Trueno-Viz".to_string(),
        categories: vec![
            FrameworkCategory {
                name: "Experiment Tracking".to_string(),
                components: vec![
                    FrameworkComponent::new(
                        "wandb.init()",
                        "Run initialization",
                        "Entrenar::ExperimentRun::new()",
                    ),
                    FrameworkComponent::new(
                        "wandb.log()",
                        "Metric logging",
                        "ExperimentRun::log_metric()",
                    ),
                    FrameworkComponent::new(
                        "wandb.config",
                        "Hyperparameter config",
                        "ExperimentRun::hyperparameters",
                    ),
                ],
            },
            FrameworkCategory {
                name: "Visualization".to_string(),
                components: vec![
                    FrameworkComponent::new("wandb.plot", "Custom plots", "Trueno-Viz::Chart"),
                    FrameworkComponent::new("Tables", "Data tables", "Trueno-Viz::DataGrid"),
                    FrameworkComponent::new(
                        "Media logging",
                        "Images/audio/video",
                        "Presentar::MediaView",
                    ),
                ],
            },
            FrameworkCategory {
                name: "Sweeps".to_string(),
                components: vec![FrameworkComponent::with_subs(
                    "wandb.sweep()",
                    "Hyperparameter search",
                    "Entrenar::HyperparameterSearch",
                    vec!["grid", "random", "bayes"],
                )],
            },
            FrameworkCategory {
                name: "Artifacts".to_string(),
                components: vec![FrameworkComponent::new(
                    "wandb.Artifact",
                    "Dataset/model versioning",
                    "SovereignArtifact",
                )],
            },
        ],
    }
}

/// Build Neptune tree
pub fn build_neptune_tree() -> ExperimentTree {
    ExperimentTree {
        framework: "Neptune.ai".to_string(),
        replacement: "Entrenar".to_string(),
        categories: vec![
            FrameworkCategory {
                name: "Experiment Tracking".to_string(),
                components: vec![
                    FrameworkComponent::new(
                        "neptune.init_run()",
                        "Run initialization",
                        "Entrenar::ExperimentRun::new()",
                    ),
                    FrameworkComponent::new(
                        "run[\"metric\"].log()",
                        "Metric logging",
                        "ExperimentRun::log_metric()",
                    ),
                ],
            },
            FrameworkCategory {
                name: "Metadata Store".to_string(),
                components: vec![
                    FrameworkComponent::new(
                        "Namespace hierarchy",
                        "Nested metadata",
                        "ExperimentRun::hyperparameters (JSON)",
                    ),
                    FrameworkComponent::new(
                        "System metrics",
                        "CPU/GPU/memory",
                        "EnergyMetrics + ComputeDevice",
                    ),
                ],
            },
        ],
    }
}

/// Build DVC tree
pub fn build_dvc_tree() -> ExperimentTree {
    ExperimentTree {
        framework: "DVC".to_string(),
        replacement: "Batuta + Trueno-DB".to_string(),
        categories: vec![
            FrameworkCategory {
                name: "Data Versioning".to_string(),
                components: vec![
                    FrameworkComponent::new(
                        "dvc add",
                        "Track data files",
                        "Trueno-DB::DatasetVersion",
                    ),
                    FrameworkComponent::new(
                        "dvc push/pull",
                        "Remote storage",
                        "SovereignDistribution",
                    ),
                ],
            },
            FrameworkCategory {
                name: "Experiment Tracking".to_string(),
                components: vec![
                    FrameworkComponent::new("dvc exp run", "Run experiments", "Batuta orchestrate"),
                    FrameworkComponent::new(
                        "dvc exp show",
                        "Compare experiments",
                        "Batuta experiment tree",
                    ),
                    FrameworkComponent::new(
                        "dvc metrics",
                        "Metric tracking",
                        "ExperimentRun::metrics",
                    ),
                ],
            },
            FrameworkCategory {
                name: "Pipelines".to_string(),
                components: vec![
                    FrameworkComponent::new("dvc.yaml", "Pipeline definition", "batuta.toml"),
                    FrameworkComponent::new(
                        "dvc repro",
                        "Reproduce pipeline",
                        "Batuta transpile + validate",
                    ),
                ],
            },
        ],
    }
}

/// Build all integration mappings
pub fn build_integration_mappings() -> Vec<IntegrationMapping> {
    vec![
        // Experiment Tracking
        IntegrationMapping::rep(
            "Entrenar::ExperimentRun",
            "mlflow.start_run()",
            "Experiment Tracking",
        ),
        IntegrationMapping::rep(
            "ExperimentRun::log_metric()",
            "mlflow.log_metrics()",
            "Experiment Tracking",
        ),
        IntegrationMapping::rep(
            "ExperimentRun::log_param()",
            "mlflow.log_params()",
            "Experiment Tracking",
        ),
        IntegrationMapping::rep("ExperimentRun::tags", "mlflow.set_tags()", "Experiment Tracking"),
        IntegrationMapping::rep("Entrenar::ExperimentRun", "wandb.init()", "Experiment Tracking"),
        IntegrationMapping::rep(
            "Entrenar::ExperimentRun",
            "neptune.init_run()",
            "Experiment Tracking",
        ),
        // Model Registry
        IntegrationMapping::rep(
            "SovereignDistribution",
            "mlflow.register_model()",
            "Model Registry",
        ),
        IntegrationMapping::rep("SovereignArtifact", "mlflow.pyfunc.log_model()", "Model Registry"),
        IntegrationMapping::rep("SovereignArtifact", "wandb.Artifact", "Model Registry"),
        // Cost & Energy
        IntegrationMapping::rep("EnergyMetrics", "System metrics (wandb/neptune)", "Cost & Energy"),
        IntegrationMapping::rep("CostMetrics", "N/A (not in MLflow)", "Cost & Energy"),
        IntegrationMapping::rep(
            "CostPerformanceBenchmark",
            "N/A (Pareto frontier)",
            "Cost & Energy",
        ),
        IntegrationMapping::rep("ComputeDevice", "mlflow.system_metrics", "Cost & Energy"),
        // Visualization
        IntegrationMapping::rep("Trueno-Viz::Chart", "wandb.plot", "Visualization"),
        IntegrationMapping::rep("Trueno-Viz::DataGrid", "wandb.Table", "Visualization"),
        IntegrationMapping::rep("Presentar::Dashboard", "MLflow UI", "Visualization"),
        // LLM / GenAI
        IntegrationMapping::rep("Realizar::trace()", "mlflow.tracing", "LLM / GenAI"),
        IntegrationMapping::rep("Entrenar::evaluate()", "mlflow.evaluate()", "LLM / GenAI"),
        IntegrationMapping::rep(
            "Realizar::PromptTemplate",
            "MLflow Prompt Registry",
            "LLM / GenAI",
        ),
        // Deployment
        IntegrationMapping::rep("Batuta serve", "mlflow models serve", "Deployment"),
        IntegrationMapping::rep("SpilloverRouter", "MLflow Gateway", "Deployment"),
        IntegrationMapping::rep("SovereignDistribution", "MLflow Docker/K8s", "Deployment"),
        // Data Versioning
        IntegrationMapping::rep("Trueno-DB::DatasetVersion", "dvc add", "Data Versioning"),
        IntegrationMapping::rep("Batuta orchestrate", "dvc repro", "Data Versioning"),
        // Academic / Research
        IntegrationMapping::rep("ResearchArtifact", "N/A (ORCID/CRediT)", "Academic / Research"),
        IntegrationMapping::rep("CitationMetadata", "N/A (BibTeX/CFF)", "Academic / Research"),
        IntegrationMapping::rep("PreRegistration", "N/A (reproducibility)", "Academic / Research"),
    ]
}

/// Format a category's components as ASCII tree lines.
fn format_category_components(
    output: &mut String,
    category: &FrameworkCategory,
    cat_continuation: &str,
) {
    for (comp_idx, component) in category.components.iter().enumerate() {
        let is_last_comp = comp_idx == category.components.len() - 1;
        let comp_prefix = if is_last_comp { "└──" } else { "├──" };

        output.push_str(&format!(
            "{}{} {} → {}\n",
            cat_continuation, comp_prefix, component.name, component.replacement
        ));

        if !component.sub_components.is_empty() {
            let sub_cont = if is_last_comp {
                format!("{}    ", cat_continuation)
            } else {
                format!("{}│   ", cat_continuation)
            };
            for (sub_idx, sub) in component.sub_components.iter().enumerate() {
                let sub_prefix = if sub_idx == component.sub_components.len() - 1 {
                    "└──"
                } else {
                    "├──"
                };
                output.push_str(&format!("{}{} {}\n", sub_cont, sub_prefix, sub));
            }
        }
    }
}

/// Format a single framework tree as ASCII
pub fn format_framework_tree(tree: &ExperimentTree) -> String {
    let mut output = String::new();
    output.push_str(&format!("{} (Python) → {} (Rust)\n", tree.framework, tree.replacement));

    for (cat_idx, category) in tree.categories.iter().enumerate() {
        let is_last_cat = cat_idx == tree.categories.len() - 1;
        let cat_prefix = if is_last_cat { "└──" } else { "├──" };
        let cat_continuation = if is_last_cat { "    " } else { "│   " };

        output.push_str(&format!("{} {}\n", cat_prefix, category.name));
        format_category_components(&mut output, category, cat_continuation);
    }

    output
}

/// Format all frameworks
pub fn format_all_frameworks() -> String {
    let mut output = String::new();
    output.push_str("EXPERIMENT TRACKING FRAMEWORKS ECOSYSTEM\n");
    output.push_str("========================================\n\n");

    output.push_str(&format_framework_tree(&build_mlflow_tree()));
    output.push('\n');
    output.push_str(&format_framework_tree(&build_wandb_tree()));
    output.push('\n');
    output.push_str(&format_framework_tree(&build_neptune_tree()));
    output.push('\n');
    output.push_str(&format_framework_tree(&build_dvc_tree()));

    output.push_str(&format!(
        "\nSummary: {} Python frameworks replaced by sovereign Rust stack\n",
        ExperimentFramework::all().len()
    ));

    output
}

/// Format integration mappings
pub fn format_integration_mappings() -> String {
    let mappings = build_integration_mappings();
    let mut output = String::new();

    output.push_str("PAIML REPLACEMENTS FOR PYTHON EXPERIMENT TRACKING\n");
    output.push_str("=================================================\n\n");

    // Group by category
    let categories = [
        "Experiment Tracking",
        "Model Registry",
        "Cost & Energy",
        "Visualization",
        "LLM / GenAI",
        "Deployment",
        "Data Versioning",
        "Academic / Research",
    ];

    for category in categories {
        let cat_mappings: Vec<_> = mappings.iter().filter(|m| m.category == category).collect();

        if cat_mappings.is_empty() {
            continue;
        }

        output.push_str(&format!("{}\n", category.to_uppercase()));

        for (idx, mapping) in cat_mappings.iter().enumerate() {
            let is_last = idx == cat_mappings.len() - 1;
            let prefix = if is_last { "└──" } else { "├──" };

            output.push_str(&format!(
                "{} [{}] {} ← {}\n",
                prefix,
                mapping.integration_type.code(),
                mapping.paiml_component,
                mapping.python_component
            ));
        }
        output.push('\n');
    }

    output.push_str("Legend: [REP]=Replaces (Python eliminated)\n\n");
    output.push_str(&format!(
        "Summary: {} Python components replaced by sovereign Rust alternatives\n",
        mappings.len()
    ));
    output.push_str("         Zero Python dependencies in production\n");

    output
}

/// Format as JSON
pub fn format_json(framework: Option<ExperimentFramework>, integration: bool) -> String {
    if integration {
        let mappings = build_integration_mappings();
        serde_json::to_string_pretty(&mappings).unwrap_or_default()
    } else {
        match framework {
            Some(ExperimentFramework::MLflow) => {
                serde_json::to_string_pretty(&build_mlflow_tree()).unwrap_or_default()
            }
            Some(ExperimentFramework::WandB) => {
                serde_json::to_string_pretty(&build_wandb_tree()).unwrap_or_default()
            }
            Some(ExperimentFramework::Neptune) => {
                serde_json::to_string_pretty(&build_neptune_tree()).unwrap_or_default()
            }
            Some(ExperimentFramework::Dvc) => {
                serde_json::to_string_pretty(&build_dvc_tree()).unwrap_or_default()
            }
            Some(fw @ (ExperimentFramework::CometML | ExperimentFramework::Sacred)) => {
                // Minimal trees for these
                let tree = ExperimentTree {
                    framework: fw.name().to_string(),
                    replacement: fw.replacement().to_string(),
                    categories: vec![],
                };
                serde_json::to_string_pretty(&tree).unwrap_or_default()
            }
            None => {
                let trees = vec![
                    build_mlflow_tree(),
                    build_wandb_tree(),
                    build_neptune_tree(),
                    build_dvc_tree(),
                ];
                serde_json::to_string_pretty(&trees).unwrap_or_default()
            }
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn test_EXP_TREE_001_framework_names() {
        assert_eq!(ExperimentFramework::MLflow.name(), "MLflow");
        assert_eq!(ExperimentFramework::WandB.name(), "Weights & Biases");
        assert_eq!(ExperimentFramework::Neptune.name(), "Neptune.ai");
        assert_eq!(ExperimentFramework::Dvc.name(), "DVC");
    }

    #[test]
    fn test_EXP_TREE_002_framework_replacements() {
        assert_eq!(ExperimentFramework::MLflow.replacement(), "Entrenar + Batuta");
        assert_eq!(ExperimentFramework::WandB.replacement(), "Entrenar + Trueno-Viz");
        assert_eq!(ExperimentFramework::Neptune.replacement(), "Entrenar");
        assert_eq!(ExperimentFramework::Dvc.replacement(), "Batuta + Trueno-DB");
    }

    #[test]
    fn test_EXP_TREE_003_all_frameworks() {
        let all = ExperimentFramework::all();
        assert_eq!(all.len(), 6);
    }

    #[test]
    fn test_EXP_TREE_004_integration_type_code() {
        assert_eq!(IntegrationType::Replaces.code(), "REP");
    }

    #[test]
    fn test_EXP_TREE_005_mlflow_tree_structure() {
        let tree = build_mlflow_tree();
        assert_eq!(tree.framework, "MLflow");
        assert_eq!(tree.replacement, "Entrenar + Batuta");
        assert!(!tree.categories.is_empty());

        // Check experiment tracking category exists
        let exp_tracking = tree.categories.iter().find(|c| c.name == "Experiment Tracking");
        assert!(exp_tracking.is_some());
    }

    #[test]
    fn test_EXP_TREE_006_mlflow_has_genai() {
        let tree = build_mlflow_tree();
        let genai = tree.categories.iter().find(|c| c.name == "GenAI / LLM");
        assert!(genai.is_some());

        let genai = genai.expect("unexpected failure");
        assert!(genai.components.iter().any(|c| c.name == "mlflow.tracing"));
    }

    #[test]
    fn test_EXP_TREE_007_wandb_tree_structure() {
        let tree = build_wandb_tree();
        assert_eq!(tree.framework, "Weights & Biases");
        assert_eq!(tree.replacement, "Entrenar + Trueno-Viz");

        // Check visualization category
        let viz = tree.categories.iter().find(|c| c.name == "Visualization");
        assert!(viz.is_some());
    }

    #[test]
    fn test_EXP_TREE_008_neptune_tree_structure() {
        let tree = build_neptune_tree();
        assert_eq!(tree.framework, "Neptune.ai");
        assert_eq!(tree.replacement, "Entrenar");
    }

    #[test]
    fn test_EXP_TREE_009_dvc_tree_structure() {
        let tree = build_dvc_tree();
        assert_eq!(tree.framework, "DVC");
        assert_eq!(tree.replacement, "Batuta + Trueno-DB");

        // Check data versioning
        let data_ver = tree.categories.iter().find(|c| c.name == "Data Versioning");
        assert!(data_ver.is_some());
    }

    #[test]
    fn test_EXP_TREE_010_integration_mappings_count() {
        let mappings = build_integration_mappings();
        assert!(mappings.len() >= 20);
    }

    #[test]
    fn test_EXP_TREE_011_integration_mappings_categories() {
        let mappings = build_integration_mappings();
        let categories: std::collections::HashSet<_> =
            mappings.iter().map(|m| m.category.as_str()).collect();

        assert!(categories.contains("Experiment Tracking"));
        assert!(categories.contains("Model Registry"));
        assert!(categories.contains("Visualization"));
        assert!(categories.contains("LLM / GenAI"));
    }

    #[test]
    fn test_EXP_TREE_012_format_framework_tree() {
        let tree = build_mlflow_tree();
        let output = format_framework_tree(&tree);

        assert!(output.contains("MLflow (Python) → Entrenar + Batuta (Rust)"));
        assert!(output.contains("Experiment Tracking"));
        assert!(output.contains("mlflow.start_run()"));
    }

    #[test]
    fn test_EXP_TREE_013_format_all_frameworks() {
        let output = format_all_frameworks();

        assert!(output.contains("EXPERIMENT TRACKING FRAMEWORKS ECOSYSTEM"));
        assert!(output.contains("MLflow"));
        assert!(output.contains("Weights & Biases"));
        assert!(output.contains("Neptune.ai"));
        assert!(output.contains("DVC"));
        assert!(output.contains("Summary:"));
    }

    #[test]
    fn test_EXP_TREE_014_format_integration_mappings() {
        let output = format_integration_mappings();

        assert!(output.contains("PAIML REPLACEMENTS"));
        assert!(output.contains("[REP]"));
        assert!(output.contains("Entrenar::ExperimentRun"));
        assert!(output.contains("mlflow.start_run()"));
        assert!(output.contains("Legend:"));
    }

    #[test]
    fn test_EXP_TREE_015_json_output_single() {
        let json = format_json(Some(ExperimentFramework::MLflow), false);
        assert!(json.contains("MLflow"));
        assert!(json.contains("Entrenar"));

        // Verify it's valid JSON
        let parsed: Result<ExperimentTree, _> = serde_json::from_str(&json);
        assert!(parsed.is_ok());
    }

    #[test]
    fn test_EXP_TREE_016_json_output_all() {
        let json = format_json(None, false);

        // Verify it's valid JSON array
        let parsed: Result<Vec<ExperimentTree>, _> = serde_json::from_str(&json);
        assert!(parsed.is_ok());
        assert_eq!(parsed.expect("unexpected failure").len(), 4);
    }

    #[test]
    fn test_EXP_TREE_017_json_output_integration() {
        let json = format_json(None, true);

        // Verify it's valid JSON array
        let parsed: Result<Vec<IntegrationMapping>, _> = serde_json::from_str(&json);
        assert!(parsed.is_ok());
        assert!(parsed.expect("unexpected failure").len() >= 20);
    }

    #[test]
    fn test_EXP_TREE_018_all_mappings_have_replacements() {
        let mappings = build_integration_mappings();
        for mapping in &mappings {
            assert_eq!(mapping.integration_type, IntegrationType::Replaces);
            assert!(!mapping.paiml_component.is_empty());
            assert!(!mapping.python_component.is_empty());
        }
    }

    #[test]
    fn test_EXP_TREE_019_academic_features() {
        let mappings = build_integration_mappings();
        let academic: Vec<_> =
            mappings.iter().filter(|m| m.category == "Academic / Research").collect();

        assert!(academic.len() >= 3);
        assert!(academic.iter().any(|m| m.paiml_component.contains("ResearchArtifact")));
        assert!(academic.iter().any(|m| m.paiml_component.contains("CitationMetadata")));
    }

    #[test]
    fn test_EXP_TREE_020_cost_energy_features() {
        let mappings = build_integration_mappings();
        let cost_energy: Vec<_> =
            mappings.iter().filter(|m| m.category == "Cost & Energy").collect();

        assert!(cost_energy.len() >= 3);
        assert!(cost_energy.iter().any(|m| m.paiml_component.contains("EnergyMetrics")));
        assert!(cost_energy.iter().any(|m| m.paiml_component.contains("CostMetrics")));
    }
}
