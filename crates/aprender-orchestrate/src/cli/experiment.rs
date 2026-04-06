//! Experiment tracking command implementations
//!
//! This module contains all experiment tracking-related CLI commands extracted from main.rs.

#![cfg(feature = "native")]

use crate::experiment;

/// Experiment tracking subcommands
#[derive(Debug, Clone, clap::Subcommand)]
pub enum ExperimentCommand {
    /// Display experiment tracking frameworks ecosystem tree
    Tree {
        /// Filter by framework (mlflow, wandb, neptune, dvc)
        #[arg(long)]
        framework: Option<String>,

        /// Show PAIML replacement mappings
        #[arg(long)]
        integration: bool,

        /// Output format (ascii, json)
        #[arg(long, default_value = "ascii")]
        format: String,
    },
}

/// Main experiment command dispatcher
pub fn cmd_experiment(command: ExperimentCommand) -> anyhow::Result<()> {
    match command {
        ExperimentCommand::Tree { framework, integration, format } => {
            cmd_experiment_tree(framework.as_deref(), integration, &format)?;
        }
    }
    Ok(())
}

fn cmd_experiment_tree(
    framework: Option<&str>,
    integration: bool,
    format: &str,
) -> anyhow::Result<()> {
    use experiment::tree::{
        build_dvc_tree, build_integration_mappings, build_mlflow_tree, build_neptune_tree,
        build_wandb_tree, format_all_frameworks, format_framework_tree,
        format_integration_mappings,
    };

    if integration {
        // Show PAIML replacement mappings
        let output = match format {
            "json" => {
                let mappings = build_integration_mappings();
                serde_json::to_string_pretty(&mappings)?
            }
            _ => format_integration_mappings(),
        };
        println!("{}", output);
    } else if let Some(framework_name) = framework {
        // Show specific framework tree
        let fw = framework_name.to_lowercase();
        let tree = match fw.as_str() {
            "mlflow" => build_mlflow_tree(),
            "wandb" => build_wandb_tree(),
            "neptune" => build_neptune_tree(),
            "dvc" => build_dvc_tree(),
            _ => {
                anyhow::bail!(
                    "Unknown framework: {}. Valid options: mlflow, wandb, neptune, dvc",
                    framework_name
                );
            }
        };
        let output = match format {
            "json" => serde_json::to_string_pretty(&tree)?,
            _ => format_framework_tree(&tree),
        };
        println!("{}", output);
    } else {
        // Show all frameworks
        let output = match format {
            "json" => {
                let trees = vec![
                    build_mlflow_tree(),
                    build_wandb_tree(),
                    build_neptune_tree(),
                    build_dvc_tree(),
                ];
                serde_json::to_string_pretty(&trees)?
            }
            _ => format_all_frameworks(),
        };
        println!("{}", output);
    }

    Ok(())
}
