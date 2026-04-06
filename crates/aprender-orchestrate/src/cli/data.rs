//! Data platforms command implementations
//!
//! This module contains all data platform-related CLI commands extracted from main.rs.

#![cfg(feature = "native")]

use crate::data;

/// Data Platforms subcommands
#[derive(Debug, Clone, clap::Subcommand)]
pub enum DataCommand {
    /// Display data platforms ecosystem tree
    Tree {
        /// Filter by platform (databricks, snowflake, aws, huggingface)
        #[arg(long)]
        platform: Option<String>,

        /// Show PAIML integration mappings
        #[arg(long)]
        integration: bool,

        /// Output format (ascii, json)
        #[arg(long, default_value = "ascii")]
        format: String,
    },
}

/// Main data command dispatcher
pub fn cmd_data(command: DataCommand) -> anyhow::Result<()> {
    match command {
        DataCommand::Tree { platform, integration, format } => {
            cmd_data_tree(platform.as_deref(), integration, &format)?;
        }
    }
    Ok(())
}

fn cmd_data_tree(platform: Option<&str>, integration: bool, format: &str) -> anyhow::Result<()> {
    use data::tree::{
        build_aws_tree, build_databricks_tree, build_huggingface_tree, build_integration_mappings,
        build_snowflake_tree, format_all_platforms, format_integration_mappings,
        format_platform_tree,
    };

    if integration {
        // Show PAIML integration mappings
        let output = match format {
            "json" => {
                let mappings = build_integration_mappings();
                serde_json::to_string_pretty(&mappings)?
            }
            _ => format_integration_mappings(),
        };
        println!("{}", output);
    } else if let Some(platform_name) = platform {
        // Show specific platform tree
        let platform = platform_name.to_lowercase();
        let tree = match platform.as_str() {
            "databricks" => build_databricks_tree(),
            "snowflake" => build_snowflake_tree(),
            "aws" => build_aws_tree(),
            "huggingface" | "hf" => build_huggingface_tree(),
            _ => {
                anyhow::bail!(
                    "Unknown platform: {}. Valid options: databricks, snowflake, aws, huggingface",
                    platform_name
                );
            }
        };
        let output = match format {
            "json" => serde_json::to_string_pretty(&tree)?,
            _ => format_platform_tree(&tree),
        };
        println!("{}", output);
    } else {
        // Show all platforms
        let output = match format {
            "json" => {
                let trees = vec![
                    build_databricks_tree(),
                    build_snowflake_tree(),
                    build_aws_tree(),
                    build_huggingface_tree(),
                ];
                serde_json::to_string_pretty(&trees)?
            }
            _ => format_all_platforms(),
        };
        println!("{}", output);
    }

    Ok(())
}
