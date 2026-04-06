//! Stack command implementations
//!
//! This module contains all stack-related CLI commands extracted from main.rs.

#![cfg(feature = "native")]

use crate::ansi_colors::Colorize;
use crate::stack;
use std::path::PathBuf;

#[path = "stack_check.rs"]
mod check;
#[path = "stack_drift.rs"]
mod drift;
#[path = "stack_status.rs"]
mod status;
#[path = "stack_versions.rs"]
mod versions;

use check::{cmd_stack_check, cmd_stack_release};
use drift::{cmd_stack_comply, cmd_stack_drift, extract_minor_version};
use status::{cmd_stack_gate, cmd_stack_quality, cmd_stack_status, cmd_stack_sync, cmd_stack_tree};
use versions::{cmd_stack_publish_status, cmd_stack_versions, format_downloads};

/// Stack output format
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum)]
pub enum StackOutputFormat {
    #[default]
    Text,
    Json,
    Markdown,
}

/// Version bump type
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum BumpType {
    Patch,
    Minor,
    Major,
}

/// Stack subcommand
#[derive(Debug, Clone, clap::Subcommand)]
pub enum StackCommand {
    /// Check dependency health across the stack
    Check {
        /// Specific project to check (default: scan workspace)
        #[arg(long)]
        project: Option<String>,
        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: StackOutputFormat,
        /// Strict mode (fail on warnings)
        #[arg(long)]
        strict: bool,
        /// Verify against crates.io
        #[arg(long)]
        verify_published: bool,
        /// Offline mode (use cached data)
        #[arg(long)]
        offline: bool,
        /// Workspace directory
        #[arg(long)]
        workspace: Option<PathBuf>,
    },
    /// Prepare a stack release
    Release {
        /// Crate to release
        crate_name: Option<String>,
        /// Release all crates
        #[arg(long)]
        all: bool,
        /// Dry run (show what would happen)
        #[arg(long)]
        dry_run: bool,
        /// Version bump type
        #[arg(long, value_enum)]
        bump: Option<BumpType>,
        /// Skip verification
        #[arg(long)]
        no_verify: bool,
        /// Skip confirmation
        #[arg(long, short)]
        yes: bool,
        /// Publish to crates.io
        #[arg(long)]
        publish: bool,
    },
    /// Show stack status
    Status {
        /// Simple text output (no TUI)
        #[arg(long)]
        simple: bool,
        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: StackOutputFormat,
        /// Show dependency tree
        #[arg(long)]
        tree: bool,
    },
    /// Sync stack dependencies
    Sync {
        /// Crate to sync
        crate_name: Option<String>,
        /// Sync all crates
        #[arg(long)]
        all: bool,
        /// Dry run
        #[arg(long)]
        dry_run: bool,
        /// Align to specific dependency version
        #[arg(long)]
        align: Option<String>,
    },
    /// Show dependency tree
    Tree {
        /// Output format (ascii, json, dot)
        #[arg(long, default_value = "ascii")]
        format: String,
        /// Include health indicators
        #[arg(long)]
        health: bool,
        /// Filter by layer
        #[arg(long)]
        filter: Option<String>,
    },
    /// Check code quality across stack
    Quality {
        /// Specific component to check
        #[arg(long)]
        component: Option<String>,
        /// Strict mode (fail if not A+)
        #[arg(long)]
        strict: bool,
        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: StackOutputFormat,
        /// Verify hero implementation matching
        #[arg(long)]
        verify_hero: bool,
        /// Verbose output
        #[arg(long, short)]
        verbose: bool,
        /// Workspace directory
        #[arg(long)]
        workspace: Option<PathBuf>,
    },
    /// Quality gate check for CI/pre-commit
    Gate {
        /// Workspace directory
        #[arg(long)]
        workspace: Option<PathBuf>,
        /// Quiet mode (minimal output)
        #[arg(long, short)]
        quiet: bool,
    },
    /// Check latest versions of PAIML stack crates
    Versions {
        /// Show only outdated crates
        #[arg(long)]
        outdated: bool,
        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: StackOutputFormat,
        /// Offline mode (use cached data only)
        #[arg(long)]
        offline: bool,
        /// Include pre-release versions
        #[arg(long)]
        include_prerelease: bool,
    },
    /// Check publish status of stack crates
    PublishStatus {
        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: StackOutputFormat,
        /// Workspace directory
        #[arg(long)]
        workspace: Option<PathBuf>,
        /// Clear cache and refresh
        #[arg(long)]
        clear_cache: bool,
    },
    /// Detect version drift in stack dependencies
    Drift {
        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: StackOutputFormat,
        /// Generate fix commands
        #[arg(long)]
        fix: bool,
        /// Workspace directory
        #[arg(long)]
        workspace: Option<PathBuf>,
    },
    /// Check cross-project compliance (Makefiles, Cargo.toml, CI)
    Comply {
        /// Specific rule to check
        #[arg(long)]
        rule: Option<String>,
        /// Attempt to auto-fix violations
        #[arg(long)]
        fix: bool,
        /// Dry run (show what would be fixed)
        #[arg(long)]
        dry_run: bool,
        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: ComplyOutputFormat,
        /// Workspace directory
        #[arg(long)]
        workspace: Option<PathBuf>,
        /// List available rules
        #[arg(long)]
        list_rules: bool,
    },
}

/// Comply output format
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum)]
pub enum ComplyOutputFormat {
    #[default]
    Text,
    Json,
    Markdown,
    Html,
}

/// Execute read-only stack commands (check, status, tree, versions).
fn dispatch_stack_info(command: StackCommand) -> anyhow::Result<()> {
    match command {
        StackCommand::Check { project, format, strict, verify_published, offline, workspace } => {
            cmd_stack_check(project, format, strict, verify_published, offline, workspace)
        }
        StackCommand::Status { simple, format, tree } => cmd_stack_status(simple, format, tree),
        StackCommand::Tree { format, health, filter } => {
            cmd_stack_tree(&format, health, filter.as_deref())
        }
        StackCommand::Versions { outdated, format, offline, include_prerelease } => {
            cmd_stack_versions(outdated, format, offline, include_prerelease)
        }
        _ => unreachable!(),
    }
}

/// Execute mutating/quality stack commands.
fn dispatch_stack_action(command: StackCommand) -> anyhow::Result<()> {
    match command {
        StackCommand::Release { crate_name, all, dry_run, bump, no_verify, yes, publish } => {
            cmd_stack_release(crate_name, all, dry_run, bump, no_verify, yes, publish)
        }
        StackCommand::Sync { crate_name, all, dry_run, align } => {
            cmd_stack_sync(crate_name, all, dry_run, align)
        }
        StackCommand::Quality { component, strict, format, workspace, .. } => {
            cmd_stack_quality(component, strict, format, workspace)
        }
        StackCommand::Gate { workspace, quiet } => cmd_stack_gate(workspace, quiet),
        StackCommand::PublishStatus { format, workspace, clear_cache } => {
            cmd_stack_publish_status(format, workspace, clear_cache)
        }
        StackCommand::Drift { format, fix, workspace } => cmd_stack_drift(format, fix, workspace),
        StackCommand::Comply { rule, fix, dry_run, format, workspace, list_rules } => {
            cmd_stack_comply(rule, fix, dry_run, format, workspace, list_rules)
        }
        StackCommand::Check { .. }
        | StackCommand::Status { .. }
        | StackCommand::Tree { .. }
        | StackCommand::Versions { .. } => unreachable!(),
    }
}

/// Main stack command dispatcher
pub fn cmd_stack(command: StackCommand) -> anyhow::Result<()> {
    match &command {
        StackCommand::Check { .. }
        | StackCommand::Status { .. }
        | StackCommand::Tree { .. }
        | StackCommand::Versions { .. } => dispatch_stack_info(command),
        _ => dispatch_stack_action(command),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_minor_version_simple() {
        assert_eq!(extract_minor_version("0.10.1"), "0.10");
        assert_eq!(extract_minor_version("1.2.3"), "1.2");
    }

    #[test]
    fn test_extract_minor_version_with_prefix() {
        assert_eq!(extract_minor_version("^0.10"), "0.10");
        assert_eq!(extract_minor_version("~1.2"), "1.2");
        assert_eq!(extract_minor_version("=2.0.0"), "2.0");
    }

    #[test]
    fn test_format_downloads() {
        assert_eq!(format_downloads(500), "500");
        assert_eq!(format_downloads(1500), "1.5K");
        assert_eq!(format_downloads(1_500_000), "1.5M");
    }
}
