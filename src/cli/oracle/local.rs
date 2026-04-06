//! Local workspace oracle commands
//!
//! This module contains local workspace discovery and multi-project intelligence
//! for managing PAIML stack projects.

use crate::ansi_colors::Colorize;
use crate::oracle;

use super::types::OracleOutputFormat;

// ============================================================================
// Helper Functions
// ============================================================================

fn local_print_summary(summary: &oracle::local_workspace::WorkspaceSummary) {
    println!(
        "{}: {} PAIML projects discovered in ~/src",
        "Summary".bright_yellow(),
        summary.total_projects
    );
    println!(
        "  {} {} (use crates.io versions for deps)",
        summary.projects_with_changes.to_string().bright_red(),
        "dirty".bright_red()
    );
    println!(
        "  {} {} (ready to push)",
        summary.projects_with_unpushed.to_string().bright_yellow(),
        "unpushed".bright_yellow()
    );
    println!(
        "  {} {} (using local versions)",
        (summary.total_projects - summary.projects_with_changes - summary.projects_with_unpushed)
            .to_string()
            .bright_green(),
        "clean".bright_green()
    );
    println!();
}

fn local_show_dirty_summary(projects: &[&oracle::local_workspace::LocalProject]) {
    use oracle::local_workspace::DevState;

    let dirty: Vec<_> = projects.iter().filter(|p| p.dev_state == DevState::Dirty).collect();

    println!("{} {} projects with uncommitted changes:", "*".bright_red(), dirty.len());
    println!();
    for project in &dirty {
        println!(
            "  {} {} ({} files)",
            "*".bright_red(),
            project.name.bright_white().bold(),
            project.git_status.modified_count
        );
    }
    println!();
    println!(
        "{}",
        "These projects are in active development - stack uses crates.io versions for deps"
            .dimmed()
    );
}

fn local_show_project_details(project: &oracle::local_workspace::LocalProject) {
    use oracle::local_workspace::DevState;

    let (status_icon, state_label) = match project.dev_state {
        DevState::Dirty => ("*".bright_red(), "DIRTY".bright_red()),
        DevState::Unpushed => ("o".bright_yellow(), "UNPUSHED".bright_yellow()),
        DevState::Clean => ("o".bright_green(), "clean".bright_green()),
    };

    let version_info = match &project.published_version {
        Some(pub_v) if pub_v == &project.local_version => {
            format!("v{}", project.local_version).bright_green().to_string()
        }
        Some(pub_v) => {
            format!("v{} -> v{}", pub_v, project.local_version).bright_yellow().to_string()
        }
        None => format!("v{} (unpublished)", project.local_version).dimmed().to_string(),
    };

    println!(
        "  {} {} {} [{}] {}",
        status_icon,
        project.name.bright_white().bold(),
        version_info,
        project.git_status.branch.dimmed(),
        state_label
    );

    if project.git_status.has_changes {
        println!(
            "      {} modified files",
            project.git_status.modified_count.to_string().bright_red()
        );
    }
    if project.git_status.unpushed_commits > 0 {
        println!(
            "      {} unpushed commits",
            project.git_status.unpushed_commits.to_string().bright_yellow()
        );
    }
    if !project.paiml_dependencies.is_empty() {
        let deps: Vec<_> = project
            .paiml_dependencies
            .iter()
            .map(|d| {
                if d.is_path_dep {
                    format!("{}(path)", d.name)
                } else {
                    format!("{}@{}", d.name, d.required_version)
                }
            })
            .collect();
        println!("      deps: {}", deps.join(", ").dimmed());
    }
}

fn local_show_drift(drifts: &[oracle::local_workspace::VersionDrift]) {
    use oracle::local_workspace::DriftType;

    if drifts.is_empty() {
        return;
    }

    println!("{}", "Version Drift".bright_cyan().bold());
    println!("{}", "─".repeat(50).dimmed());

    for drift in drifts {
        let icon = match drift.drift_type {
            DriftType::LocalAhead => "^".bright_green(),
            DriftType::LocalBehind => "v".bright_red(),
            DriftType::NotPublished => "o".dimmed(),
            DriftType::InSync => "+".bright_green(),
        };
        let msg = match drift.drift_type {
            DriftType::LocalAhead => "ready to publish",
            DriftType::LocalBehind => "needs update",
            DriftType::NotPublished => "not published",
            DriftType::InSync => "in sync",
        };
        println!(
            "  {} {} {} -> {} ({})",
            icon,
            drift.name.bright_white(),
            drift.published_version.dimmed(),
            drift.local_version.bright_yellow(),
            msg.dimmed()
        );
    }
    println!();
}

fn local_show_publish_order_text(order: &oracle::local_workspace::PublishOrder) {
    println!("{}", "Suggested Publish Order".bright_cyan().bold());
    println!("{}", "─".repeat(50).dimmed());
    println!();

    if !order.cycles.is_empty() {
        println!("{} Dependency cycles detected:", "!".bright_yellow());
        for cycle in &order.cycles {
            println!("  {}", cycle.join(" -> ").bright_red());
        }
        println!();
    }

    let needs_publish: Vec<_> = order.order.iter().filter(|s| s.needs_publish).collect();

    if needs_publish.is_empty() {
        println!("{}", "All projects are up to date with crates.io".bright_green());
    } else {
        println!("Crates to publish ({} total):", needs_publish.len().to_string().bright_yellow());
        println!();

        for (i, step) in needs_publish.iter().enumerate() {
            let blocked = if step.blocked_by.is_empty() {
                "ready".bright_green().to_string()
            } else {
                format!("after: {}", step.blocked_by.join(", ")).dimmed().to_string()
            };

            println!(
                "  {}. {} v{} ({})",
                (i + 1).to_string().bright_cyan(),
                step.name.bright_white().bold(),
                step.version.bright_yellow(),
                blocked
            );
        }
    }
    println!();
}

fn local_show_usage() {
    println!("{}", "Usage:".bright_yellow());
    println!("  {} {}", "batuta oracle --local".cyan(), "# Show all local PAIML projects".dimmed());
    println!(
        "  {} {}",
        "batuta oracle --publish-order".cyan(),
        "# Show suggested publish order".dimmed()
    );
    println!("  {} {}", "batuta oracle --local --publish-order".cyan(), "# Show both".dimmed());
}

// ============================================================================
// Public Commands
// ============================================================================

/// Local workspace discovery and multi-project intelligence
pub fn cmd_oracle_local(
    show_status: bool,
    show_dirty: bool,
    show_publish_order: bool,
    format: OracleOutputFormat,
) -> anyhow::Result<()> {
    use oracle::local_workspace::{DevState, LocalWorkspaceOracle};

    // GH-47: Status headers go to stderr to avoid contaminating --format json output
    eprintln!("{}", "Local Workspace Oracle".bright_cyan().bold());
    eprintln!("{}", "─".repeat(50).dimmed());
    eprintln!();

    let mut oracle_ws = LocalWorkspaceOracle::new()?;
    oracle_ws.discover_projects()?;

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(oracle_ws.fetch_published_versions())?;

    let summary = oracle_ws.summary();
    let projects = oracle_ws.projects();

    local_print_summary(&summary);

    let filtered_projects: Vec<_> = if show_dirty {
        projects.values().filter(|p| p.dev_state == DevState::Dirty).collect()
    } else {
        projects.values().collect()
    };

    if show_dirty && !show_status {
        local_show_dirty_summary(&filtered_projects);
        return Ok(());
    }

    if show_status {
        match format {
            OracleOutputFormat::Code | OracleOutputFormat::CodeSvg => {
                eprintln!("No code available for workspace status (try --format text)");
                std::process::exit(1);
            }
            OracleOutputFormat::Json => {
                let output = serde_json::json!({
                    "summary": summary,
                    "projects": projects,
                    "drift": oracle_ws.detect_drift(),
                });
                println!("{}", serde_json::to_string_pretty(&output)?);
            }
            OracleOutputFormat::Markdown | OracleOutputFormat::Text => {
                println!("{}", "Projects".bright_cyan().bold());
                println!("{}", "─".repeat(50).dimmed());

                let mut sorted_projects = filtered_projects.clone();
                sorted_projects.sort_by(|a, b| a.name.cmp(&b.name));

                for project in sorted_projects {
                    local_show_project_details(project);
                }
                println!();

                local_show_drift(&oracle_ws.detect_drift());
            }
        }
    }

    if show_publish_order {
        let order = oracle_ws.suggest_publish_order();

        match format {
            OracleOutputFormat::Code | OracleOutputFormat::CodeSvg => {
                eprintln!("No code available for publish order (try --format text)");
                std::process::exit(1);
            }
            OracleOutputFormat::Json => {
                println!("{}", serde_json::to_string_pretty(&order)?);
            }
            OracleOutputFormat::Markdown | OracleOutputFormat::Text => {
                local_show_publish_order_text(&order);
            }
        }
    }

    if !show_status && !show_publish_order {
        local_show_usage();
    }

    Ok(())
}
