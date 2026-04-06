//! Stack versions and publish status command implementations.

use crate::ansi_colors::Colorize;
use crate::stack;
use std::path::PathBuf;

use super::StackOutputFormat;

#[derive(Debug, serde::Serialize)]
pub(super) struct CrateVersionInfo {
    name: String,
    latest: String,
    description: Option<String>,
    updated: String,
    downloads: u64,
}

#[derive(Debug, serde::Serialize)]
pub(super) struct VersionReport {
    crates: Vec<CrateVersionInfo>,
    total_checked: usize,
    total_found: usize,
    timestamp: String,
}

pub(super) fn cmd_stack_versions(
    outdated_only: bool,
    format: StackOutputFormat,
    offline: bool,
    _include_prerelease: bool,
) -> anyhow::Result<()> {
    use stack::crates_io::CratesIoClient;
    use stack::PAIML_CRATES;

    if !matches!(format, StackOutputFormat::Json) {
        println!("{}", "📦 PAIML Stack Versions".bright_cyan().bold());
        println!("{}", "═".repeat(60).dimmed());
        if offline {
            println!("{}", "📴 Offline mode - using cached data".yellow());
        }
        println!();
        println!("{}", "Fetching latest versions from crates.io...".dimmed());
        println!();
    }

    let rt = tokio::runtime::Runtime::new()?;

    let mut client = CratesIoClient::new().with_persistent_cache();
    if offline {
        client.set_offline(true);
    }

    let mut crate_infos: Vec<CrateVersionInfo> = Vec::new();
    let mut found_count = 0;

    rt.block_on(async {
        for crate_name in PAIML_CRATES {
            match client.get_crate(crate_name).await {
                Ok(response) => {
                    found_count += 1;
                    crate_infos.push(CrateVersionInfo {
                        name: (*crate_name).to_string(),
                        latest: response.krate.max_version.clone(),
                        description: response.krate.description.clone(),
                        updated: response.krate.updated_at.clone(),
                        downloads: response.krate.downloads,
                    });
                }
                Err(_) => {
                    if !outdated_only && !matches!(format, StackOutputFormat::Json) {
                        // Skip unpublished crates in normal output
                    }
                }
            }
        }
    });

    let report = VersionReport {
        crates: crate_infos,
        total_checked: PAIML_CRATES.len(),
        total_found: found_count,
        timestamp: chrono::Utc::now().to_rfc3339(),
    };

    match format {
        StackOutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        StackOutputFormat::Text | StackOutputFormat::Markdown => {
            display_version_report_text(&report, outdated_only);
        }
    }

    Ok(())
}

fn truncate_description(desc: Option<&String>) -> String {
    desc.map(|d| {
        if d.len() > 35 {
            let mut end = 32;
            while end > 0 && !d.is_char_boundary(end) {
                end -= 1;
            }
            format!("{}...", &d[..end])
        } else {
            d.clone()
        }
    })
    .unwrap_or_else(|| "-".to_string())
}

fn display_version_report_text(report: &VersionReport, outdated_only: bool) {
    use stack::PAIML_CRATES;

    println!(
        "{:<20} {:>12} {:>12} {}",
        "Crate".bright_yellow().bold(),
        "Latest".bright_yellow().bold(),
        "Downloads".bright_yellow().bold(),
        "Description".bright_yellow().bold()
    );
    println!("{}", "─".repeat(80).dimmed());

    for info in &report.crates {
        println!(
            "{:<20} {:>12} {:>12} {}",
            info.name.cyan(),
            info.latest.green(),
            format_downloads(info.downloads),
            truncate_description(info.description.as_ref()).dimmed()
        );
    }

    println!("{}", "─".repeat(80).dimmed());
    println!();
    println!(
        "📊 Found {} of {} PAIML crates on crates.io",
        report.total_found.to_string().green(),
        report.total_checked
    );

    let unpublished: Vec<_> = PAIML_CRATES
        .iter()
        .filter(|name| !report.crates.iter().any(|c| c.name == **name))
        .collect();

    if !unpublished.is_empty() && !outdated_only {
        println!();
        println!("{}", "📝 Not yet published:".dimmed());
        for name in unpublished {
            println!("   {} {}", "•".dimmed(), name.dimmed());
        }
    }

    println!();
    println!("💡 Tip: Use {} to update dependencies", "cargo update".cyan());
}

/// Format download count for display (e.g., 1.2M, 45K)
pub(super) fn format_downloads(downloads: u64) -> String {
    if downloads >= 1_000_000 {
        format!("{:.1}M", downloads as f64 / 1_000_000.0)
    } else if downloads >= 1_000 {
        format!("{:.1}K", downloads as f64 / 1_000.0)
    } else {
        downloads.to_string()
    }
}

pub(super) fn cmd_stack_publish_status(
    format: StackOutputFormat,
    workspace: Option<PathBuf>,
    clear_cache: bool,
) -> anyhow::Result<()> {
    use stack::publish_status::{format_report_json, PublishStatusCache, PublishStatusScanner};

    let workspace_path = workspace.unwrap_or_else(|| {
        std::env::current_dir()
            .expect("Failed to get current directory")
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from(".."))
    });

    if !matches!(format, StackOutputFormat::Json) {
        println!("{}", "📦 PAIML Stack Publish Status".bright_cyan().bold());
        println!("{}", "═".repeat(65).dimmed());
        if clear_cache {
            println!("{}", "🗑️  Clearing cache...".yellow());
        }
        println!();
    }

    if clear_cache {
        let mut cache = PublishStatusCache::load();
        cache.clear();
        let _ = cache.save();
    }

    let mut scanner = PublishStatusScanner::new(workspace_path).with_crates_io();
    let report = scanner.scan_sync()?;

    match format {
        StackOutputFormat::Json => {
            println!("{}", format_report_json(&report)?);
        }
        StackOutputFormat::Text | StackOutputFormat::Markdown => {
            println!(
                "{:<20} {:>10} {:>10} {:>10} {:>12}",
                "Crate".bright_yellow().bold(),
                "Local".bright_yellow().bold(),
                "crates.io".bright_yellow().bold(),
                "Git".bright_yellow().bold(),
                "Action".bright_yellow().bold()
            );
            println!("{}", "─".repeat(65).dimmed());

            for status in &report.crates {
                let local = status.local_version.as_deref().unwrap_or("-");
                let remote = status.crates_io_version.as_deref().unwrap_or("-");
                let git = status.git_status.summary();

                let action_colored = match status.action {
                    stack::PublishAction::UpToDate => "✓ up to date".green(),
                    stack::PublishAction::NeedsPublish => "📦 PUBLISH".bright_red().bold(),
                    stack::PublishAction::NeedsCommit => "📝 commit".yellow(),
                    stack::PublishAction::LocalBehind => "⚠️  behind".yellow(),
                    stack::PublishAction::NotPublished => "🆕 new".cyan(),
                    stack::PublishAction::Error => "❌ error".red(),
                };

                println!(
                    "{:<20} {:>10} {:>10} {:>10} {}",
                    status.name.cyan(),
                    local,
                    remote,
                    git.dimmed(),
                    action_colored
                );
            }

            println!("{}", "─".repeat(65).dimmed());
            println!();

            println!(
                "📊 {} crates: {} {}, {} {}, {} up-to-date",
                report.total,
                report.needs_publish.to_string().bright_red().bold(),
                "publish".red(),
                report.needs_commit.to_string().yellow(),
                "commit".yellow(),
                report.up_to_date.to_string().green()
            );
            println!(
                "⚡ {}ms (cache: {} hits, {} misses)",
                report.elapsed_ms,
                report.cache_hits.to_string().green(),
                report.cache_misses
            );
        }
    }

    Ok(())
}
