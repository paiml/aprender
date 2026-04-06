//! Stack check and release command implementations.

use crate::ansi_colors::Colorize;
use crate::stack;
use std::path::PathBuf;

use super::{BumpType, StackOutputFormat};

pub(super) fn cmd_stack_check(
    _project: Option<String>,
    format: StackOutputFormat,
    strict: bool,
    verify_published: bool,
    offline: bool,
    workspace: Option<PathBuf>,
) -> anyhow::Result<()> {
    use stack::checker::{
        format_report_json, format_report_markdown, format_report_text, StackChecker,
    };
    use stack::crates_io::CratesIoClient;

    println!("{}", "🔍 PAIML Stack Health Check".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    if offline {
        println!("{}", "📴 Offline mode - using cached data".yellow());
    }
    println!();

    let workspace_path = workspace.unwrap_or_else(|| PathBuf::from("."));

    let mut checker = StackChecker::from_workspace(&workspace_path)?
        .verify_published(verify_published && !offline)
        .strict(strict);

    let rt = tokio::runtime::Runtime::new()?;

    let report = rt.block_on(async {
        let mut client = CratesIoClient::new().with_persistent_cache();
        if offline {
            client.set_offline(true);
        }
        checker.check(&mut client).await
    })?;

    let output = match format {
        StackOutputFormat::Text => format_report_text(&report),
        StackOutputFormat::Json => format_report_json(&report)?,
        StackOutputFormat::Markdown => format_report_markdown(&report),
    };

    println!("{}", output);

    if strict && !report.is_healthy() {
        std::process::exit(1);
    }

    Ok(())
}

/// Run quality gate check before release, returns error if blocked
pub(super) fn release_quality_gate() -> anyhow::Result<()> {
    use stack::{tree::LAYER_DEFINITIONS, QualityChecker, StackQualityReport};

    println!("{}", "🔒 Running quality gate check...".dimmed());

    let workspace_path = std::env::current_dir()?
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));

    let rt = tokio::runtime::Runtime::new()?;
    let mut components = Vec::new();

    for (_layer_name, layer_components) in LAYER_DEFINITIONS.iter() {
        for comp_name in *layer_components {
            let comp_path = workspace_path.join(comp_name);
            if comp_path.join("Cargo.toml").exists() {
                let checker = QualityChecker::new(comp_path);
                if let Ok(quality) = rt.block_on(async { checker.check_component(comp_name).await })
                {
                    components.push(quality);
                }
            }
        }
    }

    let report = StackQualityReport::from_components(components);

    if !report.release_ready {
        println!();
        println!("{}", "❌ RELEASE BLOCKED - Quality gate failed".bright_red().bold());
        println!();
        println!(
            "The following {} component(s) are below A- threshold (SQI < 85):",
            report.blocked_components.len()
        );
        for comp in &report.blocked_components {
            println!("  • {}", comp.bright_yellow());
        }
        println!();
        println!(
            "Fix quality issues before releasing, or use --no-verify to skip (not recommended)."
        );
        anyhow::bail!(
            "Release blocked: {} component(s) below A- threshold",
            report.blocked_components.len()
        );
    }

    println!("{}", "✅ Quality gate passed".bright_green());
    println!();
    Ok(())
}

pub(super) fn cmd_stack_release(
    crate_name: Option<String>,
    all: bool,
    dry_run: bool,
    bump: Option<BumpType>,
    no_verify: bool,
    _yes: bool,
    publish: bool,
) -> anyhow::Result<()> {
    use stack::checker::StackChecker;
    use stack::releaser::{format_plan_text, ReleaseConfig, ReleaseOrchestrator};

    println!("{}", "📦 PAIML Stack Release".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    if !all && crate_name.is_none() {
        println!("{}", "❌ Specify a crate name or use --all".red());
        return Ok(());
    }

    if !no_verify {
        release_quality_gate()?;
    } else {
        println!("{}", "⚠️  SKIPPING quality gate check (--no-verify)".yellow());
        println!();
    }

    if dry_run {
        println!("{}", "⚠️  DRY RUN - No changes will be made".yellow().bold());
        println!();
    }

    let workspace_path = PathBuf::from(".");

    let bump_type = bump.map(|b| match b {
        BumpType::Patch => stack::releaser::BumpType::Patch,
        BumpType::Minor => stack::releaser::BumpType::Minor,
        BumpType::Major => stack::releaser::BumpType::Major,
    });

    let config = ReleaseConfig { bump_type, no_verify, dry_run, publish, ..Default::default() };

    let checker = StackChecker::from_workspace(&workspace_path)?;
    let mut orchestrator = ReleaseOrchestrator::new(checker, config);

    let plan = if all {
        orchestrator.plan_all_releases()?
    } else {
        let name = crate_name.expect("crate_name validated at function start");
        orchestrator.plan_release(&name)?
    };

    let plan_text = format_plan_text(&plan);
    println!("{}", plan_text);

    if dry_run {
        println!();
        println!("{}", "Dry run complete. Use without --dry-run to execute.".dimmed());
        return Ok(());
    }

    println!("Executing release plan...");
    println!();

    let result = orchestrator.execute(&plan)?;
    print_release_result(&result);

    Ok(())
}

fn print_release_result(result: &stack::releaser::ReleaseResult) {
    if !result.success {
        return;
    }
    println!("{} {}", "✓".bright_green(), result.message.bright_green());
    for crate_info in &result.released_crates {
        let publish_status = if crate_info.published {
            "published to crates.io".bright_green().to_string()
        } else {
            "version bumped (not published)".yellow().to_string()
        };
        println!(
            "  {} {} v{} - {}",
            "•".dimmed(),
            crate_info.name.cyan(),
            crate_info.version,
            publish_status
        );
    }
}
