#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    dead_code,
    clippy::needless_borrows_for_generic_args,
    clippy::match_single_binding,
    clippy::single_char_pattern
)]
/// Local Workspace Oracle demonstration
/// Discovers and analyzes PAIML projects in ~/src with development state awareness
use batuta::oracle::local_workspace::{
    DevState, LocalProject, LocalWorkspaceOracle, PublishOrder, WorkspaceSummary,
};
use std::collections::HashMap;

const SEPARATOR: &str = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";

fn print_section(num: u32, title: &str) {
    println!("{}", SEPARATOR);
    println!("{}. {}", num, title);
    println!("{}\n", SEPARATOR);
}

fn dev_state_icon(state: &DevState) -> &'static str {
    match state {
        DevState::Clean => "✅",
        DevState::Dirty => "🔧",
        DevState::Unpushed => "📤",
    }
}

fn display_discovered_projects(projects: &HashMap<String, LocalProject>) {
    print_section(1, "DISCOVERING LOCAL PROJECTS");
    println!("📁 Found {} PAIML projects in ~/src:\n", projects.len());

    for project in projects.values() {
        let icon = dev_state_icon(&project.dev_state);
        println!(
            "  {} {} v{} ({:?})",
            icon, project.name, project.local_version, project.dev_state
        );

        let status = &project.git_status;
        if status.modified_count > 0 || status.unpushed_commits > 0 {
            println!(
                "     └─ {} modified, {} ahead of remote",
                status.modified_count, status.unpushed_commits
            );
        }
    }
    println!();
}

fn display_dev_state_legend(projects: &HashMap<String, LocalProject>) {
    print_section(2, "DEVELOPMENT STATE AWARENESS");

    println!("Understanding DevState:");
    println!("  ✅ Clean    - No uncommitted changes, safe to use local version");
    println!("  🔧 Dirty    - Active development, use crates.io version for deps");
    println!("  📤 Unpushed - Clean but has unpushed commits\n");

    let clean = projects.values().filter(|p| p.dev_state == DevState::Clean).count();
    let dirty = projects.values().filter(|p| p.dev_state == DevState::Dirty).count();
    let unpushed = projects.values().filter(|p| p.dev_state == DevState::Unpushed).count();

    println!("📊 Project States:");
    println!("  ✅ Clean:    {}", clean);
    println!("  🔧 Dirty:    {}", dirty);
    println!("  📤 Unpushed: {}", unpushed);
    println!();
}

fn display_dirty_projects(projects: &HashMap<String, LocalProject>) {
    print_section(3, "DIRTY PROJECTS (Active Development)");

    println!("🔧 Projects with uncommitted changes:\n");
    let dirty_projects: Vec<_> =
        projects.values().filter(|p| p.dev_state == DevState::Dirty).collect();

    if dirty_projects.is_empty() {
        println!("  (none - all projects are clean!)");
    } else {
        for project in &dirty_projects {
            println!("  🔧 {}", project.name);
            println!("     {} modified files", project.git_status.modified_count);
            println!("     Local:     v{}", project.local_version);
            if let Some(crates_ver) = &project.published_version {
                println!("     Crates.io: v{} (stable - use this for deps)", crates_ver);
            }
            println!();
        }
    }

    println!("💡 Key Insight: Dirty projects don't block the stack!");
    println!("   The crates.io version is stable and should be used for dependencies.\n");
}

fn display_version_drift(projects: &HashMap<String, LocalProject>) {
    print_section(4, "VERSION DRIFT DETECTION");

    println!("🔍 Comparing local versions vs crates.io:\n");

    for project in projects.values() {
        match &project.published_version {
            Some(published) if project.local_version != *published => {
                let (icon, desc) = if project.local_version > *published {
                    ("📈", "LocalAhead")
                } else {
                    ("📉", "LocalBehind")
                };
                println!("  {} {} ({})", icon, project.name, desc);
                println!("     Local: v{}  →  Crates.io: v{}\n", project.local_version, published);
            }
            None => {
                println!("  🆕 {} (NotPublished)", project.name);
                println!("     Local: v{}\n", project.local_version);
            }
            _ => {} // In sync, skip
        }
    }
}

fn display_publish_order(publish_order: &PublishOrder) {
    print_section(5, "PUBLISH ORDER (Topological Sort)");

    println!("📦 Safe publish order (respects dependencies):\n");
    for (i, step) in publish_order.order.iter().enumerate() {
        let icon = if step.needs_publish { "📤" } else { "✅" };
        println!("  {}. {} {} v{}", i + 1, icon, step.name, step.version);
        for blocker in &step.blocked_by {
            println!("     ⚠️  Blocked by: {}", blocker);
        }
    }

    if !publish_order.cycles.is_empty() {
        println!("\n⚠️  Detected dependency cycles:");
        for cycle in &publish_order.cycles {
            println!("     {}", cycle.join(" → "));
        }
    }
    println!();
}

fn display_workspace_summary(summary: &WorkspaceSummary) {
    print_section(6, "WORKSPACE SUMMARY");

    println!("📊 Workspace Overview:");
    println!("  Total PAIML projects:   {}", summary.total_projects);
    println!("  With uncommitted:       {}", summary.projects_with_changes);
    println!("  With unpushed commits:  {}", summary.projects_with_unpushed);
    println!("  Workspace projects:     {}", summary.workspace_count);
    println!();

    println!("✅ Local Workspace Oracle ready!");
    println!("   Run: batuta oracle --local");
    println!("   Run: batuta oracle --dirty");
    println!("   Run: batuta oracle --publish-order");
}

fn main() -> anyhow::Result<()> {
    println!("🏠 Local Workspace Oracle Demo");
    println!("Discover PAIML projects and their development state\n");

    let mut oracle = LocalWorkspaceOracle::new()?;
    let projects = oracle.discover_projects()?;

    display_discovered_projects(projects);
    display_dev_state_legend(oracle.projects());
    display_dirty_projects(oracle.projects());
    display_version_drift(oracle.projects());
    display_publish_order(&oracle.suggest_publish_order());
    display_workspace_summary(&oracle.summary());

    Ok(())
}
