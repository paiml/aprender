//! Publish Status Demo
//!
//! Demonstrates the O(1) cached publish status scanner for the PAIML stack.
//!
//! ## Features
//!
//! - **O(1) Cache**: Hash-based invalidation for instant repeated queries
//! - **Content-Addressable Keys**: Cache invalidates on Cargo.toml or git changes
//! - **Parallel Fetches**: Cold cache fetches crates.io versions in parallel
//! - **TTL**: crates.io data refreshes after 15 minutes
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example publish_status_demo --features native
//! ```
//!
//! ## Performance
//!
//! - Cold cache: ~7s (parallel crates.io fetches)
//! - Warm cache: <100ms (O(1) hash checks)

#[cfg(feature = "native")]
use batuta::stack::publish_status::{
    determine_action, format_report_text, CrateStatus, GitStatus, PublishAction,
    PublishStatusCache, PublishStatusReport, PublishStatusScanner,
};
#[cfg(feature = "native")]
use std::path::PathBuf;
#[cfg(feature = "native")]
use std::time::Instant;

#[cfg(feature = "native")]
fn main() -> anyhow::Result<()> {
    println!("===============================================================");
    println!("     Publish Status Scanner - O(1) Cache Demo");
    println!("===============================================================\n");

    // =========================================================================
    // Phase 1: Understanding PublishAction
    // =========================================================================
    println!("+-------------------------------------------------------------+");
    println!("| Phase 1: Publish Actions                                    |");
    println!("+-------------------------------------------------------------+\n");

    demo_publish_actions();

    // =========================================================================
    // Phase 2: GitStatus Parsing
    // =========================================================================
    println!("\n+-------------------------------------------------------------+");
    println!("| Phase 2: Git Status Parsing                                 |");
    println!("+-------------------------------------------------------------+\n");

    demo_git_status();

    // =========================================================================
    // Phase 3: Action Determination Logic
    // =========================================================================
    println!("\n+-------------------------------------------------------------+");
    println!("| Phase 3: Action Determination                               |");
    println!("+-------------------------------------------------------------+\n");

    demo_action_determination();

    // =========================================================================
    // Phase 4: Cache Performance
    // =========================================================================
    println!("\n+-------------------------------------------------------------+");
    println!("| Phase 4: Cache Performance                                  |");
    println!("+-------------------------------------------------------------+\n");

    demo_cache_performance();

    // =========================================================================
    // Phase 5: Report Generation
    // =========================================================================
    println!("\n+-------------------------------------------------------------+");
    println!("| Phase 5: Report Generation                                  |");
    println!("+-------------------------------------------------------------+\n");

    demo_report_generation();

    // =========================================================================
    // Phase 6: Live Scan (if in workspace)
    // =========================================================================
    println!("\n+-------------------------------------------------------------+");
    println!("| Phase 6: Live Workspace Scan                                |");
    println!("+-------------------------------------------------------------+\n");

    demo_live_scan()?;

    println!("\n===============================================================");
    println!("     Publish Status demo completed!");
    println!("===============================================================\n");

    Ok(())
}

#[cfg(feature = "native")]
fn demo_publish_actions() {
    println!("  Available Actions:");
    println!();

    let actions = [
        PublishAction::UpToDate,
        PublishAction::NeedsCommit,
        PublishAction::NeedsPublish,
        PublishAction::LocalBehind,
        PublishAction::NotPublished,
        PublishAction::Error,
    ];

    println!("  {:<15} {:<8} {:<20}", "Action", "Symbol", "Description");
    println!("  {}", "-".repeat(45));

    for action in actions {
        println!(
            "  {:<15} {:<8} {:<20}",
            format!("{:?}", action),
            action.symbol(),
            action.description()
        );
    }
}

#[cfg(feature = "native")]
fn demo_git_status() {
    println!("  Git Status Parsing Examples:");
    println!();

    let examples = [
        (
            "Clean repo",
            GitStatus {
                modified: 0,
                untracked: 0,
                staged: 0,
                head_sha: "abc123f".to_string(),
                is_clean: true,
            },
        ),
        (
            "Modified files",
            GitStatus {
                modified: 5,
                untracked: 0,
                staged: 0,
                head_sha: "def456a".to_string(),
                is_clean: false,
            },
        ),
        (
            "Mixed changes",
            GitStatus {
                modified: 3,
                untracked: 2,
                staged: 1,
                head_sha: "789bcd0".to_string(),
                is_clean: false,
            },
        ),
    ];

    println!("  {:<20} {:<10} {:<15}", "Scenario", "Summary", "Total");
    println!("  {}", "-".repeat(50));

    for (name, status) in examples {
        println!("  {:<20} {:<10} {:<15}", name, status.summary(), status.total_changes());
    }
}

#[cfg(feature = "native")]
fn demo_action_determination() {
    println!("  Action Determination Logic:");
    println!();

    let scenarios = [
        ("Local = Remote, clean", Some("1.0.0"), Some("1.0.0"), true),
        ("Local > Remote, clean", Some("1.0.1"), Some("1.0.0"), true),
        ("Local = Remote, dirty", Some("1.0.0"), Some("1.0.0"), false),
        ("Local > Remote, dirty", Some("1.0.1"), Some("1.0.0"), false),
        ("Not published, clean", Some("1.0.0"), None, true),
        ("Not published, dirty", Some("1.0.0"), None, false),
        ("Local < Remote", Some("1.0.0"), Some("1.0.1"), true),
    ];

    println!("  {:<25} {:<8} {:<8} {:<8} {:<15}", "Scenario", "Local", "Remote", "Clean", "Action");
    println!("  {}", "-".repeat(70));

    for (name, local, remote, is_clean) in scenarios {
        let git_status =
            GitStatus { is_clean, modified: if is_clean { 0 } else { 3 }, ..Default::default() };
        let action = determine_action(local, remote, &git_status);
        println!(
            "  {:<25} {:<8} {:<8} {:<8} {:<15}",
            name,
            local.unwrap_or("-"),
            remote.unwrap_or("-"),
            if is_clean { "yes" } else { "no" },
            action.description()
        );
    }
}

#[cfg(feature = "native")]
fn demo_cache_performance() {
    println!("  Cache Key Computation:");
    println!();
    println!("  Cache key = hash(Cargo.toml content || git HEAD || mtime)");
    println!();
    println!("  Invalidation triggers:");
    println!("    1. Cargo.toml modified");
    println!("    2. Git HEAD changed (new commit)");
    println!("    3. crates.io TTL expired (15 min)");
    println!();
    println!("  Performance targets:");
    println!("    Cold cache: <5s (parallel fetches)");
    println!("    Warm cache: <100ms (O(1) hash check)");
    println!();

    // Demo cache operations
    let cache = PublishStatusCache::default();

    println!("  Cache operations:");
    println!("    Initial entries: {}", cache.get("test", "key1").is_none());

    // Simulating cache miss/hit would require more setup
    println!("    Cache is content-addressable");
    println!("    Different key = automatic invalidation");
}

#[cfg(feature = "native")]
fn demo_report_generation() {
    let statuses = vec![
        CrateStatus {
            name: "trueno".to_string(),
            local_version: Some("0.8.1".to_string()),
            crates_io_version: Some("0.8.1".to_string()),
            git_status: GitStatus { is_clean: true, ..Default::default() },
            action: PublishAction::UpToDate,
            path: PathBuf::from("../trueno"),
            error: None,
        },
        CrateStatus {
            name: "pacha".to_string(),
            local_version: Some("0.1.2".to_string()),
            crates_io_version: Some("0.1.1".to_string()),
            git_status: GitStatus { is_clean: true, ..Default::default() },
            action: PublishAction::NeedsPublish,
            path: PathBuf::from("../pacha"),
            error: None,
        },
        CrateStatus {
            name: "depyler".to_string(),
            local_version: Some("3.21.0".to_string()),
            crates_io_version: Some("3.20.0".to_string()),
            git_status: GitStatus {
                modified: 8,
                untracked: 3,
                is_clean: false,
                ..Default::default()
            },
            action: PublishAction::NeedsCommit,
            path: PathBuf::from("../depyler"),
            error: None,
        },
        CrateStatus {
            name: "certeza".to_string(),
            local_version: Some("0.1.0".to_string()),
            crates_io_version: None,
            git_status: GitStatus { is_clean: true, ..Default::default() },
            action: PublishAction::NotPublished,
            path: PathBuf::from("../certeza"),
            error: None,
        },
    ];

    let report = PublishStatusReport::from_statuses(statuses, 3, 85);

    println!("  Sample Report (text format):");
    println!();
    println!("{}", format_report_text(&report));

    println!("\n  Report Summary:");
    println!("    Total crates: {}", report.total);
    println!("    Needs publish: {}", report.needs_publish);
    println!("    Needs commit: {}", report.needs_commit);
    println!("    Up to date: {}", report.up_to_date);
    println!("    Cache hits: {}", report.cache_hits);
    println!("    Elapsed: {}ms", report.elapsed_ms);
}

#[cfg(feature = "native")]
fn demo_live_scan() -> anyhow::Result<()> {
    // Try to find workspace root
    let current_dir = std::env::current_dir()?;
    let workspace_root =
        current_dir.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from(".."));

    println!("  Workspace: {}", workspace_root.display());
    println!();

    // Check if we're in a valid workspace
    let trueno_path = workspace_root.join("trueno").join("Cargo.toml");
    if !trueno_path.exists() {
        println!("  [SKIP] Not in PAIML workspace (trueno not found)");
        println!("  Run from batuta directory with sibling crates present.");
        return Ok(());
    }

    println!("  Running live scan...");
    println!();

    let start = Instant::now();
    let mut scanner = PublishStatusScanner::new(workspace_root).with_crates_io();
    let report = scanner.scan_sync()?;
    let elapsed = start.elapsed();

    println!("{}", format_report_text(&report));
    println!();
    println!("  Actual elapsed: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!(
        "  Performance: {} (expected: {})",
        if elapsed.as_millis() < 200 {
            "EXCELLENT"
        } else if elapsed.as_millis() < 1000 {
            "GOOD"
        } else {
            "COLD CACHE"
        },
        if report.cache_hits > 0 { "<100ms" } else { "~7s" }
    );

    Ok(())
}

#[cfg(not(feature = "native"))]
fn main() {
    println!("This example requires the 'native' feature.");
    println!("Run with: cargo run --example publish_status_demo --features native");
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;

    #[test]
    fn test_action_symbols_not_empty() {
        let actions = [
            PublishAction::UpToDate,
            PublishAction::NeedsCommit,
            PublishAction::NeedsPublish,
            PublishAction::LocalBehind,
            PublishAction::NotPublished,
            PublishAction::Error,
        ];

        for action in actions {
            assert!(!action.symbol().is_empty());
            assert!(!action.description().is_empty());
        }
    }

    #[test]
    fn test_git_status_summary() {
        let clean = GitStatus { is_clean: true, ..Default::default() };
        assert_eq!(clean.summary(), "clean");

        let dirty = GitStatus { modified: 3, is_clean: false, ..Default::default() };
        assert_eq!(dirty.summary(), "3M");
    }

    #[test]
    fn test_action_determination_up_to_date() {
        let git = GitStatus { is_clean: true, ..Default::default() };
        let action = determine_action(Some("1.0.0"), Some("1.0.0"), &git);
        assert_eq!(action, PublishAction::UpToDate);
    }

    #[test]
    fn test_action_determination_needs_publish() {
        let git = GitStatus { is_clean: true, ..Default::default() };
        let action = determine_action(Some("1.0.1"), Some("1.0.0"), &git);
        assert_eq!(action, PublishAction::NeedsPublish);
    }

    #[test]
    fn test_report_counts() {
        let statuses = vec![
            CrateStatus {
                name: "a".to_string(),
                local_version: Some("1.0.0".to_string()),
                crates_io_version: Some("1.0.0".to_string()),
                git_status: GitStatus::default(),
                action: PublishAction::UpToDate,
                path: PathBuf::from("."),
                error: None,
            },
            CrateStatus {
                name: "b".to_string(),
                local_version: Some("1.0.1".to_string()),
                crates_io_version: Some("1.0.0".to_string()),
                git_status: GitStatus::default(),
                action: PublishAction::NeedsPublish,
                path: PathBuf::from("."),
                error: None,
            },
        ];

        let report = PublishStatusReport::from_statuses(statuses, 1, 50);
        assert_eq!(report.total, 2);
        assert_eq!(report.up_to_date, 1);
        assert_eq!(report.needs_publish, 1);
    }
}
