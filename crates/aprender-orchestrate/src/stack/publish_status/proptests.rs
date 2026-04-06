//! Property-based tests for publish status module.

use std::path::PathBuf;

use proptest::prelude::*;

use super::cache::PublishStatusCache;
use super::format::format_report_text;
use super::scanner::PublishStatusScanner;
use super::types::{CacheEntry, CrateStatus, GitStatus, PublishAction, PublishStatusReport};
use std::path::Path;

proptest! {
    /// PROPERTY: GitStatus total_changes is sum of components
    #[test]
    fn prop_git_status_total(m in 0usize..100, u in 0usize..100, s in 0usize..100) {
        let status = GitStatus {
            modified: m,
            untracked: u,
            staged: s,
            head_sha: String::new(),
            is_clean: m + u + s == 0,
        };
        prop_assert_eq!(status.total_changes(), m + u + s);
    }

    /// PROPERTY: Clean status has zero changes
    #[test]
    fn prop_clean_status_zero_changes(sha in "[a-f0-9]{7}") {
        let status = GitStatus {
            modified: 0,
            untracked: 0,
            staged: 0,
            head_sha: sha,
            is_clean: true,
        };
        prop_assert_eq!(status.total_changes(), 0);
        prop_assert_eq!(status.summary(), "clean");
    }

    /// PROPERTY: Report counts are consistent
    #[test]
    fn prop_report_counts_consistent(
        up in 0usize..10,
        publish in 0usize..10,
        commit in 0usize..10
    ) {
        let mut statuses = Vec::new();

        for i in 0..up {
            statuses.push(CrateStatus {
                name: format!("up{}", i),
                local_version: Some("1.0.0".to_string()),
                crates_io_version: Some("1.0.0".to_string()),
                git_status: GitStatus::default(),
                action: PublishAction::UpToDate,
                path: PathBuf::from("."),
                error: None,
            });
        }

        for i in 0..publish {
            statuses.push(CrateStatus {
                name: format!("pub{}", i),
                local_version: Some("1.0.1".to_string()),
                crates_io_version: Some("1.0.0".to_string()),
                git_status: GitStatus::default(),
                action: PublishAction::NeedsPublish,
                path: PathBuf::from("."),
                error: None,
            });
        }

        for i in 0..commit {
            statuses.push(CrateStatus {
                name: format!("commit{}", i),
                local_version: Some("1.0.0".to_string()),
                crates_io_version: Some("1.0.0".to_string()),
                git_status: GitStatus { modified: 1, is_clean: false, ..Default::default() },
                action: PublishAction::NeedsCommit,
                path: PathBuf::from("."),
                error: None,
            });
        }

        let report = PublishStatusReport::from_statuses(statuses, 0, 0);

        prop_assert_eq!(report.total, up + publish + commit);
        prop_assert_eq!(report.up_to_date, up);
        prop_assert_eq!(report.needs_publish, publish);
        prop_assert_eq!(report.needs_commit, commit);
    }
}

// ========================================================================
// PUB-015: Additional Scanner and Edge Case Tests
// ========================================================================

#[test]
fn test_pub_015_check_crate_error_path() {
    let mut scanner = PublishStatusScanner::new(PathBuf::from("/nonexistent"));
    let status = scanner.check_crate("test", Path::new("/nonexistent/path"));

    assert_eq!(status.action, PublishAction::Error);
    assert!(status.error.is_some());
}

#[test]
fn test_pub_015_check_crate_with_valid_path() {
    let temp_dir = std::env::temp_dir().join("batuta_check_crate_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    // Create a Cargo.toml
    std::fs::write(
        temp_dir.join("Cargo.toml"),
        r#"[package]
name = "test-crate"
version = "1.0.0"
"#,
    )
    .unwrap();

    // Initialize git repo
    let _ = std::process::Command::new("git").args(["init"]).current_dir(&temp_dir).output();

    let mut scanner = PublishStatusScanner::new(temp_dir.parent().unwrap().to_path_buf());
    let status = scanner.check_crate("test-crate", &temp_dir);

    assert_eq!(status.local_version, Some("1.0.0".to_string()));
    assert_eq!(status.name, "test-crate");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_pub_015_scanner_find_crate_dirs_with_paiml() {
    let temp_dir = std::env::temp_dir().join("batuta_paiml_dirs_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    // Create a "trueno" directory with Cargo.toml (matching PAIML_CRATES)
    let trueno_dir = temp_dir.join("trueno");
    std::fs::create_dir_all(&trueno_dir).unwrap();
    std::fs::write(
        trueno_dir.join("Cargo.toml"),
        r#"[package]
name = "trueno"
version = "0.8.0"
"#,
    )
    .unwrap();

    let scanner = PublishStatusScanner::new(temp_dir.clone());
    let dirs = scanner.find_crate_dirs();

    assert_eq!(dirs.len(), 1);
    assert_eq!(dirs[0].0, "trueno");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_pub_015_all_action_symbols() {
    // Test all symbols for complete coverage
    assert_eq!(PublishAction::UpToDate.symbol(), "✓");
    assert_eq!(PublishAction::NeedsCommit.symbol(), "📝");
    assert_eq!(PublishAction::NeedsPublish.symbol(), "📦");
    assert_eq!(PublishAction::LocalBehind.symbol(), "⚠️");
    assert_eq!(PublishAction::NotPublished.symbol(), "🆕");
    assert_eq!(PublishAction::Error.symbol(), "❌");
}

#[test]
fn test_pub_015_all_action_descriptions() {
    assert_eq!(PublishAction::UpToDate.description(), "up to date");
    assert_eq!(PublishAction::NeedsCommit.description(), "commit changes");
    assert_eq!(PublishAction::NeedsPublish.description(), "PUBLISH");
    assert_eq!(PublishAction::LocalBehind.description(), "local behind");
    assert_eq!(PublishAction::NotPublished.description(), "not published");
    assert_eq!(PublishAction::Error.description(), "error");
}

#[test]
fn test_pub_015_cache_entry_serialization() {
    let entry = CacheEntry {
        cache_key: "abc123".to_string(),
        status: CrateStatus {
            name: "test".to_string(),
            local_version: Some("1.0.0".to_string()),
            crates_io_version: None,
            git_status: GitStatus::default(),
            action: PublishAction::NotPublished,
            path: PathBuf::from("/test"),
            error: None,
        },
        crates_io_checked_at: 1234567890,
        created_at: 1234567890,
    };

    let json = serde_json::to_string(&entry).unwrap();
    let parsed: CacheEntry = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.cache_key, "abc123");
    assert_eq!(parsed.status.name, "test");
}

#[test]
fn test_pub_015_cache_default_path() {
    let cache = PublishStatusCache::default();
    // Just verify it doesn't panic
    assert!(cache.entries.is_empty());
}

#[test]
fn test_pub_015_format_report_all_actions() {
    let statuses = vec![
        CrateStatus {
            name: "a".to_string(),
            local_version: Some("1.0.0".to_string()),
            crates_io_version: Some("1.0.0".to_string()),
            git_status: GitStatus { is_clean: true, ..Default::default() },
            action: PublishAction::UpToDate,
            path: PathBuf::new(),
            error: None,
        },
        CrateStatus {
            name: "b".to_string(),
            local_version: Some("1.0.1".to_string()),
            crates_io_version: Some("1.0.0".to_string()),
            git_status: GitStatus { is_clean: true, ..Default::default() },
            action: PublishAction::NeedsPublish,
            path: PathBuf::new(),
            error: None,
        },
        CrateStatus {
            name: "c".to_string(),
            local_version: Some("1.0.0".to_string()),
            crates_io_version: Some("1.0.0".to_string()),
            git_status: GitStatus { is_clean: false, modified: 1, ..Default::default() },
            action: PublishAction::NeedsCommit,
            path: PathBuf::new(),
            error: None,
        },
        CrateStatus {
            name: "d".to_string(),
            local_version: Some("0.9.0".to_string()),
            crates_io_version: Some("1.0.0".to_string()),
            git_status: GitStatus { is_clean: true, ..Default::default() },
            action: PublishAction::LocalBehind,
            path: PathBuf::new(),
            error: None,
        },
        CrateStatus {
            name: "e".to_string(),
            local_version: Some("1.0.0".to_string()),
            crates_io_version: None,
            git_status: GitStatus { is_clean: true, ..Default::default() },
            action: PublishAction::NotPublished,
            path: PathBuf::new(),
            error: None,
        },
        CrateStatus {
            name: "f".to_string(),
            local_version: None,
            crates_io_version: None,
            git_status: GitStatus::default(),
            action: PublishAction::Error,
            path: PathBuf::new(),
            error: Some("Test error".to_string()),
        },
    ];

    let report = PublishStatusReport::from_statuses(statuses, 2, 100);
    let text = format_report_text(&report);

    // Verify all actions appear in output
    assert!(text.contains("✓"));
    assert!(text.contains("📦"));
    assert!(text.contains("📝"));
    assert!(text.contains("⚠️"));
    assert!(text.contains("🆕"));
    assert!(text.contains("❌"));
    assert!(text.contains("cache: 2 hits, 4 misses"));
}

#[test]
fn test_pub_015_git_status_summary_all_types() {
    // Mixed changes
    let status = GitStatus {
        modified: 2,
        untracked: 1,
        staged: 3,
        is_clean: false,
        head_sha: "abc1234".to_string(),
    };
    let summary = status.summary();
    assert!(summary.contains("2M"));
    assert!(summary.contains("1?"));
    assert!(summary.contains("3+"));
}

#[test]
fn test_pub_015_report_cache_stats_valid() {
    // Test with valid cache stats (cache_hits <= total)
    let statuses = vec![
        CrateStatus {
            name: "cached".to_string(),
            local_version: Some("1.0.0".to_string()),
            crates_io_version: Some("1.0.0".to_string()),
            git_status: GitStatus::default(),
            action: PublishAction::UpToDate,
            path: PathBuf::new(),
            error: None,
        },
        CrateStatus {
            name: "fresh".to_string(),
            local_version: Some("1.0.0".to_string()),
            crates_io_version: Some("1.0.0".to_string()),
            git_status: GitStatus::default(),
            action: PublishAction::UpToDate,
            path: PathBuf::new(),
            error: None,
        },
    ];
    let report = PublishStatusReport::from_statuses(statuses, 1, 50);
    assert_eq!(report.total, 2);
    assert_eq!(report.cache_hits, 1);
    assert_eq!(report.cache_misses, 1);
    assert_eq!(report.elapsed_ms, 50);
}

#[test]
fn test_pub_015_report_with_cache_hits() {
    let statuses = vec![CrateStatus {
        name: "test".to_string(),
        local_version: Some("1.0.0".to_string()),
        crates_io_version: Some("1.0.0".to_string()),
        git_status: GitStatus::default(),
        action: PublishAction::UpToDate,
        path: PathBuf::new(),
        error: None,
    }];
    let report = PublishStatusReport::from_statuses(statuses, 1, 10);
    assert_eq!(report.cache_hits, 1);
    assert_eq!(report.cache_misses, 0);
    assert_eq!(report.elapsed_ms, 10);
}
