//! Tests for publish status module (PUB-001 through PUB-008).

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use super::cache::PublishStatusCache;
use super::format::{format_report_json, format_report_text};
use super::git::{determine_action, get_local_version};
use super::types::{CacheEntry, CrateStatus, GitStatus, PublishAction, PublishStatusReport};

// ========================================================================
// PUB-001: PublishAction tests
// ========================================================================

#[test]
fn test_pub_001_action_symbols() {
    assert_eq!(PublishAction::UpToDate.symbol(), "✓");
    assert_eq!(PublishAction::NeedsCommit.symbol(), "📝");
    assert_eq!(PublishAction::NeedsPublish.symbol(), "📦");
    assert_eq!(PublishAction::LocalBehind.symbol(), "⚠️");
    assert_eq!(PublishAction::NotPublished.symbol(), "🆕");
    assert_eq!(PublishAction::Error.symbol(), "❌");
}

#[test]
fn test_pub_001_action_descriptions() {
    assert_eq!(PublishAction::UpToDate.description(), "up to date");
    assert_eq!(PublishAction::NeedsPublish.description(), "PUBLISH");
}

// ========================================================================
// PUB-002: GitStatus tests
// ========================================================================

#[test]
fn test_pub_002_git_status_clean() {
    let status = GitStatus {
        modified: 0,
        untracked: 0,
        staged: 0,
        head_sha: "abc123".to_string(),
        is_clean: true,
    };
    assert_eq!(status.total_changes(), 0);
    assert_eq!(status.summary(), "clean");
}

#[test]
fn test_pub_002_git_status_dirty() {
    let status = GitStatus {
        modified: 3,
        untracked: 2,
        staged: 1,
        head_sha: "abc123".to_string(),
        is_clean: false,
    };
    assert_eq!(status.total_changes(), 6);
    assert_eq!(status.summary(), "3M 2? 1+");
}

#[test]
fn test_pub_002_git_status_modified_only() {
    let status = GitStatus {
        modified: 5,
        untracked: 0,
        staged: 0,
        head_sha: "def456".to_string(),
        is_clean: false,
    };
    assert_eq!(status.summary(), "5M");
}

// ========================================================================
// PUB-003: Cache tests
// ========================================================================

#[test]
fn test_pub_003_cache_entry_stale() {
    let old_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() - (20 * 60); // 20 minutes ago

    let entry = CacheEntry {
        cache_key: "test".to_string(),
        status: CrateStatus {
            name: "test".to_string(),
            local_version: Some("1.0.0".to_string()),
            crates_io_version: Some("1.0.0".to_string()),
            git_status: GitStatus::default(),
            action: PublishAction::UpToDate,
            path: PathBuf::from("."),
            error: None,
        },
        crates_io_checked_at: old_time,
        created_at: old_time,
    };

    assert!(entry.is_crates_io_stale());
}

#[test]
fn test_pub_003_cache_entry_fresh() {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

    let entry = CacheEntry {
        cache_key: "test".to_string(),
        status: CrateStatus {
            name: "test".to_string(),
            local_version: Some("1.0.0".to_string()),
            crates_io_version: Some("1.0.0".to_string()),
            git_status: GitStatus::default(),
            action: PublishAction::UpToDate,
            path: PathBuf::from("."),
            error: None,
        },
        crates_io_checked_at: now,
        created_at: now,
    };

    assert!(!entry.is_crates_io_stale());
}

#[test]
fn test_pub_003_cache_hit_miss() {
    let mut cache = PublishStatusCache::default();

    // Miss
    assert!(cache.get("test", "key1").is_none());

    // Insert
    let entry = CacheEntry {
        cache_key: "key1".to_string(),
        status: CrateStatus {
            name: "test".to_string(),
            local_version: Some("1.0.0".to_string()),
            crates_io_version: None,
            git_status: GitStatus::default(),
            action: PublishAction::NotPublished,
            path: PathBuf::from("."),
            error: None,
        },
        crates_io_checked_at: 0,
        created_at: 0,
    };
    cache.insert("test".to_string(), entry);

    // Hit with same key
    assert!(cache.get("test", "key1").is_some());

    // Miss with different key (invalidation)
    assert!(cache.get("test", "key2").is_none());
}

// ========================================================================
// PUB-004: Action determination tests
// ========================================================================

#[test]
fn test_pub_004_determine_action_up_to_date() {
    let git = GitStatus { is_clean: true, ..Default::default() };
    let action = determine_action(Some("1.0.0"), Some("1.0.0"), &git);
    assert_eq!(action, PublishAction::UpToDate);
}

#[test]
fn test_pub_004_determine_action_needs_publish() {
    let git = GitStatus { is_clean: true, ..Default::default() };
    let action = determine_action(Some("1.0.1"), Some("1.0.0"), &git);
    assert_eq!(action, PublishAction::NeedsPublish);
}

#[test]
fn test_pub_004_determine_action_needs_commit() {
    let git = GitStatus { is_clean: false, modified: 5, ..Default::default() };
    let action = determine_action(Some("1.0.1"), Some("1.0.0"), &git);
    assert_eq!(action, PublishAction::NeedsCommit);
}

#[test]
fn test_pub_004_determine_action_local_behind() {
    let git = GitStatus { is_clean: true, ..Default::default() };
    let action = determine_action(Some("1.0.0"), Some("1.0.1"), &git);
    assert_eq!(action, PublishAction::LocalBehind);
}

#[test]
fn test_pub_004_determine_action_not_published() {
    let git = GitStatus { is_clean: true, ..Default::default() };
    let action = determine_action(Some("1.0.0"), None, &git);
    assert_eq!(action, PublishAction::NotPublished);
}

#[test]
fn test_pub_004_determine_action_no_local() {
    let git = GitStatus::default();
    let action = determine_action(None, Some("1.0.0"), &git);
    assert_eq!(action, PublishAction::Error);
}

// ========================================================================
// PUB-005: Report tests
// ========================================================================

#[test]
fn test_pub_005_report_from_statuses() {
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
        CrateStatus {
            name: "c".to_string(),
            local_version: Some("1.0.0".to_string()),
            crates_io_version: Some("1.0.0".to_string()),
            git_status: GitStatus { modified: 3, is_clean: false, ..Default::default() },
            action: PublishAction::NeedsCommit,
            path: PathBuf::from("."),
            error: None,
        },
    ];

    let report = PublishStatusReport::from_statuses(statuses, 2, 50);

    assert_eq!(report.total, 3);
    assert_eq!(report.up_to_date, 1);
    assert_eq!(report.needs_publish, 1);
    assert_eq!(report.needs_commit, 1);
    assert_eq!(report.cache_hits, 2);
    assert_eq!(report.cache_misses, 1);
    assert_eq!(report.elapsed_ms, 50);
}

// ========================================================================
// PUB-006: Formatting tests
// ========================================================================

#[test]
fn test_pub_006_format_report_text() {
    let statuses = vec![CrateStatus {
        name: "trueno".to_string(),
        local_version: Some("0.8.1".to_string()),
        crates_io_version: Some("0.8.1".to_string()),
        git_status: GitStatus { is_clean: true, ..Default::default() },
        action: PublishAction::UpToDate,
        path: PathBuf::from("."),
        error: None,
    }];

    let report = PublishStatusReport::from_statuses(statuses, 1, 10);
    let text = format_report_text(&report);

    assert!(text.contains("trueno"));
    assert!(text.contains("0.8.1"));
    assert!(text.contains("✓"));
    assert!(text.contains("up to date"));
}

#[test]
fn test_pub_006_format_report_json() {
    let statuses = vec![CrateStatus {
        name: "test".to_string(),
        local_version: Some("1.0.0".to_string()),
        crates_io_version: Some("1.0.0".to_string()),
        git_status: GitStatus::default(),
        action: PublishAction::UpToDate,
        path: PathBuf::from("."),
        error: None,
    }];

    let report = PublishStatusReport::from_statuses(statuses, 0, 5);
    let json = format_report_json(&report).unwrap();

    assert!(json.contains("\"name\": \"test\""));
    assert!(json.contains("\"total\": 1"));
}

// ========================================================================
// PUB-007: Cache persistence tests
// ========================================================================

#[test]
fn test_pub_007_cache_save_load() {
    let temp_dir = std::env::temp_dir().join("batuta_cache_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let cache_path = temp_dir.join("test-cache.json");

    // Create and save cache
    let mut cache =
        PublishStatusCache { cache_path: Some(cache_path.clone()), ..Default::default() };
    cache.insert(
        "test_crate".to_string(),
        CacheEntry {
            cache_key: "abc123".to_string(),
            status: CrateStatus {
                name: "test_crate".to_string(),
                local_version: Some("1.0.0".to_string()),
                crates_io_version: Some("1.0.0".to_string()),
                git_status: GitStatus::default(),
                action: PublishAction::UpToDate,
                path: PathBuf::from("."),
                error: None,
            },
            crates_io_checked_at: 0,
            created_at: 0,
        },
    );
    cache.save().unwrap();

    // Load and verify
    let loaded = PublishStatusCache::load_from(&cache_path).unwrap();
    assert!(loaded.get("test_crate", "abc123").is_some());

    // Cleanup
    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_pub_007_cache_load_nonexistent() {
    let result = PublishStatusCache::load_from(Path::new("/nonexistent/path/cache.json"));
    assert!(result.is_ok()); // Returns default cache
    assert!(result.unwrap().entries.is_empty());
}

#[test]
fn test_pub_007_cache_clear() {
    let mut cache = PublishStatusCache::default();
    cache.insert(
        "test".to_string(),
        CacheEntry {
            cache_key: "key".to_string(),
            status: CrateStatus {
                name: "test".to_string(),
                local_version: None,
                crates_io_version: None,
                git_status: GitStatus::default(),
                action: PublishAction::Error,
                path: PathBuf::from("."),
                error: None,
            },
            crates_io_checked_at: 0,
            created_at: 0,
        },
    );

    assert!(cache.get("test", "key").is_some());
    cache.clear();
    assert!(cache.get("test", "key").is_none());
}

// ========================================================================
// PUB-008: Version parsing tests
// ========================================================================

#[test]
fn test_pub_008_get_local_version() {
    let temp_dir = std::env::temp_dir().join("batuta_version_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let cargo_toml = temp_dir.join("Cargo.toml");
    std::fs::write(
        &cargo_toml,
        r#"[package]
name = "test"
version = "1.2.3"
"#,
    )
    .unwrap();

    let version = get_local_version(&temp_dir).unwrap();
    assert_eq!(version, "1.2.3");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_pub_008_get_local_version_not_found() {
    let temp_dir = std::env::temp_dir().join("batuta_no_version_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let cargo_toml = temp_dir.join("Cargo.toml");
    std::fs::write(&cargo_toml, "[package]\nname = \"test\"\n").unwrap();

    let result = get_local_version(&temp_dir);
    assert!(result.is_err());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_pub_008_get_local_version_no_file() {
    let result = get_local_version(Path::new("/nonexistent/path"));
    assert!(result.is_err());
}
