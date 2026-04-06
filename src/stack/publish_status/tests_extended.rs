//! Extended tests for publish status module (PUB-009 through PUB-014).

use std::path::{Path, PathBuf};

use super::cache::{compute_cache_key, PublishStatusCache};
use super::format::format_report_text;
use super::git::{determine_action, get_local_version};
use super::scanner::PublishStatusScanner;
use super::types::{CrateStatus, GitStatus, PublishAction, PublishStatusReport};

// ========================================================================
// PUB-009: Cache key computation tests
// ========================================================================

#[test]
fn test_pub_009_compute_cache_key() {
    let temp_dir = std::env::temp_dir().join("batuta_cache_key_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let cargo_toml = temp_dir.join("Cargo.toml");
    std::fs::write(&cargo_toml, "[package]\nname = \"test\"\nversion = \"1.0.0\"\n").unwrap();

    // Initialize git repo
    let _ = std::process::Command::new("git").args(["init"]).current_dir(&temp_dir).output();
    let _ = std::process::Command::new("git").args(["add", "."]).current_dir(&temp_dir).output();
    let _ = std::process::Command::new("git")
        .args(["commit", "-m", "initial"])
        .current_dir(&temp_dir)
        .output();

    let key = compute_cache_key(&temp_dir);
    assert!(key.is_ok());
    // Key should be a hex string
    assert!(key.unwrap().chars().all(|c| c.is_ascii_hexdigit()));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_pub_009_compute_cache_key_no_cargo_toml() {
    let result = compute_cache_key(Path::new("/nonexistent/path"));
    assert!(result.is_err());
}

#[test]
fn test_pub_009_compute_cache_key_no_git() {
    let temp_dir = std::env::temp_dir().join("batuta_no_git_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let cargo_toml = temp_dir.join("Cargo.toml");
    std::fs::write(&cargo_toml, "[package]\nname = \"test\"\nversion = \"1.0.0\"\n").unwrap();

    // Should still work, using "no-git" as HEAD
    let key = compute_cache_key(&temp_dir);
    assert!(key.is_ok());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ========================================================================
// PUB-010: Scanner tests
// ========================================================================

#[test]
fn test_pub_010_scanner_find_crate_dirs_empty() {
    let temp_dir = std::env::temp_dir().join("batuta_scanner_empty_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let scanner = PublishStatusScanner::new(temp_dir.clone());
    let dirs = scanner.find_crate_dirs();
    assert!(dirs.is_empty());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_pub_010_scanner_new() {
    let scanner = PublishStatusScanner::new(PathBuf::from("/tmp"));
    assert_eq!(scanner.workspace_root, PathBuf::from("/tmp"));
}

// ========================================================================
// PUB-011: GitStatus additional tests
// ========================================================================

#[test]
fn test_pub_011_git_status_summary_untracked_only() {
    let status = GitStatus {
        modified: 0,
        untracked: 3,
        staged: 0,
        head_sha: String::new(),
        is_clean: false,
    };
    assert_eq!(status.summary(), "3?");
}

#[test]
fn test_pub_011_git_status_summary_staged_only() {
    let status = GitStatus {
        modified: 0,
        untracked: 0,
        staged: 2,
        head_sha: String::new(),
        is_clean: false,
    };
    assert_eq!(status.summary(), "2+");
}

#[test]
fn test_pub_011_git_status_default() {
    let status = GitStatus::default();
    // Default has is_clean = false (since the Default derive sets it to false)
    assert_eq!(status.total_changes(), 0);
    assert_eq!(status.modified, 0);
    assert_eq!(status.untracked, 0);
    assert_eq!(status.staged, 0);
}

// ========================================================================
// PUB-012: CrateStatus tests
// ========================================================================

#[test]
fn test_pub_012_crate_status_with_error() {
    let status = CrateStatus {
        name: "broken".to_string(),
        local_version: None,
        crates_io_version: None,
        git_status: GitStatus::default(),
        action: PublishAction::Error,
        path: PathBuf::from("/broken"),
        error: Some("Test error".to_string()),
    };

    assert_eq!(status.action, PublishAction::Error);
    assert!(status.error.is_some());
}

#[test]
fn test_pub_012_crate_status_serialization() {
    let status = CrateStatus {
        name: "test".to_string(),
        local_version: Some("1.0.0".to_string()),
        crates_io_version: Some("1.0.0".to_string()),
        git_status: GitStatus::default(),
        action: PublishAction::UpToDate,
        path: PathBuf::from("."),
        error: None,
    };

    let json = serde_json::to_string(&status).unwrap();
    assert!(json.contains("\"name\":\"test\""));

    let deserialized: CrateStatus = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.name, "test");
}

// ========================================================================
// PUB-013: Report edge cases
// ========================================================================

#[test]
fn test_pub_013_report_empty() {
    let report = PublishStatusReport::from_statuses(vec![], 0, 0);
    assert_eq!(report.total, 0);
    assert_eq!(report.needs_publish, 0);
    assert_eq!(report.needs_commit, 0);
    assert_eq!(report.up_to_date, 0);
}

#[test]
fn test_pub_013_report_all_errors() {
    let statuses = vec![
        CrateStatus {
            name: "a".to_string(),
            local_version: None,
            crates_io_version: None,
            git_status: GitStatus::default(),
            action: PublishAction::Error,
            path: PathBuf::from("."),
            error: Some("error 1".to_string()),
        },
        CrateStatus {
            name: "b".to_string(),
            local_version: None,
            crates_io_version: None,
            git_status: GitStatus::default(),
            action: PublishAction::Error,
            path: PathBuf::from("."),
            error: Some("error 2".to_string()),
        },
    ];

    let report = PublishStatusReport::from_statuses(statuses, 0, 0);
    assert_eq!(report.total, 2);
    assert_eq!(report.needs_publish, 0);
    assert_eq!(report.up_to_date, 0);
}

#[test]
fn test_pub_013_format_report_text_with_missing() {
    let statuses = vec![CrateStatus {
        name: "missing".to_string(),
        local_version: None,
        crates_io_version: None,
        git_status: GitStatus::default(),
        action: PublishAction::Error,
        path: PathBuf::from("."),
        error: None,
    }];

    let report = PublishStatusReport::from_statuses(statuses, 0, 0);
    let text = format_report_text(&report);

    assert!(text.contains("missing"));
    assert!(text.contains('-')); // Missing versions shown as "-"
}

// ========================================================================
// PUB-014: Action edge cases
// ========================================================================

#[test]
fn test_pub_014_determine_action_not_published_dirty() {
    let git = GitStatus { is_clean: false, modified: 1, ..Default::default() };
    let action = determine_action(Some("1.0.0"), None, &git);
    assert_eq!(action, PublishAction::NeedsCommit);
}

#[test]
fn test_pub_014_determine_action_same_version_dirty() {
    let git = GitStatus { is_clean: false, staged: 1, ..Default::default() };
    let action = determine_action(Some("1.0.0"), Some("1.0.0"), &git);
    assert_eq!(action, PublishAction::NeedsCommit);
}

#[test]
fn test_pub_014_determine_action_invalid_semver() {
    let git = GitStatus { is_clean: true, ..Default::default() };
    // Invalid semver versions
    let action = determine_action(Some("not-a-version"), Some("also-not-valid"), &git);
    assert_eq!(action, PublishAction::UpToDate); // Falls through to UpToDate
}

// ========================================================================
// PUB-EXT-015: Scanner check_crate tests
// ========================================================================

#[test]
fn test_pub_ext_015_scanner_check_crate_cache_miss() {
    // ARRANGE
    let temp_dir = std::env::temp_dir().join("batuta_ext_check_crate_miss_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let cargo_toml = temp_dir.join("Cargo.toml");
    std::fs::write(&cargo_toml, "[package]\nname = \"test-crate\"\nversion = \"1.2.3\"\n").unwrap();

    // Init git
    let _ = std::process::Command::new("git").args(["init"]).current_dir(&temp_dir).output();
    let _ = std::process::Command::new("git").args(["add", "."]).current_dir(&temp_dir).output();
    let _ = std::process::Command::new("git")
        .args(["commit", "-m", "initial"])
        .current_dir(&temp_dir)
        .output();

    let mut scanner = PublishStatusScanner::new(temp_dir.parent().unwrap().to_path_buf());

    // ACT
    let status = scanner.check_crate("test-crate", &temp_dir);

    // ASSERT
    assert_eq!(status.name, "test-crate");
    assert_eq!(status.local_version, Some("1.2.3".to_string()));
    assert!(status.error.is_none());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_pub_ext_015_scanner_check_crate_cache_hit() {
    // ARRANGE
    let temp_dir = std::env::temp_dir().join("batuta_ext_check_crate_hit_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let cargo_toml = temp_dir.join("Cargo.toml");
    std::fs::write(&cargo_toml, "[package]\nname = \"cached-crate\"\nversion = \"2.0.0\"\n")
        .unwrap();

    // Init git
    let _ = std::process::Command::new("git").args(["init"]).current_dir(&temp_dir).output();
    let _ = std::process::Command::new("git").args(["add", "."]).current_dir(&temp_dir).output();
    let _ = std::process::Command::new("git")
        .args(["commit", "-m", "initial"])
        .current_dir(&temp_dir)
        .output();

    let mut scanner = PublishStatusScanner::new(temp_dir.parent().unwrap().to_path_buf());

    // First call - cache miss
    let status1 = scanner.check_crate("cached-crate", &temp_dir);

    // Second call - should be cache hit
    let status2 = scanner.check_crate("cached-crate", &temp_dir);

    // ASSERT
    assert_eq!(status1.name, status2.name);
    assert_eq!(status1.local_version, status2.local_version);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_pub_ext_015_scanner_check_crate_error_path() {
    // ARRANGE
    let mut scanner = PublishStatusScanner::new(PathBuf::from("/nonexistent"));

    // ACT - check a crate in a nonexistent path
    let status = scanner.check_crate("bad-crate", Path::new("/nonexistent/bad-crate"));

    // ASSERT
    assert_eq!(status.name, "bad-crate");
    assert_eq!(status.action, PublishAction::Error);
    assert!(status.error.is_some());
}

// ========================================================================
// PUB-016: Scanner refresh_crate tests
// ========================================================================

#[test]
fn test_pub_ext_016_scanner_refresh_crate() {
    // ARRANGE
    let temp_dir = std::env::temp_dir().join("batuta_refresh_crate_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let cargo_toml = temp_dir.join("Cargo.toml");
    std::fs::write(&cargo_toml, "[package]\nname = \"refresh-test\"\nversion = \"0.1.0\"\n")
        .unwrap();

    // Init git
    let _ = std::process::Command::new("git").args(["init"]).current_dir(&temp_dir).output();
    let _ = std::process::Command::new("git").args(["add", "."]).current_dir(&temp_dir).output();
    let _ = std::process::Command::new("git")
        .args(["commit", "-m", "initial"])
        .current_dir(&temp_dir)
        .output();

    let mut scanner = PublishStatusScanner::new(temp_dir.parent().unwrap().to_path_buf());

    // ACT
    let status = scanner.refresh_crate("refresh-test", &temp_dir, "test-cache-key");

    // ASSERT
    assert_eq!(status.name, "refresh-test");
    assert_eq!(status.local_version, Some("0.1.0".to_string()));
    assert!(status.crates_io_version.is_none()); // Not fetched in refresh_crate
    assert!(status.error.is_none());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_pub_ext_016_scanner_refresh_crate_no_version() {
    // ARRANGE
    let temp_dir = std::env::temp_dir().join("batuta_refresh_no_version_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    // Cargo.toml without version field
    let cargo_toml = temp_dir.join("Cargo.toml");
    std::fs::write(&cargo_toml, "[package]\nname = \"no-version\"\n").unwrap();

    let mut scanner = PublishStatusScanner::new(temp_dir.parent().unwrap().to_path_buf());

    // ACT
    let status = scanner.refresh_crate("no-version", &temp_dir, "test-key");

    // ASSERT
    assert_eq!(status.name, "no-version");
    assert!(status.local_version.is_none());
    // Should result in Error action due to missing version
    assert_eq!(status.action, PublishAction::Error);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ========================================================================
// PUB-017: Scanner with_crates_io tests
// ========================================================================

#[cfg(feature = "native")]
#[test]
fn test_pub_ext_017_scanner_with_crates_io() {
    // ARRANGE & ACT
    let scanner = PublishStatusScanner::new(PathBuf::from("/tmp")).with_crates_io();

    // ASSERT
    assert!(scanner.crates_io.is_some());
}

#[cfg(feature = "native")]
#[test]
fn test_pub_ext_017_scanner_default_no_crates_io() {
    // ARRANGE & ACT
    let scanner = PublishStatusScanner::new(PathBuf::from("/tmp"));

    // ASSERT
    assert!(scanner.crates_io.is_none());
}

// ========================================================================
// PUB-018: PublishAction additional tests
// ========================================================================

#[test]
fn test_pub_ext_018_action_local_behind_symbol() {
    assert_eq!(PublishAction::LocalBehind.symbol(), "⚠️");
}

#[test]
fn test_pub_ext_018_action_not_published_description() {
    assert_eq!(PublishAction::NotPublished.description(), "not published");
}

#[test]
fn test_pub_ext_018_action_error_description() {
    assert_eq!(PublishAction::Error.description(), "error");
}

#[test]
fn test_pub_ext_018_action_needs_commit_description() {
    assert_eq!(PublishAction::NeedsCommit.description(), "commit changes");
}

#[test]
fn test_pub_ext_018_action_local_behind_description() {
    assert_eq!(PublishAction::LocalBehind.description(), "local behind");
}

// ========================================================================
// PUB-019: Cache additional tests
// ========================================================================

#[test]
fn test_pub_ext_019_cache_invalidation_on_key_change() {
    // ARRANGE
    let mut cache = PublishStatusCache::default();
    let entry = super::types::CacheEntry {
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
        crates_io_checked_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        created_at: 0,
    };
    cache.insert("test".to_string(), entry);

    // ACT & ASSERT
    // Hit with matching key
    assert!(cache.get("test", "key1").is_some());
    // Miss with different key (cache invalidation)
    assert!(cache.get("test", "key2").is_none());
}

#[test]
fn test_pub_ext_019_cache_entry_created_at() {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();

    let entry = super::types::CacheEntry {
        cache_key: "test".to_string(),
        status: CrateStatus {
            name: "test".to_string(),
            local_version: None,
            crates_io_version: None,
            git_status: GitStatus::default(),
            action: PublishAction::Error,
            path: PathBuf::from("."),
            error: None,
        },
        crates_io_checked_at: now,
        created_at: now,
    };

    assert_eq!(entry.created_at, now);
}

// ========================================================================
// PUB-020: Report statistics tests
// ========================================================================

#[test]
fn test_pub_ext_020_report_cache_statistics() {
    let statuses = vec![
        CrateStatus {
            name: "a".to_string(),
            local_version: Some("1.0.0".to_string()),
            crates_io_version: Some("1.0.0".to_string()),
            git_status: GitStatus { is_clean: true, ..Default::default() },
            action: PublishAction::UpToDate,
            path: PathBuf::from("."),
            error: None,
        },
        CrateStatus {
            name: "b".to_string(),
            local_version: Some("2.0.0".to_string()),
            crates_io_version: Some("1.0.0".to_string()),
            git_status: GitStatus { is_clean: true, ..Default::default() },
            action: PublishAction::NeedsPublish,
            path: PathBuf::from("."),
            error: None,
        },
    ];

    let report = PublishStatusReport::from_statuses(statuses, 1, 100);

    assert_eq!(report.cache_hits, 1);
    assert_eq!(report.cache_misses, 1); // 2 statuses - 1 hit = 1 miss
    assert_eq!(report.elapsed_ms, 100);
}

#[test]
fn test_pub_ext_020_report_all_local_behind() {
    let statuses = vec![
        CrateStatus {
            name: "old1".to_string(),
            local_version: Some("0.1.0".to_string()),
            crates_io_version: Some("1.0.0".to_string()),
            git_status: GitStatus { is_clean: true, ..Default::default() },
            action: PublishAction::LocalBehind,
            path: PathBuf::from("."),
            error: None,
        },
        CrateStatus {
            name: "old2".to_string(),
            local_version: Some("0.2.0".to_string()),
            crates_io_version: Some("2.0.0".to_string()),
            git_status: GitStatus { is_clean: true, ..Default::default() },
            action: PublishAction::LocalBehind,
            path: PathBuf::from("."),
            error: None,
        },
    ];

    let report = PublishStatusReport::from_statuses(statuses, 0, 50);

    assert_eq!(report.total, 2);
    assert_eq!(report.up_to_date, 0);
    assert_eq!(report.needs_publish, 0);
    assert_eq!(report.needs_commit, 0);
}

// ========================================================================
// PUB-021: Scanner find_crate_dirs with PAIML directories
// ========================================================================

#[test]
fn test_pub_ext_021_scanner_find_crate_dirs_with_multiple_paiml() {
    // ARRANGE
    let temp_dir = std::env::temp_dir().join("batuta_ext_021_find_dirs");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    // Create directories matching PAIML crate names with Cargo.toml
    for name in &["trueno", "aprender", "realizar"] {
        let crate_dir = temp_dir.join(name);
        std::fs::create_dir_all(&crate_dir).unwrap();
        std::fs::write(
            crate_dir.join("Cargo.toml"),
            format!("[package]\nname = \"{}\"\nversion = \"0.1.0\"\n", name),
        )
        .unwrap();
    }

    // Create a non-PAIML directory (should be ignored)
    let non_paiml = temp_dir.join("serde");
    std::fs::create_dir_all(&non_paiml).unwrap();
    std::fs::write(
        non_paiml.join("Cargo.toml"),
        "[package]\nname = \"serde\"\nversion = \"1.0.0\"\n",
    )
    .unwrap();

    let scanner = PublishStatusScanner::new(temp_dir.clone());

    // ACT
    let dirs = scanner.find_crate_dirs();

    // ASSERT
    assert_eq!(dirs.len(), 3);
    let names: Vec<&str> = dirs.iter().map(|(n, _)| n.as_str()).collect();
    assert!(names.contains(&"trueno"));
    assert!(names.contains(&"aprender"));
    assert!(names.contains(&"realizar"));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_pub_ext_021_scanner_find_crate_dirs_missing_cargo_toml() {
    // ARRANGE: directory exists but no Cargo.toml
    let temp_dir = std::env::temp_dir().join("batuta_ext_021_no_cargo");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let trueno_dir = temp_dir.join("trueno");
    std::fs::create_dir_all(&trueno_dir).unwrap();
    // No Cargo.toml in trueno directory

    let scanner = PublishStatusScanner::new(temp_dir.clone());

    // ACT
    let dirs = scanner.find_crate_dirs();

    // ASSERT: should not include directory without Cargo.toml
    assert!(dirs.is_empty());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ========================================================================
// PUB-022: Scanner refresh_crate cache insertion verification
// ========================================================================

#[test]
fn test_pub_ext_022_refresh_crate_inserts_to_cache() {
    // ARRANGE
    let temp_dir = std::env::temp_dir().join("batuta_ext_022_cache_insert");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let cargo_toml = temp_dir.join("Cargo.toml");
    std::fs::write(&cargo_toml, "[package]\nname = \"cache-insert-test\"\nversion = \"3.0.0\"\n")
        .unwrap();

    let mut scanner = PublishStatusScanner::new(temp_dir.parent().unwrap().to_path_buf());

    // ACT
    let _status = scanner.refresh_crate("cache-insert-test", &temp_dir, "unique-cache-key");

    // ASSERT: verify the cache now has an entry
    let cached = scanner.cache.get("cache-insert-test", "unique-cache-key");
    assert!(cached.is_some(), "Cache should have entry after refresh");
    assert_eq!(cached.unwrap().status.local_version, Some("3.0.0".to_string()));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_pub_ext_022_refresh_crate_sets_crates_io_none() {
    // ARRANGE
    let temp_dir = std::env::temp_dir().join("batuta_ext_022_no_crates_io");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    std::fs::write(
        temp_dir.join("Cargo.toml"),
        "[package]\nname = \"local-only\"\nversion = \"1.0.0\"\n",
    )
    .unwrap();

    let mut scanner = PublishStatusScanner::new(temp_dir.parent().unwrap().to_path_buf());

    // ACT
    let status = scanner.refresh_crate("local-only", &temp_dir, "key");

    // ASSERT: crates_io_version is always None in refresh_crate (filled by async scan)
    assert!(status.crates_io_version.is_none());
    assert!(status.error.is_none());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ========================================================================
// PUB-023: Scanner check_crate with stale cache entry
// ========================================================================

#[test]
fn test_pub_ext_023_check_crate_stale_cache_refreshes() {
    // ARRANGE
    let temp_dir = std::env::temp_dir().join("batuta_ext_023_stale");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    std::fs::write(
        temp_dir.join("Cargo.toml"),
        "[package]\nname = \"stale-test\"\nversion = \"1.0.0\"\n",
    )
    .unwrap();

    // Init git so compute_cache_key works
    let _ = std::process::Command::new("git").args(["init"]).current_dir(&temp_dir).output();
    let _ = std::process::Command::new("git").args(["add", "."]).current_dir(&temp_dir).output();
    let _ = std::process::Command::new("git")
        .args(["commit", "-m", "initial"])
        .current_dir(&temp_dir)
        .output();

    let mut scanner = PublishStatusScanner::new(temp_dir.parent().unwrap().to_path_buf());

    // First call - populates cache
    let status1 = scanner.check_crate("stale-test", &temp_dir);
    assert_eq!(status1.local_version, Some("1.0.0".to_string()));

    // Manually mark the cache entry as stale (set crates_io_checked_at far in the past)
    if let Some(entry) = scanner.cache.entries.get_mut("stale-test") {
        entry.crates_io_checked_at = 0; // Very stale
    }

    // Second call - should refresh because cache is stale
    let status2 = scanner.check_crate("stale-test", &temp_dir);
    assert_eq!(status2.local_version, Some("1.0.0".to_string()));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ========================================================================
// PUB-024: PublishStatusCache load_from and save with path
// ========================================================================

#[test]
fn test_pub_ext_024_cache_load_from_valid() {
    // ARRANGE
    let temp_dir = std::env::temp_dir().join("batuta_ext_024_load_from");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let cache_path = temp_dir.join("test-publish-cache.json");

    // Create a cache, save it, load from specific path
    let mut cache =
        PublishStatusCache { cache_path: Some(cache_path.clone()), ..Default::default() };
    cache.insert(
        "load-from-test".to_string(),
        super::types::CacheEntry {
            cache_key: "key123".to_string(),
            status: CrateStatus {
                name: "load-from-test".to_string(),
                local_version: Some("5.0.0".to_string()),
                crates_io_version: Some("4.0.0".to_string()),
                git_status: GitStatus::default(),
                action: PublishAction::NeedsPublish,
                path: PathBuf::from("."),
                error: None,
            },
            crates_io_checked_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            created_at: 0,
        },
    );
    cache.save().unwrap();

    // ACT
    let loaded = PublishStatusCache::load_from(&cache_path).unwrap();

    // ASSERT
    let entry = loaded.get("load-from-test", "key123");
    assert!(entry.is_some());
    assert_eq!(entry.unwrap().status.local_version, Some("5.0.0".to_string()));
    assert_eq!(entry.unwrap().status.action, PublishAction::NeedsPublish);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_pub_ext_024_cache_load_from_corrupt() {
    // ARRANGE
    let temp_dir = std::env::temp_dir().join("batuta_ext_024_corrupt");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let cache_path = temp_dir.join("corrupt-cache.json");
    std::fs::write(&cache_path, "{{invalid json").unwrap();

    // ACT
    let result = PublishStatusCache::load_from(&cache_path);

    // ASSERT: corrupt JSON should return an error
    assert!(result.is_err());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_pub_ext_024_cache_save_and_reload_preserves_path() {
    // ARRANGE
    let temp_dir = std::env::temp_dir().join("batuta_ext_024_path");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let cache_path = temp_dir.join("path-test-cache.json");

    let cache = PublishStatusCache { cache_path: Some(cache_path.clone()), ..Default::default() };

    // ACT: save should create the file at the specified path
    cache.save().unwrap();

    // ASSERT
    assert!(cache_path.exists());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ========================================================================
// PUB-025: Cache save without cache_path (uses default_cache_path)
// ========================================================================

#[test]
fn test_pub_ext_025_cache_save_without_path_uses_default() {
    // ARRANGE: Create a cache with no cache_path set
    let cache = PublishStatusCache::default();
    assert!(cache.cache_path.is_none());

    // ACT: Saving should use default_cache_path and not panic
    // We don't assert success because the default path may or may not be writable
    // The important thing is coverage of the unwrap_or_else branch
    let _ = cache.save();
}

#[test]
fn test_pub_ext_025_cache_default_cache_path_format() {
    // ACT: The default path should end with the expected structure
    let path = PublishStatusCache::default_cache_path();

    // ASSERT: Should contain "batuta" and "publish-status.json"
    let path_str = path.to_string_lossy();
    assert!(
        path_str.contains("batuta"),
        "Default cache path should contain 'batuta': {}",
        path_str
    );
    assert!(
        path_str.ends_with("publish-status.json"),
        "Default cache path should end with 'publish-status.json': {}",
        path_str
    );
}

// ========================================================================
// PUB-026: get_git_head coverage (failure path)
// ========================================================================

#[test]
fn test_pub_ext_026_get_git_head_non_git_dir() {
    // ARRANGE: Create a temp dir that is NOT a git repo
    let temp_dir = std::env::temp_dir().join("batuta_ext_026_no_git");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    // ACT: get_git_head should fail for non-git directory
    let result = super::cache::get_git_head(&temp_dir);

    // ASSERT
    assert!(result.is_err(), "get_git_head should fail for non-git directory");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_pub_ext_026_get_git_head_valid_repo() {
    // ARRANGE: Create a git repo with a commit (disable pre-commit hook)
    let temp_dir = std::env::temp_dir().join("batuta_ext_026_valid_git");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let _ = std::process::Command::new("git").args(["init"]).current_dir(&temp_dir).output();
    let _ = std::process::Command::new("git")
        .args(["config", "user.email", "test@test.com"])
        .current_dir(&temp_dir)
        .output();
    let _ = std::process::Command::new("git")
        .args(["config", "user.name", "Test"])
        .current_dir(&temp_dir)
        .output();
    std::fs::write(temp_dir.join("test.txt"), "hello").unwrap();
    let _ = std::process::Command::new("git").args(["add", "."]).current_dir(&temp_dir).output();
    let _ = std::process::Command::new("git")
        .args(["commit", "--no-verify", "-m", "init"])
        .current_dir(&temp_dir)
        .output();

    // ACT
    let result = super::cache::get_git_head(&temp_dir);

    // ASSERT: Should return a short SHA
    assert!(result.is_ok());
    let sha = result.unwrap();
    assert!(!sha.is_empty(), "SHA should not be empty");
    assert!(sha.len() <= 12, "Short SHA should be at most 12 chars");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ========================================================================
// PUB-027: get_git_status line-level parsing coverage
// ========================================================================

#[test]
fn test_pub_ext_027_git_status_with_staged_files() {
    // ARRANGE: Create a git repo with staged files (disable pre-commit hook)
    let temp_dir = std::env::temp_dir().join("batuta_ext_027_staged");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let _ = std::process::Command::new("git").args(["init"]).current_dir(&temp_dir).output();
    let _ = std::process::Command::new("git")
        .args(["config", "user.email", "test@test.com"])
        .current_dir(&temp_dir)
        .output();
    let _ = std::process::Command::new("git")
        .args(["config", "user.name", "Test"])
        .current_dir(&temp_dir)
        .output();
    std::fs::write(temp_dir.join("file1.txt"), "hello").unwrap();
    let _ = std::process::Command::new("git")
        .args(["add", "file1.txt"])
        .current_dir(&temp_dir)
        .output();
    let _ = std::process::Command::new("git")
        .args(["commit", "--no-verify", "-m", "init"])
        .current_dir(&temp_dir)
        .output();

    // Create untracked file
    std::fs::write(temp_dir.join("untracked.txt"), "new").unwrap();
    // Modify existing file (unstaged)
    std::fs::write(temp_dir.join("file1.txt"), "modified").unwrap();
    // Create and stage another file
    std::fs::write(temp_dir.join("staged.txt"), "staged content").unwrap();
    let _ = std::process::Command::new("git")
        .args(["add", "staged.txt"])
        .current_dir(&temp_dir)
        .output();

    // ACT
    let status = super::git::get_git_status(&temp_dir);

    // ASSERT
    assert!(status.is_ok());
    let status = status.unwrap();
    assert!(!status.is_clean);
    assert!(status.untracked >= 1, "Should have at least 1 untracked file");
    assert!(status.staged >= 1, "Should have at least 1 staged file");
    assert!(status.modified >= 1, "Should have at least 1 modified file");
    assert!(!status.head_sha.is_empty());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_pub_ext_027_git_status_clean_repo() {
    // ARRANGE: Create a clean git repo (disable pre-commit hook)
    let temp_dir = std::env::temp_dir().join("batuta_ext_027_clean");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let _ = std::process::Command::new("git").args(["init"]).current_dir(&temp_dir).output();
    let _ = std::process::Command::new("git")
        .args(["config", "user.email", "test@test.com"])
        .current_dir(&temp_dir)
        .output();
    let _ = std::process::Command::new("git")
        .args(["config", "user.name", "Test"])
        .current_dir(&temp_dir)
        .output();
    std::fs::write(temp_dir.join("file.txt"), "hello").unwrap();
    let _ = std::process::Command::new("git").args(["add", "."]).current_dir(&temp_dir).output();
    let _ = std::process::Command::new("git")
        .args(["commit", "--no-verify", "-m", "init"])
        .current_dir(&temp_dir)
        .output();

    // ACT
    let status = super::git::get_git_status(&temp_dir);

    // ASSERT
    assert!(status.is_ok());
    let status = status.unwrap();
    assert!(status.is_clean);
    assert_eq!(status.total_changes(), 0);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_pub_ext_027_git_status_non_git_dir() {
    // ARRANGE: Non-git directory
    let temp_dir = std::env::temp_dir().join("batuta_ext_027_non_git");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    // ACT
    let result = super::git::get_git_status(&temp_dir);

    // ASSERT: Should fail
    assert!(result.is_err());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ========================================================================
// PUB-028: get_local_version edge cases for line-level coverage
// ========================================================================

#[test]
fn test_pub_ext_028_get_local_version_version_not_first_line() {
    // ARRANGE: version field is not on the first line
    let temp_dir = std::env::temp_dir().join("batuta_ext_028_version_later");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    std::fs::write(
        temp_dir.join("Cargo.toml"),
        "[package]\nname = \"test\"\nedition = \"2021\"\nversion = \"2.5.9\"\n",
    )
    .unwrap();

    // ACT
    let version = get_local_version(&temp_dir);

    // ASSERT
    assert!(version.is_ok());
    assert_eq!(version.unwrap(), "2.5.9");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_pub_ext_028_get_local_version_with_workspace_version() {
    // ARRANGE: Cargo.toml with version.workspace = true (no quoted version)
    let temp_dir = std::env::temp_dir().join("batuta_ext_028_workspace_ver");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    std::fs::write(
        temp_dir.join("Cargo.toml"),
        "[package]\nname = \"test\"\nversion.workspace = true\n",
    )
    .unwrap();

    // ACT: This file has 'version' keyword but no quoted string after split('"')
    let result = get_local_version(&temp_dir);

    // ASSERT: Should fail because there's no quoted version string
    assert!(result.is_err());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

// ========================================================================
// PUB-028b: PublishStatusCache save with explicit cache_path
// ========================================================================

#[test]
fn test_pub_ext_028b_cache_save_with_explicit_path_roundtrip() {
    // ARRANGE
    let temp_dir = std::env::temp_dir().join("batuta_ext_028b_save_explicit");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let cache_path = temp_dir.join("subdir").join("explicit-cache.json");

    let mut cache =
        PublishStatusCache { cache_path: Some(cache_path.clone()), ..Default::default() };

    // Insert an entry
    let entry = super::types::CacheEntry {
        cache_key: "explicit-key".to_string(),
        status: CrateStatus {
            name: "explicit-test".to_string(),
            local_version: Some("9.0.0".to_string()),
            crates_io_version: None,
            git_status: GitStatus::default(),
            action: PublishAction::NotPublished,
            path: PathBuf::from("."),
            error: None,
        },
        crates_io_checked_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        created_at: 0,
    };
    cache.insert("explicit-test".to_string(), entry);

    // ACT: save should create parent dirs and write file
    let result = cache.save();
    assert!(result.is_ok(), "save() with explicit path should succeed");

    // ASSERT: load_from should retrieve the saved data
    let loaded = PublishStatusCache::load_from(&cache_path).unwrap();
    assert!(loaded.get("explicit-test", "explicit-key").is_some());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_pub_ext_028b_cache_load_default() {
    // Exercise the load() method (uses default_cache_path internally)
    let cache = PublishStatusCache::load();
    // Should not panic; may or may not have entries depending on disk state
    let _ = cache.entries.len();
}

// ========================================================================
// PUB-029: determine_action additional edge cases
// ========================================================================

#[test]
fn test_pub_ext_029_determine_action_no_local_no_remote() {
    let git = GitStatus::default();
    let action = determine_action(None, None, &git);
    assert_eq!(action, PublishAction::Error);
}

#[test]
fn test_pub_ext_029_determine_action_not_published_clean() {
    let git = GitStatus { is_clean: true, ..Default::default() };
    let action = determine_action(Some("0.1.0"), None, &git);
    assert_eq!(action, PublishAction::NotPublished);
}

#[test]
fn test_pub_ext_029_determine_action_not_published_dirty() {
    let git = GitStatus { is_clean: false, untracked: 1, ..Default::default() };
    let action = determine_action(Some("0.1.0"), None, &git);
    assert_eq!(action, PublishAction::NeedsCommit);
}

#[test]
fn test_pub_ext_029_determine_action_different_versions() {
    let git = GitStatus { is_clean: true, ..Default::default() };
    // Local behind remote
    let action = determine_action(Some("1.0.0"), Some("1.0.1"), &git);
    assert_eq!(action, PublishAction::LocalBehind);
}
