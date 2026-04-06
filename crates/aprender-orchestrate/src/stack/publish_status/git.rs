//! Git status parsing and version functions.
//!
//! Provides utilities for parsing git status and extracting version
//! information from Cargo.toml files.

use anyhow::{anyhow, Result};
use std::path::Path;

use super::cache::get_git_head;
use super::types::{GitStatus, PublishAction};

// ============================================================================
// PUB-004: Git Status Parsing
// ============================================================================

/// Get git status for a repo
pub fn get_git_status(repo_path: &Path) -> Result<GitStatus> {
    let output = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(repo_path)
        .output()?;

    if !output.status.success() {
        return Err(anyhow!("git status failed"));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut status = GitStatus::default();

    for line in stdout.lines() {
        if line.len() < 2 {
            continue;
        }
        let index = line.chars().next().unwrap_or(' ');
        let worktree = line.chars().nth(1).unwrap_or(' ');

        match (index, worktree) {
            ('?', '?') => status.untracked += 1,
            (i, _) if i != ' ' && i != '?' => status.staged += 1,
            (_, w) if w != ' ' && w != '?' => status.modified += 1,
            _ => {}
        }
    }

    status.is_clean = status.total_changes() == 0;
    status.head_sha = get_git_head(repo_path).unwrap_or_default();

    Ok(status)
}

// ============================================================================
// PUB-005: Version Parsing
// ============================================================================

/// Extract version from Cargo.toml
pub fn get_local_version(repo_path: &Path) -> Result<String> {
    let cargo_toml = repo_path.join("Cargo.toml");
    let content = std::fs::read_to_string(&cargo_toml)?;

    for line in content.lines() {
        if line.starts_with("version") {
            if let Some(version) = line.split('"').nth(1) {
                return Ok(version.to_string());
            }
        }
    }

    Err(anyhow!("No version found in Cargo.toml"))
}

/// Compare versions and determine action
pub fn determine_action(
    local: Option<&str>,
    crates_io: Option<&str>,
    git_status: &GitStatus,
) -> PublishAction {
    match (local, crates_io) {
        (None, _) => PublishAction::Error,
        (Some(_), None) => {
            if git_status.is_clean {
                PublishAction::NotPublished
            } else {
                PublishAction::NeedsCommit
            }
        }
        (Some(local), Some(remote)) => {
            if !git_status.is_clean {
                PublishAction::NeedsCommit
            } else if local == remote {
                PublishAction::UpToDate
            } else {
                // Parse and compare versions
                match (semver::Version::parse(local), semver::Version::parse(remote)) {
                    (Ok(l), Ok(r)) if l > r => PublishAction::NeedsPublish,
                    (Ok(l), Ok(r)) if l < r => PublishAction::LocalBehind,
                    _ => PublishAction::UpToDate,
                }
            }
        }
    }
}
