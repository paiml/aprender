//! Core types for publish status tracking.
//!
//! This module defines the foundational types used throughout the publish
//! status system, including actions, git status, and crate status.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// PUB-001: Core Types
// ============================================================================

/// Recommended action for a crate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PublishAction {
    /// Already published and up to date
    UpToDate,
    /// Has uncommitted changes, needs commit first
    NeedsCommit,
    /// Committed but not published
    NeedsPublish,
    /// Local version behind crates.io (unusual)
    LocalBehind,
    /// Not yet published to crates.io
    NotPublished,
    /// Error checking status
    Error,
}
impl PublishAction {
    /// Get display symbol for action
    #[must_use]
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::UpToDate => "✓",
            Self::NeedsCommit => "📝",
            Self::NeedsPublish => "📦",
            Self::LocalBehind => "⚠️",
            Self::NotPublished => "🆕",
            Self::Error => "❌",
        }
    }

    /// Get action description
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::UpToDate => "up to date",
            Self::NeedsCommit => "commit changes",
            Self::NeedsPublish => "PUBLISH",
            Self::LocalBehind => "local behind",
            Self::NotPublished => "not published",
            Self::Error => "error",
        }
    }
}

/// Git status summary for a repo
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitStatus {
    /// Number of modified files
    pub modified: usize,
    /// Number of untracked files
    pub untracked: usize,
    /// Number of staged files
    pub staged: usize,
    /// Current HEAD commit SHA (short)
    pub head_sha: String,
    /// Is repo clean?
    pub is_clean: bool,
}

impl GitStatus {
    /// Total changed files
    #[must_use]
    pub fn total_changes(&self) -> usize {
        self.modified + self.untracked + self.staged
    }

    /// Summary string
    #[must_use]
    pub fn summary(&self) -> String {
        if self.is_clean {
            "clean".to_string()
        } else {
            let mut parts = Vec::new();
            if self.modified > 0 {
                parts.push(format!("{}M", self.modified));
            }
            if self.untracked > 0 {
                parts.push(format!("{}?", self.untracked));
            }
            if self.staged > 0 {
                parts.push(format!("{}+", self.staged));
            }
            parts.join(" ")
        }
    }
}

/// Status of a single crate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrateStatus {
    /// Crate name
    pub name: String,
    /// Local version from Cargo.toml
    pub local_version: Option<String>,
    /// Published version on crates.io
    pub crates_io_version: Option<String>,
    /// Git status
    pub git_status: GitStatus,
    /// Recommended action
    pub action: PublishAction,
    /// Path to repo
    pub path: PathBuf,
    /// Error message if any
    pub error: Option<String>,
}

/// Cache entry for a single repo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Cache key (hash of Cargo.toml + git HEAD)
    pub cache_key: String,
    /// Cached status
    pub status: CrateStatus,
    /// When crates.io was last checked (Unix timestamp)
    pub crates_io_checked_at: u64,
    /// When this entry was created
    pub created_at: u64,
}

impl CacheEntry {
    /// Check if crates.io data is stale (>15 min)
    #[must_use]
    pub fn is_crates_io_stale(&self) -> bool {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        now - self.crates_io_checked_at > 15 * 60
    }
}

/// Full publish status report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishStatusReport {
    /// Status for each crate
    pub crates: Vec<CrateStatus>,
    /// Total crates checked
    pub total: usize,
    /// Crates needing publish
    pub needs_publish: usize,
    /// Crates needing commit
    pub needs_commit: usize,
    /// Crates up to date
    pub up_to_date: usize,
    /// Cache hits (O(1) lookups)
    pub cache_hits: usize,
    /// Cache misses (required refresh)
    pub cache_misses: usize,
    /// Time to generate report (ms)
    pub elapsed_ms: u64,
}

impl PublishStatusReport {
    /// Create report from crate statuses
    #[must_use]
    pub fn from_statuses(crates: Vec<CrateStatus>, cache_hits: usize, elapsed_ms: u64) -> Self {
        let total = crates.len();
        let needs_publish =
            crates.iter().filter(|c| c.action == PublishAction::NeedsPublish).count();
        let needs_commit = crates.iter().filter(|c| c.action == PublishAction::NeedsCommit).count();
        let up_to_date = crates.iter().filter(|c| c.action == PublishAction::UpToDate).count();
        let cache_misses = total - cache_hits;

        Self {
            crates,
            total,
            needs_publish,
            needs_commit,
            up_to_date,
            cache_hits,
            cache_misses,
            elapsed_ms,
        }
    }
}
