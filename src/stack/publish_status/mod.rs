//! Publish Status Scanner with O(1) Cache
//!
//! Scans PAIML stack repositories for publish status with intelligent caching.
//! Uses content-addressable cache keys to achieve O(1) lookups for unchanged repos.
//!
//! ## Cache Invalidation
//!
//! Cache keys are computed from:
//! - Cargo.toml content hash (blake3)
//! - Git HEAD commit SHA
//! - Worktree modification time
//!
//! crates.io versions are cached with 15-minute TTL.
//!
//! ## Performance Target
//!
//! - Cold cache: <5s (parallel fetches)
//! - Warm cache: <100ms (O(1) hash checks)

mod cache;
mod format;
mod git;
mod scanner;
mod types;

#[cfg(test)]
mod proptests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_extended;

// Re-export all public types and functions
pub use cache::{compute_cache_key, PublishStatusCache};
pub use format::{format_report_json, format_report_text};
pub use git::{determine_action, get_git_status, get_local_version};
pub use scanner::PublishStatusScanner;
pub use types::{CacheEntry, CrateStatus, GitStatus, PublishAction, PublishStatusReport};
