//! Git worktree isolation for subagents (Claude-Code parity).
//!
//! PMAT-CODE-WORKTREE-001: Claude Code's Agent tool accepts an
//! `isolation: "worktree"` flag. When set, the runtime creates a
//! fresh `git worktree` checked out to a new branch, the agent runs
//! against that working tree, and at completion:
//!
//! - if the worktree is clean (no changes), it is removed and the
//!   branch is deleted — the caller sees no artifact.
//! - if the worktree is dirty, both the worktree path and the
//!   branch are returned so the caller can inspect or merge.
//!
//! This module ships the primitives (`WorktreeSession::create`,
//! `.is_dirty()`, `.auto_close_if_clean()`, `.keep()`) so
//! spawn-tool call sites can opt in.
//!
//! # Example
//!
//! ```rust,ignore
//! use batuta::agent::worktree::WorktreeSession;
//!
//! let repo = std::path::Path::new(".");
//! let session = WorktreeSession::create(repo, "agent/xyz")?;
//! // ...agent runs, edits files under session.path()...
//! if session.is_dirty()? {
//!     let kept = session.keep()?;  // returns (path, branch)
//!     println!("worktree kept at {} on branch {}", kept.0.display(), kept.1);
//! } else {
//!     session.auto_close()?;  // removes worktree + deletes branch
//! }
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

/// Errors arising from worktree lifecycle operations.
#[derive(Debug)]
pub enum WorktreeError {
    /// `git worktree add` failed.
    CreateFailed(String),
    /// `git worktree remove` failed.
    RemoveFailed(String),
    /// `git branch -D` failed.
    BranchDeleteFailed(String),
    /// `git status --porcelain` failed.
    StatusFailed(String),
    /// Spawn of the underlying git process failed (e.g. git not on PATH).
    SpawnFailed(String),
    /// Supplied branch name is empty.
    EmptyBranchName,
}

impl std::fmt::Display for WorktreeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CreateFailed(msg) => write!(f, "git worktree add failed: {msg}"),
            Self::RemoveFailed(msg) => write!(f, "git worktree remove failed: {msg}"),
            Self::BranchDeleteFailed(msg) => write!(f, "git branch -D failed: {msg}"),
            Self::StatusFailed(msg) => write!(f, "git status --porcelain failed: {msg}"),
            Self::SpawnFailed(msg) => write!(f, "failed to spawn git: {msg}"),
            Self::EmptyBranchName => write!(f, "branch name must be non-empty"),
        }
    }
}

impl std::error::Error for WorktreeError {}

/// Active git worktree session scoped to a subagent.
///
/// On drop, nothing happens by design — the caller must choose
/// `auto_close`, `auto_close_if_clean`, or `keep`. This forces an
/// explicit disposition (Poka-Yoke) rather than silent cleanup
/// that might discard agent output.
#[derive(Debug)]
pub struct WorktreeSession {
    /// Path to the worktree directory.
    path: PathBuf,
    /// Branch checked out in the worktree.
    branch: String,
    /// Repository the worktree was forked from.
    repo_root: PathBuf,
}

impl WorktreeSession {
    /// Create a new worktree at a path derived from the branch name.
    ///
    /// Worktree is placed at `<repo>/.git/apr-worktrees/<sanitized-branch>`.
    /// The branch is created fresh from HEAD.
    pub fn create(repo_root: &Path, branch: &str) -> Result<Self, WorktreeError> {
        if branch.trim().is_empty() {
            return Err(WorktreeError::EmptyBranchName);
        }
        let worktree_path = worktree_path_for(repo_root, branch);
        let output = Command::new("git")
            .arg("-C")
            .arg(repo_root)
            .arg("worktree")
            .arg("add")
            .arg("-b")
            .arg(branch)
            .arg(&worktree_path)
            .output()
            .map_err(|e| WorktreeError::SpawnFailed(e.to_string()))?;

        if !output.status.success() {
            return Err(WorktreeError::CreateFailed(
                String::from_utf8_lossy(&output.stderr).into(),
            ));
        }

        Ok(Self {
            path: worktree_path,
            branch: branch.to_string(),
            repo_root: repo_root.to_path_buf(),
        })
    }

    /// Path to the worktree directory (pass as cwd to the subagent).
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Branch name checked out in the worktree.
    pub fn branch(&self) -> &str {
        &self.branch
    }

    /// Repository root the worktree was forked from.
    pub fn repo_root(&self) -> &Path {
        &self.repo_root
    }

    /// Check whether the worktree has any uncommitted changes.
    ///
    /// Uses `git status --porcelain` — any non-empty output is dirty.
    pub fn is_dirty(&self) -> Result<bool, WorktreeError> {
        let output = Command::new("git")
            .arg("-C")
            .arg(&self.path)
            .arg("status")
            .arg("--porcelain")
            .output()
            .map_err(|e| WorktreeError::SpawnFailed(e.to_string()))?;

        if !output.status.success() {
            return Err(WorktreeError::StatusFailed(
                String::from_utf8_lossy(&output.stderr).into(),
            ));
        }

        Ok(!output.stdout.is_empty())
    }

    /// Remove the worktree and delete its branch.
    ///
    /// Safe to call whether clean or dirty. Caller must have
    /// decided they don't want to keep the work.
    pub fn auto_close(self) -> Result<(), WorktreeError> {
        remove_worktree(&self.repo_root, &self.path)?;
        delete_branch(&self.repo_root, &self.branch)?;
        Ok(())
    }

    /// Close the worktree only if clean; otherwise keep it.
    ///
    /// Returns `Ok(None)` if cleanup ran (clean case), or
    /// `Ok(Some((path, branch)))` if the caller should keep the
    /// worktree (dirty case).
    pub fn auto_close_if_clean(self) -> Result<Option<(PathBuf, String)>, WorktreeError> {
        if self.is_dirty()? {
            Ok(Some((self.path, self.branch)))
        } else {
            let path = self.path.clone();
            let branch = self.branch.clone();
            remove_worktree(&self.repo_root, &path)?;
            delete_branch(&self.repo_root, &branch)?;
            Ok(None)
        }
    }

    /// Explicitly keep the worktree, returning `(path, branch)`.
    ///
    /// The caller is now responsible for future cleanup.
    pub fn keep(self) -> (PathBuf, String) {
        (self.path, self.branch)
    }
}

fn remove_worktree(repo_root: &Path, path: &Path) -> Result<(), WorktreeError> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .arg("worktree")
        .arg("remove")
        .arg("--force")
        .arg(path)
        .output()
        .map_err(|e| WorktreeError::SpawnFailed(e.to_string()))?;

    if !output.status.success() {
        return Err(WorktreeError::RemoveFailed(String::from_utf8_lossy(&output.stderr).into()));
    }
    Ok(())
}

fn delete_branch(repo_root: &Path, branch: &str) -> Result<(), WorktreeError> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .arg("branch")
        .arg("-D")
        .arg(branch)
        .output()
        .map_err(|e| WorktreeError::SpawnFailed(e.to_string()))?;

    if !output.status.success() {
        return Err(WorktreeError::BranchDeleteFailed(
            String::from_utf8_lossy(&output.stderr).into(),
        ));
    }
    Ok(())
}

fn worktree_path_for(repo_root: &Path, branch: &str) -> PathBuf {
    let sanitized: String = branch
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '-' })
        .collect();
    repo_root.join(".git").join("apr-worktrees").join(sanitized)
}

#[cfg(test)]
mod tests;
