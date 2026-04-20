//! Unit tests for [`super::WorktreeSession`].
//!
//! Tests shell out to a real `git` binary against a throwaway temp
//! repo. If `git` is not on PATH, every test falls through a guard
//! that marks the test as skipped-but-passing.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::*;

fn git_available() -> bool {
    Command::new("git").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn init_temp_repo() -> (tempfile::TempDir, PathBuf) {
    let td = tempfile::tempdir().expect("tempdir");
    let repo = td.path().to_path_buf();
    run_git(&repo, &["init", "--initial-branch=main"]);
    run_git(&repo, &["config", "user.email", "test@example.com"]);
    run_git(&repo, &["config", "user.name", "Test"]);
    // Disable any inherited hooks path (parent repo's pmat pre-commit hook
    // would otherwise run inside the temp repo's git commit).
    run_git(&repo, &["config", "core.hooksPath", "/dev/null"]);
    fs::write(repo.join("README.md"), "seed\n").expect("seed file");
    run_git(&repo, &["add", "README.md"]);
    run_git(&repo, &["commit", "--no-verify", "-m", "seed"]);
    (td, repo)
}

fn run_git(cwd: &Path, args: &[&str]) {
    let out = Command::new("git").arg("-C").arg(cwd).args(args).output().expect("git spawn");
    assert!(
        out.status.success(),
        "git {args:?} failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn create_fails_on_empty_branch_name() {
    if !git_available() {
        return;
    }
    let (_td, repo) = init_temp_repo();
    let err = WorktreeSession::create(&repo, "   ").unwrap_err();
    assert!(matches!(err, WorktreeError::EmptyBranchName));
}

#[test]
fn create_clean_worktree_and_auto_close_if_clean_cleans_up() {
    if !git_available() {
        return;
    }
    let (_td, repo) = init_temp_repo();
    let session = WorktreeSession::create(&repo, "agent/clean").expect("create worktree");
    let wt_path = session.path().to_path_buf();
    let wt_branch = session.branch().to_string();

    assert!(wt_path.exists(), "worktree path must exist");
    assert_eq!(wt_branch, "agent/clean");
    assert!(!session.is_dirty().expect("is_dirty probe"));

    let outcome = session.auto_close_if_clean().expect("auto close clean");
    assert!(outcome.is_none(), "clean worktree should be cleaned up");
    assert!(!wt_path.exists(), "worktree dir removed");

    // Branch should be gone.
    let out = Command::new("git")
        .arg("-C")
        .arg(&repo)
        .args(["branch", "--list", "agent/clean"])
        .output()
        .expect("git branch");
    assert!(String::from_utf8_lossy(&out.stdout).trim().is_empty(), "branch must be deleted");
}

#[test]
fn create_dirty_worktree_and_auto_close_if_clean_keeps_it() {
    if !git_available() {
        return;
    }
    let (_td, repo) = init_temp_repo();
    let session = WorktreeSession::create(&repo, "agent/dirty").expect("create worktree");
    let wt_path = session.path().to_path_buf();
    // Make the worktree dirty.
    fs::write(wt_path.join("scratch.txt"), "new file\n").expect("write scratch");

    assert!(session.is_dirty().expect("is_dirty probe"));

    let kept = session
        .auto_close_if_clean()
        .expect("auto close dirty")
        .expect("dirty session should NOT auto-close");
    assert_eq!(kept.1, "agent/dirty");
    assert!(wt_path.exists(), "dirty worktree must be retained");

    // Cleanup for the test's sake.
    let out = Command::new("git")
        .arg("-C")
        .arg(&repo)
        .args(["worktree", "remove", "--force", wt_path.to_str().unwrap()])
        .output()
        .expect("cleanup spawn");
    assert!(out.status.success(), "cleanup failed");
}

#[test]
fn keep_returns_path_and_branch_without_removing() {
    if !git_available() {
        return;
    }
    let (_td, repo) = init_temp_repo();
    let session = WorktreeSession::create(&repo, "agent/kept").expect("create worktree");
    let wt_path = session.path().to_path_buf();

    let (kept_path, kept_branch) = session.keep();
    assert_eq!(kept_path, wt_path);
    assert_eq!(kept_branch, "agent/kept");
    assert!(wt_path.exists(), "kept worktree must persist");

    // Cleanup.
    let _ = Command::new("git")
        .arg("-C")
        .arg(&repo)
        .args(["worktree", "remove", "--force", wt_path.to_str().unwrap()])
        .output();
}

#[test]
fn auto_close_removes_even_if_dirty() {
    if !git_available() {
        return;
    }
    let (_td, repo) = init_temp_repo();
    let session = WorktreeSession::create(&repo, "agent/force").expect("create worktree");
    let wt_path = session.path().to_path_buf();
    fs::write(wt_path.join("scratch.txt"), "dirty\n").expect("write scratch");

    session.auto_close().expect("forced close");
    assert!(!wt_path.exists(), "worktree removed despite dirty state");
}

#[test]
fn path_derivation_sanitizes_unsafe_chars() {
    let p = worktree_path_for(Path::new("/tmp/repo"), "feature/xy/z");
    let tail = p.file_name().unwrap().to_string_lossy().to_string();
    assert_eq!(tail, "feature-xy-z", "slashes sanitized to dashes");
    assert!(p.to_string_lossy().contains(".git/apr-worktrees"));
}

#[test]
fn error_display_messages_are_informative() {
    assert!(WorktreeError::EmptyBranchName.to_string().contains("branch"));
    assert!(WorktreeError::CreateFailed("x".into()).to_string().contains("worktree add"));
    assert!(WorktreeError::RemoveFailed("x".into()).to_string().contains("worktree remove"));
    assert!(WorktreeError::BranchDeleteFailed("x".into()).to_string().contains("branch"));
    assert!(WorktreeError::StatusFailed("x".into()).to_string().contains("status"));
    assert!(WorktreeError::SpawnFailed("x".into()).to_string().contains("spawn"));
}

#[test]
fn repo_root_accessor_returns_input_path() {
    if !git_available() {
        return;
    }
    let (_td, repo) = init_temp_repo();
    let session = WorktreeSession::create(&repo, "agent/root-probe").expect("create worktree");
    assert_eq!(session.repo_root(), repo.as_path());
    session.auto_close().expect("cleanup");
}
