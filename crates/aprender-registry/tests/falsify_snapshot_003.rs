//! FALSIFY-SNAPSHOT-003 — `Registry::snapshot()` refuses to overwrite an
//! existing target file. SQLite's `VACUUM INTO` is intentionally not
//! idempotent on the destination; callers must rotate filenames. We
//! surface the refusal as `Err(_)` instead of silently truncating.
//!
//! Contract: `contracts/apr-registry-snapshot-v1.yaml`.

#![allow(clippy::unwrap_used)]

use pacha::model::{ModelCard, ModelVersion};
use pacha::registry::{Registry, RegistryConfig};
use std::fs;
use tempfile::TempDir;

#[test]
fn snapshot_refuses_to_overwrite_existing_file() {
    let src_dir = TempDir::new().unwrap();
    let registry = Registry::open(RegistryConfig::new(src_dir.path())).unwrap();
    registry
        .register_model("marker", &ModelVersion::new(1, 0, 0), b"data", ModelCard::new("marker"))
        .unwrap();

    let target_dir = TempDir::new().unwrap();
    let target = target_dir.path().join("backup.db");
    fs::write(&target, b"i was here first").unwrap();
    let prior = fs::read(&target).unwrap();

    let result = registry.snapshot(&target);
    assert!(
        result.is_err(),
        "snapshot must error when target already exists; got Ok(_) — VACUUM INTO would have clobbered the prior file",
    );

    // Sanity: the prior file content is untouched.
    let post = fs::read(&target).unwrap();
    assert_eq!(post, prior, "VACUUM INTO must not partially overwrite");
}

#[test]
fn snapshot_target_directory_must_exist() {
    // Indirect corollary: VACUUM INTO into a path under a non-existent
    // directory must error rather than create the directory tree. This
    // pins the contract's "callers manage paths explicitly" stance.
    let src_dir = TempDir::new().unwrap();
    let registry = Registry::open(RegistryConfig::new(src_dir.path())).unwrap();

    let bogus = src_dir.path().join("does/not/exist/snap.db");
    let result = registry.snapshot(&bogus);
    assert!(result.is_err(), "snapshot must error when target's parent directory is missing");
}

#[test]
fn snapshot_to_fresh_path_succeeds_after_overwrite_refusal() {
    // Operationally important: a single failed snapshot (target exists)
    // must not poison the source registry — the next call to a fresh
    // path succeeds.
    let src_dir = TempDir::new().unwrap();
    let registry = Registry::open(RegistryConfig::new(src_dir.path())).unwrap();
    registry
        .register_model("marker", &ModelVersion::new(1, 0, 0), b"data", ModelCard::new("marker"))
        .unwrap();

    let target_dir = TempDir::new().unwrap();
    let occupied = target_dir.path().join("occupied.db");
    fs::write(&occupied, b"hello").unwrap();
    assert!(registry.snapshot(&occupied).is_err());

    let fresh = target_dir.path().join("fresh.db");
    registry.snapshot(&fresh).unwrap();
    assert!(fresh.is_file());
}
