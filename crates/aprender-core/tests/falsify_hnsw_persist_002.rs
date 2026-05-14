//! FALSIFY-HNSW-PERSIST-002 — `PersistentHnsw::flush()` is atomic via
//! temp-file + fsync + rename. A crash before the rename leaves the
//! main snapshot file untouched; a corruption *of* the main file is
//! surfaced as `Err(_)`, never as a usable-looking-but-lying index.
//!
//! Contract: `contracts/apr-hnsw-persistence-v1.yaml` v1.1.0.
//!
//! Discharge strategy is multi-pronged because "atomicity" is hard to
//! prove with one assertion:
//!
//! 1. **Garbage in `.tmp` sibling** — simulate "crash during temp
//!    write" by manually scribbling bytes into `<path>.tmp`, then
//!    `open(<path>)` returns the previous good index unaffected.
//! 2. **Corruption of `<path>` itself** — write garbage directly to
//!    `<path>` and assert `open()` returns `PersistentHnswError::Decode`.
//! 3. **Structural source check** — `flush()` calls `fs::rename`,
//!    not `fs::write` direct. A drive-by refactor that drops the
//!    rename fails this gate at the source level even if the runtime
//!    tests happen to pass on a particular OS.

#![allow(clippy::unwrap_used)]

use aprender::index::{PersistentHnsw, PersistentHnswError};
use aprender::primitives::Vector;
use std::fs;
use tempfile::tempdir;

const AUTH_SOURCE: &str = include_str!("../src/index/persistent_hnsw.rs");

fn fixture(idx: &mut PersistentHnsw) {
    idx.add("a", Vector::from_slice(&[1.0, 0.0, 0.0]));
    idx.add("b", Vector::from_slice(&[0.0, 1.0, 0.0]));
    idx.add("c", Vector::from_slice(&[0.0, 0.0, 1.0]));
    idx.add("d", Vector::from_slice(&[0.6, 0.6, 0.0]));
}

#[test]
fn partial_write_does_not_silently_corrupt() {
    // Scenario 1: garbage in `.tmp` does NOT poison the main snapshot.
    let dir = tempdir().unwrap();
    let path = dir.path().join("snap.bin");
    let tmp = dir.path().join("snap.bin.tmp");

    // Write a known-good snapshot.
    let mut idx = PersistentHnsw::open(&path, 8, 64).unwrap();
    fixture(&mut idx);
    let baseline = idx.search(&Vector::from_slice(&[0.9, 0.1, 0.0]), 3);
    idx.flush().unwrap();
    drop(idx);

    // Manually scribble garbage into the temp sibling. This simulates
    // "process killed mid-temp-write".
    fs::write(&tmp, b"\xff\xfe\xfd\xfc partial bincode garbage").unwrap();

    // Open the main path. The garbage in `.tmp` MUST NOT be read.
    let reopened = PersistentHnsw::open(&path, 8, 64).unwrap();
    let got = reopened.search(&Vector::from_slice(&[0.9, 0.1, 0.0]), 3);
    assert_eq!(
        got, baseline,
        "garbage in <path>.tmp must not affect open(<path>) — that would mean \
         flush()'s rename is being ignored or open() is reading the wrong file."
    );
}

#[test]
fn corruption_of_main_path_returns_decode_error() {
    // Scenario 2: bytes that are NOT a valid bincode HNSWIndex must
    // surface as Err(Decode), never decode into a "valid-looking"
    // index that lies on read.
    let dir = tempdir().unwrap();
    let path = dir.path().join("snap.bin");
    fs::write(&path, b"definitely not a bincode payload at all").unwrap();
    let result = PersistentHnsw::open(&path, 8, 64);
    assert!(
        matches!(result, Err(PersistentHnswError::Decode(_))),
        "corrupt main snapshot must error, got: {:?}",
        result.as_ref().err(),
    );
}

#[test]
fn truncated_main_path_returns_decode_error() {
    // Scenario 2b: a *truncated* main file (the "looks like the start
    // of a valid bincode payload but stops mid-way" case). bincode
    // detects the EOF mid-deserialization and returns Err.
    let dir = tempdir().unwrap();
    let path = dir.path().join("trunc.bin");

    // Build a real snapshot.
    let mut idx = PersistentHnsw::open(&path, 8, 64).unwrap();
    fixture(&mut idx);
    idx.flush().unwrap();
    drop(idx);

    // Truncate to half-size to simulate "crash after partial fsync".
    let bytes = fs::read(&path).unwrap();
    fs::write(&path, &bytes[..bytes.len() / 2]).unwrap();

    let result = PersistentHnsw::open(&path, 8, 64);
    assert!(
        matches!(result, Err(PersistentHnswError::Decode(_))),
        "truncated snapshot must error, got: {:?}",
        result.as_ref().err(),
    );
}

#[test]
fn flush_implementation_uses_atomic_rename() {
    // Scenario 3: structural source check. `flush()` MUST go through
    // a temp file + rename, not fs::write directly to the snapshot
    // path. A regression that drops the rename also drops atomicity.
    assert!(
        AUTH_SOURCE.contains("fs::rename") || AUTH_SOURCE.contains("std::fs::rename"),
        "persistent_hnsw.rs::flush MUST call `fs::rename` for atomic \
         crash safety (FALSIFY-HNSW-PERSIST-002). If renaming was \
         extracted to a helper, that helper must live in this module \
         so this gate keeps catching regressions.",
    );
    // And it MUST NOT call `fs::write(&self.path, ...)` directly,
    // which would defeat atomicity.
    assert!(
        !AUTH_SOURCE.contains("fs::write(&self.path"),
        "persistent_hnsw.rs::flush MUST NOT write directly to self.path; \
         use a temp + rename pattern instead."
    );
}

#[test]
fn flush_implementation_calls_sync_all() {
    // Scenario 3b: data must be fsync'd to disk *before* the rename
    // — without fsync, the rename can succeed while page-cache
    // contents are unflushed, allowing a power-loss to lose recent
    // writes despite the rename succeeding.
    assert!(
        AUTH_SOURCE.contains(".sync_all()"),
        "persistent_hnsw.rs::flush MUST call sync_all() on the temp \
         file handle before the rename — required by \
         FALSIFY-HNSW-PERSIST-002 to prevent power-loss data loss."
    );
}

#[test]
fn previous_snapshot_intact_after_failed_open() {
    // Scenario: a corrupt main file is detected by open(); a
    // subsequent FRESH flush from a NEW handle (built from in-memory
    // data the caller still has) must succeed and replace the
    // corrupt main with a good one. This shows the atomic-rename
    // path is also the recovery path.
    let dir = tempdir().unwrap();
    let path = dir.path().join("recover.bin");

    // Step 1: write garbage to simulate a corrupt prior file.
    fs::write(&path, b"corrupt").unwrap();
    assert!(matches!(
        PersistentHnsw::open(&path, 8, 64),
        Err(PersistentHnswError::Decode(_)),
    ));

    // Step 2: caller decides to recreate from in-memory data. Wipe
    // the corrupt file (the operator's call) and re-open fresh.
    fs::remove_file(&path).unwrap();
    let mut idx = PersistentHnsw::open(&path, 8, 64).unwrap();
    fixture(&mut idx);
    idx.flush().unwrap();
    drop(idx);

    // Step 3: reopen — should now succeed cleanly.
    let recovered = PersistentHnsw::open(&path, 8, 64).unwrap();
    assert_eq!(recovered.len(), 4);
}
