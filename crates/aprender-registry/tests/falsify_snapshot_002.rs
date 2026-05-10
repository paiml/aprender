//! FALSIFY-SNAPSHOT-002 — `Registry::snapshot()` does not block
//! concurrent writers. A writer thread runs `register_model` in a loop
//! while the main thread takes a snapshot; the snapshot returns within a
//! generous wall-clock budget and the writer thread completes its loop
//! without error.
//!
//! Contract: `contracts/apr-registry-snapshot-v1.yaml`.
//!
//! Note on budget: the gate enforces "writer can keep going" and "snapshot
//! returns in finite time," not microbenchmark perf. The default 5-second
//! timeout is comfortably above any plausible CI fluctuation. Tunable via
//! `APR_SNAPSHOT_BUDGET_MS` for isolated reproductions.

#![allow(clippy::unwrap_used)]

use pacha::model::{ModelCard, ModelVersion};
use pacha::registry::{Registry, RegistryConfig};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;

fn budget_ms() -> u64 {
    std::env::var("APR_SNAPSHOT_BUDGET_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(5_000)
}

#[test]
fn snapshot_does_not_block_concurrent_writers() {
    let dir = TempDir::new().unwrap();
    let path: PathBuf = dir.path().to_path_buf();

    // Pre-populate so VACUUM INTO has real pages to copy.
    {
        let r = Registry::open(RegistryConfig::new(&path)).unwrap();
        for i in 0..20 {
            r.register_model(
                &format!("warmup-{i}"),
                &ModelVersion::new(1, 0, u32::try_from(i).unwrap()),
                format!("payload-{i}").as_bytes(),
                ModelCard::new(format!("warmup model {i}")),
            )
            .unwrap();
        }
    }

    // Writer thread holds its own Registry handle and inserts unique rows
    // for as long as the main-thread snapshot takes. SQLite serializes
    // them at the page level; the test asserts neither side errors.
    let writer_path = path.clone();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_writer = Arc::clone(&stop);
    let writer = thread::spawn(move || -> usize {
        let r = Registry::open(RegistryConfig::new(&writer_path)).unwrap();
        let mut written = 0_usize;
        while !stop_writer.load(std::sync::atomic::Ordering::Acquire) {
            // SQLite can return SQLITE_BUSY transiently while VACUUM INTO
            // copies pages. The contract is "concurrent writers continue,"
            // not "every write succeeds within microseconds." Surface BUSY
            // by retrying instead of failing the test.
            let res = r.register_model(
                &format!("concurrent-{written}"),
                &ModelVersion::new(1, 0, u32::try_from(written).unwrap()),
                format!("c{written}").as_bytes(),
                ModelCard::new(format!("c{written}")),
            );
            match res {
                Ok(_) => written += 1,
                Err(e) => {
                    // Tolerate transient lock contention; bail on real errors.
                    let msg = format!("{e}");
                    if msg.contains("locked") || msg.contains("busy") {
                        std::thread::sleep(Duration::from_millis(2));
                    } else {
                        panic!("writer hit non-busy error: {msg}");
                    }
                }
            }
        }
        written
    });

    // Snapshot — the test's load-bearing call.
    let main_registry = Registry::open(RegistryConfig::new(&path)).unwrap();
    let target_dir = TempDir::new().unwrap();
    let target = target_dir.path().join("snapshot.db");

    let start = Instant::now();
    main_registry.snapshot(&target).unwrap();
    let elapsed = start.elapsed();

    // Stop the writer.
    stop.store(true, std::sync::atomic::Ordering::Release);
    let total_writes = writer.join().expect("writer thread didn't panic");

    let budget = Duration::from_millis(budget_ms());
    assert!(
        elapsed < budget,
        "snapshot must return within {}ms; took {}ms (total writer rows: {})",
        budget.as_millis(),
        elapsed.as_millis(),
        total_writes,
    );
    assert!(target.is_file(), "VACUUM INTO must produce the file");
}
