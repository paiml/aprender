//! Falsifier for the `DEVICE_INIT_LOCK` reentrancy self-deadlock.
//!
//! # Why this is its own test binary
//!
//! The bug only fires when the call is the FIRST thing in the process to reach
//! `shared_instance()`, because that is the only time the `OnceLock` initializer
//! actually runs. `GpuDevice::is_available()` reaches `shared_instance()`
//! WITHOUT holding `DEVICE_INIT_LOCK`, so any earlier `is_available()` primes
//! the `OnceLock` and the reentrant acquisition never happens.
//!
//! `cargo test` runs the tests of one binary concurrently in ONE process, so a
//! probe living beside other GPU tests loses the race non-deterministically and
//! stops proving anything. Cargo gives each `tests/*.rs` file its own process,
//! so this file — holding exactly one test — is the only construction in which
//! "first GPU call in the process" is guaranteed rather than hoped for.
//!
//! # Why the timeout
//!
//! The regression is a HANG, not a panic. `#[should_panic]` cannot express it
//! and a test that simply deadlocks is an outage, not a falsifier: it burns the
//! job's whole `timeout-minutes` and reports "cancelled", which reads as
//! infrastructure flake rather than as a defect. So the call is made on a
//! worker thread and the main thread waits on a channel with a deadline; a
//! regression fails in `DEADLINE` with a diagnosis instead of hanging.
//!
//! Verified to turn RED: with the `DEVICE_INIT_LOCK.lock()` restored inside
//! `shared_instance()`'s `get_or_init` closure, this test fails on the deadline
//! (see the PR body for the recorded run); with the fix it completes.
#![cfg(all(feature = "gpu", not(target_arch = "wasm32")))]

use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Generous relative to a real enumeration (measured ~0.2 s on the RTX 4090
/// host, ~1 s cold) and still far below any CI job timeout.
const DEADLINE: Duration = Duration::from_secs(60);

/// `list_adapters()` must return when it is the first GPU call in the process.
///
/// `list_adapters()` takes `DEVICE_INIT_LOCK` and then calls `shared_instance()`,
/// whose `OnceLock` initializer used to take the SAME non-reentrant
/// `std::sync::Mutex`. First-call ⟹ initializer runs ⟹ self-deadlock, forever.
///
/// This test asserts nothing about the CONTENT of the list: a host with no GPU
/// legitimately returns zero adapters, and the deadlock is independent of
/// whether any adapter exists — the hang is in acquiring the instance, before
/// enumeration. Adapter identity is asserted by `gpu_vulkan_falsifier.rs`.
#[test]
fn list_adapters_returns_when_it_is_the_first_gpu_call() {
    let (tx, rx) = mpsc::channel();
    let started = Instant::now();

    // Must be a plain thread, not a scoped one: a scoped thread's `join` on
    // scope exit would re-introduce the very hang this test exists to report.
    std::thread::Builder::new()
        .name("first-gpu-call".into())
        .spawn(move || {
            let adapters = trueno::backends::gpu::GpuDevice::list_adapters();
            // Ignore send failure: the receiver is gone only if we already
            // blew the deadline, and the test has failed by then anyway.
            let _ = tx.send(adapters);
        })
        .expect("failed to spawn probe thread");

    match rx.recv_timeout(DEADLINE) {
        Ok(adapters) => {
            eprintln!(
                "list_adapters() returned {} adapter(s) in {:.3}s as the first GPU call",
                adapters.len(),
                started.elapsed().as_secs_f64()
            );
            for (idx, name, backend) in &adapters {
                eprintln!("  [{idx}] name={name:?} backend={backend}");
            }
        }
        Err(mpsc::RecvTimeoutError::Timeout) => panic!(
            "GpuDevice::list_adapters() did not return within {}s as the first GPU call in \
             the process. This is the DEVICE_INIT_LOCK reentrancy self-deadlock: \
             list_adapters() holds the lock and shared_instance()'s OnceLock initializer \
             re-acquires it. std::sync::Mutex is not reentrant.",
            DEADLINE.as_secs()
        ),
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            panic!("probe thread died without returning a result — see its panic message above")
        }
    }
}
