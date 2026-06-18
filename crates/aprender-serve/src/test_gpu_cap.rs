//! PMAT-779: process-global, cross-backend GPU-test concurrency cap (test-only).
//!
//! The aprender-serve `--features cuda` lib suite has ~16,920 tests; the default
//! libtest/nextest harness runs them ~20-way parallel. Two test clusters are
//! GPU-resident and use *different* GPU backends:
//!
//!   * `cuda::executor::*` (+ `CudaScheduler`, gguf cuda) → the **CUDA** driver
//!     backend, funneled through `CudaExecutor::new`.
//!   * `gpu::tests::*` (`GpuModel` / `forward_gpu`) → the **wgpu** backend,
//!     funneled through `HybridScheduler` (`GpuCompute::auto`).
//!
//! On the NVIDIA GB10 (Blackwell, sm_121, aarch64) running these clusters at full
//! parallelism collectively OVER-SUBSCRIBES the single physical GPU: a faulting /
//! spin-waiting GPU thread leaves the driver wedged on its internal real-time
//! mutex, and the whole 20-thread run deadlocks (one thread spins on a core, the
//! rest park in `futex`/`rt_mutex`). The clusters all pass in isolation /
//! single-threaded — this is purely concurrency over-subscription.
//!
//! A cap on *only* the CUDA backend is insufficient: the dominant `gpu::tests`
//! cluster is wgpu and bypasses `CudaExecutor` entirely (confirmed on GB10 —
//! serializing CUDA alone still hung, with a spinning `gpu::tests` thread). So the
//! permit is acquired by **both** GPU-resident entry points sharing ONE
//! process-global counting semaphore, capping concurrently-live GPU-resident test
//! objects across both backends to a safe `N`.
//!
//! ## Per-thread reentrancy (deadlock safety)
//!
//! A single test object can transitively build *several* GPU objects on one
//! thread — e.g. `GpuModel::new_with_cuda` constructs a `HybridScheduler` (wgpu)
//! **and** a `CudaScheduler`/`CudaExecutor` (CUDA) back-to-back. With a naive
//! counting semaphore, two such threads could each grab one permit and then
//! block forever waiting for their second — a classic multi-resource deadlock.
//!
//! To make the cap deadlock-free, each **thread** holds at most ONE real permit:
//! the first GPU object a test thread builds takes a semaphore permit; any nested
//! GPU object built on the same thread takes a cheap reentrant no-op. The real
//! permit is released only when the *outermost* GPU object on that thread drops.
//! So the semaphore counts concurrently-active GPU *test threads*, never more than
//! one per thread — exactly the quantity that over-subscribes the GPU.
//!
//! `N` defaults to 2 and is tunable without recompiling via
//! `REALIZAR_GPU_TEST_CONCURRENCY` (clamped to `>= 1`; `1` = full GPU-test
//! serialization). This entire module is `#[cfg(all(test, feature = "cuda"))]` —
//! the production `apr serve` path never compiles or executes it.

use std::cell::Cell;
use std::sync::{Condvar, Mutex, OnceLock};

/// Default cap on concurrently-active GPU-resident test *threads* (both backends).
///
/// Verified parallel-clean on the NVIDIA GB10 (Blackwell, sm_121) across repeated
/// full-suite runs at 20-way harness parallelism. `4` still intermittently tripped
/// an over-subscription kernel fault (`CUDA_ERROR_ILLEGAL_ADDRESS`); `2` is stable.
const DEFAULT_MAX_CONCURRENT: usize = 2;

struct CountingSemaphore {
    permits: Mutex<usize>,
    available: Condvar,
}

fn semaphore() -> &'static CountingSemaphore {
    static SEM: OnceLock<CountingSemaphore> = OnceLock::new();
    SEM.get_or_init(|| {
        let max = std::env::var("REALIZAR_GPU_TEST_CONCURRENCY")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .map(|n| n.max(1))
            .unwrap_or(DEFAULT_MAX_CONCURRENT);
        CountingSemaphore {
            permits: Mutex::new(max),
            available: Condvar::new(),
        }
    })
}

thread_local! {
    /// How many GPU objects the *current thread* is holding. Only the transition
    /// 0 -> 1 takes a real semaphore permit; 1 -> 0 releases it. Nested builds on
    /// the same thread are reentrant no-ops, so one thread is at most one permit.
    static GPU_OBJECTS_ON_THREAD: Cell<usize> = const { Cell::new(0) };
}

fn acquire_semaphore() {
    let sem = semaphore();
    let mut permits = sem.permits.lock().unwrap_or_else(|e| e.into_inner());
    while *permits == 0 {
        permits = sem
            .available
            .wait(permits)
            .unwrap_or_else(|e| e.into_inner());
    }
    *permits -= 1;
}

fn release_semaphore() {
    let sem = semaphore();
    let mut permits = sem.permits.lock().unwrap_or_else(|e| e.into_inner());
    *permits += 1;
    sem.available.notify_one();
}

/// RAII permit held for the entire lifetime of a GPU-resident test object
/// (`CudaExecutor` or `HybridScheduler`). Reentrant per thread via a per-thread
/// reference count: the thread holds exactly one semaphore permit whenever its
/// count is `> 0`. Acquire takes the permit on the `0 -> 1` transition; whichever
/// permit drops *last* (the `1 -> 0` transition) releases it — so correctness is
/// independent of the order in which a test's GPU objects are dropped.
pub(crate) struct GpuTestPermit {
    // Field-less marker; all state lives in GPU_OBJECTS_ON_THREAD. The Drop impl
    // is what matters.
    _private: (),
}

impl GpuTestPermit {
    pub(crate) fn acquire() -> Self {
        let depth = GPU_OBJECTS_ON_THREAD.with(Cell::get);
        if depth == 0 {
            // First GPU object on this thread: take the real permit BEFORE
            // marking the thread busy, so the block happens while this thread
            // holds zero GPU resources (deadlock-free even when a single test
            // builds several GPU objects across both backends).
            acquire_semaphore();
        }
        GPU_OBJECTS_ON_THREAD.with(|c| c.set(depth + 1));
        GpuTestPermit { _private: () }
    }
}

impl Drop for GpuTestPermit {
    fn drop(&mut self) {
        let remaining = GPU_OBJECTS_ON_THREAD.with(|c| {
            let n = c.get().saturating_sub(1);
            c.set(n);
            n
        });
        if remaining == 0 {
            release_semaphore();
        }
    }
}
