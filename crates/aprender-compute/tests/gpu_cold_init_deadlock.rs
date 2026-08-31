//! The FIRST sync GPU init in a process must not deadlock against itself.
//!
//! PMAT-778 (#2063) added `DEVICE_INIT_LOCK` to serialize native wgpu
//! instance/adapter/device creation, and in the same commit added
//! `shared_instance()` whose `OnceLock::get_or_init` closure took that same
//! lock. `std::sync::Mutex` is not reentrant, and every sync entry point
//! (`GpuDevice::new`, `new_with_adapter_index`, `list_adapters`) took the lock
//! and *then* called `shared_instance()`. So the first of them to run in a
//! process blocked on a lock it was already holding.
//!
//! Measured on mini (Apple M4, macOS 26.5), `apr serve run <gguf> --backend
//! wgpu --gpu-layers all`: 603 s elapsed, **0.0% CPU**, state S, port never
//! bound, stuck between the caller's `Initializing WGPU device...` and
//! `WGPU device ready` prints. Reproduced twice.
//!
//! WHY IT HID FOR SO LONG. It only bites when a *sync* entry point is the first
//! thing in the process to touch `shared_instance()`. `GpuDevice::is_available`,
//! `pool.rs` and `monitor/backends.rs` all reach the instance WITHOUT the guard,
//! so any run that probed availability first primed the `OnceLock` and never
//! saw it — realizar's own wgpu path does exactly that
//! (`gguf_gpu_generate.rs` calls `is_available()` before `new()`). The only two
//! cold callers left are in apr-cli behind `--features wgpu`, a feature that
//! had never compiled.
//!
//! WHY THIS IS AN INTEGRATION TEST AND NOT A UNIT TEST. `INSTANCE` is a
//! process-global `OnceLock`. Inside the lib test binary any sibling test that
//! touches the GPU primes it first, and the cold path — the only path with the
//! bug — becomes unreachable. Each `tests/*.rs` file is its own binary, so this
//! file is a guaranteed-cold process. It needs no GPU: the deadlock is upstream
//! of adapter probing, so a host with no adapter reproduces it identically.
#![cfg(all(feature = "gpu", not(target_arch = "wasm32")))]

use std::sync::mpsc;
use std::time::Duration;

/// Generous enough that a real cold adapter+device probe on a slow ICD is never
/// mistaken for the deadlock, short enough to fail a CI job rather than hang it.
const BUDGET: Duration = Duration::from_secs(120);

#[test]
fn the_first_sync_gpu_init_in_a_process_does_not_self_deadlock() {
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new()
        .name("cold-gpu-init".into())
        .spawn(move || {
            // `list_adapters` is the cheapest sync entry point that takes
            // DEVICE_INIT_LOCK, and unlike `new()` it returns a value on a
            // host with no GPU instead of an error, so this test asserts
            // LIVENESS on every host rather than the presence of hardware.
            let adapters = trueno::backends::gpu::GpuDevice::list_adapters();
            // The send may fail if the receiver already timed out; that is the
            // RED case and is reported below, not here.
            let _ = tx.send(adapters.len());
        })
        .expect("spawn cold-gpu-init");

    match rx.recv_timeout(BUDGET) {
        Ok(_n) => { /* returned — liveness is the assertion */ }
        Err(mpsc::RecvTimeoutError::Timeout) => panic!(
            "the first sync GPU init in this process did not return within {BUDGET:?}. \
             This is the PMAT-778 self-deadlock: a sync entry point took \
             DEVICE_INIT_LOCK and then called shared_instance(), whose \
             OnceLock closure took the same non-reentrant mutex. Every sync \
             entry point must go through `with_device_init_lock`, which primes \
             the instance BEFORE taking the guard."
        ),
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            panic!("the cold GPU init thread died without returning a result")
        }
    }
}
