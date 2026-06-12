//! Test-only macros for portable CUDA tests (APR-MONO §S #1981).
//!
//! Many CUDA tests construct a `CudaExecutor` / `CudaScheduler` on the first
//! line via `.expect(...)` / `.unwrap()`. On a GPU-less machine these panic with
//! `CudaNotAvailable`, turning the whole suite red even though there is no GPU to
//! test against. These macros convert that panic into a clean runtime skip: the
//! test prints a SKIP line and returns early when no device is present, and runs
//! normally when a device is available.
//!
//! This is the *runtime* guard counterpart to the compile-time `#[cfg(feature =
//! "cuda")]` gate — the feature being enabled does not imply a device exists at
//! test time (CI builders, laptops, containers without `/dev/nvidia*`).

/// Construct a `CudaExecutor` for the given device ordinal, or skip the test.
///
/// On `Err` (e.g. `CudaNotAvailable` on a GPU-less host), prints a SKIP line and
/// `return`s from the enclosing test function instead of panicking.
///
/// ```ignore
/// let mut executor = cuda_executor_or_skip!(0);
/// ```
#[macro_export]
macro_rules! cuda_executor_or_skip {
    ($device:expr) => {
        match $crate::cuda::CudaExecutor::new($device) {
            Ok(executor) => executor,
            Err(e) => {
                eprintln!(
                    "SKIP: CUDA executor unavailable on device {}: {e:?}",
                    $device
                );
                return;
            },
        }
    };
}

/// Construct a `CudaScheduler`, or skip the test.
///
/// On `Err` (e.g. `CudaNotAvailable` on a GPU-less host), prints a SKIP line and
/// `return`s from the enclosing test function instead of panicking.
///
/// ```ignore
/// let mut scheduler = cuda_scheduler_or_skip!();
/// ```
#[macro_export]
macro_rules! cuda_scheduler_or_skip {
    () => {
        match $crate::gpu::CudaScheduler::new() {
            Ok(scheduler) => scheduler,
            Err(e) => {
                eprintln!("SKIP: CUDA scheduler unavailable: {e:?}");
                return;
            },
        }
    };
}
