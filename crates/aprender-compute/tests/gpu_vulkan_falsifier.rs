//! Vulkan/wgpu GPU falsifier — the runtime half of `apr-feature-surface-v1`.
//!
//! This is the ONLY GPU test in the tree that is allowed to fail closed. The
//! 32 tests in `src/backends/gpu/tests_gpu/` all open with
//!
//! ```ignore
//! let Some(mut gpu) = get_shared_gpu() else {
//!     eprintln!("GPU not available, skipping test");
//!     return;                       // <-- PASSES with no GPU
//! };
//! ```
//!
//! and 28 of them additionally swallow the kernel's `Err`. They therefore pass
//! on a machine with no GPU *and* pass when the kernel returns an error, which
//! is why a CI lane built on them would prove nothing. This file has no skip
//! hatch when `APRENDER_REQUIRE_DISCRETE_GPU=1` is set, which is exactly how
//! the `gpu-vulkan` CI job invokes it.
//!
//! ## Why the adapter is asserted BY NAME
//!
//! `GpuDevice::new_async()` calls `request_adapter` with
//! `power_preference: HighPerformance, force_fallback_adapter: false` and **no
//! `device_type` filter**. `force_fallback_adapter: false` only means "do not
//! FORCE the software adapter" — it does not exclude one when it is the only
//! adapter present. On the intel/AMD-RADV CI host, Mesa's `lvp_icd` enumerates
//! `llvmpipe` alongside the two W5700X, so:
//!
//! * a container with no `/dev/dri` passthrough,
//! * a container whose `/dev/dri` is present but permission-denied,
//! * and a correctly-plumbed container
//!
//! all produce a *successful* `GpuDevice::new()` and a *correct* compute
//! result. Wall-clock does not separate them either (measured 27.5 ms on RADV
//! vs 29.4 ms on permission-denied llvmpipe). Asserting only "the compute
//! returned the right answer" would pass in all three cases — pure theater.
//! The discriminator has to be the adapter identity itself.
//!
//! Run it by hand:
//! ```sh
//! APRENDER_REQUIRE_DISCRETE_GPU=1 \
//!   cargo test -p aprender-compute --features gpu --test gpu_vulkan_falsifier
//! ```
#![cfg(feature = "gpu")]

use trueno::backends::gpu::{GpuBackend, GpuDevice};

/// Substrings identifying a software rasteriser. Matched case-insensitively
/// against `wgpu`'s adapter name.
const SOFTWARE_RASTERISERS: &[&str] = &["llvmpipe", "lavapipe", "swiftshader", "softpipe"];

fn requires_gpu() -> bool {
    std::env::var("APRENDER_REQUIRE_DISCRETE_GPU").as_deref() == Ok("1")
}

/// Returns `true` if the body should run.
///
/// On a dev laptop with no GPU this returns `false` and the test no-ops. In
/// CI, where `APRENDER_REQUIRE_DISCRETE_GPU=1`, absence of a GPU is a hard
/// FAILURE rather than a skip — so the lane cannot go green by losing its
/// device passthrough.
fn gate(what: &str) -> bool {
    if GpuDevice::is_available() {
        return true;
    }
    assert!(
        !requires_gpu(),
        "APRENDER_REQUIRE_DISCRETE_GPU=1 but no wgpu adapter is available ({what}). \
         In the CI lane this means the container lost its /dev/dri passthrough or the \
         RADV ICD pin is wrong — NOT that the test should be skipped."
    );
    eprintln!("no GPU present and APRENDER_REQUIRE_DISCRETE_GPU unset — skipping {what}");
    false
}

// The `DEVICE_INIT_LOCK` reentrancy self-deadlock is falsified by
// `tests/gpu_device_init_deadlock.rs`, NOT here. That bug only fires when the
// call is the first in the PROCESS to reach `shared_instance()`, and `cargo
// test` runs this file's tests concurrently in one process — the three tests
// below all call `is_available()` through `gate()`, which primes the `OnceLock`
// without holding the lock. A probe placed here would win or lose that race at
// random and prove nothing. It therefore lives alone in its own test binary,
// where cargo guarantees it runs first.

/// The ICD pin held: no software rasteriser is even enumerated.
///
/// The CI lane sets `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json`,
/// which turns a missing/denied `/dev/dri` into "0 adapters" and a hard error
/// instead of a silent llvmpipe fallback.
/// This asserts a property of the LANE'S ENVIRONMENT (the loader is pinned to
/// the hardware ICD), not of aprender's code, so it is enforced only where
/// that pin exists — i.e. under `APRENDER_REQUIRE_DISCRETE_GPU=1`. An
/// unpinned dev box legitimately enumerates llvmpipe alongside its real GPU
/// (measured on the local RTX 4090 host: `[0] NVIDIA GeForce RTX 4090`,
/// `[1] llvmpipe (LLVM 15.0.7, 256 bits)`), and that is not a defect.
///
/// `selected_adapter_is_real_hardware` below is the assertion that must hold
/// EVERYWHERE, pinned or not.
#[test]
fn enumerated_adapters_contain_no_software_rasteriser() {
    if !requires_gpu() {
        eprintln!(
            "APRENDER_REQUIRE_DISCRETE_GPU unset — ICD-pin purity is a lane property; skipping"
        );
        return;
    }
    if !gate("enumerated_adapters_contain_no_software_rasteriser") {
        return;
    }
    let adapters = GpuDevice::list_adapters();

    // Non-vacuity: an empty list must FAIL, never pass. A gate that is green
    // on n=0 is the failure mode this whole contract exists to remove.
    assert!(!adapters.is_empty(), "0 adapters enumerated — failed measurement, not a pass");

    for (idx, name, backend) in &adapters {
        let lower = name.to_lowercase();
        for bad in SOFTWARE_RASTERISERS {
            assert!(
                !lower.contains(bad),
                "adapter [{idx}] {name:?} (backend={backend}) is the software rasteriser \
                 {bad:?}. The Vulkan loader is not pinned to the hardware ICD, so this lane \
                 could pass on CPU."
            );
        }
    }
}

/// The adapter aprender ACTUALLY selects is real hardware.
///
/// Asserts on the selection made by the shipping code path, not on a
/// re-implementation of it.
#[test]
fn selected_adapter_is_real_hardware() {
    if !gate("selected_adapter_is_real_hardware") {
        return;
    }
    let adapters = GpuDevice::list_adapters();
    assert!(!adapters.is_empty(), "0 adapters enumerated — failed measurement, not a pass");

    // `list_adapters()` enumerates in the same order and under the same
    // backend mask that `new_async()`'s `request_adapter` chooses from.
    let (_, name, backend) = &adapters[0];
    let lower = name.to_lowercase();

    for bad in SOFTWARE_RASTERISERS {
        assert!(
            !lower.contains(bad),
            "SELECTED adapter is {name:?} — a software rasteriser. This lane would have \
             reported a GPU pass while running entirely on the CPU."
        );
    }

    // The lane runs Vulkan-on-RADV. Metal/DX12 are accepted so the same test
    // is meaningful on the M4 and on Windows hosts; GL must never appear
    // (PMAT-925: the GLES adapter SIGABRTs in Drop on this exact AMD box).
    assert!(
        !backend.to_lowercase().contains("gl"),
        "adapter backend is {backend:?}; Backends::PRIMARY must never yield GL/GLES"
    );

    eprintln!("SELECTED adapter name={name:?} backend={backend}");
}

/// A real dispatch produces correct values on that adapter.
///
/// Correctness alone is NOT sufficient (llvmpipe computes the right answer
/// too) — this runs in addition to the identity assertions above, not instead
/// of them.
#[test]
fn gpu_compute_dispatch_is_correct() {
    if !gate("gpu_compute_dispatch_is_correct") {
        return;
    }
    let mut gpu = GpuBackend::new();

    let a: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..1024).map(|i| (i * 2) as f32).collect();

    // No `if let Ok(..)` — an Err must fail the test, not be printed and
    // swallowed the way 28 of the tests_gpu cases do.
    let out = gpu.vec_add(&a, &b).expect("vec_add dispatch failed on the selected adapter");

    assert_eq!(out.len(), a.len(), "vec_add returned the wrong length");
    for i in 0..a.len() {
        let expected = a[i] + b[i];
        assert!((out[i] - expected).abs() < 1e-5, "vec_add[{i}] = {}, expected {expected}", out[i]);
    }
}
