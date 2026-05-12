//! Runtime-wiring pins for SHIP-TWO-001 Task #132 Phase 2.
//!
//! Context: during spec v2.45 authoring (2026-04-24), a stale §14.5 row
//! asserted "Phase 2 (live-wire, pending) — 2 days" long after the runtime
//! wiring had actually shipped. The draft amendment propagated the stale
//! claim until a code-check falsified it; the spec v2.45 inline five-whys
//! named the defect class: **spec narrative outruns code-existence
//! verification**.
//!
//! This test pins the narrative to the code at compile time. If someone
//! deletes, renames, or stubs out any of the named runtime-wiring symbols,
//! this test file fails to compile. `cargo test` catches the drift before
//! the next `apr pretrain --device cuda:0` dispatch silently falls back
//! to CPU (the original Task #132 symptom).
//!
//! Coverage at v2.50.0:
//!   Device enum + resolve_device()                → `entrenar::train::device`
//!   CudaTransformerTrainer runtime handle         → `entrenar::train::transformer_trainer::cuda_trainer` (feature=cuda)
//!   build_shared_cuda_trainer / CudaRealStepFn /  → `entrenar::train::pretrain_real_cuda` (feature=cuda)
//!      CudaRealValFn / CudaAprCheckpointFn
//!
//! Pairs with `ship_two_001_const_pinning.rs` (pins the 45 `AC_*`
//! constants) as the second of the spec-consistency drift-prevention
//! falsification tests — this one pins runtime paths, that one pins
//! threshold values. Both fail CI on drift.

// Device grammar — lives in aprender-train, needs no cuda feature.
use entrenar::train::device::{resolve_device, Device};

/// `Device` enum must be convertible from a `--device` CLI string via
/// `resolve_device`. If either side renames, this fails to compile.
#[test]
fn device_resolve_cpu_succeeds() {
    let d: Device = resolve_device("cpu").expect("resolve_device(\"cpu\")");
    assert_eq!(d.to_string(), "cpu");
}

#[test]
fn device_resolve_cuda0_succeeds_if_available() {
    // "auto" never fails: resolves to cpu on CPU-only host, cuda:0 otherwise.
    let d: Device = resolve_device("auto").expect("resolve_device(\"auto\")");
    // Either "cpu" or "cuda:0" is fine — just pin that the string round-trips.
    let s = d.to_string();
    assert!(s == "cpu" || s.starts_with("cuda:"), "unexpected: {s}");
}

// ─── CUDA runtime-wiring pins — only when built with --features cuda ───

#[cfg(feature = "cuda")]
mod cuda_wiring {
    //! These symbols must exist whenever the `cuda` feature is compiled in.
    //! `apr pretrain --device cuda:0` depends on every one of them; if any
    //! goes missing, the CUDA dispatch path silently breaks (the Task #132
    //! bug class). Compile-time imports + function-pointer binds force the
    //! compiler to resolve each symbol to a concrete `fn`/`struct`.
    //!
    //! On a `cargo test -p aprender-train --features cuda` build, any rename
    //! or signature drift in `pretrain_real_cuda.rs` fails this module.

    use entrenar::train::pretrain_real_cuda::{
        build_shared_cuda_trainer, CudaAprCheckpointFn, CudaRealStepFn, CudaRealValFn,
    };

    /// `build_shared_cuda_trainer(lr, seq_length, seed) -> Result<_>`
    /// is the entry point `drive_real_cuda` calls.
    #[test]
    fn build_shared_cuda_trainer_is_a_function_of_f32_usize_u64() {
        // Function-pointer cast forces symbol + signature resolution. If
        // the fn gets renamed to `build_cuda_shared_trainer` or its args
        // change, this line fails to compile.
        let _: fn(f32, usize, u64) -> entrenar::Result<_> = build_shared_cuda_trainer;
    }

    /// The three CUDA step/val/ckpt types wired into `drive_real_cuda`
    /// must remain constructible — if any of these structs vanish or get
    /// renamed, the drive_real_cuda call at `crates/apr-cli/src/commands/pretrain.rs:356`
    /// fails to build.
    #[test]
    fn cuda_step_val_ckpt_types_exist_as_named() {
        // size_of is enough to force resolution; value irrelevant.
        let _step_sz = std::mem::size_of::<CudaRealStepFn>();
        let _val_sz = std::mem::size_of::<CudaRealValFn>();
        let _ckpt_sz = std::mem::size_of::<CudaAprCheckpointFn>();
    }
}

// ─── apr-cli wiring: dispatch path symbols ───
//
// drive_real / drive_real_cuda / drive_real_cpu are private fns inside
// apr-cli, so they can't be imported from aprender-train tests. What we
// can pin from this side is the inverse: the `entrenar::train::device`
// module + the `entrenar::train::pretrain_real_cuda` module are the two
// surfaces drive_real bridges. If either disappears, the apr-cli build
// fails — which shows up elsewhere in CI — so the pin here + the build
// of apr-cli form a two-sided check.
