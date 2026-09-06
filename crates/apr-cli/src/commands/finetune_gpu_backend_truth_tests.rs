//! finetune_gpu_backend_truth (PMAT-991, #2906, PP-066 R-3 P_2): the CLI must
//! meet the request against the BUILD, not print a claim off the request
//! alone.
//!
//! Two defects this pins:
//!
//! 1. `--gpu-backend cuda` on a binary built WITHOUT the `cuda` feature used
//!    to fall through to the CPU path silently
//!    (feedback_apr_finetune_lora_cpu_only). It must instead REFUSE —
//!    `CliError::FeatureDisabled`, exit code 9 (see `error.rs`), naming the
//!    missing feature and the rebuild flag.
//! 2. The pre-training banner used to print
//!    "[gpu-backend] CUDA selected — using cuBLAS backward path" straight
//!    from the request, before any training step ran. The cuBLAS-backward
//!    claim is an EVENT, derived after training from
//!    `entrenar::training_backend_banner`
//!    (`entrenar::backward_kernel_launches()`), never from the request.
//!
//! These three functions are pure and gate through
//! `super` (this file is a `#[cfg(test)]` module of `commands::finetune`; the crate-private `commands` tree is not
//! reachable from an integration test, same seam as `serve_test_support` /
//! `qa_types`).

#![allow(clippy::unwrap_used)]

use super::{gpu_backend_decision, post_training_banner, pre_training_notice, GpuBackendChoice};
use crate::CliError;

/// Every row of {cuda, wgpu, auto, cpu} x {build_has_cuda} x {build_has_wgpu} —
/// both polarities of both build flags, for every requested string.
#[test]
fn gpu_backend_decision_case_table() {
    struct Row {
        requested: &'static str,
        build_has_cuda: bool,
        build_has_wgpu: bool,
        expect: Expect,
    }
    enum Expect {
        Ok(GpuBackendChoice),
        FeatureDisabled,
    }

    let rows = [
        // requested = "cuda": Ok(Cuda) iff build_has_cuda, else a refusal —
        // NEVER a silent CPU fallback — regardless of wgpu.
        Row {
            requested: "cuda",
            build_has_cuda: true,
            build_has_wgpu: true,
            expect: Expect::Ok(GpuBackendChoice::Cuda),
        },
        Row {
            requested: "cuda",
            build_has_cuda: true,
            build_has_wgpu: false,
            expect: Expect::Ok(GpuBackendChoice::Cuda),
        },
        Row {
            requested: "cuda",
            build_has_cuda: false,
            build_has_wgpu: true,
            expect: Expect::FeatureDisabled,
        },
        Row {
            requested: "cuda",
            build_has_cuda: false,
            build_has_wgpu: false,
            expect: Expect::FeatureDisabled,
        },
        // requested = "wgpu": symmetric on build_has_wgpu, regardless of cuda.
        Row {
            requested: "wgpu",
            build_has_cuda: true,
            build_has_wgpu: true,
            expect: Expect::Ok(GpuBackendChoice::Wgpu),
        },
        Row {
            requested: "wgpu",
            build_has_cuda: false,
            build_has_wgpu: true,
            expect: Expect::Ok(GpuBackendChoice::Wgpu),
        },
        Row {
            requested: "wgpu",
            build_has_cuda: true,
            build_has_wgpu: false,
            expect: Expect::FeatureDisabled,
        },
        Row {
            requested: "wgpu",
            build_has_cuda: false,
            build_has_wgpu: false,
            expect: Expect::FeatureDisabled,
        },
        // requested = "cpu": always Ok(Cpu), whatever the build.
        Row {
            requested: "cpu",
            build_has_cuda: true,
            build_has_wgpu: true,
            expect: Expect::Ok(GpuBackendChoice::Cpu),
        },
        Row {
            requested: "cpu",
            build_has_cuda: true,
            build_has_wgpu: false,
            expect: Expect::Ok(GpuBackendChoice::Cpu),
        },
        Row {
            requested: "cpu",
            build_has_cuda: false,
            build_has_wgpu: true,
            expect: Expect::Ok(GpuBackendChoice::Cpu),
        },
        Row {
            requested: "cpu",
            build_has_cuda: false,
            build_has_wgpu: false,
            expect: Expect::Ok(GpuBackendChoice::Cpu),
        },
        // requested = "auto": NEVER errors; picks the best compiled-in
        // backend, else CPU.
        Row {
            requested: "auto",
            build_has_cuda: true,
            build_has_wgpu: true,
            expect: Expect::Ok(GpuBackendChoice::Cuda),
        },
        Row {
            requested: "auto",
            build_has_cuda: true,
            build_has_wgpu: false,
            expect: Expect::Ok(GpuBackendChoice::Cuda),
        },
        Row {
            requested: "auto",
            build_has_cuda: false,
            build_has_wgpu: true,
            expect: Expect::Ok(GpuBackendChoice::Wgpu),
        },
        Row {
            requested: "auto",
            build_has_cuda: false,
            build_has_wgpu: false,
            expect: Expect::Ok(GpuBackendChoice::Cpu),
        },
    ];

    for (i, row) in rows.iter().enumerate() {
        let got = gpu_backend_decision(row.requested, true, row.build_has_cuda, row.build_has_wgpu);
        match &row.expect {
            Expect::Ok(choice) => {
                assert_eq!(
                    got.as_ref().ok(),
                    Some(choice),
                    "row {i} ({}, cuda={}, wgpu={}): expected Ok({choice:?}), got {got:?}",
                    row.requested,
                    row.build_has_cuda,
                    row.build_has_wgpu,
                );
            }
            Expect::FeatureDisabled => {
                let err = got.expect_err(&format!(
                    "row {i} ({}, cuda={}, wgpu={}): expected a refusal, got Ok",
                    row.requested, row.build_has_cuda, row.build_has_wgpu,
                ));
                assert!(
                    matches!(err, CliError::FeatureDisabled(_)),
                    "row {i}: expected CliError::FeatureDisabled, got {err:?}"
                );
                assert_eq!(
                    err.exit_code_value(),
                    9,
                    "row {i}: FeatureDisabled must exit 9 (error.rs)"
                );
            }
        }
    }
}

/// A cuda request on a cpu-only build (the exact `-m lora --gpu-backend cuda`
/// defect scenario) refuses; it never silently falls back to CPU.
#[test]
fn cuda_request_on_cpu_only_build_is_a_refusal_not_a_fallback() {
    let err = gpu_backend_decision("cuda", true, false, false)
        .expect_err("cuda requested on a cpu-only build must refuse");
    assert!(matches!(err, CliError::FeatureDisabled(_)));
    assert_eq!(err.exit_code_value(), 9);
    let msg = err.to_string();
    assert!(
        msg.contains("cuda"),
        "refusal must name the missing feature: {msg}"
    );
    assert!(
        msg.contains("--features cuda"),
        "refusal must name the rebuild flag: {msg}"
    );
}

/// The pre-training notice must never CLAIM a cuBLAS backward — that is only
/// known after training. Every choice, both polarities.
#[test]
fn pre_training_notice_never_claims_cublas_backward() {
    for choice in [
        GpuBackendChoice::Cuda,
        GpuBackendChoice::Wgpu,
        GpuBackendChoice::Cpu,
    ] {
        let notice = pre_training_notice(&choice);
        assert!(
            !notice.contains("cuBLAS backward"),
            "pre-training notice for {choice:?} claims a cuBLAS backward before training ran: \
             {notice}"
        );
    }
}

/// Registered mutation: "print the cuBLAS banner unconditionally for a cuda
/// choice" must go RED here. With zero observed device-side backward
/// launches, `post_training_banner` must be `None` for the cuda choice.
#[test]
fn post_training_banner_is_none_with_zero_launches() {
    assert_eq!(entrenar::backward_kernel_launches(), 0);

    assert_eq!(
        post_training_banner(
            &GpuBackendChoice::Cuda,
            entrenar::backward_kernel_launches()
        ),
        None,
        "zero device-side backward launches observed — the cuBLAS banner must not print"
    );
}

/// The cpu and wgpu choices never print the cuBLAS banner, launches or not —
/// `entrenar::training_backend_banner` gates on `requested == "cuda"`.
#[test]
fn post_training_banner_is_none_for_non_cuda_choices() {
    assert_eq!(post_training_banner(&GpuBackendChoice::Cpu, 0), None);
    assert_eq!(post_training_banner(&GpuBackendChoice::Wgpu, 0), None);
}

/// R-3 rule: `-m lora --gpu-backend cuda` REFUSES or trains on the GPU — never
/// a CPU run under a GPU flag. Plain LoRA has no cuBLAS path, so the explicit
/// request is `ValidationFailed` (exit code 5, read from error.rs), even on a
/// build that has the cuda feature.
#[test]
fn plain_lora_with_explicit_cuda_is_a_refusal_not_a_cpu_run() {
    let err = gpu_backend_decision("cuda", false, true, true)
        .expect_err("cuda for a method with no cuda path must refuse");
    assert!(matches!(err, CliError::ValidationFailed(_)), "got {err:?}");
    assert_eq!(err.exit_code_value(), 5);
}

/// `auto` never refuses: plain LoRA on auto is the CPU path whatever the
/// build carries; QLoRA on auto prefers cuda, then wgpu, then cpu.
#[test]
fn auto_follows_the_method_then_the_build() {
    assert_eq!(
        gpu_backend_decision("auto", false, true, true).expect("auto"),
        GpuBackendChoice::Cpu
    );
    assert_eq!(
        gpu_backend_decision("auto", true, true, true).expect("auto"),
        GpuBackendChoice::Cuda
    );
    assert_eq!(
        gpu_backend_decision("auto", true, false, true).expect("auto"),
        GpuBackendChoice::Wgpu
    );
    assert_eq!(
        gpu_backend_decision("auto", true, false, false).expect("auto"),
        GpuBackendChoice::Cpu
    );
}

/// The banner is about THIS run: a launch count snapshot taken now makes a
/// later banner None even if earlier runs in this process launched kernels.
#[test]
fn post_training_banner_is_scoped_to_the_run() {
    let now = entrenar::backward_kernel_launches();
    assert_eq!(post_training_banner(&GpuBackendChoice::Cuda, now), None);
    assert_eq!(
        post_training_banner(&GpuBackendChoice::Cuda, now.saturating_add(1)),
        None
    );
}
