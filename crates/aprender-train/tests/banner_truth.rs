//! banner_truth (PMAT-991, #2906, PP-066 R-3): the "cuBLAS backward" training
//! banner MUST be derived from device-side backward kernel launches actually
//! observed, never from the caller's request alone.
//!
//! The defect (measured 2026-09-05): the CLI printed
//! "[gpu-backend] CUDA selected — using cuBLAS backward path" straight from
//! the request, with nothing in `entrenar` recording whether a device-side
//! cuBLAS backward kernel ever launched. A banner could claim GPU training
//! while every backward ran on the CPU. This test pins the library-side
//! contract: `training_backend_banner(requested)` is `Some` iff
//! `requested == "cuda"` AND `backward_kernel_launches() > 0`.

#![allow(clippy::unwrap_used)]

use entrenar::{backward_kernel_launches, reset_backward_kernel_launches, training_backend_banner};

/// Registered mutation: "print the banner unconditionally" must go RED here.
/// A banner claiming cuBLAS backward with a zero launch count is exactly the
/// defect this ticket forecloses.
#[test]
fn banner_is_none_for_cuda_with_zero_launches() {
    reset_backward_kernel_launches();
    assert_eq!(backward_kernel_launches(), 0);
    assert_eq!(
        training_backend_banner("cuda"),
        None,
        "a cuBLAS-backward banner with ZERO observed device-side backward launches is the \
         defect PMAT-991 exists to forbid — the banner must never be printed unconditionally"
    );
}

#[cfg(not(feature = "cuda"))]
mod cpu_only {
    use entrenar::autograd::{backward, matmul, Tensor};
    use entrenar::{
        backward_kernel_launches, reset_backward_kernel_launches, training_backend_banner,
    };

    /// With no `cuda` feature compiled in, there is no device-side backward
    /// kernel that could ever launch — so the counter must stay at 0 even
    /// after driving a real (CPU) backward pass through the public autograd
    /// API, and the banner must be `None` for BOTH "cuda" (nothing to claim)
    /// and "cpu" (the banner is a cuBLAS claim, not a generic device claim).
    #[test]
    fn cpu_backward_never_increments_the_device_counter() {
        reset_backward_kernel_launches();

        // Smallest public backward drivable through the autograd API: a 2x2
        // matmul forward + backward (entrenar::autograd::ops::matmul).
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], true);
        let mut c = matmul(&a, &b, 2, 2, 2);
        backward(&mut c, None);

        assert_eq!(
            backward_kernel_launches(),
            0,
            "no cuda feature is compiled in, so NOTHING can have launched a device-side \
             backward kernel — a CPU-only backward must never move this counter"
        );
        assert_eq!(training_backend_banner("cuda"), None);
        assert_eq!(training_backend_banner("cpu"), None);
    }
}

#[cfg(feature = "cuda")]
mod cuda_only {
    use std::sync::Arc;

    use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};

    use entrenar::autograd::cuda_backward::{gemm_backward_a, init_kernel_cache};
    use entrenar::{
        backward_kernel_launches, reset_backward_kernel_launches, training_backend_banner,
    };

    /// Runs a real cuBLAS backward GEMM (2x2) through the smallest public
    /// CUDA entry point and asserts the counter — and hence the banner — is
    /// driven by that launch, not by the string "cuda".
    #[test]
    fn cuda_backward_launch_drives_the_banner() {
        if !cuda_available() {
            eprintln!("no CUDA device visible — skipping cuda_backward_launch_drives_the_banner");
            return;
        }
        let ctx = Arc::new(CudaContext::new(0).expect("CUDA device 0 required"));
        init_kernel_cache(ctx.clone()).expect("init_kernel_cache failed");
        let stream = CudaStream::new(&ctx).expect("CudaStream::new failed");

        reset_backward_kernel_launches();
        assert_eq!(backward_kernel_launches(), 0);
        assert_eq!(training_backend_banner("cuda"), None);

        // C = A @ B, A/B/grad_C all 2x2 — same fixture as
        // cuda_backward::tests::gemm::test_gemm_backward_a_basic.
        let m = 2u32;
        let k = 2u32;
        let n = 2u32;
        let grad_output_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let b_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let grad_a_data: Vec<f32> = vec![0.0; (m * k) as usize];

        let grad_output = GpuBuffer::from_host(&ctx, &grad_output_data).expect("grad_output");
        let b = GpuBuffer::from_host(&ctx, &b_data).expect("b");
        let mut grad_a = GpuBuffer::from_host(&ctx, &grad_a_data).expect("grad_a");

        gemm_backward_a(&grad_output, &b, &mut grad_a, m, k, n, &stream)
            .expect("gemm_backward_a failed");
        stream.synchronize().expect("stream.synchronize failed");

        let launches = backward_kernel_launches();
        assert!(
            launches > 0,
            "a cuBLAS backward GEMM just launched on-device; the counter must reflect it"
        );

        let banner = training_backend_banner("cuda");
        assert!(banner.is_some(), "backward launched — the banner must now be Some");
        let banner = banner.expect("checked is_some above");
        assert!(
            banner.contains("cuBLAS backward"),
            "banner must name the cuBLAS backward engagement, got: {banner}"
        );

        // Reset and re-check: a banner without ANY launch since the reset is
        // the defect again — the predicate must track the counter, not a
        // sticky "we've seen CUDA before" flag.
        reset_backward_kernel_launches();
        assert_eq!(
            training_backend_banner("cuda"),
            None,
            "after reset, zero launches have been observed — banner must be None again"
        );
    }
}
