//! #3759: a norm kernel compiled for one epsilon is never reused for another.
//!
//! Epsilon is a PTX immediate, but the module cache keyed RMSNorm by shape alone, and preload
//! compiled those keys with a hardcoded 1e-5. qwen2.5-coder-7b (rms_norm_eps 1e-6) then ran
//! RMSNorm at 1e-5. That is invisible on ordinary tokens (1.4% scale) and 0.431x on
//! `<|im_start|>`, whose embedding has mean-square ~1.05e-6, so the F2 gate rejected the GPU at
//! every special-token position. These rows feed exactly that input to the real launch paths.
//! They need a CUDA device and skip, loudly, without one.

use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
use crate::cuda::executor::CudaExecutor;
use trueno_gpu::driver::GpuBuffer;

/// A vector with the mean-square of qwen2.5-coder-7b's `<|im_start|>` embedding (~1.05e-6).
fn near_zero_input(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| if i % 2 == 0 { 1.025e-3 } else { -1.025e-3 })
        .collect()
}

/// f32 reference RMSNorm with gamma = 1.
fn reference_norm(x: &[f32], eps: f32) -> f32 {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().map(|v| (v * inv).powi(2)).sum::<f32>().sqrt()
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Serial and batched RMSNorm at `eps`, returning each output's L2 norm.
fn run_both(exec: &mut CudaExecutor, x: &[f32], eps: f32) -> (f32, f32) {
    let n = x.len();
    let input = GpuBuffer::from_host(&exec.context, x).expect("input");
    let gamma = GpuBuffer::from_host(&exec.context, &vec![1.0f32; n]).expect("gamma");
    let out = GpuBuffer::<f32>::new(&exec.context, n).expect("out");
    exec.rmsnorm_into(&input, &gamma, &out, n as u32, eps)
        .expect("rmsnorm_into");
    exec.stream.synchronize().expect("sync");
    let mut serial = vec![0.0f32; n];
    out.copy_to_host(&mut serial).expect("download");
    let bout = GpuBuffer::<f32>::new(&exec.context, n).expect("bout");
    exec.batched_rmsnorm_into(&input, &gamma, &bout, n as u32, 1, eps)
        .expect("batched_rmsnorm_into");
    exec.stream.synchronize().expect("sync");
    let mut batched = vec![0.0f32; n];
    bout.copy_to_host(&mut batched).expect("download");
    (norm(&serial), norm(&batched))
}

fn check(label: &str, got: f32, want: f32, failures: &mut Vec<String>) {
    let rel = ((got - want) / want).abs();
    if rel > 1e-3 {
        failures.push(format!(
            "{label}: output norm {got:.4}, reference {want:.4} ({:.1}% off; a 1e-5 kernel gives ~0.43x here)",
            rel * 100.0
        ));
    }
}

#[test]
fn two_epsilons_in_one_process_get_two_kernels() {
    let Ok(mut exec) = CudaExecutor::new(0) else {
        eprintln!("SKIP #3759 eps-key row: no CUDA device");
        return;
    };
    let x = near_zero_input(1536);
    let (e5, e6) = (1e-5f32, 1e-6f32);
    // Not vacuous: at this input the two epsilons must give clearly different outputs.
    assert!(reference_norm(&x, e6) / reference_norm(&x, e5) > 2.0);

    let mut failures = Vec::new();
    let (s5, b5) = run_both(&mut exec, &x, e5); // compiles and caches the 1e-5 kernels first
    check("serial eps=1e-5", s5, reference_norm(&x, e5), &mut failures);
    check(
        "batched eps=1e-5",
        b5,
        reference_norm(&x, e5),
        &mut failures,
    );
    let (s6, b6) = run_both(&mut exec, &x, e6); // must NOT reuse them
    check(
        "serial eps=1e-6 after 1e-5",
        s6,
        reference_norm(&x, e6),
        &mut failures,
    );
    check(
        "batched eps=1e-6 after 1e-5",
        b6,
        reference_norm(&x, e6),
        &mut failures,
    );
    assert!(failures.is_empty(), "#3759:\n{}", failures.join("\n"));
}

#[test]
fn preload_compiles_the_models_epsilon_not_1e5() {
    let Ok(mut exec) = CudaExecutor::new(0) else {
        eprintln!("SKIP #3759 preload row: no CUDA device");
        return;
    };
    // hidden_dim 1536 so the preloaded vectorized/batched kernels cover this input's length.
    let config = HarnessConfig {
        hidden_dim: 1536,
        intermediate_dim: 4096,
        ..HarnessConfig::default()
    };
    setup_executor_harness(&mut exec, &config).expect("harness");
    let eps = 1e-6f32;
    exec.preload_modules_for_capture(
        config.num_layers,
        config.hidden_dim as u32,
        config.intermediate_dim as u32,
        config.vocab_size as u32,
        eps,
    )
    .expect("preload");

    let x = near_zero_input(config.hidden_dim);
    let (serial, batched) = run_both(&mut exec, &x, eps);
    let want = reference_norm(&x, eps);
    let mut failures = Vec::new();
    check("serial after preload(1e-6)", serial, want, &mut failures);
    check("batched after preload(1e-6)", batched, want, &mut failures);
    assert!(failures.is_empty(), "#3759:\n{}", failures.join("\n"));
}

/// The guard itself (debug builds): one key asked for two epsilons is refused.
#[test]
fn the_module_key_guard_refuses_one_key_for_two_epsilons() {
    use crate::cuda::KernelType;
    let Ok(mut exec) = CudaExecutor::new(0) else {
        eprintln!("SKIP #3759 guard row: no CUDA device");
        return;
    };
    let e5 = KernelType::VectorizedRmsNorm {
        hidden_size: 1536,
        epsilon: 1e-5,
    };
    let e6 = KernelType::VectorizedRmsNorm {
        hidden_size: 1536,
        epsilon: 1e-6,
    };
    exec.ensure_kernel_module("guard_probe_3759", &e5)
        .expect("compile");
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        exec.ensure_kernel_module("guard_probe_3759", &e6)
    }));
    assert!(
        refused.is_err(),
        "one key, two epsilons: the guard must refuse (this is #3759's shape)"
    );
}

/// ...and a grid-only difference (same PTX) passes, so the guard has no false positive there.
#[test]
fn the_module_key_guard_passes_a_grid_only_difference() {
    use crate::cuda::KernelType;
    let Ok(mut exec) = CudaExecutor::new(0) else {
        eprintln!("SKIP #3759 guard row: no CUDA device");
        return;
    };
    for batch_size in [1u32, 8, 49] {
        let kt = KernelType::BatchedVectorizedRmsNorm {
            hidden_size: 1536,
            batch_size,
            epsilon: 1e-6,
        };
        exec.ensure_kernel_module("guard_probe_grid_3759", &kt)
            .expect("batch_size is a grid dimension, not a PTX parameter");
    }
}
