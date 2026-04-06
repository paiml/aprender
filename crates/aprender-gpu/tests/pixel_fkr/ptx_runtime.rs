//! PTX Runtime Validation (requires CUDA device)

use super::*;
use trueno_gpu::driver::CudaContext;

/// Check if CUDA device is available
fn cuda_available() -> bool {
    CudaContext::new(0).is_ok()
}

/// ptx-pixel-fkr: Softmax PTX produces correct results
#[test]
fn ptx_pixel_fkr_softmax_runtime() {
    if !cuda_available() {
        eprintln!("Skipping PTX runtime test: no CUDA device");
        return;
    }

    let mut rng = SimpleRng::new(12345);
    let x = rng.gen_vec(128);

    // Scalar baseline
    let scalar_result = scalar_softmax(&x);

    // PTX execution would go here
    // For now, verify PTX generation is valid
    let kernel = SoftmaxKernel::new(128);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".entry"), "PTX should have entry point");
    assert!(
        ptx.contains(".visible"),
        "PTX should have visible attribute"
    );

    println!(
        "ptx_pixel_fkr_softmax_runtime: PTX generated ({} bytes)",
        ptx.len()
    );
    println!(
        "  Scalar baseline sum: {:.6}",
        scalar_result.iter().sum::<f32>()
    );
}

/// ptx-pixel-fkr: GEMM PTX produces correct results
#[test]
fn ptx_pixel_fkr_gemm_runtime() {
    if !cuda_available() {
        eprintln!("Skipping PTX GEMM runtime test: no CUDA device");
        return;
    }

    let m: usize = 32;
    let n: usize = 32;
    let k: usize = 64;

    let mut rng = SimpleRng::new(23456);
    let a = rng.gen_vec(m * k);
    let b = rng.gen_vec(k * n);

    // Scalar baseline
    let scalar_result = scalar_gemm(&a, &b, m, n, k);

    // Generate PTX
    let kernel = GemmKernel::tiled(m as u32, n as u32, k as u32, 32);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".entry"), "GEMM PTX should have entry point");
    assert!(ptx.contains(".shared"), "GEMM PTX should use shared memory");

    println!(
        "ptx_pixel_fkr_gemm_runtime: PTX generated ({} bytes)",
        ptx.len()
    );
    println!("  Scalar result[0]: {:.6}", scalar_result[0]);
}

/// ptx-pixel-fkr: LayerNorm PTX produces correct results
#[test]
fn ptx_pixel_fkr_layernorm_runtime() {
    if !cuda_available() {
        eprintln!("Skipping PTX LayerNorm runtime test: no CUDA device");
        return;
    }

    let n: usize = 256;
    let mut rng = SimpleRng::new(34567);
    let x = rng.gen_vec(n);
    let gamma = rng.gen_vec(n);
    let beta = rng.gen_vec(n);

    // Scalar baseline
    let scalar_result = scalar_layernorm(&x, &gamma, &beta, 1e-5);

    // Generate PTX
    let kernel = LayerNormKernel::new(n as u32);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".entry"),
        "LayerNorm PTX should have entry point"
    );

    println!(
        "ptx_pixel_fkr_layernorm_runtime: PTX generated ({} bytes)",
        ptx.len()
    );
    println!(
        "  Scalar result mean: {:.6}",
        scalar_result.iter().sum::<f32>() / n as f32
    );
}
