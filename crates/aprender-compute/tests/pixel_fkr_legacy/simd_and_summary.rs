//! SIMD-PIXEL-FKR + WGPU-PIXEL-FKR + Summary (SPEC Section 3.5.3-3.5.5)

use super::scalar_helpers::*;
use trueno::Vector;

// ============================================================================
// SIMD-PIXEL-FKR: SIMD Validation (SPEC Section 3.5.3)
// ============================================================================

/// simd-pixel-fkr: Vector operations match scalar baseline
#[test]
fn simd_pixel_fkr_vector_ops() {
    let mut rng = SimpleRng::new(11111);
    let a = rng.gen_vec(10000);
    let b = rng.gen_vec(10000);

    // Scalar baseline
    let scalar_add: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    let scalar_mul: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();

    // SIMD implementation
    let va = Vector::from_slice(&a);
    let vb = Vector::from_slice(&b);

    let simd_add = va.add(&vb).expect("SIMD add failed");
    let simd_mul = va.mul(&vb).expect("SIMD mul failed");

    assert!(vectors_match(&scalar_add, simd_add.as_slice(), SIMD_TOLERANCE, "simd_add"));
    assert!(vectors_match(&scalar_mul, simd_mul.as_slice(), SIMD_TOLERANCE, "simd_mul"));
}

/// simd-pixel-fkr: Softmax matches scalar baseline
#[test]
fn simd_pixel_fkr_softmax() {
    let mut rng = SimpleRng::new(22222);
    let x = rng.gen_vec(2048);

    // Scalar baseline
    let scalar_result = scalar_softmax(&x);

    // SIMD implementation
    let v = Vector::from_slice(&x);
    let simd_result = v.softmax().expect("SIMD softmax failed");

    assert!(vectors_match(&scalar_result, simd_result.as_slice(), SIMD_TOLERANCE, "simd_softmax"));
}

/// simd-pixel-fkr: Unaligned input (17 elements - not divisible by SIMD width)
#[test]
fn simd_pixel_fkr_unaligned_17() {
    let mut rng = SimpleRng::new(33333);
    let a = rng.gen_vec(17);
    let b = rng.gen_vec(17);

    // Scalar baseline
    let scalar_add: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

    // SIMD implementation
    let va = Vector::from_slice(&a);
    let vb = Vector::from_slice(&b);
    let simd_add = va.add(&vb).expect("SIMD unaligned add failed");

    assert!(vectors_match(&scalar_add, simd_add.as_slice(), SIMD_TOLERANCE, "simd_unaligned_17"));
}

/// simd-pixel-fkr: Remainder handling (255 elements)
#[test]
fn simd_pixel_fkr_remainder_255() {
    let mut rng = SimpleRng::new(44444);
    let a = rng.gen_vec(255);
    let b = rng.gen_vec(255);

    // Scalar baseline
    let scalar_mul: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();

    // SIMD implementation
    let va = Vector::from_slice(&a);
    let vb = Vector::from_slice(&b);
    let simd_mul = va.mul(&vb).expect("SIMD remainder mul failed");

    assert!(vectors_match(&scalar_mul, simd_mul.as_slice(), SIMD_TOLERANCE, "simd_remainder_255"));
}

/// simd-pixel-fkr: ReLU activation
#[test]
fn simd_pixel_fkr_relu() {
    let mut rng = SimpleRng::new(55555);
    let x = rng.gen_vec(10000);

    // Scalar baseline
    let scalar_relu: Vec<f32> = x.iter().map(|v| v.max(0.0)).collect();

    // SIMD implementation
    let v = Vector::from_slice(&x);
    let simd_relu = v.relu().expect("SIMD relu failed");

    assert!(vectors_match(&scalar_relu, simd_relu.as_slice(), SIMD_TOLERANCE, "simd_relu"));
}

// ============================================================================
// WGPU-PIXEL-FKR: WebGPU Validation (SPEC Section 3.5.4)
// ============================================================================

#[cfg(feature = "gpu")]
mod wgpu_fkr {
    use super::*;
    use trueno::backends::gpu::GpuBackend;

    /// wgpu-pixel-fkr: Large vector operations
    #[test]
    fn wgpu_pixel_fkr_large_vector() {
        if !GpuBackend::is_available() {
            eprintln!("Skipping WGPU FKR: no GPU available");
            return;
        }

        let mut rng = SimpleRng::new(66666);
        let a = rng.gen_vec(100_000);
        let b = rng.gen_vec(100_000);

        // Scalar baseline
        let scalar_add: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

        // WGPU implementation (should auto-dispatch to GPU for large vectors)
        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);
        let wgpu_add = va.add(&vb).expect("WGPU add failed");

        assert!(vectors_match(
            &scalar_add,
            wgpu_add.as_slice(),
            GPU_TOLERANCE,
            "wgpu_large_vector"
        ));
    }

    /// wgpu-pixel-fkr: Matrix multiply (GPU stress test)
    #[test]
    fn wgpu_pixel_fkr_matmul() {
        if !GpuBackend::is_available() {
            eprintln!("Skipping WGPU matmul FKR: no GPU available");
            return;
        }

        let n = 128; // 128x128 matrix
        let mut rng = SimpleRng::new(77777);
        let a_data = rng.gen_vec(n * n);
        let b_data = rng.gen_vec(n * n);

        // Scalar baseline (naive O(n^3))
        let mut scalar_result = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0f32;
                for k in 0..n {
                    sum += a_data[i * n + k] * b_data[k * n + j];
                }
                scalar_result[i * n + j] = sum;
            }
        }

        // WGPU implementation via Matrix
        use trueno::Matrix;
        let a = Matrix::from_vec(n, n, a_data).expect("Matrix A creation failed");
        let b = Matrix::from_vec(n, n, b_data).expect("Matrix B creation failed");
        let wgpu_result = a.matmul(&b).expect("WGPU matmul failed");

        // Matmul accumulates errors, so use larger tolerance
        let matmul_tolerance = GPU_TOLERANCE * n as f32;
        assert!(vectors_match(
            &scalar_result,
            wgpu_result.as_slice(),
            matmul_tolerance,
            "wgpu_matmul_128x128"
        ));
    }

    /// wgpu-pixel-fkr: Softmax (numerical stability on GPU)
    #[test]
    fn wgpu_pixel_fkr_softmax() {
        if !GpuBackend::is_available() {
            eprintln!("Skipping WGPU softmax FKR: no GPU available");
            return;
        }

        let mut rng = SimpleRng::new(88888);
        let x = rng.gen_vec(4096);

        // Scalar baseline
        let scalar_result = scalar_softmax(&x);

        // WGPU implementation
        let v = Vector::from_slice(&x);
        let wgpu_result = v.softmax().expect("WGPU softmax failed");

        assert!(vectors_match(
            &scalar_result,
            wgpu_result.as_slice(),
            GPU_TOLERANCE,
            "wgpu_softmax"
        ));
    }
}

// ============================================================================
// PTX-PIXEL-FKR: CUDA Validation (SPEC Section 3.5.5)
// ============================================================================

// PTX tests are in trueno-gpu crate (requires CUDA feature)
// See trueno-gpu/tests/ptx_pixel_fkr.rs

/// Placeholder test documenting PTX FKR location
#[test]
fn ptx_pixel_fkr_location() {
    println!("PTX Pixel FKR tests are in trueno-gpu crate:");
    println!("  cargo test -p trueno-gpu --test pixel_fkr --features cuda");
    println!();
    println!("Tests validate:");
    println!("  - QuantizeKernel (Issue #67 prevention)");
    println!("  - Q4_K dequantization");
    println!("  - GEMM kernels");
    println!("  - Softmax PTX");
}

// ============================================================================
// REGRESSION SUMMARY
// ============================================================================

/// Summary test that reports all FKR status
#[test]
fn pixel_fkr_summary() {
    println!();
    println!("========================================");
    println!("  TRUENO-SPEC-013 Pixel FKR Summary");
    println!("========================================");
    println!();
    println!("  scalar-pixel-fkr: Baseline truth tests");
    println!("    - rmsnorm_4096");
    println!("    - silu_8192");
    println!("    - softmax_2048");
    println!("    - rope_512");
    println!("    - causal_mask_64x64");
    println!("    - q4k_dequant_256");
    println!();
    println!("  simd-pixel-fkr: SIMD vs scalar (+-1 ULP)");
    println!("    - vector_ops_10000");
    println!("    - softmax_2048");
    println!("    - unaligned_17");
    println!("    - remainder_255");
    println!("    - relu_10000");
    println!();
    #[cfg(feature = "gpu")]
    {
        println!("  wgpu-pixel-fkr: WGPU vs scalar (+-2 ULP)");
        println!("    - large_vector_100000");
        println!("    - matmul_128x128");
        println!("    - softmax_4096");
    }
    #[cfg(not(feature = "gpu"))]
    {
        println!("  wgpu-pixel-fkr: SKIPPED (gpu feature disabled)");
    }
    println!();
    println!("  ptx-pixel-fkr: See trueno-gpu crate");
    println!();
    println!("========================================");
}
