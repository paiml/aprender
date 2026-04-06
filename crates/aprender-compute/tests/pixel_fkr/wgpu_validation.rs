//! WGPU-PIXEL-FKR: WebGPU Validation (SPEC Section 3.5.4)

use super::helpers::*;
use trueno::backends::gpu::GpuBackend;
use trueno::Vector;

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

    assert!(vectors_match(&scalar_add, wgpu_add.as_slice(), GPU_TOLERANCE, "wgpu_large_vector"));
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

    assert!(vectors_match(&scalar_result, wgpu_result.as_slice(), GPU_TOLERANCE, "wgpu_softmax"));
}
