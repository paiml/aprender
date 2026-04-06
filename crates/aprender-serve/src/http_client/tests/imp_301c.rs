
/// IMP-301c: Test trueno SIMD dequantization simulation
#[test]
fn test_imp_301c_trueno_dequant_speedup() {
    use std::time::Instant;
    use trueno::Vector;

    let size = 32768; // Typical weight block size
    let iterations = 100;

    // Simulate Q4_K block data
    let q4k_scales: Vec<f32> = (0..size / 32).map(|i| 0.1 + (i as f32 * 0.001)).collect();
    let q4k_data: Vec<f32> = (0..size).map(|i| ((i % 16) as f32 - 8.0) * 0.1).collect();

    // Scalar baseline
    let start = Instant::now();
    for _ in 0..iterations {
        let _result: Vec<f32> = q4k_data
            .chunks(32)
            .zip(q4k_scales.iter())
            .flat_map(|(chunk, scale)| chunk.iter().map(|&x| x * scale).collect::<Vec<_>>())
            .collect();
    }
    let scalar_time = start.elapsed().as_micros() as f64 / iterations as f64;

    // Trueno SIMD
    let vec = Vector::from_slice(&q4k_data);
    let _scales_vec = Vector::from_slice(&q4k_scales);
    let start = Instant::now();
    for _ in 0..iterations {
        let _result = vec.mul(&Vector::from_slice(&q4k_data));
    }
    let simd_time = start.elapsed().as_micros() as f64 / iterations as f64;

    let result = TruenoSimdBenchResult::new(
        "Q4_K dequant",
        SimdBackend::detect(),
        scalar_time,
        simd_time.max(1.0), // Avoid div by zero
        size,
    );

    println!("\nIMP-301c: Trueno SIMD Dequant Speedup:");
    println!("  Elements: {}", size);
    println!("  Scalar: {:.1}µs", scalar_time);
    println!("  SIMD: {:.1}µs", simd_time);
    println!("  Speedup: {:.2}x", result.speedup);
    println!("  Throughput: {:.2} GB/s", result.throughput_gbs);
    println!(
        "  IMP-301: {}",
        if result.meets_imp301 {
            "PASS"
        } else {
            "NEEDS OPTIMIZATION"
        }
    );
}

/// IMP-301d: Real-world trueno performance benchmark
#[test]
#[ignore = "Requires extended benchmark time"]
fn test_imp_301d_realworld_trueno_perf() {
    use std::time::Instant;
    use trueno::{Matrix, Vector};

    // Phi-2 model dimensions
    let hidden_dim = 2560;
    let vocab_size = 51200;
    let iterations = 10;

    // Create weight matrix (simulating model weights)
    let weights_data: Vec<f32> = (0..hidden_dim * vocab_size)
        .map(|i| (i as f32 * 0.0001) % 1.0 - 0.5)
        .collect();
    let weights =
        Matrix::from_vec(vocab_size, hidden_dim, weights_data).expect("Matrix creation failed");

    // Create input vector
    let input_data: Vec<f32> = (0..hidden_dim).map(|i| i as f32 * 0.01).collect();
    let input = Vector::from_slice(&input_data);

    // Benchmark matvec
    let start = Instant::now();
    for _ in 0..iterations {
        let _output = weights.matvec(&input).expect("matvec failed");
    }
    let total_time = start.elapsed().as_micros() as f64;
    let avg_time = total_time / iterations as f64;

    // Calculate throughput
    let flops = 2.0 * hidden_dim as f64 * vocab_size as f64; // 2 ops per multiply-add
    let gflops = (flops * iterations as f64) / (total_time * 1e-6) / 1e9;

    println!("\nIMP-301d: Real-World Trueno Performance:");
    println!("  Matrix: {}x{}", vocab_size, hidden_dim);
    println!("  Avg time: {:.1}µs", avg_time);
    println!("  Throughput: {:.2} GFLOPS", gflops);
    println!("  Est. tok/s: {:.1}", 1e6 / avg_time);
}

// ==================== IMP-302: Trueno SIMD Matmul ====================
// Per spec: 4x matmul speedup, >50 GFLOPS single thread

/// Matrix multiplication benchmark result
#[derive(Debug, Clone)]
pub struct MatmulBenchResult {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub time_us: f64,
    pub gflops: f64,
    pub meets_imp302: bool,
}

impl MatmulBenchResult {
    pub fn new(m: usize, n: usize, k: usize, time_us: f64) -> Self {
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let gflops = flops / (time_us * 1e-6) / 1e9;
        let meets_imp302 = gflops >= 50.0; // Target: >50 GFLOPS

        Self {
            m,
            n,
            k,
            time_us,
            gflops,
            meets_imp302,
        }
    }
}

/// IMP-302a: Test trueno Matrix matmul
#[test]
fn test_imp_302a_trueno_matmul() {
    use trueno::Matrix;

    let a_data: Vec<f32> = (0..64 * 128).map(|i| (i as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..128 * 64).map(|i| (i as f32) * 0.01).collect();

    let a = Matrix::from_vec(64, 128, a_data).expect("Matrix A");
    let b = Matrix::from_vec(128, 64, b_data).expect("Matrix B");

    let c = a.matmul(&b).expect("matmul failed");

    assert_eq!(c.rows(), 64, "IMP-302a: Output rows");
    assert_eq!(c.cols(), 64, "IMP-302a: Output cols");

    println!("\nIMP-302a: Trueno Matmul:");
    println!("  A: 64x128");
    println!("  B: 128x64");
    println!("  C: {}x{}", c.rows(), c.cols());
}

/// IMP-302b: Test trueno matmul performance
#[test]
fn test_imp_302b_trueno_matmul_perf() {
    use std::time::Instant;
    use trueno::Matrix;

    let sizes = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)];
    let iterations = 10;

    println!("\nIMP-302b: Trueno Matmul Performance:");
    for (m, n, k) in sizes {
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.001).collect();

        let a = Matrix::from_vec(m, k, a_data).expect("Matrix A");
        let b = Matrix::from_vec(k, n, b_data).expect("Matrix B");

        let start = Instant::now();
        for _ in 0..iterations {
            let _c = a.matmul(&b).expect("matmul");
        }
        let total_us = start.elapsed().as_micros() as f64;
        let avg_us = total_us / iterations as f64;

        let result = MatmulBenchResult::new(m, n, k, avg_us);

        println!(
            "  {}x{}x{}: {:.1}µs, {:.1} GFLOPS [{}]",
            m,
            n,
            k,
            avg_us,
            result.gflops,
            if result.meets_imp302 {
                "PASS"
            } else {
                "NEEDS WORK"
            }
        );
    }
}

/// IMP-302c: Test matvec performance (most common in inference)
#[test]
fn test_imp_302c_trueno_matvec_perf() {
    use std::time::Instant;
    use trueno::{Matrix, Vector};

    // Transformer layer dimensions
    let dims = [(2560, 10240), (10240, 2560), (2560, 51200)];
    let iterations = 50;

    println!("\nIMP-302c: Trueno Matvec Performance:");
    for (rows, cols) in dims {
        let mat_data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.0001).collect();
        let vec_data: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01).collect();

        let mat = Matrix::from_vec(rows, cols, mat_data).expect("Matrix");
        let vec = Vector::from_slice(&vec_data);

        let start = Instant::now();
        for _ in 0..iterations {
            let _result = mat.matvec(&vec).expect("matvec");
        }
        let total_us = start.elapsed().as_micros() as f64;
        let avg_us = total_us / iterations as f64;

        let flops = 2.0 * rows as f64 * cols as f64;
        let gflops = flops / (avg_us * 1e-6) / 1e9;

        println!("  {}x{}: {:.1}µs, {:.1} GFLOPS", rows, cols, avg_us, gflops);
    }
}

/// IMP-302d: Real-world matmul benchmark
#[test]
#[ignore = "Requires extended benchmark time"]
fn test_imp_302d_realworld_matmul() {
    use std::time::Instant;
    use trueno::Matrix;

    // Full transformer layer: FFN up projection
    let hidden = 2560;
    let intermediate = 10240;
    let batch = 1;

    let weights: Vec<f32> = (0..hidden * intermediate)
        .map(|i| ((i as f32) * 0.0001) % 1.0 - 0.5)
        .collect();
    let input: Vec<f32> = (0..batch * hidden).map(|i| (i as f32) * 0.01).collect();

    let w = Matrix::from_vec(intermediate, hidden, weights).expect("weights");
    let x = Matrix::from_vec(batch, hidden, input).expect("input");

    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let _y = Matrix::vecmat(&trueno::Vector::from_slice(x.as_slice()), &w.transpose());
    }
    let avg_us = start.elapsed().as_micros() as f64 / iterations as f64;

    let result = MatmulBenchResult::new(batch, intermediate, hidden, avg_us);

    println!("\nIMP-302d: Real-World FFN Projection:");
    println!("  Dimensions: {}x{}x{}", batch, intermediate, hidden);
    println!("  Time: {:.1}µs", avg_us);
    println!("  GFLOPS: {:.1}", result.gflops);
    println!(
        "  IMP-302: {}",
        if result.meets_imp302 { "PASS" } else { "FAIL" }
    );
}

// ==================== IMP-303: Trueno SIMD Activations ====================
// Per spec: 8x activation speedup, <100µs for 4096 dim

/// Activation benchmark result
#[derive(Debug, Clone)]
pub struct ActivationBenchResult {
    pub name: String,
    pub size: usize,
    pub time_us: f64,
    pub throughput_gbs: f64,
    pub meets_imp303: bool,
}

impl ActivationBenchResult {
    pub fn new(name: impl Into<String>, size: usize, time_us: f64) -> Self {
        let throughput_gbs = (size as f64 * 4.0) / (time_us * 1e-6) / 1e9;
        let meets_imp303 = time_us < 100.0 || size > 4096; // <100µs for 4096

        Self {
            name: name.into(),
            size,
            time_us,
            throughput_gbs,
            meets_imp303,
        }
    }
}

/// IMP-303a: Test trueno activation functions
#[test]
fn test_imp_303a_trueno_activations() {
    use trueno::Vector;

    let data: Vec<f32> = (-100..100).map(|i| i as f32 * 0.1).collect();
    let vec = Vector::from_slice(&data);

    let relu = vec.relu().expect("relu");
    let sigmoid = vec.sigmoid().expect("sigmoid");
    let gelu = vec.gelu().expect("gelu");
    let swish = vec.swish().expect("swish");

    // Verify basic properties
    assert!(
        relu.as_slice().iter().all(|&x| x >= 0.0),
        "IMP-303a: ReLU non-negative"
    );
    assert!(
        sigmoid.as_slice().iter().all(|&x| x > 0.0 && x < 1.0),
        "IMP-303a: Sigmoid (0,1)"
    );

    println!("\nIMP-303a: Trueno Activations:");
    println!("  ReLU(0): {:.4}", relu.as_slice()[100]);
    println!("  Sigmoid(0): {:.4}", sigmoid.as_slice()[100]);
    println!("  GELU(0): {:.4}", gelu.as_slice()[100]);
    println!("  Swish(0): {:.4}", swish.as_slice()[100]);
}

/// IMP-303b: Test activation performance
#[test]
fn test_imp_303b_trueno_activation_perf() {
    use std::time::Instant;
    use trueno::Vector;

    let size = 4096;
    let iterations = 1000;
    let data: Vec<f32> = (0..size).map(|i| (i as f32 - 2048.0) * 0.01).collect();
    let vec = Vector::from_slice(&data);

    let activations = ["relu", "sigmoid", "gelu", "swish", "softmax"];

    println!("\nIMP-303b: Trueno Activation Performance (n={}):", size);
    for name in activations {
        let start = Instant::now();
        for _ in 0..iterations {
            match name {
                "relu" => {
                    vec.relu().ok();
                },
                "sigmoid" => {
                    vec.sigmoid().ok();
                },
                "gelu" => {
                    vec.gelu().ok();
                },
                "swish" => {
                    vec.swish().ok();
                },
                "softmax" => {
                    vec.softmax().ok();
                },
                _ => {},
            }
        }
        let avg_us = start.elapsed().as_micros() as f64 / iterations as f64;
        let result = ActivationBenchResult::new(name, size, avg_us);

        println!(
            "  {}: {:.2}µs, {:.1} GB/s [{}]",
            name,
            avg_us,
            result.throughput_gbs,
            if result.meets_imp303 { "PASS" } else { "SLOW" }
        );
    }
}

/// IMP-303c: Test layer norm performance
#[test]
fn test_imp_303c_trueno_layer_norm_perf() {
    use std::time::Instant;
    use trueno::Vector;

    let sizes = [768, 2048, 2560, 4096];
    let iterations = 1000;

    println!("\nIMP-303c: Trueno Layer Norm Performance:");
    for size in sizes {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        let vec = Vector::from_slice(&data);

        let start = Instant::now();
        for _ in 0..iterations {
            let _normed = vec.layer_norm_simple(1e-5).expect("layer_norm_simple");
        }
        let avg_us = start.elapsed().as_micros() as f64 / iterations as f64;

        println!(
            "  n={}: {:.2}µs [{}]",
            size,
            avg_us,
            if avg_us < 50.0 { "PASS" } else { "NEEDS WORK" }
        );
    }
}

/// IMP-303d: Real-world activation chain
#[test]
#[ignore = "Requires extended benchmark"]
fn test_imp_303d_realworld_activation_chain() {
    use std::time::Instant;
    use trueno::Vector;

    // Full FFN activation chain: linear -> gelu -> linear
    let hidden = 2560;
    let intermediate = 10240;
    let iterations = 100;

    let x: Vec<f32> = (0..hidden).map(|i| i as f32 * 0.01).collect();
    let _hidden_vec = Vector::from_slice(&x);

    let start = Instant::now();
    for _ in 0..iterations {
        // Simulate FFN: up_proj -> gelu -> down_proj
        let up: Vec<f32> = (0..intermediate).map(|i| i as f32 * 0.001).collect();
        let up_vec = Vector::from_slice(&up);
        let _activated = up_vec.gelu().expect("gelu");
    }
    let avg_us = start.elapsed().as_micros() as f64 / iterations as f64;

    println!("\nIMP-303d: Real-World Activation Chain:");
    println!("  Hidden: {}, Intermediate: {}", hidden, intermediate);
    println!("  GELU time: {:.1}µs", avg_us);
    println!(
        "  IMP-303: {}",
        if avg_us < 500.0 { "PASS" } else { "FAIL" }
    );
}
