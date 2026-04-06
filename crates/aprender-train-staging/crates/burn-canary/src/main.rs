//! Performance canary: trueno WGPU vs Burn WGPU throughput

fn main() {
    if let Err(e) = run() { eprintln!("FAIL: {e}"); std::process::exit(1); }
}

fn run() -> Result<(), String> {
    use burn::tensor::Tensor;
    use burn::backend::Wgpu;
    use std::time::Instant;
    type B = Wgpu;
    let dev = Default::default();

    eprintln!("=== Performance Canary: trueno vs Burn WGPU ===\n");

    let sizes: Vec<(usize, usize, usize)> = vec![
        (4, 2560, 9728),    // seq_len=4, hidden→intermediate (training workload)
        (32, 2560, 9728),   // seq_len=32
        (128, 2560, 9728),  // seq_len=128
        (4, 2560, 151936),  // lm_head (small seq)
        (32, 2560, 4096),   // Q projection
    ];

    let gpu = trueno::backends::gpu::GpuDevice::new().map_err(|e| format!("{e}"))?;

    for (m, k, n) in &sizes {
        let (m, k, n) = (*m, *k, *n);
        let a: Vec<f32> = (0..m*k).map(|i| ((i*7+3)%1000) as f32 / 1000.0 - 0.5).collect();
        let b: Vec<f32> = (0..k*n).map(|i| ((i*13+7)%1000) as f32 / 1000.0 - 0.5).collect();

        // Warmup
        let mut c = vec![0.0f32; m*n];
        gpu.matmul(&a, &b, &mut c, m, k, n).map_err(|e| format!("{e}"))?;
        let a_b = Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(a.clone(), [m, k]), &dev);
        let b_b = Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(b.clone(), [k, n]), &dev);
        let _ = a_b.matmul(b_b);

        // Benchmark trueno (3 runs)
        let iters = 3;
        let t0 = Instant::now();
        for _ in 0..iters {
            gpu.matmul(&a, &b, &mut c, m, k, n).map_err(|e| format!("{e}"))?;
        }
        let trueno_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // Benchmark burn (3 runs)
        let t1 = Instant::now();
        for _ in 0..iters {
            let ab = Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(a.clone(), [m, k]), &dev);
            let bb = Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(b.clone(), [k, n]), &dev);
            let _r = ab.matmul(bb).to_data();
        }
        let burn_ms = t1.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let ratio = burn_ms / trueno_ms;
        let gflops = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;
        let trueno_gflops = gflops / (trueno_ms / 1000.0);
        let burn_gflops = gflops / (burn_ms / 1000.0);

        eprintln!("  [{m}x{k}x{n}] trueno={trueno_ms:.1}ms ({trueno_gflops:.1} GFLOP/s) burn={burn_ms:.1}ms ({burn_gflops:.1} GFLOP/s) ratio={ratio:.2}x");
    }

    eprintln!("\nratio > 1.0 = trueno faster, ratio < 1.0 = burn faster");
    Ok(())
}
