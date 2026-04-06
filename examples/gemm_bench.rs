//! GEMM benchmark: naive 16×16 vs CUTLASS-style 64×64 tiled
//!
//! Run: cargo run --features gpu --example gemm_bench --release

fn main() {
    #[cfg(feature = "gpu")]
    {
        use entrenar::autograd::wgpu_training::WgpuTrainer;
        use std::time::Instant;

        let trainer = WgpuTrainer::new().expect("wgpu init failed");

        // Training-sized matrices: batch=4, seq=512 → M=2048, K=3584, N=3584
        let shapes: &[(u32, u32, u32, &str)] = &[
            (1, 3584, 3584, "decode (M=1)"),
            (4, 3584, 3584, "small batch (M=4)"),
            (32, 3584, 3584, "medium batch (M=32)"),
            (128, 3584, 3584, "large batch (M=128)"),
            (512, 3584, 3584, "training batch (M=512)"),
            (2048, 3584, 3584, "full training (M=2048)"),
            // FFN shapes
            (128, 3584, 18944, "FFN up (M=128)"),
            (128, 18944, 3584, "FFN down (M=128)"),
        ];

        println!("GEMM Benchmark — wgpu/Vulkan on GB10");
        println!("{:<30} {:>10} {:>10} {:>12}", "Shape", "Time (ms)", "GFLOPS", "Throughput");
        println!("{}", "-".repeat(65));

        for &(m, k, n, label) in shapes {
            // Create random-ish data
            let a_data: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.001).sin()).collect();
            let b_data: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.0007).cos()).collect();

            let a = trainer.upload(&a_data);
            let b = trainer.upload(&b_data);
            let c = trainer.zeros((m * n) as usize);

            // Warmup
            trainer.matmul_forward(&a, &b, &c, m, k, n);
            // Force sync
            let _ = trainer.download(&c);

            // Benchmark: 5 iterations
            let iters = 5;
            let start = Instant::now();
            for _ in 0..iters {
                trainer.matmul_forward(&a, &b, &c, m, k, n);
            }
            // Force sync after all dispatches
            let _ = trainer.download(&c);
            let elapsed = start.elapsed();

            let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / f64::from(iters);
            let flops = 2.0 * f64::from(m) * f64::from(k) * f64::from(n); // 2*M*K*N FLOPs per GEMM
            let gflops = flops / (ms_per_iter / 1000.0) / 1e9;

            println!(
                "{:<30} {:>10.1} {:>10.1} {:>10.1} tok/s",
                format!("{} [{}×{}×{}]", label, m, k, n),
                ms_per_iter,
                gflops,
                if ms_per_iter > 0.0 { f64::from(m) / (ms_per_iter / 1000.0) } else { 0.0 }
            );
        }
    }

    #[cfg(not(feature = "gpu"))]
    {
        eprintln!("Requires --features gpu");
    }
}
