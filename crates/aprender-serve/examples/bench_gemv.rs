//! Minimal GEMV benchmark - 30 second feedback loop
use std::time::Instant;

fn main() {
    let k = 4096usize;
    let n = 4096usize;

    let a: Vec<f32> = (0..k).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..(k * n)).map(|_| 0.001).collect();

    let mut sched = realizar::gpu::CudaScheduler::new().expect("CUDA init");

    // Warmup
    sched.matmul(&a, &b, 1, k, n).expect("test");

    // Bench
    let iters = 100;
    let start = Instant::now();
    for _ in 0..iters {
        sched.matmul(&a, &b, 1, k, n).expect("test");
    }
    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    let ms_per_op = total_ms / iters as f64;

    // Estimate: 32 layers × 6 matmuls/layer ≈ 192 matmuls/token
    let matmuls_per_token = 192.0;
    let tok_per_sec = 1000.0 / (ms_per_op * matmuls_per_token);

    println!("GEMV 1×4096×4096: {:.2}ms/op", ms_per_op);
    println!("  → {:.1} matmuls/s", 1000.0 / ms_per_op);
    println!("  → {:.1} tok/s (192 matmuls/tok)", tok_per_sec);
    println!("  Ollama: 228 tok/s");
}
