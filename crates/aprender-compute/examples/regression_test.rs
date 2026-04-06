#![allow(clippy::disallowed_methods)]
//! Regression test for matmul_rectangular performance
//!
//! Specifically tests 128x512 @ 512x256 case that showed +9.8% regression

use std::time::Instant;
use trueno::Matrix;

fn main() {
    let cases = [
        (1, 512, 256, "1x512x256 (vector-matrix)"),
        (32, 128, 64, "32x128x64"),
        (64, 256, 128, "64x256x128"),
        (128, 512, 256, "128x512x256 (REGRESSION?)"),
        (256, 512, 256, "256x512x256"),
    ];

    println!("Matmul Regression Test");
    println!("======================\n");

    for (m, k, n, name) in cases {
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.001).collect();
        let ma = Matrix::from_vec(m, k, a).unwrap();
        let mb = Matrix::from_vec(k, n, b).unwrap();

        // Warmup
        for _ in 0..5 {
            let _ = ma.matmul(&mb);
        }

        let iterations = if m * k * n > 10_000_000 { 10 } else { 50 };
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = ma.matmul(&mb).unwrap();
        }
        let elapsed = start.elapsed();
        let time_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        let ops = (m * k * n) as f64;
        let gflops = (2.0 * ops) / (time_ms / 1000.0) / 1e9;

        let path = if m == 1 { "vector-matrix" } else { "general" };
        println!("{:30} {:>8.2}ms {:>5.1} GFLOPS  [{}]", name, time_ms, gflops, path);
    }

    println!("\nNote: Check if 128x512x256 shows significant degradation vs baseline");
}
