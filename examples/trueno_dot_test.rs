//! Test trueno dot product speedup after 4-accumulator optimization
use std::hint::black_box;
use std::time::Instant;
use trueno::{select_best_available_backend, Vector};

fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn main() {
    let sizes = [80, 256, 2560]; // phi2 head_dim, typical, hidden_dim

    println!("=== Trueno Dot Product Speedup Test ===");
    println!("Backend: {:?}", select_best_available_backend());
    println!("\nSize\t\tScalar(ns)\tTrueno(ns)\tSpeedup");

    for &size in &sizes {
        let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.002).collect();

        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);

        let iterations = 10000;

        // Warmup
        black_box(scalar_dot(&a, &b));
        let _ = black_box(va.dot(&vb));

        // Benchmark scalar
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(scalar_dot(&a, &b));
        }
        let scalar_time = start.elapsed().as_nanos() / iterations;

        // Benchmark trueno
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = black_box(va.dot(&vb));
        }
        let trueno_time = start.elapsed().as_nanos() / iterations;

        let speedup = scalar_time as f64 / trueno_time as f64;
        println!(
            "{}\t\t{}\t\t{}\t\t{:.2}x",
            size, scalar_time, trueno_time, speedup
        );
    }
}
