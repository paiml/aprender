//! A/B Test: Scalar vs Trueno matvec
use std::hint::black_box;
use std::time::Instant;
use trueno::{select_best_available_backend, Matrix, Vector};

fn scalar_matvec(weight: &[f32], input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0; rows];
    for i in 0..rows {
        let mut sum = 0.0f32;
        for j in 0..cols {
            sum += weight[i * cols + j] * input[j];
        }
        output[i] = sum;
    }
    output
}

fn main() {
    let rows = 2560; // phi2 hidden_dim
    let cols = 2560;
    let iterations = 50;

    println!("=== Trueno A/B Matvec Test ({}x{}) ===", rows, cols);
    println!("Backend: {:?}", select_best_available_backend());

    let weight: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 % 100.0) * 0.001)
        .collect();
    let input: Vec<f32> = (0..cols).map(|i| (i as f32 % 100.0) * 0.01).collect();

    // === SCALAR BENCHMARK ===
    let _ = black_box(scalar_matvec(
        black_box(&weight),
        black_box(&input),
        rows,
        cols,
    ));
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(scalar_matvec(&weight, &input, rows, cols));
    }
    let scalar_time = start.elapsed();
    let scalar_per_iter = scalar_time.as_micros() as f64 / iterations as f64;

    println!("\n=== Scalar ===");
    println!(
        "Per matvec: {:.2}µs ({:.2}ms)",
        scalar_per_iter,
        scalar_per_iter / 1000.0
    );

    // === TRUENO BENCHMARK ===
    let weight_mat = Matrix::from_vec(rows, cols, weight.clone()).expect("test");
    let input_vec = Vector::from_slice(&input);
    let _ = black_box(weight_mat.matvec(&input_vec).expect("test")); // warmup

    let start = Instant::now();
    for _ in 0..iterations {
        black_box(weight_mat.matvec(&input_vec).expect("test"));
    }
    let trueno_time = start.elapsed();
    let trueno_per_iter = trueno_time.as_micros() as f64 / iterations as f64;

    println!("\n=== Trueno ===");
    println!(
        "Per matvec: {:.2}µs ({:.2}ms)",
        trueno_per_iter,
        trueno_per_iter / 1000.0
    );

    // === COMPARISON ===
    let speedup = scalar_per_iter / trueno_per_iter;
    println!("\n=== Results ===");
    println!("Speedup: {:.2}x", speedup);

    if speedup > 1.0 {
        println!("✅ Trueno is {:.1}x faster than scalar", speedup);
    } else {
        println!("❌ Scalar is {:.1}x faster than trueno", 1.0 / speedup);
    }

    // Verify correctness
    let scalar_result = scalar_matvec(&weight, &input, rows, cols);
    let trueno_result = weight_mat.matvec(&input_vec).expect("test");
    let max_diff: f32 = scalar_result
        .iter()
        .zip(trueno_result.as_slice().iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Max difference: {:.2e}", max_diff);
}
