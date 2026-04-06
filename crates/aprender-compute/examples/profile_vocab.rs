#![allow(clippy::disallowed_methods)]
//! Profile vocab projection with renacer
//!
//! Run with: renacer -s -- cargo run --release --features tracing --example profile_vocab

use std::time::Instant;
use trueno::Matrix;

fn main() {
    // Whisper vocab projection: 1×384 @ 384×51865
    let rows = 1;
    let inner = 384;
    let cols = 51865;

    println!("Profiling vocab projection: {rows}×{inner} @ {inner}×{cols}");

    let a: Vec<f32> = (0..rows * inner).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..inner * cols).map(|i| (i as f32) * 0.0001).collect();

    let ma = Matrix::from_vec(rows, inner, a).unwrap();
    let mb = Matrix::from_vec(inner, cols, b).unwrap();

    // Warmup
    for _ in 0..3 {
        let _ = ma.matmul(&mb);
    }

    // Profile 5 iterations
    let start = Instant::now();
    for _ in 0..5 {
        let _ = ma.matmul(&mb).unwrap();
    }
    let elapsed = start.elapsed();

    println!("5 iterations: {:.1}ms", elapsed.as_secs_f64() * 1000.0);
    println!("Per iteration: {:.1}ms", elapsed.as_secs_f64() * 1000.0 / 5.0);
}
