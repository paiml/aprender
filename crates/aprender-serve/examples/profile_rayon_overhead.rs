//! Profile Rayon overhead per operation
use rayon::prelude::*;
use std::time::Instant;

fn main() {
    // Measure rayon overhead for different work sizes
    let work_sizes = [256, 512, 1024, 1536, 2048, 4096, 8960];
    let iterations = 1000;

    println!("=== Rayon Overhead Analysis ===");
    println!(
        "{:>8} {:>12} {:>12} {:>12}",
        "Size", "Sequential", "Parallel", "Overhead"
    );

    for &size in &work_sizes {
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; size];

        // Sequential
        let start = Instant::now();
        for _ in 0..iterations {
            for (i, o) in output.iter_mut().enumerate() {
                *o = data[i] * 2.0;
            }
        }
        let seq_us = start.elapsed().as_micros() as f64 / iterations as f64;

        // Parallel (mimics our matmul structure)
        let start = Instant::now();
        for _ in 0..iterations {
            output
                .par_iter_mut()
                .enumerate()
                .with_min_len(64)
                .for_each(|(i, o)| {
                    *o = data[i] * 2.0;
                });
        }
        let par_us = start.elapsed().as_micros() as f64 / iterations as f64;

        let overhead_us = par_us - seq_us;
        println!(
            "{:>8} {:>10.1} us {:>10.1} us {:>10.1} us",
            size, seq_us, par_us, overhead_us
        );
    }

    // Measure pure rayon dispatch overhead (empty work)
    println!("\n=== Pure Rayon Dispatch Overhead ===");
    let dummy = vec![0u8; 1000];
    let iterations = 10000;

    let start = Instant::now();
    for _ in 0..iterations {
        dummy.par_iter().for_each(|_| {});
    }
    let dispatch_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!("Par dispatch (1000 items): {:.2} us", dispatch_us);

    // rayon::join overhead
    let start = Instant::now();
    for _ in 0..iterations {
        rayon::join(|| {}, || {});
    }
    let join_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!("Rayon::join (empty):       {:.2} us", join_us);

    // Impact analysis
    println!("\n=== Impact Analysis ===");
    let calls_per_token = 140; // ~140 matmuls per token
    let overhead_estimate = dispatch_us * calls_per_token as f64;
    println!(
        "Estimated overhead/token:  {:.1} us ({:.2} ms)",
        overhead_estimate,
        overhead_estimate / 1000.0
    );
    println!(
        "At 140 matmuls/token with {}us dispatch = {:.1}ms",
        dispatch_us,
        overhead_estimate / 1000.0
    );
}
