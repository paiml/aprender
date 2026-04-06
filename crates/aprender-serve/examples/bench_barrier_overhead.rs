//! PAR-126: Benchmark barrier synchronization overhead
//! Compare std::sync::Barrier vs Rayon dispatch

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

fn main() {
    let num_threads = rayon::current_num_threads();
    println!("=== Barrier vs Rayon Dispatch Overhead ===");
    println!("Threads: {}\n", num_threads);

    let iterations = 10000;

    // 1. std::sync::Barrier overhead
    let barrier = Arc::new(Barrier::new(num_threads));
    let handles: Vec<_> = (0..num_threads - 1)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                for _ in 0..iterations {
                    barrier.wait();
                    barrier.wait();
                }
            })
        })
        .collect();

    let start = Instant::now();
    for _ in 0..iterations {
        barrier.wait(); // Signal workers
        barrier.wait(); // Wait for completion
    }
    let barrier_ns = start.elapsed().as_nanos() as f64 / iterations as f64;

    for h in handles {
        h.join().unwrap();
    }
    println!(
        "std::sync::Barrier (round-trip): {:.1} ns = {:.2} us",
        barrier_ns,
        barrier_ns / 1000.0
    );

    // 2. Rayon par_iter dispatch overhead (empty work)
    use rayon::prelude::*;
    let dummy = vec![0u8; num_threads * 64];
    let start = Instant::now();
    for _ in 0..iterations {
        dummy.par_iter().for_each(|_| {});
    }
    let rayon_ns = start.elapsed().as_nanos() as f64 / iterations as f64;
    println!(
        "Rayon par_iter (empty):          {:.1} ns = {:.2} us",
        rayon_ns,
        rayon_ns / 1000.0
    );

    // 3. rayon::join overhead
    let start = Instant::now();
    for _ in 0..iterations {
        rayon::join(|| {}, || {});
    }
    let join_ns = start.elapsed().as_nanos() as f64 / iterations as f64;
    println!(
        "Rayon::join (empty):             {:.1} ns = {:.2} us",
        join_ns,
        join_ns / 1000.0
    );

    // 4. std::thread::scope (for reference)
    let start = Instant::now();
    for _ in 0..iterations {
        thread::scope(|s| {
            for _ in 0..num_threads {
                s.spawn(|| {});
            }
        });
    }
    let scope_ns = start.elapsed().as_nanos() as f64 / iterations as f64;
    println!(
        "std::thread::scope (empty):      {:.1} ns = {:.2} us",
        scope_ns,
        scope_ns / 1000.0
    );

    // Impact analysis
    println!("\n=== Impact on Forward Pass (28 layers, 5 ops/layer) ===");
    let ops = 140;
    let barrier_total_us = ops as f64 * 2.0 * barrier_ns / 1000.0; // 2 barriers per op
    let rayon_total_us = ops as f64 * rayon_ns / 1000.0;
    println!(
        "Barrier-based:   {:.1} us = {:.2} ms",
        barrier_total_us,
        barrier_total_us / 1000.0
    );
    println!(
        "Rayon par_iter:  {:.1} us = {:.2} ms",
        rayon_total_us,
        rayon_total_us / 1000.0
    );
    println!(
        "Savings:         {:.1} ms per token",
        (rayon_total_us - barrier_total_us) / 1000.0
    );
}
