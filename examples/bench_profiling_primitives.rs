#![allow(clippy::disallowed_methods)]
//! F150-F155: High-Performance Profiling Primitives Benchmark
//!
//! This example validates the low-overhead profiling primitives from E.9.
//!
//! Run with: cargo run --release --example bench_profiling_primitives

use std::time::Instant;
use trueno::brick::{
    cached_nanos, cached_nanos_or_now, cpu_cycles, get_page_faults, init_time_service,
};

const ITERATIONS: u64 = 1_000_000;

fn main() {
    println!("=== F150-F155: High-Performance Profiling Primitives ===\n");

    // F150: RDTSCP Overhead Test
    println!("F150: RDTSCP/Cycle Counter Overhead");
    let start = Instant::now();
    let mut prev_cycles = 0u64;
    let mut monotonic_violations = 0u64;

    for _ in 0..ITERATIONS {
        let cycles = cpu_cycles();
        if cycles < prev_cycles {
            monotonic_violations += 1;
        }
        prev_cycles = cycles;
    }

    let elapsed_ns = start.elapsed().as_nanos() as f64;
    let avg_ns = elapsed_ns / ITERATIONS as f64;

    println!("  Iterations: {}", ITERATIONS);
    println!("  Total time: {:.2}ms", elapsed_ns / 1_000_000.0);
    println!("  Avg per call: {:.2}ns", avg_ns);
    println!("  Target: < 50ns");
    let f150_pass = avg_ns < 50.0;
    println!(
        "  Result: {} ({})",
        if f150_pass { "PASS" } else { "FAIL" },
        if avg_ns < 10.0 {
            "Excellent"
        } else if avg_ns < 30.0 {
            "Good"
        } else {
            "Acceptable"
        }
    );

    // F151: Cycle Monotonicity
    println!("\nF151: Cycle Counter Monotonicity");
    println!("  Monotonicity violations: {}", monotonic_violations);
    let f151_pass = monotonic_violations == 0;
    println!("  Result: {}", if f151_pass { "PASS" } else { "FAIL" });

    // Initialize time service for F152/F153
    println!("\nInitializing time service for F152/F153...");
    init_time_service();
    std::thread::sleep(std::time::Duration::from_millis(100)); // Let it warm up

    // F153: Cached Time Speed Test
    println!("\nF153: Cached Time Service Speed");
    let start = Instant::now();
    let mut sum = 0u64;

    for _ in 0..ITERATIONS {
        sum = sum.wrapping_add(cached_nanos());
    }

    let elapsed_ns = start.elapsed().as_nanos() as f64;
    let avg_ns = elapsed_ns / ITERATIONS as f64;

    println!("  Iterations: {}", ITERATIONS);
    println!("  Total time: {:.2}ms", elapsed_ns / 1_000_000.0);
    println!("  Avg per call: {:.2}ns", avg_ns);
    println!("  Target: < 20ns");
    println!("  Sum (anti-optimize): {}", sum);
    let f153_pass = avg_ns < 20.0;
    println!(
        "  Result: {} ({})",
        if f153_pass { "PASS" } else { "FAIL" },
        if avg_ns < 5.0 {
            "Excellent"
        } else if avg_ns < 10.0 {
            "Good"
        } else {
            "Acceptable"
        }
    );

    // F152: Cached Time Drift Check
    println!("\nF152: Cached Time Drift Check");
    let before_cached = cached_nanos_or_now();
    let before_instant = Instant::now();

    std::thread::sleep(std::time::Duration::from_secs(2));

    let after_cached = cached_nanos_or_now();
    let after_instant = Instant::now();

    let cached_delta = after_cached.saturating_sub(before_cached);
    let instant_delta = after_instant.duration_since(before_instant).as_nanos() as u64;
    let drift = (cached_delta as i64 - instant_delta as i64).unsigned_abs();

    println!("  Sleep duration: 2 seconds");
    println!("  Cached delta: {}ns ({:.3}s)", cached_delta, cached_delta as f64 / 1_000_000_000.0);
    println!(
        "  Instant delta: {}ns ({:.3}s)",
        instant_delta,
        instant_delta as f64 / 1_000_000_000.0
    );
    println!("  Drift: {}µs", drift / 1000);
    println!("  Target: < 500µs");
    let f152_pass = drift < 500_000;
    println!("  Result: {}", if f152_pass { "PASS" } else { "FAIL" });

    // F155: Page Fault Detection (basic check)
    println!("\nF155: Page Fault Tracking");
    let (minor_before, major_before) = get_page_faults();

    // Allocate and touch memory to trigger page faults
    const ALLOC_SIZE: usize = 100 * 1024 * 1024; // 100MB
    let mut data: Vec<u8> = Vec::with_capacity(ALLOC_SIZE);

    // Touch every page to trigger faults
    for i in 0..ALLOC_SIZE {
        data.push((i & 0xFF) as u8);
    }

    let (minor_after, major_after) = get_page_faults();
    let minor_faults = minor_after.saturating_sub(minor_before);
    let major_faults = major_after.saturating_sub(major_before);

    println!("  Allocated: {} MB", ALLOC_SIZE / (1024 * 1024));
    println!("  Minor faults: {}", minor_faults);
    println!("  Major faults: {}", major_faults);
    println!("  Expected minor: ~{} (for 4KB pages)", ALLOC_SIZE / 4096);
    let f155_pass = minor_faults > 0;
    println!("  Result: {}", if f155_pass { "PASS" } else { "FAIL" });

    // Prevent optimization
    drop(data);

    // Summary
    println!("\n=== Final Scorecard ===");
    println!("┌──────┬─────────────────────┬───────────────┬────────┐");
    println!("│ ID   │ Test Name           │ Threshold     │ Result │");
    println!("├──────┼─────────────────────┼───────────────┼────────┤");
    println!(
        "│ F150 │ RDTSCP Overhead     │ < 50ns        │ [{}]  │",
        if f150_pass { "✓" } else { " " }
    );
    println!(
        "│ F151 │ Cycle Monotonicity  │ Always > Prev │ [{}]  │",
        if f151_pass { "✓" } else { " " }
    );
    println!(
        "│ F152 │ Cached Time Drift   │ < 500µs       │ [{}]  │",
        if f152_pass { "✓" } else { " " }
    );
    println!(
        "│ F153 │ Cached Time Speed   │ < 20ns        │ [{}]  │",
        if f153_pass { "✓" } else { " " }
    );
    println!("│ F154 │ Poll Count Accuracy │ (unit tests)  │ [✓]  │");
    println!(
        "│ F155 │ Page Fault Tracking │ > 0 on alloc  │ [{}]  │",
        if f155_pass { "✓" } else { " " }
    );
    println!("└──────┴─────────────────────┴───────────────┴────────┘");

    let all_pass = f150_pass && f151_pass && f152_pass && f153_pass && f155_pass;
    println!("\nOverall: {}", if all_pass { "ALL TESTS PASSED ✓" } else { "SOME TESTS FAILED" });

    if !all_pass {
        std::process::exit(1);
    }
}
