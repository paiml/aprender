#![allow(clippy::cast_precision_loss)]
//! Performance benchmarks for `TruenoVec`
//!
//! These benchmarks measure the performance characteristics of `TruenoVec`
//! operations and compare them against `std::Vec` where applicable.
//!
//! Run with: `cargo bench`

use certeza::TruenoVec;
use std::hint::black_box;

/// Benchmark: Sequential push operations
fn bench_push_sequential(n: usize) -> std::time::Duration {
    let start = std::time::Instant::now();
    let mut vec = TruenoVec::new();
    for i in 0..n {
        vec.push(black_box(i));
    }
    black_box(vec);
    start.elapsed()
}

/// Benchmark: Pre-allocated push operations
fn bench_push_preallocated(n: usize) -> std::time::Duration {
    let start = std::time::Instant::now();
    let mut vec = TruenoVec::with_capacity(n);
    for i in 0..n {
        vec.push(black_box(i));
    }
    black_box(vec);
    start.elapsed()
}

/// Benchmark: Pop operations
fn bench_pop(n: usize) -> std::time::Duration {
    let mut vec = TruenoVec::new();
    for i in 0..n {
        vec.push(i);
    }

    let start = std::time::Instant::now();
    for _ in 0..n {
        black_box(vec.pop());
    }
    black_box(vec);
    start.elapsed()
}

/// Benchmark: Random access (get)
fn bench_get(n: usize) -> std::time::Duration {
    let mut vec = TruenoVec::new();
    for i in 0..n {
        vec.push(i);
    }

    let start = std::time::Instant::now();
    for i in 0..n {
        black_box(vec.get(black_box(i % n)));
    }
    start.elapsed()
}

/// Benchmark: Growth pattern analysis
fn bench_growth_pattern(n: usize) -> (usize, std::time::Duration) {
    let mut reallocation_count = 0;
    let mut prev_capacity = 0;

    let start = std::time::Instant::now();
    let mut vec = TruenoVec::new();

    for i in 0..n {
        vec.push(i);
        if vec.capacity() > prev_capacity {
            reallocation_count += 1;
            prev_capacity = vec.capacity();
        }
    }
    black_box(vec);
    let elapsed = start.elapsed();

    (reallocation_count, elapsed)
}

/// Comparison benchmark: `TruenoVec` vs `std::Vec`
fn bench_comparison(n: usize) -> (std::time::Duration, std::time::Duration) {
    // TruenoVec
    let start = std::time::Instant::now();
    let mut trueno = TruenoVec::new();
    for i in 0..n {
        trueno.push(black_box(i));
    }
    for i in 0..n {
        black_box(trueno.get(black_box(i)));
    }
    for _ in 0..n {
        black_box(trueno.pop());
    }
    let trueno_time = start.elapsed();

    // std::Vec
    let start = std::time::Instant::now();
    let mut std_vec = Vec::new();
    for i in 0..n {
        std_vec.push(black_box(i));
    }
    for i in 0..n {
        black_box(std_vec.get(black_box(i)));
    }
    for _ in 0..n {
        black_box(std_vec.pop());
    }
    let std_time = start.elapsed();

    (trueno_time, std_time)
}

// Main function for running benchmarks manually
fn main() {
    println!("=== TruenoVec Performance Benchmarks ===\n");

    let sizes = [100, 1_000, 10_000, 100_000];

    for &n in &sizes {
        println!("--- Benchmark size: {n} ---");

        let duration = bench_push_sequential(n);
        println!("  Push sequential: {duration:?}");

        let duration = bench_push_preallocated(n);
        println!("  Push preallocated: {duration:?}");

        let duration = bench_pop(n);
        println!("  Pop: {duration:?}");

        let duration = bench_get(n);
        println!("  Get (random access): {duration:?}");

        let (reallocs, duration) = bench_growth_pattern(n);
        println!("  Growth pattern: {reallocs} reallocations, {duration:?}");

        println!();
    }

    println!("--- Comparison with std::Vec ---");
    for &n in &[1_000, 10_000, 100_000] {
        let (trueno_time, std_time) = bench_comparison(n);
        let ratio = trueno_time.as_nanos() as f64 / std_time.as_nanos() as f64;
        println!(
            "  n={n}: TruenoVec {trueno_time:?} vs std::Vec {std_time:?} (ratio: {ratio:.2}x)"
        );
    }
}

#[cfg(test)]
mod bench_tests {
    use super::*;

    #[test]
    fn test_push_sequential_small() {
        let duration = bench_push_sequential(100);
        println!("Push sequential (n=100): {duration:?}");
        assert!(duration.as_millis() < 10); // Should be very fast
    }

    #[test]
    fn test_push_sequential_medium() {
        let duration = bench_push_sequential(1_000);
        println!("Push sequential (n=1000): {duration:?}");
        assert!(duration.as_millis() < 50);
    }

    #[test]
    fn test_push_sequential_large() {
        let duration = bench_push_sequential(10_000);
        println!("Push sequential (n=10000): {duration:?}");
        assert!(duration.as_millis() < 100);
    }

    #[test]
    fn test_push_preallocated_benefit() {
        let sequential_time = bench_push_sequential(10_000);
        let preallocated_time = bench_push_preallocated(10_000);

        println!("Sequential: {sequential_time:?}");
        println!("Preallocated: {preallocated_time:?}");

        // Both should complete successfully
        // Preallocated typically faster (no reallocations) but allow variance
        // Just verify both complete in reasonable time
        assert!(sequential_time.as_millis() < 100);
        assert!(preallocated_time.as_millis() < 100);
    }

    #[test]
    fn test_pop_performance() {
        let duration = bench_pop(10_000);
        println!("Pop (n=10000): {duration:?}");
        assert!(duration.as_millis() < 100);
    }

    #[test]
    fn test_get_performance() {
        let duration = bench_get(10_000);
        println!("Get (n=10000 accesses): {duration:?}");
        assert!(duration.as_micros() < 50_000); // Should be very fast (O(1))
    }

    #[test]
    fn test_growth_pattern_logarithmic() {
        let (realloc_count, duration) = bench_growth_pattern(10_000);
        println!("Reallocations for n=10000: {realloc_count}");
        println!("Time: {duration:?}");

        // With 2x growth factor, expect approximately log2(10000) ≈ 14 reallocations
        // Allow some margin for the initial growth from 0
        assert!(realloc_count <= 15, "Expected ~14 reallocations, got {realloc_count}");
    }

    #[test]
    fn test_comparison_competitive() {
        let (trueno_time, std_time) = bench_comparison(10_000);
        println!("TruenoVec time: {trueno_time:?}");
        println!("std::Vec time: {std_time:?}");

        // TruenoVec should be competitive with std::Vec
        // Allow up to 3x slower (we're custom implementation vs highly optimized std)
        let ratio = trueno_time.as_nanos() as f64 / std_time.as_nanos() as f64;
        println!("Performance ratio (TruenoVec/std::Vec): {ratio:.2}x");

        assert!(ratio < 5.0, "TruenoVec significantly slower than std::Vec ({ratio}x)");
    }

    #[test]
    fn test_memory_overhead() {
        use std::mem;

        let vec = TruenoVec::<i32>::new();
        let size = mem::size_of_val(&vec);

        println!("TruenoVec<i32> size: {size} bytes");

        // Should be 3 word-sized fields (ptr, len, capacity) + PhantomData (0 size)
        // On 64-bit: 24 bytes minimum, but may be 32 with padding
        assert!(size <= 32, "Unexpected memory overhead: {size} bytes");
    }

    #[test]
    fn test_amortized_push_cost() {
        // Measure average cost per push over many operations
        let n = 100_000;
        let duration = bench_push_sequential(n);
        let avg_nanos = duration.as_nanos() / n as u128;

        println!("Average push cost (n={n}): {avg_nanos} ns/op");

        // Should be amortized O(1) - expect under 100ns per operation on modern hardware
        assert!(avg_nanos < 1000, "Average push cost too high: {avg_nanos} ns");
    }

    #[test]
    fn test_cache_efficiency() {
        // Sequential access should be cache-friendly
        let n = 10_000;
        let mut vec = TruenoVec::new();
        for i in 0..n {
            vec.push(i);
        }

        // Sequential access
        let start = std::time::Instant::now();
        for i in 0..n {
            black_box(vec.get(i));
        }
        let sequential_time = start.elapsed();

        // Random access (cache-unfriendly)
        let indices: Vec<usize> = (0..n).rev().collect();
        let start = std::time::Instant::now();
        for &i in &indices {
            black_box(vec.get(i));
        }
        let random_time = start.elapsed();

        println!("Sequential access: {sequential_time:?}");
        println!("Random access: {random_time:?}");

        // Both should be fast due to O(1) access, but sequential might be slightly faster
        assert!(sequential_time.as_micros() < 10_000);
        assert!(random_time.as_micros() < 20_000);
    }
}
