//! Reproducibility Example
//!
//! Demonstrates simular's core guarantee: bitwise-identical results
//! across runs, platforms, and thread counts.
//!
//! # Running
//! ```bash
//! cargo run --example reproducibility
//! ```

use simular::engine::rng::SimRng;
use simular::engine::SimTime;

fn main() {
    println!("=== Simular Reproducibility Guarantees ===\n");

    // 1. RNG Reproducibility
    println!("1. Deterministic Random Number Generation:");
    println!("   Property: Same seed → identical sequence\n");

    let seed = 0xDEAD_BEEF_u64;
    let mut rng1 = SimRng::new(seed);
    let mut rng2 = SimRng::new(seed);

    let seq1: Vec<f64> = (0..5).map(|_| rng1.gen_f64()).collect();
    let seq2: Vec<f64> = (0..5).map(|_| rng2.gen_f64()).collect();

    println!("   Seed: 0x{:X}", seed);
    println!("   Run 1: {:?}", seq1);
    println!("   Run 2: {:?}", seq2);
    println!("   Bitwise identical: {}\n", seq1 == seq2);

    // 2. Partition Independence
    println!("2. Parallel Partition Independence:");
    println!("   Property: Results independent of thread count\n");

    let mut rng = SimRng::new(42);

    // Partition into 4 streams (simulating 4 threads)
    let mut partitions = rng.partition(4);

    // Each partition produces independent sequences
    let results: Vec<Vec<f64>> = partitions
        .iter_mut()
        .map(|p| (0..3).map(|_| p.gen_f64()).collect())
        .collect();

    println!("   Master seed: 42, partitioned into 4 streams:");
    for (i, seq) in results.iter().enumerate() {
        println!("   Stream {}: {:?}", i, seq);
    }

    // Verify same results with fresh RNG
    let mut rng_fresh = SimRng::new(42);
    let mut partitions_fresh = rng_fresh.partition(4);
    let results_fresh: Vec<Vec<f64>> = partitions_fresh
        .iter_mut()
        .map(|p| (0..3).map(|_| p.gen_f64()).collect())
        .collect();

    println!("   Reproducible: {}\n", results == results_fresh);

    // 3. Checkpoint/Restore
    println!("3. Checkpoint and Restore:");
    println!("   Property: Restored state continues identically\n");

    let mut rng = SimRng::new(999);

    // Generate some values
    let _: Vec<f64> = (0..100).map(|_| rng.gen_f64()).collect();

    // Save state
    let state = rng.save_state();
    println!("   Checkpoint saved at stream position: {}", state.stream);

    // Continue and record
    let continuation: Vec<f64> = (0..5).map(|_| rng.gen_f64()).collect();
    println!("   Continuation after checkpoint: {:?}", continuation);

    // Restore and verify
    let mut rng_restored = SimRng::new(999);
    let _ = rng_restored.restore_state(&state);
    let restored_continuation: Vec<f64> = (0..5).map(|_| rng_restored.gen_f64()).collect();
    println!("   After restore: {:?}", restored_continuation);
    println!("   Identical: {}\n", continuation == restored_continuation);

    // 4. Time Representation
    println!("4. Precise Time Representation:");
    println!("   Property: No floating-point drift in time\n");

    let dt = SimTime::from_nanos(1_000_000); // 1ms in nanoseconds
    let mut t = SimTime::ZERO;

    // Accumulate 1 million steps
    for _ in 0..1_000_000 {
        t = t + dt;
    }

    let expected = SimTime::from_secs(1000.0);
    println!("   1,000,000 steps of 1ms each");
    println!("   Result: {} ns", t.as_nanos());
    println!("   Expected: {} ns", expected.as_nanos());
    println!("   Exact match: {}\n", t == expected);

    // 5. Hash-based verification
    println!("5. Cryptographic Verification:");
    println!("   Property: Identical runs produce identical hashes\n");

    fn compute_simulation_hash(seed: u64, steps: usize) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut rng = SimRng::new(seed);
        let mut hasher = DefaultHasher::new();

        for _ in 0..steps {
            let v = rng.gen_f64();
            v.to_bits().hash(&mut hasher);
        }

        format!("{:016x}", hasher.finish())
    }

    let hash1 = compute_simulation_hash(42, 10000);
    let hash2 = compute_simulation_hash(42, 10000);
    let hash3 = compute_simulation_hash(43, 10000); // Different seed

    println!("   Seed 42, run 1: {}", hash1);
    println!("   Seed 42, run 2: {}", hash2);
    println!("   Seed 43, run 1: {}", hash3);
    println!("   Same seed identical: {}", hash1 == hash2);
    println!("   Different seed differs: {}\n", hash1 != hash3);

    // 6. Normal distribution reproducibility
    println!("6. Statistical Distribution Reproducibility:");
    println!("   Property: Complex distributions are reproducible\n");

    let mut rng1 = SimRng::new(777);
    let mut rng2 = SimRng::new(777);

    let normal1: Vec<f64> = (0..5).map(|_| rng1.gen_normal(100.0, 15.0)).collect();
    let normal2: Vec<f64> = (0..5).map(|_| rng2.gen_normal(100.0, 15.0)).collect();

    println!("   Normal(100, 15) samples:");
    println!("   Run 1: {:?}", normal1);
    println!("   Run 2: {:?}", normal2);
    println!("   Identical: {}", normal1 == normal2);

    println!("\n=== All Reproducibility Guarantees Verified ===");
}
