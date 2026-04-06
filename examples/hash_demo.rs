#![allow(clippy::disallowed_methods)]
//! Hash Module Demo - SIMD-optimized key hashing for KV stores
//!
//! Run with: `cargo run --example hash_demo`
//!
//! This example demonstrates trueno's hash functions designed for
//! high-performance key-value store operations.

use trueno::{hash_key, hash_keys_batch, hash_keys_batch_with_backend, Backend};

fn main() {
    println!("=== Trueno Hash Module Demo ===\n");

    // Single key hashing
    demo_single_key();

    // Batch hashing
    demo_batch_hashing();

    // Backend selection
    demo_backend_selection();

    // Performance comparison
    demo_performance();
}

fn demo_single_key() {
    println!("1. Single Key Hashing");
    println!("   -----------------");

    let keys = ["hello", "world", "trueno", ""];

    for key in keys {
        let hash = hash_key(key);
        println!("   hash_key({:?}) = 0x{:016x}", key, hash);
    }

    // Demonstrate determinism
    let h1 = hash_key("test");
    let h2 = hash_key("test");
    println!("\n   Deterministic: hash(\"test\") == hash(\"test\"): {}", h1 == h2);
    println!();
}

fn demo_batch_hashing() {
    println!("2. Batch Hashing (SIMD-optimized)");
    println!("   ------------------------------");

    let keys = ["user:1001", "user:1002", "user:1003", "user:1004"];
    let hashes = hash_keys_batch(&keys);

    println!("   Input keys: {:?}", keys);
    println!("   Output hashes:");
    for (key, hash) in keys.iter().zip(hashes.iter()) {
        println!("     {} -> 0x{:016x}", key, hash);
    }
    println!();
}

fn demo_backend_selection() {
    println!("3. Backend Selection");
    println!("   ------------------");

    let keys = ["a", "b", "c", "d", "e", "f", "g", "h"];

    let scalar = hash_keys_batch_with_backend(&keys, Backend::Scalar);
    let auto = hash_keys_batch_with_backend(&keys, Backend::Auto);

    println!("   Backend::Scalar results: {:?}", &scalar[..4]);
    println!("   Backend::Auto results:   {:?}", &auto[..4]);
    println!("   Results match: {}", scalar == auto);
    println!();
}

fn demo_performance() {
    println!("4. Performance Demo");
    println!("   -----------------");

    // Generate test keys
    let keys: Vec<String> = (0..10000).map(|i| format!("key:{:08}", i)).collect();
    let key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();

    // Warm up
    let _ = hash_keys_batch(&key_refs);

    // Benchmark batch
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _ = hash_keys_batch(&key_refs);
    }
    let batch_time = start.elapsed();

    // Benchmark sequential
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _: Vec<u64> = key_refs.iter().map(|k| hash_key(k)).collect();
    }
    let seq_time = start.elapsed();

    println!("   10,000 keys x 100 iterations:");
    println!("     Batch (hash_keys_batch): {:?}", batch_time);
    println!("     Sequential (hash_key):   {:?}", seq_time);
    println!("     Speedup: {:.2}x", seq_time.as_nanos() as f64 / batch_time.as_nanos() as f64);
    println!();
}
