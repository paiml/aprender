//! Demonstrates the certeza testing-effectiveness framework.
//!
//! This example showcases:
//! - `TruenoVec`: a verified growable vector
//! - Chaos engineering configuration
//! - Basic arithmetic functions with property-tested guarantees
//!
//! Run with: `cargo run --example quality_check`

use certeza::chaos::{ChaosConfig, ChaosError};
use certeza::{add, multiply, subtract, TruenoVec};
use std::time::Duration;

fn main() {
    println!("=== Certeza Quality Check Demo ===\n");

    // --- TruenoVec: verified growable vector ---
    println!("--- TruenoVec (verified vector) ---");
    let mut vec = TruenoVec::new();
    for i in 1..=5 {
        vec.push(i);
    }
    println!("Pushed 1..5 -> len={}, capacity={}", vec.len(), vec.capacity());
    println!("vec[2] = {:?}", vec.get(2));
    println!("pop()  = {:?}", vec.pop());
    println!("After pop: len={}", vec.len());

    // Pre-allocated vector
    let mut preallocated = TruenoVec::with_capacity(1000);
    for i in 0..1000 {
        preallocated.push(i);
    }
    println!(
        "Pre-allocated 1000 elements: len={}, capacity={}",
        preallocated.len(),
        preallocated.capacity()
    );

    // --- Arithmetic functions (property-tested) ---
    println!("\n--- Arithmetic (property-tested) ---");
    println!("add(40, 2)       = {}", add(40, 2));
    println!("subtract(100, 58) = {}", subtract(100, 58));
    println!("multiply(6, 7)   = {}", multiply(6, 7));

    // Demonstrate commutativity (verified by proptest in the library)
    assert_eq!(add(3, 7), add(7, 3));
    assert_eq!(multiply(4, 9), multiply(9, 4));
    println!("Commutativity verified for add and multiply.");

    // --- Chaos engineering configuration ---
    println!("\n--- Chaos Engineering (renacer presets) ---");

    let gentle = ChaosConfig::gentle();
    println!(
        "Gentle:     memory={}MB, cpu={:.0}%, timeout={}s",
        gentle.memory_limit / (1024 * 1024),
        gentle.cpu_limit * 100.0,
        gentle.timeout.as_secs()
    );

    let aggressive = ChaosConfig::aggressive();
    println!(
        "Aggressive: memory={}MB, cpu={:.0}%, timeout={}s, signals={}",
        aggressive.memory_limit / (1024 * 1024),
        aggressive.cpu_limit * 100.0,
        aggressive.timeout.as_secs(),
        aggressive.signal_injection
    );

    // Custom chaos config via builder
    let custom = ChaosConfig::new()
        .with_memory_limit(256 * 1024 * 1024)
        .with_cpu_limit(0.75)
        .with_timeout(Duration::from_secs(60))
        .with_signal_injection(false)
        .build();
    println!(
        "Custom:     memory={}MB, cpu={:.0}%, timeout={}s",
        custom.memory_limit / (1024 * 1024),
        custom.cpu_limit * 100.0,
        custom.timeout.as_secs()
    );

    // Demonstrate chaos error types
    let err =
        ChaosError::Timeout { elapsed: Duration::from_secs(15), limit: Duration::from_secs(10) };
    println!("\nExample chaos error: {err}");

    println!("\nAll checks passed.");
}
