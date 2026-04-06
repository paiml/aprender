//! Example: Soft Assertions (Feature 17)
//!
//! Demonstrates: Collecting multiple assertion failures without stopping execution
//!
//! Run with: `cargo run --example soft_assertions`
//!
//! Toyota Way: Kaizen (Continuous Improvement) - Gather all issues before reporting

use jugar_probar::prelude::*;

fn main() -> ProbarResult<()> {
    println!("=== Soft Assertions Example ===\n");

    // 1. Basic soft assertions
    println!("1. Basic soft assertions...");
    let mut soft = SoftAssertions::new();

    soft.assert_eq(&5, &5, "Five equals five");
    soft.assert_true(true, "True is true");
    soft.assert_ne(&1, &2, "One is not two");
    soft.assert_some(&Some(42), "Value exists");
    soft.assert_none(&None::<i32>, "Value is none");

    println!("   {} assertions, {} failures", 5, soft.failure_count());

    // 2. Assertions that will fail
    println!("\n2. Demonstrating failure collection...");
    let mut failing = SoftAssertions::new();

    failing.assert_eq(&5, &10, "Five equals ten");
    failing.assert_true(false, "False is true");
    failing.assert_contains("hello", "world", "hello contains world");

    println!(
        "   {} failures collected (not stopped)",
        failing.failure_count()
    );

    for (i, failure) in failing.failures().iter().enumerate() {
        println!("   Failure {}: {}", i + 1, failure.message);
    }

    // 3. Verify and get summary
    println!("\n3. Generating summary...");
    let summary = failing.summary();
    println!(
        "   Total: {}, Passed: {}, Failed: {}",
        summary.total, summary.passed, summary.failed
    );

    // 4. Advanced assertions
    println!("\n4. Advanced assertions...");
    let mut advanced = SoftAssertions::new();

    advanced.assert_approx_eq(std::f64::consts::PI, 3.14158, 0.01, "Pi approximation");
    advanced.assert_in_range(50.0, 0.0, 100.0, "Value in range");
    advanced.assert_len(&[1, 2, 3], 3, "Array length");
    advanced.assert_not_empty(&[1, 2, 3], "Array not empty");

    if advanced.all_passed() {
        println!("   All advanced assertions passed!");
    }

    println!("\n✅ Soft assertions example completed!");
    Ok(())
}
