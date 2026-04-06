//! Example: Basic Test
//!
//! Demonstrates: Core test execution and basic assertions
//!
//! Run with: `cargo run --example basic_test`
//!
//! Toyota Way: Jidoka (Autonomation) - Fail-fast on quality issues

use jugar_probar::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

fn main() -> ProbarResult<()> {
    println!("=== Basic Test Example ===\n");

    // 1. Create a simple test suite
    println!("1. Setting up test suite...");
    let mut suite = TestSuite::new("example_suite");
    suite.add_test(TestCase::new("test_addition"));
    suite.add_test(TestCase::new("test_subtraction"));
    println!(
        "   Test suite '{}' created with {} tests",
        suite.name,
        suite.test_count()
    );

    // 2. Create test results
    println!("\n2. Creating test results...");
    let passed_result = TestResult::pass("test_addition");
    let failed_result = TestResult::fail("test_subtraction", "Expected 5, got 4");

    println!("   Passed test: {}", passed_result.name);
    println!(
        "   Failed test: {} - {}",
        failed_result.name,
        failed_result.error.as_deref().unwrap_or("")
    );

    // 3. Demonstrate soft assertions
    println!("\n3. Demonstrating soft assertions...");
    let mut soft = SoftAssertions::new();

    // These won't panic immediately - they collect failures
    soft.assert_eq(&5, &5, "Five equals five");
    soft.assert_true(true, "True is true");
    soft.assert_eq(&"hello", &"hello", "Strings match");

    println!("   Collected {} failures", soft.failure_count());
    if soft.all_passed() {
        println!("   All soft assertions passed!");
    }

    // 4. Demonstrate retry assertions
    println!("\n4. Demonstrating retry assertions...");
    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts_clone = Arc::clone(&attempts);

    // Create a retry assertion that succeeds on 3rd attempt
    let assertion = retry_true(
        move || {
            let count = attempts_clone.fetch_add(1, Ordering::SeqCst) + 1;
            count >= 3 // Succeed on 3rd attempt
        },
        "Waiting for condition to be true",
    )
    .with_timeout(Duration::from_millis(200))
    .with_poll_interval(Duration::from_millis(20));

    match assertion.verify() {
        Ok(result) => println!(
            "   Retry succeeded after {} attempts (took {:?})",
            result.attempts, result.duration
        ),
        Err(e) => println!("   Retry failed: {}", e.message),
    }

    // 5. Create and run test harness
    println!("\n5. Running test harness...");
    let harness = TestHarness::new();
    let results = harness.run(&suite);

    println!("   Results - all_passed: {}", results.all_passed());
    println!(
        "   Passed: {}, Failed: {}, Total: {}",
        results.passed_count(),
        results.failed_count(),
        results.total()
    );

    println!("\n✅ Basic test example completed successfully!");
    Ok(())
}
