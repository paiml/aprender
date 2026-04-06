//! Example: Retry Assertions (Feature 18)
//!
//! Demonstrates: Auto-retrying assertions for eventually-consistent states
//!
//! Run with: `cargo run --example retry_assertions`
//!
//! Toyota Way: Heijunka (Level Loading) - Consistent polling intervals

use jugar_probar::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

fn main() -> ProbarResult<()> {
    println!("=== Retry Assertions Example ===\n");

    // 1. Simple retry that succeeds
    println!("1. Retry that succeeds on 3rd attempt...");
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = counter;

    let assertion = retry_true(
        move || {
            let count = counter_clone.fetch_add(1, Ordering::SeqCst) + 1;
            count >= 3
        },
        "Counter reaches 3",
    )
    .with_timeout(Duration::from_millis(500))
    .with_poll_interval(Duration::from_millis(50));

    match assertion.verify() {
        Ok(result) => println!(
            "   Succeeded after {} attempts in {:?}",
            result.attempts, result.duration
        ),
        Err(e) => println!("   Failed: {}", e.message),
    }

    // 2. Retry with equality check
    println!("\n2. Retry until value equals expected...");
    let value = Arc::new(AtomicUsize::new(0));
    let value_clone = value;

    let assertion = retry_eq(move || value_clone.fetch_add(1, Ordering::SeqCst), 5)
        .with_timeout(Duration::from_millis(500))
        .with_poll_interval(Duration::from_millis(30));

    match assertion.verify() {
        Ok(result) => println!("   Value reached 5 after {} attempts", result.attempts),
        Err(e) => println!("   Timeout: {}", e.message),
    }

    // 3. Retry with Option check
    println!("\n3. Retry until Option becomes Some...");
    let has_value = Arc::new(AtomicUsize::new(0));
    let has_value_clone = has_value;

    let assertion = retry_some(move || {
        let count = has_value_clone.fetch_add(1, Ordering::SeqCst);
        if count >= 2 {
            Some(42)
        } else {
            None
        }
    })
    .with_timeout(Duration::from_millis(300))
    .with_poll_interval(Duration::from_millis(30));

    match assertion.verify() {
        Ok(result) => println!("   Got Some value after {} attempts", result.attempts),
        Err(e) => println!("   Remained None: {}", e.message),
    }

    // 4. Retry that times out
    println!("\n4. Retry that times out...");
    let assertion = retry_true(
        || false, // Never succeeds
        "This will timeout",
    )
    .with_timeout(Duration::from_millis(100))
    .with_poll_interval(Duration::from_millis(20));

    match assertion.verify() {
        Ok(_) => println!("   Unexpectedly succeeded"),
        Err(e) => println!(
            "   Expected timeout after {} attempts: {}",
            e.attempts, e.message
        ),
    }

    // 5. Using fast config for quick checks
    println!("\n5. Using fast retry config...");
    let fast_counter = Arc::new(AtomicUsize::new(0));
    let fast_counter_clone = fast_counter;

    let config = RetryConfig::fast();
    let assertion = retry_true(
        move || fast_counter_clone.fetch_add(1, Ordering::SeqCst) >= 2,
        "Fast check",
    )
    .with_timeout(config.timeout)
    .with_poll_interval(config.poll_interval);

    match assertion.verify() {
        Ok(result) => println!("   Fast retry succeeded in {:?}", result.duration),
        Err(e) => println!("   Fast retry failed: {}", e.message),
    }

    println!("\n✅ Retry assertions example completed!");
    Ok(())
}
