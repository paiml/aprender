//! Wait Mechanisms Example (PMAT-005)
//!
//! Demonstrates Playwright-compatible wait mechanisms:
//! - LoadState (load, domcontentloaded, networkidle)
//! - WaitOptions (timeout, polling interval)
//! - wait_for_url (URL pattern matching)
//! - wait_for_load_state (page load states)
//! - wait_for_function (custom predicates)
//! - wait_for_event (page events)
//! - NavigationOptions (navigation wait configuration)
//!
//! # Running
//!
//! ```bash
//! cargo run --example wait_mechanisms -p probar
//! ```
//!
//! # Playwright Parity
//!
//! These wait mechanisms match Playwright's API:
//! - `page.waitForLoadState('networkidle')` -> `waiter.wait_for_load_state(LoadState::NetworkIdle, &options)`
//! - `page.waitForURL(pattern)` -> `waiter.wait_for_url(&pattern, &options)`
//! - `page.waitForFunction(fn)` -> `waiter.wait_for_function(|| predicate(), &options)`

use jugar_probar::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

fn main() {
    println!("=== Probar Wait Mechanisms Example (PMAT-005) ===\n");

    // Demo 1: LoadState types
    demo_load_states();

    // Demo 2: WaitOptions configuration
    demo_wait_options();

    // Demo 3: NavigationOptions
    demo_navigation_options();

    // Demo 4: PageEvent types
    demo_page_events();

    // Demo 5: Waiter for URL pattern
    demo_wait_for_url();

    // Demo 6: Waiter for load state
    demo_wait_for_load_state();

    // Demo 7: Waiter for custom function
    demo_wait_for_function();

    // Demo 8: Waiter for events
    demo_wait_for_event();

    // Demo 9: Integration example
    demo_integration();

    println!("\n=== Wait Mechanisms Example Complete ===");
}

fn demo_load_states() {
    println!("--- Demo 1: LoadState Types ---\n");

    let states = [
        LoadState::Load,
        LoadState::DomContentLoaded,
        LoadState::NetworkIdle,
    ];

    for state in &states {
        println!("LoadState::{:?}", state);
        println!("  Event name: \"{}\"", state.event_name());
        println!("  Default timeout: {}ms", state.default_timeout_ms());
    }

    println!("\nDefault LoadState: {:?}", LoadState::default());
    println!();
}

fn demo_wait_options() {
    println!("--- Demo 2: WaitOptions Configuration ---\n");

    // Default options
    let default_opts = WaitOptions::default();
    println!("Default WaitOptions:");
    println!("  timeout_ms: {}", default_opts.timeout_ms);
    println!("  poll_interval_ms: {}", default_opts.poll_interval_ms);
    println!("  wait_until: {:?}", default_opts.wait_until);

    // Custom options with builder pattern
    let custom_opts = WaitOptions::new()
        .with_timeout(10_000)
        .with_poll_interval(100)
        .with_wait_until(LoadState::NetworkIdle);

    println!("\nCustom WaitOptions (builder pattern):");
    println!("  timeout_ms: {}", custom_opts.timeout_ms);
    println!("  poll_interval_ms: {}", custom_opts.poll_interval_ms);
    println!("  wait_until: {:?}", custom_opts.wait_until);

    // Duration accessors
    println!("\nDuration accessors:");
    println!("  timeout(): {:?}", custom_opts.timeout());
    println!("  poll_interval(): {:?}", custom_opts.poll_interval());

    println!();
}

fn demo_navigation_options() {
    println!("--- Demo 3: NavigationOptions ---\n");

    // Default
    let default_nav = NavigationOptions::default();
    println!("Default NavigationOptions:");
    println!("  timeout_ms: {}", default_nav.timeout_ms);
    println!("  wait_until: {:?}", default_nav.wait_until);
    println!("  url_pattern: {:?}", default_nav.url_pattern);

    // With all options
    let custom_nav = NavigationOptions::new()
        .with_timeout(5000)
        .with_wait_until(LoadState::DomContentLoaded)
        .with_url(UrlPattern::Contains("example.com".into()));

    println!("\nCustom NavigationOptions:");
    println!("  timeout_ms: {}", custom_nav.timeout_ms);
    println!("  wait_until: {:?}", custom_nav.wait_until);
    println!("  url_pattern: {:?}", custom_nav.url_pattern);

    println!();
}

fn demo_page_events() {
    println!("--- Demo 4: PageEvent Types ---\n");

    let events = [
        PageEvent::Load,
        PageEvent::DomContentLoaded,
        PageEvent::Close,
        PageEvent::Console,
        PageEvent::Dialog,
        PageEvent::Popup,
        PageEvent::Request,
        PageEvent::Response,
        PageEvent::PageError,
        PageEvent::Download,
    ];

    println!("Common PageEvent types:");
    for event in &events {
        println!("  PageEvent::{:?} -> \"{}\"", event, event.as_str());
    }

    println!();
}

fn demo_wait_for_url() {
    println!("--- Demo 5: Wait for URL Pattern ---\n");

    let mut waiter = Waiter::new();
    let options = WaitOptions::new().with_timeout(100);

    // Set URL to match
    waiter.set_url("https://example.com/dashboard");

    // Test different URL patterns
    let patterns = [
        (
            "Exact",
            UrlPattern::Exact("https://example.com/dashboard".into()),
        ),
        ("Contains", UrlPattern::Contains("example.com".into())),
        ("Prefix", UrlPattern::Prefix("https://example".into())),
        ("Glob", UrlPattern::Glob("https://*.com/*".into())),
    ];

    println!("Current URL: \"https://example.com/dashboard\"\n");

    for (name, pattern) in patterns {
        let result = waiter.wait_for_url(&pattern, &options);
        let status = if result.is_ok() {
            "✓ matched"
        } else {
            "✗ no match"
        };
        println!("  UrlPattern::{} -> {}", name, status);
    }

    println!();
}

fn demo_wait_for_load_state() {
    println!("--- Demo 6: Wait for Load State ---\n");

    let mut waiter = Waiter::new();
    let options = WaitOptions::new().with_timeout(100);

    // Test Load state
    waiter.set_load_state(LoadState::Load);
    let result = waiter.wait_for_load_state(LoadState::Load, &options);
    println!(
        "LoadState::Load with state=Load: {}",
        if result.is_ok() {
            "✓ satisfied"
        } else {
            "✗ timeout"
        }
    );

    // DomContentLoaded is satisfied by Load
    let result = waiter.wait_for_load_state(LoadState::DomContentLoaded, &options);
    println!(
        "LoadState::DomContentLoaded with state=Load: {}",
        if result.is_ok() {
            "✓ satisfied (Load implies DOMContentLoaded)"
        } else {
            "✗ timeout"
        }
    );

    // NetworkIdle check
    waiter.set_pending_requests(0);
    // Need to wait for idle threshold
    println!("\nNetworkIdle semantics:");
    println!("  - No pending requests: ✓");
    println!(
        "  - No activity for {}ms: checking...",
        NETWORK_IDLE_THRESHOLD_MS
    );

    println!();
}

fn demo_wait_for_function() {
    println!("--- Demo 7: Wait for Custom Function ---\n");

    // Immediate success
    let waiter = Waiter::new();
    let options = WaitOptions::new().with_timeout(100);

    let result = waiter.wait_for_function(|| true, &options);
    println!(
        "wait_for_function(|| true): {}",
        if result.is_ok() {
            "✓ immediate success"
        } else {
            "✗ failed"
        }
    );

    // Timeout case
    let result = waiter.wait_for_function(|| false, &options);
    println!(
        "wait_for_function(|| false): {}",
        if result.is_err() {
            "✓ correctly timed out"
        } else {
            "✗ unexpected success"
        }
    );

    // Counter-based wait
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = counter.clone();

    // Simulate async increment
    std::thread::spawn(move || {
        for _ in 0..5 {
            std::thread::sleep(Duration::from_millis(10));
            counter_clone.fetch_add(1, Ordering::SeqCst);
        }
    });

    let options = WaitOptions::new().with_timeout(200).with_poll_interval(10);

    let result = waiter.wait_for_function(|| counter.load(Ordering::SeqCst) >= 3, &options);
    println!(
        "wait_for_function(|| counter >= 3): {}",
        if result.is_ok() {
            "✓ condition met"
        } else {
            "✗ timeout"
        }
    );

    println!();
}

fn demo_wait_for_event() {
    println!("--- Demo 8: Wait for Events ---\n");

    let mut waiter = Waiter::new();
    let options = WaitOptions::new().with_timeout(100);

    // Record an event
    waiter.record_event(PageEvent::Load);
    waiter.record_event(PageEvent::DomContentLoaded);

    // Check for recorded events
    let result = waiter.wait_for_event(&PageEvent::Load, &options);
    println!(
        "wait_for_event(PageEvent::Load): {}",
        if result.is_ok() {
            "✓ found"
        } else {
            "✗ not found"
        }
    );

    let result = waiter.wait_for_event(&PageEvent::Popup, &options);
    println!(
        "wait_for_event(PageEvent::Popup): {}",
        if result.is_err() {
            "✓ correctly timed out (not recorded)"
        } else {
            "✗ unexpected"
        }
    );

    // Clear and re-check
    waiter.clear_events();
    let result = waiter.wait_for_event(&PageEvent::Load, &options);
    println!(
        "After clear_events(), wait for Load: {}",
        if result.is_err() {
            "✓ correctly timed out"
        } else {
            "✗ unexpected"
        }
    );

    println!();
}

fn demo_integration() {
    println!("--- Demo 9: Integration Example ---\n");

    println!("Simulating page navigation flow:\n");

    let mut waiter = Waiter::new();
    let ready = Arc::new(AtomicBool::new(false));
    let ready_clone = ready.clone();

    // Simulate navigation in background
    std::thread::spawn(move || {
        // Simulate navigation delay
        std::thread::sleep(Duration::from_millis(30));
        ready_clone.store(true, Ordering::SeqCst);
    });

    // Configure navigation options
    let nav_options = NavigationOptions::new()
        .with_timeout(200)
        .with_wait_until(LoadState::Load);

    println!("NavigationOptions configured:");
    println!("  timeout: {}ms", nav_options.timeout_ms);
    println!("  wait_until: {:?}", nav_options.wait_until);

    // Set up waiter state
    waiter.set_url("https://example.com/app");
    waiter.set_load_state(LoadState::Load);

    // Wait for navigation
    let result = waiter.wait_for_navigation(&nav_options);
    println!(
        "\nwait_for_navigation result: {}",
        if result.is_ok() {
            "✓ navigation complete"
        } else {
            "✗ navigation failed"
        }
    );

    if let Ok(wait_result) = result {
        println!("  Waited for: {}", wait_result.waited_for);
        println!("  Elapsed: {:?}", wait_result.elapsed);
    }

    // Convenience function demo
    println!("\nConvenience functions:");
    let result = wait_until(|| ready.load(Ordering::SeqCst), 100);
    println!(
        "  wait_until(predicate, 100ms): {}",
        if result.is_ok() {
            "✓ success"
        } else {
            "✗ timeout"
        }
    );

    println!("\n  wait_timeout(50ms)...");
    let start = std::time::Instant::now();
    wait_timeout(50);
    println!("  Elapsed: {:?}", start.elapsed());

    println!();
}
