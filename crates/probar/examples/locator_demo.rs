//! Locator Demo - Playwright-style Element Selection
//!
//! Demonstrates the Probar Locator API for testing WASM games
//! with auto-waiting and fluent assertions.
//!
//! # Running
//!
//! ```bash
//! cargo run --example locator_demo -p probar
//! ```
//!
//! # Features
//!
//! - CSS and entity selectors
//! - Test ID selectors
//! - Fluent `expect()` assertions
//! - Drag and drop operations

#![allow(
    clippy::uninlined_format_args,
    clippy::std_instead_of_core,
    clippy::unwrap_used
)]

use jugar_probar::{
    expect, BoundingBox, Locator, Point, Selector, DEFAULT_POLL_INTERVAL_MS, DEFAULT_TIMEOUT_MS,
};
use std::time::Duration;

fn main() {
    println!("=== Probar Locator Demo ===\n");

    // Demo 1: Selector types
    demo_selectors();

    // Demo 2: Locator configuration
    demo_locator_config();

    // Demo 3: Expect assertions
    demo_expect_assertions();

    // Demo 4: Drag operations
    demo_drag_operations();

    // Demo 5: Bounding box operations
    demo_bounding_box();

    println!("\n=== Locator Demo Complete ===");
}

fn demo_selectors() {
    println!("--- Demo 1: Selector Types ---\n");

    // CSS selector (most common)
    let css = Selector::css("#player-score");
    println!("CSS: {:?}", css);
    println!("  Query: {}", css.to_query());

    // Test ID for stable tests
    let test_id = Selector::test_id("submit-button");
    println!("\nTestId: {:?}", test_id);
    println!("  Query: {}", test_id.to_query());

    // Text content
    let text = Selector::text("Click to start!");
    println!("\nText: {:?}", text);
    println!("  Query: {}", text.to_query());

    // WASM Entity selector (game-specific)
    let entity = Selector::entity("player");
    println!("\nEntity: {:?}", entity);
    println!("  Query: {}", entity.to_query());

    // Count query
    let css_count = Selector::css(".enemy").to_count_query();
    println!("\nCount query: {}", css_count);

    println!();
}

fn demo_locator_config() {
    println!("--- Demo 2: Locator Configuration ---\n");

    // Default locator
    let locator = Locator::new(".game-canvas");

    println!("Default constants:");
    println!("  Timeout: {}ms", DEFAULT_TIMEOUT_MS);
    println!("  Poll interval: {}ms", DEFAULT_POLL_INTERVAL_MS);

    // Check options
    let opts = locator.options();
    println!("\nLocator options:");
    println!("  timeout: {:?}", opts.timeout);
    println!("  poll_interval: {:?}", opts.poll_interval);
    println!("  strict: {}", opts.strict);
    println!("  visible: {}", opts.visible);

    // Locator with custom timeout
    let custom_locator = Locator::new("button")
        .with_timeout(Duration::from_secs(10))
        .with_strict(false);
    println!("\nCustom locator:");
    println!("  timeout: {:?}", custom_locator.options().timeout);
    println!("  strict: {}", custom_locator.options().strict);

    // Chained filtering with text
    let filtered = Locator::new("button").with_text("Start Game");
    println!("\nFiltered locator: {:?}", filtered.selector());

    // Entity selection
    let entity_locator = Locator::new("canvas").entity("hero");
    println!("Entity locator: {:?}", entity_locator.selector());

    println!();
}

fn demo_expect_assertions() {
    println!("--- Demo 3: Expect Assertions (Fluent API) ---\n");

    let locator = Locator::new("[data-testid='score-display']");

    // Create expect wrapper
    let expectation = expect(locator.clone());

    println!("Available assertions:");
    println!("  expect(locator).to_be_visible()");
    println!("  expect(locator).to_be_hidden()");
    println!("  expect(locator).to_have_text(\"Score: 100\")");
    println!("  expect(locator).to_contain_text(\"Score\")");
    println!("  expect(locator).to_have_count(3)");

    // Demonstrate validation
    let has_text = expectation.to_have_text("10");
    println!(
        "\nValidating 'to_have_text(\"10\")' against '10': {:?}",
        has_text.validate("10").is_ok()
    );
    println!(
        "Validating 'to_have_text(\"10\")' against '20': {:?}",
        has_text.validate("20").is_ok()
    );

    let contains = expect(locator.clone()).to_contain_text("Score");
    println!(
        "\nValidating 'to_contain_text(\"Score\")' against 'Score: 100': {:?}",
        contains.validate("Score: 100").is_ok()
    );

    let count = expect(locator).to_have_count(3);
    println!(
        "Validating 'to_have_count(3)' with count 3: {:?}",
        count.validate_count(3).is_ok()
    );
    println!(
        "Validating 'to_have_count(3)' with count 5: {:?}",
        count.validate_count(5).is_ok()
    );

    println!();
}

fn demo_drag_operations() {
    println!("--- Demo 4: Drag and Drop Operations ---\n");

    let locator = Locator::new("canvas").entity("hero");

    // Create drag operation
    let target = Point::new(500.0, 300.0);
    let drag = locator
        .drag_to(&target)
        .steps(20)
        .duration(Duration::from_millis(800))
        .build();

    println!("Drag operation created:");
    println!("  Target: ({}, {})", target.x, target.y);
    println!("  Action type: {:?}", std::mem::discriminant(&drag));

    // Actions
    println!("\nAvailable locator actions:");
    println!("  locator.click()");
    println!("  locator.double_click()");
    println!("  locator.fill(\"text\")");
    println!("  locator.drag_to(&point).steps(10).duration(ms).build()");

    // Queries
    println!("\nAvailable locator queries:");
    println!("  locator.text_content()");
    println!("  locator.is_visible()");
    println!("  locator.bounding_box()");
    println!("  locator.count()");

    println!();
}

fn demo_bounding_box() {
    println!("--- Demo 5: Bounding Box Operations ---\n");

    // Create bounding boxes
    let box1 = BoundingBox::new(100.0, 100.0, 50.0, 50.0);

    // Center point
    let center = box1.center();
    println!(
        "Box 1: x={}, y={}, w={}, h={}",
        box1.x, box1.y, box1.width, box1.height
    );
    println!("  Center: ({}, {})", center.x, center.y);

    // Contains point check
    let inside = Point::new(110.0, 110.0);
    let outside = Point::new(200.0, 200.0);
    println!(
        "\nContains point ({}, {}): {}",
        inside.x,
        inside.y,
        box1.contains(&inside)
    );
    println!(
        "Contains point ({}, {}): {}",
        outside.x,
        outside.y,
        box1.contains(&outside)
    );

    // Edge cases
    let on_edge = Point::new(100.0, 100.0);
    println!(
        "Contains point on edge ({}, {}): {}",
        on_edge.x,
        on_edge.y,
        box1.contains(&on_edge)
    );

    println!();
}
