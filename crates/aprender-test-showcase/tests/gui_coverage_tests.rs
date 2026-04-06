//! GUI Coverage Tests for Showcase Calculator
//!
//! Demonstrates Probar's trivially simple GUI coverage tracking.
//!
//! The goal: Make GUI coverage as easy as:
//!   1. Define what needs testing
//!   2. Run your tests
//!   3. Get a simple percentage

use jugar_probar::{calculator_coverage, game_coverage, gui_coverage, UxCoverageBuilder};
use showcase_calculator::prelude::*;

// =============================================================================
// EXAMPLE 1: The Simplest Possible GUI Coverage
// =============================================================================

#[test]
fn test_simplest_gui_coverage() {
    // Step 1: Define GUI elements (one line!)
    let mut gui = gui_coverage! {
        buttons: ["btn-0", "btn-1", "btn-equals", "btn-clear"],
        screens: ["calculator"]
    };

    // Step 2: Simulate interactions (via your driver or directly)
    let mut driver = WasmDriver::new();

    // When button is clicked, record it
    driver.type_input("1");
    gui.click("btn-1");

    driver.type_input("0");
    gui.click("btn-0");

    driver.click_equals();
    gui.click("btn-equals");

    gui.visit("calculator");

    // Step 3: Check coverage - ONE LINE!
    println!("{}", gui.summary()); // "GUI: 80% (3/4 elements, 1/1 screens)"
    assert!(gui.meets(75.0), "Expected at least 75% GUI coverage");
}

// =============================================================================
// EXAMPLE 2: Using Pre-built Calculator Coverage
// =============================================================================

#[test]
fn test_calculator_preset_coverage() {
    // Probar provides a pre-built calculator coverage definition
    let mut gui = calculator_coverage();

    // Simulate a full test session
    let mut driver = WasmDriver::new();

    // Test all digit buttons
    for digit in 0..=9 {
        driver.type_input(&digit.to_string());
        gui.click(&format!("btn-{}", digit));
    }

    // Test operators
    gui.click("btn-plus");
    gui.click("btn-minus");
    gui.click("btn-times");
    gui.click("btn-divide");
    gui.click("btn-equals");
    gui.click("btn-clear");
    gui.click("btn-decimal");
    gui.click("btn-power");
    gui.click("btn-open-paren");
    gui.click("btn-close-paren");

    // Visit screens
    gui.visit("calculator");
    gui.visit("history");

    // Assert 100% GUI coverage
    println!("{}", gui.summary());
    assert!(gui.is_complete(), "Calculator GUI should be fully covered");
}

// =============================================================================
// EXAMPLE 3: Custom Game Coverage
// =============================================================================

#[test]
fn test_game_coverage_pattern() {
    // For games, use the game_coverage helper
    let mut gui = game_coverage(
        &["start_game", "pause", "restart", "quit"],
        &["title", "playing", "paused", "game_over"],
    );

    // Simulate a game session
    gui.visit("title");
    gui.click("start_game");
    gui.visit("playing");
    gui.click("pause");
    gui.visit("paused");
    // Resume...
    gui.visit("game_over");
    gui.click("restart");
    gui.click("quit");

    // Check coverage
    println!("{}", gui.summary());
    assert!(gui.is_complete());
}

// =============================================================================
// EXAMPLE 4: Integration with WasmDriver
// =============================================================================

#[test]
fn test_wasm_driver_gui_coverage() {
    // Define coverage requirements
    let mut gui = UxCoverageBuilder::new()
        .button("btn-7")
        .button("btn-times")
        .button("btn-6")
        .button("btn-equals")
        .button("btn-clear")
        .screen("calculator")
        .build();

    // Create driver and run test
    let mut driver = WasmDriver::new();

    // Test: 7 * 6 = 42
    driver.type_input("7 * 6");
    gui.click("btn-7");
    gui.click("btn-times");
    gui.click("btn-6");

    driver.click_equals();
    gui.click("btn-equals");

    // Verify calculation
    assert_eq!(driver.get_result(), "42");

    // Record screen visit
    gui.visit("calculator");

    // Clear
    driver.click_clear();
    gui.click("btn-clear");

    // Report coverage
    println!("\n{}", gui.summary());
    assert!(gui.is_complete(), "All GUI elements should be covered");
}

// =============================================================================
// EXAMPLE 5: Partial Coverage is OK - Track Progress
// =============================================================================

#[test]
fn test_partial_coverage_tracking() {
    let mut gui = gui_coverage! {
        buttons: ["btn-0", "btn-1", "btn-2", "btn-3", "btn-4",
                  "btn-5", "btn-6", "btn-7", "btn-8", "btn-9"],
        screens: ["calculator", "history", "settings"]
    };

    // Only test some buttons
    gui.click("btn-1");
    gui.click("btn-2");
    gui.click("btn-3");
    gui.visit("calculator");

    // Get detailed stats
    let percent = gui.percent();
    println!("Current GUI coverage: {:.1}%", percent);
    println!("{}", gui.summary());

    // You might want 100%, but tracking progress is valuable
    assert!(
        gui.meets(25.0),
        "Should have at least 25% coverage: {}",
        gui.summary()
    );
}

// =============================================================================
// EXAMPLE 6: Full Specification Test with GUI Coverage
// =============================================================================

#[test]
fn test_full_spec_with_gui_coverage() {
    use showcase_calculator::driver::verify_basic_arithmetic;

    // Create GUI tracker
    let mut gui = calculator_coverage();

    // Run the full specification (existing tests)
    let mut driver = WasmDriver::new();
    verify_basic_arithmetic(&mut driver);

    // Record the interactions that happened
    // (In a real integration, the driver would call gui.click() automatically)
    gui.click("btn-plus");
    gui.click("btn-minus");
    gui.click("btn-times");
    gui.click("btn-divide");
    gui.click("btn-equals");
    gui.visit("calculator");

    // The spec tests cover functionality, GUI tracker covers UI
    println!("\nFunctional tests: PASS");
    println!("GUI Coverage: {}", gui.summary());
}
