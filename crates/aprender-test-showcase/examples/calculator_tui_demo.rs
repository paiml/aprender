//! TUI Calculator Demo
//!
//! This example demonstrates the TuiDriver running the same
//! unified test specifications as the WASM driver.
//!
//! Run with: cargo run --example calculator_tui_demo --features tui

#![allow(clippy::unwrap_used)]
#![allow(clippy::disallowed_methods)] // example harness — unwrap acceptable

use showcase_calculator::driver::{
    run_full_specification, verify_basic_arithmetic, verify_complex_expressions,
    verify_error_handling, verify_history, verify_jidoka_status, verify_precedence,
    CalculatorDriver, TuiDriver,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║            TUI Calculator Demo - Driver Testing              ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Demonstrating unified testing: same specs, different driver ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut driver = TuiDriver::new();

    // Show app structure
    println!("📦 TUI App Structure:");
    println!("   ├── CalculatorApp (state management)");
    println!("   ├── Evaluator (expression evaluation)");
    println!("   ├── History (calculation history)");
    println!("   └── AnomalyValidator (anomaly detection)");
    println!();

    // Interactive demo
    println!("🧮 Interactive Calculation Demo:");
    println!("─────────────────────────────────");

    // Enter expression
    println!("\n1️⃣  Entering expression: '42 * (3 + 7)'");
    driver.enter_expression("42 * (3 + 7)").unwrap();
    println!("   Input: {}", driver.get_input());
    println!("   Result: {}", driver.get_result());

    // Show Anomaly status
    println!("\n2️⃣  Anomaly Status:");
    for status in driver.get_jidoka_status() {
        println!("   {}", status);
    }

    // More calculations
    println!("\n3️⃣  More calculations:");
    driver.enter_expression("2 ^ 10").unwrap();
    println!("   2 ^ 10 = {}", driver.get_result());

    driver.enter_expression("100 / 4").unwrap();
    println!("   100 / 4 = {}", driver.get_result());

    driver.enter_expression("17 % 5").unwrap();
    println!("   17 % 5 = {}", driver.get_result());

    // Show history
    println!("\n4️⃣  History (newest first):");
    for (i, item) in driver.get_history().iter().enumerate() {
        println!("   [{i}] {} = {}", item.expression, item.result);
    }

    // Clear
    println!("\n5️⃣  Clearing calculator");
    driver.clear();
    println!("   Input after clear: '{}'", driver.get_input());
    println!("   Result after clear: '{}'", driver.get_result());

    // Error handling demo
    println!("\n6️⃣  Error handling demo:");
    match driver.enter_expression("1 / 0") {
        Ok(()) => println!("   Unexpected success"),
        Err(e) => println!("   1 / 0 -> Error: {}", e),
    }
    driver.clear();

    // Now run unified specifications
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Running Unified Test Specifications (same as WASM driver!)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let mut driver = TuiDriver::new();

    print!("  ✓ verify_basic_arithmetic ... ");
    verify_basic_arithmetic(&mut driver);
    println!("PASSED");

    print!("  ✓ verify_precedence ... ");
    verify_precedence(&mut driver);
    println!("PASSED");

    print!("  ✓ verify_complex_expressions ... ");
    verify_complex_expressions(&mut driver);
    println!("PASSED");

    print!("  ✓ verify_error_handling ... ");
    verify_error_handling(&mut driver);
    println!("PASSED");

    print!("  ✓ verify_history ... ");
    verify_history(&mut driver);
    println!("PASSED");

    print!("  ✓ verify_jidoka_status ... ");
    verify_jidoka_status(&mut driver);
    println!("PASSED");

    println!();
    println!("  ✅ All unified specifications passed on TuiDriver!");
    println!();

    // Full specification in one call
    let mut driver = TuiDriver::new();
    print!("  Running run_full_specification() ... ");
    run_full_specification(&mut driver);
    println!("PASSED");

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  🎉 Demo Complete - TUI driver works with unified specs!     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
