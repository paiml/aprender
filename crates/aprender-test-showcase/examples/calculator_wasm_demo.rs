//! WASM Calculator Demo
//!
//! This example demonstrates the WasmDriver running the same
//! unified test specifications as the TUI driver.
//!
//! Run with: cargo run --example calculator_wasm_demo

use showcase_calculator::driver::{
    run_full_specification, verify_basic_arithmetic, verify_complex_expressions,
    verify_error_handling, verify_history, verify_jidoka_status, verify_precedence,
    CalculatorDriver,
};
use showcase_calculator::wasm::WasmDriver;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           WASM Calculator Demo - Mock DOM Testing            ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Demonstrating unified testing: same specs, different driver ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut driver = WasmDriver::new();

    // Show DOM structure
    println!("📦 Mock DOM Structure:");
    println!("   ├── calc-input    (input field)");
    println!("   ├── calc-result   (result display)");
    println!("   ├── calc-history  (history list)");
    println!("   ├── calc-status   (Anomaly status)");
    println!("   ├── btn-clear     (clear button)");
    println!("   └── btn-equals    (equals button)");
    println!();

    // Interactive demo
    println!("🧮 Interactive Calculation Demo:");
    println!("─────────────────────────────────");

    // Simulate typing
    println!("\n1️⃣  Simulating: type '42 * (3 + 7)' into input");
    driver.type_input("42 * (3 + 7)");
    println!("   DOM Input: {:?}", driver.input_element_text());

    // Simulate clicking equals
    println!("\n2️⃣  Simulating: click '=' button");
    driver.click_equals();
    println!("   DOM Result: {:?}", driver.result_element_text());
    println!("   Driver Result: {}", driver.get_result());

    // Show DOM event history
    println!("\n3️⃣  DOM Event History:");
    for (i, event) in driver.dom().event_history().iter().enumerate() {
        println!("   [{i}] {:?}", event);
    }

    // Show Anomaly status
    println!("\n4️⃣  Anomaly Status (DOM sync):");
    if let Some(status) = driver.status_element_text() {
        for line in status.lines() {
            println!("   {}", line);
        }
    }

    // More calculations to build history
    println!("\n5️⃣  Building history with more calculations:");
    driver.type_input("2 ^ 10");
    driver.press_enter();
    println!("   2 ^ 10 = {}", driver.get_result());

    driver.type_input("100 / 4");
    driver.click_equals();
    println!("   100 / 4 = {}", driver.get_result());

    // Show history in DOM
    println!("\n6️⃣  History List (from DOM):");
    for (i, item) in driver.history_list_items().iter().enumerate() {
        println!("   [{i}] {item}");
    }

    // Clear demo
    println!("\n7️⃣  Simulating: click 'C' (clear) button");
    driver.click_clear();
    println!("   Input after clear: {:?}", driver.input_element_text());
    println!("   Result after clear: {:?}", driver.result_element_text());

    // Now run unified specifications
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Running Unified Test Specifications (same as TUI driver!)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let mut driver = WasmDriver::new();

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
    println!("  ✅ All unified specifications passed on WasmDriver!");
    println!();

    // Full specification in one call
    let mut driver = WasmDriver::new();
    print!("  Running run_full_specification() ... ");
    run_full_specification(&mut driver);
    println!("PASSED");

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  🎉 Demo Complete - WASM driver works with unified specs!    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
