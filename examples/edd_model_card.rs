//! Equation Model Card (EMC) Example
//!
//! This example demonstrates how to create and use Equation Model Cards,
//! the mandatory documentation that bridges mathematical theory and code.
//!
//! Run with: cargo run --example edd_model_card

use simular::edd::{Citation, EddValidator, EmcBuilder, EquationClass};

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     Equation Model Cards: EDD Documentation                   ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    demonstrate_emc_creation();
    demonstrate_emc_validation();
    demonstrate_emc_verification_tests();

    println!("\n✓ EMC demonstration completed!");
}

fn demonstrate_emc_creation() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Creating an Equation Model Card");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create a complete EMC using the builder pattern
    let emc = EmcBuilder::new()
        .name("Little's Law")
        .version("1.0.0")
        .equation("L = \\lambda W")
        .description(
            "The fundamental theorem of queueing theory. States that the \
                      average number of items in a stable system equals the arrival \
                      rate multiplied by the average time in system.",
        )
        .class(EquationClass::Queueing)
        .citation(
            Citation::new(&["Little, J.D.C."], "Operations Research", 1961)
                .with_title("A Proof for the Queuing Formula: L = λW"),
        )
        // Add variables
        .add_variable("L", "Average number of items in system (WIP)", "items")
        .add_variable("λ", "Average arrival rate (Throughput)", "items/time")
        .add_variable("W", "Average time in system (Cycle Time)", "time")
        // Add verification tests (TDD: tests first!)
        .add_verification_test("L = λW for λ=5, W=2 => L=10", 10.0, 1e-10)
        .add_verification_test("L = λW for λ=10, W=1.5 => L=15", 15.0, 1e-10)
        .add_verification_test("L = λW for λ=2.5, W=4 => L=10", 10.0, 1e-10)
        .build()
        .expect("EMC should build successfully");

    // Display the EMC
    println!("EMC Created:");
    println!("  Name: {}", emc.name);
    println!("  Version: {}", emc.version);
    println!("  Equation: {}", emc.equation);
    println!("  Class: {:?}", emc.class);
    println!("  Citation: {}", emc.citation);
    println!("  Variables: {}", emc.variables.len());
    println!("  Verification Tests: {}", emc.verification_tests.len());

    // Show variables
    println!("\nVariables:");
    for var in &emc.variables {
        println!("  {} - {} [{}]", var.symbol, var.description, var.units);
    }

    println!();
}

fn demonstrate_emc_validation() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("EMC Validation (EDD Compliance)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let validator = EddValidator::new();

    // Test 1: Missing EMC
    println!("Test 1: Simulation without EMC");
    let result = validator.validate_simulation_has_emc(None);
    match result {
        Err(violation) => {
            println!("  ✗ {} - {}", violation.code, violation.message);
            println!("  Severity: {:?}", violation.severity);
        }
        Ok(()) => println!("  ✓ Passed"),
    }

    // Test 2: EMC without equation (should fail to build)
    println!("\nTest 2: EMC without governing equation");
    let result = EmcBuilder::new().name("Incomplete EMC").build();
    match result {
        Err(msg) => println!("  ✗ Build failed: {msg}"),
        Ok(_) => println!("  ✓ Built (unexpected)"),
    }

    // Test 3: EMC without citation
    println!("\nTest 3: EMC without citation");
    let result = EmcBuilder::new()
        .name("No Citation EMC")
        .equation("x = y")
        .build();
    match result {
        Err(msg) => println!("  ✗ Build failed: {msg}"),
        Ok(_) => println!("  ✓ Built (unexpected)"),
    }

    // Test 4: EMC without verification tests
    println!("\nTest 4: EMC without verification tests");
    let result = EmcBuilder::new()
        .name("No Tests EMC")
        .equation("x = y")
        .citation(Citation::new(&["Author"], "Journal", 2024))
        .build();
    match result {
        Err(msg) => println!("  ✗ Build failed: {msg}"),
        Ok(_) => println!("  ✓ Built (unexpected)"),
    }

    // Test 5: Complete EMC
    println!("\nTest 5: Complete EMC");
    let result = EmcBuilder::new()
        .name("Complete EMC")
        .equation("y = mx + b")
        .citation(Citation::new(&["Author"], "Journal", 2024))
        .add_verification_test("slope test", 5.0, 0.01)
        .build();
    match result {
        Err(msg) => println!("  ✗ Build failed: {msg}"),
        Ok(emc) => {
            println!("  ✓ Built successfully");
            let validate_result = validator.validate_simulation_has_emc(Some(&emc));
            println!(
                "  ✓ Passes EDD-01 validation: {:?}",
                validate_result.is_ok()
            );
        }
    }

    println!();
}

fn demonstrate_emc_verification_tests() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Running EMC Verification Tests");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create EMC with verification tests
    let emc = EmcBuilder::new()
        .name("Little's Law")
        .equation("L = \\lambda W")
        .citation(Citation::new(
            &["Little, J.D.C."],
            "Operations Research",
            1961,
        ))
        .add_variable("L", "WIP", "items")
        .add_variable("lambda", "Throughput", "items/time")
        .add_variable("W", "Cycle Time", "time")
        .add_verification_test("test_10_5_2", 10.0, 1e-6)
        .add_verification_test("test_15_10_1.5", 15.0, 1e-6)
        .add_verification_test("test_20_4_5", 20.0, 1e-6)
        .build()
        .unwrap();

    // Define an evaluator function that implements Little's Law
    let evaluator = |params: &std::collections::HashMap<String, f64>| -> f64 {
        let lambda = params.get("lambda").copied().unwrap_or(0.0);
        let w = params.get("W").copied().unwrap_or(0.0);
        lambda * w
    };

    // Run verification tests
    println!(
        "Running {} verification tests...\n",
        emc.verification_tests.len()
    );
    let results = emc.run_verification_tests(evaluator);

    println!("┌───────────────────┬──────────┬──────────┬──────────┬────────┐");
    println!("│       Test        │ Expected │  Actual  │   Tol    │ Status │");
    println!("├───────────────────┼──────────┼──────────┼──────────┼────────┤");

    for (name, passed, msg) in &results {
        // Parse expected from message (simplified)
        let status = if *passed { "✓ PASS" } else { "✗ FAIL" };
        println!(
            "│ {:^17} │   {:>5}  │   {:>5}  │  {:>5}   │ {:^6} │",
            &name[..name.len().min(17)],
            "-",
            "-",
            "-",
            status
        );
        if !passed {
            println!("│   {msg:^63} │");
        }
    }
    println!("└───────────────────┴──────────┴──────────┴──────────┴────────┘");

    let passed_count = results.iter().filter(|(_, p, _)| *p).count();
    let total = results.len();
    println!("\nResults: {passed_count}/{total} tests passed");

    if passed_count == total {
        println!("✓ All verification tests passed - EMC compliance confirmed!");
    } else {
        println!("✗ Some tests failed - review implementation");
    }
}
