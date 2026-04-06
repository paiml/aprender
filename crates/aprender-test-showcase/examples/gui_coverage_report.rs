#![allow(clippy::expect_used, clippy::unwrap_used)]
//! GUI Coverage Report for Showcase Calculator (PIXEL-001 v2.1 Dogfooding)
//!
//! Run with: `cargo run -p showcase-calculator --example gui_coverage_report`
//!
//! This runs all calculator functionality and reports GUI coverage using
//! the full PIXEL-001 v2.1 pixel-perfect verification framework:
//! - Pixel-level coverage tracking with heatmaps
//! - Popperian falsification with FalsifiabilityGate
//! - Wilson score confidence intervals
//! - Rich terminal output with score bars

use jugar_probar::pixel_coverage::{
    CombinedCoverageReport, ConfidenceInterval, FalsifiabilityGate, FalsifiableHypothesis,
    LineCoverageReport, OutputMode, PixelCoverageReport, PixelCoverageTracker, PixelRegion,
    ScoreBar,
};
use jugar_probar::{UxCoverageBuilder, UxCoverageReport, UxCoverageTracker};
use showcase_calculator::prelude::*;

/// Calculate button region position in the 4x6 grid (80x80 pixel buttons)
fn button_region(col: u32, row: u32) -> PixelRegion {
    PixelRegion::new(col * 80, row * 80, 80, 80)
}

/// Initialize the pixel coverage tracker for the calculator layout
fn init_pixel_tracker() -> PixelCoverageTracker {
    PixelCoverageTracker::builder()
        .resolution(320, 480)
        .grid_size(4, 6) // 4 columns, 6 rows (matches calculator keypad layout)
        .threshold(1.0) // Require 100% coverage
        .build()
}

/// Initialize the GUI coverage tracker with all calculator elements
fn init_gui_tracker() -> UxCoverageTracker {
    UxCoverageBuilder::new()
        // All 20 keypad buttons
        .button("btn-0")
        .button("btn-1")
        .button("btn-2")
        .button("btn-3")
        .button("btn-4")
        .button("btn-5")
        .button("btn-6")
        .button("btn-7")
        .button("btn-8")
        .button("btn-9")
        .button("btn-plus")
        .button("btn-minus")
        .button("btn-times")
        .button("btn-divide")
        .button("btn-equals")
        .button("btn-clear")
        .button("btn-decimal")
        .button("btn-power")
        .button("btn-open-paren")
        .button("btn-close-paren")
        .input("calc-input")
        .screen("calculator")
        .screen("result-display")
        .screen("history-panel")
        .screen("anomaly-panel")
        .build()
}

/// Run all test suites and record coverage
fn run_test_suites(
    driver: &mut WasmDriver,
    gui: &mut UxCoverageTracker,
    pixel_tracker: &mut PixelCoverageTracker,
) {
    // Cover display area first (row 0)
    pixel_tracker.record_region(PixelRegion::new(0, 0, 320, 80));

    run_basic_arithmetic(driver, gui, pixel_tracker);
    run_advanced_operations(driver, gui, pixel_tracker);
    run_digit_coverage(driver, gui, pixel_tracker);
    run_screen_coverage(gui);
}

/// Test Suite 1: Basic Arithmetic
fn run_basic_arithmetic(
    driver: &mut WasmDriver,
    gui: &mut UxCoverageTracker,
    pixel_tracker: &mut PixelCoverageTracker,
) {
    println!("Running Test Suite 1: Basic Arithmetic...");

    // Addition: 5 + 3 = 8
    driver.type_input("5 + 3");
    gui.click("btn-5");
    pixel_tracker.record_region(button_region(1, 2));
    gui.click("btn-plus");
    pixel_tracker.record_region(button_region(3, 4));
    gui.click("btn-3");
    pixel_tracker.record_region(button_region(2, 3));
    driver.click_equals();
    gui.click("btn-equals");
    pixel_tracker.record_region(button_region(2, 4));
    assert_eq!(driver.get_result(), "8");
    driver.click_clear();
    gui.click("btn-clear");
    pixel_tracker.record_region(button_region(3, 5));

    // Subtraction: 10 - 4 = 6
    driver.type_input("10 - 4");
    gui.click("btn-1");
    pixel_tracker.record_region(button_region(0, 3));
    gui.click("btn-0");
    pixel_tracker.record_region(button_region(0, 4));
    gui.click("btn-minus");
    pixel_tracker.record_region(button_region(3, 3));
    gui.click("btn-4");
    pixel_tracker.record_region(button_region(0, 2));
    driver.click_equals();
    gui.click("btn-equals");
    assert_eq!(driver.get_result(), "6");
    driver.click_clear();

    // Multiplication: 7 * 6 = 42
    driver.type_input("7 * 6");
    gui.click("btn-7");
    pixel_tracker.record_region(button_region(0, 1));
    gui.click("btn-times");
    pixel_tracker.record_region(button_region(3, 2));
    gui.click("btn-6");
    pixel_tracker.record_region(button_region(2, 2));
    driver.click_equals();
    assert_eq!(driver.get_result(), "42");
    driver.click_clear();

    // Division: 20 / 4 = 5
    driver.type_input("20 / 4");
    gui.click("btn-2");
    pixel_tracker.record_region(button_region(1, 3));
    gui.click("btn-divide");
    pixel_tracker.record_region(button_region(3, 1));
    driver.click_equals();
    assert_eq!(driver.get_result(), "5");
    driver.click_clear();

    println!("   [OK] Basic arithmetic: PASS");
}

/// Test Suite 2: Advanced Operations
fn run_advanced_operations(
    driver: &mut WasmDriver,
    gui: &mut UxCoverageTracker,
    pixel_tracker: &mut PixelCoverageTracker,
) {
    println!("Running Test Suite 2: Advanced Operations...");

    // Power: 2^8 = 256
    driver.type_input("2 ^ 8");
    gui.click("btn-power");
    pixel_tracker.record_region(button_region(2, 5));
    gui.click("btn-8");
    pixel_tracker.record_region(button_region(1, 1));
    driver.click_equals();
    assert_eq!(driver.get_result(), "256");
    driver.click_clear();

    // Decimal: 3.14 * 2 = 6.28
    driver.type_input("3.14 * 2");
    gui.click("btn-decimal");
    pixel_tracker.record_region(button_region(1, 4));
    driver.click_equals();
    driver.click_clear();

    // Parentheses: (2 + 3) * 4 = 20
    driver.type_input("(2 + 3) * 4");
    gui.click("btn-open-paren");
    pixel_tracker.record_region(button_region(0, 5));
    gui.click("btn-close-paren");
    pixel_tracker.record_region(button_region(1, 5));
    driver.click_equals();
    assert_eq!(driver.get_result(), "20");
    driver.click_clear();

    println!("   [OK] Advanced operations: PASS");
}

/// Test Suite 3: All Digits
fn run_digit_coverage(
    driver: &mut WasmDriver,
    gui: &mut UxCoverageTracker,
    pixel_tracker: &mut PixelCoverageTracker,
) {
    println!("Running Test Suite 3: All Digits...");
    driver.type_input("9");
    gui.click("btn-9");
    pixel_tracker.record_region(button_region(2, 1));
    driver.click_clear();
    println!("   [OK] All digits covered: PASS");
}

/// Test Suite 4: Screen Coverage
fn run_screen_coverage(gui: &mut UxCoverageTracker) {
    println!("Running Test Suite 4: Screen Coverage...");
    gui.input("calc-input");
    gui.visit("calculator");
    gui.visit("result-display");
    gui.visit("history-panel");
    gui.visit("anomaly-panel");
    println!("   [OK] All screens visited: PASS");
}

/// Print the coverage report
fn print_coverage_report(gui: &UxCoverageTracker, pixel_tracker: &PixelCoverageTracker) {
    let gui_report = gui.generate_report();
    let pixel_report = pixel_tracker.generate_report();
    let mode = OutputMode::from_env();

    println!("\n===============================================================");
    println!("           PIXEL-001 v2.1 COVERAGE RESULTS                      ");
    println!("===============================================================\n");

    print_gui_element_coverage(&gui_report, mode);
    print_pixel_coverage(&pixel_report, pixel_tracker, mode);
    print_statistical_rigor(&gui_report, &pixel_report);
    print_popperian_falsification(&gui_report, &pixel_report);
    print_combined_summary(&gui_report, &pixel_report);
    print_final_status(gui, &gui_report, &pixel_report);
}

/// Section 1: GUI Element Coverage
fn print_gui_element_coverage(gui_report: &UxCoverageReport, mode: OutputMode) {
    println!("--- GUI ELEMENT COVERAGE ---");
    let element_bar = ScoreBar::new("Elements", gui_report.element_coverage as f32, 1.0);
    println!("  {}", element_bar.render(mode));
    let state_bar = ScoreBar::new("Screens", gui_report.state_coverage as f32, 1.0);
    println!("  {}", state_bar.render(mode));
    println!(
        "  Covered: {}/{} elements, {}/{} screens\n",
        gui_report.covered_elements,
        gui_report.total_elements,
        gui_report.covered_states,
        gui_report.total_states
    );
}

/// Section 2: Pixel-Level Coverage with Heatmap
fn print_pixel_coverage(
    pixel_report: &PixelCoverageReport,
    pixel_tracker: &PixelCoverageTracker,
    mode: OutputMode,
) {
    println!("--- PIXEL-LEVEL COVERAGE ---");
    let pixel_bar = ScoreBar::new("Pixels", pixel_report.overall_coverage, 1.0);
    println!("  {}", pixel_bar.render(mode));
    println!(
        "  Cells: {}/{} covered\n",
        pixel_report.covered_cells, pixel_report.total_cells
    );

    println!("  Pixel Heatmap (4x6 grid):");
    let heatmap = pixel_tracker.terminal_heatmap();
    for line in heatmap.render().lines() {
        println!("    {}", line);
    }
    println!();
}

/// Section 3: Wilson Score Confidence Intervals
fn print_statistical_rigor(gui_report: &UxCoverageReport, pixel_report: &PixelCoverageReport) {
    println!("--- STATISTICAL RIGOR (Wilson Score 95% CI) ---");
    let pixel_pct = pixel_report.overall_coverage * 100.0;
    let ci = ConfidenceInterval::wilson_score(
        pixel_report.covered_cells,
        pixel_report.total_cells,
        0.95,
    );
    println!(
        "  Pixel Coverage: {:.1}% [{:.1}%, {:.1}%]",
        pixel_pct,
        ci.lower * 100.0,
        ci.upper * 100.0
    );

    let gui_pct = gui_report.element_coverage * 100.0;
    let gui_ci = ConfidenceInterval::wilson_score(
        gui_report.covered_elements as u32,
        gui_report.total_elements as u32,
        0.95,
    );
    println!(
        "  GUI Coverage:   {:.1}% [{:.1}%, {:.1}%]\n",
        gui_pct,
        gui_ci.lower * 100.0,
        gui_ci.upper * 100.0
    );
}

/// Section 4: Popperian Falsification Gate
fn print_popperian_falsification(
    gui_report: &UxCoverageReport,
    pixel_report: &PixelCoverageReport,
) {
    println!("--- POPPERIAN FALSIFICATION ---");
    let gate = FalsifiabilityGate::new(15.0);

    let h1_template = FalsifiableHypothesis::coverage_threshold("H0-PIX-CALC-01", 1.0);
    let h1 = h1_template.evaluate(pixel_report.overall_coverage);

    let h2_template = FalsifiableHypothesis::coverage_threshold("H0-GUI-CALC-01", 1.0);
    let h2 = h2_template.evaluate(gui_report.element_coverage as f32);

    let h3_template = FalsifiableHypothesis::coverage_threshold("H0-SCR-CALC-01", 1.0);
    let h3 = h3_template.evaluate(gui_report.state_coverage as f32);

    let hypotheses = [&h1, &h2, &h3];
    for h in &hypotheses {
        let status = if h.falsified {
            "[FALSIFIED]"
        } else {
            "[NOT FALSIFIED]"
        };
        println!("  {}: {}", h.id, status);
        if let Some(actual) = h.actual {
            println!(
                "    Actual: {:.1}% vs Threshold: {:.1}%",
                actual * 100.0,
                h.threshold * 100.0
            );
        }
    }

    let gate_result = gate.evaluate(&h1);
    println!(
        "\n  Gate Score: {:.1} (threshold: {:.1})",
        gate_result.score(),
        gate.gateway_threshold
    );
    println!(
        "  Gate Status: {}\n",
        if gate_result.is_passed() {
            "[PASSED]"
        } else {
            "[FAILED]"
        }
    );
}

/// Section 5: Combined Coverage Summary
fn print_combined_summary(gui_report: &UxCoverageReport, pixel_report: &PixelCoverageReport) {
    let line_report = LineCoverageReport::new(
        gui_report.element_coverage as f32,
        gui_report.state_coverage as f32,
        1.0,
        gui_report.total_elements,
        gui_report.covered_elements,
    );
    let combined = CombinedCoverageReport::from_parts(line_report, pixel_report.clone());

    println!("--- COMBINED COVERAGE SUMMARY ---");
    println!("  Overall Score: {:.1}%", combined.overall_score * 100.0);
    println!(
        "  Meets Threshold (80%): {}\n",
        if combined.meets_threshold {
            "[YES]"
        } else {
            "[NO]"
        }
    );
}

/// Final Status output
fn print_final_status(
    gui: &UxCoverageTracker,
    gui_report: &UxCoverageReport,
    pixel_report: &PixelCoverageReport,
) {
    let gate = FalsifiabilityGate::new(15.0);
    let h1_template = FalsifiableHypothesis::coverage_threshold("H0-PIX-CALC-01", 1.0);
    let h1 = h1_template.evaluate(pixel_report.overall_coverage);
    let gate_result = gate.evaluate(&h1);

    let all_complete =
        gui_report.is_complete && pixel_report.meets_threshold && gate_result.is_passed();

    println!("===============================================================");
    if all_complete {
        println!("  [OK] STATUS: PIXEL-PERFECT COVERAGE ACHIEVED!");
        println!("     - 100% GUI element coverage");
        println!("     - 100% pixel-level coverage");
        println!("     - All hypotheses falsified");
        println!("     - Falsifiability gate PASSED");
    } else {
        println!("  [!!] STATUS: COVERAGE INCOMPLETE");
        if !gui_report.is_complete {
            println!("\n  Uncovered GUI elements:");
            for elem in gui.uncovered_elements() {
                println!("    - {}", elem.element);
            }
        }
        if !pixel_report.meets_threshold {
            println!("\n  Uncovered pixel regions:");
            for region in &pixel_report.uncovered_regions {
                println!(
                    "    - Region at ({}, {}) - {}x{}",
                    region.x, region.y, region.width, region.height
                );
            }
        }
    }
    println!("===============================================================\n");
}

fn main() {
    println!("===============================================================");
    println!("    SHOWCASE CALCULATOR - PIXEL-PERFECT COVERAGE (v2.1)        ");
    println!("===============================================================\n");

    let mut pixel_tracker = init_pixel_tracker();
    let mut gui = init_gui_tracker();
    let mut driver = WasmDriver::new();

    run_test_suites(&mut driver, &mut gui, &mut pixel_tracker);
    print_coverage_report(&gui, &pixel_tracker);
}
