#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Example: GUI Coverage with Pixel-Perfect Verification (PIXEL-001 v2.1)
//!
//! Demonstrates: Complete GUI coverage tracking with pixel-level verification,
//! Popperian falsification, and statistical rigor.
//!
//! Run with: `cargo run --example gui_coverage -p jugar-probar`
//!
//! Probar: Complete UX verification with minimal boilerplate

use jugar_probar::pixel_coverage::{
    CombinedCoverageReport, ConfidenceInterval, FalsifiabilityGate, FalsifiableHypothesis,
    LineCoverageReport, OutputMode, PixelCoverageTracker, PixelRegion, ScoreBar,
};
use jugar_probar::prelude::*;

fn main() -> ProbarResult<()> {
    println!("=== GUI Coverage with Pixel-Perfect Verification (v2.1) ===\n");

    // =========================================================================
    // 1. Simple GUI Coverage with Pixel Tracking
    // =========================================================================
    println!("1. GUI Coverage with Pixel Tracking...");

    let mut gui = jugar_probar::gui_coverage! {
        buttons: ["start", "pause", "quit"],
        screens: ["title", "playing", "game_over"]
    };

    // Pixel tracker for 400x300 game UI (4x3 grid)
    let mut pixels = PixelCoverageTracker::builder()
        .resolution(400, 300)
        .grid_size(4, 3)
        .threshold(0.80)
        .build();

    // Simulate test interactions with pixel regions
    gui.click("start");
    pixels.record_region(PixelRegion::new(150, 200, 100, 40)); // Start button

    gui.visit("title");
    pixels.record_region(PixelRegion::new(0, 0, 400, 100)); // Title screen header

    gui.visit("playing");
    pixels.record_region(PixelRegion::new(0, 100, 400, 200)); // Game area

    println!("   GUI: {}", gui.summary());
    println!(
        "   Pixels: {:.1}% ({}/{} cells)",
        pixels.generate_report().overall_coverage * 100.0,
        pixels.generate_report().covered_cells,
        pixels.generate_report().total_cells
    );

    // =========================================================================
    // 2. Calculator with Full Pixel Coverage
    // =========================================================================
    println!("\n2. Calculator with Pixel-Perfect Coverage...");

    let mut calc = calculator_coverage();
    let mut calc_pixels = PixelCoverageTracker::builder()
        .resolution(320, 480)
        .grid_size(4, 6)
        .threshold(1.0)
        .build();

    // Button regions (80x80 each in 4x6 grid)
    let btn = |col: u32, row: u32| PixelRegion::new(col * 80, row * 80, 80, 80);

    // Cover display
    calc_pixels.record_region(PixelRegion::new(0, 0, 320, 80));

    // Click all digits with pixel tracking
    for i in 0..=9 {
        calc.click(&format!("btn-{}", i));
        let (col, row) = match i {
            7 => (0, 1),
            8 => (1, 1),
            9 => (2, 1),
            4 => (0, 2),
            5 => (1, 2),
            6 => (2, 2),
            1 => (0, 3),
            2 => (1, 3),
            3 => (2, 3),
            0 => (0, 4),
            _ => (0, 0),
        };
        calc_pixels.record_region(btn(col, row));
    }

    // Operators
    calc.click("btn-plus");
    calc_pixels.record_region(btn(3, 4));
    calc.click("btn-minus");
    calc_pixels.record_region(btn(3, 3));
    calc.click("btn-times");
    calc_pixels.record_region(btn(3, 2));
    calc.click("btn-divide");
    calc_pixels.record_region(btn(3, 1));
    calc.click("btn-equals");
    calc_pixels.record_region(btn(2, 4));
    calc.click("btn-clear");
    calc_pixels.record_region(btn(3, 5));
    calc.click("btn-decimal");
    calc_pixels.record_region(btn(1, 4));
    calc.click("btn-power");
    calc_pixels.record_region(btn(2, 5));
    calc.click("btn-open-paren");
    calc_pixels.record_region(btn(0, 5));
    calc.click("btn-close-paren");
    calc_pixels.record_region(btn(1, 5));

    calc.visit("calculator");

    let calc_pixel_report = calc_pixels.generate_report();
    println!("   GUI: {}", calc.summary());
    println!(
        "   Pixels: {:.1}% coverage",
        calc_pixel_report.overall_coverage * 100.0
    );

    // =========================================================================
    // 3. Game Coverage with Popperian Falsification
    // =========================================================================
    println!("\n3. Game Coverage with Falsification Gate...");

    let mut game = game_coverage(
        &["play", "pause", "restart", "menu", "settings"],
        &["splash", "main_menu", "gameplay", "pause_menu", "game_over"],
    );

    let mut game_pixels = PixelCoverageTracker::builder()
        .resolution(800, 600)
        .grid_size(8, 6)
        .threshold(0.85)
        .build();

    // Simulate a game session with pixel tracking
    game.visit("splash");
    game_pixels.record_region(PixelRegion::new(0, 0, 800, 600)); // Full splash

    game.click("play");
    game_pixels.record_region(PixelRegion::new(300, 400, 200, 60)); // Play button

    game.visit("main_menu");
    game_pixels.record_region(PixelRegion::new(200, 100, 400, 400)); // Menu area

    game.visit("gameplay");
    game_pixels.record_region(PixelRegion::new(0, 0, 800, 500)); // Game area

    game.click("pause");
    game_pixels.record_region(PixelRegion::new(700, 10, 80, 40)); // Pause button

    game.visit("pause_menu");
    game_pixels.record_region(PixelRegion::new(250, 150, 300, 300)); // Pause menu

    game.click("restart");
    game_pixels.record_region(PixelRegion::new(300, 350, 200, 50)); // Restart button

    game.visit("game_over");
    game_pixels.record_region(PixelRegion::new(0, 0, 800, 600)); // Game over screen

    game.click("menu");
    game_pixels.record_region(PixelRegion::new(300, 450, 200, 50)); // Menu button

    // Popperian Falsification
    let gate = FalsifiabilityGate::new(15.0);
    let game_report = game.generate_report();
    let pixel_report = game_pixels.generate_report();

    let h1 = FalsifiableHypothesis::coverage_threshold("H0-GAME-GUI", 0.80)
        .evaluate(game_report.overall_coverage as f32);
    let h2 = FalsifiableHypothesis::coverage_threshold("H0-GAME-PIX", 0.85)
        .evaluate(pixel_report.overall_coverage);

    println!("   GUI: {}", game.summary());
    println!(
        "   Pixels: {:.1}% coverage",
        pixel_report.overall_coverage * 100.0
    );
    println!(
        "   H0-GAME-GUI: {} (actual: {:.1}%)",
        if h1.falsified {
            "FALSIFIED"
        } else {
            "NOT FALSIFIED"
        },
        h1.actual.unwrap_or(0.0) * 100.0
    );
    println!(
        "   H0-GAME-PIX: {} (actual: {:.1}%)",
        if h2.falsified {
            "FALSIFIED"
        } else {
            "NOT FALSIFIED"
        },
        h2.actual.unwrap_or(0.0) * 100.0
    );
    println!(
        "   Gate: {}",
        if gate.evaluate(&h1).is_passed() {
            "PASSED"
        } else {
            "FAILED"
        }
    );

    // =========================================================================
    // 4. Custom Coverage with Wilson Confidence Intervals
    // =========================================================================
    println!("\n4. Custom Coverage with Statistical Rigor...");

    let mut custom = UxCoverageBuilder::new()
        .button("login")
        .button("logout")
        .input("username")
        .input("password")
        .screen("login_page")
        .screen("dashboard")
        .modal("confirm_logout")
        .build();

    let mut custom_pixels = PixelCoverageTracker::builder()
        .resolution(1024, 768)
        .grid_size(8, 6)
        .threshold(0.90)
        .build();

    // Simulate login flow
    custom.visit("login_page");
    custom_pixels.record_region(PixelRegion::new(0, 0, 1024, 768));

    custom.input("username");
    custom_pixels.record_region(PixelRegion::new(362, 300, 300, 40));

    custom.input("password");
    custom_pixels.record_region(PixelRegion::new(362, 360, 300, 40));

    custom.click("login");
    custom_pixels.record_region(PixelRegion::new(412, 430, 200, 50));

    custom.visit("dashboard");
    custom_pixels.record_region(PixelRegion::new(0, 0, 1024, 768));

    let custom_report = custom.generate_report();
    let custom_pixel_report = custom_pixels.generate_report();

    // Wilson score confidence intervals
    let gui_ci = ConfidenceInterval::wilson_score(
        custom_report.covered_elements as u32,
        custom_report.total_elements as u32,
        0.95,
    );
    let pixel_ci = ConfidenceInterval::wilson_score(
        custom_pixel_report.covered_cells,
        custom_pixel_report.total_cells,
        0.95,
    );

    println!("   GUI: {}", custom.summary());
    println!(
        "   GUI 95% CI: [{:.1}%, {:.1}%]",
        gui_ci.lower * 100.0,
        gui_ci.upper * 100.0
    );
    println!(
        "   Pixels: {:.1}%",
        custom_pixel_report.overall_coverage * 100.0
    );
    println!(
        "   Pixel 95% CI: [{:.1}%, {:.1}%]",
        pixel_ci.lower * 100.0,
        pixel_ci.upper * 100.0
    );

    // =========================================================================
    // 5. Full Coverage with Score Bars
    // =========================================================================
    println!("\n5. Full Coverage with Visual Score Bars...");

    let mut full = jugar_probar::gui_coverage! {
        buttons: ["a", "b", "c"],
        screens: ["home"]
    };

    let mut full_pixels = PixelCoverageTracker::builder()
        .resolution(300, 200)
        .grid_size(3, 2)
        .threshold(1.0)
        .build();

    full.click("a");
    full_pixels.record_region(PixelRegion::new(0, 100, 100, 100));
    full.click("b");
    full_pixels.record_region(PixelRegion::new(100, 100, 100, 100));
    full.click("c");
    full_pixels.record_region(PixelRegion::new(200, 100, 100, 100));
    full.visit("home");
    full_pixels.record_region(PixelRegion::new(0, 0, 300, 100));

    let full_report = full.generate_report();
    let full_pixel_report = full_pixels.generate_report();
    let mode = OutputMode::from_env();

    let gui_bar = ScoreBar::new("GUI", full_report.overall_coverage as f32, 1.0);
    let pixel_bar = ScoreBar::new("Pixels", full_pixel_report.overall_coverage, 1.0);

    println!("   {}", gui_bar.render(mode));
    println!("   {}", pixel_bar.render(mode));
    println!(
        "   Complete: {}",
        full.is_complete() && full_pixel_report.meets_threshold
    );

    full.assert_complete()?;

    // =========================================================================
    // 6. Combined Coverage Report
    // =========================================================================
    println!("\n6. Combined Coverage Report...");

    let mut detailed = jugar_probar::gui_coverage! {
        buttons: ["save", "cancel", "delete"],
        inputs: ["name", "email"],
        screens: ["form", "confirmation"],
        modals: ["delete_confirm"]
    };

    let mut detailed_pixels = PixelCoverageTracker::builder()
        .resolution(600, 400)
        .grid_size(6, 4)
        .threshold(0.85)
        .build();

    detailed.click("save");
    detailed_pixels.record_region(PixelRegion::new(400, 350, 100, 40));

    detailed.input("name");
    detailed_pixels.record_region(PixelRegion::new(200, 100, 200, 30));

    detailed.visit("form");
    detailed_pixels.record_region(PixelRegion::new(100, 50, 400, 300));

    let detailed_report = detailed.generate_report();
    let detailed_pixel_report = detailed_pixels.generate_report();

    // Combined report
    let line_report = LineCoverageReport::new(
        detailed_report.element_coverage as f32,
        detailed_report.state_coverage as f32,
        1.0,
        detailed_report.total_elements,
        detailed_report.covered_elements,
    );
    let combined = CombinedCoverageReport::from_parts(line_report, detailed_pixel_report);

    println!("{}", combined.summary());

    // =========================================================================
    // 7. User Journeys with Pixel Heatmap
    // =========================================================================
    println!("\n7. User Journey with Pixel Heatmap...");

    let mut journey = UxCoverageTracker::new();
    journey.register_screen("home");
    journey.register_screen("products");
    journey.register_screen("cart");
    journey.register_screen("checkout");

    let mut journey_pixels = PixelCoverageTracker::builder()
        .resolution(1200, 800)
        .grid_size(12, 8)
        .threshold(0.75)
        .build();

    // Journey 1: Browse and buy
    journey.visit("home");
    journey_pixels.record_region(PixelRegion::new(0, 0, 1200, 800));

    journey.visit("products");
    journey_pixels.record_region(PixelRegion::new(0, 100, 1200, 600));

    journey.visit("cart");
    journey_pixels.record_region(PixelRegion::new(800, 0, 400, 800));

    journey.visit("checkout");
    journey_pixels.record_region(PixelRegion::new(200, 100, 800, 600));

    journey.end_journey();

    // Journey 2: Just browse
    journey.visit("home");
    journey.visit("products");
    journey.end_journey();

    let _journey_pixel_report = journey_pixels.generate_report();

    println!("   Journeys recorded: {}", journey.journeys().len());
    println!("   {}", journey.summary());
    println!("   Pixel Heatmap:");
    let heatmap = journey_pixels.terminal_heatmap();
    for line in heatmap.render().lines() {
        println!("     {}", line);
    }

    // =========================================================================
    // Final Summary
    // =========================================================================
    println!("\n=== PIXEL-001 v2.1 Coverage Summary ===");
    println!("   All examples use pixel-perfect GUI testing");
    println!("   Features demonstrated:");
    println!("   - PixelCoverageTracker with grid-based tracking");
    println!("   - FalsifiabilityGate with Popperian methodology");
    println!("   - Wilson score confidence intervals");
    println!("   - ScoreBar visual indicators");
    println!("   - CombinedCoverageReport (GUI + Pixel)");
    println!("   - Terminal heatmap visualization");

    println!("\n[OK] GUI coverage with pixel-perfect verification completed!");
    Ok(())
}
