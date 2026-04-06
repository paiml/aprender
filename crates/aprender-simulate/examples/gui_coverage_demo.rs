//! GUI Coverage Tracking Demo
//!
//! This example demonstrates how to use the `GuiCoverage` module
//! to track UI element and screen coverage during E2E testing.
//!
//! # Toyota Production System Alignment
//!
//! - **Visual Control**: Coverage reports show exactly what's tested
//! - **Jidoka**: Fail tests when coverage drops below threshold
//! - **Poka-Yoke**: Preset configurations prevent missing tests
//!
//! # Run
//!
//! ```bash
//! cargo run --example gui_coverage_demo
//! ```

use simular::edd::gui_coverage::{GuiCoverage, InteractionKind};

fn main() {
    println!("=== GUI Coverage Tracking Demo ===\n");

    // Example 1: Basic coverage tracking
    demo_basic_coverage();

    // Example 2: TSP TUI preset
    demo_tsp_tui_coverage();

    // Example 3: TSP WASM preset
    demo_tsp_wasm_coverage();

    // Example 4: Custom coverage setup
    demo_custom_coverage();
}

fn demo_basic_coverage() {
    println!("--- Basic Coverage Tracking ---\n");

    let mut coverage = GuiCoverage::new("My App");

    // Register elements to track
    coverage.register_element("login_button");
    coverage.register_element("username_field");
    coverage.register_element("password_field");
    coverage.register_element("submit_button");

    // Register screens
    coverage.register_screen("login_screen");
    coverage.register_screen("dashboard");
    coverage.register_screen("settings");

    // Register user journeys
    coverage.register_journey(
        "login_flow",
        vec!["login_screen", "login_button", "dashboard"],
    );

    // Simulate test coverage
    coverage.cover_element("login_button");
    coverage.cover_element("username_field");
    coverage.cover_screen("login_screen");

    // Log interactions
    coverage.log_interaction(InteractionKind::Click, "login_button", None, 0);
    coverage.log_interaction(
        InteractionKind::Input,
        "username_field",
        Some("testuser"),
        1,
    );

    // Check coverage
    println!(
        "Element coverage: {:.1}%",
        coverage.element_coverage() * 100.0
    );
    println!(
        "Screen coverage:  {:.1}%",
        coverage.screen_coverage() * 100.0
    );
    println!(
        "Journey coverage: {:.1}%",
        coverage.journey_coverage() * 100.0
    );
    println!("Summary: {}\n", coverage.summary());
}

fn demo_tsp_tui_coverage() {
    println!("--- TSP TUI Coverage (Preset) ---\n");

    let mut coverage = GuiCoverage::tsp_tui();

    // Simulate a user testing the TUI
    // User views main screen
    coverage.cover_screen("main_view");
    coverage.cover_element("city_plot");
    coverage.cover_element("convergence_graph");
    coverage.cover_element("statistics_panel");
    coverage.cover_element("controls_panel");
    coverage.cover_element("equations_panel");

    // User presses Space to start
    coverage.cover_element("space_toggle");
    coverage.log_interaction(InteractionKind::KeyPress, "space_toggle", Some("Space"), 0);

    // User observes running state
    coverage.cover_screen("running_state");
    coverage.cover_element("tour_length_display");
    coverage.cover_element("gap_display");
    coverage.cover_element("restarts_display");

    // User presses G for single step
    coverage.cover_element("g_step");
    coverage.log_interaction(InteractionKind::KeyPress, "g_step", Some("G"), 1);

    // User completes basic_run journey
    coverage.complete_journey("basic_run");

    // Print report
    println!("{}", coverage.summary());
    println!("Interactions logged: {}", coverage.interaction_count());

    // Check threshold
    if coverage.meets_threshold(0.30) {
        println!("Coverage meets 30% threshold");
    } else {
        println!("WARNING: Coverage below 30% threshold!");
    }
    println!();
}

fn demo_tsp_wasm_coverage() {
    println!("--- TSP WASM Coverage (100% Target) ---\n");

    let mut coverage = GuiCoverage::tsp_wasm();

    // === All 13 API elements ===
    coverage.cover_element("new_from_yaml");
    coverage.cover_element("step");
    coverage.cover_element("get_tour_length");
    coverage.cover_element("get_best_tour");
    coverage.cover_element("get_cities");
    coverage.cover_element("get_gap");
    coverage.cover_element("get_status");
    coverage.cover_element("set_rcl_size");
    coverage.cover_element("set_method");
    coverage.cover_element("reset");
    coverage.cover_element("get_convergence_history");
    coverage.cover_element("get_tour_edges");
    coverage.cover_element("get_best_tour_edges");

    // === All 3 screens ===
    coverage.cover_screen("initialized");
    coverage.cover_screen("running");
    coverage.cover_screen("converged");

    // === All 2 journeys ===
    coverage.complete_journey("basic_solve");
    coverage.complete_journey("full_optimization");

    // Print detailed report
    println!("{}", coverage.detailed_report());

    // Verify 100%
    if coverage.is_complete() {
        println!("100% GUI coverage achieved!");
    }
}

fn demo_custom_coverage() {
    println!("--- Custom Coverage with Macro ---\n");

    // Use macro for quick setup
    let coverage = simular::gui_coverage!("Custom App" =>
        elements: ["btn_save", "btn_cancel", "input_name", "dropdown_type"],
        screens: ["form_view", "confirm_dialog", "success_view"]
    );

    println!("Created custom coverage tracker: {}", coverage.name());
    println!(
        "Registered {} elements, {} screens",
        4, // macro doesn't expose count, but we know we registered 4
        3
    );
    println!("Initial coverage: {}\n", coverage.summary());

    // Show uncovered elements
    let uncovered = coverage.uncovered_elements();
    println!("Uncovered elements: {:?}", uncovered);
}
