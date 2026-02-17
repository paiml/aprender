//! Probar TUI Testing - Demo
//!
//! Demonstrates probar-style TUI testing with:
//! - Playwright-style assertions
//! - Snapshot testing
//! - Frame sequence testing
//! - UX coverage tracking
//!
//! Run: cargo run -p apr-cli --features inference --example probar_tui_testing

use apr_cli::federation::{
    tui::{render_federation_dashboard, FederationApp, FederationTab, RoutingRecord},
    Capability, CircuitBreaker, HealthChecker, ModelCatalog, ModelCatalogTrait, ModelId, NodeId,
    RegionId,
};
use jugar_probar::gui_coverage;
use jugar_probar::tui::{expect_frame, FrameSequence, SnapshotManager, TuiFrame, TuiSnapshot};
use jugar_probar::ux_coverage::{InteractionType, StateId, UxCoverageBuilder};
use ratatui::backend::TestBackend;
use ratatui::Terminal;
use std::sync::Arc;
use std::time::Duration;

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    PROBAR TUI TESTING DEMO                                   ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    let app = create_test_app();
    let frame = demo_frame_rendering(&app);
    demo_playwright_assertions(&frame);
    demo_soft_assertions(&frame);
    demo_snapshot_testing(&app, &frame);
    demo_frame_sequence();
    demo_ux_coverage();
    demo_snapshot_manager(&frame);

    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    PROBAR TUI TESTING DEMO COMPLETE                          ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║  Testing Patterns Demonstrated:                                              ║");
    println!("║    • Frame rendering to test backend                                         ║");
    println!("║    • Playwright-style expect_frame() assertions                              ║");
    println!("║    • Soft assertions for collecting multiple failures                        ║");
    println!("║    • Snapshot testing with hash-based comparison                             ║");
    println!("║    • Frame sequence testing for state transitions                            ║");
    println!("║    • UX coverage tracking (100% coverage achieved)                           ║");
    println!("║    • Golden file workflow with SnapshotManager                               ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");
}

fn demo_frame_rendering(app: &FederationApp) -> TuiFrame {
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ FRAME RENDERING                                                             │");
    println!("└─────────────────────────────────────────────────────────────────────────────┘\n");

    let frame = render_frame(app, 100, 30);
    println!("  Frame dimensions: {}x{}", frame.width(), frame.height());
    println!("  Frame rendered successfully\n");
    println!("  Preview (first 10 lines):");
    for line in frame.as_text().lines().take(10) {
        println!("  │ {}", line);
    }
    println!();
    frame
}

fn print_result(label: &str, ok: bool) {
    println!("  {}... {}", label, if ok { "✓ PASS" } else { "✗ FAIL" });
}

fn demo_playwright_assertions(frame: &TuiFrame) {
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ PLAYWRIGHT-STYLE ASSERTIONS                                                 │");
    println!("└─────────────────────────────────────────────────────────────────────────────┘\n");

    let mut a = expect_frame(frame);
    print_result("to_contain_text(\"Federation Gateway\")", a.to_contain_text("Federation Gateway").is_ok());
    print_result("to_contain_text(\"Navigation\")", a.to_contain_text("Navigation").is_ok());
    print_result("to_contain_text(\"MODEL CATALOG\")", a.to_contain_text("MODEL CATALOG").is_ok());
    print_result("not_to_contain_text(\"ERROR\")", a.not_to_contain_text("ERROR").is_ok());
    print_result("to_have_size(100, 30)", a.to_have_size(100, 30).is_ok());
}

fn demo_soft_assertions(frame: &TuiFrame) {
    println!("\n┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SOFT ASSERTIONS (collect multiple failures)                                 │");
    println!("└─────────────────────────────────────────────────────────────────────────────┘\n");

    let mut soft = expect_frame(frame).soft();
    for text in ["Federation Gateway", "Catalog", "Health", "Routing", "Circuits", "Stats", "Policies", "Help"] {
        let _ = soft.to_contain_text(text);
    }
    let errors = soft.errors();
    if errors.is_empty() {
        println!("  All 8 soft assertions passed ✓");
    } else {
        println!("  {} assertions failed:", errors.len());
        for err in errors {
            println!("    ✗ {}", err);
        }
    }
    match soft.finalize() {
        Ok(()) => println!("  Soft assertion block: PASS"),
        Err(e) => println!("  Soft assertion block: FAIL ({})", e),
    }
}

fn demo_snapshot_testing(app: &FederationApp, frame: &TuiFrame) {
    println!("\n┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SNAPSHOT TESTING                                                            │");
    println!("└─────────────────────────────────────────────────────────────────────────────┘\n");

    let snapshot = TuiSnapshot::from_frame("federation_catalog", frame);
    println!("  Snapshot created:");
    println!("    Name:   {}", snapshot.name);
    println!("    Size:   {}x{}", snapshot.width, snapshot.height);
    println!("    Hash:   {}...", &snapshot.hash[..16]);

    let frame2 = render_frame(app, 100, 30);
    let snapshot2 = TuiSnapshot::from_frame("federation_catalog_2", &frame2);
    print!("\n  Comparing identical frames... ");
    println!("{}", if snapshot.matches(&snapshot2) { "✓ MATCH" } else { "✗ DIFFERENT" });

    let mut app_health = create_test_app();
    app_health.current_tab = FederationTab::Health;
    let frame_health = render_frame(&app_health, 100, 30);
    let snapshot_health = TuiSnapshot::from_frame("federation_health", &frame_health);
    print!("  Comparing different tabs...   ");
    println!("{}", if snapshot.matches(&snapshot_health) { "✓ MATCH (unexpected!)" } else { "✗ DIFFERENT (expected)" });
}

fn demo_frame_sequence() {
    println!("\n┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ FRAME SEQUENCE TESTING (Tab Navigation)                                     │");
    println!("└─────────────────────────────────────────────────────────────────────────────┘\n");

    let mut sequence = FrameSequence::new("federation_tab_navigation");
    let mut app_seq = create_test_app();
    let tabs = [
        FederationTab::Catalog, FederationTab::Health, FederationTab::Routing,
        FederationTab::Circuits, FederationTab::Stats, FederationTab::Policies, FederationTab::Help,
    ];

    println!("  Recording frame sequence...");
    for (i, tab) in tabs.iter().enumerate() {
        app_seq.current_tab = *tab;
        sequence.add_frame(&render_frame(&app_seq, 100, 30));
        println!("    Frame {}: {:?}", i + 1, tab);
    }

    println!("\n  Sequence statistics:");
    println!("    Total frames: {}", sequence.len());
    let first = sequence.first().expect("first frame");
    let last = sequence.last().expect("last frame");
    print!("    First != Last: ");
    println!("{}", if !first.matches(last) { "✓ (different content)" } else { "✗ (unexpected match)" });
}

fn demo_ux_coverage() {
    println!("\n┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ UX COVERAGE TRACKING                                                        │");
    println!("└─────────────────────────────────────────────────────────────────────────────┘\n");

    demo_ux_builder_api();
    demo_ux_macro_api();
}

fn demo_ux_builder_api() {
    println!("  Method 1: UxCoverageBuilder API\n");

    let mut tracker = UxCoverageBuilder::new()
        .clickable("tab", "catalog").clickable("tab", "health").clickable("tab", "routing")
        .clickable("tab", "circuits").clickable("tab", "stats").clickable("tab", "policies")
        .clickable("tab", "help").clickable("nav", "quit")
        .screen("catalog").screen("health").screen("routing")
        .screen("circuits").screen("stats").screen("policies").screen("help")
        .build();

    for tab in ["catalog", "health", "routing", "circuits", "stats", "policies", "help"] {
        tracker.record_interaction(&jugar_probar::ux_coverage::ElementId::new("tab", tab), InteractionType::Click);
        tracker.record_state(StateId::new("screen", tab));
    }
    tracker.record_interaction(&jugar_probar::ux_coverage::ElementId::new("nav", "quit"), InteractionType::Click);

    let report = tracker.generate_report();
    println!("    Elements covered: {}/{}", report.covered_elements, report.total_elements);
    println!("    States covered:   {}/{}", report.covered_states, report.total_states);
    println!("    Overall coverage: {:.1}%", report.overall_coverage * 100.0);
    println!("    Status:           {}", if report.is_complete { "COMPLETE ✓" } else { "INCOMPLETE" });
}

fn demo_ux_macro_api() {
    println!("\n  Method 2: gui_coverage! macro\n");

    let mut gui = gui_coverage! {
        buttons: ["tab_catalog", "tab_health", "tab_routing", "tab_circuits", "tab_stats", "tab_policies", "tab_help", "quit"],
        screens: ["catalog", "health", "routing", "circuits", "stats", "policies", "help"]
    };

    for tab in ["catalog", "health", "routing", "circuits", "stats", "policies", "help"] {
        gui.click(&format!("tab_{}", tab));
        gui.visit(tab);
    }
    gui.click("quit");

    let gui_report = gui.generate_report();
    println!("    Elements covered: {}/{}", gui_report.covered_elements, gui_report.total_elements);
    println!("    States covered:   {}/{}", gui_report.covered_states, gui_report.total_states);
    println!("    Overall coverage: {:.1}%", gui_report.overall_coverage * 100.0);
    println!("    Status:           {}", if gui.is_complete() { "COMPLETE ✓" } else { "INCOMPLETE" });
}

fn demo_snapshot_manager(frame: &TuiFrame) {
    println!("\n┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SNAPSHOT MANAGER (Golden File Workflow)                                     │");
    println!("└─────────────────────────────────────────────────────────────────────────────┘\n");

    let temp_dir = tempfile::TempDir::new().expect("temp dir");
    let manager = SnapshotManager::new(temp_dir.path());
    println!("  Snapshot directory: {:?}\n", temp_dir.path());

    print!("  First run (create golden): ");
    match manager.assert_snapshot("federation_dashboard", frame) {
        Ok(()) => println!("✓ Created"),
        Err(e) => println!("✗ Error: {}", e),
    }

    print!("  Second run (compare):      ");
    match manager.assert_snapshot("federation_dashboard", frame) {
        Ok(()) => println!("✓ Matched"),
        Err(e) => println!("✗ Mismatch: {}", e),
    }

    print!("  Golden file exists:        ");
    println!("{}", if manager.exists("federation_dashboard") { "✓ Yes" } else { "✗ No" });
}

fn create_test_app() -> FederationApp {
    let catalog = Arc::new(ModelCatalog::new());
    let health = Arc::new(HealthChecker::default());
    let circuit_breaker = Arc::new(CircuitBreaker::default());

    // Register sample models
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(async {
        catalog
            .register(
                ModelId("whisper-large-v3".to_string()),
                NodeId("us-west-gpu-01".to_string()),
                RegionId("us-west-2".to_string()),
                vec![Capability::Transcribe],
            )
            .await
            .ok();

        catalog
            .register(
                ModelId("llama-70b-chat".to_string()),
                NodeId("us-east-gpu-01".to_string()),
                RegionId("us-east-1".to_string()),
                vec![Capability::Generate],
            )
            .await
            .ok();
    });

    // Set up health
    health.register_node(NodeId("us-west-gpu-01".to_string()));
    health.register_node(NodeId("us-east-gpu-01".to_string()));
    health.report_success(
        &NodeId("us-west-gpu-01".to_string()),
        Duration::from_millis(50),
    );
    health.report_success(
        &NodeId("us-east-gpu-01".to_string()),
        Duration::from_millis(70),
    );

    let mut app = FederationApp::new(catalog, health, circuit_breaker);

    // Add routing history
    app.record_routing(RoutingRecord {
        request_id: "req-001".to_string(),
        capability: "Transcribe".to_string(),
        selected_node: "us-west-gpu-01".to_string(),
        score: 0.95,
        reason: "lowest latency".to_string(),
        timestamp: std::time::Instant::now(),
    });

    app
}

fn render_frame(app: &FederationApp, width: u16, height: u16) -> TuiFrame {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("terminal creation");
    terminal
        .draw(|f| render_federation_dashboard(f, app))
        .expect("draw");
    TuiFrame::from_buffer(terminal.backend().buffer(), 0)
}
