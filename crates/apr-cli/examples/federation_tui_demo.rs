//! Demo: Render Federation TUI frames to terminal
//!
//! Run with: cargo run -p apr-cli --features inference --example federation_tui_demo

use apr_cli::federation::{
    tui::{render_federation_dashboard, FederationApp, FederationTab, RoutingRecord},
    Capability, CircuitBreaker, CircuitBreakerTrait, HealthChecker, ModelCatalog,
    ModelCatalogTrait, ModelId, NodeId, RegionId,
};
use ratatui::backend::TestBackend;
use ratatui::Terminal;
use std::sync::Arc;
use std::time::Duration;

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    FEDERATION GATEWAY TUI DEMO                               ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    // Create components with sample data
    let catalog = Arc::new(ModelCatalog::new());
    let health = Arc::new(HealthChecker::default());
    let circuit_breaker = Arc::new(CircuitBreaker::default());

    // Register some sample models
    let rt = tokio::runtime::Runtime::new().expect("TUI setup");
    rt.block_on(async {
        // Register whisper model on two nodes
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
                ModelId("whisper-large-v3".to_string()),
                NodeId("eu-west-gpu-01".to_string()),
                RegionId("eu-west-1".to_string()),
                vec![Capability::Transcribe],
            )
            .await
            .ok();

        // Register LLM model
        catalog
            .register(
                ModelId("llama-70b-chat".to_string()),
                NodeId("us-east-gpu-01".to_string()),
                RegionId("us-east-1".to_string()),
                vec![Capability::Generate, Capability::Code],
            )
            .await
            .ok();

        // Register embedding model
        catalog
            .register(
                ModelId("bge-large-en".to_string()),
                NodeId("ap-south-cpu-01".to_string()),
                RegionId("ap-south-1".to_string()),
                vec![Capability::Embed],
            )
            .await
            .ok();
    });

    // Set up health for nodes
    health.register_node(NodeId("us-west-gpu-01".to_string()));
    health.register_node(NodeId("eu-west-gpu-01".to_string()));
    health.register_node(NodeId("us-east-gpu-01".to_string()));
    health.register_node(NodeId("ap-south-cpu-01".to_string()));

    // Report health
    for _ in 0..3 {
        health.report_success(
            &NodeId("us-west-gpu-01".to_string()),
            Duration::from_millis(45),
        );
        health.report_success(
            &NodeId("eu-west-gpu-01".to_string()),
            Duration::from_millis(120),
        );
        health.report_success(
            &NodeId("us-east-gpu-01".to_string()),
            Duration::from_millis(65),
        );
        health.report_success(
            &NodeId("ap-south-cpu-01".to_string()),
            Duration::from_millis(200),
        );
    }

    // Make one node degraded
    health.report_failure(&NodeId("ap-south-cpu-01".to_string()));

    // Trip circuit breaker on one node
    for _ in 0..5 {
        circuit_breaker.record_failure(&NodeId("eu-west-gpu-01".to_string()));
    }

    // Create app with data
    let mut app = FederationApp::new(
        Arc::clone(&catalog),
        Arc::clone(&health),
        Arc::clone(&circuit_breaker),
    );

    // Add some routing history
    app.record_routing(RoutingRecord {
        request_id: "req-001".to_string(),
        capability: "Transcribe".to_string(),
        selected_node: "us-west-gpu-01".to_string(),
        score: 0.95,
        reason: "lowest latency, healthy".to_string(),
        timestamp: std::time::Instant::now(),
    });
    app.record_routing(RoutingRecord {
        request_id: "req-002".to_string(),
        capability: "Generate".to_string(),
        selected_node: "us-east-gpu-01".to_string(),
        score: 0.88,
        reason: "only available node".to_string(),
        timestamp: std::time::Instant::now(),
    });
    app.record_routing(RoutingRecord {
        request_id: "req-003".to_string(),
        capability: "Embed".to_string(),
        selected_node: "ap-south-cpu-01".to_string(),
        score: 0.72,
        reason: "degraded, cost optimized".to_string(),
        timestamp: std::time::Instant::now(),
    });

    // Render each tab
    let tabs = [
        ("CATALOG", FederationTab::Catalog),
        ("HEALTH", FederationTab::Health),
        ("ROUTING", FederationTab::Routing),
        ("CIRCUITS", FederationTab::Circuits),
        ("STATS", FederationTab::Stats),
        ("POLICIES", FederationTab::Policies),
        ("HELP", FederationTab::Help),
    ];

    for (name, tab) in tabs {
        app.current_tab = tab;

        println!("┌─────────────────────────────────────────────────────────────────────────────┐");
        println!("│ TAB: {:<71} │", name);
        println!("└─────────────────────────────────────────────────────────────────────────────┘");

        // Render frame
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).expect("TUI setup");
        terminal
            .draw(|f| render_federation_dashboard(f, &app))
            .expect("TUI setup");

        // Print the buffer
        let buffer = terminal.backend().buffer();
        for y in 0..buffer.area.height {
            let mut line = String::new();
            for x in 0..buffer.area.width {
                let cell = &buffer[(x, y)];
                line.push_str(cell.symbol());
            }
            println!("{}", line);
        }
        println!();
    }

    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                         DEMO COMPLETE                                        ║");
    println!("║  All 7 tabs rendered successfully with sample data                           ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
}
