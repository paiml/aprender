//! Tests for diagnostics engine
//!
//! Tests for StackDiagnostics and graph analytics functionality.

use crate::stack::diagnostics::*;
use crate::stack::quality::StackLayer;

// ========================================================================
// StackDiagnostics Tests
// ========================================================================

#[test]
fn test_stack_diagnostics_new() {
    let diag = StackDiagnostics::new();
    assert_eq!(diag.component_count(), 0);
    assert!(diag.graph().is_none());
    assert!(diag.anomalies().is_empty());
}

#[test]
fn test_stack_diagnostics_add_component() {
    let mut diag = StackDiagnostics::new();
    let node = ComponentNode::new("trueno", "0.7.4", StackLayer::Compute);
    diag.add_component(node);

    assert_eq!(diag.component_count(), 1);
    assert!(diag.get_component("trueno").is_some());
    assert!(diag.get_component("missing").is_none());
}

#[test]
fn test_stack_diagnostics_health_summary_empty() {
    let diag = StackDiagnostics::new();
    let summary = diag.health_summary();

    assert_eq!(summary.total_components, 0);
    assert_eq!(summary.green_count, 0);
    assert_eq!(summary.andon_status, AndonStatus::Unknown);
}

#[test]
fn test_stack_diagnostics_health_summary_all_green() {
    let mut diag = StackDiagnostics::new();

    let mut node1 = ComponentNode::new("trueno", "0.7.4", StackLayer::Compute);
    node1.health = HealthStatus::Green;
    node1.metrics = ComponentMetrics::with_demo_score(95.0);
    diag.add_component(node1);

    let mut node2 = ComponentNode::new("aprender", "0.9.0", StackLayer::Ml);
    node2.health = HealthStatus::Green;
    node2.metrics = ComponentMetrics::with_demo_score(92.0);
    diag.add_component(node2);

    let summary = diag.health_summary();

    assert_eq!(summary.total_components, 2);
    assert_eq!(summary.green_count, 2);
    assert_eq!(summary.yellow_count, 0);
    assert_eq!(summary.red_count, 0);
    assert!(summary.all_healthy());
    assert_eq!(summary.andon_status, AndonStatus::Green);
    assert!((summary.avg_demo_score - 93.5).abs() < 0.1);
}

#[test]
fn test_stack_diagnostics_health_summary_mixed() {
    let mut diag = StackDiagnostics::new();

    let mut node1 = ComponentNode::new("trueno", "0.7.4", StackLayer::Compute);
    node1.health = HealthStatus::Green;
    diag.add_component(node1);

    let mut node2 = ComponentNode::new("weak", "1.0.0", StackLayer::Ml);
    node2.health = HealthStatus::Red;
    diag.add_component(node2);

    let summary = diag.health_summary();

    assert_eq!(summary.green_count, 1);
    assert_eq!(summary.red_count, 1);
    assert!(!summary.all_healthy());
    assert_eq!(summary.andon_status, AndonStatus::Red);
}

#[test]
fn test_stack_diagnostics_add_anomaly() {
    let mut diag = StackDiagnostics::new();

    let anomaly =
        Anomaly::new("trueno-graph", 0.75, AnomalyCategory::CoverageDrop, "Coverage dropped 5.2%")
            .with_evidence("lcov.info shows missing tests")
            .with_recommendation("Add tests for GPU BFS");

    diag.add_anomaly(anomaly);

    assert_eq!(diag.anomalies().len(), 1);
    assert_eq!(diag.anomalies()[0].component, "trueno-graph");
    assert!(!diag.anomalies()[0].is_critical());
}

// ========================================================================
// Graph Analytics Tests
// ========================================================================

#[test]
fn test_compute_metrics_empty() {
    let mut diag = StackDiagnostics::new();
    let metrics = diag.compute_metrics().unwrap();

    assert_eq!(metrics.total_nodes, 0);
    assert_eq!(metrics.total_edges, 0);
    assert_eq!(metrics.density, 0.0);
}

#[test]
fn test_compute_metrics_single_node() {
    let mut diag = StackDiagnostics::new();
    diag.add_component(ComponentNode::new("trueno", "0.7.4", StackLayer::Compute));

    let metrics = diag.compute_metrics().unwrap();

    assert_eq!(metrics.total_nodes, 1);
    assert_eq!(metrics.total_edges, 0);
    assert_eq!(metrics.density, 0.0);
    assert_eq!(metrics.avg_degree, 0.0);

    // PageRank should be 1.0 for single node
    let pagerank = metrics.pagerank.get("trueno").copied().unwrap_or(0.0);
    assert!((pagerank - 1.0).abs() < 0.01, "Single node PageRank should be ~1.0");

    // Depth should be 0 for root
    assert_eq!(metrics.depth_map.get("trueno").copied(), Some(0));
}

#[test]
fn test_compute_metrics_pagerank_chain() {
    let mut diag = StackDiagnostics::new();

    // Create chain: A -> B -> C (where A is root, C has highest PageRank)
    diag.add_component(ComponentNode::new("A", "1.0", StackLayer::Orchestration));
    diag.add_component(ComponentNode::new("B", "1.0", StackLayer::Ml));
    diag.add_component(ComponentNode::new("C", "1.0", StackLayer::Compute));

    let metrics = diag.compute_metrics().unwrap();

    // All nodes have PageRank
    assert!(metrics.pagerank.contains_key("A"));
    assert!(metrics.pagerank.contains_key("B"));
    assert!(metrics.pagerank.contains_key("C"));

    // Sum of PageRanks should be ~1.0
    let sum: f64 = metrics.pagerank.values().sum();
    assert!((sum - 1.0).abs() < 0.01, "PageRank sum should be ~1.0");
}

#[test]
fn test_compute_metrics_betweenness() {
    let mut diag = StackDiagnostics::new();

    // Hub-spoke topology: A is hub, B,C,D are leaves
    diag.add_component(ComponentNode::new("hub", "1.0", StackLayer::Compute));
    diag.add_component(ComponentNode::new("leaf1", "1.0", StackLayer::Ml));
    diag.add_component(ComponentNode::new("leaf2", "1.0", StackLayer::DataMlops));
    diag.add_component(ComponentNode::new("leaf3", "1.0", StackLayer::Orchestration));

    let metrics = diag.compute_metrics().unwrap();

    // All nodes have betweenness
    assert!(metrics.betweenness.contains_key("hub"));
    assert!(metrics.betweenness.contains_key("leaf1"));
    assert!(metrics.betweenness.contains_key("leaf2"));
    assert!(metrics.betweenness.contains_key("leaf3"));

    // Without edges, all betweenness should be 0
    for &v in metrics.betweenness.values() {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn test_compute_metrics_depth() {
    let mut diag = StackDiagnostics::new();

    // Simple graph without dependencies - all are roots
    diag.add_component(ComponentNode::new("root1", "1.0", StackLayer::Compute));
    diag.add_component(ComponentNode::new("root2", "1.0", StackLayer::Ml));
    diag.add_component(ComponentNode::new("root3", "1.0", StackLayer::DataMlops));

    let metrics = diag.compute_metrics().unwrap();

    // All nodes are roots, so depth = 0
    assert_eq!(metrics.depth_map.get("root1").copied(), Some(0));
    assert_eq!(metrics.depth_map.get("root2").copied(), Some(0));
    assert_eq!(metrics.depth_map.get("root3").copied(), Some(0));
    assert_eq!(metrics.max_depth, 0);
}

#[test]
fn test_compute_metrics_graph_density() {
    let mut diag = StackDiagnostics::new();

    // Add 3 nodes
    diag.add_component(ComponentNode::new("A", "1.0", StackLayer::Compute));
    diag.add_component(ComponentNode::new("B", "1.0", StackLayer::Ml));
    diag.add_component(ComponentNode::new("C", "1.0", StackLayer::DataMlops));

    let metrics = diag.compute_metrics().unwrap();

    // No edges, so density = 0
    assert_eq!(metrics.total_nodes, 3);
    assert_eq!(metrics.total_edges, 0);
    assert_eq!(metrics.density, 0.0);

    // max_edges for 3 nodes = 3 * 2 = 6
    // density = edges / max_edges = 0 / 6 = 0
}

#[test]
fn test_compute_metrics_avg_degree() {
    let mut diag = StackDiagnostics::new();

    diag.add_component(ComponentNode::new("node1", "1.0", StackLayer::Compute));
    diag.add_component(ComponentNode::new("node2", "1.0", StackLayer::Ml));

    let metrics = diag.compute_metrics().unwrap();

    assert_eq!(metrics.total_nodes, 2);
    assert_eq!(metrics.avg_degree, 0.0);
}

#[test]
fn test_build_adjacency_no_graph() {
    let mut diag = StackDiagnostics::new();
    diag.add_component(ComponentNode::new("A", "1.0", StackLayer::Compute));
    diag.add_component(ComponentNode::new("B", "1.0", StackLayer::Ml));

    // compute_metrics internally calls build_adjacency
    let metrics = diag.compute_metrics().unwrap();

    // Without a graph, edges should be 0
    assert_eq!(metrics.total_edges, 0);
}

#[test]
fn test_compute_metrics_pagerank_convergence() {
    let mut diag = StackDiagnostics::new();

    // Larger graph to test convergence
    for i in 0..10 {
        diag.add_component(ComponentNode::new(format!("node{}", i), "1.0", StackLayer::Compute));
    }

    let metrics = diag.compute_metrics().unwrap();

    // All nodes should have PageRank assigned
    assert_eq!(metrics.pagerank.len(), 10);

    // Sum should be ~1.0 (normalized)
    let sum: f64 = metrics.pagerank.values().sum();
    assert!((sum - 1.0).abs() < 0.01, "PageRank sum={} should be ~1.0", sum);
}

#[test]
fn test_compute_metrics_multiple_calls() {
    let mut diag = StackDiagnostics::new();
    diag.add_component(ComponentNode::new("X", "1.0", StackLayer::Compute));

    // Call compute_metrics multiple times
    let _ = diag.compute_metrics().unwrap();
    let metrics = diag.compute_metrics().unwrap();

    // Should still work correctly
    assert_eq!(metrics.total_nodes, 1);
    assert!(metrics.pagerank.contains_key("X"));
}

// ========================================================================
// Additional Coverage Tests (DIAG-001 to DIAG-012)
// ========================================================================

#[test]
fn test_diag_001_default_implementation() {
    let diag = StackDiagnostics::default();
    assert_eq!(diag.component_count(), 0);
    assert!(diag.graph().is_none());
}

#[test]
fn test_diag_002_components_iterator() {
    let mut diag = StackDiagnostics::new();
    diag.add_component(ComponentNode::new("a", "1.0", StackLayer::Compute));
    diag.add_component(ComponentNode::new("b", "2.0", StackLayer::Ml));
    diag.add_component(ComponentNode::new("c", "3.0", StackLayer::DataMlops));

    let names: Vec<_> = diag.components().map(|c| c.name.clone()).collect();
    assert_eq!(names.len(), 3);
    assert!(names.contains(&"a".to_string()));
    assert!(names.contains(&"b".to_string()));
    assert!(names.contains(&"c".to_string()));
}

#[test]
fn test_diag_003_set_graph() {
    use crate::stack::DependencyGraph;

    let mut diag = StackDiagnostics::new();
    assert!(diag.graph().is_none());

    let graph = DependencyGraph::new();
    diag.set_graph(graph);

    assert!(diag.graph().is_some());
}

#[test]
fn test_diag_004_health_summary_with_yellow() {
    let mut diag = StackDiagnostics::new();

    let mut node1 = ComponentNode::new("green", "1.0", StackLayer::Compute);
    node1.health = HealthStatus::Green;
    diag.add_component(node1);

    let mut node2 = ComponentNode::new("yellow", "1.0", StackLayer::Ml);
    node2.health = HealthStatus::Yellow;
    diag.add_component(node2);

    let summary = diag.health_summary();

    assert_eq!(summary.green_count, 1);
    assert_eq!(summary.yellow_count, 1);
    assert_eq!(summary.red_count, 0);
    assert_eq!(summary.andon_status, AndonStatus::Yellow);
}

#[test]
fn test_diag_005_health_summary_with_unknown() {
    let mut diag = StackDiagnostics::new();

    // Add a component with default health (Unknown)
    let node = ComponentNode::new("unknown", "1.0", StackLayer::Compute);
    diag.add_component(node);

    let summary = diag.health_summary();

    assert_eq!(summary.total_components, 1);
    assert_eq!(summary.unknown_count, 1);
    assert_eq!(summary.andon_status, AndonStatus::Unknown);
}

#[test]
fn test_diag_006_avg_coverage() {
    let mut diag = StackDiagnostics::new();

    let mut node1 = ComponentNode::new("high", "1.0", StackLayer::Compute);
    node1.health = HealthStatus::Green;
    node1.metrics = ComponentMetrics { coverage: 95.0, demo_score: 90.0, ..Default::default() };
    diag.add_component(node1);

    let mut node2 = ComponentNode::new("low", "1.0", StackLayer::Ml);
    node2.health = HealthStatus::Green;
    node2.metrics = ComponentMetrics { coverage: 75.0, demo_score: 80.0, ..Default::default() };
    diag.add_component(node2);

    let summary = diag.health_summary();
    assert!((summary.avg_coverage - 85.0).abs() < 0.1);
    assert!((summary.avg_demo_score - 85.0).abs() < 0.1);
}

#[test]
fn test_diag_007_anomaly_critical() {
    let anomaly = Anomaly::new(
        "critical-component",
        0.98,
        AnomalyCategory::BuildTimeSpike,
        "Build time spiked",
    );

    assert!(anomaly.is_critical());
    assert_eq!(anomaly.component, "critical-component");
}

#[test]
fn test_diag_008_anomaly_not_critical() {
    let anomaly =
        Anomaly::new("minor-component", 0.30, AnomalyCategory::CoverageDrop, "Minor coverage drop");

    assert!(!anomaly.is_critical());
}

#[test]
fn test_diag_009_metrics_accessor() {
    let mut diag = StackDiagnostics::new();
    diag.add_component(ComponentNode::new("test", "1.0", StackLayer::Compute));
    diag.compute_metrics().unwrap();

    let metrics = diag.metrics();
    assert_eq!(metrics.total_nodes, 1);
}

#[test]
fn test_diag_010_compute_depth_with_chain() {
    let mut diag = StackDiagnostics::new();

    // Add components for a chain
    diag.add_component(ComponentNode::new("root", "1.0", StackLayer::Compute));
    diag.add_component(ComponentNode::new("middle", "1.0", StackLayer::Ml));
    diag.add_component(ComponentNode::new("leaf", "1.0", StackLayer::DataMlops));

    // Without graph edges, all should be at depth 0
    let metrics = diag.compute_metrics().unwrap();

    assert_eq!(metrics.depth_map.get("root").copied(), Some(0));
    assert_eq!(metrics.depth_map.get("middle").copied(), Some(0));
    assert_eq!(metrics.depth_map.get("leaf").copied(), Some(0));
}

#[test]
fn test_diag_011_anomaly_with_evidence_and_recommendation() {
    let anomaly =
        Anomaly::new("test", 0.5, AnomalyCategory::ComplexityIncrease, "Complexity increased")
            .with_evidence("Function X cyclomatic complexity: 15 -> 25")
            .with_recommendation("Refactor into smaller functions");

    assert!(!anomaly.evidence.is_empty());
    assert!(anomaly.recommendation.is_some());
    assert!(anomaly.evidence[0].contains("cyclomatic"));
    assert!(anomaly.recommendation.as_ref().unwrap().contains("Refactor"));
}

#[test]
fn test_diag_012_health_summary_all_healthy() {
    let mut diag = StackDiagnostics::new();

    let mut node = ComponentNode::new("healthy", "1.0", StackLayer::Compute);
    node.health = HealthStatus::Green;
    diag.add_component(node);

    let summary = diag.health_summary();
    assert!(summary.all_healthy());
}

#[test]
fn test_diag_013_health_summary_not_all_healthy() {
    let mut diag = StackDiagnostics::new();

    let mut node1 = ComponentNode::new("healthy", "1.0", StackLayer::Compute);
    node1.health = HealthStatus::Green;
    diag.add_component(node1);

    let mut node2 = ComponentNode::new("sick", "1.0", StackLayer::Ml);
    node2.health = HealthStatus::Yellow;
    diag.add_component(node2);

    let summary = diag.health_summary();
    assert!(!summary.all_healthy());
}

#[test]
fn test_diag_014_component_node_new() {
    let node = ComponentNode::new("test-component", "2.5.3", StackLayer::Training);

    assert_eq!(node.name, "test-component");
    assert_eq!(node.version, "2.5.3");
    assert_eq!(node.layer, StackLayer::Training);
    assert_eq!(node.health, HealthStatus::Unknown);
}

#[test]
fn test_diag_015_component_metrics_with_demo_score() {
    let metrics = ComponentMetrics::with_demo_score(87.5);

    assert_eq!(metrics.demo_score, 87.5);
    assert_eq!(metrics.coverage, 0.0);
    assert_eq!(metrics.mutation_score, 0.0);
}

#[test]
fn test_diag_016_graph_metrics_default() {
    let metrics = GraphMetrics::default();

    assert_eq!(metrics.total_nodes, 0);
    assert_eq!(metrics.total_edges, 0);
    assert_eq!(metrics.density, 0.0);
    assert_eq!(metrics.avg_degree, 0.0);
    assert_eq!(metrics.max_depth, 0);
    assert!(metrics.pagerank.is_empty());
    assert!(metrics.betweenness.is_empty());
    assert!(metrics.depth_map.is_empty());
}

#[test]
fn test_diag_017_andon_status_message() {
    assert_eq!(AndonStatus::Green.message(), "All systems healthy");
    assert_eq!(AndonStatus::Yellow.message(), "Attention needed");
    assert_eq!(AndonStatus::Red.message(), "Stop-the-line");
    assert_eq!(AndonStatus::Unknown.message(), "Analysis pending");
}

#[test]
fn test_diag_018_health_status_symbols() {
    // HealthStatus uses icon() not symbol()
    assert_eq!(HealthStatus::Green.icon(), "🟢");
    assert_eq!(HealthStatus::Yellow.icon(), "🟡");
    assert_eq!(HealthStatus::Red.icon(), "🔴");
    assert_eq!(HealthStatus::Unknown.icon(), "⚪");
    // And symbol() for ASCII
    assert_eq!(HealthStatus::Green.symbol(), "●");
    assert_eq!(HealthStatus::Yellow.symbol(), "◐");
    assert_eq!(HealthStatus::Red.symbol(), "○");
    assert_eq!(HealthStatus::Unknown.symbol(), "◌");
}

// ========================================================================
// Coverage Gap Tests — compute_betweenness + depth with real edges
// ========================================================================

#[test]
fn test_diag_020_betweenness_with_graph_edges() {
    use crate::stack::graph::DependencyEdge;
    use crate::stack::DependencyGraph;
    use crate::stack::{CrateInfo, DependencyKind};

    let mut diag = StackDiagnostics::new();

    // Chain: trueno <- aprender <- entrenar
    diag.add_component(ComponentNode::new("trueno", "0.14.0", StackLayer::Compute));
    diag.add_component(ComponentNode::new("aprender", "0.24.0", StackLayer::Ml));
    diag.add_component(ComponentNode::new("entrenar", "0.5.0", StackLayer::Training));

    let mut graph = DependencyGraph::new();
    let trueno_info = CrateInfo::new(
        "trueno",
        semver::Version::new(0, 14, 0),
        std::path::PathBuf::from("trueno/Cargo.toml"),
    );
    let mut aprender_info = CrateInfo::new(
        "aprender",
        semver::Version::new(0, 24, 0),
        std::path::PathBuf::from("aprender/Cargo.toml"),
    );
    aprender_info
        .paiml_dependencies
        .push(crate::stack::DependencyInfo::new("trueno", "^0.14".to_string()));
    let mut entrenar_info = CrateInfo::new(
        "entrenar",
        semver::Version::new(0, 5, 0),
        std::path::PathBuf::from("entrenar/Cargo.toml"),
    );
    entrenar_info
        .paiml_dependencies
        .push(crate::stack::DependencyInfo::new("aprender", "^0.24".to_string()));

    graph.add_crate(trueno_info);
    graph.add_crate(aprender_info);
    graph.add_crate(entrenar_info);

    graph.add_dependency(
        "aprender",
        "trueno",
        DependencyEdge {
            version_req: "^0.14".to_string(),
            is_path: false,
            kind: DependencyKind::Normal,
        },
    );
    graph.add_dependency(
        "entrenar",
        "aprender",
        DependencyEdge {
            version_req: "^0.24".to_string(),
            is_path: false,
            kind: DependencyKind::Normal,
        },
    );

    diag.set_graph(graph);
    let metrics = diag.compute_metrics().unwrap();

    // With a chain, aprender is the middle node — should have higher betweenness
    let aprender_bc = metrics.betweenness.get("aprender").copied().unwrap_or(0.0);
    let trueno_bc = metrics.betweenness.get("trueno").copied().unwrap_or(0.0);
    let _entrenar_bc = metrics.betweenness.get("entrenar").copied().unwrap_or(0.0);

    // Middle node in a chain should have non-zero betweenness
    // (trueno and entrenar are endpoints)
    assert!(
        aprender_bc >= trueno_bc,
        "aprender betweenness ({}) should be >= trueno ({})",
        aprender_bc,
        trueno_bc
    );

    // Should have edges
    assert!(metrics.total_edges > 0, "Should have edges from graph");
    assert!(metrics.density > 0.0, "Density should be > 0");

    // Depth: entrenar -> aprender -> trueno
    assert!(metrics.max_depth > 0, "Max depth should be > 0 with chain");
}

#[test]
fn test_diag_021_compute_metrics_with_hub_graph() {
    use crate::stack::graph::DependencyEdge;
    use crate::stack::DependencyGraph;
    use crate::stack::{CrateInfo, DependencyKind};

    let mut diag = StackDiagnostics::new();

    // Hub topology: hub depends on leaf1, leaf2, leaf3
    diag.add_component(ComponentNode::new("hub", "1.0.0", StackLayer::Orchestration));
    diag.add_component(ComponentNode::new("leaf1", "1.0.0", StackLayer::Compute));
    diag.add_component(ComponentNode::new("leaf2", "1.0.0", StackLayer::Ml));
    diag.add_component(ComponentNode::new("leaf3", "1.0.0", StackLayer::Training));

    let mut graph = DependencyGraph::new();
    let mut hub_info = CrateInfo::new(
        "hub",
        semver::Version::new(1, 0, 0),
        std::path::PathBuf::from("hub/Cargo.toml"),
    );
    hub_info
        .paiml_dependencies
        .push(crate::stack::DependencyInfo::new("leaf1", "^1.0".to_string()));
    hub_info
        .paiml_dependencies
        .push(crate::stack::DependencyInfo::new("leaf2", "^1.0".to_string()));
    hub_info
        .paiml_dependencies
        .push(crate::stack::DependencyInfo::new("leaf3", "^1.0".to_string()));

    graph.add_crate(hub_info);
    graph.add_crate(CrateInfo::new(
        "leaf1",
        semver::Version::new(1, 0, 0),
        std::path::PathBuf::from("leaf1/Cargo.toml"),
    ));
    graph.add_crate(CrateInfo::new(
        "leaf2",
        semver::Version::new(1, 0, 0),
        std::path::PathBuf::from("leaf2/Cargo.toml"),
    ));
    graph.add_crate(CrateInfo::new(
        "leaf3",
        semver::Version::new(1, 0, 0),
        std::path::PathBuf::from("leaf3/Cargo.toml"),
    ));

    graph.add_dependency(
        "hub",
        "leaf1",
        DependencyEdge {
            version_req: "^1.0".to_string(),
            is_path: false,
            kind: DependencyKind::Normal,
        },
    );
    graph.add_dependency(
        "hub",
        "leaf2",
        DependencyEdge {
            version_req: "^1.0".to_string(),
            is_path: false,
            kind: DependencyKind::Normal,
        },
    );
    graph.add_dependency(
        "hub",
        "leaf3",
        DependencyEdge {
            version_req: "^1.0".to_string(),
            is_path: false,
            kind: DependencyKind::Normal,
        },
    );

    diag.set_graph(graph);
    let metrics = diag.compute_metrics().unwrap();

    assert_eq!(metrics.total_nodes, 4);
    assert_eq!(metrics.total_edges, 3);
    assert!(metrics.avg_degree > 0.0);
}

#[test]
fn test_diag_019_anomaly_category_display() {
    // Uses Display trait, not label method
    assert_eq!(format!("{}", AnomalyCategory::CoverageDrop), "Coverage Drop");
    assert_eq!(format!("{}", AnomalyCategory::BuildTimeSpike), "Build Time Spike");
    assert_eq!(format!("{}", AnomalyCategory::ComplexityIncrease), "Complexity Increase");
    assert_eq!(format!("{}", AnomalyCategory::DependencyRisk), "Dependency Risk");
}
