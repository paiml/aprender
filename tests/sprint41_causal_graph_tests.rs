//! Integration tests for Sprint 41: Causal Graph Construction
//!
//! This tests the causal graph module with realistic trace scenarios.

use renacer::causal_graph::CausalGraph;
use renacer::span_record::{SpanKind, SpanRecord, StatusCode};
use std::collections::HashMap;

fn create_span(
    span_id: u8,
    parent_id: Option<u8>,
    logical_clock: u64,
    duration_nanos: u64,
    name: &str,
) -> SpanRecord {
    SpanRecord::new(
        [1; 16],
        [span_id; 8],
        parent_id.map(|p| [p; 8]),
        name.to_string(),
        SpanKind::Internal,
        logical_clock * 1000,
        logical_clock * 1000 + duration_nanos,
        logical_clock,
        StatusCode::Ok,
        String::new(),
        HashMap::new(),
        HashMap::new(),
        1234,
        5678,
    )
}

fn create_span_u16(
    span_id: u16,
    parent_id: Option<u16>,
    logical_clock: u64,
    duration_nanos: u64,
    name: &str,
) -> SpanRecord {
    // Create span_id as [u8; 8] from u16
    let mut sid = [0u8; 8];
    sid[0..2].copy_from_slice(&span_id.to_le_bytes());

    let parent_sid = parent_id.map(|p| {
        let mut psid = [0u8; 8];
        psid[0..2].copy_from_slice(&p.to_le_bytes());
        psid
    });

    SpanRecord::new(
        [1; 16],
        sid,
        parent_sid,
        name.to_string(),
        SpanKind::Internal,
        logical_clock * 1000,
        logical_clock * 1000 + duration_nanos,
        logical_clock,
        StatusCode::Ok,
        String::new(),
        HashMap::new(),
        HashMap::new(),
        1234,
        5678,
    )
}

#[test]
fn test_realistic_microservice_trace() {
    // Realistic scenario: HTTP request → DB query → Cache check → Response
    let spans = vec![
        create_span(1, None, 0, 50_000, "http.request"),
        create_span(2, Some(1), 1, 30_000, "db.query"),
        create_span(3, Some(2), 2, 5_000, "cache.check"),
        create_span(4, Some(1), 3, 10_000, "http.response"),
    ];

    let graph = CausalGraph::from_spans(&spans).expect("test");

    // Verify structure
    assert_eq!(graph.node_count(), 4);
    assert_eq!(graph.edge_count(), 3); // 1→2, 2→3, 1→4
    assert_eq!(graph.roots().len(), 1);

    // Verify DAG property
    assert!(graph.is_dag().expect("test"));

    // Verify span metadata retrieval
    let root_span = graph.get_span_by_id(&[1; 8]).expect("test");
    assert_eq!(root_span.span_name, "http.request");
    assert_eq!(root_span.duration_nanos, 50_000);
}

#[test]
fn test_parallel_execution_graph() {
    // Scenario: Fan-out pattern (1 parent, multiple parallel children)
    let spans = vec![
        create_span(1, None, 0, 100_000, "orchestrator"),
        create_span(2, Some(1), 1, 40_000, "worker.1"),
        create_span(3, Some(1), 2, 60_000, "worker.2"),
        create_span(4, Some(1), 3, 50_000, "worker.3"),
    ];

    let graph = CausalGraph::from_spans(&spans).expect("test");

    assert_eq!(graph.node_count(), 4);
    assert_eq!(graph.edge_count(), 3);

    // Root should have 3 children
    let root_node = graph.get_node_by_span_id(&[1; 8]).expect("test");
    let children = graph.children(root_node).expect("test");
    assert_eq!(children.len(), 3);

    // Verify edge weights (should be child durations for critical path)
    let worker2_node = graph.get_node_by_span_id(&[3; 8]).expect("test");
    let (_, weight) = children.iter().find(|(node, _)| *node == worker2_node).expect("test");
    assert_eq!(*weight as u64, 60_000); // worker.2 duration
}

#[test]
fn test_deep_call_stack() {
    // Scenario: Deep recursive call stack
    let mut spans = vec![];
    for i in 0..20 {
        let parent = if i == 0 { None } else { Some(i - 1) };
        spans.push(create_span(i, parent, i as u64, 100, "recursive_call"));
    }

    let graph = CausalGraph::from_spans(&spans).expect("test");

    assert_eq!(graph.node_count(), 20);
    assert_eq!(graph.edge_count(), 19); // Linear chain
    assert!(graph.is_dag().expect("test"));

    // Verify linear structure
    for i in 0..19 {
        let node = graph.get_node_by_span_id(&[i; 8]).expect("test");
        let children = graph.children(node).expect("test");
        assert_eq!(children.len(), 1, "Node {} should have 1 child", i);
    }

    // Last node should have no children
    let last_node = graph.get_node_by_span_id(&[19; 8]).expect("test");
    let children = graph.children(last_node).expect("test");
    assert_eq!(children.len(), 0);
}

#[test]
fn test_multiple_root_traces() {
    // Scenario: Multiple independent traces in same dataset
    let spans = vec![
        // Trace 1
        create_span(1, None, 0, 1000, "trace1.root"),
        create_span(2, Some(1), 1, 500, "trace1.child"),
        // Trace 2
        create_span(3, None, 2, 2000, "trace2.root"),
        create_span(4, Some(3), 3, 800, "trace2.child"),
    ];

    let graph = CausalGraph::from_spans(&spans).expect("test");

    assert_eq!(graph.node_count(), 4);
    assert_eq!(graph.roots().len(), 2); // Two independent traces
    assert!(graph.is_dag().expect("test"));
}

#[test]
fn test_complex_distributed_trace() {
    // Scenario: Complex distributed system with multiple services
    //
    // Gateway (1) → [Auth (2), API (3)]
    // Auth (2) → UserDB (4)
    // API (3) → [ProductDB (5), Cache (6)]
    // ProductDB (5) → Index (7)
    let spans = vec![
        create_span(1, None, 0, 100_000, "gateway"),
        create_span(2, Some(1), 1, 30_000, "auth.service"),
        create_span(3, Some(1), 2, 50_000, "api.service"),
        create_span(4, Some(2), 3, 20_000, "userdb.query"),
        create_span(5, Some(3), 4, 40_000, "productdb.query"),
        create_span(6, Some(3), 5, 5_000, "cache.get"),
        create_span(7, Some(5), 6, 15_000, "index.scan"),
    ];

    let graph = CausalGraph::from_spans(&spans).expect("test");

    assert_eq!(graph.node_count(), 7);
    assert!(graph.is_dag().expect("test"));

    // Verify gateway has 2 children
    let gateway_node = graph.get_node_by_span_id(&[1; 8]).expect("test");
    let gateway_children = graph.children(gateway_node).expect("test");
    assert_eq!(gateway_children.len(), 2);

    // Verify API service has 2 children
    let api_node = graph.get_node_by_span_id(&[3; 8]).expect("test");
    let api_children = graph.children(api_node).expect("test");
    assert_eq!(api_children.len(), 2);

    // Verify leaf nodes have no children
    let cache_node = graph.get_node_by_span_id(&[6; 8]).expect("test");
    let cache_children = graph.children(cache_node).expect("test");
    assert_eq!(cache_children.len(), 0);
}

#[test]
fn test_lamport_clock_causality() {
    // Scenario: Verify Lamport clock ordering is preserved
    let spans = vec![
        create_span(1, None, 0, 1000, "event.0"),
        create_span(2, Some(1), 1, 1000, "event.1"),
        create_span(3, Some(2), 2, 1000, "event.2"),
        create_span(4, Some(3), 3, 1000, "event.3"),
    ];

    let graph = CausalGraph::from_spans(&spans).expect("test");

    // Verify causally ordered
    for i in 1..=4 {
        let span = graph.get_span_by_id(&[i; 8]).expect("test");
        assert_eq!(span.logical_clock, (i - 1) as u64);
    }

    // Verify parent happens-before child
    for i in 2..=4 {
        let child = graph.get_span_by_id(&[i; 8]).expect("test");
        let parent = graph.get_span_by_id(&[(i - 1); 8]).expect("test");
        assert!(parent.logical_clock < child.logical_clock);
    }
}

#[test]
fn test_graph_statistics() {
    // Scenario: Verify graph statistics are accurate
    let spans = vec![
        create_span(1, None, 0, 5000, "root"),
        create_span(2, Some(1), 1, 3000, "child1"),
        create_span(3, Some(1), 2, 2000, "child2"),
        create_span(4, Some(2), 3, 1000, "grandchild1"),
    ];

    let graph = CausalGraph::from_spans(&spans).expect("test");

    assert_eq!(graph.node_count(), 4);
    assert_eq!(graph.edge_count(), 3);
    assert_eq!(graph.roots().len(), 1);

    // Verify all spans are retrievable
    for i in 1..=4 {
        let span = graph.get_span_by_id(&[i; 8]);
        assert!(span.is_some(), "Span {} should be retrievable", i);
    }
}

#[test]
fn test_edge_weight_accuracy() {
    // Scenario: Verify edge weights represent child span durations
    let spans =
        vec![create_span(1, None, 0, 10_000, "parent"), create_span(2, Some(1), 1, 5_000, "child")];

    let graph = CausalGraph::from_spans(&spans).expect("test");

    let parent_node = graph.get_node_by_span_id(&[1; 8]).expect("test");
    let children = graph.children(parent_node).expect("test");

    assert_eq!(children.len(), 1);
    let (child_node, weight) = children[0];

    // Weight should equal child span duration
    let child_span = graph.get_span(child_node).expect("test");
    assert_eq!(weight as u64, child_span.duration_nanos);
    assert_eq!(weight as u64, 5_000);
}

#[test]
fn test_performance_1k_spans() {
    // Scenario: Performance test - graph construction should be <100ms for 1K spans
    use std::time::Instant;

    let mut spans = vec![];

    // Create a balanced binary tree of 1023 spans (2^10 - 1)
    for i in 0..1023 {
        let parent = if i == 0 { None } else { Some((i - 1) / 2) };
        spans.push(create_span_u16(i, parent, i as u64, 100, "node"));
    }

    let start = Instant::now();
    let graph = CausalGraph::from_spans(&spans).expect("test");
    let duration = start.elapsed();

    assert_eq!(graph.node_count(), 1023);
    assert!(graph.is_dag().expect("test"));

    // Target: <100ms for 1K spans
    println!("Graph construction for 1023 spans took: {:?}", duration);
    assert!(
        duration.as_millis() < 100,
        "Graph construction took {}ms, expected <100ms",
        duration.as_millis()
    );
}
