//! Integration tests for Sprint 41: Critical Path Analysis
//!
//! This tests the critical path module with realistic trace scenarios.

use renacer::causal_graph::CausalGraph;
use renacer::critical_path::find_critical_path;
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

#[test]
fn test_linear_critical_path() {
    // Scenario: Simple linear chain - entire chain is critical path
    let spans = vec![
        create_span(1, None, 0, 1000, "step1"),
        create_span(2, Some(1), 1, 2000, "step2"),
        create_span(3, Some(2), 2, 1500, "step3"),
    ];

    let graph = CausalGraph::from_spans(&spans).expect("test");
    let result = find_critical_path(&graph).expect("test");

    assert_eq!(result.path.len(), 3);
    assert_eq!(result.total_duration, 1000 + 2000 + 1500);
    assert_eq!(result.span_names, vec!["step1", "step2", "step3"]);

    // All nodes should be on critical path
    for i in 0..3 {
        let node = graph.get_node_by_span_id(&[i + 1; 8]).expect("test");
        assert!(result.is_on_critical_path(node));
    }
}

#[test]
fn test_fan_out_critical_path() {
    // Scenario: Parent with multiple children - longest child defines critical path
    // Root (100ns) → [Child1 (50ns), Child2 (200ns), Child3 (80ns)]
    let spans = vec![
        create_span(1, None, 0, 100, "root"),
        create_span(2, Some(1), 1, 50, "child1"),
        create_span(3, Some(1), 2, 200, "child2"),
        create_span(4, Some(1), 3, 80, "child3"),
    ];

    let graph = CausalGraph::from_spans(&spans).expect("test");
    let result = find_critical_path(&graph).expect("test");

    // Critical path: root → child2
    assert_eq!(result.path.len(), 2);
    assert_eq!(result.total_duration, 100 + 200);
    assert!(result.span_names.contains(&"root".to_string()));
    assert!(result.span_names.contains(&"child2".to_string()));

    // child1 and child3 should NOT be on critical path
    let child1_node = graph.get_node_by_span_id(&[2; 8]).expect("test");
    let child3_node = graph.get_node_by_span_id(&[4; 8]).expect("test");
    assert!(!result.is_on_critical_path(child1_node));
    assert!(!result.is_on_critical_path(child3_node));
}

#[test]
fn test_microservice_critical_path() {
    // Realistic scenario: HTTP request → [Auth (fast), API (slow)]
    // API → [DB (slow), Cache (fast)]
    // Critical path should be: HTTP → API → DB
    let spans = vec![
        create_span(1, None, 0, 50_000, "http.request"),
        create_span(2, Some(1), 1, 10_000, "auth.verify"),
        create_span(3, Some(1), 2, 80_000, "api.handler"),
        create_span(4, Some(3), 3, 60_000, "db.query"),
        create_span(5, Some(3), 4, 5_000, "cache.get"),
    ];

    let graph = CausalGraph::from_spans(&spans).expect("test");
    let result = find_critical_path(&graph).expect("test");

    // Critical path: http.request → api.handler → db.query
    assert_eq!(result.path.len(), 3);
    assert_eq!(result.total_duration, 50_000 + 80_000 + 60_000);
    assert_eq!(result.span_names, vec!["http.request", "api.handler", "db.query"]);

    // auth.verify and cache.get should NOT be on critical path
    let auth_node = graph.get_node_by_span_id(&[2; 8]).expect("test");
    let cache_node = graph.get_node_by_span_id(&[5; 8]).expect("test");
    assert!(!result.is_on_critical_path(auth_node));
    assert!(!result.is_on_critical_path(cache_node));

    // api.handler should be the longest span
    let (longest_node, longest_duration) = result.longest_span().expect("test");
    assert_eq!(longest_duration, 80_000);
    let longest_span = graph.get_span(longest_node).expect("test");
    assert_eq!(longest_span.span_name, "api.handler");
}

#[test]
fn test_complex_distributed_trace_critical_path() {
    // Complex scenario: Multi-level distributed system
    // Gateway → [Auth, API]
    // Auth → UserDB
    // API → [ProductDB, Cache]
    // ProductDB → Index
    //
    // Expected critical path: Gateway → API → ProductDB → Index
    let spans = vec![
        create_span(1, None, 0, 10_000, "gateway"),
        create_span(2, Some(1), 1, 20_000, "auth.service"),
        create_span(3, Some(1), 2, 50_000, "api.service"),
        create_span(4, Some(2), 3, 15_000, "userdb.query"),
        create_span(5, Some(3), 4, 40_000, "productdb.query"),
        create_span(6, Some(3), 5, 5_000, "cache.get"),
        create_span(7, Some(5), 6, 30_000, "index.scan"),
    ];

    let graph = CausalGraph::from_spans(&spans).expect("test");
    let result = find_critical_path(&graph).expect("test");

    // Critical path: gateway → api.service → productdb.query → index.scan
    assert_eq!(result.path.len(), 4);
    assert_eq!(result.total_duration, 10_000 + 50_000 + 40_000 + 30_000);
    assert_eq!(result.span_names, vec!["gateway", "api.service", "productdb.query", "index.scan"]);

    // Verify auth path is NOT on critical path
    let auth_node = graph.get_node_by_span_id(&[2; 8]).expect("test");
    let userdb_node = graph.get_node_by_span_id(&[4; 8]).expect("test");
    assert!(!result.is_on_critical_path(auth_node));
    assert!(!result.is_on_critical_path(userdb_node));
}

#[test]
fn test_critical_path_percentage() {
    // Scenario: Calculate what percentage of total trace time is critical path
    let spans = vec![
        create_span(1, None, 0, 100_000, "root"),
        create_span(2, Some(1), 1, 50_000, "child1"),
        create_span(3, Some(1), 2, 150_000, "child2"),
    ];

    let graph = CausalGraph::from_spans(&spans).expect("test");
    let result = find_critical_path(&graph).expect("test");

    // Critical path: root → child2 = 250,000ns
    // Total trace duration (assuming sequential): 100,000 + 150,000 = 250,000ns
    let total_trace_duration = 250_000;
    let percentage = result.critical_path_percentage(total_trace_duration);

    assert_eq!(percentage, 100.0);

    // If total trace was longer (due to parallel execution)
    let total_with_parallel = 300_000;
    let percentage2 = result.critical_path_percentage(total_with_parallel);
    assert!((percentage2 - 83.33).abs() < 0.01);
}

#[test]
fn test_critical_path_with_equal_branches() {
    // Scenario: Two equally long branches - either could be critical path
    let spans = vec![
        create_span(1, None, 0, 100, "root"),
        create_span(2, Some(1), 1, 200, "child1"),
        create_span(3, Some(1), 2, 200, "child2"),
    ];

    let graph = CausalGraph::from_spans(&spans).expect("test");
    let result = find_critical_path(&graph).expect("test");

    // Critical path should be root + one of the children
    assert_eq!(result.path.len(), 2);
    assert_eq!(result.total_duration, 300);
    assert!(result.span_names.contains(&"root".to_string()));
    assert!(
        result.span_names.contains(&"child1".to_string())
            || result.span_names.contains(&"child2".to_string())
    );
}

#[test]
fn test_deep_recursion_critical_path() {
    // Scenario: Deep recursive call stack - entire stack is critical path
    let mut spans = vec![];
    let depth = 50;
    for i in 0..depth {
        let parent = if i == 0 { None } else { Some(i - 1) };
        spans.push(create_span(i, parent, i as u64, 100, "recursive"));
    }

    let graph = CausalGraph::from_spans(&spans).expect("test");
    let result = find_critical_path(&graph).expect("test");

    assert_eq!(result.path.len(), depth as usize);
    assert_eq!(result.total_duration, depth as u64 * 100);

    // All nodes should be on critical path
    for i in 0..depth {
        let node = graph.get_node_by_span_id(&[i; 8]).expect("test");
        assert!(result.is_on_critical_path(node));
    }
}

#[test]
fn test_parallel_workers_critical_path() {
    // Scenario: Map-reduce pattern - slowest worker defines critical path
    // Coordinator → [Worker1, Worker2, Worker3, Worker4] → Reducer
    let spans = vec![
        create_span(1, None, 0, 10_000, "coordinator"),
        create_span(2, Some(1), 1, 50_000, "worker1"),
        create_span(3, Some(1), 2, 120_000, "worker2"),
        create_span(4, Some(1), 3, 80_000, "worker3"),
        create_span(5, Some(1), 4, 60_000, "worker4"),
        // Reducer depends on coordinator (simplified model)
        create_span(6, Some(1), 5, 20_000, "reducer"),
    ];

    let graph = CausalGraph::from_spans(&spans).expect("test");
    let result = find_critical_path(&graph).expect("test");

    // Critical path should go through slowest worker (worker2)
    assert!(result.span_names.contains(&"coordinator".to_string()));
    assert!(result.span_names.contains(&"worker2".to_string()));

    // worker2 should be the longest span
    let (_longest_node, longest_duration) = result.longest_span().expect("test");
    assert_eq!(longest_duration, 120_000);
}

#[test]
fn test_critical_path_node_durations() {
    // Scenario: Verify per-node durations are captured correctly
    let spans = vec![
        create_span(1, None, 0, 1000, "a"),
        create_span(2, Some(1), 1, 2000, "b"),
        create_span(3, Some(2), 2, 3000, "c"),
    ];

    let graph = CausalGraph::from_spans(&spans).expect("test");
    let result = find_critical_path(&graph).expect("test");

    // Verify each node's duration is recorded
    assert_eq!(result.node_durations.len(), 3);

    let node_a = graph.get_node_by_span_id(&[1; 8]).expect("test");
    let node_b = graph.get_node_by_span_id(&[2; 8]).expect("test");
    let node_c = graph.get_node_by_span_id(&[3; 8]).expect("test");

    assert_eq!(result.node_durations.get(&node_a), Some(&1000));
    assert_eq!(result.node_durations.get(&node_b), Some(&2000));
    assert_eq!(result.node_durations.get(&node_c), Some(&3000));
}

#[test]
fn test_bottleneck_identification() {
    // Scenario: Identify the primary bottleneck in a distributed trace
    // Gateway (100ms) → API (50ms) → DB (500ms) → PostProcess (10ms)
    //
    // DB query is the clear bottleneck
    let spans = vec![
        create_span(1, None, 0, 100_000_000, "gateway"),
        create_span(2, Some(1), 1, 50_000_000, "api"),
        create_span(3, Some(2), 2, 500_000_000, "db.query"),
        create_span(4, Some(3), 3, 10_000_000, "postprocess"),
    ];

    let graph = CausalGraph::from_spans(&spans).expect("test");
    let result = find_critical_path(&graph).expect("test");

    // DB query should be the longest span on critical path
    let (bottleneck_node, bottleneck_duration) = result.longest_span().expect("test");
    assert_eq!(bottleneck_duration, 500_000_000);

    let bottleneck_span = graph.get_span(bottleneck_node).expect("test");
    assert_eq!(bottleneck_span.span_name, "db.query");

    // Critical path should include all spans
    assert_eq!(result.path.len(), 4);
    assert_eq!(result.total_duration, 100_000_000 + 50_000_000 + 500_000_000 + 10_000_000);
}
