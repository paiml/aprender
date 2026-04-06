//! Graph Construction Benchmark (Sprint 41)
//!
//! Target: <100ms for 1K spans
//!
//! This benchmark validates the performance requirements for causal graph
//! construction using trueno-graph's CSR format.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use renacer::causal_graph::CausalGraph;
use renacer::critical_path::find_critical_path;
use renacer::span_record::{SpanKind, SpanRecord, StatusCode};
use std::collections::HashMap;

fn create_span(
    span_id: u16,
    parent_id: Option<u16>,
    logical_clock: u64,
    duration_nanos: u64,
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
        format!("span_{}", span_id),
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

/// Benchmark: Graph construction for 1K spans (linear chain)
///
/// Target: <100ms
fn bench_graph_construction_1k_linear(c: &mut Criterion) {
    let mut spans = vec![];
    for i in 0..1024 {
        let parent = if i == 0 { None } else { Some(i - 1) };
        spans.push(create_span(i, parent, i as u64, 100));
    }

    c.bench_function("graph_construction_1k_linear", |b| {
        b.iter(|| {
            let graph = CausalGraph::from_spans(&spans).expect("test");
            black_box(graph);
        });
    });
}

/// Benchmark: Graph construction for 1K spans (balanced tree)
///
/// Target: <100ms
fn bench_graph_construction_1k_tree(c: &mut Criterion) {
    let mut spans = vec![];
    for i in 0..1023 {
        let parent = if i == 0 { None } else { Some((i - 1) / 2) };
        spans.push(create_span(i, parent, i as u64, 100));
    }

    c.bench_function("graph_construction_1k_tree", |b| {
        b.iter(|| {
            let graph = CausalGraph::from_spans(&spans).expect("test");
            black_box(graph);
        });
    });
}

/// Benchmark: Graph construction for 1K spans (fan-out)
///
/// Target: <100ms
fn bench_graph_construction_1k_fanout(c: &mut Criterion) {
    let mut spans = vec![];

    // Root
    spans.push(create_span(0, None, 0, 1000));

    // 1023 children of root (fan-out)
    for i in 1..1024 {
        spans.push(create_span(i, Some(0), i as u64, 100));
    }

    c.bench_function("graph_construction_1k_fanout", |b| {
        b.iter(|| {
            let graph = CausalGraph::from_spans(&spans).expect("test");
            black_box(graph);
        });
    });
}

/// Benchmark: Graph construction + critical path for 1K spans
///
/// Target: <100ms total
fn bench_graph_and_critical_path_1k(c: &mut Criterion) {
    let mut spans = vec![];
    for i in 0..1024 {
        let parent = if i == 0 { None } else { Some(i - 1) };
        spans.push(create_span(i, parent, i as u64, 100));
    }

    c.bench_function("graph_and_critical_path_1k", |b| {
        b.iter(|| {
            let graph = CausalGraph::from_spans(&spans).expect("test");
            let result = find_critical_path(&graph).expect("test");
            black_box(result);
        });
    });
}

/// Benchmark: Graph construction for 10K spans
///
/// Scalability test
fn bench_graph_construction_10k(c: &mut Criterion) {
    let mut spans = vec![];
    for i in 0..10_000 {
        let parent = if i == 0 { None } else { Some(i - 1) };
        spans.push(create_span(i, parent, i as u64, 100));
    }

    c.bench_function("graph_construction_10k", |b| {
        b.iter(|| {
            let graph = CausalGraph::from_spans(&spans).expect("test");
            black_box(graph);
        });
    });
}

/// Benchmark: Graph construction for complex distributed trace (1K spans)
///
/// Realistic scenario with multiple roots and complex structure
fn bench_graph_construction_distributed_1k(c: &mut Criterion) {
    let mut spans = vec![];

    // Create 10 independent traces, each with ~100 spans
    for trace_idx in 0..10 {
        let root_id = trace_idx * 100;
        spans.push(create_span(root_id, None, root_id as u64, 1000));

        // Each trace has a tree structure
        for i in 1..100 {
            let span_id = root_id + i;
            let parent_id = if i < 3 {
                root_id // First 2 children of root
            } else {
                root_id + ((i - 1) / 2) // Tree structure
            };
            spans.push(create_span(span_id, Some(parent_id), span_id as u64, 100));
        }
    }

    c.bench_function("graph_construction_distributed_1k", |b| {
        b.iter(|| {
            let graph = CausalGraph::from_spans(&spans).expect("test");
            black_box(graph);
        });
    });
}

/// Benchmark: Graph node and edge queries
///
/// Query performance after construction
fn bench_graph_queries_1k(c: &mut Criterion) {
    let mut spans = vec![];
    for i in 0..1024 {
        let parent = if i == 0 { None } else { Some((i - 1) / 2) };
        spans.push(create_span(i, parent, i as u64, 100));
    }

    let graph = CausalGraph::from_spans(&spans).expect("test");

    c.bench_function("graph_queries_1k", |b| {
        b.iter(|| {
            // Query all children
            for i in 0..512 {
                let node = trueno_graph::NodeId(i);
                let children = graph.children(node).expect("test");
                black_box(children);
            }
        });
    });
}

/// Benchmark: DAG validation for 1K spans
///
/// Cycle detection performance
fn bench_dag_validation_1k(c: &mut Criterion) {
    let mut spans = vec![];
    for i in 0..1024 {
        let parent = if i == 0 { None } else { Some(i - 1) };
        spans.push(create_span(i, parent, i as u64, 100));
    }

    let graph = CausalGraph::from_spans(&spans).expect("test");

    c.bench_function("dag_validation_1k", |b| {
        b.iter(|| {
            let is_dag = graph.is_dag().expect("test");
            black_box(is_dag);
        });
    });
}

criterion_group!(
    benches,
    bench_graph_construction_1k_linear,
    bench_graph_construction_1k_tree,
    bench_graph_construction_1k_fanout,
    bench_graph_and_critical_path_1k,
    bench_graph_construction_10k,
    bench_graph_construction_distributed_1k,
    bench_graph_queries_1k,
    bench_dag_validation_1k,
);
criterion_main!(benches);
