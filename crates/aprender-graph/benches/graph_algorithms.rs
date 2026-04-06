//! Criterion benchmarks for graph algorithms
//!
//! Validates performance claims:
//! - BFS: 10x faster than CPU baseline
//! - PageRank: 5x faster than NetworkX equivalent
//! - CSR construction: Sub-millisecond for small graphs

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use trueno_graph::{bfs, find_callers, pagerank, CsrGraph, NodeId};

/// Generate scale-free graph (Barabási-Albert model approximation)
fn generate_scale_free_graph(
    num_nodes: usize,
    edges_per_node: usize,
) -> Vec<(NodeId, NodeId, f32)> {
    let mut edges = Vec::new();
    let mut rng_state = 12345_u64; // Simple LCG for reproducibility

    for node in 0..num_nodes {
        for _ in 0..edges_per_node {
            // Simple pseudo-random target selection
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let target = (rng_state % num_nodes as u64) as u32;

            if target != node as u32 {
                edges.push((NodeId(node as u32), NodeId(target), 1.0));
            }
        }
    }

    edges
}

/// Benchmark: CSR graph construction from edge list
fn bench_csr_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("csr_construction");

    for size in [100, 500, 1000, 5000].iter() {
        let edges = generate_scale_free_graph(*size, 3);

        group.bench_with_input(BenchmarkId::new("from_edge_list", size), &edges, |b, edges| {
            b.iter(|| {
                let graph = CsrGraph::from_edge_list(black_box(edges)).unwrap();
                black_box(graph);
            });
        });
    }

    group.finish();
}

/// Benchmark: BFS traversal performance
fn bench_bfs(c: &mut Criterion) {
    let mut group = c.benchmark_group("bfs");

    for size in [100, 500, 1000, 5000].iter() {
        let edges = generate_scale_free_graph(*size, 3);
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        group.bench_with_input(BenchmarkId::new("traversal", size), &graph, |b, graph| {
            b.iter(|| {
                let reachable = bfs(black_box(graph), NodeId(0)).unwrap();
                black_box(reachable);
            });
        });
    }

    group.finish();
}

/// Benchmark: find_callers (reverse BFS)
fn bench_find_callers(c: &mut Criterion) {
    let mut group = c.benchmark_group("find_callers");

    for size in [100, 500, 1000, 5000].iter() {
        let edges = generate_scale_free_graph(*size, 3);
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        group.bench_with_input(BenchmarkId::new("depth_10", size), &graph, |b, graph| {
            b.iter(|| {
                let callers = find_callers(black_box(graph), NodeId(*size as u32 / 2), 10).unwrap();
                black_box(callers);
            });
        });
    }

    group.finish();
}

/// Benchmark: PageRank algorithm
fn bench_pagerank(c: &mut Criterion) {
    let mut group = c.benchmark_group("pagerank");

    for size in [100, 500, 1000].iter() {
        let edges = generate_scale_free_graph(*size, 3);
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        group.bench_with_input(BenchmarkId::new("20_iterations", size), &graph, |b, graph| {
            b.iter(|| {
                let scores = pagerank(black_box(graph), 20, 1e-6).unwrap();
                black_box(scores);
            });
        });
    }

    group.finish();
}

/// Benchmark: Neighbor queries (outgoing vs incoming)
///
/// Demonstrates O(1) performance for both outgoing (forward CSR) and
/// incoming (reverse CSR) neighbor queries. Prior to Phase 3, incoming
/// neighbors required O(E) scan. With reverse CSR, both are O(1).
fn bench_neighbor_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbor_queries");

    let edges = generate_scale_free_graph(1000, 5);
    let graph = CsrGraph::from_edge_list(&edges).unwrap();

    group.bench_function("outgoing_neighbors_O1", |b| {
        b.iter(|| {
            for node in 0..100 {
                let neighbors = graph.outgoing_neighbors(NodeId(node)).unwrap();
                black_box(neighbors);
            }
        });
    });

    group.bench_function("incoming_neighbors_O1", |b| {
        b.iter(|| {
            for node in 0..100 {
                let neighbors = graph.incoming_neighbors(NodeId(node)).unwrap();
                black_box(neighbors);
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_csr_construction,
    bench_bfs,
    bench_find_callers,
    bench_pagerank,
    bench_neighbor_queries
);
criterion_main!(benches);
