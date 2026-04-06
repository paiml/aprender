//! Phase 4 algorithm benchmarks
//!
//! Validates performance for:
//! - Louvain community detection
//! - Pattern matching (God Class, Circular Dependencies, Dead Code)
//!
//! Run with: `cargo bench --bench phase4_algorithms`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use trueno_graph::{find_patterns, louvain, CsrGraph, NodeId, Pattern};

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

/// Generate graph with intentional communities
fn generate_community_graph(num_communities: usize, nodes_per_community: usize) -> CsrGraph {
    let mut graph = CsrGraph::new();

    for comm in 0..num_communities {
        let base = (comm * nodes_per_community) as u32;

        // Dense intra-community connections
        for i in 0..nodes_per_community {
            for j in (i + 1)..nodes_per_community {
                let src = NodeId(base + i as u32);
                let dst = NodeId(base + j as u32);
                graph.add_edge(src, dst, 1.0).unwrap();
                graph.add_edge(dst, src, 1.0).unwrap(); // Bidirectional
            }
        }

        // Sparse inter-community connections
        if comm < num_communities - 1 {
            let next_base = ((comm + 1) * nodes_per_community) as u32;
            graph.add_edge(NodeId(base), NodeId(next_base), 1.0).unwrap();
        }
    }

    graph
}

/// Benchmark: Louvain community detection
fn bench_louvain(c: &mut Criterion) {
    let mut group = c.benchmark_group("louvain");

    for (num_communities, nodes_per_comm) in [(2, 10), (3, 15), (4, 20), (5, 25)].iter() {
        let total_nodes = num_communities * nodes_per_comm;
        let graph = generate_community_graph(*num_communities, *nodes_per_comm);

        group.bench_with_input(
            BenchmarkId::new("community_detection", total_nodes),
            &graph,
            |b, graph| {
                b.iter(|| {
                    let result = louvain(black_box(graph)).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Pattern matching - God Class
fn bench_pattern_god_class(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_god_class");

    for size in [100, 500, 1000, 5000].iter() {
        let edges = generate_scale_free_graph(*size, 3);
        let graph = CsrGraph::from_edge_list(&edges).unwrap();
        let pattern = Pattern::god_class(10);

        group.bench_with_input(
            BenchmarkId::new("find_god_class", size),
            &(&graph, &pattern),
            |b, (graph, pattern)| {
                b.iter(|| {
                    let matches = find_patterns(black_box(graph), black_box(pattern)).unwrap();
                    black_box(matches);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Pattern matching - Circular Dependencies
fn bench_pattern_circular_dependency(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_circular_dependency");

    for size in [100, 500, 1000].iter() {
        let edges = generate_scale_free_graph(*size, 3);
        let graph = CsrGraph::from_edge_list(&edges).unwrap();
        let pattern = Pattern::circular_dependency(3);

        group.bench_with_input(
            BenchmarkId::new("find_cycles", size),
            &(&graph, &pattern),
            |b, (graph, pattern)| {
                b.iter(|| {
                    let matches = find_patterns(black_box(graph), black_box(pattern)).unwrap();
                    black_box(matches);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Pattern matching - Dead Code
fn bench_pattern_dead_code(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_dead_code");

    for size in [100, 500, 1000, 5000].iter() {
        let edges = generate_scale_free_graph(*size, 3);
        let graph = CsrGraph::from_edge_list(&edges).unwrap();
        let pattern = Pattern::dead_code();

        group.bench_with_input(
            BenchmarkId::new("find_dead_code", size),
            &(&graph, &pattern),
            |b, (graph, pattern)| {
                b.iter(|| {
                    let matches = find_patterns(black_box(graph), black_box(pattern)).unwrap();
                    black_box(matches);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_louvain,
    bench_pattern_god_class,
    bench_pattern_circular_dependency,
    bench_pattern_dead_code
);
criterion_main!(benches);
