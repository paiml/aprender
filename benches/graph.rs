use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use aprender::graph::Graph;

// Benchmark helper functions
fn generate_random_edges(n_nodes: usize, n_edges: usize, seed: u64) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    let mut rng_state = seed;

    for _ in 0..n_edges {
        // Simple LCG for deterministic random numbers
        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        let u = (rng_state % n_nodes as u64) as usize;

        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        let v = (rng_state % n_nodes as u64) as usize;

        if u != v {
            edges.push((u, v));
        }
    }

    edges
}

fn generate_weighted_edges(
    n_nodes: usize,
    n_edges: usize,
    seed: u64,
) -> Vec<(usize, usize, f64)> {
    let mut edges = Vec::new();
    let mut rng_state = seed;

    for _ in 0..n_edges {
        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        let u = (rng_state % n_nodes as u64) as usize;

        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        let v = (rng_state % n_nodes as u64) as usize;

        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        let weight = (rng_state % 100) as f64 / 10.0;

        if u != v {
            edges.push((u, v, weight));
        }
    }

    edges
}

// Pathfinding Benchmarks
fn bench_shortest_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("shortest_path");

    for size in [100, 500, 1000].iter() {
        let edges = generate_random_edges(*size, size * 5, 12345);
        let graph = Graph::from_edges(&edges, false);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(graph.shortest_path(0, size / 2))
            });
        });
    }

    group.finish();
}

fn bench_dijkstra(c: &mut Criterion) {
    let mut group = c.benchmark_group("dijkstra");

    for size in [100, 500, 1000].iter() {
        let edges = generate_weighted_edges(*size, size * 5, 12345);
        let graph = Graph::from_weighted_edges(&edges, false);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(graph.dijkstra(0, size / 2))
            });
        });
    }

    group.finish();
}

fn bench_a_star(c: &mut Criterion) {
    let mut group = c.benchmark_group("a_star");

    for size in [100, 500, 1000].iter() {
        let edges = generate_weighted_edges(*size, size * 5, 12345);
        let graph = Graph::from_weighted_edges(&edges, false);
        let target = size / 2;

        // Simple heuristic: constant for benchmark consistency
        let heuristic = |_node: usize| 5.0;

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(graph.a_star(0, target, heuristic))
            });
        });
    }

    group.finish();
}

fn bench_all_pairs_shortest_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("all_pairs_shortest_paths");

    // Smaller sizes for O(n²) algorithm
    for size in [50, 100, 200].iter() {
        let edges = generate_random_edges(*size, size * 3, 12345);
        let graph = Graph::from_edges(&edges, false);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(graph.all_pairs_shortest_paths())
            });
        });
    }

    group.finish();
}

// Traversal Benchmarks
fn bench_dfs(c: &mut Criterion) {
    let mut group = c.benchmark_group("dfs");

    for size in [100, 500, 1000, 5000].iter() {
        let edges = generate_random_edges(*size, size * 3, 12345);
        let graph = Graph::from_edges(&edges, false);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(graph.dfs(0))
            });
        });
    }

    group.finish();
}

// Component Benchmarks
fn bench_connected_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("connected_components");

    for size in [100, 500, 1000, 5000].iter() {
        let edges = generate_random_edges(*size, size * 3, 12345);
        let graph = Graph::from_edges(&edges, false);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(graph.connected_components())
            });
        });
    }

    group.finish();
}

fn bench_strongly_connected_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("strongly_connected_components");

    for size in [100, 500, 1000, 5000].iter() {
        let edges = generate_random_edges(*size, size * 3, 12345);
        let graph = Graph::from_edges(&edges, true); // directed

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(graph.strongly_connected_components())
            });
        });
    }

    group.finish();
}

fn bench_topological_sort(c: &mut Criterion) {
    let mut group = c.benchmark_group("topological_sort");

    for size in [100, 500, 1000].iter() {
        // Generate DAG edges (u -> v where u < v)
        let mut edges = Vec::new();
        let mut rng_state = 12345u64;

        for _ in 0..(size * 2) {
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let u = (rng_state % *size as u64) as usize;

            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let v = (rng_state % *size as u64) as usize;

            if u < v {
                edges.push((u, v));
            }
        }

        let graph = Graph::from_edges(&edges, true); // directed

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(graph.topological_sort())
            });
        });
    }

    group.finish();
}

// Community Detection Benchmarks
fn bench_label_propagation(c: &mut Criterion) {
    let mut group = c.benchmark_group("label_propagation");

    for size in [100, 500, 1000, 5000].iter() {
        let edges = generate_random_edges(*size, size * 5, 12345);
        let graph = Graph::from_edges(&edges, false);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(graph.label_propagation(10, Some(42)))
            });
        });
    }

    group.finish();
}

fn bench_louvain(c: &mut Criterion) {
    let mut group = c.benchmark_group("louvain");

    // Louvain is more expensive, use smaller sizes
    for size in [100, 500, 1000].iter() {
        let edges = generate_random_edges(*size, size * 5, 12345);
        let graph = Graph::from_edges(&edges, false);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(graph.louvain())
            });
        });
    }

    group.finish();
}

// Link Prediction Benchmarks
fn bench_common_neighbors(c: &mut Criterion) {
    let mut group = c.benchmark_group("common_neighbors");

    for avg_degree in [10, 50, 100].iter() {
        let size = 1000;
        let edges = generate_random_edges(size, size * avg_degree, 12345);
        let graph = Graph::from_edges(&edges, false);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("degree_{}", avg_degree)),
            avg_degree,
            |b, _| {
                b.iter(|| {
                    black_box(graph.common_neighbors(0, size / 2))
                });
            },
        );
    }

    group.finish();
}

fn bench_adamic_adar_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("adamic_adar_index");

    for avg_degree in [10, 50, 100].iter() {
        let size = 1000;
        let edges = generate_random_edges(size, size * avg_degree, 12345);
        let graph = Graph::from_edges(&edges, false);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("degree_{}", avg_degree)),
            avg_degree,
            |b, _| {
                b.iter(|| {
                    black_box(graph.adamic_adar_index(0, size / 2))
                });
            },
        );
    }

    group.finish();
}

// Centrality Benchmarks
fn bench_degree_centrality(c: &mut Criterion) {
    let mut group = c.benchmark_group("degree_centrality");

    for size in [100, 500, 1000, 5000].iter() {
        let edges = generate_random_edges(*size, size * 5, 12345);
        let graph = Graph::from_edges(&edges, false);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(graph.degree_centrality())
            });
        });
    }

    group.finish();
}

fn bench_betweenness_centrality(c: &mut Criterion) {
    let mut group = c.benchmark_group("betweenness_centrality");

    // Betweenness is expensive, use smaller sizes
    for size in [50, 100, 200].iter() {
        let edges = generate_random_edges(*size, size * 3, 12345);
        let graph = Graph::from_edges(&edges, false);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(graph.betweenness_centrality())
            });
        });
    }

    group.finish();
}

fn bench_pagerank(c: &mut Criterion) {
    let mut group = c.benchmark_group("pagerank");

    for size in [100, 500, 1000].iter() {
        let edges = generate_random_edges(*size, size * 5, 12345);
        let graph = Graph::from_edges(&edges, true); // directed for PageRank

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(graph.pagerank(0.85, 100, 1e-6))
            });
        });
    }

    group.finish();
}

// Structural Analysis Benchmarks
fn bench_clustering_coefficient(c: &mut Criterion) {
    let mut group = c.benchmark_group("clustering_coefficient");

    for size in [100, 500, 1000].iter() {
        let edges = generate_random_edges(*size, size * 5, 12345);
        let graph = Graph::from_edges(&edges, false);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(graph.clustering_coefficient())
            });
        });
    }

    group.finish();
}

fn bench_diameter(c: &mut Criterion) {
    let mut group = c.benchmark_group("diameter");

    // Diameter is expensive, use smaller sizes
    for size in [50, 100, 200].iter() {
        let edges = generate_random_edges(*size, size * 3, 12345);
        let graph = Graph::from_edges(&edges, false);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(graph.diameter())
            });
        });
    }

    group.finish();
}

criterion_group!(
    pathfinding,
    bench_shortest_path,
    bench_dijkstra,
    bench_a_star,
    bench_all_pairs_shortest_paths
);

criterion_group!(
    traversal,
    bench_dfs,
    bench_topological_sort
);

criterion_group!(
    components,
    bench_connected_components,
    bench_strongly_connected_components
);

criterion_group!(
    community,
    bench_label_propagation,
    bench_louvain
);

criterion_group!(
    link_prediction,
    bench_common_neighbors,
    bench_adamic_adar_index
);

criterion_group!(
    centrality,
    bench_degree_centrality,
    bench_betweenness_centrality,
    bench_pagerank
);

criterion_group!(
    structural,
    bench_clustering_coefficient,
    bench_diameter
);

criterion_main!(
    pathfinding,
    traversal,
    components,
    community,
    link_prediction,
    centrality,
    structural
);
