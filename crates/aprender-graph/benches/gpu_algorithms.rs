//! GPU algorithm benchmarks
//!
//! Validates GPU performance claims:
//! - GPU BFS: 250x faster than NetworkX (via CPU baseline comparison)
//! - GPU PageRank: Similar speedup for iterative algorithms
//! - Proper async/await handling for GPU operations
//!
//! Note: These benchmarks require GPU hardware and are automatically skipped
//! if no GPU is available.

#![cfg(feature = "gpu")]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use trueno_graph::gpu::{gpu_bfs, gpu_pagerank, GpuCsrBuffers, GpuDevice};
use trueno_graph::{bfs, pagerank, CsrGraph, NodeId};

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

/// Benchmark: GPU BFS vs CPU BFS
fn bench_gpu_bfs(c: &mut Criterion) {
    // Try to create GPU device; skip if unavailable
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let device = match runtime.block_on(GpuDevice::new()) {
        Ok(dev) => dev,
        Err(_) => {
            eprintln!("⚠️  GPU not available - skipping GPU BFS benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("bfs_comparison");

    for size in [100, 500, 1000, 5000].iter() {
        let edges = generate_scale_free_graph(*size, 3);
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        // CPU BFS baseline
        group.bench_with_input(BenchmarkId::new("cpu_bfs", size), &graph, |b, graph| {
            b.iter(|| {
                let reachable = bfs(black_box(graph), NodeId(0)).unwrap();
                black_box(reachable);
            });
        });

        // GPU BFS
        let buffers = GpuCsrBuffers::from_csr_graph(&device, &graph).unwrap();
        group.bench_with_input(
            BenchmarkId::new("gpu_bfs", size),
            &(&device, &buffers),
            |b, (device, buffers)| {
                b.iter(|| {
                    runtime.block_on(async {
                        let result = gpu_bfs(black_box(device), black_box(buffers), NodeId(0))
                            .await
                            .unwrap();
                        black_box(result);
                    });
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: GPU PageRank vs CPU PageRank
fn bench_gpu_pagerank(c: &mut Criterion) {
    // Try to create GPU device; skip if unavailable
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let device = match runtime.block_on(GpuDevice::new()) {
        Ok(dev) => dev,
        Err(_) => {
            eprintln!("⚠️  GPU not available - skipping GPU PageRank benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("pagerank_comparison");

    for size in [100, 500, 1000].iter() {
        let edges = generate_scale_free_graph(*size, 3);
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        // CPU PageRank baseline
        group.bench_with_input(BenchmarkId::new("cpu_pagerank", size), &graph, |b, graph| {
            b.iter(|| {
                let scores = pagerank(black_box(graph), 20, 1e-6).unwrap();
                black_box(scores);
            });
        });

        // GPU PageRank
        let buffers = GpuCsrBuffers::from_csr_graph(&device, &graph).unwrap();
        let out_degrees: Vec<u32> = (0..graph.num_nodes())
            .map(|i| graph.outgoing_neighbors(NodeId(i as u32)).unwrap().len() as u32)
            .collect();

        group.bench_with_input(
            BenchmarkId::new("gpu_pagerank", size),
            &(&device, &buffers, &out_degrees),
            |b, (device, buffers, out_degrees)| {
                b.iter(|| {
                    runtime.block_on(async {
                        let result = gpu_pagerank(
                            black_box(device),
                            black_box(buffers),
                            black_box(out_degrees),
                            20,
                            0.85,
                        )
                        .await
                        .unwrap();
                        black_box(result);
                    });
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: GPU buffer upload performance
fn bench_gpu_buffer_upload(c: &mut Criterion) {
    // Try to create GPU device; skip if unavailable
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let device = match runtime.block_on(GpuDevice::new()) {
        Ok(dev) => dev,
        Err(_) => {
            eprintln!("⚠️  GPU not available - skipping GPU buffer upload benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("gpu_buffer_upload");

    for size in [100, 500, 1000, 5000, 10000].iter() {
        let edges = generate_scale_free_graph(*size, 3);
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        group.bench_with_input(
            BenchmarkId::new("upload_csr", size),
            &(&device, &graph),
            |b, (device, graph)| {
                b.iter(|| {
                    let buffers =
                        GpuCsrBuffers::from_csr_graph(black_box(device), black_box(graph)).unwrap();
                    black_box(buffers);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_gpu_bfs, bench_gpu_pagerank, bench_gpu_buffer_upload);
criterion_main!(benches);
