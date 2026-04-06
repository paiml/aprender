//! Demonstration of GPU memory paging for large graphs
//!
//! Shows how to process graphs larger than VRAM using tile-based processing.

use trueno_graph::gpu::{gpu_bfs_paged, GpuDevice, GpuMemoryLimits, PagingCoordinator};
use trueno_graph::{CsrGraph, NodeId};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 GPU Memory Paging Demo\n");

    // Initialize GPU device
    let device = GpuDevice::new().await?;
    println!("✅ GPU device initialized");

    // Detect memory limits
    let limits = GpuMemoryLimits::detect(&device)?;
    println!("\n📊 GPU Memory Limits:");
    println!("   Total VRAM: {:.2} GB", limits.total_vram as f64 / 1_073_741_824.0);
    println!("   Usable VRAM: {:.2} GB", limits.usable_vram as f64 / 1_073_741_824.0);
    println!("   Morsel size: {} MB", limits.morsel_size / 1_048_576);
    println!("   Max morsels: {}", limits.max_morsels);

    // Create a large graph (ring of 10,000 nodes)
    println!("\n🔧 Creating large graph (10,000 nodes)...");
    let mut graph = CsrGraph::new();
    for i in 0..10_000 {
        graph.add_edge(NodeId(i), NodeId((i + 1) % 10_000), 1.0)?;
        if i % 1000 == 0 && i > 0 {
            print!(".");
            std::io::Write::flush(&mut std::io::stdout())?;
        }
    }
    println!(" Done!");
    println!("   Nodes: {}", graph.num_nodes());

    // Create paging coordinator
    println!("\n📦 Setting up paging coordinator...");
    let coordinator = PagingCoordinator::new(&device, &graph)?;
    println!("   Number of tiles: {}", coordinator.num_tiles());
    println!("   Tile size: {} nodes", coordinator.tile_size());
    println!("   Fits in VRAM: {}", coordinator.fits_in_vram());

    // Run paged BFS
    println!("\n🏃 Running paged BFS from node 0...");
    let result = gpu_bfs_paged(&device, &graph, NodeId(0)).await?;

    println!("✅ BFS Complete!");
    println!("   Visited nodes: {}", result.visited_count);
    println!("   Distance to node 100: {:?}", result.distance(NodeId(100)));
    println!("   Distance to node 5000: {:?}", result.distance(NodeId(5000)));
    println!("   Distance to node 9999: {:?}", result.distance(NodeId(9999)));

    // Demonstrate tile information
    println!("\n📋 Tile Details (first 3 tiles):");
    for tile in coordinator.tiles().take(3) {
        println!(
            "   Tile {}: nodes {}..{} ({} nodes, {:.2} MB)",
            tile.id,
            tile.start_node,
            tile.end_node,
            tile.num_nodes,
            tile.size_bytes() as f64 / 1_048_576.0
        );
    }

    println!("\n✨ Phase 5 (GPU Memory Paging) Complete!");

    Ok(())
}
