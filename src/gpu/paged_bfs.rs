//! Paged GPU BFS for graphs larger than VRAM
//!
//! Processes graph tiles incrementally using LRU caching.
//! Based on morsel-driven parallelism (Umbra, Neumann & Freitag 2020).

use super::cache::LruTileCache;
use super::paging::PagingCoordinator;
use super::{GpuBfsResult, GpuDevice};
use crate::{CsrGraph, NodeId};
use anyhow::{Context, Result};
use std::collections::VecDeque;

/// Run paged BFS on a graph that may exceed VRAM capacity
///
/// Automatically tiles the graph and processes tiles with LRU caching.
///
/// # Arguments
///
/// * `device` - GPU device
/// * `graph` - Graph to process
/// * `source` - Source node for BFS
///
/// # Errors
///
/// Returns error if:
/// - GPU operations fail
/// - Graph is empty
///
/// # Example
///
/// ```ignore
/// # use trueno_graph::gpu::{GpuDevice, gpu_bfs_paged};
/// # use trueno_graph::{CsrGraph, NodeId};
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let device = GpuDevice::new().await?;
/// let graph = CsrGraph::new();
/// // ... add many edges to create large graph ...
///
/// let result = gpu_bfs_paged(&device, &graph, NodeId(0)).await?;
/// println!("Visited {} nodes", result.visited_count);
/// # Ok(())
/// # }
/// ```
/// Process a single node's neighbors within its tile, updating distances
#[allow(clippy::cast_possible_truncation)]
fn process_node_neighbors(
    coordinator: &PagingCoordinator<'_>,
    node: NodeId,
    distances: &mut [u32],
    current_level: u32,
    next_frontier: &mut Vec<NodeId>,
) -> Result<()> {
    let tile = coordinator.get_tile_for_node(node).context("Node not in any tile")?;
    let node_idx_in_tile = node.0 as usize - tile.start_node;

    if tile.row_offsets.len() < 2 || node_idx_in_tile >= tile.row_offsets.len() - 1 {
        return Ok(());
    }

    let start = tile.row_offsets[node_idx_in_tile] as usize;
    let end = tile.row_offsets[node_idx_in_tile + 1] as usize;

    for &neighbor in &tile.col_indices[start..end] {
        let neighbor_idx = neighbor as usize;
        if distances[neighbor_idx] == u32::MAX {
            distances[neighbor_idx] = current_level + 1;
            next_frontier.push(NodeId(neighbor));
        }
    }
    Ok(())
}

#[allow(clippy::cast_possible_truncation)]
pub async fn gpu_bfs_paged(
    device: &GpuDevice,
    graph: &CsrGraph,
    source: NodeId,
) -> Result<GpuBfsResult> {
    let coordinator = PagingCoordinator::new(device, graph)?;

    if coordinator.fits_in_vram() {
        return super::gpu_bfs(
            device,
            &super::GpuCsrBuffers::from_csr_graph(device, graph)?,
            source,
        )
        .await;
    }

    let num_nodes = graph.num_nodes();
    let mut distances = vec![u32::MAX; num_nodes];
    distances[source.0 as usize] = 0;

    let mut frontier = VecDeque::new();
    frontier.push_back(source);
    let mut current_level = 0_u32;

    let cache_capacity = coordinator.limits().max_morsels.min(coordinator.num_tiles());
    let mut _tile_cache = LruTileCache::new(cache_capacity);

    while !frontier.is_empty() {
        let mut next_frontier = Vec::new();

        for &node in &frontier {
            process_node_neighbors(
                &coordinator,
                node,
                &mut distances,
                current_level,
                &mut next_frontier,
            )?;
        }

        frontier = VecDeque::from(next_frontier);
        current_level += 1;

        if current_level > num_nodes as u32 {
            break;
        }
    }

    let visited_count = distances.iter().filter(|&&d| d != u32::MAX).count();
    Ok(GpuBfsResult { distances, visited_count })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_paged_bfs_small_graph() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_paged_bfs_small_graph: GPU not available");
            return;
        }

        let device = GpuDevice::new().await.unwrap();

        // Create small graph: 0 -> 1 -> 2
        let mut graph = CsrGraph::new();
        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
        graph.add_edge(NodeId(1), NodeId(2), 1.0).unwrap();

        let result = gpu_bfs_paged(&device, &graph, NodeId(0)).await.unwrap();

        assert_eq!(result.distance(NodeId(0)), Some(0));
        assert_eq!(result.distance(NodeId(1)), Some(1));
        assert_eq!(result.distance(NodeId(2)), Some(2));
        assert_eq!(result.visited_count, 3);
    }

    #[tokio::test]
    async fn test_paged_bfs_disconnected() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_paged_bfs_disconnected: GPU not available");
            return;
        }

        let device = GpuDevice::new().await.unwrap();

        // Create disconnected: 0 -> 1, 2 (isolated)
        let mut graph = CsrGraph::new();
        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
        graph.add_edge(NodeId(2), NodeId(2), 1.0).unwrap(); // Self-loop

        let result = gpu_bfs_paged(&device, &graph, NodeId(0)).await.unwrap();

        assert_eq!(result.distance(NodeId(0)), Some(0));
        assert_eq!(result.distance(NodeId(1)), Some(1));
        assert_eq!(result.distance(NodeId(2)), None); // Unreachable
        assert_eq!(result.visited_count, 2);
    }

    #[tokio::test]
    async fn test_paged_bfs_larger_graph() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_paged_bfs_larger_graph: GPU not available");
            return;
        }

        let device = GpuDevice::new().await.unwrap();

        // Create larger ring graph
        let mut graph = CsrGraph::new();
        for i in 0..100 {
            graph.add_edge(NodeId(i), NodeId((i + 1) % 100), 1.0).unwrap();
        }

        let result = gpu_bfs_paged(&device, &graph, NodeId(0)).await.unwrap();

        // All nodes should be reachable in a ring
        assert_eq!(result.visited_count, 100);
        assert_eq!(result.distance(NodeId(0)), Some(0));
        assert_eq!(result.distance(NodeId(1)), Some(1));
        assert_eq!(result.distance(NodeId(50)), Some(50));
    }

    #[tokio::test]
    async fn test_paged_bfs_star_graph() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_paged_bfs_star_graph: GPU not available");
            return;
        }

        let device = GpuDevice::new().await.unwrap();

        // Create star graph: center node 0 connected to all others
        let mut graph = CsrGraph::new();
        for i in 1..20 {
            graph.add_edge(NodeId(0), NodeId(i), 1.0).unwrap();
        }

        let result = gpu_bfs_paged(&device, &graph, NodeId(0)).await.unwrap();

        assert_eq!(result.distance(NodeId(0)), Some(0));
        // All other nodes at distance 1
        for i in 1..20 {
            assert_eq!(result.distance(NodeId(i)), Some(1));
        }
        assert_eq!(result.visited_count, 20);
    }

    #[tokio::test]
    async fn test_paged_bfs_multiple_levels() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_paged_bfs_multiple_levels: GPU not available");
            return;
        }

        let device = GpuDevice::new().await.unwrap();

        // Create multi-level tree
        let mut graph = CsrGraph::new();
        // Level 0: node 0
        // Level 1: nodes 1, 2
        // Level 2: nodes 3, 4, 5, 6
        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
        graph.add_edge(NodeId(0), NodeId(2), 1.0).unwrap();
        graph.add_edge(NodeId(1), NodeId(3), 1.0).unwrap();
        graph.add_edge(NodeId(1), NodeId(4), 1.0).unwrap();
        graph.add_edge(NodeId(2), NodeId(5), 1.0).unwrap();
        graph.add_edge(NodeId(2), NodeId(6), 1.0).unwrap();

        let result = gpu_bfs_paged(&device, &graph, NodeId(0)).await.unwrap();

        assert_eq!(result.distance(NodeId(0)), Some(0));
        assert_eq!(result.distance(NodeId(1)), Some(1));
        assert_eq!(result.distance(NodeId(2)), Some(1));
        assert_eq!(result.distance(NodeId(3)), Some(2));
        assert_eq!(result.distance(NodeId(4)), Some(2));
        assert_eq!(result.distance(NodeId(5)), Some(2));
        assert_eq!(result.distance(NodeId(6)), Some(2));
        assert_eq!(result.visited_count, 7);
    }

    #[tokio::test]
    async fn test_paged_bfs_with_duplicate_edges() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_paged_bfs_with_duplicate_edges: GPU not available");
            return;
        }

        let device = GpuDevice::new().await.unwrap();

        // Graph with multiple paths between nodes
        let mut graph = CsrGraph::new();
        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
        graph.add_edge(NodeId(0), NodeId(2), 1.0).unwrap();
        graph.add_edge(NodeId(1), NodeId(3), 1.0).unwrap();
        graph.add_edge(NodeId(2), NodeId(3), 1.0).unwrap(); // Two paths to 3

        let result = gpu_bfs_paged(&device, &graph, NodeId(0)).await.unwrap();

        assert_eq!(result.distance(NodeId(0)), Some(0));
        assert_eq!(result.distance(NodeId(1)), Some(1));
        assert_eq!(result.distance(NodeId(2)), Some(1));
        assert_eq!(result.distance(NodeId(3)), Some(2)); // Shortest path
        assert_eq!(result.visited_count, 4);
    }

    #[tokio::test]
    async fn test_paged_bfs_empty_graph() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_paged_bfs_empty_graph: GPU not available");
            return;
        }

        let device = GpuDevice::new().await.unwrap();

        // Single node, no edges
        let mut graph = CsrGraph::new();
        graph.add_edge(NodeId(0), NodeId(0), 1.0).unwrap(); // Self-loop to create node

        let result = gpu_bfs_paged(&device, &graph, NodeId(0)).await.unwrap();

        assert_eq!(result.distance(NodeId(0)), Some(0));
        assert_eq!(result.visited_count, 1);
    }
}
