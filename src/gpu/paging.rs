//! Graph paging system for handling graphs larger than VRAM
//!
//! Implements morsel-driven parallelism and tile-based processing.
//! Based on Umbra (Neumann & Freitag, CIDR 2020) and Funke et al. (SIGMOD 2018).

use super::cache::TileId;
use super::memory::GpuMemoryLimits;
use super::GpuDevice;
use crate::{CsrGraph, NodeId};
use anyhow::Result;

/// Graph tile (subset of nodes and their edges)
#[derive(Debug, Clone)]
pub struct GraphTile {
    /// Tile ID
    pub id: TileId,

    /// Starting node index (inclusive)
    pub start_node: usize,

    /// Ending node index (exclusive)
    pub end_node: usize,

    /// Number of nodes in this tile
    pub num_nodes: usize,

    /// CSR row offsets for this tile
    pub row_offsets: Vec<u32>,
    /// CSR column indices (edge targets)
    pub col_indices: Vec<u32>,
    /// Edge weights
    pub edge_weights: Vec<f32>,
}

impl GraphTile {
    /// Calculate memory footprint in bytes
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        let row_offsets_size = self.row_offsets.len() * std::mem::size_of::<u32>();
        let col_indices_size = self.col_indices.len() * std::mem::size_of::<u32>();
        let weights_size = self.edge_weights.len() * std::mem::size_of::<f32>();
        row_offsets_size + col_indices_size + weights_size
    }
}

/// Paging coordinator for managing graph tiles
///
/// Splits large graphs into tiles and manages their lifecycle in GPU memory.
pub struct PagingCoordinator {
    /// GPU memory limits
    limits: GpuMemoryLimits,

    /// All tiles in the graph
    tiles: Vec<GraphTile>,

    /// Tile size in nodes
    tile_size: usize,
}

impl PagingCoordinator {
    /// Create paging coordinator for a graph
    ///
    /// # Errors
    ///
    /// Returns error if memory limits cannot be detected
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(device: &GpuDevice, graph: &CsrGraph) -> Result<Self> {
        let limits = GpuMemoryLimits::detect(device)?;

        // Calculate tile size based on VRAM limits
        let (row_offsets, col_indices, edge_weights) = graph.csr_components();
        let total_nodes = graph.num_nodes();

        // Estimate bytes per node (amortized)
        let total_graph_bytes = std::mem::size_of_val(row_offsets)
            + std::mem::size_of_val(col_indices)
            + std::mem::size_of_val(edge_weights);

        let bytes_per_node = if total_nodes > 0 {
            total_graph_bytes / total_nodes
        } else {
            1000 // Default estimate
        };

        let tile_size = limits.recommended_tile_size(bytes_per_node);

        // Split graph into tiles
        let mut tiles = Vec::new();

        for (tile_id, start_node) in (0..total_nodes).step_by(tile_size).enumerate() {
            let end_node = (start_node + tile_size).min(total_nodes);
            let num_nodes = end_node - start_node;

            // Extract CSR data for this tile
            let (tile_row_offsets, tile_col_indices, tile_edge_weights) =
                Self::extract_tile_csr(graph, start_node, end_node);

            tiles.push(GraphTile {
                id: tile_id,
                start_node,
                end_node,
                num_nodes,
                row_offsets: tile_row_offsets,
                col_indices: tile_col_indices,
                edge_weights: tile_edge_weights,
            });
        }

        Ok(Self { limits, tiles, tile_size })
    }

    /// Extract CSR data for a tile (nodes `start_node..end_node`)
    #[allow(clippy::cast_possible_truncation)]
    fn extract_tile_csr(
        graph: &CsrGraph,
        start_node: usize,
        end_node: usize,
    ) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
        let (graph_row_offsets, graph_col_indices, graph_edge_weights) = graph.csr_components();

        let mut tile_row_offsets = vec![0];
        let mut tile_col_indices = Vec::new();
        let mut tile_edge_weights = Vec::new();

        for node_idx in start_node..end_node {
            let start = graph_row_offsets[node_idx] as usize;
            let end = graph_row_offsets[node_idx + 1] as usize;

            // Copy edges for this node
            tile_col_indices.extend_from_slice(&graph_col_indices[start..end]);
            tile_edge_weights.extend_from_slice(&graph_edge_weights[start..end]);

            // Update row offset
            tile_row_offsets.push(tile_col_indices.len() as u32);
        }

        (tile_row_offsets, tile_col_indices, tile_edge_weights)
    }

    /// Get tile containing a specific node
    #[must_use]
    pub fn get_tile_for_node(&self, node: NodeId) -> Option<&GraphTile> {
        let node_idx = node.0 as usize;
        self.tiles.iter().find(|tile| node_idx >= tile.start_node && node_idx < tile.end_node)
    }

    /// Get tile by ID
    #[must_use]
    pub fn get_tile(&self, tile_id: TileId) -> Option<&GraphTile> {
        self.tiles.get(tile_id)
    }

    /// Get total number of tiles
    #[must_use]
    pub fn num_tiles(&self) -> usize {
        self.tiles.len()
    }

    /// Check if graph fits entirely in VRAM (no paging needed)
    #[must_use]
    pub fn fits_in_vram(&self) -> bool {
        let total_bytes: usize = self.tiles.iter().map(GraphTile::size_bytes).sum();
        self.limits.fits_in_vram(total_bytes)
    }

    /// Get tile size in nodes
    #[must_use]
    pub fn tile_size(&self) -> usize {
        self.tile_size
    }

    /// Get memory limits
    #[must_use]
    pub const fn limits(&self) -> &GpuMemoryLimits {
        &self.limits
    }

    /// Get iterator over all tiles
    pub fn tiles(&self) -> impl Iterator<Item = &GraphTile> {
        self.tiles.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_paging_coordinator_small_graph() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_paging_coordinator_small_graph: GPU not available");
            return;
        }

        let device = GpuDevice::new().await.unwrap();

        // Create small graph: 0 -> 1 -> 2
        let mut graph = CsrGraph::new();
        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
        graph.add_edge(NodeId(1), NodeId(2), 1.0).unwrap();

        let coordinator = PagingCoordinator::new(&device, &graph).unwrap();

        assert!(coordinator.num_tiles() >= 1);
        assert!(coordinator.fits_in_vram()); // Small graph should fit

        // Check tile for node 0
        let tile = coordinator.get_tile_for_node(NodeId(0)).unwrap();
        assert!(tile.start_node <= 0);
        assert!(tile.end_node > 0);
    }

    #[tokio::test]
    async fn test_paging_coordinator_tiling() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_paging_coordinator_tiling: GPU not available");
            return;
        }

        let device = GpuDevice::new().await.unwrap();

        // Create larger graph
        let mut graph = CsrGraph::new();
        for i in 0..100 {
            graph.add_edge(NodeId(i), NodeId((i + 1) % 100), 1.0).unwrap();
        }

        let coordinator = PagingCoordinator::new(&device, &graph).unwrap();

        println!("Graph nodes: {}", graph.num_nodes());
        println!("Num tiles: {}", coordinator.num_tiles());
        println!("Tile size: {}", coordinator.tile_size());

        assert!(coordinator.num_tiles() >= 1);

        // Verify all nodes are covered by tiles
        for node_id in 0..100 {
            assert!(
                coordinator.get_tile_for_node(NodeId(node_id)).is_some(),
                "Node {} not in any tile",
                node_id
            );
        }
    }

    #[tokio::test]
    async fn test_paging_coordinator_get_tile() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_paging_coordinator_get_tile: GPU not available");
            return;
        }

        let device = GpuDevice::new().await.unwrap();

        let mut graph = CsrGraph::new();
        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
        graph.add_edge(NodeId(1), NodeId(2), 1.0).unwrap();

        let coordinator = PagingCoordinator::new(&device, &graph).unwrap();

        // Get tile by ID
        let tile0 = coordinator.get_tile(0);
        assert!(tile0.is_some());

        // Invalid tile ID
        let invalid_tile = coordinator.get_tile(999);
        assert!(invalid_tile.is_none());
    }

    #[tokio::test]
    async fn test_paging_coordinator_tiles_iterator() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_paging_coordinator_tiles_iterator: GPU not available");
            return;
        }

        let device = GpuDevice::new().await.unwrap();

        let mut graph = CsrGraph::new();
        for i in 0..50 {
            graph.add_edge(NodeId(i), NodeId(i + 1), 1.0).unwrap();
        }

        let coordinator = PagingCoordinator::new(&device, &graph).unwrap();

        // Count tiles using iterator
        let tile_count = coordinator.tiles().count();
        assert_eq!(tile_count, coordinator.num_tiles());

        // Verify all tiles have valid ranges
        for tile in coordinator.tiles() {
            assert!(tile.start_node < tile.end_node);
            assert_eq!(tile.num_nodes, tile.end_node - tile.start_node);
        }
    }

    #[tokio::test]
    async fn test_paging_coordinator_limits() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_paging_coordinator_limits: GPU not available");
            return;
        }

        let device = GpuDevice::new().await.unwrap();

        let mut graph = CsrGraph::new();
        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();

        let coordinator = PagingCoordinator::new(&device, &graph).unwrap();

        let limits = coordinator.limits();
        assert!(limits.total_vram > 0);
        assert!(limits.usable_vram > 0);
        assert!(limits.max_morsels > 0);
    }

    #[test]
    fn test_graph_tile_size() {
        let tile = GraphTile {
            id: 0,
            start_node: 0,
            end_node: 100,
            num_nodes: 100,
            row_offsets: vec![0; 101],
            col_indices: vec![0; 200],
            edge_weights: vec![0.0; 200],
        };

        let size = tile.size_bytes();
        assert!(size > 0);

        // 101 u32 + 200 u32 + 200 f32 = 101*4 + 200*4 + 200*4 = 2004 bytes
        assert_eq!(size, 101 * 4 + 200 * 4 + 200 * 4);
    }

    #[tokio::test]
    async fn test_paging_coordinator_empty_graph() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_paging_coordinator_empty_graph: GPU not available");
            return;
        }

        let device = GpuDevice::new().await.unwrap();
        let graph = CsrGraph::new();

        let coordinator = PagingCoordinator::new(&device, &graph).unwrap();

        assert_eq!(coordinator.num_tiles(), 0);
        assert!(coordinator.fits_in_vram()); // Empty graph always fits
        assert!(coordinator.get_tile_for_node(NodeId(0)).is_none());
    }

    #[tokio::test]
    async fn test_paging_with_single_tile() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_paging_with_single_tile: GPU not available");
            return;
        }

        let device = GpuDevice::new().await.unwrap();

        // Small graph that fits in one tile
        let mut graph = CsrGraph::new();
        for i in 0..5 {
            graph.add_edge(NodeId(i), NodeId(i + 1), 1.0).unwrap();
        }

        let coordinator = PagingCoordinator::new(&device, &graph).unwrap();

        assert_eq!(coordinator.num_tiles(), 1);
        let tile = coordinator.get_tile(0).unwrap();
        assert_eq!(tile.id, 0);
        assert_eq!(tile.num_nodes, 6);
    }
}
