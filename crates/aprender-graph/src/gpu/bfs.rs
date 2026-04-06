//! GPU BFS (Breadth-First Search) implementation
//!
//! Frontier-based BFS using WGSL compute shaders.
//! Based on Gunrock (Wang et al., ACM `ToPC` 2017).

use super::{GpuCsrBuffers, GpuDevice};
use crate::NodeId;
use anyhow::{Context, Result};

/// BFS parameters for GPU shader
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BfsParams {
    num_nodes: u32,
    current_level: u32,
    source_node: u32,
    _padding: u32,
}

/// GPU BFS result
#[derive(Debug, Clone)]
pub struct GpuBfsResult {
    /// Distance from source to each node (`u32::MAX` for unreachable)
    pub distances: Vec<u32>,

    /// Number of nodes visited
    pub visited_count: usize,
}

impl GpuBfsResult {
    /// Get distance to a specific node
    #[must_use]
    pub fn distance(&self, node: NodeId) -> Option<u32> {
        self.distances.get(node.0 as usize).copied().filter(|&d| d != u32::MAX)
    }

    /// Check if node is reachable from source
    #[must_use]
    pub fn is_reachable(&self, node: NodeId) -> bool {
        self.distance(node).is_some()
    }
}

/// Helper: Read single u32 from GPU buffer
async fn read_buffer_u32(device: &GpuDevice, buffer: &wgpu::Buffer) -> Result<u32> {
    let staging_buffer = device.create_buffer(
        "Staging Buffer",
        4,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    )?;

    let mut encoder =
        device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, 4);
    device.queue().submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();

    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });

    device.device().poll(wgpu::Maintain::Wait);
    rx.receive().await.context("Failed to receive map result")?.context("Buffer mapping failed")?;

    let data = buffer_slice.get_mapped_range();
    let value = u32::from_ne_bytes(data[0..4].try_into()?);
    drop(data);
    staging_buffer.unmap();

    Ok(value)
}

/// Helper: Read distances array from GPU buffer
async fn read_distances(
    device: &GpuDevice,
    distances_buffer: &wgpu::Buffer,
    num_nodes: usize,
) -> Result<Vec<u32>> {
    let size = (num_nodes * std::mem::size_of::<u32>()) as u64;
    let staging_buffer = device.create_buffer(
        "Distances Staging",
        size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    )?;

    let mut encoder =
        device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(distances_buffer, 0, &staging_buffer, 0, size);
    device.queue().submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();

    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });

    device.device().poll(wgpu::Maintain::Wait);
    rx.receive().await.context("Failed to receive map result")?.context("Buffer mapping failed")?;

    let data = buffer_slice.get_mapped_range();
    let distances: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    Ok(distances)
}

/// Run GPU BFS from source node
///
/// # Errors
///
/// Returns error if:
/// - GPU shader compilation fails
/// - Buffer creation fails
/// - Shader dispatch fails
/// - Result readback fails
///
/// # Example
///
/// ```ignore
/// # use trueno_graph::gpu::{GpuDevice, GpuCsrBuffers, gpu_bfs};
/// # use trueno_graph::{CsrGraph, NodeId};
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let device = GpuDevice::new().await?;
/// let mut graph = CsrGraph::new();
/// graph.add_edge(NodeId(0), NodeId(1), 1.0)?;
/// graph.add_edge(NodeId(1), NodeId(2), 1.0)?;
///
/// let buffers = GpuCsrBuffers::from_csr_graph(&device, &graph)?;
/// let result = gpu_bfs(&device, &buffers, NodeId(0)).await?;
///
/// assert_eq!(result.distance(NodeId(0)), Some(0));
/// assert_eq!(result.distance(NodeId(1)), Some(1));
/// assert_eq!(result.distance(NodeId(2)), Some(2));
/// # Ok(())
/// # }
/// ```
#[allow(clippy::too_many_lines)]
#[allow(clippy::cast_possible_truncation)]
pub async fn gpu_bfs(
    device: &GpuDevice,
    buffers: &GpuCsrBuffers,
    source: NodeId,
) -> Result<GpuBfsResult> {
    // Step 1: Load WGSL shader
    const SHADER: &str = include_str!("shaders/bfs_simple.wgsl");
    let shader_module = device.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("BFS Shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    // Step 2: Create bind group layout
    let bind_group_layout =
        device.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BFS Bind Group Layout"),
            entries: &[
                // @binding(0): uniform params
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(1): storage row_offsets (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(2): storage col_indices (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(3): storage distances (read_write, atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(4): storage updated (read_write, atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    // Step 3: Create compute pipeline
    let pipeline_layout = device.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("BFS Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline =
        device.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BFS Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "bfs_level",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    // Step 4: Create auxiliary buffers
    let num_nodes = buffers.num_nodes();

    // Params buffer (uniform)
    let params_buffer = device.create_buffer_init(
        "BFS Params",
        bytemuck::bytes_of(&BfsParams {
            num_nodes: num_nodes as u32,
            current_level: 0,
            source_node: source.0,
            _padding: 0,
        }),
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    )?;

    // Distances buffer (storage, atomic<u32>)
    let mut initial_distances = vec![u32::MAX; num_nodes];
    if (source.0 as usize) < num_nodes {
        initial_distances[source.0 as usize] = 0;
    }

    let distances_buffer = device.create_buffer_init(
        "BFS Distances",
        bytemuck::cast_slice(&initial_distances),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    )?;

    // Updated flag buffer (storage, atomic<u32>)
    let updated_buffer = device.create_buffer_init(
        "BFS Updated Flag",
        bytemuck::bytes_of(&0u32),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    )?;

    // Step 5: Create bind group
    let bind_group = device.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("BFS Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buffers.row_offsets.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buffers.col_indices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: distances_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: updated_buffer.as_entire_binding() },
        ],
    });

    // Step 6: BFS dispatch loop
    let workgroup_size = 256;
    let num_workgroups = (num_nodes as u32).div_ceil(workgroup_size).max(1);

    for level in 0..num_nodes {
        // Reset updated flag to 0
        device.queue().write_buffer(&updated_buffer, 0, bytemuck::bytes_of(&0u32));

        // Update params with current level
        device.queue().write_buffer(
            &params_buffer,
            0,
            bytemuck::bytes_of(&BfsParams {
                num_nodes: num_nodes as u32,
                current_level: level as u32,
                source_node: source.0,
                _padding: 0,
            }),
        );

        // Create command encoder
        let mut encoder = device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("BFS Command Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BFS Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        // Submit commands
        device.queue().submit(Some(encoder.finish()));

        // Wait for GPU to ensure correctness (future optimization: async polling)
        device.device().poll(wgpu::Maintain::Wait);

        // Read updated flag
        let updated_value = read_buffer_u32(device, &updated_buffer).await?;

        // If no updates, BFS is complete
        if updated_value == 0 {
            break;
        }
    }

    // Step 7: Read back results
    let distances = read_distances(device, &distances_buffer, num_nodes).await?;
    let visited_count = distances.iter().filter(|&&d| d != u32::MAX).count();

    Ok(GpuBfsResult { distances, visited_count })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CsrGraph;

    #[tokio::test]
    async fn test_gpu_bfs_simple_chain() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_gpu_bfs_simple_chain: GPU not available");
            return;
        }

        let device = GpuDevice::new().await.unwrap();

        // Create chain: 0 -> 1 -> 2
        let mut graph = CsrGraph::new();
        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
        graph.add_edge(NodeId(1), NodeId(2), 1.0).unwrap();

        let buffers = GpuCsrBuffers::from_csr_graph(&device, &graph).unwrap();
        let result = gpu_bfs(&device, &buffers, NodeId(0)).await.unwrap();

        // Verify BFS distances
        assert_eq!(result.distance(NodeId(0)), Some(0));
        assert_eq!(result.distance(NodeId(1)), Some(1));
        assert_eq!(result.distance(NodeId(2)), Some(2));
    }

    #[tokio::test]
    async fn test_gpu_bfs_disconnected() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_gpu_bfs_disconnected: GPU not available");
            return;
        }

        let device = GpuDevice::new().await.unwrap();

        // Create disconnected: 0 -> 1, 2 (isolated)
        let mut graph = CsrGraph::new();
        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
        graph.add_edge(NodeId(2), NodeId(2), 1.0).unwrap(); // Self-loop

        let buffers = GpuCsrBuffers::from_csr_graph(&device, &graph).unwrap();
        let result = gpu_bfs(&device, &buffers, NodeId(0)).await.unwrap();

        assert_eq!(result.distance(NodeId(0)), Some(0));
        assert!(!result.is_reachable(NodeId(2))); // Should be unreachable
    }

    #[test]
    fn test_gpu_bfs_result_api() {
        let result = GpuBfsResult { distances: vec![0, 1, u32::MAX], visited_count: 2 };

        assert_eq!(result.distance(NodeId(0)), Some(0));
        assert_eq!(result.distance(NodeId(1)), Some(1));
        assert_eq!(result.distance(NodeId(2)), None); // Unreachable

        assert!(result.is_reachable(NodeId(0)));
        assert!(result.is_reachable(NodeId(1)));
        assert!(!result.is_reachable(NodeId(2)));
    }
}
