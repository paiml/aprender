//! GPU `PageRank` implementation
//!
//! Sparse matrix-vector multiplication (`SpMV`) based `PageRank`.
//! Based on Page et al. (1999) and `GraphBLAST` (Yang et al., ACM `ToMS` 2022).

use super::{GpuCsrBuffers, GpuDevice};
use anyhow::{Context, Result};

/// `PageRank` parameters for GPU shader
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PageRankParams {
    num_nodes: u32,
    damping: f32,
    iteration: u32,
    dangling_sum: f32,
}

/// GPU `PageRank` result
#[derive(Debug, Clone)]
pub struct GpuPageRankResult {
    /// `PageRank` scores for each node
    pub scores: Vec<f32>,

    /// Number of iterations performed
    pub iterations: usize,
}

impl GpuPageRankResult {
    /// Get `PageRank` score for a specific node
    #[must_use]
    pub fn score(&self, node_id: usize) -> Option<f32> {
        self.scores.get(node_id).copied()
    }
}

/// Helper: Read scores array from GPU buffer
async fn read_scores(
    device: &GpuDevice,
    scores_buffer: &wgpu::Buffer,
    num_nodes: usize,
) -> Result<Vec<f32>> {
    let size = (num_nodes * std::mem::size_of::<f32>()) as u64;
    let staging_buffer = device.create_buffer(
        "Scores Staging",
        size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    )?;

    let mut encoder =
        device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(scores_buffer, 0, &staging_buffer, 0, size);
    device.queue().submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();

    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });

    device.device().poll(wgpu::Maintain::Wait);
    rx.receive().await.context("Failed to receive map result")?.context("Buffer mapping failed")?;

    let data = buffer_slice.get_mapped_range();
    let scores: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    Ok(scores)
}

/// Run GPU `PageRank` algorithm
///
/// # Arguments
///
/// * `device` - GPU device
/// * `buffers` - CSR graph buffers
/// * `out_degrees` - Out-degree for each node (computed from CSR)
/// * `max_iterations` - Maximum number of iterations (typically 20)
/// * `damping` - Damping factor (typically 0.85)
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
/// # use trueno_graph::gpu::{GpuDevice, GpuCsrBuffers, gpu_pagerank};
/// # use trueno_graph::CsrGraph;
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let device = GpuDevice::new().await?;
/// let mut graph = CsrGraph::new();
/// // ... add edges ...
///
/// let buffers = GpuCsrBuffers::from_csr_graph(&device, &graph)?;
/// let out_degrees: Vec<u32> = (0..graph.num_nodes())
///     .map(|i| graph.outgoing_neighbors(i).len() as u32)
///     .collect();
/// let result = gpu_pagerank(&device, &buffers, &out_degrees, 20, 0.85).await?;
///
/// println!("Node 0 score: {:?}", result.score(0));
/// # Ok(())
/// # }
/// ```
#[allow(clippy::too_many_lines)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_precision_loss)]
pub async fn gpu_pagerank(
    device: &GpuDevice,
    buffers: &GpuCsrBuffers,
    out_degrees: &[u32],
    max_iterations: usize,
    damping: f32,
) -> Result<GpuPageRankResult> {
    // Step 1: Load WGSL shader
    const SHADER: &str = include_str!("shaders/pagerank.wgsl");
    let shader_module = device.device().create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("PageRank Shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    // Step 2: Create bind group layout
    let bind_group_layout =
        device.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PageRank Bind Group Layout"),
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
                // @binding(3): storage current_scores (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(4): storage next_scores (read_write)
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
                // @binding(5): storage out_degrees (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    // Step 3: Create compute pipeline
    let pipeline_layout = device.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("PageRank Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline =
        device.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PageRank Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "pagerank_iteration",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    // Step 4: Create auxiliary buffers
    let num_nodes = buffers.num_nodes();

    // Initialize scores to 1/N
    let initial_score = 1.0 / num_nodes as f32;

    // Compute initial dangling sum (nodes with out_degree = 0)
    let dangling_count = out_degrees.iter().filter(|&&d| d == 0).count();
    let initial_dangling_sum = dangling_count as f32 * initial_score;

    // Params buffer (uniform)
    let params_buffer = device.create_buffer_init(
        "PageRank Params",
        bytemuck::bytes_of(&PageRankParams {
            num_nodes: num_nodes as u32,
            damping,
            iteration: 0,
            dangling_sum: initial_dangling_sum,
        }),
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    )?;
    let initial_scores = vec![initial_score; num_nodes];

    // Current scores buffer (storage)
    let current_scores_buffer = device.create_buffer_init(
        "PageRank Current Scores",
        bytemuck::cast_slice(&initial_scores),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    )?;

    // Next scores buffer (storage)
    let next_scores_buffer = device.create_buffer_init(
        "PageRank Next Scores",
        bytemuck::cast_slice(&initial_scores),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    )?;

    // Out-degrees buffer (storage)
    let out_degrees_buffer = device.create_buffer_init(
        "PageRank Out Degrees",
        bytemuck::cast_slice(out_degrees),
        wgpu::BufferUsages::STORAGE,
    )?;

    // Step 5: Create bind groups (need to recreate each iteration for buffer swap)
    let workgroup_size = 256;
    let num_workgroups = (num_nodes as u32).div_ceil(workgroup_size).max(1);

    // Iteration loop
    for iteration in 0..max_iterations {
        // Compute dangling sum (sum of ranks from nodes with out_degree = 0)
        let current_scores = read_scores(device, &current_scores_buffer, num_nodes).await?;
        let dangling_sum: f32 =
            (0..num_nodes).filter(|&i| out_degrees[i] == 0).map(|i| current_scores[i]).sum();

        // Update params with current iteration and dangling sum
        device.queue().write_buffer(
            &params_buffer,
            0,
            bytemuck::bytes_of(&PageRankParams {
                num_nodes: num_nodes as u32,
                damping,
                iteration: iteration as u32,
                dangling_sum,
            }),
        );

        // Create bind group for this iteration
        let bind_group = device.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PageRank Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.row_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.col_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: current_scores_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: next_scores_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: out_degrees_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder
        let mut encoder = device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("PageRank Command Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PageRank Compute Pass"),
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

        // Swap buffers: copy next_scores to current_scores
        let mut encoder = device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("PageRank Buffer Swap"),
        });

        let buffer_size = (num_nodes * std::mem::size_of::<f32>()) as u64;
        encoder.copy_buffer_to_buffer(
            &next_scores_buffer,
            0,
            &current_scores_buffer,
            0,
            buffer_size,
        );

        device.queue().submit(Some(encoder.finish()));
        device.device().poll(wgpu::Maintain::Wait);
    }

    // Step 6: Read back final results
    let scores = read_scores(device, &current_scores_buffer, num_nodes).await?;

    Ok(GpuPageRankResult { scores, iterations: max_iterations })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CsrGraph, NodeId};

    #[tokio::test]
    #[allow(clippy::cast_possible_truncation)]
    async fn test_gpu_pagerank_simple_chain() {
        if !GpuDevice::is_gpu_available().await {
            eprintln!("⚠️  Skipping test_gpu_pagerank_simple_chain: GPU not available");
            return;
        }

        let device = GpuDevice::new().await.unwrap();

        // Create chain: 0 -> 1 -> 2
        let mut graph = CsrGraph::new();
        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
        graph.add_edge(NodeId(1), NodeId(2), 1.0).unwrap();

        let buffers = GpuCsrBuffers::from_csr_graph(&device, &graph).unwrap();
        let out_degrees: Vec<u32> = (0..graph.num_nodes())
            .map(|i| graph.outgoing_neighbors(NodeId(i as u32)).unwrap().len() as u32)
            .collect();
        let result = gpu_pagerank(&device, &buffers, &out_degrees, 20, 0.85).await.unwrap();

        // Verify scores are reasonable
        let score_0 = result.score(0).unwrap();
        let score_1 = result.score(1).unwrap();
        let score_2 = result.score(2).unwrap();

        println!("GPU PageRank scores: node0={score_0}, node1={score_1}, node2={score_2}");

        // Scores should be positive and non-zero
        assert!(score_0 > 0.0, "Score 0 should be positive");
        assert!(score_1 > 0.0, "Score 1 should be positive");
        assert!(score_2 > 0.0, "Score 2 should be positive");

        // Node 2 (sink) should have highest score in chain graph
        assert!(score_2 > score_1);
        assert!(score_1 > score_0);

        // Verify scores sum to approximately 1.0 (within tolerance for GPU floating point)
        let sum = score_0 + score_1 + score_2;
        println!("Sum: {sum}");
        assert!((sum - 1.0).abs() < 0.1, "Sum should be approximately 1.0, got {sum}");
    }

    #[test]
    fn test_gpu_pagerank_result_api() {
        let result = GpuPageRankResult { scores: vec![0.1, 0.3, 0.6], iterations: 20 };

        assert_eq!(result.score(0), Some(0.1));
        assert_eq!(result.score(1), Some(0.3));
        assert_eq!(result.score(2), Some(0.6));
        assert_eq!(result.score(3), None); // Out of bounds
    }
}
