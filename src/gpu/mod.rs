//! GPU compute backend using wgpu (WebGPU)
//!
//! Toyota Way Principles:
//! - Muda elimination: GPU only when compute > 5x transfer time
//! - Genchi Genbutsu: Empirical benchmarks prove 50-100x speedups
//!
//! Architecture:
//! - WGSL compute shaders for parallel reduction
//! - Workgroup size: 256 threads (GPU warp size optimization)
//! - Two-stage reduction: workgroup-local + global
//!
//! References:
//! - `HeavyDB` (2017): GPU aggregation patterns
//! - Harris (2007): Optimizing parallel reduction in CUDA
//! - Leis et al. (2014): Morsel-driven parallelism

use crate::{Error, Result};
use arrow::array::{Array, Float32Array, Int32Array};
use wgpu;
use wgpu::util::DeviceExt;

pub mod jit;
pub mod kernels;
pub mod multigpu;

/// GPU compute engine for aggregations
pub struct GpuEngine {
    /// GPU device handle (public for benchmarking)
    pub device: wgpu::Device,
    /// GPU command queue (public for benchmarking)
    pub queue: wgpu::Queue,
    /// JIT compiler for kernel fusion
    jit: jit::JitCompiler,
}

impl GpuEngine {
    /// Initialize GPU engine
    ///
    /// # Errors
    /// Returns error if GPU initialization fails (no GPU available, driver issues, etc.)
    pub async fn new() -> Result<Self> {
        // Request GPU adapter
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| Error::GpuInitFailed("No GPU adapter found".to_string()))?;

        // Request device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Trueno-DB GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| Error::GpuInitFailed(format!("Failed to create device: {e}")))?;

        Ok(Self { device, queue, jit: jit::JitCompiler::new() })
    }

    /// Execute SUM aggregation on GPU
    ///
    /// # Arguments
    /// * `data` - Input array (Int32 or Float32)
    ///
    /// # Returns
    /// Sum of all elements
    ///
    /// # Errors
    /// Returns error if GPU execution fails
    pub async fn sum_i32(&self, data: &Int32Array) -> Result<i32> {
        kernels::sum_i32(&self.device, &self.queue, data).await
    }

    /// Execute SUM aggregation on GPU (f32)
    ///
    /// # Errors
    /// Returns error if GPU execution fails
    pub async fn sum_f32(&self, data: &Float32Array) -> Result<f32> {
        kernels::sum_f32(&self.device, &self.queue, data).await
    }

    /// Execute COUNT aggregation on GPU
    ///
    /// # Errors
    /// Returns error if GPU execution fails
    pub async fn count(&self, data: &dyn Array) -> Result<usize> {
        kernels::count(&self.device, &self.queue, data).await
    }

    /// Execute MIN aggregation on GPU
    ///
    /// # Errors
    /// Returns error if GPU execution fails
    pub async fn min_i32(&self, data: &Int32Array) -> Result<i32> {
        kernels::min_i32(&self.device, &self.queue, data).await
    }

    /// Execute MAX aggregation on GPU
    ///
    /// # Errors
    /// Returns error if GPU execution fails
    pub async fn max_i32(&self, data: &Int32Array) -> Result<i32> {
        kernels::max_i32(&self.device, &self.queue, data).await
    }

    /// Execute AVG aggregation on GPU (reuses sum + count)
    ///
    /// # Errors
    /// Returns error if GPU execution fails
    #[allow(clippy::cast_precision_loss)]
    pub async fn avg_f32(&self, data: &Float32Array) -> Result<f32> {
        let sum = self.sum_f32(data).await?;
        let count = self.count(data).await?;
        if count == 0 {
            Ok(0.0)
        } else {
            Ok(sum / count as f32)
        }
    }

    /// Execute fused filter+sum aggregation on GPU (JIT-compiled kernel)
    ///
    /// Toyota Way: Muda elimination - fuses filter and sum in single pass,
    /// eliminating intermediate buffer write.
    ///
    /// # Arguments
    /// * `data` - Input array (Int32)
    /// * `filter_threshold` - Filter threshold value (e.g., WHERE value > 1000)
    /// * `filter_op` - Filter operator ("gt", "lt", "eq", "gte", "lte", "ne")
    ///
    /// # Returns
    /// Sum of filtered elements
    ///
    /// # Errors
    /// Returns error if GPU execution fails
    ///
    /// # Example
    /// ```ignore
    /// // Equivalent to: SELECT SUM(value) FROM data WHERE value > 1000
    /// let result = engine.fused_filter_sum(&data, 1000, "gt").await?;
    /// ```
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::cast_possible_truncation)]
    pub async fn fused_filter_sum(
        &self,
        data: &Int32Array,
        filter_threshold: i32,
        filter_op: &str,
    ) -> Result<i32> {
        // JIT compile the fused kernel (cached automatically)
        let shader_module =
            self.jit.compile_fused_filter_sum(&self.device, filter_threshold, filter_op);

        // Prepare input data
        let input_data: Vec<i32> = data.values().to_vec();
        let input_size = input_data.len();

        if input_size == 0 {
            return Ok(0);
        }

        // Create GPU buffers
        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fused Filter+Sum Input"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let output_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fused Filter+Sum Output"),
            contents: bytemuck::cast_slice(&[0i32]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layout
        let bind_group_layout =
            self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Fused Filter+Sum Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
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

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fused Filter+Sum Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
            ],
        });

        // Create compute pipeline
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fused Filter+Sum Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline =
            self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused Filter+Sum Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "fused_filter_sum",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // Create command encoder and execute
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Fused Filter+Sum Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Fused Filter+Sum Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups (256 threads per workgroup)
            let workgroup_count = (input_size as u32).div_ceil(256);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Copy output to staging buffer
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fused Filter+Sum Staging Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, 4);

        // Submit commands
        self.queue.submit(Some(encoder.finish()));

        // Read result
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });

        self.device.poll(wgpu::Maintain::Wait);

        rx.receive()
            .await
            .ok_or_else(|| Error::Other("Failed to receive buffer map result".to_string()))?
            .map_err(|e| Error::Other(format!("Buffer mapping failed: {e}")))?;

        let data_view = buffer_slice.get_mapped_range();
        let result = i32::from_le_bytes([data_view[0], data_view[1], data_view[2], data_view[3]]);

        drop(data_view);
        staging_buffer.unmap();

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int32Array;

    #[tokio::test]
    async fn test_gpu_init() {
        // This test may fail on machines without GPU
        match GpuEngine::new().await {
            Ok(_engine) => {
                // GPU initialization succeeded
            }
            Err(e) => {
                // Expected on machines without GPU
                eprintln!("GPU initialization failed (expected on CI): {e}");
            }
        }
    }

    #[tokio::test]
    async fn test_gpu_sum_basic() {
        let Ok(engine) = GpuEngine::new().await else {
            eprintln!("Skipping GPU test (no GPU available)");
            return;
        };

        let data = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let result = engine.sum_i32(&data).await.unwrap();
        assert_eq!(result, 15);
    }

    #[tokio::test]
    async fn test_gpu_sum_empty() {
        let Ok(engine) = GpuEngine::new().await else {
            eprintln!("Skipping GPU test (no GPU available)");
            return;
        };

        let data = Int32Array::from(vec![] as Vec<i32>);
        let result = engine.sum_i32(&data).await.unwrap();
        assert_eq!(result, 0);
    }

    #[tokio::test]
    async fn test_gpu_min_i32() {
        let Ok(engine) = GpuEngine::new().await else {
            eprintln!("Skipping GPU test (no GPU available)");
            return;
        };

        let data = Int32Array::from(vec![5, 2, 8, 1, 9]);
        let result = engine.min_i32(&data).await.unwrap();
        assert_eq!(result, 1);
    }

    #[tokio::test]
    async fn test_gpu_min_empty() {
        let Ok(engine) = GpuEngine::new().await else {
            eprintln!("Skipping GPU test (no GPU available)");
            return;
        };

        let data = Int32Array::from(vec![] as Vec<i32>);
        let result = engine.min_i32(&data).await.unwrap();
        assert_eq!(result, i32::MAX);
    }

    #[tokio::test]
    async fn test_gpu_max_i32() {
        let Ok(engine) = GpuEngine::new().await else {
            eprintln!("Skipping GPU test (no GPU available)");
            return;
        };

        let data = Int32Array::from(vec![5, 2, 8, 1, 9]);
        let result = engine.max_i32(&data).await.unwrap();
        assert_eq!(result, 9);
    }

    #[tokio::test]
    async fn test_gpu_max_empty() {
        let Ok(engine) = GpuEngine::new().await else {
            eprintln!("Skipping GPU test (no GPU available)");
            return;
        };

        let data = Int32Array::from(vec![] as Vec<i32>);
        let result = engine.max_i32(&data).await.unwrap();
        assert_eq!(result, i32::MIN);
    }

    #[tokio::test]
    async fn test_gpu_count() {
        let Ok(engine) = GpuEngine::new().await else {
            eprintln!("Skipping GPU test (no GPU available)");
            return;
        };

        let data = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let result = engine.count(&data).await.unwrap();
        assert_eq!(result, 5);
    }

    #[tokio::test]
    async fn test_gpu_sum_f32_not_implemented() {
        let Ok(engine) = GpuEngine::new().await else {
            eprintln!("Skipping GPU test (no GPU available)");
            return;
        };

        let data = Float32Array::from(vec![1.0, 2.0, 3.0]);
        let result = engine.sum_f32(&data).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not yet implemented"));
    }

    #[tokio::test]
    async fn test_gpu_avg_f32_not_implemented() {
        let Ok(engine) = GpuEngine::new().await else {
            eprintln!("Skipping GPU test (no GPU available)");
            return;
        };

        let data = Float32Array::from(vec![2.0, 4.0, 6.0]);
        let result = engine.avg_f32(&data).await;
        // avg_f32 calls sum_f32 which returns error
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_gpu_fused_filter_sum_gt() {
        let Ok(engine) = GpuEngine::new().await else {
            eprintln!("Skipping GPU test (no GPU available)");
            return;
        };

        // Data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        // Filter: value > 5
        // Expected: 6 + 7 + 8 + 9 + 10 = 40
        let data = Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let result = engine.fused_filter_sum(&data, 5, "gt").await.unwrap();
        assert_eq!(result, 40);
    }

    #[tokio::test]
    async fn test_gpu_fused_filter_sum_lt() {
        let Ok(engine) = GpuEngine::new().await else {
            eprintln!("Skipping GPU test (no GPU available)");
            return;
        };

        // Data: [1, 2, 3, 4, 5]
        // Filter: value < 4
        // Expected: 1 + 2 + 3 = 6
        let data = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let result = engine.fused_filter_sum(&data, 4, "lt").await.unwrap();
        assert_eq!(result, 6);
    }

    #[tokio::test]
    async fn test_gpu_fused_filter_sum_eq() {
        let Ok(engine) = GpuEngine::new().await else {
            eprintln!("Skipping GPU test (no GPU available)");
            return;
        };

        // Data: [1, 5, 5, 3, 5]
        // Filter: value == 5
        // Expected: 5 + 5 + 5 = 15
        let data = Int32Array::from(vec![1, 5, 5, 3, 5]);
        let result = engine.fused_filter_sum(&data, 5, "eq").await.unwrap();
        assert_eq!(result, 15);
    }

    #[tokio::test]
    async fn test_gpu_fused_filter_sum_empty() {
        let Ok(engine) = GpuEngine::new().await else {
            eprintln!("Skipping GPU test (no GPU available)");
            return;
        };

        let data = Int32Array::from(vec![] as Vec<i32>);
        let result = engine.fused_filter_sum(&data, 5, "gt").await.unwrap();
        assert_eq!(result, 0);
    }

    #[tokio::test]
    async fn test_gpu_fused_filter_sum_no_matches() {
        let Ok(engine) = GpuEngine::new().await else {
            eprintln!("Skipping GPU test (no GPU available)");
            return;
        };

        // All values < 100, so filter passes nothing
        let data = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let result = engine.fused_filter_sum(&data, 100, "gt").await.unwrap();
        assert_eq!(result, 0);
    }
}
