//! GPU matrix multiplication operations

use super::super::GpuDevice;
#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
use crate::backends::gpu::runtime;
use crate::backends::gpu::shaders;

impl GpuDevice {
    /// Execute matrix multiplication on GPU (sync, native only)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), String> {
        runtime::block_on(async { self.matmul_async(a, b, result, m, k, n).await })
    }

    /// Execute matrix multiplication on GPU (async, works on all platforms)
    pub async fn matmul_async(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), String> {
        contract_pre_matmul!();
        // Guard: if B exceeds max buffer binding, chunk along N dimension.
        // Each chunk computes result[:, n_start..n_end] = A @ B[:, n_start..n_end]
        // This handles lm_head (152064 × 3584 × 4 = 2.18 GB > 2 GB limit).
        let max_binding = self.device.limits().max_storage_buffer_binding_size as u64;
        let b_bytes = (b.len() * 4) as u64;
        if b_bytes > max_binding {
            // Chunk B along N: each chunk has at most max_n_chunk columns
            let max_elements = max_binding as usize / 4; // max f32 elements per buffer
            let max_n_chunk = max_elements / k; // max columns per chunk
            let max_n_chunk = max_n_chunk.max(1);

            let mut n_start = 0;
            while n_start < n {
                let n_end = (n_start + max_n_chunk).min(n);
                let chunk_n = n_end - n_start;

                // Extract B chunk: B[:, n_start..n_end] from row-major B[K, N]
                let mut b_chunk = vec![0.0f32; k * chunk_n];
                for row in 0..k {
                    for col in 0..chunk_n {
                        b_chunk[row * chunk_n + col] = b[row * n + n_start + col];
                    }
                }

                // Compute C_chunk = A @ B_chunk
                let mut c_chunk = vec![0.0f32; m * chunk_n];
                // Use recursive call — chunk fits in buffer now
                Box::pin(self.matmul_async(a, &b_chunk, &mut c_chunk, m, k, chunk_n)).await?;

                // Copy chunk into result: result[:, n_start..n_end]
                for row in 0..m {
                    for col in 0..chunk_n {
                        result[row * n + n_start + col] = c_chunk[row * chunk_n + col];
                    }
                }

                n_start = n_end;
            }
            return Ok(());
        }

        // Create shader module
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Matmul Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::MATMUL_SHADER.into()),
        });

        // Create buffers
        let a_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix A"),
            size: std::mem::size_of_val(a) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix B"),
            size: std::mem::size_of_val(b) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let c_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix C"),
            size: std::mem::size_of_val(result) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Dimensions uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Dimensions {
            m: u32,
            k: u32,
            n: u32,
            _padding: u32,
        }

        let dims = Dimensions { m: m as u32, k: k as u32, n: n as u32, _padding: 0 };

        let dims_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dimensions"),
            size: std::mem::size_of::<Dimensions>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write data to buffers
        self.queue.write_buffer(&a_buffer, 0, bytemuck::cast_slice(a));
        self.queue.write_buffer(&b_buffer, 0, bytemuck::cast_slice(b));
        self.queue.write_buffer(&dims_buffer, 0, bytemuck::bytes_of(&dims));

        // Create bind group layout
        let bind_group_layout =
            self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Matmul Bind Group Layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Matmul Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: c_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dims_buffer.as_entire_binding() },
            ],
        });

        // Create pipeline
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Matmul Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Matmul Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create staging buffer for reading results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: std::mem::size_of_val(result) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Matmul Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Matmul Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups (16x16 threads per workgroup)
            let workgroup_size_x = 16;
            let workgroup_size_y = 16;
            let num_workgroups_x = (m as u32).div_ceil(workgroup_size_x);
            let num_workgroups_y = (n as u32).div_ceil(workgroup_size_y);

            compute_pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
        }

        // Copy result to staging buffer
        encoder.copy_buffer_to_buffer(
            &c_buffer,
            0,
            &staging_buffer,
            0,
            std::mem::size_of_val(result) as u64,
        );

        // Submit commands
        self.queue.submit(Some(encoder.finish()));

        // Read back results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        // Poll device to ensure GPU work completes and callbacks are invoked
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();

        receiver
            .receive()
            .await
            .ok_or("Failed to receive mapping result")?
            .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;

        {
            let data = buffer_slice.get_mapped_range();
            result.copy_from_slice(bytemuck::cast_slice(&data));
        }

        staging_buffer.unmap();

        contract_post_matmul!(result);
        Ok(())
    }
}
