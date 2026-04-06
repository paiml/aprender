//! GPU vector addition operations

use super::super::GpuDevice;
#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
use crate::backends::gpu::runtime;
use crate::backends::gpu::shaders;

impl GpuDevice {
    /// Execute vector addition on GPU: c = a + b (sync, native only)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn vec_add(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<(), String> {
        runtime::block_on(async { self.vec_add_async(a, b, result).await })
    }

    /// Execute vector addition on GPU: c = a + b (async, works on all platforms)
    pub async fn vec_add_async(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
    ) -> Result<(), String> {
        let len = a.len();

        // Create shader module
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vec Add Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::VEC_ADD_SHADER.into()),
        });

        // Create buffers
        let a_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vector A"),
            size: std::mem::size_of_val(a) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vector B"),
            size: std::mem::size_of_val(b) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let c_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vector C"),
            size: std::mem::size_of_val(result) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write data to buffers
        self.queue.write_buffer(&a_buffer, 0, bytemuck::cast_slice(a));
        self.queue.write_buffer(&b_buffer, 0, bytemuck::cast_slice(b));

        // Create bind group layout
        let bind_group_layout =
            self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Vec Add Bind Group Layout"),
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
                ],
            });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Vec Add Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: c_buffer.as_entire_binding() },
            ],
        });

        // Create pipeline
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Vec Add Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Vec Add Pipeline"),
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
            label: Some("Vec Add Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Vec Add Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups (256 threads per workgroup)
            let workgroup_size = 256;
            let num_workgroups = (len as u32).div_ceil(workgroup_size);

            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
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

        Ok(())
    }
}
