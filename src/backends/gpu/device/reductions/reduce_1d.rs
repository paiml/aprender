//! 1D parallel reduction operations (max, sum)
//!
//! These are internal helpers used by activation functions (softmax, log_softmax).

use super::super::super::shaders;
use super::super::GpuDevice;

impl GpuDevice {
    /// Helper: Parallel max reduction
    pub(in crate::backends::gpu::device) async fn reduce_max(
        &self,
        input: &[f32],
    ) -> Result<f32, String> {
        let len = input.len();
        let workgroup_size = 256;
        let num_workgroups = (len as u32).div_ceil(workgroup_size);

        // Create shader module
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Max Reduction Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::MAX_REDUCTION_SHADER.into()),
        });

        // Create input buffer
        let input_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Max Reduction Input"),
            size: std::mem::size_of_val(input) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Result buffer for partial maxes
        let partial_results = vec![f32::NEG_INFINITY; num_workgroups as usize];
        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Max Partial Results"),
            size: std::mem::size_of_val(partial_results.as_slice()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.queue.write_buffer(&input_buffer, 0, bytemuck::cast_slice(input));

        // Create bind group layout
        let bind_group_layout =
            self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Max Reduction Bind Group Layout"),
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

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Max Reduction Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: result_buffer.as_entire_binding() },
            ],
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Max Reduction Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Max Reduction Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Max Reduction Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Max Reduction Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        // Create staging buffer
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Max Staging Buffer"),
            size: std::mem::size_of_val(partial_results.as_slice()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &result_buffer,
            0,
            &staging_buffer,
            0,
            std::mem::size_of_val(partial_results.as_slice()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

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
            .ok_or("Channel receive failed")?
            .map_err(|e| format!("Buffer map failed: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        // Final reduction on CPU
        Ok(result.iter().copied().fold(f32::NEG_INFINITY, f32::max))
    }

    /// Helper: Parallel sum reduction
    pub(in crate::backends::gpu::device) async fn reduce_sum(
        &self,
        input: &[f32],
    ) -> Result<f32, String> {
        let len = input.len();
        let workgroup_size = 256;
        let num_workgroups = (len as u32).div_ceil(workgroup_size);

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sum Reduction Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::SUM_REDUCTION_SHADER.into()),
        });

        let input_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sum Reduction Input"),
            size: std::mem::size_of_val(input) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let partial_results = vec![0.0f32; num_workgroups as usize];
        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sum Partial Results"),
            size: std::mem::size_of_val(partial_results.as_slice()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.queue.write_buffer(&input_buffer, 0, bytemuck::cast_slice(input));

        let bind_group_layout =
            self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Sum Reduction Bind Group Layout"),
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

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sum Reduction Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: result_buffer.as_entire_binding() },
            ],
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sum Reduction Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sum Reduction Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Sum Reduction Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sum Reduction Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sum Staging Buffer"),
            size: std::mem::size_of_val(partial_results.as_slice()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &result_buffer,
            0,
            &staging_buffer,
            0,
            std::mem::size_of_val(partial_results.as_slice()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

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
            .ok_or("Channel receive failed")?
            .map_err(|e| format!("Buffer map failed: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        // Final reduction on CPU
        Ok(result.iter().sum())
    }
}
