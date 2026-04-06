//! Generic 2D tiled reduction helper
//!
//! Shared implementation for tiled sum/max/min reductions on GPU.

use super::super::GpuDevice;

impl GpuDevice {
    /// Generic 2D tiled reduction helper
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn tiled_reduce_2d_async<F>(
        &self,
        data: &[f32],
        width: usize,
        height: usize,
        shader_source: &str,
        op_name: &str,
        identity: f32,
        combine: F,
    ) -> Result<f32, String>
    where
        F: Fn(&[f32]) -> f32,
    {
        if data.is_empty() || width == 0 || height == 0 {
            return Ok(identity);
        }

        // Calculate workgroup dimensions (16x16 tiles)
        let workgroup_size_x: u32 = 16;
        let workgroup_size_y: u32 = 16;
        let num_workgroups_x = (width as u32).div_ceil(workgroup_size_x);
        let num_workgroups_y = (height as u32).div_ceil(workgroup_size_y);
        let total_workgroups = (num_workgroups_x * num_workgroups_y) as usize;

        // Create shader module
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{} Shader", op_name)),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create input buffer
        let input_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} Input", op_name)),
            size: std::mem::size_of_val(data) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create partial results buffer
        let partial_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} Partial Results", op_name)),
            size: (total_workgroups * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Dimensions uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Dimensions {
            width: u32,
            height: u32,
        }

        let dims = Dimensions { width: width as u32, height: height as u32 };

        let dims_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} Dimensions", op_name)),
            size: std::mem::size_of::<Dimensions>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write data
        self.queue.write_buffer(&input_buffer, 0, bytemuck::cast_slice(data));
        self.queue.write_buffer(&dims_buffer, 0, bytemuck::bytes_of(&dims));

        // Create bind group layout
        let bind_group_layout =
            self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("{} Bind Group Layout", op_name)),
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
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
            label: Some(&format!("{} Bind Group", op_name)),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: partial_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dims_buffer.as_entire_binding() },
            ],
        });

        // Create pipeline
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{} Pipeline Layout", op_name)),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{} Pipeline", op_name)),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create staging buffer
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} Staging", op_name)),
            size: (total_workgroups * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("{} Encoder", op_name)),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{} Pass", op_name)),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
        }

        // Copy result to staging buffer
        encoder.copy_buffer_to_buffer(
            &partial_buffer,
            0,
            &staging_buffer,
            0,
            (total_workgroups * std::mem::size_of::<f32>()) as u64,
        );

        // Submit commands
        self.queue.submit(Some(encoder.finish()));

        // Read back results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        // Poll device
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();

        receiver
            .receive()
            .await
            .ok_or("Failed to receive mapping result")?
            .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;

        let final_result = {
            let data = buffer_slice.get_mapped_range();
            let partials: &[f32] = bytemuck::cast_slice(&data);
            combine(partials)
        };

        staging_buffer.unmap();

        Ok(final_result)
    }
}
