//! GPU 2D convolution operations

use super::super::GpuDevice;
#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
use crate::backends::gpu::runtime;
use crate::backends::gpu::shaders;

impl GpuDevice {
    /// Perform 2D convolution on GPU (sync, native only)
    ///
    /// # Arguments
    ///
    /// * `input` - Input image (row-major)
    /// * `kernel` - Convolution kernel (row-major)
    /// * `result` - Output buffer (row-major)
    /// * `input_rows` - Number of rows in input
    /// * `input_cols` - Number of columns in input
    /// * `kernel_rows` - Number of rows in kernel
    /// * `kernel_cols` - Number of columns in kernel
    ///
    /// Output dimensions: (input_rows - kernel_rows + 1) x (input_cols - kernel_cols + 1)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    #[allow(clippy::too_many_arguments)]
    pub fn convolve2d(
        &self,
        input: &[f32],
        kernel: &[f32],
        result: &mut [f32],
        input_rows: usize,
        input_cols: usize,
        kernel_rows: usize,
        kernel_cols: usize,
    ) -> Result<(), String> {
        runtime::block_on(async {
            self.convolve2d_async(
                input,
                kernel,
                result,
                input_rows,
                input_cols,
                kernel_rows,
                kernel_cols,
            )
            .await
        })
    }

    /// Perform 2D convolution on GPU (async, works on all platforms)
    #[allow(clippy::too_many_arguments)]
    pub async fn convolve2d_async(
        &self,
        input: &[f32],
        kernel: &[f32],
        result: &mut [f32],
        input_rows: usize,
        input_cols: usize,
        kernel_rows: usize,
        kernel_cols: usize,
    ) -> Result<(), String> {
        if kernel_rows > input_rows || kernel_cols > input_cols {
            return Err(format!(
                "Kernel size ({}x{}) larger than input ({}x{})",
                kernel_rows, kernel_cols, input_rows, input_cols
            ));
        }
        let output_rows = input_rows - kernel_rows + 1;
        let output_cols = input_cols - kernel_cols + 1;

        // Create shader module
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Convolve2D Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::CONVOLVE2D_SHADER.into()),
        });

        // Create buffers
        let input_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Input Image"),
            size: std::mem::size_of_val(input) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let kernel_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Kernel"),
            size: std::mem::size_of_val(kernel) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output"),
            size: std::mem::size_of_val(result) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Dimensions uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct ConvDimensions {
            input_rows: u32,
            input_cols: u32,
            kernel_rows: u32,
            kernel_cols: u32,
            output_rows: u32,
            output_cols: u32,
        }

        let dims = ConvDimensions {
            input_rows: input_rows as u32,
            input_cols: input_cols as u32,
            kernel_rows: kernel_rows as u32,
            kernel_cols: kernel_cols as u32,
            output_rows: output_rows as u32,
            output_cols: output_cols as u32,
        };

        let dims_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Conv Dimensions"),
            size: std::mem::size_of::<ConvDimensions>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write data to buffers
        self.queue.write_buffer(&input_buffer, 0, bytemuck::cast_slice(input));
        self.queue.write_buffer(&kernel_buffer, 0, bytemuck::cast_slice(kernel));
        self.queue.write_buffer(&dims_buffer, 0, bytemuck::bytes_of(&dims));

        // Create bind group layout
        let bind_group_layout =
            self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Convolve2D Bind Group Layout"),
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
            label: Some("Convolve2D Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: kernel_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dims_buffer.as_entire_binding() },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Convolve2D Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Convolve2D Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Convolve2D Encoder"),
        });

        // Compute pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Convolve2D Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups: 16x16 threads per workgroup
            let workgroup_size_x = 16;
            let workgroup_size_y = 16;
            let num_workgroups_x = (output_rows as u32).div_ceil(workgroup_size_x);
            let num_workgroups_y = (output_cols as u32).div_ceil(workgroup_size_y);
            compute_pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
        }

        // Create staging buffer for result readback
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: std::mem::size_of_val(result) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            std::mem::size_of_val(result) as u64,
        );

        // Submit commands
        self.queue.submit(Some(encoder.finish()));

        // Read result back
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).expect("oneshot channel receiver dropped");
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
            let output_data: &[f32] = bytemuck::cast_slice(&data);
            result.copy_from_slice(output_data);
        }

        staging_buffer.unmap();

        Ok(())
    }
}
