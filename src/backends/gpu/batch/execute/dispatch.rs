//! Shader dispatch infrastructure for GPU compute operations
//!
//! Contains pipeline-cached dispatch functions that encode operations into
//! a shared command encoder.  Pipelines are compiled once per unique shader
//! and reused across all operations in a batch (KAIZEN-022).

use super::super::GpuCommandBatch;
use std::collections::HashMap;

/// Workgroup size for element-wise compute shaders (threads per workgroup)
const WORKGROUP_SIZE: u32 = 256;

/// Cached GPU pipeline: shader module → compute pipeline + bind group layout.
/// Created once per unique shader source, reused for all operations using that shader.
pub struct CachedPipeline {
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
}

/// Pipeline cache keyed by shader source pointer address.
///
/// Safe because all shader sources are `&'static str` constants with stable addresses.
/// Create one per training session and pass to `GpuCommandBatch::execute_with_cache()`
/// to avoid recompiling shaders across batch executions (KAIZEN-023).
pub type PipelineCache = HashMap<usize, CachedPipeline>;

/// Compute a cache key from a shader source string.
/// Uses pointer address since all sources are `&'static str` constants.
fn cache_key(shader_source: &str) -> usize {
    shader_source.as_ptr() as usize
}

impl GpuCommandBatch {
    /// Encode a unary operation (one input, one output) into the command encoder.
    ///
    /// Pipeline is cached per shader source — first call compiles, subsequent calls reuse.
    pub(crate) fn encode_unary_op<T: bytemuck::Pod>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        cache: &mut PipelineCache,
        shader_source: &str,
        label: &str,
        input_buffer: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        size: usize,
        params: Option<&T>,
    ) -> Result<(), String> {
        let key = cache_key(shader_source);
        let has_params = params.is_some();

        // Get or create cached pipeline
        if !cache.contains_key(&key) {
            let shader = self.device.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{} Shader", label)),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

            let mut layout_entries = vec![
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
            ];

            if has_params {
                layout_entries.push(wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });
            }

            let bind_group_layout =
                self.device.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{} Layout", label)),
                    entries: &layout_entries,
                });

            let pipeline_layout =
                self.device.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(&format!("{} PipelineLayout", label)),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

            let pipeline =
                self.device.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("{} Pipeline", label)),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

            cache.insert(key, CachedPipeline { pipeline, bind_group_layout });
        }

        let cached = cache.get(&key).expect("pipeline just inserted");

        // Create uniform buffer if params provided (per-call, not cached)
        let params_buffer = if let Some(params_data) = params {
            let buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{} Params", label)),
                size: std::mem::size_of::<T>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.device.queue.write_buffer(&buffer, 0, bytemuck::bytes_of(params_data));
            Some(buffer)
        } else {
            None
        };

        // Create bind group (per-call — references specific buffers)
        let mut bind_entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
        ];

        if let Some(ref buffer) = params_buffer {
            bind_entries
                .push(wgpu::BindGroupEntry { binding: 2, resource: buffer.as_entire_binding() });
        }

        let bind_group = self.device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} BindGroup", label)),
            layout: &cached.bind_group_layout,
            entries: &bind_entries,
        });

        // Encode compute pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{} Pass", label)),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((size as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }

        Ok(())
    }

    /// Encode a matrix multiplication into the command encoder.
    ///
    /// Pipeline is cached — first matmul compiles the tiled shader, subsequent matmuls reuse it.
    pub(crate) fn encode_matmul_op(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        cache: &mut PipelineCache,
        shader_source: &str,
        label: &str,
        a: &super::super::BufferId,
        b: &super::super::BufferId,
        output: &super::super::BufferId,
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<(), String> {
        contract_pre_tiled_naive_equivalence!();
        let a_info = self.buffers.get(a).ok_or("Invalid buffer A ID")?;
        let b_info = self.buffers.get(b).ok_or("Invalid buffer B ID")?;
        let output_info = self.buffers.get(output).ok_or("Invalid output buffer ID")?;

        let a_buffer = a_info.gpu_buffer.as_ref().ok_or("Buffer A not created")?;
        let b_buffer = b_info.gpu_buffer.as_ref().ok_or("Buffer B not created")?;
        let output_buffer = output_info.gpu_buffer.as_ref().ok_or("Output buffer not created")?;

        let key = cache_key(shader_source);

        // Get or create cached pipeline
        if !cache.contains_key(&key) {
            let shader = self.device.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{} Shader", label)),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

            let bind_group_layout =
                self.device.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{} Layout", label)),
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

            let pipeline_layout =
                self.device.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(&format!("{} PipelineLayout", label)),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

            let pipeline =
                self.device.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("{} Pipeline", label)),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

            cache.insert(key, CachedPipeline { pipeline, bind_group_layout });
        }

        let cached = cache.get(&key).expect("pipeline just inserted");

        // Create dimensions uniform buffer (per-call — dimensions differ per matmul)
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct MatmulDims {
            m: u32,
            k: u32,
            n: u32,
            _pad: u32,
        }

        let dims = MatmulDims { m, k, n, _pad: 0 };
        let dims_buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} Dims", label)),
            size: std::mem::size_of::<MatmulDims>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.device.queue.write_buffer(&dims_buffer, 0, bytemuck::bytes_of(&dims));

        // Create bind group (per-call — references specific buffers)
        let bind_group = self.device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} BindGroup", label)),
            layout: &cached.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dims_buffer.as_entire_binding() },
            ],
        });

        // Encode compute pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{} Pass", label)),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(m.div_ceil(16), n.div_ceil(16), 1);
        }

        contract_post_tiled_naive_equivalence!(());
        Ok(())
    }

    /// Encode a binary operation (two inputs, one output) into the command encoder.
    ///
    /// Pipeline is cached per shader source.
    pub(crate) fn encode_binary_op(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        cache: &mut PipelineCache,
        shader_source: &str,
        label: &str,
        a_buffer: &wgpu::Buffer,
        b_buffer: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        size: usize,
    ) -> Result<(), String> {
        let key = cache_key(shader_source);

        // Get or create cached pipeline
        if !cache.contains_key(&key) {
            let shader = self.device.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{} Shader", label)),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

            let bind_group_layout =
                self.device.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{} Layout", label)),
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

            let pipeline_layout =
                self.device.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(&format!("{} PipelineLayout", label)),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

            let pipeline =
                self.device.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("{} Pipeline", label)),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

            cache.insert(key, CachedPipeline { pipeline, bind_group_layout });
        }

        let cached = cache.get(&key).expect("pipeline just inserted");

        // Create bind group (per-call)
        let bind_group = self.device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} BindGroup", label)),
            layout: &cached.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output_buffer.as_entire_binding() },
            ],
        });

        // Encode compute pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{} Pass", label)),
                timestamp_writes: None,
            });
            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((size as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }

        Ok(())
    }
}
