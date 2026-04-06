//! PMAT-322: Cached GPU matmul with persistent weight buffers.
//!
//! The default `matmul_async` creates all GPU objects per call (~8ms overhead).
//! This module pre-uploads weight matrices and caches the pipeline, reducing
//! per-call overhead to: upload input + dispatch + download output (~0.1ms).

use std::collections::HashMap;

/// Cached matmul state: pipeline + pre-uploaded weight buffers + persistent I/O.
///
/// PMAT-323: Three levels of buffer persistence:
/// 1. Weight buffers — uploaded once at model init (PMAT-322)
/// 2. I/O buffers — pre-allocated to max size, reused across calls (PMAT-323)
/// 3. Pipeline + bind group layout — created once, reused forever
pub struct GpuMatmulCache {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    /// CUTLASS-style tiled GEMM pipeline for M>16 (training batch, prefill)
    tiled_pipeline: wgpu::ComputePipeline,
    /// PMAT-326: Dedicated GEMV pipeline for M=1 (cooperative K-reduction)
    gemv_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Pre-uploaded weight buffers keyed by name
    weight_buffers: HashMap<String, WeightEntry>,
    /// PMAT-323: Persistent I/O buffers (grow-only, never deallocated)
    input_buffer: Option<wgpu::Buffer>,
    input_size: u64,
    output_buffer: Option<wgpu::Buffer>,
    output_size: u64,
    dims_buffer: Option<wgpu::Buffer>,
    /// Reusable staging buffer (grows as needed)
    staging_size: u64,
    staging_buffer: Option<wgpu::Buffer>,
}

struct WeightEntry {
    buffer: wgpu::Buffer,
    rows: usize,
    cols: usize,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Dimensions {
    m: u32,
    k: u32,
    n: u32,
    /// Alpha scaling factor for tiled GEMM epilogue. Reinterpreted as f32.
    /// Old naive shader ignores this field (_padding). Tiled GEMM reads it as `dims.alpha`.
    alpha_bits: u32,
}

impl GpuMatmulCache {
    /// Create a new cached matmul context from an existing GpuDevice.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CachedMatmul Shader"),
            source: wgpu::ShaderSource::Wgsl(crate::backends::gpu::shaders::MATMUL_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("CachedMatmul BGL"),
            entries: &[
                bgl_entry(0, true),  // A (input, read-only)
                bgl_entry(1, true),  // B (weight, read-only)
                bgl_entry(2, false), // C (output, read-write)
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("CachedMatmul PL"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("CachedMatmul Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // CUTLASS-style tiled GEMM pipeline (64×64 tiles, 4×4 thread micro-tiles)
        let tiled_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TiledGEMM Shader"),
            source: wgpu::ShaderSource::Wgsl(
                crate::backends::gpu::shaders::TILED_GEMM_SHADER.into(),
            ),
        });
        let tiled_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("TiledGEMM Pipeline"),
            layout: Some(&pipeline_layout),
            module: &tiled_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // PMAT-326: GEMV pipeline (cooperative K-reduction, optimal for M=1)
        let gemv_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GEMV Shader"),
            source: wgpu::ShaderSource::Wgsl(crate::backends::gpu::shaders::GEMV_SHADER.into()),
        });
        let gemv_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GEMV Pipeline"),
            layout: Some(&pipeline_layout),
            module: &gemv_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            pipeline,
            tiled_pipeline,
            gemv_pipeline,
            bind_group_layout,
            weight_buffers: HashMap::new(),
            input_buffer: None,
            input_size: 0,
            output_buffer: None,
            output_size: 0,
            dims_buffer: None,
            staging_size: 0,
            staging_buffer: None,
        }
    }

    /// Pre-upload a weight matrix (call once at model init).
    /// Weight is stored in row-major f32: shape [rows, cols].
    /// Silently skips weights that exceed the device's max buffer binding size.
    pub fn upload_weight(&mut self, name: &str, data: &[f32], rows: usize, cols: usize) {
        assert_eq!(data.len(), rows * cols, "weight size mismatch");
        let size_bytes = (data.len() * 4) as u64;
        let max_binding = self.device.limits().max_storage_buffer_binding_size as u64;
        if size_bytes > max_binding {
            eprintln!(
                "[wgpu] Skipping weight '{}' ({:.1} MB > {:.1} MB max binding) — will use CPU fallback",
                name,
                size_bytes as f64 / 1e6,
                max_binding as f64 / 1e6
            );
            return;
        }
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size: size_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, bytemuck::cast_slice(data));
        self.weight_buffers.insert(name.to_string(), WeightEntry { buffer, rows, cols });
    }

    /// Number of pre-uploaded weights.
    pub fn weight_count(&self) -> usize {
        self.weight_buffers.len()
    }

    /// Total VRAM used by weight buffers (bytes).
    pub fn weight_bytes(&self) -> usize {
        self.weight_buffers.values().map(|w| w.rows * w.cols * 4).sum()
    }

    /// PMAT-323: Ensure persistent I/O buffers are at least `size` bytes.
    /// Grows only — never shrinks. Returns reference to the buffer.
    fn ensure_input_buffer(&mut self, size: u64) {
        if self.input_size < size {
            self.input_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("persistent_input"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.input_size = size;
        }
    }

    fn ensure_output_buffer(&mut self, size: u64) {
        if self.output_size < size {
            self.output_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("persistent_output"),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.output_size = size;
        }
    }

    fn ensure_dims_buffer(&mut self) {
        if self.dims_buffer.is_none() {
            self.dims_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("persistent_dims"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
    }

    fn ensure_staging_buffer(&mut self, size: u64) {
        if self.staging_size < size {
            self.staging_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("persistent_staging"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.staging_size = size;
        }
    }

    /// PMAT-323: Zero-alloc matmul using persistent I/O buffers.
    /// Only creates bind group per call (required by WGPU — bind groups reference
    /// specific buffer instances). Everything else is reused.
    pub fn matmul_cached(
        &mut self,
        weight_name: &str,
        input: &[f32],
        output: &mut [f32],
        m: usize,
    ) -> Result<(), String> {
        // Extract weight dims first to avoid borrow conflict
        let (k, n) = {
            let entry = self
                .weight_buffers
                .get(weight_name)
                .ok_or_else(|| format!("Weight '{}' not uploaded", weight_name))?;
            (entry.cols, entry.rows)
        };

        if input.len() < m * k {
            return Err(format!("input too small: need {}, have {}", m * k, input.len()));
        }
        if output.len() < m * n {
            return Err(format!("output too small: need {}, have {}", m * n, output.len()));
        }

        let input_bytes = (m * k * 4) as u64;
        let output_bytes = (m * n * 4) as u64;

        // Ensure persistent buffers are large enough (may alloc on first call / size increase)
        self.ensure_input_buffer(input_bytes);
        self.ensure_output_buffer(output_bytes);
        self.ensure_dims_buffer();
        self.ensure_staging_buffer(output_bytes);

        // Write input + dims to persistent buffers (just memcpy, no alloc)
        let input_buf = self.input_buffer.as_ref().unwrap();
        self.queue.write_buffer(input_buf, 0, bytemuck::cast_slice(&input[..m * k]));

        // PMAT-346: GEMV shader expects Params { n (output dim), k, _, _ }
        // but Dimensions struct has { m, k, n, _ }. When m=1, params.n reads m=1
        // instead of the actual output dimension. Write different layout for GEMV.
        let dims = if m == 1 {
            Dimensions { m: n as u32, k: k as u32, n: 0, alpha_bits: 1.0_f32.to_bits() }
        } else {
            Dimensions { m: m as u32, k: k as u32, n: n as u32, alpha_bits: 1.0_f32.to_bits() }
        };
        let dims_buf = self.dims_buffer.as_ref().unwrap();
        self.queue.write_buffer(dims_buf, 0, bytemuck::bytes_of(&dims));

        // Bind group (per-call — WGPU requires new bind group when buffer references change)
        let output_buf = self.output_buffer.as_ref().unwrap();
        let weight_buf = &self
            .weight_buffers
            .get(weight_name)
            .ok_or_else(|| {
                format!("weight '{}' not loaded — call load_weight() first", weight_name)
            })?
            .buffer;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: weight_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dims_buf.as_entire_binding() },
            ],
        });

        let staging = self.staging_buffer.as_ref().unwrap();

        // Encode + dispatch
        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul"),
                timestamp_writes: None,
            });
            if m == 1 {
                // PMAT-326: Use GEMV shader for M=1 — cooperative K-reduction
                // Each workgroup handles 1 output row with 256 threads reducing K
                // Dispatch N workgroups (one per output element)
                pass.set_pipeline(&self.gemv_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(n as u32, 1, 1);
            } else if m >= 4 {
                // CUTLASS-style tiled GEMM for M>=4 (training batch, prefill)
                // 64×64 tiles, 4×4 thread micro-tiles, double-buffered shared memory
                // ~10-30x faster than naive 16×16 for large M
                pass.set_pipeline(&self.tiled_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((n as u32).div_ceil(64), (m as u32).div_ceil(64), 1);
            } else {
                // Naive 16×16 tiled GEMM for small M (2-3 rows)
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((m as u32).div_ceil(16), (n as u32).div_ceil(16), 1);
            }
        }

        encoder.copy_buffer_to_buffer(output_buf, 0, staging, 0, output_bytes);
        self.queue.submit(Some(encoder.finish()));

        // Readback
        let slice = staging.slice(..output_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
        rx.recv().map_err(|e| format!("recv: {e}"))?.map_err(|e| format!("map: {e:?}"))?;

        {
            let data = slice.get_mapped_range();
            output[..m * n].copy_from_slice(bytemuck::cast_slice(&data));
        }
        staging.unmap();

        Ok(())
    }
}

fn bgl_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimensions_layout() {
        let dims = Dimensions { m: 1, k: 1536, n: 1536, alpha_bits: 1.0_f32.to_bits() };
        let bytes = bytemuck::bytes_of(&dims);
        assert_eq!(bytes.len(), 16); // 4 × u32
                                     // Verify field order matches shader uniform layout
        assert_eq!(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]), 1);
        assert_eq!(u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]), 1536);
    }

    #[test]
    fn test_gemv_params_layout() {
        // PMAT-346: When m=1, first field must be n (output dim), not m
        let m = 1usize;
        let k = 1536usize;
        let n = 256usize;
        let dims = if m == 1 {
            Dimensions { m: n as u32, k: k as u32, n: 0, alpha_bits: 1.0_f32.to_bits() }
        } else {
            Dimensions { m: m as u32, k: k as u32, n: n as u32, alpha_bits: 1.0_f32.to_bits() }
        };
        let bytes = bytemuck::bytes_of(&dims);
        // GEMV shader reads params.n (offset 0) as output dimension
        let gemv_n = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(gemv_n, 256, "GEMV params.n must be output dimension, not m");
    }

    #[test]
    fn test_matmul_params_layout() {
        let dims = Dimensions { m: 4, k: 1536, n: 1536, alpha_bits: 1.0_f32.to_bits() };
        let bytes = bytemuck::bytes_of(&dims);
        // Matmul shader reads dims.M, dims.K, dims.N
        assert_eq!(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]), 4); // M
        assert_eq!(u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]), 1536); // K
        assert_eq!(u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]), 1536);
        // N
    }
}
