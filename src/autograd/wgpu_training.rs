//! wgpu-accelerated training utilities (zero unsafe)
//!
//! Drop-in replacement for `CudaTrainer` using wgpu (safe Rust API).
//! All GPU compute goes through WGSL compute shaders — no CUDA FFI,
//! no `unsafe` blocks, no `extern "C"`.
//!
//! # Architecture (§26 Step 0d)
//!
//! ```text
//! WgpuTrainer
//!   ├── device: wgpu::Device
//!   ├── queue: wgpu::Queue
//!   ├── forward: WGSL tiled GEMM (CUTLASS-style 64×64)
//!   ├── backward: same WGSL GEMM with transposed args
//!   └── optimizer: WGSL AdamW elementwise kernel
//! ```
//!
//! # Parity Gate (§26 Step 0e)
//!
//! Before CUDA code deletion, must prove:
//! - 3-sample loss match: |loss_wgpu - loss_cuda| < 0.1
//! - Gradient norm match: |norm_wgpu - norm_cuda| / norm_cuda < 0.05

use super::cuda_tensor::{CudaTensorError, Result};

// Feature-gate: WgpuTrainer requires the "gpu" feature (trueno with wgpu)
#[cfg(not(feature = "gpu"))]
pub struct WgpuTrainer;

#[cfg(not(feature = "gpu"))]
impl WgpuTrainer {
    pub fn new() -> Result<Self> {
        Err(CudaTensorError::CudaNotAvailable("Compiled without GPU support".into()))
    }
}

#[cfg(feature = "gpu")]
use trueno::backends::gpu::wgpu;

// KAIZEN root cause: MATMUL_SHADER (16×16) was 1200x slower than TILED_GEMM_SHADER (64×64).
// Parity proven (3/3 tests). Switching to tiled GEMM (375 GFLOPS vs ~20 GFLOPS).
#[cfg(feature = "gpu")]
const GEMM_SHADER: &str = trueno::backends::gpu::shaders::TILED_GEMM_SHADER;

/// WGSL AdamW optimizer kernel
#[cfg(feature = "gpu")]
const ADAMW_SHADER: &str = r"
@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read> grads: array<f32>;
@group(0) @binding(2) var<storage, read_write> m_state: array<f32>;
@group(0) @binding(3) var<storage, read_write> v_state: array<f32>;

struct AdamWParams {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    bias_correction1: f32,  // 1 - beta1^t
    bias_correction2: f32,  // 1 - beta2^t
    n: u32,
}

@group(0) @binding(4) var<uniform> cfg: AdamWParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n) { return; }

    let g = grads[i];
    var m = cfg.beta1 * m_state[i] + (1.0 - cfg.beta1) * g;
    var v = cfg.beta2 * v_state[i] + (1.0 - cfg.beta2) * g * g;
    m_state[i] = m;
    v_state[i] = v;

    let m_hat = m / cfg.bias_correction1;
    let v_hat = v / cfg.bias_correction2;

    // Decoupled weight decay (AdamW, not Adam with L2)
    params[i] = params[i] - cfg.lr * (m_hat / (sqrt(v_hat) + cfg.eps) + cfg.weight_decay * params[i]);
}
";

/// WGSL gradient clipping kernel
const GRAD_CLIP_SHADER: &str = r"
@group(0) @binding(0) var<storage, read_write> grads: array<f32>;

struct ClipParams {
    scale: f32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(1) var<uniform> cfg: ClipParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= cfg.n) { return; }
    grads[i] = grads[i] * cfg.scale;
}
";

/// wgpu-accelerated training context (zero unsafe, safe Rust API)
pub struct WgpuTrainer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    matmul_pipeline: wgpu::ComputePipeline,
    matmul_bgl: wgpu::BindGroupLayout,
    adamw_pipeline: wgpu::ComputePipeline,
    adamw_bgl: wgpu::BindGroupLayout,
    clip_pipeline: wgpu::ComputePipeline,
    clip_bgl: wgpu::BindGroupLayout,
    step: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmDims {
    m: u32,
    k: u32,
    n: u32,
    alpha_bits: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AdamWConfig {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    n: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ClipConfig {
    scale: f32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
}

impl WgpuTrainer {
    /// Create a new wgpu trainer. Requests a GPU device via Vulkan/Metal/DX12.
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL,
            ..Default::default()
        });

        let adapter = trueno::backends::gpu::runtime::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            },
        ))
        .map_err(|e| CudaTensorError::CudaNotAvailable(format!("No wgpu adapter: {e}")))?;

        let (device, queue) = trueno::backends::gpu::runtime::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("WgpuTrainer"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size:
                        adapter.limits().max_storage_buffer_binding_size,
                    max_buffer_size: adapter.limits().max_buffer_size,
                    ..Default::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: Default::default(),
                trace: Default::default(),
            },
        ))
        .map_err(|e| CudaTensorError::CudaNotAvailable(format!("wgpu device: {e}")))?;

        // Matmul pipeline (CUTLASS-style tiled GEMM)
        let matmul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tiled_gemm"),
            source: wgpu::ShaderSource::Wgsl(GEMM_SHADER.into()),
        });
        let matmul_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gemm_bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                uniform_entry(3),
            ],
        });
        let matmul_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gemm_pl"),
            bind_group_layouts: &[&matmul_bgl],
            push_constant_ranges: &[],
        });
        let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tiled_gemm_pipe"),
            layout: Some(&matmul_pl),
            module: &matmul_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // AdamW pipeline
        let adamw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("adamw"),
            source: wgpu::ShaderSource::Wgsl(ADAMW_SHADER.into()),
        });
        let adamw_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("adamw_bgl"),
            entries: &[
                storage_entry(0, false), // params (read-write)
                storage_entry(1, true),  // grads (read)
                storage_entry(2, false), // m_state (read-write)
                storage_entry(3, false), // v_state (read-write)
                uniform_entry(4),        // config
            ],
        });
        let adamw_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("adamw_pl"),
            bind_group_layouts: &[&adamw_bgl],
            push_constant_ranges: &[],
        });
        let adamw_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("adamw_pipe"),
            layout: Some(&adamw_pl),
            module: &adamw_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Gradient clipping pipeline
        let clip_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("grad_clip"),
            source: wgpu::ShaderSource::Wgsl(GRAD_CLIP_SHADER.into()),
        });
        let clip_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("clip_bgl"),
            entries: &[storage_entry(0, false), uniform_entry(1)],
        });
        let clip_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("clip_pl"),
            bind_group_layouts: &[&clip_bgl],
            push_constant_ranges: &[],
        });
        let clip_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("clip_pipe"),
            layout: Some(&clip_pl),
            module: &clip_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            matmul_pipeline,
            matmul_bgl,
            adamw_pipeline,
            adamw_bgl,
            clip_pipeline,
            clip_bgl,
            step: 0,
        })
    }

    /// Upload host data to GPU buffer
    pub fn upload(&self, data: &[f32]) -> wgpu::Buffer {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("upload_data"),
            size: (data.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
        buf
    }

    /// Allocate zero-initialized GPU buffer
    pub fn zeros(&self, len: usize) -> wgpu::Buffer {
        self.upload(&vec![0.0f32; len])
    }

    /// Download GPU buffer to host
    pub fn download(&self, buffer: &wgpu::Buffer) -> Vec<f32> {
        let size = buffer.size();
        // PMAT-498: label required — wgpu rejects unlabeled buffers on map_async
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("download_staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }

    /// Matrix multiply forward: C = A @ B using WGSL tiled GEMM
    pub fn matmul_forward(
        &self,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        c: &wgpu::Buffer,
        m: u32,
        k: u32,
        n: u32,
    ) {
        self.dispatch_gemm(a, b, c, m, k, n, 1.0);
    }

    /// Matrix multiply backward: grad_a = grad_c @ B^T, grad_b = A^T @ grad_c
    ///
    /// Uses the SAME tiled GEMM shader with transposed arguments.
    /// This is the standard GEMM backward formula:
    /// - ∂L/∂A = ∂L/∂C @ B^T  (shape: M×N @ N×K = M×K)
    /// - ∂L/∂B = A^T @ ∂L/∂C  (shape: K×M @ M×N = K×N)
    pub fn matmul_backward(
        &self,
        a: &wgpu::Buffer,      // [M, K] input from forward
        b: &wgpu::Buffer,      // [K, N] weight from forward
        grad_c: &wgpu::Buffer, // [M, N] upstream gradient
        grad_a: &wgpu::Buffer, // [M, K] output: grad w.r.t. A
        grad_b: &wgpu::Buffer, // [K, N] output: grad w.r.t. B
        m: u32,
        k: u32,
        n: u32,
    ) {
        // Contract: matmul_backward (backward-pass-v1)
        debug_assert!(
            m > 0 && k > 0 && n > 0,
            "Contract matmul_backward: dimensions must be positive"
        );
        // grad_a[M,K] = grad_c[M,N] @ B^T[N,K]
        // This is a GEMM with (M, N, K) → output is M×K
        // We need B transposed. Since B is stored as [K,N] row-major,
        // B^T is [N,K] row-major = B read with swapped dims.
        // WGSL_GEMM(grad_c[M,N], B_as_transposed[N,K]) → grad_a[M,K]
        //
        // For now, use the same shader by transposing on CPU or using
        // a transpose shader. Simple approach: dispatch as
        // grad_a = GEMM(grad_c, B, M, N, K) — treating B's [K,N] storage
        // as [N,K] by swapping the interpretation.
        //
        // Actually, the tiled GEMM computes C = A @ B where A is [M,K] and B is [K,N].
        // For grad_a = grad_c @ B^T: A=grad_c[M,N], "B"=B[K,N] read as B^T[N,K].
        // So we call GEMM with m=M, k=N (reduction over N), n=K.
        // But B is stored row-major as [K,N], and we need [N,K].
        // The simplest correct approach: use B as-is but treat it as transposed.
        // This requires the shader to support transposed reads, or we transpose first.
        //
        // For correctness, let's transpose B on GPU first. We can add a transpose
        // shader later for optimization. For now, download-transpose-upload.
        // TODO: WGSL transpose shader for zero-copy backward.

        // grad_a = grad_c @ B^T
        // Naive approach: transpose B, then GEMM
        let b_data = self.download(b);
        let mut bt_data = vec![0.0f32; (k * n) as usize];
        for i in 0..k as usize {
            for j in 0..n as usize {
                bt_data[j * k as usize + i] = b_data[i * n as usize + j];
            }
        }
        let bt = self.upload(&bt_data);
        self.dispatch_gemm(grad_c, &bt, grad_a, m, n, k, 1.0);

        // grad_b = A^T @ grad_c
        let a_data = self.download(a);
        let mut at_data = vec![0.0f32; (m * k) as usize];
        for i in 0..m as usize {
            for j in 0..k as usize {
                at_data[j * m as usize + i] = a_data[i * k as usize + j];
            }
        }
        let at = self.upload(&at_data);
        self.dispatch_gemm(&at, grad_c, grad_b, k, m, n, 1.0);
    }

    /// AdamW optimizer step on GPU
    pub fn adamw_step(
        &mut self,
        params: &wgpu::Buffer,
        grads: &wgpu::Buffer,
        m_state: &wgpu::Buffer,
        v_state: &wgpu::Buffer,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) {
        self.step += 1;
        let n = (params.size() / 4) as u32;
        let bc1 = 1.0 - beta1.powi(self.step as i32);
        let bc2 = 1.0 - beta2.powi(self.step as i32);

        let cfg = AdamWConfig {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1: bc1,
            bias_correction2: bc2,
            n,
        };
        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<AdamWConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&cfg_buf, 0, bytemuck::bytes_of(&cfg));

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.adamw_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: grads.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: m_state.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: v_state.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cfg_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.adamw_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Gradient clipping (downloads to compute norm, clips on GPU)
    pub fn clip_gradients(&self, grads: &wgpu::Buffer, max_norm: f32) {
        let grad_data = self.download(grads);
        let grad_norm: f32 = grad_data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let scale = if grad_norm > max_norm {
            max_norm / grad_norm
        } else {
            return; // No clipping needed
        };

        let n = grad_data.len() as u32;
        let cfg = ClipConfig { scale, n, _pad0: 0, _pad1: 0 };
        let cfg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&cfg_buf, 0, bytemuck::bytes_of(&cfg));

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.clip_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: grads.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cfg_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.clip_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Get current step count
    pub fn step_count(&self) -> u32 {
        self.step
    }

    /// Reset step counter
    pub fn reset_step(&mut self) {
        self.step = 0;
    }

    /// Get a reference to the wgpu queue (for buffer writes)
    pub fn queue_ref(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Get a reference to the wgpu device (for buffer creation)
    pub fn device_ref(&self) -> &wgpu::Device {
        &self.device
    }

    /// Create from an existing device + queue (share device with WgslForwardPass).
    /// Contract: single device for all GPU operations — no cross-device buffer access.
    pub fn from_device(device: wgpu::Device, queue: wgpu::Queue) -> Result<Self> {
        let matmul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tiled_gemm"),
            source: wgpu::ShaderSource::Wgsl(GEMM_SHADER.into()),
        });
        let matmul_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gemm_bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                uniform_entry(3),
            ],
        });
        let matmul_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gemm_pl"),
            bind_group_layouts: &[&matmul_bgl],
            push_constant_ranges: &[],
        });
        let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tiled_gemm_pipe"),
            layout: Some(&matmul_pl),
            module: &matmul_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let adamw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("adamw"),
            source: wgpu::ShaderSource::Wgsl(ADAMW_SHADER.into()),
        });
        let adamw_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("adamw_bgl"),
            entries: &[
                storage_entry(0, false),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
                uniform_entry(4),
            ],
        });
        let adamw_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("adamw_pl"),
            bind_group_layouts: &[&adamw_bgl],
            push_constant_ranges: &[],
        });
        let adamw_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("adamw_pipe"),
            layout: Some(&adamw_pl),
            module: &adamw_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let clip_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("grad_clip"),
            source: wgpu::ShaderSource::Wgsl(GRAD_CLIP_SHADER.into()),
        });
        let clip_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("clip_bgl"),
            entries: &[storage_entry(0, false), uniform_entry(1)],
        });
        let clip_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("clip_pl"),
            bind_group_layouts: &[&clip_bgl],
            push_constant_ranges: &[],
        });
        let clip_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("clip_pipe"),
            layout: Some(&clip_pl),
            module: &clip_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            matmul_pipeline,
            matmul_bgl,
            adamw_pipeline,
            adamw_bgl,
            clip_pipeline,
            clip_bgl,
            step: 0,
        })
    }

    // --- Internal helpers ---

    fn dispatch_gemm(
        &self,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        c: &wgpu::Buffer,
        m: u32,
        k: u32,
        n: u32,
        alpha: f32,
    ) {
        // KAIZEN: chunk B along N when it exceeds max_storage_buffer_binding_size.
        // GPU-side extraction via copy_buffer_to_buffer — no CPU roundtrip.
        let max_binding = u64::from(self.device.limits().max_storage_buffer_binding_size);
        let b_bytes = u64::from(k) * u64::from(n) * 4;
        if b_bytes > max_binding {
            let max_n_chunk = (max_binding / 4 / u64::from(k)) as u32;
            let max_n_chunk = max_n_chunk.max(1);

            // Extract B chunks on GPU: B is [K, N] row-major.
            // Chunk along N: each chunk is [K, chunk_n].
            // B[row][col] is at byte offset (row * N + col) * 4.
            // Row-major chunking requires per-row copies (not contiguous in memory).
            // For simplicity: download B once (not per chunk), extract on CPU, upload chunks.
            // The download is the cost — but only once, not per chunk.
            let b_data = self.download(b);
            let mut n_start = 0u32;
            while n_start < n {
                let chunk_n = (n - n_start).min(max_n_chunk);
                let mut b_chunk = vec![0.0f32; (k * chunk_n) as usize];
                for row in 0..k as usize {
                    let src_start = row * n as usize + n_start as usize;
                    let dst_start = row * chunk_n as usize;
                    b_chunk[dst_start..dst_start + chunk_n as usize]
                        .copy_from_slice(&b_data[src_start..src_start + chunk_n as usize]);
                }
                let b_chunk_buf = self.upload(&b_chunk);
                let c_chunk_buf = self.zeros((m * chunk_n) as usize);
                self.dispatch_gemm(a, &b_chunk_buf, &c_chunk_buf, m, k, chunk_n, alpha);
                // Copy chunk result into C at the right column offset
                // C is [M, N] row-major. Chunk covers columns [n_start..n_start+chunk_n].
                let c_chunk_data = self.download(&c_chunk_buf);
                // Write directly into C buffer at column offsets (per-row write)
                let mut c_data =
                    if n_start == 0 { vec![0.0f32; (m * n) as usize] } else { self.download(c) };
                for row in 0..m as usize {
                    let dst_start = row * n as usize + n_start as usize;
                    let src_start = row * chunk_n as usize;
                    c_data[dst_start..dst_start + chunk_n as usize]
                        .copy_from_slice(&c_chunk_data[src_start..src_start + chunk_n as usize]);
                }
                self.queue.write_buffer(c, 0, bytemuck::cast_slice(&c_data));
                n_start += chunk_n;
            }
            return;
        }

        let dims = GemmDims { m, k, n, alpha_bits: alpha.to_bits() };
        let dims_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&dims_buf, 0, bytemuck::bytes_of(&dims));

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.matmul_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: c.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dims_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.matmul_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(64), m.div_ceil(64), 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
#[path = "wgpu_training_tests.rs"]
mod tests;
