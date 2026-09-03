#![allow(dead_code, clippy::many_single_char_names)]
//! PMAT-324: WGSL transformer forward pass — multi-pass single submission.
//!
//! Instead of one matmul per CPU call (2ms roundtrip each), this encodes
//! ALL operations for one transformer layer into a single command encoder.
//! Only one submit + one readback per layer (or per full forward pass).
//!
//! Architecture: separate WGSL kernels per operation type, dispatched
//! sequentially within one command encoder. All intermediate data stays
//! GPU-resident in persistent buffers.

use std::collections::HashMap;

/// Saved activations for one transformer layer's backward pass.
///
/// Contains the 7 tensors needed for LoRA gradient computation
/// without replaying the forward pass (§26.11.5, falsification-verified).
pub struct LayerActivations {
    /// Input to Q/K/V projections (RMSNorm output). [seq, hidden]
    pub attn_norm_out: wgpu::Buffer,
    /// Input to O projection (attention output). [seq, q_dim]
    pub attn_output: wgpu::Buffer,
    /// Input to gate/up/down projections (FFN RMSNorm output). [seq, hidden]
    pub ffn_norm_out: wgpu::Buffer,
    /// Input to down projection (SiLU(gate)×up). [seq, intermediate]
    pub silu_gate_output: wgpu::Buffer,
    /// RMSNorm reciprocal std for attention norm. [seq]
    pub rstd_attn: wgpu::Buffer,
    /// RMSNorm reciprocal std for FFN norm. [seq]
    pub rstd_ffn: wgpu::Buffer,
    /// Softmax logsumexp for attention backward. [num_heads, seq]
    pub softmax_logsumexp: wgpu::Buffer,
}

/// Optional LoRA buffers for Q/K/V projections in a layer's forward pass.
pub struct QkvLoRA<'a> {
    pub q_a: &'a wgpu::Buffer,
    pub q_b: &'a wgpu::Buffer,
    pub k_a: &'a wgpu::Buffer,
    pub k_b: &'a wgpu::Buffer,
    pub v_a: &'a wgpu::Buffer,
    pub v_b: &'a wgpu::Buffer,
    pub rank: u32,
    pub scale: f32,
    pub in_dim: u32,
    pub q_dim: u32,
    pub kv_dim: u32,
    pub lora_pipeline: &'a wgpu::ComputePipeline,
    pub lora_bgl: &'a wgpu::BindGroupLayout,
}

/// GPU-resident transformer layer state.
/// All buffers persist across tokens — only input/output change per step.
pub struct WgslForwardPass {
    device: wgpu::Device,
    queue: wgpu::Queue,

    // Kernels (compiled once)
    matmul_pipeline: wgpu::ComputePipeline,
    /// CUTLASS-style tiled GEMM for M>=4 (training batch, prefill)
    tiled_matmul_pipeline: wgpu::ComputePipeline,
    /// PMAT-327: GEMV pipeline for M=1 decode (cooperative K-reduction)
    gemv_pipeline: wgpu::ComputePipeline,
    /// C-WGPU-Q4K-001: Q4K GEMV pipeline — dequantize-on-the-fly, no F32 weights
    q4k_gemv_pipeline: wgpu::ComputePipeline,
    /// Causal attention pipeline for training (full sequence, no KV cache)
    attention_pipeline: wgpu::ComputePipeline,
    attention_bgl: wgpu::BindGroupLayout,
    rmsnorm_pipeline: wgpu::ComputePipeline,
    silu_mul_pipeline: wgpu::ComputePipeline,
    rope_pipeline: wgpu::ComputePipeline,
    batch_rope_pipeline: wgpu::ComputePipeline,
    batch_rope_bgl: wgpu::BindGroupLayout,
    residual_pipeline: wgpu::ComputePipeline,

    // Bind group layouts
    matmul_bgl: wgpu::BindGroupLayout,
    elementwise_bgl: wgpu::BindGroupLayout,

    // Weight buffers (persistent, uploaded once)
    weight_buffers: HashMap<String, wgpu::Buffer>,
    /// GH-560: Raw Q4K weight buffers for fused dequant+GEMV.
    q4k_weights: HashMap<String, wgpu::Buffer>,
    /// PMAT-342: CPU-side bias data (small, not worth GPU dispatch)
    cpu_biases: HashMap<String, Vec<f32>>,
    /// GH-560: Per-layer GPU KV cache buffers.
    kv_cache_k: Vec<wgpu::Buffer>,
    /// GH-560: Per-layer GPU KV cache buffers (values).
    kv_cache_v: Vec<wgpu::Buffer>,

    // Intermediate buffers (persistent, reused across calls)
    // For 1.5B: hidden=1536, kv=256, intermediate=8960
    hidden_buf: wgpu::Buffer,   // [hidden_dim] working state
    q_buf: wgpu::Buffer,        // [q_dim]
    k_buf: wgpu::Buffer,        // [kv_dim]
    v_buf: wgpu::Buffer,        // [kv_dim]
    attn_out_buf: wgpu::Buffer, // [hidden_dim]
    ffn_gate_buf: wgpu::Buffer, // [intermediate_dim]
    ffn_up_buf: wgpu::Buffer,   // [intermediate_dim]
    ffn_silu_buf: wgpu::Buffer, // [intermediate_dim] — SiLU(gate)×up output (can't alias inputs)
    ffn_out_buf: wgpu::Buffer,  // [hidden_dim]
    norm_buf: wgpu::Buffer,     // [hidden_dim] for RMSNorm output
    staging_buf: wgpu::Buffer,  // readback

    // Config
    hidden_dim: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    intermediate_dim: u32,
}

// WGSL shader source for RMSNorm (multi-row via workgroup_id.y)
// Dispatch: (1, seq_len, 1) — one workgroup per row.
const RMSNORM_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // (dim, 0, 0, 0)

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wg_id: vec3<u32>) {
    let dim = params.x;
    let row = wg_id.y;
    let base = row * dim;
    let tid = lid.x;

    // Compute sum of squares (reduction) for this row
    var local_sum: f32 = 0.0;
    var i = tid;
    while (i < dim) {
        let val = input[base + i];
        local_sum += val * val;
        i += 256u;
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    // Tree reduction
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    let rms = sqrt(shared_sum[0] / f32(dim) + 1e-6);

    // Normalize and scale
    i = tid;
    while (i < dim) {
        output[base + i] = (input[base + i] / rms) * weight[i];
        i += 256u;
    }
}
"#;

// WGSL shader for SiLU(gate) * up
const SILU_MUL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // (dim, 0, 0, 0)

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.x) { return; }
    let g = gate[idx];
    let silu_g = g / (1.0 + exp(-g));
    output[idx] = silu_g * up[idx];
}
"#;

// WGSL shader for residual add: output = a + b
const RESIDUAL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // (dim, 0, 0, 0)

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.x) { return; }
    output[idx] = a[idx] + b[idx];
}
"#;

// Batch RoPE shader — applies RoPE to all positions in a sequence at once.
// PMAT-509: Training forward path was missing RoPE entirely, causing loss > random.
// Input: qk[seq_len * num_heads * head_dim], applies position-dependent rotation.
const BATCH_ROPE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> qk: array<f32>;

struct RopeParams {
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    _pad: u32,
}

@group(0) @binding(1) var<uniform> params: RopeParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.seq_len * params.num_heads * params.head_dim;
    if (idx >= total) { return; }

    let head_dim = params.head_dim;
    let half_hd = head_dim / 2u;

    // Decompose idx into (position, head, pos_in_head)
    let elements_per_pos = params.num_heads * head_dim;
    let position = idx / elements_per_pos;
    let within_pos = idx % elements_per_pos;
    let head_idx = within_pos / head_dim;
    let pos_in_head = within_pos % head_dim;

    // Only process the first half of each head (pairs with second half)
    if (pos_in_head >= half_hd) { return; }

    let theta = pow(1000000.0, -f32(pos_in_head * 2u) / f32(head_dim));
    let angle = f32(position) * theta;
    let cos_a = cos(angle);
    let sin_a = sin(angle);

    let base = position * elements_per_pos + head_idx * head_dim;
    let i0 = base + pos_in_head;
    let i1 = i0 + half_hd;

    let x0 = qk[i0];
    let x1 = qk[i1];
    qk[i0] = x0 * cos_a - x1 * sin_a;
    qk[i1] = x0 * sin_a + x1 * cos_a;
}
"#;

// RoPE shader (NeoX-style interleaved) — single position (inference)
const ROPE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> qk: array<f32>;
@group(0) @binding(1) var<uniform> params: vec4<u32>; // (dim, position, num_heads, head_dim)

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let dim = params.x;
    let position = params.y;
    let head_dim = params.w;

    if (idx >= dim) { return; }

    let half_hd = head_dim / 2u;
    let head_idx = idx / head_dim;
    let pos_in_head = idx % head_dim;

    if (pos_in_head >= half_hd) { return; }

    let theta = pow(1000000.0, -f32(pos_in_head * 2u) / f32(head_dim));
    let angle = f32(position) * theta;
    let cos_a = cos(angle);
    let sin_a = sin(angle);

    let i0 = head_idx * head_dim + pos_in_head;
    let i1 = i0 + half_hd;

    let x0 = qk[i0];
    let x1 = qk[i1];
    qk[i0] = x0 * cos_a - x1 * sin_a;
    qk[i1] = x0 * sin_a + x1 * cos_a;
}
"#;

impl WgslForwardPass {
    /// Get the shader sources for external inspection/testing
    pub fn rmsnorm_shader() -> &'static str {
        RMSNORM_SHADER
    }
    pub fn silu_mul_shader() -> &'static str {
        SILU_MUL_SHADER
    }
    pub fn residual_shader() -> &'static str {
        RESIDUAL_SHADER
    }
    pub fn rope_shader() -> &'static str {
        ROPE_SHADER
    }

    /// PMAT-325: Create a new WGSL forward pass context.
    ///
    /// Compiles all shader pipelines and allocates persistent intermediate buffers.
    /// Call once at model init. All GPU resources persist until dropped.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_dim: usize,
    ) -> Self {
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // Compile shaders
        let matmul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul"),
            source: wgpu::ShaderSource::Wgsl(crate::backends::gpu::shaders::MATMUL_SHADER.into()),
        });
        let rmsnorm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rmsnorm"),
            source: wgpu::ShaderSource::Wgsl(RMSNORM_SHADER.into()),
        });
        let silu_mul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("silu_mul"),
            source: wgpu::ShaderSource::Wgsl(SILU_MUL_SHADER.into()),
        });
        let rope_shader_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rope"),
            source: wgpu::ShaderSource::Wgsl(ROPE_SHADER.into()),
        });
        let residual_shader_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("residual"),
            source: wgpu::ShaderSource::Wgsl(RESIDUAL_SHADER.into()),
        });

        // Bind group layouts
        let matmul_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("matmul_bgl"),
            entries: &[
                bgl_storage(0, true),
                bgl_storage(1, true),
                bgl_storage(2, false),
                bgl_uniform(3),
            ],
        });
        let elementwise_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ew_bgl"),
            entries: &[
                bgl_storage(0, true),
                bgl_storage(1, true),
                bgl_storage(2, false),
                bgl_uniform(3),
            ],
        });

        // Pipelines
        let make_pipeline =
            |shader: &wgpu::ShaderModule, bgl: &wgpu::BindGroupLayout, label: &str| {
                let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[bgl],
                    push_constant_ranges: &[],
                });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pl),
                    module: shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                })
            };

        let matmul_pipeline = make_pipeline(&matmul_shader, &matmul_bgl, "matmul_pipe");

        // CUTLASS-style tiled GEMM for M>=4 (training batch, prefill)
        let tiled_matmul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tiled_matmul"),
            source: wgpu::ShaderSource::Wgsl(
                crate::backends::gpu::shaders::TILED_GEMM_SHADER.into(),
            ),
        });
        let tiled_matmul_pipeline =
            make_pipeline(&tiled_matmul_shader, &matmul_bgl, "tiled_matmul_pipe");

        // Causal attention pipeline for training
        let attention_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("causal_attention"),
            source: wgpu::ShaderSource::Wgsl(
                crate::backends::gpu::shaders::CAUSAL_ATTENTION_SHADER.into(),
            ),
        });
        let attention_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("attn_bgl"),
            entries: &[
                bgl_storage(0, true),  // Q
                bgl_storage(1, true),  // K
                bgl_storage(2, true),  // V
                bgl_storage(3, false), // output
                bgl_uniform(4),        // params
            ],
        });
        let attention_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("attn_pl"),
            bind_group_layouts: &[&attention_bgl],
            push_constant_ranges: &[],
        });
        let attention_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("attn_pipe"),
            layout: Some(&attention_pl),
            module: &attention_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // PMAT-327: GEMV pipeline — same bind group layout as matmul but cooperative reduction
        let gemv_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gemv"),
            source: wgpu::ShaderSource::Wgsl(crate::backends::gpu::shaders::GEMV_SHADER.into()),
        });
        let gemv_pipeline = make_pipeline(&gemv_shader, &matmul_bgl, "gemv_pipe");

        // C-WGPU-Q4K-001: Q4K GEMV — dequantize on-the-fly, no F32 weight buffer
        let q4k_gemv_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("q4k_gemv"),
            source: wgpu::ShaderSource::Wgsl(crate::backends::gpu::shaders::Q4K_GEMV_SHADER.into()),
        });
        let q4k_gemv_pipeline = make_pipeline(&q4k_gemv_shader, &matmul_bgl, "q4k_gemv_pipe");

        let rmsnorm_pipeline = make_pipeline(&rmsnorm_shader, &elementwise_bgl, "rmsnorm_pipe");
        let silu_mul_pipeline = make_pipeline(&silu_mul_shader, &elementwise_bgl, "silu_pipe");
        let residual_pipeline = make_pipeline(&residual_shader_mod, &elementwise_bgl, "res_pipe");

        // RoPE has a 2-binding layout (in-place + uniform)
        let rope_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rope_bgl"),
            entries: &[bgl_storage(0, false), bgl_uniform(1)],
        });
        let rope_pipeline = {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rope_pl"),
                bind_group_layouts: &[&rope_bgl],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("rope_pipe"),
                layout: Some(&pl),
                module: &rope_shader_mod,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        // PMAT-509: Batch RoPE for training (all positions at once)
        let batch_rope_shader_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("batch_rope"),
            source: wgpu::ShaderSource::Wgsl(BATCH_ROPE_SHADER.into()),
        });
        let batch_rope_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("batch_rope_bgl"),
            entries: &[bgl_storage(0, false), bgl_uniform(1)],
        });
        let batch_rope_pipeline = {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("batch_rope_pl"),
                bind_group_layouts: &[&batch_rope_bgl],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("batch_rope_pipe"),
                layout: Some(&pl),
                module: &batch_rope_shader_mod,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        // Allocate persistent intermediate buffers
        let buf = |size: usize, label: &str| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (size * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        // Buffer sizes: max_seq × dim for training, or 1 × dim for inference.
        // Training calls forward_layer_training with seq_len > 1.
        // Allocate for max_seq=2048 to support both.
        let max_seq = 2048;
        let hidden_buf = buf(max_seq * hidden_dim, "hidden");
        let q_buf = buf(max_seq * q_dim, "q");
        let k_buf = buf(max_seq * kv_dim, "k");
        let v_buf = buf(max_seq * kv_dim, "v");
        let attn_out_buf = buf(max_seq * hidden_dim, "attn_out");
        let ffn_gate_buf = buf(max_seq * intermediate_dim, "ffn_gate");
        let ffn_up_buf = buf(max_seq * intermediate_dim, "ffn_up");
        let ffn_silu_buf = buf(max_seq * intermediate_dim, "ffn_silu");
        let ffn_out_buf = buf(max_seq * hidden_dim, "ffn_out");
        let norm_buf = buf(max_seq * hidden_dim, "norm");

        let max_out = max_seq * hidden_dim.max(intermediate_dim);
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (max_out * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            matmul_pipeline,
            tiled_matmul_pipeline,
            attention_pipeline,
            attention_bgl,
            gemv_pipeline,
            q4k_gemv_pipeline,
            rmsnorm_pipeline,
            silu_mul_pipeline,
            rope_pipeline,
            batch_rope_pipeline,
            batch_rope_bgl,
            residual_pipeline,
            matmul_bgl,
            elementwise_bgl,
            weight_buffers: HashMap::new(),
            q4k_weights: HashMap::new(),
            kv_cache_k: Vec::new(),
            kv_cache_v: Vec::new(),
            cpu_biases: HashMap::new(),
            hidden_buf,
            q_buf,
            k_buf,
            v_buf,
            attn_out_buf,
            ffn_gate_buf,
            ffn_up_buf,
            ffn_silu_buf,
            ffn_out_buf,
            norm_buf,
            staging_buf,
            hidden_dim: hidden_dim as u32,
            num_heads: num_heads as u32,
            num_kv_heads: num_kv_heads as u32,
            head_dim: head_dim as u32,
            intermediate_dim: intermediate_dim as u32,
        }
    }

    /// Upload a weight matrix (call once per layer at init).
    /// PMAT-342: Bias weights (name contains "bias") are stored CPU-side.
    pub fn upload_weight(&mut self, name: &str, data: &[f32]) {
        if name.contains("bias") {
            // Biases are small, keep on CPU for easy access in attention
            self.cpu_biases.insert(name.to_string(), data.to_vec());
            return;
        }
        // Skip weights that exceed the device's max buffer binding size (e.g., lm_head > 2 GB)
        let size_bytes = (data.len() * 4) as u64;
        let max_binding = self.device.limits().max_storage_buffer_binding_size as u64;
        if size_bytes > max_binding {
            eprintln!(
                "[wgpu] Skipping weight '{}' ({:.1} MB > {:.1} MB limit) — CPU fallback",
                name,
                size_bytes as f64 / 1e6,
                max_binding as f64 / 1e6
            );
            return;
        }
        use wgpu::util::DeviceExt;
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(name),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        self.weight_buffers.insert(name.to_string(), buffer);
    }

    /// GH-560: Upload raw Q4K weight bytes for fused dequant+GEMV on GPU.
    pub fn upload_q4k_weight(&mut self, name: &str, data: &[u8]) {
        use wgpu::util::DeviceExt;
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(name),
            contents: data,
            usage: wgpu::BufferUsages::STORAGE,
        });
        self.q4k_weights.insert(name.to_string(), buffer);
    }

    /// GH-560: Initialize per-layer KV cache buffers on GPU.
    pub fn init_kv_cache(&mut self, num_layers: usize) {
        let kv_dim = (self.num_kv_heads * self.head_dim) as u64;
        let max_seq = 2048u64;
        for _ in 0..num_layers {
            let k = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("kv_cache_k"),
                size: max_seq * kv_dim * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let v = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("kv_cache_v"),
                size: max_seq * kv_dim * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.kv_cache_k.push(k);
            self.kv_cache_v.push(v);
        }
    }

    /// Number of uploaded weight buffers.
    pub fn weight_count(&self) -> usize {
        self.weight_buffers.len()
    }

    /// Access a dequantized weight buffer by name (e.g. "layer.0.down_proj").
    /// Used by backward pass for gradient propagation through frozen base weights.
    pub fn weight_buffer(&self, name: &str) -> Option<&wgpu::Buffer> {
        self.weight_buffers.get(name)
    }

    /// Reference to the wgpu device.
    pub fn device_ref(&self) -> &wgpu::Device {
        &self.device
    }

    /// Reference to the wgpu queue.
    pub fn queue_ref(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Reference to the hidden state buffer (for writing input).
    pub fn hidden_buffer(&self) -> &wgpu::Buffer {
        &self.hidden_buf
    }

    /// Reference to Q buffer (for LoRA addmm after Q projection).
    pub fn q_buffer(&self) -> &wgpu::Buffer {
        &self.q_buf
    }

    /// Reference to K buffer.
    pub fn k_buffer(&self) -> &wgpu::Buffer {
        &self.k_buf
    }

    /// Reference to V buffer.
    pub fn v_buffer(&self) -> &wgpu::Buffer {
        &self.v_buf
    }

    /// Elementwise add: output = a + b. Dispatches residual add shader.
    pub fn gpu_residual_add(
        &self,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        output: &wgpu::Buffer,
        len: u32,
    ) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.encode_residual(&mut encoder, a, b, output, len);
        self.queue.submit(Some(encoder.finish()));
    }

    /// Apply RMSNorm on GPU: normed = rmsnorm(hidden_buf, weight) → output_buf.
    /// Contract: gpu-output-norm-v1 / gpu_resident — hidden state never leaves GPU.
    pub fn gpu_rmsnorm(&self, weight: &wgpu::Buffer, output: &wgpu::Buffer, _seq_len: u32) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.encode_rmsnorm(&mut encoder, &self.hidden_buf, weight, output, self.hidden_dim);
        self.queue.submit(Some(encoder.finish()));
    }

    /// Download hidden state from GPU.
    pub fn download_hidden(&self, len: usize) -> Vec<f32> {
        let size = (len * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hidden_download"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.hidden_buf, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
        rx.recv()
            .expect("GPU map_async callback channel disconnected")
            .expect("GPU buffer mapping failed");

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data)[..len].to_vec();
        drop(data);
        staging.unmap();
        result
    }

    /// Total VRAM used by all buffers (bytes).
    pub fn total_vram_bytes(&self) -> usize {
        let weight_bytes: usize = self.weight_buffers.values().map(|b| b.size() as usize).sum();
        let intermediate_bytes = (self.hidden_dim as usize * 4) * 4  // hidden, attn_out, ffn_out, norm
            + (self.num_heads as usize * self.head_dim as usize * 4) // q
            + (self.num_kv_heads as usize * self.head_dim as usize * 4) * 2 // k, v
            + (self.intermediate_dim as usize * 4) * 2; // gate, up
        weight_bytes + intermediate_bytes
    }

    /// PMAT-336: Full model forward — embedding + all layers + output norm + LM head.
    ///
    /// Returns logits [vocab_size] for the given token at the given position.
    /// Embedding lookup and final LM head are CPU-side (not yet GPU-accelerated).
    /// PMAT-344: Added kv_caches for multi-token context
    #[provable_contracts_macros::contract("wgpu-forward-pass-v1", equation = "rmsnorm_correctness")]
    pub fn forward_model(
        &self,
        token_id: u32,
        position: usize,
        num_layers: usize,
        token_embedding: &[f32],
        output_norm_weight: &[f32],
        lm_head_weight: &[f32],
        vocab_size: usize,
        eps: f32,
        kv_caches: &mut Vec<(Vec<f32>, Vec<f32>)>,
    ) -> Result<Vec<f32>, String> {
        let hd = self.hidden_dim as usize;

        // 1. Embedding lookup (CPU)
        let embed_start = token_id as usize * hd;
        if embed_start + hd > token_embedding.len() {
            return Err(format!(
                "Token {} out of range (embedding size {})",
                token_id,
                token_embedding.len() / hd
            ));
        }
        let mut hidden: Vec<f32> = token_embedding[embed_start..embed_start + hd].to_vec();

        // 2. Transformer layers (GPU via forward_layer with KV cache)
        // Initialize KV caches if empty
        while kv_caches.len() < num_layers {
            kv_caches.push((Vec::new(), Vec::new()));
        }
        for layer_idx in 0..num_layers {
            let prefix = format!("layer.{layer_idx}");
            let (ref mut k_cache, ref mut v_cache) = kv_caches[layer_idx];
            self.forward_layer(&mut hidden, &prefix, position, k_cache, v_cache)?;
        }

        // 3. Output RMSNorm (CPU — small, not worth GPU dispatch)
        let rms = (hidden.iter().map(|x| x * x).sum::<f32>() / hd as f32 + eps).sqrt();
        for i in 0..hd {
            hidden[i] = (hidden[i] / rms) * output_norm_weight[i];
        }

        // 4. LM head — CPU matmul
        // PMAT-346: GPU tiled GEMM expects weight in [K,N] layout but lm_head is [N,K].
        // CPU path reads weight[v * hd + j] which matches the [vocab, hidden] layout.
        // GPU LM head via GEMV is blocked by vocab > 65535 dispatch limit.
        // Deferred (PMAT-751): add upload_weight_transposed() for GPU-accelerated LM head.
        let mut logits = vec![0.0f32; vocab_size];
        for v in 0..vocab_size {
            let mut sum = 0.0f32;
            let row_start = v * hd;
            for j in 0..hd {
                sum += lm_head_weight[row_start + j] * hidden[j];
            }
            logits[v] = sum;
        }
        Ok(logits)
    }

    /// PMAT-325: Execute one transformer layer — 14 passes, 1 submit, 1 readback.
    ///
    /// Input: hidden state [hidden_dim] on CPU.
    /// Output: updated hidden state [hidden_dim] on CPU.
    /// All intermediate computation stays GPU-resident.
    /// PMAT-344: KV cache parameters for multi-token context
    pub fn forward_layer(
        &self,
        hidden: &mut [f32],
        layer_prefix: &str,
        _position: usize,
        kv_cache_k: &mut Vec<f32>, // accumulated K: [seq_len * kv_dim]
        kv_cache_v: &mut Vec<f32>, // accumulated V: [seq_len * kv_dim]
    ) -> Result<(), String> {
        let hd = self.hidden_dim;

        // Upload hidden state
        self.queue.write_buffer(&self.hidden_buf, 0, bytemuck::cast_slice(hidden));

        let mut encoder = self.device.create_command_encoder(&Default::default());

        // Pass 1: RMSNorm(hidden → norm_buf)
        let norm_w = self
            .weight_buffers
            .get(&format!("{layer_prefix}.attn_norm"))
            .ok_or_else(|| format!("Missing {layer_prefix}.attn_norm"))?;
        self.encode_rmsnorm(&mut encoder, &self.hidden_buf, norm_w, &self.norm_buf, hd);

        // Passes 2-4: Q/K/V projections (norm_buf × W → q/k/v_buf)
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        self.encode_matmul(
            &mut encoder,
            &self.norm_buf,
            layer_prefix,
            "q_proj",
            &self.q_buf,
            1,
            hd,
            q_dim,
        );
        self.encode_matmul(
            &mut encoder,
            &self.norm_buf,
            layer_prefix,
            "k_proj",
            &self.k_buf,
            1,
            hd,
            kv_dim,
        );
        self.encode_matmul(
            &mut encoder,
            &self.norm_buf,
            layer_prefix,
            "v_proj",
            &self.v_buf,
            1,
            hd,
            kv_dim,
        );

        // PMAT-342: Submit Q/K/V projections, readback, do attention on CPU
        // GPU handles the heavy matmuls; CPU handles attention (small at M=1)
        let q_bytes = (q_dim * 4) as u64;
        let kv_bytes = (kv_dim * 4) as u64;

        // Readback Q/K/V from GPU
        let q_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("q_stg"),
            size: q_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let k_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k_stg"),
            size: kv_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let v_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("v_stg"),
            size: kv_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&self.q_buf, 0, &q_staging, 0, q_bytes);
        encoder.copy_buffer_to_buffer(&self.k_buf, 0, &k_staging, 0, kv_bytes);
        encoder.copy_buffer_to_buffer(&self.v_buf, 0, &v_staging, 0, kv_bytes);
        self.queue.submit(Some(encoder.finish()));

        // Readback Q
        let mut q_data = vec![0.0f32; q_dim as usize];
        {
            let slice = q_staging.slice(..q_bytes);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).ok();
            });
            self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
            rx.recv().map_err(|e| format!("q recv: {e}"))?.map_err(|e| format!("q map: {e:?}"))?;
            let data = slice.get_mapped_range();
            q_data.copy_from_slice(&bytemuck::cast_slice::<u8, f32>(&data)[..q_dim as usize]);
        }
        q_staging.unmap();

        // Readback K
        let mut k_data = vec![0.0f32; kv_dim as usize];
        {
            let slice = k_staging.slice(..kv_bytes);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).ok();
            });
            self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
            rx.recv().map_err(|e| format!("k recv: {e}"))?.map_err(|e| format!("k map: {e:?}"))?;
            let data = slice.get_mapped_range();
            k_data.copy_from_slice(&bytemuck::cast_slice::<u8, f32>(&data)[..kv_dim as usize]);
        }
        k_staging.unmap();

        // Readback V
        let mut v_data = vec![0.0f32; kv_dim as usize];
        {
            let slice = v_staging.slice(..kv_bytes);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).ok();
            });
            self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
            rx.recv().map_err(|e| format!("v recv: {e}"))?.map_err(|e| format!("v map: {e:?}"))?;
            let data = slice.get_mapped_range();
            v_data.copy_from_slice(&bytemuck::cast_slice::<u8, f32>(&data)[..kv_dim as usize]);
        }
        v_staging.unmap();

        // PMAT-342: Add QKV biases (required for Qwen2)
        if let Some(q_bias) = self.cpu_biases.get(&format!("{layer_prefix}.q_bias")) {
            for (q, b) in q_data.iter_mut().zip(q_bias.iter()) {
                *q += *b;
            }
        }
        if let Some(k_bias) = self.cpu_biases.get(&format!("{layer_prefix}.k_bias")) {
            for (k, b) in k_data.iter_mut().zip(k_bias.iter()) {
                *k += *b;
            }
        }
        if let Some(v_bias) = self.cpu_biases.get(&format!("{layer_prefix}.v_bias")) {
            for (v, b) in v_data.iter_mut().zip(v_bias.iter()) {
                *v += *b;
            }
        }

        // PMAT-343: Apply RoPE (NeoX-style interleaved) to Q and K
        let head_dim = self.head_dim as usize;
        let position = _position; // Use the position parameter
        let rope_theta = 1_000_000.0f64; // Qwen2 rope_theta

        // RoPE on Q (num_heads × head_dim)
        for h in 0..(self.num_heads as usize) {
            let offset = h * head_dim;
            let half = head_dim / 2;
            for i in 0..half {
                let theta = rope_theta.powf(-((2 * i) as f64) / head_dim as f64);
                let angle = position as f64 * theta;
                let cos_a = angle.cos() as f32;
                let sin_a = angle.sin() as f32;
                let x0 = q_data[offset + i];
                let x1 = q_data[offset + i + half];
                q_data[offset + i] = x0 * cos_a - x1 * sin_a;
                q_data[offset + i + half] = x0 * sin_a + x1 * cos_a;
            }
        }

        // RoPE on K (num_kv_heads × head_dim)
        for h in 0..(self.num_kv_heads as usize) {
            let offset = h * head_dim;
            let half = head_dim / 2;
            for i in 0..half {
                let theta = rope_theta.powf(-((2 * i) as f64) / head_dim as f64);
                let angle = position as f64 * theta;
                let cos_a = angle.cos() as f32;
                let sin_a = angle.sin() as f32;
                let x0 = k_data[offset + i];
                let x1 = k_data[offset + i + half];
                k_data[offset + i] = x0 * cos_a - x1 * sin_a;
                k_data[offset + i + half] = x0 * sin_a + x1 * cos_a;
            }
        }

        // PMAT-344: Append K,V to cache and compute full attention
        let head_dim = self.head_dim as usize;
        let num_heads = self.num_heads as usize;
        let num_kv_heads = self.num_kv_heads as usize;
        let kv_dim_usize = kv_dim as usize;

        kv_cache_k.extend_from_slice(&k_data);
        kv_cache_v.extend_from_slice(&v_data);
        let seq_len = kv_cache_k.len() / kv_dim_usize;

        // Scaled dot-product attention with GQA
        let kv_group = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut attn_out = vec![0.0f32; q_dim as usize];

        for h in 0..num_heads {
            let kv_h = h / kv_group;
            let q_offset = h * head_dim;
            let kv_offset = kv_h * head_dim;

            // Compute attention scores: Q[h] · K[kv_h, :seq_len]^T / sqrt(d)
            let scores = Self::attention_scores(
                &q_data[q_offset..q_offset + head_dim],
                &kv_cache_k,
                kv_offset,
                kv_dim_usize,
                seq_len,
                scale,
            );

            // Weighted sum of V
            let out_offset = h * head_dim;
            Self::attention_weighted_v(
                &scores,
                &kv_cache_v,
                &mut attn_out[out_offset..out_offset + head_dim],
                kv_offset,
                kv_dim_usize,
            );
        }

        // Upload attention output back to GPU for O projection
        self.queue.write_buffer(&self.q_buf, 0, bytemuck::cast_slice(&attn_out));

        // New encoder for remaining passes
        let mut encoder = self.device.create_command_encoder(&Default::default());

        // Pass 7: O projection (attn_out × W_o → attn_out_buf)
        self.encode_matmul(
            &mut encoder,
            &self.q_buf,
            layer_prefix,
            "o_proj",
            &self.attn_out_buf,
            1,
            q_dim,
            hd,
        );

        // Pass 8: Residual(hidden + attn_out → hidden)
        self.encode_residual(
            &mut encoder,
            &self.hidden_buf,
            &self.attn_out_buf,
            &self.ffn_out_buf,
            hd,
        );

        // Pass 9: FFN RMSNorm(ffn_out → norm_buf)
        let ffn_norm_w = self
            .weight_buffers
            .get(&format!("{layer_prefix}.ffn_norm"))
            .ok_or_else(|| format!("Missing {layer_prefix}.ffn_norm"))?;
        self.encode_rmsnorm(&mut encoder, &self.ffn_out_buf, ffn_norm_w, &self.norm_buf, hd);

        // Passes 10-11: Gate + Up projections
        let inter = self.intermediate_dim;
        self.encode_matmul(
            &mut encoder,
            &self.norm_buf,
            layer_prefix,
            "gate_proj",
            &self.ffn_gate_buf,
            1,
            hd,
            inter,
        );
        self.encode_matmul(
            &mut encoder,
            &self.norm_buf,
            layer_prefix,
            "up_proj",
            &self.ffn_up_buf,
            1,
            hd,
            inter,
        );

        // Pass 12: SiLU(gate) × up → ffn_silu_buf [intermediate_dim]
        // BUG FIX: was writing to attn_out_buf (hidden_dim=3584) but needs intermediate_dim=18944.
        // attn_out_buf is only hidden_dim — wgpu robustness silently drops OOB writes,
        // then down_proj reads zeros past hidden_dim → 81% of FFN truncated → garbage output.
        // Cannot alias gate/up buffers (WGSL read/write aliasing UB), so use dedicated buffer.
        self.encode_silu_mul(
            &mut encoder,
            &self.ffn_gate_buf,
            &self.ffn_up_buf,
            &self.ffn_silu_buf,
            inter,
        );

        // Pass 13: Down projection (reads ffn_silu_buf [intermediate_dim] → norm_buf [hidden_dim])
        self.encode_matmul(
            &mut encoder,
            &self.ffn_silu_buf,
            layer_prefix,
            "down_proj",
            &self.norm_buf,
            1,
            inter,
            hd,
        );

        // Pass 14: Residual(ffn_out + down → hidden)
        self.encode_residual(&mut encoder, &self.ffn_out_buf, &self.norm_buf, &self.hidden_buf, hd);

        // Single readback
        encoder.copy_buffer_to_buffer(&self.hidden_buf, 0, &self.staging_buf, 0, (hd * 4) as u64);
        self.queue.submit(Some(encoder.finish()));

        // Readback
        let slice = self.staging_buf.slice(..(hd as u64 * 4));
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
        rx.recv().map_err(|e| format!("recv: {e}"))?.map_err(|e| format!("map: {e:?}"))?;
        {
            let data = slice.get_mapped_range();
            hidden.copy_from_slice(
                &bytemuck::cast_slice::<u8, f32>(&data)[..self.hidden_dim as usize],
            );
        }
        self.staging_buf.unmap();

        Ok(())
    }

    /// Softmaxed attention scores for one head over `seq_len` cached steps.
    ///
    /// `kv_offset` is this head's offset inside each cached step, and each
    /// cached step is `kv_dim` floats wide. Head dimension is `q_head.len()`.
    fn attention_scores(
        q_head: &[f32],
        kv_cache_k: &[f32],
        kv_offset: usize,
        kv_dim: usize,
        seq_len: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut scores = vec![0.0f32; seq_len];
        for (s, score) in scores.iter_mut().enumerate() {
            let k_offset = s * kv_dim + kv_offset;
            let mut dot = 0.0f32;
            for (d, q) in q_head.iter().enumerate() {
                dot += q * kv_cache_k[k_offset + d];
            }
            *score = dot * scale;
        }

        // Softmax
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_score).exp();
            sum += *s;
        }
        if sum > 0.0 {
            for s in scores.iter_mut() {
                *s /= sum;
            }
        }
        scores
    }

    /// Weighted sum of V for one head, written into `out_head`.
    ///
    /// `kv_offset` and `kv_dim` are as in `attention_scores`.
    fn attention_weighted_v(
        scores: &[f32],
        kv_cache_v: &[f32],
        out_head: &mut [f32],
        kv_offset: usize,
        kv_dim: usize,
    ) {
        for (d, out) in out_head.iter_mut().enumerate() {
            let mut val = 0.0f32;
            for (s, score) in scores.iter().enumerate() {
                let v_offset = s * kv_dim + kv_offset;
                val += score * kv_cache_v[v_offset + d];
            }
            *out = val;
        }
    }

    /// Training forward pass for a single transformer layer.
    ///
    /// Unlike `forward_layer` (M=1 decode), this processes the full sequence
    /// at once (M=seq_len) and keeps everything on GPU. No CPU readback.
    ///
    /// Saves `norm_output` (pre-projection activations) for backward pass.
    ///
    /// # Arguments
    /// - `seq_len`: number of tokens in the sequence
    /// - `layer_prefix`: e.g. "model.layers.0"
    /// - `saved_norm_attn`: OUTPUT — saved pre-attention norm for backward (wgpu::Buffer, [seq×hidden])
    /// - `saved_norm_ffn`: OUTPUT — saved pre-FFN norm for backward (wgpu::Buffer, [seq×hidden])
    ///
    /// Forward one layer into an EXISTING encoder (no submit).
    /// Caller batches multiple layers into one encoder, submits once.
    pub fn encode_forward_layer_training(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        seq_len: u32,
        layer_prefix: &str,
        saved: &LayerActivations,
        lora: Option<&QkvLoRA<'_>>,
    ) -> Result<(), String> {
        let hd = self.hidden_dim;
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        let inter = self.intermediate_dim;
        let s = seq_len as usize;

        // Pass 1: RMSNorm
        let norm_w = self
            .weight_buffers
            .get(&format!("{layer_prefix}.attn_norm"))
            .ok_or_else(|| format!("Missing {layer_prefix}.attn_norm"))?;
        self.encode_rmsnorm(encoder, &self.hidden_buf, norm_w, &self.norm_buf, hd);

        // SAVE attn_norm_out
        encoder.copy_buffer_to_buffer(
            &self.norm_buf,
            0,
            &saved.attn_norm_out,
            0,
            (s * hd as usize * 4) as u64,
        );

        // Q/K/V projections
        self.encode_matmul(
            encoder,
            &self.norm_buf,
            layer_prefix,
            "q_proj",
            &self.q_buf,
            seq_len,
            hd,
            q_dim,
        );
        self.encode_matmul(
            encoder,
            &self.norm_buf,
            layer_prefix,
            "k_proj",
            &self.k_buf,
            seq_len,
            hd,
            kv_dim,
        );
        self.encode_matmul(
            encoder,
            &self.norm_buf,
            layer_prefix,
            "v_proj",
            &self.v_buf,
            seq_len,
            hd,
            kv_dim,
        );

        // LoRA addmm on Q/K/V: output += (saved_input @ A) @ B * scale
        // Must happen BEFORE attention consumes Q/K/V buffers.
        if let Some(lora) = lora {
            self.encode_lora_addmm(
                encoder,
                &saved.attn_norm_out,
                lora.q_a,
                lora.q_b,
                &self.q_buf,
                seq_len,
                lora.in_dim,
                lora.rank,
                lora.q_dim,
                lora.scale,
                lora.lora_pipeline,
                lora.lora_bgl,
            );
            self.encode_lora_addmm(
                encoder,
                &saved.attn_norm_out,
                lora.k_a,
                lora.k_b,
                &self.k_buf,
                seq_len,
                lora.in_dim,
                lora.rank,
                lora.kv_dim,
                lora.scale,
                lora.lora_pipeline,
                lora.lora_bgl,
            );
            self.encode_lora_addmm(
                encoder,
                &saved.attn_norm_out,
                lora.v_a,
                lora.v_b,
                &self.v_buf,
                seq_len,
                lora.in_dim,
                lora.rank,
                lora.kv_dim,
                lora.scale,
                lora.lora_pipeline,
                lora.lora_bgl,
            );
        }

        // PMAT-509: Apply QKV biases (required for Qwen2)
        if let Some(q_bias) = self.cpu_biases.get(&format!("{layer_prefix}.q_bias")) {
            self.encode_broadcast_bias(encoder, &self.q_buf, q_bias, seq_len);
        }
        if let Some(k_bias) = self.cpu_biases.get(&format!("{layer_prefix}.k_bias")) {
            self.encode_broadcast_bias(encoder, &self.k_buf, k_bias, seq_len);
        }
        if let Some(v_bias) = self.cpu_biases.get(&format!("{layer_prefix}.v_bias")) {
            self.encode_broadcast_bias(encoder, &self.v_buf, v_bias, seq_len);
        }

        // PMAT-509: Apply RoPE to Q and K before attention.
        self.encode_batch_rope(encoder, &self.q_buf, seq_len, self.num_heads, self.head_dim);
        self.encode_batch_rope(encoder, &self.k_buf, seq_len, self.num_kv_heads, self.head_dim);

        // Attention — wgpu handles execution ordering within the encoder.
        self.encode_attention(encoder, seq_len);

        // SAVE attn_output
        encoder.copy_buffer_to_buffer(
            &self.attn_out_buf,
            0,
            &saved.attn_output,
            0,
            (s * q_dim as usize * 4) as u64,
        );

        // O projection
        self.encode_matmul(
            encoder,
            &self.attn_out_buf,
            layer_prefix,
            "o_proj",
            &self.q_buf,
            seq_len,
            q_dim,
            hd,
        );

        // Residual
        self.encode_residual(
            encoder,
            &self.hidden_buf,
            &self.q_buf,
            &self.ffn_out_buf,
            hd * seq_len,
        );

        // FFN RMSNorm
        let ffn_norm_w = self
            .weight_buffers
            .get(&format!("{layer_prefix}.ffn_norm"))
            .ok_or_else(|| format!("Missing {layer_prefix}.ffn_norm"))?;
        self.encode_rmsnorm(encoder, &self.ffn_out_buf, ffn_norm_w, &self.norm_buf, hd);

        // SAVE ffn_norm_out
        encoder.copy_buffer_to_buffer(
            &self.norm_buf,
            0,
            &saved.ffn_norm_out,
            0,
            (s * hd as usize * 4) as u64,
        );

        // Gate + Up
        self.encode_matmul(
            encoder,
            &self.norm_buf,
            layer_prefix,
            "gate_proj",
            &self.ffn_gate_buf,
            seq_len,
            hd,
            inter,
        );
        self.encode_matmul(
            encoder,
            &self.norm_buf,
            layer_prefix,
            "up_proj",
            &self.ffn_up_buf,
            seq_len,
            hd,
            inter,
        );

        // SiLU
        self.encode_silu_mul(
            encoder,
            &self.ffn_gate_buf,
            &self.ffn_up_buf,
            &self.ffn_silu_buf,
            inter * seq_len,
        );

        // SAVE silu_gate_output
        encoder.copy_buffer_to_buffer(
            &self.ffn_silu_buf,
            0,
            &saved.silu_gate_output,
            0,
            (s * inter as usize * 4) as u64,
        );

        // Down projection
        self.encode_matmul(
            encoder,
            &self.ffn_silu_buf,
            layer_prefix,
            "down_proj",
            &self.norm_buf,
            seq_len,
            inter,
            hd,
        );

        // Residual
        self.encode_residual(
            encoder,
            &self.ffn_out_buf,
            &self.norm_buf,
            &self.hidden_buf,
            hd * seq_len,
        );

        Ok(())
    }

    /// Run one layer with per-operation GPU timing (submit+poll between each op group).
    /// Contract: forward-pass-perf-v1 / bottleneck_identified
    pub fn forward_layer_traced(
        &self,
        seq_len: u32,
        layer_prefix: &str,
        saved: &LayerActivations,
        lora: Option<&QkvLoRA<'_>>,
    ) -> Result<(), String> {
        let hd = self.hidden_dim;
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        let inter = self.intermediate_dim;
        let s = seq_len as usize;

        let norm_w = self
            .weight_buffers
            .get(&format!("{layer_prefix}.attn_norm"))
            .ok_or_else(|| format!("Missing {layer_prefix}.attn_norm"))?;

        let mut trace = Vec::new();
        let mut run = |name: &str, f: &dyn Fn(&mut wgpu::CommandEncoder)| {
            let mut enc = self.device.create_command_encoder(&Default::default());
            f(&mut enc);
            self.queue.submit(Some(enc.finish()));
            let t = std::time::Instant::now();
            self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
            trace.push((name.to_string(), t.elapsed().as_millis() as u64));
        };

        run("rmsnorm1", &|e| self.encode_rmsnorm(e, &self.hidden_buf, norm_w, &self.norm_buf, hd));
        {
            let mut e = self.device.create_command_encoder(&Default::default());
            e.copy_buffer_to_buffer(
                &self.norm_buf,
                0,
                &saved.attn_norm_out,
                0,
                (s * hd as usize * 4) as u64,
            );
            self.queue.submit(Some(e.finish()));
        }
        run("q_proj", &|e| {
            self.encode_matmul(
                e,
                &self.norm_buf,
                layer_prefix,
                "q_proj",
                &self.q_buf,
                seq_len,
                hd,
                q_dim,
            )
        });
        run("k_proj", &|e| {
            self.encode_matmul(
                e,
                &self.norm_buf,
                layer_prefix,
                "k_proj",
                &self.k_buf,
                seq_len,
                hd,
                kv_dim,
            )
        });
        run("v_proj", &|e| {
            self.encode_matmul(
                e,
                &self.norm_buf,
                layer_prefix,
                "v_proj",
                &self.v_buf,
                seq_len,
                hd,
                kv_dim,
            )
        });
        if let Some(lr) = lora {
            run("lora_qkv", &|e| {
                self.encode_lora_addmm(
                    e,
                    &saved.attn_norm_out,
                    lr.q_a,
                    lr.q_b,
                    &self.q_buf,
                    seq_len,
                    lr.in_dim,
                    lr.rank,
                    lr.q_dim,
                    lr.scale,
                    lr.lora_pipeline,
                    lr.lora_bgl,
                );
                self.encode_lora_addmm(
                    e,
                    &saved.attn_norm_out,
                    lr.k_a,
                    lr.k_b,
                    &self.k_buf,
                    seq_len,
                    lr.in_dim,
                    lr.rank,
                    lr.kv_dim,
                    lr.scale,
                    lr.lora_pipeline,
                    lr.lora_bgl,
                );
                self.encode_lora_addmm(
                    e,
                    &saved.attn_norm_out,
                    lr.v_a,
                    lr.v_b,
                    &self.v_buf,
                    seq_len,
                    lr.in_dim,
                    lr.rank,
                    lr.kv_dim,
                    lr.scale,
                    lr.lora_pipeline,
                    lr.lora_bgl,
                );
            });
        }
        // PMAT-509: QKV biases + RoPE before attention
        if let Some(q_bias) = self.cpu_biases.get(&format!("{layer_prefix}.q_bias")) {
            run("q_bias", &|e| self.encode_broadcast_bias(e, &self.q_buf, q_bias, seq_len));
        }
        if let Some(k_bias) = self.cpu_biases.get(&format!("{layer_prefix}.k_bias")) {
            run("k_bias", &|e| self.encode_broadcast_bias(e, &self.k_buf, k_bias, seq_len));
        }
        if let Some(v_bias) = self.cpu_biases.get(&format!("{layer_prefix}.v_bias")) {
            run("v_bias", &|e| self.encode_broadcast_bias(e, &self.v_buf, v_bias, seq_len));
        }
        run("rope_q", &|e| {
            self.encode_batch_rope(e, &self.q_buf, seq_len, self.num_heads, self.head_dim)
        });
        run("rope_k", &|e| {
            self.encode_batch_rope(e, &self.k_buf, seq_len, self.num_kv_heads, self.head_dim)
        });
        run("attention", &|e| self.encode_attention(e, seq_len));
        {
            let mut e = self.device.create_command_encoder(&Default::default());
            e.copy_buffer_to_buffer(
                &self.attn_out_buf,
                0,
                &saved.attn_output,
                0,
                (s * q_dim as usize * 4) as u64,
            );
            self.queue.submit(Some(e.finish()));
        }
        run("o_proj", &|e| {
            self.encode_matmul(
                e,
                &self.attn_out_buf,
                layer_prefix,
                "o_proj",
                &self.q_buf,
                seq_len,
                q_dim,
                hd,
            )
        });
        run("residual1", &|e| {
            self.encode_residual(e, &self.hidden_buf, &self.q_buf, &self.ffn_out_buf, hd * seq_len)
        });
        let ffn_norm_w = self
            .weight_buffers
            .get(&format!("{layer_prefix}.ffn_norm"))
            .ok_or_else(|| format!("Missing {layer_prefix}.ffn_norm"))?;
        run("rmsnorm2", &|e| {
            self.encode_rmsnorm(e, &self.ffn_out_buf, ffn_norm_w, &self.norm_buf, hd)
        });
        {
            let mut e = self.device.create_command_encoder(&Default::default());
            e.copy_buffer_to_buffer(
                &self.norm_buf,
                0,
                &saved.ffn_norm_out,
                0,
                (s * hd as usize * 4) as u64,
            );
            self.queue.submit(Some(e.finish()));
        }
        run("gate_proj", &|e| {
            self.encode_matmul(
                e,
                &self.norm_buf,
                layer_prefix,
                "gate_proj",
                &self.ffn_gate_buf,
                seq_len,
                hd,
                inter,
            )
        });
        run("up_proj", &|e| {
            self.encode_matmul(
                e,
                &self.norm_buf,
                layer_prefix,
                "up_proj",
                &self.ffn_up_buf,
                seq_len,
                hd,
                inter,
            )
        });
        run("silu", &|e| {
            self.encode_silu_mul(
                e,
                &self.ffn_gate_buf,
                &self.ffn_up_buf,
                &self.ffn_silu_buf,
                inter * seq_len,
            )
        });
        {
            let mut e = self.device.create_command_encoder(&Default::default());
            e.copy_buffer_to_buffer(
                &self.ffn_silu_buf,
                0,
                &saved.silu_gate_output,
                0,
                (s * inter as usize * 4) as u64,
            );
            self.queue.submit(Some(e.finish()));
        }
        run("down_proj", &|e| {
            self.encode_matmul(
                e,
                &self.ffn_silu_buf,
                layer_prefix,
                "down_proj",
                &self.norm_buf,
                seq_len,
                inter,
                hd,
            )
        });
        run("residual2", &|e| {
            self.encode_residual(
                e,
                &self.ffn_out_buf,
                &self.norm_buf,
                &self.hidden_buf,
                hd * seq_len,
            )
        });

        let total: u64 = trace.iter().map(|(_, ms)| ms).sum();
        let parts: Vec<String> = trace.iter().map(|(n, ms)| format!("{n}={ms}")).collect();
        eprintln!("[OP-TRACE] layer {} total={}ms: {}", layer_prefix, total, parts.join(" "));
        Ok(())
    }

    /// Allocate saved activations for one layer.
    pub fn alloc_layer_activations(&self, seq_len: u32) -> LayerActivations {
        let s = seq_len as usize;
        let buf = |size: usize, label: &str| -> wgpu::Buffer {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (size * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        LayerActivations {
            attn_norm_out: buf(s * self.hidden_dim as usize, "saved_attn_norm"),
            attn_output: buf(s * (self.num_heads * self.head_dim) as usize, "saved_attn_out"),
            ffn_norm_out: buf(s * self.hidden_dim as usize, "saved_ffn_norm"),
            silu_gate_output: buf(s * self.intermediate_dim as usize, "saved_silu"),
            rstd_attn: buf(s, "saved_rstd_attn"),
            rstd_ffn: buf(s, "saved_rstd_ffn"),
            softmax_logsumexp: buf(self.num_heads as usize * s, "saved_logsumexp"),
        }
    }

    /// Forward one layer with its own encoder + submit (original API, kept for compat).
    pub fn forward_layer_training(
        &self,
        seq_len: u32,
        layer_prefix: &str,
    ) -> Result<LayerActivations, String> {
        let saved = self.alloc_layer_activations(seq_len);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.encode_forward_layer_training(&mut encoder, seq_len, layer_prefix, &saved, None)?;
        self.queue.submit(Some(encoder.finish()));
        Ok(saved)
    }

    /// Forward ALL layers in one encoder submit. 28 layers → 1 GPU sync.
    pub fn forward_all_layers_training(
        &self,
        seq_len: u32,
        num_layers: usize,
    ) -> Result<Vec<LayerActivations>, String> {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let mut all_saved = Vec::with_capacity(num_layers);

        for layer_idx in 0..num_layers {
            let prefix = format!("layer.{layer_idx}");
            let saved = self.alloc_layer_activations(seq_len);
            self.encode_forward_layer_training(&mut encoder, seq_len, &prefix, &saved, None)?;
            all_saved.push(saved);
        }

        // ONE submit for all 28 layers — eliminates 27 GPU sync barriers
        self.queue.submit(Some(encoder.finish()));
        Ok(all_saved)
    }
    // --- Encode helpers (add compute passes to an existing encoder) ---

    /// Encode causal multi-head attention on GPU.
    /// Q: [seq_len, num_heads * head_dim], K/V: [seq_len, num_kv_heads * head_dim]
    /// Output written to q_buf (reused as attn output).
    /// PMAT-509: Add broadcast bias to a [seq_len, dim] buffer.
    /// bias has shape [dim], applied to each of seq_len rows.
    pub fn encode_broadcast_bias(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        buf: &wgpu::Buffer,
        bias: &[f32],
        seq_len: u32,
    ) {
        let dim = bias.len();
        // Create a full-size bias buffer by repeating the bias per position
        let mut full_bias = Vec::with_capacity(seq_len as usize * dim);
        for _ in 0..seq_len {
            full_bias.extend_from_slice(bias);
        }
        let bias_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("broadcast_bias"),
            size: (full_bias.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&bias_buf, 0, bytemuck::cast_slice(&full_bias));

        // Use existing residual: out = buf + bias_buf (into a temp, then copy back)
        let total = seq_len * dim as u32;
        let tmp = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bias_tmp"),
            size: (total as usize * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.encode_residual(encoder, buf, &bias_buf, &tmp, total);
        encoder.copy_buffer_to_buffer(&tmp, 0, buf, 0, (total as u64) * 4);
    }

    /// PMAT-509: Encode batch RoPE for all positions in a sequence.
    /// Applies position-dependent rotation to Q or K buffer in-place.
    fn encode_batch_rope(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        qk_buf: &wgpu::Buffer,
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
    ) {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct RopeParams {
            seq_len: u32,
            num_heads: u32,
            head_dim: u32,
            _pad: u32,
        }
        let params = RopeParams { seq_len, num_heads, head_dim, _pad: 0 };
        let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("batch_rope_params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("batch_rope_bg"),
            layout: &self.batch_rope_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: qk_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: params_buf.as_entire_binding() },
            ],
        });
        let total = seq_len * num_heads * head_dim;
        let wg = total.div_ceil(256);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.batch_rope_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(wg, 1, 1);
    }

    fn encode_attention(&self, encoder: &mut wgpu::CommandEncoder, seq_len: u32) {
        let params = [seq_len, self.num_heads, self.num_kv_heads, self.head_dim];
        let params_buf = self.make_uniform(&params);
        let _q_dim = self.num_heads * self.head_dim;

        // Attention reads Q and writes to attn_out_buf.
        // Then O projection reads attn_out_buf → writes to another buffer.
        // We can safely write to norm_buf here since it's not read during attention.
        // After attention, we'll copy norm_buf → q_buf for the O projection to read.
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.attention_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.k_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.v_buf.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.attn_out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.attention_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        // One workgroup per (head, position)
        pass.dispatch_workgroups(self.num_heads, seq_len, 1);
    }

    /// Encode LoRA addmm: output += (input @ A) @ B * scale
    ///
    /// KAIZEN: replaced fused shader (0.11 GFLOPS) with two tiled GEMM dispatches (1000+ GFLOPS).
    /// Step 1: temp = input @ A  [seq, rank] via tiled GEMM
    /// Step 2: output += scale * (temp @ B) [seq, out_dim] via tiled GEMM with alpha=scale
    ///
    /// The second GEMM uses alpha=scale in the tiled GEMM shader (C = alpha * A @ B).
    /// But we need ADD (+=), not overwrite (=). We use a temp buffer for the delta,
    /// then add to output via an elementwise shader.
    #[allow(clippy::too_many_arguments)]
    fn encode_lora_addmm(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        lora_a: &wgpu::Buffer,
        lora_b: &wgpu::Buffer,
        output: &wgpu::Buffer,
        seq_len: u32,
        in_dim: u32,
        rank: u32,
        out_dim: u32,
        scale: f32,
        _pipeline: &wgpu::ComputePipeline,
        _bgl: &wgpu::BindGroupLayout,
    ) {
        // Step 1: temp[seq, rank] = input[seq, in_dim] @ A[in_dim, rank]
        let temp_size = (seq_len * rank) as u64 * 4;
        let temp = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lora_temp"),
            size: temp_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.encode_tiled_gemm(encoder, input, lora_a, &temp, seq_len, in_dim, rank, 1.0);

        // Step 2: delta[seq, out_dim] = scale * temp[seq, rank] @ B[rank, out_dim]
        let delta_size = (seq_len * out_dim) as u64 * 4;
        let delta = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lora_delta"),
            size: delta_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.encode_tiled_gemm(encoder, &temp, lora_b, &delta, seq_len, rank, out_dim, scale);

        // Step 3: output += delta (elementwise add, via temp to avoid aliasing)
        let sum_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lora_sum"),
            size: delta_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.encode_residual(encoder, output, &delta, &sum_buf, seq_len * out_dim);
        encoder.copy_buffer_to_buffer(&sum_buf, 0, output, 0, delta_size);
    }

    /// Encode tiled GEMM: C = alpha * A[M,K] @ B[K,N]. Uses CUTLASS-style 64×64 tiles.
    fn encode_tiled_gemm(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        c: &wgpu::Buffer,
        m: u32,
        k: u32,
        n: u32,
        alpha: f32,
    ) {
        let params = [m, k, n, alpha.to_bits()];
        let params_buf = self.make_uniform(&params);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.matmul_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: c.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.tiled_matmul_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(n.div_ceil(64), m.div_ceil(64), 1);
    }

    fn encode_rmsnorm(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        weight: &wgpu::Buffer,
        output: &wgpu::Buffer,
        dim: u32,
    ) {
        let params = [dim, 0u32, 0, 0];
        let params_buf = self.make_uniform(&params);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.elementwise_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: weight.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });
        // Dispatch: (1, num_rows, 1). Each workgroup processes one row via wg_id.y.
        // For inference (M=1): dispatch (1,1,1). For training (M=seq_len): dispatch (1,seq_len,1).
        let num_rows = (input.size() / (dim as u64 * 4)).max(1) as u32;
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.rmsnorm_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(1, num_rows, 1);
    }

    fn encode_matmul(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        layer_prefix: &str,
        proj_name: &str,
        output: &wgpu::Buffer,
        m: u32,
        k: u32,
        n: u32,
    ) {
        // C-WGPU-Q4K-001: Try Q4K GEMV first for M=1 decode (7x less VRAM)
        if m == 1 && self.encode_q4k_gemv(encoder, input, output, layer_prefix, proj_name, n, k) {
            return;
        }
        let weight_key = format!("{layer_prefix}.{proj_name}");
        let weight = match self.weight_buffers.get(&weight_key) {
            Some(w) => w,
            None => return, // Skip missing weights silently
        };
        // PMAT-346: GEMV and matmul have different uniform struct layouts.
        // GEMV: Params { n (output dim), k (input dim), _, _ }
        // Matmul: Dimensions { M, K, N, _ }
        // Tiled GEMM: Dimensions { M, K, N, alpha_bits }
        let params = if m == 1 { [n, k, 0u32, 0u32] } else { [m, k, n, 1.0_f32.to_bits()] };
        let params_buf = self.make_uniform(&params);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.matmul_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: weight.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        if m == 1 {
            // PMAT-327: GEMV for M=1 — cooperative K-reduction, N workgroups
            pass.set_pipeline(&self.gemv_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n, 1, 1);
        } else if m >= 4 {
            // CUTLASS-style tiled GEMM for M>=4 (training batch, prefill)
            // 64×64 tiles, 4×4 thread micro-tiles, 10-30x faster than naive
            pass.set_pipeline(&self.tiled_matmul_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(64), m.div_ceil(64), 1);
        } else {
            // Naive 16×16 GEMM for small M (2-3)
            pass.set_pipeline(&self.matmul_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(m.div_ceil(16), n.div_ceil(16), 1);
        }
    }

    /// C-WGPU-Q4K-001: Encode Q4K GEMV — reads raw Q4K weight bytes, dequantizes on-the-fly.
    /// Falls back to F32 GEMV if no Q4K weight found for this layer.
    /// Returns true if Q4K path was used.
    fn encode_q4k_gemv(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        layer_prefix: &str,
        proj_name: &str,
        n: u32,
        k: u32,
    ) -> bool {
        let weight_key = format!("{layer_prefix}.{proj_name}");
        let weight = match self.q4k_weights.get(&weight_key) {
            Some(w) => w,
            None => return false,
        };
        let num_superblocks = (k + 255) / 256;
        let params = [n, k, num_superblocks, 0u32];
        let params_buf = self.make_uniform(&params);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.matmul_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: weight.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.q4k_gemv_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(n, 1, 1);
        true
    }

    fn encode_silu_mul(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gate: &wgpu::Buffer,
        up: &wgpu::Buffer,
        output: &wgpu::Buffer,
        dim: u32,
    ) {
        let params = [dim, 0u32, 0, 0];
        let params_buf = self.make_uniform(&params);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.elementwise_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gate.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: up.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.silu_mul_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(dim.div_ceil(256), 1, 1);
    }

    fn encode_residual(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        output: &wgpu::Buffer,
        dim: u32,
    ) {
        let params = [dim, 0u32, 0, 0];
        let params_buf = self.make_uniform(&params);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.elementwise_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.residual_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(dim.div_ceil(256), 1, 1);
    }

    fn make_uniform(&self, data: &[u32; 4]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }
}

fn bgl_storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
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
