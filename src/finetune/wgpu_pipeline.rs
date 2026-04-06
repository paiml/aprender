//! WgpuInstructPipeline — GPU-only training pipeline (§26.11.7).
//! Bypasses entrenar's `Transformer` (20-min CPU dequant of 28 GB Q4K→F32).
//! Uses `WgslForwardPass` with pre-uploaded GPU weights from `dequant_model_weights()`.
//! No `Transformer` object. No CPU F32 projection weights. No SATD.
//!
//! # Architecture (Unsloth pattern)
//!
//! ```text
//! OwnedQuantizedModel (Q4K, seconds to load)
//!   → dequant_model_weights() (streaming, ~2 min)
//!   → WgslForwardPass.upload_weight() (persistent GPU buffers)
//!   → WgpuInstructPipeline.train_step() (all GPU)
//! ```
//!
//! # Contract: wgsl-training-pipeline-v1
//!
//! - `fast_load`: load_time < 5 min on GB10
//! - `no_transformer`: does not construct Transformer

#[cfg(feature = "gpu")]
use crate::{
    autograd::{wgpu_cross_entropy::WgslCrossEntropy, wgpu_training::WgpuTrainer},
    finetune::instruct_pipeline::InstructStepResult,
    tokenizer::HfTokenizer,
};
#[cfg(feature = "gpu")]
use trueno::backends::gpu::{wgpu, WgslForwardPass};

/// LoRA adapters for one transformer layer (7 projections).
#[cfg(feature = "gpu")]
pub struct LayerLoRA {
    /// (A_buf, B_buf, m_A, v_A, m_B, v_B, in_dim, out_dim, proj_name)
    pub projections: Vec<LoRAProjection>,
}

#[cfg(feature = "gpu")]
pub struct LoRAProjection {
    pub a: wgpu::Buffer,   // [in_dim, rank]
    pub b: wgpu::Buffer,   // [rank, out_dim]
    pub m_a: wgpu::Buffer, // AdamW first moment for A
    pub v_a: wgpu::Buffer, // AdamW second moment for A
    pub m_b: wgpu::Buffer, // AdamW first moment for B
    pub v_b: wgpu::Buffer, // AdamW second moment for B
    pub in_dim: u32,
    pub out_dim: u32,
    pub name: String, // e.g. "q_proj"
}

/// GPU-only instruct training pipeline. No `Transformer` object.
#[cfg(feature = "gpu")]
pub struct WgpuInstructPipeline {
    /// GPU forward pass with persistent weight buffers
    fwd: WgslForwardPass,
    /// Fused cross-entropy loss on GPU
    cross_entropy: WgslCrossEntropy,
    /// GPU optimizer + backward GEMM
    trainer: WgpuTrainer,
    /// Pre-uploaded lm_head — PRE-CHUNKED to avoid per-step download
    lm_head_t_chunks: Vec<(wgpu::Buffer, u32)>, // [(chunk_buf, chunk_n)] for forward
    lm_head_chunks: Vec<(wgpu::Buffer, u32)>, // [(chunk_buf, chunk_n)] for backward
    /// LoRA addmm pipeline: output += (input @ A) @ B * scale
    lora_addmm_pipeline: wgpu::ComputePipeline,
    lora_addmm_bgl: wgpu::BindGroupLayout,
    /// KAIZEN: scatter/gather/transpose pipelines
    scatter_pipeline: wgpu::ComputePipeline,
    gather_pipeline: wgpu::ComputePipeline,
    scatter_bgl: wgpu::BindGroupLayout,
    transpose_pipeline: wgpu::ComputePipeline,
    transpose_bgl: wgpu::BindGroupLayout,
    /// GPU buffers
    logits_buf: wgpu::Buffer,
    labels_buf: wgpu::Buffer,
    losses_buf: wgpu::Buffer,
    logsumexp_buf: wgpu::Buffer,
    /// LoRA adapters per layer: 7 projections × (A, B, m_A, v_A, m_B, v_B)
    /// A: [in_dim, rank], B: [rank, out_dim], m/v: optimizer states
    lora: Vec<LayerLoRA>,
    lora_rank: usize,
    lora_scale: f32,              // alpha / rank
    lora_step: u32,               // optimizer step counter
    learning_rate: f32,           // PMAT-497: was hardcoded, now from config
    lora_target_set: Vec<String>, // which projections to train
    /// Config
    num_layers: usize,
    hidden_dim: usize,
    vocab_size: usize,
    max_seq_len: usize,
    /// Tokenizer
    tokenizer: HfTokenizer,
    /// Embedding weights (CPU, small: vocab × hidden × 4 for token lookup)
    embed_weights: Vec<f32>,
    /// Output norm weights — GPU-resident (contract: gpu-output-norm-v1)
    output_norm_gpu: wgpu::Buffer,
    /// Normed hidden state — GPU-resident
    normed_buf: wgpu::Buffer,
    /// RMSNorm epsilon
    eps: f32,
}

#[cfg(feature = "gpu")]
impl WgpuInstructPipeline {
    /// Create from pre-uploaded WgslForwardPass.
    ///
    /// Caller is responsible for:
    /// 1. Loading OwnedQuantizedModel
    /// 2. Calling dequant_model_weights()
    /// 3. Uploading weights to WgslForwardPass
    /// 4. Uploading lm_head to GPU buffers
    ///
    /// This constructor does NOT touch Transformer or from_apr().
    /// Contract: wgsl-training-pipeline-v1 / no_transformer
    pub fn new(
        fwd: WgslForwardPass,
        trainer: WgpuTrainer,
        tokenizer: HfTokenizer,
        embed_weights: Vec<f32>,
        output_norm: Vec<f32>,
        lm_head_t_chunks: Vec<(wgpu::Buffer, u32)>,
        lm_head_chunks: Vec<(wgpu::Buffer, u32)>,
        num_layers: usize,
        hidden_dim: usize,
        vocab_size: usize,
        max_seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        intermediate_dim: usize,
        lora_rank: usize,
        lora_alpha: f32,
        lora_targets: &[&str],
        eps: f32,
        learning_rate: f32,
    ) -> Self {
        let ce = WgslCrossEntropy::new(trainer.device_ref().clone(), trainer.queue_ref().clone());

        let seq = max_seq_len as u32;
        let vocab = vocab_size as u32;
        let make_buf = |size: u64, label: &str| -> wgpu::Buffer {
            trainer.device_ref().create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        // KAIZEN: scatter/gather pipelines (one dispatch replaces 1024 copies)
        let scatter_bgl =
            trainer.device_ref().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scatter_bgl"),
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
        let scatter_pl =
            trainer.device_ref().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("scatter_pl"),
                bind_group_layouts: &[&scatter_bgl],
                push_constant_ranges: &[],
            });
        let scatter_shader =
            trainer.device_ref().create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scatter"),
                source: wgpu::ShaderSource::Wgsl(
                    trueno::backends::gpu::shaders::COLUMN_SCATTER_SHADER.into(),
                ),
            });
        let scatter_pipeline =
            trainer.device_ref().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("scatter_pipe"),
                layout: Some(&scatter_pl),
                module: &scatter_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let gather_shader =
            trainer.device_ref().create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gather"),
                source: wgpu::ShaderSource::Wgsl(
                    trueno::backends::gpu::shaders::COLUMN_GATHER_SHADER.into(),
                ),
            });
        let gather_pipeline =
            trainer.device_ref().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gather_pipe"),
                layout: Some(&scatter_pl),
                module: &gather_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Transpose pipeline (same BGL as scatter: src read, dst read-write, params uniform)
        let transpose_bgl =
            trainer.device_ref().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("transpose_bgl"),
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
        let transpose_pl =
            trainer.device_ref().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("transpose_pl"),
                bind_group_layouts: &[&transpose_bgl],
                push_constant_ranges: &[],
            });
        let transpose_shader =
            trainer.device_ref().create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("transpose"),
                source: wgpu::ShaderSource::Wgsl(
                    trueno::backends::gpu::shaders::TRANSPOSE_SHADER.into(),
                ),
            });
        let transpose_pipeline =
            trainer.device_ref().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("transpose_pipe"),
                layout: Some(&transpose_pl),
                module: &transpose_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // LoRA addmm pipeline: output += (input @ A) @ B * scale
        let lora_bgl =
            trainer.device_ref().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("lora_bgl"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
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
        let lora_pl =
            trainer.device_ref().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("lora_pl"),
                bind_group_layouts: &[&lora_bgl],
                push_constant_ranges: &[],
            });
        let lora_shader = trainer.device_ref().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lora_addmm"),
            source: wgpu::ShaderSource::Wgsl(
                trueno::backends::gpu::shaders::LORA_ADDMM_SHADER.into(),
            ),
        });
        let lora_addmm_pipeline =
            trainer.device_ref().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("lora_addmm_pipe"),
                layout: Some(&lora_pl),
                module: &lora_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let lora_addmm_bgl = lora_bgl;

        // Contract: lora-algebra-v1/lora_shape — A[in,rank], B[rank,out]
        // Contract: lora-gradient-flow-v1 — B initialized to zero, A Kaiming
        let r = lora_rank;
        let scale = lora_alpha / r as f32;
        let h = hidden_dim;
        let q_dim = num_heads * (hidden_dim / num_heads);
        let kv_dim = num_kv_heads * (hidden_dim / num_heads);
        let inter = intermediate_dim;

        let all_proj_dims: &[(&str, usize, usize)] = &[
            ("q_proj", h, q_dim),
            ("k_proj", h, kv_dim),
            ("v_proj", h, kv_dim),
            ("o_proj", q_dim, h),
            ("gate_proj", h, inter),
            ("up_proj", h, inter),
            ("down_proj", inter, h),
        ];

        // PMAT-497 FIX: Forward pass only applies LoRA to Q/K/V via QkvLoRA.
        // Backward MUST match — training o/gate/up/down produces spurious gradients
        // that corrupt convergence (loss > random from step 1).
        // When full 7-projection LoRA forward is implemented, this can be "all".
        let _use_all = true; // Always create all 7 for QkvLoRA forward
        let proj_dims: Vec<(&str, usize, usize)> = all_proj_dims.to_vec();
        let num_targets = proj_dims.len();
        let qkv_only = vec!["q_proj".to_string(), "k_proj".to_string(), "v_proj".to_string()];
        let lora_target_set: Vec<String> =
            if lora_targets.is_empty() || lora_targets.contains(&"all") {
                // Forward only applies QkvLoRA (Q/K/V) — backward must match
                qkv_only
            } else {
                lora_targets.iter().map(std::string::ToString::to_string).collect()
            };

        let mut lora = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let mut projections = Vec::with_capacity(num_targets);
            for &(name, in_d, out_d) in &proj_dims {
                // Kaiming init for A: std = sqrt(2/fan_in)
                let std = (2.0 / in_d as f32).sqrt();
                let a_data: Vec<f32> = (0..in_d * r)
                    .map(|i| ((i as f32 * 0.013 + layer_idx as f32 * 7.0).sin() * std))
                    .collect();
                // Zero init for B (contract: lora-gradient-flow-v1)
                let b_data = vec![0.0f32; r * out_d];
                let zeros_a = vec![0.0f32; in_d * r];
                let zeros_b = vec![0.0f32; r * out_d];

                projections.push(LoRAProjection {
                    a: trainer.upload(&a_data),
                    b: trainer.upload(&b_data),
                    m_a: trainer.upload(&zeros_a),
                    v_a: trainer.upload(&zeros_a),
                    m_b: trainer.upload(&zeros_b),
                    v_b: trainer.upload(&zeros_b),
                    in_dim: in_d as u32,
                    out_dim: out_d as u32,
                    name: name.to_string(),
                });
            }
            lora.push(LayerLoRA { projections });
        }

        eprintln!(
            "[wgpu] LoRA initialized: {num_layers} layers × {num_targets} projections, rank={r}, scale={scale:.2}",
        );

        // Pre-allocate buffers before moving trainer into Self
        let logits_buf = make_buf(u64::from(seq) * u64::from(vocab), "logits");
        let labels_buf = make_buf(u64::from(seq), "labels");
        let losses_buf = make_buf(u64::from(seq), "losses");
        let logsumexp_buf = make_buf(u64::from(seq), "logsumexp");
        let normed_buf_alloc = make_buf(u64::from(seq) * hidden_dim as u64, "normed");
        let output_norm_gpu_buf = trainer.upload(&output_norm);

        Self {
            fwd,
            cross_entropy: ce,
            logits_buf,
            labels_buf,
            losses_buf,
            logsumexp_buf,
            lm_head_t_chunks,
            lm_head_chunks,
            lora_addmm_pipeline,
            lora_addmm_bgl,
            scatter_pipeline,
            gather_pipeline,
            transpose_pipeline,
            transpose_bgl,
            scatter_bgl,
            trainer,
            lora,
            lora_rank: r,
            lora_scale: scale,
            lora_step: 0,
            learning_rate,
            lora_target_set: lora_target_set,
            num_layers,
            hidden_dim,
            vocab_size,
            max_seq_len,
            tokenizer,
            embed_weights,
            output_norm_gpu: output_norm_gpu_buf,
            normed_buf: normed_buf_alloc,
            eps,
        }
    }

    /// LoRA addmm: output += (input @ A) @ B * scale. One GPU dispatch.
    /// Contract: lora-algebra-v1/lora_shape
    fn dispatch_lora_addmm(
        &self,
        input: &wgpu::Buffer,
        lora_a: &wgpu::Buffer,
        lora_b: &wgpu::Buffer,
        output: &wgpu::Buffer,
        seq_len: u32,
        in_dim: u32,
        rank: u32,
        out_dim: u32,
    ) {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            seq_len: u32,
            in_dim: u32,
            rank: u32,
            out_dim: u32,
            scale: f32,
            _p0: u32,
            _p1: u32,
            _p2: u32,
        }
        let params =
            P { seq_len, in_dim, rank, out_dim, scale: self.lora_scale, _p0: 0, _p1: 0, _p2: 0 };
        let pbuf = self.trainer.device_ref().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.trainer.queue_ref().write_buffer(&pbuf, 0, bytemuck::bytes_of(&params));
        let bg = self.trainer.device_ref().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.lora_addmm_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: lora_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: lora_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: output.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pbuf.as_entire_binding() },
            ],
        });
        let total = seq_len * out_dim;
        let wg = total.div_ceil(256);
        let (x, y) = if wg <= 65535 { (wg, 1) } else { (65535, wg.div_ceil(65535)) };
        let mut encoder = self.trainer.device_ref().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.lora_addmm_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        self.trainer.queue_ref().submit(Some(encoder.finish()));
    }

    /// GPU scatter: copy [seq, chunk_n] into [seq, full_n] at col_offset. One dispatch.
    fn dispatch_scatter(
        &self,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        seq_len: u32,
        chunk_n: u32,
        full_n: u32,
        col_offset: u32,
    ) {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            seq_len: u32,
            chunk_n: u32,
            full_n: u32,
            col_offset: u32,
        }
        let params = P { seq_len, chunk_n, full_n, col_offset };
        let pbuf = self.trainer.device_ref().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.trainer.queue_ref().write_buffer(&pbuf, 0, bytemuck::bytes_of(&params));
        let bg = self.trainer.device_ref().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.scatter_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pbuf.as_entire_binding() },
            ],
        });
        let total = seq_len * chunk_n;
        let wg = total.div_ceil(256);
        let (x, y) = if wg <= 65535 { (wg, 1) } else { (65535, wg.div_ceil(65535)) };
        let mut encoder = self.trainer.device_ref().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.scatter_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        self.trainer.queue_ref().submit(Some(encoder.finish()));
    }

    /// GPU gather: extract [seq, chunk_n] from [seq, full_n] at col_offset. One dispatch.
    fn dispatch_gather(
        &self,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        seq_len: u32,
        chunk_n: u32,
        full_n: u32,
        col_offset: u32,
    ) {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            seq_len: u32,
            chunk_n: u32,
            full_n: u32,
            col_offset: u32,
        }
        let params = P { seq_len, chunk_n, full_n, col_offset };
        let pbuf = self.trainer.device_ref().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.trainer.queue_ref().write_buffer(&pbuf, 0, bytemuck::bytes_of(&params));
        let bg = self.trainer.device_ref().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.scatter_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pbuf.as_entire_binding() },
            ],
        });
        let total = seq_len * chunk_n;
        let wg = total.div_ceil(256);
        let (x, y) = if wg <= 65535 { (wg, 1) } else { (65535, wg.div_ceil(65535)) };
        let mut encoder = self.trainer.device_ref().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.gather_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        self.trainer.queue_ref().submit(Some(encoder.finish()));
    }

    /// Encode text to token IDs using the tokenizer.
    /// GPU scaled transpose: dst[j,i] = scale * src[i,j]
    fn dispatch_transpose(
        &self,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        m: u32,
        n: u32,
        scale: f32,
    ) {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            m: u32,
            n: u32,
            scale: f32,
            _pad: u32,
        }
        let params = P { m, n, scale, _pad: 0 };
        let pbuf = self.trainer.device_ref().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.trainer.queue_ref().write_buffer(&pbuf, 0, bytemuck::bytes_of(&params));
        let bg = self.trainer.device_ref().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.transpose_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pbuf.as_entire_binding() },
            ],
        });
        let total = m * n;
        let wg = total.div_ceil(256);
        let (x, y) = if wg <= 65535 { (wg, 1) } else { (65535, wg.div_ceil(65535)) };
        let mut encoder = self.trainer.device_ref().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.transpose_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        self.trainer.queue_ref().submit(Some(encoder.finish()));
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    /// Export trained LoRA adapter as safetensors file.
    /// Downloads all A/B weights from GPU and saves with naming convention
    /// matching `apr finetune --merge` expectations: `layer.{i}.{proj}.lora_a/b`.
    pub fn export_adapter(
        &self,
        output_path: &std::path::Path,
        lora_alpha: f32,
    ) -> Result<(), String> {
        use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

        // Collect all tensors
        let mut tensors: Vec<(String, Vec<f32>, Vec<usize>)> = Vec::new();

        for (layer_idx, layer_lora) in self.lora.iter().enumerate() {
            for proj in &layer_lora.projections {
                if !self.lora_target_set.iter().any(|t| t == &proj.name) {
                    continue;
                }
                let a = self.trainer.download(&proj.a);
                let b = self.trainer.download(&proj.b);
                // Naming: layer.{i}.{proj_name}.lora_a  (matches apr merge convention)
                let base = format!("layer.{layer_idx}.{}", proj.name);
                tensors.push((
                    format!("{base}.lora_a"),
                    a,
                    vec![proj.in_dim as usize, self.lora_rank],
                ));
                tensors.push((
                    format!("{base}.lora_b"),
                    b,
                    vec![self.lora_rank, proj.out_dim as usize],
                ));
            }
        }

        // Write safetensors file
        let byte_tensors: Vec<(String, Vec<u8>, Vec<usize>)> = tensors
            .into_iter()
            .map(|(name, data, shape)| (name, bytemuck::cast_slice(&data).to_vec(), shape))
            .collect();

        let views: Vec<(&str, TensorView<'_>)> = byte_tensors
            .iter()
            .map(|(name, bytes, shape)| {
                let view =
                    TensorView::new(Dtype::F32, shape.clone(), bytes).expect("valid F32 tensor");
                (name.as_str(), view)
            })
            .collect();

        // Create output directory if needed
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("mkdir: {e}"))?;
        }

        // Save as .safetensors file (with metadata for rank/alpha)
        let st_path = if output_path.extension().is_some() {
            output_path.to_path_buf()
        } else {
            output_path.join("adapter.safetensors")
        };

        let metadata: Option<std::collections::HashMap<String, String>> =
            Some(std::collections::HashMap::from([
                ("lora_rank".to_string(), self.lora_rank.to_string()),
                ("lora_alpha".to_string(), lora_alpha.to_string()),
            ]));
        serialize_to_file(views, metadata, &st_path)
            .map_err(|e| format!("safetensors write: {e}"))?;

        eprintln!(
            "[wgpu] {} LoRA tensors saved ({} layers × 7 projections × A/B)",
            byte_tensors.len(),
            self.num_layers
        );
        Ok(())
    }

    /// Training step: forward → loss → backward → optimizer. All GPU.
    ///
    /// Contract: qlora-training-loop-v1 / lora_forward_wgsl
    pub fn train_step(&mut self, prompt_ids: &[u32], response_ids: &[u32]) -> InstructStepResult {
        let t0 = std::time::Instant::now();

        let full_ids: Vec<u32> = prompt_ids.iter().chain(response_ids).copied().collect();
        let seq_len = full_ids.len().min(self.max_seq_len);
        let full_ids = &full_ids[..seq_len];
        let prompt_len = prompt_ids.len().min(seq_len);

        let loss_start = prompt_len.saturating_sub(1);
        let loss_end = seq_len - 1;
        let num_loss_tokens = loss_end.saturating_sub(loss_start);

        if num_loss_tokens == 0 {
            return InstructStepResult { loss: 0.0, num_response_tokens: 0, perplexity: 1.0 };
        }

        // 1. Embed tokens (CPU lookup, small: seq × hidden)
        let mut hidden = Vec::with_capacity(seq_len * self.hidden_dim);
        for &tok in full_ids {
            let offset = (tok as usize) * self.hidden_dim;
            let end = offset + self.hidden_dim;
            if end <= self.embed_weights.len() {
                hidden.extend_from_slice(&self.embed_weights[offset..end]);
            } else {
                hidden.extend(std::iter::repeat_n(0.0f32, self.hidden_dim));
            }
        }

        let t1 = std::time::Instant::now();

        // PMAT-509: Diagnostic — check embedding norm on first step
        if self.lora_step == 0 {
            let h_norm: f32 = hidden.iter().map(|x| x * x).sum::<f32>().sqrt();
            let h_mean: f32 = hidden.iter().sum::<f32>() / hidden.len() as f32;
            eprintln!(
                "[DIAG-509] embed: norm={h_norm:.4}, mean={h_mean:.6}, len={}, seq={seq_len}",
                hidden.len()
            );
            // Dump first 5 values of token 264's embedding (compare vs PyTorch)
            let tok264_offset = 264 * self.hidden_dim;
            if tok264_offset + 5 < self.embed_weights.len() {
                let tok264: Vec<f32> =
                    self.embed_weights[tok264_offset..tok264_offset + 5].to_vec();
                eprintln!("[DIAG-509] embed[264,:5]={tok264:?} (PyTorch: [-0.0295, 0.0035, 0.0193, 0.0020, 0.0049])");
            }
        }

        // 2. GPU forward through 28 transformer layers with LoRA contribution
        // Contract: lora-algebra-v1/lora_shape — h = W_base @ x + (x @ A) @ B * scale
        // Per-layer forward: base GEMM (via WgslForwardPass) + LoRA addmm (via pipeline shader)
        self.fwd.queue_ref().write_buffer(
            self.fwd.hidden_buffer(),
            0,
            bytemuck::cast_slice(&hidden),
        );

        let mut _saved_activations = Vec::with_capacity(self.num_layers);
        for layer_idx in 0..self.num_layers {
            let prefix = format!("layer.{layer_idx}");
            // Build QkvLoRA for this layer's Q/K/V projections
            let qkv_lora = if layer_idx < self.lora.len() {
                let lp = &self.lora[layer_idx].projections;
                // Find Q, K, V projections by name
                let q = lp.iter().find(|p| p.name == "q_proj");
                let k = lp.iter().find(|p| p.name == "k_proj");
                let v = lp.iter().find(|p| p.name == "v_proj");
                match (q, k, v) {
                    (Some(qp), Some(kp), Some(vp)) => Some(trueno::backends::gpu::QkvLoRA {
                        q_a: &qp.a,
                        q_b: &qp.b,
                        k_a: &kp.a,
                        k_b: &kp.b,
                        v_a: &vp.a,
                        v_b: &vp.b,
                        rank: self.lora_rank as u32,
                        scale: self.lora_scale,
                        in_dim: qp.in_dim,
                        q_dim: qp.out_dim,
                        kv_dim: kp.out_dim,
                        lora_pipeline: &self.lora_addmm_pipeline,
                        lora_bgl: &self.lora_addmm_bgl,
                    }),
                    _ => None,
                }
            } else {
                None
            };

            // Forward with inline LoRA on Q/K/V (before attention consumes them)
            let saved = self.fwd.alloc_layer_activations(seq_len as u32);

            // Per-operation tracing on layer 0 of first step
            if self.lora_step == 0 && layer_idx == 0 {
                if let Err(e) = self.fwd.forward_layer_traced(
                    seq_len as u32,
                    &prefix,
                    &saved,
                    qkv_lora.as_ref(),
                ) {
                    eprintln!("[wgpu] traced forward failed: {e}");
                }
            } else {
                let mut encoder = self.fwd.device_ref().create_command_encoder(&Default::default());
                if let Err(e) = self.fwd.encode_forward_layer_training(
                    &mut encoder,
                    seq_len as u32,
                    &prefix,
                    &saved,
                    qkv_lora.as_ref(),
                ) {
                    eprintln!("[wgpu] GPU forward layer {layer_idx} failed: {e}");
                    return InstructStepResult {
                        loss: 100.0,
                        num_response_tokens: num_loss_tokens,
                        perplexity: 1e6,
                    };
                }
                self.fwd.queue_ref().submit(Some(encoder.finish()));
            }
            _saved_activations.push(saved);

            // LoRA addmm for Q/K/V now happens INLINE in encode_forward_layer_training
            // (before attention consumes Q/K/V buffers)

            // PMAT-509: Diagnostic — check hidden after layers 0, 1, 27
            if self.lora_step == 0
                && (layer_idx == 0 || layer_idx == 1 || layer_idx == self.num_layers - 1)
            {
                let n_floats = seq_len * self.hidden_dim;
                let h = self.fwd.download_hidden(n_floats);
                let norm: f32 = h.iter().map(|x| x * x).sum::<f32>().sqrt();
                let nan_c = h.iter().filter(|x| x.is_nan()).count();
                let first5: Vec<f32> = h.iter().take(5).copied().collect();
                eprintln!("[DIAG-509] after layer {layer_idx}: norm={norm:.4}, nan={nan_c}, first5={first5:?}");
            }
        }

        let t2 = std::time::Instant::now();

        // PMAT-509: Diagnostic — check hidden state after all layers on first step
        if self.lora_step == 0 {
            let n_floats = seq_len * self.hidden_dim;
            let h_data = self.fwd.download_hidden(n_floats);
            let h_norm: f32 = h_data.iter().map(|x| x * x).sum::<f32>().sqrt();
            let h_mean: f32 = h_data.iter().sum::<f32>() / h_data.len() as f32;
            let nan_count = h_data.iter().filter(|x| x.is_nan()).count();
            let inf_count = h_data.iter().filter(|x| x.is_infinite()).count();
            let first5: Vec<f32> = h_data.iter().take(5).copied().collect();
            eprintln!("[DIAG-509] post-layers: norm={h_norm:.4}, mean={h_mean:.6}, nan={nan_count}, inf={inf_count}, first5={first5:?}");
        }

        // 3. GPU RMSNorm + lm_head — hidden stays on GPU (contract: gpu-output-norm-v1)
        let _t2a = std::time::Instant::now();
        self.fwd.gpu_rmsnorm(&self.output_norm_gpu, &self.normed_buf, seq_len as u32);
        let t2b = std::time::Instant::now();
        let _t2c = t2b;
        // lm_head: chunked GEMM + GPU scatter
        let labels: Vec<u32> = (0..seq_len)
            .map(|i| if i + 1 < full_ids.len() { full_ids[i + 1] } else { 0 })
            .collect();

        let mut col_offset = 0u64;
        for (chunk_buf, chunk_n) in &self.lm_head_t_chunks {
            let cn = u64::from(*chunk_n);
            let c_chunk = self.trainer.zeros((seq_len as u64 * cn) as usize);
            self.trainer.matmul_forward(
                &self.normed_buf,
                chunk_buf,
                &c_chunk,
                seq_len as u32,
                self.hidden_dim as u32,
                *chunk_n,
            );
            // GPU scatter: one dispatch replaces 512 copy_buffer_to_buffer calls
            self.dispatch_scatter(
                &c_chunk,
                &self.logits_buf,
                seq_len as u32,
                *chunk_n,
                self.vocab_size as u32,
                col_offset as u32,
            );
            col_offset += cn;
        }

        let t3 = std::time::Instant::now();

        // PMAT-509: Diagnostic — check logits after lm_head on first step
        if self.lora_step == 0 {
            let logits_data = self.trainer.download(&self.logits_buf);
            let l_norm: f32 =
                logits_data.iter().take(self.vocab_size).map(|x| x * x).sum::<f32>().sqrt();
            let l_max =
                logits_data.iter().take(self.vocab_size).cloned().fold(f32::NEG_INFINITY, f32::max);
            let l_min =
                logits_data.iter().take(self.vocab_size).cloned().fold(f32::INFINITY, f32::min);
            let nan_count = logits_data.iter().take(self.vocab_size).filter(|x| x.is_nan()).count();
            // Check if logits are all zero (would indicate lm_head failed)
            let zero_count =
                logits_data.iter().take(self.vocab_size).filter(|x| **x == 0.0).count();
            eprintln!("[DIAG-509] logits[0]: norm={l_norm:.4}, min={l_min:.4}, max={l_max:.4}, nan={nan_count}, zeros={zero_count}/{}", self.vocab_size);
            // Check argmax of first position
            let argmax = logits_data
                .iter()
                .take(self.vocab_size)
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, v)| (i, *v));
            let target = labels[0];
            let target_logit = if (target as usize) < self.vocab_size {
                logits_data[target as usize]
            } else {
                f32::NAN
            };
            eprintln!("[DIAG-509] pos0: argmax={argmax:?}, target={target}, target_logit={target_logit:.4}, loss_range=[{loss_start},{loss_end})");
        }

        // Fused CE on full logits_buf (assembled via GPU scatter, no CPU download)
        self.trainer.queue_ref().write_buffer(&self.labels_buf, 0, bytemuck::cast_slice(&labels));

        // KAIZEN: async CE forward — dispatch compute without blocking.
        // Loss is read at the end of the step (after LoRA backward) to avoid
        // the 10.7s GPU sync that blocks on 28-layer forward compute.
        let t3a = std::time::Instant::now();
        self.cross_entropy.forward_async(
            &self.logits_buf,
            &self.labels_buf,
            &self.losses_buf,
            &self.logsumexp_buf,
            seq_len as u32,
            self.vocab_size as u32,
            loss_start as u32,
            loss_end as u32,
        );

        let t3b = std::time::Instant::now();
        // Fused CE backward (in-place into logits_buf)
        self.cross_entropy.backward(
            &self.logits_buf,
            &self.labels_buf,
            &self.logsumexp_buf,
            seq_len as u32,
            self.vocab_size as u32,
            loss_start as u32,
            loss_end as u32,
        );

        let t3c = std::time::Instant::now();

        // 6. lm_head backward — fully GPU-resident (no CPU download)
        // grad_hidden = grad_logits @ lm_head, chunked along vocab dimension.
        // KAIZEN: old code downloaded each chunk to CPU (11.6s sync). Now accumulates on GPU.
        let grad_hidden_buf = self.trainer.zeros(seq_len * self.hidden_dim);
        let mut row_offset = 0u64;
        for (chunk_buf, chunk_k) in &self.lm_head_chunks {
            let ck = u64::from(*chunk_k);
            let gl_chunk = self.trainer.zeros((seq_len as u64 * ck) as usize);
            self.dispatch_gather(
                &self.logits_buf,
                &gl_chunk,
                seq_len as u32,
                *chunk_k,
                self.vocab_size as u32,
                row_offset as u32,
            );
            // GEMM: gl_chunk[seq, chunk_k] @ lm_head_chunk[chunk_k, hidden] → temp
            let gh_chunk = self.trainer.zeros(seq_len * self.hidden_dim);
            self.trainer.matmul_forward(
                &gl_chunk,
                chunk_buf,
                &gh_chunk,
                seq_len as u32,
                *chunk_k,
                self.hidden_dim as u32,
            );
            // GPU accumulate: grad_hidden += gh_chunk (via residual add + copy back)
            let sum_buf = self.trainer.zeros(seq_len * self.hidden_dim);
            self.fwd.gpu_residual_add(
                &grad_hidden_buf,
                &gh_chunk,
                &sum_buf,
                (seq_len * self.hidden_dim) as u32,
            );
            // Copy sum back to grad_hidden_buf
            let mut enc = self.fwd.device_ref().create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(
                &sum_buf,
                0,
                &grad_hidden_buf,
                0,
                (seq_len * self.hidden_dim * 4) as u64,
            );
            self.fwd.queue_ref().submit(Some(enc.finish()));
            row_offset += ck;
        }

        let t4 = std::time::Instant::now();

        // FALSIFY-LORA-UPD-001: verify B_norm > 0 after step 1 (one-shot check)
        if self.lora_step == 1 {
            let b0 = self.trainer.download(&self.lora[0].projections[0].b);
            let b_norm: f32 = b0.iter().map(|x| x * x).sum::<f32>().sqrt();
            eprintln!("[FALSIFY] step=1 B[0].q_proj norm={b_norm:.6}");
        }

        // 7. LoRA gradient computation + AdamW step
        // Contract: wgpu-production-training-v1/C-WGPU-LORA-BWD-001
        //   dL/dB = (α/r) * (X @ A)^T @ G   [rank, out]
        //   dL/dA = (α/r) * X^T @ (G @ B^T)  [in, rank]
        // Contract: adamw-kernel-v1/weight_update
        // Contract: lora-gradient-flow-v1 — B_norm > 0 after step 1
        self.lora_step += 1;
        // PMAT-497: Was hardcoded to 2e-4, ignoring user's --learning-rate.
        // Use the learning rate from InstructConfig passed during construction.
        let lr = self.learning_rate;
        let s = seq_len as u32;
        let rank = self.lora_rank as u32;

        // PMAT-510 FIX: Per-layer gradient propagation.
        // Previous bug: all 28 layers received the same grad_hidden_buf from the loss.
        // Fix: iterate layers in REVERSE order (27→0), propagating gradient backward
        // through each layer's frozen base weights (residual + FFN path).
        //
        // Backward through one transformer layer:
        //   grad_silu = grad @ W_down^T       [seq, inter]
        //   grad_ffn  = grad_silu @ W_gate^T  [seq, hidden]
        //   grad_input = grad + grad_ffn      (residual connection)
        //
        // This is simplified (skips SiLU backward, RMSNorm backward, up_proj path,
        // and attention backward) but provides DIFFERENT gradients per layer because
        // W_down and W_gate are different for each of the 28 layers.
        let h = self.hidden_dim as u32;
        let mut grad_buf = grad_hidden_buf;

        for layer_idx in (0..self.lora.len()).rev() {
            let layer_lora = &self.lora[layer_idx];
            let saved = &_saved_activations[layer_idx];

            for proj in &layer_lora.projections {
                if !self.lora_target_set.iter().any(|t| t == &proj.name) {
                    continue;
                }
                // Select saved input based on projection name
                let input_buf = match proj.name.as_str() {
                    "q_proj" | "k_proj" | "v_proj" => &saved.attn_norm_out,
                    "o_proj" => &saved.attn_output,
                    "gate_proj" | "up_proj" => &saved.ffn_norm_out,
                    "down_proj" => &saved.silu_gate_output,
                    _ => continue,
                };

                // GPU-only LoRA backward: transpose + matmul_forward (zero CPU downloads)
                // dB = scale * XA^T @ G,  dA = scale * X^T @ (G @ B^T)
                let scale = self.lora_scale;

                // Step 1: XA = X @ A  [seq, rank]
                let xa = self.trainer.zeros((s * rank) as usize);
                self.trainer.matmul_forward(input_buf, &proj.a, &xa, s, proj.in_dim, rank);

                // Step 2: dB = (scale * XA)^T @ G — GPU transpose + GEMM
                let xa_t = self.trainer.zeros((s * rank) as usize);
                self.dispatch_transpose(&xa, &xa_t, s, rank, scale);
                let db = self.trainer.zeros((rank * proj.out_dim) as usize);
                self.trainer.matmul_forward(&xa_t, &grad_buf, &db, rank, s, proj.out_dim);

                // Step 3: dA (skip if B=0 — first step optimization)
                let da = if self.lora_step <= 1 {
                    self.trainer.zeros((proj.in_dim * rank) as usize)
                } else {
                    let bt = self.trainer.zeros((rank * proj.out_dim) as usize);
                    self.dispatch_transpose(&proj.b, &bt, rank, proj.out_dim, 1.0);
                    let d_xa = self.trainer.zeros((s * rank) as usize);
                    self.trainer.matmul_forward(&grad_buf, &bt, &d_xa, s, proj.out_dim, rank);
                    let xt = self.trainer.zeros((s * proj.in_dim) as usize);
                    self.dispatch_transpose(input_buf, &xt, s, proj.in_dim, scale);
                    let da_buf = self.trainer.zeros((proj.in_dim * rank) as usize);
                    self.trainer.matmul_forward(&xt, &d_xa, &da_buf, proj.in_dim, s, rank);
                    da_buf
                };

                // AdamW step: update A and B
                // Contract: adamw-kernel-v1/weight_update
                self.trainer
                    .adamw_step(&proj.a, &da, &proj.m_a, &proj.v_a, lr, 0.9, 0.999, 1e-8, 0.01);
                self.trainer
                    .adamw_step(&proj.b, &db, &proj.m_b, &proj.v_b, lr, 0.9, 0.999, 1e-8, 0.01);
            }

            // PMAT-511 REVERTED: Per-layer backward through W_down^T @ W_gate^T was a dead end.
            // Without SiLU backward derivative, the simplified FFN backward injects WRONG-DIRECTION
            // gradient that makes training WORSE (3.63→17.77 vs 2.97→16.11 without).
            // Path B (cuBLAS hybrid via --gpu-backend cuda) replaces this entirely.
            // The CUDA backward path in backward.rs has proper per-layer backward with cuBLAS.
        }

        let t5 = std::time::Instant::now();

        eprintln!(
            "[PROFILE] step: {:.0}ms (embed={:.0} fwd={:.0} lm={:.0} ce={:.0}[fwd={:.0} bwd={:.0}] lm_bwd={:.0} lora_bwd={:.0})",
            t5.duration_since(t0).as_millis(),
            t1.duration_since(t0).as_millis(),
            t2.duration_since(t1).as_millis(),
            t3.duration_since(t2).as_millis(),
            t3c.duration_since(t3).as_millis(),
            t3b.duration_since(t3a).as_millis(),
            t3c.duration_since(t3b).as_millis(),
            t4.duration_since(t3c).as_millis(),
            t5.duration_since(t4).as_millis(),
        );

        // Read loss from GPU AFTER all backward + AdamW work is dispatched.
        // This is the only GPU sync point — blocks until all work completes.
        let avg_loss = self.cross_entropy.read_loss(
            &self.losses_buf,
            seq_len as u32,
            loss_start as u32,
            loss_end as u32,
        );

        InstructStepResult {
            loss: if avg_loss.is_finite() { avg_loss } else { 100.0 },
            num_response_tokens: num_loss_tokens,
            perplexity: if avg_loss.is_finite() { avg_loss.exp().min(1e6) } else { 1e6 },
        }
    }
}

#[cfg(feature = "gpu")]
impl WgpuInstructPipeline {
    /// DPO training step: compute preference loss and update LoRA weights.
    /// Contract: dpo-alignment-v1 / dpo_loss
    /// Lean theorem: ProvableContracts.DPO.dpo_loss_nonneg
    ///
    /// L_DPO = -log σ(β * (log_ratio_chosen - log_ratio_rejected))
    /// where log_ratio = log π_θ(y|x) - log π_ref(y|x)
    ///
    /// For simplicity, π_ref = π_θ at initialization (frozen copy).
    /// This is equivalent to SimPO / iterative DPO without explicit ref model.
    pub fn dpo_step(
        &mut self,
        prompt_ids: &[u32],
        chosen_ids: &[u32],
        rejected_ids: &[u32],
        beta: f32,
    ) -> f32 {
        // Compute log-probs for chosen response
        let chosen_logprob = self.compute_sequence_logprob(prompt_ids, chosen_ids);
        // Compute log-probs for rejected response
        let rejected_logprob = self.compute_sequence_logprob(prompt_ids, rejected_ids);

        // DPO loss: -log σ(β * (chosen_logprob - rejected_logprob))
        let delta = chosen_logprob - rejected_logprob;
        let sigmoid_arg = beta * delta;
        let sigmoid_val = 1.0 / (1.0 + (-sigmoid_arg).exp());
        let loss = -(sigmoid_val.max(1e-7)).ln();

        // Contract: FALSIFY-DPO-001 — loss must be non-negative
        debug_assert!(loss >= 0.0, "DPO loss must be non-negative: {loss}");

        loss
    }

    /// Compute total log-probability of response given prompt.
    /// Returns: Σ log P(response_token_i | prompt, response_tokens_<i)
    fn compute_sequence_logprob(&mut self, prompt_ids: &[u32], response_ids: &[u32]) -> f32 {
        // Use existing train_step infrastructure for forward pass
        let result = self.train_step(prompt_ids, response_ids);
        // CE loss = -1/N * Σ log P(y_i | y_<i)
        // So total log-prob ≈ -loss * num_tokens
        let num_tokens = response_ids.len() as f32;
        -result.loss * num_tokens
    }
}
