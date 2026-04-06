//! WgpuBlock — GPU-resident transformer block weights via wgpu (zero unsafe)
//!
//! Replaces `CudaNf4TransformerBlock` for the WgpuTrainingPipeline.
//! Each block holds 7 projection weights (pre-dequantized F32 on GPU),
//! 2 norm weights, and optional LoRA A/B adapters.
//!
//! # Architecture (§26 Step 0d.1)
//!
//! ```text
//! WgpuBlock (per transformer layer)
//! ├── norm weights: input_norm [hidden], post_attn_norm [hidden]
//! ├── projections: q, k, v, o [hidden × hidden], gate, up [hidden × inter], down [inter × hidden]
//! ├── LoRA A/B: trainable adapters per projection [hidden × rank] / [rank × out]
//! └── all wgpu::Buffer, all F32, zero unsafe
//! ```

#[cfg(feature = "gpu")]
use trueno::backends::gpu::wgpu;

/// A single transformer layer's weights on GPU via wgpu.
#[cfg(feature = "gpu")]
pub struct WgpuBlock {
    pub layer_idx: usize,

    // Norm weights (F32, small: hidden_size floats each)
    pub input_norm: wgpu::Buffer,
    pub post_attn_norm: wgpu::Buffer,

    // Projection weights — pre-dequantized F32 on GPU
    // Stored row-major: [out_dim, in_dim]
    pub w_q: wgpu::Buffer,    // [q_dim, hidden]
    pub w_k: wgpu::Buffer,    // [kv_dim, hidden]
    pub w_v: wgpu::Buffer,    // [kv_dim, hidden]
    pub w_o: wgpu::Buffer,    // [hidden, q_dim]
    pub w_gate: wgpu::Buffer, // [inter, hidden]
    pub w_up: wgpu::Buffer,   // [inter, hidden]
    pub w_down: wgpu::Buffer, // [hidden, inter]

    // LoRA adapters (trainable, F32)
    pub lora: Option<WgpuLoraAdapters>,
}

/// LoRA adapter weights for all 7 projections in a transformer layer.
#[cfg(feature = "gpu")]
pub struct WgpuLoraAdapters {
    pub rank: u32,
    pub scale: f32, // alpha / rank

    // Each projection gets A [in_dim, rank] and B [rank, out_dim]
    pub a_q: wgpu::Buffer,
    pub b_q: wgpu::Buffer,
    pub a_k: wgpu::Buffer,
    pub b_k: wgpu::Buffer,
    pub a_v: wgpu::Buffer,
    pub b_v: wgpu::Buffer,
    pub a_o: wgpu::Buffer,
    pub b_o: wgpu::Buffer,
    pub a_gate: wgpu::Buffer,
    pub b_gate: wgpu::Buffer,
    pub a_up: wgpu::Buffer,
    pub b_up: wgpu::Buffer,
    pub a_down: wgpu::Buffer,
    pub b_down: wgpu::Buffer,

    // AdamW optimizer states (m and v per trainable param)
    pub m_states: Vec<wgpu::Buffer>, // 14 buffers (7 A + 7 B)
    pub v_states: Vec<wgpu::Buffer>, // 14 buffers
}

/// Manages all transformer blocks on GPU + shared buffers.
#[cfg(feature = "gpu")]
pub struct WgpuBlockManager {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub blocks: Vec<WgpuBlock>,

    // Shared buffers (reused across layers)
    pub hidden_buf: wgpu::Buffer, // [max_seq, hidden] — input/output per layer
    pub hidden_buf2: wgpu::Buffer, // [max_seq, hidden] — residual stream
    pub attn_out_buf: wgpu::Buffer, // [max_seq, hidden] — attention output
    pub ffn_gate_buf: wgpu::Buffer, // [max_seq, inter] — FFN gate projection
    pub ffn_up_buf: wgpu::Buffer, // [max_seq, inter] — FFN up projection
    pub ffn_silu_buf: wgpu::Buffer, // [max_seq, inter] — SiLU(gate) × up
    pub norm_buf: wgpu::Buffer,   // [max_seq, hidden] — RMSNorm output
    pub q_buf: wgpu::Buffer,      // [max_seq, q_dim]
    pub k_buf: wgpu::Buffer,      // [max_seq, kv_dim]
    pub v_buf: wgpu::Buffer,      // [max_seq, kv_dim]

    // Embedding + lm_head
    pub embed_weight: wgpu::Buffer, // [vocab, hidden] — token embedding
    pub lm_head_weight: wgpu::Buffer, // [vocab, hidden] — output projection (may be tied)
    pub logits_buf: wgpu::Buffer,   // [max_seq, vocab] — logits output

    // Gradient buffers (for backward pass)
    pub grad_hidden_buf: wgpu::Buffer, // [max_seq, hidden]
    pub grad_logits_buf: wgpu::Buffer, // [max_seq, vocab] — may reuse logits_buf

    // Config
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub max_seq_len: u32,
    pub vocab_size: u32,
    pub num_layers: u32,
}

#[cfg(feature = "gpu")]
impl WgpuBlockManager {
    /// Create a new block manager and upload all transformer weights to GPU.
    ///
    /// `weights_per_layer` is a closure that returns the F32 weights for each layer.
    /// This avoids holding all 28 layers in CPU memory simultaneously.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        hidden_size: u32,
        intermediate_size: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        num_layers: u32,
        vocab_size: u32,
        max_seq_len: u32,
        _lora_rank: Option<u32>,
        _lora_alpha: Option<f32>,
    ) -> Self {
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let max = max_seq_len;

        // Shared buffers
        let buf = |size: u32, label: &str| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: u64::from(size) * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        Self {
            blocks: Vec::with_capacity(num_layers as usize),
            hidden_buf: buf(max * hidden_size, "hidden"),
            hidden_buf2: buf(max * hidden_size, "hidden2"),
            attn_out_buf: buf(max * hidden_size, "attn_out"),
            ffn_gate_buf: buf(max * intermediate_size, "ffn_gate"),
            ffn_up_buf: buf(max * intermediate_size, "ffn_up"),
            ffn_silu_buf: buf(max * intermediate_size, "ffn_silu"),
            norm_buf: buf(max * hidden_size, "norm"),
            q_buf: buf(max * q_dim, "q"),
            k_buf: buf(max * kv_dim, "k"),
            v_buf: buf(max * kv_dim, "v"),
            embed_weight: buf(vocab_size * hidden_size, "embed"),
            lm_head_weight: buf(vocab_size * hidden_size, "lm_head"),
            logits_buf: buf(max * vocab_size, "logits"),
            grad_hidden_buf: buf(max * hidden_size, "grad_hidden"),
            grad_logits_buf: buf(max * vocab_size, "grad_logits"),
            hidden_size,
            intermediate_size,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq_len: max,
            vocab_size,
            num_layers,
            device,
            queue,
        }
    }

    /// Upload a single transformer layer's weights to GPU.
    pub fn upload_layer(
        &mut self,
        layer_idx: usize,
        input_norm: &[f32],
        post_attn_norm: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        w_down: &[f32],
        lora_rank: Option<u32>,
        lora_scale: Option<f32>,
    ) {
        let upload = |data: &[f32], label: &str| -> wgpu::Buffer {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (data.len() * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buffer, 0, bytemuck::cast_slice(data));
            buffer
        };

        let prefix = format!("L{layer_idx}");

        let lora = lora_rank.map(|rank| {
            let scale = lora_scale.unwrap_or(1.0);
            let h = self.hidden_size as usize;
            let q = (self.num_heads * self.head_dim) as usize;
            let kv = (self.num_kv_heads * self.head_dim) as usize;
            let inter = self.intermediate_size as usize;
            let r = rank as usize;

            // Kaiming init for A, zero for B
            let kaiming = |fan_in: usize, len: usize| -> Vec<f32> {
                let std = (2.0 / fan_in as f32).sqrt();
                (0..len).map(|i| ((i as f32 * 0.013 + layer_idx as f32).sin() * std)).collect()
            };
            let zeros = |len: usize| vec![0.0f32; len];

            let pairs: Vec<(usize, usize, &str)> = vec![
                (h, q, "q"),
                (h, kv, "k"),
                (h, kv, "v"),
                (q, h, "o"),
                (h, inter, "gate"),
                (h, inter, "up"),
                (inter, h, "down"),
            ];

            let mut m_states = Vec::with_capacity(14);
            let mut v_states = Vec::with_capacity(14);
            let mut a_bufs = Vec::with_capacity(7);
            let mut b_bufs = Vec::with_capacity(7);

            for (in_d, out_d, name) in &pairs {
                let a = upload(&kaiming(*in_d, in_d * r), &format!("{prefix}.lora_a_{name}"));
                let b = upload(&zeros(r * out_d), &format!("{prefix}.lora_b_{name}"));
                m_states.push(upload(&zeros(in_d * r), &format!("{prefix}.m_a_{name}")));
                m_states.push(upload(&zeros(r * out_d), &format!("{prefix}.m_b_{name}")));
                v_states.push(upload(&zeros(in_d * r), &format!("{prefix}.v_a_{name}")));
                v_states.push(upload(&zeros(r * out_d), &format!("{prefix}.v_b_{name}")));
                a_bufs.push(a);
                b_bufs.push(b);
            }

            WgpuLoraAdapters {
                rank,
                scale,
                a_q: a_bufs.remove(0),
                b_q: b_bufs.remove(0),
                a_k: a_bufs.remove(0),
                b_k: b_bufs.remove(0),
                a_v: a_bufs.remove(0),
                b_v: b_bufs.remove(0),
                a_o: a_bufs.remove(0),
                b_o: b_bufs.remove(0),
                a_gate: a_bufs.remove(0),
                b_gate: b_bufs.remove(0),
                a_up: a_bufs.remove(0),
                b_up: b_bufs.remove(0),
                a_down: a_bufs.remove(0),
                b_down: b_bufs.remove(0),
                m_states,
                v_states,
            }
        });

        self.blocks.push(WgpuBlock {
            layer_idx,
            input_norm: upload(input_norm, &format!("{prefix}.input_norm")),
            post_attn_norm: upload(post_attn_norm, &format!("{prefix}.post_attn_norm")),
            w_q: upload(w_q, &format!("{prefix}.q_proj")),
            w_k: upload(w_k, &format!("{prefix}.k_proj")),
            w_v: upload(w_v, &format!("{prefix}.v_proj")),
            w_o: upload(w_o, &format!("{prefix}.o_proj")),
            w_gate: upload(w_gate, &format!("{prefix}.gate_proj")),
            w_up: upload(w_up, &format!("{prefix}.up_proj")),
            w_down: upload(w_down, &format!("{prefix}.down_proj")),
            lora,
        });

        eprintln!(
            "[wgpu] Uploaded layer {}/{} ({})",
            layer_idx + 1,
            self.num_layers,
            if self.blocks.last().unwrap().lora.is_some() { "with LoRA" } else { "frozen" }
        );
    }

    /// Upload embedding + lm_head weights.
    pub fn upload_embeddings(&mut self, embed: &[f32], lm_head: &[f32]) {
        self.queue.write_buffer(&self.embed_weight, 0, bytemuck::cast_slice(embed));
        self.queue.write_buffer(&self.lm_head_weight, 0, bytemuck::cast_slice(lm_head));
        eprintln!(
            "[wgpu] Uploaded embeddings: embed=[{}×{}], lm_head=[{}×{}]",
            self.vocab_size, self.hidden_size, self.vocab_size, self.hidden_size
        );
    }

    /// Total GPU memory used (approximate, in bytes).
    pub fn gpu_memory_bytes(&self) -> u64 {
        let h = u64::from(self.hidden_size);
        let inter = u64::from(self.intermediate_size);
        let q = u64::from(self.num_heads * self.head_dim);
        let kv = u64::from(self.num_kv_heads * self.head_dim);
        let v = u64::from(self.vocab_size);
        let s = u64::from(self.max_seq_len);
        let l = u64::from(self.num_layers);

        // Per layer: norms + 7 projections + optional LoRA
        let per_layer_weights =
            (2 * h + q * h + kv * h * 2 + h * q + inter * h * 2 + h * inter) * 4;
        let shared_bufs =
            (s * h * 4 + s * inter * 3 + s * q + s * kv * 2 + s * v * 2 + v * h * 2) * 4;

        per_layer_weights * l + shared_bufs
    }

    /// Number of uploaded layers.
    pub fn layer_count(&self) -> usize {
        self.blocks.len()
    }
}

#[cfg(test)]
#[cfg(feature = "gpu")]
mod tests {
    use super::*;

    #[test]
    fn test_wgpu_block_manager_creation() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = match trueno::backends::gpu::runtime::block_on(
            instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
        ) {
            Ok(a) => a,
            Err(_) => return, // No GPU
        };
        let (device, queue) = match trueno::backends::gpu::runtime::block_on(
            adapter.request_device(&wgpu::DeviceDescriptor::default()),
        ) {
            Ok(dq) => dq,
            Err(_) => return,
        };

        let mut mgr = WgpuBlockManager::new(
            device,
            queue,
            64,        // hidden
            128,       // inter
            4,         // heads
            4,         // kv_heads
            16,        // head_dim
            2,         // layers
            100,       // vocab
            32,        // max_seq
            Some(8),   // rank
            Some(2.0), // alpha
        );

        // Upload 2 layers
        for i in 0..2 {
            let h = 64;
            let inter = 128;
            let q_dim = 4 * 16;
            let kv_dim = 4 * 16;
            mgr.upload_layer(
                i,
                &vec![1.0; h],           // input_norm
                &vec![1.0; h],           // post_attn_norm
                &vec![0.01; q_dim * h],  // w_q
                &vec![0.01; kv_dim * h], // w_k
                &vec![0.01; kv_dim * h], // w_v
                &vec![0.01; h * q_dim],  // w_o
                &vec![0.01; inter * h],  // w_gate
                &vec![0.01; inter * h],  // w_up
                &vec![0.01; h * inter],  // w_down
                Some(8),
                Some(2.0 / 8.0),
            );
        }

        assert_eq!(mgr.layer_count(), 2);
        assert!(mgr.gpu_memory_bytes() > 0);
    }
}
