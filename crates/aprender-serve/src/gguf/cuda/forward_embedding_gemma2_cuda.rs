//! APR-EMBED-001 EG-4 (#4903): the EmbeddingGemma 2 forward on the GPU.
//!
//! The device twin of [`EmbeddingGemma2::embed`], op for op in the same order: token rows
//! times `sqrt(hidden)`, the per-layer inputs, each layer's bidirectional (windowed) attention,
//! GELU FFN and per-layer branch, then `output_norm`, the `output` projection and the mean over
//! tokens. The unit-length step runs on the host through the CPU's own [`unit_length`], so a
//! zero or non-finite pooled vector is refused exactly as on the CPU (I-5).
//!
//! I-5: every projection is [`CudaExecutor::eg2_project`], f32 end to end; an executor whose
//! fp16 GEMM is armed is refused by name at construction and again at each projection. There
//! is no CPU fallback: an op the device cannot run is an error, never a quiet host path.
//!
//! I-10: the kernels are the shared `aprender-gpu` ones. A weight whose ggml type has no
//! device f32 dequant is dequantized once at load by the CPU's own `dequantize_at` (the one
//! quant table) and uploaded as f32; this file never branches on a quant type.

use super::{OwnedQuantizedTensor, RealizarError, Result};
use crate::cuda::types::WeightQuantType;
use crate::cuda::CudaExecutor;
use crate::cuda::EG2_F16_REFUSAL;
use crate::gguf::forward_embedding_gemma2::{table_row, unit_length, EmbeddingGemma2, Layer};
use crate::gguf::GGUFModel;
use trueno_gpu::driver::GpuBuffer;

/// The most attention-score floats one pass holds (128 MiB); longer inputs run in passes.
const SCORES_CAP: usize = 32 << 20;

/// The ggml type id of f32, the type a host-dequantized weight is uploaded as.
const GGML_F32: u32 = 0;

/// Map a GPU error into the crate error type with the operation that raised it.
fn gpu_err(operation: &str, e: &trueno_gpu::GpuError) -> RealizarError {
    RealizarError::UnsupportedOperation {
        operation: operation.to_string(),
        reason: format!("{e}"),
    }
}

fn to_u32(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| RealizarError::UnsupportedOperation {
        operation: "embedding_gemma2_cuda".to_string(),
        reason: format!("{what} = {v} exceeds the 32-bit launch range"),
    })
}

/// A weight resident in the executor's cache: `[n × k]`, row-major.
#[derive(Debug, Clone, Copy)]
struct CudaWeight {
    ptr: u64,
    qtype: WeightQuantType,
    n: u32,
    k: u32,
}

/// A norm weight on the device: rows as wide as the gamma.
#[derive(Debug, Clone, Copy)]
struct Gamma {
    ptr: u64,
    width: u32,
}

/// One layer's device weights and shapes; the norm buffers live in [`CudaLayer::bufs`].
#[derive(Debug, Clone, Copy)]
struct LayerPlan {
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    n_rot: u32,
    theta_scale: f32,
    half_window: Option<u32>,
    out_scale: f32,
    q: CudaWeight,
    k: CudaWeight,
    v: CudaWeight,
    o: CudaWeight,
    gate: CudaWeight,
    up: CudaWeight,
    down: CudaWeight,
    inp_gate: CudaWeight,
    pl_proj: CudaWeight,
    /// Rows `l·p .. (l+1)·p` of `per_layer_model_proj`: this layer's input projection.
    pl_input: CudaWeight,
    attn_norm: Gamma,
    q_norm: Gamma,
    k_norm: Gamma,
    ones_d: Gamma,
    post_attn_norm: Gamma,
    ffn_norm: Gamma,
    post_ffn_norm: Gamma,
    pl_post_norm: Gamma,
}

struct CudaLayer {
    plan: LayerPlan,
    /// The buffers behind the plan's [`Gamma`]s, kept alive with it.
    _bufs: Vec<GpuBuffer<f32>>,
}

/// The EmbeddingGemma 2 encoder resident on one GPU.
pub struct EmbeddingGemma2Cuda {
    hidden: u32,
    per_layer: u32,
    ff: u32,
    out: u32,
    eps: f32,
    context: Option<usize>,
    token_embd: OwnedQuantizedTensor,
    pl_norm: Gamma,
    output_norm: Gamma,
    output: CudaWeight,
    layers: Vec<CudaLayer>,
    /// The buffers behind `pl_norm` and `output_norm`.
    _bufs: Vec<GpuBuffer<f32>>,
    /// Declared last so every buffer above is freed while its context is alive.
    executor: CudaExecutor,
}

/// Upload rows `row0 .. row0 + rows` of `tensor` as the executor weight `name`.
///
/// The device dequant is tried first; a type without one is dequantized on the host and
/// uploaded as f32.
fn upload_rows(
    executor: &mut CudaExecutor,
    name: &str,
    tensor: &OwnedQuantizedTensor,
    row0: usize,
    rows: usize,
) -> Result<CudaWeight> {
    let op = "embedding_gemma2_cuda_upload";
    if tensor.data.is_empty() || tensor.out_dim == 0 || row0 + rows > tensor.out_dim {
        return Err(RealizarError::UnsupportedOperation {
            operation: op.to_string(),
            reason: format!(
                "'{name}': rows {row0}..{} of {}",
                row0 + rows,
                tensor.out_dim
            ),
        });
    }
    let row_bytes = tensor.data.len() / tensor.out_dim;
    let bytes = &tensor.data[row0 * row_bytes..(row0 + rows) * row_bytes];
    let (n, k) = (to_u32(rows, "rows")?, to_u32(tensor.in_dim, "in_dim")?);
    if let Some(qtype) = WeightQuantType::from_ggml_type(tensor.qtype) {
        executor
            .load_quantized_weights_with_type(name, bytes, tensor.qtype)
            .map_err(|e| gpu_err(op, &e))?;
        let ptr = executor
            .get_quantized_weight_ptr(name)
            .map_err(|e| gpu_err(op, &e))?;
        if executor.eg2_has_dequant(qtype, ptr, n, k) {
            return Ok(CudaWeight { ptr, qtype, n, k });
        }
    }
    let values = GGUFModel::dequantize_at(tensor.qtype, bytes, 0, rows * tensor.in_dim)?;
    let f32_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let f32_name = format!("{name}.f32");
    executor
        .load_quantized_weights_with_type(&f32_name, &f32_bytes, GGML_F32)
        .map_err(|e| gpu_err(op, &e))?;
    let ptr = executor
        .get_quantized_weight_ptr(&f32_name)
        .map_err(|e| gpu_err(op, &e))?;
    Ok(CudaWeight {
        ptr,
        qtype: WeightQuantType::F32,
        n,
        k,
    })
}

fn upload_whole(
    executor: &mut CudaExecutor,
    name: &str,
    t: &OwnedQuantizedTensor,
) -> Result<CudaWeight> {
    upload_rows(executor, name, t, 0, t.out_dim)
}

/// Upload a norm weight, keeping its buffer in `bufs`.
fn upload_gamma(
    executor: &CudaExecutor,
    bufs: &mut Vec<GpuBuffer<f32>>,
    v: &[f32],
) -> Result<Gamma> {
    let buf = GpuBuffer::from_host(executor.context(), v)
        .map_err(|e| gpu_err("embedding_gemma2_cuda_upload", &e))?;
    let gamma = Gamma {
        ptr: buf.as_ptr(),
        width: to_u32(v.len(), "norm width")?,
    };
    bufs.push(buf);
    Ok(gamma)
}

fn alloc(executor: &CudaExecutor, len: usize) -> Result<GpuBuffer<f32>> {
    GpuBuffer::new(executor.context(), len.max(1))
        .map_err(|e| gpu_err("embedding_gemma2_cuda_alloc", &e))
}

/// The device pointer `floats` f32 values into `buf`.
fn at(buf: &GpuBuffer<f32>, floats: usize) -> u64 {
    buf.as_ptr() + (floats * std::mem::size_of::<f32>()) as u64
}

type GpuResult = std::result::Result<(), trueno_gpu::GpuError>;

/// The per-call device scratch, sized for one input.
struct Scratch {
    x: GpuBuffer<f32>,
    h: GpuBuffer<f32>,
    tmp: GpuBuffer<f32>,
    attn_out: GpuBuffer<f32>,
    q: GpuBuffer<f32>,
    k: GpuBuffer<f32>,
    v: GpuBuffer<f32>,
    ctx: GpuBuffer<f32>,
    gate: GpuBuffer<f32>,
    up: GpuBuffer<f32>,
    gated: GpuBuffer<f32>,
    /// `[layers][rows][per_layer]`: layer `l`'s inputs are contiguous.
    inputs: GpuBuffer<f32>,
    pooled: GpuBuffer<f32>,
    ones: GpuBuffer<f32>,
    scores: GpuBuffer<f32>,
}

impl EmbeddingGemma2Cuda {
    /// Upload `model` onto `executor`'s device.
    ///
    /// # Errors
    /// The executor's fp16 GEMM is armed (I-5, [`EG2_F16_REFUSAL`]), a shape exceeds the
    /// 32-bit launch range, or an upload fails.
    pub fn new(model: &EmbeddingGemma2, mut executor: CudaExecutor) -> Result<Self> {
        if executor.qwen35_prefill_f16() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "embedding_gemma2_cuda".to_string(),
                reason: EG2_F16_REFUSAL.to_string(),
            });
        }
        let hp = &model.hp;
        let mut layers = Vec::with_capacity(model.layers.len());
        for (l, layer) in model.layers.iter().enumerate() {
            layers.push(Self::upload_layer(&mut executor, model, l, layer)?);
        }
        let output = upload_whole(&mut executor, "eg2.output", &model.output)?;
        let mut bufs = Vec::new();
        let pl_norm = upload_gamma(&executor, &mut bufs, &model.per_layer_proj_norm)?;
        let output_norm = upload_gamma(&executor, &mut bufs, &model.output_norm)?;
        Ok(Self {
            hidden: to_u32(hp.hidden, "hidden")?,
            per_layer: to_u32(hp.per_layer, "per_layer")?,
            ff: to_u32(hp.ff, "ff")?,
            out: to_u32(hp.out, "out")?,
            eps: hp.eps,
            context: hp.context,
            token_embd: model.token_embd.clone(),
            pl_norm,
            output_norm,
            output,
            layers,
            _bufs: bufs,
            executor,
        })
    }

    fn upload_layer(
        executor: &mut CudaExecutor,
        model: &EmbeddingGemma2,
        l: usize,
        layer: &Layer,
    ) -> Result<CudaLayer> {
        let p = model.hp.per_layer;
        let name = |t: &str| format!("eg2.blk.{l}.{t}");
        let shape = layer.shape;
        let mut bufs = Vec::new();
        let plan = LayerPlan {
            heads: to_u32(layer.heads, "heads")?,
            kv_heads: to_u32(layer.kv_heads, "kv_heads")?,
            head_dim: to_u32(shape.head_dim, "head_dim")?,
            n_rot: to_u32(shape.n_rot, "n_rot")?,
            // The CPU's own theta step, computed once with f32::powf.
            theta_scale: shape.rope_base.powf(-2.0 / shape.n_rot as f32),
            half_window: layer
                .half_window
                .map(|w| to_u32(w, "half_window"))
                .transpose()?,
            out_scale: layer.per_layer.out_scale,
            q: upload_whole(executor, &name("attn_q"), &layer.attn.q)?,
            k: upload_whole(executor, &name("attn_k"), &layer.attn.k)?,
            v: upload_whole(executor, &name("attn_v"), &layer.attn.v)?,
            o: upload_whole(executor, &name("attn_output"), &layer.attn.o)?,
            gate: upload_whole(executor, &name("ffn_gate"), &layer.ffn.gate)?,
            up: upload_whole(executor, &name("ffn_up"), &layer.ffn.up)?,
            down: upload_whole(executor, &name("ffn_down"), &layer.ffn.down)?,
            inp_gate: upload_whole(executor, &name("inp_gate"), &layer.per_layer.inp_gate)?,
            pl_proj: upload_whole(executor, &name("proj"), &layer.per_layer.proj)?,
            pl_input: upload_rows(
                executor,
                &name("per_layer_model_proj"),
                &model.per_layer_model_proj,
                l * p,
                p,
            )?,
            attn_norm: upload_gamma(executor, &mut bufs, &layer.attn.norm)?,
            q_norm: upload_gamma(executor, &mut bufs, &layer.attn.q_norm)?,
            k_norm: upload_gamma(executor, &mut bufs, &layer.attn.k_norm)?,
            ones_d: upload_gamma(executor, &mut bufs, &vec![1.0; shape.head_dim])?,
            post_attn_norm: upload_gamma(executor, &mut bufs, &layer.attn.post_norm)?,
            ffn_norm: upload_gamma(executor, &mut bufs, &layer.ffn.norm)?,
            post_ffn_norm: upload_gamma(executor, &mut bufs, &layer.ffn.post_norm)?,
            pl_post_norm: upload_gamma(executor, &mut bufs, &layer.per_layer.post_norm)?,
        };
        Ok(CudaLayer { plan, _bufs: bufs })
    }

    /// The width of a pooled vector before truncation.
    #[must_use]
    pub fn output_dim(&self) -> usize {
        self.out as usize
    }

    /// The unit-length mean-pooled embedding of `tokens`, computed on the GPU.
    ///
    /// # Errors
    /// The same refusals as [`EmbeddingGemma2::embed`], or a device error.
    pub fn embed(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let seq = tokens.len();
        if seq == 0 {
            return Err(RealizarError::UnsupportedOperation {
                operation: "embedding_gemma2".to_string(),
                reason: "no tokens to embed".to_string(),
            });
        }
        if let Some(maximum) = self.context.filter(|&m| seq > m) {
            return Err(RealizarError::ContextLimitExceeded {
                provided: seq,
                maximum,
            });
        }
        let hidden = self.hidden as usize;
        let scale = (hidden as f32).sqrt();
        let mut x = Vec::with_capacity(seq * hidden);
        for &token in tokens {
            let row = table_row(&self.token_embd, token as usize)?;
            x.extend(row.iter().map(|v| v * scale));
        }
        let rows = to_u32(seq, "tokens")?;
        to_u32(seq * self.widest(), "tokens × widest row")?;
        let mut s = self.scratch(seq)?;
        s.x.copy_from_host(&x)
            .map_err(|e| gpu_err("embedding_gemma2_cuda_upload", &e))?;
        self.per_layer_inputs(&s, rows)
            .map_err(|e| gpu_err("eg2_per_layer_inputs", &e))?;
        for l in 0..self.layers.len() {
            let plan = self.layers[l].plan;
            self.layer_forward(l, &plan, &s, rows)
                .map_err(|e| gpu_err("eg2_layer", &e))?;
        }
        self.pool(&s, rows).map_err(|e| gpu_err("eg2_pool", &e))?;
        self.executor
            .synchronize_all()
            .map_err(|e| gpu_err("eg2_sync", &e))?;
        let mut pooled = vec![0.0f32; self.out as usize];
        s.pooled
            .copy_to_host(&mut pooled)
            .map_err(|e| gpu_err("eg2_download", &e))?;
        unit_length(pooled)
    }

    /// The widest per-token row any buffer holds.
    fn widest(&self) -> usize {
        let mut w = (self.hidden.max(self.ff).max(self.out) as usize)
            .max(self.per_layer as usize * self.layers.len());
        for l in &self.layers {
            w = w.max((l.plan.heads.max(l.plan.kv_heads) * l.plan.head_dim) as usize);
        }
        w
    }

    fn scratch(&self, seq: usize) -> Result<Scratch> {
        let e = &self.executor;
        let (h, ff, p) = (
            self.hidden as usize,
            self.ff as usize,
            self.per_layer as usize,
        );
        let plans = || self.layers.iter().map(|l| l.plan);
        let q_max = plans()
            .map(|l| (l.heads * l.head_dim) as usize)
            .max()
            .unwrap_or(0);
        let kv_max = plans()
            .map(|l| (l.kv_heads * l.head_dim) as usize)
            .max()
            .unwrap_or(0);
        // A whole-input pass when it fits; else the cap, and the passes split the queries.
        let full = plans()
            .map(|l| (l.heads / l.kv_heads.max(1)) as usize * seq * seq)
            .max()
            .unwrap_or(1);
        let ones = GpuBuffer::from_host(e.context(), &vec![1.0f32; seq])
            .map_err(|err| gpu_err("embedding_gemma2_cuda_alloc", &err))?;
        Ok(Scratch {
            x: alloc(e, seq * h)?,
            h: alloc(e, seq * h)?,
            tmp: alloc(e, seq * h.max(self.out as usize))?,
            attn_out: alloc(e, seq * h)?,
            q: alloc(e, seq * q_max)?,
            k: alloc(e, seq * kv_max)?,
            v: alloc(e, seq * kv_max)?,
            ctx: alloc(e, seq * q_max)?,
            gate: alloc(e, seq * ff)?,
            up: alloc(e, seq * ff)?,
            gated: alloc(e, seq * p)?,
            inputs: alloc(e, self.layers.len() * seq * p)?,
            pooled: alloc(e, self.out as usize)?,
            ones,
            scores: alloc(e, full.min(SCORES_CAP))?,
        })
    }

    fn project(&mut self, w: CudaWeight, x: u64, y: u64, rows: u32, alpha: f32) -> GpuResult {
        self.executor
            .eg2_project(w.qtype, w.ptr, x, y, rows, w.n, w.k, alpha)
    }

    /// RMSNorm of the `len` floats at `x` into `y`, in rows as wide as `gamma`.
    fn rms(&mut self, x: u64, gamma: Gamma, y: u64, len: u32) -> GpuResult {
        self.executor
            .eg2_rms_rows(x, gamma.ptr, y, gamma.width, len / gamma.width, self.eps)
    }

    /// Layer `l`'s inputs: `rms(per_layer_model_proj_l · x / sqrt(hidden))` in chunks of `p`.
    fn per_layer_inputs(&mut self, s: &Scratch, rows: u32) -> GpuResult {
        let len = rows * self.per_layer;
        let inv = 1.0 / (self.hidden as f32).sqrt();
        for l in 0..self.layers.len() {
            let dst = at(&s.inputs, l * len as usize);
            self.project(self.layers[l].plan.pl_input, s.x.as_ptr(), dst, rows, inv)?;
            self.rms(dst, self.pl_norm, dst, len)?;
        }
        Ok(())
    }

    /// One layer, `x` in place: the CPU's `layer_forward`.
    fn layer_forward(&mut self, l: usize, ly: &LayerPlan, s: &Scratch, rows: u32) -> GpuResult {
        let n_h = rows * self.hidden;
        let (x, h, tmp, ao) = (
            s.x.as_ptr(),
            s.h.as_ptr(),
            s.tmp.as_ptr(),
            s.attn_out.as_ptr(),
        );
        // attn_out = rms(attention(rms(x)), post_attn) + x
        self.rms(x, ly.attn_norm, h, n_h)?;
        self.attention(ly, s, rows)?;
        self.project(ly.o, s.ctx.as_ptr(), tmp, rows, 1.0)?;
        self.rms(tmp, ly.post_attn_norm, tmp, n_h)?;
        self.executor.eg2_add(tmp, x, ao, n_h)?;

        // pe_in (in h) = rms(down(gelu(gate·h) ⊙ up·h), post_ffn) + attn_out
        let n_ff = rows * self.ff;
        let (gate, up) = (s.gate.as_ptr(), s.up.as_ptr());
        self.rms(ao, ly.ffn_norm, h, n_h)?;
        self.project(ly.gate, h, gate, rows, 1.0)?;
        self.project(ly.up, h, up, rows, 1.0)?;
        self.executor.eg2_gelu(gate, gate, n_ff)?;
        self.executor.eg2_mul(gate, up, gate, n_ff)?;
        self.project(ly.down, gate, tmp, rows, 1.0)?;
        self.rms(tmp, ly.post_ffn_norm, tmp, n_h)?;
        self.executor.eg2_add(tmp, ao, h, n_h)?;

        // x = (pe_in + rms(proj(gelu(inp_gate·pe_in) ⊙ input_l), pl_post)) · out_scale
        let n_p = rows * self.per_layer;
        let gated = s.gated.as_ptr();
        let input = at(&s.inputs, l * n_p as usize);
        self.project(ly.inp_gate, h, gated, rows, 1.0)?;
        self.executor.eg2_gelu(gated, gated, n_p)?;
        self.executor.eg2_mul(gated, input, gated, n_p)?;
        self.project(ly.pl_proj, gated, tmp, rows, 1.0)?;
        self.rms(tmp, ly.pl_post_norm, tmp, n_h)?;
        self.executor.eg2_add(h, tmp, x, n_h)?;
        self.executor.eg2_scale(x, x, ly.out_scale, n_h)
    }

    /// The attention context of the normed input in `h`, into `ctx`: the CPU's `attention`.
    fn attention(&mut self, ly: &LayerPlan, s: &Scratch, rows: u32) -> GpuResult {
        let (heads, kv, d) = (ly.heads, ly.kv_heads, ly.head_dim);
        let (q, k, v) = (s.q.as_ptr(), s.k.as_ptr(), s.v.as_ptr());
        let h = s.h.as_ptr();
        self.project(ly.q, h, q, rows, 1.0)?;
        self.rms(q, ly.q_norm, q, rows * heads * d)?;
        self.project(ly.k, h, k, rows, 1.0)?;
        self.rms(k, ly.k_norm, k, rows * kv * d)?;
        self.project(ly.v, h, v, rows, 1.0)?;
        self.rms(v, ly.ones_d, v, rows * kv * d)?;
        let e = &mut self.executor;
        e.qwen35_rope_rows(q, heads, d, ly.n_rot, heads * d, rows, 0, ly.theta_scale)?;
        e.qwen35_rope_rows(k, kv, d, ly.n_rot, kv * d, rows, 0, ly.theta_scale)?;
        e.eg2_attention(
            q,
            k,
            v,
            s.ctx.as_ptr(),
            s.scores.as_ptr(),
            s.scores.len(),
            rows,
            heads,
            kv,
            d,
            1.0,
            ly.half_window,
        )
    }

    /// `pooled = mean_rows(output · rms(x, output_norm))`.
    fn pool(&mut self, s: &Scratch, rows: u32) -> GpuResult {
        let (h, tmp) = (s.h.as_ptr(), s.tmp.as_ptr());
        self.rms(s.x.as_ptr(), self.output_norm, h, rows * self.hidden)?;
        self.project(self.output, h, tmp, rows, 1.0)?;
        self.executor
            .eg2_mean_rows(tmp, s.ones.as_ptr(), s.pooled.as_ptr(), rows, self.out)
    }
}

#[cfg(test)]
#[path = "forward_embedding_gemma2_cuda_tests.rs"]
mod tests;
