//! APR-EMBED-001 EG-2 (#4901): the EmbeddingGemma 2 CPU forward.
//!
//! One unit-length vector per input, following llama.cpp's `gemma-embedding2` graph (b11490):
//!
//! 1. Token rows scaled by `sqrt(hidden)`.
//! 2. Per-layer inputs: `per_layer_model_proj · x / sqrt(hidden)`, RMS-normed in chunks of
//!    `per_layer` with `per_layer_proj_norm`. Layer `l` of token `t` is chunk `l` of row `t`.
//! 3. Each layer: RMSNorm → Q/K/V; per-head RMSNorm of Q and K (weighted) and V (unweighted);
//!    NEOX RoPE with the layer's `n_rot` and base; bidirectional attention with scale 1.0, over
//!    `|i - j| <= sliding_window / 2` on a local layer and every position on a global one;
//!    output projection; post-attention norm plus residual; RMSNorm → `down(gelu(gate·x) ⊙ up·x)`
//!    → post-FFN norm plus residual; the per-layer branch
//!    `post_norm(proj(gelu(inp_gate·h) ⊙ input_l))` added; the sum times `layer_output_scale`.
//! 4. `output_norm`, the `output` projection per token, the mean over tokens, unit L2 length.
//!
//! Every block it runs is a shared one (I-10): `matmul_any` for each projection (the one quant
//! dispatcher), `dequantize_at` for a token row, `parallel_multihead_attention_windowed`,
//! `ops::rms_norm`, `ops::gelu` and `apply_partial_neox_rope`. This file reads a tensor's ggml type
//! only to size its rows; it never branches on it.
//!
//! A file that is not exactly this graph is refused: another architecture, a pooling type other
//! than mean, a causal flag, RoPE scaling, shared KV layers, a sliding-window pattern that
//! disagrees with the query widths, a tensor with other dims, or a tensor count other than the
//! 5 + 17 per layer the graph consumes (llama.cpp refuses the same file).

use crate::error::{RealizarError, Result};
use crate::gguf::forward_qwen35::apply_partial_neox_rope;
use crate::gguf::ggml_type_table;
use crate::gguf::keys::{self, arch_key};
use crate::gguf::ops::{gelu, rms_norm};
use crate::gguf::{
    GGUFModel, GGUFValue, MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedTensor,
};
use crate::model_kind::{self, LayerAttention};

/// The Matryoshka widths a pooled vector may be truncated to.
pub const MATRYOSHKA_DIMS: [usize; 4] = [768, 512, 256, 128];

/// Tensors outside the blocks: `token_embd`, `per_layer_model_proj`, `per_layer_proj_norm`,
/// `output_norm`, `output`.
const NON_BLOCK_TENSORS: usize = 5;

/// Tensors in each block.
const BLOCK_TENSORS: usize = 17;

/// The pooling type llama.cpp writes for mean pooling.
const POOLING_MEAN: usize = 1;

/// The period of llama.cpp's default sliding-window pattern when the file carries none
/// (`load_swa_pattern(ml, 6)`): layer `i` is local unless `i % 6 == 5`.
const DEFAULT_SWA_PERIOD: usize = 6;

fn refuse(reason: impl Into<String>) -> RealizarError {
    RealizarError::UnsupportedOperation {
        operation: "embedding_gemma2".to_string(),
        reason: reason.into(),
    }
}

fn meta_usize(model: &GGUFModel, arch: &str, suffix: &str) -> Result<usize> {
    let key = arch_key(arch, suffix);
    let value = model
        .metadata
        .get(&key)
        .ok_or_else(|| refuse(format!("{key} missing")))?;
    let wide = match value {
        GGUFValue::UInt8(v) => Some(u64::from(*v)),
        GGUFValue::UInt16(v) => Some(u64::from(*v)),
        GGUFValue::UInt32(v) => Some(u64::from(*v)),
        GGUFValue::UInt64(v) => Some(*v),
        GGUFValue::Int32(v) => u64::try_from(*v).ok(),
        GGUFValue::Int64(v) => u64::try_from(*v).ok(),
        _ => None,
    };
    wide.and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| refuse(format!("{key} is not a non-negative integer: {value:?}")))
}

fn meta_f32(model: &GGUFModel, arch: &str, suffix: &str) -> Result<f32> {
    let key = arch_key(arch, suffix);
    match model.metadata.get(&key) {
        Some(GGUFValue::Float32(v)) if v.is_finite() => Ok(*v),
        Some(GGUFValue::Float64(v)) if v.is_finite() => Ok(*v as f32),
        Some(other) => Err(refuse(format!("{key} is not a finite float: {other:?}"))),
        None => Err(refuse(format!("{key} missing"))),
    }
}

/// The model-wide hyperparameters, read and checked once.
#[derive(Debug, Clone, Copy)]
struct Hparams {
    hidden: usize,
    per_layer: usize,
    layers: usize,
    heads: usize,
    ff: usize,
    out: usize,
    eps: f32,
    half_window: usize,
    context: Option<usize>,
}

impl Hparams {
    fn read(model: &GGUFModel, arch: &str) -> Result<Self> {
        let pooling = meta_usize(model, arch, "pooling_type")?;
        if pooling != POOLING_MEAN {
            return Err(refuse(format!(
                "pooling_type {pooling}; this forward pools by mean (1)"
            )));
        }
        if let Some(GGUFValue::Bool(true)) = model.metadata.get(&arch_key(arch, "attention.causal"))
        {
            return Err(refuse(
                "attention.causal = true on an embedding architecture",
            ));
        }
        Self::refuse_unsupported_keys(model, arch)?;
        let window = meta_usize(model, arch, "attention.sliding_window")?;
        let hp = Self {
            hidden: meta_usize(model, arch, keys::EMBEDDING_LENGTH)?,
            per_layer: meta_usize(model, arch, "embedding_length_per_layer_input")?,
            layers: meta_usize(model, arch, keys::BLOCK_COUNT)?,
            heads: meta_usize(model, arch, keys::ATTENTION_HEAD_COUNT)?,
            ff: meta_usize(model, arch, keys::FEED_FORWARD_LENGTH)?,
            out: meta_usize(model, arch, "embedding_length_out")?,
            eps: meta_f32(model, arch, keys::ATTENTION_LAYER_NORM_RMS_EPSILON)?,
            half_window: window / 2,
            context: meta_usize(model, arch, keys::CONTEXT_LENGTH).ok(),
        };
        let widths = [
            hp.hidden,
            hp.per_layer,
            hp.layers,
            hp.heads,
            hp.ff,
            hp.out,
            window,
        ];
        if widths.contains(&0) {
            return Err(refuse(format!(
                "a zero width among hidden/per_layer/layers/heads/ff/out/window: {widths:?}"
            )));
        }
        Ok(hp)
    }

    /// RoPE scaling and shared KV layers change the graph; this forward runs neither.
    fn refuse_unsupported_keys(model: &GGUFModel, arch: &str) -> Result<()> {
        let scaling = arch_key(arch, "rope.scaling.");
        if let Some(key) = model.metadata.keys().find(|k| k.starts_with(&scaling)) {
            return Err(refuse(format!(
                "{key} present; this forward runs RoPE unscaled"
            )));
        }
        let shared = arch_key(arch, "attention.shared_kv_layers");
        if model.metadata.contains_key(&shared)
            && meta_usize(model, arch, "attention.shared_kv_layers")? != 0
        {
            return Err(refuse(format!(
                "{shared} is not 0; this forward shares no KV"
            )));
        }
        Ok(())
    }
}

/// The head width, rotated width and RoPE base of one attention kind.
#[derive(Debug, Clone, Copy, PartialEq)]
struct AttentionShape {
    head_dim: usize,
    n_rot: usize,
    rope_base: f32,
}

impl AttentionShape {
    /// `tail` is `""` for the global layers and `"_swa"` for the local ones.
    fn read(model: &GGUFModel, arch: &str, tail: &str) -> Result<Self> {
        let head_dim = meta_usize(model, arch, &format!("attention.key_length{tail}"))?;
        let value_dim = meta_usize(model, arch, &format!("attention.value_length{tail}"))?;
        if value_dim != head_dim {
            return Err(refuse(format!(
                "attention.value_length{tail} {value_dim} differs from key_length{tail} {head_dim}"
            )));
        }
        let n_rot = meta_usize(model, arch, &format!("rope.dimension_count{tail}"))?;
        if n_rot == 0 || n_rot % 2 != 0 || n_rot > head_dim {
            return Err(refuse(format!(
                "rope.dimension_count{tail} {n_rot} is not an even width in 1..={head_dim}"
            )));
        }
        let rope_base = meta_f32(model, arch, &format!("rope.freq_base{tail}"))?;
        Ok(Self {
            head_dim,
            n_rot,
            rope_base,
        })
    }
}

fn tensor_dims(model: &GGUFModel, name: &str) -> Result<Vec<usize>> {
    let info = model
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| refuse(format!("tensor {name} missing")))?;
    info.dims
        .iter()
        .map(|&d| {
            usize::try_from(d).map_err(|_| refuse(format!("{name}: dim {d} overflows usize")))
        })
        .collect()
}

/// The weight `name`, `out_dim` rows of `in_dim`, its bytes copied out of `data`. The reader
/// reverses GGUF's `ne0, ne1`, so it holds the shape as `[out_dim, in_dim]`, as everywhere else
/// in this crate. Rows must be whole blocks of the tensor's type.
fn weight(
    model: &GGUFModel,
    data: &[u8],
    name: &str,
    in_dim: usize,
    out_dim: usize,
) -> Result<OwnedQuantizedTensor> {
    let dims = tensor_dims(model, name)?;
    if dims != [out_dim, in_dim] {
        return Err(refuse(format!(
            "{name} has dims {dims:?}, expected [{out_dim}, {in_dim}]"
        )));
    }
    let info = model
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| refuse(format!("tensor {name} missing")))?;
    let traits = ggml_type_table::traits(info.qtype).ok_or_else(|| {
        refuse(format!(
            "{name}: ggml type {} has no block layout",
            info.qtype
        ))
    })?;
    if traits.blck_size == 0 || in_dim % traits.blck_size != 0 {
        return Err(refuse(format!(
            "{name}: a row of {in_dim} is not whole {} blocks of {}",
            traits.name, traits.blck_size
        )));
    }
    let bytes = (in_dim / traits.blck_size)
        .checked_mul(traits.type_size)
        .and_then(|row| row.checked_mul(out_dim))
        .ok_or_else(|| refuse(format!("{name}: byte size overflows usize")))?;
    let start = usize::try_from(info.offset)
        .ok()
        .and_then(|offset| model.tensor_data_start.checked_add(offset))
        .ok_or_else(|| refuse(format!("{name}: offset {} overflows usize", info.offset)))?;
    let slice = start
        .checked_add(bytes)
        .and_then(|end| data.get(start..end))
        .ok_or_else(|| {
            refuse(format!(
                "{name}: {bytes} bytes at {start} run past the file"
            ))
        })?;
    Ok(OwnedQuantizedTensor {
        data: slice.to_vec(),
        in_dim,
        out_dim,
        qtype: info.qtype,
    })
}

/// The one-dimensional tensor `name` of `len` elements, as f32.
fn vector(model: &GGUFModel, data: &[u8], name: &str, len: usize) -> Result<Vec<f32>> {
    let dims = tensor_dims(model, name)?;
    if dims != [len] {
        return Err(refuse(format!(
            "{name} has dims {dims:?}, expected [{len}]"
        )));
    }
    let values = model.get_tensor_f32(name, data)?;
    if values.len() != len {
        return Err(refuse(format!(
            "{name} read {} values, expected {len}",
            values.len()
        )));
    }
    Ok(values)
}

/// Row `row` of `table`, dequantized: `in_dim` values from its own bytes.
fn table_row(table: &OwnedQuantizedTensor, row: usize) -> Result<Vec<f32>> {
    if row >= table.out_dim {
        return Err(refuse(format!(
            "token {row} is outside the vocabulary of {}",
            table.out_dim
        )));
    }
    let row_bytes = table.data.len() / table.out_dim;
    GGUFModel::dequantize_at(table.qtype, &table.data, row * row_bytes, table.in_dim)
}

/// Each layer is local unless the file's pattern (or llama.cpp's default period of 6) says
/// global; the query widths must say the same.
fn check_swa_pattern(model: &GGUFModel, arch: &str, kinds: &[LayerAttention]) -> Result<()> {
    let key = arch_key(arch, "attention.sliding_window_pattern");
    let declared: Vec<bool> = match model.metadata.get(&key) {
        None => (0..kinds.len())
            .map(|i| i % DEFAULT_SWA_PERIOD != DEFAULT_SWA_PERIOD - 1)
            .collect(),
        Some(GGUFValue::Array(items)) => items
            .iter()
            .map(|item| match item {
                GGUFValue::Bool(local) => Ok(*local),
                other => Err(refuse(format!("{key} holds a non-bool: {other:?}"))),
            })
            .collect::<Result<_>>()?,
        Some(other) => return Err(refuse(format!("{key} is not an array: {other:?}"))),
    };
    if declared.len() != kinds.len() {
        return Err(refuse(format!(
            "{key} has {} entries for {} layers",
            declared.len(),
            kinds.len()
        )));
    }
    for (i, (&local, kind)) in declared.iter().zip(kinds).enumerate() {
        if local != (*kind == LayerAttention::Local) {
            return Err(refuse(format!(
                "layer {i}: the sliding-window pattern says local = {local}, the attn_q width says {kind:?}"
            )));
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct AttnWeights {
    norm: Vec<f32>,
    q: OwnedQuantizedTensor,
    k: OwnedQuantizedTensor,
    v: OwnedQuantizedTensor,
    o: OwnedQuantizedTensor,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    post_norm: Vec<f32>,
}

#[derive(Debug, Clone)]
struct FfnWeights {
    norm: Vec<f32>,
    gate: OwnedQuantizedTensor,
    up: OwnedQuantizedTensor,
    down: OwnedQuantizedTensor,
    post_norm: Vec<f32>,
}

#[derive(Debug, Clone)]
struct PerLayerWeights {
    inp_gate: OwnedQuantizedTensor,
    proj: OwnedQuantizedTensor,
    post_norm: Vec<f32>,
    out_scale: f32,
}

#[derive(Debug, Clone)]
struct Layer {
    heads: usize,
    kv_heads: usize,
    shape: AttentionShape,
    half_window: Option<usize>,
    attn: AttnWeights,
    ffn: FfnWeights,
    per_layer: PerLayerWeights,
}

/// The tensors and hyperparameters a layer is loaded from.
struct Source<'a> {
    model: &'a GGUFModel,
    data: &'a [u8],
    hp: Hparams,
}

impl Source<'_> {
    fn weight(
        &self,
        layer: usize,
        tensor: &str,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<OwnedQuantizedTensor> {
        weight(
            self.model,
            self.data,
            &format!("blk.{layer}.{tensor}.weight"),
            in_dim,
            out_dim,
        )
    }

    fn vector(&self, layer: usize, tensor: &str, len: usize) -> Result<Vec<f32>> {
        vector(
            self.model,
            self.data,
            &format!("blk.{layer}.{tensor}.weight"),
            len,
        )
    }

    fn attn(&self, i: usize, d: usize, kv_heads: usize) -> Result<AttnWeights> {
        let (h, q_width, kv_width) = (self.hp.hidden, self.hp.heads * d, kv_heads * d);
        Ok(AttnWeights {
            norm: self.vector(i, "attn_norm", h)?,
            q: self.weight(i, "attn_q", h, q_width)?,
            k: self.weight(i, "attn_k", h, kv_width)?,
            v: self.weight(i, "attn_v", h, kv_width)?,
            o: self.weight(i, "attn_output", q_width, h)?,
            q_norm: self.vector(i, "attn_q_norm", d)?,
            k_norm: self.vector(i, "attn_k_norm", d)?,
            post_norm: self.vector(i, "post_attention_norm", h)?,
        })
    }

    fn ffn(&self, i: usize) -> Result<FfnWeights> {
        let (h, ff) = (self.hp.hidden, self.hp.ff);
        Ok(FfnWeights {
            norm: self.vector(i, "ffn_norm", h)?,
            gate: self.weight(i, "ffn_gate", h, ff)?,
            up: self.weight(i, "ffn_up", h, ff)?,
            down: self.weight(i, "ffn_down", ff, h)?,
            post_norm: self.vector(i, "post_ffw_norm", h)?,
        })
    }

    fn per_layer(&self, i: usize) -> Result<PerLayerWeights> {
        let (h, p) = (self.hp.hidden, self.hp.per_layer);
        Ok(PerLayerWeights {
            inp_gate: self.weight(i, "inp_gate", h, p)?,
            proj: self.weight(i, "proj", p, h)?,
            post_norm: self.vector(i, "post_norm", h)?,
            out_scale: self.vector(i, "layer_output_scale", 1)?[0],
        })
    }

    fn layer(
        &self,
        i: usize,
        kind: LayerAttention,
        shape: AttentionShape,
        kv_heads: usize,
    ) -> Result<Layer> {
        if kv_heads == 0 || self.hp.heads % kv_heads != 0 {
            return Err(refuse(format!(
                "layer {i}: {} query heads over {kv_heads} KV heads",
                self.hp.heads
            )));
        }
        let half_window = match kind {
            LayerAttention::Local => Some(self.hp.half_window),
            LayerAttention::Global => None,
        };
        Ok(Layer {
            heads: self.hp.heads,
            kv_heads,
            shape,
            half_window,
            attn: self.attn(i, shape.head_dim, kv_heads)?,
            ffn: self.ffn(i)?,
            per_layer: self.per_layer(i)?,
        })
    }
}

/// An EmbeddingGemma 2 model, loaded for the CPU forward.
#[derive(Debug, Clone)]
pub struct EmbeddingGemma2 {
    hp: Hparams,
    token_embd: OwnedQuantizedTensor,
    per_layer_model_proj: OwnedQuantizedTensor,
    per_layer_proj_norm: Vec<f32>,
    layers: Vec<Layer>,
    output_norm: Vec<f32>,
    output: OwnedQuantizedTensor,
}

impl EmbeddingGemma2 {
    /// Load from a mapped GGUF file.
    ///
    /// # Errors
    /// See [`Self::from_gguf`].
    pub fn from_mapped(mapped: &MappedGGUFModel) -> Result<Self> {
        Self::from_gguf(&mapped.model, mapped.data())
    }

    /// Load from a parsed GGUF file and its bytes.
    ///
    /// # Errors
    /// Any file that is not exactly the `gemma-embedding2` graph (see the module docs).
    pub fn from_gguf(model: &GGUFModel, data: &[u8]) -> Result<Self> {
        let arch = model
            .architecture()
            .ok_or_else(|| refuse("general.architecture missing"))?;
        if !model_kind::is_embedding_gemma2(arch) {
            return Err(refuse(format!(
                "architecture '{arch}' is not {}",
                model_kind::EMBEDDING_GEMMA2_ARCH
            )));
        }
        let hp = Hparams::read(model, arch)?;
        let expected = NON_BLOCK_TENSORS + BLOCK_TENSORS * hp.layers;
        if model.tensors.len() != expected {
            return Err(refuse(format!(
                "{} tensors; the graph consumes {expected} ({NON_BLOCK_TENSORS} + {BLOCK_TENSORS} x {} layers)",
                model.tensors.len(),
                hp.layers
            )));
        }
        let kv_heads = model_kind::num_kv_heads_per_layer(model)
            .ok_or_else(|| refuse("attention.head_count_kv missing"))?
            .map_err(refuse)?;
        let q_widths = (0..hp.layers)
            .map(|i| {
                tensor_dims(model, &format!("blk.{i}.attn_q.weight"))
                    .map(|d| d.first().copied().unwrap_or(0))
            })
            .collect::<Result<Vec<_>>>()?;
        let kinds = model_kind::layer_attention_kinds(&q_widths).map_err(refuse)?;
        check_swa_pattern(model, arch, &kinds)?;
        let global = AttentionShape::read(model, arch, "")?;
        let local = AttentionShape::read(model, arch, "_swa")?;
        let src = Source { model, data, hp };
        let layers = kinds
            .iter()
            .zip(&kv_heads)
            .enumerate()
            .map(|(i, (&kind, &kv))| {
                let shape = if kind == LayerAttention::Global {
                    global
                } else {
                    local
                };
                src.layer(i, kind, shape, kv)
            })
            .collect::<Result<Vec<_>>>()?;
        let vocab = tensor_dims(model, "token_embd.weight")?
            .first()
            .copied()
            .unwrap_or(0);
        Ok(Self {
            hp,
            token_embd: weight(model, data, "token_embd.weight", hp.hidden, vocab)?,
            per_layer_model_proj: weight(
                model,
                data,
                "per_layer_model_proj.weight",
                hp.hidden,
                hp.per_layer * hp.layers,
            )?,
            per_layer_proj_norm: vector(model, data, "per_layer_proj_norm.weight", hp.per_layer)?,
            layers,
            output_norm: vector(model, data, "output_norm.weight", hp.hidden)?,
            output: weight(model, data, "output.weight", hp.hidden, hp.out)?,
        })
    }

    /// The width of a pooled vector before truncation.
    #[must_use]
    pub fn output_dim(&self) -> usize {
        self.hp.out
    }

    /// The number of token rows.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.token_embd.out_dim
    }

    /// The unit-length mean-pooled embedding of `tokens`, `output_dim()` wide.
    ///
    /// # Errors
    /// No tokens, more than the context length, a token outside the vocabulary, a block error,
    /// or a pooled vector of zero or non-finite length.
    pub fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let seq = tokens.len();
        if seq == 0 {
            return Err(refuse("no tokens to embed"));
        }
        if let Some(maximum) = self.hp.context.filter(|&m| seq > m) {
            return Err(RealizarError::ContextLimitExceeded {
                provided: seq,
                maximum,
            });
        }
        let scale = (self.hp.hidden as f32).sqrt();
        let mut x = Vec::with_capacity(seq * self.hp.hidden);
        for &token in tokens {
            let row = table_row(&self.token_embd, token as usize)?;
            x.extend(row.iter().map(|v| v * scale));
        }
        let inputs = self.per_layer_inputs(&x)?;
        for (l, layer) in self.layers.iter().enumerate() {
            x = self.layer_forward(l, layer, &x, &inputs, seq)?;
        }
        let normed = rms_norm(&x, &self.output_norm, self.hp.eps);
        let projected = OwnedQuantizedModel::matmul_any(&normed, &self.output)?;
        unit_length(mean_rows(&projected, self.hp.out))
    }

    /// `[seq, layers * per_layer]`: chunk `l` of row `t` is layer `l`'s input for token `t`.
    fn per_layer_inputs(&self, x: &[f32]) -> Result<Vec<f32>> {
        let mut proj = OwnedQuantizedModel::matmul_any(x, &self.per_layer_model_proj)?;
        let inv = 1.0 / (self.hp.hidden as f32).sqrt();
        for v in &mut proj {
            *v *= inv;
        }
        Ok(rms_norm(&proj, &self.per_layer_proj_norm, self.hp.eps))
    }

    fn layer_forward(
        &self,
        l: usize,
        layer: &Layer,
        x: &[f32],
        inputs: &[f32],
        seq: usize,
    ) -> Result<Vec<f32>> {
        let eps = self.hp.eps;
        let attn = self.attention(layer, &rms_norm(x, &layer.attn.norm, eps), seq)?;
        let attn_out = add(&rms_norm(&attn, &layer.attn.post_norm, eps), x);
        let ffn = gelu_ffn(&rms_norm(&attn_out, &layer.ffn.norm, eps), &layer.ffn)?;
        let pe_in = add(&rms_norm(&ffn, &layer.ffn.post_norm, eps), &attn_out);

        let (p, stride) = (self.hp.per_layer, self.hp.per_layer * self.hp.layers);
        let mut gated = OwnedQuantizedModel::matmul_any(&pe_in, &layer.per_layer.inp_gate)?;
        gelu(&mut gated);
        for (t, row) in gated.chunks_exact_mut(p).enumerate() {
            let input = &inputs[t * stride + l * p..][..p];
            row.iter_mut().zip(input).for_each(|(g, i)| *g *= i);
        }
        let projected = OwnedQuantizedModel::matmul_any(&gated, &layer.per_layer.proj)?;
        let pe = rms_norm(&projected, &layer.per_layer.post_norm, eps);
        let scale = layer.per_layer.out_scale;
        Ok(pe_in
            .iter()
            .zip(&pe)
            .map(|(a, b)| (a + b) * scale)
            .collect())
    }

    fn attention(&self, layer: &Layer, h: &[f32], seq: usize) -> Result<Vec<f32>> {
        let eps = self.hp.eps;
        let shape = layer.shape;
        let d = shape.head_dim;
        let mut q = rms_norm(
            &OwnedQuantizedModel::matmul_any(h, &layer.attn.q)?,
            &layer.attn.q_norm,
            eps,
        );
        let mut k = rms_norm(
            &OwnedQuantizedModel::matmul_any(h, &layer.attn.k)?,
            &layer.attn.k_norm,
            eps,
        );
        let v = rms_norm(
            &OwnedQuantizedModel::matmul_any(h, &layer.attn.v)?,
            &vec![1.0; d],
            eps,
        );
        for (pos, row) in q.chunks_exact_mut(layer.heads * d).enumerate() {
            apply_partial_neox_rope(row, layer.heads, d, shape.n_rot, pos, shape.rope_base);
        }
        for (pos, row) in k.chunks_exact_mut(layer.kv_heads * d).enumerate() {
            apply_partial_neox_rope(row, layer.kv_heads, d, shape.n_rot, pos, shape.rope_base);
        }
        let ctx = OwnedQuantizedModel::parallel_multihead_attention_windowed(
            &q,
            &k,
            &v,
            seq,
            layer.heads,
            layer.kv_heads,
            d,
            1.0,
            layer.half_window,
        )?;
        OwnedQuantizedModel::matmul_any(&ctx, &layer.attn.o)
    }
}

/// `down(gelu(gate·x) ⊙ up·x)`: llama.cpp's `LLM_FFN_GELU` + `LLM_FFN_PAR`.
fn gelu_ffn(x: &[f32], ffn: &FfnWeights) -> Result<Vec<f32>> {
    let mut gate = OwnedQuantizedModel::matmul_any(x, &ffn.gate)?;
    let up = OwnedQuantizedModel::matmul_any(x, &ffn.up)?;
    gelu(&mut gate);
    gate.iter_mut().zip(&up).for_each(|(g, u)| *g *= u);
    OwnedQuantizedModel::matmul_any(&gate, &ffn.down)
}

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

fn mean_rows(x: &[f32], width: usize) -> Vec<f32> {
    let mut mean = vec![0.0f32; width];
    let mut rows = 0usize;
    for row in x.chunks_exact(width) {
        mean.iter_mut().zip(row).for_each(|(m, v)| *m += v);
        rows += 1;
    }
    let inv = 1.0 / rows.max(1) as f32;
    for m in &mut mean {
        *m *= inv;
    }
    mean
}

/// `v / |v|`, the norm summed in f64 as llama.cpp's `common_embd_normalize` does.
fn unit_length(mut v: Vec<f32>) -> Result<Vec<f32>> {
    let norm = v
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    if !(norm.is_finite() && norm > 0.0) {
        return Err(RealizarError::InferenceError(format!(
            "pooled embedding has length {norm}: cannot scale it to unit length"
        )));
    }
    let inv = (1.0 / norm) as f32;
    for x in &mut v {
        *x *= inv;
    }
    Ok(v)
}

/// The first `dim` values of a pooled vector, scaled back to unit length (Matryoshka).
///
/// # Errors
/// A `dim` outside [`MATRYOSHKA_DIMS`] or wider than the vector, or a prefix of zero length.
pub fn truncate_to(vector: &[f32], dim: usize) -> Result<Vec<f32>> {
    if !MATRYOSHKA_DIMS.contains(&dim) {
        return Err(refuse(format!(
            "--dim {dim}; the model is trained at {MATRYOSHKA_DIMS:?}"
        )));
    }
    let prefix = vector.get(..dim).ok_or_else(|| {
        refuse(format!(
            "--dim {dim} is wider than the {}-wide vector",
            vector.len()
        ))
    })?;
    unit_length(prefix.to_vec())
}

/// One text through the whole EmbeddingGemma 2 path: the file's tokenizer under `prompt`,
/// the forward, and, when `dim` is given, [`truncate_to`]. Returns the token count and the
/// vector.
///
/// `apr embed` and `apr serve` both call this, so the CLI and the HTTP routes hand the
/// model the same tokens and return the same vector for the same text (EG-3, #4902).
///
/// # Errors
/// The forward refuses the tokens, or `dim` is not a width the vector truncates to.
pub fn embed_text(
    encoder: &EmbeddingGemma2,
    tokenizer: &crate::gguf::embedding_gemma2_input::EmbedTokenizer,
    prompt: &crate::gguf::embedding_gemma2_input::Prompt<'_>,
    text: &str,
    dim: Option<usize>,
) -> Result<(usize, Vec<f32>)> {
    let tokens = tokenizer.tokens(prompt, text);
    let full = encoder.embed(&tokens)?;
    let vector = match dim {
        None => full,
        Some(dim) => truncate_to(&full, dim)?,
    };
    Ok((tokens.len(), vector))
}

#[cfg(test)]
#[path = "forward_embedding_gemma2_tests.rs"]
pub(crate) mod tests;
