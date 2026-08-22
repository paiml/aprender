use super::attention_gqa::{add_positional_encoding, slice_pe};
#[allow(clippy::wildcard_imports)]
use super::*;

impl TransformerDecoderLayer {
    /// Create a new Transformer Decoder Layer.
    #[must_use]
    pub fn new(d_model: usize, nhead: usize, dim_feedforward: usize) -> Self {
        Self {
            self_attn: MultiHeadAttention::new(d_model, nhead),
            cross_attn: MultiHeadAttention::new(d_model, nhead),
            linear1: Linear::new(d_model, dim_feedforward),
            linear2: Linear::new(dim_feedforward, d_model),
            norm1: LayerNorm::new(&[d_model]),
            norm2: LayerNorm::new(&[d_model]),
            norm3: LayerNorm::new(&[d_model]),
            dropout: Dropout::new(0.1),
            dropout1: Dropout::new(0.1),
            dropout2: Dropout::new(0.1),
            dropout3: Dropout::new(0.1),
            d_model,
            training: true,
        }
    }

    /// Forward with memory from encoder.
    ///
    /// # Arguments
    ///
    /// * `tgt` - Target sequence [batch, `tgt_len`, `d_model`]
    /// * `memory` - Encoder output [batch, `src_len`, `d_model`]
    /// * `tgt_mask` - Optional causal mask for target
    /// * `memory_mask` - Optional mask for encoder-decoder attention
    pub fn forward_with_memory(
        &self,
        tgt: &Tensor,
        memory: &Tensor,
        tgt_mask: Option<&Tensor>,
        memory_mask: Option<&Tensor>,
    ) -> Tensor {
        // Self-attention (masked)
        let tgt_norm = self.norm1.forward(tgt);
        let (attn_out, _) = self.self_attn.forward_self(&tgt_norm, tgt_mask);
        let attn_out = self.dropout1.forward(&attn_out);
        let tgt = tgt.add(&attn_out);

        // Cross-attention
        let tgt_norm = self.norm2.forward(&tgt);
        let (cross_out, _) = self
            .cross_attn
            .forward_qkv(&tgt_norm, memory, memory, memory_mask);
        let cross_out = self.dropout2.forward(&cross_out);
        let tgt = tgt.add(&cross_out);

        // Feed-forward
        let tgt_norm = self.norm3.forward(&tgt);
        let ff_out = self.linear1.forward(&tgt_norm);
        // PMAT-922 (decoder twin of PMAT-921): use the autograd-aware Tensor::gelu,
        // NOT the local `gelu` helper / nn::functional::gelu, which builds its
        // output via Tensor::from_vec and SEVERS the graph — freezing linear1 +
        // norm3 in any end-to-end decoder training run. Both paths use the
        // identical tanh GELU approximation, so forward numerics are unchanged;
        // only the backward edge is restored.
        let ff_out = ff_out.gelu();
        let ff_out = self.dropout.forward(&ff_out);
        let ff_out = self.linear2.forward(&ff_out);
        let ff_out = self.dropout3.forward(&ff_out);

        tgt.add(&ff_out)
    }
}

impl Module for TransformerDecoderLayer {
    fn forward(&self, input: &Tensor) -> Tensor {
        // For single input, use it as both target and memory
        self.forward_with_memory(input, input, None, None)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.self_attn.parameters();
        params.extend(self.cross_attn.parameters());
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params.extend(self.norm3.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = self.self_attn.parameters_mut();
        params.extend(self.cross_attn.parameters_mut());
        params.extend(self.linear1.parameters_mut());
        params.extend(self.linear2.parameters_mut());
        params.extend(self.norm1.parameters_mut());
        params.extend(self.norm2.parameters_mut());
        params.extend(self.norm3.parameters_mut());
        params
    }

    fn train(&mut self) {
        self.training = true;
        self.self_attn.train();
        self.cross_attn.train();
        self.dropout.train();
        self.dropout1.train();
        self.dropout2.train();
        self.dropout3.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.self_attn.eval();
        self.cross_attn.eval();
        self.dropout.eval();
        self.dropout1.eval();
        self.dropout2.eval();
        self.dropout3.eval();
    }

    fn training(&self) -> bool {
        self.training
    }
}

impl std::fmt::Debug for TransformerDecoderLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransformerDecoderLayer")
            .field("d_model", &self.d_model)
            .field("self_attn", &self.self_attn)
            .field("cross_attn", &self.cross_attn)
            .finish_non_exhaustive()
    }
}

/// Sinusoidal Positional Encoding (Vaswani et al., 2017).
///
/// Adds position information to input embeddings using sine and cosine
/// functions of different frequencies.
///
/// ```text
/// PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
/// ```
#[derive(Debug)]
pub struct PositionalEncoding {
    d_model: usize,
    max_len: usize,
    dropout: Dropout,
    /// Pre-computed positional encodings
    pe: Tensor,
    training: bool,
}

impl PositionalEncoding {
    /// Create positional encoding.
    ///
    /// # Arguments
    ///
    /// * `d_model` - Dimension of the model
    /// * `max_len` - Maximum sequence length to pre-compute
    #[must_use]
    pub fn new(d_model: usize, max_len: usize) -> Self {
        let pe = compute_positional_encoding(d_model, max_len);

        Self {
            d_model,
            max_len,
            dropout: Dropout::new(0.1),
            pe,
            training: true,
        }
    }

    /// Set dropout probability.
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = Dropout::new(dropout);
        self
    }
}

impl Module for PositionalEncoding {
    fn forward(&self, input: &Tensor) -> Tensor {
        let seq_len = input.shape()[1];
        assert!(
            seq_len <= self.max_len,
            "Sequence length {seq_len} exceeds max_len {}",
            self.max_len
        );

        // Get positional encodings for this sequence length
        let pe_slice = slice_pe(&self.pe, seq_len, self.d_model);

        // Add to input and apply dropout
        let output = add_positional_encoding(input, &pe_slice);
        self.dropout.forward(&output)
    }

    fn train(&mut self) {
        self.training = true;
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.dropout.eval();
    }

    fn training(&self) -> bool {
        self.training
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Transpose the last two dimensions of a tensor.
///
/// ONE PATH: Canonical ND transpose for attention operations (UCBD §4).
pub(crate) fn transpose_last_two(x: &Tensor) -> Tensor {
    let shape = x.shape();
    let ndim = shape.len();

    if ndim < 2 {
        return x.clone();
    }

    let last = shape[ndim - 1];
    let second_last = shape[ndim - 2];

    // Compute new shape
    let mut new_shape = shape.to_vec();
    new_shape[ndim - 2] = last;
    new_shape[ndim - 1] = second_last;

    // Compute batch dimensions
    let batch_size: usize = shape[..ndim - 2].iter().product();
    let matrix_size = last * second_last;

    let mut output = vec![0.0; x.data().len()];

    // Tiled transpose: process TILE×TILE blocks per batch slice to stay in L1 cache.
    // src_base hoisting reduces multiplies in the inner loop.
    const TILE: usize = 32;
    let src = x.data();
    for b in 0..batch_size {
        let offset = b * matrix_size;
        for i0 in (0..second_last).step_by(TILE) {
            let i_end = (i0 + TILE).min(second_last);
            for j0 in (0..last).step_by(TILE) {
                let j_end = (j0 + TILE).min(last);
                for i in i0..i_end {
                    let src_base = offset + i * last;
                    for j in j0..j_end {
                        output[offset + j * second_last + i] = src[src_base + j];
                    }
                }
            }
        }
    }

    let mut result = Tensor::from_vec(output, &new_shape);

    // PMAT-914 / OBLIG-ATTENTION-BACKWARD-GRAD-FLOW: record the transpose edge so
    // gradient flows back through K^T to the key projection.
    record_attention_grad(x, &mut result, || {
        std::sync::Arc::new(crate::autograd::grad_fn::TransposeLastTwoBackward {
            input_shape: shape.to_vec(),
        })
    });

    result
}

/// Record a single-input attention helper backward edge on the autograd tape
/// (PMAT-914). No-op unless grad tracking is on AND the input requires grad.
fn record_attention_grad<F>(input: &Tensor, result: &mut Tensor, make: F)
where
    F: FnOnce() -> std::sync::Arc<dyn crate::autograd::GradFn>,
{
    use crate::autograd::{is_grad_enabled, with_graph};
    if is_grad_enabled() && input.requires_grad_enabled() {
        result.requires_grad_(true);
        let grad_fn = make();
        result.set_grad_fn(grad_fn.clone());
        with_graph(|graph| {
            graph.register_tensor(input.clone());
            graph.record(result.id(), grad_fn, vec![input.id()]);
        });
    }
}

/// Record a two-input attention helper backward edge (PMAT-914).
/// No-op unless grad tracking is on AND at least one input requires grad.
fn record_attention_grad2<F>(a: &Tensor, b: &Tensor, result: &mut Tensor, make: F)
where
    F: FnOnce() -> std::sync::Arc<dyn crate::autograd::GradFn>,
{
    use crate::autograd::{is_grad_enabled, with_graph};
    if is_grad_enabled() && (a.requires_grad_enabled() || b.requires_grad_enabled()) {
        result.requires_grad_(true);
        let grad_fn = make();
        result.set_grad_fn(grad_fn.clone());
        with_graph(|graph| {
            graph.register_tensor(a.clone());
            graph.register_tensor(b.clone());
            graph.record(result.id(), grad_fn, vec![a.id(), b.id()]);
        });
    }
}

/// Batched matrix multiplication using SIMD-accelerated Trueno.
/// For 4D tensors [batch, heads, m, k] @ [batch, heads, k, n] -> [batch, heads, m, n]
///
/// ONE PATH: Canonical batched matmul for attention operations (UCBD §4).
///
/// Uses `trueno::Matrix::batched_matmul_4d` for efficient SIMD computation.
/// Per spec §2.4.1 Compute Backend Hierarchy: SIMD before naive loops.
///
/// # Panics
/// Panics if `batched_matmul_4d` fails after dimension validation. This should
/// never happen as dimensions are validated by the assert above. If it does,
/// it indicates a bug in the trueno library.
#[allow(clippy::expect_used)]
pub(crate) fn matmul_batched(a: &Tensor, b: &Tensor) -> Tensor {
    let a_shape = a.shape();
    let b_shape = b.shape();

    // Handle 4D tensors: [batch, heads, seq, dim]
    if a_shape.len() == 4 && b_shape.len() == 4 {
        let (batch, heads, m, k1) = (a_shape[0], a_shape[1], a_shape[2], a_shape[3]);
        let k2 = b_shape[2];
        let n = b_shape[3];

        assert_eq!(k1, k2, "Inner dimensions must match for matmul");

        // Use Trueno's SIMD batched matmul for 4D attention tensors
        let output = Matrix::batched_matmul_4d(a.data(), b.data(), batch, heads, m, k1, n)
            .expect("batched_matmul_4d failed: dimensions validated but operation failed");

        let mut result = Tensor::from_vec(output, &[batch, heads, m, n]);

        // PMAT-914 / OBLIG-ATTENTION-BACKWARD-GRAD-FLOW: record the 4D matmul edge
        // (QK^T and attn@V) so gradient flows to Q, K, and V.
        record_attention_grad2(a, b, &mut result, || {
            std::sync::Arc::new(crate::autograd::grad_fn::BatchedMatmul4dBackward {
                a: a.clone(),
                b: b.clone(),
            })
        });

        result
    } else {
        // Fallback for 2D/3D - uses Tensor's SIMD matmul
        a.matmul(b)
    }
}

/// Scale tensor by scalar (SIMD-accelerated).
pub(super) fn scale_tensor(x: &Tensor, scale: f32) -> Tensor {
    x.mul_scalar(scale)
}

/// Materialize `mask` broadcast to exactly `target_shape`, as a CONSTANT tensor.
///
/// Right-aligned (numpy/torch) broadcasting: the mask's trailing dimensions are
/// aligned with the target's trailing dimensions; a mask dimension of extent 1 is
/// repeated across the corresponding target dimension.
///
/// The result deliberately does NOT require grad — an attention mask has no
/// differentiable input. It is a constant, exactly like the `additive_attention_mask`
/// builder. Graph connectivity is provided by the caller applying it through the
/// autograd-aware `Tensor::add` (the PMAT-922 pattern).
///
/// Non-broadcastable shapes are a programming error inside this crate (`add_mask` is
/// `pub(super)` with a single caller). They trip a `debug_assert!` in every test and
/// debug build. In release the index is clamped into range, which is deterministic and
/// panic-free; it deliberately does NOT fall back to "return scores unmodified",
/// because that would silently DISABLE masking and let padded keys contribute to
/// attention (threat T-1-18, information disclosure).
fn broadcast_mask_to(target_shape: &[usize], mask: &Tensor) -> Tensor {
    let mask_shape = mask.shape();
    let mask_data = mask.data();
    let rank = target_shape.len();
    let m_rank = mask_shape.len();

    debug_assert!(
        m_rank <= rank,
        "add_mask: mask rank {m_rank} exceeds scores rank {rank} — not broadcastable"
    );

    // Row-major strides over the mask.
    let mut m_strides = vec![0usize; m_rank];
    let mut acc = 1usize;
    for (stride, &extent) in m_strides.iter_mut().zip(mask_shape.iter()).rev() {
        *stride = acc;
        acc *= extent;
    }

    // Right-alignment. `offset` covers the usual case (mask rank <= scores rank);
    // `skip` only matters in the pathological over-rank case the debug_assert catches.
    let offset = rank.saturating_sub(m_rank);
    let skip = m_rank.saturating_sub(rank);

    let total: usize = target_shape.iter().product();
    let mut out = vec![0.0f32; total];
    let mut idx = vec![0usize; rank];

    for slot in &mut out {
        let mut m_off = 0usize;
        for (d, (&extent, &stride)) in mask_shape
            .iter()
            .zip(m_strides.iter())
            .enumerate()
            .skip(skip)
        {
            let t_d = d - skip + offset;
            let i = if extent == 1 {
                0
            } else if extent == target_shape[t_d] {
                idx[t_d]
            } else {
                debug_assert!(
                    false,
                    "add_mask: mask dim {d} (extent {extent}) is not broadcastable \
                     against scores dim {t_d} (extent {})",
                    target_shape[t_d]
                );
                idx[t_d].min(extent - 1)
            };
            m_off += i * stride;
        }
        *slot = mask_data[m_off];

        // Odometer increment over the target index.
        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < target_shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }

    Tensor::from_vec(out, target_shape)
}

/// Add an additive attention mask to attention scores, broadcasting the mask over the
/// batch, head and query axes as needed.
///
/// Contract: `setfit-encoder-conformance-v1`, equation `apply_additive_mask`.
/// This function is ON THE ENC-03 GRADIENT PATH.
///
/// # Repaired defect (plan 01-09, amendment A-02)
///
/// The previous body returned `scores.add(mask)` — which records a graph edge — ONLY
/// when the two shapes matched exactly, and otherwise fell through to
/// `Tensor::from_vec(scores.data().iter().zip(mask.data().iter())...)`. That fallback
/// had two independent defects:
///
/// 1. `.zip()` truncates to the SHORTER iterator, so a `[B,1,1,S]` mask against
///    `[B,heads,T,S]` scores produced `B*S` values where `B*heads*T*S` were needed. It
///    never broadcast at all; `Tensor::from_vec` asserts on the length, so the call
///    panicked outright at every realistic attention shape.
/// 2. It built the result with a bare `Tensor::from_vec` and recorded no grad_fn, so
///    masking SEVERED the autograd tape (the PMAT-913/914/922 class).
///
/// The repair expands the mask into a constant of exactly the scores' shape and then
/// applies it with the autograd-aware `Tensor::add`. Because the expanded tensor
/// matches shape exactly, the existing `AddBackward` records the edge back to `scores`
/// — no new backward struct, and no new severable code path.
///
/// Any future edit MUST keep both the elementwise-correctness and the
/// graph-preservation tests in `tests_attention_mask_broadcast.rs` green.
#[provable_contracts_macros::contract(
    "setfit-encoder-conformance-v1",
    equation = "apply_additive_mask"
)]
pub(super) fn add_mask(scores: &Tensor, mask: &Tensor) -> Tensor {
    contract_pre_apply_additive_mask!(scores.data());

    // Mask holds 0 for valid positions and a large negative value for masked positions.
    let result = if scores.shape() == mask.shape() {
        // Fast path: shapes already agree, SIMD add records AddBackward directly.
        scores.add(mask)
    } else {
        let expanded = broadcast_mask_to(scores.shape(), mask);
        scores.add(&expanded)
    };

    contract_post_apply_additive_mask!(result.data());
    result
}

/// Softmax over last dimension.
///
/// ONE PATH: Delegates to `nn::functional::softmax` (UCBD §4).
pub(super) fn softmax_last_dim(x: &Tensor) -> Tensor {
    let mut result = crate::nn::functional::softmax(x, -1);

    // PMAT-914 / OBLIG-ATTENTION-BACKWARD-GRAD-FLOW: record the softmax edge so
    // gradient flows back through the attention weights to Q and K.
    let out_for_grad = Tensor::from_vec(result.data().to_vec(), result.shape());
    record_attention_grad(x, &mut result, move || {
        std::sync::Arc::new(crate::autograd::grad_fn::SoftmaxLastDimBackward {
            output: out_for_grad,
        })
    });

    result
}

/// ONE PATH: Delegates to `nn::functional::dropout` (UCBD §4).
pub(super) fn apply_dropout(x: &Tensor, p: f32) -> Tensor {
    crate::nn::functional::dropout(x, p, true)
}

/// Seeded variant of [`apply_dropout`] (plan 01-06, A5).
///
/// `nn::functional::dropout(x, p, training)` takes **no seed** — confirmed by
/// inspection in 01-03's spike and by 01-09 — so the attention-probs dropout
/// inside `scaled_dot_product_attention` was not reproducible. This is the hook
/// that makes it so, added ADDITIVELY:
///
/// * `None` delegates to [`apply_dropout`] verbatim, so every existing caller is
///   byte-for-byte unchanged. That is the whole default path.
/// * `Some(seed)` routes through [`crate::nn::Dropout::with_seed`], the crate's
///   audited seeded dropout — ONE PATH, not a second hand-rolled RNG loop. Its
///   mask construction is the same PMAT-922 constant-mask `mul` that
///   `functional::dropout` uses, so the only difference is which RNG produced
///   the mask; the autograd edge is identical.
///
/// The result is a pure function of `(seed, p, x.len())`. `MultiHeadAttention`
/// therefore mixes a per-call counter into the seed it passes, so the stream
/// ADVANCES across forward passes instead of replaying one fixed mask.
pub(super) fn apply_dropout_seeded(x: &Tensor, p: f32, seed: Option<u64>) -> Tensor {
    match seed {
        None => apply_dropout(x, p),
        Some(seed) => {
            use crate::nn::module::Module as _;
            crate::nn::Dropout::with_seed(p, seed).forward(x)
        }
    }
}

/// Reshape for multi-head attention: [batch, seq, embed] -> [batch, heads, seq, `head_dim`]
pub(super) fn reshape_for_attention(
    x: &Tensor,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Tensor {
    let mut output = vec![0.0; batch * num_heads * seq_len * head_dim];

    for b in 0..batch {
        for s in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    // Input: [b, s, h * head_dim + d]
                    // Output: [b, h, s, d]
                    let in_idx = b * seq_len * (num_heads * head_dim)
                        + s * (num_heads * head_dim)
                        + h * head_dim
                        + d;
                    let out_idx = b * num_heads * seq_len * head_dim
                        + h * seq_len * head_dim
                        + s * head_dim
                        + d;
                    output[out_idx] = x.data()[in_idx];
                }
            }
        }
    }

    let mut result = Tensor::from_vec(output, &[batch, num_heads, seq_len, head_dim]);

    // PMAT-914 / OBLIG-ATTENTION-BACKWARD-GRAD-FLOW: record the head-split edge.
    record_attention_grad(x, &mut result, || {
        std::sync::Arc::new(crate::autograd::grad_fn::ReshapeForAttentionBackward {
            batch,
            seq_len,
            num_heads,
            head_dim,
        })
    });

    result
}

/// Reshape from multi-head attention: [batch, heads, seq, `head_dim`] -> [batch, seq, embed]
///
/// ONE PATH: Canonical concat-heads for attention operations (UCBD §4).
pub(crate) fn reshape_from_attention(
    x: &Tensor,
    batch: usize,
    seq_len: usize,
    embed_dim: usize,
) -> Tensor {
    let num_heads = x.shape()[1];
    let head_dim = x.shape()[3];

    let mut output = vec![0.0; batch * seq_len * embed_dim];

    for b in 0..batch {
        for s in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    // Input: [b, h, s, d]
                    // Output: [b, s, h * head_dim + d]
                    let in_idx = b * num_heads * seq_len * head_dim
                        + h * seq_len * head_dim
                        + s * head_dim
                        + d;
                    let out_idx = b * seq_len * embed_dim + s * embed_dim + h * head_dim + d;
                    output[out_idx] = x.data()[in_idx];
                }
            }
        }
    }

    let mut result = Tensor::from_vec(output, &[batch, seq_len, embed_dim]);

    // PMAT-914 / OBLIG-ATTENTION-BACKWARD-GRAD-FLOW: record the head-concat edge.
    record_attention_grad(x, &mut result, || {
        std::sync::Arc::new(crate::autograd::grad_fn::ReshapeFromAttentionBackward {
            batch,
            seq_len,
            num_heads,
            head_dim,
        })
    });

    result
}

/// Compute sinusoidal positional encoding.
fn compute_positional_encoding(d_model: usize, max_len: usize) -> Tensor {
    let mut pe = vec![0.0; max_len * d_model];

    for pos in 0..max_len {
        for i in 0..d_model / 2 {
            let angle = pos as f32 / 10000_f32.powf(2.0 * i as f32 / d_model as f32);
            pe[pos * d_model + 2 * i] = angle.sin();
            pe[pos * d_model + 2 * i + 1] = angle.cos();
        }
    }

    Tensor::new(&pe, &[max_len, d_model])
}
