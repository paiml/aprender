//! Differentiable operations a ModernBERT training forward needs (APR-LAYA-TRAIN-001 LT-2).
//!
//! Contract: `contracts/modernbert-train-v1.yaml`. Each differentiable op records a
//! backward on the tape using the same pattern as `Tensor::exp`; each is enforced by a
//! central finite-difference gradcheck with a sign-flipped mutant proof in
//! `train_ops_tests.rs`.
//!
//! All 2D tensors are row-major (LAYOUT-001).

use std::sync::Arc;

use super::OpError;
use super::NEG_MASK;
use crate::autograd::grad_fn::GradFn;
use crate::autograd::tensor::{Tensor, TensorId};
use crate::autograd::{is_grad_enabled, with_graph};

/// Record `grad_fn` for `result` on the tape when any input requires grad.
fn record_op(result: &mut Tensor, grad_fn: Arc<dyn GradFn>, inputs: &[&Tensor]) {
    let needs = inputs.iter().any(|t| t.requires_grad_enabled());
    if !is_grad_enabled() || !needs {
        return;
    }
    result.requires_grad_(true);
    result.set_grad_fn(grad_fn.clone());
    with_graph(|graph| {
        for t in inputs {
            graph.register_tensor((*t).clone());
        }
        graph.record(
            result.id(),
            grad_fn,
            inputs.iter().map(|t| t.id()).collect(),
        );
    });
}

/// `(rows, cols)` of a 2D tensor.
fn dims2(x: &Tensor, what: &str) -> (usize, usize) {
    assert_eq!(
        x.ndim(),
        2,
        "{what}: expected a 2D tensor, got {:?}",
        x.shape()
    );
    (x.shape()[0], x.shape()[1])
}

/// Copy the window `[start, start+len)` along `dim` of a `rows x cols` buffer.
fn copy_window(
    src: &[f32],
    rows: usize,
    cols: usize,
    dim: usize,
    start: usize,
    len: usize,
) -> Vec<f32> {
    if dim == 0 {
        return src[start * cols..(start + len) * cols].to_vec();
    }
    let mut out = Vec::with_capacity(rows * len);
    for r in 0..rows {
        out.extend_from_slice(&src[r * cols + start..r * cols + start + len]);
    }
    out
}

/// Write `win` (the window `[start, start+len)` along `dim`) into a `rows x cols` buffer.
fn write_window(dst: &mut [f32], cols: usize, dim: usize, start: usize, len: usize, win: &[f32]) {
    if dim == 0 {
        dst[start * cols..(start + len) * cols].copy_from_slice(win);
        return;
    }
    for (r, chunk) in win.chunks_exact(len).enumerate() {
        dst[r * cols + start..r * cols + start + len].copy_from_slice(chunk);
    }
}

// ============================================================================
// Backward functions
// ============================================================================

/// Backward for `narrow`: scatter the gradient into zeros at the window.
pub(crate) struct NarrowBackward {
    rows: usize,
    cols: usize,
    dim: usize,
    start: usize,
    len: usize,
}

impl GradFn for NarrowBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let mut dx = vec![0.0f32; self.rows * self.cols];
        write_window(
            &mut dx,
            self.cols,
            self.dim,
            self.start,
            self.len,
            grad_output.data(),
        );
        vec![Tensor::from_vec(dx, &[self.rows, self.cols])]
    }

    fn name(&self) -> &'static str {
        "NarrowBackward"
    }
}

/// Backward for `concat`: split the gradient back into each input's window.
pub(crate) struct ConcatBackward {
    shapes: Vec<(usize, usize)>,
    dim: usize,
}

impl GradFn for ConcatBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let (rows, cols) = (grad_output.shape()[0], grad_output.shape()[1]);
        let mut offset = 0;
        let mut grads = Vec::with_capacity(self.shapes.len());
        for &(r, c) in &self.shapes {
            let len = if self.dim == 0 { r } else { c };
            let win = copy_window(grad_output.data(), rows, cols, self.dim, offset, len);
            grads.push(Tensor::from_vec(win, &[r, c]));
            offset += len;
        }
        grads
    }

    fn name(&self) -> &'static str {
        "ConcatBackward"
    }
}

/// Backward for `clamp`: pass the gradient where `lo <= x <= hi` (torch, inclusive).
pub(crate) struct ClampBackward {
    x: Tensor,
    lo: Option<f32>,
    hi: Option<f32>,
}

impl GradFn for ClampBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        let data: Vec<f32> = grad_output
            .data()
            .iter()
            .zip(self.x.data())
            .map(|(&g, &x)| {
                if clamp_passes(x, self.lo, self.hi) {
                    g
                } else {
                    0.0
                }
            })
            .collect();
        vec![Tensor::from_vec(data, grad_output.shape())]
    }

    fn name(&self) -> &'static str {
        "ClampBackward"
    }
}

fn clamp_passes(x: f32, lo: Option<f32>, hi: Option<f32>) -> bool {
    lo.map_or(true, |m| x >= m) && hi.map_or(true, |m| x <= m)
}

/// Backward for `rope_rotate_half`: the inverse (transpose) rotation of the gradient.
pub(crate) struct RopeRotateHalfBackward {
    table: RopeTable,
    shape: Vec<usize>,
}

impl GradFn for RopeRotateHalfBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // dx = g*cos + rotate_half^T(g*sin), rotate_half^T(y) = cat(y2, -y1):
        // the same pairwise rotation with the sine negated.
        let data = self.table.rotate(grad_output.data(), self.shape[1], -1.0);
        vec![Tensor::from_vec(data, &self.shape)]
    }

    fn name(&self) -> &'static str {
        "RopeRotateHalfBackward"
    }
}

// ============================================================================
// Rotary table
// ============================================================================

/// Per-position cos/sin of `p * inv_freq_i`, `inv_freq_i = theta^(-2i/dh)`, `i < dh/2`.
#[derive(Clone)]
pub(crate) struct RopeTable {
    head_dim: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl RopeTable {
    /// Computed in f32 the way HF `RotaryEmbedding` does: `inv_freq` in f32, then
    /// `angle = position * inv_freq` in f32, then cos/sin in f32.
    fn new(positions: &[usize], head_dim: usize, theta: f32) -> Self {
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / head_dim as f32))
            .collect();
        let mut cos = Vec::with_capacity(positions.len() * half);
        let mut sin = Vec::with_capacity(positions.len() * half);
        for &p in positions {
            for &f in &inv_freq {
                let angle = p as f32 * f;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        Self { head_dim, cos, sin }
    }

    /// `out1 = x1*c - sign*x2*s`, `out2 = x2*c + sign*x1*s` for every head of every row.
    fn rotate(&self, x: &[f32], width: usize, sign: f32) -> Vec<f32> {
        let half = self.head_dim / 2;
        let mut out = vec![0.0f32; x.len()];
        for (s, row) in x.chunks_exact(width).enumerate() {
            let cs = &self.cos[s * half..(s + 1) * half];
            let sn = &self.sin[s * half..(s + 1) * half];
            let dst = &mut out[s * width..(s + 1) * width];
            for (head, dhead) in row
                .chunks_exact(self.head_dim)
                .zip(dst.chunks_exact_mut(self.head_dim))
            {
                rotate_head(head, dhead, cs, sn, sign);
            }
        }
        out
    }
}

fn rotate_head(x: &[f32], out: &mut [f32], cos: &[f32], sin: &[f32], sign: f32) {
    let half = cos.len();
    for i in 0..half {
        let (x1, x2) = (x[i], x[half + i]);
        let s = sign * sin[i];
        out[i] = x1 * cos[i] - x2 * s;
        out[half + i] = x2 * cos[i] + x1 * s;
    }
}

// ============================================================================
// Tensor methods
// ============================================================================

impl Tensor {
    /// Narrow a 2D tensor to `[start, start+len)` along `dim` (0 = rows, 1 = columns).
    ///
    /// Backward scatters the gradient into zeros at the window (`NarrowBackward`).
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not 2D, `dim > 1`, `len == 0`, or the window exceeds
    /// the dimension.
    #[must_use]
    #[provable_contracts_macros::contract("modernbert-train-v1", equation = "narrow")]
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Tensor {
        let (rows, cols) = dims2(self, "narrow");
        assert!(dim <= 1, "narrow: dim must be 0 or 1, got {dim}");
        let extent = if dim == 0 { rows } else { cols };
        assert!(
            len > 0 && start.checked_add(len).is_some_and(|end| end <= extent),
            "narrow: window [{start}, {start}+{len}) outside dim {dim} of size {extent}"
        );
        let data = copy_window(self.data(), rows, cols, dim, start, len);
        let shape = if dim == 0 { [len, cols] } else { [rows, len] };
        let mut result = Tensor::from_vec(data, &shape);
        let grad_fn = Arc::new(NarrowBackward {
            rows,
            cols,
            dim,
            start,
            len,
        });
        record_op(&mut result, grad_fn, &[self]);
        result
    }

    /// Split a 2D tensor into `n` equal chunks along the last dim (columns).
    ///
    /// Built from [`Tensor::narrow`], so each chunk carries its own backward edge and
    /// their gradients accumulate into disjoint windows of the input.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not 2D, `n == 0`, or the column count is not divisible
    /// by `n`.
    #[must_use]
    #[provable_contracts_macros::contract("modernbert-train-v1", equation = "chunk")]
    pub fn chunk(&self, n: usize) -> Vec<Tensor> {
        let (_, cols) = dims2(self, "chunk");
        assert!(
            n > 0 && cols % n == 0,
            "chunk: {cols} columns not divisible into {n} chunks"
        );
        let width = cols / n;
        (0..n).map(|k| self.narrow(1, k * width, width)).collect()
    }

    /// Elementwise clamp to `[lo, hi]`; either bound may be `None`.
    ///
    /// Backward passes the gradient where `lo <= x <= hi` — INCLUSIVE at the bound,
    /// matching `torch.clamp` — and is zero strictly outside.
    ///
    /// # Panics
    ///
    /// Panics if both bounds are given and `lo > hi`.
    #[must_use]
    #[provable_contracts_macros::contract("modernbert-train-v1", equation = "clamp")]
    pub fn clamp(&self, lo: Option<f32>, hi: Option<f32>) -> Tensor {
        if let (Some(l), Some(h)) = (lo, hi) {
            assert!(l <= h, "clamp: lo {l} > hi {h}");
        }
        let data: Vec<f32> = self
            .data()
            .iter()
            .map(|&x| {
                let x = lo.map_or(x, |l| x.max(l));
                hi.map_or(x, |h| x.min(h))
            })
            .collect();
        let mut result = Tensor::from_vec(data, self.shape());
        let grad_fn = Arc::new(ClampBackward {
            x: self.clone(),
            lo,
            hi,
        });
        record_op(&mut result, grad_fn, &[self]);
        result
    }

    /// Elementwise `max(x, c)`. Defined as `clamp(Some(c), None)`, so its backward
    /// passes the gradient where `x >= c` (torch `clamp(min=c)` semantics).
    #[must_use]
    #[provable_contracts_macros::contract("modernbert-train-v1", equation = "max_scalar")]
    pub fn max_scalar(&self, c: f32) -> Tensor {
        self.clamp(Some(c), None)
    }

    /// Rotary position embedding, rotate-half layout, on `x: [S, heads*head_dim]`.
    ///
    /// `out = x*cos + rotate_half(x)*sin`, `rotate_half(x) = cat(-x2, x1)` per head,
    /// with angle `positions[s] * theta^(-2i/head_dim)`. `theta` is a parameter so the
    /// same op serves the global (160000) and local (10000) rotary bases.
    ///
    /// Backward: `dx = g*cos + rotate_half^T(g*sin)`, `rotate_half^T(y) = cat(y2, -y1)`.
    ///
    /// # Panics
    ///
    /// Panics if `x` is not 2D, `head_dim` is zero or odd, the width is not a multiple
    /// of `head_dim`, or `positions.len() != S`.
    #[must_use]
    #[provable_contracts_macros::contract("modernbert-train-v1", equation = "rope_rotate_half")]
    pub fn rope_rotate_half(&self, positions: &[usize], head_dim: usize, theta: f32) -> Tensor {
        let (seq, width) = dims2(self, "rope_rotate_half");
        assert!(
            head_dim > 0 && head_dim % 2 == 0 && width % head_dim == 0,
            "rope_rotate_half: head_dim {head_dim} must be even and divide width {width}"
        );
        assert_eq!(
            positions.len(),
            seq,
            "rope_rotate_half: positions.len() != seq"
        );
        let table = RopeTable::new(positions, head_dim, theta);
        let data = table.rotate(self.data(), width, 1.0);
        let mut result = Tensor::from_vec(data, self.shape());
        let grad_fn = Arc::new(RopeRotateHalfBackward {
            table,
            shape: self.shape().to_vec(),
        });
        record_op(&mut result, grad_fn, &[self]);
        result
    }
}

// ============================================================================
// Free functions
// ============================================================================

/// Concatenate 2D tensors along `dim` (0 = rows, 1 = columns).
///
/// Backward splits the gradient into each input's window (`ConcatBackward`).
///
/// # Panics
///
/// Panics if `tensors` is empty, any tensor is not 2D, `dim > 1`, or the inputs
/// disagree on the non-concatenated dimension.
#[must_use]
#[provable_contracts_macros::contract("modernbert-train-v1", equation = "concat")]
pub fn concat(tensors: &[&Tensor], dim: usize) -> Tensor {
    assert!(!tensors.is_empty(), "concat: no tensors");
    assert!(dim <= 1, "concat: dim must be 0 or 1, got {dim}");
    let shapes: Vec<(usize, usize)> = tensors.iter().map(|t| dims2(t, "concat")).collect();
    let keep = |s: &(usize, usize)| if dim == 0 { s.1 } else { s.0 };
    let fixed = keep(&shapes[0]);
    assert!(
        shapes.iter().all(|s| keep(s) == fixed),
        "concat: inputs disagree on the non-concatenated dim: {shapes:?}"
    );
    let total: usize = shapes
        .iter()
        .map(|s| if dim == 0 { s.0 } else { s.1 })
        .sum();
    let out_shape = if dim == 0 {
        [total, fixed]
    } else {
        [fixed, total]
    };
    let mut data = vec![0.0f32; total * fixed];
    let mut offset = 0;
    for (t, s) in tensors.iter().zip(&shapes) {
        let len = if dim == 0 { s.0 } else { s.1 };
        write_window(&mut data, out_shape[1], dim, offset, len, t.data());
        offset += len;
    }
    let mut result = Tensor::from_vec(data, &out_shape);
    let grad_fn = Arc::new(ConcatBackward { shapes, dim });
    record_op(&mut result, grad_fn, tensors);
    result
}

/// `true` when key `j` is inside query `i`'s inclusive local window.
fn in_window(i: usize, j: usize, half_window: usize) -> bool {
    i.abs_diff(j) <= half_window
}

/// The `[seq, seq]` sliding-window additive mask: `0` where `|i-j| <= half_window`
/// (INCLUSIVE), [`NEG_MASK`] elsewhere.
///
/// A CONSTANT: no differentiable input, so no `grad_fn` and `requires_grad == false`.
/// Uses the same finite `NEG_MASK` as `additive_attention_mask` so the two compose
/// without `-inf` arithmetic. The diagonal is always kept, so no row is fully masked.
///
/// # Errors
///
/// [`OpError::ZeroDimension`] when `seq == 0`.
#[provable_contracts_macros::contract("modernbert-train-v1", equation = "local_window_mask")]
pub fn local_window_mask(seq: usize, half_window: usize) -> Result<Tensor, OpError> {
    if seq == 0 {
        return Err(OpError::ZeroDimension { which: "seq" });
    }
    let data: Vec<f32> = (0..seq * seq)
        .map(|k| {
            if in_window(k / seq, k % seq, half_window) {
                0.0
            } else {
                NEG_MASK
            }
        })
        .collect();
    Ok(Tensor::from_vec(data, &[seq, seq]))
}

/// The `[batch, 1, seq, seq]` local-attention mask for a `[batch, seq]` binary
/// padding mask: `0` where the key is kept AND inside the inclusive window,
/// [`NEG_MASK`] otherwise. Broadcasts over `[B, heads, S, S]` scores.
///
/// The padding mask is validated by `additive_attention_mask` (binary values, one
/// kept position per row, length `batch*seq`). A CONSTANT, like
/// [`local_window_mask`].
///
/// # Errors
///
/// Every error `additive_attention_mask` returns for a malformed padding mask.
pub fn local_window_padding_mask(
    mask: &[u8],
    batch: usize,
    seq: usize,
    half_window: usize,
) -> Result<Tensor, OpError> {
    let pad = super::additive_attention_mask(mask, batch, seq)?;
    let window = local_window_mask(seq, half_window)?;
    let mut data = Vec::with_capacity(batch * seq * seq);
    for b in 0..batch {
        let keys = &pad.data()[b * seq..(b + 1) * seq];
        for (k, &w) in window.data().iter().enumerate() {
            let kept = w == 0.0 && keys[k % seq] == 0.0;
            data.push(if kept { 0.0 } else { NEG_MASK });
        }
    }
    Ok(Tensor::from_vec(data, &[batch, 1, seq, seq]))
}

/// Global-L2 gradient clipping over the autograd gradient store
/// (`torch.nn.utils.clip_grad_norm_` semantics).
///
/// `total = sqrt(sum over every listed gradient of g^2)` (accumulated in f64); when
/// `max_norm / (total + 1e-6) < 1` every listed gradient is scaled by that
/// coefficient. Returns the PRE-clip `total`. Ids with no stored gradient are
/// skipped; ids should be distinct. A non-finite total is returned without scaling.
///
/// # Panics
///
/// Panics if `max_norm` is not finite and positive.
#[provable_contracts_macros::contract("modernbert-train-v1", equation = "clip_grad_norm")]
pub fn clip_grad_norm_(params: &[TensorId], max_norm: f32) -> f32 {
    assert!(
        max_norm.is_finite() && max_norm > 0.0,
        "clip_grad_norm_: max_norm must be finite and > 0, got {max_norm}"
    );
    with_graph(|graph| {
        let sum_sq: f64 = params
            .iter()
            .filter_map(|&id| graph.get_grad(id))
            .flat_map(|g| {
                g.data()
                    .iter()
                    .map(|&v| f64::from(v) * f64::from(v))
                    .collect::<Vec<_>>()
            })
            .sum();
        let total = sum_sq.sqrt();
        let coef = f64::from(max_norm) / (total + 1e-6);
        if coef < 1.0 {
            for &id in params {
                scale_stored_grad(graph, id, coef as f32);
            }
        }
        total as f32
    })
}

fn scale_stored_grad(graph: &mut crate::autograd::ComputationGraph, id: TensorId, coef: f32) {
    let Some(tensor) = graph.get_tensor_mut(id) else {
        return;
    };
    let Some(grad) = tensor.grad() else {
        return;
    };
    let scaled: Vec<f32> = grad.data().iter().map(|&g| g * coef).collect();
    let scaled = Tensor::from_vec(scaled, grad.shape());
    tensor.clear_grad();
    tensor.accumulate_grad(scaled);
}
