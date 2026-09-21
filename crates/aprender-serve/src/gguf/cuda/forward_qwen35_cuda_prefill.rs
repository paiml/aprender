//! PMAT-3596 (#3596): batched (chunked) prefill for the Qwen3.5 hybrid on CUDA.
//!
//! [`Qwen35CudaModel::forward_single`] advances one token: every projection is a GEMV
//! that reads the whole weight for one activation row, every position runs the
//! `lm_head` and downloads a vocabulary of logits, and full attention walks the cache
//! with one block per head. Looping it over a prompt is what made a ~5.7k-token brief
//! take 235 s on the 9B (#3596). [`Qwen35CudaModel::prefill`] runs the same layer
//! programs over `T` tokens at a time:
//!
//! - every projection is ONE `dequant → f32 SGEMM` over the chunk's `T` rows
//!   (`CudaExecutor::qwen35_project_rows`, f32 end to end — see that function);
//! - the Gated `DeltaNet` recurrence is ONE launch of the chunk-resident scan, which is
//!   bitwise-equal to `T` per-token launches (the #3596 ruling's "chunked form");
//! - causal conv1d, the L2 norm, the gates and the partial RoPE run over rows with the
//!   same arithmetic as their per-token twins; the norms, splits and gates that are
//!   contiguous per head run as `T × heads` heads through the per-token wrappers;
//! - full attention is QKᵀ → causal softmax → PV with cuBLAS over the resident cache,
//!   after this chunk's K/V rows are written into it;
//! - only the LAST row of the LAST chunk goes through the output norm and `lm_head`.
//!
//! The per-layer op order is `deltanet_layer_inner` / `attention_layer_inner`,
//! verbatim; this file only changes how many rows each op sees. The state
//! (`conv`, `ssm`, `kv`, `kv_len`) is left exactly where `T` calls of `forward_single`
//! would leave it, so decode continues from it unchanged.

use super::super::{RealizarError, Result};
use super::{gpu_err, CudaLayer, Qwen35CudaDims, Qwen35CudaModel, Qwen35CudaState};
use crate::gguf::forward_qwen35::{Qwen35Model, Qwen35OwnedLayer};
use trueno_gpu::driver::GpuBuffer;

/// Rows per chunk when the attention scores do not bound it lower.
pub const PREFILL_MAX_CHUNK_ROWS: usize = 512;

/// Rows per chunk never go below this, however long the context: a chunk of a few
/// rows is back to GEMV-shaped work.
pub const PREFILL_MIN_CHUNK_ROWS: usize = 16;

/// Bytes the attention scores of one KV group may take (`heads_per_kv × T × L × 4`).
pub const PREFILL_SCORES_BUDGET_BYTES: usize = 1 << 30;

/// A non-owning device view: never frees what it points at.
struct View(std::mem::ManuallyDrop<GpuBuffer<f32>>);

impl View {
    fn at(ptr: u64, elems: usize) -> Self {
        // SAFETY: every call site passes a pointer into a live allocation with at
        // least `elems` floats after it, and `ManuallyDrop` means the view never frees.
        Self(std::mem::ManuallyDrop::new(unsafe {
            GpuBuffer::<f32>::from_raw_parts(ptr, elems)
        }))
    }
}

impl std::ops::Deref for View {
    type Target = GpuBuffer<f32>;
    fn deref(&self) -> &GpuBuffer<f32> {
        &self.0
    }
}

/// Device buffers for one chunk of `rows` rows, allocated once per [`prefill`] call.
///
/// [`prefill`]: Qwen35CudaModel::prefill
struct PrefillBuffers {
    rows: usize,
    /// `[rows][hidden]` — the residual stream.
    x: GpuBuffer<f32>,
    normed: GpuBuffer<f32>,
    post_normed: GpuBuffer<f32>,
    /// `[rows][hidden]` — the output of `ssm_out` / `attn_output` / `ffn_down`.
    proj: GpuBuffer<f32>,
    // Gated DeltaNet
    conv_in: GpuBuffer<f32>,
    conv_out: GpuBuffer<f32>,
    alpha_raw: GpuBuffer<f32>,
    beta_raw: GpuBuffer<f32>,
    dt: GpuBuffer<f32>,
    beta: GpuBuffer<f32>,
    gate: GpuBuffer<f32>,
    out_h: GpuBuffer<f32>,
    ssm_out_in: GpuBuffer<f32>,
    // full attention
    q_full: GpuBuffer<f32>,
    q: GpuBuffer<f32>,
    q_normed: GpuBuffer<f32>,
    attn_gate: GpuBuffer<f32>,
    k_raw: GpuBuffer<f32>,
    attn_out_in: GpuBuffer<f32>,
    scores: GpuBuffer<f32>,
    // FFN
    ffn_gate: GpuBuffer<f32>,
    ffn_up: GpuBuffer<f32>,
    ffn_act: GpuBuffer<f32>,
}

/// Rows per chunk for a prompt ending at `total_positions` — one formula for the
/// running prefill and the pre-load capacity plan.
fn chunk_rows_for(d: Qwen35CudaDims, total_positions: usize) -> usize {
    let hpk = (d.num_heads / d.num_kv_heads.max(1)).max(1) as usize;
    let by_scores = PREFILL_SCORES_BUDGET_BYTES / (4 * hpk * total_positions.max(1));
    by_scores.clamp(PREFILL_MIN_CHUNK_ROWS, PREFILL_MAX_CHUNK_ROWS)
}

/// Device bytes a prefill ending at `total_positions` allocates beyond weights and
/// state: its chunk buffers, the attention scores and the f32 dequant scratch.
fn workspace_bytes_for(
    d: Qwen35CudaDims,
    largest_projection: usize,
    total_positions: usize,
) -> usize {
    let rows = chunk_rows_for(d, total_positions);
    let hpk = (d.num_heads / d.num_kv_heads.max(1)).max(1) as usize;
    let q_dim = (d.num_heads * d.attn_head_dim) as usize;
    let kv_dim = (d.num_kv_heads * d.attn_head_dim) as usize;
    let (hidden, inter) = (d.hidden_dim as usize, d.intermediate_dim as usize);
    let (conv, v, nv) = (
        d.conv_dim as usize,
        d.v_dim as usize,
        d.num_v_heads as usize,
    );
    let per_row = 4 * hidden
        + 2 * conv
        + 4 * nv
        + 3 * v
        + 2 * q_dim // q_full
        + 4 * q_dim // q, q_normed, attn_gate, attn_out_in
        + kv_dim
        + 3 * inter;
    let scores = hpk * rows * total_positions;
    4 * (rows * per_row + scores + largest_projection)
}

/// Bytes an [`OwnedQuantizedTensor`](crate::gguf::OwnedQuantizedTensor) occupies once
/// uploaded, and its `n x k`.
fn quant_bytes(t: &crate::gguf::OwnedQuantizedTensor) -> (u64, usize) {
    (t.data.len() as u64, t.out_dim * t.in_dim)
}

impl<'a> Qwen35CudaModel<'a> {
    /// Rows per prefill chunk for a prompt that ends at `total_positions`.
    ///
    /// The scores of one KV group are `heads_per_kv × T × L` floats and must fit
    /// [`PREFILL_SCORES_BUDGET_BYTES`] at the LAST chunk, where `L` is largest; the
    /// chunk size is fixed for the whole call so every chunk but the last reuses one
    /// set of compiled shapes.
    #[must_use]
    pub fn prefill_chunk_rows(&self, total_positions: usize) -> usize {
        chunk_rows_for(self.dims, total_positions)
    }

    /// Device bytes a [`Self::prefill`] of a prompt ending at `total_positions`
    /// allocates on top of the weights and the state.
    #[must_use]
    pub fn prefill_workspace_bytes(&self, total_positions: usize) -> usize {
        workspace_bytes_for(self.dims, self.largest_projection_elems(), total_positions)
    }

    /// The capacity-plan inputs for serving `model` on a device with `gpu_free` of
    /// `gpu_total` bytes, for a request of `seq_len` positions (prompt + generated +
    /// 1) — computed from the HOST model, before anything is uploaded (#3596).
    ///
    /// Weights are the bytes [`Self::with_max_seq_len`] uploads (every layer's
    /// quantized projections and f32 vectors, the output norm twice, the `lm_head`);
    /// the KV term covers only the layers that have a cache (the full-attention ones);
    /// the workspace is the prefill's at this length plus the decode state's recurrent
    /// windows and the model's own capped state.
    #[must_use]
    pub fn capacity_inputs(
        model: &Qwen35Model<'_>,
        seq_len: usize,
        gpu_free: u64,
        gpu_total: u64,
    ) -> crate::capacity::CapacityInputs {
        let d = Self::dims_of(model);
        let f32s = |v: &[f32]| 4 * v.len() as u64;
        let mut weights = 0u64;
        let mut largest = 0usize;
        let mut attn_layers = 0u64;
        for layer in &model.layers {
            let (vecs, quants): (Vec<&[f32]>, Vec<&crate::gguf::OwnedQuantizedTensor>) = match layer
            {
                Qwen35OwnedLayer::DeltaNet(l) => (
                    vec![
                        &l.attn_norm,
                        &l.ssm_a,
                        &l.ssm_dt_bias,
                        &l.ssm_conv1d_weight,
                        &l.ssm_norm_weight,
                        &l.post_attention_norm,
                    ],
                    vec![
                        &l.attn_qkv,
                        &l.attn_gate,
                        &l.ssm_alpha,
                        &l.ssm_beta,
                        &l.ssm_out,
                        &l.ffn_gate,
                        &l.ffn_up,
                        &l.ffn_down,
                    ],
                ),
                Qwen35OwnedLayer::Attention(l) => {
                    attn_layers += 1;
                    (
                        vec![
                            &l.attn_norm,
                            &l.attn_q_norm,
                            &l.attn_k_norm,
                            &l.post_attention_norm,
                        ],
                        vec![
                            &l.attn_q,
                            &l.attn_k,
                            &l.attn_v,
                            &l.attn_output,
                            &l.ffn_gate,
                            &l.ffn_up,
                            &l.ffn_down,
                        ],
                    )
                },
            };
            weights += vecs.iter().map(|v| f32s(v)).sum::<u64>();
            for q in quants {
                let (bytes, elems) = quant_bytes(q);
                weights += bytes;
                largest = largest.max(elems);
            }
        }
        weights +=
            2 * f32s(model.base.output_norm_weight()) + quant_bytes(model.base.lm_head_weight()).0;

        let kv_row = u64::from(d.num_kv_heads * d.attn_head_dim);
        let kv_bytes_per_token_f32 = attn_layers * 2 * kv_row * 4;
        let recurrent_per_state = model.layers.len() as u64
            * 4
            * u64::from(
                d.conv_dim * (d.conv_kernel - 1) + d.num_v_heads * d.head_v_dim * d.head_k_dim,
            );
        let own_state = recurrent_per_state
            + kv_bytes_per_token_f32 * seq_len.min(super::DEFAULT_MAX_SEQ_LEN) as u64;
        let per_token_scratch = 4 * u64::from(
            8 * d.hidden_dim
                + 2 * d.conv_dim
                + 4 * d.v_dim
                + 8 * d.num_heads * d.attn_head_dim
                + 4 * d.intermediate_dim
                + d.vocab_size,
        );
        let workspace = workspace_bytes_for(d, largest, seq_len) as u64
            + recurrent_per_state // the decode state's conv/ssm (its KV is the KV term)
            + own_state
            + per_token_scratch;
        crate::capacity::CapacityInputs {
            weights_bytes: weights,
            kv_bytes_per_token_f32,
            seq_len: seq_len as u64,
            workspace_bytes: workspace,
            overhead_bytes: crate::capacity::OVERHEAD_BYTES,
            gpu_free_bytes: gpu_free,
            gpu_total_bytes: gpu_total,
            // Flipped by #3725, whose split-K decode reads an f16 cache; until then a
            // plan that needs f16 refuses and names it.
            f16_kv_decode_available: false,
        }
    }

    /// `n × k` of the largest projection — the size of the f32 dequant scratch.
    fn largest_projection_elems(&self) -> usize {
        self.layers
            .iter()
            .flat_map(|l| match l {
                CudaLayer::DeltaNet(w) => vec![
                    &w.attn_qkv,
                    &w.ssm_alpha,
                    &w.ssm_beta,
                    &w.attn_gate,
                    &w.ssm_out,
                    &w.ffn_gate,
                    &w.ffn_up,
                    &w.ffn_down,
                ],
                CudaLayer::Attention(w) => vec![
                    &w.attn_q,
                    &w.attn_k,
                    &w.attn_v,
                    &w.attn_output,
                    &w.ffn_gate,
                    &w.ffn_up,
                    &w.ffn_down,
                ],
            })
            .map(|q| q.n as usize * q.k as usize)
            .max()
            .unwrap_or(0)
    }

    fn alloc_prefill(&self, rows: usize, total_positions: usize) -> Result<PrefillBuffers> {
        let d = self.dims;
        let ctx = self.executor.context();
        // Uninitialised: every buffer is written by a copy or a kernel before any op
        // reads it, and a zero-fill would cost a host vector the size of the scores.
        let z = |n: usize| {
            GpuBuffer::<f32>::new(ctx, n.max(1))
                .map_err(|e| gpu_err("qwen35_cuda_prefill_alloc", &e))
        };
        let hpk = (d.num_heads / d.num_kv_heads.max(1)).max(1) as usize;
        let q_dim = (d.num_heads * d.attn_head_dim) as usize;
        let kv_dim = (d.num_kv_heads * d.attn_head_dim) as usize;
        let (hidden, inter) = (d.hidden_dim as usize, d.intermediate_dim as usize);
        let (conv, v, nv) = (
            d.conv_dim as usize,
            d.v_dim as usize,
            d.num_v_heads as usize,
        );
        Ok(PrefillBuffers {
            rows,
            x: z(rows * hidden)?,
            normed: z(rows * hidden)?,
            post_normed: z(rows * hidden)?,
            proj: z(rows * hidden)?,
            conv_in: z(rows * conv)?,
            conv_out: z(rows * conv)?,
            alpha_raw: z(rows * nv)?,
            beta_raw: z(rows * nv)?,
            dt: z(rows * nv)?,
            beta: z(rows * nv)?,
            gate: z(rows * v)?,
            out_h: z(rows * v)?,
            ssm_out_in: z(rows * v)?,
            q_full: z(rows * 2 * q_dim)?,
            q: z(rows * q_dim)?,
            q_normed: z(rows * q_dim)?,
            attn_gate: z(rows * q_dim)?,
            k_raw: z(rows * kv_dim)?,
            attn_out_in: z(rows * q_dim)?,
            scores: z(hpk * rows * total_positions)?,
            ffn_gate: z(rows * inter)?,
            ffn_up: z(rows * inter)?,
            ffn_act: z(rows * inter)?,
        })
    }

    /// Prefill `tokens` at positions `pos0..pos0 + tokens.len()` and return the logits
    /// of the LAST position — what `tokens.len()` calls of [`Self::forward_single`]
    /// return from their last call, with `state` advanced to the same place.
    ///
    /// # Errors
    /// An empty prompt, a position past the state's `max_seq_len`, a token outside the
    /// vocabulary, a projection whose quantization has no dequant kernel, or any
    /// device failure. Nothing is half-applied that a caller could mistake for a
    /// finished prefill: on `Err` the state must be discarded.
    pub fn prefill(
        &mut self,
        tokens: &[u32],
        state: &mut Qwen35CudaState,
        pos0: usize,
    ) -> Result<Vec<f32>> {
        self.prefill_check(tokens, state, pos0)?;
        let end = pos0 + tokens.len();
        let rows = self.prefill_chunk_rows(end);
        let bufs = self.alloc_prefill(rows, end)?;
        let mut pos = pos0;
        let mut last_rows = 0;
        for chunk in tokens.chunks(rows) {
            self.prefill_chunk(&bufs, chunk, state, pos)?;
            pos += chunk.len();
            last_rows = chunk.len();
        }
        self.prefill_tail(&bufs, last_rows - 1)
    }

    /// [`Self::prefill`], returning the logits of EVERY position — for the F2 guard,
    /// which compares the GPU against the CPU position by position over its probe.
    ///
    /// Each row pays an `lm_head` GEMV, as the per-token path does; a real prompt goes
    /// through [`Self::prefill`], which pays one.
    ///
    /// # Errors
    /// As [`Self::prefill`].
    pub fn prefill_logits_every_row(
        &mut self,
        tokens: &[u32],
        state: &mut Qwen35CudaState,
        pos0: usize,
    ) -> Result<Vec<Vec<f32>>> {
        self.prefill_check(tokens, state, pos0)?;
        let end = pos0 + tokens.len();
        let rows = self.prefill_chunk_rows(end);
        let bufs = self.alloc_prefill(rows, end)?;
        let mut pos = pos0;
        let mut out = Vec::with_capacity(tokens.len());
        for chunk in tokens.chunks(rows) {
            self.prefill_chunk(&bufs, chunk, state, pos)?;
            for r in 0..chunk.len() {
                out.push(self.prefill_tail(&bufs, r)?);
            }
            pos += chunk.len();
        }
        Ok(out)
    }

    /// The refusals every prefill entry point shares.
    fn prefill_check(&self, tokens: &[u32], state: &Qwen35CudaState, pos0: usize) -> Result<()> {
        if tokens.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "qwen35_cuda prefill: the prompt is empty".to_string(),
            });
        }
        let end = pos0 + tokens.len();
        if end > state.max_seq_len {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "qwen35_cuda prefill: positions {pos0}..{end} are past the KV cache ({} rows)",
                    state.max_seq_len
                ),
            });
        }
        let hidden = self.dims.hidden_dim as usize;
        let vocab_rows = self.model.base.token_embedding().len() / hidden;
        if let Some(bad) = tokens.iter().find(|&&t| t as usize >= vocab_rows) {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "qwen35_cuda prefill: token {bad} is outside the {vocab_rows}-row embedding table"
                ),
            });
        }

        Ok(())
    }

    /// Output norm + `lm_head` on row `row` of the chunk `bufs.x` holds.
    fn prefill_tail(&mut self, bufs: &PrefillBuffers, row: usize) -> Result<Vec<f32>> {
        let d = self.dims;
        let hidden = d.hidden_dim as usize;
        let last = View::at(bufs.x.as_ptr() + (row * hidden * 4) as u64, hidden);
        let err = |e: trueno_gpu::GpuError| gpu_err("qwen35_cuda_prefill_tail", &e);
        self.executor
            .rmsnorm_into(
                &last,
                &self.output_norm,
                &self.out_normed,
                d.hidden_dim,
                d.eps,
            )
            .map_err(err)?;
        self.executor
            .gemv_dispatch(
                self.lm_head.qtype,
                self.lm_head.ptr,
                &self.out_normed,
                &self.logits_buf,
                self.lm_head.n,
                self.lm_head.k,
            )
            .map_err(err)?;
        self.executor.sync_stream().map_err(err)?;
        let mut logits = vec![0.0f32; d.vocab_size as usize];
        self.logits_buf.copy_to_host(&mut logits).map_err(err)?;
        Ok(logits)
    }

    /// One chunk of `chunk.len()` rows at positions `pos..pos + chunk.len()`.
    fn prefill_chunk(
        &mut self,
        bufs: &PrefillBuffers,
        chunk: &[u32],
        state: &mut Qwen35CudaState,
        pos: usize,
    ) -> Result<()> {
        let hidden = self.dims.hidden_dim as usize;
        let n = chunk.len();
        debug_assert!(n <= bufs.rows);
        let err = |e: trueno_gpu::GpuError| gpu_err("qwen35_cuda_prefill", &e);

        // The previous chunk's kernels read `x` until its last layer; the host copy
        // below is not ordered with this model's (non-blocking) stream, so wait.
        self.executor.sync_stream().map_err(err)?;
        let table = self.model.base.token_embedding();
        let mut host = Vec::with_capacity(n * hidden);
        for &t in chunk {
            let start = t as usize * hidden;
            host.extend_from_slice(&table[start..start + hidden]);
        }
        let mut x = View::at(bufs.x.as_ptr(), bufs.rows * hidden);
        x.0.copy_from_host_at(&host, 0).map_err(err)?;

        for il in 0..self.layers.len() {
            match self.layers[il] {
                CudaLayer::DeltaNet(_) => self
                    .prefill_deltanet(bufs, state, il, n)
                    .map_err(|e| gpu_err("qwen35_cuda_prefill_deltanet", &e))?,
                CudaLayer::Attention(_) => self
                    .prefill_attention(bufs, state, il, n, pos)
                    .map_err(|e| gpu_err("qwen35_cuda_prefill_attention", &e))?,
            }
        }
        state.kv_len = state.kv_len.max(pos + n);
        Ok(())
    }

    /// `deltanet_layer_inner` over `n` rows.
    #[allow(clippy::too_many_lines)]
    fn prefill_deltanet(
        &mut self,
        b: &PrefillBuffers,
        state: &Qwen35CudaState,
        il: usize,
        n: usize,
    ) -> std::result::Result<(), trueno_gpu::GpuError> {
        let d = self.dims;
        let CudaLayer::DeltaNet(w) = &self.layers[il] else {
            unreachable!("the caller matched a DeltaNet layer")
        };
        let ex = &mut self.executor;
        let rows = n as u32;
        let f = 4u64;

        // rms_norm(x, attn_norm)
        ex.batched_rmsnorm_into(&b.x, &w.attn_norm, &b.normed, d.hidden_dim, rows, d.eps)?;

        // attn_qkv . normed -> conv_in
        let p = &w.attn_qkv;
        ex.qwen35_project_rows(
            p.qtype,
            p.ptr,
            b.normed.as_ptr(),
            b.conv_in.as_ptr(),
            rows,
            p.n,
            p.k,
            p.n,
        )?;

        // causal conv1d + SiLU over the chunk (the window advances in place)
        ex.qwen35_conv1d_rows(
            b.conv_in.as_ptr(),
            state.conv[il].as_ptr(),
            w.conv1d_weight.as_ptr(),
            b.conv_out.as_ptr(),
            d.conv_dim,
            d.conv_kernel,
            rows,
        )?;

        // per-head L2 on the q and k sections of every conv_out row
        let conv_base = b.conv_out.as_ptr();
        ex.qwen35_l2_norm_rows(
            conv_base,
            d.head_k_dim,
            d.num_k_heads,
            d.eps,
            d.conv_dim,
            rows,
        )?;
        ex.qwen35_l2_norm_rows(
            conv_base + u64::from(d.k_dim) * f,
            d.head_k_dim,
            d.num_k_heads,
            d.eps,
            d.conv_dim,
            rows,
        )?;

        // dt = softplus(ssm_alpha . x + dt_bias) * a ; beta = sigmoid(ssm_beta . x)
        let p = &w.ssm_alpha;
        ex.qwen35_project_rows(
            p.qtype,
            p.ptr,
            b.normed.as_ptr(),
            b.alpha_raw.as_ptr(),
            rows,
            p.n,
            p.k,
            p.n,
        )?;
        let p = &w.ssm_beta;
        ex.qwen35_project_rows(
            p.qtype,
            p.ptr,
            b.normed.as_ptr(),
            b.beta_raw.as_ptr(),
            rows,
            p.n,
            p.k,
            p.n,
        )?;
        ex.qwen35_gates_rows(
            b.alpha_raw.as_ptr(),
            w.ssm_dt_bias.as_ptr(),
            w.ssm_a.as_ptr(),
            b.beta_raw.as_ptr(),
            b.dt.as_ptr(),
            b.beta.as_ptr(),
            d.num_v_heads,
            rows,
        )?;

        // attn_gate . x
        let p = &w.attn_gate;
        ex.qwen35_project_rows(
            p.qtype,
            p.ptr,
            b.normed.as_ptr(),
            b.gate.as_ptr(),
            rows,
            p.n,
            p.k,
            p.n,
        )?;

        // the delta rule over the chunk (the recurrent state stays on chip)
        ex.qwen35_delta_rule_scan(
            conv_base,
            conv_base + u64::from(d.k_dim) * f,
            conv_base + u64::from(2 * d.k_dim) * f,
            b.beta.as_ptr(),
            b.dt.as_ptr(),
            state.ssm[il].as_ptr(),
            b.out_h.as_ptr(),
            (d.num_k_heads, d.head_k_dim, d.num_v_heads, d.head_v_dim),
            d.conv_dim,
            rows,
        )?;

        // gated rmsnorm (rows x num_v_heads heads, weight shared), ssm_out, residual
        ex.gdn_gated_rmsnorm_into(
            &b.out_h,
            &b.gate,
            &w.ssm_norm_weight,
            &b.ssm_out_in,
            d.head_v_dim,
            rows * d.num_v_heads,
            d.eps,
        )?;
        let p = &w.ssm_out;
        ex.qwen35_project_rows(
            p.qtype,
            p.ptr,
            b.ssm_out_in.as_ptr(),
            b.proj.as_ptr(),
            rows,
            p.n,
            p.k,
            p.n,
        )?;
        ex.residual_add_into(&b.x, &b.proj, &b.x, rows * d.hidden_dim)?;

        Self::prefill_ffn_rows(
            ex,
            b,
            d,
            rows,
            &w.post_attention_norm,
            [&w.ffn_gate, &w.ffn_up, &w.ffn_down],
        )
    }

    /// `attention_layer_inner` over `n` rows at positions `pos..pos + n`.
    #[allow(clippy::too_many_lines)]
    fn prefill_attention(
        &mut self,
        b: &PrefillBuffers,
        state: &Qwen35CudaState,
        il: usize,
        n: usize,
        pos: usize,
    ) -> std::result::Result<(), trueno_gpu::GpuError> {
        let d = self.dims;
        let CudaLayer::Attention(w) = &self.layers[il] else {
            unreachable!("the caller matched an attention layer")
        };
        let ex = &mut self.executor;
        let rows = n as u32;
        let q_dim = d.num_heads * d.attn_head_dim;
        let kv_dim = d.num_kv_heads * d.attn_head_dim;
        let (k_cache, v_cache) = state.kv[il]
            .as_ref()
            .expect("an attention layer owns a KV cache");
        // This chunk's rows of the cache: [pos, pos + n) x kv_dim, contiguous.
        let row_off = (pos as u64) * u64::from(kv_dim) * 4;
        let k_rows = View::at(k_cache.as_ptr() + row_off, n * kv_dim as usize);
        let v_rows_ptr = v_cache.as_ptr() + row_off;

        // rms_norm(x, attn_norm)
        ex.batched_rmsnorm_into(&b.x, &w.attn_norm, &b.normed, d.hidden_dim, rows, d.eps)?;

        // attn_q -> [q | gate] per head; attn_k -> k_raw; attn_v -> straight into the cache
        let p = &w.attn_q;
        ex.qwen35_project_rows(
            p.qtype,
            p.ptr,
            b.normed.as_ptr(),
            b.q_full.as_ptr(),
            rows,
            p.n,
            p.k,
            p.n,
        )?;
        let p = &w.attn_k;
        ex.qwen35_project_rows(
            p.qtype,
            p.ptr,
            b.normed.as_ptr(),
            b.k_raw.as_ptr(),
            rows,
            p.n,
            p.k,
            p.n,
        )?;
        let p = &w.attn_v;
        ex.qwen35_project_rows(
            p.qtype,
            p.ptr,
            b.normed.as_ptr(),
            v_rows_ptr,
            rows,
            p.n,
            p.k,
            kv_dim,
        )?;

        // split [q | gate] over rows x num_heads heads
        ex.gdn_split_interleaved_into(
            &b.q_full,
            &b.q,
            &b.attn_gate,
            rows * d.num_heads,
            d.attn_head_dim,
        )?;

        // per-head RMSNorm on q, and on k straight into its cache rows
        ex.per_head_rmsnorm_into(
            &b.q,
            &w.attn_q_norm,
            &b.q_normed,
            d.attn_head_dim,
            rows * d.num_heads,
            d.eps,
        )?;
        ex.per_head_rmsnorm_into(
            &b.k_raw,
            &w.attn_k_norm,
            &k_rows,
            d.attn_head_dim,
            rows * d.num_kv_heads,
            d.eps,
        )?;

        // partial NEOX RoPE, row t at position pos + t
        let pos32 = u32::try_from(pos).unwrap_or(u32::MAX);
        ex.qwen35_rope_rows(
            b.q_normed.as_ptr(),
            d.num_heads,
            d.attn_head_dim,
            d.n_rot,
            q_dim,
            rows,
            pos32,
            d.theta_scale,
        )?;
        ex.qwen35_rope_rows(
            k_rows.as_ptr(),
            d.num_kv_heads,
            d.attn_head_dim,
            d.n_rot,
            kv_dim,
            rows,
            pos32,
            d.theta_scale,
        )?;

        // causal attention over cache rows 0..pos + n
        ex.qwen35_prefill_attention(
            b.q_normed.as_ptr(),
            k_cache.as_ptr(),
            v_cache.as_ptr(),
            b.attn_out_in.as_ptr(),
            b.scores.as_ptr(),
            rows,
            pos32,
            d.num_heads,
            d.num_kv_heads,
            d.attn_head_dim,
        )?;

        // the output gate, attn_output, the first residual
        ex.gdn_sigmoid_gate_into(&b.attn_out_in, &b.attn_gate, rows * q_dim)?;
        let p = &w.attn_output;
        ex.qwen35_project_rows(
            p.qtype,
            p.ptr,
            b.attn_out_in.as_ptr(),
            b.proj.as_ptr(),
            rows,
            p.n,
            p.k,
            p.n,
        )?;
        ex.residual_add_into(&b.x, &b.proj, &b.x, rows * d.hidden_dim)?;

        Self::prefill_ffn_rows(
            ex,
            b,
            d,
            rows,
            &w.post_attention_norm,
            [&w.ffn_gate, &w.ffn_up, &w.ffn_down],
        )
    }

    /// post_attention_norm → SwiGLU FFN → the second residual, over `rows` rows. Both
    /// layer kinds share it, as their per-token bodies do.
    fn prefill_ffn_rows(
        ex: &mut crate::cuda::CudaExecutor,
        b: &PrefillBuffers,
        d: super::Qwen35CudaDims,
        rows: u32,
        post_norm: &GpuBuffer<f32>,
        [gate, up, down]: [&super::CudaQuantWeight; 3],
    ) -> std::result::Result<(), trueno_gpu::GpuError> {
        ex.batched_rmsnorm_into(&b.x, post_norm, &b.post_normed, d.hidden_dim, rows, d.eps)?;
        ex.qwen35_project_rows(
            gate.qtype,
            gate.ptr,
            b.post_normed.as_ptr(),
            b.ffn_gate.as_ptr(),
            rows,
            gate.n,
            gate.k,
            gate.n,
        )?;
        ex.qwen35_project_rows(
            up.qtype,
            up.ptr,
            b.post_normed.as_ptr(),
            b.ffn_up.as_ptr(),
            rows,
            up.n,
            up.k,
            up.n,
        )?;
        ex.fused_swiglu_into(
            &b.ffn_gate,
            &b.ffn_up,
            &b.ffn_act,
            rows * d.intermediate_dim,
        )?;
        ex.qwen35_project_rows(
            down.qtype,
            down.ptr,
            b.ffn_act.as_ptr(),
            b.proj.as_ptr(),
            rows,
            down.n,
            down.k,
            down.n,
        )?;
        ex.residual_add_into(&b.x, &b.proj, &b.x, rows * d.hidden_dim)
    }
}

/// Batched prefill against the per-token path on the real files.
#[cfg(test)]
#[path = "forward_qwen35_cuda_prefill_tests.rs"]
mod prefill_tests;
