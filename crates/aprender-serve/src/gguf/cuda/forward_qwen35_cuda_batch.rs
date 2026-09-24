//! #4234: one decode step for several independent Qwen3.5 sequences on one device.
//!
//! [`Qwen35CudaModel::forward_single`] advances one sequence by one token. A server
//! with `B` streams in flight would call it `B` times per step: `B` host syncs, `B`
//! logits downloads, and every layer's weights streamed from DRAM `B` times, one
//! whole forward apart.
//!
//! [`Qwen35CudaModel::forward_batch`] runs the same step for `B` sequences
//! **layer-major**: layer `il` runs for sequence 0, then sequence 1, … before layer
//! `il + 1` runs for anyone. Each sequence keeps its own [`Qwen35CudaState`] (conv
//! windows, recurrent states, KV caches) and its own position, so the sequences can be
//! at different lengths. What the batch buys:
//!
//! - ONE sync and ONE logits download for the whole step, not `B`;
//! - a layer's weights are read by `B` back-to-back GEMVs, so the reads after the
//!   first hit L2 when the layer fits there (the 0.8B's layers do on a 72 MB-L2
//!   RTX 4090).
//!
//! What it deliberately does NOT change: every op is the same kernel on the same
//! inputs as in `forward_single`, launched in the same per-sequence order on the one
//! stream, and the single-row scratch is reused sequence after sequence in stream
//! order. So each sequence's logits are **bitwise** the logits `forward_single` would
//! give it — the test asserts equal bits, not a tolerance. A batched GEMV kernel (one
//! weight read for all `B` rows) is a separate step: it changes the reduction order,
//! and it has to be measured against this bitwise baseline, not assumed.

use super::super::{RealizarError, Result};
use super::{gpu_err, CudaLayer, Qwen35CudaModel, Qwen35CudaState};
use crate::cuda::types::WeightQuantType;
use trueno_gpu::driver::GpuBuffer;

/// `forward_batch`'s packed per-step buffers, reused across steps: the hidden
/// rows and the normed rows (`[rows][hidden_dim]`, the latter shared by the FFN
/// input and the output norm), the FFN's gate / up / activation
/// (`[rows][intermediate_dim]`) and down projection (`[rows][hidden_dim]`), and
/// the logits (`[rows][vocab_size]`), and the DeltaNet mixer's projections: qkv
/// (`[rows][conv_dim]`), gate and the gated-norm output (`[rows][v_dim]`), alpha
/// and beta (`[rows][num_v_heads]`), and the attention mixer's `[q | gate]`
/// (`[rows][2 * q_dim]`), raw k (`[rows][kv_dim]`) and gated attention output
/// (`[rows][q_dim]`). A step of `B <= rows` uses the first `B` rows of each.
pub(super) struct BatchIo {
    hidden: GpuBuffer<f32>,
    normed: GpuBuffer<f32>,
    dn_qkv: GpuBuffer<f32>,
    dn_gate: GpuBuffer<f32>,
    dn_alpha: GpuBuffer<f32>,
    dn_beta: GpuBuffer<f32>,
    dn_core: GpuBuffer<f32>,
    at_q_full: GpuBuffer<f32>,
    at_k_raw: GpuBuffer<f32>,
    at_out_in: GpuBuffer<f32>,
    ffn_gate: GpuBuffer<f32>,
    ffn_up: GpuBuffer<f32>,
    ffn_act: GpuBuffer<f32>,
    ffn_down: GpuBuffer<f32>,
    logits: GpuBuffer<f32>,
    rows: usize,
}

impl Qwen35CudaModel<'_> {
    /// Advance `B = tokens.len()` independent sequences by one token each and return
    /// their logits, `B` rows of `vocab_size`, in input order.
    ///
    /// Sequence `b` feeds `tokens[b]` at `positions[b]` into `states[b]`, exactly as
    /// `forward_single(tokens[b], states[b], positions[b])` would, and its logits are
    /// bitwise equal to what that call returns (see the module docs).
    ///
    /// Every input is checked before anything is enqueued, so a refused batch leaves
    /// every state untouched. A GPU failure mid-step leaves the states partly
    /// advanced, as a failed `forward_single` does; the caller resets them.
    ///
    /// # Errors
    /// An empty batch, slices of different lengths, a token id
    /// outside the vocabulary, a position past its state's `max_seq_len`, or any
    /// kernel / GEMV / transfer failure.
    pub fn forward_batch(
        &mut self,
        tokens: &[u32],
        states: &mut [&mut Qwen35CudaState],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        self.check_batch(tokens, states, positions)?;
        let batch = tokens.len();
        let hidden_dim = self.dims.hidden_dim as usize;
        let vocab = self.dims.vocab_size as usize;

        // Every sequence's embedding row, uploaded as ONE `[B][hidden_dim]` buffer.
        let embedding = self.model.base.token_embedding();
        let mut rows = Vec::with_capacity(batch * hidden_dim);
        for &token in tokens {
            let start = (token as usize) * hidden_dim;
            rows.extend_from_slice(&embedding[start..start + hidden_dim]);
        }
        let mut io = self.take_batch_io(batch)?;
        let outcome = io
            .hidden
            .copy_from_host_at(&rows, 0)
            .map_err(|e| gpu_err("qwen35_cuda_forward_batch", &e))
            .and_then(|()| self.forward_batch_enqueued(&io, batch, states, positions));
        let mut host = vec![0.0f32; batch * vocab];
        let downloaded = outcome.and_then(|()| {
            // The ONE sync of the whole step, in front of the ONE download — of
            // this batch's rows only; a wider earlier batch sized the buffer.
            self.executor
                .sync_stream()
                .map_err(|e| gpu_err("qwen35_cuda_forward_batch", &e))?;
            io.logits
                .copy_to_host(&mut host)
                .map_err(|e| gpu_err("qwen35_cuda_forward_batch", &e))
        });
        self.batch_io = Some(io);
        downloaded?;

        for (state, &position) in states.iter_mut().zip(positions) {
            state.kv_len = state.kv_len.max(position + 1);
        }
        Ok(host.chunks_exact(vocab).map(<[f32]>::to_vec).collect())
    }

    /// The step's buffers, holding at least `batch` rows: the kept pair, or a new
    /// pair when this batch is wider than any before it.
    fn take_batch_io(&mut self, batch: usize) -> Result<BatchIo> {
        if let Some(io) = self.batch_io.take().filter(|io| io.rows >= batch) {
            return Ok(io);
        }
        let ctx = self.executor.context();
        let alloc = |len: usize| {
            GpuBuffer::<f32>::new(ctx, len).map_err(|e| gpu_err("qwen35_cuda_forward_batch", &e))
        };
        let hidden = batch * self.dims.hidden_dim as usize;
        let inter = batch * self.dims.intermediate_dim as usize;
        let v = batch * self.dims.v_dim as usize;
        let nv = batch * self.dims.num_v_heads as usize;
        let d = self.dims;
        let q_dim = batch * (d.num_heads * d.attn_head_dim) as usize;
        Ok(BatchIo {
            hidden: alloc(hidden)?,
            normed: alloc(hidden)?,
            dn_qkv: alloc(batch * self.dims.conv_dim as usize)?,
            dn_gate: alloc(v)?,
            dn_alpha: alloc(nv)?,
            dn_beta: alloc(nv)?,
            dn_core: alloc(v)?,
            at_q_full: alloc(2 * q_dim)?,
            at_k_raw: alloc(batch * (d.num_kv_heads * d.attn_head_dim) as usize)?,
            at_out_in: alloc(q_dim)?,
            ffn_gate: alloc(inter)?,
            ffn_up: alloc(inter)?,
            ffn_act: alloc(inter)?,
            ffn_down: alloc(hidden)?,
            logits: alloc(batch * self.dims.vocab_size as usize)?,
            rows: batch,
        })
    }

    /// Refuse a batch before any work is enqueued.
    fn check_batch(
        &self,
        tokens: &[u32],
        states: &[&mut Qwen35CudaState],
        positions: &[usize],
    ) -> Result<()> {
        let refuse = |reason: String| Err(RealizarError::InvalidShape { reason });
        if tokens.is_empty() {
            return refuse("qwen35_cuda_forward_batch: the batch is empty".to_string());
        }
        if states.len() != tokens.len() || positions.len() != tokens.len() {
            return refuse(format!(
                "qwen35_cuda_forward_batch: {} tokens, {} states and {} positions — one of each \
                 per sequence",
                tokens.len(),
                states.len(),
                positions.len()
            ));
        }
        let hidden_dim = self.dims.hidden_dim as usize;
        let vocab_rows = self.model.base.token_embedding().len() / hidden_dim;
        for (b, ((&token, state), &position)) in
            tokens.iter().zip(states).zip(positions).enumerate()
        {
            if token as usize >= vocab_rows {
                return refuse(format!(
                    "qwen35_cuda_forward_batch: sequence {b}: token {token} is outside the \
                     {vocab_rows}-row embedding table"
                ));
            }
            if position >= state.max_seq_len {
                return refuse(format!(
                    "qwen35_cuda_forward_batch: sequence {b}: position {position} is past the KV \
                     cache ({} rows)",
                    state.max_seq_len
                ));
            }
        }
        Ok(())
    }

    /// Enqueue the whole step, layer-major. In each layer every sequence runs its
    /// own token mixer (conv + recurrence, or attention over its own KV cache),
    /// then the SwiGLU FFN runs ONCE over all of the packed rows ([`Self::ffn_batched`]).
    /// Then every sequence's output norm goes into its `normed` row and ONE
    /// batched `lm_head` runs over all of them. Nothing here syncs.
    fn forward_batch_enqueued(
        &mut self,
        io: &BatchIo,
        batch: usize,
        states: &mut [&mut Qwen35CudaState],
        positions: &[usize],
    ) -> Result<()> {
        let d = self.dims;
        let rows: Vec<GpuBuffer<f32>> = (0..batch)
            .map(|b| Self::view(&io.hidden, b as u32 * d.hidden_dim, d.hidden_dim))
            .collect();
        let normed: Vec<GpuBuffer<f32>> = (0..batch)
            .map(|b| Self::view(&io.normed, b as u32 * d.hidden_dim, d.hidden_dim))
            .collect();

        let result = (|| {
            for il in 0..self.layers.len() {
                if matches!(self.layers[il], CudaLayer::DeltaNet(_)) {
                    self.deltanet_batched(io, &rows, &normed, states, il)
                        .map_err(|e| gpu_err("qwen35_cuda_deltanet", &e))?;
                } else {
                    self.attention_batched(io, &rows, &normed, states, positions, il)
                        .map_err(|e| gpu_err("qwen35_cuda_attention", &e))?;
                }
                self.ffn_batched(io, &rows, &normed, il)
                    .map_err(|e| gpu_err("qwen35_cuda_ffn", &e))?;
            }
            // The tail of `forward_single`: the same norm per sequence, then the
            // lm_head over the packed `[B][hidden_dim]` rows in one dispatch.
            for (row, out) in rows.iter().zip(&normed) {
                self.executor
                    .rmsnorm_into(row, &self.output_norm, out, d.hidden_dim, d.eps)
                    .map_err(|e| gpu_err("qwen35_cuda_lm_head", &e))?;
            }
            let (qtype, ptr, n, k) = (
                self.lm_head.qtype,
                self.lm_head.ptr,
                self.lm_head.n,
                self.lm_head.k,
            );
            self.gemv_batched(qtype, ptr, &io.normed, &io.logits, batch as u32, n, k)
                .map_err(|e| gpu_err("qwen35_cuda_lm_head", &e))
        })();

        // The views borrow `hidden` / `normed`; they must not free that memory.
        rows.into_iter().chain(normed).for_each(std::mem::forget);
        result
    }

    /// DeltaNet layer `il`'s mixer — the first half of `deltanet_layer_inner` — over
    /// every row of the step. The norm runs per row; the qkv, alpha, beta, gate and
    /// ssm_out projections are one batched GEMV each; the causal conv, the gates,
    /// the per-head L2 norms, the delta rule and the gated norm run per sequence,
    /// because they read or update that sequence's conv window and recurrent state.
    /// Each runs on its sequence's rows exactly as `deltanet_layer_inner` runs it,
    /// with the model's single-sequence scratch for the per-sequence intermediates.
    /// The first residual is elementwise, so it is one launch over the packed rows.
    fn deltanet_batched(
        &mut self,
        io: &BatchIo,
        rows: &[GpuBuffer<f32>],
        normed: &[GpuBuffer<f32>],
        states: &mut [&mut Qwen35CudaState],
        il: usize,
    ) -> std::result::Result<(), trueno_gpu::GpuError> {
        let d = self.dims;
        let m = rows.len() as u32;
        let CudaLayer::DeltaNet(w) = &self.layers[il] else {
            unreachable!("the caller matched a DeltaNet layer")
        };
        let proj = |w: &super::CudaQuantWeight| (w.qtype, w.ptr, w.n, w.k);
        let (qkv, alpha, beta, gate, out) = (
            proj(&w.attn_qkv),
            proj(&w.ssm_alpha),
            proj(&w.ssm_beta),
            proj(&w.attn_gate),
            proj(&w.ssm_out),
        );
        for (row, out) in rows.iter().zip(normed) {
            self.executor
                .rmsnorm_into(row, &w.attn_norm, out, d.hidden_dim, d.eps)?;
        }
        self.gemv_batched(qkv.0, qkv.1, &io.normed, &io.dn_qkv, m, qkv.2, qkv.3)?;
        self.gemv_batched(
            alpha.0,
            alpha.1,
            &io.normed,
            &io.dn_alpha,
            m,
            alpha.2,
            alpha.3,
        )?;
        self.gemv_batched(beta.0, beta.1, &io.normed, &io.dn_beta, m, beta.2, beta.3)?;
        self.gemv_batched(gate.0, gate.1, &io.normed, &io.dn_gate, m, gate.2, gate.3)?;

        let CudaLayer::DeltaNet(w) = &self.layers[il] else {
            unreachable!("the caller matched a DeltaNet layer")
        };
        let s = &self.scratch;
        let ex = &mut self.executor;
        for (b, state) in (0u32..).zip(states.iter()) {
            let qkv_row = Self::view(&io.dn_qkv, b * d.conv_dim, d.conv_dim);
            let alpha_row = Self::view(&io.dn_alpha, b * d.num_v_heads, d.num_v_heads);
            let beta_row = Self::view(&io.dn_beta, b * d.num_v_heads, d.num_v_heads);
            let gate_row = Self::view(&io.dn_gate, b * d.v_dim, d.v_dim);
            let core_row = Self::view(&io.dn_core, b * d.v_dim, d.v_dim);
            let q_view = Self::view(&s.conv_out, 0, d.k_dim);
            let k_view = Self::view(&s.conv_out, d.k_dim, d.k_dim);
            let v_view = Self::view(&s.conv_out, d.k_dim * 2, d.v_dim);
            let run = (|| {
                ex.gdn_causal_conv1d_silu_into(
                    &qkv_row,
                    &state.conv[il],
                    &w.conv1d_weight,
                    &s.conv_out,
                    d.conv_dim,
                    d.conv_kernel,
                )?;
                ex.gdn_per_head_l2_norm_into(&q_view, d.head_k_dim, d.num_k_heads, d.eps)?;
                ex.gdn_per_head_l2_norm_into(&k_view, d.head_k_dim, d.num_k_heads, d.eps)?;
                ex.gdn_gates_into(
                    &alpha_row,
                    &w.ssm_dt_bias,
                    &w.ssm_a,
                    &beta_row,
                    &s.dt,
                    &s.beta,
                    d.num_v_heads,
                )?;
                ex.gdn_delta_rule_into(
                    &q_view,
                    &k_view,
                    &v_view,
                    &s.beta,
                    &s.dt,
                    &state.ssm[il],
                    &s.out_h,
                    d.num_k_heads,
                    d.head_k_dim,
                    d.num_v_heads,
                    d.head_v_dim,
                )?;
                ex.gdn_gated_rmsnorm_into(
                    &s.out_h,
                    &gate_row,
                    &w.ssm_norm_weight,
                    &core_row,
                    d.head_v_dim,
                    d.num_v_heads,
                    d.eps,
                )
            })();
            // The views borrow the step's and the scratch buffers; they must not
            // free that memory.
            [
                qkv_row, alpha_row, beta_row, gate_row, core_row, q_view, k_view, v_view,
            ]
            .into_iter()
            .for_each(std::mem::forget);
            run?;
        }

        self.gemv_batched(out.0, out.1, &io.dn_core, &io.ffn_down, m, out.2, out.3)?;
        self.executor
            .residual_add_into(&io.hidden, &io.ffn_down, &io.hidden, m * d.hidden_dim)
    }

    /// Attention layer `il`'s mixer — the first half of `attention_layer_inner` —
    /// over every row of the step. The norm runs per row; the q and k projections
    /// and the output projection are one batched GEMV each. The v projection stays
    /// per sequence: it writes straight into that sequence's KV cache row, as
    /// `attention_layer_inner` does. The split, the q/k norms, RoPE, the decode
    /// attention and the output gate run per sequence, on its rows, exactly as the
    /// single-sequence body runs them. The first residual is one launch over the
    /// packed rows.
    fn attention_batched(
        &mut self,
        io: &BatchIo,
        rows: &[GpuBuffer<f32>],
        normed: &[GpuBuffer<f32>],
        states: &mut [&mut Qwen35CudaState],
        positions: &[usize],
        il: usize,
    ) -> std::result::Result<(), trueno_gpu::GpuError> {
        let d = self.dims;
        let m = rows.len() as u32;
        let q_dim = d.num_heads * d.attn_head_dim;
        let kv_dim = d.num_kv_heads * d.attn_head_dim;
        let CudaLayer::Attention(w) = &self.layers[il] else {
            unreachable!("the caller matched an attention layer")
        };
        let proj = |w: &super::CudaQuantWeight| (w.qtype, w.ptr, w.n, w.k);
        let (q, k, out) = (proj(&w.attn_q), proj(&w.attn_k), proj(&w.attn_output));
        for (row, out) in rows.iter().zip(normed) {
            self.executor
                .rmsnorm_into(row, &w.attn_norm, out, d.hidden_dim, d.eps)?;
        }
        self.gemv_batched(q.0, q.1, &io.normed, &io.at_q_full, m, q.2, q.3)?;
        self.gemv_batched(k.0, k.1, &io.normed, &io.at_k_raw, m, k.2, k.3)?;

        let CudaLayer::Attention(w) = &self.layers[il] else {
            unreachable!("the caller matched an attention layer")
        };
        let a = &self.attn_scratch;
        let ex = &mut self.executor;
        for (((b, state), normed_row), &position) in
            (0u32..).zip(states.iter_mut()).zip(normed).zip(positions)
        {
            let (k_cache, v_cache) = state.kv[il]
                .as_ref()
                .expect("an attention layer owns a KV cache");
            let pos32 = u32::try_from(position).unwrap_or(u32::MAX);
            let k_row = Self::view(
                k_cache,
                u32::try_from(position).unwrap_or(0) * kv_dim,
                kv_dim,
            );
            let v_row = Self::view(
                v_cache,
                u32::try_from(position).unwrap_or(0) * kv_dim,
                kv_dim,
            );
            let q_full_row = Self::view(&io.at_q_full, b * 2 * q_dim, 2 * q_dim);
            let k_raw_row = Self::view(&io.at_k_raw, b * kv_dim, kv_dim);
            let out_in_row = Self::view(&io.at_out_in, b * q_dim, q_dim);
            let run = (|| {
                ex.gemv_dispatch(
                    w.attn_v.qtype,
                    w.attn_v.ptr,
                    normed_row,
                    &v_row,
                    w.attn_v.n,
                    w.attn_v.k,
                )?;
                ex.gdn_split_interleaved_into(
                    &q_full_row,
                    &a.q,
                    &a.gate,
                    d.num_heads,
                    d.attn_head_dim,
                )?;
                ex.per_head_rmsnorm_into(
                    &a.q,
                    &w.attn_q_norm,
                    &a.q_normed,
                    d.attn_head_dim,
                    d.num_heads,
                    d.eps,
                )?;
                ex.per_head_rmsnorm_into(
                    &k_raw_row,
                    &w.attn_k_norm,
                    &k_row,
                    d.attn_head_dim,
                    d.num_kv_heads,
                    d.eps,
                )?;
                ex.gdn_partial_neox_rope_into(
                    &a.q_normed,
                    d.num_heads,
                    d.attn_head_dim,
                    d.n_rot,
                    pos32,
                    d.theta_scale,
                )?;
                ex.gdn_partial_neox_rope_into(
                    &k_row,
                    d.num_kv_heads,
                    d.attn_head_dim,
                    d.n_rot,
                    pos32,
                    d.theta_scale,
                )?;
                ex.gdn_decode_attention_into(
                    &a.q_normed,
                    k_cache,
                    v_cache,
                    &out_in_row,
                    d.num_heads,
                    d.num_kv_heads,
                    d.attn_head_dim,
                    pos32 + 1,
                )?;
                ex.gdn_sigmoid_gate_into(&out_in_row, &a.gate, q_dim)
            })();
            // The views borrow the cache, the step's and the scratch buffers; they
            // must not free that memory.
            [k_row, v_row, q_full_row, k_raw_row, out_in_row]
                .into_iter()
                .for_each(std::mem::forget);
            run?;
            // The same host-side bookkeeping as `attention_layer_inner`.
            state.kv_len = state.kv_len.max(position + 1);
        }

        self.gemv_batched(out.0, out.1, &io.at_out_in, &io.ffn_down, m, out.2, out.3)?;
        self.executor
            .residual_add_into(&io.hidden, &io.ffn_down, &io.hidden, m * d.hidden_dim)
    }

    /// Layer `il`'s FFN tail — `post_attention_norm`, SwiGLU, the second residual —
    /// over every row of the step. The norm runs per row (it reduces over one
    /// row); gate, up and down are one batched GEMV each; SwiGLU and the residual
    /// are elementwise, so one launch over the packed rows computes each element
    /// exactly as the per-row launch does.
    fn ffn_batched(
        &mut self,
        io: &BatchIo,
        rows: &[GpuBuffer<f32>],
        normed: &[GpuBuffer<f32>],
        il: usize,
    ) -> std::result::Result<(), trueno_gpu::GpuError> {
        let d = self.dims;
        let (norm, gate, up, down) = match &self.layers[il] {
            CudaLayer::DeltaNet(w) => (&w.post_attention_norm, &w.ffn_gate, &w.ffn_up, &w.ffn_down),
            CudaLayer::Attention(w) => {
                (&w.post_attention_norm, &w.ffn_gate, &w.ffn_up, &w.ffn_down)
            },
        };
        let proj = |w: &super::CudaQuantWeight| (w.qtype, w.ptr, w.n, w.k);
        let (gate, up, down) = (proj(gate), proj(up), proj(down));
        for (row, out) in rows.iter().zip(normed) {
            self.executor
                .rmsnorm_into(row, norm, out, d.hidden_dim, d.eps)?;
        }
        let m = rows.len() as u32;
        self.gemv_batched(gate.0, gate.1, &io.normed, &io.ffn_gate, m, gate.2, gate.3)?;
        self.gemv_batched(up.0, up.1, &io.normed, &io.ffn_up, m, up.2, up.3)?;
        self.executor.fused_swiglu_into(
            &io.ffn_gate,
            &io.ffn_up,
            &io.ffn_act,
            m * d.intermediate_dim,
        )?;
        self.gemv_batched(down.0, down.1, &io.ffn_act, &io.ffn_down, m, down.2, down.3)?;
        self.executor
            .residual_add_into(&io.hidden, &io.ffn_down, &io.hidden, m * d.hidden_dim)
    }

    /// `gemv_dispatch` for `m` packed input rows — `input` `[m][k]` into `output`
    /// `[m][n]` — with each output row bitwise what `gemv_dispatch` computes for its
    /// input row alone.
    ///
    /// A Q4_K matrix, or a Q6_K one with `k % 256 == 0`, whose single-vector
    /// dispatch takes the float multi-warp (`Mwv`) kernel — what this model pins —
    /// and every Q5_K matrix (its one kernel) take their batched twins: the weights
    /// are read once per launch, not once per row. Every other case runs
    /// `gemv_dispatch` row by row.
    #[allow(clippy::too_many_arguments)]
    fn gemv_batched(
        &mut self,
        qtype: WeightQuantType,
        ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> std::result::Result<(), trueno_gpu::GpuError> {
        use crate::cuda::gpu_profile::{Q4kVariant, Q6kVariant};
        let profile = &self.executor.gpu_profile;
        match qtype {
            WeightQuantType::Q4K if profile.q4k == Q4kVariant::Mwv => {
                return self
                    .executor
                    .batched_mwv_q4k_gemv_into(ptr, input, output, m, n, k);
            },
            WeightQuantType::Q5K => {
                return self
                    .executor
                    .batched_q5k_gemv_into(ptr, input, output, m, n, k);
            },
            WeightQuantType::Q6K if profile.q6k == Q6kVariant::Mwv && k.is_multiple_of(256) => {
                return self
                    .executor
                    .batched_mwv_q6k_gemv_into(ptr, input, output, m, n, k);
            },
            _ => {},
        }
        for r in 0..m {
            let x = Self::view(input, r * k, k);
            let y = Self::view(output, r * n, n);
            let run = self.executor.gemv_dispatch(qtype, ptr, &x, &y, n, k);
            std::mem::forget(x);
            std::mem::forget(y);
            run?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "forward_qwen35_cuda_batch_tests.rs"]
mod batch_tests;
