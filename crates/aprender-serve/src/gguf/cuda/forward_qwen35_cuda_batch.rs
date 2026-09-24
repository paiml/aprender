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

/// `forward_batch`'s device input (`[rows][hidden_dim]`), normed output
/// (`[rows][hidden_dim]`) and logits (`[rows][vocab_size]`), reused across steps.
/// A step of `B <= rows` uses the first `B` rows of each.
pub(super) struct BatchIo {
    hidden: GpuBuffer<f32>,
    normed: GpuBuffer<f32>,
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
        Ok(BatchIo {
            hidden: alloc(batch * self.dims.hidden_dim as usize)?,
            normed: alloc(batch * self.dims.hidden_dim as usize)?,
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

    /// Enqueue the whole step — every layer layer-major, then every sequence's
    /// output norm into its `normed` row and ONE batched `lm_head` over all of them.
    /// Nothing here syncs.
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
                for ((state, row), &position) in states.iter_mut().zip(&rows).zip(positions) {
                    match self.layers[il] {
                        CudaLayer::DeltaNet(_) => self.deltanet_layer(state, il, row)?,
                        CudaLayer::Attention(_) => {
                            self.attention_layer(state, il, row, position)?;
                        },
                    }
                }
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

    /// `gemv_dispatch` for `m` packed input rows — `input` `[m][k]` into `output`
    /// `[m][n]` — with each output row bitwise what `gemv_dispatch` computes for its
    /// input row alone.
    ///
    /// A Q4_K matrix, or a Q6_K one with `k % 256 == 0`, whose single-vector
    /// dispatch takes the float multi-warp (`Mwv`) kernel — what this model pins —
    /// takes its batched twin: the weights are read once per launch, not once per
    /// row. Every other case runs `gemv_dispatch` row by row.
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
