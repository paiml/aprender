//! aprender#5279 P1 (#5289): one greedy decode step for m sequences at once.
//!
//! [`Qwen35CudaModel::forward_batch_greedy`] is the seam the serve batch
//! scheduler drives. Today it loops [`Qwen35CudaModel::forward_single_greedy`]
//! over the slots, so each slot's tokens are exactly the ones it would get
//! decoded alone. Later phases replace the loop with batched projections and
//! slot-dimensioned `DeltaNet`/attention kernels behind the same signature,
//! and are held bitwise to this one.

use super::{gpu_err, CudaLayer, Qwen35CudaDims, Qwen35CudaModel, Qwen35CudaState, Result};
use crate::cuda::CudaExecutor;
use crate::error::RealizarError;
use trueno_gpu::driver::GpuBuffer;
use trueno_gpu::GpuError;

/// Refuse a batch whose three per-slot inputs disagree in length, or that has
/// no slots, before any device work.
pub(crate) fn check_batch_shape(tokens: usize, states: usize, positions: usize) -> Result<()> {
    if tokens == 0 || tokens != states || tokens != positions {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "qwen35_cuda_batch: {tokens} tokens, {states} states and {positions} positions; \
                 the three must be equal and non-zero"
            ),
        });
    }
    Ok(())
}

/// One slot's result of a decode step.
#[derive(Debug, Clone, PartialEq)]
pub enum Qwen35StepOut {
    /// The token temperature 0 picks, chosen on the device.
    Token(u32),
    /// The slot's logits on the host, for a caller that samples, penalizes or
    /// records them; bitwise the ones [`Qwen35CudaModel::forward_single`]
    /// returns.
    Logits(Vec<f32>),
}

impl Qwen35CudaModel<'_> {
    /// The token temperature 0 picks for each of m slots: slot `i` runs
    /// `tokens[i]` at `positions[i]` against `states[i]`, which is updated in
    /// place exactly as [`Self::forward_single_greedy`] updates it.
    ///
    /// # Errors
    /// A shape mismatch (refused before any device work), or the first slot's
    /// [`Self::forward_single_greedy`] error. Slots before the failing one have
    /// already advanced.
    pub fn forward_batch_greedy(
        &mut self,
        tokens: &[u32],
        states: &mut [Qwen35CudaState],
        positions: &[usize],
    ) -> Result<Vec<u32>> {
        let host = vec![false; tokens.len()];
        let outs = self.forward_batch(tokens, states, positions, &host)?;
        Ok(outs
            .into_iter()
            .map(|out| match out {
                Qwen35StepOut::Token(t) => t,
                Qwen35StepOut::Logits(_) => unreachable!("no slot asked for its logits"),
            })
            .collect())
    }

    /// One decode step for m slots, as [`Self::forward_batch_greedy`]; slot
    /// `i` gets its logits on the host when `host_logits[i]`, and its greedy
    /// token otherwise. A slot that needs its logits steps with the others,
    /// so logprobs, a penalty or sampling do not take it out of the batch.
    ///
    /// # Errors
    /// As [`Self::forward_batch_greedy`]; `host_logits` must be as long as
    /// `tokens`.
    pub fn forward_batch(
        &mut self,
        tokens: &[u32],
        states: &mut [Qwen35CudaState],
        positions: &[usize],
        host_logits: &[bool],
    ) -> Result<Vec<Qwen35StepOut>> {
        check_batch_shape(tokens.len(), states.len(), positions.len())?;
        check_batch_shape(tokens.len(), host_logits.len(), positions.len())?;
        if tokens.len() > 1 && !self.timing_on {
            return self.batched_step(tokens, states, positions, host_logits);
        }
        tokens
            .iter()
            .zip(states.iter_mut())
            .zip(positions)
            .zip(host_logits)
            .map(|(((&token, state), &position), &host)| {
                Ok(if host {
                    Qwen35StepOut::Logits(self.forward_single(token, state, position)?)
                } else {
                    Qwen35StepOut::Token(self.forward_single_greedy(token, state, position)?)
                })
            })
            .collect()
    }
}

/// #5303: the m-row device buffers one batched decode step works in: the m
/// residual streams side by side, and an m-row twin of every buffer the FFN
/// and the `lm_head` write.
pub(crate) struct Qwen35BatchBufs {
    /// Rows every buffer holds.
    rows: usize,
    /// `[rows × hidden_dim]`, slot `i`'s residual stream at row `i`.
    hidden: GpuBuffer<f32>,
    /// `[rows × hidden_dim]`
    post_normed: GpuBuffer<f32>,
    /// `[rows × intermediate_dim]`
    ffn_gate: GpuBuffer<f32>,
    /// `[rows × intermediate_dim]`
    ffn_up: GpuBuffer<f32>,
    /// `[rows × intermediate_dim]`
    ffn_act: GpuBuffer<f32>,
    /// `[rows × hidden_dim]`
    ffn_down: GpuBuffer<f32>,
    /// `[rows × hidden_dim]`
    out_normed: GpuBuffer<f32>,
    /// `[rows × vocab_size]`
    logits: GpuBuffer<f32>,
    /// The m-row buffers the token mixers write.
    mixer: mixer::Qwen35MixerBufs,
    /// `[layers × 2 × m]` device pointers, uploaded at the start of each step:
    /// entry `(il * 2) * m + i` is slot `i`'s conv window at layer `il`, entry
    /// `(il * 2 + 1) * m + i` its recurrent state; at a full-attention layer the
    /// two are its K and V caches.
    state_table: GpuBuffer<u64>,
    /// `[m]` the slots' positions, uploaded with `state_table`.
    positions: GpuBuffer<u32>,
    /// `[m × hidden_dim]` on the host: the slots' embedding rows, uploaded
    /// to `hidden` in one copy.
    host_hidden: Vec<f32>,
    /// `[m × argmax_blocks]` first-pass argmax results, values and indices.
    argmax_vals: GpuBuffer<f32>,
    argmax_idxs: GpuBuffer<u32>,
    /// `[m]` the slots' greedy token ids, downloaded in one copy.
    argmax_out: GpuBuffer<u32>,
}

impl Qwen35BatchBufs {
    fn new(executor: &CudaExecutor, d: Qwen35CudaDims, rows: usize, layers: usize) -> Result<Self> {
        let hidden = rows * d.hidden_dim as usize;
        let inter = rows * d.intermediate_dim as usize;
        let zeros = |len| Qwen35CudaModel::zeros(executor, len);
        let blocks = rows * CudaExecutor::argmax_blocks(d.vocab_size) as usize;
        Ok(Self {
            rows,
            hidden: zeros(hidden)?,
            post_normed: zeros(hidden)?,
            ffn_gate: zeros(inter)?,
            ffn_up: zeros(inter)?,
            ffn_act: zeros(inter)?,
            ffn_down: zeros(hidden)?,
            out_normed: zeros(hidden)?,
            logits: zeros(rows * d.vocab_size as usize)?,
            mixer: mixer::Qwen35MixerBufs::new(executor, d, rows)?,
            state_table: GpuBuffer::new(executor.context(), 2 * layers * rows)
                .map_err(|e| gpu_err("qwen35_cuda_alloc", &e))?,
            positions: GpuBuffer::new(executor.context(), rows)
                .map_err(|e| gpu_err("qwen35_cuda_alloc", &e))?,
            host_hidden: vec![0.0; hidden],
            argmax_vals: zeros(blocks)?,
            argmax_idxs: GpuBuffer::new(executor.context(), blocks)
                .map_err(|e| gpu_err("qwen35_cuda_alloc", &e))?,
            argmax_out: GpuBuffer::new(executor.context(), rows)
                .map_err(|e| gpu_err("qwen35_cuda_alloc", &e))?,
        })
    }
}

/// Row `row` of an m-row buffer whose rows are `width` wide. The view must be
/// `std::mem::forget`-ed: it does not own the allocation.
fn row_view(buf: &GpuBuffer<f32>, row: usize, width: u32) -> GpuBuffer<f32> {
    Qwen35CudaModel::view(buf, row as u32 * width, width)
}

impl Qwen35CudaModel<'_> {
    /// One decode step for m > 1 slots, each slot's tokens bitwise the ones
    /// [`Self::forward_single_greedy`] gives it: every op is the single-row op
    /// on that slot's row, or a batched kernel whose row `r` is bitwise the
    /// single-row kernel on row `r`. Every projection, the FFN and the
    /// `lm_head` run once for all m rows, so their weights are read once per
    /// step instead of m times; the ops on a slot's own conv window,
    /// recurrent state and KV cache run per slot.
    fn batched_step(
        &mut self,
        tokens: &[u32],
        states: &mut [Qwen35CudaState],
        positions: &[usize],
        host_logits: &[bool],
    ) -> Result<Vec<Qwen35StepOut>> {
        let m = tokens.len();
        let mut b = match self.batch_bufs.take() {
            Some(b) if b.rows >= m => b,
            _ => {
                // #5279: a graph captured on the old buffers must never match the
                // new ones, even at a reused address.
                self.batch_graphs.clear();
                Qwen35BatchBufs::new(&self.executor, self.dims, m, self.layers.len())?
            },
        };
        let run = self.batched_step_in(&mut b, tokens, states, positions, host_logits);
        self.batch_bufs = Some(b);
        let next = run?;
        for (state, &position) in states.iter_mut().zip(positions) {
            state.kv_len = state.kv_len.max(position + 1);
        }
        Ok(next)
    }

    fn batched_step_in(
        &mut self,
        b: &mut Qwen35BatchBufs,
        tokens: &[u32],
        states: &mut [Qwen35CudaState],
        positions: &[usize],
        host_logits: &[bool],
    ) -> Result<Vec<Qwen35StepOut>> {
        let width = self.dims.hidden_dim as usize;
        let m = tokens.len();
        for (i, (&token, state)) in tokens.iter().zip(states.iter()).enumerate() {
            let row = self.embedding_row(token, state, positions[i])?;
            b.host_hidden[i * width..(i + 1) * width].copy_from_slice(&row);
            self.embed_row = row;
        }
        b.hidden
            .copy_from_host_at(&b.host_hidden[..m * width], 0)
            .map_err(|e| gpu_err("qwen35_cuda_batch", &e))?;
        // Every kernel that reads the table is queued after this blocking copy,
        // and the previous step ended in a sync, so nothing in flight reads it.
        let mut table = Vec::with_capacity(2 * self.layers.len() * tokens.len());
        for il in 0..self.layers.len() {
            // An attention layer's pair is its K and V caches instead.
            table.extend(states.iter().map(|s| match &s.kv[il] {
                Some((k, _)) => k.as_ptr(),
                None => s.conv[il].as_ptr(),
            }));
            table.extend(states.iter().map(|s| match &s.kv[il] {
                Some((_, v)) => v.as_ptr(),
                None => s.ssm[il].as_ptr(),
            }));
        }
        b.state_table
            .copy_from_host_at(&table, 0)
            .map_err(|e| gpu_err("qwen35_cuda_batch", &e))?;
        let pos32: Vec<u32> = positions
            .iter()
            .map(|&p| u32::try_from(p).unwrap_or(u32::MAX))
            .collect();
        b.positions
            .copy_from_host_at(&pos32, 0)
            .map_err(|e| gpu_err("qwen35_cuda_batch", &e))?;
        let argmax = if self.batched_graphed(b, states, positions)? {
            true
        } else {
            let argmax = host_logits.contains(&false);
            self.batched_body(b, states, positions, argmax)
                .map_err(|e| gpu_err("qwen35_cuda_batch", &e))?;
            argmax
        };
        self.batched_head_read(b, host_logits, argmax)
            .map_err(|e| gpu_err("qwen35_cuda_batch", &e))
    }

    /// Every layer, the output norm and the `lm_head` for every slot, then
    /// each row's argmax when `argmax`: the whole step up to its downloads.
    fn batched_body(
        &mut self,
        b: &Qwen35BatchBufs,
        states: &[Qwen35CudaState],
        positions: &[usize],
        argmax: bool,
    ) -> std::result::Result<(), GpuError> {
        for il in 0..self.layers.len() {
            self.batched_layer(b, il, states, positions)?;
        }
        self.batched_head_launch(b, states.len(), argmax)
    }

    /// Layer `il` for every slot: the mixer, then the FFN.
    fn batched_layer(
        &mut self,
        b: &Qwen35BatchBufs,
        il: usize,
        states: &[Qwen35CudaState],
        positions: &[usize],
    ) -> std::result::Result<(), GpuError> {
        if matches!(self.layers[il], CudaLayer::Attention(_)) {
            self.batched_attention_mixer(b, il, states, positions)?;
        } else {
            self.batched_deltanet_mixer(b, il, states)?;
        }
        self.batched_ffn(b, il, states.len())
    }

    /// [`Self::ffn_inner`] over m rows of `b.hidden`.
    fn batched_ffn(
        &mut self,
        b: &Qwen35BatchBufs,
        il: usize,
        m: usize,
    ) -> std::result::Result<(), GpuError> {
        let d = self.dims;
        let w = self.layers[il].ffn();
        let ex = &mut self.executor;
        let rows = m as u32;
        ex.rmsnorm_rows_into(&b.hidden, w.norm, &b.post_normed, d.hidden_dim, rows, d.eps)?;
        for (proj, out) in [(w.gate, &b.ffn_gate), (w.up, &b.ffn_up)] {
            ex.batched_gemv_dispatch(
                proj.qtype,
                proj.ptr,
                &b.post_normed,
                out,
                rows,
                proj.n,
                proj.k,
            )?;
        }
        ex.fused_swiglu_into(
            &b.ffn_gate,
            &b.ffn_up,
            &b.ffn_act,
            rows * d.intermediate_dim,
        )?;
        ex.batched_gemv_dispatch(
            w.down.qtype,
            w.down.ptr,
            &b.ffn_act,
            &b.ffn_down,
            rows,
            w.down.n,
            w.down.k,
        )?;
        ex.residual_add_into(&b.hidden, &b.ffn_down, &b.hidden, rows * d.hidden_dim)
    }

    /// The output norm per row, the `lm_head` once, then each row's argmax
    /// when `argmax`.
    fn batched_head_launch(
        &mut self,
        b: &Qwen35BatchBufs,
        m: usize,
        argmax: bool,
    ) -> std::result::Result<(), GpuError> {
        let d = self.dims;
        let ex = &mut self.executor;
        ex.rmsnorm_rows_into(
            &b.hidden,
            &self.output_norm,
            &b.out_normed,
            d.hidden_dim,
            m as u32,
            d.eps,
        )?;
        ex.batched_gemv_dispatch(
            self.lm_head.qtype,
            self.lm_head.ptr,
            &b.out_normed,
            &b.logits,
            m as u32,
            self.lm_head.n,
            self.lm_head.k,
        )?;
        if argmax {
            // Every row's greedy id in one pair of launches; the rows that
            // return logits ignore theirs.
            ex.argmax_rows_into(
                &b.logits,
                (&b.argmax_vals, &b.argmax_idxs),
                &b.argmax_out,
                d.vocab_size,
                m as u32,
            )?;
        }
        Ok(())
    }

    /// Each row's greedy token, or its logits on the host where `host_logits`
    /// asks for them; `argmax` says the step ran the argmax.
    fn batched_head_read(
        &mut self,
        b: &Qwen35BatchBufs,
        host_logits: &[bool],
        argmax: bool,
    ) -> std::result::Result<Vec<Qwen35StepOut>, GpuError> {
        let d = self.dims;
        let m = host_logits.len();
        // The one sync in front of the downloads, as `forward_single`.
        self.executor.sync_stream()?;
        let mut tokens = vec![0u32; m];
        if argmax && host_logits.contains(&false) {
            b.argmax_out.copy_to_host_at(&mut tokens, 0)?;
        }
        let vocab = d.vocab_size as usize;
        host_logits
            .iter()
            .zip(tokens)
            .enumerate()
            .map(|(i, (&host, token))| {
                if host {
                    let mut logits = vec![0.0f32; vocab];
                    b.logits.copy_to_host_at(&mut logits, i * vocab)?;
                    Ok(Qwen35StepOut::Logits(logits))
                } else {
                    Ok(Qwen35StepOut::Token(token))
                }
            })
            .collect()
    }
}
#[path = "forward_qwen35_cuda_batch_mixer.rs"]
mod mixer;

#[path = "forward_qwen35_cuda_batch_graph.rs"]
mod graph;
pub(crate) use graph::Qwen35BatchGraphs;

#[cfg(test)]
#[path = "forward_qwen35_cuda_batch_tests.rs"]
mod tests;
