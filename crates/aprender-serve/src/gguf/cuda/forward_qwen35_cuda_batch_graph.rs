//! #5279: the batched decode step as one CUDA graph per batch shape.
//!
//! The eager batched step issues every layer's launches from the host, one
//! by one. Its arguments no longer change from step to step: the slots'
//! residual rows, their state table and their positions are uploaded into
//! fixed buffers before the step, and every kernel reads them there. What is
//! left is the shape: the slot count `m`, the attention partials' stride
//! (`max_splits`, a power of two, see
//! [`CudaExecutor::reserve_attention_slots`]) and the buffers it all lives
//! in. A graph captured at one shape replays at every step of that shape.
//!
//! Capture is the manual recorder (trueno#243), as for the single-slot graph
//! (#4233): the capture step runs eagerly with recording on, so its tokens
//! are real, and the graph is built from the recorded launches. The argmax
//! always runs inside the graph; the downloads stay outside it.
//!
//! On by default; `QWEN35_CUDA_BATCH_GRAPH=0` keeps the batched step eager.

use super::{gpu_err, CudaLayer, Qwen35BatchBufs, Qwen35CudaModel, Qwen35CudaState, Result};
use crate::cuda::CudaExecutor;
use std::collections::HashMap;
use trueno_gpu::driver::CudaGraphExec;

/// Graphs kept at once; past this many shapes the cache starts over.
const MAX_GRAPHS: usize = 32;

/// `m`, `max_splits`, the batch buffers, then every executor scratch buffer
/// ([`CudaExecutor::graph_scratch_ptrs`]): the launch arguments that differ
/// between two graphs, or that another path may move under one.
type GraphKey = Vec<u64>;

/// Where the scratch pointers start in a [`GraphKey`].
const SCRATCH_AT: usize = 3;

/// The batched step's captured graphs, one per [`GraphKey`].
pub(crate) struct Qwen35BatchGraphs {
    /// Route the batched step through a graph.
    enabled: bool,
    graphs: HashMap<GraphKey, CudaGraphExec>,
    /// Steps that replayed a captured graph.
    replays: u64,
    /// Steps that captured one.
    captures: u64,
}

impl Default for Qwen35BatchGraphs {
    fn default() -> Self {
        Self {
            enabled: std::env::var("QWEN35_CUDA_BATCH_GRAPH").map_or(true, |v| v != "0"),
            graphs: HashMap::new(),
            replays: 0,
            captures: 0,
        }
    }
}

impl Qwen35BatchGraphs {
    /// #5279: drop every captured graph (the batch buffers they read are
    /// about to be replaced).
    pub(super) fn clear(&mut self) {
        self.graphs.clear();
    }
}

impl Qwen35CudaModel<'_> {
    /// #5279: turn the captured batched step on or off. Turning it off drops
    /// every captured graph.
    pub fn set_batch_graph(&mut self, on: bool) {
        self.batch_graphs.enabled = on;
        if !on {
            self.batch_graphs.graphs.clear();
        }
    }

    /// `(replays, captures)` of the batched step's graphs: the evidence that
    /// a batched step actually ran through one.
    pub fn batch_graph_counts(&self) -> (u64, u64) {
        (self.batch_graphs.replays, self.batch_graphs.captures)
    }

    /// The step's [`GraphKey`], with the attention partials grown to fit
    /// first; `None` when the step cannot run as a graph (graphs off, or the
    /// attention runs per slot from host-computed pointers).
    fn batch_graph_key(
        &mut self,
        b: &Qwen35BatchBufs,
        positions: &[usize],
    ) -> std::result::Result<Option<GraphKey>, trueno_gpu::GpuError> {
        let d = self.dims;
        if !self.batch_graphs.enabled
            || !CudaExecutor::gdn_decode_attention_has_slots(d.attn_head_dim)
        {
            return Ok(None);
        }
        let m = positions.len() as u32;
        // Reserving the partials first makes the scratch list below hold the
        // pointer this step launches with.
        let (splits, _) = if self
            .layers
            .iter()
            .any(|l| matches!(l, CudaLayer::Attention(_)))
        {
            let max_pos = positions.iter().copied().max().unwrap_or(0);
            self.executor.reserve_attention_slots(
                (d.num_heads, d.num_kv_heads, d.attn_head_dim),
                u32::try_from(max_pos).unwrap_or(u32::MAX),
                m,
            )?
        } else {
            (0, 0)
        };
        let mut key = vec![u64::from(m), u64::from(splits), b.hidden.as_ptr()];
        key.extend(self.executor.graph_scratch_ptrs());
        Ok(Some(key))
    }

    /// The step up to its downloads through a captured graph: replay the
    /// graph of this step's shape, or capture it. `false` when the step
    /// cannot run as a graph, and nothing was launched.
    pub(super) fn batched_graphed(
        &mut self,
        b: &Qwen35BatchBufs,
        states: &[Qwen35CudaState],
        positions: &[usize],
    ) -> Result<bool> {
        let key = self
            .batch_graph_key(b, positions)
            .map_err(|e| gpu_err("qwen35_cuda_batch_graph", &e))?;
        let Some(key) = key else {
            return Ok(false);
        };
        if let Some(exec) = self.batch_graphs.graphs.get(&key) {
            self.executor
                .launch_graph_exec(exec)
                .map_err(|e| gpu_err("qwen35_cuda_batch_graph_replay", &e))?;
            self.batch_graphs.replays += 1;
            return Ok(true);
        }
        // Every step ends in a sync, so no graph dropped here is in flight.
        if self.batch_graphs.graphs.len() >= MAX_GRAPHS {
            self.batch_graphs.graphs.clear();
        }
        self.executor.forget_q8_activation();
        self.executor.begin_graph_recording();
        let body = self.batched_body(b, states, positions, true);
        // Always stop recording, even on a failed body, so no later launch is
        // silently appended to a dead recording.
        let built = self.executor.end_graph_recording();
        body.map_err(|e| gpu_err("qwen35_cuda_batch_graph_capture", &e))?;
        let kernels = built.map_err(|e| gpu_err("qwen35_cuda_batch_graph_build", &e))?;
        // The capture step already ran; with nothing recorded there is no
        // graph to keep, and the next step of this shape captures again.
        let exec = if kernels > 0 {
            self.executor.take_decode_graph()
        } else {
            None
        };
        // A scratch buffer that grew during the capture moved under the
        // launches recorded before it grew, so that graph would replay into
        // freed memory. Drop it; the next step of this shape captures again
        // on settled buffers.
        let settled = key[SCRATCH_AT..] == self.executor.graph_scratch_ptrs()[..];
        if let (Some(exec), true) = (exec, settled) {
            self.batch_graphs.graphs.insert(key, exec);
            self.batch_graphs.captures += 1;
        }
        Ok(true)
    }
}

#[cfg(test)]
#[path = "forward_qwen35_cuda_batch_graph_tests.rs"]
mod tests;
