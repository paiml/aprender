//! aprender#4233: the Qwen3.5 CUDA decode step as ONE CUDA graph.
//!
//! The eager step enqueues ~20 launches per layer, so on a 24-layer model the
//! host spends most of a token issuing kernels. This replays the whole step —
//! every layer, the output norm and the `lm_head` GEMV — as one `cuGraphLaunch`.
//!
//! A graph freezes every kernel argument, so everything that changes per token
//! is moved behind a FIXED device address the host rewrites before each replay:
//!
//! | per-token value | eager step | graph step |
//! |---|---|---|
//! | embedding row | a fresh `GpuBuffer::from_host` each token | uploaded into `io.hidden` |
//! | position (RoPE, attention `seq_len`) | a `u32` kernel argument | read from `io.pos` on the device |
//! | this token's K/V cache row | a host-computed row view | fixed `io.k_row`/`io.v_row`, scattered at `*io.pos` |
//!
//! The recurrent state (conv windows, `DeltaNet` state, KV caches) is updated
//! in place at fixed addresses, so it needs nothing — but it belongs to ONE
//! [`Qwen35CudaState`]. A graph is therefore keyed by the device address of every
//! state buffer and re-captured when a different state arrives.
//!
//! Capture is the manual recorder (trueno#243): the capture token runs eagerly
//! with recording on, so it produces real logits, and the graph is built from
//! the recorded launches. Every executor op this step reaches records itself
//! (audited for #4233: the pinned MWV GEMVs, rmsnorm, per-head rmsnorm,
//! residual add, fused SwiGLU, and all GDN ops); `CPU_RMSNORM=1` must not be set.
//!
//! Opt-in with `QWEN35_CUDA_GRAPH=1` until the token-identity and tok/s
//! evidence on #4233 is in.

use super::{gpu_err, CudaLayer, GpuBuffer, Qwen35CudaModel, Qwen35CudaState, Result};
use trueno_gpu::driver::CudaGraphExec;

/// `QWEN35_CUDA_GRAPH=1` routes `forward_single` through the graph.
pub(super) fn graph_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("QWEN35_CUDA_GRAPH").is_ok_and(|v| v == "1"))
}

/// The fixed device addresses a captured step reads its per-token inputs from.
pub(super) struct Qwen35GraphIo {
    /// `[hidden_dim]`, the residual stream; the embedding row is uploaded here.
    pub(super) hidden: GpuBuffer<f32>,
    /// `[1]`, this token's position.
    pub(super) pos: GpuBuffer<u32>,
    /// `[num_kv_heads * attn_head_dim]`, this token's K row before the append.
    pub(super) k_row: GpuBuffer<f32>,
    /// `[num_kv_heads * attn_head_dim]`, this token's V row before the append.
    pub(super) v_row: GpuBuffer<f32>,
}

/// A captured decode step and the state it was captured against.
pub(super) struct Qwen35DecodeGraph {
    io: Qwen35GraphIo,
    exec: Option<CudaGraphExec>,
    /// The device address of every buffer of the state the graph was captured
    /// against; a different key means a different state and a re-capture.
    key: Vec<u64>,
    kernels: usize,
    replays: u64,
}

impl Qwen35CudaModel<'_> {
    /// The device address of every buffer a decode step mutates in `state`.
    fn graph_state_key(state: &Qwen35CudaState) -> Vec<u64> {
        let mut key: Vec<u64> = state.conv.iter().map(GpuBuffer::as_ptr).collect();
        key.extend(state.ssm.iter().map(GpuBuffer::as_ptr));
        for (k, v) in state.kv.iter().flatten() {
            key.push(k.as_ptr());
            key.push(v.as_ptr());
        }
        key
    }

    fn new_decode_graph(&self) -> Result<Qwen35DecodeGraph> {
        let d = self.dims;
        let kv_dim = (d.num_kv_heads * d.attn_head_dim) as usize;
        let pos = GpuBuffer::from_host(self.executor.context(), &[0u32])
            .map_err(|e| gpu_err("qwen35_cuda_graph_alloc", &e))?;
        Ok(Qwen35DecodeGraph {
            io: Qwen35GraphIo {
                hidden: Self::zeros(&self.executor, d.hidden_dim as usize)?,
                pos,
                k_row: Self::zeros(&self.executor, kv_dim)?,
                v_row: Self::zeros(&self.executor, kv_dim)?,
            },
            exec: None,
            key: Vec::new(),
            kernels: 0,
            replays: 0,
        })
    }

    /// [`Self::forward_single`] through the captured graph: capture on the first
    /// token of a state, replay on every token after it.
    pub(super) fn forward_single_graphed(
        &mut self,
        token: u32,
        state: &mut Qwen35CudaState,
        position: usize,
    ) -> Result<Vec<f32>> {
        let row = self.embedding_row(token, state, position)?;
        let mut graph = match self.decode_graph.take() {
            Some(g) => g,
            None => self.new_decode_graph()?,
        };
        let result = self.graph_step(&mut graph, row, state, position);
        self.decode_graph = Some(graph);
        let logits = result?;
        state.kv_len = state.kv_len.max(position + 1);
        Ok(logits)
    }

    fn graph_step(
        &mut self,
        graph: &mut Qwen35DecodeGraph,
        row: &[f32],
        state: &mut Qwen35CudaState,
        position: usize,
    ) -> Result<Vec<f32>> {
        let pos = [u32::try_from(position).unwrap_or(u32::MAX)];
        let key = Self::graph_state_key(state);
        self.executor
            .upload_on_stream(&mut graph.io.hidden, row)
            .and_then(|()| self.executor.upload_on_stream(&mut graph.io.pos, &pos))
            .map_err(|e| gpu_err("qwen35_cuda_graph_upload", &e))?;

        match graph.exec.as_ref().filter(|_| graph.key == key) {
            Some(exec) => {
                self.executor
                    .launch_graph_exec(exec)
                    .map_err(|e| gpu_err("qwen35_cuda_graph_replay", &e))?;
                if graph.replays == 0 {
                    eprintln!(
                        "[aprender#4233] qwen35 decode graph: replay launched ({} kernels, position {position})",
                        graph.kernels
                    );
                }
                graph.replays += 1;
            },
            None => self.graph_capture(graph, key, state, position)?,
        }

        // The ONE sync of the whole token, in front of the ONE download. It also
        // retires the async uploads, whose host slices live until here.
        self.executor
            .sync_stream()
            .map_err(|e| gpu_err("qwen35_cuda_forward", &e))?;
        let mut logits = vec![0.0f32; self.dims.vocab_size as usize];
        self.logits_buf
            .copy_to_host(&mut logits)
            .map_err(|e| gpu_err("qwen35_cuda_forward", &e))?;
        Ok(logits)
    }

    /// Run the step eagerly with recording on, then build the graph from the
    /// recorded launches. The capture token's logits are real.
    fn graph_capture(
        &mut self,
        graph: &mut Qwen35DecodeGraph,
        key: Vec<u64>,
        state: &mut Qwen35CudaState,
        position: usize,
    ) -> Result<()> {
        graph.exec = None;
        self.executor.begin_graph_recording();
        let body = self.graph_body(&graph.io, state, position);
        // Always stop recording, even on a failed body, so no later launch is
        // silently appended to a dead recording.
        let built = self.executor.end_graph_recording();
        body.map_err(|e| gpu_err("qwen35_cuda_graph_capture", &e))?;
        graph.kernels = built.map_err(|e| gpu_err("qwen35_cuda_graph_build", &e))?;
        graph.exec = self.executor.take_decode_graph();
        graph.key = key;
        graph.replays = 0;
        eprintln!(
            "[aprender#4233] qwen35 decode graph: captured {} kernels at position {position}",
            graph.kernels
        );
        Ok(())
    }

    /// Every layer, then the `lm_head` tail, reading per-token inputs from `io`.
    fn graph_body(
        &mut self,
        io: &Qwen35GraphIo,
        state: &mut Qwen35CudaState,
        position: usize,
    ) -> std::result::Result<(), trueno_gpu::GpuError> {
        for il in 0..self.layers.len() {
            match self.layers[il] {
                CudaLayer::DeltaNet(_) => self.deltanet_layer_inner(state, il, &io.hidden)?,
                CudaLayer::Attention(_) => {
                    self.attention_layer_inner(state, il, &io.hidden, position, Some(io))?;
                },
            }
        }
        self.lm_head_tail(&io.hidden)
    }
}
