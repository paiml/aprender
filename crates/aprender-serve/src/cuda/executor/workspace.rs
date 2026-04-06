//! Transformer workspace and GEMV buffer pool management
//!
//! This module implements:
//! - PAR-044: Transformer Workspace (zero-allocation forward pass)
//! - PAR-111: Batched workspace for multi-sequence processing
//! - PAR-007: GEMV Buffer Pool (avoid per-call allocation)

use super::*;

impl CudaExecutor {
    // ========================================================================
    // PAR-044: Transformer Workspace (zero-allocation forward pass)
    // ========================================================================

    /// Initialize workspace buffers for zero-allocation forward pass
    ///
    /// MUST be called after `build_indexed_weights()` and before first forward pass.
    /// Allocates all intermediate buffers once; they are reused for every token.
    ///
    /// # Arguments
    ///
    /// * `hidden_dim` - Model hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    ///
    /// # Errors
    ///
    /// Returns error if GPU allocation fails.
    pub fn init_workspace(
        &mut self,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> Result<(), GpuError> {
        // PAR-200: Skip reallocation if workspace already initialized with matching dims.
        // Prefill workspace (buffer_capacity >= 1) is always large enough for decode (batch_size=1).
        // Reusing prefill buffers preserves GPU pointers, enabling CUDA graph reuse across requests.
        if self.workspace.initialized
            && self.workspace.buffer_capacity >= 1
            && self.workspace.hidden_dim == hidden_dim
            && self.workspace.intermediate_dim == intermediate_dim
        {
            // PMAT-058: Reset batch_size to 1 for M=1 decode path.
            // After generate_batched_streaming sets batch_size=4, the early return
            // preserves batch_size=4 which is wrong for single-request decode.
            self.workspace.batch_size = 1;
            return Ok(());
        }

        let q_dim = self.kv_num_heads * self.kv_head_dim;
        let kv_dim = self.kv_num_kv_heads * self.kv_head_dim;

        // GH-215 FIX: Pad workspace buffers to Q4K super-block boundary (256 elements).
        // Q4K GEMV kernels read activations at sb_idx*256+val_idx, which can exceed the
        // logical dimension for non-256-aligned models (e.g., hidden_dim=896 reads up to 1024).
        // Without padding, this causes out-of-bounds GPU memory reads → garbage output.
        let pad256 = |dim: usize| ((dim + 255) / 256) * 256;

        // Allocate all workspace buffers (10 buffers total for zero-allocation forward)
        // PAR-051: Added attn_out_buf to eliminate 28 allocations per token
        self.workspace.hidden_buf1 = Some(GpuBuffer::new(&self.context, pad256(hidden_dim))?);
        self.workspace.hidden_buf2 = Some(GpuBuffer::new(&self.context, pad256(hidden_dim))?);
        self.workspace.input_staging = Some(GpuBuffer::new(&self.context, pad256(hidden_dim))?);
        self.workspace.q_buf = Some(GpuBuffer::new(&self.context, pad256(q_dim))?);
        self.workspace.k_buf = Some(GpuBuffer::new(&self.context, pad256(kv_dim))?);
        self.workspace.v_buf = Some(GpuBuffer::new(&self.context, pad256(kv_dim))?);
        self.workspace.attn_out_buf = Some(GpuBuffer::new(&self.context, pad256(q_dim))?); // PAR-051
        self.workspace.ffn_gate_buf =
            Some(GpuBuffer::new(&self.context, pad256(intermediate_dim))?);
        self.workspace.ffn_up_buf = Some(GpuBuffer::new(&self.context, pad256(intermediate_dim))?);
        self.workspace.ffn_act_buf = Some(GpuBuffer::new(&self.context, pad256(intermediate_dim))?);

        // PAR-114/realizr#219: positions_buf for batched RoPE (needed even at M=1)
        self.workspace.positions_buf = Some(GpuBuffer::new(&self.context, 1)?);
        // PAR-111: normed_hidden_buf for output norm before LM head
        self.workspace.normed_hidden_buf = Some(GpuBuffer::new(&self.context, pad256(hidden_dim))?);

        // PAR-PERF-DP4A: Pre-allocate Q8_1 activation buffer for DP4A GEMV
        let max_input_dim = hidden_dim.max(intermediate_dim).max(q_dim);
        let q8_num_blocks = (max_input_dim + 31) / 32;
        let q8_bytes = q8_num_blocks * 36;
        self.workspace.q8_activation_buf = Some(GpuBuffer::new(&self.context, q8_bytes)?);

        self.workspace.hidden_dim = hidden_dim;
        self.workspace.q_dim = q_dim;
        self.workspace.kv_dim = kv_dim;
        self.workspace.intermediate_dim = intermediate_dim;
        self.workspace.batch_size = 1;
        self.workspace.buffer_capacity = 1;
        self.workspace.initialized = true;

        Ok(())
    }

    /// PAR-111: Initialize batched workspace for multi-sequence processing
    ///
    /// Allocates M× larger buffers for processing batch_size sequences in parallel.
    /// Used with batched GEMV kernels to achieve 16x speedup over sequential.
    ///
    /// # Arguments
    ///
    /// * `hidden_dim` - Model hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    /// * `batch_size` - Number of sequences to process in parallel (typically 4)
    ///
    /// # Errors
    ///
    /// Returns error if GPU allocation fails.
    pub fn init_batched_workspace(
        &mut self,
        hidden_dim: usize,
        intermediate_dim: usize,
        batch_size: usize,
    ) -> Result<(), GpuError> {
        // PAR-129: Extended to M=32 via 4-warp kernel
        if batch_size == 0 || batch_size > 32 {
            return Err(GpuError::InvalidParameter(format!(
                "PAR-111: batch_size must be 1-32, got {}",
                batch_size
            )));
        }

        // PMAT-045: Skip reallocation if workspace buffers are already large enough.
        // Preserves GPU buffer addresses → CUDA graph stays valid → no recapture.
        // Uses buffer_capacity (high-water mark) instead of batch_size to prevent
        // thrashing: prefill sets batch_size=30, decode sets batch_size=4, next
        // request's prefill would see 4<30 and reallocate. buffer_capacity stays
        // at the high-water mark (30), so the skip fires correctly.
        if self.workspace.initialized
            && self.workspace.buffer_capacity >= batch_size
            && self.workspace.hidden_dim == hidden_dim
            && self.workspace.intermediate_dim == intermediate_dim
        {
            // Update logical batch size for decode kernels (they check == m)
            self.workspace.batch_size = batch_size;

            // GH-141: Ensure Q8 buffer is sized for batched DP4A (M vectors).
            // init_prefill_workspace doesn't allocate Q8 buffer, so on first
            // decode after prefill the Q8 buffer may still be M=1 sized.
            let q_dim = self.kv_num_heads * self.kv_head_dim;
            let max_input_dim = hidden_dim.max(intermediate_dim).max(q_dim);
            let q8_num_blocks = (max_input_dim + 31) / 32;
            let q8_bytes_needed = q8_num_blocks * 36 * batch_size;
            let q8_current = self
                .workspace
                .q8_activation_buf
                .as_ref()
                .map_or(0, |b| b.len());
            if q8_current < q8_bytes_needed {
                self.workspace.q8_activation_buf =
                    Some(GpuBuffer::new(&self.context, q8_bytes_needed)?);
            }

            return Ok(());
        }

        let q_dim = self.kv_num_heads * self.kv_head_dim;
        let kv_dim = self.kv_num_kv_heads * self.kv_head_dim;

        // PAR-111: Allocate M× larger buffers for batched processing
        let m = batch_size;
        self.workspace.hidden_buf1 = Some(GpuBuffer::new(&self.context, hidden_dim * m)?);
        self.workspace.hidden_buf2 = Some(GpuBuffer::new(&self.context, hidden_dim * m)?);
        self.workspace.input_staging = Some(GpuBuffer::new(&self.context, hidden_dim * m)?);
        self.workspace.q_buf = Some(GpuBuffer::new(&self.context, q_dim * m)?);
        self.workspace.k_buf = Some(GpuBuffer::new(&self.context, kv_dim * m)?);
        self.workspace.v_buf = Some(GpuBuffer::new(&self.context, kv_dim * m)?);
        self.workspace.attn_out_buf = Some(GpuBuffer::new(&self.context, q_dim * m)?);
        self.workspace.ffn_gate_buf = Some(GpuBuffer::new(&self.context, intermediate_dim * m)?);
        self.workspace.ffn_up_buf = Some(GpuBuffer::new(&self.context, intermediate_dim * m)?);
        self.workspace.ffn_act_buf = Some(GpuBuffer::new(&self.context, intermediate_dim * m)?);
        // PAR-111: normed_hidden_buf for output norm before LM head
        self.workspace.normed_hidden_buf = Some(GpuBuffer::new(&self.context, hidden_dim * m)?);
        // PAR-114: positions buffer for batched RoPE
        self.workspace.positions_buf = Some(GpuBuffer::new(&self.context, m)?);

        // GH-141: Scale Q8_1 activation buffer for batched DP4A GEMV
        // Batched kernel needs M vectors × (K/32 blocks × 36 bytes)
        let max_input_dim = hidden_dim.max(intermediate_dim).max(q_dim);
        let q8_num_blocks = (max_input_dim + 31) / 32;
        let q8_bytes = q8_num_blocks * 36 * m;
        self.workspace.q8_activation_buf = Some(GpuBuffer::new(&self.context, q8_bytes)?);

        self.workspace.hidden_dim = hidden_dim;
        self.workspace.q_dim = q_dim;
        self.workspace.kv_dim = kv_dim;
        self.workspace.intermediate_dim = intermediate_dim;
        self.workspace.batch_size = batch_size;
        self.workspace.buffer_capacity = batch_size;
        self.workspace.initialized = true;

        // PMAT-044 FIX: Buffer reallocation invalidates any captured CUDA graph
        // (graph holds hardcoded GPU pointers to old buffers). Clear graph so next
        // decode re-captures with the new buffer addresses. Poka-yoke at source.
        self.decode_graph = None;
        self.decode_token_count = 0;
        self.graph_input_buf = None;
        self.position_buf = None;
        self.seq_len_buf = None;
        self.graph_capture_failed = false;
        // PMAT-045: Also clear batched decode graphs (stale pointers)
        self.batched_decode_graphs.clear();
        self.batched_graph_batch_size = 0;

        eprintln!(
            "[PAR-111] Initialized batched workspace: batch_size={}, hidden={}×{}, q={}×{}, kv={}×{}, ffn={}×{}",
            batch_size,
            hidden_dim, m,
            q_dim, m,
            kv_dim, m,
            intermediate_dim, m
        );

        Ok(())
    }

    /// Check if workspace is initialized
    #[must_use]
    pub fn has_workspace(&self) -> bool {
        self.workspace.initialized
    }

    /// PAR-111: Get the batch size of the current workspace
    #[must_use]
    pub fn workspace_batch_size(&self) -> usize {
        self.workspace.batch_size
    }

    /// PAR-062: Check if CUDA decode graph has been captured
    ///
    /// Returns true if the decode graph is ready for replay.
    /// The graph is captured on first forward pass with `forward_all_layers_gpu_to_logits_graphed`.
    #[must_use]
    pub fn has_decode_graph(&self) -> bool {
        self.decode_graph.is_some()
    }

    /// PMAT-283: Initialize the decode completion event for CPU-GPU pipelining.
    ///
    /// Call once after graph capture succeeds. The event is recorded after each
    /// graph launch and can be queried non-blockingly via `decode_event_complete()`.
    pub fn init_decode_event(&mut self) -> Result<(), GpuError> {
        if self.decode_event.is_none() {
            self.decode_event = Some(CudaEvent::new()?);
        }
        Ok(())
    }

    /// PMAT-283: Query whether the last decode step has completed (non-blocking).
    ///
    /// Returns `true` if the GPU has finished the last graph replay,
    /// `false` if still in progress. Returns `true` if no event exists.
    #[must_use]
    pub fn decode_event_complete(&self) -> bool {
        match &self.decode_event {
            Some(event) => event.is_complete().unwrap_or(true),
            None => true,
        }
    }

    /// Clear workspace buffers (releases GPU memory)
    pub fn clear_workspace(&mut self) {
        self.workspace = TransformerWorkspace::default();
    }

    /// CORRECTNESS-015: Force next init_workspace to fully reallocate.
    ///
    /// After batched prefill, workspace buffers are M×-sized (from init_prefill_workspace).
    /// The PAR-200 early-return in init_workspace keeps these oversized buffers, but some
    /// GPU state in the reused buffers corrupts subsequent CUDA graph captures.
    /// Resetting buffer_capacity forces init_workspace to reallocate fresh M=1 buffers.
    pub fn force_workspace_reinit(&mut self) {
        self.workspace.buffer_capacity = 0;
        self.workspace.initialized = false;
    }

    /// Clear decode graph and related state
    ///
    /// Call this before starting a new generation session to ensure
    /// the graph is recaptured with fresh state.
    pub fn clear_decode_graph(&mut self) {
        self.decode_graph = None;
        self.decode_token_count = 0;
        self.graph_input_buf = None;
        self.position_buf = None;
        self.seq_len_buf = None;
        // PAR-118: Reset capture failure flag so next generation can attempt capture
        self.graph_capture_failed = false;
    }

    /// PMAT-045: Clear batched decode graphs (stale after workspace reallocation)
    pub fn clear_batched_decode_graphs(&mut self) {
        self.batched_decode_graphs.clear();
        self.batched_graph_batch_size = 0;
    }

    /// GH-219: Validate PTX parity for all batched kernels at startup
    ///
    /// Uses the executor's current model dimensions (from KV cache init) to
    /// construct all 6 batched kernels and validate structural parity with
    /// their single-vector references.
    ///
    /// Returns the parity report. Callers decide whether to warn or error.
    /// Toyota Way: Poka-Yoke — catch PTX generation bugs at init, not runtime.
    pub fn validate_kernel_parity(
        &self,
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
    ) -> crate::ptx_parity::PtxParityReport {
        let dims = crate::ptx_parity::KernelDimensions {
            hidden_dim,
            intermediate_dim,
            num_heads: self.kv_num_heads as u32,
            head_dim: self.kv_head_dim as u32,
            rope_theta: self.rope_theta,
            epsilon,
        };
        crate::ptx_parity::validate_all_kernel_pairs(&dims)
    }

    // ========================================================================
    // PAR-007: GEMV Buffer Pool (avoid per-call allocation)
    // ========================================================================

    /// ALB-110: Ensure GEMV input buffer has at least required_size capacity.
    ///
    /// Grow-only: allocates once at the high-water mark, reuses for all
    /// smaller dimensions. Eliminates ~356K cuMemAlloc/cuMemFree per request
    /// that fragment the CUDA allocator and crash after ~65 sustained completions.
    pub(crate) fn ensure_gemv_input_buffer(
        &mut self,
        required_size: usize,
    ) -> Result<u64, GpuError> {
        if self.gemv_input_size < required_size {
            self.gemv_input_buffer = Some(GpuBuffer::new(&self.context, required_size)?);
            self.gemv_input_size = required_size;
        }
        Ok(self
            .gemv_input_buffer
            .as_ref()
            .expect("buffer just created")
            .as_ptr())
    }

    /// ALB-110: Ensure GEMV output buffer has at least required_size capacity.
    /// Grow-only semantics (see ensure_gemv_input_buffer).
    pub(crate) fn ensure_gemv_output_buffer(
        &mut self,
        required_size: usize,
    ) -> Result<u64, GpuError> {
        if self.gemv_output_size < required_size {
            self.gemv_output_buffer = Some(GpuBuffer::new(&self.context, required_size)?);
            self.gemv_output_size = required_size;
        }
        Ok(self
            .gemv_output_buffer
            .as_ref()
            .expect("buffer just created")
            .as_ptr())
    }

    /// Copy input data to cached GEMV input buffer
    pub(crate) fn copy_to_gemv_input(&mut self, input: &[f32]) -> Result<(), GpuError> {
        let buf = self
            .gemv_input_buffer
            .as_mut()
            .expect("buffer should exist");
        buf.copy_from_host(input)
    }

    /// Copy output data from cached GEMV output buffer
    pub(crate) fn copy_from_gemv_output(&self, output: &mut [f32]) -> Result<(), GpuError> {
        let buf = self
            .gemv_output_buffer
            .as_ref()
            .expect("buffer should exist");
        buf.copy_to_host(output)
    }

    /// ALB-111: Ensure GEMV output buffer B has at least required_size capacity.
    /// Grow-only semantics (see ensure_gemv_input_buffer).
    pub(crate) fn ensure_gemv_output_buffer_b(
        &mut self,
        required_size: usize,
    ) -> Result<u64, GpuError> {
        if self.gemv_output_size_b < required_size {
            self.gemv_output_buffer_b = Some(GpuBuffer::new(&self.context, required_size)?);
            self.gemv_output_size_b = required_size;
        }
        Ok(self
            .gemv_output_buffer_b
            .as_ref()
            .expect("buffer just created")
            .as_ptr())
    }

    /// ALB-111: Ensure GEMV output buffer C has at least required_size capacity.
    /// Grow-only semantics (see ensure_gemv_input_buffer).
    pub(crate) fn ensure_gemv_output_buffer_c(
        &mut self,
        required_size: usize,
    ) -> Result<u64, GpuError> {
        if self.gemv_output_size_c < required_size {
            self.gemv_output_buffer_c = Some(GpuBuffer::new(&self.context, required_size)?);
            self.gemv_output_size_c = required_size;
        }
        Ok(self
            .gemv_output_buffer_c
            .as_ref()
            .expect("buffer just created")
            .as_ptr())
    }

    /// Get GEMV buffer pool statistics
    #[must_use]
    pub fn gemv_buffer_stats(&self) -> (usize, usize) {
        // ALB-111: Include extra output buffers in total output bytes
        let output_bytes =
            (self.gemv_output_size + self.gemv_output_size_b + self.gemv_output_size_c) * 4;
        (self.gemv_input_size * 4, output_bytes)
    }

    /// Clear GEMV buffers (releases GPU memory)
    pub fn clear_gemv_buffers(&mut self) {
        self.gemv_input_buffer = None;
        self.gemv_output_buffer = None;
        self.gemv_input_size = 0;
        self.gemv_output_size = 0;
        // ALB-111: Clear extra output buffers
        self.gemv_output_buffer_b = None;
        self.gemv_output_buffer_c = None;
        self.gemv_output_size_b = 0;
        self.gemv_output_size_c = 0;
    }
}

include!("workspace_tests.rs");
