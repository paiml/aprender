//! Batched prefill for prompt processing
//!
//! PMAT-PREFILL: Process all S prompt tokens through all transformer layers
//! in a single pass, replacing the serial token-by-token prefill loop.
//!
//! Expected improvement: Prefill 510ms → ~50ms (10x) for 20-token prompts.
//!
//! Uses the existing batched GEMV infrastructure (`transformer_layer_batched`)
//! which already handles M tokens at different positions. For prefill, M=S.

#![allow(clippy::wildcard_imports)]

use super::super::*;

impl CudaExecutor {
    /// PMAT-PREFILL: Initialize workspace for batched prefill
    ///
    /// Allocates workspace buffers sized for `max_seq_len` tokens.
    /// This is separate from the decode workspace because prefill
    /// processes many more tokens simultaneously.
    ///
    /// # Arguments
    ///
    /// * `max_seq_len` - Maximum prompt length to support
    /// * `hidden_dim` - Model hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    ///
    /// # Errors
    ///
    /// Returns error if GPU allocation fails.
    pub fn init_prefill_workspace(
        &mut self,
        max_seq_len: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> Result<(), GpuError> {
        if max_seq_len == 0 {
            return Err(GpuError::InvalidParameter(
                "PMAT-PREFILL: max_seq_len must be > 0".to_string(),
            ));
        }

        // PAR-200: Skip reallocation if workspace already large enough.
        // Eliminates GPU malloc churn when batched prefill runs per-request.
        // PMAT-045: Check buffer_capacity (high-water mark) instead of batch_size
        // to prevent thrashing between prefill and decode phases.
        if self.workspace.initialized
            && self.workspace.buffer_capacity >= max_seq_len
            && self.workspace.hidden_dim == hidden_dim
            && self.workspace.intermediate_dim == intermediate_dim
        {
            return Ok(());
        }

        // CORRECTNESS-014: Workspace reallocation invalidates decode graph.
        // Graph captures pointers to workspace buffers (hidden_buf1, q_buf, etc.).
        // When init_prefill_workspace allocates NEW buffers (longer prompt exceeds
        // buffer_capacity), the captured graph has stale pointers → ILLEGAL_ADDRESS.
        // Fix: clear ALL graphs before reallocating workspace.
        self.decode_graph = None;
        self.graph_capture_failed = false;
        // PMAT-075: Also clear batched decode graphs — stale workspace pointers.
        // Previously only M=1 decode graph was cleared here, leaving batched graphs
        // with stale addresses if a recycled slot had a longer prompt.
        self.batched_decode_graphs.clear();
        self.batched_graph_batch_size = 0;

        let q_dim = self.kv_num_heads * self.kv_head_dim;
        let kv_dim = self.kv_num_kv_heads * self.kv_head_dim;
        let m = max_seq_len;

        // Allocate M× sized buffers for prefill
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
        self.workspace.normed_hidden_buf = Some(GpuBuffer::new(&self.context, hidden_dim * m)?);
        self.workspace.positions_buf = Some(GpuBuffer::new(&self.context, m)?);
        // Logits only for last token (vocab_size allocated in decode workspace)
        // We don't need logits_buf for prefill since we only cache KV

        self.workspace.hidden_dim = hidden_dim;
        self.workspace.q_dim = q_dim;
        self.workspace.kv_dim = kv_dim;
        self.workspace.intermediate_dim = intermediate_dim;
        self.workspace.batch_size = m;
        self.workspace.buffer_capacity = m;
        self.workspace.initialized = true;

        Ok(())
    }

    /// PMAT-PREFILL: Process all prompt tokens through all layers in one pass
    ///
    /// Replaces the serial prefill loop that processes tokens one at a time.
    /// Uses batched GEMV kernels to process S tokens simultaneously through
    /// each transformer layer.
    ///
    /// PMAT-050: On first call for a given S, captures a CUDA graph. Subsequent
    /// calls with same S replay the graph (728 kernel launches → 1 graph launch).
    ///
    /// After prefill completes:
    /// - KV cache is populated with entries at positions 0..S-1
    /// - No logits are returned (prefill only caches KV, doesn't predict)
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn prefill_all_layers_gpu(
        &mut self,
        embeddings: &[f32],
        positions: &[u32],
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        contract_pre_prefill_incremental!();
        let s = positions.len();
        if s == 0 {
            return Ok(());
        }

        let expected_input_len = s * hidden_dim as usize;
        if embeddings.len() != expected_input_len {
            return Err(GpuError::InvalidParameter(format!(
                "PMAT-PREFILL: embeddings.len() {} != S*hidden_dim = {}",
                embeddings.len(),
                expected_input_len
            )));
        }

        // Verify workspace initialized for this batch size
        // PMAT-045: Check buffer_capacity (actual allocation) not batch_size (logical)
        if !self.workspace.initialized || self.workspace.buffer_capacity < s {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PMAT-PREFILL: Workspace not initialized for S={} (have buffer_capacity={})",
                s, self.workspace.buffer_capacity
            )));
        }

        // PMAT-059: Prefill graph DISABLED by default — cuBLAS uses workspace-free
        // algorithms during graph capture, which are 7x slower than eager cuBLAS
        // (541ms vs 78ms for S=125 on RTX 4060L). Enable with PREFILL_GRAPH=1.
        let graph_enabled = std::env::var("PREFILL_GRAPH").as_deref() == Ok("1")
            && std::env::var("CUBLAS_PREFILL").as_deref() != Ok("0");

        if graph_enabled {
            // PMAT-059: Always try replay first — graph_capture_failed must NOT
            // block replay of already-captured graphs. Prior bug: capture failure
            // for S=30 blocked replay of S=125 graph → 561ms TTFT regression.
            if self.prefill_graphs.contains_key(&s) {
                return self.prefill_graphed_replay(embeddings, s, num_layers, hidden_dim);
            }
            // Only attempt new capture if no previous prefill capture failure
            if !self.prefill_graph_capture_failed {
                match self.try_prefill_graph_capture(
                    s,
                    num_layers,
                    hidden_dim,
                    intermediate_dim,
                    epsilon,
                ) {
                    Ok(()) => {
                        return self.prefill_graphed_replay(embeddings, s, num_layers, hidden_dim);
                    },
                    Err(e) => {
                        eprintln!(
                            "[PREFILL-GRAPH] Capture failed for S={}: {}. Falling back to eager.",
                            s, e
                        );
                        // PMAT-059: Only set prefill-specific flag, NOT shared graph_capture_failed.
                        // Shared flag would prevent decode graph capture → decode regression.
                        self.prefill_graph_capture_failed = true;
                        // Reset KV cache lengths (capture may have modified them)
                        for layer_idx in 0..num_layers {
                            self.kv_cache_lengths.insert(layer_idx, 0);
                        }
                        // Fall through to eager path
                    },
                }
            }
        }

        // Eager path (fallback or when graphs disabled)
        self.prefill_eager(
            embeddings,
            positions,
            num_layers,
            hidden_dim,
            intermediate_dim,
            epsilon,
        )
    }

    /// Eager (non-graphed) prefill — original implementation
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn prefill_eager(
        &mut self,
        embeddings: &[f32],
        positions: &[u32],
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        let s = positions.len();

        // 1. Upload all S embeddings to GPU
        // CORRECTNESS-013 FIX: Use copy_from_host_async on self.stream instead of
        // GpuBuffer::from_host (which uses cuMemcpyHtoD on legacy default stream 0).
        // Five-Whys root cause: CU_STREAM_NON_BLOCKING streams have NO ordering
        // guarantee with stream 0. The RMSNorm kernel on self.stream could read
        // stale/zero data at later positions because the stream 0 H2D copy has no
        // happens-before relationship with the non-blocking stream kernel launch.
        // Fix: copy_from_host_async submits the DMA on self.stream, ensuring
        // same-stream ordering with all subsequent kernel launches.
        let mut input_buf = GpuBuffer::new(&self.context, embeddings.len())?;
        // SAFETY: embeddings slice is valid for the duration of prefill_eager.
        // cuMemcpyHtoDAsync with pageable memory stages data synchronously, then
        // submits DMA on self.stream. The slice can be safely read afterward.
        unsafe {
            input_buf.copy_from_host_async(embeddings, &self.stream)?;
        }

        // PMAT-049: Hoist workspace extraction out of layer loop.
        self.validate_batched_workspace(s as u32, positions.len())?;
        let q_dim = (self.kv_num_heads * self.kv_head_dim) as u32;
        let kv_dim = (self.kv_num_kv_heads * self.kv_head_dim) as u32;

        // Extract raw (ptr, len) pairs ONCE from workspace
        let hidden_buf1_ptr = self
            .workspace
            .hidden_buf1
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PMAT-049: hidden_buf1 missing".to_string())
            })?
            .as_ptr();
        let hidden_buf1_len = self.workspace.hidden_buf1.as_ref().unwrap().len();
        let hidden_buf2_ptr = self
            .workspace
            .hidden_buf2
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PMAT-049: hidden_buf2 missing".to_string())
            })?
            .as_ptr();
        let hidden_buf2_len = self.workspace.hidden_buf2.as_ref().unwrap().len();
        let input_staging_ptr = self
            .workspace
            .input_staging
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PMAT-049: input_staging missing".to_string())
            })?
            .as_ptr();
        let input_staging_len = self.workspace.input_staging.as_ref().unwrap().len();
        let q_buf_ptr = self
            .workspace
            .q_buf
            .as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PMAT-049: q_buf missing".to_string()))?
            .as_ptr();
        let q_buf_len = self.workspace.q_buf.as_ref().unwrap().len();
        let k_buf_ptr = self
            .workspace
            .k_buf
            .as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PMAT-049: k_buf missing".to_string()))?
            .as_ptr();
        let k_buf_len = self.workspace.k_buf.as_ref().unwrap().len();
        let v_buf_ptr = self
            .workspace
            .v_buf
            .as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PMAT-049: v_buf missing".to_string()))?
            .as_ptr();
        let v_buf_len = self.workspace.v_buf.as_ref().unwrap().len();
        let ffn_gate_ptr = self
            .workspace
            .ffn_gate_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PMAT-049: ffn_gate_buf missing".to_string())
            })?
            .as_ptr();
        let ffn_gate_len = self.workspace.ffn_gate_buf.as_ref().unwrap().len();
        let ffn_up_ptr = self
            .workspace
            .ffn_up_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PMAT-049: ffn_up_buf missing".to_string())
            })?
            .as_ptr();
        let ffn_up_len = self.workspace.ffn_up_buf.as_ref().unwrap().len();
        let ffn_act_ptr = self
            .workspace
            .ffn_act_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PMAT-049: ffn_act_buf missing".to_string())
            })?
            .as_ptr();
        let ffn_act_len = self.workspace.ffn_act_buf.as_ref().unwrap().len();
        let attn_out_ptr = self
            .workspace
            .attn_out_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PMAT-049: attn_out_buf missing".to_string())
            })?
            .as_ptr();
        let attn_out_len = self.workspace.attn_out_buf.as_ref().unwrap().len();

        // Create non-owning GpuBuffer wrappers ONCE (not 28× per layer)
        // SAFETY: All pointers valid from workspace allocation, lengths verified at init
        let hidden_buf1 =
            unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_buf1_ptr, hidden_buf1_len) };
        let hidden_buf2 =
            unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_buf2_ptr, hidden_buf2_len) };
        let input_staging =
            unsafe { GpuBuffer::<f32>::from_raw_parts(input_staging_ptr, input_staging_len) };
        let q_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(q_buf_ptr, q_buf_len) };
        let k_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(k_buf_ptr, k_buf_len) };
        let v_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(v_buf_ptr, v_buf_len) };
        let ffn_gate_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ffn_gate_ptr, ffn_gate_len) };
        let ffn_up_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ffn_up_ptr, ffn_up_len) };
        let ffn_act_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ffn_act_ptr, ffn_act_len) };
        let attn_out_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(attn_out_ptr, attn_out_len) };

        // PMAT-032: Initialize cuBLAS for parallel prefill attention
        if std::env::var("CUBLAS_PREFILL").as_deref() != Ok("0") {
            self.ensure_cublas()?;
        }

        // GH-94: Suppress flash decoding during prefill (small seq_lens cause errors)
        self.is_prefilling = true;

        let prefill_trace = std::env::var("PREFILL_TRACE").is_ok();

        // 2. Process all layers — call phase functions directly (bypass transformer_layer_batched)
        for layer_idx in 0..num_layers {
            if layer_idx >= self.indexed_layer_weights.len() {
                return Err(GpuError::InvalidLaunchConfig(format!(
                    "PMAT-PREFILL: Layer {} weights not indexed (have {})",
                    layer_idx,
                    self.indexed_layer_weights.len()
                )));
            }
            let layer_weights = self.get_indexed_layer(layer_idx).clone();

            let layer_input = if layer_idx == 0 {
                &input_buf
            } else {
                &hidden_buf2
            };

            let layer_start = if prefill_trace {
                Some(std::time::Instant::now())
            } else {
                None
            };

            // Phase 1: RMSNorm + QKV projections + bias + RoPE
            self.batched_qkv_rope_phase(
                layer_input,
                &hidden_buf1,
                &q_buf,
                &k_buf,
                &v_buf,
                q_buf_ptr,
                k_buf_ptr,
                v_buf_ptr,
                hidden_buf1_ptr,
                layer_idx,
                &layer_weights,
                s as u32,
                positions,
                hidden_dim,
                q_dim,
                kv_dim,
                epsilon,
            )?;

            let phase1_done = if prefill_trace && layer_idx == 0 {
                self.stream.synchronize()?;
                Some(std::time::Instant::now())
            } else {
                None
            };

            // Phase 2: Attention + output projection + residuals + FFN
            self.batched_attn_ffn_phase(
                layer_input,
                &hidden_buf1,
                &hidden_buf2,
                &input_staging,
                &q_buf,
                &k_buf,
                &v_buf,
                &attn_out_buf,
                &ffn_gate_buf,
                &ffn_up_buf,
                &ffn_act_buf,
                q_buf_ptr,
                k_buf_ptr,
                v_buf_ptr,
                attn_out_ptr,
                hidden_buf1_ptr,
                ffn_gate_ptr,
                ffn_up_ptr,
                ffn_act_ptr,
                layer_idx,
                &layer_weights,
                s as u32,
                positions,
                hidden_dim,
                intermediate_dim,
                q_dim,
                kv_dim,
                epsilon,
            )?;

            if let Some(t) = layer_start {
                if layer_idx == 0 {
                    self.stream.synchronize()?;
                    let total_ms = t.elapsed().as_secs_f64() * 1000.0;
                    let phase1_ms =
                        phase1_done.map_or(0.0, |p| p.duration_since(t).as_secs_f64() * 1000.0);
                    eprintln!(
                        "[PREFILL-TRACE] Layer 0 (synced): {:.2}ms (QKV+RoPE={:.2}ms, Attn+FFN={:.2}ms, S={})",
                        total_ms, phase1_ms, total_ms - phase1_ms, s
                    );
                }
            }
        }

        self.is_prefilling = false;

        // After all layers, output is in hidden_buf2 [S × hidden_dim]
        // KV cache has been populated by batched_attn_ffn_phase for each layer.

        let sync_start = if prefill_trace {
            Some(std::time::Instant::now())
        } else {
            None
        };
        self.stream.synchronize()?;
        if let Some(t) = sync_start {
            eprintln!(
                "[PREFILL-TRACE] Final sync: {:.2}ms",
                t.elapsed().as_secs_f64() * 1000.0
            );
        }

        // PMAT-049: Forget all wrappers ONCE (not 28× per layer)
        std::mem::forget(hidden_buf1);
        std::mem::forget(hidden_buf2);
        std::mem::forget(input_staging);
        std::mem::forget(q_buf);
        std::mem::forget(k_buf);
        std::mem::forget(v_buf);
        std::mem::forget(attn_out_buf);
        std::mem::forget(ffn_gate_buf);
        std::mem::forget(ffn_up_buf);
        std::mem::forget(ffn_act_buf);

        Ok(())
    }

    /// PMAT-051: Multi-prompt batched prefill — read weights once for all prompts.
    ///
    /// Five-Whys: c=4 TTFT = 256ms (10.7x vs llama.cpp 24ms).
    /// Why? 4 sequential prefills × 59ms = 236ms.
    /// Why? Each prefill reads all weights (3 GB FP16) independently.
    /// Why? prefill_and_scatter loops over prompts, calling run_prefill per slot.
    /// Fix: Concatenate all prompts' tokens into M_total, run GEMM once per layer.
    /// Only attention is per-prompt (requires prompt-local causal mask).
    ///
    /// Expected: 4×59ms = 236ms → ~65ms (~3.5x TTFT improvement).
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    /// PMAT-088d: `slot_indices` maps prompt_idx → batched KV cache slot.
    /// When None, uses prompt_idx (0,1,2,...) — existing behavior.
    /// When Some, uses slot_indices[prompt_idx] — enables recycling arbitrary slots.
    pub fn prefill_multi_prompt(
        &mut self,
        embeddings: &[f32],
        positions: &[u32],
        prompt_offsets: &[usize],
        prompt_lengths: &[usize],
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
        slot_indices: Option<&[usize]>,
    ) -> Result<(), GpuError> {
        let m_total = positions.len();
        let num_prompts = prompt_offsets.len();
        if m_total == 0 || num_prompts == 0 {
            return Ok(());
        }

        let expected_input_len = m_total * hidden_dim as usize;
        if embeddings.len() != expected_input_len {
            return Err(GpuError::InvalidParameter(format!(
                "PMAT-051: embeddings.len() {} != M_total*hidden_dim = {}",
                embeddings.len(),
                expected_input_len
            )));
        }

        // Verify workspace initialized for M_total
        if !self.workspace.initialized || self.workspace.buffer_capacity < m_total {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PMAT-051: Workspace not initialized for M_total={} (have buffer_capacity={})",
                m_total, self.workspace.buffer_capacity
            )));
        }

        let q_dim = (self.kv_num_heads * self.kv_head_dim) as u32;
        let kv_dim = (self.kv_num_kv_heads * self.kv_head_dim) as u32;

        // 1. Upload all embeddings to GPU
        // CORRECTNESS-013 FIX: Same as prefill_eager — use async H2D on self.stream
        // to ensure same-stream ordering with subsequent kernel launches.
        let mut input_buf = GpuBuffer::new(&self.context, embeddings.len())?;
        // SAFETY: embeddings valid for duration of prefill_multi_prompt
        unsafe {
            input_buf.copy_from_host_async(embeddings, &self.stream)?;
        }

        // Extract workspace buffer pointers ONCE
        let hidden_buf1_ptr = self
            .workspace
            .hidden_buf1
            .as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PMAT-051: hidden_buf1 missing".into()))?
            .as_ptr();
        let hidden_buf1_len = self.workspace.hidden_buf1.as_ref().unwrap().len();
        let hidden_buf2_ptr = self
            .workspace
            .hidden_buf2
            .as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PMAT-051: hidden_buf2 missing".into()))?
            .as_ptr();
        let hidden_buf2_len = self.workspace.hidden_buf2.as_ref().unwrap().len();
        let input_staging_ptr = self
            .workspace
            .input_staging
            .as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PMAT-051: input_staging missing".into()))?
            .as_ptr();
        let input_staging_len = self.workspace.input_staging.as_ref().unwrap().len();
        let q_buf_ptr = self
            .workspace
            .q_buf
            .as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PMAT-051: q_buf missing".into()))?
            .as_ptr();
        let q_buf_len = self.workspace.q_buf.as_ref().unwrap().len();
        let k_buf_ptr = self
            .workspace
            .k_buf
            .as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PMAT-051: k_buf missing".into()))?
            .as_ptr();
        let k_buf_len = self.workspace.k_buf.as_ref().unwrap().len();
        let v_buf_ptr = self
            .workspace
            .v_buf
            .as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PMAT-051: v_buf missing".into()))?
            .as_ptr();
        let v_buf_len = self.workspace.v_buf.as_ref().unwrap().len();
        let ffn_gate_ptr = self
            .workspace
            .ffn_gate_buf
            .as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PMAT-051: ffn_gate_buf missing".into()))?
            .as_ptr();
        let ffn_gate_len = self.workspace.ffn_gate_buf.as_ref().unwrap().len();
        let ffn_up_ptr = self
            .workspace
            .ffn_up_buf
            .as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PMAT-051: ffn_up_buf missing".into()))?
            .as_ptr();
        let ffn_up_len = self.workspace.ffn_up_buf.as_ref().unwrap().len();
        let ffn_act_ptr = self
            .workspace
            .ffn_act_buf
            .as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PMAT-051: ffn_act_buf missing".into()))?
            .as_ptr();
        let ffn_act_len = self.workspace.ffn_act_buf.as_ref().unwrap().len();
        let attn_out_ptr = self
            .workspace
            .attn_out_buf
            .as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PMAT-051: attn_out_buf missing".into()))?
            .as_ptr();
        let attn_out_len = self.workspace.attn_out_buf.as_ref().unwrap().len();

        // SAFETY: All pointers valid from workspace allocation
        let hidden_buf1 =
            unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_buf1_ptr, hidden_buf1_len) };
        let hidden_buf2 =
            unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_buf2_ptr, hidden_buf2_len) };
        let input_staging =
            unsafe { GpuBuffer::<f32>::from_raw_parts(input_staging_ptr, input_staging_len) };
        let q_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(q_buf_ptr, q_buf_len) };
        let k_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(k_buf_ptr, k_buf_len) };
        let v_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(v_buf_ptr, v_buf_len) };
        let ffn_gate_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ffn_gate_ptr, ffn_gate_len) };
        let ffn_up_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ffn_up_ptr, ffn_up_len) };
        let ffn_act_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ffn_act_ptr, ffn_act_len) };
        let attn_out_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(attn_out_ptr, attn_out_len) };

        // Initialize cuBLAS for HGEMM
        self.ensure_cublas()?;
        self.is_prefilling = true;

        let trace = std::env::var("PMAT051_TRACE").is_ok();
        let mut t_qkv_total = 0u128;
        let mut t_attn_total = 0u128;
        let mut t_ffn_total = 0u128;

        for layer_idx in 0..num_layers {
            if layer_idx >= self.indexed_layer_weights.len() {
                self.is_prefilling = false;
                return Err(GpuError::InvalidLaunchConfig(format!(
                    "PMAT-051: Layer {} weights not indexed (have {})",
                    layer_idx,
                    self.indexed_layer_weights.len()
                )));
            }
            let layer_weights = self.get_indexed_layer(layer_idx).clone();
            let layer_input = if layer_idx == 0 {
                &input_buf
            } else {
                &hidden_buf2
            };

            let t0 = if trace {
                self.stream.synchronize()?;
                Some(std::time::Instant::now())
            } else {
                None
            };

            // Phase 1: RMSNorm + QKV + bias + RoPE on ALL M_total tokens (weight read ONCE)
            self.batched_qkv_rope_phase(
                layer_input,
                &hidden_buf1,
                &q_buf,
                &k_buf,
                &v_buf,
                q_buf_ptr,
                k_buf_ptr,
                v_buf_ptr,
                hidden_buf1_ptr,
                layer_idx,
                &layer_weights,
                m_total as u32,
                positions,
                hidden_dim,
                q_dim,
                kv_dim,
                epsilon,
            )?;

            if let Some(t) = t0 {
                self.stream.synchronize()?;
                t_qkv_total += t.elapsed().as_nanos();
            }
            let t1 = if trace {
                Some(std::time::Instant::now())
            } else {
                None
            };

            // Phase 2a: Per-prompt attention from packed buffer + direct scatter to batched KV
            // PMAT-051 v2: Eliminates 56,000 D2D copies (bulk_scatter_kv) + 448 D2D copies
            // (scatter_single_kv_to_batched_layer) by:
            // 1. Running cuBLAS attention directly from packed K/V buffer (lda=kv_dim)
            // 2. Scattering K/V to batched cache via PTX kernel (1 launch per prompt)
            for prompt_idx in 0..num_prompts {
                let offset = prompt_offsets[prompt_idx];
                let s_i = prompt_lengths[prompt_idx];
                if s_i == 0 {
                    continue;
                }

                // Compute offset pointers into packed QKV/attn_out buffers
                let q_off = (offset * q_dim as usize * std::mem::size_of::<f32>()) as u64;
                let kv_off = (offset * kv_dim as usize * std::mem::size_of::<f32>()) as u64;
                let attn_off = (offset * q_dim as usize * std::mem::size_of::<f32>()) as u64;

                // Attention from packed buffer + scatter to batched KV cache in one method.
                // No intermediate single KV cache. Zero D2D copies for attention,
                // single kernel launch for scatter.
                // PMAT-088d: Map prompt_idx to target batched KV slot
                let target_slot = slot_indices.map_or(prompt_idx, |s| s[prompt_idx]);
                self.prefill_attention_from_packed(
                    layer_idx,
                    q_buf_ptr + q_off,
                    k_buf_ptr + kv_off,
                    v_buf_ptr + kv_off,
                    attn_out_ptr + attn_off,
                    s_i as u32,
                    q_dim,
                    kv_dim,
                    target_slot,
                )?;
            }

            if let Some(t) = t1 {
                self.stream.synchronize()?;
                t_attn_total += t.elapsed().as_nanos();
            }
            let t2 = if trace {
                Some(std::time::Instant::now())
            } else {
                None
            };

            // Phase 2b: Output projection + residuals + FFN on ALL M_total tokens (weight read ONCE)

            // 5. Output Projection
            self.batched_gemv_or_gemm(
                layer_weights.attn_output_qtype,
                layer_weights.attn_output_ptr,
                &attn_out_buf,
                &hidden_buf1,
                attn_out_ptr,
                hidden_buf1_ptr,
                m_total as u32,
                hidden_dim,
                q_dim,
            )?;

            // 6. First Residual
            self.batched_residual_add_into(
                layer_input,
                &hidden_buf1,
                &input_staging,
                hidden_dim,
                m_total as u32,
            )?;

            // 7. Pre-FFN RMSNorm
            self.batched_rmsnorm_ptr_into(
                &input_staging,
                layer_weights.ffn_norm_ptr,
                layer_weights.ffn_norm_len,
                &hidden_buf1,
                hidden_dim,
                m_total as u32,
                epsilon,
            )?;

            // 8. FFN Gate/Up (no fused DP4A during prefill — is_prefilling=true skips it)
            self.batched_gemv_or_gemm(
                layer_weights.ffn_gate_qtype,
                layer_weights.ffn_gate_ptr,
                &hidden_buf1,
                &ffn_gate_buf,
                hidden_buf1_ptr,
                ffn_gate_ptr,
                m_total as u32,
                intermediate_dim,
                hidden_dim,
            )?;
            self.batched_gemv_or_gemm(
                layer_weights.ffn_up_qtype,
                layer_weights.ffn_up_ptr,
                &hidden_buf1,
                &ffn_up_buf,
                hidden_buf1_ptr,
                ffn_up_ptr,
                m_total as u32,
                intermediate_dim,
                hidden_dim,
            )?;

            // 9. SwiGLU
            self.batched_swiglu_into(
                &ffn_gate_buf,
                &ffn_up_buf,
                &ffn_act_buf,
                intermediate_dim,
                m_total as u32,
            )?;

            // 10. FFN Down
            self.batched_gemv_or_gemm(
                layer_weights.ffn_down_qtype,
                layer_weights.ffn_down_ptr,
                &ffn_act_buf,
                &hidden_buf1,
                ffn_act_ptr,
                hidden_buf1_ptr,
                m_total as u32,
                hidden_dim,
                intermediate_dim,
            )?;

            // 11. Second Residual
            self.batched_residual_add_into(
                &input_staging,
                &hidden_buf1,
                &hidden_buf2,
                hidden_dim,
                m_total as u32,
            )?;

            if let Some(t) = t2 {
                self.stream.synchronize()?;
                t_ffn_total += t.elapsed().as_nanos();
            }
        }

        self.is_prefilling = false;
        self.stream.synchronize()?;

        if trace {
            eprintln!(
                "[PMAT-051-TRACE] QKV={:.1}ms, Attn+Scatter={:.1}ms, FFN={:.1}ms, total={:.1}ms",
                t_qkv_total as f64 / 1e6,
                t_attn_total as f64 / 1e6,
                t_ffn_total as f64 / 1e6,
                (t_qkv_total + t_attn_total + t_ffn_total) as f64 / 1e6,
            );
        }

        // Forget all non-owning wrappers
        std::mem::forget(hidden_buf1);
        std::mem::forget(hidden_buf2);
        std::mem::forget(input_staging);
        std::mem::forget(q_buf);
        std::mem::forget(k_buf);
        std::mem::forget(v_buf);
        std::mem::forget(attn_out_buf);
        std::mem::forget(ffn_gate_buf);
        std::mem::forget(ffn_up_buf);
        std::mem::forget(ffn_act_buf);

        Ok(())
    }

    /// PMAT-050: Warmup all resources needed for prefill graph capture.
    ///
    /// Must be called BEFORE begin_capture() to avoid memory allocations
    /// and PTX compilation inside the capture region (which cause error 901).
    #[allow(clippy::too_many_arguments)]
    fn warmup_prefill_for_capture(
        &mut self,
        s: usize,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        let q_dim = (self.kv_num_heads * self.kv_head_dim) as u32;
        let vocab_size = 0; // Not needed for prefill (no LM head)

        // 1. cuBLAS handle
        self.ensure_cublas()?;

        // 2. FP16 weight cache (all layers, all 7 matrices per layer)
        self.warmup_hgemm_cache(num_layers, hidden_dim, intermediate_dim, vocab_size)?;

        // 3. Allocate stable input buffer for graph capture
        let input_size = s * hidden_dim as usize;
        if self
            .prefill_graph_input_buf
            .as_ref()
            .map_or(true, |b| b.len() < input_size)
        {
            self.prefill_graph_input_buf = Some(GpuBuffer::new(&self.context, input_size)?);
        }

        // 4. Run one eager prefill to warm up ALL lazy state:
        // - PTX module compilation (f32_to_f16, causal_mask_softmax, rmsnorm, etc.)
        // - FP16 activation scratch allocation
        // - Attention score scratch allocation
        // - Any other lazy allocations inside kernel dispatch
        // (KV cache scatter addresses, bias broadcast modules, etc.)
        // Use dummy embeddings — the captured graph will use the stable input buffer.
        let dummy_embeddings = vec![0.0f32; input_size];
        let positions: Vec<u32> = (0..s).map(|i| i as u32).collect();

        // Reset KV cache lengths before warmup
        for layer_idx in 0..num_layers {
            self.kv_cache_lengths.insert(layer_idx, 0);
        }

        self.prefill_eager(
            &dummy_embeddings,
            &positions,
            num_layers,
            hidden_dim,
            intermediate_dim,
            epsilon,
        )?;

        Ok(())
    }

    /// PMAT-050: Capture prefill forward pass into a CUDA graph.
    ///
    /// Five-Whys Root Cause:
    /// 1. Why is TTFT 4.5x worse? 728 kernel launches × 65μs CPU overhead = 47ms.
    /// 2. Why 65μs per launch? cuBLAS HGEMM + host-side dispatch + argument packing.
    /// 3. Why not reduce launches? Can't fuse cuBLAS calls or eliminate layers.
    /// 4. Why not reduce per-launch overhead? cuBLAS internal overhead is fixed.
    /// 5. Why not batch all launches? → CUDA graph: capture 728 launches, replay as 1.
    ///
    /// Expected: 47ms CPU overhead → ~1ms graph launch overhead.
    /// TTFT: 78ms → ~32ms (within 2x of llama.cpp's 17ms).
    #[allow(clippy::too_many_arguments)]
    fn try_prefill_graph_capture(
        &mut self,
        s: usize,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        let start = std::time::Instant::now();

        // Warmup: compile modules, allocate buffers, cache weights
        self.warmup_prefill_for_capture(s, num_layers, hidden_dim, intermediate_dim, epsilon)?;

        // Reset KV cache lengths to 0 before capture
        for layer_idx in 0..num_layers {
            self.kv_cache_lengths.insert(layer_idx, 0);
        }

        // Pre-upload positions (0..S-1) — these are always the same for prefill
        let positions: Vec<u32> = (0..s).map(|i| i as u32).collect();
        if let Some(ref mut pos_buf) = self.workspace.positions_buf {
            if pos_buf.len() >= s {
                let mut wrapper = unsafe { GpuBuffer::<u32>::from_raw_parts(pos_buf.as_ptr(), s) };
                wrapper.copy_from_host(&positions)?;
                std::mem::forget(wrapper);
            }
        }

        // Copy dummy embeddings to stable input buffer (addresses must be valid during capture)
        let input_size = s * hidden_dim as usize;
        let dummy = vec![0.0f32; input_size];
        if let Some(ref mut input_buf) = self.prefill_graph_input_buf {
            input_buf.copy_from_host(&dummy)?;
        }

        // Set capture flags
        self.is_capturing = true;
        self.is_prefilling = true;

        // Begin graph capture
        self.stream.begin_capture(CaptureMode::Global)?;

        // Run the forward pass — all GPU operations will be recorded into the graph
        let capture_result = self.prefill_forward_captured(
            s,
            num_layers,
            hidden_dim,
            intermediate_dim,
            epsilon,
            &positions,
        );

        // End capture (must happen even if forward failed)
        let graph = self.stream.end_capture()?;
        self.is_capturing = false;
        self.is_prefilling = false;

        // Check if forward pass succeeded during capture
        capture_result?;

        // Instantiate the captured graph
        let graph_exec = graph.instantiate()?;
        self.prefill_graphs.insert(s, graph_exec);

        // Reset KV cache lengths (capture populated them, but with dummy data)
        for layer_idx in 0..num_layers {
            self.kv_cache_lengths.insert(layer_idx, 0);
        }

        eprintln!(
            "[PREFILL-GRAPH] Captured graph for S={} in {:.1}ms (728 launches → 1 graph)",
            s,
            start.elapsed().as_secs_f64() * 1000.0
        );

        Ok(())
    }

    /// PMAT-050: Run the prefill forward pass using stable buffers (for graph capture).
    ///
    /// Same as prefill_eager but:
    /// - Uses prefill_graph_input_buf instead of GpuBuffer::from_host
    /// - No PREFILL_TRACE timing (stream.synchronize() breaks capture)
    /// - No early return on error (must reach end_capture)
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn prefill_forward_captured(
        &mut self,
        s: usize,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
        positions: &[u32],
    ) -> Result<(), GpuError> {
        let q_dim = (self.kv_num_heads * self.kv_head_dim) as u32;
        let kv_dim = (self.kv_num_kv_heads * self.kv_head_dim) as u32;

        // Use stable input buffer (address captured by graph)
        let input_ptr = self
            .prefill_graph_input_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(
                    "PMAT-050: prefill_graph_input_buf missing".to_string(),
                )
            })?
            .as_ptr();
        let input_len = s * hidden_dim as usize;
        // SAFETY: Pointer from valid allocation, length = S × hidden_dim
        let input_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(input_ptr, input_len) };

        // Extract workspace buffers (same as prefill_eager)
        self.validate_batched_workspace(s as u32, positions.len())?;

        let hidden_buf1_ptr = self
            .workspace
            .hidden_buf1
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PMAT-050: hidden_buf1 missing".to_string())
            })?
            .as_ptr();
        let hidden_buf1_len = self.workspace.hidden_buf1.as_ref().unwrap().len();
        let hidden_buf2_ptr = self
            .workspace
            .hidden_buf2
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PMAT-050: hidden_buf2 missing".to_string())
            })?
            .as_ptr();
        let hidden_buf2_len = self.workspace.hidden_buf2.as_ref().unwrap().len();
        let input_staging_ptr = self
            .workspace
            .input_staging
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PMAT-050: input_staging missing".to_string())
            })?
            .as_ptr();
        let input_staging_len = self.workspace.input_staging.as_ref().unwrap().len();
        let q_buf_ptr = self
            .workspace
            .q_buf
            .as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PMAT-050: q_buf missing".to_string()))?
            .as_ptr();
        let q_buf_len = self.workspace.q_buf.as_ref().unwrap().len();
        let k_buf_ptr = self
            .workspace
            .k_buf
            .as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PMAT-050: k_buf missing".to_string()))?
            .as_ptr();
        let k_buf_len = self.workspace.k_buf.as_ref().unwrap().len();
        let v_buf_ptr = self
            .workspace
            .v_buf
            .as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PMAT-050: v_buf missing".to_string()))?
            .as_ptr();
        let v_buf_len = self.workspace.v_buf.as_ref().unwrap().len();
        let ffn_gate_ptr = self
            .workspace
            .ffn_gate_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PMAT-050: ffn_gate_buf missing".to_string())
            })?
            .as_ptr();
        let ffn_gate_len = self.workspace.ffn_gate_buf.as_ref().unwrap().len();
        let ffn_up_ptr = self
            .workspace
            .ffn_up_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PMAT-050: ffn_up_buf missing".to_string())
            })?
            .as_ptr();
        let ffn_up_len = self.workspace.ffn_up_buf.as_ref().unwrap().len();
        let ffn_act_ptr = self
            .workspace
            .ffn_act_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PMAT-050: ffn_act_buf missing".to_string())
            })?
            .as_ptr();
        let ffn_act_len = self.workspace.ffn_act_buf.as_ref().unwrap().len();
        let attn_out_ptr = self
            .workspace
            .attn_out_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PMAT-050: attn_out_buf missing".to_string())
            })?
            .as_ptr();
        let attn_out_len = self.workspace.attn_out_buf.as_ref().unwrap().len();

        // SAFETY: All pointers valid from workspace allocation
        let hidden_buf1 =
            unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_buf1_ptr, hidden_buf1_len) };
        let hidden_buf2 =
            unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_buf2_ptr, hidden_buf2_len) };
        let input_staging =
            unsafe { GpuBuffer::<f32>::from_raw_parts(input_staging_ptr, input_staging_len) };
        let q_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(q_buf_ptr, q_buf_len) };
        let k_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(k_buf_ptr, k_buf_len) };
        let v_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(v_buf_ptr, v_buf_len) };
        let ffn_gate_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ffn_gate_ptr, ffn_gate_len) };
        let ffn_up_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ffn_up_ptr, ffn_up_len) };
        let ffn_act_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ffn_act_ptr, ffn_act_len) };
        let attn_out_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(attn_out_ptr, attn_out_len) };

        // Process all layers (same as prefill_eager but no timing)
        for layer_idx in 0..num_layers {
            if layer_idx >= self.indexed_layer_weights.len() {
                // Must forget wrappers before returning error
                std::mem::forget(input_buf);
                std::mem::forget(hidden_buf1);
                std::mem::forget(hidden_buf2);
                std::mem::forget(input_staging);
                std::mem::forget(q_buf);
                std::mem::forget(k_buf);
                std::mem::forget(v_buf);
                std::mem::forget(attn_out_buf);
                std::mem::forget(ffn_gate_buf);
                std::mem::forget(ffn_up_buf);
                std::mem::forget(ffn_act_buf);
                return Err(GpuError::InvalidLaunchConfig(format!(
                    "PMAT-050: Layer {} weights not indexed (have {})",
                    layer_idx,
                    self.indexed_layer_weights.len()
                )));
            }
            let layer_weights = self.get_indexed_layer(layer_idx).clone();
            let layer_input = if layer_idx == 0 {
                &input_buf
            } else {
                &hidden_buf2
            };

            // Phase 1: RMSNorm + QKV + bias + RoPE
            self.batched_qkv_rope_phase(
                layer_input,
                &hidden_buf1,
                &q_buf,
                &k_buf,
                &v_buf,
                q_buf_ptr,
                k_buf_ptr,
                v_buf_ptr,
                hidden_buf1_ptr,
                layer_idx,
                &layer_weights,
                s as u32,
                positions,
                hidden_dim,
                q_dim,
                kv_dim,
                epsilon,
            )?;

            // Phase 2: Attention + output proj + residuals + FFN
            self.batched_attn_ffn_phase(
                layer_input,
                &hidden_buf1,
                &hidden_buf2,
                &input_staging,
                &q_buf,
                &k_buf,
                &v_buf,
                &attn_out_buf,
                &ffn_gate_buf,
                &ffn_up_buf,
                &ffn_act_buf,
                q_buf_ptr,
                k_buf_ptr,
                v_buf_ptr,
                attn_out_ptr,
                hidden_buf1_ptr,
                ffn_gate_ptr,
                ffn_up_ptr,
                ffn_act_ptr,
                layer_idx,
                &layer_weights,
                s as u32,
                positions,
                hidden_dim,
                intermediate_dim,
                q_dim,
                kv_dim,
                epsilon,
            )?;
        }

        // Forget all wrappers
        std::mem::forget(input_buf);
        std::mem::forget(hidden_buf1);
        std::mem::forget(hidden_buf2);
        std::mem::forget(input_staging);
        std::mem::forget(q_buf);
        std::mem::forget(k_buf);
        std::mem::forget(v_buf);
        std::mem::forget(attn_out_buf);
        std::mem::forget(ffn_gate_buf);
        std::mem::forget(ffn_up_buf);
        std::mem::forget(ffn_act_buf);

        Ok(())
    }

    /// PMAT-050: Replay captured prefill graph with new embeddings.
    ///
    /// 1. Copy embeddings to stable input buffer (async memcpy, doesn't invalidate graph)
    /// 2. Reset KV cache lengths to 0 (prefill always starts fresh)
    /// 3. Launch captured graph (replays all 728 kernel launches as one operation)
    /// 4. Synchronize
    /// 5. Set KV cache lengths to S (so decode knows where to continue)
    fn prefill_graphed_replay(
        &mut self,
        embeddings: &[f32],
        s: usize,
        num_layers: usize,
        hidden_dim: u32,
    ) -> Result<(), GpuError> {
        // 1. Copy new embeddings to stable input buffer
        // CORRECTNESS-013 FIX: Use async H2D on self.stream for same-stream ordering
        // with the graph launch below. copy_from_host uses cuMemcpyHtoD (stream 0)
        // which has no ordering guarantee with CU_STREAM_NON_BLOCKING graph launches.
        if let Some(ref mut input_buf) = self.prefill_graph_input_buf {
            // SAFETY: embeddings valid for duration of this function
            unsafe {
                input_buf.copy_from_host_async(embeddings, &self.stream)?;
            }
        } else {
            return Err(GpuError::InvalidLaunchConfig(
                "PMAT-050: prefill_graph_input_buf missing for replay".to_string(),
            ));
        }

        // 2. Positions are always 0..S-1, pre-uploaded during capture — no update needed.
        // The workspace.positions_buf has the correct values baked in.

        // 3. Reset KV cache lengths (prefill starts from empty cache)
        for layer_idx in 0..num_layers {
            self.kv_cache_lengths.insert(layer_idx, 0);
        }

        // 4. Launch the captured graph
        if let Some(graph_exec) = self.prefill_graphs.get(&s) {
            graph_exec.launch(self.stream.raw())?;
        } else {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PMAT-050: No captured graph for S={}",
                s
            )));
        }

        // 5. Synchronize — wait for all GPU work to complete
        self.stream.synchronize()?;

        // 6. Update KV cache lengths to S (so subsequent decode knows cache is populated)
        for layer_idx in 0..num_layers {
            self.kv_cache_lengths.insert(layer_idx, s);
        }

        Ok(())
    }

    /// PMAT-083: Extract first predicted token from prefill hidden state.
    ///
    /// Runs output RMSNorm + LM head GEMV + GPU argmax on the last position's
    /// hidden state (still on GPU from prefill_eager). Eliminates the separate
    /// first decode step, saving ~7ms TTFT.
    ///
    /// Must be called AFTER prefill_eager but BEFORE force_workspace_reinit,
    /// while hidden_buf2 still contains valid prefill output.
    pub(crate) fn prefill_extract_first_token(
        &mut self,
        last_pos: usize,
        hidden_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<u32, GpuError> {
        // 1. Get pointer to last position's hidden state in hidden_buf2
        let hidden_buf2_ptr = self
            .workspace
            .hidden_buf2
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(
                    "PMAT-083: hidden_buf2 missing for first token extraction".to_string(),
                )
            })?
            .as_ptr();
        let offset_bytes = last_pos as u64 * hidden_dim as u64 * 4; // f32 = 4 bytes
        let last_hidden_ptr = hidden_buf2_ptr + offset_bytes;

        // Create non-owning view of last position's hidden state (1 × hidden_dim)
        let last_hidden =
            unsafe { GpuBuffer::<f32>::from_raw_parts(last_hidden_ptr, hidden_dim as usize) };

        // 2. RMSNorm (output norm)
        let output_norm_ptr = self.output_norm_ptr;
        let output_norm_len = self.output_norm_len;
        if output_norm_ptr == 0 {
            std::mem::forget(last_hidden);
            return Err(GpuError::InvalidLaunchConfig(
                "PMAT-083: output_norm not loaded".to_string(),
            ));
        }

        // Allocate temporary buffer for normed output (1 × hidden_dim)
        let normed_buf = GpuBuffer::<f32>::new(&self.context, hidden_dim as usize)?;
        self.rmsnorm_ptr_into(
            &last_hidden,
            output_norm_ptr,
            output_norm_len,
            &normed_buf,
            hidden_dim,
            epsilon,
        )?;
        std::mem::forget(last_hidden);

        // 3. LM head GEMV (vocab_size × hidden_dim → vocab_size logits)
        let lm_head_ptr = self.lm_head_ptr;
        let lm_head_qtype =
            WeightQuantType::from_size(self.lm_head_len, vocab_size as usize, hidden_dim as usize)
                .unwrap_or(self.lm_head_qtype);

        if lm_head_ptr == 0 {
            return Err(GpuError::InvalidLaunchConfig(
                "PMAT-083: lm_head not loaded".to_string(),
            ));
        }

        let logits_buf = GpuBuffer::<f32>::new(&self.context, vocab_size as usize)?;
        self.q8_activation_valid = false; // LM head input differs from layer GEMVs
        self.gemv_dispatch(
            lm_head_qtype,
            lm_head_ptr,
            &normed_buf,
            &logits_buf,
            vocab_size,
            hidden_dim,
        )?;

        // 4. LM head bias (if present)
        if self.lm_head_bias_ptr != 0 && self.lm_head_bias_len > 0 {
            let bias_buf = unsafe {
                GpuBuffer::<f32>::from_raw_parts(self.lm_head_bias_ptr, self.lm_head_bias_len)
            };
            self.residual_add_into(&logits_buf, &bias_buf, &logits_buf, vocab_size)?;
            std::mem::forget(bias_buf);
        }

        // 5. GPU argmax → token ID
        let token = self.gpu_argmax(logits_buf.as_ptr(), vocab_size)?;

        Ok(token)
    }
}
