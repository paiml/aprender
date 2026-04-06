
impl CudaExecutor {
    /// PAR-058: Debug helper — sync stream, copy GPU buffer to host, check for NaN.
    /// Extracted to reduce cyclomatic/cognitive complexity in layer functions.
    fn debug_check_buf(
        &mut self,
        buf: &GpuBuffer<f32>,
        label: &str,
        layer_idx: usize,
    ) -> Result<(), GpuError> {
        self.stream.synchronize()?;
        let mut host = vec![0.0f32; buf.len()];
        buf.copy_to_host(&mut host)?;
        let nan_count = host.iter().filter(|x| x.is_nan()).count();
        if nan_count > 0 {
            eprintln!("[PAR-058-L{}] {} has {} NaN", layer_idx, label, nan_count);
        } else {
            eprintln!(
                "[PAR-058-L{}] {} OK, first 3: {:?}",
                layer_idx, label, &host[..3.min(host.len())]
            );
        }
        Ok(())
    }

    /// Dispatch a GEMV operation based on weight quantization type.
    ///
    /// PMAT-232 CONTRACT: Exhaustive dispatch — no catch-all.
    /// See tensor-layout-v1.yaml quant_dispatch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gemv_dispatch(
        &mut self,
        qtype: WeightQuantType,
        weight_ptr: u64,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        match qtype {
            WeightQuantType::Q4_0 => self.q4_0_gemv_into(weight_ptr, input, output, n, k),
            WeightQuantType::Q4_1 => self.q4_1_gemv_into(weight_ptr, input, output, n, k),
            WeightQuantType::Q5_0 => self.q5_0_gemv_into(weight_ptr, input, output, n, k),
            WeightQuantType::Q4K => self.q4k_gemv_into(weight_ptr, input, output, n, k),
            WeightQuantType::Q5K => self.q5k_gemv_into(weight_ptr, input, output, n, k),
            WeightQuantType::Q6K => self.q6k_gemv_into(weight_ptr, input, output, n, k),
            WeightQuantType::Q8_0 => self.q8_0_gemv_into(weight_ptr, input, output, n, k),
            WeightQuantType::F32 => self.f32_gemv_into(weight_ptr, input, output, n, k),
        }
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub(crate) fn transformer_layer_workspace_inner(
        &mut self,
        input: &GpuBuffer<f32>,
        layer_idx: usize,
        layer_weights: &ValidatedLayerWeights,
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
        position: u32, // PAR-070: Explicit position for RoPE and KV cache
        skip_debug: bool,
    ) -> Result<(), GpuError> {
        // Verify workspace is initialized
        if !self.workspace.initialized {
            return Err(GpuError::InvalidLaunchConfig(
                "PAR-044: Workspace not initialized. Call init_workspace() first.".to_string(),
            ));
        }

        // PROHIBITION-OF-MIRACLES (T-COV-95): Validate pointers BEFORE kernel launch
        // Null pointers corrupt GPU context - fail loudly at API boundary
        if layer_weights.attn_norm_ptr == 0 {
            return Err(GpuError::InvalidParameter(
                "attn_norm_ptr is null (0)".into(),
            ));
        }
        if layer_weights.attn_q_ptr == 0 {
            return Err(GpuError::InvalidParameter("attn_q_ptr is null (0)".into()));
        }
        if layer_weights.attn_k_ptr == 0 {
            return Err(GpuError::InvalidParameter("attn_k_ptr is null (0)".into()));
        }
        if layer_weights.attn_v_ptr == 0 {
            return Err(GpuError::InvalidParameter("attn_v_ptr is null (0)".into()));
        }
        if layer_weights.attn_output_ptr == 0 {
            return Err(GpuError::InvalidParameter(
                "attn_output_ptr is null (0)".into(),
            ));
        }
        if layer_weights.ffn_norm_ptr == 0 {
            return Err(GpuError::InvalidParameter(
                "ffn_norm_ptr is null (0)".into(),
            ));
        }
        if layer_weights.ffn_gate_ptr == 0 {
            return Err(GpuError::InvalidParameter(
                "ffn_gate_ptr is null (0)".into(),
            ));
        }
        if layer_weights.ffn_up_ptr == 0 {
            return Err(GpuError::InvalidParameter("ffn_up_ptr is null (0)".into()));
        }
        if layer_weights.ffn_down_ptr == 0 {
            return Err(GpuError::InvalidParameter(
                "ffn_down_ptr is null (0)".into(),
            ));
        }

        // Get dimension info
        let q_dim = (self.kv_num_heads * self.kv_head_dim) as u32;
        let kv_dim = (self.kv_num_kv_heads * self.kv_head_dim) as u32;

        // PAR-044: Get buffer pointers/lengths to avoid borrow conflicts
        // SAFETY: All workspace buffers are initialized (verified above) and remain valid
        let hidden_buf1_ptr = self
            .workspace
            .hidden_buf1
            .as_ref()
            .expect("hidden_buf1 must be initialized")
            .as_ptr();
        let hidden_buf1_len = self
            .workspace
            .hidden_buf1
            .as_ref()
            .expect("hidden_buf1 must be initialized")
            .len();
        let hidden_buf2_ptr = self
            .workspace
            .hidden_buf2
            .as_ref()
            .expect("hidden_buf2 must be initialized")
            .as_ptr();
        let hidden_buf2_len = self
            .workspace
            .hidden_buf2
            .as_ref()
            .expect("hidden_buf2 must be initialized")
            .len();
        // PAR-044 FIX: Use input_staging as scratch for residual1 to avoid read/write conflict
        // when input aliases hidden_buf2 (layers 1+)
        let input_staging_ptr = self
            .workspace
            .input_staging
            .as_ref()
            .expect("input_staging must be initialized")
            .as_ptr();
        let input_staging_len = self
            .workspace
            .input_staging
            .as_ref()
            .expect("input_staging must be initialized")
            .len();
        let q_buf_ptr = self
            .workspace
            .q_buf
            .as_ref()
            .expect("q_buf must be initialized")
            .as_ptr();
        let q_buf_len = self
            .workspace
            .q_buf
            .as_ref()
            .expect("q_buf must be initialized")
            .len();
        let k_buf_ptr = self
            .workspace
            .k_buf
            .as_ref()
            .expect("k_buf must be initialized")
            .as_ptr();
        let k_buf_len = self
            .workspace
            .k_buf
            .as_ref()
            .expect("k_buf must be initialized")
            .len();
        let v_buf_ptr = self
            .workspace
            .v_buf
            .as_ref()
            .expect("v_buf must be initialized")
            .as_ptr();
        let v_buf_len = self
            .workspace
            .v_buf
            .as_ref()
            .expect("v_buf must be initialized")
            .len();
        let ffn_gate_ptr = self
            .workspace
            .ffn_gate_buf
            .as_ref()
            .expect("ffn_gate_buf must be initialized")
            .as_ptr();
        let ffn_gate_len = self
            .workspace
            .ffn_gate_buf
            .as_ref()
            .expect("ffn_gate_buf must be initialized")
            .len();
        let ffn_up_ptr = self
            .workspace
            .ffn_up_buf
            .as_ref()
            .expect("ffn_up_buf must be initialized")
            .as_ptr();
        let ffn_up_len = self
            .workspace
            .ffn_up_buf
            .as_ref()
            .expect("ffn_up_buf must be initialized")
            .len();
        let ffn_act_ptr = self
            .workspace
            .ffn_act_buf
            .as_ref()
            .expect("ffn_act_buf must be initialized")
            .as_ptr();
        let ffn_act_len = self
            .workspace
            .ffn_act_buf
            .as_ref()
            .expect("ffn_act_buf must be initialized")
            .len();
        // PAR-051: Attention output workspace buffer
        let attn_out_ptr = self
            .workspace
            .attn_out_buf
            .as_ref()
            .expect("attn_out_buf must be initialized")
            .as_ptr();
        let attn_out_len = self
            .workspace
            .attn_out_buf
            .as_ref()
            .expect("attn_out_buf must be initialized")
            .len();

        // Create temporary non-owning buffer wrappers
        // These will be forgotten at the end to avoid freeing borrowed memory
        let hidden_buf1 =
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_buf1_ptr, hidden_buf1_len) };
        let hidden_buf2 =
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_buf2_ptr, hidden_buf2_len) };
        let input_staging =
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe { GpuBuffer::<f32>::from_raw_parts(input_staging_ptr, input_staging_len) };
        // PAR-060: Q/K buffers for RoPE application
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let q_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(q_buf_ptr, q_buf_len) };
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let k_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(k_buf_ptr, k_buf_len) };
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let v_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(v_buf_ptr, v_buf_len) };
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let ffn_gate_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ffn_gate_ptr, ffn_gate_len) };
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let ffn_up_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ffn_up_ptr, ffn_up_len) };
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let ffn_act_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ffn_act_ptr, ffn_act_len) };
        // PAR-051: Attention output buffer (eliminates 28 allocations per token)
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let attn_out_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(attn_out_ptr, attn_out_len) };

        // PAR-073: Check if profiling is enabled (avoid overhead when disabled)
        let profiling = self.profiler.is_enabled();

        // Phase 1-2: RMSNorm + QKV projections + bias + RoPE
        self.workspace_qkv_rope_phase(
            input, &hidden_buf1, &q_buf, &k_buf, &v_buf,
            layer_idx, layer_weights, hidden_dim, q_dim, kv_dim,
            epsilon, position, skip_debug, profiling,
        )?;

        // Phase 3-5: Attention + output projection + residual1
        self.workspace_attention_residual_phase(
            input, &hidden_buf1, &q_buf, &k_buf, &v_buf,
            &attn_out_buf, &input_staging,
            layer_idx, layer_weights, hidden_dim, q_dim,
            skip_debug, profiling,
        )?;

        // Phase 6-10: FFN RMSNorm + gate/up + SwiGLU + down + residual2
        self.workspace_ffn_phase(
            &hidden_buf1, &hidden_buf2, &input_staging,
            &ffn_gate_buf, &ffn_up_buf, &ffn_act_buf,
            layer_idx, layer_weights, hidden_dim, intermediate_dim,
            epsilon, skip_debug, profiling,
        )?;

        // Prevent Drop from freeing the borrowed memory
        std::mem::forget(hidden_buf1);
        std::mem::forget(hidden_buf2);
        std::mem::forget(input_staging);
        std::mem::forget(q_buf);
        std::mem::forget(k_buf);
        std::mem::forget(v_buf);
        std::mem::forget(attn_out_buf); // PAR-051
        std::mem::forget(ffn_gate_buf);
        std::mem::forget(ffn_up_buf);
        std::mem::forget(ffn_act_buf);

        // Output is now in hidden_buf2
        Ok(())
    }
}

include!("indexed_transformer.rs");
include!("apply.rs");
include!("phase_attention.rs");
include!("indexed_ffn.rs");
