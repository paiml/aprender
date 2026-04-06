impl CudaExecutor {
    /// Validate workspace is initialized and large enough for batch size M.
    pub(crate) fn validate_batched_workspace(&self, m: u32, positions_len: usize) -> Result<(), GpuError> {
        if !self.workspace.initialized {
            return Err(GpuError::InvalidLaunchConfig(
                "PAR-111: Workspace not initialized".to_string(),
            ));
        }
        if self.workspace.buffer_capacity < m as usize {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-111: Workspace buffer_capacity {} < m {}",
                self.workspace.buffer_capacity, m
            )));
        }
        if positions_len != m as usize {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-111: positions.len() {} != m {}",
                positions_len, m
            )));
        }
        Ok(())
    }

    /// PAR-111: Batched forward pass for M sequences through a single layer.
    ///
    /// Processes M sequences in parallel using batched GEMV kernels.
    /// Each sequence has independent KV cache state.
    ///
    /// # Performance Benefit
    ///
    /// Batched GEMV reads/dequantizes weights ONCE for all M inputs:
    /// - M=1: Baseline throughput (~360 tok/s)
    /// - M=4: 16x GEMV speedup → 857+ tok/s aggregate
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn transformer_layer_batched(
        &mut self,
        input: &GpuBuffer<f32>,
        layer_idx: usize,
        layer_weights: &ValidatedLayerWeights,
        m: u32,
        positions: &[u32],
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        self.validate_batched_workspace(m, positions.len())?;

        let q_dim = (self.kv_num_heads * self.kv_head_dim) as u32;
        let kv_dim = (self.kv_num_kv_heads * self.kv_head_dim) as u32;

        // Get batched buffer pointers (M× larger buffers allocated by init_batched_workspace)
        let hidden_buf1_ptr = self.workspace.hidden_buf1.as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PAR-111: hidden_buf1 not initialized".to_string()))?.as_ptr();
        let hidden_buf1_len = self.workspace.hidden_buf1.as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PAR-111: hidden_buf1 not initialized".to_string()))?.len();
        let hidden_buf2_ptr = self.workspace.hidden_buf2.as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PAR-111: hidden_buf2 not initialized".to_string()))?.as_ptr();
        let hidden_buf2_len = self.workspace.hidden_buf2.as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PAR-111: hidden_buf2 not initialized".to_string()))?.len();
        let input_staging_ptr = self.workspace.input_staging.as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PAR-111: input_staging not initialized".to_string()))?.as_ptr();
        let input_staging_len = self.workspace.input_staging.as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PAR-111: input_staging not initialized".to_string()))?.len();
        let q_buf_ptr = self.workspace.q_buf.as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PAR-111: q_buf not initialized".to_string()))?.as_ptr();
        let q_buf_len = self.workspace.q_buf.as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PAR-111: q_buf not initialized".to_string()))?.len();
        let k_buf_ptr = self.workspace.k_buf.as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PAR-111: k_buf not initialized".to_string()))?.as_ptr();
        let k_buf_len = self.workspace.k_buf.as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PAR-111: k_buf not initialized".to_string()))?.len();
        let v_buf_ptr = self.workspace.v_buf.as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PAR-111: v_buf not initialized".to_string()))?.as_ptr();
        let v_buf_len = self.workspace.v_buf.as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PAR-111: v_buf not initialized".to_string()))?.len();
        let ffn_gate_ptr = self.workspace.ffn_gate_buf.as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PAR-111: ffn_gate_buf not initialized".to_string()))?.as_ptr();
        let ffn_gate_len = self.workspace.ffn_gate_buf.as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PAR-111: ffn_gate_buf not initialized".to_string()))?.len();
        let ffn_up_ptr = self.workspace.ffn_up_buf.as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PAR-111: ffn_up_buf not initialized".to_string()))?.as_ptr();
        let ffn_up_len = self.workspace.ffn_up_buf.as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PAR-111: ffn_up_buf not initialized".to_string()))?.len();
        let ffn_act_ptr = self.workspace.ffn_act_buf.as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PAR-111: ffn_act_buf not initialized".to_string()))?.as_ptr();
        let ffn_act_len = self.workspace.ffn_act_buf.as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PAR-111: ffn_act_buf not initialized".to_string()))?.len();
        let attn_out_ptr = self.workspace.attn_out_buf.as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PAR-111: attn_out_buf not initialized".to_string()))?.as_ptr();
        let attn_out_len = self.workspace.attn_out_buf.as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PAR-111: attn_out_buf not initialized".to_string()))?.len();

        // Create temporary buffer wrappers (M× sized)
        // SAFETY: Pointers valid from workspace allocation, lengths verified
        let hidden_buf1 = unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_buf1_ptr, hidden_buf1_len) };
        let hidden_buf2 = unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_buf2_ptr, hidden_buf2_len) };
        let input_staging = unsafe { GpuBuffer::<f32>::from_raw_parts(input_staging_ptr, input_staging_len) };
        let q_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(q_buf_ptr, q_buf_len) };
        let k_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(k_buf_ptr, k_buf_len) };
        // SAFETY: v_buf_ptr valid from workspace allocation, v_buf_len verified
        let v_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(v_buf_ptr, v_buf_len) };
        // SAFETY: pointers valid from workspace allocation above, lengths verified at allocation
        let ffn_gate_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ffn_gate_ptr, ffn_gate_len) };
        let ffn_up_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ffn_up_ptr, ffn_up_len) };
        let ffn_act_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ffn_act_ptr, ffn_act_len) };
        let attn_out_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(attn_out_ptr, attn_out_len) };

        // Phase 1: RMSNorm + QKV projections + bias + RoPE
        self.batched_qkv_rope_phase(
            input, &hidden_buf1, &q_buf, &k_buf, &v_buf,
            q_buf_ptr, k_buf_ptr, v_buf_ptr, hidden_buf1_ptr,
            layer_idx, layer_weights, m, positions,
            hidden_dim, q_dim, kv_dim, epsilon,
        )?;

        // Phase 2: Attention + output projection + residuals + FFN
        self.batched_attn_ffn_phase(
            input, &hidden_buf1, &hidden_buf2, &input_staging,
            &q_buf, &k_buf, &v_buf, &attn_out_buf,
            &ffn_gate_buf, &ffn_up_buf, &ffn_act_buf,
            q_buf_ptr, k_buf_ptr, v_buf_ptr,
            attn_out_ptr, hidden_buf1_ptr, ffn_gate_ptr, ffn_up_ptr, ffn_act_ptr,
            layer_idx, layer_weights, m, positions,
            hidden_dim, intermediate_dim, q_dim, kv_dim, epsilon,
        )?;

        // Prevent Drop from freeing borrowed memory
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

        // Output is in hidden_buf2 (M × hidden_dim packed)
        Ok(())
    }
}

include!("batched_qkv.rs");
include!("batched_ffn.rs");
