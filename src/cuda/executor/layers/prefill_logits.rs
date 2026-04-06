// realizr#203: Batched logits extraction from prefill hidden states.
// Extracted from batched_forward.rs to keep file under 500 lines.
// include!()-ed from rmsnorm.rs into CudaExecutor impl block.

    /// realizr#203: Extract all logits from prefill hidden states for perplexity.
    ///
    /// After `prefill_all_layers_gpu` completes, `hidden_buf2[S × hidden_dim]`
    /// contains the final hidden states for all S positions. This function applies
    /// output RMSNorm + LM head GEMM and downloads all S × vocab_size logits.
    ///
    /// NOTE: Has correctness issues at large S due to batched_gemv_or_gemm layout.
    /// Currently unused in favor of per-position hidden_to_logits() approach.
    /// Kept for future optimization when batched LM head is fixed.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn prefill_extract_all_logits(
        &mut self,
        s: usize,
        hidden_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<Vec<f32>, GpuError> {
        if s == 0 {
            return Ok(Vec::new());
        }

        let output_norm_buf = self.rmsnorm_cache.get("output_norm.gamma").ok_or_else(|| {
            GpuError::InvalidLaunchConfig("realizr#203: output_norm not cached".to_string())
        })?;
        let output_norm_ptr = output_norm_buf.as_ptr();
        let output_norm_len = hidden_dim as usize;

        let hidden_buf2_ptr = self.workspace.hidden_buf2.as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("realizr#203: hidden_buf2 missing".to_string()))?
            .as_ptr();
        let hidden_buf2_len = s * hidden_dim as usize;

        let normed_buf = if let Some(ref buf) = self.workspace.normed_hidden_buf {
            if buf.len() >= hidden_buf2_len { buf.as_ptr() }
            else {
                let tmp = GpuBuffer::<f32>::new(&self.context, hidden_buf2_len)?;
                let ptr = tmp.as_ptr();
                self.workspace.normed_hidden_buf = Some(tmp);
                ptr
            }
        } else {
            let tmp = GpuBuffer::<f32>::new(&self.context, hidden_buf2_len)?;
            let ptr = tmp.as_ptr();
            self.workspace.normed_hidden_buf = Some(tmp);
            ptr
        };

        let hidden_wrapper = unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_buf2_ptr, hidden_buf2_len) };
        let normed_wrapper = unsafe { GpuBuffer::<f32>::from_raw_parts(normed_buf, hidden_buf2_len) };

        self.batched_rmsnorm_ptr_into(
            &hidden_wrapper, output_norm_ptr, output_norm_len,
            &normed_wrapper, hidden_dim, s as u32, epsilon,
        )?;

        std::mem::forget(hidden_wrapper);
        std::mem::forget(normed_wrapper);

        if self.lm_head_ptr == 0 {
            return Err(GpuError::InvalidLaunchConfig("realizr#203: LM head not indexed".to_string()));
        }
        let lm_head_ptr = self.lm_head_ptr;
        let lm_head_qtype = self.lm_head_qtype;
        let logits_size = s * vocab_size as usize;
        let logits_gpu = GpuBuffer::<f32>::new(&self.context, logits_size)?;
        let normed_wrapper2 = unsafe { GpuBuffer::<f32>::from_raw_parts(normed_buf, hidden_buf2_len) };

        self.batched_gemv_or_gemm(
            lm_head_qtype, lm_head_ptr,
            &normed_wrapper2, &logits_gpu,
            normed_buf, logits_gpu.as_ptr(),
            s as u32, vocab_size, hidden_dim,
        )?;

        std::mem::forget(normed_wrapper2);

        self.stream.synchronize()?;
        let mut logits = vec![0.0f32; logits_size];
        logits_gpu.copy_to_host(&mut logits)?;
        Ok(logits)
    }
