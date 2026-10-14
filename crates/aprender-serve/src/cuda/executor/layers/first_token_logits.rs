//! #4971: the prefill's first token with the logits its argmax read.
//!
//! `prefill_extract_first_token` (PMAT-083, `prefill.rs`) chooses the first
//! token on the device and drops the logits. A request for logprobs needs those
//! logits, so this module runs the same steps (output RMSNorm, LM head GEMV,
//! LM head bias, GPU argmax) on one logits buffer and copies that buffer to the
//! host after the argmax. The token and the logits come from the same buffer,
//! so the reported distribution is the one the choice was made from.

#![allow(clippy::wildcard_imports)]

use super::super::*;

impl CudaExecutor {
    /// [`Self::prefill_extract_first_token`], with the logits the argmax read
    /// copied into `out` (resized to `vocab_size`) after the argmax (#4971).
    ///
    /// Same preconditions: call it after `prefill_eager` and before
    /// `force_workspace_reinit`, while `hidden_buf2` holds the prefill output.
    pub(crate) fn prefill_extract_first_token_and_logits(
        &mut self,
        last_pos: usize,
        hidden_dim: u32,
        vocab_size: u32,
        epsilon: f32,
        out: &mut Vec<f32>,
    ) -> Result<u32, GpuError> {
        let logits_buf = self.prefill_last_row_logits(last_pos, hidden_dim, vocab_size, epsilon)?;
        let token = self.gpu_argmax(logits_buf.as_ptr(), vocab_size)?;
        out.clear();
        out.resize(vocab_size as usize, 0.0);
        self.stream.synchronize()?;
        logits_buf.copy_to_host(out)?;
        Ok(token)
    }

    /// Steps 1-4 of [`Self::prefill_extract_first_token`]: the last prompt
    /// row's logits, left on the device.
    fn prefill_last_row_logits(
        &mut self,
        last_pos: usize,
        hidden_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        let normed_buf = self.prefill_last_row_normed(last_pos, hidden_dim, epsilon)?;
        let lm_head_ptr = self.lm_head_ptr;
        // #3908: a consistent declaration wins over the size guess (BF16 == F16 in size).
        let lm_head_qtype = WeightQuantType::resolve_declared_or_sized(
            Some(self.lm_head_qtype),
            self.lm_head_len,
            vocab_size as usize,
            hidden_dim as usize,
        )
        .unwrap_or(self.lm_head_qtype);
        if lm_head_ptr == 0 {
            return Err(GpuError::InvalidLaunchConfig(
                "#4971: lm_head not loaded".to_string(),
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

        if self.lm_head_bias_ptr != 0 && self.lm_head_bias_len > 0 {
            // SAFETY: constructs a non-owning `GpuBuffer` view over an already-allocated device region (`ptr`, element count `len`) that stays live for the kernel call; the view is `leak()`ed afterwards so its Drop never frees the borrowed device allocation (no double-free).
            let bias_buf = unsafe {
                GpuBuffer::<f32>::from_raw_parts(self.lm_head_bias_ptr, self.lm_head_bias_len)
            };
            self.residual_add_into(&logits_buf, &bias_buf, &logits_buf, vocab_size)?;
            std::mem::forget(bias_buf);
        }
        Ok(logits_buf)
    }

    /// The last prompt row of `hidden_buf2`, through the output RMSNorm.
    fn prefill_last_row_normed(
        &mut self,
        last_pos: usize,
        hidden_dim: u32,
        epsilon: f32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        let hidden_buf2_ptr = self
            .workspace
            .hidden_buf2
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(
                    "#4971: hidden_buf2 missing for first token extraction".to_string(),
                )
            })?
            .as_ptr();
        let offset_bytes = last_pos as u64 * hidden_dim as u64 * 4; // f32 = 4 bytes
        let last_hidden_ptr = hidden_buf2_ptr + offset_bytes;

        let output_norm_ptr = self.output_norm_ptr;
        let output_norm_len = self.output_norm_len;
        if output_norm_ptr == 0 {
            return Err(GpuError::InvalidLaunchConfig(
                "#4971: output_norm not loaded".to_string(),
            ));
        }

        // Non-owning view of the last position's hidden state (1 × hidden_dim).
        let last_hidden =
            // SAFETY: constructs a non-owning `GpuBuffer` view over an already-allocated device region (`ptr`, element count `len`) that stays live for the kernel call; the view is `leak()`ed afterwards so its Drop never frees the borrowed device allocation (no double-free).
            unsafe { GpuBuffer::<f32>::from_raw_parts(last_hidden_ptr, hidden_dim as usize) };
        let normed = GpuBuffer::<f32>::new(&self.context, hidden_dim as usize).and_then(|normed| {
            self.rmsnorm_ptr_into(
                &last_hidden,
                output_norm_ptr,
                output_norm_len,
                &normed,
                hidden_dim,
                epsilon,
            )
            .map(|()| normed)
        });
        std::mem::forget(last_hidden);
        normed
    }
}
