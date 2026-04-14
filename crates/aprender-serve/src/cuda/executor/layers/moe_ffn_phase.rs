/// SPEC-MOE-APR-001 Phase 8: All-GPU MoE FFN phase (placeholder).
///
/// Skips FFN computation and copies hidden through (residual only).
/// This validates the attention-on-GPU pipeline for correctness.
///
/// TODO: Wire fused_moe_gemv for actual expert dispatch on GPU.
/// Contract: moe-cuda-kernel-v1.yaml (FALSIFY-CUDA-MOE-002: >= 61 tok/s)
impl super::super::CudaExecutor {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn workspace_moe_ffn_phase(
        &mut self,
        _hidden_buf1: &trueno_gpu::driver::GpuBuffer<f32>,
        hidden_buf2: &trueno_gpu::driver::GpuBuffer<f32>,
        input_staging: &trueno_gpu::driver::GpuBuffer<f32>,
        _layer_idx: usize,
        _layer_weights: &crate::cuda::types::ValidatedLayerWeights,
        hidden_dim: u32,
        _epsilon: f32,
        _skip_debug: bool,
    ) -> Result<(), super::super::GpuError> {
        // Placeholder: copy input_staging → hidden_buf2 (skip FFN, residual only)
        self.stream.memcpy_dtod_sync(
            hidden_buf2.as_ptr(),
            input_staging.as_ptr(),
            hidden_dim as usize * std::mem::size_of::<f32>(),
        )?;
        Ok(())
    }
}
