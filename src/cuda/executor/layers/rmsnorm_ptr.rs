
impl CudaExecutor {

    /// PAR-023: RMSNorm using raw device pointer for gamma
    pub(crate) fn rmsnorm_gpu_ptr(
        &mut self,
        input: &GpuBuffer<f32>,
        gamma_ptr: u64, // CUdeviceptr
        gamma_len: usize,
        hidden_dim: u32,
        epsilon: f32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        if gamma_ptr == 0 {
            return Err(GpuError::InvalidLaunchConfig(
                "null gamma pointer in rmsnorm_gpu_ptr".to_string(),
            ));
        }
        // Create temporary non-owning buffer wrapper
        // SAFETY: gamma_ptr points to valid GPU memory owned by rmsnorm_cache
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let gamma = unsafe { GpuBuffer::from_raw_parts(gamma_ptr, gamma_len) };

        let result = self.rmsnorm_gpu(input, &gamma, hidden_dim, epsilon)?;

        // Prevent Drop from freeing the borrowed memory
        std::mem::forget(gamma);

        Ok(result)
    }

    /// PAR-044: RMSNorm using raw pointer into existing output buffer
    pub(crate) fn rmsnorm_ptr_into(
        &mut self,
        input: &GpuBuffer<f32>,
        gamma_ptr: u64,
        gamma_len: usize,
        output: &GpuBuffer<f32>,
        hidden_dim: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        if gamma_ptr == 0 {
            return Err(GpuError::InvalidLaunchConfig(
                "null gamma pointer in rmsnorm_ptr_into".to_string(),
            ));
        }
        // SAFETY: Memory safety ensured by bounds checking and alignment
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let gamma = unsafe { GpuBuffer::from_raw_parts(gamma_ptr, gamma_len) };
        self.rmsnorm_into(input, &gamma, output, hidden_dim, epsilon)?;
        std::mem::forget(gamma);
        Ok(())
    }
}

include!("batched_forward.rs");
include!("par-121.rs");
include!("transformer_layer_batched.rs");
