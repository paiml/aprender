impl CudaKernels {
    /// Create a new CUDA kernel generator with the device's SM target.
    ///
    /// PMAT-044: sm_target MUST match the device (e.g., "sm_89").
    /// All generated PTX uses this target — no hardcoded sm_70.
    #[must_use]
    pub fn with_target(sm_target: String) -> Self {
        Self { sm_target }
    }

    /// Create a new CUDA kernel generator (legacy, defaults to sm_70).
    ///
    /// Only for tests that don't run on a device. Production code
    /// MUST use `with_target()`.
    #[must_use]
    pub fn new() -> Self {
        Self { sm_target: "sm_70".to_string() }
    }

    /// Generate PTX source for the specified kernel
    ///
    /// Returns PTX assembly that can be loaded by the CUDA driver API.
    /// Delegates to category-specific helpers to keep cyclomatic complexity low.
    /// PMAT-044: All PTX targets self.sm_target (device compute capability).
    #[must_use]
    pub fn generate_ptx(&self, kernel_type: &KernelType) -> String {
        contract_pre_no_hardcoded_targets!();
        let target = &self.sm_target;
        Self::generate_gemm_ptx(kernel_type, target)
            .or_else(|| Self::generate_gemv_ptx(kernel_type, target))
            .or_else(|| Self::generate_q4k_gemv_ptx(kernel_type, target))
            .or_else(|| Self::generate_attention_ptx(kernel_type, target))
            .or_else(|| Self::generate_norm_rope_ptx(kernel_type, target))
            .or_else(|| Self::generate_activation_misc_ptx(kernel_type, target))
            .unwrap_or_default()
    }
}

include!("kernels_generate_gemm_cuda.rs");
