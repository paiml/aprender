
/// CUDA kernel generator
///
/// Generates PTX assembly for various GPU kernels using trueno-gpu.
/// PMAT-044: sm_target is set from device compute capability at init.
/// All PTX generation uses this target — no hardcoded sm_70.
pub struct CudaKernels {
    /// PTX target matching the device (e.g., "sm_89", "sm_87").
    /// Set once at executor init from `GpuProfile::sm_target`.
    pub sm_target: String,
}
