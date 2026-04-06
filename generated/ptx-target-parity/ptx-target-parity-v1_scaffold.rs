/// Contract: PTX target must match device compute capability — no hardcoded SM targets in runtime kernel generation v1.0.0
/// Paper: PMAT-044: Batched decode state corruption from PTX JIT error 700
/// Paper: trueno-gpu Kernel trait (src/kernels/mod.rs) — emit_ptx_for_target()
/// Paper: realizar CudaKernels (src/cuda/kernel_generator.rs) — sm_target field
/// Paper: realizar GpuProfile (src/cuda/gpu_profile.rs) — sm_target from compute_capability()
/// Paper: CUDA PTX ISA — .target directive must be <= device SM version for JIT compilation
pub trait KernelContract {
    /// cuModuleLoadDataEx(ptx, target=device_sm) returns CUDA_SUCCESS
    /// Domain: All PTX modules loaded during model serving
    /// INVARIANT: Error 700 (CUDA_ERROR_INVALID_SOURCE) must never occur at runtime
    /// INVARIANT: Error 222 (CUDA_ERROR_INVALID_PTX) must never occur at runtime
    /// INVARIANT: PTX JIT failure corrupts CUDA context — all subsequent requests fail silently
    /// INVARIANT (Target parity): for all kernel K loaded at runtime: K.ptx_target == executor.gpu_profile.sm_target
    /// INVARIANT (No hardcoded emit_ptx in executor): grep -c 'emit_ptx()' src/cuda/executor/**/*.rs == 0
    /// INVARIANT (CudaKernels constructed with device target): CudaKernels::with_target(gpu_profile.sm_target) at executor init
    /// INVARIANT (JIT success for all kernels): for all K: compile_ptx(K.ptx) == Ok(_)
    fn jit_compilation_success(&self, input: &[f32], output: &mut [f32]);
    /// count(emit_ptx() calls in executor/) == 0
    /// Domain: All .rs files in src/cuda/executor/
    /// INVARIANT: All kernel PTX uses emit_ptx_for_target(sm_target) or generate_ptx(kernel_type)
    /// INVARIANT: generate_ptx() reads sm_target from CudaKernels struct, never hardcodes
    /// INVARIANT: Raw PTX string literals may use sm_70 only for basic instructions (no SM-specific features)
    /// INVARIANT (Target parity): for all kernel K loaded at runtime: K.ptx_target == executor.gpu_profile.sm_target
    /// INVARIANT (No hardcoded emit_ptx in executor): grep -c 'emit_ptx()' src/cuda/executor/**/*.rs == 0
    /// INVARIANT (CudaKernels constructed with device target): CudaKernels::with_target(gpu_profile.sm_target) at executor init
    /// INVARIANT (JIT success for all kernels): for all K: compile_ptx(K.ptx) == Ok(_)
    fn no_hardcoded_targets(&self, input: &[f32], output: &mut [f32]);
    /// ptx_target == device_compute_capability
    /// Domain: ptx_target in {sm_70, sm_75, sm_80, sm_86, sm_87, sm_89, sm_90, ...}, device_cc from cuDeviceGetAttribute
    /// Codomain: Boolean
    /// INVARIANT: Every PTX module loaded at runtime has .target matching the device
    /// INVARIANT: CudaKernels.sm_target is set from GpuProfile.sm_target at executor init
    /// INVARIANT: GpuProfile.sm_target is set from context.compute_capability() at executor init
    /// INVARIANT: No runtime PTX generation path calls emit_ptx() (hardcoded sm_70)
    /// INVARIANT (Target parity): for all kernel K loaded at runtime: K.ptx_target == executor.gpu_profile.sm_target
    /// INVARIANT (No hardcoded emit_ptx in executor): grep -c 'emit_ptx()' src/cuda/executor/**/*.rs == 0
    /// INVARIANT (CudaKernels constructed with device target): CudaKernels::with_target(gpu_profile.sm_target) at executor init
    /// INVARIANT (JIT success for all kernels): for all K: compile_ptx(K.ptx) == Ok(_)
    fn target_parity(&self, input: &[f32], output: &mut [f32]);
}
