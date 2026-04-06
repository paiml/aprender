//! CUDA module loading, function lookup, and kernel launch tests

use super::*;

#[test]
fn test_module_from_ptx() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");

    let ptx = r#".version 8.0
.target sm_80
.address_size 64

.visible .entry noop() {
    ret;
}
"#;

    let module = CudaModule::from_ptx(&ctx, ptx).expect("Module from_ptx MUST succeed");
    assert!(!module.raw().is_null());
}

#[test]
fn test_module_get_function() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");

    let ptx = r#".version 8.0
.target sm_80
.address_size 64

.visible .entry test_func() {
    ret;
}
"#;

    let mut module = CudaModule::from_ptx(&ctx, ptx).expect("Module MUST succeed");
    let func = module.get_function("test_func").expect("get_function MUST succeed");
    assert!(!func.is_null());
}

#[test]
fn test_module_has_function() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");

    let ptx = r#".version 8.0
.target sm_80
.address_size 64

.visible .entry existing_func() {
    ret;
}
"#;

    let mut module = CudaModule::from_ptx(&ctx, ptx).expect("Module MUST succeed");
    assert!(module.has_function("existing_func"));
    assert!(!module.has_function("nonexistent_func"));
}

#[test]
fn test_module_launch_noop_kernel() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let stream = CudaStream::new(&ctx).expect("Stream creation MUST succeed");

    let ptx = r#".version 8.0
.target sm_80
.address_size 64

.visible .entry noop() {
    ret;
}
"#;

    let mut module = CudaModule::from_ptx(&ctx, ptx).expect("Module MUST succeed");
    let config = LaunchConfig::linear(1, 1);
    let mut args: [*mut c_void; 0] = [];

    unsafe {
        stream
            .launch_kernel(&mut module, "noop", &config, &mut args)
            .expect("Kernel launch MUST succeed");
    }

    stream.synchronize().expect("Sync MUST succeed");
}

#[test]
fn test_module_cached_functions() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");

    let ptx = r#".version 8.0
.target sm_80
.address_size 64

.visible .entry func_a() {
    ret;
}

.visible .entry func_b() {
    ret;
}
"#;

    let mut module = CudaModule::from_ptx(&ctx, ptx).expect("Module MUST succeed");

    // Look up both functions
    module.get_function("func_a").expect("func_a MUST exist");
    module.get_function("func_b").expect("func_b MUST exist");

    // Test cached_functions method
    let cached = module.cached_functions();
    assert_eq!(cached.len(), 2);
    assert!(cached.contains(&"func_a"));
    assert!(cached.contains(&"func_b"));
}

#[test]
fn test_module_get_function_cached() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");

    let ptx = r#".version 8.0
.target sm_80
.address_size 64

.visible .entry cached_test() {
    ret;
}
"#;

    let mut module = CudaModule::from_ptx(&ctx, ptx).expect("Module MUST succeed");

    // First lookup
    let func1 = module.get_function("cached_test").expect("First lookup MUST succeed");

    // Second lookup (from cache)
    let func2 = module.get_function("cached_test").expect("Second lookup MUST succeed");

    // Should return same function handle
    assert_eq!(func1, func2);
}

#[test]
fn test_module_invalid_ptx_error() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");

    // Invalid PTX should fail
    let invalid_ptx = "this is not valid PTX";
    let result = CudaModule::from_ptx(&ctx, invalid_ptx);
    assert!(result.is_err());
}

#[test]
fn test_module_function_not_found() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");

    let ptx = r#".version 8.0
.target sm_80
.address_size 64

.visible .entry existing() {
    ret;
}
"#;

    let mut module = CudaModule::from_ptx(&ctx, ptx).expect("Module MUST succeed");

    // Non-existent function should fail
    let result = module.get_function("nonexistent_function");
    assert!(result.is_err());
}

#[test]
fn test_module_drop_multiple() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");

    let ptx = r#".version 8.0
.target sm_80
.address_size 64

.visible .entry drop_test() {
    ret;
}
"#;

    // Create and drop multiple modules
    for _ in 0..5 {
        let module = CudaModule::from_ptx(&ctx, ptx).expect("Module MUST succeed");
        assert!(!module.raw().is_null());
        // Module dropped here
    }
}

/// trueno#207/#231: Verify real backward kernels compile on sm_89 via linker path.
/// BatchedRmsNormBackwardKernel uses mul_wide_u32_reg (PtxOp::MulWide) which
/// previously caused CUDA_ERROR_ILLEGAL_ADDRESS (trueno#226).
#[test]
fn test_module_backward_kernel_ptx_sm89() {
    use crate::kernels::backward::BatchedRmsNormBackwardKernel;
    use crate::kernels::Kernel;

    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let kernel = BatchedRmsNormBackwardKernel::new(64, 128, 1e-6);
    let ptx = kernel.emit_ptx();

    // Verify the PTX compiles and loads via the linker path on the real GPU
    let mut module =
        CudaModule::from_ptx(&ctx, &ptx).expect("Backward kernel PTX MUST compile on sm_89");
    assert!(!module.raw().is_null());

    // Verify the entry point is accessible
    let func = module.get_function("batched_rms_norm_backward");
    assert!(func.is_ok(), "Entry point MUST be found: {:?}", func.err());
}
