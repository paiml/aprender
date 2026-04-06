use super::*;

// ============ ReluKernel Tests ============

#[test]
fn test_relu_kernel_name() {
    let kernel = ReluKernel::new(2048);
    assert_eq!(kernel.name(), "relu");
}

#[test]
fn test_relu_ptx_generation() {
    let kernel = ReluKernel::new(2048);
    let ptx = kernel.emit_ptx();

    // Verify entry point
    assert!(ptx.contains(".entry relu"));

    // Verify max operation for ReLU
    assert!(ptx.contains("max.f32"));
}

#[test]
fn test_relu_kernel_debug() {
    let kernel = ReluKernel::new(1024);
    let debug_str = format!("{:?}", kernel);
    assert!(debug_str.contains("ReluKernel"));
    assert!(debug_str.contains("1024"));
}

#[test]
fn test_relu_kernel_clone() {
    let kernel = ReluKernel::new(512);
    let cloned = kernel.clone();
    assert_eq!(cloned.n, 512);
}

#[test]
fn test_relu_kernel_ptx_contains_bounds_check() {
    let kernel = ReluKernel::new(100);
    let ptx = kernel.emit_ptx();
    // Verify bounds check with setp
    assert!(ptx.contains("setp.lt.u32"));
    // Verify branch instruction
    assert!(ptx.contains("@!"));
}

#[test]
fn test_relu_kernel_edge_case_n_zero() {
    let kernel = ReluKernel::new(0);
    let ptx = kernel.emit_ptx();
    // Should still generate valid PTX
    assert!(ptx.contains(".entry relu"));
}

#[test]
fn test_relu_kernel_edge_case_n_one() {
    let kernel = ReluKernel::new(1);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry relu"));
    assert!(ptx.contains("max.f32"));
}

#[test]
fn test_relu_kernel_large_n() {
    let kernel = ReluKernel::new(u32::MAX);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry relu"));
}

// ============ SiluKernel Tests ============

#[test]
fn test_silu_kernel_name() {
    let kernel = SiluKernel::new(2048);
    assert_eq!(kernel.name(), "silu");
}

#[test]
fn test_silu_ptx_generation() {
    let kernel = SiluKernel::new(2048);
    let ptx = kernel.emit_ptx();

    // Verify entry point
    assert!(ptx.contains(".entry silu"));

    // Verify sigmoid computation (exp and division)
    assert!(ptx.contains("ex2.approx.f32"));
    assert!(ptx.contains("div.rn.f32"));

    // Verify final multiply (x * sigmoid)
    assert!(ptx.contains("mul.f32"));
}

#[test]
fn test_silu_kernel_debug() {
    let kernel = SiluKernel::new(4096);
    let debug_str = format!("{:?}", kernel);
    assert!(debug_str.contains("SiluKernel"));
    assert!(debug_str.contains("4096"));
}

#[test]
fn test_silu_kernel_clone() {
    let kernel = SiluKernel::new(256);
    let cloned = kernel.clone();
    assert_eq!(cloned.n, 256);
}

#[test]
fn test_silu_kernel_contains_log2e_constant() {
    let kernel = SiluKernel::new(1000);
    let ptx = kernel.emit_ptx();
    // Verify we use ex2 for exp approximation
    assert!(ptx.contains("ex2.approx.f32"));
}

#[test]
fn test_silu_kernel_ptx_structure() {
    let kernel = SiluKernel::new(512);
    let ptx = kernel.emit_ptx();
    // Verify parameter declarations
    assert!(ptx.contains(".param .u64 input_ptr"));
    assert!(ptx.contains(".param .u64 output_ptr"));
    assert!(ptx.contains(".param .u32 n"));
    // Verify exit label
    assert!(ptx.contains("exit:"));
}

// ============ GeluKernel Tests ============

#[test]
fn test_gelu_kernel_name() {
    let kernel = GeluKernel::new(2048);
    assert_eq!(kernel.name(), "gelu");
}

#[test]
fn test_gelu_ptx_generation() {
    let kernel = GeluKernel::new(2048);
    let ptx = kernel.emit_ptx();

    // Verify entry point
    assert!(ptx.contains(".entry gelu"));

    // Verify tanh computation via sigmoid (exp)
    assert!(ptx.contains("ex2.approx.f32"));

    // Verify x^3 computation (two multiplies)
    assert!(ptx.contains("mul.f32"));
}

#[test]
fn test_gelu_kernel_debug() {
    let kernel = GeluKernel::new(8192);
    let debug_str = format!("{:?}", kernel);
    assert!(debug_str.contains("GeluKernel"));
    assert!(debug_str.contains("8192"));
}

#[test]
fn test_gelu_kernel_clone() {
    let kernel = GeluKernel::new(128);
    let cloned = kernel.clone();
    assert_eq!(cloned.n, 128);
}

#[test]
fn test_gelu_kernel_ptx_contains_tanh_approximation() {
    let kernel = GeluKernel::new(1000);
    let ptx = kernel.emit_ptx();
    // GELU uses tanh via 2*sigmoid - 1
    assert!(ptx.contains("div.rn.f32")); // Division for sigmoid
    assert!(ptx.contains("sub.f32")); // Subtraction for tanh
}

#[test]
fn test_gelu_kernel_edge_case_n_zero() {
    let kernel = GeluKernel::new(0);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry gelu"));
}

// ============ ElementwiseMulKernel Tests ============

#[test]
fn test_elementwise_mul_kernel_name() {
    let kernel = ElementwiseMulKernel::new(2048);
    assert_eq!(kernel.name(), "elementwise_mul");
}

#[test]
fn test_elementwise_mul_ptx_generation() {
    let kernel = ElementwiseMulKernel::new(2048);
    let ptx = kernel.emit_ptx();

    // Verify entry point
    assert!(ptx.contains(".entry elementwise_mul"));

    // Verify two input parameters
    assert!(ptx.contains(".param .u64 input1_ptr"));
    assert!(ptx.contains(".param .u64 input2_ptr"));
    assert!(ptx.contains(".param .u64 output_ptr"));
    assert!(ptx.contains(".param .u32 n"));

    // Verify multiply operation
    assert!(ptx.contains("mul.f32"));
}

#[test]
fn test_elementwise_mul_kernel_debug() {
    let kernel = ElementwiseMulKernel::new(1024);
    let debug_str = format!("{:?}", kernel);
    assert!(debug_str.contains("ElementwiseMulKernel"));
    assert!(debug_str.contains("1024"));
}

#[test]
fn test_elementwise_mul_kernel_clone() {
    let kernel = ElementwiseMulKernel::new(64);
    let cloned = kernel.clone();
    assert_eq!(cloned.n, 64);
}

#[test]
fn test_elementwise_mul_kernel_ptx_contains_bounds_check() {
    let kernel = ElementwiseMulKernel::new(500);
    let ptx = kernel.emit_ptx();
    // Verify bounds check
    assert!(ptx.contains("setp.lt.u32"));
}

#[test]
fn test_elementwise_mul_kernel_ptx_loads_two_inputs() {
    let kernel = ElementwiseMulKernel::new(100);
    let ptx = kernel.emit_ptx();
    // Verify two global loads
    let load_count = ptx.matches("ld.global.f32").count();
    assert_eq!(load_count, 2, "Should have exactly 2 global loads");
}

#[test]
fn test_elementwise_mul_kernel_edge_case_n_one() {
    let kernel = ElementwiseMulKernel::new(1);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry elementwise_mul"));
    assert!(ptx.contains("mul.f32"));
}

#[test]
fn test_elementwise_mul_kernel_large_n() {
    let kernel = ElementwiseMulKernel::new(1_000_000);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry elementwise_mul"));
}

// ============ ScaleKernel Tests ============

#[test]
fn test_scale_kernel_name() {
    let kernel = ScaleKernel::new(2048);
    assert_eq!(kernel.name(), "scale");
}

#[test]
fn test_scale_ptx_generation() {
    let kernel = ScaleKernel::new(2048);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".entry scale"));
    assert!(ptx.contains(".param .f32 scale"));
    assert!(ptx.contains("mul.f32"));
}

#[test]
fn test_scale_kernel_debug() {
    let kernel = ScaleKernel::new(512);
    let debug_str = format!("{:?}", kernel);
    assert!(debug_str.contains("ScaleKernel"));
    assert!(debug_str.contains("512"));
}

#[test]
fn test_scale_kernel_clone() {
    let kernel = ScaleKernel::new(32);
    let cloned = kernel.clone();
    assert_eq!(cloned.n, 32);
}

#[test]
fn test_scale_kernel_ptx_structure() {
    let kernel = ScaleKernel::new(256);
    let ptx = kernel.emit_ptx();
    // Verify parameter order
    assert!(ptx.contains(".param .u64 input_ptr"));
    assert!(ptx.contains(".param .u64 output_ptr"));
    assert!(ptx.contains(".param .f32 scale"));
    assert!(ptx.contains(".param .u32 n"));
}

#[test]
fn test_scale_kernel_edge_case_n_zero() {
    let kernel = ScaleKernel::new(0);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry scale"));
}

#[test]
fn test_scale_kernel_ptx_uses_f32_scale_param() {
    let kernel = ScaleKernel::new(100);
    let ptx = kernel.emit_ptx();
    // Verify f32 scale parameter is loaded
    assert!(ptx.contains(".param .f32 scale"));
    // And used in multiplication
    assert!(ptx.contains("mul.f32"));
}
