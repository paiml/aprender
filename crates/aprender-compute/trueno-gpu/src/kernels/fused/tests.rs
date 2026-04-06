use super::*;

#[test]
fn test_fused_gemm_bias_gelu_kernel_builds() {
    let kernel = FusedGemmBiasGeluKernel::new(1500, 1536, 384);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains("fused_gemm_bias_gelu"));
    assert!(ptx.contains(".entry"));
    // Verify GELU constants are present (hex format: 0F{bits:08X})
    // sqrt(2/π) ≈ 0.7978846 -> 0F3F4C422A
    // 0.044715 -> 0F3D372713
    assert!(ptx.contains("0F3F4C422A"), "Missing sqrt(2/π) constant");
    assert!(ptx.contains("0F3D372713"), "Missing 0.044715 constant");
}

#[test]
fn test_fused_qkv_kernel_builds() {
    let kernel = FusedQKVKernel::new(3584, 512);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains("fused_qkv_gemv"));
    assert!(ptx.contains(".entry"));
}

#[test]
fn test_fused_gate_up_kernel_builds() {
    let kernel = FusedGateUpKernel::new(3584, 18944);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains("fused_gate_up_swiglu"));
    assert!(ptx.contains(".entry"));
}

#[test]
fn test_fused_qkv_kernel_name() {
    let kernel = FusedQKVKernel::new(1024, 256);
    assert_eq!(kernel.name(), "fused_qkv_gemv");
}

#[test]
fn test_fused_gate_up_kernel_name() {
    let kernel = FusedGateUpKernel::new(1024, 4096);
    assert_eq!(kernel.name(), "fused_gate_up_swiglu");
}

#[test]
fn test_fused_qkv_kernel_clone() {
    let kernel = FusedQKVKernel::new(1024, 256);
    let cloned = kernel.clone();
    assert_eq!(cloned.hidden_size, kernel.hidden_size);
    assert_eq!(cloned.kv_dim, kernel.kv_dim);
}

#[test]
fn test_fused_gate_up_kernel_clone() {
    let kernel = FusedGateUpKernel::new(1024, 4096);
    let cloned = kernel.clone();
    assert_eq!(cloned.hidden_size, kernel.hidden_size);
    assert_eq!(cloned.intermediate_size, kernel.intermediate_size);
}

#[test]
fn test_fused_qkv_kernel_debug() {
    let kernel = FusedQKVKernel::new(1024, 256);
    let debug = format!("{:?}", kernel);
    assert!(debug.contains("FusedQKVKernel"));
    assert!(debug.contains("1024"));
}

#[test]
fn test_fused_gate_up_kernel_debug() {
    let kernel = FusedGateUpKernel::new(1024, 4096);
    let debug = format!("{:?}", kernel);
    assert!(debug.contains("FusedGateUpKernel"));
    assert!(debug.contains("4096"));
}
