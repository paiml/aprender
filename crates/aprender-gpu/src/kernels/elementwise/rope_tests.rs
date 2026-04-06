//! RoPE Kernel Tests (PMAT-018: Coverage Killer Remediation)
//!
//! Tests for all RoPE kernel variants to achieve coverage.

use super::rope::{
    BatchedRopeKernel, PreciseRopeIndirectKernel, PreciseRopeKernel, RopeIndirectKernel,
    RopeKernel, RopeNeoxIndirectKernel, RopeNeoxKernel,
};
use crate::kernels::Kernel;

// ============================================================================
// RopeKernel Tests
// ============================================================================

#[test]
fn test_rope_kernel_basic() {
    let kernel = RopeKernel::new(8, 64, 10000.0);

    assert_eq!(kernel.name(), "rope");
    assert_eq!(kernel.num_heads, 8);
    assert_eq!(kernel.head_dim, 64);
    assert_eq!(kernel.theta, 10000.0);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
    assert!(ptx.contains(".entry rope"));
    assert!(ptx.contains("x_ptr"));
}

#[test]
fn test_rope_kernel_llama_config() {
    // Llama-style: 32 heads, 128 head_dim
    let kernel = RopeKernel::new(32, 128, 10000.0);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

#[test]
fn test_rope_kernel_small() {
    let kernel = RopeKernel::new(4, 32, 10000.0);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

// ============================================================================
// RopeIndirectKernel Tests
// ============================================================================

#[test]
fn test_rope_indirect_kernel() {
    let kernel = RopeIndirectKernel::new(8, 64, 10000.0);

    assert!(kernel.name().contains("indirect"));

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
    // Should have indirect position parameter
    assert!(ptx.contains("pos_ptr"));
}

#[test]
fn test_rope_indirect_kernel_large() {
    let kernel = RopeIndirectKernel::new(32, 128, 10000.0);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

// ============================================================================
// RopeNeoxKernel Tests (split halves style)
// ============================================================================

#[test]
fn test_rope_neox_kernel() {
    let kernel = RopeNeoxKernel::new(8, 64, 10000.0);

    assert!(kernel.name().contains("neox"));

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

#[test]
fn test_rope_neox_kernel_gpt_config() {
    // GPT-NeoX style config
    let kernel = RopeNeoxKernel::new(16, 96, 10000.0);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

// ============================================================================
// RopeNeoxIndirectKernel Tests
// ============================================================================

#[test]
fn test_rope_neox_indirect_kernel() {
    let kernel = RopeNeoxIndirectKernel::new(8, 64, 10000.0);

    assert!(kernel.name().contains("indirect"));

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
    assert!(ptx.contains("pos_ptr"));
}

// ============================================================================
// BatchedRopeKernel Tests
// ============================================================================

#[test]
fn test_batched_rope_kernel() {
    let kernel = BatchedRopeKernel::new(8, 64, 4, 10000.0);

    assert!(kernel.name().contains("batched"));
    assert_eq!(kernel.batch_size, 4);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

#[test]
fn test_batched_rope_kernel_large_batch() {
    let kernel = BatchedRopeKernel::new(16, 128, 32, 10000.0);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

// ============================================================================
// PreciseRopeKernel Tests (high precision for theta=1M)
// ============================================================================

#[test]
fn test_precise_rope_kernel() {
    // Qwen2.5 style: theta=1_000_000
    let kernel = PreciseRopeKernel::new(8, 64, 1_000_000.0);

    assert!(kernel.name().contains("precise"));

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

#[test]
fn test_precise_rope_kernel_qwen_config() {
    // Full Qwen2.5 config
    let kernel = PreciseRopeKernel::new(28, 128, 1_000_000.0);

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
}

// ============================================================================
// PreciseRopeIndirectKernel Tests
// ============================================================================

#[test]
fn test_precise_rope_indirect_kernel() {
    let kernel = PreciseRopeIndirectKernel::new(8, 64, 1_000_000.0);

    assert!(kernel.name().contains("indirect"));

    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
    assert!(ptx.contains("pos_ptr"));
}

// ============================================================================
// Comprehensive Configuration Matrix
// ============================================================================

#[test]
fn test_all_rope_kernel_variants() {
    let configs = vec![(8, 64, 10000.0), (16, 128, 10000.0), (32, 64, 1_000_000.0)];

    for (num_heads, head_dim, theta) in configs {
        // Standard RoPE
        let k1 = RopeKernel::new(num_heads, head_dim, theta);
        assert!(k1.emit_ptx().contains(".version"));

        // Indirect RoPE
        let k2 = RopeIndirectKernel::new(num_heads, head_dim, theta);
        assert!(k2.emit_ptx().contains(".version"));

        // NEOX RoPE
        let k3 = RopeNeoxKernel::new(num_heads, head_dim, theta);
        assert!(k3.emit_ptx().contains(".version"));

        // NEOX Indirect RoPE
        let k4 = RopeNeoxIndirectKernel::new(num_heads, head_dim, theta);
        assert!(k4.emit_ptx().contains(".version"));

        // Batched RoPE
        let k5 = BatchedRopeKernel::new(num_heads, head_dim, 4, theta);
        assert!(k5.emit_ptx().contains(".version"));

        // Precise RoPE
        let k6 = PreciseRopeKernel::new(num_heads, head_dim, theta);
        assert!(k6.emit_ptx().contains(".version"));

        // Precise Indirect RoPE
        let k7 = PreciseRopeIndirectKernel::new(num_heads, head_dim, theta);
        assert!(k7.emit_ptx().contains(".version"));
    }
}

#[test]
fn test_rope_theta_values() {
    // Standard theta
    let k1 = RopeKernel::new(8, 64, 10000.0);
    assert_eq!(k1.theta, 10000.0);

    // Extended theta for longer context
    let k2 = RopeKernel::new(8, 64, 500000.0);
    assert_eq!(k2.theta, 500000.0);

    // Qwen2.5 theta
    let k3 = RopeKernel::new(8, 64, 1_000_000.0);
    assert_eq!(k3.theta, 1_000_000.0);
}
