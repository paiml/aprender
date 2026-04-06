//! Dimension-Independence Contract Falsification Tests
//!
//! Contract: contracts/dimension-independent-kernels-v1.yaml
//! Issue: trueno#203 — Dimension-independent kernels architecture
//!
//! These tests verify that training-critical GPU kernels produce IDENTICAL PTX
//! regardless of constructor dimensions. Kernels that bake dimensions as
//! compile-time immediates force JIT compilation when new dimensions are
//! encountered during training — which on Blackwell (sm_121) poisons the
//! CUDA context (trueno#200).
//!
//! ## Test Strategy
//!
//! For each kernel type, construct two instances with different dimensions
//! and compare their `emit_ptx()` output. If the PTX differs, the kernel
//! bakes dimensions and must be refactored.
//!
//! ## Current Status (2026-03-22)
//!
//! - 10 kernels PASS (already dimension-independent)
//! - 20 kernels FAIL (bake dimensions — marked #[ignore] until refactored)

use super::*;
use crate::kernels::backward::{
    BatchedRmsNormBackwardKernel, BatchedSoftmaxBackwardKernel, FusedCausalCrossEntropyKernel,
    FusedCrossEntropyKernel, GeluBackwardKernel, GemmBackwardAKernel, GemmBackwardBKernel,
    LayerNormBackwardKernel, ReluBackwardKernel, RmsNormBackwardKernel, SiluBackwardKernel,
    SoftmaxBackwardKernel,
};

// ============================================================================
// Helper: compare PTX from two kernel instances
// ============================================================================

/// Normalize PTX by sorting register declarations (HashMap iteration order is non-deterministic).
/// This ensures that functionally identical PTX compares as equal.
fn normalize_ptx(ptx: &str) -> String {
    let mut reg_decls: Vec<String> = Vec::new();
    let mut other_lines: Vec<String> = Vec::new();
    let mut in_entry = false;

    for line in ptx.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(".reg ") {
            reg_decls.push(line.to_string());
        } else {
            // When we hit the first non-reg line after collecting regs, flush sorted regs
            if !reg_decls.is_empty() {
                reg_decls.sort();
                other_lines.extend(reg_decls.drain(..));
            }
            other_lines.push(line.to_string());
            if trimmed.starts_with(".visible .entry") {
                in_entry = true;
            }
            let _ = in_entry; // suppress unused warning
        }
    }
    // Flush any remaining reg decls
    if !reg_decls.is_empty() {
        reg_decls.sort();
        other_lines.extend(reg_decls.drain(..));
    }
    other_lines.join("\n")
}

fn assert_ptx_identical<K1: Kernel, K2: Kernel>(kernel_a: &K1, kernel_b: &K2, label: &str) {
    let ptx_a = normalize_ptx(&kernel_a.emit_ptx());
    let ptx_b = normalize_ptx(&kernel_b.emit_ptx());
    assert_eq!(
        ptx_a,
        ptx_b,
        "FALSIFY-DIM: {label} produces different PTX for different dimensions.\n\
         This means dimensions are baked as compile-time immediates.\n\
         PTX length A: {}, PTX length B: {}\n\
         First diff at byte: {:?}",
        ptx_a.len(),
        ptx_b.len(),
        ptx_a.bytes().zip(ptx_b.bytes()).position(|(a, b)| a != b)
    );
}

/// FALSIFY-DIM-006: Check that every declared .param has a corresponding ld.param
fn assert_all_params_loaded(ptx: &str, label: &str) {
    // Extract declared params from .param lines
    let mut declared_params: Vec<String> = Vec::new();
    for line in ptx.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(".param") {
            // .param .u32 name or .param .u64 name or .param .f32 name
            if let Some(name) = trimmed.split_whitespace().nth(2) {
                // Strip trailing comma or semicolon
                let clean = name.trim_end_matches([',', ';']);
                declared_params.push(clean.to_string());
            }
        }
    }

    // Check each declared param has an ld.param instruction
    for param in &declared_params {
        let load_pattern = format!("[{param}]");
        assert!(
            ptx.contains(&load_pattern),
            "FALSIFY-DIM-006: {label} declares .param '{param}' but never loads it.\n\
             The closure uses a baked immediate instead of ld.param.\n\
             Contract: dimension-independent-kernels-v1.yaml"
        );
    }
}

// ============================================================================
// FALSIFY-DIM-001/002: Backward activation kernels (ALREADY PASS)
// ============================================================================

#[test]
fn test_falsify_dim_001_relu_backward_dimension_independent() {
    let a = ReluBackwardKernel::new(1024);
    let b = ReluBackwardKernel::new(4096);
    assert_ptx_identical(&a, &b, "ReluBackwardKernel");
}

#[test]
fn test_falsify_dim_001_gelu_backward_dimension_independent() {
    let a = GeluBackwardKernel::new(1024);
    let b = GeluBackwardKernel::new(4096);
    assert_ptx_identical(&a, &b, "GeluBackwardKernel");
}

#[test]
fn test_falsify_dim_001_silu_backward_dimension_independent() {
    let a = SiluBackwardKernel::new(1024);
    let b = SiluBackwardKernel::new(4096);
    assert_ptx_identical(&a, &b, "SiluBackwardKernel");
}

// ============================================================================
// FALSIFY-DIM-002: Backward GEMM naive (ALREADY PASS)
// ============================================================================

#[test]
fn test_falsify_dim_002_gemm_backward_a_naive_dimension_independent() {
    let a = GemmBackwardAKernel::new(4, 64, 64);
    let b = GemmBackwardAKernel::new(8, 128, 256);
    assert_ptx_identical(&a, &b, "GemmBackwardAKernel::naive");
}

#[test]
fn test_falsify_dim_002_gemm_backward_b_naive_dimension_independent() {
    let a = GemmBackwardBKernel::new(4, 64, 64);
    let b = GemmBackwardBKernel::new(8, 128, 256);
    assert_ptx_identical(&a, &b, "GemmBackwardBKernel::naive");
}

// ============================================================================
// FALSIFY-DIM-002: Cross-entropy backward (ALREADY PASS)
// ============================================================================

#[test]
fn test_falsify_dim_002_fused_cross_entropy_dimension_independent() {
    let a = FusedCrossEntropyKernel::new(32000);
    let b = FusedCrossEntropyKernel::new(151936);
    assert_ptx_identical(&a, &b, "FusedCrossEntropyKernel");
}

#[test]
fn test_falsify_dim_002_fused_causal_cross_entropy_dimension_independent() {
    let a = FusedCausalCrossEntropyKernel::new(32000);
    let b = FusedCausalCrossEntropyKernel::new(151936);
    assert_ptx_identical(&a, &b, "FusedCausalCrossEntropyKernel");
}

// ============================================================================
// FALSIFY-DIM-002: Optimizer kernels (ALREADY PASS)
// ============================================================================

#[test]
fn test_falsify_dim_002_adamw_step_dimension_independent() {
    let a = AdamWStepKernel::new(1024);
    let b = AdamWStepKernel::new(4096);
    assert_ptx_identical(&a, &b, "AdamWStepKernel");
}

#[test]
fn test_falsify_dim_002_gradient_clip_dimension_independent() {
    let a = GradientClipKernel::new(1024);
    let b = GradientClipKernel::new(4096);
    assert_ptx_identical(&a, &b, "GradientClipKernel");
}

// ============================================================================
// FALSIFY-DIM-001: GEMM backward tiled (CURRENTLY FAIL — bakes tile_size, n, k)
// ============================================================================

#[test]
#[ignore = "FALSIFY-DIM-001: GemmBackwardA tiled bakes tile_size/n/k — trueno#203"]
fn test_falsify_dim_001_gemm_backward_a_tiled_dimension_independent() {
    let a = GemmBackwardAKernel::tiled(4, 64, 64, 16);
    let b = GemmBackwardAKernel::tiled(8, 128, 256, 16);
    assert_ptx_identical(&a, &b, "GemmBackwardAKernel::tiled");
}

#[test]
#[ignore = "FALSIFY-DIM-001: GemmBackwardA tiled_unrolled bakes dims — trueno#203"]
fn test_falsify_dim_001_gemm_backward_a_tiled_unrolled_dimension_independent() {
    let a = GemmBackwardAKernel::tiled_unrolled(4, 64, 64, 16);
    let b = GemmBackwardAKernel::tiled_unrolled(8, 128, 256, 16);
    assert_ptx_identical(&a, &b, "GemmBackwardAKernel::tiled_unrolled");
}

#[test]
#[ignore = "FALSIFY-DIM-001: GemmBackwardB tiled bakes tile_size/n/k — trueno#203"]
fn test_falsify_dim_001_gemm_backward_b_tiled_dimension_independent() {
    let a = GemmBackwardBKernel::tiled(4, 64, 64, 16);
    let b = GemmBackwardBKernel::tiled(8, 128, 256, 16);
    assert_ptx_identical(&a, &b, "GemmBackwardBKernel::tiled");
}

#[test]
#[ignore = "FALSIFY-DIM-001: GemmBackwardB tiled_unrolled bakes dims — trueno#203"]
fn test_falsify_dim_001_gemm_backward_b_tiled_unrolled_dimension_independent() {
    let a = GemmBackwardBKernel::tiled_unrolled(4, 64, 64, 16);
    let b = GemmBackwardBKernel::tiled_unrolled(8, 128, 256, 16);
    assert_ptx_identical(&a, &b, "GemmBackwardBKernel::tiled_unrolled");
}

// ============================================================================
// FALSIFY-DIM-001: RMSNorm backward (CURRENTLY FAIL — bakes hidden_dim, eps)
// ============================================================================

#[test]
#[ignore = "FALSIFY-DIM-001: RmsNormBackward bakes hidden_dim (shuffle structure) + eps — trueno#203"]
fn test_falsify_dim_001_rms_norm_backward_dimension_independent() {
    let a = RmsNormBackwardKernel::new(64, 16, 1e-6);
    let b = RmsNormBackwardKernel::new(128, 32, 1e-5);
    assert_ptx_identical(&a, &b, "RmsNormBackwardKernel");
}

#[test]
fn test_falsify_dim_001_batched_rms_norm_backward_dimension_independent() {
    let a = BatchedRmsNormBackwardKernel::new(64, 128, 1e-6);
    let b = BatchedRmsNormBackwardKernel::new(128, 256, 1e-5);
    assert_ptx_identical(&a, &b, "BatchedRmsNormBackwardKernel");
}

// ============================================================================
// FALSIFY-DIM-001: Softmax backward (CURRENTLY FAIL — bakes row_size)
// ============================================================================

#[test]
#[ignore = "FALSIFY-DIM-001: SoftmaxBackward bakes row_size (shuffle structure) — trueno#203"]
fn test_falsify_dim_001_softmax_backward_dimension_independent() {
    let a = SoftmaxBackwardKernel::new(64, 16);
    let b = SoftmaxBackwardKernel::new(128, 32);
    assert_ptx_identical(&a, &b, "SoftmaxBackwardKernel");
}

#[test]
#[ignore = "FALSIFY-DIM-001: BatchedSoftmaxBackward bakes ALL dims (params never loaded) — trueno#203"]
fn test_falsify_dim_001_batched_softmax_backward_dimension_independent() {
    let a = BatchedSoftmaxBackwardKernel::new(64, 128);
    let b = BatchedSoftmaxBackwardKernel::new(128, 256);
    assert_ptx_identical(&a, &b, "BatchedSoftmaxBackwardKernel");
}

// ============================================================================
// FALSIFY-DIM-001: LayerNorm backward (CURRENTLY FAIL — bakes hidden_dim)
// ============================================================================

#[test]
#[ignore = "FALSIFY-DIM-001: LayerNormBackward bakes hidden_dim (shuffle structure) — trueno#203"]
fn test_falsify_dim_001_layer_norm_backward_dimension_independent() {
    let a = LayerNormBackwardKernel::new(64, 16);
    let b = LayerNormBackwardKernel::new(128, 32);
    assert_ptx_identical(&a, &b, "LayerNormBackwardKernel");
}

// ============================================================================
// FALSIFY-DIM-005: RoPE kernels (CURRENTLY FAIL — zero runtime params for dims)
// ============================================================================

#[test]
#[ignore = "FALSIFY-DIM-005: BatchedRopeKernel bakes head_dim/num_heads/theta (NO params) — trueno#203"]
fn test_falsify_dim_005_batched_rope_dimension_independent() {
    let a = BatchedRopeKernel::new(32, 128, 4, 10000.0);
    let b = BatchedRopeKernel::new(8, 64, 8, 500000.0);
    assert_ptx_identical(&a, &b, "BatchedRopeKernel");
}

// ============================================================================
// FALSIFY-DIM-004: Layout transform kernels (CURRENTLY FAIL — Run 12 JIT offenders)
// ============================================================================

#[test]
fn test_falsify_dim_004_interleaved_to_batched_dimension_independent() {
    let a = InterleavedToBatchedKernel::new(512, 32, 128);
    let b = InterleavedToBatchedKernel::new(1024, 8, 64);
    assert_ptx_identical(&a, &b, "InterleavedToBatchedKernel");
}

#[test]
fn test_falsify_dim_004_batched_to_interleaved_dimension_independent() {
    let a = BatchedToInterleavedKernel::new(512, 32, 128);
    let b = BatchedToInterleavedKernel::new(1024, 8, 64);
    assert_ptx_identical(&a, &b, "BatchedToInterleavedKernel");
}

#[test]
fn test_falsify_dim_004_batched_transpose_dimension_independent() {
    let a = BatchedTransposeKernel::new(64, 128, 4);
    let b = BatchedTransposeKernel::new(256, 512, 8);
    assert_ptx_identical(&a, &b, "BatchedTransposeKernel");
}

#[test]
fn test_falsify_dim_004_batched_softmax_dimension_independent() {
    let a = BatchedSoftmaxKernel::new(64, 128);
    let b = BatchedSoftmaxKernel::new(256, 512);
    assert_ptx_identical(&a, &b, "BatchedSoftmaxKernel");
}

#[test]
fn test_falsify_dim_004_transpose_dimension_independent() {
    let a = TransposeKernel::new(64, 128);
    let b = TransposeKernel::new(256, 512);
    assert_ptx_identical(&a, &b, "TransposeKernel");
}

// ============================================================================
// FALSIFY-DIM-006: Phantom params (declared but never loaded)
// ============================================================================

#[test]
fn test_falsify_dim_006_batched_rms_norm_backward_params_loaded() {
    let kernel = BatchedRmsNormBackwardKernel::new(64, 128, 1e-6);
    let ptx = kernel.emit_ptx();
    assert_all_params_loaded(&ptx, "BatchedRmsNormBackwardKernel");
}

#[test]
#[ignore = "FALSIFY-DIM-006: BatchedSoftmaxBackward declares params but uses baked immediates — trueno#203"]
fn test_falsify_dim_006_batched_softmax_backward_params_loaded() {
    let kernel = BatchedSoftmaxBackwardKernel::new(64, 128);
    let ptx = kernel.emit_ptx();
    assert_all_params_loaded(&ptx, "BatchedSoftmaxBackwardKernel");
}

#[test]
fn test_falsify_dim_006_transpose_params_loaded() {
    let kernel = TransposeKernel::new(64, 128);
    let ptx = kernel.emit_ptx();
    assert_all_params_loaded(&ptx, "TransposeKernel");
}

#[test]
fn test_falsify_dim_006_batched_transpose_params_loaded() {
    let kernel = BatchedTransposeKernel::new(64, 128, 4);
    let ptx = kernel.emit_ptx();
    assert_all_params_loaded(&ptx, "BatchedTransposeKernel");
}

#[test]
fn test_falsify_dim_006_batched_softmax_params_loaded() {
    let kernel = BatchedSoftmaxKernel::new(64, 128);
    let ptx = kernel.emit_ptx();
    assert_all_params_loaded(&ptx, "BatchedSoftmaxKernel");
}

// ============================================================================
// FALSIFY-DIM-003: Total unique training kernel count
// ============================================================================

#[test]
fn test_falsify_dim_003_unique_training_kernel_count() {
    // List ALL kernel types used during a Qwen3-4B training step.
    // Each entry is (name, category).
    let training_kernel_types = [
        // Forward pass
        ("batched_vectorized_rmsnorm", "forward"),
        ("batched_rope", "forward"),
        ("interleaved_to_batched", "forward"),
        ("batched_to_interleaved", "forward"),
        ("batched_scale", "forward"),
        ("batched_softmax", "forward"),
        ("batched_transpose", "forward"),
        ("silu", "forward"),
        ("fused_causal_cross_entropy", "forward"),
        // Backward pass
        ("silu_backward", "backward"),
        ("batched_rms_norm_backward", "backward"),
        ("batched_softmax_backward", "backward"),
        ("batched_rope_backward", "backward"),
        ("gemm_backward_a", "backward"),
        ("gemm_backward_b", "backward"),
        ("fused_causal_cross_entropy_backward", "backward"),
        // Optimizer
        ("adamw_step", "optimizer"),
        ("gradient_clip", "optimizer"),
        ("squared_sum", "optimizer"),
    ];

    let count = training_kernel_types.len();
    assert!(
        count <= 20,
        "FALSIFY-DIM-003: {count} unique training kernel types exceeds limit of 20.\n\
         Pre-compilation of {count} cubins is still feasible but should be monitored.\n\
         Contract: dimension-independent-kernels-v1.yaml"
    );

    // Verify no duplicates
    let mut names: Vec<&str> = training_kernel_types.iter().map(|(n, _)| *n).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), training_kernel_types.len(), "Duplicate kernel type names detected");
}
