//! PTX-PIXEL-FKR location and regression summary

// PTX tests are in trueno-gpu crate (requires CUDA feature)
// See trueno-gpu/tests/ptx_pixel_fkr.rs

/// Placeholder test documenting PTX FKR location
#[test]
fn ptx_pixel_fkr_location() {
    println!("PTX Pixel FKR tests are in trueno-gpu crate:");
    println!("  cargo test -p trueno-gpu --test pixel_fkr --features cuda");
    println!();
    println!("Tests validate:");
    println!("  - QuantizeKernel (Issue #67 prevention)");
    println!("  - Q4_K dequantization");
    println!("  - GEMM kernels");
    println!("  - Softmax PTX");
}

// ============================================================================
// REGRESSION SUMMARY
// ============================================================================

/// Summary test that reports all FKR status
#[test]
fn pixel_fkr_summary() {
    println!();
    println!("========================================");
    println!("  TRUENO-SPEC-013 Pixel FKR Summary");
    println!("========================================");
    println!();
    println!("  scalar-pixel-fkr: Baseline truth tests");
    println!("    - rmsnorm_4096");
    println!("    - silu_8192");
    println!("    - softmax_2048");
    println!("    - rope_512");
    println!("    - causal_mask_64x64");
    println!("    - q4k_dequant_256");
    println!();
    println!("  simd-pixel-fkr: SIMD vs scalar (+-1 ULP)");
    println!("    - vector_ops_10000");
    println!("    - softmax_2048");
    println!("    - unaligned_17");
    println!("    - remainder_255");
    println!("    - relu_10000");
    println!();
    #[cfg(feature = "gpu")]
    {
        println!("  wgpu-pixel-fkr: WGPU vs scalar (+-2 ULP)");
        println!("    - large_vector_100000");
        println!("    - matmul_128x128");
        println!("    - softmax_4096");
    }
    #[cfg(not(feature = "gpu"))]
    {
        println!("  wgpu-pixel-fkr: SKIPPED (gpu feature disabled)");
    }
    println!();
    println!("  ptx-pixel-fkr: See trueno-gpu crate");
    println!();
    println!("========================================");
}
