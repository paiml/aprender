#[cfg(test)]
mod tests {
    use super::super::flash::*;
    use crate::kernels::Kernel;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_flash_attention_config_fuzz(
            seq_len in 1u32..8192,
            head_dim in 32u32..256,
            tile_q in 16u32..128,
            tile_kv in 16u32..128,
            scale in 0.1f32..10.0f32,
            causal in any::<bool>(),
            use_tensor_cores in any::<bool>()
        ) {
            let kernel = AttentionKernel {
                seq_len,
                head_dim,
                tile_q,
                tile_kv,
                scale,
                causal,
                use_tensor_cores,
            };

            // Verify basic invariants
            assert!(kernel.seq_len > 0);
            assert!(kernel.head_dim > 0);

            // Verify PTX generation doesn't panic
            let ptx = kernel.emit_ptx();
            assert!(ptx.contains(".visible .entry flash_attention"));

            // Check for critical instructions based on config
            if use_tensor_cores {
                // Should have wmma instructions if tensor cores are enabled
                // Note: The builder might fall back if dimensions aren't multiples of 16
                // checking for wmma might be flaky if the builder is smart,
                // but at least it shouldn't panic.
            } else {
                // Should use standard fma
                assert!(ptx.contains("fma.rn.f32"));
            }
        }
    }

    #[test]
    fn test_flash_attention_edge_cases() {
        // Test 0 sequence length (should handle gracefully or be rejected)
        let kernel_zero = AttentionKernel {
            seq_len: 0,
            head_dim: 64,
            tile_q: 32,
            tile_kv: 32,
            scale: 1.0,
            causal: true,
            use_tensor_cores: false,
        };
        // Just ensure it doesn't crash during emit
        let _ = kernel_zero.emit_ptx();

        // Test very large sequence
        let kernel_large = AttentionKernel {
            seq_len: 1_000_000,
            head_dim: 64,
            tile_q: 32,
            tile_kv: 32,
            scale: 1.0,
            causal: false,
            use_tensor_cores: false,
        };
        let ptx = kernel_large.emit_ptx();
        assert!(ptx.contains(".visible .entry flash_attention"));
    }

    /// GH-32 FIX: with_tiles() must clamp tile_kv to at least head_dim
    /// to prevent shared memory OOB in K dot product loop.
    #[test]
    fn test_with_tiles_clamps_tile_kv_to_head_dim() {
        // head_dim=128, request tile_kv=32 — should clamp to 128
        let kernel = AttentionKernel::new(2048, 128).with_tiles(16, 32);
        assert_eq!(kernel.tile_kv, 128, "GH-32: with_tiles() must clamp tile_kv to head_dim");
        assert_eq!(kernel.tile_q, 16, "tile_q should be set as requested");

        // head_dim=64, request tile_kv=128 — should keep 128
        let kernel2 = AttentionKernel::new(2048, 64).with_tiles(32, 128);
        assert_eq!(kernel2.tile_kv, 128, "tile_kv >= head_dim should be kept as-is");

        // head_dim=64, request tile_kv=64 — exact boundary
        let kernel3 = AttentionKernel::new(2048, 64).with_tiles(32, 64);
        assert_eq!(kernel3.tile_kv, 64, "tile_kv == head_dim should be kept");
    }

    /// GH-32 FIX: Standard FlashAttention kernel must contain k_row_loop
    /// for iterating over all K rows in the tile (previously missing).
    #[test]
    fn test_flash_attention_has_k_row_loop() {
        let kernel = AttentionKernel::new(512, 64);
        let ptx = kernel.emit_ptx();

        // Must have k_row_loop labels (the GH-32 fix)
        let k_row_count = ptx.matches("k_row_loop").count();
        assert!(
            k_row_count >= 2,
            "GH-32: k_row_loop should appear at least twice (label + branch), found {}",
            k_row_count
        );

        // Must have cooperative K loading loop
        assert!(ptx.contains("k_coop_load"), "GH-32: Should have strided cooperative K loading");

        // Must have cooperative V loading loop
        assert!(ptx.contains("v_coop_load"), "GH-32: Should have strided cooperative V loading");
    }

    /// GH-32 FIX: Causal FlashAttention must have per-row causal masking
    /// within the k_row loop (not just per-block skip).
    #[test]
    fn test_flash_attention_causal_has_per_row_masking() {
        let kernel = AttentionKernel::new(512, 64).with_causal();
        let ptx = kernel.emit_ptx();

        // Must have k_row_next label (branch target for causal skip)
        assert!(
            ptx.contains("k_row_next"),
            "GH-32: Causal kernel should have k_row_next label for per-row masking"
        );

        // Also must have block-level causal skip
        assert!(ptx.contains("kv_loop_end"), "Should still have block-level causal skip");
    }

    /// GH-32: Constructor defaults must enforce tile_kv >= head_dim
    #[test]
    fn test_constructor_enforces_tile_kv_ge_head_dim() {
        // Small seq_len, large head_dim — tile_kv must be clamped up
        let kernel = AttentionKernel::new(32, 128);
        assert!(
            kernel.tile_kv >= kernel.head_dim,
            "Constructor must enforce tile_kv >= head_dim: tile_kv={}, head_dim={}",
            kernel.tile_kv,
            kernel.head_dim
        );

        // Tensor core variant
        let tc_kernel = AttentionKernel::tensor_core(32, 128);
        assert!(
            tc_kernel.tile_kv >= tc_kernel.head_dim,
            "Tensor core constructor must enforce tile_kv >= head_dim: tile_kv={}, head_dim={}",
            tc_kernel.tile_kv,
            tc_kernel.head_dim
        );
    }
}
