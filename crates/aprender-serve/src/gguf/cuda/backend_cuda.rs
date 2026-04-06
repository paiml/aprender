
// ============================================================================
// Tests (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    // =========================================================================
    // Constructor and Builder Tests
    // =========================================================================

    #[test]
    fn test_cuda_backend_new() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        assert_eq!(backend.m, 1);
        assert_eq!(backend.n, 4096);
        assert_eq!(backend.k, 4096);
        assert_eq!(backend.head_dim, 64);
        assert_eq!(backend.num_heads, 32); // default
        assert_eq!(backend.max_seq_len, 2048); // default
    }

    #[test]
    fn test_cuda_backend_with_num_heads() {
        let backend = CudaBackend::new(1, 4096, 4096, 64).with_num_heads(16);
        assert_eq!(backend.num_heads, 16);
    }

    #[test]
    fn test_cuda_backend_with_max_seq_len() {
        let backend = CudaBackend::new(1, 4096, 4096, 64).with_max_seq_len(4096);
        assert_eq!(backend.max_seq_len, 4096);
    }

    #[test]
    fn test_cuda_backend_builder_chain() {
        let backend = CudaBackend::new(1, 4096, 4096, 128)
            .with_num_heads(8)
            .with_max_seq_len(1024);
        assert_eq!(backend.num_heads, 8);
        assert_eq!(backend.max_seq_len, 1024);
        assert_eq!(backend.head_dim, 128);
    }

    #[test]
    fn test_cuda_backend_clone() {
        let backend = CudaBackend::new(2, 1024, 2048, 64);
        let cloned = backend.clone();
        assert_eq!(cloned.m, 2);
        assert_eq!(cloned.n, 1024);
        assert_eq!(cloned.k, 2048);
    }

    #[test]
    fn test_cuda_backend_debug() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        let debug_str = format!("{:?}", backend);
        assert!(debug_str.contains("CudaBackend"));
        assert!(debug_str.contains("4096"));
    }

    // =========================================================================
    // Q4_K Kernel Tests
    // =========================================================================

    #[test]
    fn test_q4k_gemm_kernel_name() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        assert_eq!(backend.q4k_gemm_kernel_name(), "q4k_gemm_fused");
    }

    #[test]
    fn test_q4k_blocks_per_row() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        assert_eq!(backend.q4k_blocks_per_row(), 128); // 4096 / 32
    }

    #[test]
    fn test_q4k_blocks_per_row_small() {
        let backend = CudaBackend::new(1, 1024, 256, 64);
        assert_eq!(backend.q4k_blocks_per_row(), 8); // 256 / 32
    }

    #[test]
    fn test_q4k_weight_bytes() {
        let backend = CudaBackend::new(1, 1024, 1024, 64);
        // blocks_per_row = 1024 / 32 = 32
        // bytes_per_row = 32 * 18 = 576
        // total = 1024 * 576 = 589,824
        assert_eq!(backend.q4k_weight_bytes(), 589_824);
    }

    #[test]
    fn test_q4k_weight_bytes_large() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        // blocks_per_row = 4096 / 32 = 128
        // bytes_per_row = 128 * 18 = 2304
        // total = 4096 * 2304 = 9,437,184
        assert_eq!(backend.q4k_weight_bytes(), 9_437_184);
    }

    #[test]
    fn test_q4k_gemm_ptx_generation() {
        let backend = CudaBackend::new(1, 1024, 256, 64);
        let ptx = backend.q4k_gemm_ptx();
        assert!(!ptx.is_empty());
        assert!(ptx.contains(".version"));
    }

    #[test]
    fn test_q4k_gemm_ptx_caching() {
        let backend = CudaBackend::new(1, 1024, 256, 64);
        let ptx1 = backend.q4k_gemm_ptx();
        let ptx2 = backend.q4k_gemm_ptx(); // Should use cache
        assert_eq!(ptx1, ptx2);
    }

    // =========================================================================
    // FlashAttention Tests
    // =========================================================================

    #[test]
    fn test_flash_attention_kernel_name_causal() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        assert_eq!(
            backend.flash_attention_kernel_name(true),
            "flash_attention_causal"
        );
    }

    #[test]
    fn test_flash_attention_kernel_name_non_causal() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        assert_eq!(
            backend.flash_attention_kernel_name(false),
            "flash_attention"
        );
    }

    #[test]
    fn test_flash_attention_smem_bytes() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        // tile_q = 64, tile_kv = 64, d = 64
        // smem = (64*64 + 64*64*2) * 4 = (4096 + 8192) * 4 = 49152
        assert_eq!(backend.flash_attention_smem_bytes(), 49152);
    }

    #[test]
    fn test_flash_attention_smem_bytes_large_head_dim() {
        let backend = CudaBackend::new(1, 4096, 4096, 128);
        // tile_q = 64, tile_kv = 64, d = 128
        // smem = (64*128 + 64*128*2) * 4 = (8192 + 16384) * 4 = 98304
        assert_eq!(backend.flash_attention_smem_bytes(), 98304);
    }

    #[test]
    fn test_flash_attention_ptx_generation() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        let ptx = backend.flash_attention_ptx(512, 64, true);
        assert!(!ptx.is_empty());
        assert!(ptx.contains(".version"));
    }

    #[test]
    fn test_flash_attention_causal_ptx_caching() {
        let backend = CudaBackend::new(1, 4096, 4096, 64).with_max_seq_len(512);
        let ptx1 = backend.flash_attention_causal_ptx();
        let ptx2 = backend.flash_attention_causal_ptx(); // Should use cache
        assert_eq!(ptx1, ptx2);
    }

    // =========================================================================
    // KV Cache Tests
    // =========================================================================

    #[test]
    fn test_kv_cache_bytes_per_layer() {
        let backend = CudaBackend::new(1, 4096, 4096, 64)
            .with_num_heads(32)
            .with_max_seq_len(2048);
        // K: 32 * 2048 * 64 * 4 = 16,777,216
        // V: 32 * 2048 * 64 * 4 = 16,777,216
        // Total: 33,554,432
        assert_eq!(backend.kv_cache_bytes_per_layer(), 33_554_432);
    }

    #[test]
    fn test_kv_cache_total_bytes() {
        let backend = CudaBackend::new(1, 4096, 4096, 64)
            .with_num_heads(32)
            .with_max_seq_len(2048);
        let per_layer = backend.kv_cache_bytes_per_layer();
        assert_eq!(backend.kv_cache_total_bytes(22), per_layer * 22);
    }

    #[test]
    fn test_kv_cache_page_tokens() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        assert_eq!(backend.kv_cache_page_tokens(), 64);
    }

    #[test]
    fn test_kv_cache_pages_needed_exact() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        // 64 tokens = 1 page
        assert_eq!(backend.kv_cache_pages_needed(64), 1);
        // 128 tokens = 2 pages
        assert_eq!(backend.kv_cache_pages_needed(128), 2);
    }

    #[test]
    fn test_kv_cache_pages_needed_round_up() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        // 65 tokens = 2 pages (rounds up)
        assert_eq!(backend.kv_cache_pages_needed(65), 2);
        // 100 tokens = 2 pages
        assert_eq!(backend.kv_cache_pages_needed(100), 2);
        // 129 tokens = 3 pages
        assert_eq!(backend.kv_cache_pages_needed(129), 3);
    }

    #[test]
    fn test_kv_cache_pages_needed_large() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        // 2048 tokens = 32 pages
        assert_eq!(backend.kv_cache_pages_needed(2048), 32);
    }

    // =========================================================================
    // Launch Configuration Tests
    // =========================================================================

    #[test]
    fn test_q4k_gemm_launch_config() {
        let backend = CudaBackend::new(1, 1024, 4096, 64);
        let (grid, block) = backend.q4k_gemm_launch_config();
        // grid = (n/32, m/32, 1) = (1024/32, 1/32 rounded up, 1) = (32, 1, 1)
        assert_eq!(grid, (32, 1, 1));
        // block = (32*32, 1, 1) = (1024, 1, 1)
        assert_eq!(block, (1024, 1, 1));
    }

    #[test]
    fn test_flash_attention_launch_config() {
        let backend = CudaBackend::new(1, 4096, 4096, 64).with_num_heads(8);
        let (grid, block) = backend.flash_attention_launch_config(256);
        // grid = (256/64, 8, 1) = (4, 8, 1)
        assert_eq!(grid, (4, 8, 1));
        // block = (64 * 64, 1, 1) = (4096, 1, 1)
        assert_eq!(block, (4096, 1, 1));
    }

    // =========================================================================
    // Validation Tests
    // =========================================================================

    #[test]
    fn test_validate_dimensions_valid() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        assert!(backend.validate_dimensions());
    }

    #[test]
    fn test_validate_dimensions_k_not_divisible() {
        let backend = CudaBackend::new(1, 4096, 4095, 64); // k not divisible by 32
        assert!(!backend.validate_dimensions());
    }

    #[test]
    fn test_validate_dimensions_head_dim_not_power_of_two() {
        let backend = CudaBackend::new(1, 4096, 4096, 80); // 80 is not power of 2
        assert!(!backend.validate_dimensions());
    }

    #[test]
    fn test_validate_dimensions_zero_m() {
        let backend = CudaBackend::new(0, 4096, 4096, 64);
        assert!(!backend.validate_dimensions());
    }

    #[test]
    fn test_validate_dimensions_zero_n() {
        let backend = CudaBackend::new(1, 0, 4096, 64);
        assert!(!backend.validate_dimensions());
    }

    #[test]
    fn test_validate_dimensions_zero_k() {
        let backend = CudaBackend::new(1, 4096, 0, 64);
        assert!(!backend.validate_dimensions());
    }

    #[test]
    fn test_validate_dimensions_zero_head_dim() {
        let backend = CudaBackend::new(1, 4096, 4096, 0);
        assert!(!backend.validate_dimensions());
    }

    // =========================================================================
    // PTX Target Tests
    // =========================================================================

    #[test]
    fn test_ptx_target() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        assert_eq!(backend.ptx_target(), "sm_89");
    }

    #[test]
    fn test_ptx_version() {
        let backend = CudaBackend::new(1, 4096, 4096, 64);
        assert_eq!(backend.ptx_version(), (8, 0));
    }
}
