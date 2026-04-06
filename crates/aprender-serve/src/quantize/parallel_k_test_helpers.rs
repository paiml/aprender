
    // ============================================================================
    // HELPER FUNCTIONS FOR TEST DATA GENERATION
    // ============================================================================

    /// Create valid Q4_K weight data for testing
    fn create_q4k_weights(out_dim: usize, in_dim: usize) -> Vec<u8> {
        let super_blocks_per_row = in_dim.div_ceil(QK_K);
        let bytes_per_row = super_blocks_per_row * 144;
        let total_bytes = out_dim * bytes_per_row;

        let mut data = vec![0u8; total_bytes];

        // Fill each row's super-blocks with valid data
        for row in 0..out_dim {
            for sb in 0..super_blocks_per_row {
                let sb_start = row * bytes_per_row + sb * 144;

                // Set d = 0.1 (f16: ~0x2E66)
                data[sb_start..sb_start + 2].copy_from_slice(&0x2E66u16.to_le_bytes());

                // Set dmin = 0.01 (f16: ~0x211F)
                data[sb_start + 2..sb_start + 4].copy_from_slice(&0x211Fu16.to_le_bytes());

                // Set scales to small values
                for i in 0..12 {
                    data[sb_start + 4 + i] = 0x11;
                }

                // Set qs to pattern
                for i in 0..128 {
                    data[sb_start + 16 + i] = ((row + i) % 256) as u8;
                }
            }
        }

        data
    }

    /// Create valid Q5_K weight data for testing
    fn create_q5k_weights(out_dim: usize, in_dim: usize) -> Vec<u8> {
        let super_blocks_per_row = in_dim.div_ceil(QK_K);
        let bytes_per_row = super_blocks_per_row * 176;
        let total_bytes = out_dim * bytes_per_row;

        let mut data = vec![0u8; total_bytes];

        for row in 0..out_dim {
            for sb in 0..super_blocks_per_row {
                let sb_start = row * bytes_per_row + sb * 176;

                // Set d and dmin
                data[sb_start..sb_start + 2].copy_from_slice(&0x3C00u16.to_le_bytes());
                data[sb_start + 2..sb_start + 4].copy_from_slice(&0x3800u16.to_le_bytes());

                // Set scales
                for i in 0..12 {
                    data[sb_start + 4 + i] = 0x11;
                }
            }
        }

        data
    }

    /// Create valid Q6_K weight data for testing
    fn create_q6k_weights(out_dim: usize, in_dim: usize) -> Vec<u8> {
        let super_blocks_per_row = in_dim.div_ceil(QK_K);
        let bytes_per_row = super_blocks_per_row * 210;
        let total_bytes = out_dim * bytes_per_row;

        let mut data = vec![0u8; total_bytes];

        for row in 0..out_dim {
            for sb in 0..super_blocks_per_row {
                let sb_start = row * bytes_per_row + sb * 210;

                // Set d at offset 208
                data[sb_start + 208..sb_start + 210].copy_from_slice(&0x3C00u16.to_le_bytes());

                // Set scales
                for i in 0..16 {
                    data[sb_start + 192 + i] = 1;
                }
            }
        }

        data
    }

    // ============================================================================
    // TILED MATVEC TESTS
    // ============================================================================

    #[test]
    fn test_fused_q4k_tiled_matvec_basic() {
        let in_dim: usize = 256; // One super-block
        let out_dim: usize = 8;

        let weights = create_q4k_weights(out_dim, in_dim);
        let activations = vec![0.1f32; in_dim];

        let result = fused_q4k_tiled_matvec(&weights, &activations, in_dim, out_dim, None);
        assert!(result.is_ok());
        let output = result.expect("output");
        assert_eq!(output.len(), out_dim);
    }

    #[test]
    fn test_fused_q4k_tiled_matvec_weight_too_small() {
        let in_dim: usize = 256;
        let out_dim: usize = 8;

        let weights = vec![0u8; 100]; // Too small
        let activations = vec![0.1f32; in_dim];

        let result = fused_q4k_tiled_matvec(&weights, &activations, in_dim, out_dim, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too small"));
    }

    #[test]
    fn test_fused_q4k_tiled_matvec_activation_mismatch() {
        let in_dim: usize = 256;
        let out_dim: usize = 8;

        let weights = create_q4k_weights(out_dim, in_dim);
        let activations = vec![0.1f32; 128]; // Wrong size

        let result = fused_q4k_tiled_matvec(&weights, &activations, in_dim, out_dim, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("doesn't match"));
    }

    #[test]
    fn test_fused_q4k_tiled_matvec_custom_tile_size() {
        let in_dim: usize = 256;
        let out_dim: usize = 128;

        let weights = create_q4k_weights(out_dim, in_dim);
        let activations = vec![0.1f32; in_dim];

        // Use custom tile size
        let result = fused_q4k_tiled_matvec(&weights, &activations, in_dim, out_dim, Some(32));
        assert!(result.is_ok());
        assert_eq!(result.expect("expected value").len(), out_dim);
    }

    // ============================================================================
    // PARALLEL Q4_K MATVEC TESTS
    // ============================================================================

    #[test]
    fn test_fused_q4k_parallel_matvec_basic() {
        let in_dim: usize = 256;
        let out_dim: usize = 64;

        let weights = create_q4k_weights(out_dim, in_dim);
        let activations = vec![0.1f32; in_dim];

        let result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim);
        assert!(result.is_ok());
        let output = result.expect("output");
        assert_eq!(output.len(), out_dim);
    }

    #[test]
    fn test_fused_q4k_parallel_matvec_sequential_path() {
        // out_dim < 256 uses sequential path
        let in_dim: usize = 256;
        let out_dim: usize = 32;

        let weights = create_q4k_weights(out_dim, in_dim);
        let activations = vec![0.1f32; in_dim];

        let result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim);
        assert!(result.is_ok());
        assert_eq!(result.expect("expected value").len(), out_dim);
    }

    #[test]
    fn test_fused_q4k_parallel_matvec_parallel_path() {
        // out_dim >= 256 uses parallel path
        let in_dim: usize = 256;
        let out_dim: usize = 512;

        let weights = create_q4k_weights(out_dim, in_dim);
        let activations = vec![0.1f32; in_dim];

        let result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim);
        assert!(result.is_ok());
        assert_eq!(result.expect("expected value").len(), out_dim);
    }

    #[test]
    fn test_fused_q4k_parallel_matvec_weight_too_small() {
        let in_dim: usize = 256;
        let out_dim: usize = 64;

        let weights = vec![0u8; 100]; // Too small
        let activations = vec![0.1f32; in_dim];

        let result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim);
        assert!(result.is_err());
    }

    #[test]
    fn test_fused_q4k_parallel_matvec_into_basic() {
        let in_dim: usize = 256;
        let out_dim: usize = 64;

        let weights = create_q4k_weights(out_dim, in_dim);
        let activations = vec![0.1f32; in_dim];
        let mut output = vec![0.0f32; out_dim];

        let result =
            fused_q4k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fused_q4k_parallel_matvec_into_output_too_small() {
        let in_dim: usize = 256;
        let out_dim: usize = 64;

        let weights = create_q4k_weights(out_dim, in_dim);
        let activations = vec![0.1f32; in_dim];
        let mut output = vec![0.0f32; 32]; // Too small

        let result =
            fused_q4k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too small"));
    }

    // ============================================================================
    // PARALLEL Q5_K MATVEC TESTS
    // ============================================================================

    #[test]
    fn test_fused_q5k_parallel_matvec_basic() {
        let in_dim: usize = 256;
        let out_dim: usize = 64;

        let weights = create_q5k_weights(out_dim, in_dim);
        let activations = vec![0.1f32; in_dim];

        let result = fused_q5k_parallel_matvec(&weights, &activations, in_dim, out_dim);
        assert!(result.is_ok());
        assert_eq!(result.expect("expected value").len(), out_dim);
    }

    #[test]
    fn test_fused_q5k_parallel_matvec_weight_too_small() {
        let in_dim: usize = 256;
        let out_dim: usize = 64;

        let weights = vec![0u8; 100]; // Too small
        let activations = vec![0.1f32; in_dim];

        let result = fused_q5k_parallel_matvec(&weights, &activations, in_dim, out_dim);
        assert!(result.is_err());
    }

    #[test]
    fn test_fused_q5k_parallel_matvec_into_basic() {
        let in_dim: usize = 256;
        let out_dim: usize = 64;

        let weights = create_q5k_weights(out_dim, in_dim);
        let activations = vec![0.1f32; in_dim];
        let mut output = vec![0.0f32; out_dim];

        let result =
            fused_q5k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fused_q5k_parallel_matvec_into_output_too_small() {
        let in_dim: usize = 256;
        let out_dim: usize = 64;

        let weights = create_q5k_weights(out_dim, in_dim);
        let activations = vec![0.1f32; in_dim];
        let mut output = vec![0.0f32; 32]; // Too small

        let result =
            fused_q5k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
        assert!(result.is_err());
    }

    // ============================================================================
    // PARALLEL Q6_K MATVEC TESTS
    // ============================================================================

    #[test]
    fn test_fused_q6k_parallel_matvec_basic() {
        let in_dim: usize = 256;
        let out_dim: usize = 64;

        let weights = create_q6k_weights(out_dim, in_dim);
        let activations = vec![0.1f32; in_dim];

        let result = fused_q6k_parallel_matvec(&weights, &activations, in_dim, out_dim);
        assert!(result.is_ok());
        assert_eq!(result.expect("expected value").len(), out_dim);
    }

    #[test]
    fn test_fused_q6k_parallel_matvec_weight_too_small() {
        let in_dim: usize = 256;
        let out_dim: usize = 64;

        let weights = vec![0u8; 100]; // Too small
        let activations = vec![0.1f32; in_dim];

        let result = fused_q6k_parallel_matvec(&weights, &activations, in_dim, out_dim);
        assert!(result.is_err());
    }

    #[test]
    fn test_fused_q6k_parallel_matvec_into_basic() {
        let in_dim: usize = 256;
        let out_dim: usize = 64;

        let weights = create_q6k_weights(out_dim, in_dim);
        let activations = vec![0.1f32; in_dim];
        let mut output = vec![0.0f32; out_dim];

        let result =
            fused_q6k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fused_q6k_parallel_matvec_into_output_too_small() {
        let in_dim: usize = 256;
        let out_dim: usize = 64;

        let weights = create_q6k_weights(out_dim, in_dim);
        let activations = vec![0.1f32; in_dim];
        let mut output = vec![0.0f32; 32]; // Too small

        let result =
            fused_q6k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
        assert!(result.is_err());
    }

    // LAYOUT-002: Backwards-compat alias tests DELETED (2026-02-03)
    // ONE WAY ONLY: Use fused_q{4,5,6}k_parallel_matvec* functions directly

    // ============================================================================
    // Q4K × Q8K PARALLEL MATVEC TESTS
    // ============================================================================

    #[test]
    fn test_fused_q4k_q8k_parallel_matvec_into_basic() {
        let in_dim: usize = 256;
        let out_dim: usize = 64;

        let weights = create_q4k_weights(out_dim, in_dim);

        // Create Q8K quantized activations
        let num_super_blocks = in_dim.div_ceil(QK_K);
        let q8k_scales = vec![0.1f32; num_super_blocks];
        let q8k_quants = vec![10i8; in_dim];
        let mut output = vec![0.0f32; out_dim];

        let result = fused_q4k_q8k_parallel_matvec_into(
            &weights,
            &q8k_scales,
            &q8k_quants,
            in_dim,
            out_dim,
            &mut output,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_fused_q4k_q8k_parallel_matvec_into_weight_too_small() {
        let in_dim: usize = 256;
        let out_dim: usize = 64;

        let weights = vec![0u8; 100]; // Too small
        let q8k_scales = vec![0.1f32; 1];
        let q8k_quants = vec![10i8; in_dim];
        let mut output = vec![0.0f32; out_dim];

        let result = fused_q4k_q8k_parallel_matvec_into(
            &weights,
            &q8k_scales,
            &q8k_quants,
            in_dim,
            out_dim,
            &mut output,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_fused_q4k_q8k_parallel_matvec_into_output_too_small() {
        let in_dim: usize = 256;
        let out_dim: usize = 64;

        let weights = create_q4k_weights(out_dim, in_dim);
        let q8k_scales = vec![0.1f32; 1];
        let q8k_quants = vec![10i8; in_dim];
        let mut output = vec![0.0f32; 32]; // Too small

        let result = fused_q4k_q8k_parallel_matvec_into(
            &weights,
            &q8k_scales,
            &q8k_quants,
            in_dim,
            out_dim,
            &mut output,
        );
        assert!(result.is_err());
    }
