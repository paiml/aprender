
#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // FFN UP+GATE FUSED TESTS
    // ============================================================================

    #[test]
    fn test_fused_q4k_q8k_ffn_up_gate_into_basic() {
        let in_dim: usize = 256;
        let out_dim: usize = 64;

        let up_weights = create_q4k_weights(out_dim, in_dim);
        let gate_weights = create_q4k_weights(out_dim, in_dim);

        let num_super_blocks = in_dim.div_ceil(QK_K);
        let q8k_scales = vec![0.1f32; num_super_blocks];
        let q8k_quants = vec![10i8; in_dim];

        let mut up_output = vec![0.0f32; out_dim];
        let mut gate_output = vec![0.0f32; out_dim];

        let result = fused_q4k_q8k_ffn_up_gate_into(
            &up_weights,
            &gate_weights,
            &q8k_scales,
            &q8k_quants,
            in_dim,
            out_dim,
            &mut up_output,
            &mut gate_output,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_fused_q4k_q8k_ffn_up_gate_into_weight_too_small() {
        let in_dim: usize = 256;
        let out_dim: usize = 64;

        let up_weights = vec![0u8; 100]; // Too small
        let gate_weights = create_q4k_weights(out_dim, in_dim);

        let q8k_scales = vec![0.1f32; 1];
        let q8k_quants = vec![10i8; in_dim];

        let mut up_output = vec![0.0f32; out_dim];
        let mut gate_output = vec![0.0f32; out_dim];

        let result = fused_q4k_q8k_ffn_up_gate_into(
            &up_weights,
            &gate_weights,
            &q8k_scales,
            &q8k_quants,
            in_dim,
            out_dim,
            &mut up_output,
            &mut gate_output,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_fused_q4k_q8k_ffn_up_gate_into_output_too_small() {
        let in_dim: usize = 256;
        let out_dim: usize = 64;

        let up_weights = create_q4k_weights(out_dim, in_dim);
        let gate_weights = create_q4k_weights(out_dim, in_dim);

        let q8k_scales = vec![0.1f32; 1];
        let q8k_quants = vec![10i8; in_dim];

        let mut up_output = vec![0.0f32; 32]; // Too small
        let mut gate_output = vec![0.0f32; out_dim];

        let result = fused_q4k_q8k_ffn_up_gate_into(
            &up_weights,
            &gate_weights,
            &q8k_scales,
            &q8k_quants,
            in_dim,
            out_dim,
            &mut up_output,
            &mut gate_output,
        );
        assert!(result.is_err());
    }

    // ============================================================================
    // EDGE CASE TESTS
    // ============================================================================

    #[test]
    fn test_parallel_matvec_multiple_super_blocks_per_row() {
        // in_dim = 512 means 2 super-blocks per row
        let in_dim: usize = 512;
        let out_dim: usize = 32;

        let weights = create_q4k_weights(out_dim, in_dim);
        let activations = vec![0.1f32; in_dim];

        let result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim);
        assert!(result.is_ok());
        assert_eq!(result.expect("result").len(), out_dim);
    }

    #[test]
    fn test_parallel_matvec_large_output_dimension() {
        // Test with larger output dimension (uses parallel path)
        let in_dim: usize = 256;
        let out_dim: usize = 1024;

        let weights = create_q4k_weights(out_dim, in_dim);
        let activations = vec![0.1f32; in_dim];

        let result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim);
        assert!(result.is_ok());
        assert_eq!(result.expect("result").len(), out_dim);
    }

    #[test]
    fn test_parallel_matvec_at_parallel_threshold() {
        // Exactly at the threshold (256)
        let in_dim: usize = 256;
        let out_dim: usize = 256;

        let weights = create_q4k_weights(out_dim, in_dim);
        let activations = vec![0.1f32; in_dim];

        let result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim);
        assert!(result.is_ok());
        assert_eq!(result.expect("result").len(), out_dim);
    }

    #[test]
    fn test_parallel_matvec_just_below_parallel_threshold() {
        // Just below threshold (255) - uses sequential path
        let in_dim: usize = 256;
        let out_dim: usize = 255;

        let weights = create_q4k_weights(out_dim, in_dim);
        let activations = vec![0.1f32; in_dim];

        let result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim);
        assert!(result.is_ok());
        assert_eq!(result.expect("result").len(), out_dim);
    }
include!("parallel_k_test_helpers.rs");
}
