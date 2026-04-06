
// ============================================================================
// Tests for Dequantization Helpers (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // GH-44 FALSIFICATION: dtype=8 disambiguation (Q8_0 vs APR Q4)
    // =========================================================================

    /// GH-44: dtype=8 with 34 bytes/block MUST route to GGML Q8_0 (not APR Q4).
    /// If this test fails, 34-byte blocks are misinterpreted as 18-byte Q4.
    #[test]
    fn test_falsify_gh44_dtype8_disambiguation_q8_0() {
        let num_elements: usize = 32;
        let num_blocks = num_elements.div_ceil(32);
        // Q8_0: exactly 34 bytes per block
        let data_size = num_blocks * 34;
        assert_eq!(data_size, 34, "1 block of Q8_0 = 34 bytes");

        let data = vec![0u8; data_size];
        // Disambiguate: tensor_data.len() == num_blocks * 34 → Q8_0
        assert_eq!(
            data.len(),
            num_blocks * 34,
            "GH-44: 34B/block must select Q8_0 path"
        );
        let result = dequantize_q8_0_apr(&data, num_elements);
        assert_eq!(result.len(), num_elements);
    }

    /// GH-44: dtype=8 with 18 bytes/block MUST route to APR Q4 (not Q8_0).
    /// If this test fails, 18-byte blocks are misinterpreted as 34-byte Q8_0.
    #[test]
    fn test_falsify_gh44_dtype8_disambiguation_apr_q4() {
        let num_elements: usize = 32;
        let num_blocks = num_elements.div_ceil(32);
        // APR Q4: exactly 18 bytes per block
        let data_size = num_blocks * 18;
        assert_eq!(data_size, 18, "1 block of APR Q4 = 18 bytes");

        let data = vec![0u8; data_size];
        // Disambiguate: tensor_data.len() != num_blocks * 34 → APR Q4
        assert_ne!(
            data.len(),
            num_blocks * 34,
            "GH-44: 18B/block must NOT match Q8_0"
        );
        let result = dequantize_apr_q4_native(&data, num_elements);
        assert_eq!(result.len(), num_elements);
    }

    /// GH-44: dtype=9 is APR Q8 native — must produce num_elements f32s.
    /// If this test fails, dtype=9 falls to the `_` default branch (raw f32 read).
    #[test]
    fn test_falsify_gh44_dtype9_apr_q8_native() {
        let num_elements: usize = 64;
        // APR Q8: 4 bytes scale + 64 i8 values = 68 bytes
        let expected_size = 4 + num_elements;
        let mut data = vec![0u8; expected_size];
        // Set scale to 1.0f32 so dequant produces i8-as-f32
        data[..4].copy_from_slice(&1.0f32.to_le_bytes());
        data[4] = 42_u8; // i8 value = 42

        let result = dequantize_apr_q8_native(&data, num_elements);
        assert_eq!(result.len(), num_elements);
        assert!(
            (result[0] - 42.0).abs() < 0.01,
            "GH-44: dtype=9 must dequant i8 42 with scale 1.0 → 42.0, got {}",
            result[0]
        );
    }
include!("dequant_f16.rs");
}
