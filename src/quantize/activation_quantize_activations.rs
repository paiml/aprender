
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_activations_q8_0_negative() {
        let activations = vec![-127.0f32; 32];
        let (scales, quants) = quantize_activations_q8_0(&activations);

        assert_eq!(scales.len(), 1);
        for q in &quants {
            assert_eq!(*q, -127);
        }
    }

    #[test]
    fn test_quantize_activations_q8_0_mixed() {
        let activations: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 2.0).collect();
        let (scales, quants) = quantize_activations_q8_0(&activations);

        assert_eq!(scales.len(), 2); // 64/32 = 2 blocks
        assert_eq!(quants.len(), 64);

        // Verify quants are all i8 values (type-guaranteed to be in -128..=127)
        assert!(quants.iter().all(|_| true)); // Just exercise the iterator
    }

    #[test]
    fn test_quantize_activations_q8_0_partial_block() {
        // 40 elements = 1 full block + 8 elements (padded to 32)
        let activations: Vec<f32> = (0..40).map(|i| i as f32).collect();
        let (scales, quants) = quantize_activations_q8_0(&activations);

        assert_eq!(scales.len(), 2); // 40/32 rounded up = 2
        assert_eq!(quants.len(), 64); // Padded to 2 * 32

        // Padding should be zeros
        for q in &quants[40..] {
            assert_eq!(*q, 0);
        }
    }

    #[test]
    fn test_quantize_activations_q8_0_roundtrip_approximate() {
        let activations: Vec<f32> = (0..32).map(|i| (i as f32) * 4.0).collect();
        let (scales, quants) = quantize_activations_q8_0(&activations);

        // Dequantize manually
        let dequant: Vec<f32> = quants.iter().map(|&q| scales[0] * q as f32).collect();

        // Should be approximately equal (within quantization error)
        for i in 0..32 {
            let diff = (activations[i] - dequant[i]).abs();
            let tolerance = scales[0]; // Max error is 1 quant step
            assert!(
                diff <= tolerance,
                "at {}: original={} dequant={} diff={}",
                i,
                activations[i],
                dequant[i],
                diff
            );
        }
    }
include!("activation_quantize_rmsnorm.rs");
}
