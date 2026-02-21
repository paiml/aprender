
/// Dequantize block back to f32
pub fn dequantize(block: &QuantizedBlock) -> Result<Vec<f32>> {
    match block.quant_type {
        QuantType::Q8_0 => Q8_0Quantizer.dequantize(block),
        QuantType::Q4_0 => Q4_0Quantizer.dequantize(block),
        QuantType::Q4_1 => Err(AprenderError::FormatError {
            message: "Q4_1 dequantization not yet implemented".to_string(),
        }),
        QuantType::Q8Tensor => Err(AprenderError::FormatError {
            message: "Q8Tensor dequantization not yet implemented".to_string(),
        }),
        QuantType::Custom => Err(AprenderError::FormatError {
            message: "Custom dequantization requires a custom Quantizer implementation".to_string(),
        }),
    }
}

/// Calculate mean squared error between original and dequantized values
pub fn quantization_mse(original: &[f32], dequantized: &[f32]) -> f32 {
    if original.len() != dequantized.len() || original.is_empty() {
        return f32::NAN;
    }

    let sum_sq_error: f32 = original
        .iter()
        .zip(dequantized.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();

    sum_sq_error / original.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_type_from_u8() {
        assert_eq!(QuantType::from_u8(0x01), Some(QuantType::Q8_0));
        assert_eq!(QuantType::from_u8(0x02), Some(QuantType::Q4_0));
        assert_eq!(QuantType::from_u8(0x03), Some(QuantType::Q4_1));
        assert_eq!(QuantType::from_u8(0x10), Some(QuantType::Q8Tensor));
        assert_eq!(QuantType::from_u8(0xFF), Some(QuantType::Custom));
        assert_eq!(QuantType::from_u8(0x99), None);
    }

    #[test]
    fn test_quant_type_bits_per_weight() {
        assert!((QuantType::Q8_0.bits_per_weight() - 8.5).abs() < 0.01);
        assert!((QuantType::Q4_0.bits_per_weight() - 4.5).abs() < 0.01);
        assert!((QuantType::Q4_1.bits_per_weight() - 5.0).abs() < 0.01);
        assert!((QuantType::Q8Tensor.bits_per_weight() - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_q8_0_roundtrip_simple() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let shape = vec![8];

        let quantizer = Q8_0Quantizer;
        let quantized = quantizer.quantize(&data, &shape).expect("quantize");
        let dequantized = quantizer.dequantize(&quantized).expect("dequantize");

        assert_eq!(dequantized.len(), data.len());

        // Check values are close (quantization introduces error)
        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            assert!(
                (orig - deq).abs() < 0.1,
                "Values differ too much: {} vs {}",
                orig,
                deq
            );
        }
    }

    #[test]
    fn test_q8_0_roundtrip_large() {
        // Test with more than one block (32 elements)
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1 - 5.0).collect();
        let shape = vec![100];

        let quantized = quantize(&data, &shape, QuantType::Q8_0).expect("quantize");
        let dequantized = dequantize(&quantized).expect("dequantize");

        assert_eq!(dequantized.len(), data.len());

        let mse = quantization_mse(&data, &dequantized);
        assert!(mse < 0.01, "MSE too high: {}", mse);
    }

    #[test]
    fn test_q8_0_block_size() {
        let data: Vec<f32> = vec![1.0; 64]; // 2 blocks
        let shape = vec![64];

        let quantized = quantize(&data, &shape, QuantType::Q8_0).expect("quantize");

        assert_eq!(quantized.num_blocks(), 2);
        assert_eq!(quantized.blocks.len(), 2 * Q8_0_BLOCK_BYTES);
    }

    #[test]
    fn test_q8_0_compression_ratio() {
        let data: Vec<f32> = vec![1.0; 128]; // 4 blocks
        let shape = vec![128];

        let quantized = quantize(&data, &shape, QuantType::Q8_0).expect("quantize");

        // Original: 128 * 4 = 512 bytes
        // Quantized: 4 blocks * 34 bytes = 136 bytes
        // Ratio: 512 / 136 ≈ 3.76
        let ratio = quantized.compression_ratio();
        assert!(ratio > 3.5, "Compression ratio too low: {}", ratio);
    }

    #[test]
    fn test_q4_0_roundtrip_simple() {
        let data: Vec<f32> = vec![1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 0.0, 0.25];
        let shape = vec![8];

        let quantizer = Q4_0Quantizer;
        let quantized = quantizer.quantize(&data, &shape).expect("quantize");
        let dequantized = quantizer.dequantize(&quantized).expect("dequantize");

        assert_eq!(dequantized.len(), data.len());

        // Q4_0 has lower precision, allow more error
        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            assert!(
                (orig - deq).abs() < 0.5,
                "Values differ too much: {} vs {}",
                orig,
                deq
            );
        }
    }

    #[test]
    fn test_q4_0_roundtrip_large() {
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.1 - 5.0).collect();
        let shape = vec![100];

        let quantized = quantize(&data, &shape, QuantType::Q4_0).expect("quantize");
        let dequantized = dequantize(&quantized).expect("dequantize");

        assert_eq!(dequantized.len(), data.len());

        // Q4_0 has lower precision, MSE will be higher
        let mse = quantization_mse(&data, &dequantized);
        assert!(mse < 0.5, "MSE too high for Q4_0: {}", mse);
    }

    #[test]
    fn test_q4_0_block_size() {
        let data: Vec<f32> = vec![1.0; 64]; // 2 blocks
        let shape = vec![64];

        let quantized = quantize(&data, &shape, QuantType::Q4_0).expect("quantize");

        assert_eq!(quantized.num_blocks(), 2);
        assert_eq!(quantized.blocks.len(), 2 * Q4_0_BLOCK_BYTES);
    }

    #[test]
    fn test_q4_0_compression_ratio() {
        let data: Vec<f32> = vec![1.0; 128]; // 4 blocks
        let shape = vec![128];

        let quantized = quantize(&data, &shape, QuantType::Q4_0).expect("quantize");

        // Original: 128 * 4 = 512 bytes
        // Quantized: 4 blocks * 18 bytes = 72 bytes
        // Ratio: 512 / 72 ≈ 7.1
        let ratio = quantized.compression_ratio();
        assert!(ratio > 6.0, "Compression ratio too low: {}", ratio);
    }

    #[test]
    fn test_quantize_zeros() {
        let data: Vec<f32> = vec![0.0; 32];
        let shape = vec![32];

        let quantized = quantize(&data, &shape, QuantType::Q8_0).expect("quantize");
        let dequantized = dequantize(&quantized).expect("dequantize");

        for val in &dequantized {
            assert!((val.abs()) < 0.001, "Expected zero, got {}", val);
        }
    }

    #[test]
    fn test_quantize_shape_mismatch() {
        let data: Vec<f32> = vec![1.0; 10];
        let shape = vec![20]; // Wrong shape

        let result = quantize(&data, &shape, QuantType::Q8_0);
        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_wrong_type() {
        let data: Vec<f32> = vec![1.0; 32];
        let shape = vec![32];

        let mut quantized = quantize(&data, &shape, QuantType::Q8_0).expect("quantize");
        quantized.quant_type = QuantType::Q4_0; // Wrong type

        let result = Q8_0Quantizer.dequantize(&quantized);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantization_mse() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.1, 2.1, 3.1, 4.1];

        let mse = quantization_mse(&a, &b);
        assert!((mse - 0.01).abs() < 0.001);
    }

    #[test]
    fn test_quantization_mse_empty() {
        let mse = quantization_mse(&[], &[]);
        assert!(mse.is_nan());
    }

    #[test]
    fn test_quantization_mse_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];

        let mse = quantization_mse(&a, &b);
        assert!(mse.is_nan());
    }

    #[test]
    fn test_quantization_info_default() {
        let info = QuantizationInfo::default();
        assert_eq!(info.quant_type, QuantType::Q8_0);
        assert_eq!(info.calibration_method, "minmax");
        assert_eq!(info.original_dtype, "f32");
    }

    #[test]
    fn test_q8_0_negative_values() {
        let data: Vec<f32> = vec![-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 7.0];
        let shape = vec![8];

        let quantized = quantize(&data, &shape, QuantType::Q8_0).expect("quantize");
        let dequantized = dequantize(&quantized).expect("dequantize");

        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            assert!(
                (orig - deq).abs() < 0.1,
                "Values differ: {} vs {}",
                orig,
                deq
            );
        }
    }

    #[test]
    fn test_q4_0_exact_block_boundary() {
        // Test exactly 32 elements (one block)
        let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let shape = vec![32];

        let quantized = quantize(&data, &shape, QuantType::Q4_0).expect("quantize");
        assert_eq!(quantized.num_blocks(), 1);

        let dequantized = dequantize(&quantized).expect("dequantize");
        assert_eq!(dequantized.len(), 32);
    }

    #[test]
    fn test_q8_0_multidimensional_shape() {
        let data: Vec<f32> = (0..96).map(|i| i as f32 * 0.01).collect();
        let shape = vec![4, 24]; // 4x24 = 96 elements

        let quantized = quantize(&data, &shape, QuantType::Q8_0).expect("quantize");
        assert_eq!(quantized.shape, vec![4, 24]);
        assert_eq!(quantized.num_elements(), 96);

        let dequantized = dequantize(&quantized).expect("dequantize");
        assert_eq!(dequantized.len(), 96);
    }
}
