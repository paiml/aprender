//! GGUF quantization helpers
//!
//! Provides entrenar's quantization enum and byte-encoding functions that bridge
//! entrenar's Q4_0/Q8_0 quant structs to raw GGUF block bytes. Binary GGUF
//! serialization is delegated to `aprender::format::gguf::export_tensors_to_gguf`.

use aprender::format::gguf::GgmlType;

use crate::quant::{GGUF_BLOCK_SIZE, Q4_0, Q8_0};

/// GGUF quantization mode for export
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufQuantization {
    /// No quantization — store as F32
    None,
    /// Quantize to Q4_0 (4-bit, 32-element blocks)
    Q4_0,
    /// Quantize to Q8_0 (8-bit, 32-element blocks)
    Q8_0,
}

/// Quantize f32 data according to `quant` mode and return raw GGUF bytes + dtype.
///
/// For `GgufQuantization::None`, returns the f32 data as little-endian bytes with `GgmlType::F32`.
/// For Q4_0/Q8_0, quantizes via entrenar's quant module and encodes to GGUF block format.
pub fn quantize_to_gguf_bytes(data: &[f32], quant: GgufQuantization) -> (Vec<u8>, GgmlType) {
    contract_pre_quantize!(data);
    match quant {
        GgufQuantization::None => {
            let bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
            (bytes, GgmlType::F32)
        }
        GgufQuantization::Q4_0 => {
            let quantized = Q4_0::quantize(data);
            (encode_q4_0_blocks(&quantized), GgmlType::Q4_0)
        }
        GgufQuantization::Q8_0 => {
            let quantized = Q8_0::quantize(data);
            (encode_q8_0_blocks(&quantized), GgmlType::Q8_0)
        }
    };
    contract_post_quantize_precision_bound!(&result);
    result
}

/// Encode Q4_0 quantized data into GGUF binary block format
/// Each block: f16 scale (2 bytes) + 16 bytes packed 4-bit data = 18 bytes
fn encode_q4_0_blocks(q: &Q4_0) -> Vec<u8> {
    let num_blocks = q.num_blocks();
    let mut bytes = Vec::with_capacity(num_blocks * 18);

    for block_idx in 0..num_blocks {
        // Scale as f16
        let scale_f16 = half::f16::from_f32(q.scales[block_idx]);
        bytes.extend_from_slice(&scale_f16.to_le_bytes());

        // 16 bytes of packed 4-bit data
        let data_start = block_idx * 16;
        let data_end = (data_start + 16).min(q.data.len());
        bytes.extend_from_slice(&q.data[data_start..data_end]);

        // Pad if the last block is short
        let pad = 16 - (data_end - data_start);
        bytes.extend(std::iter::repeat_n(0u8, pad));
    }

    bytes
}

/// Encode Q8_0 quantized data into GGUF binary block format
/// Each block: f16 scale (2 bytes) + 32 bytes i8 data = 34 bytes
fn encode_q8_0_blocks(q: &Q8_0) -> Vec<u8> {
    let num_blocks = q.num_blocks();
    let mut bytes = Vec::with_capacity(num_blocks * 34);

    for block_idx in 0..num_blocks {
        // Scale as f16
        let scale_f16 = half::f16::from_f32(q.scales[block_idx]);
        bytes.extend_from_slice(&scale_f16.to_le_bytes());

        // 32 bytes of i8 data
        let data_start = block_idx * GGUF_BLOCK_SIZE;
        let data_end = (data_start + GGUF_BLOCK_SIZE).min(q.data.len());
        let block_data = &q.data[data_start..data_end];

        // Cast i8 slice to u8 slice for extending
        for &val in block_data {
            bytes.push(val as u8);
        }

        // Pad incomplete blocks
        let pad = GGUF_BLOCK_SIZE - block_data.len();
        bytes.extend(std::iter::repeat_n(0u8, pad));
    }

    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_encode_q4_0_block_size() {
        let values: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let q = Q4_0::quantize(&values);
        let bytes = encode_q4_0_blocks(&q);
        // 1 block * 18 bytes
        assert_eq!(bytes.len(), 18);
    }

    #[test]
    fn test_encode_q8_0_block_size() {
        let values: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let q = Q8_0::quantize(&values);
        let bytes = encode_q8_0_blocks(&q);
        // 1 block * 34 bytes
        assert_eq!(bytes.len(), 34);
    }

    #[test]
    fn test_quantize_to_gguf_bytes_none() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let (bytes, dtype) = quantize_to_gguf_bytes(&data, GgufQuantization::None);
        assert_eq!(dtype, GgmlType::F32);
        assert_eq!(bytes.len(), 16); // 4 floats * 4 bytes
                                     // Verify first float roundtrips
        let val = f32::from_le_bytes(bytes[0..4].try_into().expect("conversion should succeed"));
        assert!((val - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_quantize_to_gguf_bytes_q4_0() {
        let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (bytes, dtype) = quantize_to_gguf_bytes(&data, GgufQuantization::Q4_0);
        assert_eq!(dtype, GgmlType::Q4_0);
        assert_eq!(bytes.len(), 18); // 1 block
    }

    #[test]
    fn test_quantize_to_gguf_bytes_q8_0() {
        let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (bytes, dtype) = quantize_to_gguf_bytes(&data, GgufQuantization::Q8_0);
        assert_eq!(dtype, GgmlType::Q8_0);
        assert_eq!(bytes.len(), 34); // 1 block
    }

    // =====================================================================
    // Falsification: edge cases for quantize_to_gguf_bytes
    // =====================================================================

    #[test]
    fn test_falsify_quantize_empty_data_none() {
        let (bytes, dtype) = quantize_to_gguf_bytes(&[], GgufQuantization::None);
        assert_eq!(dtype, GgmlType::F32);
        assert!(bytes.is_empty());
    }

    #[test]
    fn test_falsify_quantize_empty_data_q4_0() {
        let (bytes, dtype) = quantize_to_gguf_bytes(&[], GgufQuantization::Q4_0);
        assert_eq!(dtype, GgmlType::Q4_0);
        assert!(
            bytes.is_empty(),
            "empty input must produce empty output, got {} bytes",
            bytes.len()
        );
    }

    #[test]
    fn test_falsify_quantize_empty_data_q8_0() {
        let (bytes, dtype) = quantize_to_gguf_bytes(&[], GgufQuantization::Q8_0);
        assert_eq!(dtype, GgmlType::Q8_0);
        assert!(
            bytes.is_empty(),
            "empty input must produce empty output, got {} bytes",
            bytes.len()
        );
    }

    #[test]
    fn test_falsify_quantize_single_element_q4_0() {
        let (bytes, dtype) = quantize_to_gguf_bytes(&[42.0], GgufQuantization::Q4_0);
        assert_eq!(dtype, GgmlType::Q4_0);
        // 1 element → 1 block → 18 bytes (2 scale + 16 packed data)
        assert_eq!(bytes.len(), 18);
    }

    #[test]
    fn test_falsify_quantize_single_element_q8_0() {
        let (bytes, dtype) = quantize_to_gguf_bytes(&[42.0], GgufQuantization::Q8_0);
        assert_eq!(dtype, GgmlType::Q8_0);
        // 1 element → 1 block → 34 bytes (2 scale + 32 i8 data)
        assert_eq!(bytes.len(), 34);
    }

    #[test]
    fn test_falsify_quantize_33_elements_q4_0() {
        // 33 elements = 2 blocks (32 + 1), second block is mostly padding
        let data: Vec<f32> = (0..33).map(|i| i as f32 * 0.1).collect();
        let (bytes, dtype) = quantize_to_gguf_bytes(&data, GgufQuantization::Q4_0);
        assert_eq!(dtype, GgmlType::Q4_0);
        assert_eq!(bytes.len(), 2 * 18); // exactly 2 blocks
    }

    #[test]
    fn test_falsify_quantize_33_elements_q8_0() {
        let data: Vec<f32> = (0..33).map(|i| i as f32 * 0.1).collect();
        let (bytes, dtype) = quantize_to_gguf_bytes(&data, GgufQuantization::Q8_0);
        assert_eq!(dtype, GgmlType::Q8_0);
        assert_eq!(bytes.len(), 2 * 34); // exactly 2 blocks
    }

    #[test]
    fn test_falsify_quantize_63_elements_q4_0() {
        // 63 elements = 2 blocks (32 + 31)
        let data: Vec<f32> = (0..63).map(|i| i as f32 * 0.01).collect();
        let (bytes, dtype) = quantize_to_gguf_bytes(&data, GgufQuantization::Q4_0);
        assert_eq!(dtype, GgmlType::Q4_0);
        assert_eq!(bytes.len(), 2 * 18);
    }

    #[test]
    fn test_falsify_quantize_all_zeros_q4_0() {
        // All zeros — scale should be 0, data all zeros
        let data = [0.0f32; 64];
        let (bytes, dtype) = quantize_to_gguf_bytes(&data, GgufQuantization::Q4_0);
        assert_eq!(dtype, GgmlType::Q4_0);
        assert_eq!(bytes.len(), 2 * 18);
        // Scale bytes (first 2 of each 18-byte block) should encode 0.0
        let scale0 = half::f16::from_le_bytes([bytes[0], bytes[1]]);
        assert_eq!(scale0.to_f32(), 0.0, "scale for zero data must be 0");
    }

    #[test]
    fn test_falsify_quantize_all_zeros_q8_0() {
        let data = [0.0f32; 32];
        let (bytes, dtype) = quantize_to_gguf_bytes(&data, GgufQuantization::Q8_0);
        assert_eq!(dtype, GgmlType::Q8_0);
        assert_eq!(bytes.len(), 34);
        let scale0 = half::f16::from_le_bytes([bytes[0], bytes[1]]);
        assert_eq!(scale0.to_f32(), 0.0, "scale for zero data must be 0");
    }

    #[test]
    fn test_falsify_quantize_extreme_range_q4_0() {
        // Values spanning huge range — tests f16 scale saturation
        let mut data = vec![0.0f32; 32];
        data[0] = 1e30;
        data[1] = -1e30;
        let (bytes, dtype) = quantize_to_gguf_bytes(&data, GgufQuantization::Q4_0);
        assert_eq!(dtype, GgmlType::Q4_0);
        assert_eq!(bytes.len(), 18);
        // Scale should be finite or infinity but not NaN
        let scale = half::f16::from_le_bytes([bytes[0], bytes[1]]);
        assert!(!scale.to_f32().is_nan(), "scale must not be NaN for extreme values");
    }

    #[test]
    fn test_falsify_quantize_extreme_range_q8_0() {
        let mut data = vec![0.0f32; 32];
        data[0] = 1e30;
        data[1] = -1e30;
        let (bytes, dtype) = quantize_to_gguf_bytes(&data, GgufQuantization::Q8_0);
        assert_eq!(dtype, GgmlType::Q8_0);
        assert_eq!(bytes.len(), 34);
        let scale = half::f16::from_le_bytes([bytes[0], bytes[1]]);
        assert!(!scale.to_f32().is_nan(), "scale must not be NaN for extreme values");
    }

    #[test]
    fn test_falsify_quantize_f32_exact_byte_layout() {
        // Verify F32 mode produces exact little-endian layout
        let data = [std::f32::consts::PI, std::f32::consts::E];
        let (bytes, dtype) = quantize_to_gguf_bytes(&data, GgufQuantization::None);
        assert_eq!(dtype, GgmlType::F32);
        assert_eq!(bytes.len(), 8);
        let pi_bytes = std::f32::consts::PI.to_le_bytes();
        let e_bytes = std::f32::consts::E.to_le_bytes();
        assert_eq!(&bytes[0..4], &pi_bytes);
        assert_eq!(&bytes[4..8], &e_bytes);
    }

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(50))]

        #[test]
        fn prop_q4_0_encode_correct_block_count(
            n_elements in 1usize..256,
        ) {
            let data: Vec<f32> = vec![1.0; n_elements];
            let q = Q4_0::quantize(&data);
            let bytes = encode_q4_0_blocks(&q);
            let expected_blocks = n_elements.div_ceil(GGUF_BLOCK_SIZE);
            prop_assert_eq!(bytes.len(), expected_blocks * 18);
        }

        #[test]
        fn prop_q8_0_encode_correct_block_count(
            n_elements in 1usize..256,
        ) {
            let data: Vec<f32> = vec![1.0; n_elements];
            let q = Q8_0::quantize(&data);
            let bytes = encode_q8_0_blocks(&q);
            let expected_blocks = n_elements.div_ceil(GGUF_BLOCK_SIZE);
            prop_assert_eq!(bytes.len(), expected_blocks * 34);
        }

        #[test]
        fn prop_falsify_quantize_none_preserves_all_bytes(
            n_elements in 1usize..128,
        ) {
            let data: Vec<f32> = (0..n_elements).map(|i| i as f32 * 0.7 - 50.0).collect();
            let (bytes, dtype) = quantize_to_gguf_bytes(&data, GgufQuantization::None);
            prop_assert_eq!(dtype, GgmlType::F32);
            prop_assert_eq!(bytes.len(), n_elements * 4);
            // Verify every float
            for (i, &expected) in data.iter().enumerate() {
                let actual = f32::from_le_bytes(bytes[i*4..(i+1)*4].try_into().expect("conversion should succeed"));
                prop_assert!(
                    (actual - expected).abs() < f32::EPSILON,
                    "element {i}: expected {expected}, got {actual}"
                );
            }
        }

        #[test]
        fn prop_falsify_quantize_q4_0_byte_size_invariant(
            n_elements in 1usize..512,
        ) {
            let data: Vec<f32> = vec![0.5; n_elements];
            let (bytes, dtype) = quantize_to_gguf_bytes(&data, GgufQuantization::Q4_0);
            prop_assert_eq!(dtype, GgmlType::Q4_0);
            let expected_blocks = n_elements.div_ceil(GGUF_BLOCK_SIZE);
            prop_assert_eq!(bytes.len(), expected_blocks * 18);
        }

        #[test]
        fn prop_falsify_quantize_q8_0_byte_size_invariant(
            n_elements in 1usize..512,
        ) {
            let data: Vec<f32> = vec![0.5; n_elements];
            let (bytes, dtype) = quantize_to_gguf_bytes(&data, GgufQuantization::Q8_0);
            prop_assert_eq!(dtype, GgmlType::Q8_0);
            let expected_blocks = n_elements.div_ceil(GGUF_BLOCK_SIZE);
            prop_assert_eq!(bytes.len(), expected_blocks * 34);
        }
    }
}
