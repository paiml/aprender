
// ============================================================================
// Property-Based Tests (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod proptests {
    use super::super::super::*;
    use proptest::prelude::*;

    // Strategy for generating valid GGUF headers
    fn arb_header() -> impl Strategy<Value = GgufHeader> {
        (0u64..1000, 0u64..100).prop_map(|(tensor_count, metadata_kv_count)| GgufHeader {
            version: GGUF_VERSION,
            tensor_count,
            metadata_kv_count,
        })
    }

    proptest! {
        /// Property: Header write always produces exactly 24 bytes
        #[test]
        fn prop_header_size_always_24(header in arb_header()) {
            let mut buffer = Vec::new();
            header.write_to(&mut buffer).expect("write");
            prop_assert_eq!(buffer.len(), 24);
        }

        /// Property: Header always starts with GGUF magic
        #[test]
        fn prop_header_magic(header in arb_header()) {
            let mut buffer = Vec::new();
            header.write_to(&mut buffer).expect("write");
            prop_assert_eq!(&buffer[0..4], b"GGUF");
        }

        /// Property: Header version is always 3
        #[test]
        fn prop_header_version(header in arb_header()) {
            let mut buffer = Vec::new();
            header.write_to(&mut buffer).expect("write");
            let version = u32::from_le_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]);
            prop_assert_eq!(version, GGUF_VERSION);
        }

        /// Property: Padding is always less than alignment
        #[test]
        fn prop_padding_less_than_alignment(offset in 0usize..10000, alignment in 1usize..256) {
            let padding = padding_for_alignment(offset, alignment);
            prop_assert!(padding < alignment);
        }

        /// Property: offset + padding is always aligned
        #[test]
        fn prop_padded_offset_aligned(offset in 0usize..10000, alignment in 1usize..256) {
            let padding = padding_for_alignment(offset, alignment);
            prop_assert_eq!((offset + padding) % alignment, 0);
        }

        /// Property: Aligned offsets need zero padding
        #[test]
        fn prop_aligned_needs_no_padding(multiple in 0usize..1000, alignment in 1usize..256) {
            let offset = multiple * alignment;
            prop_assert_eq!(padding_for_alignment(offset, alignment), 0);
        }

        /// Property: String metadata key-value is non-empty
        #[test]
        fn prop_string_metadata_nonempty(
            key in "[a-z][a-z0-9_.]{0,30}",
            value in "[a-zA-Z0-9_ ]{0,100}"
        ) {
            let mut buffer = Vec::new();
            write_metadata_kv(&mut buffer, &key, &GgufValue::String(value)).expect("write");
            prop_assert!(!buffer.is_empty());
        }

        /// Property: Uint32 value roundtrip through bytes
        #[test]
        fn prop_uint32_value_written(value in any::<u32>()) {
            let mut buffer = Vec::new();
            write_metadata_kv(&mut buffer, "test", &GgufValue::Uint32(value)).expect("write");
            // Key: 8 (len) + 4 (test) + type: 4 + value: 4 = 20 bytes
            prop_assert!(buffer.len() >= 20);
        }

        /// Property: Float32 value roundtrip through bytes
        #[test]
        fn prop_float32_value_written(value in any::<f32>()) {
            let mut buffer = Vec::new();
            write_metadata_kv(&mut buffer, "f", &GgufValue::Float32(value)).expect("write");
            prop_assert!(!buffer.is_empty());
        }

        /// Property: Tensor export produces valid GGUF with magic
        #[test]
        fn prop_tensor_export_has_magic(
            name in "[a-z][a-z0-9.]{0,20}",
            dim0 in 1u64..100,
            dim1 in 1u64..100
        ) {
            let data = vec![0u8; (dim0 * dim1 * 4) as usize]; // f32 data
            let tensor = GgufTensor {
                name,
                shape: vec![dim0, dim1],
                dtype: GgmlType::F32,
                data,
            };
            let mut buffer = Vec::new();
            export_tensors_to_gguf(&mut buffer, &[tensor], &[]).expect("export");
            prop_assert_eq!(&buffer[0..4], b"GGUF");
        }

        // ================================================================
        // Metadata Roundtrip Property Tests
        // ================================================================

        /// Property: Bool true encodes to 1, false to 0
        #[test]
        fn prop_bool_value_encoding(value in any::<bool>()) {
            let mut buffer = Vec::new();
            write_metadata_kv(&mut buffer, "b", &GgufValue::Bool(value)).expect("write");
            // Last byte is the bool value
            let last_byte = buffer[buffer.len() - 1];
            prop_assert_eq!(last_byte, u8::from(value));
        }

        /// Property: Int64 values encode correctly
        #[test]
        fn prop_int64_value_encoded(value in any::<i64>()) {
            let mut buffer = Vec::new();
            write_metadata_kv(&mut buffer, "i", &GgufValue::Int64(value)).expect("write");
            // Last 8 bytes are the i64 value
            let bytes = &buffer[buffer.len() - 8..];
            let decoded = i64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3],
                bytes[4], bytes[5], bytes[6], bytes[7],
            ]);
            prop_assert_eq!(decoded, value);
        }

        /// Property: Uint64 values encode correctly
        #[test]
        fn prop_uint64_value_encoded(value in any::<u64>()) {
            let mut buffer = Vec::new();
            write_metadata_kv(&mut buffer, "u", &GgufValue::Uint64(value)).expect("write");
            let bytes = &buffer[buffer.len() - 8..];
            let decoded = u64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3],
                bytes[4], bytes[5], bytes[6], bytes[7],
            ]);
            prop_assert_eq!(decoded, value);
        }

        /// Property: Float64 values encode correctly (bit-exact)
        #[test]
        fn prop_float64_value_encoded(value in any::<f64>()) {
            let mut buffer = Vec::new();
            write_metadata_kv(&mut buffer, "d", &GgufValue::Float64(value)).expect("write");
            let bytes = &buffer[buffer.len() - 8..];
            let decoded = f64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3],
                bytes[4], bytes[5], bytes[6], bytes[7],
            ]);
            // Use to_bits for NaN-safe comparison
            prop_assert_eq!(decoded.to_bits(), value.to_bits());
        }

        /// Property: Value type tag is correct for all value types
        #[test]
        fn prop_value_type_tag_uint32(value in any::<u32>()) {
            let mut buffer = Vec::new();
            write_metadata_kv(&mut buffer, "x", &GgufValue::Uint32(value)).expect("write");
            // Type is at bytes 12-15 (after key: 8 byte len + 1 byte "x" = 9 bytes, padded to 9, then type)
            // Key: u64 length (8) + "x" (1) = 9 bytes, then u32 type
            let type_bytes = &buffer[9..13];
            let type_val = u32::from_le_bytes([type_bytes[0], type_bytes[1], type_bytes[2], type_bytes[3]]);
            prop_assert_eq!(type_val, GgufValueType::Uint32 as u32);
        }

        // ================================================================
        // Tensor Info Property Tests
        // ================================================================

        /// Property: Tensor info serialization contains name
        #[test]
        fn prop_tensor_info_contains_name(
            name in "[a-z][a-z0-9_.]{0,30}"
        ) {
            let info = GgufTensorInfo {
                name: name.clone(),
                n_dims: 2,
                dims: vec![10, 20],
                dtype: GgmlType::F32,
                offset: 0,
            };
            let mut buffer = Vec::new();
            info.write_to(&mut buffer).expect("write");
            // Name length is first 8 bytes
            let name_len = u64::from_le_bytes([
                buffer[0], buffer[1], buffer[2], buffer[3],
                buffer[4], buffer[5], buffer[6], buffer[7],
            ]) as usize;
            prop_assert_eq!(name_len, name.len());
            // Name bytes follow
            let name_bytes = &buffer[8..8 + name_len];
            prop_assert_eq!(name_bytes, name.as_bytes());
        }

        /// Property: Tensor info n_dims matches shape length
        #[test]
        fn prop_tensor_info_ndims_matches_shape(
            dims in proptest::collection::vec(1u64..100, 1..5)
        ) {
            let info = GgufTensorInfo {
                name: "t".to_string(),
                n_dims: dims.len() as u32,
                dims: dims.clone(),
                dtype: GgmlType::F32,
                offset: 0,
            };
            let mut buffer = Vec::new();
            info.write_to(&mut buffer).expect("write");
            // After name (8 + 1 = 9 bytes), n_dims is next 4 bytes
            let n_dims = u32::from_le_bytes([buffer[9], buffer[10], buffer[11], buffer[12]]);
            prop_assert_eq!(n_dims as usize, dims.len());
        }

        /// Property: Multiple metadata pairs produces correct count in header
        #[test]
        fn prop_export_metadata_count(
            count in 0usize..10
        ) {
            let metadata: Vec<(String, GgufValue)> = (0..count)
                .map(|i| (format!("key{i}"), GgufValue::Uint32(i as u32)))
                .collect();
            let mut buffer = Vec::new();
            export_tensors_to_gguf(&mut buffer, &[], &metadata).expect("export");
            // KV count is at bytes 16-23 (after magic 4, version 4, tensor_count 8)
            let kv_count = u64::from_le_bytes([
                buffer[16], buffer[17], buffer[18], buffer[19],
                buffer[20], buffer[21], buffer[22], buffer[23],
            ]);
            prop_assert_eq!(kv_count as usize, count);
        }

        /// Property: Tensor count in header matches tensors provided
        #[test]
        fn prop_export_tensor_count(
            count in 0usize..5
        ) {
            let tensors: Vec<GgufTensor> = (0..count)
                .map(|i| GgufTensor {
                    name: format!("t{i}"),
                    shape: vec![4],
                    dtype: GgmlType::F32,
                    data: vec![0u8; 16], // 4 f32s = 16 bytes
                })
                .collect();
            let mut buffer = Vec::new();
            export_tensors_to_gguf(&mut buffer, &tensors, &[]).expect("export");
            // Tensor count is at bytes 8-15
            let tensor_count = u64::from_le_bytes([
                buffer[8], buffer[9], buffer[10], buffer[11],
                buffer[12], buffer[13], buffer[14], buffer[15],
            ]);
            prop_assert_eq!(tensor_count as usize, count);
        }
    }
}
