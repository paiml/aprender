use super::*;
use proptest::prelude::*;

proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(100))]

    #[test]
    fn prop_falsify_arbitrary_f32_tensors_roundtrip(
        n_tensors in 1usize..6,
        n_elements in 1usize..64,
    ) {
        let tensors: Vec<GgufTensor> = (0..n_tensors)
            .map(|i| {
                let data: Vec<f32> = (0..n_elements)
                    .map(|j| (i * n_elements + j) as f32 * 0.1)
                    .collect();
                GgufTensor {
                    name: format!("tensor.{i}"),
                    shape: vec![n_elements as u64],
                    dtype: GgmlType::F32,
                    data: bytemuck::cast_slice(&data).to_vec(),
                }
            })
            .collect();

        let gguf_bytes = write_gguf(&tensors, &[]);
        let summary = verify_gguf(&gguf_bytes).expect("operation should succeed");

        prop_assert_eq!(summary.tensor_count, n_tensors as u64);
        prop_assert_eq!(summary.tensors.len(), n_tensors);

        // Verify every tensor's name, shape, dtype
        for (i, info) in summary.tensors.iter().enumerate() {
            prop_assert_eq!(&info.name, &format!("tensor.{i}"));
            prop_assert_eq!(&info.shape, &vec![n_elements as u64]);
            prop_assert_eq!(info.dtype, 0); // F32
        }

        // Verify actual f32 data roundtrips
        let data_start = find_data_section_start(&gguf_bytes, &summary);
        for (i, info) in summary.tensors.iter().enumerate() {
            let recovered = extract_f32_tensor_data(&gguf_bytes, data_start, info, n_elements);
            for (j, &val) in recovered.iter().enumerate() {
                let expected = (i * n_elements + j) as f32 * 0.1;
                prop_assert!(
                    (val - expected).abs() < 1e-6,
                    "tensor {i} element {j}: expected {expected}, got {val}"
                );
            }
        }
    }

    #[test]
    fn prop_falsify_metadata_count_always_matches(
        n_metadata in 0usize..8,
    ) {
        let metadata: Vec<(String, GgufValue)> = (0..n_metadata)
            .map(|i| (format!("key.{i}"), GgufValue::Uint32(i as u32)))
            .collect();
        let gguf_bytes = write_gguf(&[], &metadata);
        let summary = verify_gguf(&gguf_bytes).expect("operation should succeed");
        prop_assert_eq!(summary.metadata_count, n_metadata as u64);
        prop_assert_eq!(summary.tensor_count, 0);
    }

    #[test]
    fn prop_falsify_q4_0_roundtrip_preserves_dtype(
        n_elements in 1usize..128,
    ) {
        let data: Vec<f32> = vec![0.42; n_elements];
        let (bytes, dtype) = quantize_to_gguf_bytes(&data, GgufQuantization::Q4_0);
        let tensors = vec![GgufTensor {
            name: "q4".into(),
            shape: vec![n_elements as u64],
            dtype,
            data: bytes,
        }];
        let gguf_bytes = write_gguf(&tensors, &[]);
        let summary = verify_gguf(&gguf_bytes).expect("operation should succeed");
        prop_assert_eq!(summary.tensors[0].dtype, 2); // Q4_0
        prop_assert_eq!(&summary.tensors[0].shape, &vec![n_elements as u64]);
    }

    #[test]
    fn prop_falsify_q8_0_roundtrip_preserves_dtype(
        n_elements in 1usize..128,
    ) {
        let data: Vec<f32> = vec![0.42; n_elements];
        let (bytes, dtype) = quantize_to_gguf_bytes(&data, GgufQuantization::Q8_0);
        let tensors = vec![GgufTensor {
            name: "q8".into(),
            shape: vec![n_elements as u64],
            dtype,
            data: bytes,
        }];
        let gguf_bytes = write_gguf(&tensors, &[]);
        let summary = verify_gguf(&gguf_bytes).expect("operation should succeed");
        prop_assert_eq!(summary.tensors[0].dtype, 8); // Q8_0
        prop_assert_eq!(&summary.tensors[0].shape, &vec![n_elements as u64]);
    }

    #[test]
    fn prop_falsify_gguf_header_always_valid(
        n_tensors in 0usize..5,
        n_metadata in 0usize..5,
        n_elements in 1usize..32,
    ) {
        let metadata: Vec<(String, GgufValue)> = (0..n_metadata)
            .map(|i| (format!("m.{i}"), GgufValue::String(format!("v{i}"))))
            .collect();
        let tensors: Vec<GgufTensor> = (0..n_tensors)
            .map(|i| GgufTensor {
                name: format!("t.{i}"),
                shape: vec![n_elements as u64],
                dtype: GgmlType::F32,
                data: vec![0u8; n_elements * 4],
            })
            .collect();

        let gguf_bytes = write_gguf(&tensors, &metadata);

        // Header must always be valid
        prop_assert_eq!(&gguf_bytes[0..4], b"GGUF");
        prop_assert_eq!(
            u32::from_le_bytes(gguf_bytes[4..8].try_into().expect("conversion should succeed")),
            3
        );
        prop_assert_eq!(
            u64::from_le_bytes(gguf_bytes[8..16].try_into().expect("conversion should succeed")),
            n_tensors as u64
        );
        prop_assert_eq!(
            u64::from_le_bytes(gguf_bytes[16..24].try_into().expect("conversion should succeed")),
            n_metadata as u64
        );

        // Must verify cleanly
        let summary = verify_gguf(&gguf_bytes).expect("operation should succeed");
        prop_assert_eq!(summary.version, 3);
        prop_assert_eq!(summary.tensor_count, n_tensors as u64);
        prop_assert_eq!(summary.metadata_count, n_metadata as u64);
    }

    #[test]
    fn prop_falsify_tensor_sort_always_alphabetical(
        seed in 0u64..1000,
    ) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        // Generate deterministic pseudo-random names from seed
        let names: Vec<String> = (0..20).map(|i| {
            let mut h = DefaultHasher::new();
            (seed, i).hash(&mut h);
            format!("tensor_{:016x}", h.finish())
        }).collect();

        let mut weights =
            crate::hf_pipeline::export::weights::ModelWeights::new();
        for (i, name) in names.iter().enumerate() {
            weights.add_tensor(name.clone(), vec![i as f32], vec![1]);
        }

        let dir = tempfile::tempdir().expect("temp file creation should succeed");
        let exporter = crate::hf_pipeline::export::exporter::Exporter::new()
            .output_dir(dir.path())
            .include_metadata(false);
        exporter
            .export(
                &weights,
                crate::hf_pipeline::export::format::ExportFormat::GGUF,
                "sort.gguf",
            )
            .expect("operation should succeed");

        let file_data = std::fs::read(dir.path().join("sort.gguf")).expect("file read should succeed");
        let summary = verify_gguf(&file_data).expect("operation should succeed");

        let mut sorted_names = names.clone();
        sorted_names.sort();

        prop_assert_eq!(summary.tensors.len(), 20);
        for (i, expected_name) in sorted_names.iter().enumerate() {
            prop_assert_eq!(&summary.tensors[i].name, expected_name);
        }
    }
}
