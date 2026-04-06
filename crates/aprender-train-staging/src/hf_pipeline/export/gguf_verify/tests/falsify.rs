use super::*;

#[test]
fn test_falsify_f32_tensor_data_survives_roundtrip() {
    // Adversarial: verify actual float bytes, not just structural metadata
    let original: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.0137).collect();
    let tensors = vec![GgufTensor {
        name: "weights".into(),
        shape: vec![16, 16],
        dtype: GgmlType::F32,
        data: bytemuck::cast_slice(&original).to_vec(),
    }];
    let data = write_gguf(&tensors, &[]);
    let summary = verify_gguf(&data).expect("operation should succeed");
    let data_start = find_data_section_start(&data, &summary);
    let recovered = extract_f32_tensor_data(&data, data_start, &summary.tensors[0], 256);
    assert_eq!(original, recovered, "f32 tensor data must survive roundtrip exactly");
}

#[test]
fn test_falsify_special_float_values_survive() {
    // Edge case floats: 0, -0, subnormals, max, min, inf, -inf, NaN
    let special: Vec<f32> = vec![
        0.0,
        -0.0,
        f32::MIN_POSITIVE, // smallest positive normal
        f32::EPSILON,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        // Pad to block size of 32
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        17.0,
        18.0,
        19.0,
        20.0,
        21.0,
        22.0,
        23.0,
        24.0,
    ];
    let tensors = vec![GgufTensor {
        name: "special".into(),
        shape: vec![special.len() as u64],
        dtype: GgmlType::F32,
        data: bytemuck::cast_slice(&special).to_vec(),
    }];
    let data = write_gguf(&tensors, &[]);
    let summary = verify_gguf(&data).expect("operation should succeed");
    let data_start = find_data_section_start(&data, &summary);
    let recovered = extract_f32_tensor_data(&data, data_start, &summary.tensors[0], special.len());

    for (i, (&orig, &rec)) in special.iter().zip(recovered.iter()).enumerate() {
        if orig.is_nan() {
            assert!(rec.is_nan(), "index {i}: NaN must survive roundtrip");
        } else {
            assert_eq!(
                orig.to_bits(),
                rec.to_bits(),
                "index {i}: bitwise equality failed for {orig}"
            );
        }
    }
}

#[test]
fn test_falsify_multi_tensor_ordering_preserved() {
    // Verify tensor order is deterministic and matches insertion order
    let names: Vec<String> = (0..8).map(|i| format!("layer.{i}.weight")).collect();
    let tensors: Vec<GgufTensor> = names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let val = (i + 1) as f32;
            GgufTensor {
                name: name.clone(),
                shape: vec![1],
                dtype: GgmlType::F32,
                data: bytemuck::cast_slice(&[val]).to_vec(),
            }
        })
        .collect();
    let data = write_gguf(&tensors, &[]);
    let summary = verify_gguf(&data).expect("operation should succeed");

    assert_eq!(summary.tensor_count, 8);
    for (i, info) in summary.tensors.iter().enumerate() {
        assert_eq!(info.name, names[i], "tensor {i} name mismatch");
        assert_eq!(info.dtype, 0, "tensor {i} should be F32");
    }

    // Verify actual data values
    let data_start = find_data_section_start(&data, &summary);
    for (i, info) in summary.tensors.iter().enumerate() {
        let recovered = extract_f32_tensor_data(&data, data_start, info, 1);
        let expected = (i + 1) as f32;
        assert!(
            (recovered[0] - expected).abs() < f32::EPSILON,
            "tensor {i} data mismatch: expected {expected}, got {}",
            recovered[0]
        );
    }
}

#[test]
fn test_falsify_mixed_metadata_types_roundtrip() {
    // All metadata types in one file -- verify none corrupt the parse
    let metadata = vec![
        ("str.key".into(), GgufValue::String("hello world".into())),
        ("u32.key".into(), GgufValue::Uint32(42)),
        ("u64.key".into(), GgufValue::Uint64(u64::MAX)),
        ("f32.key".into(), GgufValue::Float32(std::f32::consts::PI)),
        ("i32.key".into(), GgufValue::Int32(-999)),
        ("bool.key".into(), GgufValue::Bool(true)),
    ];
    let tensors = vec![GgufTensor {
        name: "t".into(),
        shape: vec![4],
        dtype: GgmlType::F32,
        data: bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0]).to_vec(),
    }];
    let data = write_gguf(&tensors, &metadata);
    let summary = verify_gguf(&data).expect("operation should succeed");

    assert_eq!(summary.metadata_count, 6);
    assert_eq!(summary.tensor_count, 1);
    assert_eq!(summary.tensors[0].name, "t");
    assert_eq!(summary.tensors[0].shape, vec![4]);
}

#[test]
fn test_falsify_q4_0_q8_0_f32_mixed_in_single_file() {
    // Mix all three quantization types in one GGUF file
    let f32_data = [1.0f32; 32];
    let (q4_bytes, q4_dtype) = quantize_to_gguf_bytes(&[0.5; 64], GgufQuantization::Q4_0);
    let (q8_bytes, q8_dtype) = quantize_to_gguf_bytes(&[0.3; 32], GgufQuantization::Q8_0);

    let tensors = vec![
        GgufTensor {
            name: "f32_tensor".into(),
            shape: vec![32],
            dtype: GgmlType::F32,
            data: bytemuck::cast_slice(&f32_data).to_vec(),
        },
        GgufTensor { name: "q4_tensor".into(), shape: vec![64], dtype: q4_dtype, data: q4_bytes },
        GgufTensor { name: "q8_tensor".into(), shape: vec![32], dtype: q8_dtype, data: q8_bytes },
    ];
    let data = write_gguf(&tensors, &[]);
    let summary = verify_gguf(&data).expect("operation should succeed");

    assert_eq!(summary.tensor_count, 3);
    assert_eq!(summary.tensors[0].dtype, 0); // F32
    assert_eq!(summary.tensors[1].dtype, 2); // Q4_0
    assert_eq!(summary.tensors[2].dtype, 8); // Q8_0

    // Verify f32 data is intact
    let data_start = find_data_section_start(&data, &summary);
    let recovered = extract_f32_tensor_data(&data, data_start, &summary.tensors[0], 32);
    assert_eq!(recovered, f32_data.to_vec());
}

#[test]
fn test_falsify_long_tensor_name() {
    // Adversarial: 1000-char tensor name
    let long_name = "a".repeat(1000);
    let tensors = vec![GgufTensor {
        name: long_name.clone(),
        shape: vec![1],
        dtype: GgmlType::F32,
        data: bytemuck::cast_slice(&[42.0f32]).to_vec(),
    }];
    let data = write_gguf(&tensors, &[]);
    let summary = verify_gguf(&data).expect("operation should succeed");
    assert_eq!(summary.tensors[0].name, long_name);
}

#[test]
fn test_falsify_high_dimensional_shape() {
    // 5D tensor shape
    let shape = vec![2u64, 3, 4, 5, 6];
    let num_elements: u64 = shape.iter().product();
    let values: Vec<f32> = (0..num_elements).map(|i| i as f32).collect();
    let tensors = vec![GgufTensor {
        name: "5d".into(),
        shape: shape.clone(),
        dtype: GgmlType::F32,
        data: bytemuck::cast_slice(&values).to_vec(),
    }];
    let data = write_gguf(&tensors, &[]);
    let summary = verify_gguf(&data).expect("operation should succeed");
    assert_eq!(summary.tensors[0].shape, shape);

    let data_start = find_data_section_start(&data, &summary);
    let recovered =
        extract_f32_tensor_data(&data, data_start, &summary.tensors[0], num_elements as usize);
    assert_eq!(values, recovered);
}

#[test]
fn test_falsify_exporter_gguf_roundtrip_via_file() {
    // Full pipeline: Exporter.export_gguf() -> read file -> verify_gguf() -> check
    use crate::hf_pipeline::export::{ExportFormat, Exporter, ModelWeights};

    let mut weights = ModelWeights::new();
    weights.add_tensor("attn.q", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    weights.add_tensor("attn.k", vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    weights.metadata.architecture = Some("llama".into());
    weights.metadata.model_name = Some("test-falsify".into());
    weights.metadata.num_params = 8;
    weights.metadata.hidden_size = Some(2);
    weights.metadata.num_layers = Some(1);

    let dir = tempfile::tempdir().expect("temp file creation should succeed");
    let path = dir.path().join("falsify.gguf");

    let exporter = Exporter::new().output_dir(dir.path()).gguf_quantization(GgufQuantization::None);
    let result = exporter
        .export(&weights, ExportFormat::GGUF, "falsify.gguf")
        .expect("operation should succeed");

    assert_eq!(result.num_tensors, 2);
    assert!(result.size_bytes > 0);

    let file_data = std::fs::read(&path).expect("file read should succeed");
    let summary = verify_gguf(&file_data).expect("operation should succeed");

    assert_eq!(summary.version, 3);
    assert_eq!(summary.tensor_count, 2);
    // Metadata: architecture + name + parameter_count + hidden_size + num_layers = 5
    assert_eq!(summary.metadata_count, 5);

    // Tensors should be sorted alphabetically
    assert_eq!(summary.tensors[0].name, "attn.k");
    assert_eq!(summary.tensors[1].name, "attn.q");
    assert_eq!(summary.tensors[0].dtype, 0); // F32
    assert_eq!(summary.tensors[1].dtype, 0);
    assert_eq!(summary.tensors[0].shape, vec![2, 2]);
    assert_eq!(summary.tensors[1].shape, vec![2, 2]);

    // Verify actual data
    let data_start = find_data_section_start(&file_data, &summary);
    let k_data = extract_f32_tensor_data(&file_data, data_start, &summary.tensors[0], 4);
    let q_data = extract_f32_tensor_data(&file_data, data_start, &summary.tensors[1], 4);
    assert_eq!(k_data, vec![5.0, 6.0, 7.0, 8.0]);
    assert_eq!(q_data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_falsify_exporter_q4_0_roundtrip_via_file() {
    use crate::hf_pipeline::export::{ExportFormat, Exporter, ModelWeights};

    let mut weights = ModelWeights::new();
    let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
    weights.add_tensor("quantized_layer", data, vec![64]);

    let dir = tempfile::tempdir().expect("temp file creation should succeed");
    let exporter = Exporter::new().output_dir(dir.path()).gguf_quantization(GgufQuantization::Q4_0);
    let result =
        exporter.export(&weights, ExportFormat::GGUF, "q4.gguf").expect("operation should succeed");
    assert_eq!(result.num_tensors, 1);

    let file_data = std::fs::read(dir.path().join("q4.gguf")).expect("file read should succeed");
    let summary = verify_gguf(&file_data).expect("operation should succeed");
    assert_eq!(summary.tensors[0].dtype, 2); // Q4_0
    assert_eq!(summary.tensors[0].shape, vec![64]);
}

#[test]
fn test_falsify_exporter_q8_0_roundtrip_via_file() {
    use crate::hf_pipeline::export::{ExportFormat, Exporter, ModelWeights};

    let mut weights = ModelWeights::new();
    let data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
    weights.add_tensor("q8_layer", data, vec![128]);

    let dir = tempfile::tempdir().expect("temp file creation should succeed");
    let exporter = Exporter::new().output_dir(dir.path()).gguf_quantization(GgufQuantization::Q8_0);
    let result =
        exporter.export(&weights, ExportFormat::GGUF, "q8.gguf").expect("operation should succeed");
    assert_eq!(result.num_tensors, 1);

    let file_data = std::fs::read(dir.path().join("q8.gguf")).expect("file read should succeed");
    let summary = verify_gguf(&file_data).expect("operation should succeed");
    assert_eq!(summary.tensors[0].dtype, 8); // Q8_0
    assert_eq!(summary.tensors[0].shape, vec![128]);
}

#[test]
fn test_falsify_no_metadata_mode() {
    // Exporter with include_metadata=false must produce 0 metadata entries
    use crate::hf_pipeline::export::{ExportFormat, Exporter, ModelWeights};

    let mut weights = ModelWeights::new();
    weights.add_tensor("w", vec![1.0], vec![1]);
    weights.metadata.architecture = Some("llama".into());
    weights.metadata.model_name = Some("should-not-appear".into());

    let dir = tempfile::tempdir().expect("temp file creation should succeed");
    let exporter = Exporter::new().output_dir(dir.path()).include_metadata(false);
    exporter.export(&weights, ExportFormat::GGUF, "nometa.gguf").expect("operation should succeed");

    let file_data =
        std::fs::read(dir.path().join("nometa.gguf")).expect("file read should succeed");
    let summary = verify_gguf(&file_data).expect("operation should succeed");
    assert_eq!(summary.metadata_count, 0);
    assert_eq!(summary.tensor_count, 1);
}

#[test]
fn test_falsify_minimal_metadata_only_param_count() {
    // All optional metadata fields are None -- only num_params should appear
    use crate::hf_pipeline::export::{ExportFormat, Exporter, ModelWeights};

    let mut weights = ModelWeights::new();
    weights.add_tensor("w", vec![1.0, 2.0], vec![2]);
    weights.metadata.num_params = 2;
    // architecture, model_name, hidden_size, num_layers all None

    let dir = tempfile::tempdir().expect("temp file creation should succeed");
    let exporter = Exporter::new().output_dir(dir.path());
    exporter
        .export(&weights, ExportFormat::GGUF, "minimal.gguf")
        .expect("operation should succeed");

    let file_data =
        std::fs::read(dir.path().join("minimal.gguf")).expect("file read should succeed");
    let summary = verify_gguf(&file_data).expect("operation should succeed");
    // Only general.parameter_count should be present
    assert_eq!(summary.metadata_count, 1);
    assert_eq!(summary.tensor_count, 1);
}

#[test]
fn test_falsify_exporter_alphabetical_tensor_sort() {
    // Tensors added in reverse order must appear alphabetically in GGUF
    use crate::hf_pipeline::export::{ExportFormat, Exporter, ModelWeights};

    let mut weights = ModelWeights::new();
    // Add in reverse alphabetical order
    weights.add_tensor("z_layer", vec![3.0], vec![1]);
    weights.add_tensor("m_layer", vec![2.0], vec![1]);
    weights.add_tensor("a_layer", vec![1.0], vec![1]);

    let dir = tempfile::tempdir().expect("temp file creation should succeed");
    let exporter = Exporter::new().output_dir(dir.path()).include_metadata(false);
    exporter.export(&weights, ExportFormat::GGUF, "sorted.gguf").expect("operation should succeed");

    let file_data =
        std::fs::read(dir.path().join("sorted.gguf")).expect("file read should succeed");
    let summary = verify_gguf(&file_data).expect("operation should succeed");
    assert_eq!(summary.tensor_count, 3);
    assert_eq!(summary.tensors[0].name, "a_layer");
    assert_eq!(summary.tensors[1].name, "m_layer");
    assert_eq!(summary.tensors[2].name, "z_layer");

    // Verify data follows the sorted order (a=1.0, m=2.0, z=3.0)
    let data_start = find_data_section_start(&file_data, &summary);
    let a_data = extract_f32_tensor_data(&file_data, data_start, &summary.tensors[0], 1);
    let m_data = extract_f32_tensor_data(&file_data, data_start, &summary.tensors[1], 1);
    let z_data = extract_f32_tensor_data(&file_data, data_start, &summary.tensors[2], 1);
    assert!((a_data[0] - 1.0).abs() < f32::EPSILON, "a_layer should be 1.0");
    assert!((m_data[0] - 2.0).abs() < f32::EPSILON, "m_layer should be 2.0");
    assert!((z_data[0] - 3.0).abs() < f32::EPSILON, "z_layer should be 3.0");
}

#[test]
fn test_falsify_utf8_tensor_names() {
    // Non-ASCII tensor names must roundtrip correctly
    let tensors = vec![
        GgufTensor {
            name: "layer.\u{706B}.weight".into(),
            shape: vec![1],
            dtype: GgmlType::F32,
            data: bytemuck::cast_slice(&[1.0f32]).to_vec(),
        },
        GgufTensor {
            name: "\u{43C}\u{43E}\u{434}\u{435}\u{43B}\u{44C}.bias".into(),
            shape: vec![1],
            dtype: GgmlType::F32,
            data: bytemuck::cast_slice(&[2.0f32]).to_vec(),
        },
    ];
    let data = write_gguf(&tensors, &[]);
    let summary = verify_gguf(&data).expect("operation should succeed");
    assert_eq!(summary.tensor_count, 2);
    assert_eq!(summary.tensors[0].name, "layer.\u{706B}.weight");
    assert_eq!(summary.tensors[1].name, "\u{43C}\u{43E}\u{434}\u{435}\u{43B}\u{44C}.bias");
}

#[test]
fn test_falsify_utf8_metadata_values() {
    let metadata = vec![
        (
            "general.name".into(),
            GgufValue::String("\u{6A21}\u{578B}-\u{30C6}\u{30B9}\u{30C8}".into()),
        ),
        (
            "general.architecture".into(),
            GgufValue::String(
                "\u{442}\u{440}\u{430}\u{43D}\u{441}\u{444}\u{43E}\u{440}\u{43C}\u{435}\u{440}"
                    .into(),
            ),
        ),
    ];
    let data = write_gguf(&[], &metadata);
    let summary = verify_gguf(&data).expect("operation should succeed");
    assert_eq!(summary.metadata_count, 2);
}

#[test]
fn test_falsify_10d_tensor_shape() {
    // 10-dimensional shape must survive roundtrip
    let shape = vec![2u64, 2, 2, 2, 2, 2, 2, 2, 2, 2]; // 1024 elements
    let num_elements: u64 = shape.iter().product();
    let values: Vec<f32> = (0..num_elements).map(|i| i as f32).collect();
    let tensors = vec![GgufTensor {
        name: "10d".into(),
        shape: shape.clone(),
        dtype: GgmlType::F32,
        data: bytemuck::cast_slice(&values).to_vec(),
    }];
    let data = write_gguf(&tensors, &[]);
    let summary = verify_gguf(&data).expect("operation should succeed");
    assert_eq!(summary.tensors[0].shape, shape);
    assert_eq!(summary.tensors[0].shape.len(), 10);
}
