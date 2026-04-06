use super::*;

// =====================================================================
// TIER 3: Stress & boundary tests
// =====================================================================

#[test]
fn test_falsify_stress_100_tensors() {
    let mut weights = crate::hf_pipeline::export::weights::ModelWeights::new();
    for i in 0..100 {
        weights.add_tensor(format!("layer.{i:03}.weight"), vec![i as f32; 32], vec![32]);
    }

    let dir = tempfile::tempdir().expect("temp file creation should succeed");
    let exporter = crate::hf_pipeline::export::exporter::Exporter::new()
        .output_dir(dir.path())
        .include_metadata(false);
    let result = exporter
        .export(&weights, crate::hf_pipeline::export::format::ExportFormat::GGUF, "stress100.gguf")
        .expect("operation should succeed");
    assert_eq!(result.num_tensors, 100);

    let file_data =
        std::fs::read(dir.path().join("stress100.gguf")).expect("file read should succeed");
    let summary = verify_gguf(&file_data).expect("operation should succeed");
    assert_eq!(summary.tensor_count, 100);
    assert_eq!(summary.tensors.len(), 100);

    // Verify alphabetical sort with zero-padded names
    for i in 0..100 {
        assert_eq!(
            summary.tensors[i].name,
            format!("layer.{i:03}.weight"),
            "tensor {i} not in sorted position"
        );
    }
}

#[test]
fn test_falsify_stress_5000_char_tensor_name() {
    let long_name = "x".repeat(5000);
    let tensors = vec![GgufTensor {
        name: long_name.clone(),
        shape: vec![1],
        dtype: GgmlType::F32,
        data: bytemuck::cast_slice(&[99.0f32]).to_vec(),
    }];
    let data = write_gguf(&tensors, &[]);
    let summary = verify_gguf(&data).expect("operation should succeed");
    assert_eq!(summary.tensors[0].name, long_name);
    assert_eq!(summary.tensor_count, 1);
}

#[test]
fn test_falsify_stress_block_boundary_exact_sizes() {
    // Tensor sizes that are exact multiples of block size (32)
    for n_blocks in [1, 2, 4, 8, 16, 32] {
        let n_elements = n_blocks * 32;
        let data: Vec<f32> = (0..n_elements).map(|i| i as f32 * 0.01).collect();
        let (q4_bytes, q4_dtype) = quantize_to_gguf_bytes(&data, GgufQuantization::Q4_0);
        let (q8_bytes, q8_dtype) = quantize_to_gguf_bytes(&data, GgufQuantization::Q8_0);

        assert_eq!(q4_bytes.len(), n_blocks * 18, "Q4_0 at {n_blocks} blocks");
        assert_eq!(q8_bytes.len(), n_blocks * 34, "Q8_0 at {n_blocks} blocks");

        // Verify through GGUF roundtrip
        let tensors = vec![
            GgufTensor {
                name: "q4".into(),
                shape: vec![n_elements as u64],
                dtype: q4_dtype,
                data: q4_bytes,
            },
            GgufTensor {
                name: "q8".into(),
                shape: vec![n_elements as u64],
                dtype: q8_dtype,
                data: q8_bytes,
            },
        ];
        let gguf = write_gguf(&tensors, &[]);
        let summary = verify_gguf(&gguf).expect("operation should succeed");
        assert_eq!(summary.tensor_count, 2, "at {n_blocks} blocks");
        assert_eq!(summary.tensors[0].dtype, 2); // Q4_0
        assert_eq!(summary.tensors[1].dtype, 8); // Q8_0
    }
}

#[test]
fn test_falsify_all_16_metadata_combinations() {
    use crate::hf_pipeline::export::{ExportFormat, Exporter, ModelWeights};

    // All 16 combinations of (arch, name, hidden_size, num_layers) being Some/None
    for mask in 0u8..16 {
        let has_arch = mask & 1 != 0;
        let has_name = mask & 2 != 0;
        let has_hidden = mask & 4 != 0;
        let has_layers = mask & 8 != 0;

        let mut weights = ModelWeights::new();
        weights.add_tensor("w", vec![1.0], vec![1]);
        weights.metadata.num_params = 1;
        weights.metadata.architecture = if has_arch { Some("llama".into()) } else { None };
        weights.metadata.model_name = if has_name { Some("test".into()) } else { None };
        weights.metadata.hidden_size = if has_hidden { Some(64) } else { None };
        weights.metadata.num_layers = if has_layers { Some(4) } else { None };

        let dir = tempfile::tempdir().expect("temp file creation should succeed");
        let exporter = Exporter::new().output_dir(dir.path());
        exporter
            .export(&weights, ExportFormat::GGUF, "meta.gguf")
            .expect("operation should succeed");

        let file_data =
            std::fs::read(dir.path().join("meta.gguf")).expect("file read should succeed");
        let summary = verify_gguf(&file_data).expect("operation should succeed");

        // param_count always present + each optional field
        let expected: u64 = 1
            + u64::from(has_arch)
            + u64::from(has_name)
            + u64::from(has_hidden)
            + u64::from(has_layers);
        assert_eq!(
            summary.metadata_count, expected,
            "mask={mask:#06b}: expected {expected} metadata, got {}",
            summary.metadata_count
        );
    }
}

#[test]
fn test_falsify_magic_bytes_survive_all_quant_modes() {
    use crate::hf_pipeline::export::{ExportFormat, Exporter, ModelWeights};

    for quant in [GgufQuantization::None, GgufQuantization::Q4_0, GgufQuantization::Q8_0] {
        let mut weights = ModelWeights::new();
        weights.add_tensor("w", vec![1.0; 64], vec![64]);

        let dir = tempfile::tempdir().expect("temp file creation should succeed");
        let exporter =
            Exporter::new().output_dir(dir.path()).gguf_quantization(quant).include_metadata(false);
        exporter
            .export(&weights, ExportFormat::GGUF, "magic.gguf")
            .expect("operation should succeed");

        let file_data =
            std::fs::read(dir.path().join("magic.gguf")).expect("file read should succeed");
        assert_eq!(&file_data[0..4], b"GGUF", "magic bytes wrong for {quant:?}");
        assert_eq!(
            u32::from_le_bytes(file_data[4..8].try_into().expect("conversion should succeed")),
            3,
            "version wrong for {quant:?}"
        );
    }
}

#[test]
fn test_falsify_file_size_grows_with_tensor_count() {
    let mut prev_size = 0u64;
    for n_tensors in [1, 5, 10, 20, 50] {
        let mut weights = crate::hf_pipeline::export::weights::ModelWeights::new();
        for i in 0..n_tensors {
            weights.add_tensor(format!("t.{i:03}"), vec![1.0; 32], vec![32]);
        }

        let dir = tempfile::tempdir().expect("temp file creation should succeed");
        let exporter = crate::hf_pipeline::export::exporter::Exporter::new()
            .output_dir(dir.path())
            .include_metadata(false);
        let result = exporter
            .export(&weights, crate::hf_pipeline::export::format::ExportFormat::GGUF, "grow.gguf")
            .expect("operation should succeed");

        assert!(
            result.size_bytes > prev_size,
            "size must grow: {n_tensors} tensors = {} bytes, prev = {prev_size}",
            result.size_bytes
        );
        prev_size = result.size_bytes;
    }
}

#[test]
fn test_falsify_deterministic_output() {
    // Same weights exported twice must produce identical bytes
    use crate::hf_pipeline::export::{ExportFormat, Exporter, ModelWeights};

    let mut weights = ModelWeights::new();
    weights.add_tensor("a", vec![1.0, 2.0], vec![2]);
    weights.add_tensor("b", vec![3.0, 4.0], vec![2]);
    weights.metadata.architecture = Some("test".into());
    weights.metadata.num_params = 4;

    let dir1 = tempfile::tempdir().expect("temp file creation should succeed");
    let dir2 = tempfile::tempdir().expect("temp file creation should succeed");
    let exporter1 = Exporter::new().output_dir(dir1.path());
    let exporter2 = Exporter::new().output_dir(dir2.path());
    exporter1.export(&weights, ExportFormat::GGUF, "det.gguf").expect("operation should succeed");
    exporter2.export(&weights, ExportFormat::GGUF, "det.gguf").expect("operation should succeed");

    let bytes1 = std::fs::read(dir1.path().join("det.gguf")).expect("file read should succeed");
    let bytes2 = std::fs::read(dir2.path().join("det.gguf")).expect("file read should succeed");
    assert_eq!(bytes1, bytes2, "identical weights must produce identical GGUF files");
}

#[test]
fn test_falsify_exporter_write_error_on_readonly_path() {
    // Export to a path that cannot be created should return GgufWriteError
    use crate::hf_pipeline::export::{ExportFormat, Exporter, ModelWeights};

    let mut weights = ModelWeights::new();
    weights.add_tensor("w", vec![1.0], vec![1]);

    let exporter = Exporter::new().output_dir("/proc/nonexistent_dir");
    let result = exporter.export(&weights, ExportFormat::GGUF, "fail.gguf");
    assert!(result.is_err(), "export to invalid path must fail");
}

#[test]
fn test_falsify_exporter_creates_parent_directories() {
    // Export to a nested path should create parent directories
    use crate::hf_pipeline::export::{ExportFormat, Exporter, ModelWeights};

    let mut weights = ModelWeights::new();
    weights.add_tensor("w", vec![1.0], vec![1]);

    let dir = tempfile::tempdir().expect("temp file creation should succeed");
    let nested = dir.path().join("deep").join("nested").join("path");
    let exporter = Exporter::new().output_dir(&nested).include_metadata(false);
    let result = exporter.export(&weights, ExportFormat::GGUF, "model.gguf");
    assert!(result.is_ok(), "export should create parent directories");
    assert!(nested.join("model.gguf").exists());
}

#[test]
fn test_falsify_exporter_partial_metadata_combinations() {
    // Test all 4 combinations of architecture/model_name being Some/None
    use crate::hf_pipeline::export::{ExportFormat, Exporter, ModelWeights};

    let cases: Vec<(Option<&str>, Option<&str>, u64)> = vec![
        (None, None, 1),                      // only param_count
        (Some("llama"), None, 2),             // arch + param_count
        (None, Some("my-model"), 2),          // name + param_count
        (Some("llama"), Some("my-model"), 3), // arch + name + param_count
    ];

    for (arch, name, expected_meta) in cases {
        let mut weights = ModelWeights::new();
        weights.add_tensor("w", vec![1.0], vec![1]);
        weights.metadata.num_params = 1;
        weights.metadata.architecture = arch.map(String::from);
        weights.metadata.model_name = name.map(String::from);
        weights.metadata.hidden_size = None;
        weights.metadata.num_layers = None;

        let dir = tempfile::tempdir().expect("temp file creation should succeed");
        let exporter = Exporter::new().output_dir(dir.path());
        exporter
            .export(&weights, ExportFormat::GGUF, "meta.gguf")
            .expect("operation should succeed");

        let file_data =
            std::fs::read(dir.path().join("meta.gguf")).expect("file read should succeed");
        let summary = verify_gguf(&file_data).expect("operation should succeed");
        assert_eq!(
            summary.metadata_count, expected_meta,
            "arch={arch:?}, name={name:?}: expected {expected_meta} metadata, got {}",
            summary.metadata_count
        );
    }
}
