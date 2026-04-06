//! Tests for loader module

use super::*;
use crate::hf_pipeline::error::FetchError;
use ndarray::Array2;
use std::path::Path;

// =========================================================================
// MemoryEstimate Tests
// =========================================================================

#[test]
fn test_memory_estimate_total() {
    let est = MemoryEstimate { weights: 100, activations: 50, gradients: 25 };
    assert_eq!(est.total(), 175);
}

#[test]
fn test_memory_estimate_fits_in() {
    let est = MemoryEstimate { weights: 100, activations: 50, gradients: 0 };
    assert!(est.fits_in(200));
    assert!(est.fits_in(150));
    assert!(!est.fits_in(100));
}

#[test]
fn test_memory_estimate_fp32() {
    // 125M params in FP32 = 500MB
    let est = MemoryEstimate::fp32(125_000_000, 1, 512, 768);
    assert_eq!(est.weights, 500_000_000);
    assert!(est.activations > 0);
    assert_eq!(est.gradients, 0); // Frozen teacher
}

#[test]
fn test_memory_estimate_fp16() {
    // 125M params in FP16 = 250MB
    let est = MemoryEstimate::fp16(125_000_000, 1, 512, 768);
    assert_eq!(est.weights, 250_000_000);
}

#[test]
fn test_memory_estimate_int4() {
    // 125M params in INT4 = ~62.5MB
    let est = MemoryEstimate::int4(125_000_000, 1, 512, 768);
    assert_eq!(est.weights, 62_500_000);
}

#[test]
fn test_codebert_memory() {
    // CodeBERT: 125M params
    let est = MemoryEstimate::fp16(125_000_000, 32, 512, 768);
    // Should fit in 8GB GPU
    assert!(est.fits_in(8 * 1024 * 1024 * 1024));
}

#[test]
fn test_llama_7b_memory() {
    // Llama-7B: 7B params
    let est = MemoryEstimate::fp16(7_000_000_000, 1, 2048, 4096);
    // Needs ~14GB for weights alone
    assert!(est.weights > 10 * 1024 * 1024 * 1024);
}

#[test]
fn test_llama_7b_int4_memory() {
    // Llama-7B quantized: ~3.5GB
    let est = MemoryEstimate::int4(7_000_000_000, 1, 2048, 4096);
    assert!(est.weights < 5 * 1024 * 1024 * 1024);
}

// =========================================================================
// SafeTensorsTeacher Tests
// =========================================================================

#[test]
fn test_mock_teacher_creation() {
    let teacher = SafeTensorsTeacher::mock(12, 768);
    assert_eq!(teacher.num_layers(), 12);
    assert_eq!(teacher.hidden_size(), 768);
    assert!(teacher.param_count() > 0);
}

#[test]
fn test_teacher_forward() {
    let teacher = SafeTensorsTeacher::mock(12, 768);
    let input = Array2::<f32>::zeros((4, 768));
    let output = teacher.forward(&input);
    assert!(output.is_ok());
    assert_eq!(output.expect("operation should succeed").dim(), (4, 768));
}

#[test]
fn test_teacher_hidden_states() {
    let teacher = SafeTensorsTeacher::mock(12, 768);
    let input = Array2::<f32>::zeros((4, 768));
    let hidden = teacher.hidden_states(&input);
    assert!(hidden.is_ok());
    let hidden = hidden.expect("operation should succeed");
    assert_eq!(hidden.len(), 12); // One per layer
}

#[test]
fn test_teacher_attention_weights() {
    let teacher = SafeTensorsTeacher::mock(12, 768);
    let input = Array2::<f32>::zeros((4, 768));
    let attn = teacher.attention_weights(&input);
    assert!(attn.is_ok());
    let attn = attn.expect("operation should succeed");
    assert_eq!(attn.len(), 12);
}

#[test]
fn test_teacher_memory_estimate() {
    let teacher = SafeTensorsTeacher::mock(12, 768);
    let est = teacher.estimate_memory(32, 512);
    assert!(est.weights > 0);
    assert!(est.activations > 0);
    assert_eq!(est.gradients, 0);
}

#[test]
fn test_load_nonexistent() {
    let result = SafeTensorsTeacher::load(Path::new("/nonexistent/path"));
    assert!(matches!(result, Err(FetchError::FileNotFound { .. })));
}

// =========================================================================
// SafeTensors Parsing Tests (TDD - these define expected behavior)
// =========================================================================

#[test]
fn test_load_valid_safetensors_file() {
    use tempfile::TempDir;

    // Create a minimal valid safetensors file
    let temp_dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = temp_dir.path().join("model.safetensors");

    // Create minimal safetensors with one tensor
    let data = create_test_safetensors(&[("weight", &[2, 3])]);
    std::fs::write(&model_path, data).expect("file write should succeed");

    let teacher = SafeTensorsTeacher::load(temp_dir.path());
    assert!(teacher.is_ok(), "Should load valid safetensors file");

    let teacher = teacher.expect("operation should succeed");
    assert!(teacher.param_count() > 0, "Should have non-zero params");
}

#[test]
fn test_safetensors_extracts_tensor_names() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = temp_dir.path().join("model.safetensors");

    // Create safetensors with named tensors
    let data = create_test_safetensors(&[
        ("encoder.layer.0.attention.query.weight", &[768, 768]),
        ("encoder.layer.0.attention.key.weight", &[768, 768]),
    ]);
    std::fs::write(&model_path, data).expect("file write should succeed");

    let teacher = SafeTensorsTeacher::load(temp_dir.path()).expect("load should succeed");
    assert!(teacher.tensor_names().contains(&"encoder.layer.0.attention.query.weight".to_string()));
}

#[test]
fn test_safetensors_param_count_matches_tensors() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = temp_dir.path().join("model.safetensors");

    // Create safetensors with known parameter count
    // 2 tensors of 768x768 = 2 * 589,824 = 1,179,648 params
    let data = create_test_safetensors(&[
        ("layer.0.weight", &[768, 768]),
        ("layer.1.weight", &[768, 768]),
    ]);
    std::fs::write(&model_path, data).expect("file write should succeed");

    let teacher = SafeTensorsTeacher::load(temp_dir.path()).expect("load should succeed");
    assert_eq!(teacher.param_count(), 768 * 768 * 2);
}

#[test]
fn test_safetensors_detects_layer_count() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = temp_dir.path().join("model.safetensors");

    // Create safetensors with 12 layers
    let mut tensors: Vec<(&str, &[usize])> = Vec::new();
    let layer_names: Vec<String> =
        (0..12).map(|i| format!("encoder.layer.{i}.attention.weight")).collect();

    for name in &layer_names {
        tensors.push((name, &[768, 768]));
    }

    let data = create_test_safetensors_from_names(&tensors);
    std::fs::write(&model_path, data).expect("file write should succeed");

    let teacher = SafeTensorsTeacher::load(temp_dir.path()).expect("load should succeed");
    assert_eq!(teacher.num_layers(), 12);
}

#[test]
fn test_safetensors_detects_hidden_size() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = temp_dir.path().join("model.safetensors");

    // Create safetensors with 1024 hidden size
    let data =
        create_test_safetensors(&[("encoder.layer.0.attention.query.weight", &[1024, 1024])]);
    std::fs::write(&model_path, data).expect("file write should succeed");

    let teacher = SafeTensorsTeacher::load(temp_dir.path()).expect("load should succeed");
    assert_eq!(teacher.hidden_size(), 1024);
}

#[test]
fn test_safetensors_corrupt_file_error() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = temp_dir.path().join("model.safetensors");

    // Write garbage data
    std::fs::write(&model_path, b"not a valid safetensors file")
        .expect("file write should succeed");

    let result = SafeTensorsTeacher::load(temp_dir.path());
    assert!(result.is_err(), "Should fail on corrupt file");
}

// Helper function to create minimal safetensors for testing
fn create_test_safetensors(tensors: &[(&str, &[usize])]) -> Vec<u8> {
    use ::safetensors::tensor::{Dtype, TensorView};

    let tensor_data: Vec<(String, Vec<f32>, Vec<usize>)> = tensors
        .iter()
        .map(|(name, shape)| {
            let numel: usize = shape.iter().product();
            ((*name).to_string(), vec![0.0f32; numel], shape.to_vec())
        })
        .collect();

    let views: Vec<(&str, TensorView<'_>)> = tensor_data
        .iter()
        .map(|(name, data, shape)| {
            let view = TensorView::new(Dtype::F32, shape.clone(), bytemuck::cast_slice(data))
                .expect("operation should succeed");
            (name.as_str(), view)
        })
        .collect();

    ::safetensors::serialize(views, None::<std::collections::HashMap<String, String>>)
        .expect("operation should succeed")
}

fn create_test_safetensors_from_names(tensors: &[(&str, &[usize])]) -> Vec<u8> {
    create_test_safetensors(tensors)
}
