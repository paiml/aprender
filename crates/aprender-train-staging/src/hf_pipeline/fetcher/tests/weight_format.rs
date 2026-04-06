//! Tests for WeightFormat enum.

use crate::hf_pipeline::fetcher::WeightFormat;

#[test]
fn test_weight_format_from_safetensors() {
    let format = WeightFormat::from_filename("model.safetensors");
    assert_eq!(format, Some(WeightFormat::SafeTensors));
}

#[test]
fn test_weight_format_from_gguf() {
    let format = WeightFormat::from_filename("model.Q4_K_M.gguf");
    assert!(matches!(format, Some(WeightFormat::GGUF { .. })));
}

#[test]
fn test_weight_format_from_pytorch() {
    let format = WeightFormat::from_filename("pytorch_model.bin");
    assert_eq!(format, Some(WeightFormat::PyTorchBin));
}

#[test]
fn test_weight_format_from_onnx() {
    let format = WeightFormat::from_filename("model.onnx");
    assert_eq!(format, Some(WeightFormat::ONNX));
}

#[test]
fn test_weight_format_unknown() {
    let format = WeightFormat::from_filename("random.txt");
    assert_eq!(format, None);
}

#[test]
fn test_safetensors_is_safe() {
    assert!(WeightFormat::SafeTensors.is_safe());
}

#[test]
fn test_gguf_is_safe() {
    let format = WeightFormat::GGUF { quant_type: "Q4_K_M".into() };
    assert!(format.is_safe());
}

#[test]
fn test_pytorch_is_not_safe() {
    assert!(!WeightFormat::PyTorchBin.is_safe());
}

#[test]
fn test_onnx_is_safe() {
    assert!(WeightFormat::ONNX.is_safe());
}

#[test]
fn test_weight_format_gguf_quant_type() {
    let format = WeightFormat::GGUF { quant_type: "Q4_K_M".to_string() };
    if let WeightFormat::GGUF { quant_type } = format {
        assert_eq!(quant_type, "Q4_K_M");
    } else {
        panic!("Expected GGUF format");
    }
}

#[test]
fn test_weight_format_equality() {
    assert_eq!(WeightFormat::SafeTensors, WeightFormat::SafeTensors);
    assert_eq!(WeightFormat::PyTorchBin, WeightFormat::PyTorchBin);
    assert_eq!(WeightFormat::ONNX, WeightFormat::ONNX);
    assert_ne!(WeightFormat::SafeTensors, WeightFormat::ONNX);

    let gguf1 = WeightFormat::GGUF { quant_type: "Q4_K_M".into() };
    let gguf2 = WeightFormat::GGUF { quant_type: "Q4_K_M".into() };
    assert_eq!(gguf1, gguf2);
}

#[test]
fn test_weight_format_debug() {
    let safetensors = WeightFormat::SafeTensors;
    let debug = format!("{safetensors:?}");
    assert!(debug.contains("SafeTensors"));

    let gguf = WeightFormat::GGUF { quant_type: "Q4_K_S".into() };
    let debug_gguf = format!("{gguf:?}");
    assert!(debug_gguf.contains("Q4_K_S"));
}

#[test]
fn test_weight_format_clone() {
    let original = WeightFormat::GGUF { quant_type: "Q8_0".into() };
    let cloned = original.clone();
    assert_eq!(original, cloned);
}
