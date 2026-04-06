//! Quantize command implementation

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::{QuantMethod, QuantizeArgs};
use crate::quant::{quantize_tensor, QuantGranularity, QuantMode, QuantizedTensor};
use safetensors::SafeTensors;
use std::collections::HashMap;

/// Load and deserialize a SafeTensors model from disk.
fn load_safetensors(args: &QuantizeArgs) -> Result<Vec<u8>, String> {
    std::fs::read(&args.model).map_err(|e| format!("Failed to read model file: {e}"))
}

/// Serialize quantized tensors and write to output path (JSON format).
fn save_quantized_json(
    quantized_tensors: &HashMap<String, QuantizedTensor>,
    args: &QuantizeArgs,
) -> Result<(), String> {
    let output_data = serde_json::to_vec_pretty(quantized_tensors)
        .map_err(|e| format!("Failed to serialize: {e}"))?;

    std::fs::write(&args.output, &output_data)
        .map_err(|e| format!("Failed to write output: {e}"))?;

    Ok(())
}

/// Serialize quantized tensors to SafeTensors format with I8 dtype + scale tensors.
///
/// For each tensor `name`, outputs:
/// - `name` → I8 data (the quantized weights)
/// - `name.__scale` → F32 scale factors (per-tensor or per-channel)
fn save_quantized_safetensors(
    quantized_tensors: &HashMap<String, QuantizedTensor>,
    args: &QuantizeArgs,
) -> Result<(), String> {
    use safetensors::tensor::{Dtype, TensorView};

    // Collect data buffers that live long enough for TensorView references
    let mut i8_buffers: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();
    let mut scale_buffers: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();

    for (name, qt) in quantized_tensors {
        // I8 data: reinterpret i8 as u8 for safetensors byte storage
        let i8_bytes: Vec<u8> = qt.data.iter().map(|&v| v as u8).collect();
        i8_buffers.push((name.clone(), i8_bytes, qt.shape.clone()));

        // Scale factors as F32
        let scale_name = format!("{name}.__scale");
        let scale_bytes: Vec<u8> = qt.params.scales.iter().flat_map(|s| s.to_le_bytes()).collect();
        let scale_shape = vec![qt.params.scales.len()];
        scale_buffers.push((scale_name, scale_bytes, scale_shape));
    }

    // Build TensorViews
    let mut views: Vec<(&str, TensorView<'_>)> = Vec::new();

    for (name, bytes, shape) in &i8_buffers {
        let view = TensorView::new(Dtype::I8, shape.clone(), bytes)
            .map_err(|e| format!("Failed to create I8 TensorView for {name}: {e}"))?;
        views.push((name.as_str(), view));
    }

    for (name, bytes, shape) in &scale_buffers {
        let view = TensorView::new(Dtype::F32, shape.clone(), bytes)
            .map_err(|e| format!("Failed to create F32 TensorView for {name}: {e}"))?;
        views.push((name.as_str(), view));
    }

    // Metadata
    let mut metadata = HashMap::new();
    metadata.insert("quantization".to_string(), format!("int{}", args.bits));
    metadata.insert("method".to_string(), format!("{:?}", args.method).to_lowercase());
    metadata.insert("num_tensors".to_string(), quantized_tensors.len().to_string());

    let safetensor_bytes = safetensors::serialize(views, Some(metadata))
        .map_err(|e| format!("SafeTensors serialization failed: {e}"))?;

    std::fs::write(&args.output, safetensor_bytes)
        .map_err(|e| format!("Failed to write output: {e}"))?;

    Ok(())
}

/// Validate and convert CLI arguments to quant module types.
fn resolve_quant_params(args: &QuantizeArgs) -> Result<(QuantMode, QuantGranularity), String> {
    if args.bits != 4 && args.bits != 8 {
        return Err(format!("Unsupported bit width: {}. Use 4 or 8.", args.bits));
    }

    let mode = match args.method {
        QuantMethod::Symmetric => QuantMode::Symmetric,
        QuantMethod::Asymmetric => QuantMode::Asymmetric,
    };

    let granularity =
        if args.per_channel { QuantGranularity::PerChannel } else { QuantGranularity::PerTensor };

    Ok((mode, granularity))
}

/// Byte-size tracking for compression ratio computation.
struct ByteAccumulator {
    original: usize,
    quantized: usize,
}

impl ByteAccumulator {
    fn new() -> Self {
        Self { original: 0, quantized: 0 }
    }

    fn compression_ratio(&self) -> f64 {
        if self.quantized > 0 {
            self.original as f64 / self.quantized as f64
        } else {
            1.0
        }
    }
}

/// Quantize a single F32 tensor and return the result with byte accounting.
fn quantize_single_tensor(
    tensor: &safetensors::tensor::TensorView<'_>,
    granularity: QuantGranularity,
    mode: QuantMode,
    bits: u8,
) -> (QuantizedTensor, usize) {
    let shape: Vec<usize> = tensor.shape().to_vec();
    let num_elements: usize = shape.iter().product();
    let original_bytes = num_elements * 4; // 4 bytes per f32

    let bytes = tensor.data();
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let quantized = quantize_tensor(&values, &shape, granularity, mode, bits);
    (quantized, original_bytes)
}

/// Log verbose details about quantization arguments.
fn log_quant_args(args: &QuantizeArgs, level: LogLevel) {
    log(level, LogLevel::Verbose, &format!("  Method: {:?}", args.method));
    log(level, LogLevel::Verbose, &format!("  Per-channel: {}", args.per_channel));
    log(level, LogLevel::Verbose, &format!("  Output: {}", args.output.display()));
}

pub fn run_quantize(args: QuantizeArgs, level: LogLevel) -> Result<(), String> {
    log(
        level,
        LogLevel::Normal,
        &format!("Quantizing {} to {}-bit", args.model.display(), args.bits),
    );

    log_quant_args(&args, level);

    let (mode, granularity) = resolve_quant_params(&args)?;

    let data = load_safetensors(&args)?;
    let tensors =
        SafeTensors::deserialize(&data).map_err(|e| format!("Failed to parse safetensors: {e}"))?;

    let mut quantized_tensors: HashMap<String, QuantizedTensor> = HashMap::new();
    let mut bytes = ByteAccumulator::new();

    for name in tensors.names() {
        let tensor =
            tensors.tensor(name).map_err(|e| format!("Failed to get tensor {name}: {e}"))?;

        if tensor.dtype() != safetensors::tensor::Dtype::F32 {
            log(level, LogLevel::Verbose, &format!("  Skipping {name} (not F32)"));
            continue;
        }

        let (quantized, original_bytes) =
            quantize_single_tensor(&tensor, granularity, mode, args.bits);
        bytes.original += original_bytes;
        bytes.quantized += quantized.memory_bytes();

        log(
            level,
            LogLevel::Verbose,
            &format!(
                "  Quantized {}: {:?} -> {} bytes",
                name,
                tensor.shape(),
                quantized.memory_bytes()
            ),
        );

        quantized_tensors.insert((*name).to_string(), quantized);
    }

    if args.safetensors {
        save_quantized_safetensors(&quantized_tensors, &args)?;
    } else {
        save_quantized_json(&quantized_tensors, &args)?;
    }

    log(
        level,
        LogLevel::Normal,
        &format!(
            "Quantization complete: {} tensors, {:.1}x compression",
            quantized_tensors.len(),
            bytes.compression_ratio()
        ),
    );
    log(level, LogLevel::Normal, &format!("  Output: {}", args.output.display()));

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use safetensors::tensor::{Dtype, TensorView};

    /// Create a minimal safetensors file with known F32 data for testing.
    fn create_test_safetensors(path: &std::path::Path) {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let view = TensorView::new(Dtype::F32, vec![8, 8], &bytes).unwrap();
        let views = vec![("test_weight", view)];
        let serialized = safetensors::serialize(views, None::<HashMap<String, String>>).unwrap();
        std::fs::write(path, serialized).unwrap();
    }

    #[test]
    fn test_wasm002_quantize_safetensors_int8_output() {
        let dir = tempfile::tempdir().unwrap();
        let input_path = dir.path().join("model.safetensors");
        let output_path = dir.path().join("model_int8.safetensors");

        create_test_safetensors(&input_path);

        let args = QuantizeArgs {
            model: input_path,
            output: output_path.clone(),
            bits: 8,
            method: crate::config::QuantMethod::Symmetric,
            per_channel: false,
            calibration_data: None,
            safetensors: true,
        };

        run_quantize(args, crate::cli::LogLevel::Quiet).expect("quantize should succeed");

        // Verify output is valid safetensors with I8 tensors
        let data = std::fs::read(&output_path).unwrap();
        let tensors = SafeTensors::deserialize(&data).unwrap();

        // Should have weight tensor (I8) and scale tensor (F32)
        let names: Vec<&str> = tensors.names().into_iter().collect();
        assert!(names.contains(&"test_weight"), "Must contain weight tensor");
        assert!(names.contains(&"test_weight.__scale"), "Must contain scale tensor");

        // Verify dtype
        let weight = tensors.tensor("test_weight").unwrap();
        assert_eq!(weight.dtype(), Dtype::I8);
        assert_eq!(weight.shape(), &[8, 8]);
        assert_eq!(weight.data().len(), 64); // 64 i8 values = 64 bytes

        let scale = tensors.tensor("test_weight.__scale").unwrap();
        assert_eq!(scale.dtype(), Dtype::F32);
    }

    #[test]
    fn test_wasm002_quantize_safetensors_compression() {
        let dir = tempfile::tempdir().unwrap();
        let input_path = dir.path().join("model.safetensors");
        let output_path = dir.path().join("model_int8.safetensors");

        create_test_safetensors(&input_path);

        let args = QuantizeArgs {
            model: input_path.clone(),
            output: output_path.clone(),
            bits: 8,
            method: crate::config::QuantMethod::Symmetric,
            per_channel: false,
            calibration_data: None,
            safetensors: true,
        };

        run_quantize(args, crate::cli::LogLevel::Quiet).expect("quantize");

        let input_size = std::fs::metadata(&input_path).unwrap().len();
        let output_size = std::fs::metadata(&output_path).unwrap().len();

        // Int8 should be significantly smaller than F32 (roughly 4x)
        assert!(
            output_size < input_size,
            "Int8 output ({output_size}) must be smaller than F32 input ({input_size})"
        );
    }

    #[test]
    fn test_wasm002_quantize_json_still_works() {
        let dir = tempfile::tempdir().unwrap();
        let input_path = dir.path().join("model.safetensors");
        let output_path = dir.path().join("model_int8.json");

        create_test_safetensors(&input_path);

        let args = QuantizeArgs {
            model: input_path,
            output: output_path.clone(),
            bits: 8,
            method: crate::config::QuantMethod::Symmetric,
            per_channel: false,
            calibration_data: None,
            safetensors: false,
        };

        run_quantize(args, crate::cli::LogLevel::Quiet).expect("quantize");

        // Verify JSON output still works
        let json = std::fs::read_to_string(&output_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_object());
        assert!(parsed.get("test_weight").is_some());
    }
}
