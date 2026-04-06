
/// Load and display GGUF model information
pub fn load_gguf_model(file_data: &[u8]) -> Result<()> {
    use crate::gguf::GGUFModel;

    println!("Parsing GGUF file...");
    let gguf = GGUFModel::from_bytes(file_data)?;

    println!("Successfully parsed GGUF file");
    println!();
    println!("Model Information:");
    println!("  Version: {}", gguf.header.version);
    println!("  Tensors: {}", gguf.header.tensor_count);
    println!("  Metadata entries: {}", gguf.header.metadata_count);
    println!();

    if !gguf.metadata.is_empty() {
        println!("Metadata (first 5 entries):");
        for (key, _value) in gguf.metadata.iter().take(5) {
            println!("  - {key}");
        }
        if gguf.metadata.len() > 5 {
            println!("  ... and {} more", gguf.metadata.len() - 5);
        }
        println!();
    }

    if !gguf.tensors.is_empty() {
        println!("Tensors (first 10):");
        for tensor in gguf.tensors.iter().take(10) {
            let dims: Vec<String> = tensor
                .dims
                .iter()
                .map(std::string::ToString::to_string)
                .collect();
            println!(
                "  - {} [{}, qtype={}]",
                tensor.name,
                dims.join("x"),
                tensor.qtype
            );
        }
        if gguf.tensors.len() > 10 {
            println!("  ... and {} more", gguf.tensors.len() - 10);
        }
        println!();
    }

    println!("Model loading infrastructure is ready!");
    println!();
    println!("Next steps to complete model loading:");
    println!("  1. Extract ModelConfig from metadata (vocab_size, hidden_dim, etc.)");
    println!("  2. Map tensor names to Model layers (see src/layers.rs docs)");
    println!("  3. Load weights into each layer");
    println!();
    println!("See documentation: cargo doc --open");
    println!("Example: src/layers.rs module documentation");

    Ok(())
}

/// Load and display SafeTensors model information
pub fn load_safetensors_model(file_data: &[u8]) -> Result<()> {
    use crate::safetensors::SafetensorsModel;

    println!("Parsing Safetensors file...");
    let safetensors = SafetensorsModel::from_bytes(file_data)?;

    println!("Successfully parsed Safetensors file");
    println!();
    println!("Model Information:");
    println!("  Tensors: {}", safetensors.tensors.len());
    println!("  Data size: {} bytes", safetensors.data.len());
    println!();

    if !safetensors.tensors.is_empty() {
        println!("Tensors (first 10):");
        for (name, tensor_info) in safetensors.tensors.iter().take(10) {
            let shape: Vec<String> = tensor_info
                .shape
                .iter()
                .map(std::string::ToString::to_string)
                .collect();
            println!(
                "  - {} [{}, dtype={:?}]",
                name,
                shape.join("x"),
                tensor_info.dtype
            );
        }
        if safetensors.tensors.len() > 10 {
            println!("  ... and {} more", safetensors.tensors.len() - 10);
        }
        println!();
    }

    println!("Model loading infrastructure is ready!");
    println!();
    println!("Next steps to complete model loading:");
    println!("  1. Extract ModelConfig from tensor shapes");
    println!("  2. Map tensor names to Model layers (see src/layers.rs docs)");
    println!("  3. Load weights into each layer");
    println!();
    println!("See documentation: cargo doc --open");
    println!("Example: src/layers.rs module documentation");

    Ok(())
}

/// Load APR model file (aprender native format)
///
/// Per spec §3.1: APR is the first-class format for classical ML models.
///
/// # Arguments
///
/// * `file_data` - APR file bytes
///
/// # Errors
///
/// Returns error if:
/// - Magic bytes don't match (not "APRN")
/// - Model type is unknown
/// - File is corrupted
pub fn load_apr_model(file_data: &[u8]) -> Result<()> {
    use crate::format::{detect_format, ModelFormat};
    use crate::model_loader::read_apr_model_type;

    println!("Parsing APR file...");

    // Verify format
    let format = detect_format(file_data).map_err(|e| RealizarError::UnsupportedOperation {
        operation: "detect_apr_format".to_string(),
        reason: format!("Format detection failed: {e}"),
    })?;

    if format != ModelFormat::Apr {
        return Err(RealizarError::UnsupportedOperation {
            operation: "verify_apr_magic".to_string(),
            reason: format!("Expected APR format, got {format}"),
        });
    }

    // Extract model type
    let model_type = read_apr_model_type(file_data).unwrap_or_else(|| "Unknown".to_string());

    println!("Successfully parsed APR file");
    println!();
    println!("Model Information:");
    println!("  Format: APR (Aprender Native)");
    println!("  Model Type: {model_type}");
    println!("  File Size: {} bytes", file_data.len());
    println!();

    // APR header structure: APRN (4) + type_id (2) + version (2) = 8 bytes minimum
    if file_data.len() >= 8 {
        let version = u16::from_le_bytes([file_data[6], file_data[7]]);
        println!("  Header Version: {version}");
    }

    println!();
    println!("APR model ready for serving!");
    println!("Supported model types for inference:");
    println!("  - LogisticRegression, LinearRegression");
    println!("  - DecisionTree, RandomForest, GradientBoosting");
    println!("  - KNN, GaussianNB, LinearSVM");
    println!();
    println!("To serve this model, the serve API will auto-detect");
    println!("the model type and dispatch to the appropriate handler.");

    Ok(())
}

/// Check if a model reference is a local file path
pub fn is_local_file_path(model_ref: &str) -> bool {
    model_ref.starts_with("./")
        || model_ref.starts_with('/')
        || model_ref.ends_with(".gguf")
        || model_ref.ends_with(".safetensors")
        || model_ref.ends_with(".apr")
}

/// Simple home directory resolution
pub fn home_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(std::path::PathBuf::from)
}

/// Validate benchmark suite name
pub fn validate_suite_name(suite_name: &str) -> bool {
    BENCHMARK_SUITES.iter().any(|(name, _)| *name == suite_name)
}
