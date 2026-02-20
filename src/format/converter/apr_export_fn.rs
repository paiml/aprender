/// Export APR/SafeTensors model to another format
///
/// # Arguments
///
/// * `input` - Input model path (.apr or .safetensors)
/// * `output` - Output file path
/// * `options` - Export options
///
/// # Returns
///
/// Export report with size and format information
///
/// # Errors
///
/// Returns error if:
/// - Input file doesn't exist
/// - Format not supported
/// - Export fails
///
/// # Example
///
/// ```rust,ignore
/// use aprender::format::{apr_export, ExportOptions, ExportFormat};
///
/// let options = ExportOptions {
///     format: ExportFormat::Gguf,
///     quantize: None,
/// };
/// let report = apr_export("model.apr", "model.gguf", options)?;
/// ```
pub fn apr_export<P: AsRef<Path>>(
    input: P,
    output: P,
    options: ExportOptions,
) -> Result<ExportReport> {
    let input_path = input.as_ref();
    let output_path = output.as_ref();

    validate_export_inputs(input_path, &options)?;

    // PMAT-252: For APR→GGUF with quantized source, use raw block passthrough
    if let Some(report) = try_raw_gguf_passthrough(input_path, output_path, &options)? {
        return Ok(report);
    }

    let provenance = load_model_tensors_provenance(input_path)?;
    let original_size = calculate_tensor_size(provenance.as_map());
    let tensors = provenance.into_map();

    // PMAT-260: Capture original dtypes from SafeTensors source for round-trip preservation.
    // BF16 tensors must be written back as BF16, not widened to F32.
    let original_dtypes = extract_source_dtypes(input_path);

    let tensors = prepare_tensors_for_export(tensors, input_path, &options);
    enforce_contract_violations(&tensors)?;
    // PMAT-297: Architecture completeness gate — refuse to export incomplete models
    enforce_export_completeness(input_path, &tensors)?;
    let tensors = apply_export_quantization(tensors, input_path, &options)?;

    dispatch_export(&tensors, input_path, output_path, &options, &original_dtypes)?;

    let exported_size = if output_path.is_dir() {
        // GH-246: For directory-based exports (MLX), sum all file sizes
        dir_total_size(output_path)
    } else {
        fs::metadata(output_path)
            .map(|m| m.len() as usize)
            .unwrap_or(0)
    };
    Ok(ExportReport {
        original_size,
        exported_size,
        tensor_count: tensors.len(),
        format: options.format,
        quantization: options.quantize,
    })
}

/// PMAT-297: Architecture completeness gate for export path.
///
/// Detects the architecture from APR metadata and verifies all required tensors
/// are present before writing the output file. Missing tensor = hard error.
fn enforce_export_completeness(
    input_path: &Path,
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<()> {
    // Only check APR source files — GGUF/SafeTensors don't have APR metadata
    if input_path.extension().and_then(|e| e.to_str()) != Some("apr") {
        return Ok(());
    }
    let arch_key = match detect_apr_architecture_for_completeness(input_path) {
        Some(key) => key,
        None => return Ok(()), // Unknown architecture — skip
    };
    let num_layers = infer_layer_count(tensors.keys().map(String::as_str));
    if num_layers == 0 {
        return Ok(());
    }
    let names: Vec<&str> = tensors.keys().map(String::as_str).collect();
    crate::format::layout_contract::enforce_architecture_completeness(&names, arch_key, num_layers)
        .map_err(|e| AprenderError::FormatError {
            message: format!("PMAT-297 export completeness gate: {e}"),
        })
}

/// Detect architecture from APR metadata for completeness checking.
fn detect_apr_architecture_for_completeness(apr_path: &Path) -> Option<&'static str> {
    let data = fs::read(apr_path).ok()?;
    let reader = crate::format::v2::AprV2Reader::from_bytes(&data).ok()?;
    let metadata = reader.metadata();
    let arch = metadata
        .architecture
        .as_deref()
        .or_else(|| {
            let mt = &metadata.model_type;
            if mt.is_empty() || mt == "unknown" {
                None
            } else {
                Some(mt.as_str())
            }
        })?;
    // Map to completeness key (static str)
    let key = match arch {
        "qwen3" => "qwen3",
        "qwen3_5" | "qwen3.5" => "qwen3_5",
        "qwen2" | "qwen2.5" | "qwen" => "qwen2",
        "llama" | "llama3" => "llama",
        "mistral" => "mistral",
        "phi" | "phi3" => "phi",
        "gemma" | "gemma2" => "gemma",
        "deepseek" | "deepseek2" => "deepseek",
        "falcon_h1" | "falcon-h1" => "falcon_h1",
        "openelm" => "openelm",
        "moonshine" => "moonshine",
        "rwkv7" | "rwkv" => "rwkv7",
        "mamba" => "mamba",
        "bert" => "bert",
        "whisper" => "whisper",
        "gpt2" => "gpt2",
        _ => return None,
    };
    Some(key)
}

/// Infer number of transformer layers from tensor name patterns.
fn infer_layer_count<'a>(names: impl Iterator<Item = &'a str>) -> usize {
    let mut max_idx: Option<usize> = None;
    for name in names {
        let idx = if let Some(rest) = name.strip_prefix("blk.") {
            rest.split('.').next().and_then(|s| s.parse::<usize>().ok())
        } else if let Some(rest) = name.strip_prefix("model.layers.") {
            rest.split('.').next().and_then(|s| s.parse::<usize>().ok())
        } else {
            None
        };
        if let Some(i) = idx {
            max_idx = Some(max_idx.map_or(i, |m: usize| m.max(i)));
        }
    }
    max_idx.map_or(0, |m| m + 1)
}

/// GH-246: Calculate total size of all files in a directory (for MLX exports).
fn dir_total_size(dir: &Path) -> usize {
    fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter_map(|e| e.metadata().ok())
                .filter(|m| m.is_file())
                .map(|m| m.len() as usize)
                .sum()
        })
        .unwrap_or(0)
}

/// Validate export inputs (file exists, format supported).
fn validate_export_inputs(input_path: &Path, options: &ExportOptions) -> Result<()> {
    if !input_path.exists() {
        return Err(AprenderError::FormatError {
            message: format!("Input file not found: {}", input_path.display()),
        });
    }
    if !options.format.is_supported() {
        return Err(AprenderError::FormatError {
            message: format!(
                "Export format {:?} is not yet supported. Use 'safetensors', 'gguf', or 'mlx'.",
                options.format
            ),
        });
    }
    Ok(())
}

/// PMAT-252: Try raw block passthrough for quantized APR→GGUF.
fn try_raw_gguf_passthrough(
    input_path: &Path,
    output_path: &Path,
    options: &ExportOptions,
) -> Result<Option<ExportReport>> {
    if options.format != ExportFormat::Gguf
        || options.quantize.is_some()
        || input_path.extension().and_then(|e| e.to_str()) != Some("apr")
    {
        return Ok(None);
    }
    match detect_apr_quantization(input_path) {
        Some(detected) => {
            eprintln!(
                "[PMAT-252] Raw passthrough: detected {detected:?} in APR source. Copying blocks directly (zero loss)."
            );
            export_apr_to_gguf_raw(input_path, output_path).map(Some)
        }
        None => Ok(None),
    }
}

/// Prepare tensors: name mapping, QKV unfusing, lm_head removal.
fn prepare_tensors_for_export(
    tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    input_path: &Path,
    options: &ExportOptions,
) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    // GH-200: Map GGUF names to HF canonical format
    let tensors = if input_path.extension().and_then(|e| e.to_str()) == Some("gguf") {
        map_tensor_names(&tensors, detect_gguf_architecture(input_path))
    } else {
        tensors
    };
    let tensors = unfuse_qkv_tensors(tensors, input_path);
    if options.format == ExportFormat::SafeTensors {
        remove_tied_lm_head(tensors, input_path)
    } else {
        tensors
    }
}

/// GH-237: Validate tensor shapes against layout contract, returning error on violations.
///
/// Previously this was advisory (eprintln only). Now violations are collected and
/// returned as an error to prevent corrupt data from being written to disk.
fn enforce_contract_violations(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> Result<()> {
    let layout_contract = contract();
    let (vocab_size, hidden_dim) = infer_vocab_hidden(tensors);
    if vocab_size == 0 || hidden_dim == 0 {
        return Ok(());
    }
    let mut violations = Vec::new();
    for (name, (_data, shape)) in tensors {
        if let Err(e) = layout_contract.validate_apr_shape(name, shape, vocab_size, hidden_dim) {
            violations.push(format!("{name}: {e}"));
        }
    }
    if violations.is_empty() {
        Ok(())
    } else {
        Err(AprenderError::FormatError {
            message: format!(
                "[CONTRACT-VIOLATION] Export validation failed for {} tensor(s):\n  {}",
                violations.len(),
                violations.join("\n  ")
            ),
        })
    }
}

/// DOUBLE-QUANT-001: Apply quantization only to natively F32 tensors.
fn apply_export_quantization(
    tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    input_path: &Path,
    options: &ExportOptions,
) -> Result<BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
    let Some(ref quant_type) = options.quantize else {
        return Ok(tensors);
    };
    if let Some(detected) = detect_apr_quantization(input_path) {
        return Err(AprenderError::FormatError {
            message: format!(
                "DOUBLE-QUANT-001: Cannot re-quantize to {quant_type:?}: source is already \
                 {detected:?}. Remove --quantize flag to use raw passthrough."
            ),
        });
    }
    let native = NativeF32Tensors::new(tensors);
    Ok(quantize_tensors(&native, quant_type)?.into_inner())
}

/// Dispatch to format-specific export.
/// PMAT-260: Extract original dtype map from a SafeTensors source file.
///
/// Returns a map of tensor name → dtype string (e.g. "BF16", "F16", "F32").
/// Returns empty map for non-SafeTensors sources (APR, GGUF).
fn extract_source_dtypes(input_path: &Path) -> BTreeMap<String, String> {
    if input_path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
        return BTreeMap::new();
    }
    match crate::serialization::safetensors::MappedSafeTensors::open(input_path) {
        Ok(mapped) => mapped.dtype_map(),
        Err(_) => BTreeMap::new(),
    }
}

fn dispatch_export(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    input_path: &Path,
    output_path: &Path,
    options: &ExportOptions,
    original_dtypes: &BTreeMap<String, String>,
) -> Result<()> {
    match options.format {
        ExportFormat::SafeTensors => export_safetensors_with_companions(
            tensors,
            input_path,
            output_path,
            options,
            original_dtypes,
        ),
        ExportFormat::Gguf => {
            export_to_gguf(tensors, output_path, input_path, options.quantize.as_ref())
        }
        ExportFormat::Mlx => export_mlx(tensors, input_path, output_path, options),
        ExportFormat::Onnx
        | ExportFormat::OpenVino
        | ExportFormat::CoreMl
        | ExportFormat::TorchScript => Err(AprenderError::FormatError {
            message: format!(
                "Export format {} is not yet implemented. Supported: safetensors, gguf, mlx",
                options.format.display_name()
            ),
        }),
    }
}

include!("gguf_export_config.rs");
include!("metadata.rs");
include!("tensor.rs");
include!("name.rs");
