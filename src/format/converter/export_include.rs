/// Collect raw bytes for fusion sources from an APR reader.
/// Returns `(bytes, shapes, dtype)` or `None` if any source is missing.
fn collect_raw_fusion_sources(
    rule: &FusionExportRule,
    layer: usize,
    reader: &crate::format::v2::AprV2Reader,
) -> Option<(Vec<u8>, Vec<Vec<usize>>, crate::format::gguf::GgmlType)> {
    let mut all_bytes: Vec<u8> = Vec::new();
    let mut all_shapes: Vec<Vec<usize>> = Vec::new();
    let mut dtype = crate::format::gguf::GgmlType::F32;

    for apr_suffix in &rule.apr_suffixes {
        let apr_name = format!("model.layers.{layer}.{apr_suffix}");
        let entry = reader.get_tensor(&apr_name)?;
        let raw = reader.get_tensor_data(&apr_name)?;

        all_bytes.extend_from_slice(raw);
        all_shapes.push(entry.shape.clone());
        dtype = apr_dtype_to_ggml(entry.dtype);
    }
    Some((all_bytes, all_shapes, dtype))
}

fn build_fused_tensors_raw(
    mapper: &GgufNameMapper,
    reader: &crate::format::v2::AprV2Reader,
) -> Vec<crate::format::gguf::GgufTensor> {
    use crate::format::gguf::GgufTensor;

    let rules = mapper.fusion_rules();
    if rules.is_empty() {
        return Vec::new();
    }

    let names = reader.tensor_names();
    let num_layers = detect_num_layers_from_names(names.iter().map(|s| s.as_ref()));
    let mut fused = Vec::new();

    for rule in rules {
        for layer in 0..num_layers {
            let Some((all_bytes, all_shapes, dtype)) =
                collect_raw_fusion_sources(rule, layer, reader)
            else {
                continue;
            };

            let Some(fused_shape) = compute_fused_shape(&all_shapes) else {
                continue;
            };

            let gguf_shape = shape_to_gguf(&fused_shape);
            let gguf_name = format!("blk.{layer}.{}", rule.gguf_suffix);

            fused.push(GgufTensor {
                name: gguf_name,
                shape: gguf_shape,
                dtype,
                data: all_bytes,
            });
        }
    }

    fused
}

/// Export format options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// `SafeTensors` format (.safetensors) - `HuggingFace` ecosystem
    SafeTensors,
    /// GGUF format (.gguf) - llama.cpp / local inference
    Gguf,
    /// MLX format (directory with .safetensors + config.json) - Apple Silicon
    Mlx,
    /// ONNX format (.onnx) - Cross-framework inference (not yet implemented)
    Onnx,
    /// OpenVINO IR format (.xml + .bin) - Intel inference (not yet implemented)
    OpenVino,
    /// CoreML format (.mlpackage) - iOS/macOS deployment (not yet implemented)
    CoreMl,
    /// `TorchScript` format (.pt) - `PyTorch` deployment (not yet implemented)
    TorchScript,
}

impl std::str::FromStr for ExportFormat {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "safetensors" | "st" => Ok(Self::SafeTensors),
            "gguf" => Ok(Self::Gguf),
            "mlx" => Ok(Self::Mlx),
            "onnx" => Ok(Self::Onnx),
            "openvino" | "ov" => Ok(Self::OpenVino),
            "coreml" | "mlpackage" => Ok(Self::CoreMl),
            "torchscript" | "pt" | "torch" => Ok(Self::TorchScript),
            _ => Err(format!("Unknown export format: {s}")),
        }
    }
}

impl ExportFormat {
    /// Get default file extension
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::SafeTensors => "safetensors",
            Self::Gguf => "gguf",
            Self::Mlx => "mlx",
            Self::Onnx => "onnx",
            Self::OpenVino => "xml",
            Self::CoreMl => "mlpackage",
            Self::TorchScript => "pt",
        }
    }

    /// Check if format is supported
    #[must_use]
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::SafeTensors | Self::Gguf | Self::Mlx)
    }

    /// Human-readable name for display
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::SafeTensors => "SafeTensors",
            Self::Gguf => "GGUF",
            Self::Mlx => "MLX",
            Self::Onnx => "ONNX",
            Self::OpenVino => "OpenVINO",
            Self::CoreMl => "CoreML",
            Self::TorchScript => "TorchScript",
        }
    }

    /// All known export formats
    #[must_use]
    pub fn all() -> &'static [ExportFormat] {
        &[
            Self::SafeTensors,
            Self::Gguf,
            Self::Mlx,
            Self::Onnx,
            Self::OpenVino,
            Self::CoreMl,
            Self::TorchScript,
        ]
    }
}

/// Options for model export
#[derive(Debug, Clone)]
pub struct ExportOptions {
    /// Target format
    pub format: ExportFormat,
    /// Optional quantization
    pub quantize: Option<QuantizationType>,
    /// GH-182: Include tokenizer.json companion file (SafeTensors only)
    pub include_tokenizer: bool,
    /// GH-182: Include config.json companion file (SafeTensors only)
    pub include_config: bool,
    /// Skip PMAT-297 architecture completeness gate (for test pygmy models)
    pub skip_completeness_check: bool,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: ExportFormat::SafeTensors,
            quantize: None,
            include_tokenizer: true, // Default to true for HuggingFace compatibility
            include_config: true,
            skip_completeness_check: false,
        }
    }
}

/// Report from export operation
#[derive(Debug, Clone)]
pub struct ExportReport {
    /// Original size in bytes
    pub original_size: usize,
    /// Exported size in bytes
    pub exported_size: usize,
    /// Number of tensors exported
    pub tensor_count: usize,
    /// Export format used
    pub format: ExportFormat,
    /// Quantization applied
    pub quantization: Option<QuantizationType>,
}

/// GH-253-4: Validated GGUF metadata — compile-time enforcement via newtype.
///
/// Private inner data ensures metadata can ONLY be constructed through `validate()`,
/// which checks that required keys are present and consistent.
///
/// Required keys:
/// - `general.architecture` (always)
/// - `tokenizer.ggml.model` (if vocabulary present)
/// - `tokenizer.ggml.tokens` and `tokenizer.ggml.model` must appear together
#[derive(Debug)]
pub(crate) struct ValidatedGgufMetadata {
    inner: Vec<(String, crate::format::gguf::GgufValue)>,
}

/// GH-277/GH-279: Dedup token table for llama.cpp compatibility.
///
/// HuggingFace tokenizers (Qwen2, Qwen3, etc.) may have reserved/padding tokens
/// that share the same string (e.g., 290 copies of `<unk>`). llama.cpp requires
/// `id_to_token.size() == token_to_id.size()`, meaning every token string must be
/// unique. Following HuggingFace's `convert_hf_to_gguf.py` convention, duplicate
/// tokens are renamed to `[PAD{token_id}]`.
pub(crate) fn dedup_token_table(metadata: &mut [(String, crate::format::gguf::GgufValue)]) {
    let Some(pos) = metadata
        .iter()
        .position(|(k, _)| k == "tokenizer.ggml.tokens")
    else {
        return;
    };

    let crate::format::gguf::GgufValue::ArrayString(tokens) = &metadata[pos].1 else {
        return;
    };

    let mut seen = std::collections::HashSet::with_capacity(tokens.len());
    let mut dedup_count = 0u32;
    let deduped: Vec<String> = tokens
        .iter()
        .enumerate()
        .map(|(idx, tok)| {
            if seen.contains(tok.as_str()) {
                dedup_count += 1;
                format!("[PAD{idx}]")
            } else {
                seen.insert(tok.clone());
                tok.clone()
            }
        })
        .collect();

    if dedup_count > 0 {
        eprintln!("[GH-277] Deduped {dedup_count} duplicate token(s) → [PAD{{id}}] format");
        metadata[pos] = (
            "tokenizer.ggml.tokens".to_string(),
            crate::format::gguf::GgufValue::ArrayString(deduped),
        );
    }
}

impl ValidatedGgufMetadata {
    /// Validate and construct metadata. Fails if required keys are missing.
    pub(crate) fn validate(
        mut metadata: Vec<(String, crate::format::gguf::GgufValue)>,
    ) -> Result<Self> {
        let has_key = |k: &str| metadata.iter().any(|(name, _)| name == k);

        if !has_key("general.architecture") {
            return Err(AprenderError::FormatError {
                message: "[GH-253-4] GGUF export missing required key: general.architecture"
                    .to_string(),
            });
        }

        let has_tokens = has_key("tokenizer.ggml.tokens");
        let has_model = has_key("tokenizer.ggml.model");
        if has_tokens && !has_model {
            return Err(AprenderError::FormatError {
                message:
                    "[GH-253-4] GGUF export has tokenizer.ggml.tokens but missing tokenizer.ggml.model"
                        .to_string(),
            });
        }
        if has_model && !has_tokens {
            return Err(AprenderError::FormatError {
                message:
                    "[GH-253-4] GGUF export has tokenizer.ggml.model but missing tokenizer.ggml.tokens"
                        .to_string(),
            });
        }

        dedup_token_table(&mut metadata);

        Ok(Self { inner: metadata })
    }

    /// Access validated metadata as a slice for GGUF writing.
    pub(crate) fn as_slice(&self) -> &[(String, crate::format::gguf::GgufValue)] {
        &self.inner
    }
}

