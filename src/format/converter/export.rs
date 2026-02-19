//! APR Converter - Export to SafeTensors/GGUF (APR-SPEC §4.6)
//! PMAT-197: Extracted from mod.rs for file size reduction

use crate::error::{AprenderError, Result};
use crate::format::converter_types::{Architecture, QuantizationType};
use crate::format::gguf::GgufReader;
use crate::format::layout_contract::contract;
use crate::serialization::safetensors::{
    save_safetensors, save_safetensors_typed, save_safetensors_with_metadata,
    save_safetensors_with_metadata_typed, UserMetadata,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

// Import shared functions from parent module
use super::{
    calculate_tensor_size, load_model_tensors_provenance, map_tensor_names, quantize_tensors,
    NativeF32Tensors,
};
// NOTE: quantize_q4_k_matrix imported via super:: through mod.rs

// GH-202 FIX: Removed transpose_f32_rowmajor_to_colmajor.
// GGML data layout is C row-major with reversed shape [ne0, ne1].
// No data transposition is needed for GGUF export — only shape reversal.

/// GH-236: Resolve architecture name from APR metadata.
///
/// Checks `architecture` field first, then falls back to `model_type` (for models
/// imported without explicit architecture, e.g. GPT-2 from SafeTensors), then
/// defaults to "qwen2" as the most common architecture.
fn resolve_architecture(apr_metadata: &crate::format::v2::AprV2Metadata) -> &str {
    apr_metadata
        .architecture
        .as_deref()
        .or_else(|| {
            let mt = &apr_metadata.model_type;
            if mt.is_empty() || mt == "unknown" {
                None
            } else {
                Some(mt.as_str())
            }
        })
        .unwrap_or("qwen2")
}

/// GH-277: Resolve GGUF pre-tokenizer type from architecture and model metadata.
///
/// The `tokenizer.ggml.pre` field identifies the pre-tokenizer regex patterns
/// used by llama.cpp for text splitting before BPE. This is NOT the same as
/// the model architecture.
///
/// Reference: llama.cpp/convert_hf_to_gguf.py `get_vocab_base_pre()`
fn resolve_pre_tokenizer_type(arch: &str, model_name: &str) -> &'static str {
    let name_lower = model_name.to_lowercase();
    // SmolLM models use "default" pre-tokenizer despite llama architecture
    if name_lower.contains("smollm") {
        return "default";
    }
    match arch {
        "gpt2" => "gpt-2",
        "qwen2" | "qwen2.5" | "qwen" => "qwen2",
        "llama" | "mistral" => "default",
        "phi" | "phi3" => "default",
        "gemma" | "gemma2" => "default",
        "deepseek" | "deepseek2" => "deepseek",
        "starcoder" | "starcoder2" => "starcoder",
        _ => "default", // Safe fallback per llama.cpp convention
    }
}

/// GH-277: Check if architecture uses RoPE (vs learned position embeddings).
///
/// GPT-2 and StarCoder use learned position embeddings, not RoPE.
/// All other transformer architectures in the GGUF ecosystem use RoPE.
fn uses_rope(arch: &str) -> bool {
    !matches!(arch, "gpt2" | "starcoder")
}

/// GH-277: Contract-driven APR→GGUF tensor name mapping.
///
/// Uses the model family contract's `tensor_template` (APR names) and
/// `gguf_tensor_template` (GGUF names) to build a mapping. This replaces
/// the old hardcoded `hf_to_gguf_name()` function.
///
/// Returns a `GgufNameMapper` that converts APR tensor names to GGUF names.
/// Tensors with `None` in gguf_tensor_template are skipped.
struct GgufNameMapper {
    /// Global tensor mapping: APR name → GGUF name
    global: std::collections::HashMap<String, String>,
    /// Per-layer suffix mapping: APR suffix (after "model.layers.{n}.") → GGUF suffix
    per_layer: std::collections::HashMap<String, String>,
    /// Per-layer suffixes that should be skipped
    skip_suffixes: std::collections::HashSet<String>,
    /// GH-277: If true, transpose 2D Conv1D weight tensors to Linear layout during export.
    transpose_weights: bool,
    /// GH-277: Fusion rules — concatenate multiple APR per-layer tensors into one GGUF tensor.
    /// Each entry: (gguf_suffix, [(apr_suffix, apr_role), ...])
    fusion_rules: Vec<FusionExportRule>,
}

/// A resolved fusion rule: maps APR per-layer suffixes to a single GGUF suffix.
struct FusionExportRule {
    /// GGUF suffix for the fused tensor (e.g., "attn_qkv.weight")
    gguf_suffix: String,
    /// APR per-layer suffixes to concatenate, in order
    apr_suffixes: Vec<String>,
}

/// Build global tensor name mappings (embedding, lm_head, norms, position embedding).
fn build_global_mappings(
    tt: &crate::format::model_family::TensorTemplate,
    gt: &crate::format::model_family::GgufTensorTemplate,
) -> std::collections::HashMap<String, String> {
    let mut global = std::collections::HashMap::new();

    if let Some(gguf_name) = &gt.embedding {
        global.insert(tt.embedding.clone(), gguf_name.clone());
    }
    if let (Some(apr_name), Some(gguf_name)) = (&tt.lm_head, &gt.lm_head) {
        global.insert(apr_name.clone(), gguf_name.clone());
    }
    if let (Some(apr_name), Some(gguf_name)) = (&tt.final_norm, &gt.final_norm_weight) {
        global.insert(apr_name.clone(), gguf_name.clone());
    }
    if let (Some(apr_norm), Some(gguf_bias)) = (&tt.final_norm, &gt.final_norm_bias) {
        let apr_bias = apr_norm.replace(".weight", ".bias");
        global.insert(apr_bias, gguf_bias.clone());
    }
    if let Some(gguf_name) = &gt.position_embedding {
        if let Some(Some(apr_name)) = tt.per_layer.get("position_embedding") {
            global.insert(apr_name.clone(), gguf_name.clone());
        } else {
            global.insert(
                "model.position_embedding.weight".to_string(),
                gguf_name.clone(),
            );
        }
    }
    global
}

/// Resolve fusion rules from GGUF template against tensor template to get APR suffixes.
fn resolve_fusion_rules(
    tt: &crate::format::model_family::TensorTemplate,
    gt: &crate::format::model_family::GgufTensorTemplate,
) -> Vec<FusionExportRule> {
    gt.fuse
        .iter()
        .filter_map(|rule| {
            let apr_suffixes: Vec<String> = rule
                .source_roles
                .iter()
                .filter_map(|role| {
                    tt.per_layer.get(role.as_str()).and_then(|opt| {
                        opt.as_ref().map(|pat| {
                            pat.strip_prefix("model.layers.{n}.")
                                .unwrap_or(pat)
                                .to_string()
                        })
                    })
                })
                .collect();

            if apr_suffixes.len() == rule.source_roles.len() {
                Some(FusionExportRule {
                    gguf_suffix: rule.gguf_suffix.clone(),
                    apr_suffixes,
                })
            } else {
                eprintln!(
                    "[GH-277] WARNING: Fusion rule `{}` has unresolved source roles, skipping",
                    rule.gguf_suffix
                );
                None
            }
        })
        .collect()
}

impl GgufNameMapper {
    /// Build a mapper from a model family config's tensor template + gguf template.
    fn from_contract(config: &crate::format::model_family::ModelFamilyConfig) -> Self {
        let tt = &config.tensor_template;
        let gt = &config.gguf_tensor_template;

        let global = build_global_mappings(tt, gt);
        let mut per_layer = std::collections::HashMap::new();
        let mut skip_suffixes = std::collections::HashSet::new();

        for (role, apr_pattern_opt) in &tt.per_layer {
            let Some(apr_pattern) = apr_pattern_opt else {
                continue;
            };
            if role == "position_embedding" {
                continue;
            }

            let apr_suffix = apr_pattern
                .strip_prefix("model.layers.{n}.")
                .unwrap_or(apr_pattern);

            match gt.per_layer.get(role) {
                Some(Some(gguf_suffix)) => {
                    per_layer.insert(apr_suffix.to_string(), gguf_suffix.clone());
                }
                Some(None) => {
                    skip_suffixes.insert(apr_suffix.to_string());
                }
                None => {}
            }
        }

        let fusion_rules = resolve_fusion_rules(tt, gt);

        Self {
            global,
            per_layer,
            skip_suffixes,
            transpose_weights: gt.transpose_weights,
            fusion_rules,
        }
    }

    /// Map an APR tensor name to its GGUF equivalent.
    /// Returns `None` if the tensor should be skipped (e.g., causal attention mask).
    fn map_name(&self, apr_name: &str) -> Option<String> {
        // Check global mappings first
        if let Some(gguf_name) = self.global.get(apr_name) {
            return Some(gguf_name.clone());
        }

        // Check per-layer: extract layer number and suffix
        if let Some(rest) = apr_name.strip_prefix("model.layers.") {
            if let Some(dot_pos) = rest.find('.') {
                let layer_num = &rest[..dot_pos];
                let suffix = &rest[dot_pos + 1..];

                // Check skip list
                if self.skip_suffixes.contains(suffix) {
                    return None;
                }

                // Look up suffix mapping
                if let Some(gguf_suffix) = self.per_layer.get(suffix) {
                    return Some(format!("blk.{layer_num}.{gguf_suffix}"));
                }
            }
        }

        // Fallback: pass through with warning
        eprintln!(
            "[GH-277] WARNING: No GGUF mapping for tensor '{}', passing through",
            apr_name
        );
        Some(apr_name.to_string())
    }

    /// GH-277: Get fusion rules for this mapper.
    fn fusion_rules(&self) -> &[FusionExportRule] {
        &self.fusion_rules
    }

    /// GH-277: Whether to transpose 2D weight tensors from Conv1D to Linear layout.
    fn needs_transpose(&self) -> bool {
        self.transpose_weights
    }
}

/// Build a `GgufNameMapper` from architecture name by looking up the family contract.
fn build_gguf_mapper(arch: &str) -> GgufNameMapper {
    let registry = crate::format::model_family::build_default_registry();

    // Try to find the family by architecture name
    let family = registry
        .get(arch)
        .or_else(|| registry.detect_from_model_type(arch));

    match family {
        Some(f) => {
            let config = f.config();
            if config.gguf_tensor_template.embedding.is_some() {
                eprintln!(
                    "[GH-277] Using contract-driven GGUF mapping for family '{}'",
                    config.family
                );
                GgufNameMapper::from_contract(config)
            } else {
                eprintln!(
                    "[GH-277] Family '{}' has no gguf_tensor_template, using legacy mapping",
                    config.family
                );
                build_legacy_mapper()
            }
        }
        None => {
            eprintln!(
                "[GH-277] No family contract for arch '{}', using legacy mapping",
                arch
            );
            build_legacy_mapper()
        }
    }
}

/// Build a legacy mapper that replicates the old `hf_to_gguf_name()` behavior
/// for architectures that don't have gguf_tensor_template in their contracts.
fn build_legacy_mapper() -> GgufNameMapper {
    let mut global = std::collections::HashMap::new();
    global.insert(
        "model.embed_tokens.weight".to_string(),
        "token_embd.weight".to_string(),
    );
    global.insert("lm_head.weight".to_string(), "output.weight".to_string());
    global.insert(
        "model.norm.weight".to_string(),
        "output_norm.weight".to_string(),
    );

    let mut per_layer = std::collections::HashMap::new();
    per_layer.insert(
        "self_attn.q_proj.weight".to_string(),
        "attn_q.weight".to_string(),
    );
    per_layer.insert(
        "self_attn.q_proj.bias".to_string(),
        "attn_q.bias".to_string(),
    );
    per_layer.insert(
        "self_attn.k_proj.weight".to_string(),
        "attn_k.weight".to_string(),
    );
    per_layer.insert(
        "self_attn.k_proj.bias".to_string(),
        "attn_k.bias".to_string(),
    );
    per_layer.insert(
        "self_attn.v_proj.weight".to_string(),
        "attn_v.weight".to_string(),
    );
    per_layer.insert(
        "self_attn.v_proj.bias".to_string(),
        "attn_v.bias".to_string(),
    );
    per_layer.insert(
        "self_attn.o_proj.weight".to_string(),
        "attn_output.weight".to_string(),
    );
    per_layer.insert(
        "self_attn.o_proj.bias".to_string(),
        "attn_output.bias".to_string(),
    );
    per_layer.insert(
        "self_attn.qkv_proj.weight".to_string(),
        "attn_qkv.weight".to_string(),
    );
    per_layer.insert(
        "self_attn.qkv_proj.bias".to_string(),
        "attn_qkv.bias".to_string(),
    );
    per_layer.insert(
        "input_layernorm.weight".to_string(),
        "attn_norm.weight".to_string(),
    );
    per_layer.insert(
        "self_attn.q_norm.weight".to_string(),
        "attn_q_norm.weight".to_string(),
    );
    per_layer.insert(
        "self_attn.k_norm.weight".to_string(),
        "attn_k_norm.weight".to_string(),
    );
    per_layer.insert(
        "mlp.gate_proj.weight".to_string(),
        "ffn_gate.weight".to_string(),
    );
    per_layer.insert(
        "mlp.up_proj.weight".to_string(),
        "ffn_up.weight".to_string(),
    );
    per_layer.insert(
        "mlp.down_proj.weight".to_string(),
        "ffn_down.weight".to_string(),
    );
    per_layer.insert(
        "post_attention_layernorm.weight".to_string(),
        "ffn_norm.weight".to_string(),
    );

    GgufNameMapper {
        global,
        per_layer,
        skip_suffixes: std::collections::HashSet::new(),
        transpose_weights: false,
        fusion_rules: Vec::new(),
    }
}

/// GH-277: Detect the number of layers from the tensor names.
/// Scans for "model.layers.{n}." and returns max(n) + 1.
fn detect_num_layers_from_names<'a>(names: impl Iterator<Item = &'a str>) -> usize {
    let mut max_layer = 0usize;
    for name in names {
        if let Some(rest) = name.strip_prefix("model.layers.") {
            if let Some(dot_pos) = rest.find('.') {
                if let Ok(n) = rest[..dot_pos].parse::<usize>() {
                    max_layer = max_layer.max(n + 1);
                }
            }
        }
    }
    max_layer
}

/// Transpose a 2D f32 matrix from `[rows, cols]` to `[cols, rows]` (Conv1D → Linear).
///
/// PMAT-285: Delegates to trueno's cache-blocked transpose.
fn transpose_2d_f32(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    trueno::blis::transpose::transpose(rows, cols, data, &mut out)
        .expect("transpose_2d_f32: dimension mismatch");
    out
}

/// Compute the concatenated shape from per-source shapes (concat along dim 0).
/// Returns `None` if shapes are incompatible or empty.
fn compute_fused_shape(shapes: &[Vec<usize>]) -> Option<Vec<usize>> {
    let first = shapes.first()?;
    match first.len() {
        2 => {
            let total_rows: usize = shapes.iter().map(|s| s[0]).sum();
            Some(vec![total_rows, first[1]])
        }
        1 => {
            let total: usize = shapes.iter().map(|s| s[0]).sum();
            Some(vec![total])
        }
        _ => None,
    }
}

/// Convert a row-major shape to GGUF ne-order (reversed for 2D).
fn shape_to_gguf(shape: &[usize]) -> Vec<u64> {
    if shape.len() == 2 {
        vec![shape[1] as u64, shape[0] as u64]
    } else {
        shape.iter().map(|&d| d as u64).collect()
    }
}

/// Collect and optionally transpose source tensors for one fusion rule + layer.
/// Returns `(concatenated_data, per_source_shapes)` or `None` if any source is missing.
fn collect_fusion_sources(
    rule: &FusionExportRule,
    layer: usize,
    tensors: &std::collections::BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    needs_transpose: bool,
) -> Option<(Vec<f32>, Vec<Vec<usize>>)> {
    let is_weight = rule.gguf_suffix.ends_with(".weight");
    let mut all_data: Vec<f32> = Vec::new();
    let mut all_shapes: Vec<Vec<usize>> = Vec::new();

    for apr_suffix in &rule.apr_suffixes {
        let apr_name = format!("model.layers.{layer}.{apr_suffix}");
        let (data, shape) = tensors.get(&apr_name)?;

        if needs_transpose && is_weight && shape.len() == 2 {
            let transposed = transpose_2d_f32(data, shape[0], shape[1]);
            all_data.extend_from_slice(&transposed);
            all_shapes.push(vec![shape[1], shape[0]]);
        } else {
            all_data.extend_from_slice(data);
            all_shapes.push(shape.clone());
        }
    }
    Some((all_data, all_shapes))
}

/// GH-277: Build fused tensors for the F32 export path.
///
/// For each fusion rule and each layer, looks up source tensors by APR name,
/// concatenates their f32 data, and returns the fused GGUF tensors.
fn build_fused_tensors_f32(
    mapper: &GgufNameMapper,
    tensors: &std::collections::BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    use_q4k: bool,
) -> Vec<crate::format::gguf::GgufTensor> {
    use crate::format::gguf::{GgmlType, GgufTensor};

    let rules = mapper.fusion_rules();
    if rules.is_empty() {
        return Vec::new();
    }

    let num_layers = detect_num_layers_from_names(tensors.keys().map(|s| s.as_str()));
    let needs_transpose = mapper.needs_transpose();
    let mut fused = Vec::new();

    for rule in rules {
        for layer in 0..num_layers {
            let Some((all_data, all_shapes)) =
                collect_fusion_sources(rule, layer, tensors, needs_transpose)
            else {
                continue;
            };

            let Some(fused_shape) = compute_fused_shape(&all_shapes) else {
                continue;
            };

            let gguf_shape = shape_to_gguf(&fused_shape);
            let gguf_name = format!("blk.{layer}.{}", rule.gguf_suffix);

            let (dtype, bytes) = if use_q4k && fused_shape.len() == 2 && all_data.len() >= 256 {
                let gguf_shape_usize = vec![fused_shape[1], fused_shape[0]];
                let q4k_bytes = super::quantize_q4_k_matrix(&all_data, &gguf_shape_usize);
                (GgmlType::Q4K, q4k_bytes)
            } else {
                let f32_bytes: Vec<u8> = all_data.iter().flat_map(|f| f.to_le_bytes()).collect();
                (GgmlType::F32, f32_bytes)
            };

            eprintln!(
                "[GH-277] Fused `{}` from {} sources ({} elements)",
                gguf_name,
                rule.apr_suffixes.len(),
                all_data.len()
            );

            fused.push(GgufTensor {
                name: gguf_name,
                shape: gguf_shape,
                dtype,
                data: bytes,
            });
        }
    }

    fused
}

/// GH-277: Build fused tensors for the raw APR→GGUF export path.
///
/// For each fusion rule and each layer, reads raw tensor bytes from the APR reader,
/// concatenates them, and returns fused GGUF tensors.
/// Map APR tensor dtype to GGML type for raw byte fusion.
fn apr_dtype_to_ggml(dtype: crate::format::v2::TensorDType) -> crate::format::gguf::GgmlType {
    use crate::format::gguf::GgmlType;
    use crate::format::v2::TensorDType;
    match dtype {
        TensorDType::F32 => GgmlType::F32,
        TensorDType::F16 => GgmlType::F16,
        TensorDType::Q4K => GgmlType::Q4K,
        TensorDType::Q6K => GgmlType::Q6K,
        TensorDType::Q8 => GgmlType::Q8_0,
        _ => GgmlType::F32,
    }
}

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
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: ExportFormat::SafeTensors,
            quantize: None,
            include_tokenizer: true, // Default to true for HuggingFace compatibility
            include_config: true,
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

include!("export_part_02.rs");
include!("export_part_03.rs");
include!("export_part_04.rs");
include!("export_part_05.rs");
