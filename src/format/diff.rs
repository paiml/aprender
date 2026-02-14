//! Model diff library (TOOL-APR-002)
//!
//! Provides format-agnostic model comparison for APR, GGUF, and SafeTensors.
//!
//! Toyota Way: Kaizen - Continuous improvement through comparison.
//!
//! # Supported Formats
//!
//! | Format | Extension | Notes |
//! |--------|-----------|-------|
//! | APR | `.apr` | Native format, full metadata |
//! | GGUF | `.gguf` | llama.cpp format |
//! | SafeTensors | `.safetensors` | HuggingFace format |
//!
//! # Example
//!
//! ```rust,ignore
//! use aprender::format::diff::{diff_models, DiffOptions};
//!
//! let report = diff_models("model_a.gguf", "model_b.gguf", DiffOptions::default())?;
//! if report.is_identical() {
//!     println!("Models are identical");
//! } else {
//!     for diff in &report.differences {
//!         println!("{}: {} → {}", diff.field, diff.value1, diff.value2);
//!     }
//! }
//! ```

use crate::error::{AprenderError, Result};
use crate::format::rosetta::{FormatType, InspectionReport, RosettaStone};
use serde::Serialize;
use std::path::Path;

// ============================================================================
// Types
// ============================================================================

/// A single difference between two models
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct DiffEntry {
    /// Name of the differing field
    pub field: String,
    /// Value from first model
    pub value1: String,
    /// Value from second model
    pub value2: String,
    /// Category of difference
    pub category: DiffCategory,
}

/// Category of difference for filtering
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum DiffCategory {
    /// Format-level difference (magic, version)
    Format,
    /// Metadata difference (name, description, architecture)
    Metadata,
    /// Tensor-level difference (count, names, shapes)
    Tensor,
    /// Quantization difference
    Quantization,
    /// Size difference
    Size,
}

impl DiffCategory {
    /// Get human-readable name
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Format => "format",
            Self::Metadata => "metadata",
            Self::Tensor => "tensor",
            Self::Quantization => "quantization",
            Self::Size => "size",
        }
    }
}

/// Options for diff operation
#[derive(Debug, Clone)]
pub struct DiffOptions {
    /// Include tensor-level comparison
    pub compare_tensors: bool,
    /// Include metadata comparison
    pub compare_metadata: bool,
    /// Compare tensor statistics (min, max, mean, std)
    pub compare_stats: bool,
    /// Tensor name filter (regex pattern)
    pub tensor_filter: Option<String>,
}

impl Default for DiffOptions {
    fn default() -> Self {
        Self {
            compare_tensors: true,
            compare_metadata: true,
            compare_stats: false,
            tensor_filter: None,
        }
    }
}

impl DiffOptions {
    /// Create new options with defaults
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable tensor comparison
    #[must_use]
    pub const fn with_tensors(mut self) -> Self {
        self.compare_tensors = true;
        self
    }

    /// Disable tensor comparison
    #[must_use]
    pub const fn without_tensors(mut self) -> Self {
        self.compare_tensors = false;
        self
    }

    /// Enable metadata comparison
    #[must_use]
    pub const fn with_metadata(mut self) -> Self {
        self.compare_metadata = true;
        self
    }

    /// Disable metadata comparison
    #[must_use]
    pub const fn without_metadata(mut self) -> Self {
        self.compare_metadata = false;
        self
    }

    /// Enable tensor statistics comparison
    #[must_use]
    pub const fn with_stats(mut self) -> Self {
        self.compare_stats = true;
        self
    }

    /// Set tensor name filter
    #[must_use]
    pub fn with_filter(mut self, pattern: &str) -> Self {
        self.tensor_filter = Some(pattern.to_string());
        self
    }
}

/// Complete diff report between two models
#[derive(Debug, Clone, Serialize)]
pub struct DiffReport {
    /// Path to first model
    pub path1: String,
    /// Path to second model
    pub path2: String,
    /// Format of first model
    pub format1: String,
    /// Format of second model
    pub format2: String,
    /// All differences found
    pub differences: Vec<DiffEntry>,
    /// First model inspection
    #[serde(skip)]
    pub inspection1: Option<InspectionReport>,
    /// Second model inspection
    #[serde(skip)]
    pub inspection2: Option<InspectionReport>,
}

impl DiffReport {
    /// Check if models are identical
    #[must_use]
    pub fn is_identical(&self) -> bool {
        self.differences.is_empty()
    }

    /// Get count of differences
    #[must_use]
    pub fn diff_count(&self) -> usize {
        self.differences.len()
    }

    /// Get differences by category
    #[must_use]
    pub fn differences_by_category(&self, category: DiffCategory) -> Vec<&DiffEntry> {
        self.differences
            .iter()
            .filter(|d| d.category == category)
            .collect()
    }

    /// Check if formats are the same
    #[must_use]
    pub fn same_format(&self) -> bool {
        self.format1 == self.format2
    }

    /// Get summary string
    #[must_use]
    pub fn summary(&self) -> String {
        if self.is_identical() {
            format!("{} and {} are IDENTICAL", self.path1, self.path2)
        } else {
            format!(
                "{} and {} differ in {} fields",
                self.path1,
                self.path2,
                self.diff_count()
            )
        }
    }
}

// ============================================================================
// Main Entry Points
// ============================================================================

/// Compare two model files
///
/// Supports APR, GGUF, and SafeTensors formats. Format is auto-detected
/// from magic bytes.
///
/// # Arguments
///
/// * `path1` - Path to first model
/// * `path2` - Path to second model
/// * `options` - Diff options
///
/// # Returns
///
/// `DiffReport` containing all differences
///
/// # Errors
///
/// Returns error if files cannot be read or formats are unsupported
pub fn diff_models<P: AsRef<Path>>(path1: P, path2: P, options: DiffOptions) -> Result<DiffReport> {
    let path1 = path1.as_ref();
    let path2 = path2.as_ref();

    // Validate paths
    validate_path(path1)?;
    validate_path(path2)?;

    // Detect formats
    let format1 = FormatType::from_magic(path1).or_else(|_| FormatType::from_extension(path1))?;
    let format2 = FormatType::from_magic(path2).or_else(|_| FormatType::from_extension(path2))?;

    // Get inspections
    let rosetta = RosettaStone::new();
    let inspection1 = rosetta.inspect(path1)?;
    let inspection2 = rosetta.inspect(path2)?;

    // Compute differences
    let differences = compute_differences(&inspection1, &inspection2, &options);

    Ok(DiffReport {
        path1: path1.display().to_string(),
        path2: path2.display().to_string(),
        format1: format1.to_string(),
        format2: format2.to_string(),
        differences,
        inspection1: Some(inspection1),
        inspection2: Some(inspection2),
    })
}

/// Compare two inspections directly (for advanced use)
///
/// # Arguments
///
/// * `inspection1` - First model inspection
/// * `inspection2` - Second model inspection
/// * `path1` - Path/name for first model
/// * `path2` - Path/name for second model
/// * `options` - Diff options
///
/// # Returns
///
/// `DiffReport` containing all differences
#[must_use]
pub fn diff_inspections(
    inspection1: &InspectionReport,
    inspection2: &InspectionReport,
    path1: &str,
    path2: &str,
    options: DiffOptions,
) -> DiffReport {
    let differences = compute_differences(inspection1, inspection2, &options);

    DiffReport {
        path1: path1.to_string(),
        path2: path2.to_string(),
        format1: inspection1.format.to_string(),
        format2: inspection2.format.to_string(),
        differences,
        inspection1: None,
        inspection2: None,
    }
}

// ============================================================================
// Internal Functions
// ============================================================================

/// Validate that path exists and is a file
fn validate_path(path: &Path) -> Result<()> {
    if !path.exists() {
        return Err(AprenderError::FormatError {
            message: format!("File not found: {}", path.display()),
        });
    }
    if !path.is_file() {
        return Err(AprenderError::FormatError {
            message: format!("Not a file: {}", path.display()),
        });
    }
    Ok(())
}

/// Compute all differences between two inspections
fn compute_differences(
    i1: &InspectionReport,
    i2: &InspectionReport,
    options: &DiffOptions,
) -> Vec<DiffEntry> {
    let mut diffs = Vec::new();

    // Format differences
    if i1.format != i2.format {
        diffs.push(DiffEntry {
            field: "format".to_string(),
            value1: i1.format.to_string(),
            value2: i2.format.to_string(),
            category: DiffCategory::Format,
        });
    }

    // Size differences
    if i1.file_size != i2.file_size {
        diffs.push(DiffEntry {
            field: "file_size".to_string(),
            value1: format_size(i1.file_size),
            value2: format_size(i2.file_size),
            category: DiffCategory::Size,
        });
    }

    if i1.total_params != i2.total_params {
        diffs.push(DiffEntry {
            field: "total_params".to_string(),
            value1: format_params(i1.total_params),
            value2: format_params(i2.total_params),
            category: DiffCategory::Size,
        });
    }

    // Architecture
    if i1.architecture != i2.architecture {
        diffs.push(DiffEntry {
            field: "architecture".to_string(),
            value1: i1
                .architecture
                .clone()
                .unwrap_or_else(|| "(none)".to_string()),
            value2: i2
                .architecture
                .clone()
                .unwrap_or_else(|| "(none)".to_string()),
            category: DiffCategory::Metadata,
        });
    }

    // Quantization
    if i1.quantization != i2.quantization {
        diffs.push(DiffEntry {
            field: "quantization".to_string(),
            value1: i1
                .quantization
                .clone()
                .unwrap_or_else(|| "(none)".to_string()),
            value2: i2
                .quantization
                .clone()
                .unwrap_or_else(|| "(none)".to_string()),
            category: DiffCategory::Quantization,
        });
    }

    // Metadata comparison
    if options.compare_metadata {
        compare_metadata(&i1.metadata, &i2.metadata, &mut diffs);
    }

    // Tensor comparison
    if options.compare_tensors {
        compare_tensors(&i1.tensors, &i2.tensors, options, &mut diffs);
    }

    diffs
}

/// Compare metadata maps
fn compare_metadata(
    m1: &std::collections::BTreeMap<String, String>,
    m2: &std::collections::BTreeMap<String, String>,
    diffs: &mut Vec<DiffEntry>,
) {
    // Keys in m1 but not m2 or different
    for (key, val1) in m1 {
        match m2.get(key) {
            Some(val2) if val1 != val2 => {
                diffs.push(DiffEntry {
                    field: format!("metadata.{key}"),
                    value1: truncate_value(val1, 50),
                    value2: truncate_value(val2, 50),
                    category: DiffCategory::Metadata,
                });
            }
            None => {
                diffs.push(DiffEntry {
                    field: format!("metadata.{key}"),
                    value1: truncate_value(val1, 50),
                    value2: "(missing)".to_string(),
                    category: DiffCategory::Metadata,
                });
            }
            _ => {}
        }
    }

    // Keys in m2 but not m1
    for (key, val2) in m2 {
        if !m1.contains_key(key) {
            diffs.push(DiffEntry {
                field: format!("metadata.{key}"),
                value1: "(missing)".to_string(),
                value2: truncate_value(val2, 50),
                category: DiffCategory::Metadata,
            });
        }
    }
}

/// GH-202 FIX: Map GGUF tensor names to APR canonical names for cross-format comparison.
/// Returns both the mapped name and whether mapping was applied.
fn map_gguf_to_apr_name(name: &str) -> (String, bool) {
    // Handle layer-specific tensors (blk.N.*)
    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some(dot_pos) = rest.find('.') {
            let layer_num = &rest[..dot_pos];
            let suffix = &rest[dot_pos + 1..];

            let apr_suffix = match suffix {
                "attn_q.weight" => "self_attn.q_proj.weight",
                "attn_q.bias" => "self_attn.q_proj.bias",
                "attn_k.weight" => "self_attn.k_proj.weight",
                "attn_k.bias" => "self_attn.k_proj.bias",
                "attn_v.weight" => "self_attn.v_proj.weight",
                "attn_v.bias" => "self_attn.v_proj.bias",
                "attn_output.weight" => "self_attn.o_proj.weight",
                "attn_output.bias" => "self_attn.o_proj.bias",
                "attn_norm.weight" => "input_layernorm.weight",
                "ffn_gate.weight" => "mlp.gate_proj.weight",
                "ffn_up.weight" => "mlp.up_proj.weight",
                "ffn_down.weight" => "mlp.down_proj.weight",
                "ffn_norm.weight" => "post_attention_layernorm.weight",
                _ => return (name.to_string(), false),
            };
            return (format!("model.layers.{layer_num}.{apr_suffix}"), true);
        }
    }

    // Handle non-layer tensors
    match name {
        "token_embd.weight" => ("model.embed_tokens.weight".to_string(), true),
        "output.weight" => ("lm_head.weight".to_string(), true),
        "output_norm.weight" => ("model.norm.weight".to_string(), true),
        _ => (name.to_string(), false),
    }
}

/// GH-202 FIX: Build cross-format name mapping for tensor comparison.
/// Creates a HashMap from APR canonical names to model 2 tensors.
fn build_cross_format_map(
    tensors: &[crate::format::rosetta::TensorInfo],
) -> std::collections::HashMap<String, &crate::format::rosetta::TensorInfo> {
    let mut map = std::collections::HashMap::new();
    for t in tensors {
        // Add both original name and mapped name
        map.insert(t.name.clone(), t);
        let (mapped, was_mapped) = map_gguf_to_apr_name(&t.name);
        if was_mapped {
            map.insert(mapped, t);
        }
    }
    map
}

/// GH-202 FIX: Check if two tensor shapes are compatible across formats.
/// GGUF uses [in, out] while APR uses [out, in], so transposed 2D shapes are equivalent.
fn shapes_are_compatible(shape1: &[usize], shape2: &[usize]) -> bool {
    shape1 == shape2
        || (shape1.len() == 2
            && shape2.len() == 2
            && shape1[0] == shape2[1]
            && shape1[1] == shape2[0])
}

/// Compare shapes of two matched tensors and push a diff if incompatible.
fn compare_tensor_shapes(
    tensor1: &crate::format::rosetta::TensorInfo,
    tensor2: &crate::format::rosetta::TensorInfo,
    diffs: &mut Vec<DiffEntry>,
) {
    if !shapes_are_compatible(&tensor1.shape, &tensor2.shape) {
        diffs.push(DiffEntry {
            field: format!("tensor.{}.shape", tensor1.name),
            value1: format!("{:?}", tensor1.shape),
            value2: format!("{:?} (mapped: {})", tensor2.shape, tensor2.name),
            category: DiffCategory::Tensor,
        });
    }
}

/// Compare dtypes of two matched tensors, allowing compatible quantization variants.
/// GH-202 FIX: Show normalized dtype names for readability.
fn compare_tensor_dtypes(
    tensor1: &crate::format::rosetta::TensorInfo,
    tensor2: &crate::format::rosetta::TensorInfo,
    diffs: &mut Vec<DiffEntry>,
) {
    let dtypes_compatible =
        tensor1.dtype == tensor2.dtype || is_compatible_quant(&tensor1.dtype, &tensor2.dtype);

    if !dtypes_compatible {
        // GH-256: Categorize dtype diffs as Quantization, not Tensor.
        // This lets consumers distinguish structural issues (missing/extra tensors,
        // shape mismatches) from expected dtype differences (int4 vs int8).
        diffs.push(DiffEntry {
            field: format!("tensor.{}.dtype", tensor1.name),
            value1: normalize_dtype(&tensor1.dtype),
            value2: normalize_dtype(&tensor2.dtype),
            category: DiffCategory::Quantization,
        });
    }
}

/// Look up a tensor by name in a cross-format map, trying direct match then mapped name.
fn find_tensor_in_map<'a>(
    name: &str,
    map: &'a std::collections::HashMap<String, &'a crate::format::rosetta::TensorInfo>,
) -> Option<&'a crate::format::rosetta::TensorInfo> {
    map.get(name).copied().or_else(|| {
        let (mapped, _) = map_gguf_to_apr_name(name);
        map.get(&mapped).copied()
    })
}

/// Report a tensor that exists in only one model.
fn report_missing_tensor(
    name: &str,
    shape: &[usize],
    dtype: &str,
    present_in_first: bool,
    diffs: &mut Vec<DiffEntry>,
) {
    let (v1, v2) = if present_in_first {
        (format!("{shape:?} {dtype}"), "(missing)".to_string())
    } else {
        ("(missing)".to_string(), format!("{shape:?} {dtype}"))
    };
    diffs.push(DiffEntry {
        field: format!("tensor.{name}"),
        value1: v1,
        value2: v2,
        category: DiffCategory::Tensor,
    });
}

/// Collect tensors only in model 2 that were not matched during the forward pass.
fn collect_unmatched_from_t2(
    t2: &[crate::format::rosetta::TensorInfo],
    matched_t2: &std::collections::HashSet<&str>,
    map1: &std::collections::HashMap<String, &crate::format::rosetta::TensorInfo>,
    matches_filter: &dyn Fn(&str) -> bool,
    diffs: &mut Vec<DiffEntry>,
) {
    for tensor2 in t2 {
        if !matches_filter(&tensor2.name) {
            continue;
        }
        if matched_t2.contains(tensor2.name.as_str()) {
            continue;
        }
        let can_match = find_tensor_in_map(&tensor2.name, map1).is_some();
        if !can_match {
            report_missing_tensor(&tensor2.name, &tensor2.shape, &tensor2.dtype, false, diffs);
        }
    }
}

/// Compare tensor lists with cross-format name mapping (GH-202 FIX)
fn compare_tensors(
    t1: &[crate::format::rosetta::TensorInfo],
    t2: &[crate::format::rosetta::TensorInfo],
    options: &DiffOptions,
    diffs: &mut Vec<DiffEntry>,
) {
    // Tensor count
    if t1.len() != t2.len() {
        diffs.push(DiffEntry {
            field: "tensor_count".to_string(),
            value1: t1.len().to_string(),
            value2: t2.len().to_string(),
            category: DiffCategory::Tensor,
        });
    }

    // GH-202 FIX: Build cross-format maps for lookup
    // This allows GGUF blk.0.attn_q.weight to match APR model.layers.0.self_attn.q_proj.weight
    let map1 = build_cross_format_map(t1);
    let map2 = build_cross_format_map(t2);

    // Filter function
    let matches_filter = |name: &str| -> bool {
        options
            .tensor_filter
            .as_ref()
            .map_or(true, |pattern| name.contains(pattern.as_str()))
    };

    // Track which tensors in t2 have been matched
    let mut matched_t2: std::collections::HashSet<&str> = std::collections::HashSet::new();

    // Check tensors in model 1
    for tensor1 in t1 {
        if !matches_filter(&tensor1.name) {
            continue;
        }

        let Some(tensor2) = find_tensor_in_map(&tensor1.name, &map2) else {
            report_missing_tensor(&tensor1.name, &tensor1.shape, &tensor1.dtype, true, diffs);
            continue;
        };

        matched_t2.insert(&tensor2.name);
        compare_tensor_shapes(tensor1, tensor2, diffs);
        compare_tensor_dtypes(tensor1, tensor2, diffs);

        if options.compare_stats {
            compare_tensor_stats(tensor1, tensor2, diffs);
        }
    }

    // Check for tensors only in model 2 (not matched via name mapping)
    collect_unmatched_from_t2(t2, &matched_t2, &map1, &matches_filter, diffs);
}

/// GH-202 FIX: Normalize GGUF numeric dtype to string name
fn normalize_dtype(dtype: &str) -> String {
    // GGUF numeric codes: https://github.com/ggerganov/ggml/blob/master/include/ggml.h
    match dtype {
        "0" | "f32" | "F32" => "F32".to_string(),
        "1" | "f16" | "F16" => "F16".to_string(),
        "2" | "q4_0" | "Q4_0" => "Q4_0".to_string(),
        "3" | "q4_1" | "Q4_1" => "Q4_1".to_string(),
        "6" | "q5_0" | "Q5_0" => "Q5_0".to_string(),
        "7" | "q5_1" | "Q5_1" => "Q5_1".to_string(),
        "8" | "q8_0" | "Q8_0" => "Q8_0".to_string(),
        "9" | "q8_1" | "Q8_1" => "Q8_1".to_string(),
        "10" | "q2_k" | "Q2_K" | "q2k" | "Q2K" => "Q2_K".to_string(),
        "11" | "q3_k" | "Q3_K" | "q3k" | "Q3K" => "Q3_K".to_string(),
        "12" | "q4_k" | "Q4_K" | "q4k" | "Q4K" => "Q4_K".to_string(),
        "13" | "q5_k" | "Q5_K" | "q5k" | "Q5K" => "Q5_K".to_string(),
        "14" | "q6_k" | "Q6_K" | "q6k" | "Q6K" => "Q6_K".to_string(),
        "15" | "q8_k" | "Q8_K" | "q8k" | "Q8K" => "Q8_K".to_string(),
        "16" | "iq2_xxs" | "IQ2_XXS" => "IQ2_XXS".to_string(),
        "17" | "iq2_xs" | "IQ2_XS" => "IQ2_XS".to_string(),
        "18" | "iq3_xxs" | "IQ3_XXS" => "IQ3_XXS".to_string(),
        "19" | "iq1_s" | "IQ1_S" => "IQ1_S".to_string(),
        "bf16" | "BF16" => "BF16".to_string(),
        other => other.to_uppercase(),
    }
}

/// GH-202 FIX: Check if two quantization types are compatible
fn is_compatible_quant(dtype1: &str, dtype2: &str) -> bool {
    // Normalize both to comparable format
    let d1 = normalize_dtype(dtype1);
    let d2 = normalize_dtype(dtype2);

    // Same type after normalization
    if d1 == d2 {
        return true;
    }

    // Q5_0/Q5_K/Q5_1 are all 5-bit quantization variants
    let is_q5 = |d: &str| d.starts_with("Q5");

    // Q4_0/Q4_K/Q4_1 are all 4-bit quantization variants
    let is_q4 = |d: &str| d.starts_with("Q4");

    // Q6_K is 6-bit quantization
    let is_q6 = |d: &str| d.starts_with("Q6");

    // Q8_0/Q8_K is 8-bit quantization
    let is_q8 = |d: &str| d.starts_with("Q8");

    // Allow Q5→Q6 conversion (common import path)
    if (is_q5(&d1) && is_q6(&d2)) || (is_q6(&d1) && is_q5(&d2)) {
        return true;
    }

    // Allow Q4→Q4 variants
    if is_q4(&d1) && is_q4(&d2) {
        return true;
    }

    // Allow Q8→Q6 (downgrade) or Q5→Q4 (downgrade)
    if (is_q8(&d1) && is_q6(&d2)) || (is_q6(&d1) && is_q8(&d2)) {
        return true;
    }

    false
}

/// Compare tensor statistics
fn compare_tensor_stats(
    t1: &crate::format::rosetta::TensorInfo,
    t2: &crate::format::rosetta::TensorInfo,
    diffs: &mut Vec<DiffEntry>,
) {
    match (&t1.stats, &t2.stats) {
        (Some(s1), Some(s2)) => {
            let epsilon = 1e-4;

            if (s1.min - s2.min).abs() > epsilon {
                diffs.push(DiffEntry {
                    field: format!("tensor.{}.min", t1.name),
                    value1: format!("{:.6}", s1.min),
                    value2: format!("{:.6}", s2.min),
                    category: DiffCategory::Tensor,
                });
            }

            if (s1.max - s2.max).abs() > epsilon {
                diffs.push(DiffEntry {
                    field: format!("tensor.{}.max", t1.name),
                    value1: format!("{:.6}", s1.max),
                    value2: format!("{:.6}", s2.max),
                    category: DiffCategory::Tensor,
                });
            }

            if (s1.mean - s2.mean).abs() > epsilon {
                diffs.push(DiffEntry {
                    field: format!("tensor.{}.mean", t1.name),
                    value1: format!("{:.6}", s1.mean),
                    value2: format!("{:.6}", s2.mean),
                    category: DiffCategory::Tensor,
                });
            }

            if (s1.std - s2.std).abs() > epsilon {
                diffs.push(DiffEntry {
                    field: format!("tensor.{}.std", t1.name),
                    value1: format!("{:.6}", s1.std),
                    value2: format!("{:.6}", s2.std),
                    category: DiffCategory::Tensor,
                });
            }
        }
        (Some(_), None) => {
            diffs.push(DiffEntry {
                field: format!("tensor.{}.stats", t1.name),
                value1: "present".to_string(),
                value2: "(none)".to_string(),
                category: DiffCategory::Tensor,
            });
        }
        (None, Some(_)) => {
            diffs.push(DiffEntry {
                field: format!("tensor.{}.stats", t1.name),
                value1: "(none)".to_string(),
                value2: "present".to_string(),
                category: DiffCategory::Tensor,
            });
        }
        (None, None) => {}
    }
}

/// Format byte size for display
fn format_size(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Format parameter count for display
fn format_params(params: usize) -> String {
    const K: usize = 1_000;
    const M: usize = K * 1_000;
    const B: usize = M * 1_000;

    if params >= B {
        format!("{:.2}B", params as f64 / B as f64)
    } else if params >= M {
        format!("{:.2}M", params as f64 / M as f64)
    } else if params >= K {
        format!("{:.2}K", params as f64 / K as f64)
    } else {
        params.to_string()
    }
}

/// Truncate value for display
fn truncate_value(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", &s[..max_len])
    } else {
        s.to_string()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[path = "diff_tests.rs"]
mod tests;

