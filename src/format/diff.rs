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
mod tests {
    use super::*;

    // ========================================================================
    // DiffOptions Tests
    // ========================================================================

    #[test]
    fn test_diff_options_default() {
        let opts = DiffOptions::default();
        assert!(opts.compare_tensors);
        assert!(opts.compare_metadata);
        assert!(!opts.compare_stats);
        assert!(opts.tensor_filter.is_none());
    }

    #[test]
    fn test_diff_options_new() {
        let opts = DiffOptions::new();
        assert!(opts.compare_tensors);
        assert!(opts.compare_metadata);
    }

    #[test]
    fn test_diff_options_with_tensors() {
        let opts = DiffOptions::new().without_tensors().with_tensors();
        assert!(opts.compare_tensors);
    }

    #[test]
    fn test_diff_options_without_tensors() {
        let opts = DiffOptions::new().without_tensors();
        assert!(!opts.compare_tensors);
    }

    #[test]
    fn test_diff_options_with_metadata() {
        let opts = DiffOptions::new().without_metadata().with_metadata();
        assert!(opts.compare_metadata);
    }

    #[test]
    fn test_diff_options_without_metadata() {
        let opts = DiffOptions::new().without_metadata();
        assert!(!opts.compare_metadata);
    }

    #[test]
    fn test_diff_options_with_stats() {
        let opts = DiffOptions::new().with_stats();
        assert!(opts.compare_stats);
    }

    #[test]
    fn test_diff_options_with_filter() {
        let opts = DiffOptions::new().with_filter("embed");
        assert_eq!(opts.tensor_filter, Some("embed".to_string()));
    }

    // ========================================================================
    // DiffCategory Tests
    // ========================================================================

    #[test]
    fn test_diff_category_names() {
        assert_eq!(DiffCategory::Format.name(), "format");
        assert_eq!(DiffCategory::Metadata.name(), "metadata");
        assert_eq!(DiffCategory::Tensor.name(), "tensor");
        assert_eq!(DiffCategory::Quantization.name(), "quantization");
        assert_eq!(DiffCategory::Size.name(), "size");
    }

    // ========================================================================
    // DiffEntry Tests
    // ========================================================================

    #[test]
    fn test_diff_entry_serialization() {
        let entry = DiffEntry {
            field: "version".to_string(),
            value1: "1.0".to_string(),
            value2: "2.0".to_string(),
            category: DiffCategory::Format,
        };
        let json = serde_json::to_string(&entry).expect("serialize");
        assert!(json.contains("version"));
        assert!(json.contains("1.0"));
        assert!(json.contains("2.0"));
        assert!(json.contains("Format"));
    }

    #[test]
    fn test_diff_entry_equality() {
        let entry1 = DiffEntry {
            field: "test".to_string(),
            value1: "a".to_string(),
            value2: "b".to_string(),
            category: DiffCategory::Metadata,
        };
        let entry2 = entry1.clone();
        assert_eq!(entry1, entry2);
    }

    // ========================================================================
    // DiffReport Tests
    // ========================================================================

    #[test]
    fn test_diff_report_identical() {
        let report = DiffReport {
            path1: "a.apr".to_string(),
            path2: "b.apr".to_string(),
            format1: "APR".to_string(),
            format2: "APR".to_string(),
            differences: vec![],
            inspection1: None,
            inspection2: None,
        };
        assert!(report.is_identical());
        assert_eq!(report.diff_count(), 0);
        assert!(report.same_format());
    }

    #[test]
    fn test_diff_report_with_differences() {
        let report = DiffReport {
            path1: "a.apr".to_string(),
            path2: "b.gguf".to_string(),
            format1: "APR".to_string(),
            format2: "GGUF".to_string(),
            differences: vec![
                DiffEntry {
                    field: "format".to_string(),
                    value1: "APR".to_string(),
                    value2: "GGUF".to_string(),
                    category: DiffCategory::Format,
                },
                DiffEntry {
                    field: "tensor_count".to_string(),
                    value1: "10".to_string(),
                    value2: "12".to_string(),
                    category: DiffCategory::Tensor,
                },
            ],
            inspection1: None,
            inspection2: None,
        };
        assert!(!report.is_identical());
        assert_eq!(report.diff_count(), 2);
        assert!(!report.same_format());
    }

    #[test]
    fn test_diff_report_by_category() {
        let report = DiffReport {
            path1: "a.apr".to_string(),
            path2: "b.apr".to_string(),
            format1: "APR".to_string(),
            format2: "APR".to_string(),
            differences: vec![
                DiffEntry {
                    field: "tensor_count".to_string(),
                    value1: "10".to_string(),
                    value2: "12".to_string(),
                    category: DiffCategory::Tensor,
                },
                DiffEntry {
                    field: "metadata.name".to_string(),
                    value1: "model_a".to_string(),
                    value2: "model_b".to_string(),
                    category: DiffCategory::Metadata,
                },
                DiffEntry {
                    field: "tensor.embed.shape".to_string(),
                    value1: "[100]".to_string(),
                    value2: "[200]".to_string(),
                    category: DiffCategory::Tensor,
                },
            ],
            inspection1: None,
            inspection2: None,
        };

        let tensor_diffs = report.differences_by_category(DiffCategory::Tensor);
        assert_eq!(tensor_diffs.len(), 2);

        let metadata_diffs = report.differences_by_category(DiffCategory::Metadata);
        assert_eq!(metadata_diffs.len(), 1);

        let format_diffs = report.differences_by_category(DiffCategory::Format);
        assert_eq!(format_diffs.len(), 0);
    }

    #[test]
    fn test_diff_report_summary_identical() {
        let report = DiffReport {
            path1: "a.apr".to_string(),
            path2: "b.apr".to_string(),
            format1: "APR".to_string(),
            format2: "APR".to_string(),
            differences: vec![],
            inspection1: None,
            inspection2: None,
        };
        assert!(report.summary().contains("IDENTICAL"));
    }

    #[test]
    fn test_diff_report_summary_different() {
        let report = DiffReport {
            path1: "a.apr".to_string(),
            path2: "b.apr".to_string(),
            format1: "APR".to_string(),
            format2: "APR".to_string(),
            differences: vec![DiffEntry {
                field: "test".to_string(),
                value1: "a".to_string(),
                value2: "b".to_string(),
                category: DiffCategory::Metadata,
            }],
            inspection1: None,
            inspection2: None,
        };
        assert!(report.summary().contains("differ"));
        assert!(report.summary().contains("1"));
    }

    #[test]
    fn test_diff_report_serialization() {
        let report = DiffReport {
            path1: "a.apr".to_string(),
            path2: "b.apr".to_string(),
            format1: "APR".to_string(),
            format2: "APR".to_string(),
            differences: vec![],
            inspection1: None,
            inspection2: None,
        };
        let json = serde_json::to_string(&report).expect("serialize");
        assert!(json.contains("a.apr"));
        assert!(json.contains("b.apr"));
    }

    // ========================================================================
    // Helper Function Tests
    // ========================================================================

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(100), "100 B");
        assert_eq!(format_size(0), "0 B");
    }

    #[test]
    fn test_format_size_kb() {
        assert_eq!(format_size(1024), "1.00 KB");
        assert_eq!(format_size(2048), "2.00 KB");
        assert_eq!(format_size(1536), "1.50 KB");
    }

    #[test]
    fn test_format_size_mb() {
        assert_eq!(format_size(1024 * 1024), "1.00 MB");
        assert_eq!(format_size(10 * 1024 * 1024), "10.00 MB");
    }

    #[test]
    fn test_format_size_gb() {
        assert_eq!(format_size(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(format_size(2 * 1024 * 1024 * 1024), "2.00 GB");
    }

    #[test]
    fn test_format_params() {
        assert_eq!(format_params(100), "100");
        assert_eq!(format_params(1_000), "1.00K");
        assert_eq!(format_params(1_500), "1.50K");
        assert_eq!(format_params(1_000_000), "1.00M");
        assert_eq!(format_params(7_000_000_000), "7.00B");
    }

    #[test]
    fn test_truncate_value() {
        assert_eq!(truncate_value("short", 10), "short");
        assert_eq!(
            truncate_value("this is a very long string", 10),
            "this is a ..."
        );
    }

    // ========================================================================
    // Validate Path Tests
    // ========================================================================

    #[test]
    fn test_validate_path_not_found() {
        let result = validate_path(Path::new("/nonexistent/model.apr"));
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_path_is_directory() {
        use tempfile::tempdir;
        let dir = tempdir().expect("create dir");
        let result = validate_path(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_path_valid() {
        use tempfile::NamedTempFile;
        let file = NamedTempFile::new().expect("create file");
        let result = validate_path(file.path());
        assert!(result.is_ok());
    }

    // ========================================================================
    // Metadata Comparison Tests
    // ========================================================================

    #[test]
    fn test_compare_metadata_identical() {
        use std::collections::BTreeMap;
        let mut m1 = BTreeMap::new();
        m1.insert("key1".to_string(), "value1".to_string());
        m1.insert("key2".to_string(), "value2".to_string());

        let m2 = m1.clone();
        let mut diffs = Vec::new();
        compare_metadata(&m1, &m2, &mut diffs);
        assert!(diffs.is_empty());
    }

    #[test]
    fn test_compare_metadata_different_value() {
        use std::collections::BTreeMap;
        let mut m1 = BTreeMap::new();
        m1.insert("key1".to_string(), "value1".to_string());

        let mut m2 = BTreeMap::new();
        m2.insert("key1".to_string(), "value2".to_string());

        let mut diffs = Vec::new();
        compare_metadata(&m1, &m2, &mut diffs);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].field.contains("key1"));
    }

    #[test]
    fn test_compare_metadata_missing_key() {
        use std::collections::BTreeMap;
        let mut m1 = BTreeMap::new();
        m1.insert("key1".to_string(), "value1".to_string());
        m1.insert("key2".to_string(), "value2".to_string());

        let mut m2 = BTreeMap::new();
        m2.insert("key1".to_string(), "value1".to_string());

        let mut diffs = Vec::new();
        compare_metadata(&m1, &m2, &mut diffs);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].field.contains("key2"));
        assert!(diffs[0].value2.contains("missing"));
    }

    #[test]
    fn test_compare_metadata_extra_key() {
        use std::collections::BTreeMap;
        let mut m1 = BTreeMap::new();
        m1.insert("key1".to_string(), "value1".to_string());

        let mut m2 = BTreeMap::new();
        m2.insert("key1".to_string(), "value1".to_string());
        m2.insert("key2".to_string(), "value2".to_string());

        let mut diffs = Vec::new();
        compare_metadata(&m1, &m2, &mut diffs);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].field.contains("key2"));
        assert!(diffs[0].value1.contains("missing"));
    }

    // ========================================================================
    // Tensor Comparison Tests
    // ========================================================================

    #[test]
    fn test_compare_tensors_identical() {
        use crate::format::rosetta::TensorInfo;

        let t1 = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10, 20],
            size_bytes: 800,
            stats: None,
        }];
        let t2 = t1.clone();

        let mut diffs = Vec::new();
        let options = DiffOptions::default();
        compare_tensors(&t1, &t2, &options, &mut diffs);
        assert!(diffs.is_empty());
    }

    #[test]
    fn test_compare_tensors_different_count() {
        use crate::format::rosetta::TensorInfo;

        let t1 = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10, 20],
            size_bytes: 800,
            stats: None,
        }];
        let t2 = vec![
            TensorInfo {
                name: "weight".to_string(),
                dtype: "F32".to_string(),
                shape: vec![10, 20],
                size_bytes: 800,
                stats: None,
            },
            TensorInfo {
                name: "bias".to_string(),
                dtype: "F32".to_string(),
                shape: vec![10],
                size_bytes: 40,
                stats: None,
            },
        ];

        let mut diffs = Vec::new();
        let options = DiffOptions::default();
        compare_tensors(&t1, &t2, &options, &mut diffs);

        // Should have count diff and missing tensor diff
        assert!(diffs.iter().any(|d| d.field == "tensor_count"));
        assert!(diffs.iter().any(|d| d.field.contains("bias")));
    }

    #[test]
    fn test_compare_tensors_different_shape() {
        use crate::format::rosetta::TensorInfo;

        let t1 = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10, 20],
            size_bytes: 800,
            stats: None,
        }];
        let t2 = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![20, 20],
            size_bytes: 1600,
            stats: None,
        }];

        let mut diffs = Vec::new();
        let options = DiffOptions::default();
        compare_tensors(&t1, &t2, &options, &mut diffs);

        assert!(diffs.iter().any(|d| d.field.contains("shape")));
    }

    #[test]
    fn test_compare_tensors_different_dtype() {
        use crate::format::rosetta::TensorInfo;

        let t1 = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10, 20],
            size_bytes: 800,
            stats: None,
        }];
        let t2 = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "Q8_0".to_string(),
            shape: vec![10, 20],
            size_bytes: 400,
            stats: None,
        }];

        let mut diffs = Vec::new();
        let options = DiffOptions::default();
        compare_tensors(&t1, &t2, &options, &mut diffs);

        assert!(diffs.iter().any(|d| d.field.contains("dtype")));
    }

    #[test]
    fn test_compare_tensors_with_filter() {
        use crate::format::rosetta::TensorInfo;

        let t1 = vec![
            TensorInfo {
                name: "embed.weight".to_string(),
                dtype: "F32".to_string(),
                shape: vec![100],
                size_bytes: 400,
                stats: None,
            },
            TensorInfo {
                name: "lm_head.weight".to_string(),
                dtype: "F32".to_string(),
                shape: vec![100],
                size_bytes: 400,
                stats: None,
            },
        ];
        let t2 = vec![
            TensorInfo {
                name: "embed.weight".to_string(),
                dtype: "F32".to_string(),
                shape: vec![200], // Different
                size_bytes: 800,
                stats: None,
            },
            TensorInfo {
                name: "lm_head.weight".to_string(),
                dtype: "F32".to_string(),
                shape: vec![200], // Different
                size_bytes: 800,
                stats: None,
            },
        ];

        let mut diffs = Vec::new();
        let options = DiffOptions::new().with_filter("embed");
        compare_tensors(&t1, &t2, &options, &mut diffs);

        // Should only report embed differences due to filter
        assert!(diffs.iter().all(|d| d.field.contains("embed")));
    }

    #[test]
    fn test_compare_tensors_with_stats() {
        use crate::format::rosetta::{TensorInfo, TensorStats};

        let t1 = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10],
            size_bytes: 40,
            stats: Some(TensorStats {
                min: 0.0,
                max: 1.0,
                mean: 0.5,
                std: 0.1,
            }),
        }];
        let t2 = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10],
            size_bytes: 40,
            stats: Some(TensorStats {
                min: 0.0,
                max: 2.0,  // Different
                mean: 0.6, // Different
                std: 0.1,
            }),
        }];

        let mut diffs = Vec::new();
        let options = DiffOptions::new().with_stats();
        compare_tensors(&t1, &t2, &options, &mut diffs);

        assert!(diffs.iter().any(|d| d.field.contains("max")));
        assert!(diffs.iter().any(|d| d.field.contains("mean")));
    }

    #[test]
    fn test_compare_tensor_stats_one_missing() {
        use crate::format::rosetta::{TensorInfo, TensorStats};

        let t1 = TensorInfo {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10],
            size_bytes: 40,
            stats: Some(TensorStats {
                min: 0.0,
                max: 1.0,
                mean: 0.5,
                std: 0.1,
            }),
        };
        let t2 = TensorInfo {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10],
            size_bytes: 40,
            stats: None,
        };

        let mut diffs = Vec::new();
        compare_tensor_stats(&t1, &t2, &mut diffs);

        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].field.contains("stats"));
    }

    // ========================================================================
    // Integration Tests (File-based, minimal)
    // ========================================================================

    #[test]
    fn test_diff_models_file_not_found() {
        let result = diff_models(
            Path::new("/nonexistent/a.apr"),
            Path::new("/nonexistent/b.apr"),
            DiffOptions::default(),
        );
        assert!(result.is_err());
    }

    // ====================================================================
    // Coverage: compute_differences all branches
    // ====================================================================

    fn make_report(
        format: FormatType,
        size: usize,
        params: usize,
        arch: Option<&str>,
        quant: Option<&str>,
    ) -> InspectionReport {
        InspectionReport {
            format,
            file_size: size,
            metadata: std::collections::BTreeMap::new(),
            tensors: Vec::new(),
            total_params: params,
            quantization: quant.map(String::from),
            architecture: arch.map(String::from),
        }
    }

    #[test]
    fn test_compute_differences_identical() {
        let r = make_report(FormatType::Apr, 1000, 100, Some("llama"), None);
        let diffs = compute_differences(&r, &r, &DiffOptions::default());
        assert!(diffs.is_empty());
    }

    #[test]
    fn test_compute_differences_format_differs() {
        let r1 = make_report(FormatType::Apr, 1000, 100, None, None);
        let r2 = make_report(FormatType::Gguf, 1000, 100, None, None);
        let diffs = compute_differences(&r1, &r2, &DiffOptions::default());
        assert!(diffs.iter().any(|d| d.field == "format"));
    }

    #[test]
    fn test_compute_differences_size_differs() {
        let r1 = make_report(FormatType::Apr, 1000, 100, None, None);
        let r2 = make_report(FormatType::Apr, 2000, 100, None, None);
        let diffs = compute_differences(&r1, &r2, &DiffOptions::default());
        assert!(diffs.iter().any(|d| d.field == "file_size"));
    }

    #[test]
    fn test_compute_differences_params_differs() {
        let r1 = make_report(FormatType::Apr, 1000, 100, None, None);
        let r2 = make_report(FormatType::Apr, 1000, 200, None, None);
        let diffs = compute_differences(&r1, &r2, &DiffOptions::default());
        assert!(diffs.iter().any(|d| d.field == "total_params"));
    }

    #[test]
    fn test_compute_differences_architecture_differs() {
        let r1 = make_report(FormatType::Apr, 1000, 100, Some("llama"), None);
        let r2 = make_report(FormatType::Apr, 1000, 100, Some("qwen2"), None);
        let diffs = compute_differences(&r1, &r2, &DiffOptions::default());
        let arch_diff = diffs.iter().find(|d| d.field == "architecture").unwrap();
        assert!(arch_diff.value1.contains("llama"));
        assert!(arch_diff.value2.contains("qwen2"));
    }

    #[test]
    fn test_compute_differences_architecture_one_none() {
        let r1 = make_report(FormatType::Apr, 1000, 100, Some("llama"), None);
        let r2 = make_report(FormatType::Apr, 1000, 100, None, None);
        let diffs = compute_differences(&r1, &r2, &DiffOptions::default());
        let arch_diff = diffs.iter().find(|d| d.field == "architecture").unwrap();
        assert!(arch_diff.value2.contains("(none)"));
    }

    #[test]
    fn test_compute_differences_quantization_differs() {
        let r1 = make_report(FormatType::Apr, 1000, 100, None, Some("Q4_K"));
        let r2 = make_report(FormatType::Apr, 1000, 100, None, Some("Q8_0"));
        let diffs = compute_differences(&r1, &r2, &DiffOptions::default());
        assert!(diffs.iter().any(|d| d.field == "quantization"));
    }

    #[test]
    fn test_compute_differences_quantization_one_none() {
        let r1 = make_report(FormatType::Apr, 1000, 100, None, Some("Q4_K"));
        let r2 = make_report(FormatType::Apr, 1000, 100, None, None);
        let diffs = compute_differences(&r1, &r2, &DiffOptions::default());
        let q_diff = diffs.iter().find(|d| d.field == "quantization").unwrap();
        assert!(q_diff.value2.contains("(none)"));
    }

    #[test]
    fn test_compute_differences_multiple() {
        let r1 = make_report(FormatType::Apr, 1000, 100, Some("llama"), Some("F32"));
        let r2 = make_report(FormatType::Gguf, 2000, 200, Some("qwen2"), Some("Q4_K"));
        let diffs = compute_differences(&r1, &r2, &DiffOptions::default());
        assert!(diffs.len() >= 4);
    }

    #[test]
    fn test_compute_differences_no_tensors() {
        let r = make_report(FormatType::Apr, 1000, 100, None, None);
        let opts = DiffOptions::new().without_tensors();
        let diffs = compute_differences(&r, &r, &opts);
        assert!(diffs.is_empty());
    }

    // ====================================================================
    // Coverage: compare_tensor_stats all stat branches
    // ====================================================================

    #[test]
    fn test_compare_tensor_stats_min_differs() {
        use crate::format::rosetta::{TensorInfo as RTI, TensorStats as RTS};
        let mut diffs = Vec::new();
        let t1 = RTI {
            name: "w".to_string(),
            shape: vec![4],
            dtype: "F32".to_string(),
            size_bytes: 16,
            stats: Some(RTS {
                min: 0.0,
                max: 1.0,
                mean: 0.5,
                std: 0.1,
            }),
        };
        let t2 = RTI {
            name: "w".to_string(),
            shape: vec![4],
            dtype: "F32".to_string(),
            size_bytes: 16,
            stats: Some(RTS {
                min: 0.5,
                max: 1.0,
                mean: 0.5,
                std: 0.1,
            }),
        };
        compare_tensor_stats(&t1, &t2, &mut diffs);
        assert!(diffs.iter().any(|d| d.field.contains("min")));
        assert!(!diffs.iter().any(|d| d.field.contains("max")));
    }

    #[test]
    fn test_compare_tensor_stats_all_differ() {
        use crate::format::rosetta::{TensorInfo as RTI, TensorStats as RTS};
        let mut diffs = Vec::new();
        let t1 = RTI {
            name: "w".to_string(),
            shape: vec![4],
            dtype: "F32".to_string(),
            size_bytes: 16,
            stats: Some(RTS {
                min: 0.0,
                max: 1.0,
                mean: 0.5,
                std: 0.1,
            }),
        };
        let t2 = RTI {
            name: "w".to_string(),
            shape: vec![4],
            dtype: "F32".to_string(),
            size_bytes: 16,
            stats: Some(RTS {
                min: 1.0,
                max: 2.0,
                mean: 1.5,
                std: 0.5,
            }),
        };
        compare_tensor_stats(&t1, &t2, &mut diffs);
        assert_eq!(diffs.len(), 4); // min, max, mean, std
    }

    #[test]
    fn test_compare_tensor_stats_none_none() {
        use crate::format::rosetta::TensorInfo as RTI;
        let mut diffs = Vec::new();
        let t = RTI {
            name: "w".to_string(),
            shape: vec![4],
            dtype: "F32".to_string(),
            size_bytes: 16,
            stats: None,
        };
        compare_tensor_stats(&t, &t, &mut diffs);
        assert!(diffs.is_empty());
    }

    // ====================================================================
    // Coverage: DiffReport additional method tests
    // ====================================================================

    #[test]
    fn test_diff_report_by_category_filtering() {
        let report = DiffReport {
            path1: "a".to_string(),
            path2: "b".to_string(),
            format1: "APR".to_string(),
            format2: "APR".to_string(),
            differences: vec![
                DiffEntry {
                    field: "file_size".to_string(),
                    value1: "100".to_string(),
                    value2: "200".to_string(),
                    category: DiffCategory::Size,
                },
                DiffEntry {
                    field: "architecture".to_string(),
                    value1: "llama".to_string(),
                    value2: "qwen2".to_string(),
                    category: DiffCategory::Metadata,
                },
            ],
            inspection1: None,
            inspection2: None,
        };
        assert_eq!(report.differences_by_category(DiffCategory::Size).len(), 1);
        assert_eq!(
            report.differences_by_category(DiffCategory::Metadata).len(),
            1
        );
        assert_eq!(
            report.differences_by_category(DiffCategory::Format).len(),
            0
        );
    }

    // ====================================================================
    // Coverage: normalize_dtype exhaustive branch tests
    // ====================================================================

    #[test]
    fn test_normalize_dtype_numeric_codes() {
        // All GGUF numeric codes
        assert_eq!(normalize_dtype("0"), "F32");
        assert_eq!(normalize_dtype("1"), "F16");
        assert_eq!(normalize_dtype("2"), "Q4_0");
        assert_eq!(normalize_dtype("3"), "Q4_1");
        assert_eq!(normalize_dtype("6"), "Q5_0");
        assert_eq!(normalize_dtype("7"), "Q5_1");
        assert_eq!(normalize_dtype("8"), "Q8_0");
        assert_eq!(normalize_dtype("9"), "Q8_1");
        assert_eq!(normalize_dtype("10"), "Q2_K");
        assert_eq!(normalize_dtype("11"), "Q3_K");
        assert_eq!(normalize_dtype("12"), "Q4_K");
        assert_eq!(normalize_dtype("13"), "Q5_K");
        assert_eq!(normalize_dtype("14"), "Q6_K");
        assert_eq!(normalize_dtype("15"), "Q8_K");
        assert_eq!(normalize_dtype("16"), "IQ2_XXS");
        assert_eq!(normalize_dtype("17"), "IQ2_XS");
        assert_eq!(normalize_dtype("18"), "IQ3_XXS");
        assert_eq!(normalize_dtype("19"), "IQ1_S");
    }

    #[test]
    fn test_normalize_dtype_string_lowercase() {
        assert_eq!(normalize_dtype("f32"), "F32");
        assert_eq!(normalize_dtype("f16"), "F16");
        assert_eq!(normalize_dtype("q4_0"), "Q4_0");
        assert_eq!(normalize_dtype("q4_1"), "Q4_1");
        assert_eq!(normalize_dtype("q5_0"), "Q5_0");
        assert_eq!(normalize_dtype("q5_1"), "Q5_1");
        assert_eq!(normalize_dtype("q8_0"), "Q8_0");
        assert_eq!(normalize_dtype("q8_1"), "Q8_1");
        assert_eq!(normalize_dtype("q2_k"), "Q2_K");
        assert_eq!(normalize_dtype("q3_k"), "Q3_K");
        assert_eq!(normalize_dtype("q4_k"), "Q4_K");
        assert_eq!(normalize_dtype("q5_k"), "Q5_K");
        assert_eq!(normalize_dtype("q6_k"), "Q6_K");
        assert_eq!(normalize_dtype("q8_k"), "Q8_K");
        assert_eq!(normalize_dtype("iq2_xxs"), "IQ2_XXS");
        assert_eq!(normalize_dtype("iq2_xs"), "IQ2_XS");
        assert_eq!(normalize_dtype("iq3_xxs"), "IQ3_XXS");
        assert_eq!(normalize_dtype("iq1_s"), "IQ1_S");
    }

    #[test]
    fn test_normalize_dtype_string_uppercase() {
        assert_eq!(normalize_dtype("F32"), "F32");
        assert_eq!(normalize_dtype("F16"), "F16");
        assert_eq!(normalize_dtype("Q4_0"), "Q4_0");
        assert_eq!(normalize_dtype("Q4_1"), "Q4_1");
        assert_eq!(normalize_dtype("Q5_0"), "Q5_0");
        assert_eq!(normalize_dtype("Q5_1"), "Q5_1");
        assert_eq!(normalize_dtype("Q8_0"), "Q8_0");
        assert_eq!(normalize_dtype("Q8_1"), "Q8_1");
        assert_eq!(normalize_dtype("Q2_K"), "Q2_K");
        assert_eq!(normalize_dtype("Q3_K"), "Q3_K");
        assert_eq!(normalize_dtype("Q4_K"), "Q4_K");
        assert_eq!(normalize_dtype("Q5_K"), "Q5_K");
        assert_eq!(normalize_dtype("Q6_K"), "Q6_K");
        assert_eq!(normalize_dtype("Q8_K"), "Q8_K");
        assert_eq!(normalize_dtype("IQ2_XXS"), "IQ2_XXS");
        assert_eq!(normalize_dtype("IQ2_XS"), "IQ2_XS");
        assert_eq!(normalize_dtype("IQ3_XXS"), "IQ3_XXS");
        assert_eq!(normalize_dtype("IQ1_S"), "IQ1_S");
    }

    #[test]
    fn test_normalize_dtype_short_aliases() {
        // Short aliases without underscore (q2k, Q2K, etc.)
        assert_eq!(normalize_dtype("q2k"), "Q2_K");
        assert_eq!(normalize_dtype("Q2K"), "Q2_K");
        assert_eq!(normalize_dtype("q3k"), "Q3_K");
        assert_eq!(normalize_dtype("Q3K"), "Q3_K");
        assert_eq!(normalize_dtype("q4k"), "Q4_K");
        assert_eq!(normalize_dtype("Q4K"), "Q4_K");
        assert_eq!(normalize_dtype("q5k"), "Q5_K");
        assert_eq!(normalize_dtype("Q5K"), "Q5_K");
        assert_eq!(normalize_dtype("q6k"), "Q6_K");
        assert_eq!(normalize_dtype("Q6K"), "Q6_K");
        assert_eq!(normalize_dtype("q8k"), "Q8_K");
        assert_eq!(normalize_dtype("Q8K"), "Q8_K");
    }

    #[test]
    fn test_normalize_dtype_bf16() {
        assert_eq!(normalize_dtype("bf16"), "BF16");
        assert_eq!(normalize_dtype("BF16"), "BF16");
    }

    #[test]
    fn test_normalize_dtype_unknown_fallback() {
        // Catch-all: unknown types get uppercased
        assert_eq!(normalize_dtype("custom_type"), "CUSTOM_TYPE");
        assert_eq!(normalize_dtype("fp8"), "FP8");
        assert_eq!(normalize_dtype("int4"), "INT4");
        assert_eq!(normalize_dtype("26"), "26"); // Numeric code not in map
    }

    // ====================================================================
    // Coverage: is_compatible_quant exhaustive branch tests
    // ====================================================================

    #[test]
    fn test_is_compatible_quant_same_after_normalization() {
        // Same type after normalization returns true
        assert!(is_compatible_quant("f32", "F32"));
        assert!(is_compatible_quant("0", "F32"));
        assert!(is_compatible_quant("q4_k", "Q4_K"));
        assert!(is_compatible_quant("12", "Q4_K"));
        assert!(is_compatible_quant("q8_0", "Q8_0"));
        assert!(is_compatible_quant("8", "Q8_0"));
    }

    #[test]
    fn test_is_compatible_quant_q5_q6_compatible() {
        // Q5 <-> Q6 compatibility (common import path)
        assert!(is_compatible_quant("Q5_0", "Q6_K"));
        assert!(is_compatible_quant("Q6_K", "Q5_0"));
        assert!(is_compatible_quant("Q5_K", "Q6_K"));
        assert!(is_compatible_quant("Q6_K", "Q5_K"));
        assert!(is_compatible_quant("Q5_1", "Q6_K"));
        assert!(is_compatible_quant("Q6_K", "Q5_1"));
    }

    #[test]
    fn test_is_compatible_quant_q4_variants() {
        // Q4 <-> Q4 variants compatible
        assert!(is_compatible_quant("Q4_0", "Q4_K"));
        assert!(is_compatible_quant("Q4_K", "Q4_0"));
        assert!(is_compatible_quant("Q4_1", "Q4_K"));
        assert!(is_compatible_quant("Q4_K", "Q4_1"));
        assert!(is_compatible_quant("Q4_0", "Q4_1"));
    }

    #[test]
    fn test_is_compatible_quant_q8_q6_compatible() {
        // Q8 <-> Q6 compatibility (downgrade)
        assert!(is_compatible_quant("Q8_0", "Q6_K"));
        assert!(is_compatible_quant("Q6_K", "Q8_0"));
        assert!(is_compatible_quant("Q8_K", "Q6_K"));
        assert!(is_compatible_quant("Q6_K", "Q8_K"));
    }

    #[test]
    fn test_is_compatible_quant_incompatible_pairs() {
        // Truly incompatible pairs
        assert!(!is_compatible_quant("F32", "Q4_K"));
        assert!(!is_compatible_quant("Q4_K", "F32"));
        assert!(!is_compatible_quant("Q2_K", "Q8_0"));
        assert!(!is_compatible_quant("Q8_0", "Q2_K"));
        assert!(!is_compatible_quant("F16", "Q4_0"));
        assert!(!is_compatible_quant("BF16", "Q6_K"));
        assert!(!is_compatible_quant("Q3_K", "Q8_0"));
        assert!(!is_compatible_quant("F32", "F16"));
        assert!(!is_compatible_quant("IQ2_XXS", "Q4_K"));
    }

    #[test]
    fn test_is_compatible_quant_with_numeric_codes() {
        // Using numeric GGUF codes should also work through normalization
        assert!(is_compatible_quant("12", "Q4_0")); // 12 = Q4_K, compatible with Q4_0
        assert!(is_compatible_quant("6", "14")); // Q5_0 <-> Q6_K compatible
        assert!(!is_compatible_quant("0", "12")); // F32 <-> Q4_K incompatible
    }

    // ====================================================================
    // Coverage: map_gguf_to_apr_name all branches
    // ====================================================================

    #[test]
    fn test_map_gguf_to_apr_name_attn_q_weight() {
        let (name, mapped) = map_gguf_to_apr_name("blk.0.attn_q.weight");
        assert_eq!(name, "model.layers.0.self_attn.q_proj.weight");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_attn_q_bias() {
        let (name, mapped) = map_gguf_to_apr_name("blk.3.attn_q.bias");
        assert_eq!(name, "model.layers.3.self_attn.q_proj.bias");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_attn_k_weight() {
        let (name, mapped) = map_gguf_to_apr_name("blk.1.attn_k.weight");
        assert_eq!(name, "model.layers.1.self_attn.k_proj.weight");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_attn_k_bias() {
        let (name, mapped) = map_gguf_to_apr_name("blk.7.attn_k.bias");
        assert_eq!(name, "model.layers.7.self_attn.k_proj.bias");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_attn_v_weight() {
        let (name, mapped) = map_gguf_to_apr_name("blk.2.attn_v.weight");
        assert_eq!(name, "model.layers.2.self_attn.v_proj.weight");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_attn_v_bias() {
        let (name, mapped) = map_gguf_to_apr_name("blk.5.attn_v.bias");
        assert_eq!(name, "model.layers.5.self_attn.v_proj.bias");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_attn_output_weight() {
        let (name, mapped) = map_gguf_to_apr_name("blk.4.attn_output.weight");
        assert_eq!(name, "model.layers.4.self_attn.o_proj.weight");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_attn_output_bias() {
        let (name, mapped) = map_gguf_to_apr_name("blk.0.attn_output.bias");
        assert_eq!(name, "model.layers.0.self_attn.o_proj.bias");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_attn_norm() {
        let (name, mapped) = map_gguf_to_apr_name("blk.6.attn_norm.weight");
        assert_eq!(name, "model.layers.6.input_layernorm.weight");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_ffn_gate() {
        let (name, mapped) = map_gguf_to_apr_name("blk.0.ffn_gate.weight");
        assert_eq!(name, "model.layers.0.mlp.gate_proj.weight");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_ffn_up() {
        let (name, mapped) = map_gguf_to_apr_name("blk.10.ffn_up.weight");
        assert_eq!(name, "model.layers.10.mlp.up_proj.weight");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_ffn_down() {
        let (name, mapped) = map_gguf_to_apr_name("blk.31.ffn_down.weight");
        assert_eq!(name, "model.layers.31.mlp.down_proj.weight");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_ffn_norm() {
        let (name, mapped) = map_gguf_to_apr_name("blk.0.ffn_norm.weight");
        assert_eq!(name, "model.layers.0.post_attention_layernorm.weight");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_unknown_layer_suffix() {
        // Unknown suffix within blk.N.* returns unchanged
        let (name, mapped) = map_gguf_to_apr_name("blk.0.unknown_suffix.weight");
        assert_eq!(name, "blk.0.unknown_suffix.weight");
        assert!(!mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_token_embd() {
        let (name, mapped) = map_gguf_to_apr_name("token_embd.weight");
        assert_eq!(name, "model.embed_tokens.weight");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_output_weight() {
        let (name, mapped) = map_gguf_to_apr_name("output.weight");
        assert_eq!(name, "lm_head.weight");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_output_norm() {
        let (name, mapped) = map_gguf_to_apr_name("output_norm.weight");
        assert_eq!(name, "model.norm.weight");
        assert!(mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_unknown_non_layer() {
        // Unknown non-layer tensor returns unchanged
        let (name, mapped) = map_gguf_to_apr_name("some_other_tensor");
        assert_eq!(name, "some_other_tensor");
        assert!(!mapped);
    }

    #[test]
    fn test_map_gguf_to_apr_name_already_apr_style() {
        // APR-style names pass through unchanged
        let (name, mapped) = map_gguf_to_apr_name("model.layers.0.self_attn.q_proj.weight");
        assert_eq!(name, "model.layers.0.self_attn.q_proj.weight");
        assert!(!mapped);
    }

    // ====================================================================
    // Coverage: build_cross_format_map tests
    // ====================================================================

    #[test]
    fn test_build_cross_format_map_gguf_names() {
        use crate::format::rosetta::TensorInfo;

        let tensors = vec![
            TensorInfo {
                name: "blk.0.attn_q.weight".to_string(),
                dtype: "Q4_K".to_string(),
                shape: vec![4096, 4096],
                size_bytes: 1000,
                stats: None,
            },
            TensorInfo {
                name: "token_embd.weight".to_string(),
                dtype: "F16".to_string(),
                shape: vec![32000, 4096],
                size_bytes: 2000,
                stats: None,
            },
        ];

        let map = build_cross_format_map(&tensors);

        // Original GGUF names should be present
        assert!(map.contains_key("blk.0.attn_q.weight"));
        assert!(map.contains_key("token_embd.weight"));

        // Mapped APR names should also be present
        assert!(map.contains_key("model.layers.0.self_attn.q_proj.weight"));
        assert!(map.contains_key("model.embed_tokens.weight"));
    }

    #[test]
    fn test_build_cross_format_map_hf_names() {
        use crate::format::rosetta::TensorInfo;

        // HF/APR names that won't be mapped (no blk. prefix, not in non-layer map)
        let tensors = vec![TensorInfo {
            name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            dtype: "F16".to_string(),
            shape: vec![4096, 4096],
            size_bytes: 1000,
            stats: None,
        }];

        let map = build_cross_format_map(&tensors);

        // Original name present
        assert!(map.contains_key("model.layers.0.self_attn.q_proj.weight"));
        // Only one entry (no mapping was applied)
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_build_cross_format_map_empty() {
        let tensors: Vec<crate::format::rosetta::TensorInfo> = vec![];
        let map = build_cross_format_map(&tensors);
        assert!(map.is_empty());
    }

    #[test]
    fn test_build_cross_format_map_unknown_suffix() {
        use crate::format::rosetta::TensorInfo;

        // Unknown suffix within blk.N.* -- no mapping added
        let tensors = vec![TensorInfo {
            name: "blk.0.custom_thing.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![100],
            size_bytes: 400,
            stats: None,
        }];

        let map = build_cross_format_map(&tensors);

        // Only the original name is present (no mapping)
        assert!(map.contains_key("blk.0.custom_thing.weight"));
        assert_eq!(map.len(), 1);
    }

    // ====================================================================
    // Coverage: diff_inspections public API tests
    // ====================================================================

    #[test]
    fn test_diff_inspections_identical() {
        let r = make_report(FormatType::Apr, 1000, 100, Some("llama"), None);
        let report = diff_inspections(&r, &r, "model_a.apr", "model_b.apr", DiffOptions::default());
        assert!(report.is_identical());
        assert_eq!(report.path1, "model_a.apr");
        assert_eq!(report.path2, "model_b.apr");
        assert_eq!(report.format1, "APR");
        assert_eq!(report.format2, "APR");
        assert!(report.inspection1.is_none());
        assert!(report.inspection2.is_none());
    }

    #[test]
    fn test_diff_inspections_different_formats() {
        let r1 = make_report(FormatType::Apr, 1000, 100, Some("llama"), Some("Q4_K"));
        let r2 = make_report(FormatType::Gguf, 2000, 200, Some("qwen2"), Some("Q8_0"));
        let report = diff_inspections(&r1, &r2, "a.apr", "b.gguf", DiffOptions::default());
        assert!(!report.is_identical());
        assert!(!report.same_format());
        assert!(report.differences.iter().any(|d| d.field == "format"));
        assert!(report.differences.iter().any(|d| d.field == "file_size"));
        assert!(report.differences.iter().any(|d| d.field == "total_params"));
        assert!(report.differences.iter().any(|d| d.field == "architecture"));
        assert!(report.differences.iter().any(|d| d.field == "quantization"));
    }

    #[test]
    fn test_diff_inspections_with_tensors() {
        use crate::format::rosetta::TensorInfo;

        let mut r1 = make_report(FormatType::Apr, 1000, 100, None, None);
        r1.tensors = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10, 20],
            size_bytes: 800,
            stats: None,
        }];

        let mut r2 = make_report(FormatType::Apr, 1000, 100, None, None);
        r2.tensors = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![20, 10], // Transposed -- compatible
            size_bytes: 800,
            stats: None,
        }];

        let report = diff_inspections(&r1, &r2, "a.apr", "b.apr", DiffOptions::default());
        // Transposed shapes are compatible, so no shape diff
        assert!(report.is_identical());
    }

    #[test]
    fn test_diff_inspections_no_metadata() {
        use std::collections::BTreeMap;

        let mut r1 = make_report(FormatType::Apr, 1000, 100, None, None);
        r1.metadata = {
            let mut m = BTreeMap::new();
            m.insert("key".to_string(), "val1".to_string());
            m
        };
        let mut r2 = make_report(FormatType::Apr, 1000, 100, None, None);
        r2.metadata = {
            let mut m = BTreeMap::new();
            m.insert("key".to_string(), "val2".to_string());
            m
        };

        // With metadata comparison disabled, no diff
        let report = diff_inspections(
            &r1,
            &r2,
            "a.apr",
            "b.apr",
            DiffOptions::new().without_metadata(),
        );
        assert!(report.is_identical());

        // With metadata comparison enabled, diff present
        let report2 = diff_inspections(
            &r1,
            &r2,
            "a.apr",
            "b.apr",
            DiffOptions::new().with_metadata(),
        );
        assert!(!report2.is_identical());
    }

    // ====================================================================
    // Coverage: compare_tensor_stats None->Some branch
    // ====================================================================

    #[test]
    fn test_compare_tensor_stats_none_some() {
        use crate::format::rosetta::{TensorInfo as RTI, TensorStats as RTS};
        let mut diffs = Vec::new();
        let t1 = RTI {
            name: "w".to_string(),
            shape: vec![4],
            dtype: "F32".to_string(),
            size_bytes: 16,
            stats: None,
        };
        let t2 = RTI {
            name: "w".to_string(),
            shape: vec![4],
            dtype: "F32".to_string(),
            size_bytes: 16,
            stats: Some(RTS {
                min: 0.0,
                max: 1.0,
                mean: 0.5,
                std: 0.1,
            }),
        };
        compare_tensor_stats(&t1, &t2, &mut diffs);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].field.contains("stats"));
        assert_eq!(diffs[0].value1, "(none)");
        assert_eq!(diffs[0].value2, "present");
    }

    // ====================================================================
    // Coverage: compare_tensor_stats std differs alone
    // ====================================================================

    #[test]
    fn test_compare_tensor_stats_std_differs_only() {
        use crate::format::rosetta::{TensorInfo as RTI, TensorStats as RTS};
        let mut diffs = Vec::new();
        let t1 = RTI {
            name: "w".to_string(),
            shape: vec![4],
            dtype: "F32".to_string(),
            size_bytes: 16,
            stats: Some(RTS {
                min: 0.0,
                max: 1.0,
                mean: 0.5,
                std: 0.1,
            }),
        };
        let t2 = RTI {
            name: "w".to_string(),
            shape: vec![4],
            dtype: "F32".to_string(),
            size_bytes: 16,
            stats: Some(RTS {
                min: 0.0,
                max: 1.0,
                mean: 0.5,
                std: 0.9,
            }),
        };
        compare_tensor_stats(&t1, &t2, &mut diffs);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].field.contains("std"));
    }

    // ====================================================================
    // Coverage: cross-format tensor comparison with GGUF name mapping
    // ====================================================================

    #[test]
    fn test_compare_tensors_cross_format_gguf_to_apr() {
        use crate::format::rosetta::TensorInfo;

        // Model 1 uses GGUF naming
        let t1 = vec![
            TensorInfo {
                name: "blk.0.attn_q.weight".to_string(),
                dtype: "Q4_K".to_string(),
                shape: vec![4096, 4096],
                size_bytes: 1000,
                stats: None,
            },
            TensorInfo {
                name: "token_embd.weight".to_string(),
                dtype: "F16".to_string(),
                shape: vec![32000, 4096],
                size_bytes: 2000,
                stats: None,
            },
        ];

        // Model 2 uses APR/HF naming with same shapes
        let t2 = vec![
            TensorInfo {
                name: "model.layers.0.self_attn.q_proj.weight".to_string(),
                dtype: "Q4_K".to_string(),
                shape: vec![4096, 4096],
                size_bytes: 1000,
                stats: None,
            },
            TensorInfo {
                name: "model.embed_tokens.weight".to_string(),
                dtype: "F16".to_string(),
                shape: vec![32000, 4096],
                size_bytes: 2000,
                stats: None,
            },
        ];

        let mut diffs = Vec::new();
        let options = DiffOptions::default();
        compare_tensors(&t1, &t2, &options, &mut diffs);

        // Cross-format mapping should make these match
        assert!(
            diffs.is_empty(),
            "Expected no diffs for cross-format name mapping, got: {diffs:?}"
        );
    }

    #[test]
    fn test_compare_tensors_transposed_shapes_compatible() {
        use crate::format::rosetta::TensorInfo;

        let t1 = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![100, 200],
            size_bytes: 800,
            stats: None,
        }];
        let t2 = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![200, 100], // Transposed
            size_bytes: 800,
            stats: None,
        }];

        let mut diffs = Vec::new();
        compare_tensors(&t1, &t2, &DiffOptions::default(), &mut diffs);

        // Transposed 2D shapes are considered compatible
        assert!(diffs.is_empty());
    }

    #[test]
    fn test_compare_tensors_compatible_quant_no_diff() {
        use crate::format::rosetta::TensorInfo;

        let t1 = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "Q5_0".to_string(),
            shape: vec![10, 20],
            size_bytes: 400,
            stats: None,
        }];
        let t2 = vec![TensorInfo {
            name: "weight".to_string(),
            dtype: "Q6_K".to_string(),
            shape: vec![10, 20],
            size_bytes: 400,
            stats: None,
        }];

        let mut diffs = Vec::new();
        compare_tensors(&t1, &t2, &DiffOptions::default(), &mut diffs);

        // Q5_0 and Q6_K are considered compatible
        assert!(
            diffs.is_empty(),
            "Expected no dtype diff for compatible quants Q5_0 and Q6_K, got: {diffs:?}"
        );
    }

    #[test]
    fn test_compare_tensors_only_in_model1() {
        use crate::format::rosetta::TensorInfo;

        let t1 = vec![TensorInfo {
            name: "unique_tensor".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10],
            size_bytes: 40,
            stats: None,
        }];
        let t2: Vec<TensorInfo> = vec![];

        let mut diffs = Vec::new();
        compare_tensors(&t1, &t2, &DiffOptions::default(), &mut diffs);

        assert!(diffs
            .iter()
            .any(|d| d.field.contains("unique_tensor") && d.value2 == "(missing)"));
    }

    #[test]
    fn test_compare_tensors_only_in_model2() {
        use crate::format::rosetta::TensorInfo;

        let t1: Vec<TensorInfo> = vec![];
        let t2 = vec![TensorInfo {
            name: "extra_tensor".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10],
            size_bytes: 40,
            stats: None,
        }];

        let mut diffs = Vec::new();
        compare_tensors(&t1, &t2, &DiffOptions::default(), &mut diffs);

        assert!(diffs
            .iter()
            .any(|d| d.field.contains("extra_tensor") && d.value1 == "(missing)"));
    }
}
