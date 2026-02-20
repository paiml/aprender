//! Model Family YAML Contract Loader (PMAT-242)
//!
//! Parses model family YAML files at runtime without external dependencies.
//! This is the runtime fallback; build.rs codegen (PMAT-250) is the preferred path.
//!
//! # Contract
//!
//! See `contracts/model-families/*.yaml` and
//! `docs/specifications/compiler-enforced-model-types-model-oracle.md` §4

use std::collections::HashMap;
use std::path::Path;

use crate::error::{AprenderError, Result};
use crate::format::model_family::{
    Activation, AttentionType, CertificationConfig, ChatTemplateConfig, DynModelFamily,
    FamilyRegistry, GgufFusionRule, GgufTensorTemplate, MlpType, ModelConstraints,
    ModelFamilyConfig, ModelSizeConfig, NormType, PositionalEncoding, ShapeTemplate,
    TensorTemplate,
};

// ============================================================================
// Minimal YAML Parser
// ============================================================================

/// A simple YAML value representation
#[derive(Debug, Clone)]
enum YamlValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Null,
    Sequence(Vec<YamlValue>),
    Mapping(Vec<(String, YamlValue)>),
}

impl YamlValue {
    fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int(n) => Some(*n),
            Self::Float(f) => Some(*f as i64),
            _ => None,
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(f) => Some(*f),
            Self::Int(n) => Some(*n as f64),
            _ => None,
        }
    }

    fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    fn as_usize(&self) -> Option<usize> {
        self.as_i64().and_then(|n| usize::try_from(n).ok())
    }

    fn as_sequence(&self) -> Option<&[YamlValue]> {
        match self {
            Self::Sequence(s) => Some(s),
            _ => None,
        }
    }

    #[cfg(test)]
    fn as_mapping(&self) -> Option<&[(String, YamlValue)]> {
        match self {
            Self::Mapping(m) => Some(m),
            _ => None,
        }
    }

    fn get(&self, key: &str) -> Option<&YamlValue> {
        match self {
            Self::Mapping(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key)?.as_str()
    }

    fn get_usize(&self, key: &str) -> Option<usize> {
        self.get(key)?.as_usize()
    }

    fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key)?.as_f64()
    }

    fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key)?.as_bool()
    }
}

/// Parse a YAML string into a `YamlValue`.
/// Handles the subset of YAML used in model family contracts:
/// - Top-level mapping
/// - Nested mappings (indentation-based)
/// - Sequences (- item)
/// - Scalars: strings, ints, floats, bools, null
fn parse_yaml(input: &str) -> Result<YamlValue> {
    let lines: Vec<&str> = input.lines().collect();
    let (val, _) = parse_mapping(&lines, 0, 0)?;
    Ok(val)
}

/// Skip blank lines and comments, returning the index of the next content line.
fn skip_to_content(lines: &[&str], start: usize) -> usize {
    let mut i = start;
    while i < lines.len() {
        let trimmed = lines[i].trim();
        if !trimmed.is_empty() && !trimmed.starts_with('#') {
            break;
        }
        i += 1;
    }
    i
}

/// Parse a nested YAML value (sequence, mapping, or null) after an empty colon.
fn parse_nested_value(
    lines: &[&str],
    start: usize,
    parent_indent: usize,
) -> Result<(YamlValue, usize)> {
    let next_i = skip_to_content(lines, start);

    if next_i >= lines.len() {
        return Ok((YamlValue::Null, next_i));
    }

    let next_line = lines[next_i];
    let next_indent = next_line.len() - next_line.trim_start().len();
    let next_trimmed = next_line.trim();

    if next_trimmed.starts_with("- ") {
        parse_sequence(lines, next_i, next_indent)
    } else if next_indent > parent_indent {
        parse_mapping(lines, next_i, next_indent)
    } else {
        Ok((YamlValue::Null, next_i))
    }
}

/// Classify a YAML line's relationship to the current mapping scope.
enum MappingLineAction {
    /// Skip blank or comment lines.
    Skip,
    /// Line is dedented and entries exist: the mapping is complete.
    EndMapping,
    /// Line is dedented but no entries yet: skip it.
    SkipDedented,
    /// Line is over-indented and entries exist: the mapping is complete.
    EndOverindent,
    /// Line is at the correct indent level: parse it as a key-value entry.
    ParseEntry,
}

fn classify_mapping_line(
    trimmed: &str,
    line_indent: usize,
    indent: usize,
    has_entries: bool,
) -> MappingLineAction {
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return MappingLineAction::Skip;
    }
    if line_indent < indent && has_entries {
        return MappingLineAction::EndMapping;
    }
    if line_indent < indent {
        return MappingLineAction::SkipDedented;
    }
    if line_indent > indent && has_entries {
        return MappingLineAction::EndOverindent;
    }
    MappingLineAction::ParseEntry
}

/// Parse a single key-value entry from a YAML mapping line.
/// Returns the parsed entry and the next line index.
fn parse_mapping_entry(
    lines: &[&str],
    trimmed: &str,
    current_idx: usize,
    indent: usize,
) -> Result<Option<((String, YamlValue), usize)>> {
    let Some(colon_pos) = trimmed.find(':') else {
        return Ok(None);
    };
    let key = trimmed[..colon_pos].trim().to_string();
    let after_colon = trimmed[colon_pos + 1..].trim();

    if after_colon.is_empty() {
        let (value, new_i) = parse_nested_value(lines, current_idx + 1, indent)?;
        Ok(Some(((key, value), new_i)))
    } else {
        let value = parse_scalar(after_colon);
        Ok(Some(((key, value), current_idx + 1)))
    }
}

fn parse_mapping(lines: &[&str], start: usize, indent: usize) -> Result<(YamlValue, usize)> {
    let mut entries = Vec::new();
    let mut i = start;

    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();
        let line_indent = line.len() - line.trim_start().len();

        match classify_mapping_line(trimmed, line_indent, indent, !entries.is_empty()) {
            MappingLineAction::Skip | MappingLineAction::SkipDedented => {
                i += 1;
            }
            MappingLineAction::EndMapping | MappingLineAction::EndOverindent => {
                return Ok((YamlValue::Mapping(entries), i));
            }
            MappingLineAction::ParseEntry => {
                if let Some((entry, new_i)) = parse_mapping_entry(lines, trimmed, i, indent)? {
                    entries.push(entry);
                    i = new_i;
                } else {
                    i += 1;
                }
            }
        }
    }

    Ok((YamlValue::Mapping(entries), i))
}

fn parse_sequence(lines: &[&str], start: usize, indent: usize) -> Result<(YamlValue, usize)> {
    let mut items = Vec::new();
    let mut i = start;

    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();

        if trimmed.is_empty() || trimmed.starts_with('#') {
            i += 1;
            continue;
        }

        let line_indent = line.len() - line.trim_start().len();
        if line_indent < indent {
            break;
        }

        if let Some(stripped) = trimmed.strip_prefix("- ") {
            let item_str = stripped.trim();
            items.push(parse_scalar(item_str));
            i += 1;
        } else {
            break;
        }
    }

    Ok((YamlValue::Sequence(items), i))
}

fn parse_scalar(s: &str) -> YamlValue {
    let s = s.trim();

    // Handle quoted strings
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        return YamlValue::String(s[1..s.len() - 1].to_string());
    }

    // Handle null
    if s == "null" || s == "~" {
        return YamlValue::Null;
    }

    // Handle booleans
    match s.to_lowercase().as_str() {
        "true" | "yes" => return YamlValue::Bool(true),
        "false" | "no" => return YamlValue::Bool(false),
        _ => {}
    }

    // Handle integers
    if let Ok(n) = s.parse::<i64>() {
        return YamlValue::Int(n);
    }

    // Handle floats
    if let Ok(f) = s.parse::<f64>() {
        return YamlValue::Float(f);
    }

    // Default to string
    YamlValue::String(s.to_string())
}

// ============================================================================
// YAML to ModelFamilyConfig conversion
// ============================================================================

/// Load a `ModelFamilyConfig` from a YAML file path.
///
/// # Errors
///
/// Returns `AprenderError::FormatError` if the file cannot be read or
/// contains invalid YAML for a model family contract.
pub fn load_family_yaml(path: &Path) -> Result<ModelFamilyConfig> {
    let content = std::fs::read_to_string(path).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to read YAML file {}: {e}", path.display()),
    })?;
    parse_family_yaml(&content, path)
}

/// Parse a YAML string into a `ModelFamilyConfig`.
///
/// # Errors
///
/// Returns `AprenderError::FormatError` if the YAML content is missing
/// required fields or contains invalid values.
pub fn parse_family_yaml(content: &str, source: &Path) -> Result<ModelFamilyConfig> {
    let yaml = parse_yaml(content)?;
    yaml_to_config(&yaml, source)
}

fn yaml_to_config(yaml: &YamlValue, source: &Path) -> Result<ModelFamilyConfig> {
    let err = |msg: &str| -> AprenderError {
        AprenderError::FormatError {
            message: format!("{}: {msg}", source.display()),
        }
    };

    let family = yaml
        .get_str("family")
        .ok_or_else(|| err("missing required field: family"))?
        .to_string();
    let display_name = yaml
        .get_str("display_name")
        .ok_or_else(|| err("missing required field: display_name"))?
        .to_string();
    let vendor = yaml
        .get_str("vendor")
        .ok_or_else(|| err("missing required field: vendor"))?
        .to_string();
    let hf_pattern = yaml
        .get_str("hf_pattern")
        .ok_or_else(|| err("missing required field: hf_pattern"))?
        .to_string();

    // Parse architectures sequence
    let architectures = yaml
        .get("architectures")
        .and_then(YamlValue::as_sequence)
        .ok_or_else(|| err("missing required field: architectures"))?
        .iter()
        .filter_map(YamlValue::as_str)
        .map(String::from)
        .collect();

    // Parse size_variants
    let size_variants_yaml = yaml
        .get("size_variants")
        .ok_or_else(|| err("missing required field: size_variants"))?;

    let mut size_variants = HashMap::new();
    if let YamlValue::Mapping(entries) = size_variants_yaml {
        for (name, variant) in entries {
            let config = parse_size_config(variant, &family, name)?;
            size_variants.insert(name.clone(), config);
        }
    } else {
        return Err(err("size_variants must be a mapping"));
    }

    // Parse constraints
    let constraints_yaml = yaml
        .get("constraints")
        .ok_or_else(|| err("missing required field: constraints"))?;
    let constraints = parse_constraints(constraints_yaml)?;

    // Parse tensor_template
    let template_yaml = yaml
        .get("tensor_template")
        .ok_or_else(|| err("missing required field: tensor_template"))?;
    let tensor_template = parse_tensor_template(template_yaml)?;

    // GH-277: Parse gguf_tensor_template (optional)
    let gguf_tensor_template = yaml
        .get("gguf_tensor_template")
        .map(parse_gguf_tensor_template)
        .unwrap_or_default();

    // Parse shape_template
    let shape_yaml = yaml.get("shape_template");
    let shape_template = if let Some(sy) = shape_yaml {
        parse_shape_template(sy)
    } else {
        ShapeTemplate {
            shapes: HashMap::new(),
        }
    };

    // Parse quantizations
    let quantizations = yaml
        .get("quantizations")
        .and_then(YamlValue::as_sequence)
        .map(|seq| {
            seq.iter()
                .filter_map(YamlValue::as_str)
                .map(String::from)
                .collect()
        })
        .unwrap_or_default();

    // Parse chat_template (optional)
    let chat_template = yaml
        .get("chat_template")
        .and_then(|ct| parse_chat_template(ct).ok());

    // Parse certification (optional)
    let certification = yaml
        .get("certification")
        .and_then(|c| parse_certification(c).ok());

    Ok(ModelFamilyConfig {
        family,
        display_name,
        vendor,
        architectures,
        hf_pattern,
        size_variants,
        constraints,
        tensor_template,
        gguf_tensor_template,
        shape_template,
        quantizations,
        chat_template,
        certification,
    })
}

include!("parsing.rs");
include!("model_family_loader_part_03.rs");
