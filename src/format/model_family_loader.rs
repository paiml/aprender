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
    FamilyRegistry, MlpType, ModelConstraints, ModelFamilyConfig, ModelSizeConfig, NormType,
    PositionalEncoding, ShapeTemplate, TensorTemplate,
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

fn parse_mapping(lines: &[&str], start: usize, indent: usize) -> Result<(YamlValue, usize)> {
    let mut entries = Vec::new();
    let mut i = start;

    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();

        // Skip empty lines and comments
        if trimmed.is_empty() || trimmed.starts_with('#') {
            i += 1;
            continue;
        }

        // Calculate indentation
        let line_indent = line.len() - line.trim_start().len();

        // If indentation decreased, we're done with this mapping
        if line_indent < indent && !entries.is_empty() {
            return Ok((YamlValue::Mapping(entries), i));
        }

        // Skip if indentation is less than expected (shouldn't happen at top level)
        if line_indent < indent {
            i += 1;
            continue;
        }

        // If indentation increased beyond expected, we're done
        if line_indent > indent && !entries.is_empty() {
            return Ok((YamlValue::Mapping(entries), i));
        }

        // Parse key: value
        if let Some(colon_pos) = trimmed.find(':') {
            let key = trimmed[..colon_pos].trim().to_string();
            let after_colon = trimmed[colon_pos + 1..].trim();

            if after_colon.is_empty() {
                // Value is on next lines (nested mapping or sequence)
                i += 1;

                // Peek at next non-empty line to determine type
                let mut next_i = i;
                while next_i < lines.len() {
                    let next_trimmed = lines[next_i].trim();
                    if !next_trimmed.is_empty() && !next_trimmed.starts_with('#') {
                        break;
                    }
                    next_i += 1;
                }

                if next_i < lines.len() {
                    let next_line = lines[next_i];
                    let next_indent = next_line.len() - next_line.trim_start().len();
                    let next_trimmed = next_line.trim();

                    if next_trimmed.starts_with("- ") {
                        // Sequence
                        let (seq, new_i) = parse_sequence(lines, next_i, next_indent)?;
                        entries.push((key, seq));
                        i = new_i;
                    } else if next_indent > indent {
                        // Nested mapping
                        let (mapping, new_i) = parse_mapping(lines, next_i, next_indent)?;
                        entries.push((key, mapping));
                        i = new_i;
                    } else {
                        entries.push((key, YamlValue::Null));
                    }
                } else {
                    entries.push((key, YamlValue::Null));
                }
            } else {
                // Inline value
                let value = parse_scalar(after_colon);
                entries.push((key, value));
                i += 1;
            }
        } else {
            i += 1;
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
        shape_template,
        quantizations,
        chat_template,
        certification,
    })
}

fn parse_size_config(yaml: &YamlValue, family: &str, size: &str) -> Result<ModelSizeConfig> {
    let err = |msg: &str| -> AprenderError {
        AprenderError::FormatError {
            message: format!("Family {family}, size {size}: {msg}"),
        }
    };

    Ok(ModelSizeConfig {
        parameters: yaml
            .get_str("parameters")
            .ok_or_else(|| err("missing parameters"))?
            .to_string(),
        hidden_dim: yaml
            .get_usize("hidden_dim")
            .or_else(|| yaml.get_usize("d_model"))
            .ok_or_else(|| err("missing hidden_dim"))?,
        num_layers: yaml
            .get_usize("num_layers")
            .or_else(|| yaml.get_usize("encoder_layers"))
            .ok_or_else(|| err("missing num_layers"))?,
        num_heads: yaml
            .get_usize("num_heads")
            .or_else(|| yaml.get_usize("encoder_attention_heads"))
            .ok_or_else(|| err("missing num_heads"))?,
        num_kv_heads: yaml
            .get_usize("num_kv_heads")
            .or_else(|| yaml.get_usize("num_heads"))
            .or_else(|| yaml.get_usize("encoder_attention_heads"))
            .ok_or_else(|| err("missing num_kv_heads"))?,
        intermediate_dim: yaml
            .get_usize("intermediate_dim")
            .or_else(|| yaml.get_usize("encoder_ffn_dim"))
            .ok_or_else(|| err("missing intermediate_dim"))?,
        vocab_size: yaml
            .get_usize("vocab_size")
            .ok_or_else(|| err("missing vocab_size"))?,
        max_position_embeddings: yaml
            .get_usize("max_position_embeddings")
            .or_else(|| yaml.get_usize("max_source_positions"))
            .unwrap_or(0),
        head_dim: yaml.get_usize("head_dim").unwrap_or_else(|| {
            let hidden = yaml
                .get_usize("hidden_dim")
                .or_else(|| yaml.get_usize("d_model"))
                .unwrap_or(0);
            let heads = yaml
                .get_usize("num_heads")
                .or_else(|| yaml.get_usize("encoder_attention_heads"))
                .unwrap_or(0);
            if heads > 0 {
                hidden / heads
            } else {
                0
            }
        }),
        rope_theta: yaml.get_f64("rope_theta").unwrap_or(0.0),
        norm_eps: yaml
            .get_f64("rms_norm_eps")
            .or_else(|| yaml.get_f64("layer_norm_eps"))
            .or_else(|| yaml.get_f64("norm_eps"))
            .unwrap_or(1e-5),
    })
}

fn parse_constraints(yaml: &YamlValue) -> Result<ModelConstraints> {
    Ok(ModelConstraints {
        attention_type: AttentionType::from_str_contract(
            yaml.get_str("attention_type").unwrap_or("mha"),
        )?,
        activation: Activation::from_str_contract(yaml.get_str("activation").unwrap_or("gelu"))?,
        norm_type: NormType::from_str_contract(yaml.get_str("norm_type").unwrap_or("layernorm"))?,
        has_bias: yaml.get_bool("has_bias").unwrap_or(false),
        tied_embeddings: yaml.get_bool("tied_embeddings").unwrap_or(false),
        positional_encoding: PositionalEncoding::from_str_contract(
            yaml.get_str("positional_encoding").unwrap_or("absolute"),
        )?,
        mlp_type: MlpType::from_str_contract(yaml.get_str("mlp_type").unwrap_or("gelu_mlp"))?,
    })
}

fn parse_tensor_template(yaml: &YamlValue) -> Result<TensorTemplate> {
    let embedding = yaml.get_str("embedding").unwrap_or("").to_string();

    let lm_head = yaml.get("lm_head").and_then(|v| match v {
        YamlValue::Null => None,
        YamlValue::String(s) => Some(s.clone()),
        _ => None,
    });

    let final_norm = yaml.get("final_norm").and_then(|v| match v {
        YamlValue::Null => None,
        YamlValue::String(s) => Some(s.clone()),
        _ => None,
    });

    let mut per_layer = HashMap::new();
    if let Some(YamlValue::Mapping(pl)) = yaml.get("per_layer") {
        for (key, val) in pl {
            let value = match val {
                YamlValue::Null => None,
                YamlValue::String(s) => Some(s.clone()),
                _ => None,
            };
            per_layer.insert(key.clone(), value);
        }
    }

    Ok(TensorTemplate {
        embedding,
        lm_head,
        final_norm,
        per_layer,
    })
}

fn parse_shape_template(yaml: &YamlValue) -> ShapeTemplate {
    let mut shapes = HashMap::new();
    if let YamlValue::Mapping(mapping) = yaml {
        for (key, val) in mapping {
            if let Some(s) = val.as_str() {
                shapes.insert(key.clone(), s.to_string());
            }
        }
    }
    ShapeTemplate { shapes }
}

fn parse_chat_template(yaml: &YamlValue) -> Result<ChatTemplateConfig> {
    Ok(ChatTemplateConfig {
        format: yaml.get_str("format").unwrap_or("").to_string(),
        template: yaml.get_str("template").unwrap_or("").to_string(),
        bos_token: yaml.get_str("bos_token").unwrap_or("").to_string(),
        eos_token: yaml.get_str("eos_token").unwrap_or("").to_string(),
        special_tokens: yaml
            .get("special_tokens")
            .and_then(|st| {
                if let YamlValue::Mapping(m) = st {
                    Some(
                        m.iter()
                            .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                            .collect(),
                    )
                } else {
                    None
                }
            })
            .unwrap_or_default(),
    })
}

fn parse_certification(yaml: &YamlValue) -> Result<CertificationConfig> {
    Ok(CertificationConfig {
        playbook_path: yaml.get_str("playbook_path").unwrap_or("").to_string(),
        csv_family_key: yaml.get_str("csv_family_key").unwrap_or("").to_string(),
        size_categories: yaml
            .get("size_categories")
            .and_then(|sc| {
                if let YamlValue::Mapping(m) = sc {
                    Some(
                        m.iter()
                            .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                            .collect(),
                    )
                } else {
                    None
                }
            })
            .unwrap_or_default(),
    })
}

// ============================================================================
// Directory Loading
// ============================================================================

/// Load all model family contracts from a directory.
///
/// Reads all `.yaml` files in the `model-families/` subdirectory
/// (skipping `_`-prefixed files) and returns a populated `FamilyRegistry`.
///
/// # Errors
///
/// Returns `AprenderError::FormatError` if the directory cannot be read
/// or any YAML file fails to parse.
pub fn load_family_registry(contracts_dir: &Path) -> Result<FamilyRegistry> {
    let families_dir = contracts_dir.join("model-families");
    let mut registry = FamilyRegistry::new();

    if !families_dir.exists() {
        return Ok(registry);
    }

    let entries = std::fs::read_dir(&families_dir).map_err(|e| AprenderError::FormatError {
        message: format!(
            "Failed to read contracts directory {}: {e}",
            families_dir.display()
        ),
    })?;

    for entry in entries {
        let entry = entry.map_err(|e| AprenderError::FormatError {
            message: format!("Failed to read directory entry: {e}"),
        })?;

        let path = entry.path();
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        // Skip non-YAML files and _-prefixed files
        let ext_is_yaml = Path::new(file_name)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("yaml") || ext.eq_ignore_ascii_case("yml"));
        if !ext_is_yaml {
            continue;
        }
        if file_name.starts_with('_') {
            continue;
        }

        let config = load_family_yaml(&path)?;
        registry.register(Box::new(DynModelFamily::new(config)));
    }

    Ok(registry)
}

/// Get the default contracts directory path relative to a project root.
#[must_use]
pub fn default_contracts_dir(project_root: &Path) -> std::path::PathBuf {
    project_root.join("contracts")
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_scalar_types() {
        assert!(matches!(parse_scalar("42"), YamlValue::Int(42)));
        assert!(matches!(parse_scalar("3.14"), YamlValue::Float(_)));
        assert!(matches!(parse_scalar("true"), YamlValue::Bool(true)));
        assert!(matches!(parse_scalar("false"), YamlValue::Bool(false)));
        assert!(matches!(parse_scalar("null"), YamlValue::Null));
        assert!(matches!(parse_scalar("\"hello\""), YamlValue::String(_)));
        assert!(matches!(parse_scalar("hello"), YamlValue::String(_)));
    }

    #[test]
    fn test_parse_simple_yaml() {
        let yaml = r#"
family: qwen2
display_name: "Qwen2"
vendor: Alibaba
"#;
        let result = parse_yaml(yaml).expect("parse yaml");
        assert_eq!(result.get_str("family"), Some("qwen2"));
        assert_eq!(result.get_str("display_name"), Some("Qwen2"));
        assert_eq!(result.get_str("vendor"), Some("Alibaba"));
    }

    #[test]
    fn test_parse_nested_yaml() {
        let yaml = r#"
constraints:
  attention_type: gqa
  has_bias: true
  activation: silu
"#;
        let result = parse_yaml(yaml).expect("parse yaml");
        let constraints = result.get("constraints").expect("constraints");
        assert_eq!(constraints.get_str("attention_type"), Some("gqa"));
        assert_eq!(constraints.get_bool("has_bias"), Some(true));
    }

    #[test]
    fn test_parse_sequence_yaml() {
        let yaml = r#"
architectures:
  - Qwen2ForCausalLM
  - Qwen2ForSequenceClassification
"#;
        let result = parse_yaml(yaml).expect("parse yaml");
        let arch = result
            .get("architectures")
            .and_then(YamlValue::as_sequence)
            .expect("architectures");
        assert_eq!(arch.len(), 2);
        assert_eq!(arch[0].as_str(), Some("Qwen2ForCausalLM"));
    }

    #[test]
    fn test_load_qwen2_yaml() {
        let contracts_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("contracts");
        let yaml_path = contracts_dir.join("model-families/qwen2.yaml");
        if yaml_path.exists() {
            let config = load_family_yaml(&yaml_path).expect("load qwen2 yaml");
            assert_eq!(config.family, "qwen2");
            assert_eq!(config.vendor, "Alibaba");
            assert!(!config.size_variants.is_empty());
            assert!(config.size_variants.contains_key("0.5b"));

            let half_b = &config.size_variants["0.5b"];
            assert_eq!(half_b.hidden_dim, 896);
            assert_eq!(half_b.num_layers, 24);
        }
    }

    #[test]
    fn test_load_family_registry() {
        let contracts_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("contracts");
        if contracts_dir.join("model-families").exists() {
            let registry = load_family_registry(&contracts_dir).expect("load registry");
            assert!(
                !registry.is_empty(),
                "Registry should have at least one family"
            );

            let names = registry.family_names();
            assert!(names.contains(&"qwen2"), "Registry should contain qwen2");
        }
    }

    #[test]
    fn test_parse_null_values() {
        let yaml = r#"
tensor_template:
  embedding: model.embed_tokens.weight
  per_layer:
    q_proj: "model.layers.{n}.self_attn.q_proj.weight"
    q_proj_bias: null
"#;
        let result = parse_yaml(yaml).expect("parse yaml");
        let template = parse_tensor_template(result.get("tensor_template").expect("template"))
            .expect("parse template");
        assert!(template.per_layer.get("q_proj").expect("q_proj").is_some());
        assert!(template
            .per_layer
            .get("q_proj_bias")
            .expect("q_proj_bias")
            .is_none());
    }

    #[test]
    fn test_parse_scalar_quoted_single() {
        assert!(matches!(parse_scalar("'hello'"), YamlValue::String(_)));
        if let YamlValue::String(s) = parse_scalar("'hello'") {
            assert_eq!(s, "hello");
        }
    }

    #[test]
    fn test_parse_scalar_tilde_null() {
        assert!(matches!(parse_scalar("~"), YamlValue::Null));
    }

    #[test]
    fn test_parse_scalar_yes_no() {
        assert!(matches!(parse_scalar("yes"), YamlValue::Bool(true)));
        assert!(matches!(parse_scalar("no"), YamlValue::Bool(false)));
    }

    #[test]
    fn test_yaml_value_as_i64() {
        let v = YamlValue::Int(42);
        assert_eq!(v.as_i64(), Some(42));
        let v = YamlValue::Float(3.9);
        assert_eq!(v.as_i64(), Some(3));
        let v = YamlValue::String("nope".to_string());
        assert_eq!(v.as_i64(), None);
    }

    #[test]
    fn test_yaml_value_as_f64() {
        let v = YamlValue::Float(3.14);
        assert!((v.as_f64().expect("f64") - 3.14).abs() < f64::EPSILON);
        let v = YamlValue::Int(42);
        assert!((v.as_f64().expect("f64") - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_yaml_value_as_bool() {
        let v = YamlValue::Bool(true);
        assert_eq!(v.as_bool(), Some(true));
        let v = YamlValue::String("nope".to_string());
        assert_eq!(v.as_bool(), None);
    }

    #[test]
    fn test_yaml_value_as_usize() {
        let v = YamlValue::Int(10);
        assert_eq!(v.as_usize(), Some(10));
        let v = YamlValue::Int(-1);
        assert_eq!(v.as_usize(), None);
    }

    #[test]
    fn test_yaml_value_as_sequence() {
        let v = YamlValue::Sequence(vec![YamlValue::Int(1)]);
        assert!(v.as_sequence().is_some());
        let v = YamlValue::Int(1);
        assert!(v.as_sequence().is_none());
    }

    #[test]
    fn test_yaml_value_as_mapping() {
        let v = YamlValue::Mapping(vec![("key".to_string(), YamlValue::Int(1))]);
        assert!(v.as_mapping().is_some());
        let v = YamlValue::Int(1);
        assert!(v.as_mapping().is_none());
    }

    #[test]
    fn test_yaml_value_get_on_non_mapping() {
        let v = YamlValue::Int(42);
        assert!(v.get("key").is_none());
        assert!(v.get_str("key").is_none());
        assert!(v.get_usize("key").is_none());
        assert!(v.get_f64("key").is_none());
        assert!(v.get_bool("key").is_none());
    }

    #[test]
    fn test_yaml_value_as_str_on_non_string() {
        let v = YamlValue::Int(42);
        assert!(v.as_str().is_none());
    }

    #[test]
    fn test_parse_empty_yaml() {
        let yaml = "";
        let result = parse_yaml(yaml).expect("parse yaml");
        assert!(result.get("anything").is_none());
    }

    #[test]
    fn test_parse_yaml_comments_only() {
        let yaml = "# just a comment\n# another comment\n";
        let result = parse_yaml(yaml).expect("parse yaml");
        assert!(result.get("anything").is_none());
    }

    #[test]
    fn test_default_contracts_dir() {
        let root = Path::new("/tmp/project");
        let dir = default_contracts_dir(root);
        assert_eq!(dir, Path::new("/tmp/project/contracts"));
    }

    #[test]
    fn test_load_family_registry_missing_dir() {
        let result = load_family_registry(Path::new("/nonexistent/path"));
        assert!(result.is_ok());
        assert!(result.expect("registry").is_empty());
    }

    #[test]
    fn test_parse_family_yaml_missing_family() {
        let yaml = r#"
display_name: "Test"
vendor: Test
architectures:
  - TestArch
hf_pattern: "test/*"
size_variants:
  small:
    parameters: "1B"
    hidden_dim: 768
    num_layers: 12
    num_heads: 12
    intermediate_dim: 3072
    vocab_size: 30000
constraints:
  attention_type: mha
tensor_template:
  embedding: "model.embed.weight"
"#;
        let result = parse_family_yaml(yaml, Path::new("test.yaml"));
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_shape_template_flat() {
        let yaml = r#"
shape_template:
  embedding: "[vocab_size, hidden_dim]"
  lm_head: "[vocab_size, hidden_dim]"
"#;
        let result = parse_yaml(yaml).expect("parse yaml");
        let shapes = parse_shape_template(result.get("shape_template").expect("shape_template"));
        assert_eq!(
            shapes.shapes.get("embedding"),
            Some(&"[vocab_size, hidden_dim]".to_string())
        );
        assert_eq!(
            shapes.shapes.get("lm_head"),
            Some(&"[vocab_size, hidden_dim]".to_string())
        );
    }

    #[test]
    fn test_parse_chat_template() {
        let yaml = r#"
chat_template:
  format: chatml
  template: "test template"
  bos_token: "<s>"
  eos_token: "</s>"
  special_tokens:
    pad: "<pad>"
"#;
        let result = parse_yaml(yaml).expect("parse yaml");
        let ct = parse_chat_template(result.get("chat_template").expect("chat_template"))
            .expect("parse chat template");
        assert_eq!(ct.format, "chatml");
        assert_eq!(ct.bos_token, "<s>");
        assert_eq!(ct.special_tokens.get("pad"), Some(&"<pad>".to_string()));
    }

    #[test]
    fn test_parse_certification() {
        let yaml = r#"
certification:
  playbook_path: "path/to/playbook.yaml"
  csv_family_key: "test"
  size_categories:
    small: tiny
    large: xlarge
"#;
        let result = parse_yaml(yaml).expect("parse yaml");
        let cert = parse_certification(result.get("certification").expect("certification"))
            .expect("parse certification");
        assert_eq!(cert.playbook_path, "path/to/playbook.yaml");
        assert_eq!(cert.csv_family_key, "test");
        assert_eq!(cert.size_categories.get("small"), Some(&"tiny".to_string()));
    }
}
