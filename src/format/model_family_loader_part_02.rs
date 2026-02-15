
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
