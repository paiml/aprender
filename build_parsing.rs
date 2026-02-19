// ============================================================================
// Data Types
// ============================================================================

struct FamilyData {
    family: String,
    display_name: String,
    vendor: String,
    architectures: Vec<String>,
    hf_pattern: String,
    sizes: Vec<SizeData>,
    constraints: ConstraintsData,
    embedding_tensor: String,
    lm_head_tensor: Option<String>,
    final_norm_tensor: Option<String>,
    per_layer_tensors: Vec<(String, String)>, // (role, pattern)
    quantizations: Vec<String>,
    chat_format: Option<String>,
    // GH-277: GGUF tensor name template for contract-driven export
    gguf_embedding: Option<String>,
    gguf_position_embedding: Option<String>,
    gguf_lm_head: Option<String>,
    gguf_final_norm_weight: Option<String>,
    gguf_final_norm_bias: Option<String>,
    gguf_per_layer: Vec<(String, String)>, // (role, gguf_suffix) - only non-null entries
    gguf_skip_roles: Vec<String>,          // roles with explicit null in gguf template
    gguf_transpose_weights: bool,          // GH-277: transpose Conv1D→Linear during export
    gguf_fuse: Vec<(String, Vec<String>)>, // GH-277: (gguf_suffix, [source_role, ...])
}

struct SizeData {
    name: String,
    parameters: String,
    hidden_dim: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    intermediate_dim: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    head_dim: usize,
    rope_theta: f64,
    norm_eps: f64,
}

struct ConstraintsData {
    attention: String,
    activation: String,
    norm: String,
    bias: bool,
    tied: bool,
    position: String,
    mlp: String,
    qk_norm: bool,
}

// ============================================================================
// Minimal YAML Parser (build.rs can't depend on the crate)
// ============================================================================

/// Given a trimmed YAML line like `key: "value"`, strip the key prefix and colon,
/// returning the raw value portion (e.g. `"value"` or `bare_value`).
fn strip_yaml_key<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let rest = line.strip_prefix(key)?;
    let rest = rest.trim_start();
    let rest = rest.strip_prefix(':')?;
    Some(rest.trim())
}

/// Interpret a raw YAML scalar value: remove surrounding quotes, or return
/// the bare value if it is not empty and not an array/object opener.
fn interpret_yaml_scalar(val: &str) -> Option<&str> {
    if val.starts_with('"') && val.ends_with('"') && val.len() >= 2 {
        return Some(&val[1..val.len() - 1]);
    }
    if !val.is_empty() && !val.starts_with('[') && !val.starts_with('{') {
        return Some(val);
    }
    None
}

fn get_str<'a>(content: &'a str, key: &str) -> Option<&'a str> {
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(val) = strip_yaml_key(trimmed, key) {
            if let Some(result) = interpret_yaml_scalar(val) {
                return Some(result);
            }
        }
    }
    None
}

fn get_usize(content: &str, key: &str) -> Option<usize> {
    get_str(content, key).and_then(|v| v.parse().ok())
}

fn get_f64(content: &str, key: &str) -> Option<f64> {
    get_str(content, key).and_then(|v| v.parse().ok())
}

fn get_bool(content: &str, key: &str) -> Option<bool> {
    get_str(content, key).map(|v| matches!(v, "true" | "yes"))
}

fn parse_family_yaml(content: &str, path: &Path) -> FamilyData {
    let err = |msg: &str| -> ! { panic!("PMAT-250: {}: {msg}", path.display()) };

    let family = get_str(content, "family").unwrap_or_else(|| err("missing 'family'"));
    let display_name =
        get_str(content, "display_name").unwrap_or_else(|| err("missing 'display_name'"));
    let vendor = get_str(content, "vendor").unwrap_or_else(|| err("missing 'vendor'"));
    let hf_pattern = get_str(content, "hf_pattern").unwrap_or("");

    // Parse architectures (list)
    let architectures = parse_list_section(content, "architectures");

    // Parse quantizations
    let quantizations = parse_list_section(content, "quantizations");

    // Parse constraints
    let constraints_section = extract_section(content, "constraints");
    let constraints = ConstraintsData {
        attention: get_str(&constraints_section, "attention_type")
            .unwrap_or("mha")
            .to_string(),
        activation: get_str(&constraints_section, "activation")
            .unwrap_or("silu")
            .to_string(),
        norm: get_str(&constraints_section, "norm_type")
            .unwrap_or("rmsnorm")
            .to_string(),
        bias: get_bool(&constraints_section, "has_bias").unwrap_or(false),
        tied: get_bool(&constraints_section, "tied_embeddings").unwrap_or(false),
        position: get_str(&constraints_section, "positional_encoding")
            .unwrap_or("rope")
            .to_string(),
        mlp: get_str(&constraints_section, "mlp_type")
            .unwrap_or("swiglu")
            .to_string(),
        qk_norm: get_bool(&constraints_section, "qk_norm").unwrap_or(false),
    };

    // Parse tensor_template
    let tt_section = extract_section(content, "tensor_template");

    // Look for embedding tensor: flat "embedding:" or nested first tensor value
    let embedding_tensor = get_str(&tt_section, "embedding")
        .map(String::from)
        .unwrap_or_else(|| {
            // Try nested: embeddings.word_embeddings (BERT) or encoder.conv1_weight (Whisper)
            // Find first quoted value in the section
            find_first_tensor_value(&tt_section).unwrap_or_default()
        });
    let lm_head_tensor = get_str(&tt_section, "lm_head")
        .filter(|s| *s != "null")
        .map(String::from);
    let final_norm_tensor = get_str(&tt_section, "final_norm")
        .filter(|s| *s != "null")
        .map(String::from);

    // Parse per_layer in tensor_template
    let per_layer_section = extract_section(&tt_section, "per_layer");
    let per_layer_tensors = parse_key_values(&per_layer_section);

    // Parse size_variants
    let sizes = parse_size_variants(content, path);

    // Parse chat template format
    let ct_section = extract_section(content, "chat_template");
    let chat_format = get_str(&ct_section, "format").map(String::from);

    // GH-277: Parse gguf_tensor_template
    let gguf_section = extract_section(content, "gguf_tensor_template");
    let gguf_embedding = get_str(&gguf_section, "embedding")
        .filter(|s| *s != "null")
        .map(String::from);
    let gguf_position_embedding = get_str(&gguf_section, "position_embedding")
        .filter(|s| *s != "null")
        .map(String::from);
    let gguf_lm_head = get_str(&gguf_section, "lm_head")
        .filter(|s| *s != "null")
        .map(String::from);
    let gguf_final_norm_weight = get_str(&gguf_section, "final_norm_weight")
        .filter(|s| *s != "null")
        .map(String::from);
    let gguf_final_norm_bias = get_str(&gguf_section, "final_norm_bias")
        .filter(|s| *s != "null")
        .map(String::from);

    let gguf_pl_section = extract_section(&gguf_section, "per_layer");
    let gguf_all_kv = parse_key_values_with_null(&gguf_pl_section);
    let mut gguf_per_layer = Vec::new();
    let mut gguf_skip_roles = Vec::new();
    for (role, val) in gguf_all_kv {
        if val == "null" || val.is_empty() {
            gguf_skip_roles.push(role);
        } else {
            gguf_per_layer.push((role, val));
        }
    }

    // GH-277: Parse transpose_weights and fuse rules from gguf_tensor_template
    let gguf_transpose_weights =
        get_str(&gguf_section, "transpose_weights").is_some_and(|s| s == "true");
    let gguf_fuse = parse_fuse_rules(&gguf_section);

    FamilyData {
        family: family.to_string(),
        display_name: display_name.to_string(),
        vendor: vendor.to_string(),
        architectures,
        hf_pattern: hf_pattern.to_string(),
        sizes,
        constraints,
        embedding_tensor,
        lm_head_tensor,
        final_norm_tensor,
        per_layer_tensors,
        quantizations,
        chat_format,
        gguf_embedding,
        gguf_position_embedding,
        gguf_lm_head,
        gguf_final_norm_weight,
        gguf_final_norm_bias,
        gguf_per_layer,
        gguf_skip_roles,
        gguf_transpose_weights,
        gguf_fuse,
    }
}

fn find_first_tensor_value(section: &str) -> Option<String> {
    for line in section.lines() {
        let trimmed = line.trim();
        // Look for lines with quoted values like: key: "tensor.name.weight"
        if let Some(colon_pos) = trimmed.find(':') {
            let val = trimmed[colon_pos + 1..].trim();
            if val.starts_with('"') && val.ends_with('"') && val.len() > 2 {
                return Some(val[1..val.len() - 1].to_string());
            }
        }
    }
    None
}

fn parse_list_section(content: &str, section: &str) -> Vec<String> {
    let mut items = Vec::new();
    let mut in_section = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(&format!("{section}:")) {
            in_section = true;
            continue;
        }
        if in_section {
            if let Some(rest) = trimmed.strip_prefix("- ") {
                let val = rest.trim().trim_matches('"');
                items.push(val.to_string());
            } else if !trimmed.is_empty() && !trimmed.starts_with('#') {
                break;
            }
        }
    }
    items
}

fn extract_section(content: &str, section: &str) -> String {
    let mut lines = Vec::new();
    let mut in_section = false;
    let mut section_indent = 0;

    for line in content.lines() {
        if !in_section {
            let trimmed = line.trim();
            if trimmed.starts_with(&format!("{section}:")) {
                in_section = true;
                section_indent = line.len() - line.trim_start().len();
            }
        } else if line.trim().is_empty() {
            lines.push(String::new());
        } else {
            let indent = line.len() - line.trim_start().len();
            if indent <= section_indent {
                break;
            }
            lines.push(line.to_string());
        }
    }
    lines.join("\n")
}

fn parse_key_values(content: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(colon_pos) = trimmed.find(':') {
            let key = trimmed[..colon_pos].trim();
            let val = trimmed[colon_pos + 1..].trim().trim_matches('"');
            if !key.is_empty() && val != "null" && !val.is_empty() {
                pairs.push((key.to_string(), val.to_string()));
            }
        }
    }
    pairs
}

/// Like `parse_key_values` but preserves "null" entries instead of skipping them.
fn parse_key_values_with_null(content: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(colon_pos) = trimmed.find(':') {
            let key = trimmed[..colon_pos].trim();
            let val = trimmed[colon_pos + 1..].trim().trim_matches('"');
            if !key.is_empty() {
                pairs.push((key.to_string(), val.to_string()));
            }
        }
    }
    pairs
}

/// GH-277: Parse fuse rules from the gguf_tensor_template section.
///
/// Expects YAML like:
/// ```yaml
/// fuse:
///   - gguf_name: "attn_qkv.weight"
///     sources: [q_proj_weight, k_proj_weight, v_proj_weight]
/// ```
/// Parse a YAML inline array like `[a, b, c]` from a line containing brackets.
fn parse_yaml_inline_array(line: &str) -> Vec<String> {
    let Some(start) = line.find('[') else {
        return Vec::new();
    };
    let Some(end) = line.find(']') else {
        return Vec::new();
    };
    line[start + 1..end]
        .split(',')
        .map(|s| s.trim().trim_matches('"').to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn parse_fuse_rules(gguf_section: &str) -> Vec<(String, Vec<String>)> {
    let fuse_section = extract_section(gguf_section, "fuse");
    if fuse_section.trim().is_empty() {
        return Vec::new();
    }

    let mut rules = Vec::new();
    let mut current_gguf_name: Option<String> = None;
    let mut current_sources: Vec<String> = Vec::new();

    for line in fuse_section.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("- gguf_name:") || trimmed.starts_with("-  gguf_name:") {
            if let Some(name) = current_gguf_name.take() {
                if !current_sources.is_empty() {
                    rules.push((name, std::mem::take(&mut current_sources)));
                }
            }
            let val = trimmed
                .split(':')
                .nth(1)
                .unwrap_or("")
                .trim()
                .trim_matches('"');
            current_gguf_name = Some(val.to_string());
        } else if trimmed.starts_with("sources:") {
            current_sources = parse_yaml_inline_array(trimmed);
        }
    }

    if let Some(name) = current_gguf_name {
        if !current_sources.is_empty() {
            rules.push((name, current_sources));
        }
    }

    rules
}

fn parse_size_variants(content: &str, path: &Path) -> Vec<SizeData> {
    let section = extract_section(content, "size_variants");
    let mut sizes = Vec::new();

    // Find size names (lines that end with ":" at the top indent level)
    let mut current_name: Option<String> = None;
    let mut current_block = String::new();

    for line in section.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        let trimmed = line.trim();

        // Size variant names are at indent 2 (relative to section) and end with ":"
        if indent <= 4 && trimmed.ends_with(':') && !trimmed.contains(' ') {
            // Save previous block
            if let Some(name) = current_name.take() {
                sizes.push(parse_size_block(&name, &current_block, path));
            }
            current_name = Some(trimmed.trim_end_matches(':').to_string());
            current_block = String::new();
        } else if current_name.is_some() {
            current_block.push_str(line);
            current_block.push('\n');
        }
    }

    // Don't forget the last block
    if let Some(name) = current_name {
        sizes.push(parse_size_block(&name, &current_block, path));
    }

    sizes
}

fn parse_size_block(name: &str, block: &str, path: &Path) -> SizeData {
    let warn = |field: &str| {
        eprintln!(
            "cargo:warning=PMAT-250: {}: size_variants.{name}.{field} not found, using default",
            path.display()
        );
    };

    let hidden_dim = get_usize(block, "hidden_dim")
        .or_else(|| get_usize(block, "d_model"))
        .unwrap_or_else(|| {
            warn("hidden_dim");
            0
        });
    let num_layers = get_usize(block, "num_layers")
        .or_else(|| get_usize(block, "encoder_layers"))
        .unwrap_or_else(|| {
            warn("num_layers");
            0
        });
    let num_heads = get_usize(block, "num_heads")
        .or_else(|| get_usize(block, "encoder_attention_heads"))
        .unwrap_or_else(|| {
            warn("num_heads");
            0
        });
    let num_kv_heads = get_usize(block, "num_kv_heads").unwrap_or(num_heads);
    let intermediate_dim = get_usize(block, "intermediate_dim")
        .or_else(|| get_usize(block, "encoder_ffn_dim"))
        .unwrap_or(0);
    let vocab_size = get_usize(block, "vocab_size").unwrap_or(0);
    let max_pos = get_usize(block, "max_position_embeddings").unwrap_or(0);
    let head_dim = get_usize(block, "head_dim").unwrap_or_else(|| {
        if num_heads > 0 {
            hidden_dim / num_heads
        } else {
            0
        }
    });
    let rope_theta = get_f64(block, "rope_theta").unwrap_or(0.0);
    let norm_eps = get_f64(block, "rms_norm_eps")
        .or_else(|| get_f64(block, "norm_eps"))
        .or_else(|| get_f64(block, "layer_norm_eps"))
        .unwrap_or(1e-6);

    let parameters = get_str(block, "parameters")
        .unwrap_or("unknown")
        .to_string();

    SizeData {
        name: name.to_string(),
        parameters,
        hidden_dim,
        num_layers,
        num_heads,
        num_kv_heads,
        intermediate_dim,
        vocab_size,
        max_position_embeddings: max_pos,
        head_dim,
        rope_theta,
        norm_eps,
    }
}

