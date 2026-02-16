//! Build script for aprender (PMAT-250)
//!
//! Reads model family contract YAML files from `contracts/model-families/`
//! and generates Rust code with compiled-in family data.
//!
//! This catches YAML/Rust contract drift at build time:
//! - If a YAML file has missing/invalid fields, the build fails
//! - If YAML schema changes break the parser, the build fails
//! - Generated code includes `build_default_registry()` for zero-overhead
//!   family detection (no runtime YAML parsing needed)

use std::fs;
use std::path::Path;

fn main() {
    let families_dir = Path::new("contracts/model-families");

    // Tell Cargo to re-run if any YAML changes
    println!("cargo:rerun-if-changed=contracts/model-families");

    if !families_dir.exists() {
        // No contracts directory — generate empty registry
        let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
        let out_path = Path::new(&out_dir).join("model_families_generated.rs");
        fs::write(
            &out_path,
            "// No contracts/model-families/ directory found at build time\n\
             // Using empty generated registry\n\
             \n\
             /// Known model family names (generated at build time)\n\
             pub const KNOWN_FAMILIES: &[&str] = &[];\n",
        )
        .expect("write generated code");
        return;
    }

    let mut families: Vec<FamilyData> = Vec::new();

    let entries = fs::read_dir(families_dir).expect("read contracts/model-families");
    for entry in entries {
        let entry = entry.expect("dir entry");
        let path = entry.path();

        // Skip non-YAML and _-prefixed files
        if path.extension().map_or(true, |e| e != "yaml") {
            continue;
        }
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .map_or(true, |s| s.starts_with('_'))
        {
            continue;
        }

        let content = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));

        let family = parse_family_yaml(&content, &path);
        families.push(family);
    }

    // Sort by family name for deterministic output
    families.sort_by(|a, b| a.family.cmp(&b.family));

    // Generate Rust source
    let generated = generate_rust(&families);

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = Path::new(&out_dir).join("model_families_generated.rs");
    fs::write(&out_path, generated).expect("write generated code");
}

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
}

// ============================================================================
// Minimal YAML Parser (build.rs can't depend on the crate)
// ============================================================================

fn get_str<'a>(content: &'a str, key: &str) -> Option<&'a str> {
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix(key) {
            let rest = rest.trim_start();
            if let Some(rest) = rest.strip_prefix(':') {
                let val = rest.trim();
                // Remove surrounding quotes
                if val.starts_with('"') && val.ends_with('"') && val.len() >= 2 {
                    return Some(&val[1..val.len() - 1]);
                }
                if !val.is_empty() && !val.starts_with('[') && !val.starts_with('{') {
                    return Some(val);
                }
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

// ============================================================================
// Code Generation
// ============================================================================

fn generate_rust(families: &[FamilyData]) -> String {
    let mut out = String::new();

    out.push_str("// AUTO-GENERATED by build.rs (PMAT-250)\n");
    out.push_str("// DO NOT EDIT — regenerated from contracts/model-families/*.yaml\n");
    out.push_str("//\n");
    out.push_str("// This file is included by src/format/model_family.rs via include!\n\n");

    // Generate KNOWN_FAMILIES constant
    out.push_str("/// Known model family names (generated at build time from YAML contracts)\n");
    out.push_str("pub const KNOWN_FAMILIES: &[&str] = &[\n");
    for f in families {
        out.push_str(&format!("    \"{}\",\n", f.family));
    }
    out.push_str("];\n\n");

    // Generate per-family constants
    for f in families {
        let upper = f.family.to_uppercase();
        out.push_str(&format!("/// {} family display name\n", f.display_name));
        out.push_str(&format!(
            "pub const {upper}_DISPLAY_NAME: &str = \"{}\";\n",
            f.display_name
        ));
        out.push_str(&format!(
            "pub const {upper}_VENDOR: &str = \"{}\";\n",
            f.vendor
        ));

        // Size variant constants
        for s in &f.sizes {
            let size_upper = s.name.replace('.', "_").to_uppercase();
            let prefix = format!("{upper}_{size_upper}");
            out.push_str(&format!(
                "pub const {prefix}_HIDDEN_DIM: usize = {};\n",
                s.hidden_dim
            ));
            out.push_str(&format!(
                "pub const {prefix}_NUM_LAYERS: usize = {};\n",
                s.num_layers
            ));
            out.push_str(&format!(
                "pub const {prefix}_NUM_HEADS: usize = {};\n",
                s.num_heads
            ));
            out.push_str(&format!(
                "pub const {prefix}_NUM_KV_HEADS: usize = {};\n",
                s.num_kv_heads
            ));
            out.push_str(&format!(
                "pub const {prefix}_INTERMEDIATE_DIM: usize = {};\n",
                s.intermediate_dim
            ));
            out.push_str(&format!(
                "pub const {prefix}_VOCAB_SIZE: usize = {};\n",
                s.vocab_size
            ));
            out.push_str(&format!(
                "pub const {prefix}_HEAD_DIM: usize = {};\n",
                s.head_dim
            ));
            out.push_str(&format!(
                "pub const {prefix}_MAX_POSITION_EMBEDDINGS: usize = {};\n",
                s.max_position_embeddings
            ));
        }
        out.push('\n');

        // Compile-time algebraic proofs (§3.14, §5.6 of spec)
        out.push_str(&generate_algebraic_proofs(f));
    }

    // Generate build_default_registry function
    out.push_str("/// Build a `FamilyRegistry` populated with all families from YAML contracts.\n");
    out.push_str("///\n");
    out.push_str("/// This function uses compiled-in data from build.rs — no runtime YAML\n");
    out.push_str("/// parsing is needed. The data was validated at build time.\n");
    out.push_str("#[must_use]\n");
    out.push_str("pub fn build_default_registry() -> FamilyRegistry {\n");
    out.push_str("    let mut registry = FamilyRegistry::new();\n\n");

    for f in families {
        out.push_str(&generate_family_registration(f));
    }

    out.push_str("    registry\n");
    out.push_str("}\n");

    out
}

fn generate_family_registration(f: &FamilyData) -> String {
    let mut out = String::new();

    out.push_str("    {\n");
    out.push_str("        let mut size_variants = std::collections::HashMap::new();\n");

    for s in &f.sizes {
        out.push_str(&format!(
            "        size_variants.insert(\"{}\".to_string(), ModelSizeConfig {{\n",
            s.name
        ));
        out.push_str(&format!(
            "            parameters: \"{}\".to_string(),\n",
            s.parameters
        ));
        out.push_str(&format!("            hidden_dim: {},\n", s.hidden_dim));
        out.push_str(&format!("            num_layers: {},\n", s.num_layers));
        out.push_str(&format!("            num_heads: {},\n", s.num_heads));
        out.push_str(&format!("            num_kv_heads: {},\n", s.num_kv_heads));
        out.push_str(&format!(
            "            intermediate_dim: {},\n",
            s.intermediate_dim
        ));
        out.push_str(&format!("            vocab_size: {},\n", s.vocab_size));
        out.push_str(&format!(
            "            max_position_embeddings: {},\n",
            s.max_position_embeddings
        ));
        out.push_str(&format!("            head_dim: {},\n", s.head_dim));
        out.push_str(&format!(
            "            rope_theta: {}_f64,\n",
            format_f64(s.rope_theta)
        ));
        out.push_str(&format!(
            "            norm_eps: {}_f64,\n",
            format_f64(s.norm_eps)
        ));
        out.push_str("        });\n");
    }

    // Per-layer tensors
    out.push_str("        let mut per_layer = std::collections::HashMap::new();\n");
    for (role, pattern) in &f.per_layer_tensors {
        out.push_str(&format!(
            "        per_layer.insert(\"{role}\".to_string(), Some(\"{pattern}\".to_string()));\n"
        ));
    }

    // Shape template (empty — shapes are in YAML but we don't need them at codegen level)
    out.push_str("        let shapes = std::collections::HashMap::new();\n");

    // Chat template
    if f.chat_format.is_some() {
        out.push_str("        // Chat template parsed at runtime if needed\n");
    }

    out.push_str(&format!(
        "        let config = ModelFamilyConfig {{\n\
         \x20           family: \"{}\".to_string(),\n\
         \x20           display_name: \"{}\".to_string(),\n\
         \x20           vendor: \"{}\".to_string(),\n\
         \x20           architectures: vec![{}],\n\
         \x20           hf_pattern: \"{}\".to_string(),\n\
         \x20           size_variants,\n\
         \x20           constraints: ModelConstraints {{\n\
         \x20               attention_type: AttentionType::from_str_contract(\"{}\").unwrap_or(AttentionType::Mha),\n\
         \x20               activation: Activation::from_str_contract(\"{}\").unwrap_or(Activation::Silu),\n\
         \x20               norm_type: NormType::from_str_contract(\"{}\").unwrap_or(NormType::RmsNorm),\n\
         \x20               has_bias: {},\n\
         \x20               tied_embeddings: {},\n\
         \x20               positional_encoding: PositionalEncoding::from_str_contract(\"{}\").unwrap_or(PositionalEncoding::Rope),\n\
         \x20               mlp_type: MlpType::from_str_contract(\"{}\").unwrap_or(MlpType::SwiGlu),\n\
         \x20           }},\n\
         \x20           tensor_template: TensorTemplate {{\n\
         \x20               embedding: \"{}\".to_string(),\n\
         \x20               lm_head: {},\n\
         \x20               final_norm: {},\n\
         \x20               per_layer,\n\
         \x20           }},\n\
         \x20           shape_template: ShapeTemplate {{ shapes }},\n\
         \x20           quantizations: vec![{}],\n\
         \x20           chat_template: None,\n\
         \x20           certification: None,\n\
         \x20       }};\n",
        f.family,
        f.display_name,
        f.vendor,
        f.architectures
            .iter()
            .map(|a| format!("\"{a}\".to_string()"))
            .collect::<Vec<_>>()
            .join(", "),
        f.hf_pattern,
        f.constraints.attention,
        f.constraints.activation,
        f.constraints.norm,
        f.constraints.bias,
        f.constraints.tied,
        f.constraints.position,
        f.constraints.mlp,
        f.embedding_tensor,
        f.lm_head_tensor
            .as_ref()
            .map_or("None".to_string(), |s| format!("Some(\"{s}\".to_string())")),
        f.final_norm_tensor
            .as_ref()
            .map_or("None".to_string(), |s| format!("Some(\"{s}\".to_string())")),
        f.quantizations
            .iter()
            .map(|q| format!("\"{q}\".to_string()"))
            .collect::<Vec<_>>()
            .join(", "),
    ));

    out.push_str("        registry.register(Box::new(DynModelFamily::new(config)));\n    }\n\n");

    out
}

fn format_f64(v: f64) -> String {
    if v == 0.0 {
        "0.0".to_string()
    } else if v.fract() == 0.0 {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

// ============================================================================
// Compile-Time Algebraic Proofs (Spec §3.14, §5.6)
//
// These generate `const _: () = assert!(...)` statements that are evaluated
// by the Rust compiler at build time. If any invariant is violated, the build
// fails — the binary cannot exist in a state that violates these theorems.
// ============================================================================

fn generate_algebraic_proofs(f: &FamilyData) -> String {
    let mut out = String::new();
    let upper = f.family.to_uppercase();

    out.push_str(&format!("// ── Algebraic proofs for {} ──\n", f.family));

    for s in &f.sizes {
        let size_upper = s.name.replace('.', "_").to_uppercase();
        let prefix = format!("{upper}_{size_upper}");

        // FALSIFY-ALG-005: Non-degeneracy constraints
        // These are UNCONDITIONAL — a model with hidden_dim=0 is always invalid.
        // Previous version had tautological guards (if x > 0 { assert!(x > 0) })
        // that silently passed degenerate models. Found via self-falsification.
        out.push_str(&format!(
            "const _: () = assert!({prefix}_HIDDEN_DIM > 0, \
             \"non-degeneracy: {}/{} hidden_dim must be positive\");\n",
            f.family, s.name
        ));
        out.push_str(&format!(
            "const _: () = assert!({prefix}_NUM_LAYERS > 0, \
             \"non-degeneracy: {}/{} num_layers must be positive\");\n",
            f.family, s.name
        ));
        out.push_str(&format!(
            "const _: () = assert!({prefix}_NUM_HEADS > 0, \
             \"non-degeneracy: {}/{} num_heads must be positive\");\n",
            f.family, s.name
        ));
        out.push_str(&format!(
            "const _: () = assert!({prefix}_VOCAB_SIZE > 0, \
             \"non-degeneracy: {}/{} vocab_size must be positive\");\n",
            f.family, s.name
        ));
        out.push_str(&format!(
            "const _: () = assert!({prefix}_NUM_KV_HEADS > 0, \
             \"non-degeneracy: {}/{} num_kv_heads must be positive\");\n",
            f.family, s.name
        ));

        // FALSIFY-ALG-008: KV head ordering constraint
        // num_kv_heads <= num_heads (GQA reduces heads, never adds)
        out.push_str(&format!(
            "const _: () = assert!({prefix}_NUM_KV_HEADS <= {prefix}_NUM_HEADS, \
             \"GQA ordering: {}/{} num_kv_heads must be <= num_heads\");\n",
            f.family, s.name
        ));

        // FALSIFY-ALG-001: Attention head divisibility (Vaswani, 2017)
        // hidden_dim % num_heads == 0
        // Unconditional — non-degeneracy asserts above guarantee nonzero divisor
        out.push_str(&format!(
            "const _: () = assert!({prefix}_HIDDEN_DIM % {prefix}_NUM_HEADS == 0, \
             \"Vaswani (2017): {}/{} hidden_dim must be divisible by num_heads\");\n",
            f.family, s.name
        ));

        // FALSIFY-ALG-002: GQA group divisibility (Ainslie et al., 2023)
        // num_heads % num_kv_heads == 0
        // Skip when num_kv_heads == 1 (MQA) to avoid clippy::modulo_one
        if s.num_kv_heads > 1 {
            out.push_str(&format!(
                "const _: () = assert!({prefix}_NUM_HEADS % {prefix}_NUM_KV_HEADS == 0, \
                 \"Ainslie (2023) GQA: {}/{} num_heads must be divisible by num_kv_heads\");\n",
                f.family, s.name
            ));
        }

        // FALSIFY-ALG-003: Head dimension bounds
        // head_dim >= hidden_dim / num_heads (lower bound)
        // head_dim <= 2 * (hidden_dim / num_heads) (upper bound — Gemma uses 1.33x)
        out.push_str(&format!(
            "const _: () = assert!({prefix}_HEAD_DIM >= {prefix}_HIDDEN_DIM / {prefix}_NUM_HEADS, \
             \"head_dim underflow: {}/{} head_dim must be >= hidden_dim/num_heads\");\n",
            f.family, s.name
        ));
        out.push_str(&format!(
            "const _: () = assert!({prefix}_HEAD_DIM <= 2 * ({prefix}_HIDDEN_DIM / {prefix}_NUM_HEADS), \
             \"head_dim overflow: {}/{} head_dim must be <= 2x hidden_dim/num_heads\");\n",
            f.family, s.name
        ));

        // FALSIFY-ALG-004: FFN expansion ratio (Shazeer, 2020)
        // intermediate_dim > hidden_dim
        out.push_str(&format!(
            "const _: () = assert!({prefix}_INTERMEDIATE_DIM > {prefix}_HIDDEN_DIM, \
             \"Shazeer (2020) FFN expansion: {}/{} intermediate_dim must exceed hidden_dim\");\n",
            f.family, s.name
        ));

        // NOTE: max_position_embeddings > 0 is only enforced for RoPE models (ALG-007).
        // Non-RoPE models like Whisper define context size via different fields
        // (max_source_positions, max_target_positions), making max_position_embeddings=0 valid.
    }

    // FALSIFY-ALG-006: Activation-MLP consistency (Shazeer, 2020)
    // SwiGLU requires SiLU, GeGLU/GatedMlp requires GELU, GeluMlp requires GELU.
    // The match lists VALID combinations — anything else is INVALID.
    let activation_mlp_valid = match (
        f.constraints.mlp.as_str(),
        f.constraints.activation.as_str(),
    ) {
        ("swiglu", "silu") => true,
        ("gelu_mlp", "gelu") => true,
        ("gated_mlp", "gelu") => true,
        ("gated_mlp", "silu") => true, // Moonshine decoder: SiLU-gated MLP
        // Unknown MLP types pass (future-proof for new architectures)
        (mlp, _) if mlp != "swiglu" && mlp != "gelu_mlp" && mlp != "gated_mlp" => true,
        // Known MLP type with WRONG activation — this is the bug we catch
        _ => false,
    };
    assert!(
        activation_mlp_valid,
        "PMAT-250: {} has inconsistent activation/MLP: activation={}, mlp={} \
         (Shazeer 2020: swiglu→silu, gelu_mlp→gelu, gated_mlp→gelu)",
        f.family, f.constraints.activation, f.constraints.mlp
    );

    // FALSIFY-ALG-007: RoPE requirements (Su et al., 2024)
    if f.constraints.position == "rope" {
        for s in &f.sizes {
            let size_upper = s.name.replace('.', "_").to_uppercase();
            let prefix = format!("{upper}_{size_upper}");

            // head_dim must be even for cos/sin pairs — UNCONDITIONAL
            out.push_str(&format!(
                "const _: () = assert!({prefix}_HEAD_DIM % 2 == 0, \
                 \"Su (2024) RoPE: {}/{} head_dim must be even for cos/sin pairs\");\n",
                f.family, s.name
            ));

            // max_position_embeddings must be positive — UNCONDITIONAL
            out.push_str(&format!(
                "const _: () = assert!({prefix}_MAX_POSITION_EMBEDDINGS > 0, \
                 \"Su (2024) RoPE: {}/{} max_position_embeddings must be positive\");\n",
                f.family, s.name
            ));

            // rope_theta > 0 is checked at parse time (f64, not const-friendly)
            // We validate it in build.rs directly:
            assert!(
                s.rope_theta > 0.0,
                "PMAT-250: {}/{} has rope_theta={} but positional_encoding=rope \
                 (Su et al., 2024 requires theta > 0)",
                f.family,
                s.name,
                s.rope_theta
            );
            assert!(
                s.rope_theta.is_finite(),
                "PMAT-250: {}/{} has non-finite rope_theta={}",
                f.family,
                s.name,
                s.rope_theta
            );
        }
    }

    // FALSIFY-ALG-009: Norm epsilon positivity (Zhang & Sennrich, 2019)
    // RMSNorm computes x / sqrt(mean(x²) + eps) — eps=0 causes division by zero
    // on zero inputs. LayerNorm has the same requirement.
    for s in &f.sizes {
        assert!(
            s.norm_eps > 0.0,
            "PMAT-250: {}/{} has norm_eps={} — must be positive \
             (Zhang & Sennrich 2019: RMSNorm requires eps > 0 to prevent division by zero)",
            f.family,
            s.name,
            s.norm_eps
        );
        assert!(
            s.norm_eps < 1.0,
            "PMAT-250: {}/{} has norm_eps={} — must be < 1.0 \
             (values near 1.0 collapse all activations to zero in RMSNorm)",
            f.family,
            s.name,
            s.norm_eps
        );
        assert!(
            s.norm_eps.is_finite(),
            "PMAT-250: {}/{} has non-finite norm_eps={}",
            f.family,
            s.name,
            s.norm_eps
        );
    }

    out.push('\n');
    out
}
