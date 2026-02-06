//! Oracle command implementation (PMAT-244..247)
//!
//! Model Oracle: identifies model family, size variant, constraints, and contract
//! compliance from local files, HuggingFace URIs, or contract descriptions.
//!
//! Toyota Way: Genchi Genbutsu - Go to the source to understand the model.

use crate::error::CliError;
use crate::output;
use aprender::format::model_family::{
    FamilyRegistry, ModelFamily, ModelFamilyConfig, ModelSizeConfig,
};
use aprender::format::model_family_loader::load_family_registry;
use aprender::format::rosetta::RosettaStone;
use serde::Serialize;
use std::path::{Path, PathBuf};

// ============================================================================
// Report Types (spec §6.5)
// ============================================================================

/// Analysis mode for the oracle
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OracleMode {
    /// Local file analysis (PMAT-244)
    Local,
    /// HuggingFace API query (PMAT-245)
    HuggingFace,
    /// Contract description (PMAT-246)
    Family,
}

/// Complete oracle report for a model (spec §6.5)
#[derive(Debug, Clone, Serialize)]
pub struct ModelOracleReport {
    /// Source path, HF URI, or family name
    pub source: String,
    /// Analysis mode
    pub mode: OracleMode,
    /// Detected or specified model family
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<FamilyInfo>,
    /// Detected size variant
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_variant: Option<SizeVariantInfo>,
    /// Format information (for local files)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<FormatInfo>,
    /// Contract compliance result
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compliance: Option<ComplianceResult>,
    /// Certification status
    #[serde(skip_serializing_if = "Option::is_none")]
    pub certification: Option<CertificationInfo>,
    /// Tensor list (for --tensors flag)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensors: Option<Vec<TensorComplianceEntry>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FamilyInfo {
    pub name: String,
    pub display_name: String,
    pub vendor: String,
    pub architectures: Vec<String>,
    pub constraints: ConstraintsSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template_format: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConstraintsSummary {
    pub attention: String,
    pub activation: String,
    pub norm: String,
    pub bias: bool,
    pub tied_embeddings: bool,
    pub mlp: String,
    pub positional_encoding: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SizeVariantInfo {
    pub name: String,
    pub parameters: String,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub intermediate_dim: usize,
    pub vocab_size: usize,
    pub expected_tensor_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct FormatInfo {
    pub format_type: String,
    pub file_size: usize,
    pub tensor_count: usize,
    pub total_params: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ComplianceResult {
    pub is_compliant: bool,
    pub tensor_count_match: bool,
    pub missing_tensors: Vec<String>,
    pub unexpected_tensors: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CertificationInfo {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub playbook_path: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TensorComplianceEntry {
    pub name: String,
    pub present: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shape: Option<Vec<usize>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

// ============================================================================
// Command Entry Point
// ============================================================================

/// Run the oracle command.
///
/// Dispatches to the appropriate mode based on arguments:
/// - Local file: `apr oracle model.gguf`
/// - HuggingFace: `apr oracle hf://org/repo`
/// - Family: `apr oracle --family qwen2`
#[allow(clippy::fn_params_excessive_bools)]
pub(crate) fn run(
    source: Option<&String>,
    family_name: Option<&String>,
    size_filter: Option<&String>,
    show_compliance: bool,
    show_tensors: bool,
    json_output: bool,
    verbose: bool,
    offline: bool,
) -> Result<(), CliError> {
    // Mode 3: Contract description (--family)
    if let Some(family) = family_name {
        return run_family_mode(family, size_filter.map(String::as_str), json_output, verbose);
    }

    // Require a source for modes 1 and 2
    let source = source.ok_or_else(|| {
        CliError::InvalidFormat(
            "Either <SOURCE> or --family is required. Usage: apr oracle <FILE|hf://...> or apr oracle --family <NAME>".to_string(),
        )
    })?;

    // Mode 2: HuggingFace API query
    if source.starts_with("hf://") || source.starts_with("huggingface://") {
        if offline {
            return Err(CliError::NetworkError(
                "Cannot query HuggingFace API in --offline mode".to_string(),
            ));
        }
        return run_hf_mode(source, json_output, verbose);
    }

    // Mode 1: Local file analysis
    let path = PathBuf::from(source);
    run_local_mode(&path, show_compliance, show_tensors, json_output, verbose)
}

// ============================================================================
// Mode 1: Local File Analysis (PMAT-244)
// ============================================================================

fn run_local_mode(
    path: &Path,
    show_compliance: bool,
    show_tensors: bool,
    json_output: bool,
    verbose: bool,
) -> Result<(), CliError> {
    // Validate file exists
    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }
    if !path.is_file() {
        return Err(CliError::NotAFile(path.to_path_buf()));
    }

    // Inspect using RosettaStone
    let rosetta = RosettaStone::new();
    let report = rosetta.inspect(path).map_err(|e| {
        CliError::InvalidFormat(format!("Failed to inspect {}: {e}", path.display()))
    })?;

    // Load family registry
    let registry = load_registry()?;

    // Collect tensor names for family detection
    let tensor_names: Vec<&str> = report.tensors.iter().map(|t| t.name.as_str()).collect();

    // Detect family
    let detected_family = registry.detect_family(&tensor_names);

    // Build format info
    let format_info = FormatInfo {
        format_type: format!("{}", report.format),
        file_size: report.file_size,
        tensor_count: report.tensors.len(),
        total_params: report.total_params,
        quantization: report.quantization.clone(),
        architecture: report.architecture.clone(),
    };

    // Build family info and size variant
    let (family_info, size_variant_info) = if let Some(family) = detected_family {
        let config = family.config();
        let fi = build_family_info(config);

        // Detect size from metadata
        let size_info = detect_size_from_inspection(family, &report);

        (Some(fi), size_info)
    } else {
        (None, None)
    };

    // Build compliance result
    let compliance = if show_compliance {
        detected_family.map(|family| {
            build_compliance(
                family,
                &tensor_names,
                size_variant_info.as_ref().map(|s| s.name.as_str()),
            )
        })
    } else {
        None
    };

    // Build tensor list
    let tensors_list = if show_tensors {
        Some(build_tensor_list(&report, detected_family))
    } else {
        None
    };

    // Build certification info (PMAT-247)
    let certification = detected_family.and_then(|family| {
        build_certification(
            family.config(),
            size_variant_info.as_ref().map(|s| s.name.as_str()),
        )
    });

    let oracle_report = ModelOracleReport {
        source: path.display().to_string(),
        mode: OracleMode::Local,
        family: family_info,
        size_variant: size_variant_info,
        format: Some(format_info),
        compliance,
        certification,
        tensors: tensors_list,
    };

    if json_output {
        output_json(&oracle_report)?;
    } else {
        output_text(&oracle_report, verbose);
    }

    Ok(())
}

// ============================================================================
// Mode 2: HuggingFace API Query (PMAT-245)
// ============================================================================

fn run_hf_mode(source: &str, json_output: bool, verbose: bool) -> Result<(), CliError> {
    // Parse HF URI → org/repo
    let repo = source
        .strip_prefix("hf://")
        .or_else(|| source.strip_prefix("huggingface://"))
        .ok_or_else(|| CliError::InvalidFormat(format!("Invalid HF URI: {source}")))?;

    // Fetch config.json from HuggingFace
    let config_url = format!("https://huggingface.co/{repo}/raw/main/config.json");
    let config_json = fetch_hf_config(&config_url)?;

    // Load family registry
    let registry = load_registry()?;

    // Extract model_type from config.json
    let model_type = extract_json_string(&config_json, "model_type");
    let hidden_size = extract_json_usize(&config_json, "hidden_size");
    let num_layers = extract_json_usize(&config_json, "num_hidden_layers");

    // Detect family from model_type
    let detected_family = model_type
        .as_deref()
        .and_then(|mt| registry.detect_from_model_type(mt));

    // Build family info and size variant
    let (family_info, size_variant_info) = if let Some(family) = detected_family {
        let config = family.config();
        let fi = build_family_info(config);

        // Detect size from config.json values
        let size_info = match (hidden_size, num_layers) {
            (Some(h), Some(l)) => family.detect_size(h, l).and_then(|size_name| {
                family.size_config(&size_name).map(|sc| build_size_info(&size_name, sc, family))
            }),
            _ => None,
        };

        (Some(fi), size_info)
    } else {
        (None, None)
    };

    // Build certification info
    let certification = detected_family.and_then(|family| {
        build_certification(
            family.config(),
            size_variant_info.as_ref().map(|s| s.name.as_str()),
        )
    });

    let oracle_report = ModelOracleReport {
        source: source.to_string(),
        mode: OracleMode::HuggingFace,
        family: family_info,
        size_variant: size_variant_info,
        format: None,
        compliance: None,
        certification,
        tensors: None,
    };

    if json_output {
        output_json(&oracle_report)?;
    } else {
        output_text(&oracle_report, verbose);
    }

    Ok(())
}

// ============================================================================
// Mode 3: Contract Description (PMAT-246)
// ============================================================================

fn run_family_mode(
    family_name: &str,
    size_filter: Option<&str>,
    json_output: bool,
    verbose: bool,
) -> Result<(), CliError> {
    let registry = load_registry()?;
    let family = registry.detect_from_model_type(family_name).ok_or_else(|| {
        // List known families for helpful error
        let known: Vec<String> = (0..registry.len())
            .filter_map(|_| None::<String>) // We can't iterate, so list empty
            .collect();
        let _ = known; // avoid unused
        CliError::InvalidFormat(format!(
            "Unknown model family: '{family_name}'. Use tensor names or HF model_type."
        ))
    })?;

    let config = family.config();

    if json_output {
        // Build a report with family details
        let fi = build_family_info(config);

        // If size filter, include that variant only
        let size_info = size_filter.and_then(|size| {
            family
                .size_config(size)
                .map(|sc| build_size_info(size, sc, family))
        });

        let report = ModelOracleReport {
            source: family_name.to_string(),
            mode: OracleMode::Family,
            family: Some(fi),
            size_variant: size_info,
            format: None,
            compliance: None,
            certification: build_certification(config, size_filter),
            tensors: None,
        };
        output_json(&report)?;
    } else {
        output_family_description(config, size_filter, verbose);
    }

    Ok(())
}

// ============================================================================
// Registry Loading
// ============================================================================

/// Load the family registry from contracts/ directory.
/// Searches for contracts relative to the executable, then falls back to CWD.
fn load_registry() -> Result<FamilyRegistry, CliError> {
    // Try contracts/ relative to CWD first
    let cwd_contracts = PathBuf::from("contracts");
    if cwd_contracts.join("model-families").exists() {
        return load_family_registry(&cwd_contracts)
            .map_err(|e| CliError::Aprender(format!("Failed to load family contracts: {e}")));
    }

    // Try relative to executable
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let exe_contracts = exe_dir.join("contracts");
            if exe_contracts.join("model-families").exists() {
                return load_family_registry(&exe_contracts).map_err(|e| {
                    CliError::Aprender(format!("Failed to load family contracts: {e}"))
                });
            }
            // Try one level up (for cargo run scenarios)
            if let Some(parent) = exe_dir.parent() {
                let parent_contracts = parent.join("contracts");
                if parent_contracts.join("model-families").exists() {
                    return load_family_registry(&parent_contracts).map_err(|e| {
                        CliError::Aprender(format!("Failed to load family contracts: {e}"))
                    });
                }
            }
        }
    }

    // Try workspace root (common in dev)
    for ancestor in [".", "..", "../..", "../../.."] {
        let contracts = PathBuf::from(ancestor).join("contracts");
        if contracts.join("model-families").exists() {
            return load_family_registry(&contracts)
                .map_err(|e| CliError::Aprender(format!("Failed to load family contracts: {e}")));
        }
    }

    // Return empty registry rather than error — graceful degradation
    Ok(FamilyRegistry::new())
}

// ============================================================================
// Helper Functions
// ============================================================================

fn build_family_info(config: &ModelFamilyConfig) -> FamilyInfo {
    let constraints = &config.constraints;
    FamilyInfo {
        name: config.family.clone(),
        display_name: config.display_name.clone(),
        vendor: config.vendor.clone(),
        architectures: config.architectures.clone(),
        constraints: ConstraintsSummary {
            attention: format!("{}", constraints.attention_type),
            activation: format!("{}", constraints.activation),
            norm: format!("{}", constraints.norm_type),
            bias: constraints.has_bias,
            mlp: format!("{}", constraints.mlp_type),
            tied_embeddings: constraints.tied_embeddings,
            positional_encoding: format!("{}", constraints.positional_encoding),
        },
        chat_template_format: config.chat_template.as_ref().map(|ct| ct.format.clone()),
    }
}

fn build_size_info(
    size_name: &str,
    sc: &ModelSizeConfig,
    family: &dyn ModelFamily,
) -> SizeVariantInfo {
    SizeVariantInfo {
        name: size_name.to_string(),
        parameters: sc.parameters.clone(),
        hidden_dim: sc.hidden_dim,
        num_layers: sc.num_layers,
        num_heads: sc.num_heads,
        num_kv_heads: sc.num_kv_heads,
        intermediate_dim: sc.intermediate_dim,
        vocab_size: sc.vocab_size,
        expected_tensor_count: family
            .expected_tensor_count(size_name)
            .unwrap_or(0),
    }
}

fn detect_size_from_inspection(
    family: &dyn ModelFamily,
    report: &aprender::format::rosetta::InspectionReport,
) -> Option<SizeVariantInfo> {
    // Try to extract hidden_dim from metadata
    let hidden_dim = report
        .metadata
        .get("hidden_size")
        .or_else(|| report.metadata.get("embedding_length"))
        .or_else(|| report.metadata.get("hidden_dim"))
        .and_then(|v| v.parse::<usize>().ok());

    let num_layers = report
        .metadata
        .get("num_hidden_layers")
        .or_else(|| report.metadata.get("block_count"))
        .or_else(|| report.metadata.get("num_layers"))
        .and_then(|v| v.parse::<usize>().ok());

    // Also try to infer from tensor shapes (embedding shape[1] = hidden_dim)
    let hidden_from_tensors = report.tensors.iter().find_map(|t| {
        if t.name.contains("embed") && t.shape.len() == 2 {
            Some(t.shape[1])
        } else {
            None
        }
    });

    let layers_from_tensors = {
        let mut max_layer: Option<usize> = None;
        for t in &report.tensors {
            // Match patterns like "layers.N." or "blk.N."
            if let Some(rest) = t
                .name
                .strip_prefix("model.layers.")
                .or_else(|| t.name.strip_prefix("blk."))
                .or_else(|| t.name.strip_prefix("encoder.layers."))
            {
                if let Some(dot_pos) = rest.find('.') {
                    if let Ok(n) = rest[..dot_pos].parse::<usize>() {
                        max_layer = Some(max_layer.map_or(n, |m: usize| m.max(n)));
                    }
                }
            }
        }
        max_layer.map(|m| m + 1) // 0-indexed → count
    };

    let h = hidden_dim.or(hidden_from_tensors)?;
    let l = num_layers.or(layers_from_tensors)?;

    let size_name = family.detect_size(h, l)?;
    let sc = family.size_config(&size_name)?;
    Some(build_size_info(&size_name, sc, family))
}

fn build_compliance(
    family: &dyn ModelFamily,
    tensor_names: &[&str],
    size_name: Option<&str>,
) -> ComplianceResult {
    let size = size_name.unwrap_or("unknown");

    // Get expected tensors from template
    let config = family.config();
    let expected = expand_tensor_template(&config.tensor_template, config, size);

    let actual_set: std::collections::HashSet<&str> = tensor_names.iter().copied().collect();
    let expected_set: std::collections::HashSet<&str> = expected.iter().map(String::as_str).collect();

    let missing: Vec<String> = expected_set
        .difference(&actual_set)
        .map(|s| (*s).to_string())
        .collect();

    let unexpected: Vec<String> = actual_set
        .difference(&expected_set)
        .map(|s| (*s).to_string())
        .collect();

    let expected_count = family
        .expected_tensor_count(size)
        .unwrap_or(expected.len());
    let tensor_count_match = tensor_names.len() == expected_count;

    ComplianceResult {
        is_compliant: missing.is_empty() && tensor_count_match,
        tensor_count_match,
        missing_tensors: missing,
        unexpected_tensors: unexpected,
    }
}

fn expand_tensor_template(
    template: &aprender::format::model_family::TensorTemplate,
    config: &ModelFamilyConfig,
    size_name: &str,
) -> Vec<String> {
    let mut names = Vec::new();

    // Global tensors
    if !template.embedding.is_empty() {
        names.push(template.embedding.clone());
    }
    if let Some(ref lm_head) = template.lm_head {
        names.push(lm_head.clone());
    }
    if let Some(ref final_norm) = template.final_norm {
        names.push(final_norm.clone());
    }

    // Per-layer tensors
    let num_layers = config
        .size_variants
        .get(size_name)
        .map_or(0, |sc| sc.num_layers);

    for layer_idx in 0..num_layers {
        for pat in template.per_layer.values().flatten() {
            names.push(pat.replace("{n}", &layer_idx.to_string()));
        }
    }

    names
}

fn build_tensor_list(
    report: &aprender::format::rosetta::InspectionReport,
    _family: Option<&dyn ModelFamily>,
) -> Vec<TensorComplianceEntry> {
    report
        .tensors
        .iter()
        .map(|t| TensorComplianceEntry {
            name: t.name.clone(),
            present: true,
            dtype: Some(t.dtype.clone()),
            shape: Some(t.shape.clone()),
            note: None,
        })
        .collect()
}

fn build_certification(
    config: &ModelFamilyConfig,
    size_name: Option<&str>,
) -> Option<CertificationInfo> {
    let cert = config.certification.as_ref()?;

    let playbook = size_name.map(|size| {
        cert.playbook_path.replace("{size}", size)
    });

    Some(CertificationInfo {
        status: "PENDING".to_string(),
        playbook_path: playbook,
    })
}

/// Minimal JSON extraction from config.json (avoid serde_json dep at this level).
/// Looks for `"key": "value"` or `"key": number` patterns.
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\"");
    let idx = json.find(&pattern)?;
    let rest = &json[idx + pattern.len()..];

    // Skip whitespace and colon
    let rest = rest.trim_start();
    let rest = rest.strip_prefix(':')?;
    let rest = rest.trim_start();

    if let Some(rest) = rest.strip_prefix('"') {
        // String value
        let end = rest.find('"')?;
        Some(rest[..end].to_string())
    } else {
        None
    }
}

fn extract_json_usize(json: &str, key: &str) -> Option<usize> {
    let pattern = format!("\"{key}\"");
    let idx = json.find(&pattern)?;
    let rest = &json[idx + pattern.len()..];

    let rest = rest.trim_start();
    let rest = rest.strip_prefix(':')?;
    let rest = rest.trim_start();

    // Extract number
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Fetch a URL and return the body as a string.
fn fetch_hf_config(url: &str) -> Result<String, CliError> {
    // Use ureq for simple HTTP GET (already in workspace deps for HF API access)
    // Fallback: try reading with std::process::Command and curl
    let output = std::process::Command::new("curl")
        .args(["-sL", "--fail", "--max-time", "30", url])
        .output()
        .map_err(|e| CliError::NetworkError(format!("Failed to fetch {url}: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(CliError::NetworkError(format!(
            "Failed to fetch {url}: HTTP error. {stderr}"
        )));
    }

    String::from_utf8(output.stdout)
        .map_err(|e| CliError::NetworkError(format!("Invalid UTF-8 from {url}: {e}")))
}

// ============================================================================
// Output Formatting
// ============================================================================

fn output_json(report: &ModelOracleReport) -> Result<(), CliError> {
    let json = serde_json::to_string_pretty(report)
        .map_err(|e| CliError::Aprender(format!("JSON serialization failed: {e}")))?;
    println!("{json}");
    Ok(())
}

fn output_text(report: &ModelOracleReport, verbose: bool) {
    output::section("Model Oracle Report");

    output::kv("Source", &report.source);
    output::kv("Mode", format!("{:?}", report.mode));

    // Format info
    if let Some(ref fmt) = report.format {
        output::kv("Format", &fmt.format_type);
        output::kv("File Size", output::format_size(fmt.file_size as u64));
        output::kv("Tensors", fmt.tensor_count);
        output::kv("Parameters", format_params(fmt.total_params));
        if let Some(ref q) = fmt.quantization {
            output::kv("Quantization", q);
        }
        if let Some(ref arch) = fmt.architecture {
            output::kv("Architecture", arch);
        }
    }

    // Family info
    if let Some(ref family) = report.family {
        println!();
        output::kv(
            "Family",
            format!("{} ({})", family.name, family.display_name),
        );
        output::kv("Vendor", &family.vendor);

        if verbose {
            output::kv("Architectures", family.architectures.join(", "));
        }

        if let Some(ref ct) = family.chat_template_format {
            output::kv("Chat Template", ct);
        }
    } else {
        println!();
        output::kv("Family", "UNKNOWN (no matching contract)");
    }

    // Size variant
    if let Some(ref size) = report.size_variant {
        println!();
        output::kv(
            "Size",
            format!(
                "{} (hidden={}, layers={}, heads={}, kv_heads={})",
                size.parameters, size.hidden_dim, size.num_layers, size.num_heads, size.num_kv_heads
            ),
        );
        output::kv("Intermediate Dim", size.intermediate_dim);
        output::kv("Vocab Size", size.vocab_size);
        output::kv("Expected Tensors", size.expected_tensor_count);
    }

    // Constraints
    if let Some(ref family) = report.family {
        println!();
        println!("  Constraints:");
        let c = &family.constraints;
        println!(
            "    Attention: {} | Activation: {} | Norm: {}",
            c.attention, c.activation, c.norm
        );
        println!(
            "    Bias: {} | Tied: {} | MLP: {} | Position: {}",
            if c.bias { "yes" } else { "no" },
            if c.tied_embeddings { "yes" } else { "no" },
            c.mlp,
            c.positional_encoding
        );
    }

    // Compliance
    if let Some(ref compliance) = report.compliance {
        println!();
        if compliance.is_compliant {
            output::kv("Contract", "COMPLIANT");
        } else {
            output::kv("Contract", "NON-COMPLIANT");
            if !compliance.tensor_count_match {
                output::kv("Tensor Count", "MISMATCH");
            }
            if !compliance.missing_tensors.is_empty() {
                output::kv(
                    "Missing Tensors",
                    format!("{} tensor(s)", compliance.missing_tensors.len()),
                );
                if verbose {
                    for t in &compliance.missing_tensors {
                        println!("    - {t}");
                    }
                }
            }
            if !compliance.unexpected_tensors.is_empty() && verbose {
                output::kv(
                    "Unexpected Tensors",
                    format!("{} tensor(s)", compliance.unexpected_tensors.len()),
                );
                for t in &compliance.unexpected_tensors {
                    println!("    + {t}");
                }
            }
        }
    }

    // Certification
    if let Some(ref cert) = report.certification {
        println!();
        output::kv("Certification", &cert.status);
        if let Some(ref pb) = cert.playbook_path {
            output::kv("Playbook", pb);
        }
    }

    // Tensors
    if let Some(ref tensors) = report.tensors {
        println!();
        println!("  Tensors ({} total):", tensors.len());
        let max_show = if verbose { tensors.len() } else { 20 };
        for (i, t) in tensors.iter().enumerate() {
            if i >= max_show && i < tensors.len() - 2 {
                if i == max_show {
                    println!("    ... ({} more) ...", tensors.len() - max_show - 2);
                }
                continue;
            }
            let shape_str = t
                .shape
                .as_ref()
                .map(|s| format!("{s:?}"))
                .unwrap_or_default();
            let dtype_str = t.dtype.as_deref().unwrap_or("");
            println!("    {:<50} {} {}", t.name, dtype_str, shape_str);
        }
    }
}

fn output_family_description(
    config: &ModelFamilyConfig,
    size_filter: Option<&str>,
    verbose: bool,
) {
    output::section(&format!("{} Family Contract", config.display_name));

    output::kv("Family", &config.family);
    output::kv("Vendor", &config.vendor);
    output::kv("Architectures", config.architectures.join(", "));
    output::kv("HF Pattern", &config.hf_pattern);

    // Constraints
    let c = &config.constraints;
    println!();
    println!("  Constraints:");
    println!(
        "    Attention: {} | Activation: {} | Norm: {}",
        c.attention_type, c.activation, c.norm_type
    );
    println!(
        "    Bias: {} | Tied: {} | MLP: {} | Position: {}",
        if c.has_bias { "yes" } else { "no" },
        if c.tied_embeddings { "yes" } else { "no" },
        c.mlp_type,
        c.positional_encoding
    );

    // Size variants
    let variants: Vec<(&String, &ModelSizeConfig)> = if let Some(size) = size_filter {
        config
            .size_variants
            .iter()
            .filter(|(k, _)| k.as_str() == size)
            .collect()
    } else {
        let mut v: Vec<_> = config.size_variants.iter().collect();
        v.sort_by_key(|(_, sc)| sc.hidden_dim);
        v
    };

    for (name, sc) in &variants {
        println!();
        output::kv("Size Variant", format!("{name} ({})", sc.parameters));
        output::kv("  hidden_dim", sc.hidden_dim);
        output::kv("  num_layers", sc.num_layers);
        output::kv("  num_heads", sc.num_heads);
        output::kv("  num_kv_heads", sc.num_kv_heads);
        output::kv("  intermediate_dim", sc.intermediate_dim);
        output::kv("  vocab_size", sc.vocab_size);
        output::kv("  head_dim", sc.head_dim);
        if sc.rope_theta > 0.0 {
            output::kv("  rope_theta", sc.rope_theta);
        }
        output::kv("  norm_eps", sc.norm_eps);
    }

    if size_filter.is_some() && variants.is_empty() {
        println!();
        output::kv("Error", format!("Size '{}' not found", size_filter.unwrap_or("")));
        let available: Vec<&String> = config.size_variants.keys().collect();
        output::kv("Available", format!("{available:?}"));
    }

    // Tensor template
    if verbose {
        println!();
        println!("  Tensor Template:");
        println!("    Embedding: {}", config.tensor_template.embedding);
        if let Some(ref lm) = config.tensor_template.lm_head {
            println!("    LM Head: {lm}");
        }
        if let Some(ref norm) = config.tensor_template.final_norm {
            println!("    Final Norm: {norm}");
        }
        println!("    Per-layer:");
        for (role, pattern) in &config.tensor_template.per_layer {
            if let Some(pat) = pattern {
                println!("      {role}: {pat}");
            }
        }
    }

    // Quantizations
    if !config.quantizations.is_empty() {
        println!();
        output::kv("Quantizations", config.quantizations.join(", "));
    }

    // Chat template
    if let Some(ref ct) = config.chat_template {
        println!();
        output::kv("Chat Template", &ct.format);
        output::kv("  BOS", &ct.bos_token);
        output::kv("  EOS", &ct.eos_token);
    }
}

fn format_params(params: usize) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1_000_000_000.0)
    } else if params >= 1_000_000 {
        format!("{:.1}M", params as f64 / 1_000_000.0)
    } else if params >= 1_000 {
        format!("{:.1}K", params as f64 / 1_000.0)
    } else {
        format!("{params}")
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oracle_mode_serialize() {
        let mode = OracleMode::Local;
        let json = serde_json::to_string(&mode).expect("serialize mode");
        assert_eq!(json, "\"local\"");

        let mode = OracleMode::HuggingFace;
        let json = serde_json::to_string(&mode).expect("serialize mode");
        assert_eq!(json, "\"hugging_face\"");

        let mode = OracleMode::Family;
        let json = serde_json::to_string(&mode).expect("serialize mode");
        assert_eq!(json, "\"family\"");
    }

    #[test]
    fn test_format_params() {
        assert_eq!(format_params(500), "500");
        assert_eq!(format_params(1_500), "1.5K");
        assert_eq!(format_params(1_500_000), "1.5M");
        assert_eq!(format_params(1_500_000_000), "1.5B");
        assert_eq!(format_params(7_000_000_000), "7.0B");
    }

    #[test]
    fn test_extract_json_string() {
        let json = r#"{"model_type": "qwen2", "hidden_size": 1536}"#;
        assert_eq!(
            extract_json_string(json, "model_type"),
            Some("qwen2".to_string())
        );
        assert_eq!(extract_json_string(json, "nonexistent"), None);
    }

    #[test]
    fn test_extract_json_usize() {
        let json = r#"{"model_type": "qwen2", "hidden_size": 1536, "num_hidden_layers": 28}"#;
        assert_eq!(extract_json_usize(json, "hidden_size"), Some(1536));
        assert_eq!(extract_json_usize(json, "num_hidden_layers"), Some(28));
        assert_eq!(extract_json_usize(json, "nonexistent"), None);
    }

    #[test]
    fn test_family_info_build() {
        use aprender::format::model_family::*;
        use std::collections::HashMap;

        let config = ModelFamilyConfig {
            family: "test".to_string(),
            display_name: "Test Model".to_string(),
            vendor: "TestCo".to_string(),
            architectures: vec!["TestForCausalLM".to_string()],
            hf_pattern: "test/*".to_string(),
            size_variants: HashMap::new(),
            constraints: ModelConstraints {
                attention_type: AttentionType::Gqa,
                activation: Activation::Silu,
                norm_type: NormType::RmsNorm,
                has_bias: true,
                tied_embeddings: false,
                positional_encoding: PositionalEncoding::Rope,
                mlp_type: MlpType::SwiGlu,
            },
            tensor_template: TensorTemplate {
                embedding: "embed.weight".to_string(),
                lm_head: None,
                final_norm: None,
                per_layer: HashMap::new(),
            },
            shape_template: ShapeTemplate {
                shapes: HashMap::new(),
            },
            quantizations: vec!["q4_k_m".to_string()],
            chat_template: None,
            certification: None,
        };

        let fi = build_family_info(&config);
        assert_eq!(fi.name, "test");
        assert_eq!(fi.vendor, "TestCo");
        assert_eq!(fi.constraints.attention, "GQA");
        assert!(fi.constraints.bias);
    }

    #[test]
    fn test_compliance_result_serialize() {
        let cr = ComplianceResult {
            is_compliant: true,
            tensor_count_match: true,
            missing_tensors: vec![],
            unexpected_tensors: vec![],
        };
        let json = serde_json::to_string(&cr).expect("serialize");
        assert!(json.contains("\"is_compliant\":true"));
    }

    #[test]
    fn test_report_json_roundtrip() {
        let report = ModelOracleReport {
            source: "test.gguf".to_string(),
            mode: OracleMode::Local,
            family: Some(FamilyInfo {
                name: "qwen2".to_string(),
                display_name: "Qwen2".to_string(),
                vendor: "Alibaba".to_string(),
                architectures: vec!["Qwen2ForCausalLM".to_string()],
                constraints: ConstraintsSummary {
                    attention: "GQA".to_string(),
                    activation: "SiLU".to_string(),
                    norm: "RMSNorm".to_string(),
                    bias: true,
                    tied_embeddings: false,
                    mlp: "SwiGLU".to_string(),
                    positional_encoding: "RoPE".to_string(),
                },
                chat_template_format: Some("chatml".to_string()),
            }),
            size_variant: Some(SizeVariantInfo {
                name: "1.5b".to_string(),
                parameters: "1.5B".to_string(),
                hidden_dim: 1536,
                num_layers: 28,
                num_heads: 12,
                num_kv_heads: 2,
                intermediate_dim: 8960,
                vocab_size: 151936,
                expected_tensor_count: 339,
            }),
            format: Some(FormatInfo {
                format_type: "GGUF".to_string(),
                file_size: 1_000_000,
                tensor_count: 339,
                total_params: 1_500_000_000,
                quantization: Some("Q4_K_M".to_string()),
                architecture: Some("qwen2".to_string()),
            }),
            compliance: None,
            certification: None,
            tensors: None,
        };

        let json = serde_json::to_string_pretty(&report).expect("serialize");
        assert!(json.contains("\"source\": \"test.gguf\""));
        assert!(json.contains("\"mode\": \"local\""));
        assert!(json.contains("\"family\""));
        assert!(json.contains("\"hidden_dim\": 1536"));
    }

    #[test]
    fn test_certification_build_with_size() {
        use aprender::format::model_family::CertificationConfig;
        use std::collections::HashMap;

        let mut size_cats = HashMap::new();
        size_cats.insert("1.5b".to_string(), "small".to_string());

        let config = ModelFamilyConfig {
            family: "qwen2".to_string(),
            display_name: "Qwen2".to_string(),
            vendor: "Alibaba".to_string(),
            architectures: vec![],
            hf_pattern: String::new(),
            size_variants: HashMap::new(),
            constraints: aprender::format::model_family::ModelConstraints {
                attention_type: aprender::format::model_family::AttentionType::Gqa,
                activation: aprender::format::model_family::Activation::Silu,
                norm_type: aprender::format::model_family::NormType::RmsNorm,
                has_bias: false,
                tied_embeddings: false,
                positional_encoding: aprender::format::model_family::PositionalEncoding::Rope,
                mlp_type: aprender::format::model_family::MlpType::SwiGlu,
            },
            tensor_template: aprender::format::model_family::TensorTemplate {
                embedding: String::new(),
                lm_head: None,
                final_norm: None,
                per_layer: HashMap::new(),
            },
            shape_template: aprender::format::model_family::ShapeTemplate {
                shapes: HashMap::new(),
            },
            quantizations: vec![],
            chat_template: None,
            certification: Some(CertificationConfig {
                playbook_path: "../playbooks/{size}.yaml".to_string(),
                csv_family_key: "qwen2".to_string(),
                size_categories: size_cats,
            }),
        };

        let cert = build_certification(&config, Some("1.5b"));
        assert!(cert.is_some());
        let cert = cert.expect("cert exists");
        assert_eq!(cert.status, "PENDING");
        assert_eq!(
            cert.playbook_path,
            Some("../playbooks/1.5b.yaml".to_string())
        );
    }

    #[test]
    fn test_source_required_error() {
        let result = run(None, None, None, false, false, false, false, false);
        assert!(result.is_err());
        match result {
            Err(CliError::InvalidFormat(msg)) => {
                assert!(msg.contains("required"));
            }
            other => panic!("Expected InvalidFormat, got: {other:?}"),
        }
    }

    #[test]
    fn test_file_not_found() {
        let src = "/nonexistent/model.gguf".to_string();
        let result = run(
            Some(&src),
            None,
            None,
            false,
            false,
            false,
            false,
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            other => panic!("Expected FileNotFound, got: {other:?}"),
        }
    }

    #[test]
    fn test_offline_hf_rejected() {
        let src = "hf://Qwen/Qwen2.5-Coder-1.5B".to_string();
        let result = run(
            Some(&src),
            None,
            None,
            false,
            false,
            false,
            false,
            true, // offline
        );
        assert!(result.is_err());
        match result {
            Err(CliError::NetworkError(msg)) => {
                assert!(msg.contains("offline"));
            }
            other => panic!("Expected NetworkError, got: {other:?}"),
        }
    }

    #[test]
    fn test_load_registry_graceful_degradation() {
        // Should not error even if contracts dir doesn't exist nearby
        let registry = load_registry();
        // This should succeed (might be empty or populated depending on CWD)
        assert!(registry.is_ok());
    }

    #[test]
    fn test_tensor_compliance_entry_serialize() {
        let entry = TensorComplianceEntry {
            name: "model.embed_tokens.weight".to_string(),
            present: true,
            dtype: Some("F16".to_string()),
            shape: Some(vec![151936, 1536]),
            note: None,
        };
        let json = serde_json::to_string(&entry).expect("serialize");
        assert!(json.contains("embed_tokens"));
        assert!(!json.contains("note")); // skip_serializing_if
    }
}
