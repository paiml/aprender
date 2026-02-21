
// ============================================================================
// Mode 3: Contract Description (PMAT-246)
// ============================================================================

fn run_family_mode(
    family_name: &str,
    size_filter: Option<&str>,
    json_output: bool,
    verbose: bool,
    flags: OracleFlags,
) -> Result<(), CliError> {
    let registry = load_registry()?;
    let family = registry
        .detect_from_model_type(family_name)
        .ok_or_else(|| {
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

        // Build enhanced sections
        let (stats, explanation, kernel_compat) =
            build_enhanced_sections(Some(family), size_filter, flags);

        let report = ModelOracleReport {
            source: family_name.to_string(),
            mode: OracleMode::Family,
            family: Some(fi),
            size_variant: size_info,
            format: None,
            compliance: None,
            certification: build_certification(config, size_filter),
            tensors: None,
            stats,
            explanation,
            kernel_compatibility: kernel_compat,
            cross_validation: None,
            hf_data: None,
        };
        output_json(&report)?;
    } else {
        output_family_description(config, size_filter, verbose, flags, family);
    }

    Ok(())
}

// ============================================================================
// Enhanced Sections Builder
// ============================================================================

fn build_enhanced_sections(
    family: Option<&dyn ModelFamily>,
    size_name: Option<&str>,
    flags: OracleFlags,
) -> (
    Option<StatisticalAnalysis>,
    Option<ArchitectureExplanation>,
    Option<KernelCompatibility>,
) {
    let Some(family) = family else {
        return (None, None, None);
    };

    let Some(size_name) = size_name else {
        return (None, None, None);
    };

    let Some(size) = family.size_config(size_name) else {
        return (None, None, None);
    };

    let constraints = family.constraints();

    let stats = if flags.show_stats() || flags.show_explain() || flags.show_kernels() {
        Some(build_statistical_analysis(size, constraints))
    } else {
        None
    };

    let explanation = if flags.show_explain() {
        stats
            .as_ref()
            .map(|s| build_architecture_explanation(size, constraints, s))
    } else {
        None
    };

    let kernel_compat = if flags.show_kernels() {
        stats
            .as_ref()
            .map(|s| build_kernel_compatibility(size, constraints, s))
    } else {
        None
    };

    // Only return stats if explicitly requested
    let stats = if flags.show_stats() { stats } else { None };

    (stats, explanation, kernel_compat)
}

// ============================================================================
// Registry Loading
// ============================================================================

/// Load the family registry from contracts/ directory.
/// Searches for contracts relative to the executable, then falls back to CWD.
fn contracts_candidate_paths() -> Vec<PathBuf> {
    let mut candidates = vec![PathBuf::from("contracts")];

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            candidates.push(exe_dir.join("contracts"));
            if let Some(parent) = exe_dir.parent() {
                candidates.push(parent.join("contracts"));
            }
        }
    }

    for ancestor in [".", "..", "../..", "../../.."] {
        candidates.push(PathBuf::from(ancestor).join("contracts"));
    }

    candidates
}

fn load_registry() -> Result<FamilyRegistry, CliError> {
    for candidate in contracts_candidate_paths() {
        if candidate.join("model-families").exists() {
            return load_family_registry(&candidate)
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
        expected_tensor_count: family.expected_tensor_count(size_name).unwrap_or(0),
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
    let expected_set: std::collections::HashSet<&str> =
        expected.iter().map(String::as_str).collect();

    let missing: Vec<String> = expected_set
        .difference(&actual_set)
        .map(|s| (*s).to_string())
        .collect();

    let unexpected: Vec<String> = actual_set
        .difference(&expected_set)
        .map(|s| (*s).to_string())
        .collect();

    let expected_count = family.expected_tensor_count(size).unwrap_or(expected.len());
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

    let playbook = size_name.map(|size| cert.playbook_path.replace("{size}", size));

    Some(CertificationInfo {
        status: "PENDING".to_string(),
        playbook_path: playbook,
    })
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
// Output Formatting
// ============================================================================

fn output_json(report: &ModelOracleReport) -> Result<(), CliError> {
    let json = serde_json::to_string_pretty(report)
        .map_err(|e| CliError::Aprender(format!("JSON serialization failed: {e}")))?;
    println!("{json}");
    Ok(())
}

fn format_text_report(report: &ModelOracleReport) -> String {
    let mut out = String::new();
    writeln!(out, "  Source: {}", report.source).ok();
    writeln!(out, "  Mode: {:?}", report.mode).ok();
    out
}

fn output_text(report: &ModelOracleReport, verbose: bool) {
    output::header("Model Oracle Report");
    print!("{}", format_text_report(report));

    output_text_format(report.format.as_ref());
    output_text_family(report.family.as_ref(), verbose);
    output_text_size(report.size_variant.as_ref());
    output_text_constraints(report.family.as_ref());
    output_text_stats(report.stats.as_ref());
    output_text_explanation(report.explanation.as_ref());
    output_text_kernels(report.kernel_compatibility.as_ref());
    output_text_cross_validation(report.cross_validation.as_ref());
    output_text_hf(report.hf_data.as_ref(), verbose);
    output_text_compliance(report.compliance.as_ref(), verbose);
    output_text_certification(report.certification.as_ref());
    output_text_tensors(report.tensors.as_ref(), verbose);
}

fn format_text_format(fmt: &FormatInfo) -> String {
    let mut out = String::new();
    writeln!(out, "  Format: {}", fmt.format_type).ok();
    writeln!(
        out,
        "  File Size: {}",
        output::format_size(fmt.file_size as u64)
    )
    .ok();
    writeln!(out, "  Tensors: {}", fmt.tensor_count).ok();
    writeln!(out, "  Parameters: {}", format_params(fmt.total_params)).ok();
    if let Some(ref q) = fmt.quantization {
        writeln!(out, "  Quantization: {q}").ok();
    }
    if let Some(ref arch) = fmt.architecture {
        writeln!(out, "  Architecture: {arch}").ok();
    }
    out
}

fn output_text_format(fmt: Option<&FormatInfo>) {
    let Some(fmt) = fmt else { return };
    print!("{}", format_text_format(fmt));
}

fn format_text_family(family: &FamilyInfo, verbose: bool) -> String {
    let mut out = String::new();
    writeln!(out).ok();
    writeln!(out, "  Family: {} ({})", family.name, family.display_name).ok();
    writeln!(out, "  Vendor: {}", family.vendor).ok();
    if verbose {
        writeln!(out, "  Architectures: {}", family.architectures.join(", ")).ok();
    }
    if let Some(ref ct) = family.chat_template_format {
        writeln!(out, "  Chat Template: {ct}").ok();
    }
    out
}

fn output_text_family(family: Option<&FamilyInfo>, verbose: bool) {
    let Some(family) = family else {
        println!();
        output::kv("Family", "UNKNOWN (no matching contract)");
        return;
    };
    print!("{}", format_text_family(family, verbose));
}
