
pub fn build_kernel_compatibility(
    size: &ModelSizeConfig,
    constraints: &ModelConstraints,
    stats: &StatisticalAnalysis,
) -> KernelCompatibility {
    let params = stats.model_params;
    let params_b = params as f64 / 1e9;

    // Build quantization support entries
    let quant_entries = vec![
        QuantizationSupport {
            format: "F16".to_string(),
            supported: true,
            kernel: "trueno::f16_matvec".to_string(),
            bits_per_weight: 16.0,
            estimated_size_mb: (params as f64 * 2.0) / (1024.0 * 1024.0),
        },
        QuantizationSupport {
            format: "Q8_0".to_string(),
            supported: true,
            kernel: "trueno::q8_matvec".to_string(),
            bits_per_weight: 8.0,
            estimated_size_mb: (params as f64 * 1.0) / (1024.0 * 1024.0),
        },
        QuantizationSupport {
            format: "Q4_K_M".to_string(),
            supported: true,
            kernel: "fused_q4k_parallel_matvec (row-major)".to_string(),
            bits_per_weight: 4.5,
            estimated_size_mb: (params as f64 * 0.5625) / (1024.0 * 1024.0),
        },
        QuantizationSupport {
            format: "Q6_K".to_string(),
            supported: true,
            kernel: "fused_q6k_parallel_matvec (row-major)".to_string(),
            bits_per_weight: 6.5,
            estimated_size_mb: (params as f64 * 0.8125) / (1024.0 * 1024.0),
        },
    ];

    // Attention kernel description
    let attention_kernel = match constraints.attention_type {
        AttentionType::Gqa => format!(
            "GQA fused QKV (row-major): {kv}KV groups, {}Q heads per group",
            size.num_heads / size.num_kv_heads.max(1),
            kv = size.num_kv_heads
        ),
        AttentionType::Mha => format!(
            "MHA standard QKV (row-major): {} heads x {} dim",
            size.num_heads, size.head_dim
        ),
        AttentionType::Mqa => format!(
            "MQA single-KV QKV (row-major): 1 KV, {} Q heads",
            size.num_heads
        ),
    };

    // FFN kernel description
    let ffn_kernel = match constraints.mlp_type {
        MlpType::SwiGlu => "fused gated SwiGLU matvec (gate+up fused, row-major)".to_string(),
        MlpType::GatedMlp => "fused gated GeGLU matvec (gate+up fused, row-major)".to_string(),
        MlpType::GeluMlp => "standard GELU MLP matvec (row-major)".to_string(),
    };

    // Performance estimation (memory bandwidth model)
    // CPU: ~50 GB/s DDR5, GPU: ~900 GB/s HBM3
    let q4_size_gb = (params as f64 * 0.5625) / (1024.0 * 1024.0 * 1024.0);
    let estimated_tps_cpu = if params_b > 0.0 {
        Some(50.0 / q4_size_gb)
    } else {
        None
    };
    let estimated_tps_gpu = if params_b > 0.0 {
        Some(900.0 / q4_size_gb)
    } else {
        None
    };

    let memory_required_mb = (params as f64 * 0.5625) / (1024.0 * 1024.0) + stats.kv_cache_4k_mb;

    let mut notes = Vec::new();
    notes.push("All kernels use ROW-MAJOR layout (LAYOUT-001/002)".to_string());
    if constraints.has_bias {
        notes.push("Bias terms included in attention projections".to_string());
    }
    if size.num_kv_heads < size.num_heads {
        notes.push(format!(
            "GQA: {} query heads share {} KV groups ({}:1 ratio)",
            size.num_heads,
            size.num_kv_heads,
            size.num_heads / size.num_kv_heads.max(1)
        ));
    }

    KernelCompatibility {
        supported_quantizations: quant_entries,
        attention_kernel,
        ffn_kernel,
        estimated_tps_cpu,
        estimated_tps_gpu,
        memory_required_mb,
        notes,
    }
}

// ============================================================================
// Phase 6: CLI Options
// ============================================================================

/// Oracle enhancement flags (passed from CLI layer)
#[derive(Debug, Clone, Copy, Default)]
pub struct OracleFlags {
    pub stats: bool,
    pub explain: bool,
    pub kernels: bool,
    pub validate: bool,
    pub full: bool,
}

impl OracleFlags {
    fn show_stats(self) -> bool {
        self.stats || self.full
    }
    fn show_explain(self) -> bool {
        self.explain || self.full
    }
    fn show_kernels(self) -> bool {
        self.kernels || self.full
    }
    fn show_validate(self) -> bool {
        self.validate || self.full
    }
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
    flags: OracleFlags,
) -> Result<(), CliError> {
    // Mode 3: Contract description (--family)
    if let Some(family) = family_name {
        return run_family_mode(
            family,
            size_filter.map(String::as_str),
            json_output,
            verbose,
            flags,
        );
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
        return run_hf_mode(source, json_output, verbose, flags);
    }

    // Mode 1: Local file analysis
    let path = PathBuf::from(source);
    run_local_mode(
        &path,
        show_compliance,
        show_tensors,
        json_output,
        verbose,
        flags,
    )
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
    flags: OracleFlags,
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

    // Detect family — try tensor names first (SafeTensors), then GGUF architecture metadata
    let detected_family = registry.detect_family(&tensor_names).or_else(|| {
        // GGUF uses different tensor naming (blk.0.attn_q vs model.layers.0.self_attn.q_proj).
        // Fall back to GGUF architecture metadata (e.g., "qwen2" → Qwen2 family).
        report
            .architecture
            .as_deref()
            .and_then(|arch| registry.detect_from_model_type(arch))
    });

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

    // Build enhanced sections
    let (stats, explanation, kernel_compat) = build_enhanced_sections(
        detected_family,
        size_variant_info.as_ref().map(|s| s.name.as_str()),
        flags,
    );

    let oracle_report = ModelOracleReport {
        source: path.display().to_string(),
        mode: OracleMode::Local,
        family: family_info,
        size_variant: size_variant_info,
        format: Some(format_info),
        compliance,
        certification,
        tensors: tensors_list,
        stats,
        explanation,
        kernel_compatibility: kernel_compat,
        cross_validation: None, // No HF data in local mode
        hf_data: None,
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

fn run_hf_mode(
    source: &str,
    json_output: bool,
    verbose: bool,
    flags: OracleFlags,
) -> Result<(), CliError> {
    // Parse HF URI → org/repo
    let repo = source
        .strip_prefix("hf://")
        .or_else(|| source.strip_prefix("huggingface://"))
        .ok_or_else(|| CliError::InvalidFormat(format!("Invalid HF URI: {source}")))?;

    // Fetch HF data via ureq
    let hf_data = fetch_hf_data(repo)?;

    // Load family registry
    let registry = load_registry()?;

    // Extract model_type from config.json
    let model_type = hf_data.config_fields["model_type"].as_str();
    let hidden_size = hf_data.config_fields["hidden_size"]
        .as_u64()
        .map(|v| v as usize);
    let num_layers = hf_data.config_fields["num_hidden_layers"]
        .as_u64()
        .map(|v| v as usize);

    // Detect family from model_type
    let detected_family = model_type.and_then(|mt| registry.detect_from_model_type(mt));

    // Build family info and size variant
    let (family_info, size_variant_info) = if let Some(family) = detected_family {
        let config = family.config();
        let fi = build_family_info(config);

        // Detect size from config.json values
        let size_info = match (hidden_size, num_layers) {
            (Some(h), Some(l)) => family.detect_size(h, l).and_then(|size_name| {
                family
                    .size_config(&size_name)
                    .map(|sc| build_size_info(&size_name, sc, family))
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

    // Build enhanced sections
    let (stats, explanation, kernel_compat) = build_enhanced_sections(
        detected_family,
        size_variant_info.as_ref().map(|s| s.name.as_str()),
        flags,
    );

    // Cross-validation
    let cross_validation = if flags.show_validate() {
        detected_family.and_then(|family| {
            let size_name = size_variant_info.as_ref().map(|s| s.name.as_str())?;
            let size_config = family.size_config(size_name)?;
            Some(cross_validate(
                size_config,
                family.constraints(),
                &hf_data.config_fields,
            ))
        })
    } else {
        None
    };

    let oracle_report = ModelOracleReport {
        source: source.to_string(),
        mode: OracleMode::HuggingFace,
        family: family_info,
        size_variant: size_variant_info,
        format: None,
        compliance: None,
        certification,
        tensors: None,
        stats,
        explanation,
        kernel_compatibility: kernel_compat,
        cross_validation,
        hf_data: Some(hf_data),
    };

    if json_output {
        output_json(&oracle_report)?;
    } else {
        output_text(&oracle_report, verbose);
    }

    Ok(())
}
