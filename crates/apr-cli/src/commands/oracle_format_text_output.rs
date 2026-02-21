
fn format_text_size(size: &SizeVariantInfo) -> String {
    let mut out = String::new();
    writeln!(out).ok();
    writeln!(
        out,
        "  Size: {} (hidden={}, layers={}, heads={}, kv_heads={})",
        size.parameters, size.hidden_dim, size.num_layers, size.num_heads, size.num_kv_heads
    )
    .ok();
    writeln!(out, "  Intermediate Dim: {}", size.intermediate_dim).ok();
    writeln!(out, "  Vocab Size: {}", size.vocab_size).ok();
    writeln!(out, "  Expected Tensors: {}", size.expected_tensor_count).ok();
    out
}

fn output_text_size(size: Option<&SizeVariantInfo>) {
    let Some(size) = size else { return };
    print!("{}", format_text_size(size));
}

fn format_text_constraints(family: &FamilyInfo) -> String {
    let mut out = String::new();
    writeln!(out).ok();
    writeln!(out, "  Constraints:").ok();
    let c = &family.constraints;
    writeln!(
        out,
        "    Attention: {} | Activation: {} | Norm: {}",
        c.attention, c.activation, c.norm
    )
    .ok();
    writeln!(
        out,
        "    Bias: {} | Tied: {} | MLP: {} | Position: {}",
        if c.bias { "yes" } else { "no" },
        if c.tied_embeddings { "yes" } else { "no" },
        c.mlp,
        c.positional_encoding
    )
    .ok();
    out
}

fn output_text_constraints(family: Option<&FamilyInfo>) {
    let Some(family) = family else { return };
    print!("{}", format_text_constraints(family));
}

fn format_text_stats(stats: &StatisticalAnalysis) -> String {
    let mut out = String::new();
    writeln!(out).ok();
    writeln!(
        out,
        "  GQA Ratio: {:.2} ({:.0}% KV cache reduction)",
        stats.gqa_ratio,
        stats.kv_cache_reduction * 100.0
    )
    .ok();
    writeln!(
        out,
        "  Model Parameters: {}",
        format_params(stats.model_params as usize)
    )
    .ok();
    writeln!(out, "  Model Size (F16): {:.1} MB", stats.model_size_f16_mb).ok();
    writeln!(
        out,
        "  Model Size (Q4_K_M): {:.1} MB",
        stats.model_size_q4_mb
    )
    .ok();
    writeln!(
        out,
        "  KV Cache/Token: {} bytes",
        stats.kv_cache_per_token_bytes
    )
    .ok();
    writeln!(out, "  KV Cache (4K ctx): {:.1} MB", stats.kv_cache_4k_mb).ok();
    writeln!(out, "  FFN Expansion: {:.2}x", stats.ffn_expansion_ratio).ok();
    writeln!(out, "  FFN Type: {}", stats.ffn_type_explanation).ok();
    if stats.rope_max_wavelength > 0.0 {
        writeln!(out, "  RoPE Wavelength: {:.0}", stats.rope_max_wavelength).ok();
    }
    writeln!(out, "  Context Window: {}", stats.effective_context_window).ok();
    writeln!(
        out,
        "  Attn FLOPS/tok: {:.2e}",
        stats.attention_flops_per_token as f64
    )
    .ok();
    writeln!(
        out,
        "  FFN FLOPS/tok: {:.2e}",
        stats.ffn_flops_per_token as f64
    )
    .ok();
    out
}

fn output_text_stats(stats: Option<&StatisticalAnalysis>) {
    let Some(stats) = stats else { return };
    output::section("Statistical Analysis");
    print!("{}", format_text_stats(stats));
}

fn format_text_explanation(expl: &ArchitectureExplanation) -> String {
    let mut out = String::new();
    writeln!(out).ok();
    writeln!(out, "  Attention: {}", expl.attention_explanation).ok();
    writeln!(out).ok();
    writeln!(out, "  FFN: {}", expl.ffn_explanation).ok();
    writeln!(out).ok();
    writeln!(out, "  Normalization: {}", expl.norm_explanation).ok();
    writeln!(out).ok();
    writeln!(out, "  Position: {}", expl.positional_explanation).ok();
    writeln!(out).ok();
    writeln!(out, "  Scaling: {}", expl.scaling_analysis).ok();
    out
}

fn output_text_explanation(expl: Option<&ArchitectureExplanation>) {
    let Some(expl) = expl else { return };
    output::section("Architecture Explanation");
    print!("{}", format_text_explanation(expl));
}

fn format_text_kernels(kern: &KernelCompatibility) -> String {
    let mut out = String::new();
    writeln!(out, "  Attention Kernel: {}", kern.attention_kernel).ok();
    writeln!(out, "  FFN Kernel: {}", kern.ffn_kernel).ok();
    writeln!(out).ok();
    writeln!(out, "  Quantization Support:").ok();
    writeln!(
        out,
        "    {:<12} {:<10} {:<8} {:<12} Kernel",
        "Format", "Supported", "BPW", "Size (MB)"
    )
    .ok();
    for q in &kern.supported_quantizations {
        writeln!(
            out,
            "    {:<12} {:<10} {:<8.1} {:<12.1} {}",
            q.format,
            if q.supported { "yes" } else { "no" },
            q.bits_per_weight,
            q.estimated_size_mb,
            q.kernel
        )
        .ok();
    }
    if let Some(tps_cpu) = kern.estimated_tps_cpu {
        writeln!(out, "  Est. CPU tok/s (Q4_K_M): {tps_cpu:.0}").ok();
    }
    if let Some(tps_gpu) = kern.estimated_tps_gpu {
        writeln!(out, "  Est. GPU tok/s (Q4_K_M): {tps_gpu:.0}").ok();
    }
    writeln!(
        out,
        "  Memory Required (Q4+KV): {:.1} MB",
        kern.memory_required_mb
    )
    .ok();
    for note in &kern.notes {
        writeln!(out, "    * {note}").ok();
    }
    out
}

fn output_text_kernels(kern: Option<&KernelCompatibility>) {
    let Some(kern) = kern else { return };
    println!();
    output::section("Kernel Compatibility");
    print!("{}", format_text_kernels(kern));
}

fn format_text_cross_validation(cv: &CrossValidation) -> String {
    let mut out = String::new();
    if !cv.matches.is_empty() {
        writeln!(out, "  Matches ({}):", cv.matches.len()).ok();
        for entry in &cv.matches {
            writeln!(
                out,
                "    [OK] {}: {} == {}",
                entry.field, entry.contract_value, entry.hf_value
            )
            .ok();
        }
    }
    if !cv.mismatches.is_empty() {
        writeln!(out, "  Mismatches ({}):", cv.mismatches.len()).ok();
        for entry in &cv.mismatches {
            writeln!(
                out,
                "    [!!] {}: contract={} vs hf={}",
                entry.field, entry.contract_value, entry.hf_value
            )
            .ok();
        }
    }
    if !cv.contract_only.is_empty() {
        writeln!(out, "  Contract-only: {}", cv.contract_only.join(", ")).ok();
    }
    if !cv.hf_only.is_empty() {
        writeln!(out, "  HF-only: {}", cv.hf_only.join(", ")).ok();
    }
    out
}

fn output_text_cross_validation(cv: Option<&CrossValidation>) {
    let Some(cv) = cv else { return };
    println!();
    output::section("Cross-Validation (Contract vs HF)");
    print!("{}", format_text_cross_validation(cv));
}

fn format_text_hf(hf: &HuggingFaceData) -> String {
    let mut out = String::new();
    writeln!(out).ok();
    writeln!(out, "  HF Repo: {}", hf.repo).ok();
    if let Some(ref mt) = hf.model_type {
        writeln!(out, "  HF model_type: {mt}").ok();
    }
    if let Some(ref pt) = hf.pipeline_tag {
        writeln!(out, "  HF pipeline_tag: {pt}").ok();
    }
    if let Some(dl) = hf.downloads {
        writeln!(out, "  HF Downloads: {dl}").ok();
    }
    out
}

fn output_text_hf(hf: Option<&HuggingFaceData>, verbose: bool) {
    let Some(hf) = hf else { return };
    if !verbose {
        return;
    }
    print!("{}", format_text_hf(hf));
}

fn format_text_compliance(compliance: &ComplianceResult, verbose: bool) -> String {
    let mut out = String::new();
    writeln!(out).ok();
    if compliance.is_compliant {
        writeln!(out, "  Contract: COMPLIANT").ok();
        return out;
    }
    writeln!(out, "  Contract: NON-COMPLIANT").ok();
    if !compliance.tensor_count_match {
        writeln!(out, "  Tensor Count: MISMATCH").ok();
    }
    if !compliance.missing_tensors.is_empty() {
        writeln!(
            out,
            "  Missing Tensors: {} tensor(s)",
            compliance.missing_tensors.len()
        )
        .ok();
        if verbose {
            for t in &compliance.missing_tensors {
                writeln!(out, "    - {t}").ok();
            }
        }
    }
    if !compliance.unexpected_tensors.is_empty() && verbose {
        writeln!(
            out,
            "  Unexpected Tensors: {} tensor(s)",
            compliance.unexpected_tensors.len()
        )
        .ok();
        for t in &compliance.unexpected_tensors {
            writeln!(out, "    + {t}").ok();
        }
    }
    out
}

fn output_text_compliance(compliance: Option<&ComplianceResult>, verbose: bool) {
    let Some(compliance) = compliance else { return };
    print!("{}", format_text_compliance(compliance, verbose));
}

fn format_text_certification(cert: &CertificationInfo) -> String {
    let mut out = String::new();
    writeln!(out).ok();
    writeln!(out, "  Certification: {}", cert.status).ok();
    if let Some(ref pb) = cert.playbook_path {
        writeln!(out, "  Playbook: {pb}").ok();
    }
    out
}

fn output_text_certification(cert: Option<&CertificationInfo>) {
    let Some(cert) = cert else { return };
    print!("{}", format_text_certification(cert));
}

fn format_text_tensors(tensors: &[TensorComplianceEntry], verbose: bool) -> String {
    let mut out = String::new();
    writeln!(out).ok();
    writeln!(out, "  Tensors ({} total):", tensors.len()).ok();
    let max_show = if verbose { tensors.len() } else { 20 };
    for (i, t) in tensors.iter().enumerate() {
        if i >= max_show && i < tensors.len() - 2 {
            if i == max_show {
                writeln!(out, "    ... ({} more) ...", tensors.len() - max_show - 2).ok();
            }
            continue;
        }
        let shape_str = t
            .shape
            .as_ref()
            .map(|s| format!("{s:?}"))
            .unwrap_or_default();
        let dtype_str = t.dtype.as_deref().unwrap_or("");
        writeln!(out, "    {:<50} {} {}", t.name, dtype_str, shape_str).ok();
    }
    out
}

fn output_text_tensors(tensors: Option<&Vec<TensorComplianceEntry>>, verbose: bool) {
    let Some(tensors) = tensors else { return };
    print!("{}", format_text_tensors(tensors, verbose));
}

/// Format the family description header (config metadata + constraints).
fn format_family_description_header(config: &ModelFamilyConfig) -> String {
    let mut out = String::new();
    writeln!(out, "  Family: {}", config.family).ok();
    writeln!(out, "  Vendor: {}", config.vendor).ok();
    writeln!(out, "  Architectures: {}", config.architectures.join(", ")).ok();
    writeln!(out, "  HF Pattern: {}", config.hf_pattern).ok();

    let c = &config.constraints;
    writeln!(out).ok();
    writeln!(out, "  Constraints:").ok();
    writeln!(
        out,
        "    Attention: {} | Activation: {} | Norm: {}",
        c.attention_type, c.activation, c.norm_type
    )
    .ok();
    writeln!(
        out,
        "    Bias: {} | Tied: {} | MLP: {} | Position: {}",
        if c.has_bias { "yes" } else { "no" },
        if c.tied_embeddings { "yes" } else { "no" },
        c.mlp_type,
        c.positional_encoding
    )
    .ok();
    out
}

/// Format a single size variant block.
fn format_family_size_variant(name: &str, sc: &ModelSizeConfig) -> String {
    let mut out = String::new();
    writeln!(out).ok();
    writeln!(out, "  Size Variant: {name} ({})", sc.parameters).ok();
    writeln!(out, "    hidden_dim: {}", sc.hidden_dim).ok();
    writeln!(out, "    num_layers: {}", sc.num_layers).ok();
    writeln!(out, "    num_heads: {}", sc.num_heads).ok();
    writeln!(out, "    num_kv_heads: {}", sc.num_kv_heads).ok();
    writeln!(out, "    intermediate_dim: {}", sc.intermediate_dim).ok();
    writeln!(out, "    vocab_size: {}", sc.vocab_size).ok();
    writeln!(out, "    head_dim: {}", sc.head_dim).ok();
    if sc.rope_theta > 0.0 {
        writeln!(out, "    rope_theta: {}", sc.rope_theta).ok();
    }
    writeln!(out, "    norm_eps: {}", sc.norm_eps).ok();
    out
}

/// Format per-variant stats summary.
fn format_family_variant_stats(stats: &StatisticalAnalysis) -> String {
    let mut out = String::new();
    writeln!(out).ok();
    writeln!(
        out,
        "    GQA Ratio: {:.2} ({:.0}% KV reduction)",
        stats.gqa_ratio,
        stats.kv_cache_reduction * 100.0
    )
    .ok();
    writeln!(
        out,
        "    Est. Parameters: {}",
        format_params(stats.model_params as usize)
    )
    .ok();
    writeln!(
        out,
        "    Model Size (F16): {:.1} MB",
        stats.model_size_f16_mb
    )
    .ok();
    writeln!(out, "    Model Size (Q4): {:.1} MB", stats.model_size_q4_mb).ok();
    writeln!(out, "    KV Cache (4K): {:.1} MB", stats.kv_cache_4k_mb).ok();
    writeln!(out, "    FFN Ratio: {:.2}x", stats.ffn_expansion_ratio).ok();
    out
}

/// Format per-variant explanation summary.
fn format_family_variant_explain(expl: &ArchitectureExplanation) -> String {
    let mut out = String::new();
    writeln!(out).ok();
    writeln!(out, "    Attention: {}", expl.attention_explanation).ok();
    writeln!(out, "    FFN: {}", expl.ffn_explanation).ok();
    writeln!(out, "    Scaling: {}", expl.scaling_analysis).ok();
    out
}

/// Format per-variant kernel summary.
fn format_family_variant_kernels(kern: &KernelCompatibility) -> String {
    let mut out = String::new();
    writeln!(out).ok();
    writeln!(out, "    Attn Kernel: {}", kern.attention_kernel).ok();
    writeln!(out, "    FFN Kernel: {}", kern.ffn_kernel).ok();
    if let Some(tps) = kern.estimated_tps_cpu {
        writeln!(out, "    Est. CPU tok/s: {tps:.0}").ok();
    }
    if let Some(tps) = kern.estimated_tps_gpu {
        writeln!(out, "    Est. GPU tok/s: {tps:.0}").ok();
    }
    writeln!(out, "    Memory (Q4+KV): {:.1} MB", kern.memory_required_mb).ok();
    out
}
