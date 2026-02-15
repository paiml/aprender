
/// Format family description footer (tensor template, quantizations, chat template).
fn format_family_description_footer(config: &ModelFamilyConfig, verbose: bool) -> String {
    let mut out = String::new();
    if verbose {
        writeln!(out).ok();
        writeln!(out, "  Tensor Template:").ok();
        writeln!(out, "    Embedding: {}", config.tensor_template.embedding).ok();
        if let Some(ref lm) = config.tensor_template.lm_head {
            writeln!(out, "    LM Head: {lm}").ok();
        }
        if let Some(ref norm) = config.tensor_template.final_norm {
            writeln!(out, "    Final Norm: {norm}").ok();
        }
        writeln!(out, "    Per-layer:").ok();
        for (role, pattern) in &config.tensor_template.per_layer {
            if let Some(pat) = pattern {
                writeln!(out, "      {role}: {pat}").ok();
            }
        }
    }
    if !config.quantizations.is_empty() {
        writeln!(out).ok();
        writeln!(out, "  Quantizations: {}", config.quantizations.join(", ")).ok();
    }
    if let Some(ref ct) = config.chat_template {
        writeln!(out).ok();
        writeln!(out, "  Chat Template: {}", ct.format).ok();
        writeln!(out, "    BOS: {}", ct.bos_token).ok();
        writeln!(out, "    EOS: {}", ct.eos_token).ok();
    }
    out
}

fn output_family_description(
    config: &ModelFamilyConfig,
    size_filter: Option<&str>,
    verbose: bool,
    flags: OracleFlags,
    family: &dyn ModelFamily,
) {
    output::section(&format!("{} Family Contract", config.display_name));
    print!("{}", format_family_description_header(config));

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
        print!("{}", format_family_size_variant(name, sc));

        if flags.show_stats() || flags.show_explain() || flags.show_kernels() {
            let stats = build_statistical_analysis(sc, family.constraints());

            if flags.show_stats() {
                print!("{}", format_family_variant_stats(&stats));
            }

            if flags.show_explain() {
                let expl = build_architecture_explanation(sc, family.constraints(), &stats);
                print!("{}", format_family_variant_explain(&expl));
            }

            if flags.show_kernels() {
                let kern = build_kernel_compatibility(sc, family.constraints(), &stats);
                print!("{}", format_family_variant_kernels(&kern));
            }
        }
    }

    if size_filter.is_some() && variants.is_empty() {
        println!();
        output::kv(
            "Error",
            format!("Size '{}' not found", size_filter.unwrap_or("")),
        );
        let available: Vec<&String> = config.size_variants.keys().collect();
        output::kv("Available", format!("{available:?}"));
    }

    print!("{}", format_family_description_footer(config, verbose));
}
