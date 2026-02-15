
/// Print the kernel table
fn print_kernel_table(steps: &[KernelStep], kernel_filter: Option<&str>) {
    print_table_header();

    for step in steps {
        // Apply kernel filter
        if let Some(filter) = kernel_filter {
            let filter_lower = filter.to_lowercase();
            if !step.name.to_lowercase().contains(&filter_lower)
                && !step.role.to_lowercase().contains(&filter_lower)
            {
                continue;
            }
        }

        let batched_marker = if step.is_batched { " [B]" } else { "" };
        println!(
            "  {:<3} {:<34} {:<16} {:<22} {}{}",
            step.index,
            step.name,
            step.role,
            truncate_shape(&step.shape, 22),
            step.source,
            batched_marker,
        );
    }
}

/// Truncate a shape string to fit column width
fn truncate_shape(shape: &str, max_len: usize) -> String {
    if shape.len() <= max_len {
        shape.to_string()
    } else {
        format!("{}...", &shape[..max_len - 3])
    }
}

/// Print reverse lookup: kernel name → which steps use it
fn print_reverse_lookup(steps: &[KernelStep], kernel_name: &str, info: &ModelInfo) {
    let filter_lower = kernel_name.to_lowercase();
    let matching: Vec<&KernelStep> = steps
        .iter()
        .filter(|s| s.name.to_lowercase().contains(&filter_lower))
        .collect();

    if matching.is_empty() {
        println!(
            "  No kernel matching '{}' found in the forward pass.",
            kernel_name
        );
        return;
    }

    println!("  Reverse lookup: '{}'\n", kernel_name);
    println!("  {:<3} {:<8} {:<20}", "#", "Role", "Shape");
    println!("  {:-<3} {:-<8} {:-<20}", "", "", "");

    for step in &matching {
        println!(
            "  {:<3} {:<8} {:<20}",
            step.index,
            step.role,
            truncate_shape(&step.shape, 20)
        );
    }

    let launches_per_layer = matching.len();
    let total = launches_per_layer * info.num_layers;
    println!(
        "\n  {} launches/layer x {} layers = {} total launches",
        launches_per_layer, info.num_layers, total
    );
}

/// Print JSON output
fn print_json(steps: &[KernelStep], info: &ModelInfo, prefill: bool) {
    println!("{{");
    println!("  \"model\": \"{}\",", info.name);
    println!("  \"quantization\": \"{}\",", info.quant);
    println!("  \"num_layers\": {},", info.num_layers);
    println!("  \"hidden_dim\": {},", info.hidden_dim);
    println!("  \"intermediate_dim\": {},", info.intermediate_dim);
    println!("  \"num_heads\": {},", info.num_heads);
    println!("  \"num_kv_heads\": {},", info.num_kv_heads);
    println!("  \"head_dim\": {},", info.head_dim);
    println!(
        "  \"mode\": \"{}\",",
        if prefill { "prefill" } else { "decode" }
    );
    println!("  \"kernels_per_layer\": {},", steps.len());
    let total = steps.len() * info.num_layers + 2; // +2 for final norm + lm_head
    println!("  \"total_launches\": {},", total);
    println!("  \"steps\": [");
    for (i, step) in steps.iter().enumerate() {
        let comma = if i + 1 < steps.len() { "," } else { "" };
        println!("    {{");
        println!("      \"index\": {},", step.index);
        println!("      \"kernel\": \"{}\",", step.name);
        println!("      \"role\": \"{}\",", step.role);
        println!("      \"shape\": \"{}\",", step.shape);
        println!("      \"source\": \"{}\",", step.source);
        println!("      \"batched\": {}", step.is_batched);
        println!("    }}{}", comma);
    }
    println!("  ]");
    println!("}}");
}

/// Main entry point for ptx-map command
#[allow(clippy::fn_params_excessive_bools)]
pub fn run(
    model_path: &Path,
    kernel_filter: Option<&str>,
    reverse: Option<&str>,
    json: bool,
    verbose: bool,
    prefill: bool,
) -> Result<()> {
    #[cfg(feature = "inference")]
    {
        let _verbose = verbose; // reserved for future PTX snippet output
        let info = extract_model_info(model_path)?;

        let steps = if prefill {
            build_prefill_sequence(&info)
        } else {
            build_decode_sequence(&info)
        };

        // JSON output
        if json {
            print_json(&steps, &info, prefill);
            return Ok(());
        }

        // Reverse lookup mode
        if let Some(kernel_name) = reverse {
            println!(
                "\nModel: {} ({})\n  {} layers, hidden={}, intermediate={}, heads={}, head_dim={}\n",
                info.name, info.quant, info.num_layers, info.hidden_dim,
                info.intermediate_dim, info.num_heads, info.head_dim
            );
            print_reverse_lookup(&steps, kernel_name, &info);
            return Ok(());
        }

        // Default: full table
        let mode = if prefill { "Prefill" } else { "Decode" };
        println!(
            "\nModel: {} ({})\n  {} layers, hidden={}, intermediate={}, heads={}, head_dim={}\n",
            info.name,
            info.quant,
            info.num_layers,
            info.hidden_dim,
            info.intermediate_dim,
            info.num_heads,
            info.head_dim
        );
        println!(
            "{} Kernel Sequence (per transformer layer, {} launches):\n",
            mode,
            steps.len()
        );

        print_kernel_table(&steps, kernel_filter);

        // Summary
        let total = steps.len() * info.num_layers + 2; // +2 for final norm + lm_head
        println!(
            "\n  Total: {} kernels/layer x {} layers + 2 (final norm, lm_head) = {} launches",
            steps.len(),
            info.num_layers,
            total
        );

        // PTX parity summary (uses realizar's validate_all_kernel_pairs)
        {
            use realizar::ptx_parity::{validate_all_kernel_pairs, KernelDimensions};
            let dims = KernelDimensions {
                hidden_dim: info.hidden_dim,
                intermediate_dim: info.intermediate_dim,
                num_heads: info.num_heads,
                head_dim: info.head_dim,
                rope_theta: 1_000_000.0,
                epsilon: 1e-6,
            };
            let report = validate_all_kernel_pairs(&dims);
            if report.total > 0 {
                println!(
                    "  PTX Parity: {}/{} kernel pairs {}",
                    report.passed,
                    report.total,
                    if report.all_passed() { "PASS" } else { "FAIL" }
                );
            }
        }

        println!();
        Ok(())
    }

    #[cfg(not(feature = "inference"))]
    {
        let _ = (model_path, kernel_filter, reverse, json, verbose, prefill);
        Err(CliError::FeatureDisabled(
            "ptx-map requires the 'inference' feature (--features inference)".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_ptx_counts_registers() {
        let ptx = r#"
.version 7.0
.target sm_80
.reg .f32 %f<24>;
.reg .b32 %r<8>;
ld.global.f32 %f0, [%r1];
ld.global.f32 %f1, [%r2];
st.global.f32 [%r3], %f2;
"#;
        let stats = analyze_ptx(ptx);
        assert_eq!(stats.registers, 32); // 24 + 8
        assert_eq!(stats.global_loads, 2);
        assert_eq!(stats.global_stores, 1);
        assert_eq!(stats.shared_bytes, 0);
    }

    #[test]
    fn test_analyze_ptx_counts_shared_memory() {
        let ptx = ".shared .align 4 .b8 shmem[256];";
        let stats = analyze_ptx(ptx);
        assert_eq!(stats.shared_bytes, 256);
    }

    #[test]
    fn test_analyze_ptx_empty() {
        let stats = analyze_ptx("");
        assert_eq!(stats.registers, 0);
        assert_eq!(stats.shared_bytes, 0);
        assert_eq!(stats.global_loads, 0);
        assert_eq!(stats.global_stores, 0);
    }

    #[test]
    fn test_source_location_known_kernels() {
        assert_eq!(
            source_location("VectorizedRmsNormKernel"),
            "trueno-gpu/src/kernels/layernorm.rs"
        );
        assert_eq!(
            source_location("Q4KGemvKernel"),
            "trueno-gpu/src/kernels/quantize/q4k/"
        );
        assert_eq!(
            source_location("RopeKernel"),
            "trueno-gpu/src/kernels/rope.rs"
        );
        assert_eq!(
            source_location("IncrementalAttentionKernel"),
            "trueno-gpu/src/kernels/attention/mod.rs"
        );
        assert_eq!(
            source_location("ResidualAddKernel"),
            "trueno-gpu/src/kernels/elementwise/residual.rs"
        );
        assert_eq!(
            source_location("SwigluKernel"),
            "trueno-gpu/src/kernels/activation.rs"
        );
    }

    #[test]
    fn test_source_location_batched_variants() {
        assert_eq!(
            source_location("BatchedVectorizedRmsNormKernel"),
            source_location("VectorizedRmsNormKernel")
        );
        assert_eq!(
            source_location("BatchedQ4KGemvKernel"),
            source_location("Q4KGemvKernel")
        );
        assert_eq!(
            source_location("BatchedRopeKernel"),
            source_location("RopeKernel")
        );
        assert_eq!(
            source_location("BatchedResidualAddKernel"),
            source_location("ResidualAddKernel")
        );
        assert_eq!(
            source_location("BatchedSwigluKernel"),
            source_location("SwigluKernel")
        );
    }

    #[test]
    fn test_source_location_unknown() {
        assert_eq!(source_location("FakeKernel"), "unknown");
    }

    #[test]
    fn test_build_decode_sequence_7b() {
        let info = ModelInfo {
            name: "test-7b".to_string(),
            quant: "Q4_K".to_string(),
            num_layers: 28,
            hidden_dim: 3584,
            intermediate_dim: 18944,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
        };
        let steps = build_decode_sequence(&info);
        assert_eq!(steps.len(), 12);
        assert_eq!(steps[0].name, "VectorizedRmsNormKernel");
        assert_eq!(steps[1].name, "Q4KGemvKernel");
        assert_eq!(steps[1].role, "QKV proj");
        assert_eq!(steps[11].name, "ResidualAddKernel");
        assert_eq!(steps[11].role, "post-FFN residual");
    }

    #[test]
    fn test_build_prefill_sequence_7b() {
        let info = ModelInfo {
            name: "test-7b".to_string(),
            quant: "Q4_K".to_string(),
            num_layers: 28,
            hidden_dim: 3584,
            intermediate_dim: 18944,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
        };
        let steps = build_prefill_sequence(&info);
        assert_eq!(steps.len(), 12);
        assert_eq!(steps[0].name, "BatchedVectorizedRmsNormKernel");
        assert!(steps[0].is_batched);
        assert_eq!(steps[1].name, "BatchedQ4KGemvKernel");
        // AttentionKernel is not batched (uses causal mask directly)
        assert!(!steps[3].is_batched);
    }

    #[test]
    fn test_format_shared() {
        assert_eq!(format_shared(0), "0");
        assert_eq!(format_shared(256), "256B");
        assert_eq!(format_shared(1024), "1KB");
        assert_eq!(format_shared(8192), "8KB");
    }

    #[test]
    fn test_truncate_shape() {
        assert_eq!(truncate_shape("3584 -> 3584", 20), "3584 -> 3584");
        assert_eq!(
            truncate_shape("this is a very long shape string", 20),
            "this is a very lo..."
        );
    }

    #[test]
    fn test_decode_sequence_shapes_use_model_dims() {
        let info = ModelInfo {
            name: "test".to_string(),
            quant: "Q4_K".to_string(),
            num_layers: 28,
            hidden_dim: 3584,
            intermediate_dim: 18944,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
        };
        let steps = build_decode_sequence(&info);
        // Gate proj: hidden -> intermediate
        assert!(steps[7].shape.contains("3584"));
        assert!(steps[7].shape.contains("18944"));
        // Down proj: intermediate -> hidden
        assert!(steps[10].shape.contains("18944"));
        assert!(steps[10].shape.contains("3584"));
    }

    #[test]
    fn test_reverse_lookup_finds_multiple_steps() {
        let info = ModelInfo {
            name: "test".to_string(),
            quant: "Q4_K".to_string(),
            num_layers: 28,
            hidden_dim: 3584,
            intermediate_dim: 18944,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
        };
        let steps = build_decode_sequence(&info);
        let matching: Vec<&KernelStep> = steps
            .iter()
            .filter(|s| s.name.to_lowercase().contains("q4kgemv"))
            .collect();
        // Q4KGemv appears 5 times: QKV, out, gate, up, down
        assert_eq!(matching.len(), 5);
    }

    #[test]
    fn test_1_5b_model_dimensions() {
        let info = ModelInfo {
            name: "test-1.5b".to_string(),
            quant: "Q4_K".to_string(),
            num_layers: 28,
            hidden_dim: 1536,
            intermediate_dim: 8960,
            num_heads: 12,
            num_kv_heads: 2,
            head_dim: 128,
        };
        let steps = build_decode_sequence(&info);
        assert_eq!(steps.len(), 12);
        // QKV output: 12*128 + 2*2*128 = 1536 + 512 = 2048
        assert!(steps[1].shape.contains("2048"));
    }
}
