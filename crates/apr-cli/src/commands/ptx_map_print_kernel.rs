
/// Print the kernel table
fn print_kernel_table(steps: &[KernelStep], kernel_filter: Option<&str>) {
    print_table_header();

    for step in steps {
        // Apply kernel filter (same predicate `validate_kernel_filter` checks)
        if let Some(filter) = kernel_filter {
            if !step_matches_filter(step, &filter.to_lowercase(), true) {
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

/// Does this step match a user-supplied filter?
///
/// `match_role` mirrors the two call sites: the `--kernel` table filter matches
/// kernel name OR role, `--reverse` matches kernel name only.
fn step_matches_filter(step: &KernelStep, needle_lower: &str, match_role: bool) -> bool {
    step.name.to_lowercase().contains(needle_lower)
        || (match_role && step.role.to_lowercase().contains(needle_lower))
}

/// Reject a filter that matches no kernel in the map.
///
/// `--kernel BOGUSKERNEL` used to print an empty table and exit 0, which reads
/// as "this model launches no kernels" rather than "you mistyped a name"
/// (dogfood-0.63.0, issue #2399 finding 2). Listing the available kernels makes
/// the error actionable.
fn validate_kernel_filter(steps: &[KernelStep], filter: &str, match_role: bool) -> Result<()> {
    let needle = filter.to_lowercase();
    if steps
        .iter()
        .any(|s| step_matches_filter(s, &needle, match_role))
    {
        return Ok(());
    }
    let mut names: Vec<&str> = steps.iter().map(|s| s.name).collect();
    names.sort_unstable();
    names.dedup();
    Err(CliError::ValidationFailed(format!(
        "No kernel matches '{}' in this model's kernel map. Available kernels: {}",
        filter,
        names.join(", ")
    )))
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
#[provable_contracts_macros::contract("apr-cli-command-safety-v1", equation = "read_only_no_side_effects")]
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

        // Contract: apr-gpu-parity-consistency-v1.yaml — scope clarity
        if !json {
            eprintln!("Scope: PTX kernel DISPATCH map — verifies kernel launch configuration");
            eprintln!("(See also: apr parity for GPU/CPU output correctness comparison)");
            eprintln!();
        }

        let info = extract_model_info(model_path)?;

        let steps = if prefill {
            build_prefill_sequence(&info)
        } else {
            build_decode_sequence(&info)
        };

        // A filter that names no kernel is a user error in every mode (#2399).
        if let Some(filter) = kernel_filter {
            validate_kernel_filter(&steps, filter, true)?;
        }
        if let Some(kernel_name) = reverse {
            validate_kernel_filter(&steps, kernel_name, false)?;
        }

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

    /// Workspace root, from this crate's manifest dir (`crates/apr-cli`).
    fn workspace_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(std::path::Path::parent)
            .expect("crates/apr-cli always has a workspace root two levels up")
            .to_path_buf()
    }

    fn sample_info() -> ModelInfo {
        ModelInfo {
            name: "table-check".to_string(),
            quant: MeasuredQuant::from_qtype(12).unwrap_or(MeasuredQuant::UNKNOWN),
            num_layers: 28,
            hidden_dim: 3584,
            intermediate_dim: 18944,
            num_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
        }
    }

    /// Every path the Source column prints must open, and must be the file that
    /// defines that kernel.
    ///
    /// dogfood-0.63.0, issue #2399 finding 3: the whole table pointed into
    /// `trueno-gpu/src/kernels/...`, a tree the APR-MONO consolidation removed,
    /// and 3 of 6 leaf paths were wrong on top of that (`layernorm.rs` is a
    /// directory, rope lives under `elementwise/rope/`, `activation.rs` never
    /// existed). The previous tests here asserted those exact dead strings, so
    /// they were green the entire time the column was useless — they are
    /// replaced by this check against the working tree.
    #[test]
    fn source_paths_resolve_to_the_defining_file() {
        let root = workspace_root();
        assert!(
            root.join("crates/aprender-gpu/src/kernels").is_dir(),
            "kernel tree missing at {} — this test cannot prove anything",
            root.display()
        );

        let info = sample_info();
        let mut names: Vec<&str> = build_decode_sequence(&info)
            .iter()
            .chain(build_prefill_sequence(&info).iter())
            .map(|s| s.name)
            .collect();
        // Kernels reachable through the table but not in the 12-step sequences.
        names.extend([
            "Q6KGemvKernel",
            "BatchedQ6KGemvKernel",
            "TensorCoreQ4KGemmKernel",
            "KvCacheScatterKernel",
            "ArgMaxKernel",
        ]);
        names.sort_unstable();
        names.dedup();

        for name in names {
            let rel = source_location(name);
            assert_ne!(rel, "unknown", "{name} has no source location");
            let path = root.join(rel);
            assert!(
                path.is_file(),
                "{name}: source column points at {rel}, which does not exist"
            );
            let src = std::fs::read_to_string(&path).expect("kernel source must be readable");
            assert!(
                src.contains(&format!("struct {name}")),
                "{name}: {rel} exists but does not define `struct {name}`"
            );
        }
    }

    #[test]
    fn test_source_location_unknown() {
        assert_eq!(source_location("FakeKernel"), "unknown");
    }

    /// `--kernel BOGUSKERNEL` printed an empty table and exited 0, which reads
    /// as "this model launches no kernels" (#2399 finding 2, rider b).
    #[test]
    fn unknown_kernel_filter_is_rejected_and_lists_alternatives() {
        let steps = build_decode_sequence(&sample_info());
        let err = validate_kernel_filter(&steps, "BOGUSKERNEL", true)
            .expect_err("a filter matching no kernel must be an error, not an empty table");
        let msg = err.to_string();
        assert!(
            msg.contains("BOGUSKERNEL") && msg.contains("Q4KGemvKernel"),
            "error must echo the bad filter and list real kernels, got: {msg}"
        );
    }

    #[test]
    fn real_kernel_and_role_filters_are_accepted() {
        let steps = build_decode_sequence(&sample_info());
        validate_kernel_filter(&steps, "Q4KGemv", true).expect("kernel-name filter must be valid");
        validate_kernel_filter(&steps, "gate proj", true).expect("role filter must be valid");
        // --reverse matches names only, so a role must NOT satisfy it.
        validate_kernel_filter(&steps, "Q4KGemv", false).expect("reverse by kernel name is valid");
        assert!(
            validate_kernel_filter(&steps, "gate proj", false).is_err(),
            "--reverse takes a kernel name, not a role"
        );
    }

    #[test]
    fn test_build_decode_sequence_7b() {
        let info = ModelInfo {
            name: "test-7b".to_string(),
            quant: MeasuredQuant::from_qtype(12).unwrap_or(MeasuredQuant::UNKNOWN),
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
            quant: MeasuredQuant::from_qtype(12).unwrap_or(MeasuredQuant::UNKNOWN),
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
            quant: MeasuredQuant::from_qtype(12).unwrap_or(MeasuredQuant::UNKNOWN),
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
            quant: MeasuredQuant::from_qtype(12).unwrap_or(MeasuredQuant::UNKNOWN),
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
            quant: MeasuredQuant::from_qtype(12).unwrap_or(MeasuredQuant::UNKNOWN),
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

    // ========================================================================
    // #2444: ptx-map must REPORT what the file says, never reconstruct it
    // ========================================================================

    /// Minimal GGUF v3 writer — header, metadata, tensor info, tensor data.
    /// Only what `GGUFConfig::from_gguf` reads; no tokenizer, no real weights.
    struct TinyGguf {
        meta: Vec<u8>,
        meta_count: u64,
        tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)>,
    }

    impl TinyGguf {
        fn new() -> Self {
            Self {
                meta: Vec::new(),
                meta_count: 0,
                tensors: Vec::new(),
            }
        }

        fn push_str(buf: &mut Vec<u8>, s: &str) {
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }

        fn string(mut self, key: &str, value: &str) -> Self {
            Self::push_str(&mut self.meta, key);
            self.meta.extend_from_slice(&8u32.to_le_bytes()); // STRING
            Self::push_str(&mut self.meta, value);
            self.meta_count += 1;
            self
        }

        fn u32(mut self, key: &str, value: u32) -> Self {
            Self::push_str(&mut self.meta, key);
            self.meta.extend_from_slice(&4u32.to_le_bytes()); // UINT32
            self.meta.extend_from_slice(&value.to_le_bytes());
            self.meta_count += 1;
            self
        }

        /// A 2-D weight tensor of `qtype`, carrying `bytes` of payload.
        fn tensor(mut self, name: &str, dims: [u64; 2], qtype: u32, bytes: usize) -> Self {
            self.tensors
                .push((name.to_string(), dims.to_vec(), qtype, vec![0u8; bytes]));
            self
        }

        fn write(self, path: &std::path::Path) {
            let mut data = Vec::new();
            data.extend_from_slice(b"GGUF");
            data.extend_from_slice(&3u32.to_le_bytes());
            data.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
            data.extend_from_slice(&self.meta_count.to_le_bytes());
            data.extend_from_slice(&self.meta);

            let mut offset = 0u64;
            for (name, dims, qtype, payload) in &self.tensors {
                Self::push_str(&mut data, name);
                data.extend_from_slice(&(dims.len() as u32).to_le_bytes());
                for dim in dims.iter().rev() {
                    data.extend_from_slice(&dim.to_le_bytes());
                }
                data.extend_from_slice(&qtype.to_le_bytes());
                data.extend_from_slice(&offset.to_le_bytes());
                offset += payload.len() as u64;
            }
            let aligned = data.len().div_ceil(32) * 32;
            data.resize(aligned, 0);
            for (_, _, _, payload) in &self.tensors {
                data.extend_from_slice(payload);
            }
            std::fs::write(path, &data).expect("write synthetic gguf");
        }
    }

    /// A qwen2 GGUF declaring `heads` query heads and `kv_heads` KV heads,
    /// whose weights are stored as `qtype`.
    fn synth_gguf(path: &std::path::Path, heads: u32, kv_heads: u32, qtype: u32, bytes: usize) {
        TinyGguf::new()
            .string("general.architecture", "qwen2")
            .u32("qwen2.embedding_length", 896)
            .u32("qwen2.block_count", 24)
            .u32("qwen2.feed_forward_length", 4864)
            .u32("qwen2.attention.head_count", heads)
            .u32("qwen2.attention.head_count_kv", kv_heads)
            .tensor("blk.0.attn_q.weight", [64, 64], qtype, bytes)
            .tensor("blk.0.ffn_down.weight", [64, 64], qtype, bytes)
            .write(path);
    }

    /// FALSIFIER (#2444 finding 2): the KV head count `ptx-map` prints must be
    /// the one the file declares.
    ///
    /// It used to be reconstructed from a table keyed on the QUERY head count
    /// (28 → 4, 12 → 2, else → num_heads). Every model outside that table got
    /// its query head count reported as its KV head count: Qwen2.5-0.5B showed
    /// 14 (metadata says 2), Qwen3-8B showed 32 (says 8), and the error
    /// propagated into the printed QKV projection shape. The 1.5B fixture was
    /// right BY LUCK — one of the two hardcoded arms — so a single-model probe
    /// read as healthy. Two models, both off the table, is the point.
    #[test]
    #[cfg(feature = "inference")]
    fn kv_head_count_is_read_from_the_file_not_derived_from_query_heads() {
        let dir = std::env::temp_dir().join(format!("ptxmap-kv-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmpdir");

        for (heads, kv_heads) in [(14u32, 2u32), (32, 8)] {
            let path = dir.join(format!("m{heads}.gguf"));
            synth_gguf(&path, heads, kv_heads, 0, 64 * 64 * 4);
            let info = extract_model_info(&path).expect("synthetic qwen2 gguf must load");
            assert_eq!(
                info.num_kv_heads, kv_heads,
                "ptx-map must report the file's {kv_heads} KV heads, not {heads}"
            );
            assert_eq!(info.num_heads, heads);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// FALSIFIER (#2444 finding 3): the same bytes must get the same
    /// quantization label under any file name.
    ///
    /// The label used to be a case-sensitive substring match on the file name,
    /// so `ln -s q4_k_m.gguf totally-not-Q8_0.gguf` reported Q8_0 for the very
    /// same file, and a name with no quant token fell through to a hardcoded
    /// "Q4_K" default. Both names below carry a LIE; the answer must come from
    /// the tensors, which are F32 here.
    #[test]
    #[cfg(feature = "inference")]
    fn quantization_comes_from_the_tensors_not_the_file_name() {
        let dir = std::env::temp_dir().join(format!("ptxmap-quant-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmpdir");

        let lying = dir.join("totally-not-Q8_0.gguf");
        let mute = dir.join("mystery.gguf");
        synth_gguf(&lying, 14, 2, 0, 64 * 64 * 4); // qtype 0 = F32
        synth_gguf(&mute, 14, 2, 0, 64 * 64 * 4);

        let a = extract_model_info(&lying).expect("load");
        let b = extract_model_info(&mute).expect("load");
        assert_eq!(
            a.quant.as_str(),
            "F32",
            "quantization must be measured from the tensors, not read off the name"
        );
        assert_eq!(
            a.quant.as_str(),
            b.quant.as_str(),
            "renaming a file must not change what ptx-map says is inside it"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The modal-qtype rule itself: 1-D tensors (norms, biases) are not matmul
    /// weights and must not vote, and an unnameable qtype yields no label
    /// rather than a plausible default.
    #[test]
    fn dominant_weight_quant_counts_only_matmul_weights() {
        let unknown = MeasuredQuant::from_qtype(12).is_none();
        if unknown {
            return; // built without `inference`: no qtype table to consult
        }
        // Two Q6_K matmuls, one Q4_K matmul, and four 1-D Q4_K tensors that
        // must not outvote them.
        let tensors = vec![
            (2usize, 14u32),
            (2, 14),
            (2, 12),
            (1, 12),
            (1, 12),
            (1, 12),
            (1, 12),
        ];
        assert_eq!(
            dominant_weight_quant(tensors.into_iter()).map(MeasuredQuant::as_str),
            Some("Q6_K")
        );
        assert_eq!(
            dominant_weight_quant(std::iter::once((2usize, 9999u32))),
            None,
            "an unnameable qtype must produce no label at all"
        );
        assert_eq!(dominant_weight_quant(std::iter::empty()), None);
    }
}
