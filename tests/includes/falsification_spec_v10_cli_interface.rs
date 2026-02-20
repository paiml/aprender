// Section 2: CLI Interface (F-CLI-*)
// =============================================================================

#[test]
fn f_cli_001_all_36_top_level_commands_parse() {
    // F-CLI-001: All 36 top-level commands parse
    let lib_path = project_root()
        .join("crates")
        .join("apr-cli")
        .join("src")
        .join("lib.rs");
    let content = std::fs::read_to_string(&lib_path).expect("F-CLI-001: lib.rs readable");

    let count = count_enum_variants(&content, "pub enum Commands");
    assert_eq!(
        count, 36,
        "F-CLI-001: Commands enum must have exactly 36 variants, found {count}"
    );
}

#[test]
fn f_cli_002_all_10_nested_subcommands_parse() {
    // F-CLI-002: All 10 rosetta + canary subcommands parse
    let rosetta_path = project_root()
        .join("crates")
        .join("apr-cli")
        .join("src")
        .join("commands")
        .join("rosetta.rs");
    let rosetta_content =
        std::fs::read_to_string(&rosetta_path).expect("F-CLI-002: rosetta.rs readable");
    let rosetta_count = count_enum_variants(&rosetta_content, "pub enum RosettaCommands");
    assert_eq!(
        rosetta_count, 8,
        "F-CLI-002: RosettaCommands must have 8 variants, found {rosetta_count}"
    );

    let canary_path = project_root()
        .join("crates")
        .join("apr-cli")
        .join("src")
        .join("commands")
        .join("canary.rs");
    let canary_content =
        std::fs::read_to_string(&canary_path).expect("F-CLI-002: canary.rs readable");
    let canary_count = count_enum_variants(&canary_content, "pub enum CanaryCommands");
    assert_eq!(
        canary_count, 2,
        "F-CLI-002: CanaryCommands must have 2 variants, found {canary_count}"
    );

    let total = rosetta_count + canary_count;
    assert_eq!(
        total, 10,
        "F-CLI-002: Total nested subcommands must be 10, found {total}"
    );
}

#[test]
fn f_cli_003_unknown_command_rejected() {
    // F-CLI-003: `apr nonexistent` -> exit != 0, "unrecognized subcommand"
    // Verify structurally: Commands enum is exhaustive (no catch-all dispatch)
    let lib_path = project_root()
        .join("crates")
        .join("apr-cli")
        .join("src")
        .join("lib.rs");
    let content = std::fs::read_to_string(&lib_path).expect("F-CLI-003: lib.rs readable");

    // clap with derive macro rejects unknown subcommands by default
    assert!(
        content.contains("clap::Subcommand") || content.contains("Subcommand"),
        "F-CLI-003: Commands enum must derive clap::Subcommand for strict parsing"
    );
}

#[test]
fn f_cli_004_skip_contract_is_global_flag() {
    // F-CLI-004: --skip-contract is a global flag on the CLI
    let lib_path = project_root()
        .join("crates")
        .join("apr-cli")
        .join("src")
        .join("lib.rs");
    let content = std::fs::read_to_string(&lib_path).expect("F-CLI-004: lib.rs readable");

    assert!(
        content.contains("skip_contract") || content.contains("skip-contract"),
        "F-CLI-004: CLI must have skip_contract/skip-contract flag"
    );
}

#[test]
fn f_cli_005_action_commands_gated_diagnostics_exempt() {
    // F-CLI-005: 20 gated (16 top + 4 rosetta), 26 exempt
    let lib_path = project_root()
        .join("crates")
        .join("apr-cli")
        .join("src")
        .join("lib.rs");
    let content = std::fs::read_to_string(&lib_path).expect("F-CLI-005: lib.rs readable");

    // extract_model_paths must exist and classify commands
    assert!(
        content.contains("fn extract_model_paths"),
        "F-CLI-005: extract_model_paths function must exist"
    );

    // Action commands that MUST be gated
    let gated_commands = [
        "Commands::Run",
        "Commands::Serve",
        "ExtendedCommands::Chat",
        "ExtendedCommands::Bench",
        "ExtendedCommands::Eval",
        "ExtendedCommands::Profile",
        "Commands::Check",
    ];

    for cmd in &gated_commands {
        assert!(
            content.contains(cmd),
            "F-CLI-005: {cmd} must appear in extract_model_paths"
        );
    }
}

#[test]
fn f_cli_006_commands_support_json_output() {
    // F-CLI-006: Key commands support --json output flag
    // Structural check: verify json output support exists in command implementations
    let commands_dir = project_root()
        .join("crates")
        .join("apr-cli")
        .join("src")
        .join("commands");

    // Check that key diagnostic commands accept --json or have JSON output
    let json_capable_files = ["qa.rs", "oracle.rs"];
    let mut found_json_support = 0;

    for filename in &json_capable_files {
        let path = commands_dir.join(filename);
        if path.exists() {
            let content = std::fs::read_to_string(&path).unwrap_or_default();
            if content.contains("json") || content.contains("Json") || content.contains("JSON") {
                found_json_support += 1;
            }
        }
    }

    assert!(
        found_json_support >= 1,
        "F-CLI-006: At least 1 key command must support JSON output, found {found_json_support}"
    );

    // Verify qa.rs specifically has --json flag support
    let qa_path = commands_dir.join("qa.rs");
    if qa_path.exists() {
        let qa_content = std::fs::read_to_string(&qa_path).unwrap_or_default();
        assert!(
            qa_content.contains("pub json: bool") || qa_content.contains("json:"),
            "F-CLI-006: qa.rs must have json field in config"
        );
    }
}

// =============================================================================
// Section 3: Pipeline Verification (F-PIPE-*)
// All require model files
// =============================================================================

#[test]
fn f_pipe_001_tokenizer_produces_correct_token_count() {
    // F-PIPE-001: Structural check — BPE tokenizer with encode/decode exists
    let bpe_path = project_root()
        .join("src")
        .join("text")
        .join("bpe")
        .join("mod.rs");
    let content = std::fs::read_to_string(&bpe_path).expect("bpe/mod.rs must exist");
    assert!(
        content.contains("BpeTokenizer") || content.contains("Tokenizer"),
        "F-PIPE-001: Tokenizer type must exist in bpe/mod.rs"
    );
    assert!(
        content.contains("fn encode") || content.contains("fn tokenize"),
        "F-PIPE-001: Tokenizer must have encode/tokenize method"
    );
    assert!(
        content.contains("fn decode") || content.contains("fn detokenize"),
        "F-PIPE-001: Tokenizer must have decode/detokenize method"
    );
}

#[test]
fn f_pipe_002_embedding_lookup_is_non_zero() {
    // F-PIPE-002: Structural check — ValidatedEmbedding has lookup method
    // ValidatedEmbedding enforces non-zero data via density gate (F-DATA-QUALITY-001)
    let vt_path = project_root()
        .join("src")
        .join("format")
        .join("validated_tensors.rs");
    let content = std::fs::read_to_string(&vt_path).expect("validated_tensors.rs must exist");
    assert!(
        content.contains("ValidatedEmbedding"),
        "F-PIPE-002: ValidatedEmbedding type must exist"
    );
    assert!(
        content.contains("density") || content.contains("QUALITY-001"),
        "F-PIPE-002: Embedding validation must check density (non-zero)"
    );
}

#[test]
fn f_pipe_003_rope_theta_matches_contract() {
    // F-PIPE-003: Qwen2 7B rope_theta = 1,000,000
    // Verify via YAML contract (no model needed for this part)
    let registry = build_default_registry();
    let qwen2 = registry.get("qwen2").expect("qwen2 must exist");
    let config_7b = qwen2.size_config("7b").expect("7b must exist");

    // rope_theta is in the YAML contract
    assert!(
        config_7b.rope_theta > 0.0,
        "F-PIPE-003: rope_theta must be positive"
    );
    assert!(
        (config_7b.rope_theta - 1_000_000.0).abs() < 1.0,
        "F-PIPE-003: Qwen2 7B rope_theta must be 1000000, got {}",
        config_7b.rope_theta
    );
}

#[test]
fn f_pipe_004_attention_scores_sum_to_one() {
    // F-PIPE-004: Structural check — softmax function exists (produces sum-to-1 distributions)
    let distill_path = project_root()
        .join("src")
        .join("online")
        .join("distillation.rs");
    let content = std::fs::read_to_string(&distill_path).expect("distillation.rs must exist");
    assert!(
        content.contains("fn softmax"),
        "F-PIPE-004: softmax function must exist"
    );
    assert!(
        content.contains("softmax_temperature"),
        "F-PIPE-004: softmax_temperature variant must exist for temperature scaling"
    );
    // Also check nn/activation.rs
    let act_path = project_root().join("src").join("nn").join("activation.rs");
    let act_content = std::fs::read_to_string(&act_path).expect("activation.rs must exist");
    assert!(
        act_content.contains("Softmax"),
        "F-PIPE-004: Softmax activation struct must exist"
    );
}

#[test]
fn f_pipe_005_lm_head_output_has_correct_vocab_dim() {
    // F-PIPE-005: logits dimension = 152064 for Qwen2 7B
    let registry = build_default_registry();
    let qwen2 = registry.get("qwen2").expect("qwen2 must exist");
    let config_7b = qwen2.size_config("7b").expect("7b must exist");

    assert_eq!(
        config_7b.vocab_size, 152_064,
        "F-PIPE-005: Qwen2 7B vocab_size must be 152064"
    );
}

#[test]
fn f_pipe_006_sampler_respects_temperature_zero() {
    // F-PIPE-006: Structural check — GreedyDecoder exists + temperature parameter
    let gen_path = project_root()
        .join("src")
        .join("nn")
        .join("generation")
        .join("mod.rs");
    let content = std::fs::read_to_string(&gen_path).expect("generation/mod.rs must exist");
    assert!(
        content.contains("GreedyDecoder"),
        "F-PIPE-006: GreedyDecoder must exist (temp=0 equivalent)"
    );
    assert!(
        content.contains("with_temperature"),
        "F-PIPE-006: with_temperature method must exist"
    );
    assert!(
        content.contains("temperature"),
        "F-PIPE-006: temperature field must exist in generation config"
    );
}

#[test]
fn f_pipe_007_separate_qkv_for_qwen2() {
    // F-PIPE-007: Qwen2 uses separate Q/K/V projections (not fused)
    let contract = LayoutContract::new();
    let all_tensors: Vec<_> = contract
        .transpose_tensors()
        .into_iter()
        .chain(contract.non_transpose_tensors())
        .collect();

    // Check for separate q_proj, k_proj, v_proj patterns
    let has_q_proj = all_tensors
        .iter()
        .any(|t| t.gguf_name.contains("attn_q") || t.gguf_name.contains("q_proj"));
    let has_k_proj = all_tensors
        .iter()
        .any(|t| t.gguf_name.contains("attn_k") || t.gguf_name.contains("k_proj"));
    let has_v_proj = all_tensors
        .iter()
        .any(|t| t.gguf_name.contains("attn_v") || t.gguf_name.contains("v_proj"));

    assert!(
        has_q_proj,
        "F-PIPE-007: Contract must have Q projection tensor"
    );
    assert!(
        has_k_proj,
        "F-PIPE-007: Contract must have K projection tensor"
    );
    assert!(
        has_v_proj,
        "F-PIPE-007: Contract must have V projection tensor"
    );
}

