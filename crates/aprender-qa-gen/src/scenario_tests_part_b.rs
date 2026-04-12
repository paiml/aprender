/// Verify AprTool::command returns correct string for each variant
#[test]
fn test_apr_tool_command() {
    assert_eq!(AprTool::Run.command(), "run");
    assert_eq!(AprTool::Chat.command(), "chat");
    assert_eq!(AprTool::Serve.command(), "serve");
    assert_eq!(AprTool::Inspect.command(), "inspect");
    assert_eq!(AprTool::Validate.command(), "validate");
    assert_eq!(AprTool::Bench.command(), "bench");
    assert_eq!(AprTool::Profile.command(), "profile");
    assert_eq!(AprTool::Trace.command(), "trace");
    assert_eq!(AprTool::Check.command(), "check");
    assert_eq!(AprTool::Canary.command(), "canary");
}

/// Verify AprTool::requires_prompt returns true only for Run and Chat
#[test]
fn test_apr_tool_requires_prompt() {
    assert!(AprTool::Run.requires_prompt());
    assert!(AprTool::Chat.requires_prompt());
    assert!(!AprTool::Serve.requires_prompt());
    assert!(!AprTool::Inspect.requires_prompt());
    assert!(!AprTool::Validate.requires_prompt());
    assert!(!AprTool::Bench.requires_prompt());
}

/// Verify AprTool::supports_trace returns true only for Run and Trace
#[test]
fn test_apr_tool_supports_trace() {
    assert!(AprTool::Run.supports_trace());
    assert!(AprTool::Trace.supports_trace());
    assert!(!AprTool::Chat.supports_trace());
    assert!(!AprTool::Serve.supports_trace());
}

/// Verify AprTool Display trait formats as lowercase command string
#[test]
fn test_apr_tool_display() {
    assert_eq!(format!("{}", AprTool::Run), "run");
    assert_eq!(format!("{}", AprTool::Profile), "profile");
    assert_eq!(format!("{}", AprTool::Canary), "canary");
}

/// Verify Format::extension returns correct file extension for each variant
#[test]
fn test_format_extension() {
    assert_eq!(Format::Gguf.extension(), ".gguf");
    assert_eq!(Format::SafeTensors.extension(), ".safetensors");
    assert_eq!(Format::Apr.extension(), ".apr");
}

/// Verify TraceLevel::all returns all four trace level variants
#[test]
fn test_trace_level_all() {
    let all = TraceLevel::all();
    assert_eq!(all.len(), 4);
    assert!(all.contains(&TraceLevel::None));
    assert!(all.contains(&TraceLevel::Basic));
    assert!(all.contains(&TraceLevel::Layer));
    assert!(all.contains(&TraceLevel::Payload));
}

/// Verify Modality::command returns correct string for each variant
#[test]
fn test_modality_command() {
    assert_eq!(Modality::Run.command(), "run");
    assert_eq!(Modality::Chat.command(), "chat");
    assert_eq!(Modality::Serve.command(), "serve");
}

/// Verify QaScenario Debug format contains struct name
#[test]
fn test_scenario_debug() {
    let model = ModelId::new("test", "model");
    let scenario = QaScenario::new(
        model,
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );
    let debug_str = format!("{scenario:?}");
    assert!(debug_str.contains("QaScenario"));
}

/// Verify ScenarioGenerator Debug format contains struct name
#[test]
fn test_scenario_generator_debug() {
    let model = ModelId::new("test", "model");
    let generator = ScenarioGenerator::new(model);
    let debug_str = format!("{generator:?}");
    assert!(debug_str.contains("ScenarioGenerator"));
}

/// Verify escape_json escapes tab characters per RFC 8259
#[test]
fn test_escape_json_tab() {
    let result = escape_json("hello\tworld");
    assert_eq!(result, "hello\\tworld");
    assert!(!result.contains('\t'));
}

/// Verify escape_json escapes carriage return
#[test]
fn test_escape_json_cr() {
    let result = escape_json("line1\rline2");
    assert_eq!(result, "line1\\rline2");
}

/// Verify escape_json escapes other control characters as unicode escapes
#[test]
fn test_escape_json_control_chars() {
    let result = escape_json("null\x00byte");
    assert_eq!(result, "null\\u0000byte");
}

/// Verify escape_prompt returns input unchanged when no quotes present
#[test]
fn test_escape_prompt_no_quotes() {
    let result = escape_prompt("hello world");
    assert_eq!(result, "hello world");
}

/// Verify to_command includes trace flags when basic trace level is set
#[test]
fn test_scenario_to_command_with_basic_trace() {
    let model = ModelId::new("test", "model");
    let scenario = QaScenario::new(
        model,
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "Hello".to_string(),
        0,
    )
    .with_trace_level(TraceLevel::Basic);

    let cmd = scenario.to_command("model.gguf");
    assert!(cmd.contains("--trace"));
    assert!(cmd.contains("--trace-level basic"));
}

/// Verify to_command includes layer-level trace flag
#[test]
fn test_scenario_to_command_with_layer_trace() {
    let model = ModelId::new("test", "model");
    let scenario = QaScenario::new(
        model,
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "Hello".to_string(),
        0,
    )
    .with_trace_level(TraceLevel::Layer);

    let cmd = scenario.to_command("model.gguf");
    assert!(cmd.contains("--trace-level layer"));
}

/// Verify with_kernel_profile appends profile prompts to generator
#[test]
fn test_with_kernel_profile() {
    let model = ModelId::new("test", "model");
    let constraints = crate::kernel_profile::ArchConstraints {
        attention_type: Some("gqa".to_string()),
        activation: Some("silu".to_string()),
        norm_type: Some("rmsnorm".to_string()),
        has_bias: Some(true),
        ..crate::kernel_profile::ArchConstraints::default()
    };
    let profile = crate::kernel_profile::profile_from_constraints("test", &constraints, None);
    let original_len = default_prompts().len();
    let profile_prompts = profile.all_prompts().len();

    let generator = ScenarioGenerator::new(model).with_kernel_profile(&profile);
    assert_eq!(generator.prompts.len(), original_len + profile_prompts);
}

// --- Mutation-killing tests for mqs_category return values ---
/// Verify Run+Cpu scenario maps to MQS category A1
#[test]
fn test_mqs_category_run_cpu_is_a1() {
    let model = ModelId::new("t", "m");
    let s = QaScenario::new(
        model,
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "x".into(),
        0,
    );
    assert_eq!(s.mqs_category(), "A1");
    assert_ne!(s.mqs_category(), "A2");
    assert_ne!(s.mqs_category(), "A3");
}

/// Verify Run+Gpu scenario maps to MQS category A2
#[test]
fn test_mqs_category_run_gpu_is_a2() {
    let model = ModelId::new("t", "m");
    let s = QaScenario::new(
        model,
        Modality::Run,
        Backend::Gpu,
        Format::Gguf,
        "x".into(),
        0,
    );
    assert_eq!(s.mqs_category(), "A2");
    assert_ne!(s.mqs_category(), "A1");
}

/// Verify Chat+Cpu scenario maps to MQS category A3
#[test]
fn test_mqs_category_chat_cpu_is_a3() {
    let model = ModelId::new("t", "m");
    let s = QaScenario::new(
        model,
        Modality::Chat,
        Backend::Cpu,
        Format::Gguf,
        "x".into(),
        0,
    );
    assert_eq!(s.mqs_category(), "A3");
    assert_ne!(s.mqs_category(), "A4");
}

/// Verify Chat+Gpu scenario maps to MQS category A4
#[test]
fn test_mqs_category_chat_gpu_is_a4() {
    let model = ModelId::new("t", "m");
    let s = QaScenario::new(
        model,
        Modality::Chat,
        Backend::Gpu,
        Format::Gguf,
        "x".into(),
        0,
    );
    assert_eq!(s.mqs_category(), "A4");
    assert_ne!(s.mqs_category(), "A3");
}

/// Verify Serve+Cpu scenario maps to MQS category A5
#[test]
fn test_mqs_category_serve_cpu_is_a5() {
    let model = ModelId::new("t", "m");
    let s = QaScenario::new(
        model,
        Modality::Serve,
        Backend::Cpu,
        Format::Gguf,
        "x".into(),
        0,
    );
    assert_eq!(s.mqs_category(), "A5");
    assert_ne!(s.mqs_category(), "A6");
}

/// Verify Serve+Gpu scenario maps to MQS category A6
#[test]
fn test_mqs_category_serve_gpu_is_a6() {
    let model = ModelId::new("t", "m");
    let s = QaScenario::new(
        model,
        Modality::Serve,
        Backend::Gpu,
        Format::Gguf,
        "x".into(),
        0,
    );
    assert_eq!(s.mqs_category(), "A6");
    assert_ne!(s.mqs_category(), "A5");
}

// --- Mutation-killing tests for Format::class return values ---
/// Verify Format::Gguf class returns 'A'
#[test]
fn test_format_class_gguf_is_char_a() {
    let class = Format::Gguf.class();
    assert_eq!(class, 'A');
    assert_ne!(class, 'B');
    assert_ne!(class, 'X');
}

/// Verify Format::Apr class returns 'A'
#[test]
fn test_format_class_apr_is_char_a() {
    let class = Format::Apr.class();
    assert_eq!(class, 'A');
    assert_ne!(class, 'B');
}

/// Verify Format::SafeTensors class returns 'B'
#[test]
fn test_format_class_safetensors_is_char_b() {
    let class = Format::SafeTensors.class();
    assert_eq!(class, 'B');
    assert_ne!(class, 'A');
}

// --- Mutation-killing tests for escape_json ---
/// Verify escape_json doubles backslashes
#[test]
fn test_escape_json_backslash_not_empty() {
    let result = escape_json("a\\b");
    assert!(!result.is_empty());
    assert_eq!(result, "a\\\\b");
    assert!(result.len() > "a\\b".len());
}

/// Verify escape_json escapes double quotes
#[test]
fn test_escape_json_quote_not_empty() {
    let result = escape_json("say \"hi\"");
    assert!(!result.is_empty());
    assert_eq!(result, "say \\\"hi\\\"");
}

/// Verify escape_json replaces newlines with literal \n
#[test]
fn test_escape_json_newline_not_empty() {
    let result = escape_json("line1\nline2");
    assert!(!result.is_empty());
    assert_eq!(result, "line1\\nline2");
    assert!(!result.contains('\n'));
}

/// Verify escape_json handles combined backslash, quote, and newline escapes
#[test]
fn test_escape_json_all_escapes_combined() {
    let result = escape_json("a\\b\"c\nd");
    assert_eq!(result, "a\\\\b\\\"c\\nd");
}

// --- Mutation-killing tests for escape_prompt ---
/// Verify escape_prompt escapes single quotes for shell safety
#[test]
fn test_escape_prompt_single_quote() {
    let result = escape_prompt("it's");
    assert!(!result.is_empty());
    assert_eq!(result, "it'\\''s");
    assert!(result.contains("'\\''"));
}

// --- Test that Backend::flag returns correct strings ---
/// Verify Backend::Cpu flag returns empty string
#[test]
fn test_backend_cpu_flag_is_empty() {
    let flag = Backend::Cpu.flag();
    assert!(flag.is_empty());
    assert_eq!(flag, "");
}

/// Verify Backend::Gpu flag returns "--gpu"
#[test]
fn test_backend_gpu_flag_is_gpu_option() {
    let flag = Backend::Gpu.flag();
    assert!(!flag.is_empty());
    assert_eq!(flag, "--gpu");
    assert!(flag.starts_with("--"));
}

// --- Test TraceLevel::value returns correct strings ---
/// Verify TraceLevel::None value returns "none"
#[test]
fn test_trace_level_none_value() {
    assert_eq!(TraceLevel::None.value(), "none");
    assert_ne!(TraceLevel::None.value(), "basic");
}

/// Verify TraceLevel::Basic value returns "basic"
#[test]
fn test_trace_level_basic_value() {
    assert_eq!(TraceLevel::Basic.value(), "basic");
    assert_ne!(TraceLevel::Basic.value(), "none");
}

/// Verify TraceLevel::Layer value returns "layer"
#[test]
fn test_trace_level_layer_value() {
    assert_eq!(TraceLevel::Layer.value(), "layer");
    assert_ne!(TraceLevel::Layer.value(), "payload");
}

/// Verify TraceLevel::Payload value returns "payload"
#[test]
fn test_trace_level_payload_value() {
    assert_eq!(TraceLevel::Payload.value(), "payload");
    assert_ne!(TraceLevel::Payload.value(), "layer");
}

/// all_with_transformations returns exactly 7 modalities including all transformation types.
#[test]
fn test_modality_all_with_transformations_count_and_contents() {
    let all = Modality::all_with_transformations();
    assert_eq!(all.len(), 7, "Expected exactly 7 modalities including transformations");

    // Inference modalities
    assert!(all.contains(&Modality::Run));
    assert!(all.contains(&Modality::Chat));
    assert!(all.contains(&Modality::Serve));

    // Transformation modalities
    assert!(all.contains(&Modality::Quantize));
    assert!(all.contains(&Modality::Import));
    assert!(all.contains(&Modality::Prune));
    assert!(all.contains(&Modality::Distill));
}

/// all_with_transformations includes all 4 transformation modalities.
#[test]
fn test_modality_all_with_transformations_are_transformations() {
    let all = Modality::all_with_transformations();
    let transformation_count = all.iter().filter(|m| m.is_transformation()).count();
    assert_eq!(transformation_count, 4, "Expected 4 transformation modalities");
}

