#[test]
fn test_scenario_creation() {
    let model = ModelId::new("Qwen", "Qwen2.5-Coder-1.5B");
    let scenario = QaScenario::new(
        model.clone(),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "2+2=".to_string(),
        42,
    );

    assert_eq!(scenario.modality, Modality::Run);
    assert_eq!(scenario.backend, Backend::Cpu);
    assert_eq!(scenario.format, Format::Gguf);
    assert_eq!(scenario.oracle_type, "arithmetic");
}

#[test]
fn test_scenario_to_command_run() {
    let model = ModelId::new("test", "model");
    let scenario = QaScenario::new(
        model,
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "Hello".to_string(),
        0,
    );

    let cmd = scenario.to_command("model.gguf");
    assert!(cmd.contains("apr run"));
    assert!(cmd.contains("model.gguf"));
    assert!(cmd.contains("Hello"));
}

#[test]
fn test_scenario_to_command_gpu() {
    let model = ModelId::new("test", "model");
    let scenario = QaScenario::new(
        model,
        Modality::Run,
        Backend::Gpu,
        Format::Gguf,
        "Hello".to_string(),
        0,
    );

    let cmd = scenario.to_command("model.gguf");
    assert!(cmd.contains("--gpu"));
}

#[test]
fn test_scenario_generator() {
    let model = ModelId::new("test", "model");
    let generator = ScenarioGenerator::new(model).with_scenarios_per_combination(10);

    let scenarios = generator.generate();

    // 3 modalities × 2 backends × 3 formats × 10 = 180
    assert_eq!(scenarios.len(), 180);
}

#[test]
fn test_scenario_mqs_category() {
    let model = ModelId::new("test", "model");

    let run_cpu = QaScenario::new(
        model.clone(),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );
    assert_eq!(run_cpu.mqs_category(), "A1");

    let chat_gpu = QaScenario::new(
        model.clone(),
        Modality::Chat,
        Backend::Gpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );
    assert_eq!(chat_gpu.mqs_category(), "A4");

    let serve_cpu = QaScenario::new(
        model,
        Modality::Serve,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );
    assert_eq!(serve_cpu.mqs_category(), "A5");
}

#[test]
fn test_format_class() {
    assert_eq!(Format::Gguf.class(), 'A');
    assert_eq!(Format::Apr.class(), 'A');
    assert_eq!(Format::SafeTensors.class(), 'B');
}

#[test]
fn test_escape_prompt() {
    assert_eq!(escape_prompt("hello"), "hello");
    assert_eq!(escape_prompt("it's"), "it'\\''s");
}

#[test]
fn test_escape_json() {
    assert_eq!(escape_json("hello"), "hello");
    assert_eq!(escape_json("line1\nline2"), "line1\\nline2");
    assert_eq!(escape_json("say \"hi\""), "say \\\"hi\\\"");
}

#[test]
fn test_escape_json_backslash() {
    assert_eq!(escape_json("path\\file"), "path\\\\file");
}

#[test]
fn test_scenario_with_temperature() {
    let model = ModelId::new("test", "model");
    let scenario = QaScenario::new(
        model,
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    )
    .with_temperature(0.7);

    assert!((scenario.temperature - 0.7).abs() < f32::EPSILON);
}

#[test]
fn test_scenario_with_max_tokens() {
    let model = ModelId::new("test", "model");
    let scenario = QaScenario::new(
        model,
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    )
    .with_max_tokens(256);

    assert_eq!(scenario.max_tokens, 256);
}

#[test]
fn test_scenario_with_trace_level() {
    let model = ModelId::new("test", "model");
    let scenario = QaScenario::new(
        model,
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    )
    .with_trace_level(TraceLevel::Layer);

    assert_eq!(scenario.trace_level, TraceLevel::Layer);
}

#[test]
fn test_scenario_to_command_chat() {
    let model = ModelId::new("test", "model");
    let scenario = QaScenario::new(
        model,
        Modality::Chat,
        Backend::Cpu,
        Format::Gguf,
        "Hello".to_string(),
        0,
    );

    let cmd = scenario.to_command("model.gguf");
    assert!(cmd.contains("apr chat"));
    assert!(cmd.contains("echo"));
}

#[test]
fn test_scenario_to_command_serve() {
    let model = ModelId::new("test", "model");
    let scenario = QaScenario::new(
        model,
        Modality::Serve,
        Backend::Cpu,
        Format::Gguf,
        "Hello".to_string(),
        0,
    );

    let cmd = scenario.to_command("model.gguf");
    assert!(cmd.contains("apr serve"));
    assert!(cmd.contains("curl"));
    assert!(cmd.contains("/v1/completions"));
}

#[test]
fn test_scenario_to_command_with_trace() {
    let model = ModelId::new("test", "model");
    let scenario = QaScenario::new(
        model,
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "Hello".to_string(),
        0,
    )
    .with_trace_level(TraceLevel::Payload);

    let cmd = scenario.to_command("model.gguf");
    assert!(cmd.contains("--trace"));
    assert!(cmd.contains("--trace-level payload"));
}

#[test]
fn test_scenario_evaluate() {
    let model = ModelId::new("test", "model");
    let scenario = QaScenario::new(
        model,
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "2+2=".to_string(),
        0,
    );

    let result = scenario.evaluate("The answer is 4");
    assert!(matches!(result, crate::OracleResult::Corroborated { .. }));
}

#[test]
fn test_trace_level_value() {
    assert_eq!(TraceLevel::None.value(), "none");
    assert_eq!(TraceLevel::Basic.value(), "basic");
    assert_eq!(TraceLevel::Layer.value(), "layer");
    assert_eq!(TraceLevel::Payload.value(), "payload");
}

#[test]
fn test_modality_display() {
    assert_eq!(format!("{}", Modality::Run), "run");
    assert_eq!(format!("{}", Modality::Chat), "chat");
    assert_eq!(format!("{}", Modality::Serve), "serve");
}

#[test]
fn test_backend_flag() {
    assert_eq!(Backend::Cpu.flag(), "");
    assert_eq!(Backend::Gpu.flag(), "--gpu");
}

#[test]
fn test_backend_display() {
    assert_eq!(format!("{}", Backend::Cpu), "cpu");
    assert_eq!(format!("{}", Backend::Gpu), "gpu");
}

#[test]
fn test_format_display() {
    assert_eq!(format!("{}", Format::Gguf), "gguf");
    assert_eq!(format!("{}", Format::SafeTensors), "safetensors");
    assert_eq!(format!("{}", Format::Apr), "apr");
}

#[test]
fn test_modality_inference_modalities() {
    let all = Modality::inference_modalities();
    assert_eq!(all.len(), 3);
    assert!(all.contains(&Modality::Run));
    assert!(all.contains(&Modality::Chat));
    assert!(all.contains(&Modality::Serve));
    // Transformations are NOT included
    assert!(!all.contains(&Modality::Quantize));
    assert!(!all.contains(&Modality::Import));
}

#[test]
fn test_backend_all() {
    let all = Backend::all();
    assert_eq!(all.len(), 2);
    assert!(all.contains(&Backend::Cpu));
    assert!(all.contains(&Backend::Gpu));
}

#[test]
fn test_format_all() {
    let all = Format::all();
    assert_eq!(all.len(), 3);
    assert!(all.contains(&Format::Gguf));
    assert!(all.contains(&Format::SafeTensors));
    assert!(all.contains(&Format::Apr));
}

#[test]
fn test_generator_with_prompts() {
    let model = ModelId::new("test", "model");
    let prompts = vec!["prompt1".to_string(), "prompt2".to_string()];
    let generator = ScenarioGenerator::new(model)
        .with_prompts(prompts.clone())
        .with_scenarios_per_combination(2);

    assert_eq!(generator.prompts, prompts);
}

#[test]
fn test_generator_generate_for() {
    let model = ModelId::new("test", "model");
    let generator = ScenarioGenerator::new(model).with_scenarios_per_combination(5);

    let scenarios = generator.generate_for(Modality::Run, Backend::Cpu, Format::Gguf);
    assert_eq!(scenarios.len(), 5);

    for s in &scenarios {
        assert_eq!(s.modality, Modality::Run);
        assert_eq!(s.backend, Backend::Cpu);
        assert_eq!(s.format, Format::Gguf);
    }
}

#[test]
fn test_default_prompts_coverage() {
    let prompts = default_prompts();
    assert!(!prompts.is_empty());
    // Should have arithmetic prompts
    assert!(prompts.iter().any(|p| p.contains('+') || p.contains('*')));
    // Should have code prompts
    assert!(
        prompts
            .iter()
            .any(|p| p.starts_with("def ") || p.starts_with("fn "))
    );
    // Should have empty prompt for edge case
    assert!(prompts.iter().any(|p| p.is_empty()));
}

#[test]
fn test_mqs_category_all_combinations() {
    let model = ModelId::new("test", "model");

    // Run GPU
    let run_gpu = QaScenario::new(
        model.clone(),
        Modality::Run,
        Backend::Gpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );
    assert_eq!(run_gpu.mqs_category(), "A2");

    // Chat CPU
    let chat_cpu = QaScenario::new(
        model.clone(),
        Modality::Chat,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );
    assert_eq!(chat_cpu.mqs_category(), "A3");

    // Serve GPU
    let serve_gpu = QaScenario::new(
        model,
        Modality::Serve,
        Backend::Gpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );
    assert_eq!(serve_gpu.mqs_category(), "A6");
}

#[test]
fn test_scenario_clone() {
    let model = ModelId::new("test", "model");
    let scenario = QaScenario::new(
        model,
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        42,
    );

    let cloned = scenario.clone();
    assert_eq!(cloned.id, scenario.id);
    assert_eq!(cloned.seed, scenario.seed);
}

#[test]
fn test_scenario_serialize() {
    let model = ModelId::new("test", "model");
    let scenario = QaScenario::new(
        model,
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );

    let json = serde_json::to_string(&scenario).expect("serialize");
    assert!(json.contains("\"modality\":\"run\""));
    assert!(json.contains("\"backend\":\"cpu\""));
}

#[test]
fn test_apr_tool_all() {
    let all = AprTool::all();
    assert_eq!(all.len(), 10);
    assert!(all.contains(&AprTool::Run));
    assert!(all.contains(&AprTool::Canary));
}

