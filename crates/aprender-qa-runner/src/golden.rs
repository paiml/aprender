impl Executor {

    /// Run extended tests: conversion, golden rule, contracts, parity, perf, ollama, transformations.
    ///
    /// Jidoka (Issue #29): checks failure_policy between batteries. If `StopOnFirst`
    /// or `FailFast`, stops after the first battery that produces failures.
    /// If `StopOnP0`, stops after conversion failures (F-CONV-* are P0).
    fn run_extended_tests(&mut self, playbook: &Playbook) -> (usize, usize) {
        let mut total_passed = 0;
        let mut total_failed = 0;

        // Helper macro: run battery, check Jidoka stop condition
        macro_rules! run_battery {
            ($body:expr) => {{
                let (p, f) = $body;
                total_passed += p;
                total_failed += f;
                if f > 0 && self.config.failure_policy.stops_on_any_failure() {
                    return (total_passed, total_failed);
                }
            }};
        }

        // Jidoka: never silently skip an enabled battery — emit explicit evidence
        // when model_path is missing. (Bug #78)
        macro_rules! skip_no_model {
            ($gate:expr, $label:expr) => {{
                let ev = Evidence::skipped(
                    $gate,
                    QaScenario::new(
                        playbook.model_id(),
                        Modality::Run,
                        Backend::Cpu,
                        Format::SafeTensors,
                        concat!($label, ": model_path not configured").to_string(),
                        0,
                    ),
                    concat!($label, " skipped: model_path not set in ExecutionConfig"),
                );
                self.collector.add(ev);
            }};
        }

        if self.config.run_conversion_tests {
            if let Some(model_path) = self.config.model_path.clone() {
                let model_id = playbook.model_id();
                run_battery!(self.run_conversion_tests(Path::new(&model_path), &model_id));
            } else {
                skip_no_model!("F-CONV-SKIP-001", "Conversion tests");
            }
        }

        if self.config.run_golden_rule_test {
            if let Some(model_path) = self.config.model_path.clone() {
                let model_id = playbook.model_id();
                run_battery!(self.run_golden_rule_test(Path::new(&model_path), &model_id));
            } else {
                skip_no_model!("F-GOLDEN-RULE-SKIP-001", "Golden rule test");
            }
        }

        if self.config.run_contract_tests {
            if let Some(model_path) = self.config.model_path.clone() {
                let model_id = playbook.model_id();
                run_battery!(
                    self.run_contract_invariants(Path::new(&model_path), &model_id, playbook)
                );
            } else {
                skip_no_model!("F-CONTRACT-SKIP-001", "Contract tests");
            }
        }

        if self.config.run_hf_parity {
            let model_id = playbook.model_id();
            run_battery!(self.run_hf_parity_tests(&model_id));
        }

        if self.config.run_profile_ci {
            if let Some(model_path) = self.config.model_path.clone() {
                let model_id = playbook.model_id();
                run_battery!(self.run_perf_gates(Path::new(&model_path), &model_id, playbook));
            } else {
                skip_no_model!("F-PERF-SKIP-001", "Profile CI");
            }
        }

        if self.config.run_ollama_parity {
            if let Some(model_path) = self.config.model_path.clone() {
                run_battery!(self.run_ollama_parity_tests(Path::new(&model_path), playbook));
            } else {
                skip_no_model!("F-OLLAMA-SKIP-001", "Ollama parity");
            }
        }

        // Transformation tests (opt-in via playbook transformations: block)
        run_battery!(self.execute_transformation_tests(playbook));

        (total_passed, total_failed)
    }

    /// Tally evidence from a battery run: collect into the evidence store and
    /// return `(passed, failed)` counts. Skipped evidence is not counted as
    /// either passed or failed (Popper: only definitive outcomes count).
    fn tally_battery(&mut self, evidence_vec: Vec<Evidence>) -> (usize, usize) {
        let mut passed = 0;
        let mut failed = 0;
        for ev in evidence_vec {
            if ev.outcome == Outcome::Skipped {
                // Skipped is neither pass nor fail
            } else if ev.outcome.is_pass() {
                passed += 1;
            } else {
                failed += 1;
            }
            self.collector.add(ev);
        }
        (passed, failed)
    }

    /// Execute transformation tests from the playbook's `transformations:` block.
    ///
    /// Each transformation type (quantize, import, prune, distill) runs its
    /// dedicated battery of checks if configured. Returns `(passed, failed)`.
    fn execute_transformation_tests(&mut self, playbook: &Playbook) -> (usize, usize) {
        let Some(ref config) = playbook.transformations else {
            let model_id = playbook.model_id();
            let ev = Evidence::skipped(
                "F-TRANSFORM-SKIP-002",
                Self::golden_scenario(&model_id),
                "Transformation tests skipped: no transformations block in playbook",
            );
            self.collector.add(ev);
            return (0, 0);
        };
        let Some(ref model_path_str) = self.config.model_path else {
            let model_id = playbook.model_id();
            let ev = Evidence::skipped(
                "F-TRANSFORM-SKIP-001",
                Self::golden_scenario(&model_id),
                "Transformation tests skipped: model_path not set in ExecutionConfig",
            );
            self.collector.add(ev);
            return (0, 0);
        };
        let model_path = model_path_str.clone();
        let model_id = playbook.model_id();
        let mut passed = 0;
        let mut failed = 0;

        if let Some(ref q) = config.quantize {
            for scheme in &q.schemes {
                let scenario = QaScenario::new(
                    model_id.clone(), Modality::Quantize, Backend::Cpu,
                    Format::Apr, format!("quantize:{scheme}"), 0,
                );
                let ev = self.run_quantize_battery(&model_path, &scenario, scheme);
                let (p, f) = self.tally_battery(ev);
                passed += p;
                failed += f;
            }
        }

        if let Some(ref i) = config.import {
            for source_format in &i.source_formats {
                let scenario = QaScenario::new(
                    model_id.clone(), Modality::Import, Backend::Cpu,
                    Format::Apr, format!("import:{source_format}"), 0,
                );
                let ev = self.run_import_battery(&model_path, &scenario, source_format);
                let (p, f) = self.tally_battery(ev);
                passed += p;
                failed += f;
            }
        }

        if let Some(ref p_config) = config.prune {
            let scenario = QaScenario::new(
                model_id.clone(), Modality::Prune, Backend::Cpu,
                Format::Apr, format!("prune:{}:{}", p_config.method, p_config.target_ratio), 0,
            );
            let ev = self.run_prune_battery(&model_path, &scenario, &p_config.method, p_config.target_ratio);
            let (p, f) = self.tally_battery(ev);
            passed += p;
            failed += f;
        }

        if let Some(ref d) = config.distill {
            let scenario = QaScenario::new(
                model_id, Modality::Distill, Backend::Cpu,
                Format::Apr, "distill".to_string(), 0,
            );
            let ev = self.run_distill_battery(&model_path, &scenario, &d.student_model, &d.data_path);
            let (p, f) = self.tally_battery(ev);
            passed += p;
            failed += f;
        }

        (passed, failed)
    }

    /// Run P0 format conversion tests
    fn run_conversion_tests(&mut self, model_path: &Path, model_id: &ModelId) -> (usize, usize) {
        if model_path.is_file() {
            let ev = Evidence::skipped(
                "F-CONV-SKIP-002",
                Self::golden_scenario(model_id),
                "Conversion testing not applicable for single-file model (requires SafeTensors directory)",
            );
            self.collector.add(ev);
            return (0, 0);
        }

        let config = if self.config.no_gpu {
            ConversionConfig::cpu_only()
        } else {
            ConversionConfig::default()
        };

        // ISO-OUT-001: Use isolated output directory for conversion artifacts
        let executor = if let Some(ref output_dir) = self.config.output_dir {
            ConversionExecutor::new(config).with_output_dir(std::path::PathBuf::from(output_dir))
        } else {
            ConversionExecutor::new(config)
        };

        match executor.execute_all(model_path, model_id) {
            Ok(result) => {
                // Add all conversion evidence to collector
                for ev in result.evidence {
                    self.collector.add(ev);
                }
                (result.passed, result.failed)
            }
            Err(e) => {
                // Critical conversion infrastructure failure
                let ev = Evidence::falsified(
                    "F-CONV-INFRA-001",
                    apr_qa_gen::QaScenario::new(
                        model_id.clone(),
                        apr_qa_gen::Modality::Run,
                        apr_qa_gen::Backend::Cpu,
                        apr_qa_gen::Format::Gguf,
                        "Conversion infrastructure".to_string(),
                        0,
                    ),
                    format!("Conversion infrastructure failure: {e}"),
                    "N/A",
                    0,
                );
                self.collector.add(ev);
                (0, 1)
            }
        }
    }

    /// Golden Rule Test: convert model, run inference, diff against original.
    ///
    /// This is the SINGLE MOST IMPORTANT test in the entire pipeline.
    /// It encodes the only invariant that matters for format conversion:
    ///   "Converted models MUST produce the same output as the original."
    ///
    /// Would have caught: GH-186, GH-189, GH-190 (all 3 P0 conversion bugs).
    /// See: docs/five-whys/GH-190-systemic-conversion-failures.md
    fn run_golden_rule_test(&mut self, model_path: &Path, model_id: &ModelId) -> (usize, usize) {
        // Skip for actual single-file models (not applicable - no conversion to test)
        if model_path.is_file() {
            let ev = Evidence::skipped(
                "F-GOLDEN-RULE-SKIP-002",
                Self::golden_scenario(model_id),
                "Golden rule test not applicable for single-file model (requires SafeTensors directory)",
            );
            self.collector.add(ev);
            return (0, 0);
        }

        // For mock testing: if path has model extension but doesn't exist, run with path directly
        let has_model_extension = model_path
            .extension()
            .is_some_and(|e| ["gguf", "safetensors", "apr"].contains(&e.to_str().unwrap_or("")));
        if has_model_extension {
            return self.run_golden_rule_with_path(model_path, model_id);
        }

        // Resolve directory to SafeTensors model file (ground truth)
        let resolved_path = match resolve_model_path(model_path, apr_qa_gen::Format::SafeTensors) {
            Ok(p) => p,
            Err(e) => {
                let ev = Evidence::falsified(
                    "F-GOLDEN-RULE-001",
                    Self::golden_scenario(model_id),
                    format!("Golden Rule: failed to resolve model path: {e}"),
                    "N/A",
                    0,
                );
                self.collector.add(ev);
                return (0, 1);
            }
        };

        self.run_golden_rule_with_path(&resolved_path, model_id)
    }

    /// Internal helper for golden rule test with resolved path
    fn run_golden_rule_with_path(
        &mut self,
        model_path: &Path,
        model_id: &ModelId,
    ) -> (usize, usize) {
        let prompt = "What is 2+2?";
        let max_tokens = 10;

        // Step 1: Run inference on original model (SafeTensors ground truth)
        let original_result =
            self.command_runner
                .run_inference(model_path, prompt, max_tokens, false, &[]);

        if !original_result.success {
            let ev = Evidence::falsified(
                "F-GOLDEN-RULE-001",
                Self::golden_scenario(model_id),
                format!(
                    "Golden Rule: original inference failed: {}",
                    original_result.stderr
                ),
                "N/A",
                0,
            );
            self.collector.add(ev);
            return (0, 1);
        }

        // Step 2: Convert to APR
        let apr_path =
            std::path::PathBuf::from(format!("/tmp/golden-rule-test-{}.apr", model_id.name));
        let convert_result = self.command_runner.convert_model(model_path, &apr_path);

        if !convert_result.success {
            let ev = Evidence::falsified(
                "F-GOLDEN-RULE-002",
                Self::golden_scenario(model_id),
                format!("Golden Rule: conversion failed: {}", convert_result.stderr),
                "N/A",
                0,
            );
            self.collector.add(ev);
            return (0, 1);
        }

        // Step 3: Run inference on converted model
        let converted_result =
            self.command_runner
                .run_inference(&apr_path, prompt, max_tokens, false, &[]);

        if !converted_result.success {
            let ev = Evidence::falsified(
                "F-GOLDEN-RULE-003",
                Self::golden_scenario(model_id),
                format!(
                    "Golden Rule: converted inference failed: {}",
                    converted_result.stderr
                ),
                "N/A",
                0,
            );
            self.collector.add(ev);
            return (0, 1);
        }

        // Step 4: DIFF — the actual Golden Rule assertion
        // Extract just the "Output:" line from both
        let orig_text = Self::extract_output_text(&original_result.stdout);
        let conv_text = Self::extract_output_text(&converted_result.stdout);

        // Popperian: empty extraction means "Output:" marker was missing.
        // Two empty strings comparing equal is vacuous truth, not evidence.
        if orig_text.is_empty() && conv_text.is_empty() {
            let ev = Evidence::falsified(
                "F-GOLDEN-RULE-004",
                Self::golden_scenario(model_id),
                "Golden rule vacuous: both outputs missing 'Output:' marker",
                "N/A",
                0,
            );
            self.collector.add(ev);
            return (0, 1);
        }

        if orig_text == conv_text {
            let ev = Evidence::corroborated(
                "F-GOLDEN-RULE-001",
                Self::golden_scenario(model_id),
                &format!("Golden Rule PASS: identical output: {orig_text}"),
                0,
            );
            self.collector.add(ev);

            // Cleanup
            let _ = std::fs::remove_file(&apr_path);
            (1, 0)
        } else {
            let ev = Evidence::falsified(
                "F-GOLDEN-RULE-001",
                Self::golden_scenario(model_id),
                format!(
                    "Golden Rule FAIL: output differs after conversion.\n\
                     Original:  {orig_text}\n\
                     Converted: {conv_text}"
                ),
                &converted_result.stdout,
                0,
            );
            self.collector.add(ev);

            // Keep the APR file for investigation
            (0, 1)
        }
    }

    /// Extract the "Output:" text from apr run output
    fn extract_output_text(raw: &str) -> String {
        let mut capture = false;
        let mut lines = Vec::new();
        for line in raw.lines() {
            if line.starts_with("Output:") {
                capture = true;
                continue;
            }
            if capture {
                if line.starts_with("Completed in") || line.is_empty() {
                    break;
                }
                lines.push(line.trim());
            }
        }
        lines.join(" ").trim().to_string()
    }

    /// Create a scenario for golden rule evidence
    fn golden_scenario(model_id: &ModelId) -> apr_qa_gen::QaScenario {
        apr_qa_gen::QaScenario::new(
            model_id.clone(),
            apr_qa_gen::Modality::Run,
            apr_qa_gen::Backend::Cpu,
            apr_qa_gen::Format::Apr,
            "Golden Rule: convert → inference → diff".to_string(),
            0,
        )
    }

    /// Truncate a string for display purposes, respecting UTF-8 boundaries.
    fn truncate_str(s: &str, max_len: usize) -> &str {
        if s.len() <= max_len {
            s
        } else {
            let mut end = max_len;
            while end > 0 && !s.is_char_boundary(end) {
                end -= 1;
            }
            &s[..end]
        }
    }

    /// HF Parity Test: Compare Sovereign Stack outputs against HuggingFace golden corpus.
    ///
    /// This test implements Popperian falsification methodology: any divergence beyond
    /// IEEE 754 tolerance thresholds falsifies the parity hypothesis and indicates a
    /// bug that must be investigated.
    ///
    /// # Arguments
    ///
    /// * `model_id` - Model identifier for evidence reporting
    ///
    /// # Returns
    ///
    /// (passed_count, failed_count) - evidence is added to collector
    ///
    /// Run contract invariant tests I-2 through I-5.
    ///
    /// Uses the contract config from the playbook if present, otherwise
    /// defaults to all invariants (I-2 through I-5).
    fn run_contract_invariants(
        &mut self,
        model_path: &Path,
        model_id: &ModelId,
        playbook: &Playbook,
    ) -> (usize, usize) {
        // Skip for single-file models (not applicable)
        if model_path.is_file() {
            let ev = Evidence::skipped(
                "F-CONTRACT-SKIP-002",
                Self::golden_scenario(model_id),
                "Contract invariant testing not applicable for single-file model (requires SafeTensors directory)",
            );
            self.collector.add(ev);
            return (0, 0);
        }

        let config = playbook.contract_tests.clone().unwrap_or_default();

        let evidence = crate::contract::run_contract_tests(
            &self.command_runner,
            model_path,
            model_id,
            &config,
        );

        self.tally_battery(evidence)
    }

    /// Run ollama parity tests (GH-6/AC-2)
    ///
    /// For each quant x prompt: run APR inference + ollama inference, compare output tokens.
    /// Gate F-OLLAMA-001: output match. Gate F-OLLAMA-003: TTFT comparison.
    fn run_ollama_parity_tests(
        &mut self,
        model_path: &Path,
        playbook: &Playbook,
    ) -> (usize, usize) {
        let config = match &playbook.ollama_parity {
            Some(c) if c.enabled => c.clone(),
            _ => {
                let model_id = playbook.model_id();
                let ev = Evidence::skipped(
                    "F-OLLAMA-PARITY-SKIP-001",
                    QaScenario::new(
                        model_id,
                        Modality::Run,
                        Backend::Cpu,
                        Format::SafeTensors,
                        "Ollama parity testing".to_string(),
                        0,
                    ),
                    "Ollama parity tests skipped: ollama_parity not configured or disabled",
                );
                self.collector.add(ev);
                return (0, 0);
            }
        };

        let model_id = playbook.model_id();
        let mut passed = 0;
        let mut failed = 0;

        // Pull ollama model first
        let model_tag = config
            .model_tag
            .clone()
            .unwrap_or_else(|| format!("{}:latest", model_id.name));
        let pull_output = self.command_runner.pull_ollama_model(&model_tag);
        if !pull_output.success {
            let ev = Evidence::falsified(
                "F-OLLAMA-PULL-001",
                QaScenario::new(
                    model_id,
                    Modality::Run,
                    Backend::Cpu,
                    Format::SafeTensors,
                    format!("ollama pull {model_tag}"),
                    0,
                ),
                format!("Ollama pull failed: {}", pull_output.stderr),
                &pull_output.stdout,
                0,
            );
            self.collector.add(ev);
            return (0, 1);
        }

        let (p, f) = self.run_ollama_prompt_gates(model_path, &model_id, &model_tag, &config);
        passed += p;
        failed += f;

        let (p, f) = self.run_ollama_ecosystem_gates(model_path, &model_id);
        passed += p;
        failed += f;

        (passed, failed)
    }

    /// Run per-prompt ollama gates: F-OLLAMA-001 (output match) and F-OLLAMA-003 (TTFT).
    #[allow(clippy::too_many_lines)]
    fn run_ollama_prompt_gates(
        &mut self,
        model_path: &Path,
        model_id: &ModelId,
        model_tag: &str,
        config: &OllamaParityConfig,
    ) -> (usize, usize) {
        let mut passed = 0;
        let mut failed = 0;

        for prompt in &config.prompts {
            let start = std::time::Instant::now();
            let apr_output = self
                .command_runner
                .run_inference(model_path, prompt, 32, false, &[]);
            let ollama_output =
                self.command_runner
                    .run_ollama_inference(model_tag, prompt, config.temperature);
            let duration = start.elapsed().as_millis() as u64;

            let scenario = QaScenario::new(
                model_id.clone(),
                Modality::Run,
                Backend::Cpu,
                Format::SafeTensors,
                format!("ollama parity: {prompt}"),
                0,
            );

            if !apr_output.success || !ollama_output.success {
                let reason = if apr_output.success {
                    format!("Ollama inference failed: {}", ollama_output.stderr)
                } else {
                    format!("APR inference failed: {}", apr_output.stderr)
                };
                let ev = Evidence::falsified(
                    "F-OLLAMA-001",
                    scenario,
                    &reason,
                    &apr_output.stdout,
                    duration,
                );
                self.collector.add(ev);
                failed += 1;
                continue;
            }

            // F-OLLAMA-001: Compare actual output text (not just "both ran")
            let apr_text = Self::extract_output_text(&apr_output.stdout);
            let ollama_text = ollama_output.stdout.trim().to_string();
            if apr_text.is_empty() && ollama_text.is_empty() {
                let ev = Evidence::falsified(
                    "F-OLLAMA-001",
                    scenario.clone(),
                    "Ollama parity vacuous: both outputs empty",
                    "N/A",
                    duration,
                );
                self.collector.add(ev);
                failed += 1;
            } else if apr_text == ollama_text {
                let ev = Evidence::corroborated(
                    "F-OLLAMA-001",
                    scenario.clone(),
                    &format!(
                        "Ollama parity PASS: identical output ({} chars) for prompt: {prompt}",
                        apr_text.len()
                    ),
                    duration,
                );
                self.collector.add(ev);
                passed += 1;
            } else {
                // Text differs — this FALSIFIES the parity hypothesis.
                // Non-deterministic LLM output explains the divergence but does not
                // excuse it: the hypothesis "APR ≡ Ollama" was refuted by evidence.
                // Use temperature=0 on both sides for deterministic comparison.
                let ev = Evidence::falsified(
                    "F-OLLAMA-001",
                    scenario.clone(),
                    format!(
                        "Ollama parity FAIL: output differs (APR: {} chars, Ollama: {} chars) \
                         for prompt: {prompt}. APR: '{}', Ollama: '{}'",
                        apr_text.len(),
                        ollama_text.len(),
                        Self::truncate_str(&apr_text, 80),
                        Self::truncate_str(&ollama_text, 80),
                    ),
                    &format!("APR: {apr_text}\nOllama: {ollama_text}"),
                    duration,
                );
                self.collector.add(ev);
                failed += 1;
            }

            // Gate F-OLLAMA-003: TTFT comparison (time-to-first-token)
            let apr_ttft = crate::executor::parse_timing_ms(&apr_output.stdout);
            let ollama_ttft = crate::executor::parse_timing_ms(&ollama_output.stdout);
            if let (Some(apr_ms), Some(ollama_ms)) = (apr_ttft, ollama_ttft) {
                let ratio = apr_ms / ollama_ms.max(1.0);
                #[allow(clippy::cast_sign_loss)]
                let duration = apr_ms.round() as u64;
                if ratio <= 3.0 {
                    let ev = Evidence::corroborated(
                        "F-OLLAMA-003",
                        scenario.clone(),
                        &format!(
                            "TTFT ratio APR/Ollama: {ratio:.2} (APR={apr_ms:.0}ms, Ollama={ollama_ms:.0}ms)"
                        ),
                        duration,
                    );
                    self.collector.add(ev);
                    passed += 1;
                } else {
                    let ev = Evidence::falsified(
                        "F-OLLAMA-003",
                        scenario.clone(),
                        format!("TTFT ratio {ratio:.2} exceeds 3.0x threshold"),
                        &format!("APR={apr_ms:.0}ms, Ollama={ollama_ms:.0}ms"),
                        duration,
                    );
                    self.collector.add(ev);
                    failed += 1;
                }
            } else {
                // Timing data unavailable — cannot evaluate TTFT hypothesis.
                // Popperian: absence of timing data is not evidence of performance.
                let ev = Evidence::skipped(
                    "F-OLLAMA-003",
                    scenario.clone(),
                    "TTFT comparison skipped: timing data unavailable from output",
                );
                self.collector.add(ev);
            }
        }

        (passed, failed)
    }
}
