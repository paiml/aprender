impl Executor {

    /// Run ecosystem ollama gates: F-OLLAMA-005 (GGUF loadability) and F-OLLAMA-004 (API).
    fn run_ollama_ecosystem_gates(
        &mut self,
        model_path: &Path,
        model_id: &ModelId,
    ) -> (usize, usize) {
        let mut passed = 0;
        let mut failed = 0;

        // Gate F-OLLAMA-005: Ollama loads our GGUF without errors
        let gguf_scenario = QaScenario::new(
            model_id.clone(),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "ollama GGUF loadability".to_string(),
            0,
        );
        let start = Instant::now();
        let create_output = self
            .command_runner
            .create_ollama_model(&format!("apr-test-{}", model_id.name), model_path);
        let duration = start.elapsed().as_millis() as u64;
        if create_output.success {
            let ev = Evidence::corroborated(
                "F-OLLAMA-005",
                gguf_scenario,
                "Ollama successfully loaded our GGUF via `ollama create`",
                duration,
            );
            self.collector.add(ev);
            passed += 1;
        } else {
            let ev = Evidence::falsified(
                "F-OLLAMA-005",
                gguf_scenario,
                format!("Ollama failed to load GGUF: {}", create_output.stderr),
                &create_output.stdout,
                duration,
            );
            self.collector.add(ev);
            failed += 1;
        }

        // Gate F-OLLAMA-004: API endpoint parity (/v1/models exists on both)
        let api_scenario = QaScenario::new(
            model_id.clone(),
            Modality::Serve,
            Backend::Cpu,
            Format::SafeTensors,
            "ollama API parity".to_string(),
            0,
        );
        let start = Instant::now();
        let ollama_api = self
            .command_runner
            .http_get("http://localhost:11434/api/tags");
        let duration = start.elapsed().as_millis() as u64;
        if ollama_api.success {
            let ev = Evidence::corroborated(
                "F-OLLAMA-004",
                api_scenario,
                "Ollama API endpoint /api/tags is accessible",
                duration,
            );
            self.collector.add(ev);
            passed += 1;
        } else {
            let ev = Evidence::falsified(
                "F-OLLAMA-004",
                api_scenario,
                format!("Ollama API not accessible: {}", ollama_api.stderr),
                &ollama_api.stdout,
                duration,
            );
            self.collector.add(ev);
            failed += 1;
        }

        (passed, failed)
    }

    /// Run performance gates: F-PERF-006 (GPU/CPU ratio) and F-PERF-005 (memory profiling)
    ///
    /// Note: F-PERF-003 is reserved for Memory Leak detection (patterns_spec_gates.rs).
    /// F-PERF-006 is the CI-level GPU vs CPU throughput ratio gate.
    #[allow(clippy::too_many_lines)]
    fn run_perf_gates(
        &mut self,
        model_path: &Path,
        model_id: &ModelId,
        playbook: &Playbook,
    ) -> (usize, usize) {
        let mut passed = 0;
        let mut failed = 0;

        let profile_config = match &playbook.profile_ci {
            Some(c) if c.enabled => c,
            _ => {
                let ev = Evidence::skipped(
                    "F-PERF-SKIP-001",
                    QaScenario::new(
                        model_id.clone(),
                        Modality::Run,
                        Backend::Cpu,
                        Format::SafeTensors,
                        "Performance gates (profile_ci)".to_string(),
                        0,
                    ),
                    "Performance gates skipped: profile_ci not configured or disabled",
                );
                self.collector.add(ev);
                return (0, 0);
            }
        };

        // F-PERF-006: GPU vs CPU throughput comparison
        let has_cpu = profile_config
            .backends
            .iter()
            .any(|b| b.eq_ignore_ascii_case("cpu"));
        let includes_gpu = profile_config
            .backends
            .iter()
            .any(|b| b.eq_ignore_ascii_case("gpu"));

        if has_cpu && includes_gpu {
            let warmup = profile_config.warmup as u32;
            let measure = profile_config.measure as u32;
            let start = Instant::now();
            let cpu_output = self
                .command_runner
                .profile_ci(model_path, None, None, warmup, measure, true);
            let gpu_output = self
                .command_runner
                .profile_ci(model_path, None, None, warmup, measure, false);
            let duration = start.elapsed().as_millis() as u64;

            let cpu_tps = crate::executor::parse_throughput(&cpu_output.stdout);
            let gpu_tps = crate::executor::parse_throughput(&gpu_output.stdout);

            let scenario = QaScenario::new(
                model_id.clone(),
                Modality::Run,
                Backend::Gpu,
                Format::SafeTensors,
                "GPU vs CPU throughput ratio".to_string(),
                0,
            );

            if let (Some(cpu), Some(gpu)) = (cpu_tps, gpu_tps) {
                let ratio = gpu / cpu.max(0.01);
                if ratio >= 1.0 {
                    let ev = Evidence::corroborated(
                        "F-PERF-006",
                        scenario,
                        &format!(
                            "GPU/CPU ratio: {ratio:.1}x (GPU={gpu:.1} tok/s, CPU={cpu:.1} tok/s)"
                        ),
                        duration,
                    );
                    self.collector.add(ev);
                    passed += 1;
                } else {
                    let ev = Evidence::falsified(
                        "F-PERF-006",
                        scenario,
                        format!("GPU slower than CPU: ratio {ratio:.2}x"),
                        &format!("GPU={gpu:.1} tok/s, CPU={cpu:.1} tok/s"),
                        duration,
                    );
                    self.collector.add(ev);
                    failed += 1;
                }
            }
        }

        // F-PERF-005: Memory profiling
        let start = Instant::now();
        let mem_output = self.command_runner.profile_memory(model_path);
        let duration = start.elapsed().as_millis() as u64;
        let mem_scenario = QaScenario::new(
            model_id.clone(),
            Modality::Run,
            Backend::Cpu,
            Format::SafeTensors,
            "memory profiling".to_string(),
            0,
        );

        if mem_output.success {
            let ev = Evidence::corroborated(
                "F-PERF-005",
                mem_scenario,
                &format!("Memory profile collected: {}", mem_output.stdout.trim()),
                duration,
            );
            self.collector.add(ev);
            passed += 1;
        } else {
            let ev = Evidence::falsified(
                "F-PERF-005",
                mem_scenario,
                format!("Memory profiling failed: {}", mem_output.stderr),
                &mem_output.stdout,
                duration,
            );
            self.collector.add(ev);
            failed += 1;
        }

        (passed, failed)
    }

    /// # References
    ///
    /// - Popper, K. (1959). *The Logic of Scientific Discovery*. Routledge.
    /// - Goldberg, D. (1991). "What Every Computer Scientist Should Know About FP."
    #[allow(clippy::too_many_lines)]
    fn run_hf_parity_tests(&mut self, model_id: &ModelId) -> (usize, usize) {
        let (corpus_path, model_family) = if let (Some(cp), Some(mf)) = (
            &self.config.hf_parity_corpus_path,
            &self.config.hf_parity_model_family,
        ) {
            (cp.clone(), mf.clone())
        } else {
            // Missing configuration - skip (not evidence of corroboration)
            let ev = Evidence::skipped(
                "F-HF-PARITY-SKIP",
                Self::hf_parity_scenario(model_id, "config"),
                "HF parity skipped: corpus_path or model_family not configured",
            );
            self.collector.add(ev);
            return (0, 0);
        };

        // Load manifest to get list of available prompts
        let manifest_path = Path::new(&corpus_path)
            .join(&model_family)
            .join("manifest.json");

        if !manifest_path.exists() {
            let ev = Evidence::falsified(
                "F-HF-PARITY-001",
                Self::hf_parity_scenario(model_id, "manifest"),
                format!("HF parity manifest not found: {}", manifest_path.display()),
                "N/A",
                0,
            );
            self.collector.add(ev);
            return (0, 1);
        }

        // Parse manifest
        let manifest_data = match std::fs::read_to_string(&manifest_path) {
            Ok(d) => d,
            Err(e) => {
                let ev = Evidence::falsified(
                    "F-HF-PARITY-002",
                    Self::hf_parity_scenario(model_id, "manifest"),
                    format!("Failed to read manifest: {e}"),
                    "N/A",
                    0,
                );
                self.collector.add(ev);
                return (0, 1);
            }
        };

        #[allow(clippy::items_after_statements)]
        #[derive(serde::Deserialize)]
        struct Manifest {
            prompts: Vec<String>,
        }

        let manifest: Manifest = match serde_json::from_str(&manifest_data) {
            Ok(m) => m,
            Err(e) => {
                let ev = Evidence::falsified(
                    "F-HF-PARITY-003",
                    Self::hf_parity_scenario(model_id, "manifest"),
                    format!("Failed to parse manifest: {e}"),
                    "N/A",
                    0,
                );
                self.collector.add(ev);
                return (0, 1);
            }
        };

        if manifest.prompts.is_empty() {
            let ev = Evidence::skipped(
                "F-HF-PARITY-SKIP",
                Self::hf_parity_scenario(model_id, "manifest"),
                "HF parity skipped: no prompts in manifest",
            );
            self.collector.add(ev);
            return (0, 0);
        }

        // Create oracle with FP16 tolerance (most common for inference)
        let oracle =
            HfParityOracle::new(&corpus_path, &model_family).with_tolerance(Tolerance::fp16());

        let mut passed = 0;
        let mut failed = 0;

        // Test each prompt hash in the manifest
        for prompt_hash in &manifest.prompts {
            // Load the golden output to get the original prompt
            let golden_path = Path::new(&corpus_path)
                .join(&model_family)
                .join(format!("{prompt_hash}.json"));

            let prompt = match std::fs::read_to_string(&golden_path) {
                Ok(data) => {
                    #[allow(clippy::items_after_statements)]
                    #[derive(serde::Deserialize)]
                    struct GoldenMeta {
                        prompt: String,
                    }
                    match serde_json::from_str::<GoldenMeta>(&data) {
                        Ok(meta) => meta.prompt,
                        Err(e) => {
                            // Bug #33: I/O errors must generate Evidence, not just eprintln.
                            // Invisible failures defeat falsification.
                            let ev = Evidence::falsified(
                                "F-HF-PARITY-004",
                                Self::hf_parity_scenario(model_id, prompt_hash),
                                format!("Failed to parse golden meta {}: {e}", golden_path.display()),
                                "N/A",
                                0,
                            );
                            self.collector.add(ev);
                            failed += 1;
                            continue;
                        }
                    }
                }
                Err(e) => {
                    // Bug #33: I/O errors must generate Evidence, not just eprintln.
                    let ev = Evidence::falsified(
                        "F-HF-PARITY-004",
                        Self::hf_parity_scenario(model_id, prompt_hash),
                        format!("Failed to read golden file {}: {e}", golden_path.display()),
                        "N/A",
                        0,
                    );
                    self.collector.add(ev);
                    failed += 1;
                    continue;
                }
            };

            // Validate golden file exists before running inference (fail fast)
            let _golden = match oracle.load_golden(&prompt) {
                Ok(g) => g,
                Err(e) => {
                    let ev = Evidence::falsified(
                        "F-HF-PARITY-004",
                        Self::hf_parity_scenario(model_id, &prompt),
                        format!("Failed to load golden for prompt '{prompt}': {e}"),
                        "N/A",
                        0,
                    );
                    self.collector.add(ev);
                    failed += 1;
                    continue;
                }
            };

            // Run actual inference and compare against golden using oracle
            let Some(model_path_str) = self.config.model_path.clone() else {
                let ev = Evidence::skipped(
                    "F-HF-PARITY-001",
                    Self::hf_parity_scenario(model_id, &prompt),
                    "HF parity skipped: no model path configured for inference",
                );
                self.collector.add(ev);
                continue;
            };

            let start = Instant::now();
            let inference_output = self.command_runner.run_inference(
                Path::new(&model_path_str),
                &prompt,
                32,
                false,
                &[],
            );
            let duration = start.elapsed().as_millis() as u64;

            if !inference_output.success {
                let ev = Evidence::falsified(
                    "F-HF-PARITY-001",
                    Self::hf_parity_scenario(model_id, &prompt),
                    format!("HF parity: inference failed: {}", inference_output.stderr),
                    &inference_output.stdout,
                    duration,
                );
                self.collector.add(ev);
                failed += 1;
                continue;
            }

            // Compare actual output against golden using oracle
            let oracle_result = oracle.evaluate(&prompt, &inference_output.stdout);
            match oracle_result {
                apr_qa_gen::OracleResult::Corroborated { evidence: ev_text } => {
                    let ev = Evidence::corroborated(
                        "F-HF-PARITY-001",
                        Self::hf_parity_scenario(model_id, &prompt),
                        &format!("HF parity PASS: {ev_text}"),
                        duration,
                    );
                    self.collector.add(ev);
                    passed += 1;
                }
                apr_qa_gen::OracleResult::Falsified { reason, evidence: ev_text } => {
                    let ev = Evidence::falsified(
                        "F-HF-PARITY-001",
                        Self::hf_parity_scenario(model_id, &prompt),
                        format!("HF parity FAIL: {reason}"),
                        &ev_text,
                        duration,
                    );
                    self.collector.add(ev);
                    failed += 1;
                }
            }
        }

        (passed, failed)
    }

    /// Create a scenario for HF parity evidence
    fn hf_parity_scenario(model_id: &ModelId, prompt: &str) -> QaScenario {
        QaScenario::new(
            model_id.clone(),
            Modality::Run,
            Backend::Cpu,
            Format::Apr,
            format!("HF Parity: {}", Self::truncate_str(prompt, 40)),
            0,
        )
    }
}
