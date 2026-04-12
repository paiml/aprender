
impl Executor {
    /// Create a new executor with default config
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ExecutionConfig::default(),
            collector: EvidenceCollector::new(),
            command_runner: Arc::new(RealCommandRunner::new()),
        }
    }

    /// Create a new executor with custom config
    #[must_use]
    pub fn with_config(config: ExecutionConfig) -> Self {
        Self {
            config,
            collector: EvidenceCollector::new(),
            command_runner: Arc::new(RealCommandRunner::new()),
        }
    }

    /// Create a new executor with custom config and command runner
    #[must_use]
    pub fn with_runner(config: ExecutionConfig, runner: Arc<dyn CommandRunner>) -> Self {
        Self {
            config,
            collector: EvidenceCollector::new(),
            command_runner: runner,
        }
    }

    /// Execute a playbook
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails critically.
    #[allow(clippy::too_many_lines)]
    pub fn execute(&mut self, playbook: &Playbook) -> Result<ExecutionResult> {
        let scenarios = playbook.generate_scenarios();
        let total = scenarios.len();
        let start = Instant::now();

        // Metadata-only mode: skip inference, verify dimensions from config.json + SafeTensors headers
        if self.config.metadata_only {
            return self.execute_metadata_only(playbook, start);
        }

        // Pre-flight checks (integrity, implicit skips, gateways)
        if let Some(result) = self.check_pre_flight(playbook, total, start) {
            return Ok(result);
        }

        // G0-PULL: Ensure model is cached (skip when user provided --model-path)
        let (pull_passed, pull_failed) = if self.config.model_path.is_none() {
            let model_id = playbook.model_id();
            let (pp, pf, pulled_path) = self.run_g0_pull_check(&playbook.model.hf_repo, &model_id);
            if pf > 0 {
                return Ok(ExecutionResult {
                    playbook_name: playbook.name.clone(),
                    total_scenarios: total + pp + pf,
                    passed: pp,
                    failed: total + pf,
                    skipped: 0,
                    duration_ms: start.elapsed().as_millis() as u64,
                    gateway_failed: Some("G0-PULL-001: Model acquisition failed".to_string()),
                    evidence: self.collector.clone(),
                });
            }
            if let Some(ref path) = pulled_path {
                self.config.model_path = Some(path.clone());
            }
            (pp, pf)
        } else {
            (0, 0)
        };

        // G0-FORMAT, G0-VALIDATE (early return on failure), G0-TENSOR, G0-INTEGRITY, G0-LAYOUT
        let (format_passed, format_failed) = self.run_g0_format_check(playbook);

        let (validate_passed, validate_failed) =
            self.config.model_path.clone().map_or((0, 0), |model_path| {
                let model_id = playbook.model_id();
                self.run_g0_validate_check(Path::new(&model_path), &model_id)
            });
        if validate_failed > 0 {
            return Ok(ExecutionResult {
                playbook_name: playbook.name.clone(),
                total_scenarios: total + pull_passed + validate_passed + validate_failed,
                passed: pull_passed + validate_passed,
                failed: total + validate_failed,
                skipped: 0,
                duration_ms: start.elapsed().as_millis() as u64,
                gateway_failed: Some(
                    "G0-VALIDATE-001: Model physics validation failed (corrupt model)".to_string(),
                ),
                evidence: self.collector.clone(),
            });
        }

        // Jidoka: stop the line on any G0 sub-gate failure
        if format_failed > 0 {
            return Ok(ExecutionResult {
                playbook_name: playbook.name.clone(),
                total_scenarios: total + pull_passed + format_passed + format_failed + validate_passed,
                passed: pull_passed + format_passed + validate_passed,
                failed: total + format_failed,
                skipped: 0,
                duration_ms: start.elapsed().as_millis() as u64,
                gateway_failed: Some(
                    "G0-FORMAT: Model format check failed".to_string(),
                ),
                evidence: self.collector.clone(),
            });
        }

        let (tensor_passed, tensor_failed) = self.check_g0_tensor(playbook);
        if tensor_failed > 0 {
            return Ok(ExecutionResult {
                playbook_name: playbook.name.clone(),
                total_scenarios: total + pull_passed + format_passed + validate_passed + tensor_passed + tensor_failed,
                passed: pull_passed + format_passed + validate_passed + tensor_passed,
                failed: total + tensor_failed,
                skipped: 0,
                duration_ms: start.elapsed().as_millis() as u64,
                gateway_failed: Some(
                    "G0-TENSOR-001: Tensor template mismatch".to_string(),
                ),
                evidence: self.collector.clone(),
            });
        }

        let (integrity_passed, integrity_failed) =
            self.config.model_path.clone().map_or((0, 0), |model_path| {
                let model_id = playbook.model_id();
                self.run_g0_integrity_check(Path::new(&model_path), &model_id)
            });
        if integrity_failed > 0 {
            return Ok(ExecutionResult {
                playbook_name: playbook.name.clone(),
                total_scenarios: total + pull_passed + format_passed + validate_passed + tensor_passed + integrity_passed + integrity_failed,
                passed: pull_passed + format_passed + validate_passed + tensor_passed + integrity_passed,
                failed: total + integrity_failed,
                skipped: 0,
                duration_ms: start.elapsed().as_millis() as u64,
                gateway_failed: Some(
                    "G0-INTEGRITY: Config/tensor metadata mismatch".to_string(),
                ),
                evidence: self.collector.clone(),
            });
        }

        let (layout_passed, layout_failed) =
            self.config.model_path.clone().map_or((0, 0), |model_path| {
                let model_id = playbook.model_id();
                self.run_g0_layout_check(Path::new(&model_path), &model_id)
            });
        if layout_failed > 0 {
            return Ok(ExecutionResult {
                playbook_name: playbook.name.clone(),
                total_scenarios: total + pull_passed + format_passed + validate_passed + tensor_passed + integrity_passed + layout_passed + layout_failed,
                passed: pull_passed + format_passed + validate_passed + tensor_passed + integrity_passed + layout_passed,
                failed: total + layout_failed,
                skipped: 0,
                duration_ms: start.elapsed().as_millis() as u64,
                gateway_failed: Some(
                    "G0-LAYOUT: Tensor layout contract violation".to_string(),
                ),
                evidence: self.collector.clone(),
            });
        }

        // Execute scenarios
        let (passed, failed, skipped) = self.execute_scenarios(scenarios, &playbook.name);

        // Run extended tests (conversion, golden rule, contracts, parity, perf, ollama)
        let (ext_passed, ext_failed) = self.run_extended_tests(playbook);

        // Tally results
        let gate_passed = pull_passed
            + format_passed
            + validate_passed
            + tensor_passed
            + integrity_passed
            + layout_passed;
        let gate_failed = pull_failed
            + format_failed
            + validate_failed
            + tensor_failed
            + integrity_failed
            + layout_failed;

        Ok(ExecutionResult {
            playbook_name: playbook.name.clone(),
            total_scenarios: total + gate_passed + gate_failed + ext_passed + ext_failed,
            passed: passed + gate_passed + ext_passed,
            failed: failed + gate_failed + ext_failed,
            skipped,
            duration_ms: start.elapsed().as_millis() as u64,
            gateway_failed: None,
            evidence: self.collector.clone(),
        })
    }

    /// Pre-flight checks: integrity, implicit skips, gateway conditions
    fn check_pre_flight(
        &self,
        playbook: &Playbook,
        total: usize,
        start: Instant,
    ) -> Option<ExecutionResult> {
        if self.config.check_integrity {
            if let Some(ref lock_path) = self.config.lock_file_path {
                match crate::playbook::load_lock_file(lock_path) {
                    Ok(lock_file) => {
                        // Use playbook_file_path for hash verification (not the lock path)
                        let pb_path = self
                            .config
                            .playbook_file_path
                            .as_deref()
                            .unwrap_or(lock_path);
                        if let Err(e) = crate::playbook::verify_playbook_integrity(
                            pb_path,
                            &lock_file,
                            &playbook.name,
                        ) {
                            return Some(ExecutionResult {
                                playbook_name: playbook.name.clone(),
                                total_scenarios: total,
                                passed: 0,
                                failed: total,
                                skipped: 0,
                                duration_ms: start.elapsed().as_millis() as u64,
                                gateway_failed: Some(format!("Integrity check failed: {e}")),
                                evidence: self.collector.clone(),
                            });
                        }
                    }
                    Err(e) => {
                        eprintln!("[WARN] Could not load lock file '{lock_path}': {e}");
                    }
                }
            }
        }

        if self.config.warn_implicit_skips {
            let all_formats = vec![Format::Gguf, Format::SafeTensors, Format::Apr];
            let skip_files = crate::playbook::find_skip_files(Path::new("."), &playbook.name);
            let implicit =
                crate::playbook::detect_implicit_skips(playbook, &all_formats, &skip_files);
            for skip in &implicit {
                eprintln!("[WARN] Implicit skip detected: {skip}");
            }
        }

        if let Err(e) = self.check_gateways(playbook) {
            return Some(ExecutionResult {
                playbook_name: playbook.name.clone(),
                total_scenarios: total,
                passed: 0,
                failed: total,
                skipped: 0,
                duration_ms: start.elapsed().as_millis() as u64,
                gateway_failed: Some(e.to_string()),
                evidence: self.collector.clone(),
            });
        }

        None
    }

    /// Execute metadata-only dimensional verification (dim-smoke tier).
    ///
    /// Resolves the model path, then runs dimensional checks against
    /// config.json and SafeTensors headers without loading model weights.
    fn execute_metadata_only(
        &mut self,
        playbook: &Playbook,
        start: Instant,
    ) -> Result<ExecutionResult> {
        let model_id = playbook.model_id();

        // Resolve model path: prefer explicit --model-path, then try HF cache, then apr pull
        let model_path = if let Some(ref path) = self.config.model_path {
            PathBuf::from(path)
        } else if let Ok(p) = crate::conversion::resolve_hf_repo_to_cache(&playbook.model.hf_repo) {
            p
        } else {
            let (pp, pf, pulled_path) = self.run_g0_pull_check(&playbook.model.hf_repo, &model_id);
            if pf > 0 {
                return Ok(ExecutionResult {
                    playbook_name: playbook.name.clone(),
                    total_scenarios: pp + pf,
                    passed: pp,
                    failed: pf,
                    skipped: 0,
                    duration_ms: start.elapsed().as_millis() as u64,
                    gateway_failed: Some("G0-PULL-001: Model acquisition failed".to_string()),
                    evidence: self.collector.clone(),
                });
            }
            match pulled_path {
                Some(p) if !p.is_empty() => PathBuf::from(p),
                _ => {
                    return Ok(ExecutionResult {
                        playbook_name: playbook.name.clone(),
                        total_scenarios: 0,
                        passed: 0,
                        failed: 1,
                        skipped: 0,
                        duration_ms: start.elapsed().as_millis() as u64,
                        gateway_failed: Some(
                            "G0-PULL-001: Pull succeeded but returned no path".to_string(),
                        ),
                        evidence: self.collector.clone(),
                    });
                }
            }
        };

        let check_result = crate::dimensional_check::run_dimensional_check(&model_path, playbook);

        let mut passed = 0usize;
        let mut failed = 0usize;
        for check in &check_result.checks {
            let gate_id = format!("G0-DIM-{}", check.name.to_uppercase());
            let scenario = QaScenario::new(
                model_id.clone(),
                Modality::Run,
                Backend::Cpu,
                Format::SafeTensors,
                format!("Dimensional check: {}", check.name),
                0,
            );
            if check.passed {
                self.collector.add(Evidence::corroborated(
                    &gate_id,
                    scenario,
                    format!(
                        "G0 PASS: {} expected={} actual={}",
                        check.name, check.expected, check.actual
                    ),
                    check_result.duration_ms,
                ));
                passed += 1;
            } else {
                self.collector.add(Evidence::falsified(
                    &gate_id,
                    scenario,
                    format!(
                        "G0 FAIL: {} expected={} actual={}",
                        check.name, check.expected, check.actual
                    ),
                    format!("expected={} actual={}", check.expected, check.actual),
                    check_result.duration_ms,
                ));
                failed += 1;
            }
        }

        let gateway_failed = if failed > 0 {
            Some(format!(
                "G0-DIM: {failed} dimensional check(s) failed for {}",
                check_result.model_id
            ))
        } else {
            None
        };

        Ok(ExecutionResult {
            playbook_name: playbook.name.clone(),
            total_scenarios: passed + failed,
            passed,
            failed,
            skipped: 0,
            duration_ms: start.elapsed().as_millis() as u64,
            gateway_failed,
            evidence: self.collector.clone(),
        })
    }

    /// G0-FORMAT: Prepare workspace with APR cache directory structure
    fn run_g0_format_check(&mut self, playbook: &Playbook) -> (usize, usize) {
        let Some(model_path_str) = self.config.model_path.clone() else {
            let model_id = playbook.model_id();
            self.collector.add(Evidence::skipped(
                "G0-FORMAT-SKIP-001",
                QaScenario::new(
                    model_id,
                    Modality::Run,
                    Backend::Cpu,
                    Format::SafeTensors,
                    "G0 Format: prepare workspace".to_string(),
                    0,
                ),
                "G0-FORMAT skipped: model path not available",
            ));
            return (0, 0);
        };
        let path = Path::new(&model_path_str);
        let is_single_safetensors =
            path.is_file() && path.extension().is_some_and(|e| e == "safetensors");
        let is_sharded_index = path.is_file()
            && path
                .file_name()
                .is_some_and(|n| n.to_string_lossy().ends_with(".safetensors.index.json"));
        let is_flat_dir = path.is_dir() && {
            let has_st_file = path.join("model.safetensors").exists();
            let has_cache_structure = path.join("apr").exists();
            has_st_file && !has_cache_structure
        };
        let is_sharded_dir = path.is_dir() && path.join("model.safetensors.index.json").exists();

        let source_file = if is_single_safetensors || is_sharded_index {
            Some(path.to_path_buf())
        } else if is_sharded_dir {
            Some(path.join("model.safetensors.index.json"))
        } else if is_flat_dir {
            Some(path.join("model.safetensors"))
        } else {
            None
        };

        if let Some(source) = source_file {
            let model_id = playbook.model_id();
            let (workspace, fp, ff) =
                self.prepare_model_workspace(&source, &model_id, &playbook.model.formats);
            self.config.model_path = Some(workspace);
            (fp, ff)
        } else {
            (0, 0)
        }
    }

    /// G0-TENSOR: Tensor template validation against family YAML
    fn check_g0_tensor(&mut self, playbook: &Playbook) -> (usize, usize) {
        let model_id = playbook.model_id();
        let tensor_scenario = || {
            QaScenario::new(
                model_id.clone(),
                Modality::Run,
                Backend::Cpu,
                Format::SafeTensors,
                "G0 Tensor: template validation".to_string(),
                0,
            )
        };
        let model_path_str = if let Some(p) = self.config.model_path.as_ref() {
            p.clone()
        } else {
            self.collector.add(Evidence::skipped(
                "G0-TENSOR-SKIP-001",
                tensor_scenario(),
                "G0-TENSOR skipped: model path not available",
            ));
            return (0, 0);
        };
        let family = if let Some(f) = playbook.model.family.as_ref() {
            f.clone()
        } else {
            self.collector.add(Evidence::skipped(
                "G0-TENSOR-SKIP-001",
                tensor_scenario(),
                "G0-TENSOR skipped: model family not configured in playbook",
            ));
            return (0, 0);
        };
        let size_variant = if let Some(s) = playbook.model.size_variant.as_ref() {
            s.clone()
        } else {
            self.collector.add(Evidence::skipped(
                "G0-TENSOR-SKIP-001",
                tensor_scenario(),
                "G0-TENSOR skipped: size_variant not configured in playbook",
            ));
            return (0, 0);
        };
        self.run_g0_tensor_template_check(
            Path::new(&model_path_str),
            &model_id,
            &family,
            &size_variant,
            None,
        )
    }

    /// Execute scenario loop with failure policy handling.
    ///
    /// Serve scenarios are partitioned out and batched: one server lifecycle
    /// per `(format, backend)` group, with 19 endpoint checks per lifecycle.
    fn execute_scenarios(
        &mut self,
        scenarios: Vec<QaScenario>,
        playbook_name: &str,
    ) -> (usize, usize, usize) {
        let mut passed = 0;
        let mut failed = 0;
        let mut skipped = 0;

        // Partition serve vs non-serve scenarios
        let (serve_scenarios, other_scenarios): (Vec<_>, Vec<_>) = scenarios
            .into_iter()
            .partition(|s| s.modality == Modality::Serve);

        // Non-serve: execute individually (unchanged)
        for scenario in other_scenarios {
            if self.config.dry_run {
                let cmd = scenario.to_command("model.gguf");
                println!("[DRY RUN] {cmd}");
                skipped += 1;
                continue;
            }

            let evidence = self.execute_scenario(&scenario);
            let (p, f, s, stop) = self.tally_evidence(evidence, playbook_name);
            passed += p;
            failed += f;
            skipped += s;
            if stop {
                return (passed, failed, skipped);
            }
        }

        // Serve: group by (format, backend), one server per group
        let (sp, sf, ss) = self.execute_serve_groups(serve_scenarios, playbook_name);
        passed += sp;
        failed += sf;
        skipped += ss;

        (passed, failed, skipped)
    }

    /// Tally a single evidence item: update counters, check stop policy.
    ///
    /// Returns `(passed, failed, skipped, should_stop)`.
    fn tally_evidence(
        &mut self,
        evidence: Evidence,
        playbook_name: &str,
    ) -> (usize, usize, usize, bool) {
        if evidence.outcome == Outcome::Skipped {
            self.collector.add(evidence);
            return (0, 0, 1, false);
        }
        if evidence.outcome.is_pass() {
            self.collector.add(evidence);
            return (1, 0, 0, false);
        }
        let stop = self.should_stop_on_failure(&evidence, playbook_name);
        self.collector.add(evidence);
        (0, 1, 0, stop)
    }

    /// Execute serve scenario groups: one server lifecycle per (format, backend).
    fn execute_serve_groups(
        &mut self,
        serve_scenarios: Vec<QaScenario>,
        playbook_name: &str,
    ) -> (usize, usize, usize) {
        let mut passed = 0;
        let mut failed = 0;
        let mut skipped = 0;

        let mut groups: std::collections::HashMap<(Format, Backend), Vec<QaScenario>> =
            std::collections::HashMap::new();
        for s in serve_scenarios {
            groups.entry((s.format, s.backend)).or_default().push(s);
        }

        for (_key, group) in groups {
            if self.config.dry_run {
                for s in &group {
                    let cmd = s.to_command("model.gguf");
                    println!("[DRY RUN] {cmd}");
                    skipped += 1;
                }
                continue;
            }

            let first = &group[0];
            let Some(model_path) = self.resolve_model_path(first) else {
                for s in &group {
                    let gate_id = format!("F-{}-001", s.mqs_category());
                    skipped += 1;
                    self.collector.add(Evidence::skipped(
                        &gate_id,
                        s.clone(),
                        format!("Format {:?} not available for model file", s.format),
                    ));
                }
                continue;
            };
            let no_gpu = first.backend == Backend::Cpu;
            let evidence_vec = self.run_serve_battery(&model_path, first, no_gpu);
            for ev in evidence_vec {
                let (p, f, s, stop) = self.tally_evidence(ev, playbook_name);
                passed += p;
                failed += f;
                skipped += s;
                if stop {
                    return (passed, failed, skipped);
                }
            }
        }

        (passed, failed, skipped)
    }

    /// Check failure policy and return true if execution should stop
    fn should_stop_on_failure(&self, evidence: &Evidence, playbook_name: &str) -> bool {
        match self.config.failure_policy {
            FailurePolicy::StopOnFirst => true,
            FailurePolicy::FailFast => {
                self.print_fail_fast_diagnostics(evidence, playbook_name);
                true
            }
            FailurePolicy::StopOnP0 => {
                evidence.gate_id.starts_with("G0-")
                    || evidence.gate_id.starts_with("F-INT-")
                    || evidence.gate_id.starts_with("F-SEC-")
            }
            FailurePolicy::CollectAll => false,
        }
    }

    /// Print fail-fast diagnostic report (FF-REPORT-001)
    fn print_fail_fast_diagnostics(&self, evidence: &Evidence, playbook_name: &str) {
        eprintln!("\n[FAIL-FAST] Gate {} FALSIFIED", evidence.gate_id);
        eprintln!("[FAIL-FAST] Model: {}", evidence.scenario.model.hf_repo());
        eprintln!("[FAIL-FAST] Format: {:?}", evidence.scenario.format);
        eprintln!("[FAIL-FAST] Backend: {:?}", evidence.scenario.backend);
        eprintln!("[FAIL-FAST] Outcome: {:?}", evidence.outcome);
        eprintln!("[FAIL-FAST] Reason: {}", evidence.reason);

        if let Some(ref model_path) = self.config.model_path {
            let output_dir = self.config.output_dir.as_deref().unwrap_or("output");
            let reporter = FailFastReporter::new(Path::new(output_dir));
            if let Err(e) =
                reporter.generate_report(evidence, Path::new(model_path), Some(playbook_name))
            {
                eprintln!("[FAIL-FAST] Warning: Failed to generate report: {e}");
            }
        } else {
            if let Some(ref stderr) = evidence.stderr {
                eprintln!("[FAIL-FAST] Stderr:\n{stderr}");
            }
            if let Some(exit_code) = evidence.exit_code {
                eprintln!("[FAIL-FAST] Exit code: {exit_code}");
            }
            eprintln!("[FAIL-FAST] No model path - full report not generated\n");
        }
    }
}

include!("golden.rs");
include!("gates.rs");
