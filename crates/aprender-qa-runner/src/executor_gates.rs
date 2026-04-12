impl Executor {
    /// G0 Model Integrity Check: Validates config.json matches tensor metadata
    ///
    /// This pre-flight check catches corrupted configs that would pass G1 (model loads)
    /// but cause silent inference failures. Designed to detect the bug found in
    /// `~/.cache/apr-models/qwen2-5-coder-0-5b-instruct/` where config.json had:
    /// - `num_hidden_layers: 14` (should be 24)
    /// - `hidden_size: 4096` (should be 896)
    /// - `vocab_size: 896` (should be 151936)
    ///
    /// # Returns
    ///
    /// (passed_count, failed_count) - evidence is added to collector
    fn run_g0_integrity_check(&mut self, model_path: &Path, model_id: &ModelId) -> (usize, usize) {
        let start = Instant::now();
        let result = Self::run_integrity_analysis(model_path);
        let duration = start.elapsed().as_millis() as u64;
        let Some(result) = result else {
            let ev = Evidence::skipped(
                integrity::gate_ids::CONFIG,
                Self::integrity_scenario(model_id),
                "G0 SKIP: No SafeTensors files found for integrity check",
            );
            self.collector.add(ev);
            return (0, 0);
        };

        if result.passed {
            let ev = Evidence::corroborated(
                integrity::gate_ids::CONFIG,
                Self::integrity_scenario(model_id),
                "G0 PASS: config.json matches tensor metadata",
                duration,
            );
            self.collector.add(ev);
            return (1, 0);
        }

        let mut failed = 0;
        for error in &result.errors {
            let gate_id = Self::classify_integrity_gate(error);
            let ev = Evidence::falsified(
                gate_id,
                Self::integrity_scenario(model_id),
                error,
                &format!(
                    "Config: {:?}, Tensors: {:?}",
                    result.config_values, result.tensor_values
                ),
                duration,
            );
            self.collector.add(ev);
            failed += 1;
        }
        (0, failed)
    }

    /// Run integrity analysis, returning None if not applicable
    fn run_integrity_analysis(model_path: &Path) -> Option<integrity::IntegrityResult> {
        if model_path.is_file() && model_path.extension().is_some_and(|e| e == "safetensors") {
            Some(integrity::check_safetensors_file_integrity(model_path))
        } else {
            let st_dir = Self::find_safetensors_dir(model_path)?;
            Some(integrity::check_safetensors_integrity(&st_dir))
        }
    }

    /// Classify integrity error into specific gate ID
    fn classify_integrity_gate(error: &str) -> &'static str {
        if error.contains("LAYERS") {
            integrity::gate_ids::LAYERS
        } else if error.contains("HIDDEN") {
            integrity::gate_ids::HIDDEN
        } else if error.contains("VOCAB") {
            integrity::gate_ids::VOCAB
        } else {
            integrity::gate_ids::CONFIG
        }
    }

    /// Find the SafeTensors directory within a model path
    ///
    /// Supports common cache structures:
    /// - `<model_path>/safetensors/` - apr-model-qa-playbook structure
    /// - `<model_path>/` - direct HF cache structure
    fn find_safetensors_dir(model_path: &Path) -> Option<std::path::PathBuf> {
        // File mode: check parent directory for sibling .safetensors files
        if model_path.is_file() {
            if model_path.extension().is_some_and(|e| e == "safetensors") {
                return model_path.parent().map(Path::to_path_buf);
            }
            return None;
        }

        // Try explicit safetensors subdirectory first (apr cache structure)
        let st_subdir = model_path.join("safetensors");
        if st_subdir.exists() && Self::has_safetensors_files(&st_subdir) {
            return Some(st_subdir);
        }

        // Try the model path directly (HF cache structure)
        if Self::has_safetensors_files(model_path) {
            return Some(model_path.to_path_buf());
        }

        // No SafeTensors found
        None
    }

    /// Check if a directory contains .safetensors files
    fn has_safetensors_files(dir: &Path) -> bool {
        dir.read_dir()
            .map(|entries| {
                entries
                    .flatten()
                    .any(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
            })
            .unwrap_or(false)
    }

    /// Create a scenario for G0 integrity evidence
    fn integrity_scenario(model_id: &ModelId) -> apr_qa_gen::QaScenario {
        apr_qa_gen::QaScenario::new(
            model_id.clone(),
            apr_qa_gen::Modality::Run,
            apr_qa_gen::Backend::Cpu,
            apr_qa_gen::Format::SafeTensors,
            "G0 Integrity: config.json vs tensor metadata".to_string(),
            0,
        )
    }

    /// G0-LAYOUT Pre-flight Check: Validates tensor layouts against contract (Issue #4)
    ///
    /// Compares model tensor shapes against the tensor layout contract
    /// (`tensor-layout-v1.yaml`) to catch GH-202 style bugs where wrong shapes
    /// cause garbage output.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the model file or directory
    /// * `model_id` - Model identifier for evidence tracking
    ///
    /// # Returns
    ///
    /// (passed_count, failed_count) - evidence is added to collector
    fn run_g0_layout_check(&mut self, model_path: &Path, model_id: &ModelId) -> (usize, usize) {
        // Try to load the contract from the default location
        // If not found, skip the check (contract is optional)
        let Ok(contract) = load_contract_from(DEFAULT_CONTRACT_PATH) else {
            // Contract not found - expected when aprender is not a sibling directory
            let ev = Evidence::skipped(
                "G0-LAYOUT-001",
                Self::layout_scenario(model_id),
                &format!("G0 SKIP: Layout contract not found at '{DEFAULT_CONTRACT_PATH}' (aprender not present)"),
            );
            self.collector.add(ev);
            return (0, 0);
        };

        let start = Instant::now();
        let result = match validate_model(model_path, &contract) {
            Ok(r) => r,
            Err(e) => {
                // Validation itself failed - emit falsified evidence
                let ev = Evidence::falsified(
                    "G0-LAYOUT-001",
                    Self::layout_scenario(model_id),
                    &format!("Tensor layout validation error: {e}"),
                    "",
                    start.elapsed().as_millis() as u64,
                );
                self.collector.add(ev);
                return (0, 1);
            }
        };

        let duration = start.elapsed().as_millis() as u64;

        if result.passed {
            let ev = Evidence::corroborated(
                "G0-LAYOUT-001",
                Self::layout_scenario(model_id),
                &format!(
                    "G0 PASS: Tensor layouts conform to contract\n  Rules checked: {}\n  Rules passed: {}",
                    result.rules_checked, result.rules_passed
                ),
                duration,
            );
            self.collector.add(ev);
            (1, 0)
        } else {
            // Emit evidence for each failed rule
            let mut failed = 0;
            for tensor_result in &result.tensor_results {
                if !tensor_result.passed {
                    let details = Self::format_tensor_failure(tensor_result);
                    let ev = Evidence::falsified(
                        &tensor_result.rule_id,
                        Self::layout_scenario(model_id),
                        &details,
                        "",
                        duration,
                    );
                    self.collector.add(ev);
                    failed += 1;
                }
            }

            // Also emit evidence for critical failures
            for critical in &result.critical_failures {
                let ev = Evidence::falsified(
                    "G0-LAYOUT-CRITICAL",
                    Self::layout_scenario(model_id),
                    critical,
                    "",
                    duration,
                );
                self.collector.add(ev);
                failed += 1;
            }

            (0, failed.max(1)) // Ensure at least 1 failure is reported
        }
    }

    /// Create a scenario for G0-LAYOUT evidence
    fn layout_scenario(model_id: &ModelId) -> apr_qa_gen::QaScenario {
        apr_qa_gen::QaScenario::new(
            model_id.clone(),
            apr_qa_gen::Modality::Run,
            apr_qa_gen::Backend::Cpu,
            apr_qa_gen::Format::SafeTensors,
            "G0 Layout: tensor shape contract validation".to_string(),
            0,
        )
    }

    /// Format a tensor validation failure for evidence output
    fn format_tensor_failure(
        tensor_result: &crate::layout_contract::TensorValidationResult,
    ) -> String {
        match (&tensor_result.expected, &tensor_result.actual) {
            (Some(expected), Some(actual)) => {
                format!(
                    "{}: {}\n  Expected: {}\n  Actual: {}",
                    tensor_result.rule_id, tensor_result.details, expected, actual
                )
            }
            _ => format!("{}: {}", tensor_result.rule_id, tensor_result.details),
        }
    }

    /// Create a scenario for G0-VALIDATE evidence
    fn validate_scenario(model_id: &ModelId) -> apr_qa_gen::QaScenario {
        apr_qa_gen::QaScenario::new(
            model_id.clone(),
            apr_qa_gen::Modality::Run,
            apr_qa_gen::Backend::Cpu,
            apr_qa_gen::Format::SafeTensors,
            "G0 Validate: NaN/Inf/all-zeros tensor check".to_string(),
            0,
        )
    }

    /// Create a scenario for G0-PULL evidence
    fn pull_scenario(model_id: &ModelId) -> apr_qa_gen::QaScenario {
        apr_qa_gen::QaScenario::new(
            model_id.clone(),
            apr_qa_gen::Modality::Run,
            apr_qa_gen::Backend::Cpu,
            apr_qa_gen::Format::SafeTensors,
            "G0 Pull: acquire model via apr pull".to_string(),
            0,
        )
    }

    /// G0-PULL Pre-flight Check: Acquires model via `apr pull --json`
    ///
    /// Ensures the model is downloaded and cached before any validation
    /// or inference tests. Parses the `Path:` line from stdout to determine
    /// the cached model location.
    ///
    /// # Returns
    ///
    /// (passed_count, failed_count, Option<pulled_path>) - evidence is added to collector
    fn run_g0_pull_check(
        &mut self,
        hf_repo: &str,
        model_id: &ModelId,
    ) -> (usize, usize, Option<String>) {
        let start = Instant::now();
        let output = self.command_runner.pull_model(hf_repo);
        let duration = start.elapsed().as_millis() as u64;

        if output.success {
            // Parse "Path: <path>" from stdout (apr pull indents with spaces)
            // Strip ANSI escape codes since apr pull colorizes the path
            let pulled_path = output.stdout.lines().find_map(|line| {
                line.trim()
                    .strip_prefix("Path: ")
                    .map(|p| Self::strip_ansi(p.trim()))
            });

            let ev = Evidence::corroborated(
                "G0-PULL-001",
                Self::pull_scenario(model_id),
                &format!("G0 PASS: model acquired via apr pull\n{}", output.stdout),
                duration,
            );
            self.collector.add(ev);
            (1, 0, pulled_path)
        } else {
            let reason = format!("G0 FAIL: apr pull failed for {hf_repo}: {}", output.stderr);
            let ev = Evidence::falsified(
                "G0-PULL-001",
                Self::pull_scenario(model_id),
                &reason,
                &output.stdout,
                duration,
            );
            self.collector.add(ev);
            (0, 1, None)
        }
    }

    /// G0-VALIDATE Pre-flight Check: Validates model physics (NaN, Inf, all-zeros)
    ///
    /// Runs `apr validate --strict --json` on each SafeTensors file before any
    /// conversion or inference tests. Resolves directories to individual
    /// `.safetensors` files (supports multi-file sharded models).
    ///
    /// Catches corrupt model files (e.g., 6.7GB F32 zeros instead of 2.88GB BF16)
    /// that would waste qualification time producing meaningless results.
    ///
    /// # Returns
    ///
    /// (passed_count, failed_count) - evidence is added to collector
    fn run_g0_validate_check(&mut self, model_path: &Path, model_id: &ModelId) -> (usize, usize) {
        // Resolve to individual safetensors files
        let files = Self::find_safetensors_files(model_path);
        if files.is_empty() {
            let ev = Evidence::skipped(
                "G0-VALIDATE-001",
                Self::validate_scenario(model_id),
                "G0 SKIP: No safetensors files found for physics validation",
            );
            self.collector.add(ev);
            return (0, 0);
        }

        let mut passed = 0;
        let mut failed = 0;

        for file in &files {
            let start = Instant::now();
            let output = self.command_runner.validate_model_strict(file);
            let duration = start.elapsed().as_millis() as u64;
            let file_name = file
                .file_name()
                .map_or("unknown", |f| f.to_str().unwrap_or("unknown"));

            if output.success {
                let ev = Evidence::corroborated(
                    "G0-VALIDATE-001",
                    Self::validate_scenario(model_id),
                    &format!("G0 PASS: {file_name} physics validated\n{}", output.stdout),
                    duration,
                );
                self.collector.add(ev);
                passed += 1;
            } else {
                let reason = if output.stdout.is_empty() {
                    format!(
                        "G0 FAIL: {file_name} physics validation failed: {}",
                        output.stderr
                    )
                } else {
                    format!(
                        "G0 FAIL: {file_name} corrupt (NaN/Inf/all-zeros)\n{}",
                        output.stdout
                    )
                };
                let ev = Evidence::falsified(
                    "G0-VALIDATE-001",
                    Self::validate_scenario(model_id),
                    &reason,
                    &output.stdout,
                    duration,
                );
                self.collector.add(ev);
                failed += 1;
            }
        }

        (passed, failed)
    }

    /// Find all `.safetensors` files for a model path
    ///
    /// Supports:
    /// - Single file: returns `[file]` if it has `.safetensors` extension
    /// - Directory with `safetensors/` subdir (apr cache): lists files in subdir
    /// - Directory with `.safetensors` files directly (HF cache): lists files
    fn find_safetensors_files(model_path: &Path) -> Vec<std::path::PathBuf> {
        if model_path.is_file() {
            return if model_path.extension().is_some_and(|e| e == "safetensors") {
                vec![model_path.to_path_buf()]
            } else {
                Vec::new()
            };
        }

        // Find the directory containing safetensors files
        let Some(st_dir) = Self::find_safetensors_dir(model_path) else {
            return Vec::new();
        };

        // Collect all .safetensors files
        let Ok(entries) = st_dir.read_dir() else {
            return Vec::new();
        };

        let mut files: Vec<_> = entries
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
            .map(|e| e.path())
            .collect();
        files.sort();
        files
    }
}

include!("executor_gates_b.rs");
