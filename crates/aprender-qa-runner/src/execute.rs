
impl ToolExecutor {
    /// Create a new tool executor
    #[must_use]
    pub fn new(model_path: String, no_gpu: bool, timeout_ms: u64) -> Self {
        Self {
            model_path,
            no_gpu,
            timeout_ms,
            command_runner: Arc::new(RealCommandRunner::new()),
        }
    }

    /// Create a new tool executor with custom command runner
    #[must_use]
    pub fn with_runner(
        model_path: String,
        no_gpu: bool,
        timeout_ms: u64,
        runner: Arc<dyn CommandRunner>,
    ) -> Self {
        Self {
            model_path,
            no_gpu,
            timeout_ms,
            command_runner: runner,
        }
    }

    /// Execute apr rosetta inspect (works with any format)
    #[must_use]
    pub fn execute_inspect(&self) -> ToolTestResult {
        let start = std::time::Instant::now();
        let output = self
            .command_runner
            .inspect_model(Path::new(&self.model_path));
        self.build_result_from_output("inspect", output, start)
    }

    /// Execute apr rosetta inspect with metadata verification (T-GH192-01)
    ///
    /// Parses `--json` output and validates that critical model metadata
    /// fields are present and non-zero. This catches models with missing
    /// or corrupted config (e.g., num_heads=0, hidden_size=0).
    ///
    /// Gate: `F-INSPECT-META-001`
    #[must_use]
    pub fn execute_inspect_verified(&self) -> ToolTestResult {
        let start = std::time::Instant::now();

        match crate::differential::run_inspect(Path::new(&self.model_path), "apr") {
            Ok(inspect) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                let mut issues = Vec::new();

                // Verify tensor count is non-zero
                if inspect.tensor_count == 0 {
                    issues.push("tensor_count is 0".to_string());
                }

                // Verify critical metadata (if present, must be non-zero)
                if let Some(heads) = inspect.num_attention_heads {
                    if heads == 0 {
                        issues.push("num_attention_heads is 0".to_string());
                    }
                }

                if let Some(kv_heads) = inspect.num_key_value_heads {
                    if kv_heads == 0 {
                        issues.push("num_key_value_heads is 0".to_string());
                    }
                }

                if let Some(hidden) = inspect.hidden_size {
                    if hidden == 0 {
                        issues.push("hidden_size is 0".to_string());
                    }
                }

                let passed = issues.is_empty();
                let stdout = format!(
                    "tensor_count={}, num_attention_heads={:?}, num_key_value_heads={:?}, \
                     hidden_size={:?}, architecture={:?}",
                    inspect.tensor_count,
                    inspect.num_attention_heads,
                    inspect.num_key_value_heads,
                    inspect.hidden_size,
                    inspect.architecture,
                );

                ToolTestResult {
                    tool: "inspect-verified".to_string(),
                    passed,
                    exit_code: i32::from(!passed),
                    stdout,
                    stderr: if passed {
                        String::new()
                    } else {
                        format!("Metadata issues: {}", issues.join(", "))
                    },
                    duration_ms,
                    gate_id: "F-INSPECT-META-001".to_string(),
                }
            }
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                ToolTestResult {
                    tool: "inspect-verified".to_string(),
                    passed: false,
                    exit_code: -1,
                    stdout: String::new(),
                    stderr: format!("Failed to run inspect: {e}"),
                    duration_ms,
                    gate_id: "F-INSPECT-META-001".to_string(),
                }
            }
        }
    }

    /// Execute apr validate
    #[must_use]
    pub fn execute_validate(&self) -> ToolTestResult {
        let start = std::time::Instant::now();
        let output = self
            .command_runner
            .validate_model(Path::new(&self.model_path));
        self.build_result_from_output("validate", output, start)
    }

    /// Execute apr bench
    #[must_use]
    pub fn execute_bench(&self) -> ToolTestResult {
        let start = std::time::Instant::now();
        let output = self.command_runner.bench_model(Path::new(&self.model_path));
        self.build_result_from_output("bench", output, start)
    }

    /// Execute apr check
    #[must_use]
    pub fn execute_check(&self) -> ToolTestResult {
        let start = std::time::Instant::now();
        let output = self.command_runner.check_model(Path::new(&self.model_path));
        self.build_result_from_output("check", output, start)
    }

    /// Execute apr trace with specified level
    #[must_use]
    pub fn execute_trace(&self, level: &str) -> ToolTestResult {
        let start = std::time::Instant::now();
        let output = self.command_runner.run_inference(
            Path::new(&self.model_path),
            "What is 2+2?",
            8,
            self.no_gpu,
            &["--trace", "--trace-level", level],
        );
        self.build_result_from_output(&format!("trace-{level}"), output, start)
    }

    /// Execute apr profile (standalone command)
    #[must_use]
    pub fn execute_profile(&self) -> ToolTestResult {
        let start = std::time::Instant::now();
        let output = self
            .command_runner
            .profile_model(Path::new(&self.model_path), 1, 2);
        self.build_result_from_output("profile", output, start)
    }

    /// Execute apr profile in CI mode with assertions (F-PROFILE-006)
    ///
    /// Tests the CI mode features:
    /// - `--ci` flag for CI mode with assertion checks
    /// - `--assert-throughput` minimum tok/s assertion
    /// - `--warmup` and `--measure` pass counts
    ///
    /// Returns pass if CI mode runs and reports metrics correctly.
    #[must_use]
    pub fn execute_profile_ci(&self) -> ToolTestResult {
        let start = std::time::Instant::now();

        // Run apr profile in CI mode with lenient assertions
        // Use very low throughput threshold (1 tok/s) to ensure it passes
        let output = self.command_runner.profile_ci(
            Path::new(&self.model_path),
            Some(1.0), // Very lenient: 1 tok/s minimum
            None,      // No p99 assertion
            1,         // warmup
            2,         // measure
            false,     // use default backend
        );

        let duration_ms = start.elapsed().as_millis() as u64;

        // Check if CI features are available
        if output.stderr.contains("unexpected argument")
            || output.stderr.contains("unrecognized")
            || output.stderr.contains("--ci")
        {
            return ToolTestResult {
                tool: "profile-ci".to_string(),
                passed: false,
                exit_code: -2,
                stdout: output.stdout,
                stderr: "Feature not available: apr profile does not support --ci mode".to_string(),
                duration_ms,
                gate_id: "F-PROFILE-006".to_string(),
            };
        }

        // Verify JSON output contains expected CI fields
        let has_passed_field = output.stdout.contains("\"passed\"");
        let has_metrics = output.stdout.contains("throughput") || output.stdout.contains("tok_s");

        let passed = output.exit_code == 0 && (has_passed_field || has_metrics);

        ToolTestResult {
            tool: "profile-ci".to_string(),
            passed,
            exit_code: output.exit_code,
            stdout: output.stdout,
            stderr: output.stderr,
            duration_ms,
            gate_id: "F-PROFILE-006".to_string(),
        }
    }

    /// Execute apr profile CI with assertion failure test (F-PROFILE-007)
    ///
    /// Tests that CI mode correctly fails when assertions are not met.
    /// Uses an impossibly high throughput assertion to guarantee failure.
    #[must_use]
    pub fn execute_profile_ci_assertion_failure(&self) -> ToolTestResult {
        let start = std::time::Instant::now();

        // Run with impossible throughput assertion (1 million tok/s)
        let output = self.command_runner.profile_ci(
            Path::new(&self.model_path),
            Some(1_000_000.0), // Impossible: 1M tok/s
            None,
            1,     // warmup
            1,     // measure
            false, // use default backend
        );

        let duration_ms = start.elapsed().as_millis() as u64;

        // Check if CI features are available
        if output.stderr.contains("unexpected argument") || output.stderr.contains("unrecognized") {
            return ToolTestResult {
                tool: "profile-ci-assertion".to_string(),
                passed: false,
                exit_code: -2,
                stdout: output.stdout,
                stderr: "Feature not available: apr profile does not support --ci mode".to_string(),
                duration_ms,
                gate_id: "F-PROFILE-007".to_string(),
            };
        }

        // CI mode should EXIT 1 when assertion fails
        // The test PASSES if apr correctly returns non-zero exit code
        // or reports failure in output (fallback for older versions)
        let assertion_failed_correctly = output.exit_code == 1
            || output.stdout.contains("\"passed\":false")
            || output.stdout.contains("\"passed\": false")
            || output.stdout.contains("ASSERTIONS FAILED");

        ToolTestResult {
            tool: "profile-ci-assertion".to_string(),
            passed: assertion_failed_correctly,
            exit_code: output.exit_code,
            stdout: output.stdout,
            stderr: output.stderr,
            duration_ms,
            gate_id: "F-PROFILE-007".to_string(),
        }
    }

    /// Execute apr profile with p99 latency assertion (F-PROFILE-008)
    #[must_use]
    pub fn execute_profile_ci_p99(&self) -> ToolTestResult {
        let start = std::time::Instant::now();

        // Run with lenient p99 assertion (10 seconds max)
        let output = self.command_runner.profile_ci(
            Path::new(&self.model_path),
            None,           // No throughput assertion
            Some(10_000.0), // 10 seconds max p99
            1,              // warmup
            2,              // measure
            false,          // use default backend
        );

        let duration_ms = start.elapsed().as_millis() as u64;

        // Check if p99 assertion feature is available
        if output.stderr.contains("unexpected argument") || output.stderr.contains("--assert-p99") {
            return ToolTestResult {
                tool: "profile-ci-p99".to_string(),
                passed: false,
                exit_code: -2,
                stdout: output.stdout,
                stderr: "Feature not available: apr profile does not support --assert-p99"
                    .to_string(),
                duration_ms,
                gate_id: "F-PROFILE-008".to_string(),
            };
        }

        // Verify p99 metric is in output
        let has_p99 = output.stdout.contains("p99") || output.stdout.contains("latency");
        let passed = output.exit_code == 0 && has_p99;

        ToolTestResult {
            tool: "profile-ci-p99".to_string(),
            passed,
            exit_code: output.exit_code,
            stdout: output.stdout,
            stderr: output.stderr,
            duration_ms,
            gate_id: "F-PROFILE-008".to_string(),
        }
    }

    /// Execute apr profile with flamegraph output (F-PROFILE-002)
    ///
    /// Tests that profile can generate valid SVG flamegraph output.
    /// This feature may not be available in all apr versions.
    #[must_use]
    pub fn execute_profile_flamegraph(&self, output_path: &std::path::Path) -> ToolTestResult {
        let start = std::time::Instant::now();

        let svg_path = output_path.join("profile_flamegraph.svg");
        let output = self.command_runner.profile_with_flamegraph(
            Path::new(&self.model_path),
            &svg_path,
            self.no_gpu,
        );
        let duration_ms = start.elapsed().as_millis() as u64;

        // If apr doesn't support --profile-output, it will error
        if output.stderr.contains("unexpected argument") || output.stderr.contains("unrecognized") {
            return ToolTestResult {
                tool: "profile-flamegraph".to_string(),
                passed: false,
                exit_code: -2,
                stdout: output.stdout,
                stderr: "Feature not available: apr does not support --profile-output".to_string(),
                duration_ms,
                gate_id: "F-PROFILE-002".to_string(),
            };
        }

        // Check if flamegraph was generated
        let flamegraph_exists = svg_path.exists();
        let flamegraph_valid = if flamegraph_exists {
            std::fs::read_to_string(&svg_path)
                .map(|content| content.contains("<svg") && content.contains("</svg>"))
                .unwrap_or(false)
        } else {
            false
        };

        ToolTestResult {
            tool: "profile-flamegraph".to_string(),
            passed: flamegraph_valid,
            exit_code: i32::from(!flamegraph_valid),
            stdout: format!("Flamegraph exists: {flamegraph_exists}, valid: {flamegraph_valid}"),
            stderr: output.stderr,
            duration_ms,
            gate_id: "F-PROFILE-002".to_string(),
        }
    }

    /// Execute apr profile with focus filtering (F-PROFILE-003)
    ///
    /// Tests that profile --focus option works to limit scope.
    /// This feature may not be available in all apr versions.
    #[must_use]
    pub fn execute_profile_focus(&self, focus: &str) -> ToolTestResult {
        let start = std::time::Instant::now();

        let output =
            self.command_runner
                .profile_with_focus(Path::new(&self.model_path), focus, self.no_gpu);
        let duration_ms = start.elapsed().as_millis() as u64;

        // If apr doesn't support --focus, it will error
        if output.stderr.contains("unexpected argument") || output.stderr.contains("unrecognized") {
            return ToolTestResult {
                tool: "profile-focus".to_string(),
                passed: false,
                exit_code: -2,
                stdout: output.stdout,
                stderr: format!("Feature not available: apr does not support --focus {focus}"),
                duration_ms,
                gate_id: "F-PROFILE-003".to_string(),
            };
        }

        let passed = output.success;

        ToolTestResult {
            tool: "profile-focus".to_string(),
            passed,
            exit_code: output.exit_code,
            stdout: output.stdout,
            stderr: output.stderr,
            duration_ms,
            gate_id: "F-PROFILE-003".to_string(),
        }
    }
}

include!("executor_tools_backend_equivalence.rs");
