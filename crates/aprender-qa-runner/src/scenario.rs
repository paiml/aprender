
// G0 gateway checks — see executor_gates.rs
include!("executor_gates.rs");

// Serve battery: spawn once, run 19 endpoint checks, kill once
// PMAT-534: cfg-gated during monorepo migration (depends on jugar-probar)
#[cfg(feature = "serve-battery")]
include!("executor_serve_battery.rs");

// Transformation batteries: quantize, import, prune, distill
include!("executor_quantize_battery.rs");
include!("executor_import_battery.rs");
include!("executor_prune_battery.rs");
include!("executor_distill_battery.rs");

impl Executor {
    /// Execute a single scenario
    fn execute_scenario(&self, scenario: &QaScenario) -> Evidence {
        let start = Instant::now();

        let (output, stderr, exit_code, tps, skipped) = self.subprocess_execution(scenario);

        if skipped {
            let gate_id = format!("F-{}-001", scenario.mqs_category());
            return Evidence::skipped(
                &gate_id,
                scenario.clone(),
                format!("Format {:?} not available for model file", scenario.format),
            );
        }

        let duration = start.elapsed().as_millis() as u64;

        // Check for crash (negative exit code = signal)
        if exit_code < 0 {
            return Evidence::crashed(
                "G3-STABLE",
                scenario.clone(),
                stderr.as_deref().unwrap_or("Process crashed"),
                exit_code,
                duration,
            );
        }

        // Check for command failure (non-zero exit code)
        if exit_code > 0 {
            let error_msg = stderr
                .as_deref()
                .unwrap_or("Command failed with non-zero exit code");
            let mut evidence = Evidence::falsified(
                "G2-BASIC",
                scenario.clone(),
                format!("Command failed (exit {exit_code}): {error_msg}"),
                &output,
                duration,
            );
            evidence.exit_code = Some(exit_code);
            evidence.stderr = stderr;
            return evidence;
        }

        // Evaluate the output
        let oracle_result = scenario.evaluate(&output);

        let gate_id = format!("F-{}-001", scenario.mqs_category());

        match oracle_result {
            apr_qa_gen::OracleResult::Corroborated { evidence: _reason } => {
                let mut evidence =
                    Evidence::corroborated(&gate_id, scenario.clone(), &output, duration);
                evidence.metrics = PerformanceMetrics {
                    duration_ms: duration,
                    tokens_per_second: tps,
                    total_tokens: Some(32),
                    time_to_first_token_ms: None,
                    memory_peak_mb: None,
                };
                if let Some(ref err) = stderr {
                    evidence.stderr = Some(err.clone());
                }
                evidence
            }
            apr_qa_gen::OracleResult::Falsified {
                reason,
                evidence: _,
            } => {
                let mut evidence =
                    Evidence::falsified(&gate_id, scenario.clone(), reason, &output, duration);
                if let Some(ref err) = stderr {
                    evidence.stderr = Some(err.clone());
                }
                evidence
            }
        }
    }

    /// Execute via subprocess (real apr commands)
    /// On failure, re-runs with --trace for full diagnostics
    ///
    /// Returns `(stdout, stderr, exit_code, tps, skipped)`.
    /// When `skipped` is `true` the scenario format is unavailable for the
    /// model file and the caller should emit `Evidence::skipped`.
    fn subprocess_execution(
        &self,
        scenario: &QaScenario,
    ) -> (String, Option<String>, i32, Option<f64>, bool) {
        let Some(model_path) = self.resolve_model_path(scenario) else {
            return (String::new(), None, 0, None, true);
        };

        // Fix 201: Use per-scenario backend, not global no_gpu flag
        let no_gpu = scenario.backend == Backend::Cpu;

        // Fix 200: Dispatch by modality instead of always using `apr run`
        let output = match scenario.modality {
            Modality::Run => self.command_runner.run_inference(
                Path::new(&model_path),
                &scenario.prompt,
                32,
                no_gpu,
                &["--benchmark", "--json"],
            ),
            Modality::Chat => self.command_runner.run_chat(
                Path::new(&model_path),
                &scenario.prompt,
                no_gpu,
                &["--json"],
            ),
            // Serve scenarios are handled via battery in execute_scenarios().
            // If we somehow reach here, treat as unreachable with a graceful fallback.
            Modality::Serve => {
                return (
                    String::new(),
                    Some("Serve scenarios must be executed via serve battery".to_string()),
                    1,
                    None,
                    false,
                );
            }
            // Transformation scenarios are handled via dedicated batteries.
            Modality::Quantize | Modality::Import | Modality::Prune | Modality::Distill => {
                return (
                    String::new(),
                    Some("Transformation scenarios must be executed via transformation battery".to_string()),
                    1,
                    None,
                    false,
                );
            }
        };

        // Try to parse tok/s from JSON output
        let tps = Self::parse_tps_from_output(&output.stdout);

        // Extract the actual generated text (not the JSON benchmark data)
        let generated_text = Self::extract_generated_text(&output.stdout);

        // On failure, re-run with tracing for full diagnostics
        let (final_stderr, final_exit_code) = if output.success {
            (
                if output.stderr.is_empty() {
                    None
                } else {
                    Some(output.stderr)
                },
                output.exit_code,
            )
        } else {
            // Trace retry uses the same modality as the original command
            let trace_output = match scenario.modality {
                Modality::Run | Modality::Serve
                | Modality::Quantize | Modality::Import
                | Modality::Prune | Modality::Distill => self.command_runner.run_inference(
                    Path::new(&model_path),
                    &scenario.prompt,
                    32,
                    no_gpu,
                    &["--trace"],
                ),
                Modality::Chat => self.command_runner.run_chat(
                    Path::new(&model_path),
                    &scenario.prompt,
                    no_gpu,
                    &["--trace"],
                ),
            };
            let mut full_trace = output.stderr.clone();
            if !trace_output.stderr.is_empty() {
                full_trace.push_str("\n--- TRACE OUTPUT ---\n");
                full_trace.push_str(&trace_output.stderr);
            }
            if !trace_output.stdout.is_empty() {
                full_trace.push_str("\n--- TRACE STDOUT ---\n");
                full_trace.push_str(&trace_output.stdout);
            }
            (Some(full_trace), output.exit_code)
        };

        (generated_text, final_stderr, final_exit_code, tps, false)
    }
}

// Model resolution + workspace — see executor_resolution.rs
include!("executor_resolution.rs");

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

// ToolExecutor — see executor_tools.rs
include!("executor_tools.rs");

#[cfg(test)]
#[path = "executor_tests_a.rs"]
mod tests_a;

#[cfg(test)]
#[path = "executor_tests_b.rs"]
mod tests_b;

#[cfg(test)]
#[path = "executor_tests_c.rs"]
mod tests_c;

#[cfg(test)]
#[path = "executor_tests_d.rs"]
mod tests_d;

#[cfg(test)]
#[path = "executor_tests_e.rs"]
mod tests_e;

#[cfg(test)]
#[path = "executor_tests_f.rs"]
mod tests_f;

#[cfg(test)]
#[path = "executor_tests_serve_battery.rs"]
mod tests_serve_battery;

#[cfg(test)]
#[path = "executor_tests_quantize_battery.rs"]
mod tests_quantize_battery;

#[cfg(test)]
#[path = "executor_tests_import_battery.rs"]
mod tests_import_battery;

#[cfg(test)]
#[path = "executor_tests_prune_battery.rs"]
mod tests_prune_battery;

#[cfg(test)]
#[path = "executor_tests_distill_battery.rs"]
mod tests_distill_battery;

#[cfg(test)]
#[path = "executor_tests_transformation.rs"]
mod tests_transformation;
