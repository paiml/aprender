
/// Prune battery: execute weight pruning once, run 6 validation checks.
///
/// Gate IDs: T3-PRUNE-{001,SIZE-001,RATIO-001,LOAD-001,INFER-001,TENSOR-001}
impl Executor {
    /// Run a battery of 6 pruning validation checks.
    ///
    /// 1. Prune exits 0
    /// 2. Output smaller than input
    /// 3. Actual sparsity near target (within 5%)
    /// 4. Output loads via `apr validate`
    /// 5. Quick inference on pruned model produces non-garbage output
    /// 6. Tensor count preserved
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn run_prune_battery(
        &self,
        model_path: &str,
        scenario: &QaScenario,
        method: &str,
        target_ratio: f64,
    ) -> Vec<Evidence> {
        let start = Instant::now();
        let mut results = Vec::with_capacity(6);

        let output_path = PathBuf::from(format!(
            "/tmp/qa-prune-{}-{method}.apr",
            scenario.model.name
        ));

        // Check 1: Prune exits 0
        let prune_output = self.command_runner.prune_model(
            Path::new(model_path),
            &output_path,
            method,
            target_ratio,
        );
        let duration = start.elapsed().as_millis() as u64;

        if !prune_output.success {
            results.push(Evidence::falsified(
                "T3-PRUNE-001",
                scenario.clone(),
                format!("Prune failed (exit {}): {}", prune_output.exit_code, prune_output.stderr),
                &prune_output.stdout,
                duration,
            ));
            return results;
        }
        results.push(Evidence::corroborated(
            "T3-PRUNE-001",
            scenario.clone(),
            &format!("Pruning with {method} at ratio {target_ratio} succeeded"),
            duration,
        ));

        let prune_json: serde_json::Value = match serde_json::from_str(&prune_output.stdout) {
            Ok(v) => v,
            Err(e) => {
                results.push(Evidence::falsified(
                    "T3-PRUNE-001",
                    scenario.clone(),
                    format!(
                        "Prune exited 0 but produced invalid JSON: {e}. Stdout: {}",
                        Self::truncate_output(&prune_output.stdout),
                    ),
                    &prune_output.stdout,
                    start.elapsed().as_millis() as u64,
                ));
                return results;
            }
        };

        // Check 2: Output smaller than input
        let output_size = prune_json
            .get("output_size_bytes")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let input_size = Self::get_file_size(model_path);
        let duration = start.elapsed().as_millis() as u64;

        if output_size > 0 && input_size > 0 && output_size < input_size {
            results.push(Evidence::corroborated(
                "T3-PRUNE-SIZE-001",
                scenario.clone(),
                &format!(
                    "Pruned output smaller: {output_size} < {input_size} bytes ({:.1}% reduction)",
                    (1.0 - output_size as f64 / input_size as f64) * 100.0
                ),
                duration,
            ));
        } else {
            results.push(Evidence::falsified(
                "T3-PRUNE-SIZE-001",
                scenario.clone(),
                format!("Pruned output not smaller: output={output_size}, input={input_size}"),
                &prune_output.stdout,
                duration,
            ));
        }

        // Check 3: Actual sparsity near target (within 5%)
        let actual_sparsity = prune_json
            .get("actual_sparsity")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);
        let duration = start.elapsed().as_millis() as u64;
        let sparsity_diff = (actual_sparsity - target_ratio).abs();

        if sparsity_diff <= 0.05 {
            results.push(Evidence::corroborated(
                "T3-PRUNE-RATIO-001",
                scenario.clone(),
                &format!(
                    "Sparsity within tolerance: actual={actual_sparsity:.3}, target={target_ratio:.3} (diff={sparsity_diff:.3})"
                ),
                duration,
            ));
        } else {
            results.push(Evidence::falsified(
                "T3-PRUNE-RATIO-001",
                scenario.clone(),
                format!(
                    "Sparsity outside tolerance: actual={actual_sparsity:.3}, target={target_ratio:.3} (diff={sparsity_diff:.3} > 0.05)"
                ),
                &prune_output.stdout,
                duration,
            ));
        }

        // Check 4: Output loads via apr validate
        let validate_output = self.command_runner.validate_model_strict(&output_path);
        let duration = start.elapsed().as_millis() as u64;

        if validate_output.success {
            results.push(Evidence::corroborated(
                "T3-PRUNE-LOAD-001",
                scenario.clone(),
                "Pruned model validates successfully",
                duration,
            ));
        } else {
            results.push(Evidence::falsified(
                "T3-PRUNE-LOAD-001",
                scenario.clone(),
                format!("Pruned model validation failed: {}", validate_output.stderr),
                &validate_output.stdout,
                duration,
            ));
        }

        // Check 5: Quick inference on pruned model
        let infer_output = self.command_runner.run_inference(
            &output_path,
            "What is 2+2?",
            16,
            true,
            &[],
        );
        let duration = start.elapsed().as_millis() as u64;

        if infer_output.success {
            let text = Self::extract_generated_text(&infer_output.stdout);
            let oracle_result = apr_qa_gen::oracle::select_oracle("What is 2+2?");
            let eval = oracle_result.evaluate("What is 2+2?", &text);
            match eval {
                apr_qa_gen::OracleResult::Corroborated { .. } => {
                    results.push(Evidence::corroborated(
                        "T3-PRUNE-INFER-001",
                        scenario.clone(),
                        &format!("Pruned inference OK: {text}"),
                        duration,
                    ));
                }
                apr_qa_gen::OracleResult::Falsified { reason, .. } => {
                    results.push(Evidence::falsified(
                        "T3-PRUNE-INFER-001",
                        scenario.clone(),
                        format!("Pruned inference garbage: {reason}"),
                        &text,
                        duration,
                    ));
                }
            }
        } else {
            results.push(Evidence::falsified(
                "T3-PRUNE-INFER-001",
                scenario.clone(),
                format!("Pruned inference failed: {}", infer_output.stderr),
                &infer_output.stdout,
                duration,
            ));
        }

        // Check 6: Tensor count preserved
        let output_tensor_count = prune_json
            .get("tensor_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let source_inspect = self.command_runner.inspect_model_json(Path::new(model_path));
        let source_tensor_count = serde_json::from_str::<serde_json::Value>(&source_inspect.stdout)
            .ok()
            .and_then(|v| v.get("tensor_count")?.as_u64())
            .unwrap_or(0);
        let duration = start.elapsed().as_millis() as u64;

        if output_tensor_count > 0 && output_tensor_count == source_tensor_count {
            results.push(Evidence::corroborated(
                "T3-PRUNE-TENSOR-001",
                scenario.clone(),
                &format!("Tensor count preserved: {output_tensor_count}"),
                duration,
            ));
        } else {
            results.push(Evidence::falsified(
                "T3-PRUNE-TENSOR-001",
                scenario.clone(),
                format!(
                    "Tensor count mismatch: source={source_tensor_count}, output={output_tensor_count}"
                ),
                &prune_output.stdout,
                duration,
            ));
        }

        results
    }
}
