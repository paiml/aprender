
/// Import battery: execute format import once, run 5 validation checks.
///
/// Gate IDs: T2-IMPORT-{001,SIZE-001,TENSOR-001,LOAD-001,INFER-001}
impl Executor {
    /// Run a battery of 5 import validation checks.
    ///
    /// 1. Import exits 0
    /// 2. Output file reasonable size (within 2x of source)
    /// 3. Tensor count matches source
    /// 4. Output loads via `apr validate`
    /// 5. Quick inference on imported model produces non-garbage output
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn run_import_battery(
        &self,
        source_path: &str,
        scenario: &QaScenario,
        _source_format: &str,
    ) -> Vec<Evidence> {
        let start = Instant::now();
        let mut results = Vec::with_capacity(5);

        let output_path = PathBuf::from(format!(
            "/tmp/qa-import-{}.apr",
            scenario.model.name
        ));

        // Check 1: Import exits 0
        let import_output = self.command_runner.import_model(
            Path::new(source_path),
            &output_path,
        );
        let duration = start.elapsed().as_millis() as u64;

        if !import_output.success {
            results.push(Evidence::falsified(
                "T2-IMPORT-001",
                scenario.clone(),
                format!("Import failed (exit {}): {}", import_output.exit_code, import_output.stderr),
                &import_output.stdout,
                duration,
            ));
            return results;
        }
        results.push(Evidence::corroborated(
            "T2-IMPORT-001",
            scenario.clone(),
            "Import succeeded",
            duration,
        ));

        let import_json: serde_json::Value = match serde_json::from_str(&import_output.stdout) {
            Ok(v) => v,
            Err(e) => {
                results.push(Evidence::falsified(
                    "T2-IMPORT-001",
                    scenario.clone(),
                    format!(
                        "Import exited 0 but produced invalid JSON: {e}. Stdout: {}",
                        Self::truncate_output(&import_output.stdout),
                    ),
                    &import_output.stdout,
                    start.elapsed().as_millis() as u64,
                ));
                return results;
            }
        };

        // Check 2: Output file reasonable size (within 2x of source)
        let output_size = import_json
            .get("output_size_bytes")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let input_size = Self::get_file_size(source_path);
        let duration = start.elapsed().as_millis() as u64;

        if output_size > 0 && input_size > 0 && output_size <= input_size * 2 {
            results.push(Evidence::corroborated(
                "T2-IMPORT-SIZE-001",
                scenario.clone(),
                &format!(
                    "Import output reasonable size: {output_size} bytes (source: {input_size})"
                ),
                duration,
            ));
        } else {
            results.push(Evidence::falsified(
                "T2-IMPORT-SIZE-001",
                scenario.clone(),
                format!(
                    "Import output unreasonable: output={output_size}, source={input_size} (>2x)"
                ),
                &import_output.stdout,
                duration,
            ));
        }

        // Check 3: Tensor count matches source
        let output_tensor_count = import_json
            .get("tensor_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let source_inspect = self.command_runner.inspect_model_json(Path::new(source_path));
        let source_tensor_count = serde_json::from_str::<serde_json::Value>(&source_inspect.stdout)
            .ok()
            .and_then(|v| v.get("tensor_count")?.as_u64())
            .unwrap_or(0);
        let duration = start.elapsed().as_millis() as u64;

        if output_tensor_count > 0 && output_tensor_count == source_tensor_count {
            results.push(Evidence::corroborated(
                "T2-IMPORT-TENSOR-001",
                scenario.clone(),
                &format!("Tensor count preserved: {output_tensor_count}"),
                duration,
            ));
        } else {
            results.push(Evidence::falsified(
                "T2-IMPORT-TENSOR-001",
                scenario.clone(),
                format!(
                    "Tensor count mismatch: source={source_tensor_count}, output={output_tensor_count}"
                ),
                &import_output.stdout,
                duration,
            ));
        }

        // Check 4: Output loads via apr validate
        let validate_output = self.command_runner.validate_model_strict(&output_path);
        let duration = start.elapsed().as_millis() as u64;

        if validate_output.success {
            results.push(Evidence::corroborated(
                "T2-IMPORT-LOAD-001",
                scenario.clone(),
                "Imported model validates successfully",
                duration,
            ));
        } else {
            results.push(Evidence::falsified(
                "T2-IMPORT-LOAD-001",
                scenario.clone(),
                format!("Imported model validation failed: {}", validate_output.stderr),
                &validate_output.stdout,
                duration,
            ));
        }

        // Check 5: Quick inference on imported model
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
                        "T2-IMPORT-INFER-001",
                        scenario.clone(),
                        &format!("Imported inference OK: {text}"),
                        duration,
                    ));
                }
                apr_qa_gen::OracleResult::Falsified { reason, .. } => {
                    results.push(Evidence::falsified(
                        "T2-IMPORT-INFER-001",
                        scenario.clone(),
                        format!("Imported inference garbage: {reason}"),
                        &text,
                        duration,
                    ));
                }
            }
        } else {
            results.push(Evidence::falsified(
                "T2-IMPORT-INFER-001",
                scenario.clone(),
                format!("Imported inference failed: {}", infer_output.stderr),
                &infer_output.stdout,
                duration,
            ));
        }

        results
    }
}
