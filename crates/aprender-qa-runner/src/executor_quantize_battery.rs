
/// Quantize battery: execute quantization once, run 6 validation checks.
///
/// Gate IDs: T1-QUANT-{001,SIZE-001,TENSOR-001,LOAD-001,INFER-001,DTYPE-001}
impl Executor {
    /// Run a battery of 6 quantization validation checks.
    ///
    /// 1. Quantize exits 0
    /// 2. Output file smaller than input
    /// 3. Tensor count matches source
    /// 4. Output loads via `apr validate`
    /// 5. Quick inference on quantized model produces non-garbage output
    /// 6. Output dtype matches requested scheme
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn run_quantize_battery(
        &self,
        model_path: &str,
        scenario: &QaScenario,
        scheme: &str,
    ) -> Vec<Evidence> {
        let start = Instant::now();
        let mut results = Vec::with_capacity(6);

        let output_path = PathBuf::from(format!(
            "/tmp/qa-quantize-{}-{scheme}.apr",
            scenario.model.name
        ));

        // Check 1: Quantize exits 0
        let quant_output = self.command_runner.quantize_model(
            Path::new(model_path),
            &output_path,
            scheme,
        );
        let duration = start.elapsed().as_millis() as u64;

        if !quant_output.success {
            results.push(Evidence::falsified(
                "T1-QUANT-001",
                scenario.clone(),
                format!("Quantization failed (exit {}): {}", quant_output.exit_code, quant_output.stderr),
                &quant_output.stdout,
                duration,
            ));
            return results;
        }
        results.push(Evidence::corroborated(
            "T1-QUANT-001",
            scenario.clone(),
            &format!("Quantization to {scheme} succeeded"),
            duration,
        ));

        // Parse JSON output for metadata — Jidoka: invalid JSON is a falsifiable defect,
        // not something to silently default away (Bug #31).
        let quant_json: serde_json::Value = match serde_json::from_str(&quant_output.stdout) {
            Ok(v) => v,
            Err(e) => {
                results.push(Evidence::falsified(
                    "T1-QUANT-001",
                    scenario.clone(),
                    format!(
                        "Quantization exited 0 but produced invalid JSON: {e}. Stdout: {}",
                        Self::truncate_output(&quant_output.stdout),
                    ),
                    &quant_output.stdout,
                    start.elapsed().as_millis() as u64,
                ));
                return results;
            }
        };

        // Check 2: Output file smaller than input
        let output_size = quant_json
            .get("output_size_bytes")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let input_size = Self::get_file_size(model_path);
        let duration = start.elapsed().as_millis() as u64;

        if output_size > 0 && input_size > 0 && output_size < input_size {
            results.push(Evidence::corroborated(
                "T1-QUANT-SIZE-001",
                scenario.clone(),
                &format!(
                    "Quantized output smaller: {output_size} < {input_size} bytes ({:.1}% reduction)",
                    (1.0 - output_size as f64 / input_size as f64) * 100.0
                ),
                duration,
            ));
        } else {
            results.push(Evidence::falsified(
                "T1-QUANT-SIZE-001",
                scenario.clone(),
                format!("Quantized output not smaller: output={output_size}, input={input_size}"),
                &quant_output.stdout,
                duration,
            ));
        }

        // Check 3: Tensor count matches source
        let output_tensor_count = quant_json
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
                "T1-QUANT-TENSOR-001",
                scenario.clone(),
                &format!("Tensor count preserved: {output_tensor_count}"),
                duration,
            ));
        } else {
            results.push(Evidence::falsified(
                "T1-QUANT-TENSOR-001",
                scenario.clone(),
                format!(
                    "Tensor count mismatch: source={source_tensor_count}, output={output_tensor_count}"
                ),
                &quant_output.stdout,
                duration,
            ));
        }

        // Check 4: Output loads via apr validate
        let validate_output = self.command_runner.validate_model_strict(&output_path);
        let duration = start.elapsed().as_millis() as u64;

        if validate_output.success {
            results.push(Evidence::corroborated(
                "T1-QUANT-LOAD-001",
                scenario.clone(),
                "Quantized model validates successfully",
                duration,
            ));
        } else {
            results.push(Evidence::falsified(
                "T1-QUANT-LOAD-001",
                scenario.clone(),
                format!("Quantized model validation failed: {}", validate_output.stderr),
                &validate_output.stdout,
                duration,
            ));
        }

        // Check 5: Quick inference on quantized model
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
                        "T1-QUANT-INFER-001",
                        scenario.clone(),
                        &format!("Quantized inference OK: {text}"),
                        duration,
                    ));
                }
                apr_qa_gen::OracleResult::Falsified { reason, .. } => {
                    results.push(Evidence::falsified(
                        "T1-QUANT-INFER-001",
                        scenario.clone(),
                        format!("Quantized inference garbage: {reason}"),
                        &text,
                        duration,
                    ));
                }
            }
        } else {
            results.push(Evidence::falsified(
                "T1-QUANT-INFER-001",
                scenario.clone(),
                format!("Quantized inference failed: {}", infer_output.stderr),
                &infer_output.stdout,
                duration,
            ));
        }

        // Check 6: Output dtype matches requested scheme
        let dtype = quant_json
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let duration = start.elapsed().as_millis() as u64;

        if !dtype.is_empty() && dtype.eq_ignore_ascii_case(scheme) {
            results.push(Evidence::corroborated(
                "T1-QUANT-DTYPE-001",
                scenario.clone(),
                &format!("Output dtype matches: {dtype}"),
                duration,
            ));
        } else {
            results.push(Evidence::falsified(
                "T1-QUANT-DTYPE-001",
                scenario.clone(),
                format!("Dtype mismatch: expected={scheme}, actual={dtype}"),
                &quant_output.stdout,
                duration,
            ));
        }

        results
    }

    /// Get file size (or directory total) for comparison
    fn get_file_size(path: &str) -> u64 {
        let p = Path::new(path);
        if p.is_file() {
            std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)
        } else if p.is_dir() {
            // Sum up all files in directory
            std::fs::read_dir(p)
                .map(|entries| {
                    entries
                        .filter_map(std::result::Result::ok)
                        .filter_map(|e| e.metadata().ok())
                        .filter(std::fs::Metadata::is_file)
                        .map(|m| m.len())
                        .sum()
                })
                .unwrap_or(0)
        } else {
            0
        }
    }
}
