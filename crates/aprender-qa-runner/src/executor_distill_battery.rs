
/// Distill battery: execute knowledge distillation once, run 5 validation checks.
///
/// Gate IDs: T4-DISTILL-{001,SIZE-001,LOAD-001,INFER-001,LOSS-001}
impl Executor {
    /// Run a battery of 5 distillation validation checks.
    ///
    /// 1. Distill exits 0
    /// 2. Student smaller than teacher
    /// 3. Student loads via `apr validate`
    /// 4. Quick inference on student produces non-garbage output
    /// 5. Training loss decreasing (final < initial)
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn run_distill_battery(
        &self,
        teacher_path: &str,
        scenario: &QaScenario,
        student_model: &str,
        data_path: &str,
    ) -> Vec<Evidence> {
        let start = Instant::now();
        let mut results = Vec::with_capacity(5);

        let student_path = PathBuf::from(student_model);
        let output_path = PathBuf::from(format!(
            "/tmp/qa-distill-{}.apr",
            scenario.model.name
        ));

        // Check 1: Distill exits 0
        let distill_output = self.command_runner.distill_model(
            Path::new(teacher_path),
            &student_path,
            &output_path,
            data_path,
        );
        let duration = start.elapsed().as_millis() as u64;

        if !distill_output.success {
            results.push(Evidence::falsified(
                "T4-DISTILL-001",
                scenario.clone(),
                format!(
                    "Distillation failed (exit {}): {}",
                    distill_output.exit_code, distill_output.stderr
                ),
                &distill_output.stdout,
                duration,
            ));
            return results;
        }
        results.push(Evidence::corroborated(
            "T4-DISTILL-001",
            scenario.clone(),
            "Distillation succeeded",
            duration,
        ));

        let distill_json: serde_json::Value = match serde_json::from_str(&distill_output.stdout) {
            Ok(v) => v,
            Err(e) => {
                results.push(Evidence::falsified(
                    "T4-DISTILL-001",
                    scenario.clone(),
                    format!(
                        "Distill exited 0 but produced invalid JSON: {e}. Stdout: {}",
                        Self::truncate_output(&distill_output.stdout),
                    ),
                    &distill_output.stdout,
                    start.elapsed().as_millis() as u64,
                ));
                return results;
            }
        };

        // Check 2: Student smaller than teacher
        let student_size = distill_json
            .get("output_size_bytes")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let teacher_size = distill_json
            .get("teacher_size_bytes")
            .and_then(serde_json::Value::as_u64)
            .or_else(|| Some(Self::get_file_size(teacher_path)))
            .unwrap_or(0);
        let duration = start.elapsed().as_millis() as u64;

        if student_size > 0 && teacher_size > 0 && student_size < teacher_size {
            results.push(Evidence::corroborated(
                "T4-DISTILL-SIZE-001",
                scenario.clone(),
                &format!(
                    "Student smaller than teacher: {student_size} < {teacher_size} bytes ({:.1}% of teacher)",
                    student_size as f64 / teacher_size as f64 * 100.0
                ),
                duration,
            ));
        } else {
            results.push(Evidence::falsified(
                "T4-DISTILL-SIZE-001",
                scenario.clone(),
                format!(
                    "Student not smaller: student={student_size}, teacher={teacher_size}"
                ),
                &distill_output.stdout,
                duration,
            ));
        }

        // Check 3: Student loads via apr validate
        let validate_output = self.command_runner.validate_model_strict(&output_path);
        let duration = start.elapsed().as_millis() as u64;

        if validate_output.success {
            results.push(Evidence::corroborated(
                "T4-DISTILL-LOAD-001",
                scenario.clone(),
                "Distilled student model validates successfully",
                duration,
            ));
        } else {
            results.push(Evidence::falsified(
                "T4-DISTILL-LOAD-001",
                scenario.clone(),
                format!("Student model validation failed: {}", validate_output.stderr),
                &validate_output.stdout,
                duration,
            ));
        }

        // Check 4: Quick inference on student
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
                        "T4-DISTILL-INFER-001",
                        scenario.clone(),
                        &format!("Student inference OK: {text}"),
                        duration,
                    ));
                }
                apr_qa_gen::OracleResult::Falsified { reason, .. } => {
                    results.push(Evidence::falsified(
                        "T4-DISTILL-INFER-001",
                        scenario.clone(),
                        format!("Student inference garbage: {reason}"),
                        &text,
                        duration,
                    ));
                }
            }
        } else {
            results.push(Evidence::falsified(
                "T4-DISTILL-INFER-001",
                scenario.clone(),
                format!("Student inference failed: {}", infer_output.stderr),
                &infer_output.stdout,
                duration,
            ));
        }

        // Check 5: Training loss decreasing
        let initial_loss = distill_json
            .get("initial_loss")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);
        let final_loss = distill_json
            .get("final_loss")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);
        let duration = start.elapsed().as_millis() as u64;

        if initial_loss > 0.0 && final_loss > 0.0 && final_loss < initial_loss {
            results.push(Evidence::corroborated(
                "T4-DISTILL-LOSS-001",
                scenario.clone(),
                &format!(
                    "Loss decreasing: {initial_loss:.4} -> {final_loss:.4} ({:.1}% reduction)",
                    (1.0 - final_loss / initial_loss) * 100.0
                ),
                duration,
            ));
        } else {
            results.push(Evidence::falsified(
                "T4-DISTILL-LOSS-001",
                scenario.clone(),
                format!(
                    "Loss not decreasing: initial={initial_loss:.4}, final={final_loss:.4}"
                ),
                &distill_output.stdout,
                duration,
            ));
        }

        results
    }
}
