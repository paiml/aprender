/// Executor gate methods for tensor template validation
impl Executor {

    /// G0-TENSOR Pre-flight Check: Validates tensor names against family YAML template (PMAT-271)
    ///
    /// Compares actual tensor names from the model against expected names from the
    /// family contract's tensor_template. Reports missing or unexpected tensors.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the model file or directory
    /// * `model_id` - Model identifier for evidence tracking
    /// * `family` - Model family identifier (e.g., "qwen2")
    /// * `size_variant` - Size variant identifier (e.g., "0.5b", "7b")
    /// * `aprender_path` - Path to aprender contracts directory
    ///
    /// # Returns
    ///
    /// (passed_count, failed_count) - evidence is added to collector
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn run_g0_tensor_template_check(
        &mut self,
        model_path: &Path,
        model_id: &ModelId,
        family: &str,
        size_variant: &str,
        aprender_path: Option<&str>,
    ) -> (usize, usize) {
        let start = Instant::now();

        // Load family contract
        let registry_path = aprender_path.unwrap_or(crate::family_contract::DEFAULT_APRENDER_PATH);
        let mut registry = crate::family_contract::FamilyRegistry::with_path(registry_path);

        // Try to load the family contract
        let contract = match registry.load_family(family) {
            Ok(c) => c.clone(),
            Err(e) => {
                // Family contract not found — cannot validate, emit skipped (not corroborated).
                // Popperian: untested hypothesis has NOT survived falsification (Bug #79).
                let ev = Evidence::skipped(
                    "G0-TENSOR-001",
                    Self::validate_scenario(model_id),
                    &format!("G0 SKIP: Family contract not found for '{family}': {e}"),
                );
                self.collector.add(ev);
                return (0, 0);
            }
        };

        // Get expected tensor names from family YAML
        let expected_tensors = contract.required_tensors_for_size(size_variant);
        if expected_tensors.is_empty() {
            // No tensor template — cannot validate, emit skipped (not corroborated).
            // Popperian: untested hypothesis has NOT survived falsification (Bug #80).
            let ev = Evidence::skipped(
                "G0-TENSOR-001",
                Self::validate_scenario(model_id),
                &format!("G0 SKIP: No tensor template for {family}/{size_variant}"),
            );
            self.collector.add(ev);
            return (0, 0);
        }

        // Get actual tensor names from the model via inspect
        let files = Self::find_safetensors_files(model_path);
        if files.is_empty() {
            // No safetensors files — cannot validate tensors, emit skipped (not corroborated).
            // Popperian: untested hypothesis has NOT survived falsification (Bug #81).
            let ev = Evidence::skipped(
                "G0-TENSOR-001",
                Self::validate_scenario(model_id),
                "G0 SKIP: No safetensors files found for tensor template validation",
            );
            self.collector.add(ev);
            return (0, 0);
        }

        // Inspect the first safetensors file to get tensor names
        let inspect_output = self.command_runner.inspect_model_json(&files[0]);
        let duration = start.elapsed().as_millis() as u64;

        if !inspect_output.success {
            let ev = Evidence::falsified(
                "G0-TENSOR-001",
                Self::validate_scenario(model_id),
                &format!(
                    "G0 FAIL: Could not inspect model: {}",
                    inspect_output.stderr
                ),
                &inspect_output.stdout,
                duration,
            );
            self.collector.add(ev);
            return (0, 1);
        }

        // Parse tensor names from JSON output
        let actual_tensors: Vec<String> = match serde_json::from_str::<serde_json::Value>(
            &inspect_output.stdout,
        ) {
            Ok(val) => val
                .get("tensor_names")
                .cloned()
                .and_then(|v| serde_json::from_value(v).ok())
                .unwrap_or_default(),
            Err(e) => {
                // Inspect succeeded but returned malformed JSON — falsify
                let ev = Evidence::falsified(
                    "G0-TENSOR-001",
                    Self::validate_scenario(model_id),
                    &format!("G0 FAIL: Inspect returned invalid JSON: {e}"),
                    &inspect_output.stdout,
                    duration,
                );
                self.collector.add(ev);
                return (0, 1);
            }
        };

        if actual_tensors.is_empty() {
            // Inspect returned valid JSON but no tensor_names field — falsify
            let ev = Evidence::falsified(
                "G0-TENSOR-001",
                Self::validate_scenario(model_id),
                "G0 FAIL: Model inspect returned no tensor names (missing or empty tensor_names field)",
                &inspect_output.stdout,
                duration,
            );
            self.collector.add(ev);
            return (0, 1);
        }

        // Check for missing expected tensors
        let missing: Vec<_> = expected_tensors
            .iter()
            .filter(|t| !actual_tensors.contains(t))
            .collect();

        if missing.is_empty() {
            let ev = Evidence::corroborated(
                "G0-TENSOR-001",
                Self::validate_scenario(model_id),
                &format!(
                    "G0 PASS: All {} expected tensors from {}/{} template present",
                    expected_tensors.len(),
                    family,
                    size_variant
                ),
                duration,
            );
            self.collector.add(ev);
            (1, 0)
        } else {
            let missing_list = missing
                .iter()
                .take(5)
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            let more = if missing.len() > 5 {
                format!(" ... and {} more", missing.len() - 5)
            } else {
                String::new()
            };
            let ev = Evidence::falsified(
                "G0-TENSOR-001",
                Self::validate_scenario(model_id),
                &format!(
                    "G0 FAIL: Missing {} tensors from {}/{} template: {}{}",
                    missing.len(),
                    family,
                    size_variant,
                    missing_list,
                    more
                ),
                &inspect_output.stdout,
                duration,
            );
            self.collector.add(ev);
            (0, 1)
        }
    }

}
