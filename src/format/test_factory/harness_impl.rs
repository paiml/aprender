
impl ConversionTestHarness {
    /// Create a new empty harness with a fresh temp directory.
    pub(crate) fn new() -> Self {
        Self {
            dir: TempDir::new().expect("Failed to create temp dir for test harness"),
            input_path: None,
            output_path: None,
            source_tensors: Vec::new(),
            tolerance: ToleranceConfig::default(),
        }
    }

    /// Access the temp directory path.
    pub(crate) fn dir(&self) -> &Path {
        self.dir.path()
    }

    // ----------------------------------------------------------------
    // Setup: write pygmy input files
    // ----------------------------------------------------------------

    /// Write a pygmy SafeTensors file into the temp dir and record
    /// the source tensor data for later verification.
    pub(crate) fn with_safetensors(mut self, config: PygmyConfig) -> Self {
        let bytes = build_pygmy_safetensors_with_config(config.clone());
        let path = self.dir.path().join("input.safetensors");
        fs::write(&path, &bytes).expect("Failed to write pygmy safetensors");

        // Record source tensors for verification
        self.source_tensors = collect_pygmy_tensors(&config);
        self.input_path = Some(path);
        self
    }

    /// Write a pygmy APR file into the temp dir.
    pub(crate) fn with_apr(mut self, config: PygmyConfig) -> Self {
        let bytes = build_pygmy_apr_with_config(config.clone());
        let path = self.dir.path().join("input.apr");
        fs::write(&path, &bytes).expect("Failed to write pygmy apr");

        self.source_tensors = collect_pygmy_tensors(&config);
        self.input_path = Some(path);
        self
    }

    /// Write a Q4K-quantized APR file (GGUF-style names) into the temp dir.
    pub(crate) fn with_apr_q4k(mut self) -> Self {
        let bytes = super::build_pygmy_apr_q4k();
        let path = self.dir.path().join("input.apr");
        fs::write(&path, &bytes).expect("write pygmy q4k apr");
        self.input_path = Some(path);
        self
    }

    /// Write a Q6K-quantized APR file (GGUF-style names) into the temp dir.
    pub(crate) fn with_apr_q6k(mut self) -> Self {
        let bytes = super::build_pygmy_apr_q6k();
        let path = self.dir.path().join("input.apr");
        fs::write(&path, &bytes).expect("write pygmy q6k apr");
        self.input_path = Some(path);
        self
    }

    // ----------------------------------------------------------------
    // Exercise: run real pipeline
    // ----------------------------------------------------------------

    /// Import the input SafeTensors to APR using `apr_import`.
    pub(crate) fn import_to_apr(mut self, options: ImportOptions) -> Self {
        let input = self
            .input_path
            .as_ref()
            .expect("Call with_safetensors() first");
        let output = self.dir.path().join("output.apr");
        let input_str = input.to_string_lossy().to_string();

        let result = apr_import(&input_str, &output, options);
        assert!(
            result.is_ok(),
            "apr_import failed: {:?}",
            result.unwrap_err()
        );

        self.output_path = Some(output);
        self
    }

    /// Import and return the Result (for testing error paths).
    pub(crate) fn try_import_to_apr(
        &self,
        options: ImportOptions,
    ) -> crate::error::Result<crate::format::validation::ValidationReport> {
        let input = self
            .input_path
            .as_ref()
            .expect("Call with_safetensors() first");
        let output = self.dir.path().join("output.apr");
        let input_str = input.to_string_lossy().to_string();
        apr_import(&input_str, &output, options)
    }

    /// Export the output APR back to SafeTensors using `apr_export`.
    pub(crate) fn export_to_safetensors(mut self) -> Self {
        let input = self
            .output_path
            .as_ref()
            .expect("Call import_to_apr() first");
        let output = self.dir.path().join("roundtrip.safetensors");

        let options = ExportOptions {
            format: ExportFormat::SafeTensors,
            quantize: None,
            include_tokenizer: false,
            include_config: false,
            ..Default::default()
        };
        let result = apr_export(input, &output, options);
        assert!(
            result.is_ok(),
            "apr_export failed: {:?}",
            result.unwrap_err()
        );

        self.output_path = Some(output);
        self
    }

    /// Export APR -> GGUF (for T-QKV-03 and T-QKV-04 multi-hop tests).
    pub(crate) fn export_to_gguf(mut self) -> Self {
        let input = self
            .output_path
            .as_ref()
            .expect("Call import_to_apr() first");
        let output = self.dir.path().join("roundtrip.gguf");

        let options = ExportOptions {
            format: ExportFormat::Gguf,
            quantize: None,
            include_tokenizer: false,
            include_config: false,
            ..Default::default()
        };
        let result = apr_export(input, &output, options);
        assert!(
            result.is_ok(),
            "apr_export to GGUF failed: {:?}",
            result.unwrap_err()
        );

        self.output_path = Some(output);
        self
    }

    /// Import from current output (GGUF or SafeTensors) -> APR (for multi-hop chains).
    pub(crate) fn reimport_to_apr(mut self) -> Self {
        let input = self
            .output_path
            .as_ref()
            .expect("No output to reimport from");
        let input_str = input.to_string_lossy().to_string();
        let output = self.dir.path().join("reimported.apr");

        // Create synthetic tokenizer for PMAT-232 compliance (GGUF imports need tokenizer)
        let tokenizer_path = self.dir.path().join("synthetic_tokenizer.json");
        let synthetic_tokenizer = r#"{
                "version": "1.0",
                "model": {
                    "type": "BPE",
                    "vocab": {"<pad>": 0, "<eos>": 1, "<unk>": 2},
                    "merges": []
                }
            }"#;
        std::fs::write(&tokenizer_path, synthetic_tokenizer)
            .expect("Failed to write synthetic tokenizer");

        let options = ImportOptions {
            tokenizer_path: Some(tokenizer_path),
            allow_no_config: true,
            ..ImportOptions::default()
        };
        let result = apr_import(&input_str, &output, options);
        assert!(
            result.is_ok(),
            "reimport to APR failed: {:?}",
            result.unwrap_err()
        );

        self.output_path = Some(output);
        self
    }

    // ----------------------------------------------------------------
    // Verify: read back output and compare
    // ----------------------------------------------------------------

    /// Read back the output APR file from disk and verify tensor data matches source.
    ///
    /// Checks: tensor existence, shape equality, and data values within tolerance.
    /// Panics if no source tensors were recorded (empty config guard).
    pub(crate) fn verify_apr(&self) -> VerificationResult {
        self.assert_source_tensors_present();
        let output = self.require_output_path("run import first");
        let data = fs::read(output).expect("Failed to read output APR");
        let reader = AprV2Reader::from_bytes(&data).expect("Failed to parse output APR");

        let mut mismatches = Vec::new();

        for (name, expected_data, expected_shape) in &self.source_tensors {
            let actual_shape = reader.get_tensor(name).map(|e| e.shape.clone());
            let actual_data = reader.get_tensor_as_f32(name);
            Self::verify_single_tensor(
                name, expected_data, expected_shape,
                actual_shape.as_deref(), actual_data.as_deref(),
                self.tolerance.f32_atol, &mut mismatches,
            );
        }

        Self::check_extra_tensors(
            &self.source_tensors, reader.tensor_names(), &[], &mut mismatches,
        );
        VerificationResult { mismatches }
    }

    /// Read back the output SafeTensors from disk and verify tensor data matches source.
    ///
    /// Checks: tensor existence, shape equality, and data values within tolerance.
    /// Panics if no source tensors were recorded (empty config guard).
    pub(crate) fn verify_safetensors(&self) -> VerificationResult {
        self.assert_source_tensors_present();
        let output = self.require_output_path("run export first");
        let mapped = MappedSafeTensors::open(output).expect("Failed to open output SafeTensors");

        let mut mismatches = Vec::new();

        for (name, expected_data, expected_shape) in &self.source_tensors {
            let actual_shape = mapped.get_metadata(name).map(|m| m.shape.clone());
            let actual_data = mapped.get_tensor(name).ok();
            Self::verify_single_tensor(
                name, expected_data, expected_shape,
                actual_shape.as_deref(), actual_data.as_deref(),
                self.tolerance.f32_atol, &mut mismatches,
            );
        }

        Self::check_extra_tensors(
            &self.source_tensors, mapped.tensor_names(),
            &["__metadata__"], &mut mismatches,
        );
        VerificationResult { mismatches }
    }

    fn assert_source_tensors_present(&self) {
        assert!(
            !self.source_tensors.is_empty(),
            "Cannot verify with 0 source tensors -- use a non-empty PygmyConfig"
        );
    }

    fn require_output_path(&self, context: &str) -> &Path {
        self.output_path
            .as_ref()
            .unwrap_or_else(|| panic!("No output path set -- {context}"))
    }

    /// Verify a single tensor: existence → shape → data values.
    fn verify_single_tensor(
        name: &str,
        expected_data: &[f32],
        expected_shape: &[usize],
        actual_shape: Option<&[usize]>,
        actual_data: Option<&[f32]>,
        tolerance: f32,
        mismatches: &mut Vec<TensorMismatch>,
    ) {
        let Some(shape) = actual_shape else {
            mismatches.push(TensorMismatch {
                tensor_name: name.to_string(),
                kind: MismatchKind::Missing,
            });
            return;
        };

        if shape != expected_shape {
            mismatches.push(TensorMismatch {
                tensor_name: name.to_string(),
                kind: MismatchKind::ShapeMismatch {
                    expected: expected_shape.to_vec(),
                    actual: shape.to_vec(),
                },
            });
            return;
        }

        if let Some(data) = actual_data {
            if let Some((i, (&exp, &act))) = expected_data
                .iter()
                .zip(data.iter())
                .enumerate()
                .find(|(_, (&e, &a))| (e - a).abs() > tolerance)
            {
                mismatches.push(TensorMismatch {
                    tensor_name: name.to_string(),
                    kind: MismatchKind::DataMismatch {
                        index: i,
                        expected: exp,
                        actual: act,
                        tolerance,
                    },
                });
            }
        }
    }

    /// T-QKV-02: Check for EXTRA tensors in output (detects fusion/split bugs).
    fn check_extra_tensors<S: AsRef<str>>(
        source_tensors: &[(String, Vec<f32>, Vec<usize>)],
        output_names: Vec<S>,
        skip_names: &[&str],
        mismatches: &mut Vec<TensorMismatch>,
    ) {
        let expected_names: std::collections::HashSet<&str> =
            source_tensors.iter().map(|(n, _, _)| n.as_str()).collect();
        for output_name in &output_names {
            let name = output_name.as_ref();
            if skip_names.contains(&name) {
                continue;
            }
            if !expected_names.contains(name) {
                mismatches.push(TensorMismatch {
                    tensor_name: name.to_string(),
                    kind: MismatchKind::Extra,
                });
            }
        }
    }

    /// Get the output APR path (for manual inspection).
    pub(crate) fn output_path(&self) -> Option<&Path> {
        self.output_path.as_deref()
    }

    /// Get the input path.
    pub(crate) fn input_path(&self) -> Option<&Path> {
        self.input_path.as_deref()
    }

    // ----------------------------------------------------------------
    // Convenience one-liners
    // ----------------------------------------------------------------

    /// Import pygmy SafeTensors -> APR with default options and verify.
    pub(crate) fn assert_import_ok(config: PygmyConfig) {
        let h = Self::new()
            .with_safetensors(config)
            .import_to_apr(ImportOptions {
                allow_no_config: true,
                ..ImportOptions::default()
            });
        h.verify_apr().assert_passed();
    }

    /// Full round-trip: SafeTensors -> APR -> SafeTensors, verify data preserved.
    pub(crate) fn assert_roundtrip_ok(config: PygmyConfig) {
        let h = Self::new()
            .with_safetensors(config)
            .import_to_apr(ImportOptions {
                allow_no_config: true,
                ..ImportOptions::default()
            })
            .export_to_safetensors();
        h.verify_safetensors().assert_passed();
    }
}
