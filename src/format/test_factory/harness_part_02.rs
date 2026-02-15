
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
        assert!(
            !self.source_tensors.is_empty(),
            "Cannot verify with 0 source tensors -- use a non-empty PygmyConfig"
        );
        let output = self
            .output_path
            .as_ref()
            .expect("No output path set -- run import first");
        let data = fs::read(output).expect("Failed to read output APR");
        let reader = AprV2Reader::from_bytes(&data).expect("Failed to parse output APR");

        let mut mismatches = Vec::new();
        let tolerance = self.tolerance.f32_atol;

        for (name, expected_data, expected_shape) in &self.source_tensors {
            // Check tensor exists
            let entry = match reader.get_tensor(name) {
                Some(e) => e,
                None => {
                    mismatches.push(TensorMismatch {
                        tensor_name: name.clone(),
                        kind: MismatchKind::Missing,
                    });
                    continue;
                }
            };

            // Check shape
            if &entry.shape != expected_shape {
                mismatches.push(TensorMismatch {
                    tensor_name: name.clone(),
                    kind: MismatchKind::ShapeMismatch {
                        expected: expected_shape.clone(),
                        actual: entry.shape.clone(),
                    },
                });
                continue;
            }

            // Check data values
            if let Some(actual_data) = reader.get_tensor_as_f32(name) {
                for (i, (&exp, &act)) in expected_data.iter().zip(actual_data.iter()).enumerate() {
                    if (exp - act).abs() > tolerance {
                        mismatches.push(TensorMismatch {
                            tensor_name: name.clone(),
                            kind: MismatchKind::DataMismatch {
                                index: i,
                                expected: exp,
                                actual: act,
                                tolerance,
                            },
                        });
                        break; // One mismatch per tensor is enough
                    }
                }
            }
        }

        // T-QKV-02: Check for EXTRA tensors in output (detects fusion/split bugs)
        let expected_names: std::collections::HashSet<&str> = self
            .source_tensors
            .iter()
            .map(|(n, _, _)| n.as_str())
            .collect();
        for output_name in reader.tensor_names() {
            if !expected_names.contains(output_name) {
                mismatches.push(TensorMismatch {
                    tensor_name: output_name.to_string(),
                    kind: MismatchKind::Extra,
                });
            }
        }

        VerificationResult { mismatches }
    }

    /// Read back the output SafeTensors from disk and verify tensor data matches source.
    ///
    /// Checks: tensor existence, shape equality, and data values within tolerance.
    /// Panics if no source tensors were recorded (empty config guard).
    pub(crate) fn verify_safetensors(&self) -> VerificationResult {
        assert!(
            !self.source_tensors.is_empty(),
            "Cannot verify with 0 source tensors -- use a non-empty PygmyConfig"
        );
        let output = self
            .output_path
            .as_ref()
            .expect("No output path set -- run export first");
        let mapped = MappedSafeTensors::open(output).expect("Failed to open output SafeTensors");

        let mut mismatches = Vec::new();
        let tolerance = self.tolerance.f32_atol;

        for (name, expected_data, expected_shape) in &self.source_tensors {
            let meta = match mapped.get_metadata(name) {
                Some(m) => m,
                None => {
                    mismatches.push(TensorMismatch {
                        tensor_name: name.clone(),
                        kind: MismatchKind::Missing,
                    });
                    continue;
                }
            };

            if &meta.shape != expected_shape {
                mismatches.push(TensorMismatch {
                    tensor_name: name.clone(),
                    kind: MismatchKind::ShapeMismatch {
                        expected: expected_shape.clone(),
                        actual: meta.shape.clone(),
                    },
                });
                continue;
            }

            if let Ok(actual_data) = mapped.get_tensor(name) {
                for (i, (&exp, &act)) in expected_data.iter().zip(actual_data.iter()).enumerate() {
                    if (exp - act).abs() > tolerance {
                        mismatches.push(TensorMismatch {
                            tensor_name: name.clone(),
                            kind: MismatchKind::DataMismatch {
                                index: i,
                                expected: exp,
                                actual: act,
                                tolerance,
                            },
                        });
                        break;
                    }
                }
            }
        }

        // T-QKV-02: Check for EXTRA tensors in output (detects fusion/split bugs)
        let expected_names: std::collections::HashSet<&str> = self
            .source_tensors
            .iter()
            .map(|(n, _, _)| n.as_str())
            .collect();
        for output_name in mapped.tensor_names() {
            // SafeTensors __metadata__ key is not a tensor, skip it
            if output_name == "__metadata__" {
                continue;
            }
            if !expected_names.contains(output_name) {
                mismatches.push(TensorMismatch {
                    tensor_name: output_name.to_string(),
                    kind: MismatchKind::Extra,
                });
            }
        }

        VerificationResult { mismatches }
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
