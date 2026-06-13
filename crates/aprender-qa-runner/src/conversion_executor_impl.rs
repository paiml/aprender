
impl ConversionExecutor {
    /// Create a new conversion executor
    #[must_use]
    pub fn new(config: ConversionConfig) -> Self {
        Self {
            config,
            binary: default_binary(),
            output_dir: None,
        }
    }

    /// Create with default config
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(ConversionConfig::default())
    }

    /// Set the output directory for conversion artifacts (ISO-OUT-001)
    #[must_use]
    pub fn with_output_dir(mut self, output_dir: PathBuf) -> Self {
        self.output_dir = Some(output_dir);
        self
    }

    /// Execute all conversion tests for a model
    ///
    /// # Errors
    ///
    /// Returns an error if a critical conversion failure occurs.
    pub fn execute_all(
        &self,
        model_path: &Path,
        model_id: &ModelId,
    ) -> Result<ConversionExecutionResult> {
        let mut results = Vec::new();
        let mut evidence = Vec::new();
        let start = std::time::Instant::now();

        let backends: Vec<Backend> = if self.config.no_gpu {
            vec![Backend::Cpu]
        } else {
            self.config.backends.clone()
        };

        let output_dir_wrapper = self
            .output_dir
            .as_ref()
            .map(|dir| ConversionOutputDir::new(dir, model_id));

        if self.config.test_all_pairs {
            self.run_all_pairs(
                model_path,
                model_id,
                &backends,
                output_dir_wrapper.as_ref(),
                &mut results,
                &mut evidence,
            );
        }

        if self.config.test_round_trips {
            self.run_round_trips(model_path, model_id, &backends, &mut results, &mut evidence);
        }

        if self.config.test_multi_hop {
            self.run_multi_hop_chains(model_path, model_id, &backends, &mut results, &mut evidence);
            self.run_byte_level_rt(model_path, model_id, &backends, &mut results, &mut evidence);
        }

        if self.config.test_idempotency {
            self.run_idempotency(model_path, model_id, &backends, &mut results, &mut evidence);
        }

        if self.config.test_commutativity {
            self.run_commutativity(model_path, model_id, &backends, &mut results, &mut evidence);
        }

        if self.config.test_cardinality || self.config.test_tensor_names {
            self.run_structural_checks(model_path, model_id, &mut results, &mut evidence);
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        let passed = results
            .iter()
            .filter(|r| matches!(r, ConversionResult::Corroborated { .. }))
            .count();
        let failed = results.len() - passed;

        Ok(ConversionExecutionResult {
            total: results.len(),
            passed,
            failed,
            duration_ms,
            results,
            evidence,
        })
    }

    /// Test all format conversion pairs
    fn run_all_pairs(
        &self,
        model_path: &Path,
        model_id: &ModelId,
        backends: &[Backend],
        output_dir_wrapper: Option<&ConversionOutputDir>,
        results: &mut Vec<ConversionResult>,
        evidence: &mut Vec<Evidence>,
    ) {
        for (source, target) in all_conversion_pairs() {
            for backend in backends {
                let mut test = ConversionTest::new(source, target, *backend, model_id.clone());
                test.binary.clone_from(&self.binary);
                if let Some(out_dir) = &output_dir_wrapper {
                    test.output_dir = Some((*out_dir).clone());
                }

                match test.execute(model_path) {
                    Ok(result) => {
                        let ev: Evidence = result.clone().into();
                        evidence.push(ev);
                        results.push(result);
                    }
                    Err(e) => {
                        let ev = Evidence::falsified(
                            &test.gate_id(),
                            QaScenario::new(
                                model_id.clone(),
                                Modality::Run,
                                *backend,
                                target,
                                format!("Convert {source:?} to {target:?}"),
                                0,
                            ),
                            format!("Conversion infrastructure error: {e}"),
                            "N/A",
                            0,
                        );
                        evidence.push(ev);
                        results.push(ConversionResult::Falsified {
                            gate_id: test.gate_id(),
                            reason: e.to_string(),
                            evidence: ConversionEvidence {
                                source_hash: String::new(),
                                converted_hash: String::new(),
                                max_diff: f64::MAX,
                                diff_indices: vec![],
                                source_format: source,
                                target_format: target,
                                backend: *backend,
                                failure_type: None,
                                quant_type: None,
                            },
                        });
                    }
                }
            }
        }
    }

    /// Test round-trips (GGUF → APR → SafeTensors → GGUF) - F-CONV-RT-001
    fn run_round_trips(
        &self,
        model_path: &Path,
        model_id: &ModelId,
        backends: &[Backend],
        results: &mut Vec<ConversionResult>,
        evidence: &mut Vec<Evidence>,
    ) {
        for backend in backends {
            let mut rt = RoundTripTest::new(
                vec![Format::Gguf, Format::Apr, Format::SafeTensors, Format::Gguf],
                *backend,
                model_id.clone(),
            );
            rt.binary.clone_from(&self.binary);

            match rt.execute(model_path) {
                Ok(result) => {
                    let ev: Evidence = result.clone().into();
                    evidence.push(ev);
                    results.push(result);
                }
                Err(e) => {
                    let ev = Evidence::falsified(
                        "F-CONV-RT-001",
                        QaScenario::new(
                            model_id.clone(),
                            Modality::Run,
                            *backend,
                            Format::Gguf,
                            "Round-trip conversion".to_string(),
                            0,
                        ),
                        format!("Round-trip failed: {e}"),
                        "N/A",
                        0,
                    );
                    evidence.push(ev);
                    results.push(ConversionResult::Falsified {
                        gate_id: "F-CONV-RT-001".to_string(),
                        reason: e.to_string(),
                        evidence: ConversionEvidence {
                            source_hash: String::new(),
                            converted_hash: String::new(),
                            max_diff: f64::MAX,
                            diff_indices: vec![],
                            source_format: Format::Gguf,
                            target_format: Format::Gguf,
                            backend: *backend,
                            failure_type: None,
                            quant_type: None,
                        },
                    });
                }
            }
        }
    }

    /// Multi-hop chain tests (F-CONV-RT-002, RT-003, RT-004)
    fn run_multi_hop_chains(
        &self,
        model_path: &Path,
        model_id: &ModelId,
        backends: &[Backend],
        results: &mut Vec<ConversionResult>,
        evidence: &mut Vec<Evidence>,
    ) {
        let multi_hop_chains: Vec<(&str, Vec<Format>)> = vec![
            (
                "F-CONV-RT-002",
                vec![
                    Format::SafeTensors,
                    Format::Apr,
                    Format::Gguf,
                    Format::SafeTensors,
                ],
            ),
            (
                "F-CONV-RT-003",
                vec![
                    Format::SafeTensors,
                    Format::Apr,
                    Format::Gguf,
                    Format::Apr,
                    Format::SafeTensors,
                ],
            ),
            (
                "F-CONV-RT-004",
                vec![Format::SafeTensors, Format::Apr, Format::Gguf, Format::Apr],
            ),
        ];

        for (gate_id, chain) in &multi_hop_chains {
            for backend in backends {
                let mut rt = RoundTripTest::new(chain.clone(), *backend, model_id.clone());
                rt.binary.clone_from(&self.binary);

                match rt.execute(model_path) {
                    Ok(mut result) => {
                        if let ConversionResult::Falsified {
                            gate_id: ref mut gid,
                            ..
                        } = result
                        {
                            *gid = (*gate_id).to_string();
                        }
                        let ev: Evidence = result.clone().into();
                        evidence.push(ev);
                        results.push(result);
                    }
                    Err(e) => {
                        let chain_desc: Vec<_> = chain.iter().map(|f| format!("{f:?}")).collect();
                        let ev = Evidence::falsified(
                            *gate_id,
                            QaScenario::new(
                                model_id.clone(),
                                Modality::Run,
                                *backend,
                                Format::SafeTensors,
                                format!("Multi-hop: {}", chain_desc.join("→")),
                                0,
                            ),
                            format!("Multi-hop chain failed: {e}"),
                            "N/A",
                            0,
                        );
                        evidence.push(ev);
                        results.push(ConversionResult::Falsified {
                            gate_id: (*gate_id).to_string(),
                            reason: e.to_string(),
                            evidence: ConversionEvidence {
                                source_hash: String::new(),
                                converted_hash: String::new(),
                                max_diff: f64::MAX,
                                diff_indices: vec![],
                                source_format: Format::SafeTensors,
                                target_format: Format::SafeTensors,
                                backend: *backend,
                                failure_type: None,
                                quant_type: None,
                            },
                        });
                    }
                }
            }
        }
    }

    /// Byte-level round-trip test (F-CONV-RT-BYTE-001)
    fn run_byte_level_rt(
        &self,
        model_path: &Path,
        model_id: &ModelId,
        backends: &[Backend],
        results: &mut Vec<ConversionResult>,
        evidence: &mut Vec<Evidence>,
    ) {
        for backend in backends {
            let mut byte_rt = ByteLevelRoundTripTest::new(*backend, model_id.clone());
            byte_rt.binary.clone_from(&self.binary);

            match byte_rt.execute(model_path) {
                Ok(result) => {
                    let ev: Evidence = result.clone().into();
                    evidence.push(ev);
                    results.push(result);
                }
                Err(e) => {
                    let ev = Evidence::falsified(
                        "F-CONV-RT-BYTE-001",
                        QaScenario::new(
                            model_id.clone(),
                            Modality::Run,
                            *backend,
                            Format::SafeTensors,
                            "Byte-level round-trip ST→APR→GGUF→APR".to_string(),
                            0,
                        ),
                        format!("Byte-level round-trip failed: {e}"),
                        "N/A",
                        0,
                    );
                    evidence.push(ev);
                    results.push(ConversionResult::Falsified {
                        gate_id: "F-CONV-RT-BYTE-001".to_string(),
                        reason: e.to_string(),
                        evidence: ConversionEvidence {
                            source_hash: String::new(),
                            converted_hash: String::new(),
                            max_diff: f64::MAX,
                            diff_indices: vec![],
                            source_format: Format::SafeTensors,
                            target_format: Format::Apr,
                            backend: *backend,
                            failure_type: None,
                            quant_type: None,
                        },
                    });
                }
            }
        }
    }

    /// Idempotency test (F-CONV-IDEM-001)
    fn run_idempotency(
        &self,
        model_path: &Path,
        model_id: &ModelId,
        backends: &[Backend],
        results: &mut Vec<ConversionResult>,
        evidence: &mut Vec<Evidence>,
    ) {
        for backend in backends {
            let mut idem =
                IdempotencyTest::new(Format::Gguf, Format::Apr, *backend, model_id.clone());
            idem.binary.clone_from(&self.binary);

            match idem.execute(model_path) {
                Ok(result) => {
                    let ev: Evidence = result.clone().into();
                    evidence.push(ev);
                    results.push(result);
                }
                Err(e) => {
                    let ev = Evidence::falsified(
                        "F-CONV-IDEM-001",
                        QaScenario::new(
                            model_id.clone(),
                            Modality::Run,
                            *backend,
                            Format::Apr,
                            "Idempotency: GGUF→APR twice".to_string(),
                            0,
                        ),
                        format!("Idempotency test failed: {e}"),
                        "N/A",
                        0,
                    );
                    evidence.push(ev);
                    results.push(ConversionResult::Falsified {
                        gate_id: "F-CONV-IDEM-001".to_string(),
                        reason: e.to_string(),
                        evidence: ConversionEvidence {
                            source_hash: String::new(),
                            converted_hash: String::new(),
                            max_diff: f64::MAX,
                            diff_indices: vec![],
                            source_format: Format::Gguf,
                            target_format: Format::Apr,
                            backend: *backend,
                            failure_type: None,
                            quant_type: None,
                        },
                    });
                }
            }
        }
    }

    /// Commutativity test (F-CONV-COM-001)
    fn run_commutativity(
        &self,
        model_path: &Path,
        model_id: &ModelId,
        backends: &[Backend],
        results: &mut Vec<ConversionResult>,
        evidence: &mut Vec<Evidence>,
    ) {
        for backend in backends {
            let mut com = CommutativityTest::new(*backend, model_id.clone());
            com.binary.clone_from(&self.binary);

            match com.execute(model_path) {
                Ok(result) => {
                    let ev: Evidence = result.clone().into();
                    evidence.push(ev);
                    results.push(result);
                }
                Err(e) => {
                    let ev = Evidence::falsified(
                        "F-CONV-COM-001",
                        QaScenario::new(
                            model_id.clone(),
                            Modality::Run,
                            *backend,
                            Format::Apr,
                            "Commutativity: GGUF→APR vs GGUF→ST→APR".to_string(),
                            0,
                        ),
                        format!("Commutativity test failed: {e}"),
                        "N/A",
                        0,
                    );
                    evidence.push(ev);
                    results.push(ConversionResult::Falsified {
                        gate_id: "F-CONV-COM-001".to_string(),
                        reason: e.to_string(),
                        evidence: ConversionEvidence {
                            source_hash: String::new(),
                            converted_hash: String::new(),
                            max_diff: f64::MAX,
                            diff_indices: vec![],
                            source_format: Format::Gguf,
                            target_format: Format::Apr,
                            backend: *backend,
                            failure_type: None,
                            quant_type: None,
                        },
                    });
                }
            }
        }
    }

    /// Structural checks: cardinality (F-CONV-CARD-001) and tensor names (F-CONV-NAME-001)
    fn run_structural_checks(
        &self,
        model_path: &Path,
        model_id: &ModelId,
        results: &mut Vec<ConversionResult>,
        evidence: &mut Vec<Evidence>,
    ) {
        for (source, target) in all_conversion_pairs() {
            let target_ext = format_extension(target);
            // PMAT-743: must match the writer's idempotent path scheme.
            let converted_path = converted_output_path(model_path, target_ext);
            if !converted_path.exists() {
                continue;
            }

            if self.config.test_cardinality {
                self.check_cardinality_gate(
                    model_path,
                    &converted_path,
                    model_id,
                    source,
                    target,
                    results,
                    evidence,
                );
            }

            if self.config.test_tensor_names {
                self.check_tensor_name_gate(
                    model_path,
                    &converted_path,
                    model_id,
                    source,
                    target,
                    evidence,
                );
            }
        }
    }
}

include!("check.rs");
