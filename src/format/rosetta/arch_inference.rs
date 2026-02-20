impl RosettaStone {
    /// Create a new Rosetta Stone converter with default options
    #[must_use]
    pub fn new() -> Self {
        Self {
            options: ConversionOptions::default(),
        }
    }

    /// Create with custom options
    #[must_use]
    pub fn with_options(options: ConversionOptions) -> Self {
        Self { options }
    }

    /// GH-249: Infer architecture from tensor naming patterns when metadata is absent.
    fn infer_architecture_from_tensors(tensors: &[TensorInfo]) -> Option<String> {
        let names: Vec<&str> = tensors.iter().map(|t| t.name.as_str()).collect();
        let has = |pat: &str| names.iter().any(|n| n.contains(pat));

        // GPT-2: uses c_attn, c_proj, c_fc (Conv1D-style naming)
        if has("c_attn") || has("attn.c_proj") {
            return Some("gpt2".to_string());
        }
        // Qwen2: uses q_proj.bias (LLaMA doesn't have bias on Q/K/V)
        if has("q_proj") && has("q_proj.bias") {
            return Some("qwen2".to_string());
        }
        // LLaMA/SmolLM: uses q_proj without bias, gate_proj
        if has("q_proj") && has("gate_proj") {
            return Some("llama".to_string());
        }
        // Generic transformer fallback
        if has("self_attn") || has("attention") {
            return Some("transformer".to_string());
        }
        None
    }

    /// Inspect a model file (Genchi Genbutsu - go and see)
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or format is unknown
    pub fn inspect<P: AsRef<Path>>(&self, path: P) -> Result<InspectionReport> {
        let path = path.as_ref();

        // Sharded SafeTensors index detection (GH-212, resolved)
        if is_sharded_index(path) {
            return self.inspect_sharded_safetensors(path);
        }

        // Detect format from magic bytes first, fall back to extension
        let format = FormatType::from_magic(path).or_else(|_| FormatType::from_extension(path))?;

        let file_size = std::fs::metadata(path)
            .map(|m| m.len() as usize)
            .unwrap_or(0);

        match format {
            FormatType::Gguf => self.inspect_gguf(path, file_size),
            FormatType::SafeTensors => self.inspect_safetensors(path, file_size),
            FormatType::Apr => self.inspect_apr(path, file_size),
        }
    }

    /// Validate a model file for physics constraints (GH-175, PMAT-180)
    ///
    /// Checks per APR-SPEC 10.9:
    /// - NaN detection (corruption indicator)
    /// - Inf detection (overflow indicator)
    /// - All-zeros detection (uninitialized weights)
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or format is unknown
    pub fn validate<P: AsRef<Path>>(&self, path: P) -> Result<ValidationReport> {
        let path = path.as_ref();
        let start = std::time::Instant::now();

        // Detect format
        let format = FormatType::from_magic(path).or_else(|_| FormatType::from_extension(path))?;

        // Dispatch to format-specific validation
        let mut report = match format {
            FormatType::Gguf => self.validate_gguf(path)?,
            FormatType::SafeTensors => self.validate_safetensors(path)?,
            FormatType::Apr => self.validate_apr(path)?,
        };

        report.duration_ms = start.elapsed().as_millis() as u64;
        Ok(report)
    }

    /// Convert a model file to a different format
    ///
    /// # Errors
    ///
    /// Returns error if conversion fails
    pub fn convert<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        source: P,
        target: Q,
        options: Option<ConversionOptions>,
    ) -> Result<ConversionReport> {
        let source = source.as_ref();
        let target = target.as_ref();
        let opts = options.unwrap_or_else(|| self.options.clone());

        let start = std::time::Instant::now();

        // Sharded SafeTensors index detection (GH-212, resolved)
        if is_sharded_index(source) {
            let target_format = FormatType::from_extension(target)?;
            let source_inspection = self.inspect_sharded_safetensors(source)?;

            self.convert_sharded(source, target, target_format, &opts)?;

            let target_inspection = self.inspect(target)?;
            let duration_ms = start.elapsed().as_millis() as u64;

            return Ok(ConversionReport {
                path: ConversionPath::direct(FormatType::SafeTensors, target_format),
                source_inspection,
                target_inspection,
                warnings: Vec::new(),
                duration_ms,
                modified_tensors: Vec::new(),
                dropped_tensors: Vec::new(),
            });
        }

        // Detect formats
        let source_format =
            FormatType::from_magic(source).or_else(|_| FormatType::from_extension(source))?;
        let target_format = FormatType::from_extension(target)?;

        // Inspect source
        let source_inspection = self.inspect(source)?;

        // Perform conversion
        self.convert_internal(source, target, source_format, target_format, &opts)?;

        // Inspect target
        let target_inspection = self.inspect(target)?;

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(ConversionReport {
            path: ConversionPath::direct(source_format, target_format),
            source_inspection,
            target_inspection,
            warnings: Vec::new(),
            duration_ms,
            modified_tensors: Vec::new(),
            dropped_tensors: Vec::new(),
        })
    }

    /// Execute a multi-step conversion chain
    ///
    /// # Errors
    ///
    /// Returns error if any step fails or cycle detected
    pub fn chain<P: AsRef<Path>>(
        &self,
        source: P,
        chain: &[FormatType],
        work_dir: &Path,
    ) -> Result<Vec<ConversionReport>> {
        if chain.len() < 2 {
            return Err(AprenderError::FormatError {
                message: "Chain must have at least 2 formats".to_string(),
            });
        }

        // Check for cycles (Popperian: The Infinite Loop test)
        let path = ConversionPath::chain(
            chain[0],
            chain[1..chain.len() - 1].to_vec(),
            chain[chain.len() - 1],
        );
        if path.has_cycle() {
            return Err(AprenderError::FormatError {
                message: "Conversion chain contains a cycle".to_string(),
            });
        }

        let source = source.as_ref();
        let mut reports = Vec::new();
        let mut current_path = source.to_path_buf();

        for (i, window) in chain.windows(2).enumerate() {
            let target_format = window[1];
            let target_path = work_dir.join(format!("step_{i}.{}", target_format.extension()));

            let report = self.convert(&current_path, &target_path, None)?;
            reports.push(report);

            current_path = target_path;
        }

        Ok(reports)
    }

    /// Verify round-trip conversion preserves equivalence
    ///
    /// # Errors
    ///
    /// Returns error if verification fails
    pub fn verify_roundtrip<P: AsRef<Path>>(
        &self,
        source: P,
        intermediate: FormatType,
    ) -> Result<VerificationReport> {
        let source = source.as_ref();
        let source_format =
            FormatType::from_magic(source).or_else(|_| FormatType::from_extension(source))?;

        // Create temp directory for intermediate files
        let temp_dir = std::env::temp_dir().join("rosetta_verify");
        std::fs::create_dir_all(&temp_dir).map_err(|e| AprenderError::FormatError {
            message: format!("Cannot create temp dir: {e}"),
        })?;

        // Source → Intermediate
        let intermediate_path = temp_dir.join(format!("intermediate.{}", intermediate.extension()));
        self.convert(source, &intermediate_path, None)?;

        // Intermediate → Source format (round-trip)
        let roundtrip_path = temp_dir.join(format!("roundtrip.{}", source_format.extension()));
        self.convert(&intermediate_path, &roundtrip_path, None)?;

        // Compare source and round-trip
        self.compare_files(source, &roundtrip_path)
    }

    /// Load a tensor as f32 values from any supported format
    ///
    /// Handles dequantization for quantized formats (Q4_K, Q6_K, etc.)
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read, format is unknown, or tensor not found
    pub fn load_tensor_f32<P: AsRef<Path>>(&self, path: P, tensor_name: &str) -> Result<Vec<f32>> {
        let path = path.as_ref();
        let format = FormatType::from_magic(path).or_else(|_| FormatType::from_extension(path))?;

        match format {
            FormatType::Gguf => self.load_tensor_f32_gguf(path, tensor_name),
            FormatType::SafeTensors => self.load_tensor_f32_safetensors(path, tensor_name),
            FormatType::Apr => self.load_tensor_f32_apr(path, tensor_name),
        }
    }

    fn load_tensor_f32_gguf(&self, path: &Path, tensor_name: &str) -> Result<Vec<f32>> {
        use crate::format::gguf::GgufReader;

        let reader = GgufReader::from_file(path)?;
        let (data, _shape) =
            reader
                .get_tensor_f32(tensor_name)
                .map_err(|e| AprenderError::FormatError {
                    message: format!("Failed to load GGUF tensor '{}': {}", tensor_name, e),
                })?;
        Ok(data)
    }

    fn load_tensor_f32_safetensors(&self, path: &Path, tensor_name: &str) -> Result<Vec<f32>> {
        use crate::serialization::safetensors::MappedSafeTensors;

        let mapped = MappedSafeTensors::open(path).map_err(|e| AprenderError::FormatError {
            message: format!("SafeTensors open failed: {e}"),
        })?;
        mapped
            .get_tensor(tensor_name)
            .map_err(|e| AprenderError::FormatError {
                message: format!("Failed to load SafeTensors tensor '{}': {}", tensor_name, e),
            })
    }

    fn load_tensor_f32_apr(&self, path: &Path, tensor_name: &str) -> Result<Vec<f32>> {
        use crate::format::v2::AprV2Reader;

        let data = std::fs::read(path).map_err(|e| AprenderError::FormatError {
            message: format!("Cannot read APR file: {e}"),
        })?;
        let reader = AprV2Reader::from_bytes(&data).map_err(|e| AprenderError::FormatError {
            message: format!("APR parse failed: {e}"),
        })?;
        reader
            .get_tensor_as_f32(tensor_name)
            .ok_or_else(|| AprenderError::FormatError {
                message: format!("Tensor '{}' not found in APR file", tensor_name),
            })
    }

    // ========================================================================
    // Private Methods
    // ========================================================================

    // ------------------------------------------------------------------------
    // Validation Methods (GH-175, PMAT-180)
    // ------------------------------------------------------------------------

    fn validate_gguf(&self, path: &Path) -> Result<ValidationReport> {
        use crate::format::gguf::GgufReader;

        let reader = GgufReader::from_file(path)?;
        let mut tensors = Vec::new();
        let mut total_nan = 0;
        let mut total_inf = 0;
        let mut all_zero_tensors = Vec::new();

        // Get tensor names from metadata
        let tensor_names: Vec<String> = reader.tensors.iter().map(|t| t.name.clone()).collect();

        for name in &tensor_names {
            // Use GgufReader's dequantization (handles Q4K, Q6K, etc.)
            if let Ok((f32_data, _shape)) = reader.get_tensor_f32(name) {
                let tv = self.compute_tensor_validation(name, &f32_data);

                total_nan += tv.nan_count;
                total_inf += tv.inf_count;
                if tv.is_all_zeros() {
                    all_zero_tensors.push(name.clone());
                }
                tensors.push(tv);
            }
        }

        let failed_count = tensors.iter().filter(|t| !t.is_valid).count();
        let is_valid = failed_count == 0;

        Ok(ValidationReport {
            format: FormatType::Gguf,
            file_path: path.display().to_string(),
            is_valid,
            tensor_count: tensors.len(),
            failed_tensor_count: failed_count,
            total_nan_count: total_nan,
            total_inf_count: total_inf,
            all_zero_tensors,
            tensors,
            duration_ms: 0, // Set by caller
        })
    }

    fn validate_safetensors(&self, path: &Path) -> Result<ValidationReport> {
        use crate::serialization::safetensors::MappedSafeTensors;

        let mapped = MappedSafeTensors::open(path).map_err(|e| AprenderError::FormatError {
            message: format!("SafeTensors open failed: {e}"),
        })?;

        let mut tensors = Vec::new();
        let mut total_nan = 0;
        let mut total_inf = 0;
        let mut all_zero_tensors = Vec::new();

        for name in mapped.tensor_names() {
            // get_tensor returns Result<Vec<f32>, String>
            if let Ok(f32_data) = mapped.get_tensor(name) {
                let tv = self.compute_tensor_validation(name, &f32_data);

                total_nan += tv.nan_count;
                total_inf += tv.inf_count;
                if tv.is_all_zeros() {
                    all_zero_tensors.push(name.to_string());
                }
                tensors.push(tv);
            }
        }

        let failed_count = tensors.iter().filter(|t| !t.is_valid).count();
        let is_valid = failed_count == 0;

        Ok(ValidationReport {
            format: FormatType::SafeTensors,
            file_path: path.display().to_string(),
            is_valid,
            tensor_count: tensors.len(),
            failed_tensor_count: failed_count,
            total_nan_count: total_nan,
            total_inf_count: total_inf,
            all_zero_tensors,
            tensors,
            duration_ms: 0,
        })
    }
}
