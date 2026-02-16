impl RosettaStone {

    fn validate_apr(&self, path: &Path) -> Result<ValidationReport> {
        use crate::format::v2::AprV2Reader;

        let data = std::fs::read(path).map_err(|e| AprenderError::FormatError {
            message: format!("Cannot read APR file: {e}"),
        })?;

        let reader = AprV2Reader::from_bytes(&data).map_err(|e| AprenderError::FormatError {
            message: format!("APR parse failed: {e}"),
        })?;

        // GH-187: Log embedding tensor shapes for transposition detection
        let meta = reader.metadata();
        let hidden_size = meta.hidden_size.unwrap_or(0);
        let vocab_size = meta.vocab_size.unwrap_or(0);
        for name in reader.tensor_names() {
            let name_lower = name.to_lowercase();
            let is_embedding = name_lower.contains("embed")
                || name_lower.contains("wte")
                || name_lower.contains("wpe")
                || name_lower.contains("lm_head")
                || name_lower == "output.weight";
            if is_embedding {
                if let Some(entry) = reader.get_tensor(name) {
                    eprintln!(
                        "[GH-187] Embedding '{}': shape={:?}, dtype={:?}",
                        name, entry.shape, entry.dtype
                    );
                    // Detect transposition: if shape is [hidden, vocab] instead of [vocab, hidden]
                    if entry.shape.len() == 2
                        && hidden_size > 0
                        && vocab_size > 0
                        && entry.shape[0] == hidden_size
                        && entry.shape[1] == vocab_size
                    {
                        eprintln!(
                            "[GH-187] WARNING: '{}' may be transposed — shape [{}, {}] \
                             looks like [hidden, vocab] instead of [vocab, hidden]",
                            name, entry.shape[0], entry.shape[1]
                        );
                    }
                }
            }
        }

        let mut tensors = Vec::new();
        let mut total_nan = 0;
        let mut total_inf = 0;
        let mut all_zero_tensors = Vec::new();

        for name in reader.tensor_names() {
            // Use get_tensor_as_f32 which handles dequantization
            if let Some(f32_data) = reader.get_tensor_as_f32(name) {
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
            format: FormatType::Apr,
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

    /// Build an empty (valid) `TensorValidation` for tensors with no elements.
    fn empty_tensor_validation(name: &str) -> TensorValidation {
        TensorValidation {
            name: name.to_string(),
            is_valid: true,
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
            element_count: 0,
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std: 0.0,
            failures: Vec::new(),
        }
    }

    /// Clamp infinite min/max to 0.0 for reporting.
    fn clamp_infinite(v: f32) -> f32 {
        if v.is_infinite() { 0.0 } else { v }
    }

    fn compute_tensor_validation(&self, name: &str, data: &[f32]) -> TensorValidation {
        let element_count = data.len();
        if element_count == 0 {
            return Self::empty_tensor_validation(name);
        }

        let stats = Self::accumulate_tensor_stats(data);
        let valid_count = element_count - stats.nan_count - stats.inf_count;
        let mean = if valid_count > 0 { (stats.sum / valid_count as f64) as f32 } else { 0.0 };
        let std = Self::compute_std_dev(data, mean, valid_count);
        let failures = Self::collect_validation_failures(name, data, &stats, element_count, valid_count);

        TensorValidation {
            name: name.to_string(),
            is_valid: failures.is_empty(),
            nan_count: stats.nan_count,
            inf_count: stats.inf_count,
            zero_count: stats.zero_count,
            element_count,
            min: Self::clamp_infinite(stats.min),
            max: Self::clamp_infinite(stats.max),
            mean,
            std,
            failures,
        }
    }

    /// Accumulate basic statistics (min, max, sum, nan/inf/zero counts) in a single pass.
    fn accumulate_tensor_stats(data: &[f32]) -> TensorAccum {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut nan_count = 0usize;
        let mut inf_count = 0usize;
        let mut zero_count = 0usize;

        for &v in data {
            if v.is_nan() {
                nan_count += 1;
                continue;
            }
            if v.is_infinite() {
                inf_count += 1;
                continue;
            }
            if v == 0.0 {
                zero_count += 1;
            }
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            sum += f64::from(v);
        }

        TensorAccum { min, max, sum, nan_count, inf_count, zero_count }
    }

    /// Compute sample standard deviation from data, given pre-computed mean and valid count.
    fn compute_std_dev(data: &[f32], mean: f32, valid_count: usize) -> f32 {
        if valid_count <= 1 {
            return 0.0;
        }
        let mean_f64 = f64::from(mean);
        let var_sum: f64 = data.iter()
            .filter(|v| !v.is_nan() && !v.is_infinite())
            .map(|&v| {
                let diff = f64::from(v) - mean_f64;
                diff * diff
            })
            .sum();
        (var_sum / (valid_count - 1) as f64).sqrt() as f32
    }

    /// Collect all validation failures (APR-SPEC 10.9 + PMAT-235 contract gates).
    fn collect_validation_failures(
        name: &str,
        data: &[f32],
        stats: &TensorAccum,
        element_count: usize,
        valid_count: usize,
    ) -> Vec<String> {
        let mut failures = Vec::new();

        // NaN / Inf checks
        if stats.nan_count > 0 {
            failures.push(format!(
                "[F-DATA-QUALITY-002] {} NaN values detected", stats.nan_count
            ));
        }
        if stats.inf_count > 0 {
            failures.push(format!(
                "[F-DATA-QUALITY-002] {} Inf values detected", stats.inf_count
            ));
        }
        if stats.zero_count == element_count {
            failures.push("[F-DATA-QUALITY-001] All values are zero (uninitialized?)".to_string());
        }

        // Density gate (F-DATA-QUALITY-001)
        Self::check_density_gate(name, stats.zero_count, element_count, &mut failures);

        // L2 norm gate (F-DATA-QUALITY-003)
        Self::check_l2_norm_gate(data, valid_count, &mut failures);

        // Variation gate (F-DATA-QUALITY-003)
        Self::check_variation_gate(name, stats.min, stats.max, valid_count, &mut failures);

        failures
    }

    /// Density gate: embedding tensors >50% zeros, weight tensors >80% zeros.
    fn check_density_gate(
        name: &str,
        zero_count: usize,
        element_count: usize,
        failures: &mut Vec<String>,
    ) {
        if element_count == 0 || zero_count == element_count {
            return;
        }
        let zero_pct = 100.0 * zero_count as f32 / element_count as f32;
        let density_threshold = Self::density_threshold_for(name);
        if zero_pct > density_threshold {
            failures.push(format!(
                "[F-DATA-QUALITY-001] DENSITY: {zero_pct:.1}% zeros (max {density_threshold}%)"
            ));
        }
    }

    /// Return the density threshold for a tensor based on its name.
    /// Embedding and lm_head tensors use 50%; all others use 80%.
    fn density_threshold_for(name: &str) -> f32 {
        let name_lower = name.to_lowercase();
        let is_embedding = name_lower.contains("embed")
            || name_lower.contains("wte")
            || name_lower.contains("wpe")
            || name_lower.contains("position_embedding");
        // GH-234: lm_head has similar value distribution to embeddings (especially weight-tied)
        let is_lm_head = name_lower.contains("lm_head") || name_lower == "output.weight";
        if is_embedding || is_lm_head { 50.0 } else { 80.0 }
    }

    /// PMAT-235: L2 norm gate — tensor is effectively empty if L2 norm ~0.
    fn check_l2_norm_gate(data: &[f32], valid_count: usize, failures: &mut Vec<String>) {
        if valid_count == 0 {
            return;
        }
        let sum_sq: f64 = data.iter()
            .filter(|v| !v.is_nan() && !v.is_infinite())
            .map(|&v| f64::from(v) * f64::from(v))
            .sum();
        let l2_norm = sum_sq.sqrt() as f32;
        if l2_norm < 1e-6 {
            failures
                .push("[F-DATA-QUALITY-003] L2 norm ~0: tensor is effectively empty".to_string());
        }
    }

    /// PMAT-235: Variation gate — tensor has no variation (all values identical).
    /// Norm and bias tensors are exempt (constant init is correct for e.g. RMS norm).
    fn check_variation_gate(
        name: &str,
        min: f32,
        max: f32,
        valid_count: usize,
        failures: &mut Vec<String>,
    ) {
        if valid_count <= 1 || min.is_infinite() {
            return;
        }
        let name_lower = name.to_lowercase();
        let is_norm_or_bias = name_lower.contains("norm")
            || name_lower.contains("bias")
            || name_lower.contains("ln_");
        if (max - min).abs() < 1e-10 && !is_norm_or_bias {
            failures
                .push("[F-DATA-QUALITY-003] All values identical: tensor is constant".to_string());
        }
    }

    // ------------------------------------------------------------------------
    // Inspection Methods
    // ------------------------------------------------------------------------

    fn inspect_gguf(&self, path: &Path, file_size: usize) -> Result<InspectionReport> {
        use crate::format::gguf::{load_gguf_raw, GgufRawTensor};

        let result = load_gguf_raw(path)?;

        let mut meta_map: BTreeMap<String, String> = BTreeMap::new();
        // Add config info to metadata
        if let Some(ref arch) = result.model_config.architecture {
            meta_map.insert("general.architecture".to_string(), arch.clone());
        }
        if let Some(num_layers) = result.model_config.num_layers {
            meta_map.insert("n_layers".to_string(), num_layers.to_string());
        }
        if let Some(num_heads) = result.model_config.num_heads {
            meta_map.insert("n_heads".to_string(), num_heads.to_string());
        }
        if let Some(hidden_size) = result.model_config.hidden_size {
            meta_map.insert("n_embd".to_string(), hidden_size.to_string());
        }

        let tensors: Vec<TensorInfo> = result
            .tensors
            .iter()
            .map(|(name, t): (&String, &GgufRawTensor)| TensorInfo {
                name: name.clone(),
                dtype: format!("{}", t.dtype),
                shape: t.shape.clone(),
                size_bytes: t.data.len(),
                stats: None,
            })
            .collect();

        let total_params: usize = tensors
            .iter()
            .map(|t| t.shape.iter().product::<usize>())
            .sum();

        let architecture = result.model_config.architecture.clone();

        let quantization = tensors.first().map(|t| t.dtype.clone());

        Ok(InspectionReport {
            format: FormatType::Gguf,
            file_size,
            metadata: meta_map,
            tensors,
            total_params,
            quantization,
            architecture,
        })
    }

    fn inspect_safetensors(&self, path: &Path, file_size: usize) -> Result<InspectionReport> {
        use crate::serialization::safetensors::{MappedSafeTensors, TensorMetadata};

        let mapped = MappedSafeTensors::open(path).map_err(|e| AprenderError::FormatError {
            message: format!("SafeTensors open failed: {e}"),
        })?;
        let tensor_names = mapped.tensor_names();

        let mut tensors = Vec::new();
        let mut total_params: usize = 0;

        for name in tensor_names {
            if let Some(info) = mapped.get_metadata(name) {
                let info: &TensorMetadata = info;
                let shape: Vec<usize> = info.shape.clone();
                let params: usize = shape.iter().product();
                total_params += params;

                let data_len = info.data_offsets[1] - info.data_offsets[0];

                tensors.push(TensorInfo {
                    name: name.to_string(),
                    dtype: info.dtype.clone(),
                    shape,
                    size_bytes: data_len,
                    stats: None,
                });
            }
        }

        // GH-249: Infer architecture from tensor names for SafeTensors
        let architecture = Self::infer_architecture_from_tensors(&tensors);

        Ok(InspectionReport {
            format: FormatType::SafeTensors,
            file_size,
            metadata: BTreeMap::new(),
            tensors,
            total_params,
            quantization: None,
            architecture,
        })
    }

    fn inspect_apr(&self, path: &Path, file_size: usize) -> Result<InspectionReport> {
        use crate::format::v2::AprV2Reader;

        // Read file into bytes
        let data = std::fs::read(path).map_err(|e| AprenderError::FormatError {
            message: format!("Cannot read APR file: {e}"),
        })?;

        let reader = AprV2Reader::from_bytes(&data).map_err(|e| AprenderError::FormatError {
            message: format!("APR parse failed: {e}"),
        })?;

        let meta = reader.metadata();

        let mut metadata: BTreeMap<String, String> = BTreeMap::new();
        metadata.insert("format_version".to_string(), "2".to_string());
        metadata.insert("model_type".to_string(), meta.model_type.clone());
        if let Some(ref name) = meta.name {
            metadata.insert("model_name".to_string(), name.clone());
        }

        // Get tensors from tensor_names + get_tensor
        let tensor_names = reader.tensor_names();
        let mut tensors = Vec::new();
        let mut total_params: usize = 0;

        for name in tensor_names {
            if let Some(entry) = reader.get_tensor(name) {
                let params: usize = entry.shape.iter().product();
                total_params += params;
                tensors.push(TensorInfo {
                    name: entry.name.clone(),
                    dtype: format!("{:?}", entry.dtype),
                    shape: entry.shape.clone(),
                    size_bytes: entry.size as usize,
                    stats: None,
                });
            }
        }

        // GH-249: Infer architecture from tensor names when metadata is empty
        let architecture = meta
            .architecture
            .clone()
            .filter(|a| !a.is_empty())
            .or_else(|| Self::infer_architecture_from_tensors(&tensors));

        Ok(InspectionReport {
            format: FormatType::Apr,
            file_size,
            metadata,
            tensors,
            total_params,
            quantization: meta.quantization.as_ref().map(|q| q.quant_type.clone()),
            architecture,
        })
    }
}
