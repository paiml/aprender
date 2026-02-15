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

    fn compute_tensor_validation(&self, name: &str, data: &[f32]) -> TensorValidation {
        let element_count = data.len();
        if element_count == 0 {
            return TensorValidation {
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
            };
        }

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut zero_count = 0;

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

        let valid_count = element_count - nan_count - inf_count;
        let mean = if valid_count > 0 {
            (sum / valid_count as f64) as f32
        } else {
            0.0
        };

        // Compute std dev
        let mut var_sum = 0.0f64;
        for &v in data {
            if !v.is_nan() && !v.is_infinite() {
                let diff = f64::from(v) - f64::from(mean);
                var_sum += diff * diff;
            }
        }
        let std = if valid_count > 1 {
            (var_sum / (valid_count - 1) as f64).sqrt() as f32
        } else {
            0.0
        };

        // Collect failures (APR-SPEC 10.9 + PMAT-235 contract gates)
        let mut failures = Vec::new();
        if nan_count > 0 {
            failures.push(format!(
                "[F-DATA-QUALITY-002] {nan_count} NaN values detected"
            ));
        }
        if inf_count > 0 {
            failures.push(format!(
                "[F-DATA-QUALITY-002] {inf_count} Inf values detected"
            ));
        }
        if zero_count == element_count {
            failures.push("[F-DATA-QUALITY-001] All values are zero (uninitialized?)".to_string());
        }

        // Density gate (F-DATA-QUALITY-001)
        // Embedding tensors: >50% zeros indicates corrupt offset loading
        // Weight tensors: >80% zeros indicates uninitialized or zeroed memory
        let zero_pct = if element_count > 0 {
            100.0 * zero_count as f32 / element_count as f32
        } else {
            0.0
        };
        let name_lower = name.to_lowercase();
        let is_embedding = name_lower.contains("embed")
            || name_lower.contains("wte")
            || name_lower.contains("wpe")
            || name_lower.contains("position_embedding");
        // GH-234: lm_head has similar value distribution to embeddings (especially weight-tied)
        let is_lm_head = name_lower.contains("lm_head") || name_lower == "output.weight";
        let density_threshold = if is_embedding || is_lm_head {
            50.0
        } else {
            80.0
        };
        if zero_pct > density_threshold && zero_count < element_count {
            failures.push(format!(
                "[F-DATA-QUALITY-001] DENSITY: {zero_pct:.1}% zeros (max {density_threshold}%)"
            ));
        }

        // PMAT-235: L2 norm gate (F-DATA-QUALITY-003)
        let l2_norm = {
            let mut sum_sq = 0.0f64;
            for &v in data {
                if !v.is_nan() && !v.is_infinite() {
                    sum_sq += f64::from(v) * f64::from(v);
                }
            }
            sum_sq.sqrt() as f32
        };
        if valid_count > 0 && l2_norm < 1e-6 {
            failures
                .push("[F-DATA-QUALITY-003] L2 norm ~0: tensor is effectively empty".to_string());
        }

        // PMAT-235: Variation gate (F-DATA-QUALITY-003)
        // Norm and bias tensors are exempt: constant init (e.g., all 1.0 for RMS norm) is correct
        let is_norm_or_bias = name_lower.contains("norm")
            || name_lower.contains("bias")
            || name_lower.contains("ln_");
        if valid_count > 1 && (max - min).abs() < 1e-10 && !min.is_infinite() && !is_norm_or_bias {
            failures
                .push("[F-DATA-QUALITY-003] All values identical: tensor is constant".to_string());
        }

        let is_valid = failures.is_empty();

        TensorValidation {
            name: name.to_string(),
            is_valid,
            nan_count,
            inf_count,
            zero_count,
            element_count,
            min: if min.is_infinite() { 0.0 } else { min },
            max: if max.is_infinite() { 0.0 } else { max },
            mean,
            std,
            failures,
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
