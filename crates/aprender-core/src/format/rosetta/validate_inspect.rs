impl RosettaStone {

    fn validate_apr(&self, path: &Path) -> Result<ValidationReport> {
        use crate::format::v2::AprV2Reader;
        use crate::format::AprV2DequantExt; // issue #2231 re-attached accessor

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
                // PMAT-889: pass the on-disk 2-D shape so the dead-output-row gate
                // (F-DATA-QUALITY-007) can scan per-output-row L2 for output-projection
                // tensors. Falls back to the shape-free gates for non-2-D tensors.
                let shape: Vec<usize> = reader
                    .get_tensor(name)
                    .map(|e| e.shape.clone())
                    .unwrap_or_default();
                let tv = self.compute_tensor_validation_with_shape(name, &f32_data, &shape);

                total_nan += tv.nan_count;
                total_inf += tv.inf_count;
                if tv.is_all_zeros() {
                    all_zero_tensors.push(name.to_string());
                }
                tensors.push(tv);
            } else {
                // A tensor the reader cannot decode is a validation FAILURE, not a
                // tensor to omit from the count. See `unreadable_tensor_validation`.
                tensors.push(Self::unreadable_tensor_validation(
                    name,
                    "APR reader returned no data (shape/offset may exceed the file)",
                ));
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

    /// Build a FAILING `TensorValidation` for a tensor the reader could not decode.
    ///
    /// A tensor that cannot be read is the strongest possible signal that a file is
    /// corrupt, and it used to be the one signal `validate` threw away: all three
    /// `validate_*` paths wrote `if let Ok(data) = reader.get_tensor(..)` and silently
    /// dropped the tensor on the else branch, then reported
    /// `tensor_count: tensors.len()` — the number that survived. On a GGUF whose
    /// `output_norm.weight` extent overruns EOF that produced
    /// `VALID: 338 tensors checked, 0 contract violations` with exit 0, while
    /// `apr tensors` counted 339 on the same file and `inspect`, `debug`, `tree` and
    /// `diff` all rejected it outright. The gate was arithmetically incapable of
    /// failing on that corruption class, because the evidence was removed from the
    /// denominator before the comparison.
    ///
    /// Recording the failure as a tensor keeps `tensor_count` equal to the number the
    /// file declares, so a dropped tensor can no longer be invisible.
    pub(crate) fn unreadable_tensor_validation(name: &str, reason: &str) -> TensorValidation {
        TensorValidation {
            name: name.to_string(),
            is_valid: false,
            nan_count: 0,
            inf_count: 0,
            zero_count: 0,
            element_count: 0,
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std: 0.0,
            failures: vec![format!("tensor data could not be read: {reason}")],
        }
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

    /// Shape-aware tensor validation (PMAT-889).
    ///
    /// Runs the whole-tensor data-quality gates (`compute_tensor_validation`) and,
    /// when the tensor is a 2-D output projection (lm_head / embed / output), ALSO
    /// runs the per-output-row dead-row gate (F-DATA-QUALITY-007). The flat row-major
    /// `data` is interpreted as `[shape[0], shape[1]]` = `[out_units, in_dim]`.
    fn compute_tensor_validation_with_shape(
        &self,
        name: &str,
        data: &[f32],
        shape: &[usize],
    ) -> TensorValidation {
        let mut tv = self.compute_tensor_validation(name, data);
        Self::check_dead_output_row(name, data, shape, &mut tv.failures);
        tv.is_valid = tv.failures.is_empty();
        tv
    }

    /// F-DATA-QUALITY-007 (PMAT-889): reject a 2-D output-projection weight that
    /// has at least one fully-zero (L2 ~ 0) OUTPUT ROW.
    ///
    /// For an `lm_head`/`output` `[vocab, hidden]` a zero row makes that token's
    /// logit structurally a constant (its dot-product against any hidden state is
    /// 0) — a dead token that can never be emitted with positive evidence. For an
    /// `embed_tokens` `[vocab, hidden]` a zero row is a dead/unreachable token
    /// vector. The whole-tensor density / L2 / constant gates MISS this because a
    /// single zero row is a tiny fraction of a large tensor. llama.cpp / Ollama
    /// load and run such a model silently; apr fails closed.
    ///
    /// Scope (false-positive safety): the gate fires ONLY when (a) the shape is
    /// 2-D and (b) the tensor's ROLE is an output projection whose zero row is
    /// unambiguously corrupt — `lm_head` / `embed` / `output`. Other roles (q/k/v,
    /// gate/up/down, norms, biases, generic buffers) are intentionally excluded to
    /// avoid false positives on tensors where a structurally-zero row may be valid.
    fn check_dead_output_row(
        name: &str,
        data: &[f32],
        shape: &[usize],
        failures: &mut Vec<String>,
    ) {
        // Gate only on 2-D output-projection roles.
        if shape.len() != 2 || !Self::is_output_projection_role(name) {
            return;
        }
        let (out_units, in_dim) = (shape[0], shape[1]);
        // Guard against shape/data inconsistency (a separate concern handled by
        // shape gates elsewhere) — only scan when the flat length matches.
        if in_dim == 0 || out_units == 0 || out_units * in_dim != data.len() {
            return;
        }
        for row in 0..out_units {
            let start = row * in_dim;
            let slice = &data[start..start + in_dim];
            let sum_sq: f64 = slice.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
            let row_l2 = sum_sq.sqrt() as f32;
            if row_l2 < 1e-6 {
                failures.push(format!(
                    "[F-DATA-QUALITY-007] DEAD OUTPUT ROW: output row {row}/{out_units} has \
                     L2~0 (dead token — logit is structurally constant; the incumbents load \
                     and run this silently)"
                ));
                // One representative dead row is sufficient to fail closed.
                break;
            }
        }
    }

    /// Roles whose fully-zero output row is unambiguously corrupt (PMAT-889).
    /// Conservative allow-list: lm_head / output head and the token embedding.
    fn is_output_projection_role(name: &str) -> bool {
        let n = name.to_lowercase();
        n.contains("lm_head")
            || n == "output.weight"
            || n.ends_with("output.weight")
            || n.contains("embed")
            || n.contains("tok_embeddings")
            || n.contains("wte")
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

        // Contract: apr-inspect-metadata-propagation-v1 F-INSPECT-META-001 (paiml/aprender#622).
        // Surface ALL on-disk GGUF KV pairs using their authentic keys (e.g., qwen2.embedding_length,
        // general.architecture, tokenizer.ggml.model). Previously this was a 4-key hand-written stub
        // that fabricated ML-shorthand names (n_embd, n_heads, n_layers) — see Five Whys in the
        // contract YAML for full root-cause analysis.
        let meta_map: BTreeMap<String, String> = result.raw_metadata.clone();

        // Contract: apr-inspect-dtype-naming-v1 F-INSPECT-DTYPE-001 (paiml/aprender#619).
        // Render GGML dtype as a human-readable name (F32, Q4_K, Q6_K, …), not the raw u32
        // discriminant. Delegates to the same lookup used by `apr tensors` for cross-cmd parity.
        let tensors: Vec<TensorInfo> = result
            .tensors
            .iter()
            .map(|(name, t): (&String, &GgufRawTensor)| TensorInfo {
                name: name.clone(),
                dtype: crate::format::tensors::ggml_dtype_name(t.dtype).to_string(),
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

        // Contract: apr-inspect-quantization-v1 F-INSPECT-QUANT-001 (paiml/aprender#603).
        // The model's "quantization" is the dominant dtype by parameter count among its WEIGHT
        // tensors — biases and norm layers are excluded because they are typically kept in F32
        // even for heavily-quantized models. Previous code picked tensors.first() which, after
        // alphabetical BTreeMap ordering, was always blk.0.attn_k.bias (F32). See Five Whys in
        // contracts/apr-inspect-quantization-v1.yaml.
        let quantization = {
            let mut params_by_dtype: std::collections::HashMap<&str, usize> =
                std::collections::HashMap::new();
            for t in &tensors {
                let name_lower = t.name.to_lowercase();
                let is_weight = !(name_lower.contains("bias")
                    || name_lower.contains("norm")
                    || name_lower.contains("ln_"));
                if is_weight {
                    let params: usize = t.shape.iter().product();
                    *params_by_dtype.entry(t.dtype.as_str()).or_insert(0) += params;
                }
            }
            params_by_dtype
                .into_iter()
                .max_by_key(|(_, params)| *params)
                .map(|(dtype, _)| dtype.to_string())
        };

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
        let mut max_data_end: usize = 0;

        for name in tensor_names {
            if let Some(info) = mapped.get_metadata(name) {
                let info: &TensorMetadata = info;
                let shape: Vec<usize> = info.shape.clone();
                let params: usize = shape.iter().product();
                total_params += params;

                let data_len = info.data_offsets[1] - info.data_offsets[0];
                if info.data_offsets[1] > max_data_end {
                    max_data_end = info.data_offsets[1];
                }

                tensors.push(TensorInfo {
                    name: name.to_string(),
                    dtype: info.dtype.clone(),
                    shape,
                    size_bytes: data_len,
                    stats: None,
                });
            }
        }

        // PMAT-264: Detect truncated data section (valid header but payload truncated)
        let data_offset = mapped.data_offset();
        let required_size = data_offset + max_data_end;
        if required_size > file_size {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Truncated SafeTensors data: tensors require {required_size} bytes but file is only {file_size} bytes"
                ),
            });
        }

        // GH-249: Infer architecture from tensor names for SafeTensors
        let architecture = Self::infer_architecture_from_tensors(&tensors);

        Ok(InspectionReport {
            format: FormatType::SafeTensors,
            file_size,
            metadata: mapped.user_metadata().clone(),
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
                    dtype: entry.dtype.to_string(),
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
