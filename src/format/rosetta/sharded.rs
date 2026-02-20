impl RosettaStone {

    /// Bug 212: Inspect a sharded SafeTensors model via its index.json.
    /// Iterates shard files and aggregates tensor metadata.
    fn inspect_sharded_safetensors(&self, index_path: &Path) -> Result<InspectionReport> {
        use crate::format::sharded::ShardIndex;
        use crate::serialization::safetensors::{MappedSafeTensors, TensorMetadata};

        let content =
            std::fs::read_to_string(index_path).map_err(|e| AprenderError::FormatError {
                message: format!("Failed to read shard index {}: {e}", index_path.display()),
            })?;
        let index = ShardIndex::from_json(&content)?;

        let base_dir = index_path
            .parent()
            .ok_or_else(|| AprenderError::FormatError {
                message: format!(
                    "Cannot determine parent directory of {}",
                    index_path.display()
                ),
            })?;

        let mut tensors = Vec::new();
        let mut total_params: usize = 0;
        let mut total_file_size: usize = 0;
        let mut user_meta: BTreeMap<String, String> = BTreeMap::new();

        for shard_file in index.shard_files() {
            let shard_path = base_dir.join(shard_file);
            if !shard_path.exists() {
                continue;
            }

            total_file_size += std::fs::metadata(&shard_path)
                .map(|m| m.len() as usize)
                .unwrap_or(0);

            let mapped =
                MappedSafeTensors::open(&shard_path).map_err(|e| AprenderError::FormatError {
                    message: format!("SafeTensors open failed for shard {shard_file}: {e}"),
                })?;

            // GH-271: Collect user metadata from first shard that has it
            if user_meta.is_empty() {
                let shard_meta = mapped.user_metadata();
                if !shard_meta.is_empty() {
                    user_meta.clone_from(shard_meta);
                }
            }

            for name in mapped.tensor_names() {
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
        }

        Ok(InspectionReport {
            format: FormatType::SafeTensors,
            file_size: total_file_size,
            metadata: {
                user_meta.insert("shards".to_string(), index.shard_count().to_string());
                user_meta
            },
            tensors,
            total_params,
            quantization: None,
            architecture: None,
        })
    }

    /// Bug 212: Convert a sharded SafeTensors model to any target format.
    /// Routes through import (sharded ST → APR) then converts APR → target.
    fn convert_sharded(
        &self,
        source: &Path,
        target: &Path,
        target_format: FormatType,
        opts: &ConversionOptions,
    ) -> Result<()> {
        use crate::format::converter::{apr_import, ImportOptions};

        let source_str = source.to_string_lossy();
        let effective_tokenizer = opts.tokenizer_path.clone().or_else(|| {
            let sibling = source.with_file_name("tokenizer.json");
            if sibling.exists() {
                Some(sibling)
            } else {
                None
            }
        });
        let import_opts = ImportOptions {
            tokenizer_path: effective_tokenizer,
            allow_no_config: true, // Sharded models may have config.json; let import warn
            ..ImportOptions::default()
        };

        if target_format == FormatType::Apr {
            // Direct: sharded ST → APR via import
            eprintln!(
                "[BUG-212] Converting sharded SafeTensors → APR: {}",
                source.display()
            );
            apr_import(&source_str, target, import_opts)?;
        } else {
            // Sharded ST conversion via intermediate APR
            let temp_apr = std::env::temp_dir().join("rosetta_sharded_temp.apr");
            eprintln!(
                "[BUG-212] Converting sharded SafeTensors → {} (via temp APR): {}",
                target_format,
                source.display()
            );
            apr_import(&source_str, &temp_apr, import_opts)?;
            self.convert_internal(&temp_apr, target, FormatType::Apr, target_format, opts)?;
            let _ = std::fs::remove_file(&temp_apr);
        }

        Ok(())
    }

    #[allow(clippy::self_only_used_in_recursion)] // Self is needed for recursive convert calls
    fn convert_internal(
        &self,
        source: &Path,
        target: &Path,
        source_format: FormatType,
        target_format: FormatType,
        opts: &ConversionOptions,
    ) -> Result<()> {
        use crate::format::converter::{
            apr_export, apr_import, ExportFormat, ExportOptions, ImportOptions, QuantizationType,
        };

        // GH-205 FIX: Map ConversionOptions.quantization to ExportOptions.quantize
        // Previously opts was ignored, causing F32 GGUF export even when quantization requested.
        // Note: Q6_K maps to Q4K since that's what realizar's inference supports.
        let export_quantize =
            opts.quantization
                .as_ref()
                .and_then(|q| match q.to_lowercase().as_str() {
                    "q4_k" | "q4_k_m" | "int4" | "q6_k" => Some(QuantizationType::Q4K),
                    "int8" | "q8_0" => Some(QuantizationType::Int8),
                    "fp16" | "f16" => Some(QuantizationType::Fp16),
                    _ => None,
                });

        match (source_format, target_format) {
            // GGUF/SafeTensors → APR (same conversion path via apr_import)
            // GH-196: Default ImportOptions are permissive (strict=false),
            // so format conversion proceeds with warnings for unverified architectures.
            (FormatType::Gguf | FormatType::SafeTensors, FormatType::Apr) => {
                let source_str = source.to_string_lossy();
                let effective_tokenizer = opts.tokenizer_path.clone().or_else(|| {
                    let sibling = source.with_file_name("tokenizer.json");
                    if sibling.exists() {
                        Some(sibling)
                    } else {
                        None
                    }
                });
                let import_opts = ImportOptions {
                    tokenizer_path: effective_tokenizer,
                    allow_no_config: true,
                    ..ImportOptions::default()
                };
                apr_import(&source_str, target, import_opts)?;
                Ok(())
            }

            // APR → GGUF
            // GH-205 FIX: Default to Q4_K quantization for GGUF export.
            // F32 GGUF files don't work with realizar's fused matmul kernels
            // (see export.rs:532-537 comment). Q4_K is the standard format.
            (FormatType::Apr, FormatType::Gguf) => {
                let gguf_quantize = export_quantize.clone().or(Some(QuantizationType::Q4K)); // Default to Q4K for GGUF
                apr_export(
                    source,
                    target,
                    ExportOptions {
                        format: ExportFormat::Gguf,
                        quantize: gguf_quantize,
                        ..Default::default()
                    },
                )?;
                Ok(())
            }

            // APR → SafeTensors
            (FormatType::Apr, FormatType::SafeTensors) => {
                apr_export(
                    source,
                    target,
                    ExportOptions {
                        format: ExportFormat::SafeTensors,
                        ..Default::default()
                    },
                )?;
                Ok(())
            }

            // GGUF → SafeTensors (via APR)
            (FormatType::Gguf, FormatType::SafeTensors) => {
                let temp_apr = std::env::temp_dir().join("rosetta_temp.apr");
                self.convert_internal(source, &temp_apr, FormatType::Gguf, FormatType::Apr, opts)?;
                self.convert_internal(
                    &temp_apr,
                    target,
                    FormatType::Apr,
                    FormatType::SafeTensors,
                    opts,
                )?;
                let _ = std::fs::remove_file(temp_apr);
                Ok(())
            }

            // SafeTensors → GGUF (via APR)
            (FormatType::SafeTensors, FormatType::Gguf) => {
                let temp_apr = std::env::temp_dir().join("rosetta_temp.apr");
                self.convert_internal(
                    source,
                    &temp_apr,
                    FormatType::SafeTensors,
                    FormatType::Apr,
                    opts,
                )?;
                self.convert_internal(&temp_apr, target, FormatType::Apr, FormatType::Gguf, opts)?;
                let _ = std::fs::remove_file(temp_apr);
                Ok(())
            }

            // Same format - just copy
            (f1, f2) if f1 == f2 => {
                std::fs::copy(source, target).map_err(|e| AprenderError::FormatError {
                    message: format!("Copy failed: {e}"),
                })?;
                Ok(())
            }

            _ => Err(AprenderError::FormatError {
                message: format!("Conversion {source_format} → {target_format} not supported"),
            }),
        }
    }

    fn compare_files(&self, file_a: &Path, file_b: &Path) -> Result<VerificationReport> {
        let inspection_a = self.inspect(file_a)?;
        let inspection_b = self.inspect(file_b)?;

        // Compare tensor counts
        if inspection_a.tensors.len() != inspection_b.tensors.len() {
            return Ok(VerificationReport {
                is_equivalent: false,
                max_diff: f32::INFINITY,
                mean_diff: f32::INFINITY,
                tensor_diffs: BTreeMap::new(),
                changed_metadata: Vec::new(),
                failed_tensors: vec!["Tensor count mismatch".to_string()],
            });
        }

        // Compare tensor statistics (Toyota Way: no SATD, implement now)
        // Uses statistical comparison: if stats match closely, tensors are equivalent
        let mut tensor_diffs = BTreeMap::new();
        let mut max_diff: f32 = 0.0;
        let mut total_diff: f32 = 0.0;
        let mut diff_count: usize = 0;
        let mut failed_tensors = Vec::new();

        for (tensor_a, tensor_b) in inspection_a.tensors.iter().zip(inspection_b.tensors.iter()) {
            // Check tensor names match
            if tensor_a.name != tensor_b.name {
                failed_tensors.push(format!(
                    "Tensor name mismatch: {} vs {}",
                    tensor_a.name, tensor_b.name
                ));
                continue;
            }

            // Check shapes match
            if tensor_a.shape != tensor_b.shape {
                failed_tensors.push(format!(
                    "{}: shape mismatch {:?} vs {:?}",
                    tensor_a.name, tensor_a.shape, tensor_b.shape
                ));
                continue;
            }

            // Compare statistics if available
            match (&tensor_a.stats, &tensor_b.stats) {
                (Some(stats_a), Some(stats_b)) => {
                    let mean_diff = (stats_a.mean - stats_b.mean).abs();
                    let std_diff = (stats_a.std - stats_b.std).abs();
                    let min_diff = (stats_a.min - stats_b.min).abs();
                    let max_val_diff = (stats_a.max - stats_b.max).abs();

                    let tensor_max_diff = mean_diff.max(std_diff).max(min_diff).max(max_val_diff);
                    tensor_diffs.insert(tensor_a.name.clone(), tensor_max_diff);

                    max_diff = max_diff.max(tensor_max_diff);
                    total_diff += tensor_max_diff;
                    diff_count += 1;
                }
                _ => {
                    // No stats available, assume matching if shapes match
                    tensor_diffs.insert(tensor_a.name.clone(), 0.0);
                }
            }
        }

        let mean_diff = if diff_count > 0 {
            total_diff / diff_count as f32
        } else {
            0.0
        };

        // Threshold: max_diff < 1e-4 is considered equivalent (float precision)
        let is_equivalent = failed_tensors.is_empty() && max_diff < 1e-4;

        Ok(VerificationReport {
            is_equivalent,
            max_diff,
            mean_diff,
            tensor_diffs,
            changed_metadata: Vec::new(),
            failed_tensors,
        })
    }
}
