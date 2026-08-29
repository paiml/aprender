impl AprV2Model {
    /// Load a model from a .apr file using memory mapping.
    ///
    /// # Memory Efficiency
    ///
    /// For uncompressed files, uses `mmap()` for zero-copy access.
    /// The kernel manages pages via demand paging - only accessed
    /// pages are loaded into RAM. After GPU transfer, call
    /// `release_cpu_pages()` to advise the kernel to drop pages.
    ///
    /// For compressed files, falls back to heap allocation after
    /// decompression (mmap not possible for decompressed data).
    ///
    /// # References
    ///
    /// - Didona et al. (2022): mmap achieves 2.3x throughput vs read()
    /// - See docs/model-loading.md for design rationale
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        use std::io::Read;

        let path_ref = path.as_ref();

        // Read just the header first to check for compression
        let mut file = File::open(path_ref).map_err(|e| RealizarError::IoError {
            message: format!("Failed to open .apr file: {e}"),
        })?;

        let mut header_buf = [0u8; HEADER_SIZE];
        file.read_exact(&mut header_buf)
            .map_err(|e| RealizarError::IoError {
                message: format!("Failed to read .apr header: {e}"),
            })?;

        let header = AprHeader::from_bytes(&header_buf)?;

        // Check for unsupported features
        if header.flags.is_encrypted() {
            return Err(RealizarError::FormatError {
                reason: "Encrypted .apr files not yet supported".to_string(),
            });
        }

        // Choose loading strategy based on compression
        let data = if header.flags.is_compressed() {
            // Compressed: must read entire file into heap, then decompress
            drop(file); // Close file handle
            let raw_data = std::fs::read(path_ref).map_err(|e| RealizarError::IoError {
                message: format!("Failed to read compressed .apr file: {e}"),
            })?;
            let decompressed = Self::decompress_apr_data(&header, raw_data)?;
            ModelData::from_vec(decompressed)
        } else {
            // Uncompressed: use mmap for zero-copy access
            drop(file); // Close file handle before mmap
            ModelData::open_mmap(path_ref)?
        };

        // Advise sequential access pattern for parsing
        #[cfg(unix)]
        let _ = data.advise_sequential();

        Self::from_model_data(header, data)
    }

    /// Load a model from a .apr file (WASM fallback).
    #[cfg(target_arch = "wasm32")]
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let raw_data = std::fs::read(path.as_ref()).map_err(|e| RealizarError::IoError {
            message: format!("Failed to read .apr file: {e}"),
        })?;
        Self::from_bytes(raw_data)
    }

    /// Load a model from bytes (heap-allocated).
    ///
    /// Use this for:
    /// - Compressed files after decompression
    /// - Data received over network
    /// - WASM environments (no mmap support)
    ///
    /// For file-based loading with mmap support, use `load()` instead.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        // Parse header
        let header = AprHeader::from_bytes(&data)?;

        // Check for unsupported features
        if header.flags.is_encrypted() {
            return Err(RealizarError::FormatError {
                reason: "Encrypted .apr files not yet supported".to_string(),
            });
        }

        // Decompress data if needed (GH-35)
        let data = if header.flags.is_compressed() {
            Self::decompress_apr_data(&header, data)?
        } else {
            data
        };

        Self::from_model_data(header, ModelData::from_vec(data))
    }

    /// Internal: construct model from header and ModelData.
    fn from_model_data(header: AprHeader, data: ModelData) -> Result<Self> {
        let data_slice = data.as_slice();

        // Parse metadata
        let metadata_start = header.metadata_offset as usize;
        let metadata_end = metadata_start + header.metadata_size as usize;

        if data_slice.len() < metadata_end {
            return Err(RealizarError::FormatError {
                reason: format!(
                    ".apr file truncated: metadata extends to {} but file is {} bytes",
                    metadata_end,
                    data_slice.len()
                ),
            });
        }

        let metadata: AprMetadata = if header.metadata_size > 0 {
            serde_json::from_slice(&data_slice[metadata_start..metadata_end]).unwrap_or_default()
        } else {
            AprMetadata::default()
        };

        // Parse tensor index (binary format from aprender v2)
        let index_start = header.tensor_index_offset as usize;
        let index_end = header.data_offset as usize;

        let mut tensors = Vec::with_capacity(header.tensor_count as usize);
        if index_start < index_end && index_end <= data_slice.len() {
            let index_data = &data_slice[index_start..index_end];
            let mut pos = 0;

            while pos < index_data.len() && tensors.len() < header.tensor_count as usize {
                match TensorEntry::from_binary(&index_data[pos..]) {
                    Ok((entry, consumed)) => {
                        tensors.push(entry);
                        pos += consumed;
                    },
                    Err(_) => break, // Stop on parse error
                }
            }
        }

        let model = Self {
            header,
            metadata,
            tensors,
            data,
        };

        // PMAT-906 / Pillar-4 fail-closed: reject a structurally-inconsistent model
        // at LOAD rather than producing garbage at inference. llama.cpp / Ollama
        // silently load such a model; apr fails closed.
        //   - OBLIG-APR-VOCAB-EMBED-CONSISTENT: config.vocab_size == embedding rows
        //   - OBLIG-APR-WEIGHT-SHAPE-MATCHES-CONFIG: weight cols == config.hidden_size
        model.validate_config_consistency()?;

        Ok(model)
    }

    /// PMAT-906 — Pillar-4 fail-closed: verify the declared transformer config
    /// is structurally consistent with the loaded weight tensors.
    ///
    /// Two checks, mirroring the GGUF/SafeTensors fail-closed gates
    /// (OBLIG-GGUF-LOAD-NANINF, F-DATA-QUALITY-*):
    ///
    /// - **OBLIG-APR-VOCAB-EMBED-CONSISTENT**: `config.vocab_size` MUST equal the
    ///   row count of the token-embedding matrix (and of `lm_head` when present
    ///   and untied). A mismatch means token IDs index past the embedding table
    ///   (or logits are projected to the wrong vocabulary) → garbage / OOB.
    /// - **OBLIG-APR-WEIGHT-SHAPE-MATCHES-CONFIG**: the inner (column) dimension of
    ///   the token-embedding / `lm_head` matrix MUST equal `config.hidden_size`.
    ///   A mismatch means every matmul reads the hidden vector with the wrong
    ///   stride → garbage.
    ///
    /// Only enforced for transformer models that declare BOTH `vocab_size` and
    /// `hidden_size`; non-transformer / simple `predict` models are untouched.
    ///
    /// # Errors
    ///
    /// Returns [`RealizarError::FormatError`] naming the inconsistent tensor and
    /// the declared-vs-actual dimensions.
    fn validate_config_consistency(&self) -> Result<()> {
        let Some(vocab_size) = self.metadata.vocab_size else {
            return Ok(());
        };
        let Some(hidden_size) = self.metadata.hidden_size else {
            return Ok(());
        };

        // Token-embedding matrix is [vocab_size, hidden_size] (row-major).
        const EMBED_NAMES: &[&str] = &[
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "transformer.wte.weight",
            "embeddings.word_embeddings.weight",
            "tok_embeddings.weight",
            "token_embd.weight",
        ];
        if let Some(embed) = EMBED_NAMES.iter().find_map(|n| self.get_tensor(n)) {
            if embed.shape.len() == 2 {
                let (rows, cols) = (embed.shape[0], embed.shape[1]);
                // Embedding may be stored as [vocab, hidden] (HF/SafeTensors) or
                // [hidden, vocab] (some tied/GGUF layouts). Accept either
                // orientation, but vocab_size MUST be one of the two dims and
                // hidden_size the other.
                let vocab_ok = rows == vocab_size || cols == vocab_size;
                let hidden_ok = rows == hidden_size || cols == hidden_size;
                if !vocab_ok {
                    return Err(RealizarError::FormatError {
                        reason: format!(
                            "OBLIG-APR-VOCAB-EMBED-CONSISTENT: config vocab_size={vocab_size} \
                             does not match embedding tensor '{}' shape {:?} — neither dim is \
                             {vocab_size}. The declared vocabulary is inconsistent with the \
                             embedding matrix; token IDs would index out of bounds and inference \
                             would produce garbage. Re-convert the model.",
                            embed.name, embed.shape
                        ),
                    });
                }
                if !hidden_ok {
                    return Err(RealizarError::FormatError {
                        reason: format!(
                            "OBLIG-APR-WEIGHT-SHAPE-MATCHES-CONFIG: config hidden_size={hidden_size} \
                             does not match embedding tensor '{}' shape {:?} — neither dim is \
                             {hidden_size}. The hidden vector would be read with the wrong stride \
                             and inference would produce garbage. Re-convert the model.",
                            embed.name, embed.shape
                        ),
                    });
                }
            }
        }

        // Separate (untied) lm_head, when present, is [vocab_size, hidden_size].
        if let Some(lm_head) = self.get_tensor("lm_head.weight") {
            if lm_head.shape.len() == 2 {
                let (rows, cols) = (lm_head.shape[0], lm_head.shape[1]);
                if rows != vocab_size && cols != vocab_size {
                    return Err(RealizarError::FormatError {
                        reason: format!(
                            "OBLIG-APR-VOCAB-EMBED-CONSISTENT: config vocab_size={vocab_size} \
                             does not match lm_head tensor '{}' shape {:?}. The output projection \
                             targets the wrong vocabulary; inference would produce garbage. \
                             Re-convert the model.",
                            lm_head.name, lm_head.shape
                        ),
                    });
                }
                if rows != hidden_size && cols != hidden_size {
                    return Err(RealizarError::FormatError {
                        reason: format!(
                            "OBLIG-APR-WEIGHT-SHAPE-MATCHES-CONFIG: config hidden_size={hidden_size} \
                             does not match lm_head tensor '{}' shape {:?}. The output projection \
                             reads the hidden vector with the wrong stride; inference would produce \
                             garbage. Re-convert the model.",
                            lm_head.name, lm_head.shape
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    /// Decompress APR data based on compression flags (GH-35)
    ///
    /// The compressed format stores: header (64 bytes, uncompressed) + compressed payload.
    /// We decompress the payload and reconstruct the full data vector.
    #[allow(unreachable_patterns)] // Pattern varies based on apr-compression feature
    fn decompress_apr_data(header: &AprHeader, data: Vec<u8>) -> Result<Vec<u8>> {
        #[cfg(feature = "apr-compression")]
        let compressed_payload = &data[HEADER_SIZE..];

        #[cfg(feature = "apr-compression")]
        {
            let decompressed = if header.flags.is_lz4() {
                lz4_flex::decompress_size_prepended(compressed_payload).map_err(|e| {
                    RealizarError::FormatError {
                        reason: format!("LZ4 decompression failed: {e}"),
                    }
                })?
            } else if header.flags.is_zstd() {
                zstd::decode_all(compressed_payload).map_err(|e| RealizarError::FormatError {
                    reason: format!("ZSTD decompression failed: {e}"),
                })?
            } else {
                // Unknown compression - should not happen
                return Err(RealizarError::FormatError {
                    reason: "Unknown compression algorithm in APR flags".to_string(),
                });
            };

            // Reconstruct full data: header + decompressed payload
            let mut result = Vec::with_capacity(HEADER_SIZE + decompressed.len());
            result.extend_from_slice(&data[..HEADER_SIZE]);
            result.extend_from_slice(&decompressed);
            Ok(result)
        }

        #[cfg(not(feature = "apr-compression"))]
        {
            let _ = (header, &data); // Suppress unused warnings
            Err(RealizarError::FormatError {
                reason: "Compressed .apr files require 'apr-compression' feature".to_string(),
            })
        }
    }

    /// Get number of tensors
    #[must_use]
    pub fn tensor_count(&self) -> u32 {
        self.header.tensor_count
    }

    /// Get tensor names
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    /// Get metadata
    #[must_use]
    pub fn metadata(&self) -> &AprMetadata {
        &self.metadata
    }

    /// Get tensor by name
    #[must_use]
    pub fn get_tensor(&self, name: &str) -> Option<&TensorEntry> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Get tensor data as f32 slice
    pub fn get_tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let entry = self
            .get_tensor(name)
            .ok_or_else(|| RealizarError::FormatError {
                reason: format!("Tensor not found: {name}"),
            })?;

        let start = (self.header.data_offset + entry.offset) as usize;
        let end = start + entry.size as usize;
        let data_slice = self.data.as_slice();

        if end > data_slice.len() {
            return Err(RealizarError::FormatError {
                reason: format!("Tensor data out of bounds: {name}"),
            });
        }

        let bytes = &data_slice[start..end];

        // Calculate total number of elements from shape
        let num_elements: usize = entry.shape.iter().product();

        // Parse based on dtype
        // GH-250: APR-native Q8/Q4 formats differ from GGML Q8_0/Q4_K!
        //   APR Q8: [scale: f32 (4B)] + [i8 × N]  (single whole-tensor scale)
        //   APR Q4: per-32-block [scale: f16 (2B)] + [16 packed nibble bytes]
        //   GGML Q8_0: per-32-block [scale: f16 (2B)] + [32 × i8]
        //   GGML Q4_K: 256-element super-blocks with sub-block scales
        // APR Q4_K/Q6_K are passthrough GGML formats (from add_q4k_raw_tensor).
        match entry.dtype.as_str() {
            "F32" | "f32" => {
                let floats: Vec<f32> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Ok(floats)
            },
            "F16" | "f16" => Ok(dequantize_f16(bytes, num_elements)),
            // APR-native formats (different from GGML!)
            "q8" | "Q8" => Ok(dequant::dequantize_apr_q8(bytes, num_elements)),
            "q4" | "Q4" => Ok(dequant::dequantize_apr_q4(bytes, num_elements)),
            // GGML-compatible formats (passthrough from GGUF import)
            "Q8_0" | "q8_0" => Ok(dequantize_q8_0(bytes, num_elements)),
            "Q4_K" | "q4_k" => Ok(dequantize_q4_k(bytes, num_elements)),
            "Q6_K" | "q6_k" => Ok(dequantize_q6_k(bytes, num_elements)),
            dtype => Err(RealizarError::FormatError {
                reason: format!("Unsupported tensor dtype: {dtype}"),
            }),
        }
    }

    /// Get raw tensor bytes
    pub fn get_tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let entry = self
            .get_tensor(name)
            .ok_or_else(|| RealizarError::FormatError {
                reason: format!("Tensor not found: {name}"),
            })?;

        let start = (self.header.data_offset + entry.offset) as usize;
        let end = start + entry.size as usize;
        let data_slice = self.data.as_slice();

        if end > data_slice.len() {
            return Err(RealizarError::FormatError {
                reason: format!("Tensor data out of bounds: {name}"),
            });
        }

        Ok(&data_slice[start..end])
    }

    /// Release CPU pages after GPU transfer (Unix only).
    ///
    /// Advises the kernel that the mapped pages are no longer needed.
    /// The kernel will drop pages immediately (not compress to zram)
    /// and re-fault from disk if accessed again.
    ///
    /// # When to Call
    ///
    /// After all tensor data has been copied to GPU via `cuMemcpy()`.
    /// This is the key method for reducing zram pressure.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let model = AprV2Model::load("model.apr")?;
    /// for name in model.tensor_names() {
    ///     let bytes = model.get_tensor_bytes(&name)?;
    ///     cuda::memcpy_htod(gpu_ptr, bytes);
    /// }
    /// // Free CPU pages now that data is on GPU
    /// model.release_cpu_pages()?;
    /// ```
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    pub fn release_cpu_pages(&self) -> Result<()> {
        self.data.release_cpu_pages()
    }

    /// No-op on non-Unix platforms (madvise not available).
    #[cfg(not(all(unix, not(target_arch = "wasm32"))))]
    pub fn release_cpu_pages(&self) -> Result<()> {
        Ok(())
    }

    /// Check if model is using memory-mapped I/O.
    ///
    /// Returns `true` if the model was loaded via mmap (uncompressed file).
    /// Returns `false` if the model is heap-allocated (compressed file or WASM).
    #[must_use]
    pub fn is_mmap(&self) -> bool {
        self.data.is_mmap()
    }

    /// Estimate total parameters
    #[must_use]
    pub fn estimated_parameters(&self) -> usize {
        self.tensors
            .iter()
            .map(|t| t.shape.iter().product::<usize>())
            .sum()
    }

    /// Run inference on input features (for simple models)
    ///
    /// For transformer models, use `forward()` instead.
    ///
    /// # Arguments
    ///
    /// * `features` - Input feature vector
    ///
    /// # Returns
    ///
    /// Output vector
    ///
    /// # Errors
    ///
    /// Returns error if model has no tensors
    pub fn predict(&self, features: &[f32]) -> Result<Vec<f32>> {
        if self.tensors.is_empty() && self.header.tensor_count == 0 {
            let sum: f32 = features.iter().sum();
            return Ok(vec![sum]);
        }

        // Linear model: y = Wx + b (if we have weights)
        if let Some(weight) = self.get_tensor("weight") {
            let weights = self.get_tensor_f32("weight")?;
            let bias = self.get_tensor_f32("bias").unwrap_or_default();

            let output_dim = weight.shape.first().copied().unwrap_or(1);
            let input_dim = weight.shape.get(1).copied().unwrap_or(features.len());

            let mut output = vec![0.0; output_dim];
            for (i, out) in output.iter_mut().enumerate() {
                for (j, &feat) in features.iter().take(input_dim).enumerate() {
                    *out += weights.get(i * input_dim + j).copied().unwrap_or(0.0) * feat;
                }
                *out += bias.get(i).copied().unwrap_or(0.0);
            }
            return Ok(output);
        }

        let sum: f32 = features.iter().sum();
        Ok(vec![sum])
    }
}
