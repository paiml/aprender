
/// Legacy type alias for APR v2 model
pub type AprModel = AprV2Model;
/// Legacy type alias (model types are now in metadata)
pub type AprModelType = ();

/// Extract special tokens from vocabulary for atomic tokenization (GH-189)
///
/// Special tokens like `<|im_start|>`, `<|im_end|>` must be tokenized atomically,
/// not split character-by-character. This function scans the vocabulary for
/// special token patterns rather than maintaining a hardcoded list.
///
/// GH-320: Uses pattern-based detection instead of a manually-maintained list.
/// Two patterns cover all known model families:
/// - `<|...|>` — ChatML, Qwen, Phi, code models, etc.
/// - `<tag>` / `</tag>` — standard transformer tokens (`<s>`, `</s>`, `<pad>`, etc.)
#[allow(clippy::implicit_hasher)]
pub fn extract_special_tokens_from_vocab(
    token_to_id: &HashMap<String, u32>,
) -> HashMap<String, u32> {
    let mut special_tokens = HashMap::new();

    for (token, &id) in token_to_id {
        if is_special_token(token) {
            special_tokens.insert(token.clone(), id);
        }
    }

    special_tokens
}

/// GH-320: Detect special tokens by pattern instead of hardcoded list.
///
/// Matches two families of control tokens found in transformer vocabularies:
/// - `<|...|>` — model-specific tokens (e.g., `<|im_start|>`, `<|endoftext|>`)
/// - `<tag>` / `</tag>` — standard tokens where tag is short lowercase
///   alphanumeric/underscore (e.g., `<s>`, `</s>`, `<pad>`, `<unk>`, `<bos>`)
#[must_use]
fn is_special_token(token: &str) -> bool {
    // ChatML / model-specific: <|...|> (e.g., <|im_start|>, <|endoftext|>)
    if token.starts_with("<|") && token.ends_with("|>") && token.len() > 4 {
        return true;
    }
    // Standard transformer control tokens: <s>, </s>, <pad>, <unk>, <bos>, <eos>, <mask>
    // Pattern: <tag> or </tag> where tag is short, lowercase, alphanumeric/underscore
    if token.starts_with('<') && token.ends_with('>') && token.len() <= 20 {
        let inner = &token[1..token.len() - 1];
        let inner = inner.strip_prefix('/').unwrap_or(inner);
        if !inner.is_empty() && inner.chars().all(|c| c.is_ascii_lowercase() || c == '_') {
            return true;
        }
    }
    false
}

// =============================================================================

use memmap2::Mmap;

/// Memory-mapped APR model for fast loading and GPU inference
///
/// Similar to MappedGGUFModel, this provides zero-copy access to APR tensor data.
/// The file is memory-mapped for fast startup (~36x faster than full file read).
#[derive(Debug)]
pub struct MappedAprModel {
    /// APR header
    pub header: AprHeader,
    /// Model metadata
    pub metadata: AprMetadata,
    /// Tensor index
    pub tensors: Vec<TensorEntry>,
    /// Memory-mapped file data
    mmap: Mmap,
}

impl MappedAprModel {
    /// Load an APR model with memory mapping for fast startup
    ///
    /// # Arguments
    /// * `path` - Path to the .apr file
    ///
    /// # Errors
    /// Returns error if file cannot be opened or has invalid format.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref()).map_err(|e| RealizarError::IoError {
            message: format!("Failed to open .apr file: {e}"),
        })?;

        // SAFETY: File is opened read-only, callers validate format before trusting data
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| RealizarError::IoError {
                message: format!("Failed to mmap .apr file: {e}"),
            })?
        };

        Self::from_mmap(mmap)
    }

    /// Create from existing memory map
    fn from_mmap(mmap: Mmap) -> Result<Self> {
        let data = mmap.get(..).expect("mmap deref to full slice");

        // Parse header
        let header = AprHeader::from_bytes(data)?;

        // Validate magic
        if header.magic != MAGIC {
            return Err(RealizarError::FormatError {
                reason: "Invalid APR magic bytes".to_string(),
            });
        }

        // Parse metadata
        let metadata_start = header.metadata_offset as usize;
        let metadata_end = metadata_start + header.metadata_size as usize;

        if data.len() < metadata_end {
            return Err(RealizarError::FormatError {
                reason: "APR file truncated: metadata extends past EOF".to_string(),
            });
        }

        let metadata: AprMetadata = if header.metadata_size > 0 {
            serde_json::from_slice(&data[metadata_start..metadata_end]).unwrap_or_default()
        } else {
            AprMetadata::default()
        };

        // Parse tensor index
        let index_start = header.tensor_index_offset as usize;
        let index_end = header.data_offset as usize;

        let mut tensors = Vec::with_capacity(header.tensor_count as usize);
        if index_start < index_end && index_end <= data.len() {
            let index_data = &data[index_start..index_end];
            let mut pos = 0;

            while pos < index_data.len() && tensors.len() < header.tensor_count as usize {
                match TensorEntry::from_binary(&index_data[pos..]) {
                    Ok((entry, consumed)) => {
                        tensors.push(entry);
                        pos += consumed;
                    },
                    Err(_) => break,
                }
            }
        }

        Ok(Self {
            header,
            metadata,
            tensors,
            mmap,
        })
    }

    /// Get raw file data (for tensor access)
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.mmap[..]
    }

    /// Get file size in bytes
    #[must_use]
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }

    /// Get tensor count
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Get data offset (start of tensor data section)
    #[must_use]
    pub fn data_offset(&self) -> u64 {
        self.header.data_offset
    }

    /// Find tensor by name
    #[must_use]
    pub fn find_tensor(&self, name: &str) -> Option<&TensorEntry> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Get raw tensor data by name
    pub fn get_tensor_data(&self, name: &str) -> Result<&[u8]> {
        let tensor = self
            .find_tensor(name)
            .ok_or_else(|| RealizarError::FormatError {
                reason: format!("Tensor not found: {name}"),
            })?;

        let start = self.header.data_offset as usize + tensor.offset as usize;
        let end = start + tensor.size as usize;

        if end > self.mmap.len() {
            return Err(RealizarError::FormatError {
                reason: format!("Tensor {name} extends past EOF"),
            });
        }

        Ok(&self.mmap[start..end])
    }

    /// Convert APR dtype string to GGML qtype
    #[must_use]
    pub fn dtype_to_qtype(dtype: &str) -> u32 {
        match dtype {
            "F32" => 0,
            "F16" => 1,
            "Q4_0" => 2,
            "Q4_1" => 3,
            "Q5_0" => 6,
            "Q5_1" => 7,
            "Q8_0" => 8,
            "Q8_1" => 9,
            "Q2_K" => 10,
            "Q3_K" => 11,
            "Q4_K" => 12,
            "Q5_K" => 13,
            "Q6_K" => 14,
            "IQ2_XXS" => 16,
            "IQ2_XS" => 17,
            "BF16" => 30,
            _ => 0, // Default to F32
        }
    }
}

// Tests extracted to tests.rs (PMAT-802)
#[cfg(test)]
#[path = "tests.rs"]
mod apr_tests;

// Additional tests for coverage (tests_part_02.rs)
#[cfg(test)]
#[path = "tests_apr_flags.rs"]
mod apr_tests_part_02;

// T-COV-95 Additional coverage (tests_part_03.rs)
#[cfg(test)]
#[path = "tests_decode_tokens.rs"]
mod apr_tests_part_03;

// T-COV-95 Active APR Pygmy: Cross-Format Dynamic Falsification (tests_pygmy_apr.rs)
#[cfg(test)]
#[path = "tests_pygmy_apr.rs"]
mod apr_tests_pygmy_apr;

// T-COV-95 Coverage Bridge (Part 05 - AprFlags, AprHeader, TensorEntry, AprMetadata, dtype_to_qtype)
#[cfg(test)]
#[path = "tests_apr_flags_02.rs"]
mod apr_tests_part_05;

// T-COV-95 Coverage Bridge (Part 06 - from_bytes edge cases, predict, tensor access, metadata aliases)
#[cfg(test)]
#[path = "tests_apr_v2_bytes.rs"]
mod apr_tests_v2_bytes;
