
/// Parse and bounds-check the JSON metadata section (FALSIFY-PARSE-001).
///
/// All offsets/sizes here come straight from the (attacker-controllable) file
/// header — the CRC32 checksum is computed over the header itself, so a
/// corrupted file can carry a matching checksum. Therefore every offset+size is
/// validated with checked arithmetic and `slice::get` (never `[start..end]`
/// indexing, which panics on out-of-range / `start > end`).
fn parse_metadata_section(
    data: &[u8],
    metadata_offset: u64,
    metadata_size: u32,
) -> Result<AprV2Metadata, V2FormatError> {
    let start = usize::try_from(metadata_offset)
        .map_err(|_| V2FormatError::InvalidHeader("metadata_offset exceeds usize".to_string()))?;
    let end = start
        .checked_add(metadata_size as usize)
        .ok_or_else(|| V2FormatError::InvalidHeader("metadata offset+size overflow".to_string()))?;
    let slice = data
        .get(start..end)
        .ok_or_else(|| V2FormatError::InvalidHeader("file too small for metadata".to_string()))?;
    AprV2Metadata::from_json(slice)
}

/// Parse and bounds-check the tensor index section (FALSIFY-PARSE-001).
///
/// `tensor_index_offset` is attacker-controllable; previously it was used
/// directly as `&data[pos..]`, which PANICS ("range start index out of bounds")
/// when the offset points past EOF. Now the start offset is validated against
/// the file length before slicing, and each entry advances `pos` with the same
/// `slice::get` guard.
fn parse_tensor_index_section(
    data: &[u8],
    tensor_index_offset: u64,
    tensor_count: u32,
) -> Result<Vec<TensorIndexEntry>, V2FormatError> {
    let mut pos = usize::try_from(tensor_index_offset).map_err(|_| {
        V2FormatError::InvalidTensorIndex("tensor_index_offset exceeds usize".to_string())
    })?;

    let mut tensor_index = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        // `data.get(pos..)` returns None only when pos > data.len(); pos == len
        // yields an empty slice, which TensorIndexEntry::from_bytes rejects
        // cleanly. This replaces the panicking `&data[pos..]`.
        let remaining = data.get(pos..).ok_or_else(|| {
            V2FormatError::InvalidTensorIndex("tensor index offset past end of file".to_string())
        })?;
        let (entry, consumed) = TensorIndexEntry::from_bytes(remaining)?;
        tensor_index.push(entry);
        pos = pos.checked_add(consumed).ok_or_else(|| {
            V2FormatError::InvalidTensorIndex("tensor index position overflow".to_string())
        })?;
    }

    // Verify tensor names are sorted
    for i in 1..tensor_index.len() {
        if tensor_index[i].name < tensor_index[i - 1].name {
            return Err(V2FormatError::InvalidTensorIndex(
                "tensor index not sorted".to_string(),
            ));
        }
    }

    Ok(tensor_index)
}

impl AprV2Reader {
    /// Read from bytes
    ///
    /// # Errors
    /// Returns error if parsing fails.
    ///
    /// # LAYOUT-002 Jidoka Guard
    /// Rejects APR files with `LAYOUT_COLUMN_MAJOR` flag set, as these indicate
    /// improperly converted GGUF files that would produce garbage output.
    pub fn from_bytes(data: &[u8]) -> Result<Self, V2FormatError> {
        if data.len() < HEADER_SIZE_V2 {
            return Err(V2FormatError::InvalidHeader("file too small".to_string()));
        }

        // Parse header
        let header = AprV2Header::from_bytes(data)?;

        // Verify checksum
        if !header.verify_checksum() {
            return Err(V2FormatError::ChecksumMismatch);
        }

        // LAYOUT-002: Jidoka Guard - Reject "dirty" APR files with column-major layout
        if !header.flags.is_layout_valid() {
            return Err(V2FormatError::InvalidHeader(
                "LAYOUT-002 violation: APR file has LAYOUT_COLUMN_MAJOR flag set. \
                 This indicates a dirty import from GGUF without proper transpose. \
                 Re-import the model using `apr import` with LAYOUT-002 enforcement."
                    .to_string(),
            ));
        }

        // Parse metadata (FALSIFY-PARSE-001 / PMAT-822: checked arithmetic +
        // .get() so a corrupted metadata_offset/size can never panic-slice).
        let metadata = parse_metadata_section(data, header.metadata_offset, header.metadata_size)?;

        // Parse tensor index
        let tensor_index =
            parse_tensor_index_section(data, header.tensor_index_offset, header.tensor_count)?;

        Ok(Self {
            header,
            metadata,
            tensor_index,
            data: data.to_vec(),
        })
    }

    /// Read from a Read impl
    ///
    /// # Errors
    /// Returns error if read fails.
    pub fn from_reader<R: Read>(reader: &mut R) -> Result<Self, V2FormatError> {
        let mut data = Vec::new();
        reader
            .read_to_end(&mut data)
            .map_err(|e| V2FormatError::IoError(e.to_string()))?;
        Self::from_bytes(&data)
    }

    /// Get header
    #[must_use]
    pub fn header(&self) -> &AprV2Header {
        &self.header
    }

    /// Get metadata
    #[must_use]
    pub fn metadata(&self) -> &AprV2Metadata {
        &self.metadata
    }

    /// Get tensor names
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensor_index.iter().map(|e| e.name.as_str()).collect()
    }

    /// Get tensor by name
    #[must_use]
    pub fn get_tensor(&self, name: &str) -> Option<&TensorIndexEntry> {
        self.tensor_index.iter().find(|e| e.name == name)
    }

    /// Get tensor data by name
    #[must_use]
    pub fn get_tensor_data(&self, name: &str) -> Option<&[u8]> {
        let entry = self.get_tensor(name)?;
        // FALSIFY-PARSE-001 / PMAT-822: data_offset + offset (u64) and
        // start + size (usize) can both wrap for a crafted header, letting a
        // wrapped `end <= len` check pass over an OOB region. Use checked
        // arithmetic + `slice::get` so any overflow / past-EOF range → None.
        let abs_offset = self.header.data_offset.checked_add(entry.offset)?;
        let start = usize::try_from(abs_offset).ok()?;
        let end = start.checked_add(usize::try_from(entry.size).ok()?)?;
        self.data.get(start..end)
    }

    /// Get tensor as f32 slice (F32 dtype only)
    #[must_use]
    pub fn get_f32_tensor(&self, name: &str) -> Option<Vec<f32>> {
        let entry = self.get_tensor(name)?;
        if entry.dtype != TensorDType::F32 {
            return None;
        }

        let data = self.get_tensor_data(name)?;
        let floats: Vec<f32> = data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Some(floats)
    }

    /// Get tensor as f32 Vec, dequantizing if necessary
    ///
    /// Supports all tensor types:
    /// - F32: direct copy
    /// - F16: IEEE 754 half-precision → f32
    /// - Q8: 8-bit symmetric dequantization
    /// - Q4: 4-bit block dequantization
    /// - Q4K: GGUF Q4_K super-block dequantization (GH-200)
    /// - Q6K: GGUF Q6_K super-block dequantization (GH-200)
    #[must_use]
    pub fn get_tensor_as_f32(&self, name: &str) -> Option<Vec<f32>> {
        let entry = self.get_tensor(name)?;
        let data = self.get_tensor_data(name)?;
        let element_count = entry.element_count();

        match entry.dtype {
            TensorDType::F32 => {
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Some(floats)
            }
            TensorDType::F16 => {
                let floats: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|chunk| f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
                    .collect();
                Some(floats)
            }
            TensorDType::AprQ8 => {
                if data.len() < 4 {
                    return None;
                }
                let scale = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                let floats: Vec<f32> = data[4..]
                    .iter()
                    .map(|&b| f32::from(b as i8) * scale)
                    .collect();
                Some(floats)
            }
            TensorDType::AprQ4 => Some(dequantize_q4(data, element_count)),
            TensorDType::BF16 => {
                let floats: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        f32::from_bits(u32::from(bits) << 16)
                    })
                    .collect();
                Some(floats)
            }
            TensorDType::Q4K => dequantize_q4_k(data, 0, element_count).ok(),
            TensorDType::Q6K => dequantize_q6_k(data, 0, element_count).ok(),
            _ => None, // Other types not yet supported
        }
    }

    /// Check if all tensors are 64-byte aligned
    #[must_use]
    pub fn verify_alignment(&self) -> bool {
        let data_offset = self.header.data_offset as usize;
        self.tensor_index
            .iter()
            .all(|e| is_aligned_64(data_offset + e.offset as usize))
    }
}

impl<'a> AprV2ReaderRef<'a> {
    /// Read from bytes (zero-copy - borrows data)
    ///
    /// Unlike `AprV2Reader::from_bytes`, this does NOT copy the input data.
    /// The reader borrows the slice, making it ideal for use with mmap.
    ///
    /// # Errors
    /// Returns error if parsing fails.
    ///
    /// # LAYOUT-002 Jidoka Guard
    /// Rejects APR files with `LAYOUT_COLUMN_MAJOR` flag set, as these indicate
    /// improperly converted GGUF files that would produce garbage output.
    pub fn from_bytes(data: &'a [u8]) -> Result<Self, V2FormatError> {
        if data.len() < HEADER_SIZE_V2 {
            return Err(V2FormatError::InvalidHeader("file too small".to_string()));
        }

        // Parse header
        let header = AprV2Header::from_bytes(data)?;

        // Verify checksum
        if !header.verify_checksum() {
            return Err(V2FormatError::ChecksumMismatch);
        }

        // LAYOUT-002: Jidoka Guard - Reject "dirty" APR files with column-major layout
        if !header.flags.is_layout_valid() {
            return Err(V2FormatError::InvalidHeader(
                "LAYOUT-002 violation: APR file has LAYOUT_COLUMN_MAJOR flag set. \
                 This indicates a dirty import from GGUF without proper transpose. \
                 Re-import the model using `apr import` with LAYOUT-002 enforcement."
                    .to_string(),
            ));
        }

        // Parse metadata (FALSIFY-PARSE-001 / PMAT-822: checked arithmetic +
        // .get() so a corrupted metadata_offset/size can never panic-slice).
        let metadata = parse_metadata_section(data, header.metadata_offset, header.metadata_size)?;

        // Parse tensor index
        let tensor_index =
            parse_tensor_index_section(data, header.tensor_index_offset, header.tensor_count)?;

        Ok(Self {
            header,
            metadata,
            tensor_index,
            data, // Borrow, no copy!
        })
    }

    /// Get header
    #[must_use]
    pub fn header(&self) -> &AprV2Header {
        &self.header
    }

    /// Get metadata
    #[must_use]
    pub fn metadata(&self) -> &AprV2Metadata {
        &self.metadata
    }

    /// Get tensor names
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensor_index.iter().map(|e| e.name.as_str()).collect()
    }

    /// Get tensor by name
    #[must_use]
    pub fn get_tensor(&self, name: &str) -> Option<&TensorIndexEntry> {
        self.tensor_index.iter().find(|e| e.name == name)
    }

    /// Get tensor data by name (zero-copy slice into mmap)
    #[must_use]
    pub fn get_tensor_data(&self, name: &str) -> Option<&[u8]> {
        let entry = self.get_tensor(name)?;
        // FALSIFY-PARSE-001 / PMAT-822: data_offset + offset (u64) and
        // start + size (usize) can both wrap for a crafted header, letting a
        // wrapped `end <= len` check pass over an OOB region. Use checked
        // arithmetic + `slice::get` so any overflow / past-EOF range → None.
        let abs_offset = self.header.data_offset.checked_add(entry.offset)?;
        let start = usize::try_from(abs_offset).ok()?;
        let end = start.checked_add(usize::try_from(entry.size).ok()?)?;
        self.data.get(start..end)
    }

    /// Get tensor as f32 Vec (copies data from mmap to `Vec<f32>`)
    ///
    /// Note: This allocates memory for the f32 values. For very large tensors,
    /// consider using `get_tensor_data` and processing in chunks.
    #[must_use]
    pub fn get_f32_tensor(&self, name: &str) -> Option<Vec<f32>> {
        let entry = self.get_tensor(name)?;
        if entry.dtype != TensorDType::F32 {
            return None;
        }

        let data = self.get_tensor_data(name)?;
        let floats: Vec<f32> = data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Some(floats)
    }

    /// Get tensor as f32 Vec, dequantizing if necessary
    ///
    /// Supports all tensor types:
    /// - F32: direct copy
    /// - F16: IEEE 754 half-precision → f32
    /// - Q8: 8-bit symmetric dequantization
    /// - Q4: 4-bit block dequantization
    /// - Q4K: GGUF Q4_K super-block dequantization (GH-200)
    /// - Q6K: GGUF Q6_K super-block dequantization (GH-200)
    #[must_use]
    pub fn get_tensor_as_f32(&self, name: &str) -> Option<Vec<f32>> {
        let entry = self.get_tensor(name)?;
        let data = self.get_tensor_data(name)?;
        let element_count = entry.element_count();

        match entry.dtype {
            TensorDType::F32 => {
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Some(floats)
            }
            TensorDType::F16 => {
                let floats: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|chunk| f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
                    .collect();
                Some(floats)
            }
            TensorDType::AprQ8 => {
                if data.len() < 4 {
                    return None;
                }
                let scale = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                let floats: Vec<f32> = data[4..]
                    .iter()
                    .map(|&b| f32::from(b as i8) * scale)
                    .collect();
                Some(floats)
            }
            TensorDType::AprQ4 => Some(dequantize_q4(data, element_count)),
            TensorDType::BF16 => {
                let floats: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        f32::from_bits(u32::from(bits) << 16)
                    })
                    .collect();
                Some(floats)
            }
            TensorDType::Q4K => dequantize_q4_k(data, 0, element_count).ok(),
            TensorDType::Q6K => dequantize_q6_k(data, 0, element_count).ok(),
            _ => None, // Other types not yet supported
        }
    }

    /// Check if all tensors are 64-byte aligned
    #[must_use]
    pub fn verify_alignment(&self) -> bool {
        let data_offset = self.header.data_offset as usize;
        self.tensor_index
            .iter()
            .all(|e| is_aligned_64(data_offset + e.offset as usize))
    }
}

// ============================================================================
// Shard Manifest
// ============================================================================

/// Shard manifest for multi-file models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardManifest {
    /// Format version
    pub version: String,
    /// Total number of shards
    pub shard_count: usize,
    /// Total size in bytes
    pub total_size: u64,
    /// Total tensor count
    pub tensor_count: usize,
    /// Shard files
    pub shards: Vec<ShardInfo>,
    /// Tensor to shard mapping
    pub weight_map: HashMap<String, usize>,
}

/// Information about a single shard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardInfo {
    /// Shard filename
    pub filename: String,
    /// Shard index
    pub index: usize,
    /// Size in bytes
    pub size: u64,
    /// Tensor names in this shard
    pub tensors: Vec<String>,
}