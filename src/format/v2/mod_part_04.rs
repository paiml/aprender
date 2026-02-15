
impl AprV2Writer {
    /// Create new writer
    ///
    /// LAYOUT-002: All new APR files are created with LAYOUT_ROW_MAJOR flag set.
    /// This ensures realizar can safely assume row-major layout for all tensors.
    #[must_use]
    pub fn new(metadata: AprV2Metadata) -> Self {
        let mut header = AprV2Header::new();
        // LAYOUT-002: Mark all new APR files as row-major
        header.flags = header.flags.with(AprV2Flags::LAYOUT_ROW_MAJOR);
        Self {
            header,
            metadata,
            tensors: Vec::new(),
        }
    }

    /// Add tensor to the file
    pub fn add_tensor(
        &mut self,
        name: impl Into<String>,
        dtype: TensorDType,
        shape: Vec<usize>,
        data: Vec<u8>,
    ) {
        let entry = TensorIndexEntry::new(name, dtype, shape, 0, data.len() as u64);
        self.tensors.push((entry, data));
    }

    /// Add f32 tensor
    pub fn add_f32_tensor(&mut self, name: impl Into<String>, shape: Vec<usize>, data: &[f32]) {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.add_tensor(name, TensorDType::F32, shape, bytes);
    }

    /// Add f16 tensor (converts f32 → f16, 2 bytes per value)
    ///
    /// This provides true 2x compression over f32 storage with minimal precision loss
    /// for inference workloads. Uses IEEE 754 half-precision format.
    pub fn add_f16_tensor(&mut self, name: impl Into<String>, shape: Vec<usize>, data: &[f32]) {
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|&f| f32_to_f16(f).to_le_bytes())
            .collect();
        self.add_tensor(name, TensorDType::F16, shape, bytes);
    }

    /// Add Q8 tensor (8-bit symmetric quantization)
    ///
    /// Format: [scale: f32 (4 bytes)] + [quantized: i8 × n]
    /// Total size: 4 + n bytes (vs 4n for f32)
    /// Compression ratio: ~4x
    pub fn add_q8_tensor(&mut self, name: impl Into<String>, shape: Vec<usize>, data: &[f32]) {
        let name = name.into();
        if data.is_empty() {
            self.add_tensor(name, TensorDType::Q8, shape, Vec::new());
            return;
        }

        // Find scale (max absolute value)
        let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };

        // Pack: scale (4 bytes) + quantized values (1 byte each)
        let mut bytes = Vec::with_capacity(4 + data.len());
        bytes.extend_from_slice(&scale.to_le_bytes());

        for &v in data {
            let q = (v / scale).round().clamp(-127.0, 127.0) as i8;
            bytes.push(q as u8);
        }

        // CONTRACT: Q8 byte count must be scale(4) + element_count(N)
        let element_count: usize = shape.iter().product();
        assert_eq!(
            bytes.len(),
            4 + element_count,
            "Q8 CONTRACT VIOLATION: tensor '{}' packed {} bytes, expected {} (4 + {})",
            name,
            bytes.len(),
            4 + element_count,
            element_count
        );

        // CONTRACT: dequantized data must not be >99% zeros (F-DATA-QUALITY-001)
        // Catches packing bugs that produce all-zeros at write time, not read time.
        // Threshold is 99% (not 80%) because Q4K→F32→Q8 re-quantization legitimately
        // produces 80-95% zeros from dequantized sparse data. True packing bugs produce
        // ~100% zeros. Only enforced for large tensors (≥1024 elements).
        #[allow(clippy::naive_bytecount)] // No bytecount crate dependency
        if element_count >= 1024 {
            let zero_count = bytes[4..].iter().filter(|&&b| b == 0).count();
            let zero_pct = zero_count as f64 / element_count as f64;
            assert!(
                zero_pct <= 0.99,
                "Q8 DENSITY VIOLATION: tensor '{}' has {:.1}% zeros (threshold 99%)",
                name,
                zero_pct * 100.0
            );
        }

        self.add_tensor(name, TensorDType::Q8, shape, bytes);
    }

    /// Add Q4 tensor (4-bit symmetric quantization, block-wise)
    ///
    /// Format: For each block of 32 values:
    ///   [block_scale: f16 (2 bytes)] + [packed nibbles: 16 bytes]
    ///
    /// Total size per block: 18 bytes (vs 128 bytes for f32)
    /// Compression ratio: ~7x
    pub fn add_q4_tensor(&mut self, name: impl Into<String>, shape: Vec<usize>, data: &[f32]) {
        const BLOCK_SIZE: usize = 32;

        let name = name.into();
        if data.is_empty() {
            self.add_tensor(name, TensorDType::Q4, shape, Vec::new());
            return;
        }

        // Blocks: each block has 2-byte scale + 16 bytes of packed nibbles
        let num_blocks = (data.len() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let mut bytes = Vec::with_capacity(num_blocks * 18);

        for block_start in (0..data.len()).step_by(BLOCK_SIZE) {
            let block_end = (block_start + BLOCK_SIZE).min(data.len());
            let block = &data[block_start..block_end];

            // Find block scale
            let max_abs = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };

            // Store scale as f16
            bytes.extend_from_slice(&f32_to_f16(scale).to_le_bytes());

            // Quantize and pack (2 values per byte)
            let mut packed_idx = 0;
            let mut packed_buf = [0u8; 16];

            for (i, &v) in block.iter().enumerate() {
                // Quantize to 4-bit signed (-8 to 7)
                let q = (v / scale).round().clamp(-8.0, 7.0) as i8;
                // Store as unsigned nibble (0-15)
                let nibble = ((q + 8) as u8) & 0x0F;

                if i % 2 == 0 {
                    packed_buf[packed_idx] = nibble;
                } else {
                    packed_buf[packed_idx] |= nibble << 4;
                    packed_idx += 1;
                }
            }
            // Note: No need to track packed_idx for odd elements since we write all 16 bytes anyway

            // Write all 16 bytes (zero-padded for partial blocks)
            bytes.extend_from_slice(&packed_buf);
        }

        // CONTRACT: Q4 byte count must be num_blocks * 18
        let element_count: usize = shape.iter().product();
        let expected_blocks = (element_count + 31) / 32;
        assert_eq!(
            bytes.len(),
            expected_blocks * 18,
            "Q4 CONTRACT VIOLATION: tensor '{}' packed {} bytes, expected {} ({} blocks * 18)",
            name,
            bytes.len(),
            expected_blocks * 18,
            expected_blocks
        );

        // CONTRACT: dequantized data must not be >99% zeros (F-DATA-QUALITY-001)
        // For Q4, nibble value 8 (0x08) represents zero (signed 0 = unsigned 8).
        // Threshold is 99% — same rationale as Q8 density check.
        // Only enforced for large tensors (≥1024 elements).
        if element_count >= 1024 {
            let mut zero_nibbles = 0usize;
            let mut total_nibbles = 0usize;
            for block_idx in 0..num_blocks {
                let block_offset = block_idx * 18 + 2; // skip 2-byte scale
                let block_elem_count =
                    BLOCK_SIZE.min(element_count.saturating_sub(block_idx * BLOCK_SIZE));
                for i in 0..block_elem_count {
                    let byte = bytes[block_offset + i / 2];
                    let nibble = if i % 2 == 0 {
                        byte & 0x0F
                    } else {
                        (byte >> 4) & 0x0F
                    };
                    if nibble == 8 {
                        zero_nibbles += 1;
                    }
                    total_nibbles += 1;
                }
            }
            if total_nibbles > 0 {
                let zero_pct = zero_nibbles as f64 / total_nibbles as f64;
                assert!(
                    zero_pct <= 0.99,
                    "Q4 DENSITY VIOLATION: tensor '{}' has {:.1}% zeros (threshold 99%)",
                    name,
                    zero_pct * 100.0
                );
            }
        }

        self.add_tensor(name, TensorDType::Q4, shape, bytes);
    }

    /// Add raw Q4_K tensor (GGUF-compatible super-block format)
    ///
    /// This stores GGUF Q4_K data directly without re-quantization.
    /// Q4_K format: 256-element super-blocks with nested 32-element sub-blocks
    /// Each super-block: d (f16, 2B) + dmin (f16, 2B) + scales (12B) + qs (128B) = 144 bytes
    /// Effective bits per weight: ~4.5
    ///
    /// Use this when importing from GGUF to preserve exact quantization.
    pub fn add_q4k_raw_tensor(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        raw_data: Vec<u8>,
    ) {
        self.add_tensor(name, TensorDType::Q4K, shape, raw_data);
    }

    /// Add raw Q6_K tensor (GGUF-compatible super-block format)
    ///
    /// This stores GGUF Q6_K data directly without re-quantization.
    /// Q6_K format: 256-element super-blocks
    /// Each super-block: ql (128B) + qh (64B) + scales (16B) + d (f16, 2B) = 210 bytes
    /// Effective bits per weight: ~6.5
    pub fn add_q6k_raw_tensor(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        raw_data: Vec<u8>,
    ) {
        self.add_tensor(name, TensorDType::Q6K, shape, raw_data);
    }

    /// Set LZ4 compression flag
    pub fn with_lz4_compression(&mut self) -> &mut Self {
        self.header.flags = self.header.flags.with(AprV2Flags::LZ4_COMPRESSED);
        self
    }

    /// Set sharding info
    pub fn with_sharding(&mut self, shard_count: usize, shard_index: usize) -> &mut Self {
        self.header.flags = self.header.flags.with(AprV2Flags::SHARDED);
        self.metadata.sharding = Some(ShardingMetadata {
            shard_count,
            shard_index,
            total_size: 0,
            pattern: None,
        });
        self
    }

    /// Write to bytes
    ///
    /// # Errors
    /// Returns error if serialization fails.
    pub fn write(&mut self) -> Result<Vec<u8>, V2FormatError> {
        // Sort tensors by name
        self.tensors.sort_by(|a, b| a.0.name.cmp(&b.0.name));

        // Serialize metadata
        let metadata_bytes = self.metadata.to_json()?;
        let metadata_padded_size = align_64(metadata_bytes.len());

        // Build tensor index
        let mut tensor_index_bytes = Vec::new();
        let mut data_offset = 0_u64;

        for (entry, data) in &mut self.tensors {
            entry.offset = data_offset;
            entry.size = data.len() as u64;
            tensor_index_bytes.extend_from_slice(&entry.to_bytes());
            data_offset += align_64(data.len()) as u64;
        }
        let tensor_index_padded_size = align_64(tensor_index_bytes.len());

        // Calculate offsets
        let metadata_offset = HEADER_SIZE_V2;
        let tensor_index_offset = metadata_offset + metadata_padded_size;
        let data_section_offset = tensor_index_offset + tensor_index_padded_size;

        // Update header
        self.header.tensor_count = self.tensors.len() as u32;
        self.header.metadata_offset = metadata_offset as u64;
        self.header.metadata_size = metadata_bytes.len() as u32;
        self.header.tensor_index_offset = tensor_index_offset as u64;
        self.header.data_offset = data_section_offset as u64;
        self.header.update_checksum();

        // Build output
        let mut output = Vec::new();

        // Header
        output.extend_from_slice(&self.header.to_bytes());

        // Metadata (padded)
        output.extend_from_slice(&metadata_bytes);
        output.resize(metadata_offset + metadata_padded_size, 0);

        // Tensor index (padded)
        output.extend_from_slice(&tensor_index_bytes);
        output.resize(tensor_index_offset + tensor_index_padded_size, 0);

        // Tensor data (each 64-byte aligned)
        for (_, data) in &self.tensors {
            let start = output.len();
            output.extend_from_slice(data);
            let padded_size = align_64(data.len());
            output.resize(start + padded_size, 0);
        }

        // Footer checksum
        let footer_checksum = crc32(&output);
        output.extend_from_slice(&footer_checksum.to_le_bytes());

        Ok(output)
    }

    /// Write to a Write impl
    ///
    /// # Errors
    /// Returns error if write fails.
    pub fn write_to<W: Write>(&mut self, writer: &mut W) -> Result<(), V2FormatError> {
        let bytes = self.write()?;
        writer
            .write_all(&bytes)
            .map_err(|e| V2FormatError::IoError(e.to_string()))
    }
}

// ============================================================================
// Reader
// ============================================================================

/// APR v2 format reader (owns data - copies input)
#[derive(Debug)]
pub struct AprV2Reader {
    header: AprV2Header,
    metadata: AprV2Metadata,
    tensor_index: Vec<TensorIndexEntry>,
    data: Vec<u8>,
}

/// APR v2 format reader with zero-copy (borrows data - for mmap)
///
/// This reader borrows the data slice instead of copying it, enabling
/// true zero-copy access when used with memory-mapped files.
///
/// # Example
///
/// ```ignore
/// use aprender::bundle::MappedFile;
/// use aprender::format::v2::AprV2ReaderRef;
///
/// let mmap = MappedFile::open("model.apr")?;
/// let reader = AprV2ReaderRef::from_bytes(mmap.as_slice())?;
/// let weights = reader.get_f32_tensor("embed_tokens.weight")?;
/// ```
#[derive(Debug)]
pub struct AprV2ReaderRef<'a> {
    header: AprV2Header,
    metadata: AprV2Metadata,
    tensor_index: Vec<TensorIndexEntry>,
    data: &'a [u8],
}
