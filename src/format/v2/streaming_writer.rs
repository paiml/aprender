/// Streaming APR v2 writer — writes tensors to disk incrementally (realizar#136).
///
/// Unlike `AprV2Writer` which accumulates all tensor data in RAM,
/// this writer streams tensor data to a temp file, keeping only the
/// index entries (~KB) in memory. Peak RAM = largest single tensor.
///
/// # Architecture
///
/// 1. Tensor data written to temp file in insertion order, 64B aligned
/// 2. Index entries (name, dtype, shape, offset, size) accumulated in Vec (~KB)
/// 3. `finalize()` writes: header + metadata + index, then copies data from temp file
///
/// Index entries are sorted by name during `finalize()` (APR v2 contract).
/// Data in the temp file stays in insertion order; index offsets point correctly.

impl AprV2StreamingWriter {
    /// Create a new streaming writer.
    ///
    /// # Errors
    ///
    /// Returns error if the temp file cannot be created.
    pub fn new(metadata: AprV2Metadata) -> Result<Self, V2FormatError> {
        let mut header = AprV2Header::new();
        header.flags = header.flags.with(AprV2Flags::LAYOUT_ROW_MAJOR);

        let data_file = tempfile::tempfile()
            .map_err(|e| V2FormatError::IoError(format!("Failed to create temp file: {e}")))?;

        Ok(Self {
            header,
            metadata,
            index_entries: Vec::new(),
            data_writer: std::io::BufWriter::new(data_file),
            data_offset: 0,
        })
    }

    /// Add a tensor, writing its data to the temp file immediately.
    ///
    /// Only the index entry (~100 bytes) is kept in memory.
    /// The `data` slice can be dropped after this call returns.
    ///
    /// # Errors
    ///
    /// Returns error if writing to the temp file fails.
    pub fn add_tensor(
        &mut self,
        name: impl Into<String>,
        dtype: TensorDType,
        shape: Vec<usize>,
        data: &[u8],
    ) -> Result<(), V2FormatError> {
        let entry = TensorIndexEntry::new(name, dtype, shape, self.data_offset, data.len() as u64);
        self.index_entries.push(entry);

        // Write data + 64-byte alignment padding
        self.data_writer
            .write_all(data)
            .map_err(|e| V2FormatError::IoError(e.to_string()))?;

        let padded_size = align_64(data.len());
        let padding = padded_size - data.len();
        if padding > 0 {
            self.data_writer
                .write_all(&vec![0u8; padding])
                .map_err(|e| V2FormatError::IoError(e.to_string()))?;
        }

        self.data_offset += padded_size as u64;
        Ok(())
    }

    /// Add f32 tensor (streaming).
    ///
    /// # Errors
    ///
    /// Returns error if writing fails.
    pub fn add_f32_tensor(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        data: &[f32],
    ) -> Result<(), V2FormatError> {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.add_tensor(name, TensorDType::F32, shape, &bytes)
    }

    /// Add raw BF16/F16 bytes directly (zero conversion, streaming).
    ///
    /// # Errors
    ///
    /// Returns error if writing fails.
    pub fn add_raw_f16_tensor(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        data: &[u8],
        is_bf16: bool,
    ) -> Result<(), V2FormatError> {
        let dtype = if is_bf16 {
            TensorDType::BF16
        } else {
            TensorDType::F16
        };
        self.add_tensor(name, dtype, shape, data)
    }

    /// Number of tensors added so far.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.index_entries.len()
    }

    /// Total bytes of tensor data written to temp file.
    #[must_use]
    pub fn data_bytes_written(&self) -> u64 {
        self.data_offset
    }

    /// Finalize and write the complete APR v2 file.
    ///
    /// Writes header + metadata + tensor index + tensor data (streamed from temp file).
    /// The temp file is consumed and deleted automatically.
    ///
    /// # Errors
    ///
    /// Returns error if assembly or writing fails.
    pub fn finalize(mut self, output_path: &std::path::Path) -> Result<(), V2FormatError> {
        use std::io::{BufWriter, Seek, SeekFrom};

        // Serialize metadata
        let metadata_bytes = self.metadata.to_json()?;
        let metadata_padded_size = align_64(metadata_bytes.len());

        // Sort index entries by name (APR v2 contract — readers enforce sorted order).
        // Offsets are preserved from insertion time — they point to correct data positions
        // in the temp file regardless of index order.
        self.index_entries.sort_by(|a, b| a.name.cmp(&b.name));

        // Build tensor index bytes
        let mut tensor_index_bytes = Vec::new();
        for entry in &self.index_entries {
            tensor_index_bytes.extend_from_slice(&entry.to_bytes());
        }
        let tensor_index_padded_size = align_64(tensor_index_bytes.len());

        // Calculate section offsets
        let metadata_offset = HEADER_SIZE_V2;
        let tensor_index_offset = metadata_offset + metadata_padded_size;
        let data_section_offset = tensor_index_offset + tensor_index_padded_size;

        // Update header
        self.header.tensor_count = self.index_entries.len() as u32;
        self.header.metadata_offset = metadata_offset as u64;
        self.header.metadata_size = metadata_bytes.len() as u32;
        self.header.tensor_index_offset = tensor_index_offset as u64;
        self.header.data_offset = data_section_offset as u64;
        self.header.update_checksum();

        // Flush and rewind temp data file
        self.data_writer
            .flush()
            .map_err(|e| V2FormatError::IoError(e.to_string()))?;
        let mut data_file = self
            .data_writer
            .into_inner()
            .map_err(|e| V2FormatError::IoError(e.to_string()))?;
        data_file
            .seek(SeekFrom::Start(0))
            .map_err(|e| V2FormatError::IoError(e.to_string()))?;

        // Write output file
        let out_file = std::fs::File::create(output_path).map_err(|e| {
            V2FormatError::IoError(format!(
                "Failed to create {}: {e}",
                output_path.display()
            ))
        })?;
        let mut out = BufWriter::new(out_file);

        // Header
        out.write_all(&self.header.to_bytes())
            .map_err(|e| V2FormatError::IoError(e.to_string()))?;

        // Metadata (padded)
        out.write_all(&metadata_bytes)
            .map_err(|e| V2FormatError::IoError(e.to_string()))?;
        let metadata_pad = metadata_padded_size - metadata_bytes.len();
        if metadata_pad > 0 {
            out.write_all(&vec![0u8; metadata_pad])
                .map_err(|e| V2FormatError::IoError(e.to_string()))?;
        }

        // Tensor index (padded)
        out.write_all(&tensor_index_bytes)
            .map_err(|e| V2FormatError::IoError(e.to_string()))?;
        let index_pad = tensor_index_padded_size - tensor_index_bytes.len();
        if index_pad > 0 {
            out.write_all(&vec![0u8; index_pad])
                .map_err(|e| V2FormatError::IoError(e.to_string()))?;
        }

        // Tensor data — stream from temp file in 256KB chunks
        let mut buf = vec![0u8; 256 * 1024];
        loop {
            let n = data_file
                .read(&mut buf)
                .map_err(|e| V2FormatError::IoError(e.to_string()))?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])
                .map_err(|e| V2FormatError::IoError(e.to_string()))?;
        }

        // Footer checksum — flush, re-read, append CRC32
        out.flush()
            .map_err(|e| V2FormatError::IoError(e.to_string()))?;
        drop(out);

        let footer_crc = streaming_crc32_file(output_path)?;

        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(output_path)
            .map_err(|e| V2FormatError::IoError(e.to_string()))?;
        file.write_all(&footer_crc.to_le_bytes())
            .map_err(|e| V2FormatError::IoError(e.to_string()))?;

        Ok(())
    }
}

/// CRC32 over a file, read in 256KB chunks. Same polynomial as `crc32()`.
fn streaming_crc32_file(path: &std::path::Path) -> Result<u32, V2FormatError> {
    const TABLE: [u32; 256] = {
        let mut table = [0u32; 256];
        let mut i = 0;
        while i < 256 {
            let mut c = i as u32;
            let mut j = 0;
            while j < 8 {
                if c & 1 != 0 {
                    c = (c >> 1) ^ 0xEDB8_8320;
                } else {
                    c >>= 1;
                }
                j += 1;
            }
            table[i] = c;
            i += 1;
        }
        table
    };

    let mut file =
        std::fs::File::open(path).map_err(|e| V2FormatError::IoError(e.to_string()))?;
    let mut crc = 0xFFFF_FFFF_u32;
    let mut buf = vec![0u8; 256 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| V2FormatError::IoError(e.to_string()))?;
        if n == 0 {
            break;
        }
        for &byte in &buf[..n] {
            let idx = ((crc ^ u32::from(byte)) & 0xFF) as usize;
            crc = (crc >> 8) ^ TABLE[idx];
        }
    }
    Ok(!crc)
}
