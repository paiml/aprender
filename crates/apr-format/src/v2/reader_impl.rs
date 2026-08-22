//! v2 reader impls + shard manifest (issue #2231).
//!
//! Formerly `include!`d into `v2/mod.rs`; now a real module. The `AprV2Reader`
//! and `AprV2ReaderRef` struct declarations live here next to their `impl`s
//! (private fields).
//!
//! # Sovereignty seam (issue #2231)
//!
//! The dequantizing `get_tensor_as_f32` accessor (which needed the GGUF Q4_K /
//! Q6_K dequant kernels + the local f16-scaled `dequantize_q4`) is **severed**
//! from the leaf: it pulls quantization/physics that belongs to the framework.
//! The leaf exposes only the raw container bytes ([`AprV2Reader::get_tensor_data`])
//! and the trivial F32-only typed view ([`AprV2Reader::get_f32_tensor`]).
//! `aprender-core` re-attaches the dequantizing accessor as the `AprV2DequantExt`
//! extension trait.

use super::{
    is_aligned_64, AprV2Header, AprV2Metadata, TensorDType, TensorIndexEntry, V2FormatError,
    HEADER_SIZE_V2,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Read;

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
/// use apr_format::v2::AprV2ReaderRef;
///
/// let bytes: &[u8] = /* mmap'd slice */;
/// let reader = AprV2ReaderRef::from_bytes(bytes)?;
/// let weights = reader.get_f32_tensor("embed_tokens.weight")?;
/// ```
#[derive(Debug)]
pub struct AprV2ReaderRef<'a> {
    header: AprV2Header,
    metadata: AprV2Metadata,
    tensor_index: Vec<TensorIndexEntry>,
    data: &'a [u8],
}

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

/// The minimum file length the container's own header + tensor index imply
/// (issue #2612).
///
/// # The invariant
///
/// ```text
/// data_offset + max(entry.offset + align_64(entry.size)) <= file_length
/// ```
///
/// Every byte of every tensor the index declares must exist inside the file,
/// **and so must the 64-byte alignment padding that follows it** — both APR v2
/// writers (`AprV2Writer::write`, `AprV2StreamingWriter::add_tensor`) pad every
/// tensor unconditionally, the last one included. This is pure arithmetic over
/// the container's self-description: it reads no tensor data, so it costs
/// O(tensor_count) regardless of file size, and it holds for every APR v2
/// writer, because the on-disk order is header, metadata, index, data, footer —
/// the declared extent of a complete file is always bounded by EOF.
///
/// The GGUF path has enforced the same invariant since GH-707 / S1-FIX
/// (`Truncated GGUF: file is N bytes but tensor data starts at byte M`). The
/// APR path had no equivalent, and the asymmetry is exactly why a `.apr`
/// truncated to 4.5% of its length validated clean: header, metadata and index
/// all live in FRONT of the data section, so every structure the reader
/// actually parses survives the truncation intact.
///
/// # The residual, stated precisely
///
/// Both writers append a **4-byte CRC32 footer** after the padded data section,
/// so the true length of a file they produced is `required_file_len(data) + 4`.
/// This function deliberately stops short of it, and the reason is a
/// measurement, not caution: of the ten parseable APR v2 files in the local
/// corpus, **two carry no footer at all** —
/// `~/models/qwen2.5-coder-1.5b-instruct-q4k.apr` and its `-q4k-v2` sibling are
/// each exactly `required_file_len` bytes long, four short of
/// `required_file_len + 4`. Requiring the footer would report both intact files
/// as truncated. So a file missing only its last 4 bytes still passes this
/// check; catching that needs check 4 (footer CRC32), which is still a declared
/// `Skip("Footer not implemented")` stub — and which cannot simply be switched
/// on for the same reason those two files just demonstrated.
///
/// Including the padding, verified against the same corpus, removes the rest of
/// the tail slack: the unpadded bound left up to 63 further bytes undetected
/// (measured 60 on `whisper.apr/models/tiny-int8.apr`, 36 on the in-tree
/// `tests/fixtures/golden_v2.apr`), and the padded bound is `<= file_length` on
/// all ten.
///
/// # Errors
///
/// Returns [`V2FormatError`] when `data` is not an APR v2 container at all (too
/// short for the header, or wrong magic) — the caller cannot conclude anything
/// about truncation in that case — or when the tensor index itself cannot be
/// parsed, which IS evidence of a damaged file and should be reported as such.
pub fn required_file_len(data: &[u8]) -> Result<u64, V2FormatError> {
    // 64-byte alignment, in u64 so a u64 tensor size never round-trips through
    // usize (32-bit targets) on its way to the comparison.
    const ALIGN: u64 = 64;

    let header = AprV2Header::from_bytes(data)?;
    let tensor_index =
        parse_tensor_index_section(data, header.tensor_index_offset, header.tensor_count)?;

    let mut required = header.data_offset;
    for entry in &tensor_index {
        let overflow = || {
            V2FormatError::InvalidTensorIndex(format!(
                "tensor '{}' extent overflows u64 (offset {}, size {})",
                entry.name, entry.offset, entry.size
            ))
        };
        let padded_size = entry
            .size
            .checked_add(ALIGN - 1)
            .map(|v| v & !(ALIGN - 1))
            .ok_or_else(overflow)?;
        let end = header
            .data_offset
            .checked_add(entry.offset)
            .and_then(|start| start.checked_add(padded_size))
            .ok_or_else(overflow)?;
        required = required.max(end);
    }
    Ok(required)
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

    // NOTE (issue #2231): `get_tensor_as_f32` (the dequantizing accessor) is
    // SEVERED from the sovereign leaf — it needed the GGUF Q4_K/Q6_K dequant
    // kernels + the f16-scaled `dequantize_q4`, all framework/quant concerns.
    // `aprender-core` re-attaches it via the `AprV2DequantExt` extension trait.

    /// Check if all tensors are 64-byte aligned
    #[must_use]
    pub fn verify_alignment(&self) -> bool {
        let data_offset = self.header.data_offset as usize;
        self.tensor_index
            .iter()
            .all(|e| is_aligned_64(data_offset + e.offset as usize))
    }

    /// Borrow the parsed tensor index (used by the core dequant extension).
    #[must_use]
    pub fn tensor_index(&self) -> &[TensorIndexEntry] {
        &self.tensor_index
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

    // NOTE (issue #2231): `get_tensor_as_f32` severed from the leaf — see the
    // owning-reader note above. Re-attached in core via `AprV2DequantExt`.

    /// Check if all tensors are 64-byte aligned
    #[must_use]
    pub fn verify_alignment(&self) -> bool {
        let data_offset = self.header.data_offset as usize;
        self.tensor_index
            .iter()
            .all(|e| is_aligned_64(data_offset + e.offset as usize))
    }

    /// Borrow the parsed tensor index (used by the core dequant extension).
    #[must_use]
    pub fn tensor_index(&self) -> &[TensorIndexEntry] {
        &self.tensor_index
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
