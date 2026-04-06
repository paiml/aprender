//! Streaming dataset format with lazy chunk loading
//!
//! Enables lazy loading of large datasets (>100MB) by reading only
//! the header and chunk index initially, then loading chunks on demand.
//!
//! # Example
//!
//! ```ignore
//! use alimentar::format::streaming::StreamingDataset;
//!
//! // Open dataset (only reads header + index)
//! let dataset = StreamingDataset::open("large_data.ald")?;
//!
//! // Access chunks lazily
//! println!("Total rows: {}", dataset.num_rows());
//! for chunk in dataset.chunks() {
//!     println!("Chunk with {} rows", chunk?.num_rows());
//! }
//! ```

// Allow casts where truncation is intentional for file format sizes
#![allow(clippy::cast_possible_truncation)]

use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::{array::RecordBatch, datatypes::SchemaRef};
use serde::{Deserialize, Serialize};

use crate::{
    error::{Error, Result},
    format::{flags, Compression, HEADER_SIZE, MAGIC},
};

/// Default chunk size in rows for streaming format
pub const DEFAULT_CHUNK_SIZE: usize = 65536; // 64K rows per chunk

/// Minimum dataset size to recommend streaming (100MB)
pub const STREAMING_THRESHOLD: u64 = 100 * 1024 * 1024;

/// Entry in the chunk index describing one chunk's location
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkEntry {
    /// Row offset (first row in this chunk)
    pub row_offset: u64,
    /// Number of rows in this chunk
    pub num_rows: u32,
    /// Byte offset in the payload section
    pub byte_offset: u64,
    /// Compressed size in bytes
    pub compressed_size: u32,
    /// Uncompressed size in bytes
    pub uncompressed_size: u32,
}

impl ChunkEntry {
    /// Create a new chunk entry
    pub fn new(
        row_offset: u64,
        num_rows: u32,
        byte_offset: u64,
        compressed_size: u32,
        uncompressed_size: u32,
    ) -> Self {
        Self {
            row_offset,
            num_rows,
            byte_offset,
            compressed_size,
            uncompressed_size,
        }
    }

    /// Check if this chunk contains the given row
    pub fn contains_row(&self, row: u64) -> bool {
        row >= self.row_offset && row < self.row_offset + u64::from(self.num_rows)
    }

    /// Get the last row index in this chunk (exclusive)
    pub fn end_row(&self) -> u64 {
        self.row_offset + u64::from(self.num_rows)
    }
}

/// Index of all chunks in a streaming dataset
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChunkIndex {
    /// Chunk entries in order
    entries: Vec<ChunkEntry>,
    /// Total row count (cached)
    total_rows: u64,
}

impl ChunkIndex {
    /// Create a new empty chunk index
    pub fn new() -> Self {
        Self::default()
    }

    /// Create chunk index from entries
    pub fn from_entries(entries: Vec<ChunkEntry>) -> Self {
        let total_rows = entries.last().map_or(0, ChunkEntry::end_row);
        Self {
            entries,
            total_rows,
        }
    }

    /// Add a chunk entry
    pub fn push(&mut self, entry: ChunkEntry) {
        self.total_rows = entry.end_row();
        self.entries.push(entry);
    }

    /// Get number of chunks
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get total row count
    pub fn total_rows(&self) -> u64 {
        self.total_rows
    }

    /// Get chunk entry by index
    pub fn get(&self, index: usize) -> Option<&ChunkEntry> {
        self.entries.get(index)
    }

    /// Find chunk containing the given row
    pub fn find_chunk_for_row(&self, row: u64) -> Option<usize> {
        if row >= self.total_rows {
            return None;
        }
        // Binary search for efficiency
        self.entries
            .binary_search_by(|entry| {
                if row < entry.row_offset {
                    std::cmp::Ordering::Greater
                } else if row >= entry.end_row() {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .ok()
    }

    /// Iterate over entries
    pub fn iter(&self) -> impl Iterator<Item = &ChunkEntry> {
        self.entries.iter()
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        rmp_serde::to_vec(self)
            .map_err(|e| Error::Format(format!("Failed to serialize chunk index: {e}")))
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        rmp_serde::from_slice(bytes)
            .map_err(|e| Error::Format(format!("Failed to deserialize chunk index: {e}")))
    }
}

/// Streaming dataset with lazy chunk loading
///
/// Uses file seeking to load chunks on demand, minimizing memory usage.
pub struct StreamingDataset {
    /// Path to the dataset file
    path: PathBuf,
    /// Chunk index
    index: ChunkIndex,
    /// Arrow schema
    schema: SchemaRef,
    /// Compression type used
    compression: Compression,
    /// Offset where payload starts
    payload_offset: u64,
}

impl StreamingDataset {
    /// Open a streaming dataset from a file
    ///
    /// Only reads the header, metadata, and chunk index initially.
    /// Chunks are loaded on demand.
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be opened, is not a valid .ald file,
    /// or does not have the STREAMING flag set.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|e| Error::io(e, path))?;
        let mut reader = BufReader::new(file);

        // Read header
        let mut header = [0u8; HEADER_SIZE];
        reader
            .read_exact(&mut header)
            .map_err(|e| Error::io(e, path))?;

        // Verify magic bytes
        if header[0..4] != MAGIC {
            return Err(Error::Format("Invalid magic bytes".into()));
        }

        // Check streaming flag
        let header_flags = header[6];
        if header_flags & flags::STREAMING == 0 {
            return Err(Error::Format(
                "File does not have STREAMING flag set. Use regular load() instead.".into(),
            ));
        }

        // Read compression type
        let compression = Compression::from_u8(header[7])
            .ok_or_else(|| Error::Format(format!("Unknown compression type: {}", header[7])))?;

        // Read section sizes from header
        let metadata_size = u64::from(u32::from_le_bytes([
            header[12], header[13], header[14], header[15],
        ]));
        let schema_size = u64::from(u32::from_le_bytes([
            header[16], header[17], header[18], header[19],
        ]));
        let index_size = u64::from(u32::from_le_bytes([
            header[20], header[21], header[22], header[23],
        ]));

        // Calculate offsets
        let schema_offset = u64::from(HEADER_SIZE as u32) + metadata_size;
        let index_offset = schema_offset + schema_size;
        let payload_offset = index_offset + index_size;

        // Read schema
        reader
            .seek(SeekFrom::Start(schema_offset))
            .map_err(|e| Error::io(e, path))?;
        let mut schema_bytes = vec![0u8; schema_size as usize];
        reader
            .read_exact(&mut schema_bytes)
            .map_err(|e| Error::io(e, path))?;
        let schema = Self::deserialize_schema(&schema_bytes)?;

        // Read chunk index
        let mut index_bytes = vec![0u8; index_size as usize];
        reader
            .read_exact(&mut index_bytes)
            .map_err(|e| Error::io(e, path))?;
        let index = ChunkIndex::from_bytes(&index_bytes)?;

        Ok(Self {
            path: path.to_path_buf(),
            index,
            schema,
            compression,
            payload_offset,
        })
    }

    /// Get total row count
    pub fn num_rows(&self) -> u64 {
        self.index.total_rows()
    }

    /// Get number of chunks
    pub fn num_chunks(&self) -> usize {
        self.index.len()
    }

    /// Get the schema
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    /// Load a specific chunk by index
    ///
    /// # Errors
    ///
    /// Returns error if chunk index is out of bounds or decompression fails.
    pub fn get_chunk(&self, chunk_idx: usize) -> Result<RecordBatch> {
        let entry = self
            .index
            .get(chunk_idx)
            .ok_or_else(|| Error::IndexOutOfBounds {
                index: chunk_idx,
                len: self.index.len(),
            })?;

        self.load_chunk(entry)
    }

    /// Get random access to rows by range
    ///
    /// # Errors
    ///
    /// Returns error if row range is out of bounds.
    pub fn get_rows(&self, start: u64, count: u64) -> Result<RecordBatch> {
        if start >= self.num_rows() {
            return Err(Error::IndexOutOfBounds {
                index: start as usize,
                len: self.num_rows() as usize,
            });
        }

        let end = (start + count).min(self.num_rows());
        let actual_count = end - start;

        // Find chunks that contain these rows
        let start_chunk =
            self.index
                .find_chunk_for_row(start)
                .ok_or_else(|| Error::IndexOutOfBounds {
                    index: start as usize,
                    len: self.num_rows() as usize,
                })?;

        let end_chunk = self
            .index
            .find_chunk_for_row(end.saturating_sub(1))
            .unwrap_or(start_chunk);

        // Load and slice chunks
        let mut batches = Vec::new();
        let mut remaining = actual_count;
        let mut current_row = start;

        for chunk_idx in start_chunk..=end_chunk {
            let entry = self
                .index
                .get(chunk_idx)
                .ok_or_else(|| Error::IndexOutOfBounds {
                    index: chunk_idx,
                    len: self.index.len(),
                })?;

            let batch = self.load_chunk(entry)?;

            // Calculate slice within this chunk
            let chunk_start = if current_row > entry.row_offset {
                (current_row - entry.row_offset) as usize
            } else {
                0
            };

            let chunk_take = remaining.min(u64::from(entry.num_rows) - chunk_start as u64) as usize;

            let sliced = batch.slice(chunk_start, chunk_take);
            batches.push(sliced);

            remaining -= chunk_take as u64;
            current_row += chunk_take as u64;
        }

        // Concatenate batches if needed
        if batches.len() == 1 {
            Ok(batches
                .into_iter()
                .next()
                .ok_or_else(|| Error::Format("No batches loaded".into()))?)
        } else {
            use arrow::compute::concat_batches;
            concat_batches(&self.schema, &batches).map_err(Error::Arrow)
        }
    }

    /// Iterate over all chunks
    pub fn chunks(&self) -> ChunkIterator<'_> {
        ChunkIterator {
            dataset: self,
            current: 0,
        }
    }

    /// Load a chunk from the file
    fn load_chunk(&self, entry: &ChunkEntry) -> Result<RecordBatch> {
        let file = File::open(&self.path).map_err(|e| Error::io(e, &self.path))?;
        let mut reader = BufReader::new(file);

        let offset = self.payload_offset + entry.byte_offset;
        reader
            .seek(SeekFrom::Start(offset))
            .map_err(|e| Error::io(e, &self.path))?;

        let mut compressed_data = vec![0u8; entry.compressed_size as usize];
        reader
            .read_exact(&mut compressed_data)
            .map_err(|e| Error::io(e, &self.path))?;

        // Decompress
        let decompressed = self.decompress(&compressed_data, entry.uncompressed_size as usize)?;

        // Deserialize Arrow IPC
        Self::deserialize_batch(&decompressed)
    }

    /// Decompress data based on compression type
    fn decompress(&self, data: &[u8], expected_size: usize) -> Result<Vec<u8>> {
        match self.compression {
            Compression::None => Ok(data.to_vec()),
            Compression::ZstdL3 | Compression::ZstdL19 => {
                let mut output = Vec::with_capacity(expected_size);
                zstd::stream::copy_decode(data, &mut output)
                    .map_err(|e| Error::Format(format!("Zstd decompression failed: {e}")))?;
                Ok(output)
            }
            Compression::Lz4 => lz4_flex::decompress(data, expected_size)
                .map_err(|e| Error::Format(format!("LZ4 decompression failed: {e}"))),
        }
    }

    /// Deserialize Arrow schema from bytes
    fn deserialize_schema(bytes: &[u8]) -> Result<SchemaRef> {
        use std::io::Cursor;

        use arrow::ipc::reader::StreamReader;

        let cursor = Cursor::new(bytes);
        let reader = StreamReader::try_new(cursor, None).map_err(Error::Arrow)?;
        Ok(reader.schema())
    }

    /// Deserialize Arrow batch from bytes
    fn deserialize_batch(bytes: &[u8]) -> Result<RecordBatch> {
        use std::io::Cursor;

        use arrow::ipc::reader::StreamReader;

        let cursor = Cursor::new(bytes);
        let mut reader = StreamReader::try_new(cursor, None).map_err(Error::Arrow)?;

        reader
            .next()
            .ok_or_else(|| Error::Format("No batch in IPC data".into()))?
            .map_err(Error::Arrow)
    }
}

impl std::fmt::Debug for StreamingDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingDataset")
            .field("path", &self.path)
            .field("num_rows", &self.num_rows())
            .field("num_chunks", &self.num_chunks())
            .field("compression", &self.compression)
            .finish_non_exhaustive()
    }
}

/// Iterator over chunks in a streaming dataset
pub struct ChunkIterator<'a> {
    dataset: &'a StreamingDataset,
    current: usize,
}

impl Iterator for ChunkIterator<'_> {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.dataset.num_chunks() {
            return None;
        }

        let result = self.dataset.get_chunk(self.current);
        self.current += 1;
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.dataset.num_chunks() - self.current;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ChunkIterator<'_> {}

/// Save a dataset in streaming format
///
/// # Arguments
/// * `batches` - Iterator of record batches to save
/// * `schema` - Arrow schema
/// * `path` - Output file path
/// * `chunk_size` - Rows per chunk (default: 65536)
/// * `compression` - Compression type to use
///
/// # Errors
///
/// Returns error if file cannot be written.
pub fn save_streaming<P, I>(
    batches: I,
    schema: &SchemaRef,
    path: P,
    chunk_size: Option<usize>,
    compression: Compression,
) -> Result<()>
where
    P: AsRef<Path>,
    I: Iterator<Item = RecordBatch>,
{
    use std::io::{BufWriter, Write};

    let path = path.as_ref();
    let chunk_size = chunk_size.unwrap_or(DEFAULT_CHUNK_SIZE);

    let file = File::create(path).map_err(|e| Error::io(e, path))?;
    let mut writer = BufWriter::new(file);

    let chunks = collect_chunks(batches, schema, chunk_size, compression)?;

    // Build chunk index
    let index = ChunkIndex::from_entries(chunks.iter().map(|(e, _)| e.clone()).collect());
    let index_bytes = index.to_bytes()?;

    // Serialize schema
    let schema_bytes = serialize_schema(schema)?;

    // Build metadata (minimal for streaming)
    let metadata = crate::format::Metadata::default();
    let metadata_bytes = rmp_serde::to_vec(&metadata)
        .map_err(|e| Error::Format(format!("Failed to serialize metadata: {e}")))?;

    // Write header
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(&MAGIC);
    header[4] = crate::format::FORMAT_VERSION_MAJOR;
    header[5] = crate::format::FORMAT_VERSION_MINOR;
    header[6] = flags::STREAMING;
    header[7] = compression.as_u8();
    // Dataset type (Tabular default)
    header[8..10].copy_from_slice(&1u16.to_le_bytes());
    // Reserved
    header[10..12].copy_from_slice(&[0, 0]);
    // Section sizes
    header[12..16].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
    header[16..20].copy_from_slice(&(schema_bytes.len() as u32).to_le_bytes());
    header[20..24].copy_from_slice(&(index_bytes.len() as u32).to_le_bytes());
    // Payload size (sum of all chunks)
    let payload_size: u64 = chunks
        .iter()
        .map(|(e, _)| u64::from(e.compressed_size))
        .sum();
    header[24..32].copy_from_slice(&payload_size.to_le_bytes());

    writer.write_all(&header).map_err(|e| Error::io(e, path))?;
    writer
        .write_all(&metadata_bytes)
        .map_err(|e| Error::io(e, path))?;
    writer
        .write_all(&schema_bytes)
        .map_err(|e| Error::io(e, path))?;
    writer
        .write_all(&index_bytes)
        .map_err(|e| Error::io(e, path))?;

    // Write chunk data
    for (_, data) in &chunks {
        writer.write_all(data).map_err(|e| Error::io(e, path))?;
    }

    writer.flush().map_err(|e| Error::io(e, path))?;

    Ok(())
}

/// Collect all batches into compressed chunks
fn collect_chunks<I>(
    batches: I,
    schema: &SchemaRef,
    chunk_size: usize,
    compression: Compression,
) -> Result<Vec<(ChunkEntry, Vec<u8>)>>
where
    I: Iterator<Item = RecordBatch>,
{
    let mut chunks: Vec<(ChunkEntry, Vec<u8>)> = Vec::new();
    let mut current_rows: Vec<RecordBatch> = Vec::new();
    let mut current_row_count = 0usize;
    let mut total_row_offset = 0u64;
    let mut byte_offset = 0u64;

    for batch in batches {
        current_rows.push(batch.clone());
        current_row_count += batch.num_rows();

        while current_row_count >= chunk_size {
            let (chunk_batch, remaining) = split_batches(&current_rows, chunk_size, schema)?;
            let (entry, data) =
                build_chunk(&chunk_batch, total_row_offset, byte_offset, compression)?;

            total_row_offset += u64::from(entry.num_rows);
            byte_offset += u64::from(entry.compressed_size);

            chunks.push((entry, data));

            current_rows = remaining;
            current_row_count = current_rows.iter().map(RecordBatch::num_rows).sum();
        }
    }

    if !current_rows.is_empty() {
        let chunk_batch = concat_batches_vec(&current_rows, schema)?;
        let (entry, data) = build_chunk(&chunk_batch, total_row_offset, byte_offset, compression)?;
        chunks.push((entry, data));
    }

    Ok(chunks)
}

/// Build a chunk from a record batch
fn build_chunk(
    batch: &RecordBatch,
    row_offset: u64,
    byte_offset: u64,
    compression: Compression,
) -> Result<(ChunkEntry, Vec<u8>)> {
    // Serialize batch to Arrow IPC
    let uncompressed = serialize_batch(batch)?;
    let uncompressed_size = uncompressed.len();

    // Compress
    let compressed = match compression {
        Compression::None => uncompressed,
        Compression::ZstdL3 => zstd::encode_all(uncompressed.as_slice(), 3)
            .map_err(|e| Error::Format(format!("Zstd compression failed: {e}")))?,
        Compression::ZstdL19 => zstd::encode_all(uncompressed.as_slice(), 19)
            .map_err(|e| Error::Format(format!("Zstd compression failed: {e}")))?,
        Compression::Lz4 => lz4_flex::compress_prepend_size(&uncompressed),
    };

    let entry = ChunkEntry::new(
        row_offset,
        batch.num_rows() as u32,
        byte_offset,
        compressed.len() as u32,
        uncompressed_size as u32,
    );

    Ok((entry, compressed))
}

/// Serialize Arrow schema to bytes
fn serialize_schema(schema: &SchemaRef) -> Result<Vec<u8>> {
    use arrow::ipc::writer::StreamWriter;

    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, schema).map_err(Error::Arrow)?;
        writer.finish().map_err(Error::Arrow)?;
    }
    Ok(buf)
}

/// Serialize Arrow batch to bytes
fn serialize_batch(batch: &RecordBatch) -> Result<Vec<u8>> {
    use arrow::ipc::writer::StreamWriter;

    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &batch.schema()).map_err(Error::Arrow)?;
        writer.write(batch).map_err(Error::Arrow)?;
        writer.finish().map_err(Error::Arrow)?;
    }
    Ok(buf)
}

/// Split batches to get exactly chunk_size rows
fn split_batches(
    batches: &[RecordBatch],
    chunk_size: usize,
    schema: &SchemaRef,
) -> Result<(RecordBatch, Vec<RecordBatch>)> {
    use arrow::compute::concat_batches;

    // Concatenate all batches
    let combined = concat_batches(schema, batches).map_err(Error::Arrow)?;

    if combined.num_rows() <= chunk_size {
        return Ok((combined, Vec::new()));
    }

    // Split at chunk_size
    let chunk = combined.slice(0, chunk_size);
    let remaining = combined.slice(chunk_size, combined.num_rows() - chunk_size);

    Ok((chunk, vec![remaining]))
}

/// Concatenate batches into one
fn concat_batches_vec(batches: &[RecordBatch], schema: &SchemaRef) -> Result<RecordBatch> {
    use arrow::compute::concat_batches;
    concat_batches(schema, batches).map_err(Error::Arrow)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{Float64Array, Int32Array},
        datatypes::{DataType, Field, Schema},
    };
    use tempfile::NamedTempFile;

    use super::*;

    /// Helper to create a test batch with n rows
    fn make_test_batch(n: usize, offset: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Float64, false),
        ]));

        let ids: Vec<i32> = (offset..offset + n).map(|i| i as i32).collect();
        let values: Vec<f64> = (offset..offset + n).map(|i| i as f64 * 1.5).collect();

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(Float64Array::from(values)),
            ],
        )
        .expect("batch creation")
    }

    fn test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Float64, false),
        ]))
    }

    // ========== ChunkEntry tests ==========

    #[test]
    fn test_chunk_entry_new() {
        let entry = ChunkEntry::new(100, 50, 1000, 500, 800);

        assert_eq!(entry.row_offset, 100);
        assert_eq!(entry.num_rows, 50);
        assert_eq!(entry.byte_offset, 1000);
        assert_eq!(entry.compressed_size, 500);
        assert_eq!(entry.uncompressed_size, 800);
    }

    #[test]
    fn test_chunk_entry_contains_row() {
        let entry = ChunkEntry::new(100, 50, 0, 0, 0);

        assert!(!entry.contains_row(99));
        assert!(entry.contains_row(100));
        assert!(entry.contains_row(125));
        assert!(entry.contains_row(149));
        assert!(!entry.contains_row(150));
    }

    #[test]
    fn test_chunk_entry_end_row() {
        let entry = ChunkEntry::new(100, 50, 0, 0, 0);
        assert_eq!(entry.end_row(), 150);
    }

    // ========== ChunkIndex tests ==========

    #[test]
    fn test_chunk_index_new_empty() {
        let index = ChunkIndex::new();

        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        assert_eq!(index.total_rows(), 0);
    }

    #[test]
    fn test_chunk_index_push() {
        let mut index = ChunkIndex::new();

        index.push(ChunkEntry::new(0, 100, 0, 500, 800));
        assert_eq!(index.len(), 1);
        assert_eq!(index.total_rows(), 100);

        index.push(ChunkEntry::new(100, 100, 500, 500, 800));
        assert_eq!(index.len(), 2);
        assert_eq!(index.total_rows(), 200);
    }

    #[test]
    fn test_chunk_index_from_entries() {
        let entries = vec![
            ChunkEntry::new(0, 100, 0, 500, 800),
            ChunkEntry::new(100, 100, 500, 500, 800),
            ChunkEntry::new(200, 50, 1000, 250, 400),
        ];

        let index = ChunkIndex::from_entries(entries);

        assert_eq!(index.len(), 3);
        assert_eq!(index.total_rows(), 250);
    }

    #[test]
    fn test_chunk_index_get() {
        let entries = vec![
            ChunkEntry::new(0, 100, 0, 500, 800),
            ChunkEntry::new(100, 100, 500, 500, 800),
        ];
        let index = ChunkIndex::from_entries(entries);

        assert!(index.get(0).is_some());
        assert!(index.get(1).is_some());
        assert!(index.get(2).is_none());

        assert_eq!(index.get(0).map(|e| e.row_offset), Some(0));
        assert_eq!(index.get(1).map(|e| e.row_offset), Some(100));
    }

    #[test]
    fn test_chunk_index_find_chunk_for_row() {
        let entries = vec![
            ChunkEntry::new(0, 100, 0, 500, 800),
            ChunkEntry::new(100, 100, 500, 500, 800),
            ChunkEntry::new(200, 50, 1000, 250, 400),
        ];
        let index = ChunkIndex::from_entries(entries);

        assert_eq!(index.find_chunk_for_row(0), Some(0));
        assert_eq!(index.find_chunk_for_row(50), Some(0));
        assert_eq!(index.find_chunk_for_row(99), Some(0));
        assert_eq!(index.find_chunk_for_row(100), Some(1));
        assert_eq!(index.find_chunk_for_row(150), Some(1));
        assert_eq!(index.find_chunk_for_row(200), Some(2));
        assert_eq!(index.find_chunk_for_row(249), Some(2));
        assert_eq!(index.find_chunk_for_row(250), None);
        assert_eq!(index.find_chunk_for_row(1000), None);
    }

    #[test]
    fn test_chunk_index_serialization() {
        let entries = vec![
            ChunkEntry::new(0, 100, 0, 500, 800),
            ChunkEntry::new(100, 100, 500, 500, 800),
        ];
        let index = ChunkIndex::from_entries(entries);

        let bytes = index.to_bytes().expect("serialize");
        let restored = ChunkIndex::from_bytes(&bytes).expect("deserialize");

        assert_eq!(restored.len(), index.len());
        assert_eq!(restored.total_rows(), index.total_rows());
        assert_eq!(restored.get(0), index.get(0));
        assert_eq!(restored.get(1), index.get(1));
    }

    // ========== save_streaming tests ==========

    #[test]
    fn test_save_streaming_creates_file() {
        let batches = vec![make_test_batch(100, 0), make_test_batch(100, 100)];
        let schema = test_schema();

        let temp = NamedTempFile::new().expect("temp file");
        let path = temp.path();

        save_streaming(
            batches.into_iter(),
            &schema,
            path,
            Some(64),
            Compression::None,
        )
        .expect("save");

        assert!(path.exists());
        assert!(std::fs::metadata(path).expect("metadata").len() > 0);
    }

    #[test]
    fn test_save_streaming_with_compression() {
        let batches = vec![make_test_batch(1000, 0)];
        let schema = test_schema();

        let temp_none = NamedTempFile::new().expect("temp");
        let temp_zstd = NamedTempFile::new().expect("temp");

        save_streaming(
            batches.clone().into_iter(),
            &schema,
            temp_none.path(),
            Some(500),
            Compression::None,
        )
        .expect("save none");

        save_streaming(
            batches.into_iter(),
            &schema,
            temp_zstd.path(),
            Some(500),
            Compression::ZstdL3,
        )
        .expect("save zstd");

        let size_none = std::fs::metadata(temp_none.path()).expect("meta").len();
        let size_zstd = std::fs::metadata(temp_zstd.path()).expect("meta").len();

        // Compressed should be smaller
        assert!(
            size_zstd < size_none,
            "Zstd should compress: {size_zstd} >= {size_none}"
        );
    }

    // ========== StreamingDataset::open tests ==========

    #[test]
    fn test_streaming_dataset_open() {
        let batches = vec![make_test_batch(100, 0), make_test_batch(100, 100)];
        let schema = test_schema();

        let temp = NamedTempFile::new().expect("temp");
        save_streaming(
            batches.into_iter(),
            &schema,
            temp.path(),
            Some(64),
            Compression::ZstdL3,
        )
        .expect("save");

        let dataset = StreamingDataset::open(temp.path()).expect("open");

        assert_eq!(dataset.num_rows(), 200);
        assert!(dataset.num_chunks() > 0);
    }

    #[test]
    fn test_streaming_dataset_rejects_non_streaming_file() {
        // Test that opening a non-existent file fails
        let result = StreamingDataset::open("/nonexistent/path.ald");
        assert!(result.is_err());
    }

    // ========== get_chunk tests ==========

    #[test]
    fn test_get_chunk_returns_correct_data() {
        let batch1 = make_test_batch(100, 0);
        let batch2 = make_test_batch(100, 100);
        let schema = test_schema();

        let temp = NamedTempFile::new().expect("temp");
        save_streaming(
            vec![batch1, batch2].into_iter(),
            &schema,
            temp.path(),
            Some(100),
            Compression::None,
        )
        .expect("save");

        let dataset = StreamingDataset::open(temp.path()).expect("open");

        let chunk0 = dataset.get_chunk(0).expect("chunk 0");
        assert_eq!(chunk0.num_rows(), 100);

        // Check first row has id=0
        let ids = chunk0
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("downcast");
        assert_eq!(ids.value(0), 0);
    }

    #[test]
    fn test_get_chunk_out_of_bounds() {
        let batches = vec![make_test_batch(100, 0)];
        let schema = test_schema();

        let temp = NamedTempFile::new().expect("temp");
        save_streaming(
            batches.into_iter(),
            &schema,
            temp.path(),
            Some(50),
            Compression::None,
        )
        .expect("save");

        let dataset = StreamingDataset::open(temp.path()).expect("open");
        let result = dataset.get_chunk(999);

        assert!(result.is_err());
    }

    // ========== get_rows tests ==========

    #[test]
    fn test_get_rows_within_chunk() {
        let batches = vec![make_test_batch(100, 0)];
        let schema = test_schema();

        let temp = NamedTempFile::new().expect("temp");
        save_streaming(
            batches.into_iter(),
            &schema,
            temp.path(),
            Some(100),
            Compression::None,
        )
        .expect("save");

        let dataset = StreamingDataset::open(temp.path()).expect("open");
        let rows = dataset.get_rows(10, 20).expect("get_rows");

        assert_eq!(rows.num_rows(), 20);

        let ids = rows
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("downcast");
        assert_eq!(ids.value(0), 10);
        assert_eq!(ids.value(19), 29);
    }

    #[test]
    fn test_get_rows_spanning_chunks() {
        let batches = vec![make_test_batch(50, 0), make_test_batch(50, 50)];
        let schema = test_schema();

        let temp = NamedTempFile::new().expect("temp");
        save_streaming(
            batches.into_iter(),
            &schema,
            temp.path(),
            Some(50),
            Compression::None,
        )
        .expect("save");

        let dataset = StreamingDataset::open(temp.path()).expect("open");

        // Get rows that span the chunk boundary
        let rows = dataset.get_rows(40, 20).expect("get_rows");

        assert_eq!(rows.num_rows(), 20);

        let ids = rows
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("downcast");
        assert_eq!(ids.value(0), 40);
        assert_eq!(ids.value(9), 49); // End of first chunk
        assert_eq!(ids.value(10), 50); // Start of second chunk
        assert_eq!(ids.value(19), 59);
    }

    #[test]
    fn test_get_rows_out_of_bounds() {
        let batches = vec![make_test_batch(100, 0)];
        let schema = test_schema();

        let temp = NamedTempFile::new().expect("temp");
        save_streaming(
            batches.into_iter(),
            &schema,
            temp.path(),
            Some(50),
            Compression::None,
        )
        .expect("save");

        let dataset = StreamingDataset::open(temp.path()).expect("open");

        let result = dataset.get_rows(200, 10);
        assert!(result.is_err());
    }

    // ========== chunks iterator tests ==========

    #[test]
    fn test_chunks_iterator() {
        let batches = vec![
            make_test_batch(100, 0),
            make_test_batch(100, 100),
            make_test_batch(50, 200),
        ];
        let schema = test_schema();

        let temp = NamedTempFile::new().expect("temp");
        save_streaming(
            batches.into_iter(),
            &schema,
            temp.path(),
            Some(100),
            Compression::None,
        )
        .expect("save");

        let dataset = StreamingDataset::open(temp.path()).expect("open");

        let mut total_rows = 0;
        for chunk_result in dataset.chunks() {
            let chunk = chunk_result.expect("chunk");
            total_rows += chunk.num_rows();
        }

        assert_eq!(total_rows, 250);
    }

    #[test]
    fn test_chunks_iterator_size_hint() {
        let batches = vec![make_test_batch(200, 0)];
        let schema = test_schema();

        let temp = NamedTempFile::new().expect("temp");
        save_streaming(
            batches.into_iter(),
            &schema,
            temp.path(),
            Some(50),
            Compression::None,
        )
        .expect("save");

        let dataset = StreamingDataset::open(temp.path()).expect("open");
        let chunks = dataset.chunks();

        let (lower, upper) = chunks.size_hint();
        assert_eq!(lower, dataset.num_chunks());
        assert_eq!(upper, Some(dataset.num_chunks()));
    }

    #[test]
    fn test_chunk_index_iter() {
        let entries = vec![
            ChunkEntry::new(0, 100, 0, 500, 800),
            ChunkEntry::new(100, 100, 500, 500, 800),
        ];
        let index = ChunkIndex::from_entries(entries);

        let collected: Vec<_> = index.iter().collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].row_offset, 0);
        assert_eq!(collected[1].row_offset, 100);
    }

    #[test]
    fn test_chunk_index_from_entries_empty() {
        let index = ChunkIndex::from_entries(vec![]);
        assert!(index.is_empty());
        assert_eq!(index.total_rows(), 0);
    }

    #[test]
    fn test_streaming_dataset_schema() {
        let batches = vec![make_test_batch(100, 0)];
        let schema = test_schema();

        let temp = NamedTempFile::new().expect("temp");
        save_streaming(
            batches.into_iter(),
            &schema,
            temp.path(),
            Some(50),
            Compression::None,
        )
        .expect("save");

        let dataset = StreamingDataset::open(temp.path()).expect("open");
        let ds_schema = dataset.schema();
        assert_eq!(ds_schema.fields().len(), 2);
        assert_eq!(ds_schema.field(0).name(), "id");
        assert_eq!(ds_schema.field(1).name(), "value");
    }

    #[test]
    fn test_streaming_dataset_debug() {
        let batches = vec![make_test_batch(100, 0)];
        let schema = test_schema();

        let temp = NamedTempFile::new().expect("temp");
        save_streaming(
            batches.into_iter(),
            &schema,
            temp.path(),
            Some(50),
            Compression::None,
        )
        .expect("save");

        let dataset = StreamingDataset::open(temp.path()).expect("open");
        let debug = format!("{:?}", dataset);
        assert!(debug.contains("StreamingDataset"));
        assert!(debug.contains("num_rows"));
        assert!(debug.contains("num_chunks"));
    }

    #[test]
    fn test_save_streaming_with_lz4_compression() {
        let batches = vec![make_test_batch(1000, 0)];
        let schema = test_schema();

        let temp = NamedTempFile::new().expect("temp");
        save_streaming(
            batches.into_iter(),
            &schema,
            temp.path(),
            Some(500),
            Compression::Lz4,
        )
        .expect("save");

        let dataset = StreamingDataset::open(temp.path()).expect("open");
        assert_eq!(dataset.num_rows(), 1000);
    }

    #[test]
    fn test_save_streaming_with_zstd_l19_compression() {
        let batches = vec![make_test_batch(500, 0)];
        let schema = test_schema();

        let temp = NamedTempFile::new().expect("temp");
        save_streaming(
            batches.into_iter(),
            &schema,
            temp.path(),
            Some(250),
            Compression::ZstdL19,
        )
        .expect("save");

        let dataset = StreamingDataset::open(temp.path()).expect("open");
        assert_eq!(dataset.num_rows(), 500);
    }

    #[test]
    fn test_save_streaming_default_chunk_size() {
        let batches = vec![make_test_batch(100, 0)];
        let schema = test_schema();

        let temp = NamedTempFile::new().expect("temp");
        save_streaming(
            batches.into_iter(),
            &schema,
            temp.path(),
            None, // Use default chunk size
            Compression::None,
        )
        .expect("save");

        let dataset = StreamingDataset::open(temp.path()).expect("open");
        assert_eq!(dataset.num_rows(), 100);
    }

    #[test]
    fn test_get_rows_clamps_to_end() {
        let batches = vec![make_test_batch(100, 0)];
        let schema = test_schema();

        let temp = NamedTempFile::new().expect("temp");
        save_streaming(
            batches.into_iter(),
            &schema,
            temp.path(),
            Some(50),
            Compression::None,
        )
        .expect("save");

        let dataset = StreamingDataset::open(temp.path()).expect("open");
        // Request more rows than available
        let rows = dataset.get_rows(90, 50).expect("get_rows");
        assert_eq!(rows.num_rows(), 10); // Should clamp to 10 remaining rows
    }

    #[test]
    fn test_chunk_entry_clone_and_eq() {
        let entry1 = ChunkEntry::new(0, 100, 0, 500, 800);
        let entry2 = entry1.clone();
        assert_eq!(entry1, entry2);
    }

    #[test]
    fn test_chunk_index_clone() {
        let entries = vec![
            ChunkEntry::new(0, 100, 0, 500, 800),
            ChunkEntry::new(100, 100, 500, 500, 800),
        ];
        let index = ChunkIndex::from_entries(entries);
        let cloned = index.clone();

        assert_eq!(cloned.len(), index.len());
        assert_eq!(cloned.total_rows(), index.total_rows());
    }

    #[test]
    fn test_chunk_entry_debug() {
        let entry = ChunkEntry::new(0, 100, 0, 500, 800);
        let debug = format!("{:?}", entry);
        assert!(debug.contains("ChunkEntry"));
        assert!(debug.contains("row_offset"));
    }

    #[test]
    fn test_chunk_index_debug() {
        let index = ChunkIndex::new();
        let debug = format!("{:?}", index);
        assert!(debug.contains("ChunkIndex"));
    }

    #[test]
    fn test_constants() {
        assert_eq!(DEFAULT_CHUNK_SIZE, 65536);
        assert_eq!(STREAMING_THRESHOLD, 100 * 1024 * 1024);
    }

    #[test]
    fn test_chunks_iterator_exact_size() {
        let batches = vec![make_test_batch(200, 0)];
        let schema = test_schema();

        let temp = NamedTempFile::new().expect("temp");
        save_streaming(
            batches.into_iter(),
            &schema,
            temp.path(),
            Some(50),
            Compression::None,
        )
        .expect("save");

        let dataset = StreamingDataset::open(temp.path()).expect("open");
        let chunks = dataset.chunks();

        // ExactSizeIterator provides len()
        assert_eq!(chunks.len(), dataset.num_chunks());
    }
}
