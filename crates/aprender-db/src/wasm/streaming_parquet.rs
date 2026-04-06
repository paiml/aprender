//! Streaming Parquet reader with HTTP range requests for browser deployment.
//!
//! Implements footer-first reading strategy to minimize memory usage:
//! 1. Read Parquet footer (last ~1KB) to get metadata
//! 2. Read row groups on-demand based on query predicates
//! 3. Use late materialization to defer row reconstruction
//!
//! # Memory Strategy
//!
//! Browser constraint: ~2GB heap limit
//! - Footer: <1MB
//! - Per row group: <128MB
//! - Active result set: <256MB
//! Total: <400MB (leaves 1.6GB headroom)
//!
//! # Parquet File Structure
//!
//! ```text
//! [4 bytes]  Magic Number "PAR1"
//! [variable] Row Group 1
//! [variable] Row Group 2
//! ...
//! [variable] Footer (FileMetaData)
//! [4 bytes]  Footer length (little-endian)
//! [4 bytes]  Magic Number "PAR1"
//! ```
//!
//! # References
//! - Apache Parquet Format Spec v2.9.0
//! - Abadi et al. 2008: Late materialization

#![cfg(target_arch = "wasm32")]

use super::http_range::{ByteRange, RangeClient};
use wasm_bindgen::prelude::*;
use web_sys::console;

/// Parquet magic number "PAR1"
const PARQUET_MAGIC: &[u8] = b"PAR1";

/// Size of Parquet footer metadata (magic + length)
const FOOTER_METADATA_SIZE: u64 = 8; // 4 bytes length + 4 bytes magic

/// Maximum footer size (safety limit)
const MAX_FOOTER_SIZE: u64 = 10 * 1024 * 1024; // 10MB

/// Streaming Parquet reader with HTTP range requests
pub struct StreamingParquetReader {
    /// HTTP range client for remote file
    client: RangeClient,

    /// Total file size in bytes
    file_size: u64,

    /// Parsed file metadata (lazily loaded)
    metadata: Option<ParquetMetadata>,
}

impl StreamingParquetReader {
    /// Create a new streaming reader for the given URL
    ///
    /// # Example
    ///
    /// ```ignore
    /// let reader = StreamingParquetReader::new("https://cdn.example.com/data.parquet");
    /// let metadata = reader.read_metadata().await?;
    /// ```
    pub async fn new(url: impl Into<String>) -> Result<Self, JsValue> {
        let client = RangeClient::new(url);
        let file_size = client.get_file_size().await?;

        console::log_1(&format!("Parquet file size: {} bytes", file_size).into());

        Ok(Self { client, file_size, metadata: None })
    }

    /// Read and parse Parquet footer metadata
    ///
    /// This reads the last ~1KB of the file to get:
    /// - Number of row groups
    /// - Schema information
    /// - Column statistics (min/max for predicate pushdown)
    pub async fn read_metadata(&mut self) -> Result<&ParquetMetadata, JsValue> {
        if self.metadata.is_some() {
            return Ok(self.metadata.as_ref().expect("metadata checked Some above"));
        }

        // Read footer metadata (last 8 bytes)
        let footer_range =
            ByteRange::new(self.file_size - FOOTER_METADATA_SIZE, self.file_size - 1);
        let footer_bytes = self.client.fetch_range(footer_range).await?;

        // Verify magic number
        if &footer_bytes[4..8] != PARQUET_MAGIC {
            return Err(JsValue::from_str("Invalid Parquet file: missing magic number"));
        }

        // Parse footer length (little-endian i32)
        let footer_length = u32::from_le_bytes([
            footer_bytes[0],
            footer_bytes[1],
            footer_bytes[2],
            footer_bytes[3],
        ]) as u64;

        if footer_length > MAX_FOOTER_SIZE {
            return Err(JsValue::from_str(&format!(
                "Footer too large: {} bytes (max {})",
                footer_length, MAX_FOOTER_SIZE
            )));
        }

        console::log_1(&format!("Reading Parquet footer: {} bytes", footer_length).into());

        // Read full footer
        let footer_start = self.file_size - FOOTER_METADATA_SIZE - footer_length;
        let footer_range = ByteRange::new(footer_start, self.file_size - FOOTER_METADATA_SIZE - 1);
        let footer_data = self.client.fetch_range(footer_range).await?;

        // Parse footer (simplified for now - full Thrift parsing in future)
        let metadata = self.parse_footer(&footer_data)?;
        self.metadata = Some(metadata);

        Ok(self.metadata.as_ref().expect("metadata just assigned above"))
    }

    /// Parse Parquet footer bytes into metadata
    ///
    /// TODO: Full Thrift deserialization
    /// For now, returns stub metadata for demonstration
    fn parse_footer(&self, _data: &[u8]) -> Result<ParquetMetadata, JsValue> {
        // Stub implementation - real version needs Thrift deserialization
        console::log_1(&"Parsing Parquet footer (stub implementation)".into());

        Ok(ParquetMetadata { num_rows: 0, row_groups: vec![], schema: ParquetSchema::default() })
    }

    /// Read a specific row group on-demand
    ///
    /// # Arguments
    /// * `index` - Row group index (0-based)
    ///
    /// # Returns
    /// Row group data as raw bytes (to be decoded into Arrow RecordBatch)
    pub async fn read_row_group(&self, index: usize) -> Result<Vec<u8>, JsValue> {
        let metadata = self
            .metadata
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Metadata not loaded. Call read_metadata() first"))?;

        if index >= metadata.row_groups.len() {
            return Err(JsValue::from_str(&format!(
                "Row group index {} out of bounds (total: {})",
                index,
                metadata.row_groups.len()
            )));
        }

        let row_group = &metadata.row_groups[index];
        let range = ByteRange::new(
            row_group.file_offset,
            row_group.file_offset + row_group.total_byte_size - 1,
        );

        console::log_1(
            &format!(
                "Reading row group {}: {} bytes at offset {}",
                index, row_group.total_byte_size, row_group.file_offset
            )
            .into(),
        );

        self.client.fetch_range(range).await
    }

    /// Read specific columns from a row group (column pruning)
    ///
    /// Only reads the requested columns, saving bandwidth and memory.
    pub async fn read_columns(
        &self,
        row_group_index: usize,
        column_indices: &[usize],
    ) -> Result<Vec<Vec<u8>>, JsValue> {
        let metadata =
            self.metadata.as_ref().ok_or_else(|| JsValue::from_str("Metadata not loaded"))?;

        let row_group = &metadata.row_groups[row_group_index];
        let mut column_data = Vec::new();

        for &col_idx in column_indices {
            if col_idx >= row_group.columns.len() {
                return Err(JsValue::from_str(&format!("Column index {} out of bounds", col_idx)));
            }

            let column = &row_group.columns[col_idx];
            let range = ByteRange::new(
                column.file_offset,
                column.file_offset + column.total_compressed_size - 1,
            );
            let bytes = self.client.fetch_range(range).await?;
            column_data.push(bytes);
        }

        Ok(column_data)
    }
}

/// Parquet file metadata (simplified)
#[derive(Debug, Clone)]
pub struct ParquetMetadata {
    /// Total number of rows in file
    pub num_rows: i64,

    /// Row group metadata
    pub row_groups: Vec<RowGroupMetadata>,

    /// Schema information
    pub schema: ParquetSchema,
}

/// Row group metadata
#[derive(Debug, Clone)]
pub struct RowGroupMetadata {
    /// Number of rows in this row group
    pub num_rows: i64,

    /// File offset where this row group starts
    pub file_offset: u64,

    /// Total byte size of this row group
    pub total_byte_size: u64,

    /// Column chunk metadata
    pub columns: Vec<ColumnChunkMetadata>,
}

/// Column chunk metadata
#[derive(Debug, Clone)]
pub struct ColumnChunkMetadata {
    /// Column name
    pub name: String,

    /// File offset for this column chunk
    pub file_offset: u64,

    /// Compressed size in bytes
    pub total_compressed_size: u64,

    /// Statistics for predicate pushdown
    pub statistics: Option<ColumnStatistics>,
}

/// Column statistics (for predicate pushdown)
#[derive(Debug, Clone)]
pub struct ColumnStatistics {
    /// Minimum value (as bytes)
    pub min: Vec<u8>,

    /// Maximum value (as bytes)
    pub max: Vec<u8>,

    /// Null count
    pub null_count: i64,
}

/// Parquet schema (simplified)
#[derive(Debug, Clone, Default)]
pub struct ParquetSchema {
    /// Column names
    pub columns: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parquet_magic_constant() {
        assert_eq!(PARQUET_MAGIC, b"PAR1");
        assert_eq!(PARQUET_MAGIC.len(), 4);
    }

    #[test]
    fn test_footer_metadata_size() {
        assert_eq!(FOOTER_METADATA_SIZE, 8);
    }

    #[test]
    fn test_max_footer_size_reasonable() {
        // Footer should be reasonable (10MB limit)
        assert_eq!(MAX_FOOTER_SIZE, 10 * 1024 * 1024);
    }
}
