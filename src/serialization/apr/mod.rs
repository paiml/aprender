//! APR (Aprender) binary format with JSON metadata.
//!
//! A compact binary format optimized for WASM deployment with:
//! - LZ4/ZSTD compression support (GH-146)
//! - JSON metadata section (vocab, config, tokenizer, etc.)
//! - Streaming decompression capability
//!
//! Format (APR\0 - ONE format, no versioning):
//! ```text
//! [4-byte magic: "APR\0"]
//! [4-byte metadata_len: u32 little-endian]
//! [JSON metadata: arbitrary key-value pairs]
//! [4-byte n_tensors: u32 little-endian]
//! [Tensor index: name, dtype, shape, offset, size per tensor]
//! [Raw tensor data: values in little-endian]
//! [4-byte CRC32: checksum of all preceding bytes]
//! ```
//!
//! Format (compressed):
//! ```text
//! [4-byte magic: "APR\0"]
//! [1-byte compression: 0=None, 1=LZ4, 2=ZSTD]
//! [4-byte uncompressed_len: u32 little-endian]
//! [compressed payload: compressed APR data]
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// Magic bytes for APR format (uncompressed) - "APR1"
pub const APR_MAGIC: [u8; 4] = [b'A', b'P', b'R', b'1'];

/// Magic bytes for APR format (compressed) - "APR\0" (APR + null byte)
pub const APR_MAGIC_COMPRESSED: [u8; 4] = [b'A', b'P', b'R', 0];

/// Compression algorithm for .apr files
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Compression {
    /// No compression (default, backward compatible)
    #[default]
    None,
    /// LZ4 compression - fast, good for real-time (GH-146)
    #[cfg(feature = "format-compression")]
    Lz4,
    /// ZSTD compression - better ratio, slower
    #[cfg(feature = "format-compression")]
    Zstd,
}

impl Compression {
    /// Get compression type byte for header
    #[must_use]
    pub const fn as_byte(self) -> u8 {
        match self {
            Self::None => 0,
            #[cfg(feature = "format-compression")]
            Self::Lz4 => 1,
            #[cfg(feature = "format-compression")]
            Self::Zstd => 2,
        }
    }

    /// Parse compression type from byte
    #[must_use]
    pub const fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::None),
            #[cfg(feature = "format-compression")]
            1 => Some(Self::Lz4),
            #[cfg(feature = "format-compression")]
            2 => Some(Self::Zstd),
            _ => None,
        }
    }
}

/// Tensor descriptor in the index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AprTensorDescriptor {
    /// Tensor name
    pub name: String,
    /// Data type (e.g., "F32", "I8")
    pub dtype: String,
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// Byte offset in data section
    pub offset: usize,
    /// Byte size
    pub size: usize,
}

/// APR file metadata - arbitrary JSON
pub type AprMetadata = BTreeMap<String, JsonValue>;

/// APR format reader
#[derive(Debug)]
pub struct AprReader {
    /// Parsed metadata
    pub metadata: AprMetadata,
    /// Tensor descriptors
    pub tensors: Vec<AprTensorDescriptor>,
    /// Raw file data
    data: Vec<u8>,
    /// Offset to tensor data section
    tensor_data_offset: usize,
}

impl AprReader {
    /// Load APR file from path
    ///
    /// # Errors
    /// Returns error if file is invalid or cannot be read
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let data = fs::read(path).map_err(|e| format!("Failed to read file: {e}"))?;
        Self::from_bytes(data)
    }

    /// Parse APR format from bytes
    ///
    /// Automatically detects APR1 (uncompressed) or APR2 (compressed) format.
    ///
    /// # Errors
    /// Returns error if format is invalid
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, String> {
        // Validate minimum size for magic
        if data.len() < 8 {
            return Err("File too short".to_string());
        }

        // Check for APR2 (compressed) format first
        let magic = data.get(0..4).ok_or("File too short for magic")?;
        if magic == APR_MAGIC_COMPRESSED {
            return Self::from_bytes_compressed(&data);
        }

        // Check for APR1 (uncompressed) format
        if magic != APR_MAGIC {
            return Err(format!(
                "Invalid magic: expected APR1 or APR2, got {magic:?}",
            ));
        }

        Self::from_bytes_uncompressed(data)
    }

    /// Parse compressed APR2 format
    fn from_bytes_compressed(data: &[u8]) -> Result<Self, String> {
        if data.len() < 9 {
            return Err("APR2 file too short for header".to_string());
        }

        let compression_byte = data[4];
        let compression = Compression::from_byte(compression_byte)
            .ok_or_else(|| format!("Unknown compression type: {compression_byte}"))?;

        let uncompressed_len = u32::from_le_bytes([data[5], data[6], data[7], data[8]]) as usize;
        let compressed_payload = data.get(9..).ok_or("APR2 file too short for payload")?;

        // Decompress based on algorithm
        let decompressed =
            Self::decompress_payload(compression, compressed_payload, uncompressed_len)?;

        // Parse the decompressed APR1 data
        Self::from_bytes_uncompressed(decompressed)
    }

    /// Decompress payload using detected algorithm
    #[allow(unreachable_patterns)] // Pattern varies based on format-compression feature
    fn decompress_payload(
        compression: Compression,
        data: &[u8],
        _expected_len: usize,
    ) -> Result<Vec<u8>, String> {
        match compression {
            Compression::None => Ok(data.to_vec()),
            #[cfg(feature = "format-compression")]
            Compression::Lz4 => lz4_flex::decompress_size_prepended(data)
                .map_err(|e| format!("LZ4 decompression failed: {e}")),
            #[cfg(feature = "format-compression")]
            Compression::Zstd => {
                zstd::decode_all(data).map_err(|e| format!("ZSTD decompression failed: {e}"))
            }
            #[cfg(not(feature = "format-compression"))]
            _ => Err(format!(
                "Compressed APR file requires 'format-compression' feature (compression type: {})",
                compression.as_byte()
            )),
        }
    }

    /// Parse uncompressed APR1 format
    fn from_bytes_uncompressed(data: Vec<u8>) -> Result<Self, String> {
        // Validate APR1 magic
        let magic = data.get(0..4).ok_or("File too short for magic")?;
        if magic != APR_MAGIC {
            return Err(format!("Invalid magic: expected APR1, got {magic:?}",));
        }

        // Read metadata length
        let metadata_len = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

        if data.len() < 8 + metadata_len + 4 {
            return Err("File too short for metadata".to_string());
        }

        // Parse metadata JSON
        let metadata_json = &data[8..8 + metadata_len];
        let metadata: AprMetadata = if metadata_len > 0 {
            serde_json::from_slice(metadata_json)
                .map_err(|e| format!("Invalid metadata JSON: {e}"))?
        } else {
            BTreeMap::new()
        };

        // Read tensor count
        let tensor_count_offset = 8 + metadata_len;
        let n_tensors = u32::from_le_bytes([
            data[tensor_count_offset],
            data[tensor_count_offset + 1],
            data[tensor_count_offset + 2],
            data[tensor_count_offset + 3],
        ]) as usize;

        // Read tensor index (JSON array)
        let index_offset = tensor_count_offset + 4;

        // Find end of tensor index by reading the index length
        let index_len = u32::from_le_bytes([
            data[index_offset],
            data[index_offset + 1],
            data[index_offset + 2],
            data[index_offset + 3],
        ]) as usize;

        let index_data = &data[index_offset + 4..index_offset + 4 + index_len];
        let tensors: Vec<AprTensorDescriptor> = if n_tensors > 0 {
            serde_json::from_slice(index_data).map_err(|e| format!("Invalid tensor index: {e}"))?
        } else {
            Vec::new()
        };

        let tensor_data_offset = index_offset + 4 + index_len;

        Ok(Self {
            metadata,
            tensors,
            data,
            tensor_data_offset,
        })
    }

    /// Get metadata value by key
    #[must_use]
    pub fn get_metadata(&self, key: &str) -> Option<&JsonValue> {
        self.metadata.get(key)
    }

    /// Read tensor data as f32 values
    ///
    /// # Errors
    /// Returns error if tensor not found or data invalid
    pub fn read_tensor_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let desc = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("Tensor not found: {name}"))?;

        let start = self.tensor_data_offset + desc.offset;
        let end = start + desc.size;

        if end > self.data.len() {
            return Err("Tensor data out of bounds".to_string());
        }

        let bytes = &self.data[start..end];
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(values)
    }
}

/// APR format writer
#[derive(Debug, Default)]
pub struct AprWriter {
    /// Metadata to write
    metadata: AprMetadata,
    /// Tensors to write
    tensors: Vec<(AprTensorDescriptor, Vec<u8>)>,
    /// Compression algorithm (default: None for backward compatibility)
    compression: Compression,
}

impl AprWriter {
    /// Create new writer
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set compression algorithm for output
    ///
    /// Default is `Compression::None` for backward compatibility with APR1 format.
    /// When compression is enabled, writes APR2 format with compressed payload.
    #[must_use]
    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Set metadata key-value pair
    pub fn set_metadata(&mut self, key: impl Into<String>, value: JsonValue) {
        self.metadata.insert(key.into(), value);
    }

    /// Add tensor with f32 data
    pub fn add_tensor_f32(&mut self, name: impl Into<String>, shape: Vec<usize>, data: &[f32]) {
        let name = name.into();
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let size = bytes.len();

        // Calculate offset based on existing tensors
        let offset: usize = self.tensors.iter().map(|(_, d)| d.len()).sum();

        let desc = AprTensorDescriptor {
            name,
            dtype: "F32".to_string(),
            shape,
            offset,
            size,
        };

        self.tensors.push((desc, bytes));
    }

    /// Write to bytes
    ///
    /// # Errors
    /// Returns error if serialization fails
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        // Generate APR1 (uncompressed) payload first
        let apr1_data = self.to_bytes_uncompressed()?;

        // If no compression, return APR1 format directly
        if matches!(self.compression, Compression::None) {
            return Ok(apr1_data);
        }

        // Compress and wrap in APR2 format
        #[cfg(feature = "format-compression")]
        {
            let compressed = self.compress_payload(&apr1_data)?;

            let mut output = Vec::with_capacity(9 + compressed.len());
            output.extend_from_slice(&APR_MAGIC_COMPRESSED); // APR2 magic
            output.push(self.compression.as_byte()); // compression type
            output.extend_from_slice(&(apr1_data.len() as u32).to_le_bytes()); // uncompressed size
            output.extend_from_slice(&compressed);

            Ok(output)
        }

        #[cfg(not(feature = "format-compression"))]
        {
            // Compression requested but feature not enabled
            Err("Compression requested but 'format-compression' feature not enabled".to_string())
        }
    }

    /// Generate uncompressed APR1 format
    fn to_bytes_uncompressed(&self) -> Result<Vec<u8>, String> {
        let mut output = Vec::new();

        // 1. Magic
        output.extend_from_slice(&APR_MAGIC);

        // 2. Metadata
        let metadata_json = serde_json::to_string(&self.metadata)
            .map_err(|e| format!("Metadata serialization failed: {e}"))?;
        let metadata_bytes = metadata_json.as_bytes();
        output.extend_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        output.extend_from_slice(metadata_bytes);

        // 3. Tensor count
        output.extend_from_slice(&(self.tensors.len() as u32).to_le_bytes());

        // 4. Tensor index
        let descriptors: Vec<_> = self.tensors.iter().map(|(d, _)| d).collect();
        let index_json = serde_json::to_string(&descriptors)
            .map_err(|e| format!("Index serialization failed: {e}"))?;
        let index_bytes = index_json.as_bytes();
        output.extend_from_slice(&(index_bytes.len() as u32).to_le_bytes());
        output.extend_from_slice(index_bytes);

        // 5. Tensor data
        for (_, data) in &self.tensors {
            output.extend_from_slice(data);
        }

        // 6. CRC32
        let crc = crc32(&output);
        output.extend_from_slice(&crc.to_le_bytes());

        Ok(output)
    }

    /// Compress payload using selected algorithm
    #[cfg(feature = "format-compression")]
    fn compress_payload(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        match self.compression {
            Compression::None => Ok(data.to_vec()),
            Compression::Lz4 => Ok(lz4_flex::compress_prepend_size(data)),
            Compression::Zstd => {
                zstd::encode_all(data, 3).map_err(|e| format!("ZSTD compression failed: {e}"))
            }
        }
    }

    /// Write to file
    ///
    /// # Errors
    /// Returns error if write fails
    pub fn write<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let bytes = self.to_bytes()?;
        fs::write(path, bytes).map_err(|e| format!("Write failed: {e}"))
    }
}

include!("mod_part_02.rs");
