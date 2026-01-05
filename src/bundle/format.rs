//! Bundle File Format
//!
//! Defines the binary format for .apbundle files and provides read/write operations.
//!
//! # Format Structure
//!
//! ```text
//! +------------------+
//! | Magic (8 bytes)  |  "APBUNDLE"
//! +------------------+
//! | Version (4 bytes)|  u32 little-endian
//! +------------------+
//! | Manifest Length  |  u64 little-endian
//! +------------------+
//! | Manifest Data    |  Variable length
//! +------------------+
//! | Model Data       |  Concatenated model bytes
//! +------------------+
//! ```

use super::manifest::{BundleManifest, ModelEntry};
use super::BUNDLE_MAGIC;
use crate::error::{AprenderError, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Helper to create IO errors with a message.
fn io_error(msg: impl Into<String>) -> AprenderError {
    AprenderError::Other(msg.into())
}

/// Helper to create format errors.
fn format_error(msg: impl Into<String>) -> AprenderError {
    AprenderError::FormatError {
        message: msg.into(),
    }
}

/// Bundle file format constants.
#[derive(Debug, Clone, Copy)]
pub struct BundleFormat;

impl BundleFormat {
    /// Header size: magic (8) + version (4) + `manifest_len` (8) = 20 bytes.
    pub const HEADER_SIZE: usize = 20;

    /// Validate magic bytes.
    #[must_use]
    pub fn validate_magic(bytes: &[u8]) -> bool {
        bytes.len() >= 8 && &bytes[0..8] == BUNDLE_MAGIC
    }

    /// Extract version from header.
    #[must_use]
    pub fn read_version(header: &[u8]) -> Option<u32> {
        if header.len() < 12 {
            return None;
        }
        Some(u32::from_le_bytes(header[8..12].try_into().ok()?))
    }

    /// Extract manifest length from header.
    #[must_use]
    pub fn read_manifest_length(header: &[u8]) -> Option<u64> {
        if header.len() < 20 {
            return None;
        }
        Some(u64::from_le_bytes(header[12..20].try_into().ok()?))
    }
}

// ============================================================================
// Bundle Reader
// ============================================================================

/// Reader for loading bundle files.
pub struct BundleReader {
    /// Buffered reader for the bundle file.
    reader: BufReader<File>,
    /// Version number from the header.
    header_version: u32,
    /// Offset where the manifest begins.
    manifest_offset: u64,
    /// Offset where model data begins.
    data_offset: u64,
}

impl std::fmt::Debug for BundleReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BundleReader")
            .field("header_version", &self.header_version)
            .field("manifest_offset", &self.manifest_offset)
            .field("data_offset", &self.data_offset)
            .finish_non_exhaustive()
    }
}

impl BundleReader {
    /// Open a bundle file for reading.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref()).map_err(|e| {
            io_error(format!(
                "Failed to open bundle file '{}': {}",
                path.as_ref().display(),
                e
            ))
        })?;

        let mut reader = BufReader::new(file);

        // Read and validate header
        let mut header = [0u8; BundleFormat::HEADER_SIZE];
        reader
            .read_exact(&mut header)
            .map_err(|e| io_error(format!("Failed to read bundle header: {e}")))?;

        if !BundleFormat::validate_magic(&header) {
            return Err(format_error("Invalid bundle file: magic bytes mismatch"));
        }

        let version = BundleFormat::read_version(&header)
            .ok_or_else(|| format_error("Invalid bundle version"))?;

        let manifest_len = BundleFormat::read_manifest_length(&header)
            .ok_or_else(|| format_error("Invalid manifest length"))?;

        Ok(Self {
            reader,
            header_version: version,
            manifest_offset: BundleFormat::HEADER_SIZE as u64,
            data_offset: BundleFormat::HEADER_SIZE as u64 + manifest_len,
        })
    }

    /// Get the bundle version.
    #[must_use]
    pub fn version(&self) -> u32 {
        self.header_version
    }

    /// Read the bundle manifest.
    pub fn read_manifest(&mut self) -> Result<BundleManifest> {
        self.reader
            .seek(SeekFrom::Start(self.manifest_offset))
            .map_err(|e| io_error(format!("Failed to seek to manifest: {e}")))?;

        let manifest_len = (self.data_offset - self.manifest_offset) as usize;
        let mut manifest_bytes = vec![0u8; manifest_len];
        self.reader
            .read_exact(&mut manifest_bytes)
            .map_err(|e| io_error(format!("Failed to read manifest: {e}")))?;

        BundleManifest::from_bytes(&manifest_bytes)
            .ok_or_else(|| format_error("Failed to parse manifest"))
    }

    /// Read a single model's data.
    pub fn read_model(&mut self, entry: &ModelEntry) -> Result<Vec<u8>> {
        let offset = self.data_offset + entry.offset;
        self.reader
            .seek(SeekFrom::Start(offset))
            .map_err(|e| io_error(format!("Failed to seek to model data: {e}")))?;

        let mut data = vec![0u8; entry.size];
        self.reader
            .read_exact(&mut data)
            .map_err(|e| io_error(format!("Failed to read model data: {e}")))?;

        Ok(data)
    }

    /// Read all models from the bundle.
    pub fn read_all_models(
        &mut self,
        manifest: &BundleManifest,
    ) -> Result<HashMap<String, Vec<u8>>> {
        let mut models = HashMap::new();

        for entry in manifest.iter() {
            let data = self.read_model(entry)?;
            models.insert(entry.name.clone(), data);
        }

        Ok(models)
    }

    /// Get the offset where model data begins.
    #[must_use]
    pub fn data_offset(&self) -> u64 {
        self.data_offset
    }
}

// ============================================================================
// Bundle Writer
// ============================================================================

/// Writer for creating bundle files.
pub struct BundleWriter {
    /// Buffered writer for the bundle file.
    writer: BufWriter<File>,
}

impl std::fmt::Debug for BundleWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BundleWriter").finish_non_exhaustive()
    }
}

impl BundleWriter {
    /// Create a new bundle file for writing.
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::create(path.as_ref()).map_err(|e| {
            io_error(format!(
                "Failed to create bundle file '{}': {}",
                path.as_ref().display(),
                e
            ))
        })?;

        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    /// Write a complete bundle.
    pub fn write_bundle(
        mut self,
        manifest: &BundleManifest,
        models: &HashMap<String, Vec<u8>>,
    ) -> Result<()> {
        // Calculate offsets for each model
        let mut updated_manifest = manifest.clone();
        let mut current_offset = 0u64;

        for name in manifest.model_names() {
            if let Some(entry) = updated_manifest.get_model_mut(name) {
                entry.offset = current_offset;
                current_offset += entry.size as u64;
            }
        }

        // Serialize manifest
        let manifest_bytes = updated_manifest.to_bytes();

        // Write header
        self.write_header(manifest.version, manifest_bytes.len() as u64)?;

        // Write manifest
        self.writer
            .write_all(&manifest_bytes)
            .map_err(|e| io_error(format!("Failed to write manifest: {e}")))?;

        // Write model data in order
        for name in manifest.model_names() {
            if let Some(data) = models.get(name) {
                self.writer
                    .write_all(data)
                    .map_err(|e| io_error(format!("Failed to write model '{name}': {e}")))?;
            }
        }

        self.writer
            .flush()
            .map_err(|e| io_error(format!("Failed to flush bundle: {e}")))?;

        Ok(())
    }

    /// Write the bundle header.
    fn write_header(&mut self, version: u32, manifest_len: u64) -> Result<()> {
        // Magic bytes
        self.writer
            .write_all(BUNDLE_MAGIC)
            .map_err(|e| io_error(format!("Failed to write magic bytes: {e}")))?;

        // Version
        self.writer
            .write_all(&version.to_le_bytes())
            .map_err(|e| io_error(format!("Failed to write version: {e}")))?;

        // Manifest length
        self.writer
            .write_all(&manifest_len.to_le_bytes())
            .map_err(|e| io_error(format!("Failed to write manifest length: {e}")))?;

        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_bundle_format_validate_magic() {
        assert!(BundleFormat::validate_magic(BUNDLE_MAGIC));
        assert!(BundleFormat::validate_magic(b"APBUNDLEextra"));
        assert!(!BundleFormat::validate_magic(b"INVALID!"));
        assert!(!BundleFormat::validate_magic(b"SHORT"));
    }

    #[test]
    fn test_bundle_format_read_version() {
        let mut header = vec![0u8; 20];
        header[0..8].copy_from_slice(BUNDLE_MAGIC);
        header[8..12].copy_from_slice(&1u32.to_le_bytes());

        assert_eq!(BundleFormat::read_version(&header), Some(1));
    }

    #[test]
    fn test_bundle_format_read_manifest_length() {
        let mut header = vec![0u8; 20];
        header[0..8].copy_from_slice(BUNDLE_MAGIC);
        header[8..12].copy_from_slice(&1u32.to_le_bytes());
        header[12..20].copy_from_slice(&256u64.to_le_bytes());

        assert_eq!(BundleFormat::read_manifest_length(&header), Some(256));
    }

    #[test]
    fn test_bundle_writer_reader_roundtrip() {
        let temp = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp.path();

        // Create manifest and models
        let mut manifest = BundleManifest::new().with_description("Test");
        manifest.add_model(ModelEntry::new("model1", 4));
        manifest.add_model(ModelEntry::new("model2", 3));

        let mut models = HashMap::new();
        models.insert("model1".to_string(), vec![1, 2, 3, 4]);
        models.insert("model2".to_string(), vec![5, 6, 7]);

        // Write
        let writer = BundleWriter::create(path).expect("Failed to create writer");
        writer
            .write_bundle(&manifest, &models)
            .expect("Failed to write bundle");

        // Read
        let mut reader = BundleReader::open(path).expect("Failed to open reader");
        assert_eq!(reader.version(), manifest.version);

        let read_manifest = reader.read_manifest().expect("Failed to read manifest");
        assert_eq!(read_manifest.len(), 2);
        assert_eq!(read_manifest.description, "Test");

        let read_models = reader
            .read_all_models(&read_manifest)
            .expect("Failed to read models");
        assert_eq!(read_models.get("model1"), Some(&vec![1, 2, 3, 4]));
        assert_eq!(read_models.get("model2"), Some(&vec![5, 6, 7]));
    }

    #[test]
    fn test_bundle_reader_invalid_magic() {
        let temp = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp.path();

        // Write invalid data
        std::fs::write(path, b"INVALID!").expect("Failed to write");

        // Should fail to open
        let result = BundleReader::open(path);
        assert!(result.is_err());
    }

    #[test]
    fn test_bundle_read_single_model() {
        let temp = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp.path();

        let mut manifest = BundleManifest::new();
        manifest.add_model(ModelEntry::new("weights", 5));

        let mut models = HashMap::new();
        models.insert("weights".to_string(), vec![10, 20, 30, 40, 50]);

        let writer = BundleWriter::create(path).expect("Failed to create writer");
        writer
            .write_bundle(&manifest, &models)
            .expect("Failed to write");

        let mut reader = BundleReader::open(path).expect("Failed to open");
        let manifest = reader.read_manifest().expect("Failed to read manifest");
        let entry = manifest.get_model("weights").expect("Model not found");

        let data = reader.read_model(entry).expect("Failed to read model");
        assert_eq!(data, vec![10, 20, 30, 40, 50]);
    }

    // ============================================================================
    // Additional Coverage Tests
    // ============================================================================

    #[test]
    fn test_bundle_format_read_version_short_header() {
        // Header too short for version
        let header = vec![0u8; 8];
        assert_eq!(BundleFormat::read_version(&header), None);
    }

    #[test]
    fn test_bundle_format_read_manifest_length_short_header() {
        // Header too short for manifest length
        let header = vec![0u8; 15];
        assert_eq!(BundleFormat::read_manifest_length(&header), None);
    }

    #[test]
    fn test_bundle_reader_debug() {
        let temp = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp.path();

        // Create a valid bundle
        let manifest = BundleManifest::new();
        let models = HashMap::new();
        let writer = BundleWriter::create(path).expect("Failed to create writer");
        writer
            .write_bundle(&manifest, &models)
            .expect("Failed to write");

        let reader = BundleReader::open(path).expect("Failed to open");
        let debug_str = format!("{:?}", reader);
        assert!(debug_str.contains("BundleReader"));
        assert!(debug_str.contains("header_version"));
    }

    #[test]
    fn test_bundle_writer_debug() {
        let temp = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp.path();

        let writer = BundleWriter::create(path).expect("Failed to create writer");
        let debug_str = format!("{:?}", writer);
        assert!(debug_str.contains("BundleWriter"));
    }

    #[test]
    fn test_bundle_reader_data_offset() {
        let temp = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp.path();

        let manifest = BundleManifest::new().with_description("Test offset");
        let models = HashMap::new();
        let writer = BundleWriter::create(path).expect("Failed to create writer");
        writer
            .write_bundle(&manifest, &models)
            .expect("Failed to write");

        let reader = BundleReader::open(path).expect("Failed to open");
        let offset = reader.data_offset();
        // Header is 20 bytes, manifest follows
        assert!(offset >= BundleFormat::HEADER_SIZE as u64);
    }

    #[test]
    fn test_bundle_format_header_size_constant() {
        assert_eq!(BundleFormat::HEADER_SIZE, 20);
    }

    #[test]
    fn test_bundle_format_copy_clone() {
        let format = BundleFormat;
        let _cloned = format;
        let _copied = format;
        // Just testing that Copy + Clone derive works
    }

    #[test]
    fn test_bundle_reader_open_nonexistent() {
        let result = BundleReader::open("/nonexistent/path/bundle.apbundle");
        assert!(result.is_err());
    }

    #[test]
    fn test_bundle_reader_truncated_header() {
        let temp = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp.path();

        // Write only magic, no version or manifest length
        std::fs::write(path, BUNDLE_MAGIC).expect("Failed to write");

        let result = BundleReader::open(path);
        assert!(result.is_err());
    }

    #[test]
    fn test_bundle_writer_create_invalid_path() {
        let result = BundleWriter::create("/nonexistent/directory/bundle.apbundle");
        assert!(result.is_err());
    }

    #[test]
    fn test_bundle_empty_manifest() {
        let temp = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp.path();

        let manifest = BundleManifest::new();
        let models = HashMap::new();

        let writer = BundleWriter::create(path).expect("Failed to create writer");
        writer
            .write_bundle(&manifest, &models)
            .expect("Failed to write empty bundle");

        let mut reader = BundleReader::open(path).expect("Failed to open");
        let read_manifest = reader.read_manifest().expect("Failed to read manifest");
        assert_eq!(read_manifest.len(), 0);
    }

    #[test]
    fn test_bundle_multiple_models_order() {
        let temp = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp.path();

        let mut manifest = BundleManifest::new();
        manifest.add_model(ModelEntry::new("first", 3));
        manifest.add_model(ModelEntry::new("second", 4));
        manifest.add_model(ModelEntry::new("third", 5));

        let mut models = HashMap::new();
        models.insert("first".to_string(), vec![1, 2, 3]);
        models.insert("second".to_string(), vec![4, 5, 6, 7]);
        models.insert("third".to_string(), vec![8, 9, 10, 11, 12]);

        let writer = BundleWriter::create(path).expect("Failed to create writer");
        writer
            .write_bundle(&manifest, &models)
            .expect("Failed to write");

        let mut reader = BundleReader::open(path).expect("Failed to open");
        let manifest = reader.read_manifest().expect("Failed to read manifest");
        let all = reader
            .read_all_models(&manifest)
            .expect("Failed to read all");

        assert_eq!(all.len(), 3);
        assert_eq!(all.get("first"), Some(&vec![1, 2, 3]));
        assert_eq!(all.get("second"), Some(&vec![4, 5, 6, 7]));
        assert_eq!(all.get("third"), Some(&vec![8, 9, 10, 11, 12]));
    }
}
