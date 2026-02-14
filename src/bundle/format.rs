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
        bytes.len() >= 8 && bytes.get(0..8) == Some(BUNDLE_MAGIC)
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
#[path = "format_tests.rs"]
mod tests;
