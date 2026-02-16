//! APR (Aprender) binary format — ONE format.
//!
//! This module provides convenience wrappers (`AprWriter`, `AprReader`) around
//! the canonical APR binary format defined in `crate::format::v2`. All APR files
//! use the same binary layout with 64-byte aligned sections, CRC32 checksums,
//! and JSON metadata.
//!
//! See `src/format/v2/` for the canonical format implementation.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

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

/// APR format reader — delegates to `AprV2Reader` (ONE format).
#[derive(Debug)]
pub struct AprReader {
    /// Parsed metadata
    pub metadata: AprMetadata,
    /// Tensor descriptors
    pub tensors: Vec<AprTensorDescriptor>,
    /// Raw file data (owned for tensor reads)
    data: Vec<u8>,
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
    /// # Errors
    /// Returns error if format is invalid
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, String> {
        use crate::format::v2::AprV2ReaderRef;

        let reader = AprV2ReaderRef::from_bytes(&data)
            .map_err(|e| format!("Invalid APR file: {e}"))?;

        let meta = reader.metadata();

        // Build flat metadata from typed + custom fields
        let mut metadata = AprMetadata::new();
        if !meta.model_type.is_empty() {
            metadata.insert("model_type".to_string(), JsonValue::String(meta.model_type.clone()));
        }
        if let Some(ref name) = meta.name {
            metadata.insert("model_name".to_string(), JsonValue::String(name.clone()));
        }
        if let Some(ref desc) = meta.description {
            metadata.insert("description".to_string(), JsonValue::String(desc.clone()));
        }
        if let Some(ref author) = meta.author {
            metadata.insert("author".to_string(), JsonValue::String(author.clone()));
        }
        if let Some(ref license) = meta.license {
            metadata.insert("license".to_string(), JsonValue::String(license.clone()));
        }
        if let Some(ref version) = meta.version {
            metadata.insert("version".to_string(), JsonValue::String(version.clone()));
        }
        if let Some(ref arch) = meta.architecture {
            metadata.insert("architecture".to_string(), JsonValue::String(arch.clone()));
        }
        // Include custom fields
        for (k, v) in &meta.custom {
            metadata.insert(k.clone(), v.clone());
        }

        // Build tensor descriptors
        let tensor_names = reader.tensor_names();
        let mut tensors = Vec::new();
        for name in tensor_names {
            if let Some(entry) = reader.get_tensor(name) {
                tensors.push(AprTensorDescriptor {
                    name: entry.name.clone(),
                    dtype: format!("{:?}", entry.dtype),
                    shape: entry.shape.clone(),
                    offset: entry.offset as usize,
                    size: entry.size as usize,
                });
            }
        }

        Ok(Self {
            metadata,
            tensors,
            data,
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
        use crate::format::v2::AprV2ReaderRef;

        let reader = AprV2ReaderRef::from_bytes(&self.data)
            .map_err(|e| format!("Invalid APR file: {e}"))?;

        reader.get_f32_tensor(name)
            .ok_or_else(|| format!("Tensor not found or not F32: {name}"))
    }
}

/// APR format writer — delegates to `AprV2Writer` for ONE format.
///
/// This is a convenience wrapper around `AprV2Writer` that provides a simple
/// key-value metadata API. All output uses the canonical APR format (v2 binary
/// layout with header, metadata, tensor index, and aligned data sections).
#[derive(Debug, Default)]
pub struct AprWriter {
    /// Metadata key-value pairs (stored in AprV2Metadata.custom)
    metadata: AprMetadata,
    /// Tensors: (name, shape, f32 data)
    tensors: Vec<(String, Vec<usize>, Vec<f32>)>,
}

impl AprWriter {
    /// Create new writer
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set metadata key-value pair
    pub fn set_metadata(&mut self, key: impl Into<String>, value: JsonValue) {
        self.metadata.insert(key.into(), value);
    }

    /// Add tensor with f32 data
    pub fn add_tensor_f32(&mut self, name: impl Into<String>, shape: Vec<usize>, data: &[f32]) {
        self.tensors.push((name.into(), shape, data.to_vec()));
    }

    /// Write to bytes using the canonical APR format.
    ///
    /// # Errors
    /// Returns error if serialization fails
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        use crate::format::v2::{AprV2Metadata, AprV2Writer as V2Writer};

        // Build AprV2Metadata from our simple key-value metadata
        let mut v2_meta = AprV2Metadata::default();

        // Map well-known keys to typed fields, rest goes to custom
        for (key, value) in &self.metadata {
            match key.as_str() {
                "model_type" => {
                    if let Some(s) = value.as_str() {
                        v2_meta.model_type = s.to_string();
                    }
                }
                "model_name" => v2_meta.name = value.as_str().map(String::from),
                "description" => v2_meta.description = value.as_str().map(String::from),
                "author" => v2_meta.author = value.as_str().map(String::from),
                "license" => v2_meta.license = value.as_str().map(String::from),
                "version" => v2_meta.version = value.as_str().map(String::from),
                "architecture" => v2_meta.architecture = value.as_str().map(String::from),
                _ => {
                    v2_meta.custom.insert(key.clone(), value.clone());
                }
            }
        }

        let mut writer = V2Writer::new(v2_meta);

        for (name, shape, data) in &self.tensors {
            writer.add_f32_tensor(name, shape.clone(), data);
        }

        writer
            .write()
            .map_err(|e| format!("APR serialization failed: {e}"))
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

mod mod_part_02;

#[cfg(test)]
mod tests;
