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

        let reader =
            AprV2ReaderRef::from_bytes(&data).map_err(|e| format!("Invalid APR file: {e}"))?;

        let meta = reader.metadata();

        // Build flat metadata from typed + custom fields
        let mut metadata = AprMetadata::new();
        if !meta.model_type.is_empty() {
            metadata.insert(
                "model_type".to_string(),
                JsonValue::String(meta.model_type.clone()),
            );
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

    /// Load APR file, keeping only tensors that pass the filter predicate.
    ///
    /// This is the primary mechanism for inference readers to skip `__training__.*`
    /// tensors when loading checkpoint files (F-CKPT-016).
    ///
    /// # Example
    /// ```no_run
    /// # use aprender::serialization::apr::AprReader;
    /// // Load only inference tensors, skip training state
    /// let reader = AprReader::open_filtered("model.ckpt.apr", |name| {
    ///     !name.starts_with("__training__.")
    /// }).unwrap();
    /// ```
    ///
    /// # Errors
    /// Returns error if file is invalid or cannot be read
    pub fn open_filtered<P, F>(path: P, filter: F) -> Result<Self, String>
    where
        P: AsRef<Path>,
        F: Fn(&str) -> bool,
    {
        let data = fs::read(path).map_err(|e| format!("Failed to read file: {e}"))?;
        Self::from_bytes_filtered(data, filter)
    }

    /// Parse APR format from bytes, keeping only tensors that pass the filter.
    ///
    /// # Errors
    /// Returns error if format is invalid
    pub fn from_bytes_filtered<F>(data: Vec<u8>, filter: F) -> Result<Self, String>
    where
        F: Fn(&str) -> bool,
    {
        let mut reader = Self::from_bytes(data)?;
        reader.tensors.retain(|t| filter(&t.name));
        Ok(reader)
    }

    /// Get metadata value by key
    #[must_use]
    pub fn get_metadata(&self, key: &str) -> Option<&JsonValue> {
        self.metadata.get(key)
    }

    /// Read tensor data as f32 values (F32 dtype only)
    ///
    /// # Errors
    /// Returns error if tensor not found or not F32 dtype
    pub fn read_tensor_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        use crate::format::v2::AprV2ReaderRef;

        let reader =
            AprV2ReaderRef::from_bytes(&self.data).map_err(|e| format!("Invalid APR file: {e}"))?;

        reader
            .get_f32_tensor(name)
            .ok_or_else(|| format!("Tensor not found or not F32: {name}"))
    }

    /// Read tensor data as f32, dequantizing from any supported dtype (F32, F16, Q4K, Q6K, Q8, Q4).
    ///
    /// Unlike `read_tensor_f32` which only handles F32 dtype, this method handles
    /// all stored formats — essential for loading imported models stored as F16/Q4K.
    ///
    /// # Errors
    /// Returns error if tensor not found or dtype not supported
    pub fn read_tensor_as_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        use crate::format::v2::AprV2ReaderRef;

        let reader =
            AprV2ReaderRef::from_bytes(&self.data).map_err(|e| format!("Invalid APR file: {e}"))?;

        reader
            .get_tensor_as_f32(name)
            .ok_or_else(|| format!("Tensor '{name}' not found or unsupported dtype"))
    }

    /// Read tensor and validate it contains no NaN/Inf (F-CKPT-013).
    ///
    /// # Errors
    /// Returns error if tensor not found, not F32, or contains NaN/Inf
    pub fn read_tensor_f32_checked(&self, name: &str) -> Result<Vec<f32>, String> {
        let data = self.read_tensor_f32(name)?;
        for (i, &v) in data.iter().enumerate() {
            if !v.is_finite() {
                return Err(format!(
                    "F-CKPT-013: tensor '{name}' contains non-finite value at index {i}: {v}"
                ));
            }
        }
        Ok(data)
    }

    /// Validate that a tensor's element count matches an expected shape (F-CKPT-014).
    ///
    /// # Errors
    /// Returns error if tensor shape doesn't match expected dimensions
    pub fn validate_tensor_shape(
        &self,
        name: &str,
        expected_elements: usize,
    ) -> Result<(), String> {
        let desc = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("F-CKPT-014: tensor '{name}' not found"))?;
        let actual_elements: usize = desc.shape.iter().product();
        if actual_elements != expected_elements {
            return Err(format!(
                "F-CKPT-014: tensor '{name}' shape mismatch: \
                 expected {expected_elements} elements, got {actual_elements} (shape {:?})",
                desc.shape,
            ));
        }
        Ok(())
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

    /// Add tensor with f32 data (copies the slice)
    pub fn add_tensor_f32(&mut self, name: impl Into<String>, shape: Vec<usize>, data: &[f32]) {
        self.tensors.push((name.into(), shape, data.to_vec()));
    }

    /// ALB-099: Add tensor with owned f32 data — zero-copy for large tensors.
    /// Use this when the caller already owns a Vec<f32> to avoid a redundant copy.
    pub fn add_tensor_f32_owned(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        data: Vec<f32>,
    ) {
        self.tensors.push((name.into(), shape, data));
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

    /// Write to file atomically (F-CKPT-009).
    ///
    /// Writes to a `.tmp` file, fsyncs, then renames. A crash at any point
    /// leaves the original file intact.
    ///
    /// # Errors
    /// Returns error if write fails
    pub fn write<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        use std::io::Write;

        let path = path.as_ref();
        let bytes = self.to_bytes()?;

        // F-CKPT-009: Atomic write via tmp+fsync+rename
        let tmp_path = path.with_extension("apr.tmp");
        let mut file =
            fs::File::create(&tmp_path).map_err(|e| format!("Failed to create temp file: {e}"))?;
        file.write_all(&bytes)
            .map_err(|e| format!("Failed to write temp file: {e}"))?;
        file.sync_all()
            .map_err(|e| format!("Failed to fsync temp file: {e}"))?;
        drop(file);

        fs::rename(&tmp_path, path).map_err(|e| format!("Failed to rename temp file: {e}"))?;

        Ok(())
    }
}

mod crc32;

#[cfg(test)]
mod tests;
