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
    /// Byte offset to the data section (from header), cached at construction
    /// to avoid re-parsing on every tensor read (ALB-104).
    data_offset: usize,
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

        let data_offset = reader.header().data_offset as usize;

        Ok(Self {
            metadata,
            tensors,
            data,
            data_offset,
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

    /// Get raw tensor bytes by name using cached data_offset (ALB-104).
    ///
    /// Avoids re-parsing the entire APR file header/index on every tensor read.
    fn get_tensor_bytes(&self, name: &str) -> Result<(&AprTensorDescriptor, &[u8]), String> {
        let desc = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("Tensor not found: {name}"))?;
        let start = self.data_offset + desc.offset;
        let end = start + desc.size;
        if end > self.data.len() {
            return Err(format!(
                "Tensor '{name}' data out of bounds: {end} > {}",
                self.data.len()
            ));
        }
        Ok((desc, &self.data[start..end]))
    }

    /// Read tensor data as f32 values (F32 dtype only)
    ///
    /// # Errors
    /// Returns error if tensor not found or not F32 dtype
    pub fn read_tensor_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let (desc, bytes) = self.get_tensor_bytes(name)?;
        if desc.dtype != "F32" {
            return Err(format!(
                "Tensor not found or not F32: {name} (dtype={})",
                desc.dtype
            ));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Read tensor data as f32, dequantizing from any supported dtype (F32, F16, Q4K, Q6K, Q8, Q4).
    ///
    /// Unlike `read_tensor_f32` which only handles F32 dtype, this method handles
    /// all stored formats — essential for loading imported models stored as F16/Q4K.
    ///
    /// # Errors
    /// Returns error if tensor not found or dtype not supported
    pub fn read_tensor_as_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        use crate::format::gguf::dequant::{dequantize_q4_k, dequantize_q6_k};

        let (desc, bytes) = self.get_tensor_bytes(name)?;
        let element_count: usize = desc.shape.iter().product();

        match desc.dtype.as_str() {
            "F32" => Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()),
            "F16" => Ok(bytes
                .chunks_exact(2)
                .map(|c| trueno::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()),
            "BF16" => Ok(bytes
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits(u32::from(bits) << 16)
                })
                .collect()),
            "Q8" => {
                if bytes.len() < 4 {
                    return Err(format!("Tensor '{name}' Q8 data too short"));
                }
                let scale = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                Ok(bytes[4..]
                    .iter()
                    .map(|&b| f32::from(b as i8) * scale)
                    .collect())
            }
            "Q4" => Ok(Self::dequantize_q4_inline(bytes, element_count)),
            "Q4K" => dequantize_q4_k(bytes, 0, element_count)
                .map_err(|e| format!("Tensor '{name}' Q4K dequant failed: {e}")),
            "Q6K" => dequantize_q6_k(bytes, 0, element_count)
                .map_err(|e| format!("Tensor '{name}' Q6K dequant failed: {e}")),
            other => Err(format!(
                "Tensor '{name}' not found or unsupported dtype: {other}"
            )),
        }
    }

    /// Inline Q4 block dequantization (matches format::v2::dequantize_q4).
    fn dequantize_q4_inline(data: &[u8], element_count: usize) -> Vec<f32> {
        use crate::format::f16_safety::F16_MIN_NORMAL;

        const BLOCK_SIZE: usize = 32;
        let mut result = Vec::with_capacity(element_count);
        let mut pos = 0;
        let mut remaining = element_count;

        while remaining > 0 && pos + 2 <= data.len() {
            let scale_bits = u16::from_le_bytes([data[pos], data[pos + 1]]);
            let scale_raw = trueno::f16_to_f32(scale_bits);
            let scale = if scale_raw.is_nan()
                || scale_raw.is_infinite()
                || scale_raw.abs() < F16_MIN_NORMAL
            {
                0.0
            } else {
                scale_raw
            };
            pos += 2;

            let values_in_block = remaining.min(BLOCK_SIZE);
            for i in 0..values_in_block {
                let byte_idx = pos + i / 2;
                if byte_idx >= data.len() {
                    break;
                }
                let byte = data[byte_idx];
                let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                let q = (nibble as i8) - 8;
                result.push(f32::from(q) * scale);
            }

            pos += 16;
            remaining = remaining.saturating_sub(BLOCK_SIZE);
        }

        result.resize(element_count, 0.0);
        result
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

    /// Build `AprV2Metadata` from the flat key-value metadata map.
    ///
    /// Shared by `to_bytes()`, `into_bytes()`, and `write_into()` to avoid
    /// duplicating the well-known key → typed field mapping.
    fn build_v2_metadata(&self) -> crate::format::v2::AprV2Metadata {
        use crate::format::v2::AprV2Metadata;

        let mut v2_meta = AprV2Metadata::default();
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
        v2_meta
    }

    /// Write to bytes using the canonical APR format.
    ///
    /// # Errors
    /// Returns error if serialization fails
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        use crate::format::v2::AprV2Writer as V2Writer;

        let mut writer = V2Writer::new(self.build_v2_metadata());

        for (name, shape, data) in &self.tensors {
            writer.add_f32_tensor(name, shape.clone(), data);
        }

        writer
            .write()
            .map_err(|e| format!("APR serialization failed: {e}"))
    }

    /// ALB-099: Consume the writer and serialize — zero-copy for owned tensor data.
    /// Avoids the f32→bytes copy for tensors added via `add_tensor_f32_owned`.
    ///
    /// # Errors
    /// Returns error if serialization fails
    pub fn into_bytes(self) -> Result<Vec<u8>, String> {
        use crate::format::v2::AprV2Writer as V2Writer;

        let mut writer = V2Writer::new(self.build_v2_metadata());
        for (name, shape, data) in self.tensors {
            writer.add_tensor_f32_owned(name, shape, data);
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

    /// ALB-105: Consuming streaming write — streams tensor data to disk
    /// instead of serializing the entire model into a Vec<u8> then writing.
    ///
    /// Peak RAM = metadata + tensor index + largest single tensor (not all tensors).
    /// Same atomic write semantics as `write()` (F-CKPT-009): writes to a `.tmp`
    /// file via `AprV2StreamingWriter`, fsyncs, then renames.
    ///
    /// # Errors
    /// Returns error if write fails
    pub fn write_into<P: AsRef<Path>>(self, path: P) -> Result<(), String> {
        use crate::format::v2::AprV2StreamingWriter;

        let path = path.as_ref();

        // F-CKPT-009: Stream to tmp path, then rename for atomicity.
        // AprV2StreamingWriter::finalize() writes directly to the given path
        // (no internal tmp+rename), so we point it at the .tmp path and
        // rename ourselves after fsync.
        let tmp_path = path.with_extension("apr.tmp");

        let mut writer = AprV2StreamingWriter::new(self.build_v2_metadata())
            .map_err(|e| format!("Failed to create streaming writer: {e}"))?;

        for (name, shape, data) in self.tensors {
            writer
                .add_f32_tensor(name, shape, &data)
                .map_err(|e| format!("Failed to add tensor: {e}"))?;
        }

        writer
            .finalize(&tmp_path)
            .map_err(|e| format!("Failed to finalize streaming write: {e}"))?;

        // fsync the finalized file before rename
        let file = fs::File::open(&tmp_path)
            .map_err(|e| format!("Failed to open tmp file for fsync: {e}"))?;
        file.sync_all()
            .map_err(|e| format!("Failed to fsync tmp file: {e}"))?;
        drop(file);

        fs::rename(&tmp_path, path).map_err(|e| format!("Failed to rename temp file: {e}"))?;

        Ok(())
    }
}

mod crc32;

#[cfg(test)]
mod tests;
