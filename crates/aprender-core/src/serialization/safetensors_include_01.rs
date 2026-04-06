
impl MappedSafeTensors {
    /// Open a `SafeTensors` file with memory mapping.
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be opened or format is invalid.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let mmap = MappedFile::open(path).map_err(|e| format!("mmap failed: {e}"))?;
        let bytes = mmap.as_slice();

        let metadata_len = validate_and_read_header(bytes)?;
        let (metadata, user_metadata) = parse_metadata(bytes, metadata_len)?;
        let data_offset = 8 + metadata_len;

        Ok(Self {
            mmap,
            metadata,
            user_metadata,
            data_offset,
        })
    }

    /// Get the offset where tensor data begins (after header + metadata JSON).
    #[must_use]
    pub fn data_offset(&self) -> usize {
        self.data_offset
    }

    /// Get tensor metadata by name.
    #[must_use]
    pub fn get_metadata(&self, name: &str) -> Option<&TensorMetadata> {
        self.metadata.get(name)
    }

    /// Get all tensor names.
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.metadata.keys().map(String::as_str).collect()
    }

    /// Extract tensor data as f32 values (BF16/F16 are converted to F32).
    ///
    /// # Errors
    ///
    /// Returns error if tensor not found or data is invalid.
    pub fn get_tensor(&self, name: &str) -> Result<Vec<f32>, String> {
        let meta = self
            .metadata
            .get(name)
            .ok_or_else(|| format!("Tensor '{name}' not found"))?;

        let bytes = self.mmap.as_slice();
        let [start, end] = meta.data_offsets;
        let abs_start = self.data_offset + start;
        let abs_end = self.data_offset + end;

        if abs_end > bytes.len() {
            return Err(format!(
                "Tensor '{name}' data out of bounds: {abs_end} > {}",
                bytes.len()
            ));
        }

        let tensor_bytes = &bytes[abs_start..abs_end];

        // Handle different dtypes
        match meta.dtype.as_str() {
            "F32" => extract_f32(tensor_bytes),
            "BF16" => extract_bf16_to_f32(tensor_bytes),
            "F16" => extract_f16_to_f32(tensor_bytes),
            other => Err(format!("Unsupported dtype for '{name}': {other}")),
        }
    }

    /// Get raw tensor bytes (zero-copy).
    #[must_use]
    pub fn get_tensor_bytes(&self, name: &str) -> Option<&[u8]> {
        let meta = self.metadata.get(name)?;
        let [start, end] = meta.data_offsets;
        let abs_start = self.data_offset + start;
        let abs_end = self.data_offset + end;

        self.mmap.slice(abs_start, abs_end)
    }

    /// GH-205 FIX: Get tensor with original dtype preserved (no F16→F32 conversion).
    ///
    /// Returns the raw tensor bytes along with dtype and shape information.
    /// This enables F16 passthrough: SafeTensors F16 → APR F16 without precision loss.
    ///
    /// For F32 tensors, returns f32 data directly.
    /// For F16/BF16 tensors, returns the raw bytes without conversion.
    ///
    /// # Errors
    ///
    /// Returns error if tensor not found or data is invalid.
    pub fn get_tensor_raw(&self, name: &str) -> Result<RawTensorData, String> {
        let meta = self
            .metadata
            .get(name)
            .ok_or_else(|| format!("Tensor '{name}' not found"))?;

        let bytes = self.mmap.as_slice();
        let [start, end] = meta.data_offsets;
        let abs_start = self.data_offset + start;
        let abs_end = self.data_offset + end;

        if abs_end > bytes.len() {
            return Err(format!(
                "Tensor '{name}' data out of bounds: {abs_end} > {}",
                bytes.len()
            ));
        }

        let tensor_bytes = &bytes[abs_start..abs_end];

        // Parse dtype string to enum
        let dtype = match meta.dtype.as_str() {
            "F32" => SafeTensorsDType::F32,
            "F16" => SafeTensorsDType::F16,
            "BF16" => SafeTensorsDType::BF16,
            other => return Err(format!("Unsupported dtype for '{name}': {other}")),
        };

        Ok(RawTensorData {
            dtype,
            shape: meta.shape.clone(),
            bytes: tensor_bytes.to_vec(),
        })
    }

    /// Number of tensors in the file.
    #[must_use]
    pub fn len(&self) -> usize {
        self.metadata.len()
    }

    /// Check if file has no tensors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.metadata.is_empty()
    }

    /// Get user metadata from `__metadata__` header section (PMAT-223).
    ///
    /// SafeTensors files may contain arbitrary string→string metadata under
    /// the `__metadata__` key. This method exposes that data for preservation
    /// during format conversion.
    #[must_use]
    pub fn user_metadata(&self) -> &UserMetadata {
        &self.user_metadata
    }

    /// PMAT-260: Extract original dtype for each tensor.
    ///
    /// Returns a map of tensor name → dtype string (e.g., "F32", "F16", "BF16").
    /// Used by the export pipeline to preserve original dtypes during round-trip.
    #[must_use]
    pub fn dtype_map(&self) -> BTreeMap<String, String> {
        self.metadata
            .iter()
            .map(|(name, meta)| (name.clone(), meta.dtype.clone()))
            .collect()
    }
}

#[path = "safetensors_reader.rs"]
mod safetensors_reader;
pub use safetensors_reader::extract_tensor;
use safetensors_reader::{
    extract_bf16_to_f32, extract_f16_to_f32, extract_f32, parse_metadata, validate_and_read_header,
};

#[cfg(test)]
#[path = "safetensors_tests.rs"]
mod safetensors_tests;
