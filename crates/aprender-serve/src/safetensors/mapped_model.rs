
// ============================================================================
// Zero-Copy Memory-Mapped SafeTensors Model (T-QA-020)
// ============================================================================

/// Zero-copy memory-mapped SafeTensors model container
///
/// Unlike `SafetensorsModel` which copies all tensor data to the heap,
/// `MappedSafeTensorsModel` uses memory-mapping (mmap) for true zero-copy
/// access to tensor data. This is critical for fast model loading (TTFT).
///
/// # Performance Characteristics
///
/// - **Loading time**: O(1) - only parses header/metadata, no data copy
/// - **Memory**: Only RSS grows as pages are accessed (demand paging)
/// - **TTFT target**: < 500ms for 3GB model
///
/// # Example
///
/// ```rust,ignore
/// let model = MappedSafeTensorsModel::load("model.safetensors")?;
/// let weights = model.get_tensor_bytes("layer1.weight")?;
/// // weights is a zero-copy slice into the mmap'd file
/// ```
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
pub struct MappedSafeTensorsModel {
    /// Memory-mapped file data
    mmap: memmap2::Mmap,
    /// File path (for diagnostics)
    path: std::path::PathBuf,
    /// Tensor metadata (parsed from header)
    tensors: HashMap<String, SafetensorsTensorInfo>,
    /// Offset where tensor data begins (after header + JSON metadata)
    data_offset: usize,
}

#[cfg(not(target_arch = "wasm32"))]
impl MappedSafeTensorsModel {
    /// Load a SafeTensors file with zero-copy memory mapping
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the SafeTensors file
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - File cannot be opened
    /// - Memory mapping fails
    /// - Header/metadata parsing fails
    ///
    /// # Performance
    ///
    /// This method is O(1) with respect to file size - only the header
    /// and JSON metadata are parsed. Tensor data is not touched until
    /// `get_tensor_bytes()` is called.
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Open file
        let file = std::fs::File::open(&path).map_err(|e| RealizarError::UnsupportedOperation {
            operation: "open_safetensors".to_string(),
            reason: format!("Failed to open file '{}': {}", path.display(), e),
        })?;

        // Memory-map the file (zero-copy)
        // SAFETY: File is opened read-only and we don't modify it
        let mmap = unsafe {
            memmap2::MmapOptions::new().map(&file).map_err(|e| {
                RealizarError::UnsupportedOperation {
                    operation: "mmap_safetensors".to_string(),
                    reason: format!("Failed to mmap file '{}': {}", path.display(), e),
                }
            })?
        };

        // Parse header (8-byte metadata length)
        if mmap.len() < 8 {
            return Err(RealizarError::UnsupportedOperation {
                operation: "parse_safetensors_header".to_string(),
                reason: format!(
                    "File too small: {} bytes (minimum 8 for header)",
                    mmap.len()
                ),
            });
        }

        let metadata_len =
            u64::from_le_bytes(mmap[0..8].try_into().expect("slice is exactly 8 bytes"));

        let metadata_len_usize =
            usize::try_from(metadata_len).map_err(|_| RealizarError::UnsupportedOperation {
                operation: "parse_safetensors_header".to_string(),
                reason: format!("Metadata length {} exceeds platform limit", metadata_len),
            })?;

        // Verify we have enough data for metadata
        let data_offset = 8 + metadata_len_usize;
        if mmap.len() < data_offset {
            return Err(RealizarError::UnsupportedOperation {
                operation: "parse_safetensors_header".to_string(),
                reason: format!(
                    "File truncated: need {} bytes for header+metadata, have {}",
                    data_offset,
                    mmap.len()
                ),
            });
        }

        // Parse JSON metadata (from mmap'd memory, no copy)
        let json_bytes = &mmap[8..data_offset];
        let tensors = Self::parse_metadata(json_bytes)?;

        // GH-213: Validate file covers all tensor data (catches truncated downloads)
        let max_tensor_end = tensors
            .values()
            .map(|t| t.data_offsets[1])
            .max()
            .unwrap_or(0);
        let required_size = data_offset + max_tensor_end;
        if mmap.len() < required_size {
            return Err(RealizarError::UnsupportedOperation {
                operation: "validate_safetensors_size".to_string(),
                reason: format!(
                    "SafeTensors file '{}' is truncated: file has {} bytes but tensor data \
                     requires {} bytes. The file may have been partially downloaded.",
                    path.display(),
                    mmap.len(),
                    required_size
                ),
            });
        }

        Ok(Self {
            mmap,
            path,
            tensors,
            data_offset,
        })
    }

    /// Parse JSON metadata from bytes
    fn parse_metadata(json_bytes: &[u8]) -> Result<HashMap<String, SafetensorsTensorInfo>> {
        // Parse JSON as generic Value first to handle __metadata__ and other special keys
        let json_value: serde_json::Value = serde_json::from_slice(json_bytes).map_err(|e| {
            RealizarError::UnsupportedOperation {
                operation: "parse_json".to_string(),
                reason: e.to_string(),
            }
        })?;

        let json_map =
            json_value
                .as_object()
                .ok_or_else(|| RealizarError::UnsupportedOperation {
                    operation: "parse_json".to_string(),
                    reason: "Expected JSON object".to_string(),
                })?;

        // Convert to SafetensorsTensorInfo, skipping special keys like __metadata__
        let mut tensors = HashMap::new();
        for (name, value) in json_map {
            // Skip metadata keys (start with __)
            if name.starts_with("__") {
                continue;
            }

            // Parse tensor metadata
            let meta: TensorMetadata = serde_json::from_value(value.clone()).map_err(|e| {
                RealizarError::UnsupportedOperation {
                    operation: "parse_tensor_metadata".to_string(),
                    reason: format!("Failed to parse tensor '{name}': {e}"),
                }
            })?;

            tensors.insert(
                name.clone(),
                SafetensorsTensorInfo {
                    name: name.clone(),
                    dtype: meta.dtype,
                    shape: meta.shape,
                    data_offsets: meta.data_offsets,
                },
            );
        }

        Ok(tensors)
    }

    /// Get raw tensor bytes (zero-copy slice into mmap'd file)
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name
    ///
    /// # Returns
    ///
    /// Zero-copy slice into the memory-mapped file. The slice is valid
    /// as long as `self` is alive.
    ///
    /// # Errors
    ///
    /// Returns error if tensor not found or offsets are invalid.
    pub fn get_tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| RealizarError::UnsupportedOperation {
                operation: "get_tensor_bytes".to_string(),
                reason: format!("Tensor '{name}' not found"),
            })?;

        // Integrity: declared byte length must equal product(shape) * dtype_size.
        // Fail closed on a crafted/corrupt file rather than silently returning a
        // wrong-sized slice (parity with the HF safetensors library). This is the
        // shared raw-read entry point, so every dtype variant inherits the check.
        tensor.validate_shape_matches_bytes()?;

        let [start, end] = tensor.data_offsets;
        let abs_start = self.data_offset + start;
        let abs_end = self.data_offset + end;

        if abs_end > self.mmap.len() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_bytes".to_string(),
                reason: format!(
                    "Tensor '{}' data offsets [{}, {}] exceed file size {}",
                    name,
                    abs_start,
                    abs_end,
                    self.mmap.len()
                ),
            });
        }

        Ok(&self.mmap[abs_start..abs_end])
    }

    /// Get tensor as F32 values (zero-copy bytes, then convert)
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name
    ///
    /// # Errors
    ///
    /// Returns error if tensor not found or dtype is not F32.
    pub fn get_tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| RealizarError::UnsupportedOperation {
                operation: "get_tensor_f32".to_string(),
                reason: format!("Tensor '{name}' not found"),
            })?;

        if tensor.dtype != SafetensorsDtype::F32 {
            return Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_f32".to_string(),
                reason: format!(
                    "Tensor '{}' has dtype {:?}, expected F32",
                    name, tensor.dtype
                ),
            });
        }

        let bytes = self.get_tensor_bytes(name)?;

        if !bytes.len().is_multiple_of(4) {
            return Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_f32".to_string(),
                reason: format!("Data size {} is not a multiple of 4", bytes.len()),
            });
        }

        let values = bytes
            .chunks_exact(4)
            .map(|chunk| {
                f32::from_le_bytes(
                    chunk
                        .try_into()
                        .expect("chunks_exact(4) guarantees 4-byte slices"),
                )
            })
            .collect();

        Ok(values)
    }

    /// Get tensor as BF16 bytes (zero-copy, native format)
    ///
    /// Returns raw BF16 bytes for native SIMD processing without
    /// F32 conversion at boot time.
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name
    ///
    /// # Errors
    ///
    /// Returns error if tensor not found or dtype is not BF16.
    pub fn get_tensor_bf16_bytes(&self, name: &str) -> Result<&[u8]> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| RealizarError::UnsupportedOperation {
                operation: "get_tensor_bf16_bytes".to_string(),
                reason: format!("Tensor '{name}' not found"),
            })?;

        if tensor.dtype != SafetensorsDtype::BF16 {
            return Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_bf16_bytes".to_string(),
                reason: format!(
                    "Tensor '{}' has dtype {:?}, expected BF16",
                    name, tensor.dtype
                ),
            });
        }

        self.get_tensor_bytes(name)
    }

    /// Get tensor as BF16 values converted to F32
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name
    ///
    /// # Errors
    ///
    /// Returns error if tensor not found or dtype is not BF16.
    pub fn get_tensor_bf16_as_f32(&self, name: &str) -> Result<Vec<f32>> {
        let bytes = self.get_tensor_bf16_bytes(name)?;

        // Convert BF16 bytes to F32 using SIMD-accelerated conversion
        // This provides 3-4x speedup over scalar conversion
        let values = simd_bf16_to_f32(bytes);

        Ok(values)
    }

    /// Get tensor as F16 bytes (zero-copy, native format)
    ///
    /// Returns raw F16 bytes for native SIMD processing.
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name
    ///
    /// # Errors
    ///
    /// Returns error if tensor not found or dtype is not F16.
    pub fn get_tensor_f16_bytes(&self, name: &str) -> Result<&[u8]> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| RealizarError::UnsupportedOperation {
                operation: "get_tensor_f16_bytes".to_string(),
                reason: format!("Tensor '{name}' not found"),
            })?;

        if tensor.dtype != SafetensorsDtype::F16 {
            return Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_f16_bytes".to_string(),
                reason: format!(
                    "Tensor '{}' has dtype {:?}, expected F16",
                    name, tensor.dtype
                ),
            });
        }

        self.get_tensor_bytes(name)
    }

    /// GH-174: Get tensor as raw FP16 u16 values (no conversion).
    ///
    /// Returns the native FP16 representation for direct GPU upload.
    /// Use with `CudaExecutor::load_weights_f16()` to bypass F32
    /// conversion entirely — 2x memory savings on GPU.
    pub fn get_tensor_f16_native(&self, name: &str) -> Result<Vec<u16>> {
        let bytes = self.get_tensor_f16_bytes(name)?;
        let values: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();
        Ok(values)
    }

    /// Get tensor as F16 values converted to F32
    pub fn get_tensor_f16_as_f32(&self, name: &str) -> Result<Vec<f32>> {
        let bytes = self.get_tensor_f16_bytes(name)?;

        let values: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                half::f16::from_bits(bits).to_f32()
            })
            .collect();

        Ok(values)
    }

    /// GH-174: Check if a tensor exists with FP16 dtype.
    ///
    /// Returns true if the tensor exists and has F16 dtype.
    /// Used by SafeTensors loader to decide between FP16 HGEMM
    /// and FP32 SGEMM dispatch paths.
    #[must_use]
    pub fn is_tensor_f16(&self, name: &str) -> bool {
        self.tensors
            .get(name)
            .is_some_and(|t| t.dtype == SafetensorsDtype::F16)
    }

    /// Get tensor as F32 with automatic dtype conversion
    ///
    /// Supports F32, F16, and BF16 dtypes with automatic conversion to F32.
    ///
    /// PMAT-789 (peak-RSS fix): the BF16/F16 expansion paths read the on-disk
    /// bytes from the read-only mmap and convert them into an owned `Vec<f32>`.
    /// Once the owned f32 copy exists the source mmap pages are no longer needed
    /// for this tensor. Left resident, those pages accumulate (≈0.92 GiB for a
    /// 0.5B BF16 model) alongside the ≈1.84 GiB f32 weights, inflating *peak*
    /// RSS during conversion by ~0.9 GiB even though steady-state (post-convert)
    /// holds only the f32 weights. After conversion we `madvise(MADV_DONTNEED)`
    /// the tensor's page range so the kernel reclaims those clean pages
    /// immediately. This is purely a residency hint — the returned f32 data is
    /// byte-for-byte unchanged, so inference output stays bit-identical.
    ///
    /// The F32 path is intentionally NOT released: its bytes are returned as a
    /// fresh `Vec<f32>` too, but a release there would be a no-net-win for the
    /// common all-BF16 model and risks re-faulting if any F32 tensor is read
    /// twice. The BF16/F16 source bytes are never re-read after load.
    pub fn get_tensor_auto(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| RealizarError::UnsupportedOperation {
                operation: "get_tensor_auto".to_string(),
                reason: format!("Tensor '{name}' not found"),
            })?;

        match tensor.dtype {
            SafetensorsDtype::F32 => self.get_tensor_f32(name),
            SafetensorsDtype::F16 => {
                let values = self.get_tensor_f16_as_f32(name)?;
                self.release_tensor_pages(name);
                Ok(values)
            }
            SafetensorsDtype::BF16 => {
                let values = self.get_tensor_bf16_as_f32(name)?;
                self.release_tensor_pages(name);
                Ok(values)
            }
            _ => Err(RealizarError::UnsupportedOperation {
                operation: "get_tensor_auto".to_string(),
                reason: format!("Unsupported dtype {:?} for tensor '{}'", tensor.dtype, name),
            }),
        }
    }

    /// PMAT-789: Advise the kernel that this tensor's mmap pages are no longer
    /// needed, so they are reclaimed from RSS immediately after the owned f32
    /// copy has been materialized.
    ///
    /// `MADV_DONTNEED` on a read-only file-backed mapping drops the resident
    /// (clean) pages; any later access transparently re-faults them from the
    /// file. After load no consumer re-reads the BF16/F16 source bytes (the
    /// owned f32 `Vec` is the single source of truth), so this never triggers a
    /// re-fault on the hot path and never changes returned data.
    ///
    /// Best-effort: failures are silently ignored — correctness never depends on
    /// the page being dropped, only peak RSS does. The advised range is shrunk
    /// to page boundaries (start rounded up, end rounded down) so we never
    /// advise-away a page partially shared with an adjacent tensor.
    #[cfg(unix)]
    fn release_tensor_pages(&self, name: &str) {
        let Some(tensor) = self.tensors.get(name) else {
            return;
        };
        let [start, end] = tensor.data_offsets;
        let abs_start = self.data_offset + start;
        let abs_end = self.data_offset + end;
        if abs_end > self.mmap.len() || abs_start >= abs_end {
            return;
        }

        let page = Self::page_size();
        let aligned_start = abs_start.div_ceil(page) * page;
        let aligned_end = (abs_end / page) * page;
        if aligned_start >= aligned_end {
            // Sub-page tensor span — nothing whole to drop safely.
            return;
        }

        // SAFETY: `MADV_DONTNEED` is only memory-unsafe for *writable* mappings
        // (it would discard un-flushed dirty data). This mapping is created
        // read-only (`MmapOptions::map`, never `map_mut`), so every page is
        // clean and file-backed; dropping a clean page is lossless and any later
        // access re-faults it from disk. The range is page-aligned and
        // bounds-checked above.
        #[allow(unsafe_code)]
        unsafe {
            let _ = self.mmap.unchecked_advise_range(
                memmap2::UncheckedAdvice::DontNeed,
                aligned_start,
                aligned_end - aligned_start,
            );
        }
    }

    /// No-op page release on non-unix platforms (no `madvise`).
    #[cfg(not(unix))]
    fn release_tensor_pages(&self, _name: &str) {}

    /// System page size for `madvise` alignment.
    #[cfg(unix)]
    fn page_size() -> usize {
        // SAFETY: `sysconf(_SC_PAGESIZE)` takes no pointers and cannot fault.
        #[allow(unsafe_code)]
        let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if p > 0 {
            p as usize
        } else {
            4096
        }
    }

    /// Get list of tensor names
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(String::as_str).collect()
    }

    /// Get tensor info by name
    #[must_use]
    pub fn get_tensor_info(&self, name: &str) -> Option<&SafetensorsTensorInfo> {
        self.tensors.get(name)
    }

    /// Check if model has a tensor with given name
    #[must_use]
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Get the file path
    #[must_use]
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// Get the total file size in bytes
    #[must_use]
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }

    /// Get the number of tensors
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}
