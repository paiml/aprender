impl GGUFModel {
    /// Parse GGUF file from bytes
    ///
    /// # Arguments
    ///
    /// * `data` - Raw GGUF file bytes
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Invalid magic number
    /// - Unsupported version
    /// - Malformed data
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let data = std::fs::read("model.gguf")?;
    /// let model = GGUFModel::from_bytes(&data)?;
    /// println!("Loaded {} tensors", model.tensors.len());
    /// ```
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(data);

        // Parse header
        let header = Self::parse_header(&mut cursor)?;

        // Parse metadata
        let metadata = Self::parse_metadata(&mut cursor, header.metadata_count)?;

        // Parse tensor info
        let tensors = Self::parse_tensor_info(&mut cursor, header.tensor_count)?;

        // Calculate tensor data start with 32-byte alignment
        let current_pos = cursor.position() as usize;
        let tensor_data_start = current_pos.div_ceil(GGUF_ALIGNMENT) * GGUF_ALIGNMENT;

        Ok(Self {
            header,
            metadata,
            tensors,
            tensor_data_start,
        })
    }

    /// Parse GGUF header
    fn parse_header(cursor: &mut Cursor<&[u8]>) -> Result<GGUFHeader> {
        let mut buf = [0u8; 4];

        // Read magic
        cursor
            .read_exact(&mut buf)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_magic".to_string(),
                reason: e.to_string(),
            })?;
        let magic = u32::from_le_bytes(buf);

        if magic != GGUF_MAGIC {
            return Err(RealizarError::InvalidShape {
                reason: format!("Invalid GGUF magic: 0x{magic:08X}, expected 0x{GGUF_MAGIC:08X}"),
            });
        }

        // Read version
        cursor
            .read_exact(&mut buf)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_version".to_string(),
                reason: e.to_string(),
            })?;
        let version = u32::from_le_bytes(buf);

        // GH-310: Accept both v2 and v3 — same format structure, v3 just adds
        // big-endian support (all models in practice are little-endian).
        if version != GGUF_VERSION_V3 && version != GGUF_VERSION_V2 {
            return Err(RealizarError::UnsupportedOperation {
                operation: "parse_gguf".to_string(),
                reason: format!("Unsupported GGUF version: {version}, only v2/v3 supported"),
            });
        }

        // Read tensor_count
        let mut buf8 = [0u8; 8];
        cursor
            .read_exact(&mut buf8)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_tensor_count".to_string(),
                reason: e.to_string(),
            })?;
        let tensor_count = u64::from_le_bytes(buf8);

        // Bounds check: Prevent allocation attacks from corrupted headers
        // Reasonable limit: no model has >100,000 tensors
        const MAX_TENSOR_COUNT: u64 = 100_000;
        if tensor_count > MAX_TENSOR_COUNT {
            return Err(RealizarError::UnsupportedOperation {
                operation: "parse_gguf".to_string(),
                reason: format!(
                    "tensor_count {} exceeds maximum allowed {} (corrupted header?)",
                    tensor_count, MAX_TENSOR_COUNT
                ),
            });
        }

        // Read metadata_count
        cursor
            .read_exact(&mut buf8)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_metadata_count".to_string(),
                reason: e.to_string(),
            })?;
        let metadata_count = u64::from_le_bytes(buf8);

        // Bounds check: Prevent allocation attacks
        // Reasonable limit: no model has >10,000 metadata entries
        const MAX_METADATA_COUNT: u64 = 10_000;
        if metadata_count > MAX_METADATA_COUNT {
            return Err(RealizarError::UnsupportedOperation {
                operation: "parse_gguf".to_string(),
                reason: format!(
                    "metadata_count {} exceeds maximum allowed {} (corrupted header?)",
                    metadata_count, MAX_METADATA_COUNT
                ),
            });
        }

        Ok(GGUFHeader {
            magic,
            version,
            tensor_count,
            metadata_count,
        })
    }

    /// Parse metadata key-value pairs
    fn parse_metadata(
        cursor: &mut Cursor<&[u8]>,
        count: u64,
    ) -> Result<HashMap<String, GGUFValue>> {
        let mut metadata = HashMap::new();

        for _ in 0..count {
            // Read key (string: u64 length + bytes)
            let key = Self::read_string(cursor)?;

            // Read value type (u32)
            let mut buf = [0u8; 4];
            cursor
                .read_exact(&mut buf)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "read_metadata_type".to_string(),
                    reason: e.to_string(),
                })?;
            let value_type = u32::from_le_bytes(buf);

            // Read value based on type
            let value = Self::read_value(cursor, value_type)?;

            metadata.insert(key, value);
        }

        Ok(metadata)
    }

    /// Read a string: u64 length + UTF-8 bytes
    fn read_string(cursor: &mut Cursor<&[u8]>) -> Result<String> {
        let mut buf8 = [0u8; 8];
        cursor
            .read_exact(&mut buf8)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_string_length".to_string(),
                reason: e.to_string(),
            })?;
        let len_u64 = u64::from_le_bytes(buf8);
        let len = usize::try_from(len_u64).map_err(|_| RealizarError::UnsupportedOperation {
            operation: "convert_string_length".to_string(),
            reason: format!("String length {len_u64} exceeds platform usize limit"),
        })?;

        let mut string_bytes = vec![0u8; len];
        cursor
            .read_exact(&mut string_bytes)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "read_string_data".to_string(),
                reason: e.to_string(),
            })?;

        String::from_utf8(string_bytes).map_err(|e| RealizarError::UnsupportedOperation {
            operation: "parse_utf8".to_string(),
            reason: e.to_string(),
        })
    }

    /// Read a value based on type
    fn read_value(cursor: &mut Cursor<&[u8]>, value_type: u32) -> Result<GGUFValue> {
        match value_type {
            0 => Ok(GGUFValue::UInt8(Self::read_u8(cursor)?)),
            1 => Ok(GGUFValue::Int8(Self::read_i8(cursor)?)),
            2 => Ok(GGUFValue::UInt16(Self::read_u16(cursor)?)),
            3 => Ok(GGUFValue::Int16(Self::read_i16(cursor)?)),
            4 => Ok(GGUFValue::UInt32(Self::read_u32(cursor)?)),
            5 => Ok(GGUFValue::Int32(Self::read_i32(cursor)?)),
            6 => Ok(GGUFValue::Float32(Self::read_f32(cursor)?)),
            7 => Ok(GGUFValue::Bool(Self::read_bool(cursor)?)),
            8 => Ok(GGUFValue::String(Self::read_string(cursor)?)),
            9 => {
                // Array: element_type (u32) + array_len (u64) + elements
                let element_type = Self::read_u32(cursor)?;
                let array_len = Self::read_u64(cursor)?;

                // Bounds check: Limit array length to prevent allocation attacks
                const MAX_ARRAY_LEN: u64 = 10_000_000; // 10M elements max
                if array_len > MAX_ARRAY_LEN {
                    return Err(RealizarError::InvalidShape {
                        reason: format!(
                            "Array length {} exceeds maximum {} (corrupted?)",
                            array_len, MAX_ARRAY_LEN
                        ),
                    });
                }

                // Safely convert array_len to usize
                let len = usize::try_from(array_len).map_err(|_| RealizarError::InvalidShape {
                    reason: format!("Array length too large: {array_len}"),
                })?;

                let mut elements = Vec::with_capacity(len);
                for _ in 0..array_len {
                    elements.push(Self::read_value(cursor, element_type)?);
                }
                Ok(GGUFValue::Array(elements))
            },
            10 => Ok(GGUFValue::UInt64(Self::read_u64(cursor)?)),
            11 => Ok(GGUFValue::Int64(Self::read_i64(cursor)?)),
            12 => Ok(GGUFValue::Float64(Self::read_f64(cursor)?)),
            _ => Err(RealizarError::UnsupportedOperation {
                operation: "read_value".to_string(),
                reason: format!("Unsupported value type: {value_type}"),
            }),
        }
    }

    // Primitive type readers delegated to io.rs (PMAT-COMPLY)
    fn read_u8(cursor: &mut Cursor<&[u8]>) -> Result<u8> {
        super::io::read_u8(cursor)
    }
    fn read_i8(cursor: &mut Cursor<&[u8]>) -> Result<i8> {
        super::io::read_i8(cursor)
    }
    fn read_u16(cursor: &mut Cursor<&[u8]>) -> Result<u16> {
        super::io::read_u16(cursor)
    }
    fn read_i16(cursor: &mut Cursor<&[u8]>) -> Result<i16> {
        super::io::read_i16(cursor)
    }
    fn read_u32(cursor: &mut Cursor<&[u8]>) -> Result<u32> {
        super::io::read_u32(cursor)
    }
    fn read_i32(cursor: &mut Cursor<&[u8]>) -> Result<i32> {
        super::io::read_i32(cursor)
    }
    fn read_f32(cursor: &mut Cursor<&[u8]>) -> Result<f32> {
        super::io::read_f32(cursor)
    }
    fn read_bool(cursor: &mut Cursor<&[u8]>) -> Result<bool> {
        super::io::read_bool(cursor)
    }
    fn read_u64(cursor: &mut Cursor<&[u8]>) -> Result<u64> {
        super::io::read_u64(cursor)
    }
    fn read_i64(cursor: &mut Cursor<&[u8]>) -> Result<i64> {
        super::io::read_i64(cursor)
    }
    fn read_f64(cursor: &mut Cursor<&[u8]>) -> Result<f64> {
        super::io::read_f64(cursor)
    }

    /// Parse tensor info
    fn parse_tensor_info(cursor: &mut Cursor<&[u8]>, count: u64) -> Result<Vec<TensorInfo>> {
        let mut tensors = Vec::new();

        for _ in 0..count {
            // Read tensor name (string)
            let name = Self::read_string(cursor)?;

            // Read n_dims (u32)
            let n_dims = Self::read_u32(cursor)?;

            // Bounds check: Tensors have at most 8 dimensions (typically 1-4)
            const MAX_DIMS: u32 = 8;
            if n_dims > MAX_DIMS {
                return Err(RealizarError::UnsupportedOperation {
                    operation: "parse_tensor_info".to_string(),
                    reason: format!(
                        "tensor '{}' has {} dimensions, max allowed is {} (corrupted?)",
                        name, n_dims, MAX_DIMS
                    ),
                });
            }

            // Read dimensions array
            // GGUF stores dimensions in GGML order (reversed from standard row-major)
            // We need to reverse them to get the correct shape [out_dim, in_dim]
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(Self::read_u64(cursor)?);
            }
            dims.reverse();

            // Read quantization type (u32)
            let qtype = Self::read_u32(cursor)?;

            // Read offset (u64)
            let offset = Self::read_u64(cursor)?;

            tensors.push(TensorInfo {
                name,
                n_dims,
                dims,
                qtype,
                offset,
            });
        }

        Ok(tensors)
    }
}
