
impl LlamaTokenizer {
    /// Load tokenizer from GGUF file bytes.
    ///
    /// Extracts vocabulary from GGUF metadata:
    /// - `tokenizer.ggml.tokens` - vocabulary strings
    /// - `tokenizer.ggml.scores` - token priorities
    /// - `tokenizer.ggml.bos_token_id` - BOS token
    /// - `tokenizer.ggml.eos_token_id` - EOS token
    /// - `tokenizer.ggml.unknown_token_id` - UNK token
    ///
    /// # Errors
    /// Returns error if GGUF is invalid or missing tokenizer data.
    pub fn from_gguf_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 24 {
            return Err(AprenderError::FormatError {
                message: "GGUF data too short".to_string(),
            });
        }

        // Verify magic
        if data.get(0..4) != Some(b"GGUF".as_slice()) {
            return Err(AprenderError::FormatError {
                message: "Invalid GGUF magic".to_string(),
            });
        }

        // Parse header
        let metadata_count =
            u64::from_le_bytes(
                data[16..24]
                    .try_into()
                    .map_err(|_| AprenderError::FormatError {
                        message: "Failed to read metadata count".to_string(),
                    })?,
            ) as usize;

        // Parse metadata
        let mut offset = 24usize;
        let mut tokens: Option<Vec<String>> = None;
        let mut scores: Option<Vec<f32>> = None;
        let mut bos_token_id: u32 = 1;
        let mut eos_token_id: u32 = 2;
        let mut unk_token_id: u32 = 0;
        let mut tokenizer_model = TokenizerModel::SentencePiece; // Default

        for _ in 0..metadata_count {
            let (key, val_type, new_offset) = match Self::read_metadata_key(data, offset) {
                Some(v) => v,
                None => break,
            };
            offset = new_offset;

            offset = Self::apply_metadata_field(
                data, offset, &key, val_type,
                &mut tokens, &mut scores,
                &mut bos_token_id, &mut eos_token_id, &mut unk_token_id,
                &mut tokenizer_model,
            )?;
        }

        let tokens = tokens.ok_or_else(|| AprenderError::FormatError {
            message: "Missing tokenizer.ggml.tokens in GGUF".to_string(),
        })?;

        let scores = scores.unwrap_or_else(|| vec![0.0; tokens.len()]);

        let mut tokenizer = Self::new(tokens, scores, bos_token_id, eos_token_id, unk_token_id)?;
        tokenizer.set_model(tokenizer_model);
        Ok(tokenizer)
    }

    /// Read a GGUF metadata key and value type, returning (key, val_type, new_offset).
    fn read_metadata_key(data: &[u8], mut offset: usize) -> Option<(String, u32, usize)> {
        if offset + 8 > data.len() { return None; }
        let key_len = u64::from_le_bytes(data[offset..offset + 8].try_into().ok()?) as usize;
        offset += 8;
        if offset + key_len > data.len() { return None; }
        let key = String::from_utf8_lossy(&data[offset..offset + key_len]).to_string();
        offset += key_len;
        if offset + 4 > data.len() { return None; }
        let val_type = u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?);
        offset += 4;
        Some((key, val_type, offset))
    }

    /// Read a u32 from GGUF data at offset (val_type == 4).
    fn read_u32_field(data: &[u8], offset: usize, val_type: u32) -> Option<(u32, usize)> {
        if val_type == 4 && offset + 4 <= data.len() {
            let val = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap_or([0; 4]));
            Some((val, offset + 4))
        } else {
            None
        }
    }

    /// Read a string from GGUF data at offset (val_type == 8).
    fn read_string_field(data: &[u8], offset: usize, val_type: u32) -> Option<(String, usize)> {
        if val_type == 8 && offset + 8 <= data.len() {
            let str_len = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap_or([0; 8])) as usize;
            let new_offset = offset + 8;
            if new_offset + str_len <= data.len() {
                let s = String::from_utf8_lossy(&data[new_offset..new_offset + str_len]).to_string();
                return Some((s, new_offset + str_len));
            }
        }
        None
    }

    /// Try to read a u32 field, writing it to `target` if present. Returns new offset.
    fn try_apply_u32(data: &[u8], offset: usize, val_type: u32, target: &mut u32) -> Option<usize> {
        let (val, off) = Self::read_u32_field(data, offset, val_type)?;
        *target = val;
        Some(off)
    }

    /// Apply a single GGUF metadata field to tokenizer state.
    #[allow(clippy::too_many_arguments)]
    fn apply_metadata_field(
        data: &[u8],
        offset: usize,
        key: &str,
        val_type: u32,
        tokens: &mut Option<Vec<String>>,
        scores: &mut Option<Vec<f32>>,
        bos_token_id: &mut u32,
        eos_token_id: &mut u32,
        unk_token_id: &mut u32,
        tokenizer_model: &mut TokenizerModel,
    ) -> Result<usize> {
        match key {
            "tokenizer.ggml.tokens" if val_type == 9 => {
                let (arr, off) = Self::parse_string_array(data, offset)?;
                *tokens = Some(arr);
                Ok(off)
            }
            "tokenizer.ggml.scores" if val_type == 9 => {
                let (arr, off) = Self::parse_f32_array(data, offset)?;
                *scores = Some(arr);
                Ok(off)
            }
            "tokenizer.ggml.bos_token_id" => Ok(
                Self::try_apply_u32(data, offset, val_type, bos_token_id)
                    .unwrap_or_else(|| Self::skip_value(data, offset, val_type))
            ),
            "tokenizer.ggml.eos_token_id" => Ok(
                Self::try_apply_u32(data, offset, val_type, eos_token_id)
                    .unwrap_or_else(|| Self::skip_value(data, offset, val_type))
            ),
            "tokenizer.ggml.unknown_token_id" => Ok(
                Self::try_apply_u32(data, offset, val_type, unk_token_id)
                    .unwrap_or_else(|| Self::skip_value(data, offset, val_type))
            ),
            "tokenizer.ggml.model" => {
                if let Some((s, off)) = Self::read_string_field(data, offset, val_type) {
                    if s == "gpt2" { *tokenizer_model = TokenizerModel::Gpt2; }
                    return Ok(off);
                }
                Ok(Self::skip_value(data, offset, val_type))
            }
            _ => Ok(Self::skip_value(data, offset, val_type)),
        }
    }

    fn parse_string_array(data: &[u8], mut offset: usize) -> Result<(Vec<String>, usize)> {
        if offset + 12 > data.len() {
            return Err(AprenderError::FormatError {
                message: "Array header too short".to_string(),
            });
        }

        let elem_type = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap_or([0; 4]));
        offset += 4;

        if elem_type != 8 {
            return Err(AprenderError::FormatError {
                message: format!("Expected string array (type 8), got type {elem_type}"),
            });
        }

        let count =
            u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap_or([0; 8])) as usize;
        offset += 8;

        let mut result = Vec::with_capacity(count);

        for _ in 0..count {
            if offset + 8 > data.len() {
                break;
            }
            let str_len =
                u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap_or([0; 8])) as usize;
            offset += 8;

            if offset + str_len > data.len() {
                break;
            }
            let s = String::from_utf8_lossy(&data[offset..offset + str_len]).to_string();
            offset += str_len;
            result.push(s);
        }

        Ok((result, offset))
    }

    fn parse_f32_array(data: &[u8], mut offset: usize) -> Result<(Vec<f32>, usize)> {
        if offset + 12 > data.len() {
            return Err(AprenderError::FormatError {
                message: "Array header too short".to_string(),
            });
        }

        let elem_type = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap_or([0; 4]));
        offset += 4;

        if elem_type != 6 {
            return Err(AprenderError::FormatError {
                message: format!("Expected f32 array (type 6), got type {elem_type}"),
            });
        }

        let count =
            u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap_or([0; 8])) as usize;
        offset += 8;

        let mut result = Vec::with_capacity(count);

        for _ in 0..count {
            if offset + 4 > data.len() {
                break;
            }
            let f = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap_or([0; 4]));
            offset += 4;
            result.push(f);
        }

        Ok((result, offset))
    }

    /// Return the byte size of a single GGUF scalar value for the given type.
    fn gguf_scalar_size(val_type: u32) -> usize {
        match val_type {
            0 | 1 | 7 => 1,   // u8, i8, bool
            2 | 3 => 2,       // u16, i16
            4..=6 => 4,       // u32, i32, f32
            10..=12 => 8,     // u64, i64, f64
            _ => 0,
        }
    }

    /// Skip a GGUF string at `offset`, returning new offset.
    fn skip_gguf_string(data: &[u8], offset: usize) -> usize {
        if offset + 8 > data.len() { return offset; }
        let len = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap_or([0; 8])) as usize;
        offset + 8 + len
    }

    /// Skip a GGUF array at `offset`, returning new offset.
    fn skip_gguf_array(data: &[u8], mut offset: usize) -> usize {
        if offset + 12 > data.len() { return offset; }
        let elem_type = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap_or([0; 4]));
        offset += 4;
        let count = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap_or([0; 8])) as usize;
        offset += 8;

        let elem_size = Self::gguf_scalar_size(elem_type);
        if elem_size > 0 {
            offset += count * elem_size;
        } else if elem_type == 8 {
            for _ in 0..count {
                offset = Self::skip_gguf_string(data, offset);
            }
        }
        offset
    }

    fn skip_value(data: &[u8], offset: usize, val_type: u32) -> usize {
        let scalar = Self::gguf_scalar_size(val_type);
        if scalar > 0 { return offset + scalar; }
        match val_type {
            8 => Self::skip_gguf_string(data, offset),
            9 => Self::skip_gguf_array(data, offset),
            _ => offset,
        }
    }
}

// ============================================================================
// Tests (EXTREME TDD - Falsification Tests First)
// ============================================================================

#[cfg(test)]
mod tests;
