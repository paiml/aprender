
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
            if offset + 8 > data.len() {
                break;
            }

            // Read key
            let key_len = u64::from_le_bytes(data[offset..offset + 8].try_into().map_err(|_| {
                AprenderError::FormatError {
                    message: "Failed to read key length".to_string(),
                }
            })?) as usize;
            offset += 8;

            if offset + key_len > data.len() {
                break;
            }
            let key = String::from_utf8_lossy(&data[offset..offset + key_len]).to_string();
            offset += key_len;

            if offset + 4 > data.len() {
                break;
            }

            // Read value type
            let val_type =
                u32::from_le_bytes(data[offset..offset + 4].try_into().map_err(|_| {
                    AprenderError::FormatError {
                        message: "Failed to read value type".to_string(),
                    }
                })?);
            offset += 4;

            // Parse value based on type and key
            match key.as_str() {
                "tokenizer.ggml.tokens" => {
                    if val_type == 9 {
                        // Array
                        let (arr, new_offset) = Self::parse_string_array(data, offset)?;
                        tokens = Some(arr);
                        offset = new_offset;
                    }
                }
                "tokenizer.ggml.scores" => {
                    if val_type == 9 {
                        // Array
                        let (arr, new_offset) = Self::parse_f32_array(data, offset)?;
                        scores = Some(arr);
                        offset = new_offset;
                    }
                }
                "tokenizer.ggml.bos_token_id" => {
                    if val_type == 4 && offset + 4 <= data.len() {
                        bos_token_id = u32::from_le_bytes(
                            data[offset..offset + 4].try_into().unwrap_or([0; 4]),
                        );
                        offset += 4;
                    }
                }
                "tokenizer.ggml.eos_token_id" => {
                    if val_type == 4 && offset + 4 <= data.len() {
                        eos_token_id = u32::from_le_bytes(
                            data[offset..offset + 4].try_into().unwrap_or([0; 4]),
                        );
                        offset += 4;
                    }
                }
                "tokenizer.ggml.unknown_token_id" => {
                    if val_type == 4 && offset + 4 <= data.len() {
                        unk_token_id = u32::from_le_bytes(
                            data[offset..offset + 4].try_into().unwrap_or([0; 4]),
                        );
                        offset += 4;
                    }
                }
                "tokenizer.ggml.model" => {
                    // Detect tokenizer type: "gpt2" for GPT-2 BPE, "llama" for SentencePiece
                    if val_type == 8 && offset + 8 <= data.len() {
                        let str_len = u64::from_le_bytes(
                            data[offset..offset + 8].try_into().unwrap_or([0; 8]),
                        ) as usize;
                        offset += 8;
                        if offset + str_len <= data.len() {
                            let model_str =
                                String::from_utf8_lossy(&data[offset..offset + str_len]);
                            if model_str == "gpt2" {
                                tokenizer_model = TokenizerModel::Gpt2;
                            }
                            offset += str_len;
                        }
                    }
                }
                _ => {
                    // Skip other values
                    offset = Self::skip_value(data, offset, val_type);
                }
            }
        }

        let tokens = tokens.ok_or_else(|| AprenderError::FormatError {
            message: "Missing tokenizer.ggml.tokens in GGUF".to_string(),
        })?;

        let scores = scores.unwrap_or_else(|| vec![0.0; tokens.len()]);

        let mut tokenizer = Self::new(tokens, scores, bos_token_id, eos_token_id, unk_token_id)?;
        tokenizer.set_model(tokenizer_model);
        Ok(tokenizer)
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

    fn skip_value(data: &[u8], mut offset: usize, val_type: u32) -> usize {
        match val_type {
            0 | 1 | 7 => offset += 1, // u8, i8, bool
            2 | 3 => offset += 2,     // u16, i16
            4..=6 => offset += 4,     // u32, i32, f32
            8 => {
                // string
                if offset + 8 > data.len() {
                    return offset;
                }
                let len = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap_or([0; 8]))
                    as usize;
                offset += 8 + len;
            }
            9 => {
                // array
                if offset + 12 > data.len() {
                    return offset;
                }
                let elem_type =
                    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap_or([0; 4]));
                offset += 4;
                let count =
                    u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap_or([0; 8]))
                        as usize;
                offset += 8;

                match elem_type {
                    0 | 1 | 7 => offset += count,
                    2 | 3 => offset += count * 2,
                    4..=6 => offset += count * 4,
                    8 => {
                        for _ in 0..count {
                            if offset + 8 > data.len() {
                                break;
                            }
                            let len = u64::from_le_bytes(
                                data[offset..offset + 8].try_into().unwrap_or([0; 8]),
                            ) as usize;
                            offset += 8 + len;
                        }
                    }
                    10..=12 => offset += count * 8,
                    _ => {}
                }
            }
            10..=12 => offset += 8, // u64, i64, f64
            _ => {}
        }
        offset
    }
}

// ============================================================================
// Tests (EXTREME TDD - Falsification Tests First)
// ============================================================================

#[cfg(test)]
mod tests;
