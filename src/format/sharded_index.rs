
/// Parse error message to detect specific error types (GH-129)
#[cfg(feature = "hf-hub-integration")]
pub fn parse_import_error(error_msg: &str, resource: &str) -> ImportError {
    let msg_lower = error_msg.to_lowercase();

    // Check for 404 / not found
    if msg_lower.contains("404")
        || msg_lower.contains("not found")
        || msg_lower.contains("does not exist")
        || msg_lower.contains("no such")
    {
        return ImportError::NotFound {
            resource: resource.to_string(),
            status: 404,
        };
    }

    // Check for authentication / 401 / 403
    if msg_lower.contains("401")
        || msg_lower.contains("403")
        || msg_lower.contains("unauthorized")
        || msg_lower.contains("forbidden")
        || msg_lower.contains("gated")
        || msg_lower.contains("access denied")
    {
        return ImportError::AuthRequired {
            resource: resource.to_string(),
        };
    }

    // Check for rate limiting / 429
    if msg_lower.contains("429")
        || msg_lower.contains("rate limit")
        || msg_lower.contains("too many requests")
    {
        // Try to extract retry-after
        let retry_after = if let Some(pos) = msg_lower.find("retry") {
            msg_lower[pos..]
                .split_whitespace()
                .find_map(|s| s.parse::<u64>().ok())
        } else {
            None
        };
        return ImportError::RateLimited { retry_after };
    }

    // Default to download failed
    ImportError::DownloadFailed {
        source: resource.to_string(),
        reason: error_msg.to_string(),
    }
}

// ============================================================================
// GH-127: Sharded Model Support
// ============================================================================

/// Parsed sharded model index (model.safetensors.index.json)
///
/// `HuggingFace` uses this format for large models split across multiple shards.
/// Example: Llama-2-7b has 2 shards, Llama-2-70b has 15 shards.
#[derive(Debug, Clone)]
pub struct ShardedIndex {
    /// Map of tensor name → shard filename
    weight_map: std::collections::HashMap<String, String>,
    /// Optional total size in bytes
    total_size: Option<u64>,
}

impl ShardedIndex {
    /// Parse a sharded index from JSON string
    ///
    /// # Example JSON format
    /// ```json
    /// {
    ///   "metadata": {"total_size": 14000000000},
    ///   "weight_map": {
    ///     "model.encoder.weight": "model-00001-of-00002.safetensors",
    ///     "model.decoder.weight": "model-00002-of-00002.safetensors"
    ///   }
    /// }
    /// ```
    pub fn parse(json: &str) -> Result<Self> {
        // Minimal JSON parsing without serde dependency
        // Look for "weight_map" key and parse the object

        let json = json.trim();
        if !json.starts_with('{') || !json.ends_with('}') {
            return Err(AprenderError::FormatError {
                message: "Invalid JSON: expected object".to_string(),
            });
        }

        // Find weight_map section
        let weight_map_start =
            json.find("\"weight_map\"")
                .ok_or_else(|| AprenderError::FormatError {
                    message: "Missing 'weight_map' key in index.json".to_string(),
                })?;

        // Parse weight_map object
        let after_key = &json[weight_map_start + 12..]; // Skip "weight_map"
        let obj_start = after_key
            .find('{')
            .ok_or_else(|| AprenderError::FormatError {
                message: "Invalid weight_map: expected object".to_string(),
            })?;

        let obj_content = &after_key[obj_start..];
        let mut weight_map = std::collections::HashMap::new();
        let mut depth = 0;
        let mut obj_end = 0;

        for (i, c) in obj_content.char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        obj_end = i;
                        break;
                    }
                }
                _ => {}
            }
        }

        let inner = &obj_content[1..obj_end];

        // Parse key-value pairs: "tensor_name": "shard_file"
        for pair in inner.split(',') {
            let pair = pair.trim();
            if pair.is_empty() {
                continue;
            }

            let parts: Vec<&str> = pair.splitn(2, ':').collect();
            if parts.len() == 2 {
                let key = parts[0].trim().trim_matches('"');
                let val = parts[1].trim().trim_matches('"');
                if !key.is_empty() && !val.is_empty() {
                    weight_map.insert(key.to_string(), val.to_string());
                }
            }
        }

        // Parse optional total_size from metadata
        let total_size = json.find("\"total_size\"").and_then(|pos| {
            let after = &json[pos + 12..];
            let colon = after.find(':')?;
            let after_colon = after[colon + 1..].trim_start();
            let end = after_colon.find(|c: char| !c.is_ascii_digit())?;
            after_colon[..end].parse::<u64>().ok()
        });

        Ok(Self {
            weight_map,
            total_size,
        })
    }

    /// Number of unique shard files
    #[must_use]
    pub fn shard_count(&self) -> usize {
        let unique: std::collections::HashSet<_> = self.weight_map.values().collect();
        unique.len()
    }

    /// Number of tensors in the index
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }

    /// Total model size in bytes (if available)
    #[must_use]
    pub fn total_size(&self) -> Option<u64> {
        self.total_size
    }

    /// Get the shard file containing a specific tensor
    #[must_use]
    pub fn shard_for_tensor(&self, tensor_name: &str) -> Option<&str> {
        self.weight_map.get(tensor_name).map(String::as_str)
    }

    /// Get all tensor names in a specific shard
    #[must_use]
    pub fn tensors_in_shard(&self, shard_file: &str) -> Vec<&str> {
        self.weight_map
            .iter()
            .filter(|(_, v)| v.as_str() == shard_file)
            .map(|(k, _)| k.as_str())
            .collect()
    }

    /// Get sorted list of shard files
    #[must_use]
    pub fn shard_files(&self) -> Vec<&str> {
        let mut files: Vec<_> = self
            .weight_map
            .values()
            .map(String::as_str)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        files.sort_unstable();
        files
    }
}

/// Detect if a model directory contains a sharded model
///
/// Checks for `model.safetensors.index.json` which indicates sharding.
#[must_use]
pub fn detect_sharded_model(dir: &std::path::Path, base_name: &str) -> Option<PathBuf> {
    let index_name = format!("{base_name}.index.json");
    let index_path = dir.join(&index_name);

    if index_path.exists() {
        Some(index_path)
    } else {
        None
    }
}
