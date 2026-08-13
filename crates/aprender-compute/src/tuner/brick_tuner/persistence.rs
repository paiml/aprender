//! APR persistence and JSON serialization for BrickTuner.
//!
//! Implements the APR1 binary format (uncompressed) and JSON serialization
//! for saving/loading tuner models.

use super::super::error::TunerError;
use super::super::helpers::crc32_update;
use super::BrickTuner;

impl BrickTuner {
    /// APR format magic bytes (APR1 = uncompressed)
    const APR_MAGIC: [u8; 4] = *b"APR1";

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String, TunerError> {
        serde_json::to_string_pretty(self).map_err(|e| TunerError::Serialization(e.to_string()))
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self, TunerError> {
        serde_json::from_str(json).map_err(|e| TunerError::Serialization(e.to_string()))
    }

    // =========================================================================
    // APR Persistence (SOVEREIGN STACK - GH#81)
    // =========================================================================

    /// Get the default cache path for tuner models.
    ///
    /// Returns `~/.cache/trueno/tuner_model_v{VERSION}.apr`
    #[cfg(feature = "hardware-detect")]
    pub fn cache_path() -> std::path::PathBuf {
        let cache_dir =
            dirs::cache_dir().unwrap_or_else(|| std::path::PathBuf::from(".")).join("trueno");

        // Create directory if it doesn't exist
        let _ = std::fs::create_dir_all(&cache_dir);

        cache_dir.join(format!("tuner_model_v{}.apr", Self::VERSION))
    }

    /// Load tuner from cache or create new with defaults.
    ///
    /// This is the recommended way to create a BrickTuner for production use.
    /// It will:
    /// 1. Check for cached model at `~/.cache/trueno/tuner_model_v{VERSION}.apr`
    /// 2. Load if exists and version matches
    /// 3. Create new with defaults if not found or version mismatch
    #[cfg(feature = "hardware-detect")]
    pub fn load_or_default() -> Self {
        let path = Self::cache_path();

        if path.exists() {
            match Self::load_apr(&path) {
                Ok(tuner) => {
                    // Version check
                    if tuner.version == Self::VERSION {
                        return tuner;
                    }
                    // Version mismatch - create new
                }
                Err(_) => {
                    // Load failed - create new
                }
            }
        }

        Self::new()
    }

    /// Save tuner model to .apr file.
    ///
    /// APR1 format (uncompressed):
    /// - 4-byte magic: "APR1"
    /// - 4-byte metadata_len: u32 LE
    /// - JSON metadata
    /// - 4-byte CRC32: checksum
    pub fn save_apr<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), TunerError> {
        use std::io::Write;

        let json = self.to_json()?;
        let json_bytes = json.as_bytes();

        let mut file = std::fs::File::create(path).map_err(|e| TunerError::Io(e.to_string()))?;

        // Write magic
        file.write_all(&Self::APR_MAGIC).map_err(|e| TunerError::Io(e.to_string()))?;

        // Write metadata length
        let len = json_bytes.len() as u32;
        file.write_all(&len.to_le_bytes()).map_err(|e| TunerError::Io(e.to_string()))?;

        // Write JSON metadata
        file.write_all(json_bytes).map_err(|e| TunerError::Io(e.to_string()))?;

        // Calculate and write CRC32
        let mut crc = 0u32;
        crc = crc32_update(crc, &Self::APR_MAGIC);
        crc = crc32_update(crc, &len.to_le_bytes());
        crc = crc32_update(crc, json_bytes);
        file.write_all(&crc.to_le_bytes()).map_err(|e| TunerError::Io(e.to_string()))?;

        Ok(())
    }

    /// Load tuner model from .apr file.
    pub fn load_apr<P: AsRef<std::path::Path>>(path: P) -> Result<Self, TunerError> {
        use std::io::Read;

        let mut file = std::fs::File::open(path).map_err(|e| TunerError::Io(e.to_string()))?;

        // Read and verify magic
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).map_err(|e| TunerError::Io(e.to_string()))?;

        if magic != Self::APR_MAGIC {
            return Err(TunerError::InvalidFormat("Invalid APR magic bytes".to_string()));
        }

        // Read metadata length
        let mut len_bytes = [0u8; 4];
        file.read_exact(&mut len_bytes).map_err(|e| TunerError::Io(e.to_string()))?;
        let len = u32::from_le_bytes(len_bytes) as usize;

        // Read JSON metadata
        let mut json_bytes = vec![0u8; len];
        file.read_exact(&mut json_bytes).map_err(|e| TunerError::Io(e.to_string()))?;

        // Read and verify CRC32
        let mut crc_bytes = [0u8; 4];
        file.read_exact(&mut crc_bytes).map_err(|e| TunerError::Io(e.to_string()))?;
        let stored_crc = u32::from_le_bytes(crc_bytes);

        let mut computed_crc = 0u32;
        computed_crc = crc32_update(computed_crc, &Self::APR_MAGIC);
        computed_crc = crc32_update(computed_crc, &len_bytes);
        computed_crc = crc32_update(computed_crc, &json_bytes);

        if stored_crc != computed_crc {
            return Err(TunerError::InvalidFormat("CRC32 checksum mismatch".to_string()));
        }

        // Parse JSON
        let json =
            String::from_utf8(json_bytes).map_err(|e| TunerError::Serialization(e.to_string()))?;

        Self::from_json(&json)
    }

    /// Save to default cache path.
    #[cfg(feature = "hardware-detect")]
    pub fn save_to_cache(&self) -> Result<(), TunerError> {
        self.save_apr(Self::cache_path())
    }
}
