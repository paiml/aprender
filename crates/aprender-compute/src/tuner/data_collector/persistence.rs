//! Persistent training data storage (T-TUNER-003, GitHub #80).
//!
//! APR binary format: `MAGIC(4) + LEN(4) + JSON(n) + CRC32(4)`.

use crate::tuner::error::TunerError;
use crate::tuner::features::FeatureExtractor;
use crate::tuner::helpers::crc32_hash;

use super::types::TrainingSample;
use super::TunerDataCollector;

impl TunerDataCollector {
    // ========================================================================
    // T-TUNER-003: Persistent Training Data (GitHub #80)
    // ========================================================================

    /// Training data cache path
    #[cfg(feature = "hardware-detect")]
    pub fn cache_path() -> std::path::PathBuf {
        let hw_id = Self::hardware_id();
        dirs::cache_dir()
            .unwrap_or_else(|| std::path::PathBuf::from(".cache"))
            .join("trueno")
            .join(format!("training_data_{}.apr", hw_id))
    }

    /// Generate hardware fingerprint for hardware-specific models
    #[cfg(feature = "hardware-detect")]
    pub fn hardware_id() -> String {
        use crate::hardware::HardwareCapability;
        let hw = HardwareCapability::detect();

        // Create a stable fingerprint from hardware characteristics
        let fingerprint = format!(
            "{}-{:?}-{}-{}",
            hw.cpu.cores,
            hw.cpu.simd,
            hw.gpu.as_ref().map(|g| g.model.as_str()).unwrap_or("none"),
            hw.gpu.as_ref().map(|g| g.vram_gb as u32).unwrap_or(0),
        );

        // Hash to short hex string
        let hash = crc32_hash(fingerprint.as_bytes());
        format!("{:08x}", hash)
    }

    /// Load from cache or create empty
    #[cfg(feature = "hardware-detect")]
    pub fn load_or_create() -> Self {
        let path = Self::cache_path();
        if path.exists() {
            if let Ok(collector) = Self::load_apr(&path) {
                return collector;
            }
        }
        Self::new()
    }

    /// Save training data to APR format
    pub fn save_apr<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), TunerError> {
        use std::io::Write;

        // Ensure parent directory exists
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;
        }

        // Serialize samples to JSON
        let json = serde_json::to_string(&self.samples)
            .map_err(|e| TunerError::Serialization(e.to_string()))?;
        let json_bytes = json.as_bytes();

        // Create APR format: MAGIC + LEN + JSON + CRC32
        let mut file = std::fs::File::create(path.as_ref())
            .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;

        // Write magic bytes: "APR2" (version 2 for training data)
        file.write_all(b"APR2").map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;

        // Write length as u32 little-endian
        let len = json_bytes.len() as u32;
        file.write_all(&len.to_le_bytes())
            .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;

        // Write JSON
        file.write_all(json_bytes).map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;

        // Write CRC32 checksum
        let checksum = crc32_hash(json_bytes);
        file.write_all(&checksum.to_le_bytes())
            .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;

        Ok(())
    }

    /// Load training data from APR format
    pub fn load_apr<P: AsRef<std::path::Path>>(path: P) -> Result<Self, TunerError> {
        use std::io::Read;

        let mut file = std::fs::File::open(path.as_ref())
            .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;

        // Read and verify magic
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;
        if &magic != b"APR2" {
            return Err(TunerError::InvalidFormat(format!("Expected APR2 magic, got {:?}", magic)));
        }

        // Read length
        let mut len_bytes = [0u8; 4];
        file.read_exact(&mut len_bytes)
            .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;
        let len = u32::from_le_bytes(len_bytes) as usize;

        // Read JSON
        let mut json_bytes = vec![0u8; len];
        file.read_exact(&mut json_bytes)
            .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;

        // Read and verify CRC32
        let mut crc_bytes = [0u8; 4];
        file.read_exact(&mut crc_bytes)
            .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;
        let stored_crc = u32::from_le_bytes(crc_bytes);
        let computed_crc = crc32_hash(&json_bytes);

        if stored_crc != computed_crc {
            return Err(TunerError::InvalidFormat(format!(
                "CRC mismatch: stored={:08x}, computed={:08x}",
                stored_crc, computed_crc
            )));
        }

        // Deserialize samples
        let samples: Vec<TrainingSample> = serde_json::from_slice(&json_bytes)
            .map_err(|e| TunerError::Serialization(e.to_string()))?;

        Ok(Self {
            samples,
            extractor: FeatureExtractor::new(),
            retrain_threshold: 100,
            samples_at_last_train: 0,
            feedback: Vec::new(),
            online_learning_enabled: false,
            error_window: Vec::new(),
            error_window_size: Self::DEFAULT_ERROR_WINDOW_SIZE,
        })
    }

    /// Append a sample to the cached training data
    #[cfg(feature = "hardware-detect")]
    pub fn record_and_persist(
        &mut self,
        profiler: &crate::brick::BrickProfiler,
        config: &crate::tuner::features::RunConfig,
        kernel: crate::tuner::types::KernelType,
    ) -> Result<(), TunerError> {
        // Record the sample
        self.record(profiler, config, kernel);

        // Append to cache file
        let path = Self::cache_path();
        self.save_apr(&path)?;

        Ok(())
    }

    /// Import samples from JSON
    pub fn from_json(json: &str) -> Result<Self, TunerError> {
        let samples: Vec<TrainingSample> =
            serde_json::from_str(json).map_err(|e| TunerError::Serialization(e.to_string()))?;

        Ok(Self {
            samples,
            extractor: FeatureExtractor::new(),
            retrain_threshold: 100,
            samples_at_last_train: 0,
            feedback: Vec::new(),
            online_learning_enabled: false,
            error_window: Vec::new(),
            error_window_size: Self::DEFAULT_ERROR_WINDOW_SIZE,
        })
    }

    /// Import samples from the Five-Whys archive (85 labeled iterations)
    /// Bootstrap initial training data from historical analysis
    pub fn bootstrap_from_five_whys() -> Self {
        // Five-Whys archive has 85 labeled iterations from SHOWCASE-BRICK-001
        // Each iteration has: features, throughput, kernel selection, bottleneck

        // Returns empty collector -- data will be collected from real runs
        // Five-Whys archive loading is not yet implemented
        Self::new()
    }
}
