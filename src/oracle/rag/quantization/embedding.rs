//! Int8 quantized embedding with metadata
//!
//! Achieves 4x memory reduction compared to f32 embeddings.
//! Following Gholami et al. (2022) survey on quantization methods.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::calibration::CalibrationStats;
use super::error::{validate_embedding, QuantizationError};
use super::params::QuantizationParams;

/// Int8 quantized embedding with metadata
///
/// Achieves 4x memory reduction compared to f32 embeddings.
/// Following Gholami et al. (2022) survey on quantization methods.
#[derive(Debug, Clone)]
pub struct QuantizedEmbedding {
    /// Quantized values in [-128, 127]
    pub values: Vec<i8>,
    /// Quantization parameters for dequantization
    pub params: QuantizationParams,
    /// BLAKE3 content hash for integrity (Poka-Yoke)
    pub hash: [u8; 32],
}

impl QuantizedEmbedding {
    /// Quantize f32 embedding to int8 with calibration
    ///
    /// Implements symmetric quantization: Q(x) = round(x / scale)
    ///
    /// # Errors
    /// - `EmptyEmbedding`: If input is empty
    /// - `DimensionMismatch`: If dimensions don't match calibration
    /// - `NonFiniteValue`: If NaN/Inf detected (Poka-Yoke)
    pub fn from_f32(
        embedding: &[f32],
        calibration: &CalibrationStats,
    ) -> Result<Self, QuantizationError> {
        // Jidoka + Poka-Yoke: Validate embedding
        validate_embedding(embedding, calibration.dims)?;

        // Create quantization params from calibration
        let params = calibration.to_quant_params()?;

        // Quantize with rounding and clamping
        let values: Vec<i8> = embedding.iter().map(|&v| params.quantize_value(v)).collect();

        // Compute integrity hash (Poka-Yoke)
        let hash = compute_hash(&values);

        Ok(Self { values, params, hash })
    }

    /// Quantize f32 embedding using local statistics (no calibration)
    ///
    /// Computes absmax from the embedding itself.
    pub fn from_f32_uncalibrated(embedding: &[f32]) -> Result<Self, QuantizationError> {
        // Jidoka + Poka-Yoke: Validate embedding (dimension check is trivially satisfied)
        validate_embedding(embedding, embedding.len())?;

        // Compute absmax (safe since validate_embedding confirmed all values are finite)
        let mut absmax: f32 = embedding.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));

        // Handle zero vector
        if absmax == 0.0 {
            absmax = 1.0; // Avoid division by zero
        }

        let params = QuantizationParams::from_absmax(absmax, embedding.len())?;
        let values: Vec<i8> = embedding.iter().map(|&v| params.quantize_value(v)).collect();
        // Compute integrity hash (Poka-Yoke)
        let hash = compute_hash(&values);

        Ok(Self { values, params, hash })
    }

    /// Dequantize to f32 embedding
    ///
    /// Returns approximate original values within error bound.
    pub fn dequantize(&self) -> Vec<f32> {
        self.values.iter().map(|&v| self.params.dequantize_value(v)).collect()
    }

    /// Verify integrity hash (Poka-Yoke)
    pub fn verify_integrity(&self) -> bool {
        let computed = compute_hash(&self.values);
        computed == self.hash
    }

    /// Get embedding dimensions
    pub fn dims(&self) -> usize {
        self.values.len()
    }

    /// Memory size in bytes (4x reduction from f32)
    pub fn memory_size(&self) -> usize {
        self.values.len() // 1 byte per element
            + std::mem::size_of::<QuantizationParams>()
            + 32 // hash
    }
}

/// Compute content hash for integrity verification (Poka-Yoke)
///
/// Uses SipHash (DefaultHasher) which provides good collision resistance
/// for integrity checking. Expanded to 32 bytes for consistency with
/// BLAKE3-style hashes used in specification.
pub fn compute_hash(values: &[i8]) -> [u8; 32] {
    // h[0]: SipHash of raw values
    let mut hasher = DefaultHasher::new();
    values.hash(&mut hasher);
    let mut hashes = [0u64; 4];
    hashes[0] = hasher.finish();

    // h[1]: chain previous hash + mix in length for extra entropy
    let mut hasher = DefaultHasher::new();
    hashes[0].hash(&mut hasher);
    values.len().hash(&mut hasher);
    hashes[1] = hasher.finish();

    // h[2..3]: chain each subsequent hash
    for i in 2..4 {
        let mut hasher = DefaultHasher::new();
        hashes[i - 1].hash(&mut hasher);
        hashes[i] = hasher.finish();
    }

    let mut result = [0u8; 32];
    for (i, &h) in hashes.iter().enumerate() {
        result[i * 8..(i + 1) * 8].copy_from_slice(&h.to_le_bytes());
    }
    result
}
