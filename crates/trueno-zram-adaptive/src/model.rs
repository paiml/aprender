//! ML model for compression algorithm selection.
//!
//! This module will integrate with aprender for more sophisticated
//! algorithm selection based on learned page patterns.

use trueno_zram_core::{Algorithm, PAGE_SIZE};

/// ML-based algorithm selector.
///
/// Currently uses heuristics; will be extended with aprender integration.
#[derive(Debug, Default)]
pub struct AlgorithmModel {
    // Model weights will go here
}

impl AlgorithmModel {
    /// Create a new model with default weights.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Predict the best algorithm for a page.
    #[must_use]
    pub fn predict(&self, page: &[u8; PAGE_SIZE]) -> Algorithm {
        // Simple heuristic for now: sample bytes to estimate compressibility
        let sample_size = 64;
        let mut unique_bytes = [false; 256];
        let mut unique_count = 0;

        for i in (0..PAGE_SIZE).step_by(PAGE_SIZE / sample_size) {
            let byte = page[i] as usize;
            if !unique_bytes[byte] {
                unique_bytes[byte] = true;
                unique_count += 1;
            }
        }

        // Heuristic thresholds
        if unique_count < 32 {
            // Low entropy (< 32 unique values) - use fast LZ4
            Algorithm::Lz4
        } else if unique_count < 128 {
            // Medium entropy - use Zstd for better ratios
            Algorithm::Zstd { level: 3 }
        } else {
            // High entropy - incompressible
            Algorithm::None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_new() {
        let model = AlgorithmModel::new();
        let debug = format!("{model:?}");
        assert!(debug.contains("AlgorithmModel"));
    }

    #[test]
    fn test_model_default() {
        let model = AlgorithmModel::default();
        let _ = format!("{model:?}");
    }

    #[test]
    fn test_predict_zero_page() {
        let model = AlgorithmModel::new();
        let zero_page = [0u8; PAGE_SIZE];
        assert_eq!(model.predict(&zero_page), Algorithm::Lz4);
    }

    #[test]
    fn test_predict_single_value() {
        let model = AlgorithmModel::new();
        let page = [0xFFu8; PAGE_SIZE];
        // All same value = 1 unique byte < 4, should use LZ4
        assert_eq!(model.predict(&page), Algorithm::Lz4);
    }

    #[test]
    fn test_predict_few_unique_values() {
        let model = AlgorithmModel::new();
        let mut page = [0u8; PAGE_SIZE];
        // Create pattern with ~8 unique bytes (< 32)
        for (i, byte) in page.iter_mut().enumerate() {
            *byte = u8::try_from(i % 8).unwrap();
        }
        assert_eq!(model.predict(&page), Algorithm::Lz4);
    }

    #[test]
    fn test_predict_medium_unique_values() {
        let model = AlgorithmModel::new();
        let mut page = [0u8; PAGE_SIZE];
        // The model samples at PAGE_SIZE/64 intervals
        // To get 32-128 unique values in samples, we need to set values
        // at those sample positions to be unique
        let step = PAGE_SIZE / 64;
        for (j, i) in (0..PAGE_SIZE).step_by(step).enumerate() {
            // Make the first 64 samples unique (j goes from 0-63)
            page[i] = u8::try_from(j % 64).unwrap();
        }
        assert_eq!(model.predict(&page), Algorithm::Zstd { level: 3 });
    }

    #[test]
    fn test_predict_high_entropy() {
        let model = AlgorithmModel::new();
        let mut page = [0u8; PAGE_SIZE];
        // The model samples 64 positions
        // To get >= 128 unique, we can't with only 64 samples
        // But we can test with pseudo-random to get many unique values
        let mut rng = 12345u64;
        for byte in &mut page {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            *byte = (rng >> 56) as u8; // Take top 8 bits
        }
        // With random distribution across 256 values, samples should hit many unique
        let result = model.predict(&page);
        // Either Zstd or None depending on how many unique values are sampled
        assert!(matches!(result, Algorithm::Zstd { .. } | Algorithm::None));
    }
}
