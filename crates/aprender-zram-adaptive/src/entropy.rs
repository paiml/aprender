//! Shannon entropy calculation for memory pages.

use trueno_zram_core::PAGE_SIZE;

/// Calculator for Shannon entropy of byte sequences.
#[derive(Debug, Clone)]
pub struct EntropyCalculator {
    /// Byte frequency histogram (256 buckets).
    histogram: [u64; 256],
}

impl Default for EntropyCalculator {
    fn default() -> Self {
        Self {
            histogram: [0u64; 256],
        }
    }
}

impl EntropyCalculator {
    /// Create a new entropy calculator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate Shannon entropy of a page in bits per byte.
    ///
    /// Returns a value between 0.0 (all same bytes) and 8.0 (uniform random).
    #[must_use]
    pub fn calculate(&mut self, page: &[u8; PAGE_SIZE]) -> f64 {
        // Reset histogram
        self.histogram.fill(0);

        // Count byte frequencies
        for &byte in page {
            self.histogram[byte as usize] += 1;
        }

        // Calculate entropy: H(X) = -sum(p(x) * log2(p(x)))
        let total = PAGE_SIZE as f64;
        let mut entropy = 0.0;

        for &count in &self.histogram {
            if count > 0 {
                let p = count as f64 / total;
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    /// Classify entropy level for algorithm selection.
    #[must_use]
    pub fn classify(&mut self, page: &[u8; PAGE_SIZE]) -> EntropyLevel {
        let entropy = self.calculate(page);

        if entropy < 1.0 {
            EntropyLevel::VeryLow
        } else if entropy < 4.0 {
            EntropyLevel::Low
        } else if entropy < 6.5 {
            EntropyLevel::Medium
        } else if entropy < 7.5 {
            EntropyLevel::High
        } else {
            EntropyLevel::VeryHigh
        }
    }
}

/// Classification of page entropy levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntropyLevel {
    /// Entropy < 1.0 bits/byte - highly compressible (e.g., all zeros).
    VeryLow,
    /// Entropy 1.0-4.0 bits/byte - good compression expected.
    Low,
    /// Entropy 4.0-6.5 bits/byte - moderate compression.
    Medium,
    /// Entropy 6.5-7.5 bits/byte - poor compression.
    High,
    /// Entropy > 7.5 bits/byte - incompressible (random data).
    VeryHigh,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_page_entropy() {
        let mut calc = EntropyCalculator::new();
        let page = [0u8; PAGE_SIZE];
        let entropy = calc.calculate(&page);
        assert!(entropy < 0.001, "Zero page should have ~0 entropy");
    }

    #[test]
    fn test_uniform_random_entropy() {
        let mut calc = EntropyCalculator::new();
        // Create a page with uniform byte distribution
        let mut page = [0u8; PAGE_SIZE];
        for (i, byte) in page.iter_mut().enumerate() {
            *byte = u8::try_from(i % 256).unwrap();
        }
        // Repeat pattern to fill page
        let entropy = calc.calculate(&page);
        // With 4096 bytes and 256 values, each appears 16 times
        // Expected entropy = log2(256) = 8.0 for truly uniform
        assert!(
            entropy > 7.9,
            "Uniform distribution should have high entropy: {entropy}"
        );
    }

    #[test]
    fn test_entropy_classification() {
        let mut calc = EntropyCalculator::new();

        // Zero page -> VeryLow
        let zero_page = [0u8; PAGE_SIZE];
        assert_eq!(calc.classify(&zero_page), EntropyLevel::VeryLow);

        // Repeating pattern -> Low
        let mut pattern_page = [0u8; PAGE_SIZE];
        for (i, byte) in pattern_page.iter_mut().enumerate() {
            *byte = u8::try_from(i % 4).unwrap();
        }
        let level = calc.classify(&pattern_page);
        assert!(matches!(level, EntropyLevel::VeryLow | EntropyLevel::Low));
    }
}
