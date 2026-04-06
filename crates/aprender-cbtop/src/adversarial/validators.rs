//! Input and configuration validators for adversarial testing.

use super::{AdversarialError, AdversarialResult};

/// Input validator for adversarial testing
#[derive(Debug, Clone)]
pub struct InputValidator {
    /// Maximum allowed input size in bytes
    pub max_size: usize,
    /// Whether to compute and verify checksums
    pub verify_checksum: bool,
    /// Whether to detect NaN values
    pub detect_nan: bool,
    /// Whether to detect infinity values
    pub detect_inf: bool,
}

impl Default for InputValidator {
    fn default() -> Self {
        Self {
            max_size: 1024 * 1024 * 1024, // 1GB default max
            verify_checksum: true,
            detect_nan: true,
            detect_inf: true,
        }
    }
}

impl InputValidator {
    /// Create a new input validator
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum input size
    pub fn with_max_size(mut self, max_size: usize) -> Self {
        self.max_size = max_size;
        self
    }

    /// Validate a byte slice
    pub fn validate_bytes(&self, data: &[u8]) -> AdversarialResult<()> {
        // F1004: Zero-size inputs handled
        if data.is_empty() {
            return Err(AdversarialError::ZeroSizeInput);
        }

        // F1005: Maximum-size inputs handled
        if data.len() > self.max_size {
            return Err(AdversarialError::MaxSizeExceeded {
                size: data.len(),
                max: self.max_size,
            });
        }

        Ok(())
    }

    /// Validate a float slice for NaN and Inf
    pub fn validate_floats(&self, data: &[f32]) -> AdversarialResult<()> {
        // F1004: Zero-size inputs handled
        if data.is_empty() {
            return Err(AdversarialError::ZeroSizeInput);
        }

        // F1005: Maximum-size inputs handled
        let byte_size = std::mem::size_of_val(data);
        if byte_size > self.max_size {
            return Err(AdversarialError::MaxSizeExceeded {
                size: byte_size,
                max: self.max_size,
            });
        }

        // F1014: NaN propagation controlled
        if self.detect_nan {
            for (i, &v) in data.iter().enumerate() {
                if v.is_nan() {
                    return Err(AdversarialError::NaNDetected { index: i });
                }
            }
        }

        // F1015: Inf propagation controlled
        if self.detect_inf {
            for (i, &v) in data.iter().enumerate() {
                if v.is_infinite() {
                    return Err(AdversarialError::InfinityDetected {
                        index: i,
                        positive: v.is_sign_positive(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Compute a simple checksum for data validation
    pub fn compute_checksum(data: &[u8]) -> u32 {
        // Simple Adler-32-like checksum
        let mut a: u32 = 1;
        let mut b: u32 = 0;
        for &byte in data {
            a = (a.wrapping_add(u32::from(byte))) % 65521;
            b = (b.wrapping_add(a)) % 65521;
        }
        (b << 16) | a
    }

    /// Verify data against expected checksum
    pub fn verify_checksum(&self, data: &[u8], expected: u32) -> AdversarialResult<()> {
        if !self.verify_checksum {
            return Ok(());
        }

        let actual = Self::compute_checksum(data);
        if actual != expected {
            return Err(AdversarialError::CorruptedInput {
                byte_index: 0, // Can't pinpoint exact corruption
                expected_checksum: expected,
                actual_checksum: actual,
            });
        }
        Ok(())
    }
}

/// Configuration validator for fuzzing (F1008, F1009)
#[derive(Debug, Clone)]
pub struct ConfigValidator {
    /// Minimum allowed values by field name
    pub mins: std::collections::HashMap<String, f64>,
    /// Maximum allowed values by field name
    pub maxs: std::collections::HashMap<String, f64>,
}

impl Default for ConfigValidator {
    fn default() -> Self {
        Self {
            mins: std::collections::HashMap::new(),
            maxs: std::collections::HashMap::new(),
        }
    }
}

impl ConfigValidator {
    /// Create a new config validator
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a bound for a field
    pub fn with_bound(mut self, field: &str, min: f64, max: f64) -> Self {
        self.mins.insert(field.to_string(), min);
        self.maxs.insert(field.to_string(), max);
        self
    }

    /// Validate a numeric config value (F1009)
    pub fn validate_numeric(&self, field: &str, value: f64) -> AdversarialResult<f64> {
        // Check for NaN
        if value.is_nan() {
            return Err(AdversarialError::ConfigParseError {
                field: field.to_string(),
                reason: "value is NaN".to_string(),
            });
        }

        // Check bounds
        if let Some(&min) = self.mins.get(field) {
            if value < min {
                return Err(AdversarialError::ConfigOutOfBounds {
                    field: field.to_string(),
                    value: value.to_string(),
                    min: min.to_string(),
                    max: self
                        .maxs
                        .get(field)
                        .map_or("unbounded".to_string(), |m| m.to_string()),
                });
            }
        }

        if let Some(&max) = self.maxs.get(field) {
            if value > max {
                return Err(AdversarialError::ConfigOutOfBounds {
                    field: field.to_string(),
                    value: value.to_string(),
                    min: self
                        .mins
                        .get(field)
                        .map_or("unbounded".to_string(), |m| m.to_string()),
                    max: max.to_string(),
                });
            }
        }

        Ok(value)
    }

    /// Validate TOML-like string config (F1008)
    pub fn validate_toml_string(&self, input: &str) -> AdversarialResult<()> {
        // Check for common malformed TOML patterns
        let trimmed = input.trim();

        // Empty input
        if trimmed.is_empty() {
            return Err(AdversarialError::ConfigParseError {
                field: "root".to_string(),
                reason: "empty config".to_string(),
            });
        }

        // Unclosed brackets
        let open_brackets = trimmed.matches('[').count();
        let close_brackets = trimmed.matches(']').count();
        if open_brackets != close_brackets {
            return Err(AdversarialError::ConfigParseError {
                field: "root".to_string(),
                reason: format!(
                    "mismatched brackets: {} open, {} close",
                    open_brackets, close_brackets
                ),
            });
        }

        // Unclosed quotes
        let quotes = trimmed.matches('"').count();
        if !quotes.is_multiple_of(2) {
            return Err(AdversarialError::ConfigParseError {
                field: "root".to_string(),
                reason: "unclosed string literal".to_string(),
            });
        }

        Ok(())
    }
}
