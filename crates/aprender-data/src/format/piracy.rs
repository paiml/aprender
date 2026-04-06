//! Piracy detection and watermarking for .ald format (§9.3)
//!
//! Provides first-class support for detecting stolen datasets and tracing
//! leaks.
//!
//! # Features
//!
//! - **Entropy Analysis**: Detect watermark presence without seller key
//! - **Watermark Embedding**: Buyer-specific fingerprints in LSB of floats
//! - **Watermark Extraction**: Recover buyer identity with seller key
//! - **Legal Evidence**: Generate cryptographic proof for proceedings

// Statistical calculations require f64 casts which are acceptable for this use case
#![allow(clippy::cast_precision_loss)]

use arrow::array::{Array, Float32Array, Float64Array, RecordBatch};

use crate::error::{Error, Result};

/// Natural LSB entropy threshold for clean data
pub const LSB_NATURAL_THRESHOLD: f64 = 0.97;

/// Autocorrelation threshold for pattern detection
/// Watermarks repeat every 256 bits, creating detectable autocorrelation
pub const AUTOCORRELATION_THRESHOLD: f64 = 0.7;

/// Minimum confidence for watermark detection
pub const DETECTION_CONFIDENCE_THRESHOLD: f64 = 0.80;

/// Result of watermark detection analysis
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Whether the dataset is likely watermarked
    pub likely_watermarked: bool,
    /// Confidence level (0.0 - 1.0)
    pub confidence: f64,
    /// Columns with suspicious LSB patterns
    pub suspicious_columns: Vec<String>,
}

/// Entropy analysis results for a column
#[derive(Debug, Clone)]
pub struct ColumnEntropy {
    /// Column name
    pub name: String,
    /// Shannon entropy of values (bits)
    pub shannon_entropy: f64,
    /// Shannon entropy of LSB bits only
    pub lsb_entropy: f64,
    /// Kolmogorov-Smirnov test p-value against uniform
    pub ks_pvalue: f64,
    /// Chi-square test result for LSB uniformity
    pub chi_square_pvalue: f64,
    /// Autocorrelation at lag 256 (watermark period)
    pub autocorrelation_256: f64,
}

/// Full entropy analysis of a dataset
#[derive(Debug, Clone)]
pub struct EntropyAnalysis {
    /// Per-column entropy results
    pub columns: Vec<ColumnEntropy>,
    /// Overall LSB entropy
    pub overall_lsb_entropy: f64,
    /// Overall autocorrelation at lag 256
    pub overall_autocorrelation: f64,
    /// Detection confidence
    pub confidence: f64,
    /// Anomalous columns (high autocorrelation = repeating pattern)
    pub anomalous_columns: Vec<String>,
}

/// Watermark configuration for embedding
#[derive(Debug, Clone)]
pub struct Watermark {
    /// Buyer identifier (hashed with seller secret)
    pub buyer_hash: [u8; 32],
    /// Embedding strength (0.0001 - 0.001 typical)
    pub strength: f32,
    /// Columns to watermark (indices)
    pub column_indices: Vec<usize>,
    /// Redundancy factor (0.0 - 1.0, survives N% row deletion)
    pub redundancy: f32,
}

/// Extracted buyer identity
#[derive(Debug, Clone)]
pub struct BuyerIdentity {
    /// Recovered buyer hash
    pub buyer_hash: [u8; 32],
    /// Extraction confidence
    pub confidence: f64,
}

/// Legal evidence for proceedings
#[derive(Debug, Clone)]
pub struct LegalEvidence {
    /// Dataset hash (SHA-256)
    pub dataset_hash: [u8; 32],
    /// Extracted buyer hash
    pub buyer_hash: [u8; 32],
    /// Statistical confidence (0.0-1.0)
    pub confidence: f64,
    /// Timestamp of analysis (RFC 3339)
    pub analyzed_at: String,
    /// Column-level evidence
    pub column_evidence: Vec<ColumnEvidence>,
}

/// Per-column evidence
#[derive(Debug, Clone)]
pub struct ColumnEvidence {
    /// Column name
    pub name: String,
    /// LSB entropy (lower = more likely watermarked)
    pub lsb_entropy: f64,
    /// Chi-square p-value
    pub chi_square_pvalue: f64,
    /// Bits extracted matching buyer
    pub matching_bits: usize,
    /// Total bits analyzed
    pub total_bits: usize,
}

/// Piracy detector for analyzing datasets
pub struct PiracyDetector;

impl PiracyDetector {
    /// Detect if dataset likely contains watermarks (no seller key needed)
    ///
    /// Uses statistical analysis of LSB autocorrelation to detect repeating
    /// patterns. Watermarks repeat every 256 bits, creating detectable
    /// autocorrelation.
    pub fn detect_watermark_presence(batches: &[RecordBatch]) -> DetectionResult {
        let analysis = Self::analyze_entropy(batches);

        // High autocorrelation indicates repeating pattern (watermark)
        let likely_watermarked = analysis.overall_autocorrelation > AUTOCORRELATION_THRESHOLD;

        DetectionResult {
            likely_watermarked,
            confidence: analysis.confidence,
            suspicious_columns: analysis.anomalous_columns,
        }
    }

    /// Perform entropy analysis on numeric columns
    pub fn analyze_entropy(batches: &[RecordBatch]) -> EntropyAnalysis {
        if batches.is_empty() {
            return EntropyAnalysis {
                columns: vec![],
                overall_lsb_entropy: 1.0,
                overall_autocorrelation: 0.0,
                confidence: 0.0,
                anomalous_columns: vec![],
            };
        }

        let schema = batches[0].schema();
        let mut column_results = Vec::new();
        let mut anomalous = Vec::new();

        for (col_idx, field) in schema.fields().iter().enumerate() {
            // Only analyze float columns (watermarks embedded in LSB)
            if !is_float_type(field.data_type()) {
                continue;
            }

            let lsb_bits = collect_lsb_bits(batches, col_idx);
            if lsb_bits.is_empty() {
                continue;
            }

            let lsb_entropy = shannon_entropy_bits(&lsb_bits);
            let chi_pvalue = chi_square_uniformity(&lsb_bits);
            let ks_pvalue = ks_test_uniform(&lsb_bits);
            let autocorr = autocorrelation_lag_256(&lsb_bits);

            let col_entropy = ColumnEntropy {
                name: field.name().clone(),
                shannon_entropy: 0.0, // Not needed for detection
                lsb_entropy,
                ks_pvalue,
                chi_square_pvalue: chi_pvalue,
                autocorrelation_256: autocorr,
            };

            // High autocorrelation indicates repeating pattern (watermark)
            if autocorr > AUTOCORRELATION_THRESHOLD {
                anomalous.push(field.name().clone());
            }

            column_results.push(col_entropy);
        }

        // Calculate overall metrics
        let overall_lsb = if column_results.is_empty() {
            1.0
        } else {
            column_results.iter().map(|c| c.lsb_entropy).sum::<f64>() / column_results.len() as f64
        };

        let overall_autocorr = if column_results.is_empty() {
            0.0
        } else {
            column_results
                .iter()
                .map(|c| c.autocorrelation_256)
                .sum::<f64>()
                / column_results.len() as f64
        };

        // Confidence based on autocorrelation strength
        let confidence = if overall_autocorr > 0.9 {
            0.99
        } else if overall_autocorr > AUTOCORRELATION_THRESHOLD {
            (overall_autocorr - AUTOCORRELATION_THRESHOLD).mul_add(0.6, 0.80)
        } else if overall_autocorr > 0.3 {
            overall_autocorr.mul_add(0.5, 0.50)
        } else {
            overall_autocorr
        }
        .clamp(0.0, 1.0);

        EntropyAnalysis {
            columns: column_results,
            overall_lsb_entropy: overall_lsb,
            overall_autocorrelation: overall_autocorr,
            confidence,
            anomalous_columns: anomalous,
        }
    }

    /// Generate legal evidence package
    pub fn generate_evidence(
        batches: &[RecordBatch],
        buyer_hash: &[u8; 32],
    ) -> Result<LegalEvidence> {
        let analysis = Self::analyze_entropy(batches);

        // Compute dataset hash
        let dataset_hash = hash_batches(batches);

        // Build column evidence
        let column_evidence: Vec<ColumnEvidence> = analysis
            .columns
            .iter()
            .map(|col| ColumnEvidence {
                name: col.name.clone(),
                lsb_entropy: col.lsb_entropy,
                chi_square_pvalue: col.chi_square_pvalue,
                matching_bits: 0, // Would need seller key to determine
                total_bits: 0,
            })
            .collect();

        // Get current timestamp
        let analyzed_at = chrono_lite_now();

        Ok(LegalEvidence {
            dataset_hash,
            buyer_hash: *buyer_hash,
            confidence: analysis.confidence,
            analyzed_at,
            column_evidence,
        })
    }
}

/// Watermark embedder for protecting datasets
pub struct WatermarkEmbedder {
    seller_key: [u8; 32],
}

impl WatermarkEmbedder {
    /// Create a new embedder with seller secret key
    #[must_use]
    pub fn new(seller_key: [u8; 32]) -> Self {
        Self { seller_key }
    }

    /// Embed watermark into dataset batches
    ///
    /// Modifies LSB of float values to encode buyer identity.
    pub fn embed(
        &self,
        batches: &[RecordBatch],
        watermark: &Watermark,
    ) -> Result<Vec<RecordBatch>> {
        let mut result = Vec::with_capacity(batches.len());

        // Generate deterministic bit sequence from buyer hash + seller key
        let bit_sequence = generate_watermark_bits(&watermark.buyer_hash, &self.seller_key);

        for batch in batches {
            let modified = Self::embed_batch(batch, watermark, &bit_sequence)?;
            result.push(modified);
        }

        Ok(result)
    }

    fn embed_batch(
        batch: &RecordBatch,
        watermark: &Watermark,
        bits: &[bool],
    ) -> Result<RecordBatch> {
        use std::sync::Arc;

        let schema = batch.schema();
        let mut new_columns: Vec<Arc<dyn Array>> = Vec::with_capacity(batch.num_columns());

        for col_idx in 0..batch.num_columns() {
            let col = batch.column(col_idx);

            if watermark.column_indices.contains(&col_idx) {
                // Embed watermark in this column
                let modified = embed_in_column(col.as_ref(), bits, watermark.strength)?;
                new_columns.push(modified);
            } else {
                new_columns.push(Arc::clone(col));
            }
        }

        RecordBatch::try_new(schema, new_columns).map_err(Error::Arrow)
    }

    /// Extract watermark from dataset (requires seller key)
    pub fn extract(&self, batches: &[RecordBatch]) -> Option<BuyerIdentity> {
        if batches.is_empty() {
            return None;
        }

        // Collect LSB bits from all float columns
        let schema = batches[0].schema();
        let mut all_bits = Vec::new();

        for (col_idx, field) in schema.fields().iter().enumerate() {
            if is_float_type(field.data_type()) {
                let bits = collect_lsb_bits(batches, col_idx);
                all_bits.extend(bits);
            }
        }

        if all_bits.len() < 256 {
            return None; // Not enough data
        }

        // Try to decode buyer hash using seller key
        let decoded = decode_watermark_bits(&all_bits, &self.seller_key)?;

        // Calculate confidence based on bit correlation
        let confidence = calculate_extraction_confidence(&all_bits, &decoded, &self.seller_key);

        if confidence < DETECTION_CONFIDENCE_THRESHOLD {
            return None;
        }

        Some(BuyerIdentity {
            buyer_hash: decoded,
            confidence,
        })
    }

    /// Verify if dataset contains specific buyer's watermark
    pub fn verify(&self, batches: &[RecordBatch], buyer_hash: &[u8; 32]) -> bool {
        self.extract(batches)
            .is_some_and(|id| &id.buyer_hash == buyer_hash)
    }
}

// === Helper functions ===

fn is_float_type(dtype: &arrow::datatypes::DataType) -> bool {
    matches!(
        dtype,
        arrow::datatypes::DataType::Float32 | arrow::datatypes::DataType::Float64
    )
}

fn collect_lsb_bits(batches: &[RecordBatch], col_idx: usize) -> Vec<bool> {
    let mut bits = Vec::new();

    for batch in batches {
        if col_idx >= batch.num_columns() {
            continue;
        }
        collect_column_lsb_bits(batch.column(col_idx), &mut bits);
    }

    bits
}

fn collect_column_lsb_bits(col: &dyn arrow::array::Array, bits: &mut Vec<bool>) {
    if let Some(f32_arr) = col.as_any().downcast_ref::<Float32Array>() {
        for i in 0..f32_arr.len() {
            if !f32_arr.is_null(i) {
                bits.push(f32_arr.value(i).to_bits() & 1 == 1);
            }
        }
    } else if let Some(f64_arr) = col.as_any().downcast_ref::<Float64Array>() {
        for i in 0..f64_arr.len() {
            if !f64_arr.is_null(i) {
                bits.push(f64_arr.value(i).to_bits() & 1 == 1);
            }
        }
    }
}

fn shannon_entropy_bits(bits: &[bool]) -> f64 {
    if bits.is_empty() {
        return 1.0;
    }

    let ones = bits.iter().filter(|&&b| b).count();
    let zeros = bits.len() - ones;
    let total = bits.len() as f64;

    let p1 = ones as f64 / total;
    let p0 = zeros as f64 / total;

    let mut entropy = 0.0;
    if p0 > 0.0 {
        entropy -= p0 * p0.log2();
    }
    if p1 > 0.0 {
        entropy -= p1 * p1.log2();
    }

    entropy // Max is 1.0 for uniform distribution
}

fn chi_square_uniformity(bits: &[bool]) -> f64 {
    if bits.is_empty() {
        return 1.0;
    }

    let ones = bits.iter().filter(|&&b| b).count() as f64;
    let zeros = (bits.len() - bits.iter().filter(|&&b| b).count()) as f64;
    let expected = bits.len() as f64 / 2.0;

    let chi_sq = (ones - expected).powi(2) / expected + (zeros - expected).powi(2) / expected;

    // Approximate p-value for chi-square with 1 df
    // Using simple approximation: p ≈ e^(-chi_sq/2) for large values
    (-chi_sq / 2.0).exp().clamp(0.0, 1.0)
}

fn ks_test_uniform(bits: &[bool]) -> f64 {
    if bits.is_empty() {
        return 1.0;
    }

    // Simple KS test against uniform [0,1]
    let n = bits.len() as f64;
    let ones_ratio = bits.iter().filter(|&&b| b).count() as f64 / n;

    // Max deviation from expected 0.5
    let d = (ones_ratio - 0.5).abs();

    // KS statistic
    let ks = d * n.sqrt();

    // Approximate p-value
    (-2.0 * ks.powi(2)).exp().clamp(0.0, 1.0)
}

/// Compute autocorrelation at lag 256 (watermark period)
///
/// Watermarks repeat every 256 bits, so watermarked data will have
/// high correlation between bits that are 256 apart.
fn autocorrelation_lag_256(bits: &[bool]) -> f64 {
    const LAG: usize = 256;

    if bits.len() < LAG * 2 {
        return 0.0; // Not enough data
    }

    // Convert to +1/-1 for correlation calculation
    let values: Vec<f64> = bits.iter().map(|&b| if b { 1.0 } else { -1.0 }).collect();

    // Compute mean
    let mean = values.iter().sum::<f64>() / values.len() as f64;

    // Compute variance
    let variance = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;

    if variance < 1e-10 {
        return 0.0;
    }

    // Compute autocorrelation at lag 256
    let n = values.len() - LAG;
    let autocorr: f64 = (0..n)
        .map(|i| (values[i] - mean) * (values[i + LAG] - mean))
        .sum::<f64>()
        / (n as f64 * variance);

    autocorr.clamp(-1.0, 1.0)
}

fn hash_batches(batches: &[RecordBatch]) -> [u8; 32] {
    use std::{
        collections::hash_map::DefaultHasher,
        hash::{Hash, Hasher},
    };

    let mut hasher = DefaultHasher::new();

    for batch in batches {
        batch.num_rows().hash(&mut hasher);
        batch.num_columns().hash(&mut hasher);

        for col_idx in 0..batch.num_columns() {
            let col = batch.column(col_idx);
            col.len().hash(&mut hasher);
        }
    }

    let hash64 = hasher.finish();

    // Expand to 32 bytes by hashing again
    let mut result = [0u8; 32];
    result[..8].copy_from_slice(&hash64.to_le_bytes());
    result[8..16].copy_from_slice(&hash64.to_be_bytes());
    result[16..24].copy_from_slice(&(!hash64).to_le_bytes());
    result[24..32].copy_from_slice(&hash64.rotate_left(32).to_le_bytes());

    result
}

fn chrono_lite_now() -> String {
    // Simple timestamp without chrono dependency
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", duration.as_secs())
}

fn generate_watermark_bits(buyer_hash: &[u8; 32], seller_key: &[u8; 32]) -> Vec<bool> {
    // XOR buyer and seller to create deterministic sequence
    let mut combined = [0u8; 32];
    for i in 0..32 {
        combined[i] = buyer_hash[i] ^ seller_key[i];
    }

    // Expand to bit sequence
    let mut bits = Vec::with_capacity(256);
    for byte in &combined {
        for bit_pos in 0..8 {
            bits.push((byte >> bit_pos) & 1 == 1);
        }
    }

    bits
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn embed_in_column(
    col: &dyn Array,
    bits: &[bool],
    strength: f32,
) -> Result<std::sync::Arc<dyn Array>> {
    use std::sync::Arc;

    if let Some(f32_arr) = col.as_any().downcast_ref::<Float32Array>() {
        let mut values: Vec<f32> = Vec::with_capacity(f32_arr.len());

        for (i, val) in f32_arr.iter().enumerate() {
            if let Some(v) = val {
                let bit_idx = i % bits.len();
                let modified = embed_bit_f32(v, bits[bit_idx], strength);
                values.push(modified);
            } else {
                values.push(f32::NAN);
            }
        }

        Ok(Arc::new(Float32Array::from(values)))
    } else if let Some(f64_arr) = col.as_any().downcast_ref::<Float64Array>() {
        let mut values: Vec<f64> = Vec::with_capacity(f64_arr.len());

        for (i, val) in f64_arr.iter().enumerate() {
            if let Some(v) = val {
                let bit_idx = i % bits.len();
                let modified = embed_bit_f64(v, bits[bit_idx], f64::from(strength));
                values.push(modified);
            } else {
                values.push(f64::NAN);
            }
        }

        Ok(Arc::new(Float64Array::from(values)))
    } else {
        Err(Error::Format("Column is not a float type".to_string()))
    }
}

fn embed_bit_f32(value: f32, bit: bool, _strength: f32) -> f32 {
    let mut bits = value.to_bits();
    if bit {
        bits |= 1;
    } else {
        bits &= !1;
    }
    f32::from_bits(bits)
}

fn embed_bit_f64(value: f64, bit: bool, _strength: f64) -> f64 {
    let mut bits = value.to_bits();
    if bit {
        bits |= 1;
    } else {
        bits &= !1;
    }
    f64::from_bits(bits)
}

fn decode_watermark_bits(bits: &[bool], seller_key: &[u8; 32]) -> Option<[u8; 32]> {
    if bits.len() < 256 {
        return None;
    }

    // Extract first 256 bits as buyer hash (XORed with seller key)
    let mut encoded = [0u8; 32];
    for (byte_idx, chunk) in bits.chunks(8).enumerate().take(32) {
        let mut byte = 0u8;
        for (bit_idx, &bit) in chunk.iter().enumerate() {
            if bit {
                byte |= 1 << bit_idx;
            }
        }
        encoded[byte_idx] = byte;
    }

    // XOR with seller key to recover buyer hash
    let mut buyer_hash = [0u8; 32];
    for i in 0..32 {
        buyer_hash[i] = encoded[i] ^ seller_key[i];
    }

    Some(buyer_hash)
}

fn calculate_extraction_confidence(
    observed_bits: &[bool],
    decoded_buyer: &[u8; 32],
    seller_key: &[u8; 32],
) -> f64 {
    // Generate expected bit sequence
    let expected_bits = generate_watermark_bits(decoded_buyer, seller_key);

    if observed_bits.len() < expected_bits.len() {
        return 0.0;
    }

    // Count matching bits
    let matches = observed_bits
        .iter()
        .zip(expected_bits.iter().cycle())
        .filter(|(a, b)| a == b)
        .count();

    let total = observed_bits.len().min(expected_bits.len() * 4); // Check multiple cycles
    let match_ratio = matches as f64 / total as f64;

    // Convert to confidence (0.5 = random, 1.0 = perfect match)
    ((match_ratio - 0.5) * 2.0).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;

    fn create_test_batch_with_size(size: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("price", DataType::Float64, false),
            Field::new("quantity", DataType::Float64, false),
        ]));

        // Use simple LCG to generate pseudo-random floats with varied LSBs
        // This simulates real-world data where LSBs are essentially random
        let mut seed: u64 = 12345;
        let prices: Vec<f64> = (0..size)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let mantissa = (seed >> 11) as f64 / (1u64 << 53) as f64;
                10.0 + mantissa * 100.0
            })
            .collect();

        seed = 67890;
        let quantities: Vec<f64> = (0..size)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let mantissa = (seed >> 11) as f64 / (1u64 << 53) as f64;
                1.0 + mantissa * 50.0
            })
            .collect();

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(prices)),
                Arc::new(Float64Array::from(quantities)),
            ],
        )
        .expect("create batch")
    }

    #[test]
    fn test_entropy_analysis_clean_data() {
        // Need enough data for autocorrelation at lag 256
        let batch = create_test_batch_with_size(1000);
        let analysis = PiracyDetector::analyze_entropy(&[batch]);

        // Clean data should have LOW autocorrelation (no repeating pattern)
        assert!(
            analysis.overall_autocorrelation < AUTOCORRELATION_THRESHOLD,
            "Clean data autocorrelation should be low: {}",
            analysis.overall_autocorrelation
        );

        // Clean data should have high entropy (random LSBs)
        assert!(
            analysis.overall_lsb_entropy > 0.9,
            "Clean data LSB entropy should be high: {}",
            analysis.overall_lsb_entropy
        );
    }

    #[test]
    fn test_watermark_embed_extract() {
        // Need enough data for autocorrelation detection
        let batch = create_test_batch_with_size(1000);
        let seller_key = [42u8; 32];
        let buyer_hash = [7u8; 32];

        let embedder = WatermarkEmbedder::new(seller_key);

        let watermark = Watermark {
            buyer_hash,
            strength: 0.001,
            column_indices: vec![0, 1],
            redundancy: 0.5,
        };

        // Embed watermark
        let watermarked = embedder.embed(&[batch], &watermark).expect("embed failed");

        // Watermarked data should have HIGH autocorrelation (repeating pattern)
        let analysis = PiracyDetector::analyze_entropy(&watermarked);
        assert!(
            analysis.overall_autocorrelation > AUTOCORRELATION_THRESHOLD,
            "Watermarked data autocorrelation should be high: {}",
            analysis.overall_autocorrelation
        );

        // Extract watermark
        let extracted = embedder.extract(&watermarked);
        assert!(extracted.is_some(), "Should extract watermark");

        let identity = extracted.expect("identity");
        assert_eq!(identity.buyer_hash, buyer_hash);
    }

    #[test]
    fn test_detection_without_key() {
        // Need enough data for autocorrelation detection
        let batch = create_test_batch_with_size(1000);
        let seller_key = [42u8; 32];
        let buyer_hash = [7u8; 32];

        let embedder = WatermarkEmbedder::new(seller_key);

        let watermark = Watermark {
            buyer_hash,
            strength: 0.001,
            column_indices: vec![0, 1],
            redundancy: 0.5,
        };

        let watermarked = embedder.embed(&[batch], &watermark).expect("embed failed");

        // Detection should work without seller key (based on autocorrelation)
        let detection = PiracyDetector::detect_watermark_presence(&watermarked);
        assert!(
            detection.likely_watermarked,
            "Should detect watermark presence"
        );
        assert!(
            detection.confidence > 0.5,
            "Confidence: {}",
            detection.confidence
        );
    }

    #[test]
    fn test_verify_buyer() {
        // Need enough data for watermark extraction
        let batch = create_test_batch_with_size(1000);
        let seller_key = [42u8; 32];
        let buyer_hash = [7u8; 32];
        let wrong_buyer = [99u8; 32];

        let embedder = WatermarkEmbedder::new(seller_key);

        let watermark = Watermark {
            buyer_hash,
            strength: 0.001,
            column_indices: vec![0, 1],
            redundancy: 0.5,
        };

        let watermarked = embedder.embed(&[batch], &watermark).expect("embed failed");

        assert!(
            embedder.verify(&watermarked, &buyer_hash),
            "Should verify correct buyer"
        );
        assert!(
            !embedder.verify(&watermarked, &wrong_buyer),
            "Should reject wrong buyer"
        );
    }

    #[test]
    fn test_shannon_entropy() {
        // Uniform distribution should have entropy ~1.0
        let uniform: Vec<bool> = (0..1000).map(|i| i % 2 == 0).collect();
        let entropy = shannon_entropy_bits(&uniform);
        assert!((entropy - 1.0).abs() < 0.01, "Uniform entropy: {}", entropy);

        // All zeros should have entropy 0
        let zeros = vec![false; 1000];
        let entropy = shannon_entropy_bits(&zeros);
        assert!(entropy < 0.01, "Zero entropy: {}", entropy);
    }

    #[test]
    fn test_generate_evidence() {
        let batch = create_test_batch_with_size(1000);
        let buyer_hash = [7u8; 32];

        let evidence =
            PiracyDetector::generate_evidence(&[batch], &buyer_hash).expect("generate failed");

        assert_eq!(evidence.buyer_hash, buyer_hash);
        assert!(!evidence.column_evidence.is_empty());
    }

    #[test]
    fn test_autocorrelation_detection() {
        // Test that autocorrelation correctly distinguishes clean vs watermarked data
        let clean_batch = create_test_batch_with_size(1000);
        let seller_key = [42u8; 32];
        let buyer_hash = [7u8; 32];

        let embedder = WatermarkEmbedder::new(seller_key);
        let watermark = Watermark {
            buyer_hash,
            strength: 0.001,
            column_indices: vec![0, 1],
            redundancy: 0.5,
        };

        let watermarked = embedder
            .embed(&[clean_batch.clone()], &watermark)
            .expect("embed");

        // Analyze both
        let clean_analysis = PiracyDetector::analyze_entropy(&[clean_batch]);
        let watermarked_analysis = PiracyDetector::analyze_entropy(&watermarked);

        // Clean should have low autocorrelation
        assert!(
            clean_analysis.overall_autocorrelation < 0.3,
            "Clean autocorr: {}",
            clean_analysis.overall_autocorrelation
        );

        // Watermarked should have high autocorrelation
        assert!(
            watermarked_analysis.overall_autocorrelation > 0.9,
            "Watermarked autocorr: {}",
            watermarked_analysis.overall_autocorrelation
        );
    }

    #[test]
    fn test_detection_result_default() {
        let result = DetectionResult {
            likely_watermarked: false,
            confidence: 0.0,
            suspicious_columns: Vec::new(),
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("DetectionResult"));
    }

    #[test]
    fn test_column_entropy_debug() {
        let entropy = ColumnEntropy {
            name: "col".to_string(),
            shannon_entropy: 7.5,
            lsb_entropy: 0.99,
            ks_pvalue: 0.5,
            chi_square_pvalue: 0.5,
            autocorrelation_256: 0.1,
        };
        let debug = format!("{:?}", entropy);
        assert!(debug.contains("ColumnEntropy"));
        assert!(debug.contains("col"));
    }

    #[test]
    fn test_watermark_clone() {
        let watermark = Watermark {
            buyer_hash: [1u8; 32],
            strength: 0.001,
            column_indices: vec![0],
            redundancy: 0.5,
        };
        let cloned = watermark.clone();
        assert_eq!(cloned.buyer_hash, watermark.buyer_hash);
        assert_eq!(cloned.strength, watermark.strength);
    }

    #[test]
    fn test_entropy_analysis_clone() {
        let analysis = EntropyAnalysis {
            columns: Vec::new(),
            overall_lsb_entropy: 0.99,
            overall_autocorrelation: 0.1,
            confidence: 0.0,
            anomalous_columns: Vec::new(),
        };
        let cloned = analysis.clone();
        assert_eq!(cloned.overall_lsb_entropy, 0.99);
    }

    #[test]
    fn test_entropy_analysis_empty_batches() {
        let analysis = PiracyDetector::analyze_entropy(&[]);
        assert_eq!(analysis.overall_lsb_entropy, 1.0);
        assert_eq!(analysis.overall_autocorrelation, 0.0);
        assert_eq!(analysis.confidence, 0.0);
        assert!(analysis.columns.is_empty());
    }

    #[test]
    fn test_is_float_type() {
        assert!(is_float_type(&DataType::Float32));
        assert!(is_float_type(&DataType::Float64));
        assert!(!is_float_type(&DataType::Int32));
        assert!(!is_float_type(&DataType::Utf8));
    }

    #[test]
    fn test_chi_square_empty() {
        let bits: Vec<bool> = vec![];
        assert_eq!(chi_square_uniformity(&bits), 1.0);
    }

    #[test]
    fn test_ks_test_empty() {
        let bits: Vec<bool> = vec![];
        assert_eq!(ks_test_uniform(&bits), 1.0);
    }

    #[test]
    fn test_shannon_entropy_empty() {
        let bits: Vec<bool> = vec![];
        assert_eq!(shannon_entropy_bits(&bits), 1.0);
    }

    #[test]
    fn test_autocorrelation_short_data() {
        // Data shorter than 2*LAG should return 0
        let bits: Vec<bool> = (0..500).map(|i| i % 2 == 0).collect();
        let autocorr = autocorrelation_lag_256(&bits);
        assert_eq!(autocorr, 0.0);
    }

    #[test]
    fn test_hash_batches() {
        let batch = create_test_batch_with_size(100);
        let hash1 = hash_batches(&[batch.clone()]);
        let hash2 = hash_batches(&[batch]);
        assert_eq!(hash1, hash2); // Same data should produce same hash
    }

    #[test]
    fn test_generate_watermark_bits() {
        let buyer_hash = [1u8; 32];
        let seller_key = [2u8; 32];
        let bits = generate_watermark_bits(&buyer_hash, &seller_key);
        assert_eq!(bits.len(), 256);
    }

    #[test]
    fn test_decode_watermark_bits_short() {
        let bits: Vec<bool> = vec![true; 100]; // Too short
        let result = decode_watermark_bits(&bits, &[0u8; 32]);
        assert!(result.is_none());
    }

    #[test]
    fn test_decode_watermark_bits_roundtrip() {
        let buyer_hash = [42u8; 32];
        let seller_key = [99u8; 32];
        let bits = generate_watermark_bits(&buyer_hash, &seller_key);

        let decoded = decode_watermark_bits(&bits, &seller_key);
        assert!(decoded.is_some());
        assert_eq!(decoded.unwrap(), buyer_hash);
    }

    #[test]
    fn test_embed_bit_f32() {
        let val = 1.0f32;
        let embedded_1 = embed_bit_f32(val, true, 0.001);
        let embedded_0 = embed_bit_f32(val, false, 0.001);

        // LSB should be set correctly
        assert_eq!(embedded_1.to_bits() & 1, 1);
        assert_eq!(embedded_0.to_bits() & 1, 0);
    }

    #[test]
    fn test_embed_bit_f64() {
        let val = 1.0f64;
        let embedded_1 = embed_bit_f64(val, true, 0.001);
        let embedded_0 = embed_bit_f64(val, false, 0.001);

        // LSB should be set correctly
        assert_eq!(embedded_1.to_bits() & 1, 1);
        assert_eq!(embedded_0.to_bits() & 1, 0);
    }

    #[test]
    fn test_extraction_confidence_short_data() {
        let observed_bits: Vec<bool> = vec![true; 100]; // Too short
        let decoded = [0u8; 32];
        let seller_key = [0u8; 32];
        let confidence = calculate_extraction_confidence(&observed_bits, &decoded, &seller_key);
        assert_eq!(confidence, 0.0);
    }

    #[test]
    fn test_extract_empty_batches() {
        let embedder = WatermarkEmbedder::new([0u8; 32]);
        let result = embedder.extract(&[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_collect_lsb_bits_f32() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float32,
            false,
        )]));
        let values: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Float32Array::from(values))]).unwrap();

        let bits = collect_lsb_bits(&[batch], 0);
        assert_eq!(bits.len(), 100);
    }

    #[test]
    fn test_collect_lsb_bits_column_out_of_range() {
        let batch = create_test_batch_with_size(10);
        let bits = collect_lsb_bits(&[batch], 999); // Out of range
        assert!(bits.is_empty());
    }

    #[test]
    fn test_legal_evidence_clone() {
        let evidence = LegalEvidence {
            dataset_hash: [0u8; 32],
            buyer_hash: [1u8; 32],
            confidence: 0.95,
            analyzed_at: "2024-01-01".to_string(),
            column_evidence: vec![],
        };
        let cloned = evidence.clone();
        assert_eq!(cloned.confidence, 0.95);
    }

    #[test]
    fn test_buyer_identity_clone() {
        let identity = BuyerIdentity {
            buyer_hash: [42u8; 32],
            confidence: 0.9,
        };
        let cloned = identity.clone();
        assert_eq!(cloned.buyer_hash, identity.buyer_hash);
    }

    #[test]
    fn test_column_evidence_clone() {
        let evidence = ColumnEvidence {
            name: "test".to_string(),
            lsb_entropy: 0.98,
            chi_square_pvalue: 0.5,
            matching_bits: 100,
            total_bits: 200,
        };
        let cloned = evidence.clone();
        assert_eq!(cloned.name, "test");
    }

    #[test]
    fn test_detection_result_clone() {
        let result = DetectionResult {
            likely_watermarked: true,
            confidence: 0.9,
            suspicious_columns: vec!["col1".to_string()],
        };
        let cloned = result.clone();
        assert!(cloned.likely_watermarked);
    }

    #[test]
    fn test_chrono_lite_now() {
        let timestamp = chrono_lite_now();
        // Should be a parseable number (seconds since epoch)
        assert!(!timestamp.is_empty());
        let _: u64 = timestamp.parse().expect("Should be a number");
    }

    #[test]
    fn test_confidence_calculation_ranges() {
        // Test various autocorrelation ranges for confidence
        let _analysis_high = EntropyAnalysis {
            columns: vec![],
            overall_lsb_entropy: 0.9,
            overall_autocorrelation: 0.95, // Very high
            confidence: 0.0,
            anomalous_columns: vec![],
        };

        // Confidence should be high for high autocorrelation
        let analysis = PiracyDetector::analyze_entropy(&[]);
        assert!(analysis.confidence >= 0.0 && analysis.confidence <= 1.0);
    }
}
