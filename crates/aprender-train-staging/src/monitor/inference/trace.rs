//! Decision Trace (ENT-104)
//!
//! Universal decision trace structure for all APR models.

use super::path::DecisionPath;
use serde::{Deserialize, Serialize};

/// Universal decision trace structure
///
/// Records everything needed to understand and reproduce a prediction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionTrace<P: DecisionPath> {
    /// Monotonic nanosecond timestamp
    pub timestamp_ns: u64,
    /// Sequence number within session
    pub sequence: u64,
    /// FNV-1a hash of input features
    pub input_hash: u64,
    /// Model-specific decision path
    pub path: P,
    /// Final output value
    pub output: f32,
    /// Inference latency in nanoseconds
    pub latency_ns: u64,
}

impl<P: DecisionPath> DecisionTrace<P> {
    /// Create a new decision trace
    pub fn new(
        timestamp_ns: u64,
        sequence: u64,
        input_hash: u64,
        path: P,
        output: f32,
        latency_ns: u64,
    ) -> Self {
        Self { timestamp_ns, sequence, input_hash, path, output, latency_ns }
    }

    /// Get confidence from the underlying path
    pub fn confidence(&self) -> f32 {
        self.path.confidence()
    }

    /// Get human-readable explanation
    pub fn explain(&self) -> String {
        let mut explanation = format!(
            "Trace #{} @ {}ns (latency: {}ns)\n",
            self.sequence, self.timestamp_ns, self.latency_ns
        );
        explanation.push_str(&format!("Input hash: 0x{:016x}\n", self.input_hash));
        explanation.push_str(&format!("Output: {:.4}\n", self.output));
        explanation.push_str("---\n");
        explanation.push_str(&self.path.explain());
        explanation
    }

    /// Get feature contributions from the underlying path
    pub fn feature_contributions(&self) -> &[f32] {
        self.path.feature_contributions()
    }

    /// Convert to binary format
    ///
    /// Format:
    /// - `[0..8]`: timestamp_ns (u64 LE)
    /// - `[8..16]`: sequence (u64 LE)
    /// - `[16..24]`: input_hash (u64 LE)
    /// - `[24..28]`: output (f32 LE)
    /// - `[28..32]`: latency_ns (u32 LE, microsecond precision)
    /// - `[32..36]`: path_length (u32 LE)
    /// - `[36..]`: path_data
    pub fn to_bytes(&self) -> Vec<u8> {
        let path_bytes = self.path.to_bytes();

        let mut bytes = Vec::with_capacity(36 + path_bytes.len());

        bytes.extend_from_slice(&self.timestamp_ns.to_le_bytes());
        bytes.extend_from_slice(&self.sequence.to_le_bytes());
        bytes.extend_from_slice(&self.input_hash.to_le_bytes());
        bytes.extend_from_slice(&self.output.to_le_bytes());

        // Store latency as microseconds in u32 for compactness
        let latency_us = (self.latency_ns / 1000) as u32;
        bytes.extend_from_slice(&latency_us.to_le_bytes());

        bytes.extend_from_slice(&(path_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&path_bytes);

        bytes
    }

    /// Reconstruct from binary format
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, super::path::PathError>
    where
        P: DecisionPath,
    {
        if bytes.len() < 36 {
            return Err(super::path::PathError::InsufficientData {
                expected: 36,
                actual: bytes.len(),
            });
        }

        let timestamp_ns = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);

        let sequence = u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]);

        let input_hash = u64::from_le_bytes([
            bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
        ]);

        let output = f32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);

        let latency_us = u32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]);
        let latency_ns = u64::from(latency_us) * 1000;

        let path_length = u32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]) as usize;

        if bytes.len() < 36 + path_length {
            return Err(super::path::PathError::InsufficientData {
                expected: 36 + path_length,
                actual: bytes.len(),
            });
        }

        let path = P::from_bytes(&bytes[36..36 + path_length])?;

        Ok(Self { timestamp_ns, sequence, input_hash, path, output, latency_ns })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::inference::path::LinearPath;

    #[test]
    fn test_decision_trace_new() {
        let path = LinearPath::new(vec![0.5, -0.3], 0.1, 0.5, 0.87);
        let trace = DecisionTrace::new(1000, 42, 0xdeadbeef, path, 0.87, 500);

        assert_eq!(trace.timestamp_ns, 1000);
        assert_eq!(trace.sequence, 42);
        assert_eq!(trace.input_hash, 0xdeadbeef);
        assert_eq!(trace.output, 0.87);
        assert_eq!(trace.latency_ns, 500);
    }

    #[test]
    fn test_decision_trace_confidence() {
        let path = LinearPath::new(vec![0.5], 0.0, 0.0, 0.0).with_probability(0.9);
        let trace = DecisionTrace::new(0, 0, 0, path, 0.0, 0);
        assert!((trace.confidence() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_decision_trace_explain() {
        let path = LinearPath::new(vec![0.5, -0.3], 0.1, 0.5, 0.87);
        let trace = DecisionTrace::new(1000, 42, 0xdeadbeef, path, 0.87, 500);

        let explanation = trace.explain();
        assert!(explanation.contains("Trace #42"));
        assert!(explanation.contains("0x00000000deadbeef"));
        assert!(explanation.contains("Output: 0.87"));
    }

    #[test]
    fn test_decision_trace_serialization_roundtrip() {
        let path = LinearPath::new(vec![0.5, -0.3, 0.2], 0.1, 0.5, 0.87).with_probability(0.87);
        let trace = DecisionTrace::new(1_000_000_000, 42, 0xdeadbeef12345678, path, 0.87, 500_000);

        let bytes = trace.to_bytes();
        let restored: DecisionTrace<LinearPath> =
            DecisionTrace::from_bytes(&bytes).expect("Failed to deserialize");

        assert_eq!(trace.timestamp_ns, restored.timestamp_ns);
        assert_eq!(trace.sequence, restored.sequence);
        assert_eq!(trace.input_hash, restored.input_hash);
        assert!((trace.output - restored.output).abs() < 1e-6);
        // Latency has microsecond precision, so check within 1000ns
        assert!((trace.latency_ns as i64 - restored.latency_ns as i64).abs() < 1000);
    }

    #[test]
    fn test_decision_trace_feature_contributions() {
        let path = LinearPath::new(vec![0.5, -0.3, 0.2], 0.0, 0.0, 0.0);
        let trace = DecisionTrace::new(0, 0, 0, path, 0.0, 0);

        let contributions = trace.feature_contributions();
        assert_eq!(contributions.len(), 3);
        assert!((contributions[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_decision_trace_insufficient_data() {
        let result: Result<DecisionTrace<LinearPath>, _> = DecisionTrace::from_bytes(&[0; 10]);
        assert!(result.is_err());
    }
}
