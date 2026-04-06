//! HashChainCollector - Tamper-evident for safety-critical

use super::super::path::DecisionPath;
use super::super::trace::DecisionTrace;
use super::traits::TraceCollector;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A single entry in the hash chain
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChainEntry<P: DecisionPath> {
    /// Monotonic sequence number
    pub sequence: u64,
    /// Hash of previous entry (SHA-256)
    pub prev_hash: [u8; 32],
    /// The decision trace
    pub trace: DecisionTrace<P>,
    /// SHA-256(sequence || prev_hash || trace)
    pub hash: [u8; 32],
}

/// Chain verification result
#[derive(Debug, Clone)]
pub struct ChainVerification {
    /// Whether the chain is valid
    pub valid: bool,
    /// Number of entries verified
    pub entries_verified: usize,
    /// First broken link index (if any)
    pub first_break: Option<usize>,
    /// Error message (if any)
    pub error: Option<String>,
}

/// Hash-chain collector for safety-critical systems
///
/// Target: <10µs per entry
///
/// # Features
/// - SHA-256 hash chain for tamper evidence
/// - Chain verification on load
/// - Append-only storage
///
/// # Example
///
/// ```ignore
/// use entrenar::monitor::inference::{HashChainCollector, LinearPath};
///
/// let mut collector = HashChainCollector::<LinearPath>::new();
/// collector.record(trace);
///
/// let verification = collector.verify_chain();
/// assert!(verification.valid);
/// ```
pub struct HashChainCollector<P: DecisionPath> {
    pub(crate) entries: Vec<ChainEntry<P>>,
    prev_hash: [u8; 32],
    sequence: u64,
}

impl<P: DecisionPath + Serialize> HashChainCollector<P> {
    /// Create a new hash-chain collector
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            prev_hash: [0u8; 32], // Genesis block has zero hash
            sequence: 0,
        }
    }

    /// Compute hash of an entry
    fn compute_hash(sequence: u64, prev_hash: &[u8; 32], trace: &DecisionTrace<P>) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(sequence.to_le_bytes());
        hasher.update(prev_hash);
        hasher.update(trace.to_bytes());
        hasher.finalize().into()
    }

    /// Verify the entire chain
    pub fn verify_chain(&self) -> ChainVerification {
        if self.entries.is_empty() {
            return ChainVerification {
                valid: true,
                entries_verified: 0,
                first_break: None,
                error: None,
            };
        }

        let mut prev_hash = [0u8; 32]; // Genesis

        for (i, entry) in self.entries.iter().enumerate() {
            // Check sequence
            if entry.sequence != i as u64 {
                return ChainVerification {
                    valid: false,
                    entries_verified: i,
                    first_break: Some(i),
                    error: Some(format!(
                        "Sequence mismatch at index {}: expected {}, got {}",
                        i, i, entry.sequence
                    )),
                };
            }

            // Check prev_hash
            if entry.prev_hash != prev_hash {
                return ChainVerification {
                    valid: false,
                    entries_verified: i,
                    first_break: Some(i),
                    error: Some(format!("Previous hash mismatch at index {i}")),
                };
            }

            // Verify hash
            let computed_hash = Self::compute_hash(entry.sequence, &prev_hash, &entry.trace);
            if entry.hash != computed_hash {
                return ChainVerification {
                    valid: false,
                    entries_verified: i,
                    first_break: Some(i),
                    error: Some(format!("Hash mismatch at index {i}")),
                };
            }

            prev_hash = entry.hash;
        }

        ChainVerification {
            valid: true,
            entries_verified: self.entries.len(),
            first_break: None,
            error: None,
        }
    }

    /// Get all entries
    pub fn entries(&self) -> &[ChainEntry<P>] {
        &self.entries
    }

    /// Get entry by sequence number
    pub fn get(&self, sequence: u64) -> Option<&ChainEntry<P>> {
        self.entries.get(sequence as usize)
    }

    /// Get the latest hash
    pub fn latest_hash(&self) -> [u8; 32] {
        self.entries.last().map_or([0u8; 32], |e| e.hash)
    }

    /// Export chain to JSON
    pub fn to_json(&self) -> serde_json::Result<String>
    where
        P: Serialize,
    {
        serde_json::to_string_pretty(&self.entries)
    }
}

impl<P: DecisionPath + Serialize> TraceCollector<P> for HashChainCollector<P> {
    fn record(&mut self, trace: DecisionTrace<P>) {
        let hash = Self::compute_hash(self.sequence, &self.prev_hash, &trace);

        let entry = ChainEntry { sequence: self.sequence, prev_hash: self.prev_hash, trace, hash };

        self.prev_hash = hash;
        self.sequence += 1;
        self.entries.push(entry);
    }

    fn flush(&mut self) -> std::io::Result<()> {
        // Hash chain is always "flushed" (in memory)
        Ok(())
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

impl<P: DecisionPath + Serialize> Default for HashChainCollector<P> {
    fn default() -> Self {
        Self::new()
    }
}
