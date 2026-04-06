//! Counterfactual explanation structure and methods.

use serde::{Deserialize, Serialize};

use super::error::CounterfactualError;
use super::feature_change::FeatureChange;

/// Counterfactual explanation for a decision
///
/// Answers: "What minimal change would have flipped the decision?"
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Counterfactual {
    /// Original input that produced the decision
    pub original_input: Vec<f32>,
    /// Original decision/class
    pub original_decision: usize,
    /// Original confidence
    pub original_confidence: f32,
    /// Modified input that would flip the decision
    pub counterfactual_input: Vec<f32>,
    /// The alternative decision
    pub alternative_decision: usize,
    /// Alternative confidence
    pub alternative_confidence: f32,
    /// Which features changed and by how much
    pub changes: Vec<FeatureChange>,
    /// L1 distance (sparsity of changes)
    pub sparsity: f32,
    /// L2 distance (magnitude of changes)
    pub distance: f32,
}

impl Counterfactual {
    /// Create a new counterfactual
    pub fn new(
        original_input: Vec<f32>,
        original_decision: usize,
        original_confidence: f32,
        counterfactual_input: Vec<f32>,
        alternative_decision: usize,
        alternative_confidence: f32,
    ) -> Self {
        assert_eq!(original_input.len(), counterfactual_input.len(), "Input dimensions must match");

        let mut changes = Vec::new();
        let mut l1 = 0.0f32;
        let mut l2 = 0.0f32;

        for i in 0..original_input.len() {
            let delta = counterfactual_input[i] - original_input[i];
            if delta.abs() > 1e-6 {
                changes.push(FeatureChange::new(i, original_input[i], counterfactual_input[i]));
                l1 += delta.abs();
                l2 += delta * delta;
            }
        }

        Self {
            original_input,
            original_decision,
            original_confidence,
            counterfactual_input,
            alternative_decision,
            alternative_confidence,
            changes,
            sparsity: l1,
            distance: l2.sqrt(),
        }
    }

    /// Generate natural language explanation
    pub fn explain(&self) -> String {
        let mut explanation = format!(
            "Original decision: {} (confidence: {:.1}%)\n",
            self.original_decision,
            self.original_confidence * 100.0
        );
        explanation.push_str(&format!(
            "Alternative decision: {} (confidence: {:.1}%)\n",
            self.alternative_decision,
            self.alternative_confidence * 100.0
        ));
        explanation.push_str(&format!(
            "\nThe decision would have been {} if:\n",
            self.alternative_decision
        ));

        // Sort changes by absolute delta (most impactful first)
        let mut sorted_changes = self.changes.clone();
        sorted_changes.sort_by(|a, b| {
            b.abs_delta().partial_cmp(&a.abs_delta()).unwrap_or(std::cmp::Ordering::Equal)
        });

        for change in sorted_changes.iter().take(5) {
            let sign = if change.delta >= 0.0 { "+" } else { "" };
            let default_name = format!("feature[{}]", change.feature_idx);
            let name = change.feature_name.as_deref().unwrap_or(&default_name);
            explanation.push_str(&format!(
                "  - {}: {:.4} → {:.4} ({}{:.4})\n",
                name, change.original_value, change.counterfactual_value, sign, change.delta
            ));
        }

        if self.changes.len() > 5 {
            explanation.push_str(&format!("  ... and {} more changes\n", self.changes.len() - 5));
        }

        explanation.push_str(&format!("\nSparsity (L1): {:.4}\n", self.sparsity));
        explanation.push_str(&format!("Distance (L2): {:.4}\n", self.distance));

        explanation
    }

    /// Number of features that changed
    pub fn n_changes(&self) -> usize {
        self.changes.len()
    }

    /// Check if this is a valid counterfactual (decision actually flipped)
    pub fn is_valid(&self) -> bool {
        self.original_decision != self.alternative_decision
    }

    /// Set feature names for all changes
    pub fn with_feature_names(mut self, names: &[String]) -> Self {
        for change in &mut self.changes {
            if change.feature_idx < names.len() {
                change.feature_name = Some(names[change.feature_idx].clone());
            }
        }
        self
    }

    /// Convert to binary format
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(1); // version

        // Original decision info
        bytes.extend_from_slice(&(self.original_decision as u32).to_le_bytes());
        bytes.extend_from_slice(&self.original_confidence.to_le_bytes());
        bytes.extend_from_slice(&(self.alternative_decision as u32).to_le_bytes());
        bytes.extend_from_slice(&self.alternative_confidence.to_le_bytes());

        // Original input
        bytes.extend_from_slice(&(self.original_input.len() as u32).to_le_bytes());
        for v in &self.original_input {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        // Counterfactual input
        for v in &self.counterfactual_input {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        // Changes (compact: only store changed indices and deltas)
        bytes.extend_from_slice(&(self.changes.len() as u32).to_le_bytes());
        for change in &self.changes {
            bytes.extend_from_slice(&(change.feature_idx as u32).to_le_bytes());
            bytes.extend_from_slice(&change.original_value.to_le_bytes());
            bytes.extend_from_slice(&change.counterfactual_value.to_le_bytes());
            bytes.extend_from_slice(&change.delta.to_le_bytes());

            // Feature name (length-prefixed)
            if let Some(name) = &change.feature_name {
                bytes.extend_from_slice(&(name.len() as u32).to_le_bytes());
                bytes.extend_from_slice(name.as_bytes());
            } else {
                bytes.extend_from_slice(&0u32.to_le_bytes());
            }
        }

        // Metrics
        bytes.extend_from_slice(&self.sparsity.to_le_bytes());
        bytes.extend_from_slice(&self.distance.to_le_bytes());

        bytes
    }

    /// Reconstruct from binary format
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CounterfactualError> {
        if bytes.len() < 21 {
            return Err(CounterfactualError::InsufficientData {
                expected: 21,
                actual: bytes.len(),
            });
        }

        let mut reader = ByteReader::new(bytes);

        let version = reader.read_u8()?;
        if version != 1 {
            return Err(CounterfactualError::VersionMismatch { expected: 1, actual: version });
        }

        let original_decision = reader.read_u32_as_usize()?;
        let original_confidence = reader.read_f32()?;
        let alternative_decision = reader.read_u32_as_usize()?;
        let alternative_confidence = reader.read_f32()?;
        let n_features = reader.read_u32_as_usize()?;

        let original_input = reader.read_f32_vec_n(n_features)?;
        let counterfactual_input = reader.read_f32_vec_n(n_features)?;

        let n_changes = reader.read_u32_as_usize()?;
        let mut changes = Vec::with_capacity(n_changes);
        for _ in 0..n_changes {
            changes.push(reader.read_feature_change()?);
        }

        let sparsity = reader.read_f32()?;
        let distance = reader.read_f32()?;

        Ok(Self {
            original_input,
            original_decision,
            original_confidence,
            counterfactual_input,
            alternative_decision,
            alternative_confidence,
            changes,
            sparsity,
            distance,
        })
    }
}

/// Stateful byte reader that tracks offset and validates bounds.
struct ByteReader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> ByteReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    fn read_u8(&mut self) -> Result<u8, CounterfactualError> {
        self.ensure_available(1)?;
        let val = self.data[self.offset];
        self.offset += 1;
        Ok(val)
    }

    fn read_u32(&mut self) -> Result<u32, CounterfactualError> {
        self.ensure_available(4)?;
        let o = self.offset;
        let val = u32::from_le_bytes([
            self.data[o],
            self.data[o + 1],
            self.data[o + 2],
            self.data[o + 3],
        ]);
        self.offset += 4;
        Ok(val)
    }

    fn read_u32_as_usize(&mut self) -> Result<usize, CounterfactualError> {
        Ok(self.read_u32()? as usize)
    }

    fn read_f32(&mut self) -> Result<f32, CounterfactualError> {
        self.ensure_available(4)?;
        let o = self.offset;
        let val = f32::from_le_bytes([
            self.data[o],
            self.data[o + 1],
            self.data[o + 2],
            self.data[o + 3],
        ]);
        self.offset += 4;
        Ok(val)
    }

    fn read_f32_vec_n(&mut self, n: usize) -> Result<Vec<f32>, CounterfactualError> {
        let mut vec = Vec::with_capacity(n);
        for _ in 0..n {
            vec.push(self.read_f32()?);
        }
        Ok(vec)
    }

    fn read_string(&mut self, len: usize) -> Result<String, CounterfactualError> {
        self.ensure_available(len)?;
        let s = String::from_utf8_lossy(&self.data[self.offset..self.offset + len]).to_string();
        self.offset += len;
        Ok(s)
    }

    fn read_feature_change(&mut self) -> Result<FeatureChange, CounterfactualError> {
        let feature_idx = self.read_u32_as_usize()?;
        let original_value = self.read_f32()?;
        let counterfactual_value = self.read_f32()?;
        let delta = self.read_f32()?;
        let name_len = self.read_u32_as_usize()?;
        let feature_name = if name_len > 0 { Some(self.read_string(name_len)?) } else { None };
        Ok(FeatureChange { feature_idx, feature_name, original_value, counterfactual_value, delta })
    }

    fn ensure_available(&self, needed: usize) -> Result<(), CounterfactualError> {
        if self.offset + needed > self.data.len() {
            return Err(CounterfactualError::InsufficientData {
                expected: self.offset + needed,
                actual: self.data.len(),
            });
        }
        Ok(())
    }
}
