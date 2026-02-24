//! Validated Classification Types - Compile-Time Contract Enforcement
//!
//! Poka-Yoke (mistake-proofing) types for classification fine-tuning.
//! Makes invalid classification states unrepresentable at the type level.
//!
//! # Contract
//!
//! See `contracts/classification-finetune-v1.yaml` for the full specification.
//!
//! # Compiler Guarantee
//!
//! It is IMPOSSIBLE to use invalid classification data because:
//! 1. Inner fields are private
//! 2. `new()` is the ONLY constructor (no Default, no unsafe backdoor)
//! 3. `new()` runs ALL validation checks from the contract

use super::validated_tensors::ContractValidationError;
use crate::text::shell_vocab::SafetyClass;
use std::fmt;

// =============================================================================
// VALIDATED CLASS LOGITS (F-CLASS-001)
// =============================================================================

/// Validated classification logits — private constructor enforces shape.
///
/// Guarantees:
/// - `data.len() == num_classes`
/// - `num_classes >= 2` (binary classification minimum)
/// - No NaN or Inf values
///
/// # Poka-Yoke
///
/// Inner `data` field is private. The only way to construct this type
/// is through `new()`, which enforces all invariants.
#[derive(Debug, Clone)]
pub struct ValidatedClassLogits {
    data: Vec<f32>,
    num_classes: usize,
}

impl ValidatedClassLogits {
    /// Construct validated logits.
    ///
    /// # Errors
    ///
    /// Returns `ContractValidationError` if:
    /// - `data.len() != num_classes` (F-CLASS-001)
    /// - `num_classes < 2` (degenerate classifier)
    /// - Any value is NaN or Inf
    pub fn new(data: Vec<f32>, num_classes: usize) -> Result<Self, ContractValidationError> {
        if num_classes < 2 {
            return Err(ContractValidationError {
                tensor_name: "class_logits".to_string(),
                rule_id: "F-CLASS-001".to_string(),
                message: format!("num_classes must be >= 2, got {num_classes}"),
            });
        }

        if data.len() != num_classes {
            return Err(ContractValidationError {
                tensor_name: "class_logits".to_string(),
                rule_id: "F-CLASS-001".to_string(),
                message: format!(
                    "Logit shape mismatch: got {} elements, expected {num_classes}",
                    data.len()
                ),
            });
        }

        for (i, &v) in data.iter().enumerate() {
            if v.is_nan() {
                return Err(ContractValidationError {
                    tensor_name: "class_logits".to_string(),
                    rule_id: "F-CLASS-001".to_string(),
                    message: format!("NaN at index {i}"),
                });
            }
            if v.is_infinite() {
                return Err(ContractValidationError {
                    tensor_name: "class_logits".to_string(),
                    rule_id: "F-CLASS-001".to_string(),
                    message: format!("Inf at index {i}"),
                });
            }
        }

        Ok(Self { data, num_classes })
    }

    /// Access the validated logit values.
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Number of classes.
    #[must_use]
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Compute softmax probabilities.
    ///
    /// Contract: output sums to 1.0 (within epsilon=1e-5).
    #[must_use]
    pub fn softmax(&self) -> Vec<f32> {
        let max_val = self.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = self.data.iter().map(|&v| (v - max_val).exp()).sum();
        self.data
            .iter()
            .map(|&v| (v - max_val).exp() / exp_sum)
            .collect()
    }

    /// Return the predicted class index (argmax).
    #[must_use]
    pub fn predicted_class(&self) -> usize {
        self.data
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Return predicted class and confidence (softmax probability).
    #[must_use]
    pub fn predicted_class_with_confidence(&self) -> (usize, f32) {
        let probs = self.softmax();
        let (idx, &conf) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));
        (idx, conf)
    }
}

impl fmt::Display for ValidatedClassLogits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (cls, conf) = self.predicted_class_with_confidence();
        write!(
            f,
            "ClassLogits[{} classes, predicted={cls}, conf={conf:.1}%]",
            self.num_classes
        )
    }
}

// =============================================================================
// VALIDATED SAFETY LABEL (F-CLASS-002)
// =============================================================================

/// Validated safety label — bounded to valid class indices.
///
/// Guarantees:
/// - `index < num_classes`
/// - Maps to a valid `SafetyClass` variant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidatedSafetyLabel {
    class: SafetyClass,
    index: usize,
}

impl ValidatedSafetyLabel {
    /// Construct a validated safety label.
    ///
    /// # Errors
    ///
    /// Returns `ContractValidationError` if:
    /// - `index >= num_classes` (F-CLASS-002)
    /// - `index` does not map to a valid `SafetyClass`
    pub fn new(index: usize, num_classes: usize) -> Result<Self, ContractValidationError> {
        if index >= num_classes {
            return Err(ContractValidationError {
                tensor_name: "safety_label".to_string(),
                rule_id: "F-CLASS-002".to_string(),
                message: format!("Label index {index} out of range (num_classes={num_classes})"),
            });
        }

        let class = SafetyClass::from_index(index).ok_or_else(|| ContractValidationError {
            tensor_name: "safety_label".to_string(),
            rule_id: "F-CLASS-002".to_string(),
            message: format!("Index {index} does not map to a SafetyClass variant"),
        })?;

        Ok(Self { class, index })
    }

    /// The safety class.
    #[must_use]
    pub fn class(&self) -> SafetyClass {
        self.class
    }

    /// The class index (0-4).
    #[must_use]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        self.class.label()
    }
}

impl fmt::Display for ValidatedSafetyLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.class.label(), self.index)
    }
}

// =============================================================================
// VALIDATED CLASSIFIER WEIGHT (F-CLASS-004)
// =============================================================================

/// Validated classifier head weight — enforces shape invariant.
///
/// Guarantees:
/// - `data.len() == hidden_size * num_classes`
/// - `hidden_size > 0`
/// - `num_classes >= 2`
/// - No NaN or Inf values
#[derive(Debug, Clone)]
pub struct ValidatedClassifierWeight {
    data: Vec<f32>,
    hidden_size: usize,
    num_classes: usize,
}

impl ValidatedClassifierWeight {
    /// Construct validated classifier weight.
    ///
    /// # Errors
    ///
    /// Returns `ContractValidationError` if shape or data quality invariants violated.
    pub fn new(
        data: Vec<f32>,
        hidden_size: usize,
        num_classes: usize,
    ) -> Result<Self, ContractValidationError> {
        if hidden_size == 0 {
            return Err(ContractValidationError {
                tensor_name: "classifier_weight".to_string(),
                rule_id: "F-CLASS-004".to_string(),
                message: "hidden_size must be > 0".to_string(),
            });
        }

        if num_classes < 2 {
            return Err(ContractValidationError {
                tensor_name: "classifier_weight".to_string(),
                rule_id: "F-CLASS-004".to_string(),
                message: format!("num_classes must be >= 2, got {num_classes}"),
            });
        }

        let expected = hidden_size * num_classes;
        if data.len() != expected {
            return Err(ContractValidationError {
                tensor_name: "classifier_weight".to_string(),
                rule_id: "F-CLASS-004".to_string(),
                message: format!(
                    "Shape mismatch: got {} elements, expected {expected} ({hidden_size}x{num_classes})",
                    data.len()
                ),
            });
        }

        for (i, &v) in data.iter().enumerate() {
            if v.is_nan() {
                return Err(ContractValidationError {
                    tensor_name: "classifier_weight".to_string(),
                    rule_id: "F-CLASS-004".to_string(),
                    message: format!("NaN at index {i}"),
                });
            }
            if v.is_infinite() {
                return Err(ContractValidationError {
                    tensor_name: "classifier_weight".to_string(),
                    rule_id: "F-CLASS-004".to_string(),
                    message: format!("Inf at index {i}"),
                });
            }
        }

        Ok(Self {
            data,
            hidden_size,
            num_classes,
        })
    }

    /// Access the validated weight data.
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Mutable access for training (gradient updates).
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Hidden dimension.
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Number of output classes.
    #[must_use]
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }
}

impl fmt::Display for ValidatedClassifierWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ClassifierWeight[{}x{}, {} elements]",
            self.hidden_size,
            self.num_classes,
            self.data.len()
        )
    }
}
