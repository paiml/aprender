//! Context destruction ordering validation.
//!
//! CUDA contexts must be destroyed in valid orders (typically reverse
//! creation order, but other orders are sometimes valid). This module
//! generates and validates destruction orderings.

use serde::{Deserialize, Serialize};

/// A destruction ordering (permutation of context indices).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DestructionOrdering {
    /// Context indices in destruction order.
    pub order: Vec<usize>,
}

impl DestructionOrdering {
    /// Create a new destruction ordering.
    #[must_use]
    pub fn new(order: Vec<usize>) -> Self {
        Self { order }
    }

    /// Returns true if this is the reverse of creation order (LIFO).
    #[must_use]
    pub fn is_reverse(&self) -> bool {
        if self.order.is_empty() {
            return true;
        }
        let n = self.order.len();
        self.order
            .iter()
            .enumerate()
            .all(|(i, &idx)| idx == n - 1 - i)
    }

    /// Returns true if this is forward creation order (FIFO).
    #[must_use]
    pub fn is_forward(&self) -> bool {
        self.order.iter().enumerate().all(|(i, &idx)| idx == i)
    }
}

/// Generate all possible destruction orderings for N contexts.
///
/// Returns N! orderings. Use with caution for N > 8.
#[must_use]
pub fn generate_destruction_orderings(n: usize) -> Vec<DestructionOrdering> {
    if n == 0 {
        return vec![DestructionOrdering::new(vec![])];
    }

    let mut indices: Vec<usize> = (0..n).collect();
    let mut orderings = Vec::new();

    // Heap's algorithm for generating permutations
    heap_permute(&mut indices, n, &mut orderings);

    orderings
}

fn heap_permute(arr: &mut [usize], k: usize, result: &mut Vec<DestructionOrdering>) {
    if k == 1 {
        result.push(DestructionOrdering::new(arr.to_vec()));
        return;
    }

    heap_permute(arr, k - 1, result);

    for i in 0..k - 1 {
        if k.is_multiple_of(2) {
            arr.swap(i, k - 1);
        } else {
            arr.swap(0, k - 1);
        }
        heap_permute(arr, k - 1, result);
    }
}

/// Validation result for a destruction ordering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderingValidation {
    /// Ordering is valid (no dependency violations).
    Valid,
    /// Ordering violates dependencies.
    Invalid {
        /// Description of the violation.
        reason: String,
    },
}

/// Validate that an ordering is a valid permutation of 0..n.
#[must_use]
pub fn validate_ordering(ordering: &DestructionOrdering, n: usize) -> OrderingValidation {
    if ordering.order.len() != n {
        return OrderingValidation::Invalid {
            reason: format!("ordering length {} != expected {}", ordering.order.len(), n),
        };
    }

    let mut seen = vec![false; n];
    for &idx in &ordering.order {
        if idx >= n {
            return OrderingValidation::Invalid {
                reason: format!("index {idx} >= n={n}"),
            };
        }
        if seen[idx] {
            return OrderingValidation::Invalid {
                reason: format!("duplicate index {idx}"),
            };
        }
        seen[idx] = true;
    }

    OrderingValidation::Valid
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_ordering_is_reverse() {
        let ordering = DestructionOrdering::new(vec![]);
        assert!(ordering.is_reverse());
        assert!(ordering.is_forward());
    }

    #[test]
    fn reverse_ordering_detected() {
        let ordering = DestructionOrdering::new(vec![2, 1, 0]);
        assert!(ordering.is_reverse());
        assert!(!ordering.is_forward());
    }

    #[test]
    fn forward_ordering_detected() {
        let ordering = DestructionOrdering::new(vec![0, 1, 2]);
        assert!(ordering.is_forward());
        assert!(!ordering.is_reverse());
    }

    #[test]
    fn generate_orderings_factorial_count() {
        let orderings = generate_destruction_orderings(3);
        assert_eq!(orderings.len(), 6); // 3! = 6
    }

    #[test]
    fn generate_orderings_includes_reverse() {
        let orderings = generate_destruction_orderings(3);
        assert!(orderings.iter().any(DestructionOrdering::is_reverse));
    }

    #[test]
    fn validate_ordering_valid() {
        let ordering = DestructionOrdering::new(vec![1, 0, 2]);
        assert_eq!(validate_ordering(&ordering, 3), OrderingValidation::Valid);
    }

    #[test]
    fn validate_ordering_wrong_length() {
        let ordering = DestructionOrdering::new(vec![0, 1]);
        let result = validate_ordering(&ordering, 3);
        assert!(matches!(result, OrderingValidation::Invalid { .. }));
    }

    #[test]
    fn validate_ordering_duplicate() {
        let ordering = DestructionOrdering::new(vec![0, 0, 2]);
        let result = validate_ordering(&ordering, 3);
        assert!(matches!(result, OrderingValidation::Invalid { .. }));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn generated_orderings_are_valid(n in 1usize..6) {
            let orderings = generate_destruction_orderings(n);
            for ordering in &orderings {
                prop_assert_eq!(validate_ordering(ordering, n), OrderingValidation::Valid);
            }
        }
    }
}
