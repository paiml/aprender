// Noise filtering using Zeller's Delta Debugging approach
//
// Scientific Foundation:
// [7] Zeller, A. (2002). Isolating cause-effect chains from computer programs.
//     FSE-10. Delta Debugging minimizes differences between failing/passing runs.
//
// Key Insight: Some syscalls are inherently noisy (high variance) due to:
// - Network I/O latency
// - Random number generation
// - OS scheduling jitter
//
// Filtering noisy syscalls BEFORE statistical testing reduces false positives.

use std::collections::HashMap;
use trueno::Vector;

/// Represents a syscall with its baseline measurement distribution
#[derive(Debug, Clone)]
pub struct SyscallDistribution {
    pub name: String,
    pub measurements: Vec<f32>,
}

impl SyscallDistribution {
    /// Calculate coefficient of variation (CV = `std_dev` / mean)
    ///
    /// CV is a normalized measure of variability:
    /// - CV near 0: very stable (e.g., CV=0.1 means std is 10% of mean)
    /// - CV near 1: highly variable (e.g., CV=1.0 means std equals mean)
    /// - CV > 1: extreme variability
    ///
    /// # Example
    /// ```ignore
    /// use renacer::regression::noise_filter::SyscallDistribution;
    ///
    /// let stable = SyscallDistribution {
    ///     name: "mmap".to_string(),
    ///     measurements: vec![10.0, 11.0, 10.0, 12.0, 10.0],
    /// };
    /// assert!(stable.coefficient_of_variation() < 0.2); // Stable
    ///
    /// let noisy = SyscallDistribution {
    ///     name: "socket".to_string(),
    ///     measurements: vec![5.0, 50.0, 3.0, 45.0, 2.0],
    /// };
    /// assert!(noisy.coefficient_of_variation() > 0.5); // Noisy
    /// ```
    pub fn coefficient_of_variation(&self) -> f32 {
        if self.measurements.is_empty() {
            return 0.0;
        }

        let vec = Vector::from_slice(&self.measurements);

        // trueno 0.7.0 returns Result for mean and stddev
        let Ok(mean) = vec.mean() else {
            return 0.0;
        };
        let Ok(std) = vec.stddev() else {
            return 0.0;
        };

        if mean.abs() < 1e-6 {
            // Avoid division by zero (all measurements near zero)
            return 0.0;
        }

        std / mean.abs()
    }

    /// Check if this syscall is "noisy" based on CV threshold
    pub fn is_noisy(&self, threshold: f64) -> bool {
        f64::from(self.coefficient_of_variation()) > threshold
    }
}

/// Filter out noisy syscalls from baseline distributions
///
/// Uses Delta Debugging principle: Remove syscalls with high variance
/// (noisy) before statistical comparison to isolate true signal from noise.
///
/// # Arguments
/// * `distributions` - Map of syscall name → baseline measurements
/// * `noise_threshold` - CV threshold (default: 0.5)
///
/// # Returns
/// Tuple of (`stable_syscalls`, `filtered_out_syscalls`)
///
/// # Example
/// ```ignore
/// use renacer::regression::noise_filter::{filter_noisy_syscalls, SyscallDistribution};
/// use std::collections::HashMap;
///
/// let mut distributions = HashMap::new();
/// distributions.insert("mmap".to_string(), vec![10.0, 11.0, 10.0]);
/// distributions.insert("socket".to_string(), vec![5.0, 50.0, 3.0]);
///
/// let (stable, noisy) = filter_noisy_syscalls(&distributions, 0.5);
/// assert!(stable.contains_key("mmap"));  // Stable syscall kept
/// assert!(!stable.contains_key("socket")); // Noisy syscall filtered
/// assert!(noisy.contains(&"socket".to_string()));
/// ```
pub fn filter_noisy_syscalls(
    distributions: &HashMap<String, Vec<f32>>,
    noise_threshold: f64,
) -> (HashMap<String, Vec<f32>>, Vec<String>) {
    let mut stable_syscalls = HashMap::new();
    let mut filtered_out = Vec::new();

    for (name, measurements) in distributions {
        let dist = SyscallDistribution { name: name.clone(), measurements: measurements.clone() };

        if dist.is_noisy(noise_threshold) {
            filtered_out.push(name.clone());
        } else {
            stable_syscalls.insert(name.clone(), measurements.clone());
        }
    }

    (stable_syscalls, filtered_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coefficient_of_variation_stable() {
        let dist = SyscallDistribution {
            name: "mmap".to_string(),
            measurements: vec![10.0, 11.0, 10.0, 12.0, 10.0],
        };

        let cv = dist.coefficient_of_variation();
        assert!(cv < 0.2, "Stable syscall should have low CV, got {}", cv);
    }

    #[test]
    fn test_coefficient_of_variation_noisy() {
        let dist = SyscallDistribution {
            name: "socket".to_string(),
            measurements: vec![5.0, 50.0, 3.0, 45.0, 2.0],
        };

        let cv = dist.coefficient_of_variation();
        assert!(cv > 0.5, "Noisy syscall should have high CV, got {}", cv);
    }

    #[test]
    fn test_coefficient_of_variation_constant() {
        let dist = SyscallDistribution {
            name: "brk".to_string(),
            measurements: vec![100.0, 100.0, 100.0],
        };

        let cv = dist.coefficient_of_variation();
        assert_eq!(cv, 0.0, "Constant values should have CV=0");
    }

    #[test]
    fn test_coefficient_of_variation_empty() {
        let dist = SyscallDistribution { name: "empty".to_string(), measurements: vec![] };

        assert_eq!(dist.coefficient_of_variation(), 0.0);
    }

    #[test]
    fn test_is_noisy() {
        let stable = SyscallDistribution {
            name: "mmap".to_string(),
            measurements: vec![10.0, 11.0, 10.0, 12.0],
        };
        assert!(!stable.is_noisy(0.5));

        let noisy =
            SyscallDistribution { name: "socket".to_string(), measurements: vec![5.0, 50.0, 3.0] };
        assert!(noisy.is_noisy(0.5));
    }

    #[test]
    fn test_filter_noisy_syscalls() {
        let mut distributions = HashMap::new();

        // Stable syscalls
        distributions.insert("mmap".to_string(), vec![10.0, 11.0, 10.0, 12.0]);
        distributions.insert("brk".to_string(), vec![5.0, 5.0, 5.0, 5.0]);

        // Noisy syscall
        distributions.insert("socket".to_string(), vec![5.0, 50.0, 3.0, 45.0]);

        let (stable, noisy) = filter_noisy_syscalls(&distributions, 0.5);

        // Should keep stable syscalls
        assert!(stable.contains_key("mmap"));
        assert!(stable.contains_key("brk"));

        // Should filter noisy syscall
        assert!(!stable.contains_key("socket"));
        assert!(noisy.contains(&"socket".to_string()));
    }

    #[test]
    fn test_filter_noisy_syscalls_threshold_strict() {
        let mut distributions = HashMap::new();
        distributions.insert("mmap".to_string(), vec![10.0, 15.0, 12.0]); // CV ~ 0.2

        // Strict threshold (0.1): should filter out
        let (stable, noisy) = filter_noisy_syscalls(&distributions, 0.1);
        assert!(stable.is_empty());
        assert!(noisy.contains(&"mmap".to_string()));
    }

    #[test]
    fn test_filter_noisy_syscalls_threshold_permissive() {
        let mut distributions = HashMap::new();
        distributions.insert("socket".to_string(), vec![5.0, 50.0, 3.0]); // High CV

        // Permissive threshold (2.0): should NOT filter out
        let (stable, noisy) = filter_noisy_syscalls(&distributions, 2.0);
        assert!(stable.contains_key("socket"));
        assert!(noisy.is_empty());
    }
}
