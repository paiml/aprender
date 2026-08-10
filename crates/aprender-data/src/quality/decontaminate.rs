//! N-gram decontamination for benchmark safety.
//!
//! Detects overlap between training data and evaluation benchmarks
//! using n-gram fingerprinting. Training samples that exceed the
//! overlap threshold are flagged for removal.
//!
//! # Algorithm
//!
//! 1. Build n-gram set from each reference benchmark sample
//! 2. For each training sample, compute n-gram overlap ratio
//! 3. Flag samples exceeding threshold (default 50%)
//!
//! # References
//!
//! - Spec §12.1: Decontamination Protocol
//! - AC-016: <1% n-gram overlap between training and eval data
//! - GH-9: `apr validate --decontaminate`

use std::collections::HashSet;

/// Result of decontamination check on a single sample.
#[derive(Debug, Clone)]
pub struct ContaminationResult {
    /// Index of the training sample
    pub sample_index: usize,
    /// Maximum overlap ratio with any reference sample (0.0 to 1.0)
    pub max_overlap: f64,
    /// Index of the reference sample with highest overlap
    pub matched_reference: usize,
    /// Whether this sample exceeds the contamination threshold
    pub contaminated: bool,
}

/// Summary report of decontamination check.
#[derive(Debug, Clone)]
pub struct DecontaminationReport {
    /// N-gram size used
    pub ngram_size: usize,
    /// Overlap threshold used
    pub threshold: f64,
    /// Total training samples checked
    pub total_samples: usize,
    /// Training samples that yielded at least one n-gram and were therefore
    /// actually compared against the reference set.
    ///
    /// Samples shorter than `ngram_size` (and every sample when `ngram_size`
    /// is 0) produce no n-grams and are silently un-checkable. When this is 0
    /// but `total_samples` is not, the check was **vacuous**: a clean
    /// `contamination_rate` of 0.0 proves nothing.
    pub evaluated_samples: usize,
    /// Reference samples that yielded at least one n-gram. Zero here also
    /// makes the check vacuous — nothing could ever be matched against.
    pub evaluated_references: usize,
    /// Number of contaminated samples
    pub contaminated_count: usize,
    /// Contamination rate (0.0 to 1.0)
    pub contamination_rate: f64,
    /// Per-sample results (only contaminated samples included)
    pub flagged: Vec<ContaminationResult>,
}

impl DecontaminationReport {
    /// Whether the configuration made the check meaningless.
    ///
    /// True when there was input to check but nothing could be compared —
    /// e.g. `ngram_size` exceeds the length of every sample, or is 0. A
    /// caller enforcing a contamination gate must treat this as a failure,
    /// not as a pass.
    #[must_use]
    pub fn is_vacuous(&self) -> bool {
        (self.total_samples > 0 && self.evaluated_samples == 0) || self.evaluated_references == 0
    }
}

/// Extract character-level n-grams from text.
///
/// Returns an empty set when `n` is 0 or larger than the whitespace-stripped
/// text — a zero-width window has no meaning, and this function must remain
/// total: `n` reaches here from a user-supplied `--ngram` flag.
fn extract_ngrams(text: &str, n: usize) -> HashSet<Vec<char>> {
    if n == 0 {
        return HashSet::new();
    }

    let chars: Vec<char> = text
        .chars()
        .filter(|c| !c.is_whitespace())
        .flat_map(|c| c.to_lowercase())
        .collect();

    if chars.len() < n {
        return HashSet::new();
    }

    chars.windows(n).map(|w| w.to_vec()).collect()
}

/// Compute n-gram overlap ratio between two texts.
///
/// Returns the fraction of n-grams in `candidate` that also
/// appear in `reference`. Range: 0.0 (no overlap) to 1.0 (complete).
pub fn ngram_overlap(candidate: &str, reference: &str, n: usize) -> f64 {
    let cand_ngrams = extract_ngrams(candidate, n);
    if cand_ngrams.is_empty() {
        return 0.0;
    }

    let ref_ngrams = extract_ngrams(reference, n);
    let intersection = cand_ngrams.intersection(&ref_ngrams).count();

    intersection as f64 / cand_ngrams.len() as f64
}

/// Check training data against reference benchmarks for contamination.
///
/// # Arguments
///
/// * `training_samples` - Training data texts
/// * `reference_samples` - Benchmark/eval texts to check against
/// * `ngram_size` - Size of n-grams (default: 10)
/// * `threshold` - Overlap ratio above which a sample is contaminated
///
/// # Returns
///
/// `DecontaminationReport` with per-sample results and summary stats.
pub fn check_contamination(
    training_samples: &[&str],
    reference_samples: &[&str],
    ngram_size: usize,
    threshold: f64,
) -> DecontaminationReport {
    // Pre-compute reference n-gram sets
    let ref_ngram_sets: Vec<HashSet<Vec<char>>> = reference_samples
        .iter()
        .map(|s| extract_ngrams(s, ngram_size))
        .collect();

    let evaluated_references = ref_ngram_sets.iter().filter(|s| !s.is_empty()).count();

    let mut flagged = Vec::new();
    let mut evaluated_samples = 0usize;

    for (i, sample) in training_samples.iter().enumerate() {
        let cand_ngrams = extract_ngrams(sample, ngram_size);
        if cand_ngrams.is_empty() {
            continue;
        }
        evaluated_samples += 1;

        let mut max_overlap = 0.0_f64;
        let mut matched_ref = 0;

        for (j, ref_set) in ref_ngram_sets.iter().enumerate() {
            let intersection = cand_ngrams.intersection(ref_set).count();
            let overlap = intersection as f64 / cand_ngrams.len() as f64;

            if overlap > max_overlap {
                max_overlap = overlap;
                matched_ref = j;
            }
        }

        if max_overlap > threshold {
            flagged.push(ContaminationResult {
                sample_index: i,
                max_overlap,
                matched_reference: matched_ref,
                contaminated: true,
            });
        }
    }

    let contaminated_count = flagged.len();
    let total = training_samples.len();
    let rate = if total > 0 {
        contaminated_count as f64 / total as f64
    } else {
        0.0
    };

    DecontaminationReport {
        ngram_size,
        threshold,
        total_samples: total,
        evaluated_samples,
        evaluated_references,
        contaminated_count,
        contamination_rate: rate,
        flagged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_ngrams() {
        let ngrams = extract_ngrams("hello world", 3);
        // "helloworld" -> "hel", "ell", "llo", "low", "owo", "wor", "orl", "rld"
        assert_eq!(ngrams.len(), 8);
    }

    #[test]
    fn test_extract_ngrams_short_text() {
        let ngrams = extract_ngrams("hi", 10);
        assert!(ngrams.is_empty());
    }

    #[test]
    fn test_ngram_overlap_identical() {
        let overlap = ngram_overlap("def fibonacci(n):", "def fibonacci(n):", 5);
        assert!((overlap - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ngram_overlap_no_match() {
        let overlap = ngram_overlap(
            "completely different text about cooking",
            "def fibonacci(n): return n if n < 2",
            10,
        );
        assert!(overlap < 0.1);
    }

    #[test]
    fn test_ngram_overlap_partial() {
        let overlap = ngram_overlap(
            "def fibonacci(n): return n if n < 2 else fibonacci(n-1)",
            "def fibonacci(n): return fib(n-1) + fib(n-2)",
            5,
        );
        // Partial overlap from shared prefix
        assert!(overlap > 0.0);
        assert!(overlap < 1.0);
    }

    #[test]
    fn test_check_contamination_clean() {
        let training = vec![
            "def sort_list(lst): return sorted(lst)",
            "def reverse_string(s): return s[::-1]",
        ];
        let reference =
            vec!["def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)"];

        let report = check_contamination(&training, &reference, 10, 0.5);
        assert_eq!(report.contaminated_count, 0);
        assert!(report.contamination_rate < 0.01);
    }

    #[test]
    fn test_check_contamination_flagged() {
        let reference_text =
            "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)";
        let training = vec![
            "def sort_list(lst): return sorted(lst)",
            reference_text, // exact copy
        ];
        let reference = vec![reference_text];

        let report = check_contamination(&training, &reference, 10, 0.5);
        assert_eq!(report.contaminated_count, 1);
        assert_eq!(report.flagged[0].sample_index, 1);
        assert!((report.flagged[0].max_overlap - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_check_contamination_threshold() {
        let training =
            vec!["def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)"];
        let reference = vec!["def fibonacci(n): return n if n < 2 else fib(n-1) + fib(n-2)"];

        // Strict threshold should catch partial overlap
        let strict = check_contamination(&training, &reference, 5, 0.3);
        // Lenient threshold should pass
        let lenient = check_contamination(&training, &reference, 10, 0.9);

        assert!(strict.contaminated_count >= lenient.contaminated_count);
    }

    /// A zero n-gram size arrives from a user-supplied `--ngram` flag and used
    /// to reach `slice::windows(0)`, aborting the process with
    /// "window size must be non-zero". It must return, and it must not claim
    /// the corpus is clean.
    #[test]
    fn test_ngram_size_zero_does_not_panic_and_is_vacuous() {
        let text = "def fibonacci(n): return n if n < 2 else fibonacci(n-1)";
        assert!(extract_ngrams(text, 0).is_empty());
        assert!((ngram_overlap(text, text, 0) - 0.0).abs() < f64::EPSILON);

        let report = check_contamination(&[text, "unrelated gardening text"], &[text], 0, 0.5);
        assert_eq!(report.total_samples, 2);
        assert_eq!(report.evaluated_samples, 0);
        assert_eq!(report.contaminated_count, 0);
        assert!(
            report.is_vacuous(),
            "ngram_size 0 compares nothing; a 0.00% rate must not read as a pass"
        );
    }

    /// An n-gram larger than every sample also silently compares nothing —
    /// the same vacuous-pass hazard, reached without a panic.
    #[test]
    fn test_oversized_ngram_is_vacuous_not_clean() {
        let leaked = "the capital of france is paris";
        let report = check_contamination(&[leaked, leaked], &[leaked], 1000, 0.5);
        assert_eq!(report.total_samples, 2);
        assert_eq!(report.evaluated_samples, 0);
        assert_eq!(report.contaminated_count, 0);
        assert!((report.contamination_rate - 0.0).abs() < f64::EPSILON);
        assert!(
            report.is_vacuous(),
            "two byte-identical copies of the benchmark must never be reported as clean"
        );
    }

    /// A reference set whose samples are all too short compares nothing
    /// either, even when the training samples are long enough.
    #[test]
    fn test_short_references_only_is_vacuous() {
        let training = "def fibonacci(n): return n if n < 2 else fibonacci(n-1)";
        let report = check_contamination(&[training], &["hi"], 10, 0.5);
        assert_eq!(report.evaluated_samples, 1);
        assert_eq!(report.evaluated_references, 0);
        assert!(report.is_vacuous());
    }

    /// The healthy path must NOT be flagged as vacuous, or the gate would
    /// fail closed on every real corpus.
    #[test]
    fn test_real_check_is_not_vacuous() {
        let leaked = "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)";
        let report = check_contamination(
            &["def sort_list(lst): return sorted(lst)", leaked],
            &[leaked],
            10,
            0.5,
        );
        assert_eq!(report.evaluated_samples, 2);
        assert_eq!(report.evaluated_references, 1);
        assert!(!report.is_vacuous());
        assert_eq!(report.contaminated_count, 1);
    }

    #[test]
    fn test_empty_inputs() {
        let report = check_contamination(&[], &["some reference"], 10, 0.5);
        assert_eq!(report.total_samples, 0);
        assert_eq!(report.contaminated_count, 0);

        let report2 = check_contamination(&["some training"], &[], 10, 0.5);
        assert_eq!(report2.contaminated_count, 0);
    }
}
