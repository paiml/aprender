use std::collections::HashMap;

/// Type alias for N-gram sequences (vector of syscall names)
pub type NGram = Vec<String>;

/// Type alias for N-gram frequency map
pub type NGramMap = HashMap<NGram, usize>;

/// Extracts N-gram sequences from a syscall trace
///
/// For example, with N=3 (trigrams):
/// - Input trace: ["mmap", "read", "write", "close"]
/// - Output N-grams: [["mmap", "read", "write"], ["read", "write", "close"]]
///
/// # Arguments
/// * `syscalls` - Ordered sequence of syscall names from trace
/// * `n` - N-gram size (3 recommended for syscalls per Forrest et al.)
///
/// # Returns
/// `HashMap` mapping each N-gram sequence to its occurrence count
///
/// # Example
/// ```
/// use renacer::sequence::extract_ngrams;
///
/// let syscalls = vec!["mmap".to_string(), "read".to_string(), "write".to_string()];
/// let ngrams = extract_ngrams(&syscalls, 3);
///
/// assert_eq!(ngrams.len(), 1);
/// assert_eq!(ngrams.get(&vec!["mmap".to_string(), "read".to_string(), "write".to_string()]), Some(&1));
/// ```
pub fn extract_ngrams(syscalls: &[String], n: usize) -> NGramMap {
    let mut ngrams: NGramMap = HashMap::new();

    if syscalls.len() < n {
        return ngrams; // Not enough syscalls for N-gram
    }

    // Sliding window of size N
    for window in syscalls.windows(n) {
        let ngram = window.to_vec();
        *ngrams.entry(ngram).or_insert(0) += 1;
    }

    ngrams
}

/// Calculate N-gram coverage (percentage of unique N-grams vs total occurrences)
///
/// High coverage indicates diverse syscall patterns.
/// Low coverage indicates repetitive patterns (tight loops).
pub fn ngram_coverage(ngrams: &NGramMap) -> f64 {
    if ngrams.is_empty() {
        return 0.0;
    }

    let unique_count = ngrams.len();
    let total_count: usize = ngrams.values().sum();

    unique_count as f64 / total_count as f64
}

/// Find most frequent N-grams (useful for identifying hot paths)
pub fn top_ngrams(ngrams: &NGramMap, k: usize) -> Vec<(NGram, usize)> {
    let mut ngram_vec: Vec<_> =
        ngrams.iter().map(|(ngram, count)| (ngram.clone(), *count)).collect();

    // Sort by frequency (descending)
    ngram_vec.sort_by_key(|b| std::cmp::Reverse(b.1));

    ngram_vec.into_iter().take(k).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_ngrams_basic() {
        let syscalls =
            vec!["mmap".to_string(), "read".to_string(), "write".to_string(), "close".to_string()];

        let ngrams = extract_ngrams(&syscalls, 3);

        // Should have 2 trigrams
        assert_eq!(ngrams.len(), 2);

        // Verify specific N-grams
        let ngram1 = vec!["mmap".to_string(), "read".to_string(), "write".to_string()];
        let ngram2 = vec!["read".to_string(), "write".to_string(), "close".to_string()];

        assert_eq!(ngrams.get(&ngram1), Some(&1));
        assert_eq!(ngrams.get(&ngram2), Some(&1));
    }

    #[test]
    fn test_extract_ngrams_repeated() {
        let syscalls = vec![
            "mmap".to_string(),
            "read".to_string(),
            "write".to_string(),
            "mmap".to_string(),
            "read".to_string(),
            "write".to_string(),
        ];

        let ngrams = extract_ngrams(&syscalls, 3);

        // Repeated pattern should increase count
        let ngram = vec!["mmap".to_string(), "read".to_string(), "write".to_string()];
        assert_eq!(ngrams.get(&ngram), Some(&2));
    }

    #[test]
    fn test_extract_ngrams_insufficient_length() {
        let syscalls = vec!["mmap".to_string(), "read".to_string()];

        let ngrams = extract_ngrams(&syscalls, 3);

        // Not enough syscalls for trigrams
        assert_eq!(ngrams.len(), 0);
    }

    #[test]
    fn test_ngram_coverage() {
        let mut ngrams = HashMap::new();
        ngrams.insert(vec!["a".to_string(), "b".to_string()], 1);
        ngrams.insert(vec!["b".to_string(), "c".to_string()], 1);
        ngrams.insert(vec!["c".to_string(), "d".to_string()], 1);

        // 3 unique N-grams, 3 total occurrences = 100% coverage
        assert_eq!(ngram_coverage(&ngrams), 1.0);
    }

    #[test]
    fn test_ngram_coverage_repetitive() {
        let mut ngrams = HashMap::new();
        ngrams.insert(vec!["a".to_string(), "b".to_string()], 10);
        ngrams.insert(vec!["b".to_string(), "c".to_string()], 1);

        // 2 unique N-grams, 11 total occurrences = ~18% coverage
        let coverage = ngram_coverage(&ngrams);
        assert!((coverage - 0.181).abs() < 0.01);
    }

    #[test]
    fn test_top_ngrams() {
        let mut ngrams = HashMap::new();
        ngrams.insert(vec!["a".to_string(), "b".to_string()], 10);
        ngrams.insert(vec!["b".to_string(), "c".to_string()], 5);
        ngrams.insert(vec!["c".to_string(), "d".to_string()], 1);

        let top = top_ngrams(&ngrams, 2);

        assert_eq!(top.len(), 2);
        assert_eq!(top[0].1, 10); // Most frequent first
        assert_eq!(top[1].1, 5);
    }
}
