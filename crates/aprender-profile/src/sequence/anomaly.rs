use crate::cluster::Severity;
use crate::sequence::ngram::{NGram, NGramMap};

/// Type of sequence anomaly detected
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnomalyType {
    /// Sequence appears in current trace but not in baseline
    NewSequence,

    /// Sequence appears in baseline but not in current trace
    MissingSequence,

    /// Sequence frequency changed significantly between traces
    FrequencyChange,
}

/// A detected sequence anomaly with context
#[derive(Debug, Clone)]
pub struct SequenceAnomaly {
    /// The N-gram sequence that triggered the anomaly
    pub ngram: NGram,

    /// Occurrence count in baseline trace
    pub baseline_freq: usize,

    /// Occurrence count in current trace
    pub current_freq: usize,

    /// Type of anomaly detected
    pub anomaly_type: AnomalyType,

    /// Severity level for prioritization
    pub severity: Severity,
}

impl SequenceAnomaly {
    /// Calculate percentage change in frequency
    pub fn frequency_change_percent(&self) -> f64 {
        if self.baseline_freq == 0 {
            return if self.current_freq > 0 { 100.0 } else { 0.0 };
        }

        let delta = self.current_freq as f64 - self.baseline_freq as f64;
        (delta / self.baseline_freq as f64) * 100.0
    }

    /// Format anomaly as human-readable string
    pub fn to_report_string(&self) -> String {
        match self.anomaly_type {
            AnomalyType::NewSequence => {
                format!(
                    "⚠️ NEW SEQUENCE: {} ({})\n  Baseline: 0 occurrences\n  Current: {} occurrences",
                    ngram_to_string(&self.ngram),
                    severity_emoji(self.severity),
                    self.current_freq
                )
            }
            AnomalyType::MissingSequence => {
                format!(
                    "⚠️ MISSING SEQUENCE: {}\n  Baseline: {} occurrences\n  Current: 0 occurrences",
                    ngram_to_string(&self.ngram),
                    self.baseline_freq
                )
            }
            AnomalyType::FrequencyChange => {
                format!(
                    "⚠️ FREQUENCY CHANGE: {}\n  Baseline: {} occurrences\n  Current: {} occurrences ({:+.1}%)",
                    ngram_to_string(&self.ngram),
                    self.baseline_freq,
                    self.current_freq,
                    self.frequency_change_percent()
                )
            }
        }
    }
}

/// Detect sequence anomalies by comparing N-gram frequency distributions
///
/// # Arguments
/// * `baseline_ngrams` - N-grams from baseline (golden) trace
/// * `current_ngrams` - N-grams from current trace
/// * `frequency_threshold` - Minimum frequency change to flag (0.0-1.0)
///
/// # Returns
/// Vector of detected anomalies sorted by severity (Critical → Low)
///
/// # Example
/// ```
/// use renacer::sequence::{extract_ngrams, detect_sequence_anomalies};
///
/// let baseline_syscalls = vec!["mmap".to_string(), "read".to_string()];
/// let current_syscalls = vec!["socket".to_string(), "connect".to_string()];
///
/// let baseline_ngrams = extract_ngrams(&baseline_syscalls, 2);
/// let current_ngrams = extract_ngrams(&current_syscalls, 2);
///
/// let anomalies = detect_sequence_anomalies(&baseline_ngrams, &current_ngrams, 0.30);
/// assert!(anomalies.len() > 0); // Should detect new networking sequence
/// ```
pub fn detect_sequence_anomalies(
    baseline_ngrams: &NGramMap,
    current_ngrams: &NGramMap,
    frequency_threshold: f64,
) -> Vec<SequenceAnomaly> {
    let mut anomalies = Vec::new();

    // Detect new sequences (present in current, absent in baseline)
    for (ngram, &count) in current_ngrams {
        if !baseline_ngrams.contains_key(ngram) {
            anomalies.push(SequenceAnomaly {
                ngram: ngram.clone(),
                baseline_freq: 0,
                current_freq: count,
                anomaly_type: AnomalyType::NewSequence,
                severity: assess_sequence_severity(ngram),
            });
        }
    }

    // Detect missing sequences (present in baseline, absent in current)
    for (ngram, &count) in baseline_ngrams {
        if !current_ngrams.contains_key(ngram) {
            anomalies.push(SequenceAnomaly {
                ngram: ngram.clone(),
                baseline_freq: count,
                current_freq: 0,
                anomaly_type: AnomalyType::MissingSequence,
                severity: Severity::Medium,
            });
        }
    }

    // Detect frequency changes (both present, but count differs significantly)
    for (ngram, &baseline_count) in baseline_ngrams {
        if let Some(&current_count) = current_ngrams.get(ngram) {
            let freq_change =
                (current_count as f64 - baseline_count as f64) / baseline_count as f64;

            if freq_change.abs() > frequency_threshold {
                anomalies.push(SequenceAnomaly {
                    ngram: ngram.clone(),
                    baseline_freq: baseline_count,
                    current_freq: current_count,
                    anomaly_type: AnomalyType::FrequencyChange,
                    severity: if freq_change.abs() > 0.5 {
                        Severity::High
                    } else {
                        Severity::Medium
                    },
                });
            }
        }
    }

    // Sort by severity (Critical → High → Medium → Low)
    anomalies.sort_by_key(|b| std::cmp::Reverse(b.severity));

    anomalies
}

/// Assess severity of a sequence based on its content
///
/// Critical: Networking sequences (socket, connect, send)
/// High: Synchronization sequences (futex) - unexpected in single-threaded code
/// Medium: Other new sequences
fn assess_sequence_severity(ngram: &NGram) -> Severity {
    // Critical: sequences involving networking (telemetry leaks, supply chain attacks)
    if ngram.iter().any(|s| {
        s.contains("socket") || s.contains("connect") || s.contains("send") || s.contains("recv")
    }) {
        return Severity::Critical;
    }

    // High: sequences involving synchronization (unexpected in single-threaded transpilers)
    if ngram.iter().any(|s| s == "futex" || s.contains("pthread_mutex")) {
        return Severity::High;
    }

    // Medium: other new sequences
    Severity::Medium
}

fn ngram_to_string(ngram: &NGram) -> String {
    format!("[{}]", ngram.join(" → "))
}

fn severity_emoji(severity: Severity) -> &'static str {
    match severity {
        Severity::Critical => "🔴 CRITICAL",
        Severity::High => "🟠 HIGH",
        Severity::Medium => "🟡 MEDIUM",
        Severity::Low => "🟢 LOW",
    }
}

// Compile-time thread-safety verification (Sprint 59)
static_assertions::assert_impl_all!(AnomalyType: Send, Sync);
static_assertions::assert_impl_all!(SequenceAnomaly: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::extract_ngrams;

    #[test]
    fn test_detect_new_sequence() {
        let baseline_syscalls = vec!["mmap".to_string(), "read".to_string(), "write".to_string()];
        let current_syscalls =
            vec!["socket".to_string(), "connect".to_string(), "send".to_string()];

        let baseline_ngrams = extract_ngrams(&baseline_syscalls, 3);
        let current_ngrams = extract_ngrams(&current_syscalls, 3);

        let anomalies = detect_sequence_anomalies(&baseline_ngrams, &current_ngrams, 0.30);

        assert_eq!(anomalies.len(), 2); // 1 new, 1 missing
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::NewSequence);
        assert_eq!(anomalies[0].severity, Severity::Critical); // Networking
    }

    #[test]
    fn test_detect_missing_sequence() {
        let baseline_syscalls = vec!["mmap".to_string(), "read".to_string(), "write".to_string()];
        let current_syscalls = vec!["mmap".to_string(), "write".to_string()]; // Skipped read

        let baseline_ngrams = extract_ngrams(&baseline_syscalls, 2);
        let current_ngrams = extract_ngrams(&current_syscalls, 2);

        let anomalies = detect_sequence_anomalies(&baseline_ngrams, &current_ngrams, 0.30);

        // Should detect missing ["read", "write"] sequence
        let missing = anomalies.iter().find(|a| a.anomaly_type == AnomalyType::MissingSequence);
        assert!(missing.is_some());
    }

    #[test]
    fn test_detect_frequency_change() {
        let mut baseline_ngrams = NGramMap::new();
        baseline_ngrams.insert(vec!["mmap".to_string(), "read".to_string()], 10);

        let mut current_ngrams = NGramMap::new();
        current_ngrams.insert(vec!["mmap".to_string(), "read".to_string()], 50);

        let anomalies = detect_sequence_anomalies(&baseline_ngrams, &current_ngrams, 0.30);

        // Should detect +400% frequency change (10 → 50)
        assert_eq!(anomalies.len(), 1);
        assert_eq!(anomalies[0].anomaly_type, AnomalyType::FrequencyChange);
        assert_eq!(anomalies[0].severity, Severity::High); // >50% change
    }

    #[test]
    fn test_severity_assessment_networking() {
        let ngram = vec!["socket".to_string(), "connect".to_string(), "send".to_string()];
        assert_eq!(assess_sequence_severity(&ngram), Severity::Critical);
    }

    #[test]
    fn test_severity_assessment_synchronization() {
        let ngram = vec!["futex".to_string(), "read".to_string()];
        assert_eq!(assess_sequence_severity(&ngram), Severity::High);
    }

    #[test]
    fn test_severity_assessment_normal() {
        let ngram = vec!["mmap".to_string(), "read".to_string()];
        assert_eq!(assess_sequence_severity(&ngram), Severity::Medium);
    }

    #[test]
    fn test_frequency_change_percent() {
        let anomaly = SequenceAnomaly {
            ngram: vec!["a".to_string()],
            baseline_freq: 100,
            current_freq: 150,
            anomaly_type: AnomalyType::FrequencyChange,
            severity: Severity::Medium,
        };

        assert_eq!(anomaly.frequency_change_percent(), 50.0);
    }

    #[test]
    fn test_to_report_string() {
        let anomaly = SequenceAnomaly {
            ngram: vec!["socket".to_string(), "connect".to_string()],
            baseline_freq: 0,
            current_freq: 3,
            anomaly_type: AnomalyType::NewSequence,
            severity: Severity::Critical,
        };

        let report = anomaly.to_report_string();
        assert!(report.contains("NEW SEQUENCE"));
        assert!(report.contains("socket → connect"));
        assert!(report.contains("CRITICAL"));
    }
}
