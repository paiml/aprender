// N-gram Sequence Analysis for Syscall Grammar Detection
// (Section 6.1.1 of single-shot-compile-tooling-spec.md)
//
// This module implements sequence mining to detect syscall grammar violations.
// Instead of just counting syscalls, this detects when execution order changes
// (e.g., A→B→C becomes A→C→B).
//
// Scientific Foundation:
// [2] Forrest, S., Hofmeyr, S. A., Somayaji, A., & Longstaff, T. A. (1996).
//     A sense of self for unix processes. IEEE Symposium on Security and Privacy.
//
// Key Insight: Processes have a "grammar" of syscalls. Anomalies are often
// SEQUENCES disrupted, not just counts changed.

mod anomaly;
mod ngram;

pub use anomaly::{detect_sequence_anomalies, AnomalyType, SequenceAnomaly};
pub use ngram::{extract_ngrams, ngram_coverage, top_ngrams, NGram, NGramMap};

#[cfg(test)]
mod tests;
