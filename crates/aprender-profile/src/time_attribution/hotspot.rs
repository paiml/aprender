// Hotspot identification for time attribution analysis
//
// Identifies clusters that consume >5% of total execution time
// and provides actionable explanations.

use crate::time_attribution::attribution::TimeAttribution;
use std::time::Duration;

/// A performance hotspot in the trace
#[derive(Debug, Clone)]
pub struct Hotspot {
    /// Cluster name
    pub cluster: String,

    /// Total time spent in this hotspot
    pub time: Duration,

    /// Percentage of total execution time
    pub percentage: f64,

    /// Human-readable explanation
    pub explanation: String,

    /// Whether this hotspot is expected for transpilers
    pub is_expected: bool,
}

impl Hotspot {
    /// Format as human-readable report
    pub fn to_report_string(&self) -> String {
        let expected_marker = if self.is_expected { "✓" } else { "⚠️" };
        format!(
            "{} {} ({:.1}%, {:?})\n   {}",
            expected_marker, self.cluster, self.percentage, self.time, self.explanation
        )
    }
}

/// Identify performance hotspots from time attributions
///
/// Filters for clusters consuming >5% of total time and provides
/// actionable explanations for each hotspot.
///
/// # Arguments
/// * `attributions` - Time attributions sorted by total time
///
/// # Returns
/// Vector of hotspots sorted by percentage (descending)
///
/// # Example
/// ```ignore
/// use renacer::time_attribution::{calculate_time_attribution, identify_hotspots};
/// use renacer::cluster::ClusterRegistry;
/// use renacer::unified_trace::SyscallSpan;
/// use std::borrow::Cow;
///
/// let registry = ClusterRegistry::default_transpiler_clusters().expect("test");
/// let spans = vec![
///     SyscallSpan {
///         span_id: 1,
///         parent_span_id: 0,
///         name: Cow::Borrowed("read"),
///         args: vec![],
///         return_value: 0,
///         timestamp_nanos: 0,
///         duration_nanos: 10000, // 10μs - dominates
///         errno: None,
///     },
/// ];
///
/// let attributions = calculate_time_attribution(&spans, &registry);
/// let hotspots = identify_hotspots(&attributions);
/// assert!(hotspots.len() > 0); // Should identify FileIO hotspot
/// ```
pub fn identify_hotspots(attributions: &[TimeAttribution]) -> Vec<Hotspot> {
    attributions
        .iter()
        .filter(|a| a.percentage > 5.0) // Only report >5% of total time
        .map(|a| {
            let explanation = explain_hotspot(&a.cluster, a.percentage);
            let is_expected = is_expected_for_transpiler(&a.cluster);

            Hotspot {
                cluster: a.cluster.clone(),
                time: a.total_time,
                percentage: a.percentage,
                explanation,
                is_expected,
            }
        })
        .collect()
}

/// Generate explanation for a hotspot
fn explain_hotspot(cluster: &str, percentage: f64) -> String {
    match cluster {
        "FileIO" => {
            if percentage > 50.0 {
                format!(
                    "File I/O dominates execution ({percentage:.1}%). Expected for transpilers reading source files."
                )
            } else {
                format!(
                    "File I/O consumes {percentage:.1}% of time. Typical for reading source + emitting output."
                )
            }
        }
        "MemoryAllocation" => {
            format!(
                "Memory allocation uses {percentage:.1}% of time. May indicate excessive allocations or large AST structures."
            )
        }
        "DynamicLinking" => {
            if percentage > 20.0 {
                format!(
                    "Dynamic linking overhead is high ({percentage:.1}%). Consider static linking or reducing dependency count."
                )
            } else {
                format!("Dynamic linking uses {percentage:.1}% of time. Normal startup cost.")
            }
        }
        "Networking" => {
            format!(
                "❌ UNEXPECTED: Networking detected ({percentage:.1}%). Transpilers should not perform network I/O (telemetry leak?)."
            )
        }
        "GPU" => {
            format!(
                "❌ UNEXPECTED: GPU operations detected ({percentage:.1}%). Single-shot transpilers should not use GPU."
            )
        }
        "Synchronization" => {
            if percentage > 5.0 {
                format!(
                    "⚠️ UNEXPECTED: Synchronization overhead ({percentage:.1}%). Transpilers should be single-threaded."
                )
            } else {
                format!(
                    "Synchronization detected ({percentage:.1}%). May indicate unintended threading."
                )
            }
        }
        "ProcessControl" => {
            format!(
                "Process control operations use {percentage:.1}% of time. May include fork/exec if invoking external tools."
            )
        }
        "Randomness" => {
            format!(
                "Random number generation uses {percentage:.1}% of time. Should be minimal for deterministic compilation."
            )
        }
        "Unclassified" => {
            format!(
                "Unclassified syscalls consume {percentage:.1}% of time. Review clusters.toml for missing patterns."
            )
        }
        _ => {
            format!("{cluster} cluster uses {percentage:.1}% of execution time.")
        }
    }
}

/// Check if a cluster is expected for single-shot transpilers
fn is_expected_for_transpiler(cluster: &str) -> bool {
    matches!(
        cluster,
        "FileIO" | "MemoryAllocation" | "DynamicLinking" | "ProcessControl" | "Randomness"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_attribution(cluster: &str, percentage: f64) -> TimeAttribution {
        TimeAttribution {
            cluster: cluster.to_string(),
            total_time: Duration::from_secs(1),
            call_count: 100,
            percentage,
            avg_per_call: Duration::from_millis(10),
        }
    }

    #[test]
    fn test_identify_hotspots_threshold() {
        let attributions = vec![
            make_attribution("FileIO", 60.0),           // Hotspot (>5%)
            make_attribution("MemoryAllocation", 30.0), // Hotspot (>5%)
            make_attribution("DynamicLinking", 3.0),    // Not a hotspot (<5%)
        ];

        let hotspots = identify_hotspots(&attributions);

        // Should only report clusters >5%
        assert_eq!(hotspots.len(), 2);
        assert!(hotspots.iter().any(|h| h.cluster == "FileIO"));
        assert!(hotspots.iter().any(|h| h.cluster == "MemoryAllocation"));
    }

    #[test]
    fn test_is_expected_for_transpiler() {
        assert!(is_expected_for_transpiler("FileIO"));
        assert!(is_expected_for_transpiler("MemoryAllocation"));
        assert!(!is_expected_for_transpiler("Networking"));
        assert!(!is_expected_for_transpiler("GPU"));
    }

    #[test]
    fn test_explain_hotspot_file_io() {
        let explanation = explain_hotspot("FileIO", 60.0);
        assert!(explanation.contains("dominates"));
        assert!(explanation.contains("Expected"));
    }

    #[test]
    fn test_explain_hotspot_networking() {
        let explanation = explain_hotspot("Networking", 10.0);
        assert!(explanation.contains("UNEXPECTED"));
        assert!(explanation.contains("should not"));
    }

    #[test]
    fn test_explain_hotspot_gpu() {
        let explanation = explain_hotspot("GPU", 15.0);
        assert!(explanation.contains("UNEXPECTED"));
        assert!(explanation.contains("should not"));
    }

    #[test]
    fn test_hotspot_report_string() {
        let hotspot = Hotspot {
            cluster: "FileIO".to_string(),
            time: Duration::from_secs(1),
            percentage: 60.0,
            explanation: "Test explanation".to_string(),
            is_expected: true,
        };

        let report = hotspot.to_report_string();
        assert!(report.contains("FileIO"));
        assert!(report.contains("60.0%"));
        assert!(report.contains("✓")); // Expected marker
    }

    #[test]
    fn test_hotspot_unexpected_marker() {
        let hotspot = Hotspot {
            cluster: "Networking".to_string(),
            time: Duration::from_millis(100),
            percentage: 10.0,
            explanation: "Unexpected networking".to_string(),
            is_expected: false,
        };

        let report = hotspot.to_report_string();
        assert!(report.contains("⚠️")); // Unexpected marker
    }

    #[test]
    fn test_explain_hotspot_memory_allocation() {
        let explanation = explain_hotspot("MemoryAllocation", 25.0);
        assert!(explanation.contains("Memory allocation"));
        assert!(explanation.contains("25.0%"));
    }

    #[test]
    fn test_explain_hotspot_dynamic_linking_high() {
        let explanation = explain_hotspot("DynamicLinking", 30.0);
        assert!(explanation.contains("high"));
        assert!(explanation.contains("static linking"));
    }

    #[test]
    fn test_explain_hotspot_dynamic_linking_low() {
        let explanation = explain_hotspot("DynamicLinking", 10.0);
        assert!(explanation.contains("Normal startup"));
    }

    #[test]
    fn test_explain_hotspot_synchronization_high() {
        let explanation = explain_hotspot("Synchronization", 15.0);
        assert!(explanation.contains("UNEXPECTED"));
        assert!(explanation.contains("single-threaded"));
    }

    #[test]
    fn test_explain_hotspot_synchronization_low() {
        let explanation = explain_hotspot("Synchronization", 3.0);
        assert!(explanation.contains("threading"));
        assert!(!explanation.contains("UNEXPECTED"));
    }

    #[test]
    fn test_explain_hotspot_process_control() {
        let explanation = explain_hotspot("ProcessControl", 8.0);
        assert!(explanation.contains("Process control"));
        assert!(explanation.contains("fork/exec"));
    }

    #[test]
    fn test_explain_hotspot_randomness() {
        let explanation = explain_hotspot("Randomness", 5.5);
        assert!(explanation.contains("Random number"));
        assert!(explanation.contains("deterministic"));
    }

    #[test]
    fn test_explain_hotspot_unclassified() {
        let explanation = explain_hotspot("Unclassified", 12.0);
        assert!(explanation.contains("Unclassified"));
        assert!(explanation.contains("clusters.toml"));
    }

    #[test]
    fn test_explain_hotspot_file_io_moderate() {
        let explanation = explain_hotspot("FileIO", 30.0);
        assert!(explanation.contains("30.0%"));
        assert!(explanation.contains("Typical"));
    }

    #[test]
    fn test_explain_hotspot_unknown_cluster() {
        let explanation = explain_hotspot("CustomCluster", 20.0);
        assert!(explanation.contains("CustomCluster"));
        assert!(explanation.contains("20.0%"));
    }

    #[test]
    fn test_is_expected_complete_coverage() {
        // Expected clusters
        assert!(is_expected_for_transpiler("FileIO"));
        assert!(is_expected_for_transpiler("MemoryAllocation"));
        assert!(is_expected_for_transpiler("DynamicLinking"));
        assert!(is_expected_for_transpiler("ProcessControl"));
        assert!(is_expected_for_transpiler("Randomness"));

        // Unexpected clusters
        assert!(!is_expected_for_transpiler("Networking"));
        assert!(!is_expected_for_transpiler("GPU"));
        assert!(!is_expected_for_transpiler("Synchronization"));
        assert!(!is_expected_for_transpiler("Unclassified"));
        assert!(!is_expected_for_transpiler("CustomCluster"));
    }

    #[test]
    fn test_identify_hotspots_empty() {
        let attributions: Vec<TimeAttribution> = vec![];
        let hotspots = identify_hotspots(&attributions);
        assert!(hotspots.is_empty());
    }

    #[test]
    fn test_identify_hotspots_all_below_threshold() {
        let attributions = vec![
            make_attribution("FileIO", 2.0),
            make_attribution("MemoryAllocation", 3.0),
            make_attribution("Other", 4.0),
        ];

        let hotspots = identify_hotspots(&attributions);
        assert!(hotspots.is_empty());
    }
}
