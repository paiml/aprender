// Time attribution calculation for syscall clusters
//
// Attributes wall-clock time to semantic clusters, not just counts.
// Key insight: One blocking read() may dominate 1000 fast mmap() calls.

use crate::cluster::ClusterRegistry;
use crate::unified_trace::SyscallSpan;
use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

/// Time attribution for a semantic cluster
#[derive(Debug, Clone)]
pub struct TimeAttribution {
    /// Cluster name (e.g., "`MemoryAllocation`", "`FileIO`")
    pub cluster: String,

    /// Total time spent in this cluster
    pub total_time: Duration,

    /// Number of calls in this cluster
    pub call_count: usize,

    /// Percentage of total execution time
    pub percentage: f64,

    /// Average time per call
    pub avg_per_call: Duration,
}

impl fmt::Display for TimeAttribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {:.2}% ({} calls, avg {:?}/call, total {:?})",
            self.cluster, self.percentage, self.call_count, self.avg_per_call, self.total_time
        )
    }
}

/// Calculate time attribution from syscall spans
///
/// Aggregates syscall durations by semantic cluster to identify
/// where wall-clock time is actually spent.
///
/// # Arguments
/// * `spans` - Syscall spans from trace
/// * `registry` - Cluster registry for classification
///
/// # Returns
/// Vector of `TimeAttribution` sorted by `total_time` (descending)
///
/// # Example
/// ```
/// use renacer::time_attribution::calculate_time_attribution;
/// use renacer::cluster::ClusterRegistry;
/// use renacer::unified_trace::SyscallSpan;
/// use std::borrow::Cow;
///
/// let registry = ClusterRegistry::default_transpiler_clusters().expect("test");
/// let spans = vec![
///     SyscallSpan {
///         span_id: 1,
///         parent_span_id: 0,
///         name: Cow::Borrowed("mmap"),
///         args: vec![],
///         return_value: 0,
///         timestamp_nanos: 0,
///         duration_nanos: 1000,
///         errno: None,
///     },
/// ];
///
/// let attributions = calculate_time_attribution(&spans, &registry);
/// assert!(attributions.len() > 0);
/// ```
pub fn calculate_time_attribution(
    spans: &[SyscallSpan],
    registry: &ClusterRegistry,
) -> Vec<TimeAttribution> {
    if spans.is_empty() {
        return Vec::new();
    }

    // Calculate total execution time
    let total_time_nanos: u64 = spans.iter().map(|s| s.duration_nanos).sum();

    if total_time_nanos == 0 {
        return Vec::new(); // Avoid division by zero
    }

    // Aggregate time and count by cluster
    let mut cluster_time: HashMap<String, u64> = HashMap::new();
    let mut cluster_count: HashMap<String, usize> = HashMap::new();

    for span in spans {
        // Extract args as strings for classification
        let args: Vec<String> = span.args.iter().map(|(_, v)| v.clone()).collect();

        // For time attribution, we don't track FDs, so use simple classification
        let cluster_name =
            registry.classify_simple(&span.name, &args).unwrap_or("Unclassified".to_string());

        *cluster_time.entry(cluster_name.clone()).or_default() += span.duration_nanos;
        *cluster_count.entry(cluster_name).or_default() += 1;
    }

    // Calculate attributions
    let mut attributions: Vec<TimeAttribution> = cluster_time
        .into_iter()
        .map(|(cluster, time_nanos)| {
            let count = cluster_count[&cluster];
            let total_time_cluster = Duration::from_nanos(time_nanos);
            let percentage = (time_nanos as f64 / total_time_nanos as f64) * 100.0;
            let avg_per_call = Duration::from_nanos(time_nanos / count as u64);

            TimeAttribution {
                cluster,
                total_time: total_time_cluster,
                call_count: count,
                percentage,
                avg_per_call,
            }
        })
        .collect();

    // Sort by time (descending)
    attributions.sort_by_key(|b| std::cmp::Reverse(b.total_time));

    attributions
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    fn make_span(name: &'static str, duration_nanos: u64) -> SyscallSpan {
        SyscallSpan {
            span_id: 1,
            parent_span_id: 0,
            name: Cow::Borrowed(name),
            args: vec![],
            return_value: 0,
            timestamp_nanos: 0,
            duration_nanos,
            errno: None,
        }
    }

    #[test]
    fn test_time_attribution_basic() {
        let registry = ClusterRegistry::default_transpiler_clusters().expect("test");
        let spans = vec![
            make_span("mmap", 1000),  // MemoryAllocation
            make_span("read", 9000),  // FileIO (90% of time)
            make_span("write", 1000), // FileIO
        ];

        let attributions = calculate_time_attribution(&spans, &registry);

        // Should have 2 clusters
        assert_eq!(attributions.len(), 2);

        // FileIO should dominate (10ms out of 11ms)
        let file_io = attributions.iter().find(|a| a.cluster == "FileIO").expect("test");
        assert!((file_io.percentage - 90.9).abs() < 0.1); // ~90.9%
        assert_eq!(file_io.call_count, 2);
    }

    #[test]
    fn test_time_attribution_sorted() {
        let registry = ClusterRegistry::default_transpiler_clusters().expect("test");
        let spans = vec![
            make_span("mmap", 1000),  // MemoryAllocation: 1μs
            make_span("read", 9000),  // FileIO: 9μs
            make_span("write", 5000), // FileIO: 5μs
        ];

        let attributions = calculate_time_attribution(&spans, &registry);

        // Should be sorted by total time (descending)
        assert!(attributions[0].total_time >= attributions[1].total_time);
    }

    #[test]
    fn test_time_attribution_empty() {
        let registry = ClusterRegistry::default_transpiler_clusters().expect("test");
        let spans: Vec<SyscallSpan> = vec![];

        let attributions = calculate_time_attribution(&spans, &registry);
        assert!(attributions.is_empty());
    }

    #[test]
    fn test_time_attribution_zero_duration() {
        let registry = ClusterRegistry::default_transpiler_clusters().expect("test");
        let spans = vec![make_span("mmap", 0), make_span("read", 0)];

        let attributions = calculate_time_attribution(&spans, &registry);
        assert!(attributions.is_empty()); // All zero duration
    }

    #[test]
    fn test_avg_per_call() {
        let registry = ClusterRegistry::default_transpiler_clusters().expect("test");
        let spans = vec![
            make_span("read", 1000),  // FileIO
            make_span("read", 3000),  // FileIO
            make_span("write", 2000), // FileIO
        ];

        let attributions = calculate_time_attribution(&spans, &registry);
        let file_io = attributions.iter().find(|a| a.cluster == "FileIO").expect("test");

        // Total: 6000ns, 3 calls = 2000ns avg
        assert_eq!(file_io.avg_per_call, Duration::from_nanos(2000));
    }
}
