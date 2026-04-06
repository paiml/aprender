//! Benchmark metric aggregation strategies.

use super::types::{AggregationStrategy, HostBenchmark};

/// Extract three metric vectors (throughput, latency_p50, latency_p99) from benchmarks.
fn extract_metric_triple(benchmarks: &[HostBenchmark]) -> [Vec<f64>; 3] {
    let extractors: [fn(&HostBenchmark) -> f64; 3] = [
        |b| b.throughput_ops,
        |b| b.latency_p50_us,
        |b| b.latency_p99_us,
    ];
    extractors.map(|f| benchmarks.iter().map(f).collect())
}

/// Compute aggregated metrics (throughput, latency_p50, latency_p99) from benchmark results.
pub(crate) fn compute_metrics(
    benchmarks: &[HostBenchmark],
    strategy: AggregationStrategy,
) -> (f64, f64, f64) {
    let [throughputs, latencies_p50, latencies_p99] = extract_metric_triple(benchmarks);

    let apply = |vals: &[f64], init: f64, op: fn(f64, f64) -> f64| -> f64 {
        vals.iter().copied().fold(init, op)
    };

    match strategy {
        AggregationStrategy::GeometricMean => {
            let log_sum: f64 = throughputs.iter().map(|v| v.ln()).sum();
            let throughput = (log_sum / throughputs.len() as f64).exp();
            let latency_p50 = latencies_p50.iter().sum::<f64>() / latencies_p50.len() as f64;
            let latency_p99 = apply(&latencies_p99, 0.0, f64::max);
            (throughput, latency_p50, latency_p99)
        }
        AggregationStrategy::Median => {
            let median = |v: &[f64]| {
                let mut s: Vec<f64> = v.to_vec();
                s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                s[s.len() / 2]
            };
            (
                median(&throughputs),
                median(&latencies_p50),
                median(&latencies_p99),
            )
        }
        AggregationStrategy::Minimum => (
            apply(&throughputs, f64::INFINITY, f64::min),
            apply(&latencies_p50, f64::INFINITY, f64::min),
            apply(&latencies_p99, f64::INFINITY, f64::min),
        ),
        AggregationStrategy::Maximum => (
            apply(&throughputs, 0.0, f64::max),
            apply(&latencies_p50, 0.0, f64::max),
            apply(&latencies_p99, 0.0, f64::max),
        ),
    }
}
