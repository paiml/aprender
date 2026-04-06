
impl BenchmarkMatrix {
    /// Generate summary statistics
    #[must_use]
    pub fn summary(&self) -> MatrixSummary {
        let total_entries = self.entries.len();
        let available_entries = self.entries.iter().filter(|e| e.available).count();

        let mut backend_summaries = Vec::new();
        for backend in ComputeBackendType::all() {
            let entries: Vec<_> = self.entries_for_backend(backend);
            let available: Vec<_> = entries.iter().filter(|e| e.available).collect();

            let fastest = available.iter().min_by(|a, b| {
                a.p50_latency_ms
                    .partial_cmp(&b.p50_latency_ms)
                    .expect("test")
            });
            let highest_tp = available.iter().max_by(|a, b| {
                a.throughput_tps
                    .partial_cmp(&b.throughput_tps)
                    .expect("test")
            });

            backend_summaries.push(BackendSummary {
                backend,
                available_runtimes: available.len(),
                fastest_runtime: fastest.map(|e| format!("{:?}", e.runtime).to_lowercase()),
                fastest_p50_ms: fastest.map_or(0.0, |e| e.p50_latency_ms),
                highest_throughput_runtime: highest_tp
                    .map(|e| format!("{:?}", e.runtime).to_lowercase()),
                highest_throughput_tps: highest_tp.map_or(0.0, |e| e.throughput_tps),
            });
        }

        let available = self.entries.iter().filter(|e| e.available);
        let overall_fastest = available
            .clone()
            .min_by(|a, b| {
                a.p50_latency_ms
                    .partial_cmp(&b.p50_latency_ms)
                    .expect("test")
            })
            .map(|e| {
                (
                    format!("{:?}", e.runtime).to_lowercase(),
                    e.backend.to_string(),
                )
            });
        let overall_highest_throughput = available
            .max_by(|a, b| {
                a.throughput_tps
                    .partial_cmp(&b.throughput_tps)
                    .expect("test")
            })
            .map(|e| {
                (
                    format!("{:?}", e.runtime).to_lowercase(),
                    e.backend.to_string(),
                )
            });

        MatrixSummary {
            total_entries,
            available_entries,
            backend_summaries,
            overall_fastest,
            overall_highest_throughput,
        }
    }
}
