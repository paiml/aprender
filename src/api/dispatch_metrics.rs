
/// Dispatch metrics handler - returns CPU/GPU dispatch statistics (IMP-127, IMP-128)
/// Supports ?format=prometheus for Prometheus-compatible output
#[cfg(feature = "gpu")]
async fn dispatch_metrics_handler(
    State(state): State<AppState>,
    Query(query): Query<DispatchMetricsQuery>,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    if let Some(metrics) = state.dispatch_metrics() {
        let format = query.format.as_deref().unwrap_or("json");

        if format == "prometheus" {
            // IMP-128: Prometheus format
            // IMP-128: Basic dispatch counters
            // IMP-130: Add latency histograms
            let cpu_buckets = metrics.cpu_latency_buckets();
            let gpu_buckets = metrics.gpu_latency_buckets();

            // Convert to cumulative buckets for Prometheus histogram format
            // Bucket boundaries: 100µs, 500µs, 1000µs, 5000µs, +Inf
            let cpu_cumulative = [
                cpu_buckets[0],
                cpu_buckets[0] + cpu_buckets[1],
                cpu_buckets[0] + cpu_buckets[1] + cpu_buckets[2],
                cpu_buckets[0] + cpu_buckets[1] + cpu_buckets[2] + cpu_buckets[3],
                cpu_buckets[0] + cpu_buckets[1] + cpu_buckets[2] + cpu_buckets[3] + cpu_buckets[4],
            ];
            let gpu_cumulative = [
                gpu_buckets[0],
                gpu_buckets[0] + gpu_buckets[1],
                gpu_buckets[0] + gpu_buckets[1] + gpu_buckets[2],
                gpu_buckets[0] + gpu_buckets[1] + gpu_buckets[2] + gpu_buckets[3],
                gpu_buckets[0] + gpu_buckets[1] + gpu_buckets[2] + gpu_buckets[3] + gpu_buckets[4],
            ];

            let prometheus_output = format!(
                "# HELP realizar_dispatch_cpu_total Total CPU dispatch decisions\n\
                 # TYPE realizar_dispatch_cpu_total counter\n\
                 realizar_dispatch_cpu_total {}\n\
                 # HELP realizar_dispatch_gpu_total Total GPU dispatch decisions\n\
                 # TYPE realizar_dispatch_gpu_total counter\n\
                 realizar_dispatch_gpu_total {}\n\
                 # HELP realizar_dispatch_gpu_ratio Ratio of GPU dispatches (0.0 to 1.0)\n\
                 # TYPE realizar_dispatch_gpu_ratio gauge\n\
                 realizar_dispatch_gpu_ratio {:.6}\n\
                 # HELP realizar_dispatch_throughput_rps Requests per second since start or reset\n\
                 # TYPE realizar_dispatch_throughput_rps gauge\n\
                 realizar_dispatch_throughput_rps {:.6}\n\
                 # HELP realizar_dispatch_elapsed_seconds Seconds since start or last reset\n\
                 # TYPE realizar_dispatch_elapsed_seconds gauge\n\
                 realizar_dispatch_elapsed_seconds {:.6}\n\
                 # HELP realizar_dispatch_cpu_latency CPU dispatch latency in microseconds\n\
                 # TYPE realizar_dispatch_cpu_latency histogram\n\
                 realizar_dispatch_cpu_latency_bucket{{le=\"100\"}} {}\n\
                 realizar_dispatch_cpu_latency_bucket{{le=\"500\"}} {}\n\
                 realizar_dispatch_cpu_latency_bucket{{le=\"1000\"}} {}\n\
                 realizar_dispatch_cpu_latency_bucket{{le=\"5000\"}} {}\n\
                 realizar_dispatch_cpu_latency_bucket{{le=\"+Inf\"}} {}\n\
                 realizar_dispatch_cpu_latency_sum {}\n\
                 realizar_dispatch_cpu_latency_count {}\n\
                 # HELP realizar_dispatch_gpu_latency GPU dispatch latency in microseconds\n\
                 # TYPE realizar_dispatch_gpu_latency histogram\n\
                 realizar_dispatch_gpu_latency_bucket{{le=\"100\"}} {}\n\
                 realizar_dispatch_gpu_latency_bucket{{le=\"500\"}} {}\n\
                 realizar_dispatch_gpu_latency_bucket{{le=\"1000\"}} {}\n\
                 realizar_dispatch_gpu_latency_bucket{{le=\"5000\"}} {}\n\
                 realizar_dispatch_gpu_latency_bucket{{le=\"+Inf\"}} {}\n\
                 realizar_dispatch_gpu_latency_sum {}\n\
                 realizar_dispatch_gpu_latency_count {}\n",
                metrics.cpu_dispatches(),
                metrics.gpu_dispatches(),
                metrics.gpu_ratio(),
                // IMP-141: Throughput metrics
                metrics.throughput_rps(),
                metrics.elapsed_seconds(),
                // CPU latency histogram
                cpu_cumulative[0],
                cpu_cumulative[1],
                cpu_cumulative[2],
                cpu_cumulative[3],
                cpu_cumulative[4],
                metrics.cpu_latency_sum_us(),
                metrics.cpu_latency_count(),
                // GPU latency histogram
                gpu_cumulative[0],
                gpu_cumulative[1],
                gpu_cumulative[2],
                gpu_cumulative[3],
                gpu_cumulative[4],
                metrics.gpu_latency_sum_us(),
                metrics.gpu_latency_count(),
            );
            (
                StatusCode::OK,
                [("content-type", "text/plain; charset=utf-8")],
                prometheus_output,
            )
                .into_response()
        } else {
            // Default: JSON format
            Json(DispatchMetricsResponse {
                cpu_dispatches: metrics.cpu_dispatches(),
                gpu_dispatches: metrics.gpu_dispatches(),
                total_dispatches: metrics.total_dispatches(),
                gpu_ratio: metrics.gpu_ratio(),
                // IMP-131: Latency percentiles
                cpu_latency_p50_us: metrics.cpu_latency_p50_us(),
                cpu_latency_p95_us: metrics.cpu_latency_p95_us(),
                cpu_latency_p99_us: metrics.cpu_latency_p99_us(),
                gpu_latency_p50_us: metrics.gpu_latency_p50_us(),
                gpu_latency_p95_us: metrics.gpu_latency_p95_us(),
                gpu_latency_p99_us: metrics.gpu_latency_p99_us(),
                // IMP-133: Latency means
                cpu_latency_mean_us: metrics.cpu_latency_mean_us(),
                gpu_latency_mean_us: metrics.gpu_latency_mean_us(),
                // IMP-134: Latency min/max
                cpu_latency_min_us: metrics.cpu_latency_min_us(),
                cpu_latency_max_us: metrics.cpu_latency_max_us(),
                gpu_latency_min_us: metrics.gpu_latency_min_us(),
                gpu_latency_max_us: metrics.gpu_latency_max_us(),
                // IMP-135: Latency variance/stddev
                cpu_latency_variance_us: metrics.cpu_latency_variance_us(),
                cpu_latency_stddev_us: metrics.cpu_latency_stddev_us(),
                gpu_latency_variance_us: metrics.gpu_latency_variance_us(),
                gpu_latency_stddev_us: metrics.gpu_latency_stddev_us(),
                // IMP-136: Histogram bucket configuration
                bucket_boundaries_us: metrics.bucket_boundaries_us(),
                cpu_latency_bucket_counts: metrics.cpu_latency_buckets().to_vec(),
                gpu_latency_bucket_counts: metrics.gpu_latency_buckets().to_vec(),
                // IMP-140: Throughput metrics
                throughput_rps: metrics.throughput_rps(),
                elapsed_seconds: metrics.elapsed_seconds(),
            })
            .into_response()
        }
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "Dispatch metrics not available. No GPU model configured.".to_string(),
            }),
        )
            .into_response()
    }
}

/// Dispatch metrics handler stub for non-GPU builds (IMP-127)
#[cfg(not(feature = "gpu"))]
async fn dispatch_metrics_handler(
    State(_state): State<AppState>,
    Query(_query): Query<DispatchMetricsQuery>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorResponse {
            error: "Dispatch metrics not available. GPU feature not enabled.".to_string(),
        }),
    )
        .into_response()
}

// Test helpers module (compiled only in tests)
#[cfg(test)]
pub(crate) mod test_helpers;

// Tests split into parts for PMAT compliance (<2000 lines per file)
#[cfg(test)]
mod tests;
