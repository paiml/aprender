
#[test]
fn test_dispatch_metrics_variance_stddev() {
    let m = DispatchMetrics::new();
    // Need at least 2 samples for variance
    assert_eq!(m.cpu_latency_variance_us(), 0.0);
    assert_eq!(m.cpu_latency_stddev_us(), 0.0);

    m.record_cpu_latency(Duration::from_micros(100));
    assert_eq!(m.cpu_latency_variance_us(), 0.0); // only 1 sample

    m.record_cpu_latency(Duration::from_micros(200));
    // mean=150, variance = (100^2+200^2)/2 - 150^2 = 25000 - 22500 = 2500
    let var = m.cpu_latency_variance_us();
    assert!((var - 2500.0).abs() < 1.0, "variance was {}", var);
    let stddev = m.cpu_latency_stddev_us();
    assert!((stddev - 50.0).abs() < 0.1, "stddev was {}", stddev);
}

#[test]
fn test_dispatch_metrics_gpu_variance_stddev() {
    let m = DispatchMetrics::new();
    assert_eq!(m.gpu_latency_variance_us(), 0.0);
    assert_eq!(m.gpu_latency_stddev_us(), 0.0);

    m.record_gpu_latency(Duration::from_micros(50));
    m.record_gpu_latency(Duration::from_micros(150));
    // mean=100, variance = (50^2+150^2)/2 - 100^2 = 12500 - 10000 = 2500
    let var = m.gpu_latency_variance_us();
    assert!((var - 2500.0).abs() < 1.0, "variance was {}", var);
}

#[test]
fn test_dispatch_metrics_histogram_buckets() {
    let m = DispatchMetrics::new();
    // Bucket boundaries: [100, 500, 1000, 5000]
    // <100, 100-500, 500-1000, 1000-5000, >=5000

    m.record_cpu_latency(Duration::from_micros(50)); // bucket 0
    m.record_cpu_latency(Duration::from_micros(200)); // bucket 1
    m.record_cpu_latency(Duration::from_micros(700)); // bucket 2
    m.record_cpu_latency(Duration::from_micros(2000)); // bucket 3
    m.record_cpu_latency(Duration::from_micros(8000)); // bucket 4

    let buckets = m.cpu_latency_buckets();
    assert_eq!(buckets, [1, 1, 1, 1, 1]);
}

#[test]
fn test_dispatch_metrics_gpu_histogram_buckets() {
    let m = DispatchMetrics::new();
    m.record_gpu_latency(Duration::from_micros(10));
    m.record_gpu_latency(Duration::from_micros(10));

    let buckets = m.gpu_latency_buckets();
    assert_eq!(buckets[0], 2);
}

#[test]
fn test_dispatch_metrics_percentiles_empty() {
    let m = DispatchMetrics::new();
    assert_eq!(m.cpu_latency_p50_us(), 0.0);
    assert_eq!(m.cpu_latency_p95_us(), 0.0);
    assert_eq!(m.cpu_latency_p99_us(), 0.0);
    assert_eq!(m.gpu_latency_p50_us(), 0.0);
    assert_eq!(m.gpu_latency_p95_us(), 0.0);
    assert_eq!(m.gpu_latency_p99_us(), 0.0);
}

#[test]
fn test_dispatch_metrics_percentiles() {
    let m = DispatchMetrics::new();
    // All samples in bucket 0 (<100us)
    for _ in 0..100 {
        m.record_cpu_latency(Duration::from_micros(50));
    }

    let p50 = m.cpu_latency_p50_us();
    let p95 = m.cpu_latency_p95_us();
    let p99 = m.cpu_latency_p99_us();

    // All in first bucket [0-100], estimates should be in that range
    assert!(p50 >= 0.0 && p50 <= 100.0);
    assert!(p95 >= 0.0 && p95 <= 100.0);
    assert!(p99 >= 0.0 && p99 <= 100.0);
}

#[test]
fn test_dispatch_metrics_bucket_boundaries() {
    let m = DispatchMetrics::new();
    let boundaries = m.bucket_boundaries_us();
    assert_eq!(boundaries.len(), 5);
    assert_eq!(boundaries[0], "0-100");
    assert_eq!(boundaries[4], "5000+");
}

#[test]
fn test_dispatch_metrics_cv_zero_mean() {
    let m = DispatchMetrics::new();
    assert_eq!(m.cpu_latency_cv(), 0.0);
    assert_eq!(m.gpu_latency_cv(), 0.0);
}

#[test]
fn test_dispatch_metrics_cv() {
    let m = DispatchMetrics::new();
    m.record_cpu_latency(Duration::from_micros(100));
    m.record_cpu_latency(Duration::from_micros(200));
    // mean=150, stddev=50, CV = (50/150)*100 = 33.33%
    let cv = m.cpu_latency_cv();
    assert!((cv - 33.33).abs() < 1.0, "CV was {}", cv);
}

#[test]
fn test_dispatch_metrics_speedup_zero_gpu() {
    let m = DispatchMetrics::new();
    assert_eq!(m.cpu_gpu_speedup(), 0.0);
}

#[test]
fn test_dispatch_metrics_speedup() {
    let m = DispatchMetrics::new();
    m.record_cpu_latency(Duration::from_micros(200));
    m.record_gpu_latency(Duration::from_micros(50));

    let speedup = m.cpu_gpu_speedup();
    // speedup = cpu_mean / gpu_mean = 200/50 = 4.0
    assert!((speedup - 4.0).abs() < 0.01, "speedup was {}", speedup);
}

#[test]
fn test_dispatch_metrics_start_time() {
    let m = DispatchMetrics::new();
    let start = m.start_time_ms();
    assert!(start > 0);
}

#[test]
fn test_dispatch_metrics_elapsed_seconds() {
    let m = DispatchMetrics::new();
    let elapsed = m.elapsed_seconds();
    // Should be very small (just created)
    assert!(elapsed >= 0.0);
    assert!(elapsed < 1.0);
}

#[test]
fn test_dispatch_metrics_throughput_zero_elapsed() {
    let m = DispatchMetrics::new();
    // With essentially 0 elapsed time, throughput should be 0 to avoid div by zero
    let throughput = m.throughput_rps();
    assert!(throughput >= 0.0);
}

#[test]
fn test_dispatch_metrics_reset() {
    let m = DispatchMetrics::new();
    m.record_cpu_dispatch();
    m.record_gpu_dispatch();
    m.record_cpu_latency(Duration::from_micros(100));
    m.record_gpu_latency(Duration::from_micros(50));

    m.reset();

    assert_eq!(m.cpu_dispatches(), 0);
    assert_eq!(m.gpu_dispatches(), 0);
    assert_eq!(m.cpu_latency_count(), 0);
    assert_eq!(m.gpu_latency_count(), 0);
    assert_eq!(m.cpu_latency_sum_us(), 0);
    assert_eq!(m.gpu_latency_sum_us(), 0);
    assert_eq!(m.cpu_latency_max_us(), 0);
    assert_eq!(m.gpu_latency_max_us(), 0);
    // min is reset to MAX
    assert_eq!(m.cpu_latency_min_us(), 0); // getter returns 0 when count=0
    assert_eq!(m.gpu_latency_min_us(), 0);
}

#[test]
fn test_dispatch_metrics_debug() {
    let m = DispatchMetrics::new();
    let debug = format!("{:?}", m);
    assert!(debug.contains("DispatchMetrics"));
}

#[test]
fn test_dispatch_metrics_bucket_boundaries_constant() {
    assert_eq!(DispatchMetrics::BUCKET_BOUNDARIES, [100, 500, 1000, 5000]);
}
