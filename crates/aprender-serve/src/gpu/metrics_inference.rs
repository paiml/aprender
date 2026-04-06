
// ============================================================================
// Tests (T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // =========================================================================
    // InferenceMetrics Tests
    // =========================================================================

    #[test]
    fn test_inference_metrics_new() {
        let metrics = InferenceMetrics::new();
        assert_eq!(metrics.total_inferences(), 0);
        assert_eq!(metrics.total_tokens(), 0);
    }

    #[test]
    fn test_inference_metrics_default() {
        let metrics = InferenceMetrics::default();
        assert_eq!(metrics.total_inferences(), 0);
    }

    #[test]
    fn test_inference_metrics_record() {
        let mut metrics = InferenceMetrics::new();
        metrics.record_inference(Duration::from_millis(100), 50);
        assert_eq!(metrics.total_inferences(), 1);
        assert_eq!(metrics.total_tokens(), 50);

        metrics.record_inference(Duration::from_millis(200), 100);
        assert_eq!(metrics.total_inferences(), 2);
        assert_eq!(metrics.total_tokens(), 150);
    }

    #[test]
    fn test_inference_metrics_latency_percentile_empty() {
        let metrics = InferenceMetrics::new();
        assert!(metrics.latency_percentile(50).is_none());
    }

    #[test]
    fn test_inference_metrics_latency_percentile() {
        let mut metrics = InferenceMetrics::new();
        metrics.record_inference(Duration::from_millis(10), 1);
        metrics.record_inference(Duration::from_millis(20), 1);
        metrics.record_inference(Duration::from_millis(30), 1);
        metrics.record_inference(Duration::from_millis(40), 1);
        metrics.record_inference(Duration::from_millis(50), 1);

        let p50 = metrics.latency_percentile(50).expect("has data");
        assert!(p50 >= Duration::from_millis(20) && p50 <= Duration::from_millis(30));

        let p0 = metrics.latency_percentile(0).expect("has data");
        assert_eq!(p0, Duration::from_millis(10));

        let p100 = metrics.latency_percentile(100).expect("has data");
        assert_eq!(p100, Duration::from_millis(50));
    }

    #[test]
    fn test_inference_metrics_throughput_no_time() {
        let metrics = InferenceMetrics::new();
        // With no elapsed time, throughput is 0
        let throughput = metrics.throughput();
        assert!(throughput >= 0.0);
    }

    #[test]
    fn test_inference_metrics_reset() {
        let mut metrics = InferenceMetrics::new();
        metrics.record_inference(Duration::from_millis(100), 50);
        metrics.record_inference(Duration::from_millis(200), 100);

        metrics.reset();

        assert_eq!(metrics.total_inferences(), 0);
        assert_eq!(metrics.total_tokens(), 0);
    }

    // =========================================================================
    // HealthChecker Tests
    // =========================================================================

    #[test]
    fn test_health_checker_new() {
        let checker = HealthChecker::new();
        assert_eq!(checker.check_count(), 0);
        assert!(checker.is_healthy()); // No checks means healthy
    }

    #[test]
    fn test_health_checker_default() {
        let checker = HealthChecker::default();
        assert_eq!(checker.check_count(), 0);
    }

    #[test]
    fn test_health_checker_register() {
        let mut checker = HealthChecker::new();
        checker.register_check("test", Box::new(|| true));
        assert_eq!(checker.check_count(), 1);
    }

    #[test]
    fn test_health_checker_check_all() {
        let mut checker = HealthChecker::new();
        checker.register_check("always_healthy", Box::new(|| true));
        checker.register_check("always_unhealthy", Box::new(|| false));

        let results = checker.check_all();

        assert_eq!(results.len(), 2);
        assert_eq!(results.get("always_healthy"), Some(&true));
        assert_eq!(results.get("always_unhealthy"), Some(&false));
    }

    #[test]
    fn test_health_checker_is_healthy() {
        let mut checker = HealthChecker::new();

        // All healthy
        checker.register_check("check1", Box::new(|| true));
        checker.register_check("check2", Box::new(|| true));
        let _ = checker.check_all();
        assert!(checker.is_healthy());
    }

    #[test]
    fn test_health_checker_is_unhealthy() {
        let mut checker = HealthChecker::new();
        checker.register_check("check1", Box::new(|| true));
        checker.register_check("check2", Box::new(|| false)); // Unhealthy
        let _ = checker.check_all();
        assert!(!checker.is_healthy());
    }

    #[test]
    fn test_health_checker_clear() {
        let mut checker = HealthChecker::new();
        checker.register_check("test", Box::new(|| true));
        checker.check_all();

        checker.clear();

        assert_eq!(checker.check_count(), 0);
        assert!(checker.is_healthy());
    }

    // =========================================================================
    // ShutdownCoordinator Tests
    // =========================================================================

    #[test]
    fn test_shutdown_coordinator_new() {
        let coordinator = ShutdownCoordinator::new();
        assert!(!coordinator.is_shutting_down());
        assert_eq!(coordinator.pending_requests(), 0);
        assert_eq!(coordinator.handler_count(), 0);
    }

    #[test]
    fn test_shutdown_coordinator_default() {
        let coordinator = ShutdownCoordinator::default();
        assert!(!coordinator.is_shutting_down());
    }

    #[test]
    fn test_shutdown_coordinator_register_handler() {
        let mut coordinator = ShutdownCoordinator::new();
        coordinator.register_handler(Box::new(|| {}));
        assert_eq!(coordinator.handler_count(), 1);
    }

    #[test]
    fn test_shutdown_coordinator_request_tracking() {
        let mut coordinator = ShutdownCoordinator::new();

        coordinator.request_started();
        assert_eq!(coordinator.pending_requests(), 1);

        coordinator.request_started();
        assert_eq!(coordinator.pending_requests(), 2);

        coordinator.request_completed();
        assert_eq!(coordinator.pending_requests(), 1);

        coordinator.request_completed();
        assert_eq!(coordinator.pending_requests(), 0);
    }

    #[test]
    fn test_shutdown_coordinator_request_underflow() {
        let mut coordinator = ShutdownCoordinator::new();
        coordinator.request_completed(); // Should not underflow
        assert_eq!(coordinator.pending_requests(), 0);
    }

    #[test]
    fn test_shutdown_coordinator_initiate() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let mut coordinator = ShutdownCoordinator::new();
        let handler_called = Arc::new(AtomicBool::new(false));
        let handler_called_clone = Arc::clone(&handler_called);

        coordinator.register_handler(Box::new(move || {
            handler_called_clone.store(true, Ordering::SeqCst);
        }));

        coordinator.initiate_shutdown();

        assert!(coordinator.is_shutting_down());
        assert!(handler_called.load(Ordering::SeqCst));
    }

    #[test]
    fn test_shutdown_coordinator_initiate_idempotent() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let mut coordinator = ShutdownCoordinator::new();
        let call_count = Arc::new(AtomicU32::new(0));
        let call_count_clone = Arc::clone(&call_count);

        coordinator.register_handler(Box::new(move || {
            call_count_clone.fetch_add(1, Ordering::SeqCst);
        }));

        coordinator.initiate_shutdown();
        coordinator.initiate_shutdown(); // Should not call again

        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_shutdown_coordinator_is_complete() {
        let mut coordinator = ShutdownCoordinator::new();

        // Not complete until shutdown initiated
        assert!(!coordinator.is_complete());

        coordinator.request_started();
        coordinator.initiate_shutdown();

        // Not complete with pending requests
        assert!(!coordinator.is_complete());

        coordinator.request_completed();

        // Now complete
        assert!(coordinator.is_complete());
    }

    // =========================================================================
    // ComputeBackend Tests
    // =========================================================================

    #[test]
    fn test_compute_backend_default() {
        let backend: ComputeBackend = ComputeBackend::default();
        assert_eq!(backend, ComputeBackend::Auto);
    }

    #[test]
    fn test_compute_backend_equality() {
        assert_eq!(ComputeBackend::Gpu, ComputeBackend::Gpu);
        assert_eq!(ComputeBackend::Cpu, ComputeBackend::Cpu);
        assert_eq!(ComputeBackend::Auto, ComputeBackend::Auto);
        assert_ne!(ComputeBackend::Gpu, ComputeBackend::Cpu);
    }

    #[test]
    fn test_compute_backend_clone() {
        let backend = ComputeBackend::Gpu;
        let cloned = backend;
        assert_eq!(backend, cloned);
    }

    #[test]
    fn test_compute_backend_copy() {
        let backend = ComputeBackend::Cpu;
        let copied: ComputeBackend = backend;
        assert_eq!(backend, copied);
    }

    #[test]
    fn test_compute_backend_debug() {
        let debug_str = format!("{:?}", ComputeBackend::Auto);
        assert!(debug_str.contains("Auto"));
    }

    // =========================================================================
    // GpuCompute Tests
    // =========================================================================

    #[test]
    fn test_gpu_compute_cpu_backend() {
        let compute = GpuCompute::new(ComputeBackend::Cpu).expect("CPU should always work");
        assert!(!compute.is_gpu());
        assert_eq!(compute.backend(), ComputeBackend::Cpu);
    }

    #[test]
    fn test_gpu_compute_auto() {
        let compute = GpuCompute::auto().expect("auto should always work");
        // Backend depends on hardware, but should not error
        let _ = compute.backend();
    }
}
