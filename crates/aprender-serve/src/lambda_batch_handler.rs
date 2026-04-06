
#[cfg(test)]
mod tests {
    use super::*;
    use crate::apr::{HEADER_SIZE, MAGIC};

    #[test]
    #[ignore = "Mock APR format incompatible with real AprModel::from_bytes parser"]
    fn test_batch_handler_with_errors() {
        let model_bytes = create_sum_model(2);
        let handler = LambdaHandler::from_bytes(model_bytes).expect("test");

        let request = BatchLambdaRequest {
            instances: vec![
                LambdaRequest {
                    features: vec![1.0, 2.0],
                    model_id: None,
                },
                LambdaRequest {
                    features: vec![], // Empty features will fail
                    model_id: None,
                },
                LambdaRequest {
                    features: vec![5.0, 0.0],
                    model_id: None,
                },
            ],
            max_parallelism: None,
        };

        let response = handler.handle_batch(&request).expect("test");

        assert_eq!(response.predictions.len(), 3);
        assert_eq!(response.success_count, 2);
        assert_eq!(response.error_count, 1);

        // Check successful predictions
        assert!((response.predictions[0].prediction - 3.0).abs() < 0.001);
        assert!(response.predictions[1].prediction.is_nan()); // Error placeholder
        assert!((response.predictions[2].prediction - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_batch_handler_rejects_empty_batch() {
        let model_bytes = create_sum_model(2);
        let handler = LambdaHandler::from_bytes(model_bytes).expect("test");

        let request = BatchLambdaRequest {
            instances: vec![],
            max_parallelism: None,
        };

        let result = handler.handle_batch(&request);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), LambdaError::EmptyBatch);
    }

    // --------------------------------------------------------------------------
    // Test: Prometheus metrics (PROD-001)
    // Per spec §11.3
    // --------------------------------------------------------------------------

    #[test]
    fn test_metrics_new() {
        let metrics = LambdaMetrics::new();
        assert_eq!(metrics.requests_total, 0);
        assert_eq!(metrics.requests_success, 0);
        assert_eq!(metrics.requests_failed, 0);
        assert_eq!(metrics.cold_starts, 0);
        assert_eq!(metrics.batch_requests, 0);
    }

    #[test]
    fn test_metrics_record_success() {
        let mut metrics = LambdaMetrics::new();

        metrics.record_success(5.0, true);
        assert_eq!(metrics.requests_total, 1);
        assert_eq!(metrics.requests_success, 1);
        assert_eq!(metrics.cold_starts, 1);
        assert!((metrics.latency_total_ms - 5.0).abs() < 0.001);

        metrics.record_success(3.0, false);
        assert_eq!(metrics.requests_total, 2);
        assert_eq!(metrics.requests_success, 2);
        assert_eq!(metrics.cold_starts, 1); // No new cold start
        assert!((metrics.latency_total_ms - 8.0).abs() < 0.001);
    }

    #[test]
    fn test_metrics_record_failure() {
        let mut metrics = LambdaMetrics::new();

        metrics.record_failure();
        assert_eq!(metrics.requests_total, 1);
        assert_eq!(metrics.requests_failed, 1);
        assert_eq!(metrics.requests_success, 0);
    }

    #[test]
    fn test_metrics_record_batch() {
        let mut metrics = LambdaMetrics::new();

        metrics.record_batch(5, 2, 10.0);
        assert_eq!(metrics.batch_requests, 1);
        assert_eq!(metrics.requests_total, 7); // 5 + 2
        assert_eq!(metrics.requests_success, 5);
        assert_eq!(metrics.requests_failed, 2);
        assert!((metrics.latency_total_ms - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_metrics_avg_latency() {
        let mut metrics = LambdaMetrics::new();

        // Empty metrics returns 0
        assert!((metrics.avg_latency_ms() - 0.0).abs() < 0.001);

        metrics.record_success(4.0, false);
        metrics.record_success(6.0, false);

        // Average of 4.0 and 6.0 = 5.0
        assert!((metrics.avg_latency_ms() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_metrics_prometheus_format() {
        let mut metrics = LambdaMetrics::new();
        metrics.record_success(5.0, true);
        metrics.record_success(3.0, false);
        metrics.record_failure();
        metrics.record_batch(10, 2, 20.0);

        let prom = metrics.to_prometheus();

        // Verify Prometheus format
        assert!(prom.contains("# HELP lambda_requests_total"));
        assert!(prom.contains("# TYPE lambda_requests_total counter"));
        assert!(prom.contains("lambda_requests_total 15")); // 2 + 1 + 12
        assert!(prom.contains("lambda_requests_success 12")); // 2 + 10
        assert!(prom.contains("lambda_requests_failed 3")); // 1 + 2
        assert!(prom.contains("lambda_cold_starts 1"));
        assert!(prom.contains("lambda_batch_requests 1"));
        assert!(prom.contains("lambda_latency_avg_ms"));
    }

    #[test]
    fn test_metrics_default() {
        let metrics = LambdaMetrics::default();
        assert_eq!(metrics.requests_total, 0);
        assert_eq!(metrics.batch_requests, 0);
    }
include!("lambda_mock_model.rs");
}
