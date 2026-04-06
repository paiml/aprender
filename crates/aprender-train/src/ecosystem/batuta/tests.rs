//! Tests for Batuta integration.

use super::*;

#[test]
fn test_gpu_pricing_creation() {
    let pricing = GpuPricing::new("a100-80gb", 3.00, 80)
        .with_spot(true)
        .with_provider("aws")
        .with_region("us-west-2");

    assert_eq!(pricing.gpu_type, "a100-80gb");
    assert!((pricing.hourly_rate - 3.00).abs() < f64::EPSILON);
    assert_eq!(pricing.memory_gb, 80);
    assert!(pricing.is_spot);
    assert_eq!(pricing.provider, "aws");
    assert_eq!(pricing.region, "us-west-2");
}

#[test]
fn test_queue_state_creation() {
    let queue = QueueState::new(5, 2, 8).with_avg_wait(300).with_eta(600);

    assert_eq!(queue.queue_depth, 5);
    assert_eq!(queue.available_gpus, 2);
    assert_eq!(queue.total_gpus, 8);
    assert_eq!(queue.avg_wait_seconds, 300);
    assert_eq!(queue.eta_seconds, Some(600));
}

#[test]
fn test_queue_state_is_available() {
    let available = QueueState::new(0, 2, 8);
    assert!(available.is_available());

    let unavailable = QueueState::new(5, 0, 8);
    assert!(!unavailable.is_available());
}

#[test]
fn test_queue_state_utilization() {
    let empty = QueueState::new(0, 8, 8);
    assert!((empty.utilization() - 0.0).abs() < f64::EPSILON);

    let half = QueueState::new(0, 4, 8);
    assert!((half.utilization() - 0.5).abs() < f64::EPSILON);

    let full = QueueState::new(10, 0, 8);
    assert!((full.utilization() - 1.0).abs() < f64::EPSILON);

    let zero_total = QueueState::new(0, 0, 0);
    assert!((zero_total.utilization() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_fallback_pricing_default() {
    let fallback = FallbackPricing::new();

    assert!(fallback.get_rate("a100-80gb").is_some());
    assert!(fallback.get_rate("v100").is_some());
    assert!(fallback.get_rate("t4").is_some());
    assert!(fallback.get_rate("unknown-gpu").is_none());
}

#[test]
fn test_fallback_pricing_case_insensitive() {
    let fallback = FallbackPricing::new();

    // Should match regardless of case and separators
    assert!(fallback.get_rate("A100-80GB").is_some());
    assert!(fallback.get_rate("a100_80gb").is_some());
    assert!(fallback.get_rate("V100").is_some());
}

#[test]
fn test_fallback_pricing_set_rate() {
    let mut fallback = FallbackPricing::new();

    // Update existing
    let old_rate = fallback.get_rate("t4").expect("operation should succeed").hourly_rate;
    fallback.set_rate(GpuPricing::new("t4", 0.35, 16));
    let new_rate = fallback.get_rate("t4").expect("operation should succeed").hourly_rate;
    assert!((old_rate - 0.50).abs() < f64::EPSILON);
    assert!((new_rate - 0.35).abs() < f64::EPSILON);

    // Add new
    fallback.set_rate(GpuPricing::new("rtx-4090", 0.80, 24));
    assert!(fallback.get_rate("rtx-4090").is_some());
}

#[test]
fn test_batuta_client_new() {
    let client = BatutaClient::new();
    assert!(!client.is_connected());

    let pricing = client.get_hourly_rate("a100-80gb").expect("operation should succeed");
    assert!((pricing.hourly_rate - 3.00).abs() < f64::EPSILON);
}

#[test]
fn test_batuta_client_with_url() {
    let client = BatutaClient::with_url("http://batuta.local:8080");
    assert!(client.is_connected());
}

#[test]
fn test_batuta_client_unknown_gpu() {
    let client = BatutaClient::new();
    let result = client.get_hourly_rate("nonexistent-gpu");
    assert!(matches!(result, Err(BatutaError::UnknownGpuType(_))));
}

#[test]
fn test_batuta_client_get_queue_depth() {
    let client = BatutaClient::new();
    let queue = client.get_queue_depth("a100-80gb").expect("operation should succeed");

    // Default optimistic queue state
    assert_eq!(queue.queue_depth, 0);
    assert!(queue.is_available());
}

#[test]
fn test_batuta_client_get_status() {
    let client = BatutaClient::new();
    let (pricing, queue) = client.get_status("v100").expect("operation should succeed");

    assert_eq!(pricing.gpu_type, "v100");
    assert!((pricing.hourly_rate - 2.00).abs() < f64::EPSILON);
    assert!(queue.is_available());
}

#[test]
fn test_batuta_client_estimate_cost() {
    let client = BatutaClient::new();
    let cost = client.estimate_cost("a100-80gb", 10.0).expect("operation should succeed");
    assert!((cost - 30.0).abs() < f64::EPSILON);
}

#[test]
fn test_batuta_client_cheapest_gpu() {
    let client = BatutaClient::new();

    // Should return T4 (cheapest with 16GB)
    let cheapest_16 = client.cheapest_gpu(16).expect("operation should succeed");
    assert_eq!(cheapest_16.gpu_type, "t4");

    // Should return something with 80GB
    let cheapest_80 = client.cheapest_gpu(80).expect("operation should succeed");
    assert!(cheapest_80.memory_gb >= 80);

    // Should return None for impossible requirements
    let impossible = client.cheapest_gpu(1000);
    assert!(impossible.is_none());
}

#[test]
fn test_adjust_eta_available() {
    let queue = QueueState::new(0, 4, 8);
    let adjusted = adjust_eta(3600, &queue);
    assert_eq!(adjusted.as_secs(), 3600); // No adjustment
}

#[test]
fn test_adjust_eta_queue_wait() {
    let queue = QueueState::new(3, 0, 8).with_avg_wait(300);
    let adjusted = adjust_eta(3600, &queue);

    // 3600 + (3 jobs * 300s avg wait) = 4500s
    assert!(adjusted.as_secs() >= 4500);
}

#[test]
fn test_adjust_eta_high_utilization() {
    let queue = QueueState::new(0, 1, 8); // 87.5% utilization
    let base = 3600;
    let adjusted = adjust_eta(base, &queue);

    // Should be increased due to high utilization
    assert!(adjusted.as_secs() > base);
}

#[test]
fn test_adjust_eta_with_queue_eta() {
    let queue = QueueState::new(0, 4, 8).with_eta(7200);
    let adjusted = adjust_eta(3600, &queue);

    // Should use queue ETA since it's higher
    assert!(adjusted.as_secs() >= 7200);
}
