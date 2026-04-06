use super::*;

#[test]
fn test_cloud_provider_names() {
    assert_eq!(CloudProvider::Aws.name(), "AWS");
    assert_eq!(CloudProvider::Gcp.name(), "GCP");
    assert_eq!(CloudProvider::Azure.name(), "Azure");
}

#[test]
fn test_gpu_pricing() {
    let pricing = GpuPricing::new(CloudProvider::Aws, "A100-40GB", 4.10, 400.0);

    assert!((pricing.price_per_second() - 4.10 / 3600.0).abs() < 0.0001);
    assert_eq!(pricing.joules_per_second(), 400.0);
}

#[test]
fn test_energy_measurement() {
    let energy = EnergyMeasurement::from_power_duration(400.0, 3600.0);

    assert_eq!(energy.joules, 1_440_000.0);
    assert!((energy.kwh() - 0.4).abs() < 0.001);
}

#[test]
fn test_cost_calculation() {
    let mut tracker = CostTracker::new().with_gpu(CloudProvider::Aws, "A100-40GB");

    let result = tracker.calculate_cost(60.0, 10000);

    assert!(result.total_cost > 0.0);
    assert!(result.energy_joules > 0.0);
    assert!(result.carbon_g > 0.0);
    assert_eq!(result.token_count, 10000);
}

#[test]
fn test_cost_per_token() {
    let mut tracker = CostTracker::new();

    let result = tracker.calculate_cost(60.0, 1_000_000);

    assert!(result.cost_per_token > 0.0);
    assert!(result.cost_per_million_tokens > 0.0);
}

#[test]
fn test_budget_alert() {
    let mut tracker = CostTracker::new().with_budget(10.0);

    // Spend more than 80%
    for _ in 0..100 {
        tracker.calculate_cost(60.0, 1000);
    }

    let alert = tracker.check_budget();
    assert!(alert.is_some() || tracker.total_spend() < 8.0);
}

#[test]
fn test_cost_comparison() {
    let baseline = CostResult::new(1.0, 100000.0, 50.0, 60.0, 10000);
    let current = CostResult::new(1.1, 110000.0, 55.0, 60.0, 10000);

    let comparison = CostComparison::new(baseline, current);

    assert!((comparison.cost_change_percent - 10.0).abs() < 0.1);
    assert!(comparison.is_regression);
}

#[test]
fn test_export_csv() {
    let mut tracker = CostTracker::new();
    tracker.calculate_cost(60.0, 10000);

    let csv = tracker.export_csv();
    assert!(csv.contains("duration_sec"));
    assert!(csv.contains("60"));
}

#[test]
fn test_export_json() {
    let mut tracker = CostTracker::new();
    tracker.calculate_cost(60.0, 10000);

    let json = tracker.export_json();
    assert!(json.starts_with('['));
    assert!(json.contains("total_cost"));
}

#[test]
fn test_carbon_estimation() {
    let mut tracker = CostTracker::new().with_carbon_intensity(500.0); // High carbon grid

    let result = tracker.calculate_cost(3600.0, 100000);

    // 400W * 3600s = 1.44MJ = 0.4kWh
    // 0.4kWh * 500 gCO2/kWh = 200 gCO2
    assert!(result.carbon_g > 0.0);
}

#[test]
fn test_history() {
    let mut tracker = CostTracker::new();

    for _ in 0..5 {
        tracker.calculate_cost(60.0, 10000);
    }

    assert_eq!(tracker.history().len(), 5);
}
