//! Experiment Tracking Example
//!
//! Demonstrates tracking synthetic data generation experiments with
//! cost, energy, and CO2 metrics.
//!
//! Run with: cargo run --example experiment_tracking

use std::time::Duration;
use verificar::ml::{
    AppleChip, ComputeDevice, CostMetrics, CpuArchitecture, EnergyMetrics, GenerationExperiment,
    GpuVendor, TpuVersion,
};

fn main() {
    println!("=== Verificar Experiment Tracking Example ===\n");

    // Demonstrate different compute devices
    println!("--- Compute Devices ---");

    let devices: Vec<(&str, ComputeDevice)> = vec![
        (
            "x86 Server",
            ComputeDevice::Cpu {
                cores: 32,
                threads_per_core: 2,
                architecture: CpuArchitecture::X86_64,
            },
        ),
        (
            "Apple M3 Max",
            ComputeDevice::AppleSilicon {
                chip: AppleChip::M3Max,
                neural_engine_cores: 16,
                gpu_cores: 40,
                memory_gb: 64,
            },
        ),
        (
            "NVIDIA A100",
            ComputeDevice::Gpu {
                name: "A100".to_string(),
                memory_gb: 80.0,
                compute_capability: Some("8.0".to_string()),
                vendor: GpuVendor::Nvidia,
            },
        ),
        (
            "Google TPU v4",
            ComputeDevice::Tpu {
                version: TpuVersion::V4,
                cores: 4,
            },
        ),
        (
            "Jetson Edge",
            ComputeDevice::Edge {
                name: "Jetson Orin".to_string(),
                power_budget_watts: 30.0,
            },
        ),
    ];

    for (name, device) in &devices {
        println!(
            "  {:15}: {:.2} TFLOPS, {:.0}W",
            name,
            device.theoretical_flops() / 1e12,
            device.estimated_power_watts()
        );
    }

    // Energy metrics
    println!("\n--- Energy Metrics ---");

    let energy = EnergyMetrics::new(
        36000.0, // 36 kJ (10 Wh)
        100.0,   // 100W average
        150.0,   // 150W peak
        360.0,   // 6 minutes
    )
    .with_carbon_intensity(386.0) // US average g CO2/kWh
    .with_pue(1.2); // Efficient datacenter

    println!("  Total energy: {:.2} kJ", energy.total_joules / 1000.0);
    println!("  Average power: {:.0}W", energy.average_power_watts);
    println!("  CO2 emissions: {:.2}g", energy.co2_grams.unwrap_or(0.0));
    println!("  PUE factor: {:.1}", energy.pue);

    // Cost metrics
    println!("\n--- Cost Metrics ---");

    let cost = CostMetrics::new(0.50, 0.05, 0.02).with_samples(10000);

    println!("  Compute cost: ${:.2}", cost.compute_cost_usd);
    println!("  Storage cost: ${:.2}", cost.storage_cost_usd);
    println!("  Network cost: ${:.2}", cost.network_cost_usd);
    println!("  Total cost: ${:.2}", cost.total_cost_usd);
    println!("  Cost per sample: ${:.6}", cost.cost_per_sample());

    // Full experiment tracking
    println!("\n--- Generation Experiment ---");

    let device = ComputeDevice::Cpu {
        cores: 8,
        threads_per_core: 2,
        architecture: CpuArchitecture::X86_64,
    };

    let mut experiment = GenerationExperiment::new("depyler-corpus-v1", device)
        .with_hourly_rate(0.10) // $0.10/hour for on-prem
        .with_carbon_intensity(200.0); // Renewable-heavy grid

    // Simulate generating batches
    experiment.record_samples(5000, Duration::from_secs(30));
    experiment.record_samples(5000, Duration::from_secs(28));
    experiment.record_samples(5000, Duration::from_secs(32));

    let metrics = experiment.finalize();

    println!("  Experiment: {}", metrics.name);
    println!("  Samples: {}", metrics.samples_generated);
    println!("  Duration: {:.1}s", metrics.duration.as_secs_f64());
    println!(
        "  Throughput: {:.1} samples/sec",
        metrics.samples_per_second
    );
    println!("  Total cost: ${:.4}", metrics.cost.total_cost_usd);
    println!("  Cost per sample: ${:.8}", metrics.cost_per_sample());
    println!("  CO2 per sample: {:.6}g", metrics.co2_per_sample());

    // Compare different compute options
    println!("\n--- Compute Comparison ---");

    let scenarios = [
        ("On-prem CPU", 0.05, 386.0),
        ("Cloud GPU", 2.50, 386.0),
        ("Green Cloud", 3.00, 50.0),
    ];

    for (name, rate, carbon) in scenarios {
        let device = ComputeDevice::default_cpu();
        let mut exp = GenerationExperiment::new(name, device)
            .with_hourly_rate(rate)
            .with_carbon_intensity(carbon);

        exp.record_samples(100_000, Duration::from_secs(600));
        let m = exp.finalize();

        println!(
            "  {:15}: ${:.2} total, {:.1}g CO2",
            name,
            m.cost.total_cost_usd,
            m.energy.co2_grams.unwrap_or(0.0)
        );
    }

    println!("\n=== Example Complete ===");
}
