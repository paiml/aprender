#![allow(clippy::disallowed_methods)]
//! Predictive Scheduling Optimizer Demo
//!
//! Demonstrates SLO-aware workload scheduling with cost optimization.
//!
//! Run with: cargo run --example predictive_scheduler_demo -p cbtop

use cbtop::{
    HostProfile, InstanceType, PredictiveScheduler, PredictiveSchedulerConfig,
    SchedulerWorkloadSpec,
};
use std::time::Duration;

fn main() {
    println!("=== Predictive Scheduling Optimizer Demo ===\n");

    // Create scheduler with 99% SLO target
    let config = PredictiveSchedulerConfig {
        target_slo_compliance: 0.99,
        enable_spot_instances: true,
        ..Default::default()
    };
    let mut scheduler = PredictiveScheduler::new(config);

    // Register diverse host fleet
    println!("Setting up GPU cluster...");

    // On-demand H100 instances
    let mut h100 = HostProfile::new("h100-1", InstanceType::OnDemand);
    h100.compute_capacity = 2000.0;
    h100.hourly_cost = 3.0;
    h100.current_load = 0.3;
    scheduler.register_host(h100);

    // Spot A100 instances (cheaper but preemptible)
    let mut a100_spot = HostProfile::new("a100-spot-1", InstanceType::Spot);
    a100_spot.compute_capacity = 1500.0;
    a100_spot.hourly_cost = 2.0;
    a100_spot.current_load = 0.2;
    scheduler.register_host(a100_spot);

    // Reserved L40S instances
    let mut l40s = HostProfile::new("l40s-reserved", InstanceType::Reserved);
    l40s.compute_capacity = 1000.0;
    l40s.hourly_cost = 1.0;
    l40s.current_load = 0.4;
    scheduler.register_host(l40s);

    println!("Registered {} hosts\n", scheduler.hosts().count());

    // Schedule various workloads
    println!("=== Scheduling Workloads ===\n");

    let workloads = [
        ("critical-inference", 500, Duration::from_millis(100), 3),
        ("batch-processing", 2000, Duration::from_secs(5), 1),
        ("interactive-query", 100, Duration::from_millis(50), 2),
    ];

    for (name, ops, deadline, priority) in workloads {
        let mut workload = SchedulerWorkloadSpec::new(name, ops as u64);
        workload.slo_deadline = deadline;
        workload.priority = priority;

        if let Some(decision) = scheduler.schedule(&workload) {
            println!("Workload: {}", name);
            println!("  → Scheduled to: {}", decision.host_id);
            println!("  → Predicted time: {:?}", decision.predicted_time);
            println!("  → Predicted cost: ${:.4}", decision.predicted_cost);
            println!("  → SLO compliance: {:.1}%", decision.slo_compliance_prob * 100.0);
            println!("  → Reason: {}\n", decision.reason);

            // Record result (simulate execution)
            scheduler.record_result(
                &decision.host_id,
                decision.predicted_time,
                false, // No SLO violation
                decision.predicted_cost,
            );
        }
    }

    // Show metrics
    println!("=== Scheduler Metrics ===");
    let metrics = scheduler.metrics();
    println!("Total decisions: {}", metrics.total_decisions);
    println!("SLO compliance rate: {:.1}%", metrics.slo_compliance_rate() * 100.0);
    println!("Total cost: ${:.4}", metrics.total_cost);
    println!("Spot savings: ${:.4}", metrics.spot_savings);
    println!("Avg scheduling latency: {:.1}µs", metrics.avg_scheduling_latency_us);

    // Demonstrate load balancing suggestions
    println!("\n=== Load Balancing ===");
    scheduler.update_host_load("h100-1", 0.9); // Overload H100
    scheduler.update_host_load("a100-spot-1", 0.2); // Underutilized spot

    let rebalance = scheduler.suggest_rebalancing();
    if !rebalance.is_empty() {
        println!("Suggested migrations:");
        for (from, to) in &rebalance {
            println!("  Migrate from {} → {}", from, to);
        }
    } else {
        println!("No rebalancing needed");
    }

    // Show instance type characteristics
    println!("\n=== Instance Type Comparison ===");
    for itype in [
        InstanceType::OnDemand,
        InstanceType::Spot,
        InstanceType::Reserved,
        InstanceType::Preemptible,
    ] {
        println!(
            "{:?}: cost={:.0}%, reliability={:.1}%",
            itype,
            itype.cost_multiplier() * 100.0,
            itype.reliability() * 100.0
        );
    }

    println!("\n✅ Predictive scheduler demo complete!");
}
