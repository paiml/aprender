//! PMAT-051: Predictive Scheduling Optimizer
//!
//! SLO-aware workload scheduling with cost optimization for multi-host deployments.
//! Uses PMAT-033 predictive models for SLO violation risk assessment.
//!
//! # Falsifiable Hypothesis (FKR-052)
//! "Predictive scheduling achieves >99% SLO compliance while minimizing cost"
//!
//! # Features
//! - Host performance/cost modeling
//! - SLO risk prediction using regression models
//! - Cost optimization (minimize cloud spend)
//! - Load balancing with weighted capacity
//! - Spot instance and preemption support

mod scheduler;
mod types;

pub use scheduler::PredictiveScheduler;
pub use types::{
    HostProfile, InstanceType, PredictiveSchedulerConfig, SchedulerMetrics, SchedulingDecision,
    WorkloadSpec,
};

#[cfg(test)]
mod tests;
