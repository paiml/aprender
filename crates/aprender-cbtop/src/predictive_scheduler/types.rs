//! Core types for predictive scheduling: config, host profiles, workloads, and metrics.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for predictive scheduler
#[derive(Debug, Clone)]
pub struct PredictiveSchedulerConfig {
    /// Target SLO compliance rate (0.0-1.0)
    pub target_slo_compliance: f64,
    /// Maximum cost per operation (normalized units)
    pub max_cost_per_op: f64,
    /// Enable spot instance scheduling
    pub enable_spot_instances: bool,
    /// Preemption buffer time (avoid scheduling near preemption)
    pub preemption_buffer: Duration,
    /// Load balancing weight decay factor
    pub load_decay_factor: f64,
    /// Minimum host capacity utilization before overflow
    pub min_capacity_threshold: f64,
    /// SLO violation penalty multiplier for cost function
    pub slo_violation_penalty: f64,
    /// History window for performance tracking
    pub history_window: usize,
}

impl Default for PredictiveSchedulerConfig {
    fn default() -> Self {
        Self {
            target_slo_compliance: 0.99,
            max_cost_per_op: 1.0,
            enable_spot_instances: true,
            preemption_buffer: Duration::from_secs(300), // 5 minutes
            load_decay_factor: 0.9,
            min_capacity_threshold: 0.8,
            slo_violation_penalty: 10.0,
            history_window: 100,
        }
    }
}

/// Host instance type for cost modeling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InstanceType {
    /// On-demand instances (guaranteed availability)
    OnDemand,
    /// Spot instances (can be preempted)
    Spot,
    /// Reserved instances (committed capacity)
    Reserved,
    /// Preemptible instances (scheduled termination)
    Preemptible,
}

impl InstanceType {
    /// Cost multiplier relative to on-demand
    pub fn cost_multiplier(&self) -> f64 {
        match self {
            Self::OnDemand => 1.0,
            Self::Spot => 0.3,        // 70% discount
            Self::Reserved => 0.6,    // 40% discount
            Self::Preemptible => 0.2, // 80% discount
        }
    }

    /// Reliability score (probability of availability)
    pub fn reliability(&self) -> f64 {
        match self {
            Self::OnDemand => 0.9999,
            Self::Reserved => 0.9999,
            Self::Spot => 0.85,
            Self::Preemptible => 0.90,
        }
    }
}

/// Performance profile for a host
#[derive(Debug, Clone)]
pub struct HostProfile {
    /// Unique host identifier
    pub host_id: String,
    /// Instance type for cost modeling
    pub instance_type: InstanceType,
    /// Compute capacity (operations per second)
    pub compute_capacity: f64,
    /// Memory capacity (bytes)
    pub memory_capacity: u64,
    /// Current load (0.0-1.0)
    pub current_load: f64,
    /// Base cost per hour (normalized units)
    pub hourly_cost: f64,
    /// Network latency to coordinator (ms)
    pub network_latency_ms: f64,
    /// Historical SLO compliance rate
    pub historical_slo_compliance: f64,
    /// Preemption deadline (None = not preemptible)
    pub preemption_deadline: Option<Instant>,
    /// Performance variance (CV of operation times)
    pub performance_cv: f64,
}

impl HostProfile {
    /// Create a new host profile
    pub fn new(host_id: impl Into<String>, instance_type: InstanceType) -> Self {
        Self {
            host_id: host_id.into(),
            instance_type,
            compute_capacity: 1000.0,
            memory_capacity: 8 * 1024 * 1024 * 1024, // 8GB
            current_load: 0.0,
            hourly_cost: 1.0,
            network_latency_ms: 1.0,
            historical_slo_compliance: 0.99,
            preemption_deadline: None,
            performance_cv: 0.1,
        }
    }

    /// Effective cost per operation
    pub fn cost_per_op(&self) -> f64 {
        let base_cost = self.hourly_cost * self.instance_type.cost_multiplier();
        // Cost per op = hourly cost / ops per hour, adjusted for current load
        let effective_capacity = self.compute_capacity * (1.0 - self.current_load);
        if effective_capacity > 0.0 {
            base_cost / (effective_capacity * 3600.0)
        } else {
            f64::MAX
        }
    }

    /// Available capacity (0.0-1.0)
    pub fn available_capacity(&self) -> f64 {
        (1.0 - self.current_load).max(0.0)
    }

    /// Time until preemption (if applicable)
    pub fn time_until_preemption(&self) -> Option<Duration> {
        self.preemption_deadline.map(|deadline| {
            let now = Instant::now();
            if deadline > now {
                deadline - now
            } else {
                Duration::ZERO
            }
        })
    }

    /// Check if host is safe for scheduling (not near preemption)
    pub fn is_safe_for_scheduling(&self, buffer: Duration) -> bool {
        match self.time_until_preemption() {
            Some(time_left) => time_left > buffer,
            None => true, // Non-preemptible hosts are always safe
        }
    }
}

/// Workload characteristics for scheduling decisions
#[derive(Debug, Clone)]
pub struct WorkloadSpec {
    /// Unique workload identifier
    pub workload_id: String,
    /// Estimated operation count
    pub operation_count: u64,
    /// Memory requirement (bytes)
    pub memory_required: u64,
    /// SLO deadline
    pub slo_deadline: Duration,
    /// Priority (higher = more important)
    pub priority: u32,
    /// Whether workload can be preempted
    pub preemptible: bool,
    /// Estimated compute intensity (ops per byte)
    pub compute_intensity: f64,
}

impl WorkloadSpec {
    /// Create a new workload specification
    pub fn new(workload_id: impl Into<String>, operation_count: u64) -> Self {
        Self {
            workload_id: workload_id.into(),
            operation_count,
            memory_required: 1024 * 1024, // 1MB default
            slo_deadline: Duration::from_millis(100),
            priority: 1,
            preemptible: true,
            compute_intensity: 100.0,
        }
    }

    /// Estimated execution time on a host
    pub fn estimated_execution_time(&self, host: &HostProfile) -> Duration {
        let ops_per_sec = host.compute_capacity * host.available_capacity();
        if ops_per_sec > 0.0 {
            let seconds = self.operation_count as f64 / ops_per_sec;
            Duration::from_secs_f64(seconds)
        } else {
            Duration::MAX
        }
    }
}

/// Scheduling decision for a workload
#[derive(Debug, Clone)]
pub struct SchedulingDecision {
    /// Target host for execution
    pub host_id: String,
    /// Predicted execution time
    pub predicted_time: Duration,
    /// Predicted cost
    pub predicted_cost: f64,
    /// SLO compliance probability
    pub slo_compliance_prob: f64,
    /// Scheduling score (higher = better)
    pub score: f64,
    /// Reason for selection
    pub reason: String,
}

/// Scheduling metrics for monitoring
#[derive(Debug, Clone, Default)]
pub struct SchedulerMetrics {
    /// Total scheduling decisions made
    pub total_decisions: u64,
    /// Decisions resulting in SLO violations
    pub slo_violations: u64,
    /// Total cost incurred
    pub total_cost: f64,
    /// Average scheduling latency
    pub avg_scheduling_latency_us: f64,
    /// Host utilization map
    pub host_utilization: HashMap<String, f64>,
    /// Spot instance savings
    pub spot_savings: f64,
}

impl SchedulerMetrics {
    /// Current SLO compliance rate
    pub fn slo_compliance_rate(&self) -> f64 {
        if self.total_decisions > 0 {
            1.0 - (self.slo_violations as f64 / self.total_decisions as f64)
        } else {
            1.0
        }
    }

    /// Average cost per decision
    pub fn avg_cost_per_decision(&self) -> f64 {
        if self.total_decisions > 0 {
            self.total_cost / self.total_decisions as f64
        } else {
            0.0
        }
    }
}
