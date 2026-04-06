//! Predictive scheduler implementation with SLO-aware workload placement.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::types::{
    HostProfile, InstanceType, PredictiveSchedulerConfig, SchedulerMetrics, SchedulingDecision,
    WorkloadSpec,
};

/// Predictive scheduling optimizer
pub struct PredictiveScheduler {
    config: PredictiveSchedulerConfig,
    hosts: HashMap<String, HostProfile>,
    metrics: SchedulerMetrics,
    /// Historical execution times per host
    execution_history: HashMap<String, Vec<Duration>>,
    /// SLO violation history per host
    violation_history: HashMap<String, Vec<bool>>,
}

impl PredictiveScheduler {
    /// Create a new predictive scheduler
    pub fn new(config: PredictiveSchedulerConfig) -> Self {
        Self {
            config,
            hosts: HashMap::new(),
            metrics: SchedulerMetrics::default(),
            execution_history: HashMap::new(),
            violation_history: HashMap::new(),
        }
    }

    /// Register a host with the scheduler
    pub fn register_host(&mut self, profile: HostProfile) {
        let host_id = profile.host_id.clone();
        self.hosts.insert(host_id.clone(), profile);
        self.execution_history.insert(host_id.clone(), Vec::new());
        self.violation_history.insert(host_id, Vec::new());
    }

    /// Remove a host from the scheduler
    pub fn deregister_host(&mut self, host_id: &str) {
        self.hosts.remove(host_id);
        self.execution_history.remove(host_id);
        self.violation_history.remove(host_id);
    }

    /// Update host load
    pub fn update_host_load(&mut self, host_id: &str, load: f64) {
        if let Some(host) = self.hosts.get_mut(host_id) {
            host.current_load = load.clamp(0.0, 1.0);
        }
    }

    /// Update host preemption deadline
    pub fn update_preemption_deadline(&mut self, host_id: &str, deadline: Option<Instant>) {
        if let Some(host) = self.hosts.get_mut(host_id) {
            host.preemption_deadline = deadline;
        }
    }

    /// Schedule a workload to optimal host
    pub fn schedule(&mut self, workload: &WorkloadSpec) -> Option<SchedulingDecision> {
        let start = Instant::now();

        // Filter eligible hosts
        let eligible_hosts: Vec<_> = self
            .hosts
            .values()
            .filter(|h| self.is_host_eligible(h, workload))
            .collect();

        if eligible_hosts.is_empty() {
            return None;
        }

        // Score each host
        let mut best_decision: Option<SchedulingDecision> = None;
        let mut best_score = f64::NEG_INFINITY;

        for host in eligible_hosts {
            let decision = self.evaluate_host(host, workload);
            if decision.score > best_score {
                best_score = decision.score;
                best_decision = Some(decision);
            }
        }

        // Update metrics
        if let Some(ref decision) = best_decision {
            self.metrics.total_decisions += 1;
            let scheduling_time = start.elapsed().as_micros() as f64;
            let n = self.metrics.total_decisions as f64;
            self.metrics.avg_scheduling_latency_us =
                self.metrics.avg_scheduling_latency_us * (n - 1.0) / n + scheduling_time / n;

            // Track spot savings
            if let Some(host) = self.hosts.get(&decision.host_id) {
                if host.instance_type == InstanceType::Spot {
                    let on_demand_cost =
                        decision.predicted_cost / host.instance_type.cost_multiplier();
                    self.metrics.spot_savings += on_demand_cost - decision.predicted_cost;
                }
            }
        }

        best_decision
    }

    /// Check if host is eligible for workload
    fn is_host_eligible(&self, host: &HostProfile, workload: &WorkloadSpec) -> bool {
        // Check capacity
        if host.current_load >= self.config.min_capacity_threshold {
            return false;
        }

        // Check memory
        if host.memory_capacity < workload.memory_required {
            return false;
        }

        // Check preemption safety
        if !host.is_safe_for_scheduling(self.config.preemption_buffer) {
            return false;
        }

        // Check spot instance policy
        if host.instance_type == InstanceType::Spot && !self.config.enable_spot_instances {
            return false;
        }

        true
    }

    /// Evaluate a host for workload placement
    fn evaluate_host(&self, host: &HostProfile, workload: &WorkloadSpec) -> SchedulingDecision {
        let predicted_time = self.predict_execution_time(host, workload);
        let slo_compliance_prob = self.predict_slo_compliance(host, workload, predicted_time);
        let predicted_cost = self.calculate_cost(host, workload, predicted_time);

        // Multi-objective scoring
        let score = self.calculate_score(host, slo_compliance_prob, predicted_cost, workload);

        let reason = self.generate_reason(host, slo_compliance_prob, predicted_cost);

        SchedulingDecision {
            host_id: host.host_id.clone(),
            predicted_time,
            predicted_cost,
            slo_compliance_prob,
            score,
            reason,
        }
    }

    /// Predict execution time using historical data
    fn predict_execution_time(&self, host: &HostProfile, workload: &WorkloadSpec) -> Duration {
        let base_estimate = workload.estimated_execution_time(host);

        // Adjust based on historical variance
        if let Some(history) = self.execution_history.get(&host.host_id) {
            if !history.is_empty() {
                // Use exponential smoothing on historical data
                let alpha = 0.3;
                let mut smoothed = history[0].as_secs_f64();
                for duration in history.iter().skip(1) {
                    smoothed = alpha * duration.as_secs_f64() + (1.0 - alpha) * smoothed;
                }

                // Blend historical with estimate
                let blended = 0.7 * base_estimate.as_secs_f64() + 0.3 * smoothed;
                return Duration::from_secs_f64(blended);
            }
        }

        // Add safety margin based on performance CV
        let margin = 1.0 + host.performance_cv;
        Duration::from_secs_f64(base_estimate.as_secs_f64() * margin)
    }

    /// Predict SLO compliance probability
    pub(super) fn predict_slo_compliance(
        &self,
        host: &HostProfile,
        workload: &WorkloadSpec,
        predicted_time: Duration,
    ) -> f64 {
        // Base compliance from time vs deadline
        let time_ratio = predicted_time.as_secs_f64() / workload.slo_deadline.as_secs_f64();

        // Sigmoid function for compliance probability
        // P(comply) = 1 / (1 + exp(k * (time_ratio - 1)))
        let k = 10.0; // Steepness
        let base_prob = 1.0 / (1.0 + (k * (time_ratio - 0.9)).exp());

        // Adjust for host reliability
        let reliability_factor = host.instance_type.reliability();

        // Adjust for historical compliance
        let historical_factor = if let Some(history) = self.violation_history.get(&host.host_id) {
            if history.len() >= 10 {
                let recent: Vec<_> = history.iter().rev().take(10).collect();
                let violations = recent.iter().filter(|&&v| *v).count();
                1.0 - (violations as f64 / 10.0)
            } else {
                host.historical_slo_compliance
            }
        } else {
            host.historical_slo_compliance
        };

        base_prob * reliability_factor * historical_factor
    }

    /// Calculate execution cost
    fn calculate_cost(
        &self,
        host: &HostProfile,
        _workload: &WorkloadSpec,
        predicted_time: Duration,
    ) -> f64 {
        let hours = predicted_time.as_secs_f64() / 3600.0;
        let base_cost = host.hourly_cost * host.instance_type.cost_multiplier() * hours;

        // Add network cost based on latency
        let network_cost = host.network_latency_ms * 0.0001; // Small factor for latency

        base_cost + network_cost
    }

    /// Calculate multi-objective score
    fn calculate_score(
        &self,
        host: &HostProfile,
        slo_compliance_prob: f64,
        predicted_cost: f64,
        workload: &WorkloadSpec,
    ) -> f64 {
        // Priority weighting
        let priority_weight = 1.0 + (workload.priority as f64 * 0.1);

        // SLO compliance score (heavily weighted)
        let slo_score = if slo_compliance_prob >= self.config.target_slo_compliance {
            slo_compliance_prob * 100.0
        } else {
            // Penalty for below-target compliance
            slo_compliance_prob * 100.0
                - self.config.slo_violation_penalty
                    * (self.config.target_slo_compliance - slo_compliance_prob)
                    * 100.0
        };

        // Cost score (inverse - lower is better)
        let max_cost = self.config.max_cost_per_op;
        let cost_score = if predicted_cost <= max_cost {
            (1.0 - predicted_cost / max_cost) * 50.0
        } else {
            -((predicted_cost / max_cost) - 1.0) * 50.0
        };

        // Load balancing score (prefer less loaded hosts)
        let load_score = (1.0 - host.current_load) * 20.0;

        // Combine scores
        (slo_score + cost_score + load_score) * priority_weight
    }

    /// Generate human-readable reason for selection
    fn generate_reason(
        &self,
        host: &HostProfile,
        slo_compliance_prob: f64,
        predicted_cost: f64,
    ) -> String {
        let mut reasons = Vec::new();

        if slo_compliance_prob >= 0.99 {
            reasons.push("excellent SLO compliance");
        } else if slo_compliance_prob >= 0.95 {
            reasons.push("good SLO compliance");
        }

        if host.instance_type == InstanceType::Spot {
            reasons.push("cost-effective spot instance");
        } else if host.instance_type == InstanceType::Reserved {
            reasons.push("reserved capacity");
        }

        if host.current_load < 0.3 {
            reasons.push("low current load");
        }

        if predicted_cost < self.config.max_cost_per_op * 0.5 {
            reasons.push("low cost");
        }

        if reasons.is_empty() {
            "best available option".to_string()
        } else {
            reasons.join(", ")
        }
    }

    /// Record execution result for learning
    pub fn record_result(
        &mut self,
        host_id: &str,
        actual_time: Duration,
        slo_violated: bool,
        actual_cost: f64,
    ) {
        // Update execution history
        if let Some(history) = self.execution_history.get_mut(host_id) {
            history.push(actual_time);
            if history.len() > self.config.history_window {
                history.remove(0);
            }
        }

        // Update violation history
        if let Some(history) = self.violation_history.get_mut(host_id) {
            history.push(slo_violated);
            if history.len() > self.config.history_window {
                history.remove(0);
            }
        }

        // Update metrics
        if slo_violated {
            self.metrics.slo_violations += 1;
        }
        self.metrics.total_cost += actual_cost;

        // Update host utilization
        if let Some(host) = self.hosts.get(host_id) {
            self.metrics
                .host_utilization
                .insert(host_id.to_string(), host.current_load);
        }

        // Update host historical compliance
        if let Some(host) = self.hosts.get_mut(host_id) {
            if let Some(history) = self.violation_history.get(host_id) {
                let recent_violations = history
                    .iter()
                    .rev()
                    .take(self.config.history_window)
                    .filter(|&&v| v)
                    .count();
                let total = history.len().min(self.config.history_window);
                if total > 0 {
                    host.historical_slo_compliance =
                        1.0 - (recent_violations as f64 / total as f64);
                }
            }
        }
    }

    /// Get current scheduler metrics
    pub fn metrics(&self) -> &SchedulerMetrics {
        &self.metrics
    }

    /// Get all registered hosts
    pub fn hosts(&self) -> impl Iterator<Item = &HostProfile> {
        self.hosts.values()
    }

    /// Get host by ID
    pub fn get_host(&self, host_id: &str) -> Option<&HostProfile> {
        self.hosts.get(host_id)
    }

    /// Rebalance workloads across hosts (returns migration suggestions)
    pub fn suggest_rebalancing(&self) -> Vec<(String, String)> {
        let mut migrations = Vec::new();

        // Find overloaded and underloaded hosts
        let mut overloaded: Vec<_> = self
            .hosts
            .values()
            .filter(|h| h.current_load > 0.8)
            .collect();
        let mut underloaded: Vec<_> = self
            .hosts
            .values()
            .filter(|h| {
                h.current_load < 0.3 && h.is_safe_for_scheduling(self.config.preemption_buffer)
            })
            .collect();

        overloaded.sort_by(|a, b| {
            b.current_load
                .partial_cmp(&a.current_load)
                .expect("values should be comparable")
        });
        underloaded.sort_by(|a, b| {
            a.current_load
                .partial_cmp(&b.current_load)
                .expect("values should be comparable")
        });

        // Suggest migrations from overloaded to underloaded
        for (over, under) in overloaded.iter().zip(underloaded.iter()) {
            migrations.push((over.host_id.clone(), under.host_id.clone()));
        }

        migrations
    }
}
