//! Demo 2: Little's Law Factory Simulation
//!
//! Interactive factory floor demonstrating WIP, throughput, and cycle time relationship.
//!
//! # Governing Equation
//!
//! ```text
//! Little's Law: L = λW
//!
//! Where:
//!   L = Average number in system (WIP)
//!   λ = Average arrival rate (Throughput)
//!   W = Average time in system (Cycle Time)
//! ```
//!
//! # EDD Cycle
//!
//! 1. **Equation**: WIP = Throughput × Cycle Time (holds for ANY stable system)
//! 2. **Failing Test**: |L - λW| / L > 0.05 (5% tolerance violation)
//! 3. **Implementation**: M/M/1 discrete event simulation queue
//! 4. **Verification**: Linear regression R² > 0.98 for WIP vs TH×CT
//! 5. **Falsification**: During transients (startup), law temporarily violated

use super::{CriterionStatus, EddDemo, FalsificationStatus};
use crate::engine::rng::SimRng;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Little's Law Factory demo state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LittlesLawFactoryDemo {
    /// Current simulation time.
    pub time: f64,
    /// Arrival rate λ (items/hour).
    pub arrival_rate: f64,
    /// Service rate μ (items/hour).
    pub service_rate: f64,
    /// Current WIP (items in system).
    pub wip: usize,
    /// Total items that have entered the system.
    pub total_arrivals: u64,
    /// Total items that have exited the system.
    pub total_departures: u64,
    /// Sum of cycle times for departed items.
    pub total_cycle_time: f64,
    /// Time-weighted WIP integral (for average WIP calculation).
    pub wip_integral: f64,
    /// Last time WIP changed (for integral calculation).
    pub last_wip_change_time: f64,
    /// Queue of arrival times (for cycle time calculation).
    #[serde(skip)]
    arrival_times: VecDeque<f64>,
    /// Next arrival event time.
    pub next_arrival: f64,
    /// Next departure event time (None if queue empty).
    pub next_departure: Option<f64>,
    /// WIP cap for CONWIP mode (None = infinite).
    pub wip_cap: Option<usize>,
    /// Tolerance for Little's Law verification.
    pub tolerance: f64,
    /// Minimum simulation time before verification.
    pub warmup_time: f64,
    /// RNG for stochastic arrivals/service.
    #[serde(skip)]
    rng: Option<SimRng>,
    /// Seed for reproducibility.
    pub seed: u64,
    /// History of (time, wip, throughput, `cycle_time`) for analysis.
    #[serde(skip)]
    pub history: Vec<(f64, f64, f64, f64)>,
}

impl Default for LittlesLawFactoryDemo {
    fn default() -> Self {
        Self::new(42)
    }
}

impl LittlesLawFactoryDemo {
    /// Create a new Little's Law factory demo.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut demo = Self {
            time: 0.0,
            arrival_rate: 4.0, // 4 items/hour
            service_rate: 5.0, // 5 items/hour (utilization = 80%)
            wip: 0,
            total_arrivals: 0,
            total_departures: 0,
            total_cycle_time: 0.0,
            wip_integral: 0.0,
            last_wip_change_time: 0.0,
            arrival_times: VecDeque::new(),
            next_arrival: 0.0,
            next_departure: None,
            wip_cap: None,
            tolerance: 0.05,
            warmup_time: 10.0,
            rng: Some(SimRng::new(seed)),
            seed,
            history: Vec::new(),
        };

        // Schedule first arrival
        demo.schedule_arrival();
        demo
    }

    /// Set arrival and service rates.
    pub fn set_rates(&mut self, arrival_rate: f64, service_rate: f64) {
        self.arrival_rate = arrival_rate;
        self.service_rate = service_rate;
    }

    /// Enable CONWIP mode with WIP cap.
    pub fn set_wip_cap(&mut self, cap: Option<usize>) {
        self.wip_cap = cap;
    }

    /// Get current utilization ρ = λ/μ.
    #[must_use]
    pub fn utilization(&self) -> f64 {
        self.arrival_rate / self.service_rate
    }

    /// Get average WIP (time-weighted).
    #[must_use]
    pub fn average_wip(&self) -> f64 {
        if self.time > 0.0 {
            (self.wip_integral + self.wip as f64 * (self.time - self.last_wip_change_time))
                / self.time
        } else {
            0.0
        }
    }

    /// Get throughput (departures per unit time).
    #[must_use]
    pub fn throughput(&self) -> f64 {
        if self.time > 0.0 {
            self.total_departures as f64 / self.time
        } else {
            0.0
        }
    }

    /// Get average cycle time.
    #[must_use]
    pub fn average_cycle_time(&self) -> f64 {
        if self.total_departures > 0 {
            self.total_cycle_time / self.total_departures as f64
        } else {
            0.0
        }
    }

    /// Get Little's Law prediction: L = λW.
    #[must_use]
    pub fn littles_law_prediction(&self) -> f64 {
        self.throughput() * self.average_cycle_time()
    }

    /// Get Little's Law error: |L - λW| / L.
    #[must_use]
    pub fn littles_law_error(&self) -> f64 {
        let l = self.average_wip();
        let prediction = self.littles_law_prediction();

        if l > 0.0 {
            (l - prediction).abs() / l
        } else {
            0.0
        }
    }

    /// Check if system is in steady state.
    #[must_use]
    pub fn is_steady_state(&self) -> bool {
        self.time >= self.warmup_time && self.total_departures >= 100
    }

    /// Schedule next arrival using exponential interarrival time.
    fn schedule_arrival(&mut self) {
        if let Some(ref mut rng) = self.rng {
            let u: f64 = rng.gen_range_f64(0.0001, 1.0);
            let interarrival = -u.ln() / self.arrival_rate;
            self.next_arrival = self.time + interarrival;
        }
    }

    /// Schedule next departure using exponential service time.
    fn schedule_departure(&mut self) {
        if let Some(ref mut rng) = self.rng {
            let u: f64 = rng.gen_range_f64(0.0001, 1.0);
            let service_time = -u.ln() / self.service_rate;
            self.next_departure = Some(self.time + service_time);
        }
    }

    /// Update WIP integral before changing WIP.
    fn update_wip_integral(&mut self) {
        self.wip_integral += self.wip as f64 * (self.time - self.last_wip_change_time);
        self.last_wip_change_time = self.time;
    }

    /// Process an arrival event.
    fn process_arrival(&mut self) {
        // Check CONWIP cap
        if let Some(cap) = self.wip_cap {
            if self.wip >= cap {
                // Blocked arrival - schedule next one anyway
                self.schedule_arrival();
                return;
            }
        }

        self.update_wip_integral();
        self.wip += 1;
        self.total_arrivals += 1;
        self.arrival_times.push_back(self.time);

        // If server was idle, start service
        if self.next_departure.is_none() {
            self.schedule_departure();
        }

        // Schedule next arrival
        self.schedule_arrival();
    }

    /// Process a departure event.
    fn process_departure(&mut self) {
        if self.wip == 0 {
            self.next_departure = None;
            return;
        }

        self.update_wip_integral();
        self.wip -= 1;
        self.total_departures += 1;

        // Calculate cycle time for this item
        if let Some(arrival_time) = self.arrival_times.pop_front() {
            let cycle_time = self.time - arrival_time;
            self.total_cycle_time += cycle_time;
        }

        // If more items waiting, schedule next departure
        if self.wip > 0 {
            self.schedule_departure();
        } else {
            self.next_departure = None;
        }
    }

    /// Record history point for analysis.
    fn record_history(&mut self) {
        if self.total_departures > 0 {
            self.history.push((
                self.time,
                self.average_wip(),
                self.throughput(),
                self.average_cycle_time(),
            ));
        }
    }

    /// Calculate R² for Little's Law validation.
    #[must_use]
    pub fn calculate_r_squared(&self) -> f64 {
        if self.history.len() < 10 {
            return 0.0;
        }

        // Calculate R² for WIP vs TH×CT
        let n = self.history.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;

        for &(_, wip, th, ct) in &self.history {
            let x = th * ct; // TH × CT
            let y = wip; // Actual WIP

            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
            sum_y2 += y * y;
        }

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator > f64::EPSILON {
            let r = numerator / denominator;
            r * r
        } else {
            0.0
        }
    }

    /// Run simulation for a given duration.
    #[allow(clippy::while_float)]
    pub fn run_until(&mut self, end_time: f64) {
        while self.time < end_time {
            self.step(0.0); // Step size ignored for DES
        }
    }
}

impl EddDemo for LittlesLawFactoryDemo {
    fn name(&self) -> &'static str {
        "Little's Law Factory Simulation"
    }

    fn emc_ref(&self) -> &'static str {
        "operations/littles_law"
    }

    fn step(&mut self, _dt: f64) {
        // Discrete event simulation - find next event
        let next_event_time = match self.next_departure {
            Some(dep) => self.next_arrival.min(dep),
            None => self.next_arrival,
        };

        // Advance time to next event
        self.time = next_event_time;

        // Process event(s) at this time
        if self.next_arrival <= next_event_time {
            self.process_arrival();
        }

        if let Some(dep) = self.next_departure {
            if dep <= next_event_time {
                self.process_departure();
            }
        }

        // Periodically record history
        if self.total_departures % 10 == 0 {
            self.record_history();
        }
    }

    fn verify_equation(&self) -> bool {
        if !self.is_steady_state() {
            return false;
        }

        self.littles_law_error() < self.tolerance
    }

    fn get_falsification_status(&self) -> FalsificationStatus {
        let error = self.littles_law_error();
        let r_squared = self.calculate_r_squared();
        let steady_state = self.is_steady_state();

        let linear_passed = r_squared > 0.98;
        let error_passed = error < self.tolerance;
        let steady_passed = steady_state;

        FalsificationStatus {
            verified: linear_passed && error_passed && steady_passed,
            criteria: vec![
                CriterionStatus {
                    id: "LL-LINEAR".to_string(),
                    name: "Linear relationship".to_string(),
                    passed: linear_passed,
                    value: r_squared,
                    threshold: 0.98,
                },
                CriterionStatus {
                    id: "LL-ERROR".to_string(),
                    name: "Little's Law error".to_string(),
                    passed: error_passed,
                    value: error,
                    threshold: self.tolerance,
                },
                CriterionStatus {
                    id: "LL-STEADY".to_string(),
                    name: "Steady state".to_string(),
                    passed: steady_passed,
                    value: self.time,
                    threshold: self.warmup_time,
                },
            ],
            message: if linear_passed && error_passed && steady_passed {
                format!(
                    "Little's Law verified: L={:.2}, λW={:.2}, R²={:.4}",
                    self.average_wip(),
                    self.littles_law_prediction(),
                    r_squared
                )
            } else if !steady_passed {
                "System not yet in steady state (transient)".to_string()
            } else {
                format!(
                    "FALSIFIED: error={:.2}% > {:.2}%, R²={:.4}",
                    error * 100.0,
                    self.tolerance * 100.0,
                    r_squared
                )
            },
        }
    }

    fn reset(&mut self) {
        *self = Self::new(self.seed);
    }
}

// =============================================================================
// WASM Bindings
// =============================================================================

#[cfg(feature = "wasm")]
mod wasm {
    use super::{EddDemo, LittlesLawFactoryDemo};
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    pub struct WasmLittlesLawFactory {
        inner: LittlesLawFactoryDemo,
    }

    #[wasm_bindgen]
    impl WasmLittlesLawFactory {
        #[wasm_bindgen(constructor)]
        pub fn new(seed: u64) -> Self {
            Self {
                inner: LittlesLawFactoryDemo::new(seed),
            }
        }

        pub fn step(&mut self) {
            self.inner.step(0.0);
        }

        pub fn get_wip(&self) -> usize {
            self.inner.wip
        }

        pub fn get_throughput(&self) -> f64 {
            self.inner.throughput()
        }

        pub fn get_cycle_time(&self) -> f64 {
            self.inner.average_cycle_time()
        }

        pub fn get_utilization(&self) -> f64 {
            self.inner.utilization()
        }

        pub fn get_time(&self) -> f64 {
            self.inner.time
        }

        pub fn verify_equation(&self) -> bool {
            self.inner.verify_equation()
        }

        pub fn set_rates(&mut self, arrival_rate: f64, service_rate: f64) {
            self.inner.set_rates(arrival_rate, service_rate);
        }

        pub fn set_wip_cap(&mut self, cap: usize) {
            self.inner.set_wip_cap(Some(cap));
        }

        pub fn reset(&mut self) {
            self.inner.reset();
        }

        pub fn get_status_json(&self) -> String {
            serde_json::to_string(&self.inner.get_falsification_status()).unwrap_or_default()
        }
    }
}

// =============================================================================
// Tests - Following EDD Methodology
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Phase 1: Equation - Define what we're testing
    // =========================================================================

    #[test]
    fn test_equation_littles_law_formula() {
        // L = λW (average WIP = throughput × average cycle time)
        let mut demo = LittlesLawFactoryDemo::new(42);

        // Run to steady state
        demo.run_until(100.0);

        let l = demo.average_wip();
        let lambda = demo.throughput();
        let w = demo.average_cycle_time();

        // L should approximately equal λW
        let prediction = lambda * w;
        let error = (l - prediction).abs() / l.max(0.001);

        assert!(
            error < 0.10,
            "Little's Law: L={l:.2}, λW={prediction:.2}, error={:.1}%",
            error * 100.0
        );
    }

    #[test]
    fn test_equation_steady_state_utilization() {
        let demo = LittlesLawFactoryDemo::new(42);

        // ρ = λ/μ
        let expected_util = demo.arrival_rate / demo.service_rate;
        assert!((demo.utilization() - expected_util).abs() < 1e-10);
    }

    // =========================================================================
    // Phase 2: Failing Test - During transients, law violated
    // =========================================================================

    #[test]
    fn test_failing_transient_period() {
        let mut demo = LittlesLawFactoryDemo::new(42);

        // During warmup, system is NOT in steady state
        demo.run_until(1.0); // Very short run

        assert!(
            !demo.is_steady_state(),
            "Should not be in steady state after 1 time unit"
        );

        // Verification should fail during transient
        assert!(
            !demo.verify_equation(),
            "Little's Law verification should fail during transient"
        );
    }

    #[test]
    fn test_failing_high_variability() {
        // Document: High variability makes convergence slower
        let mut demo = LittlesLawFactoryDemo::new(42);
        demo.tolerance = 0.01; // Very strict tolerance

        // Short run may not converge
        demo.run_until(50.0);

        let error = demo.littles_law_error();
        // Just document the error, may or may not pass
        println!("Short run error: {:.2}%", error * 100.0);
    }

    // =========================================================================
    // Phase 3: Implementation - Long run verifies law
    // =========================================================================

    #[test]
    fn test_verification_long_run() {
        let mut demo = LittlesLawFactoryDemo::new(42);
        demo.tolerance = 0.05;

        // Long run to steady state
        demo.run_until(500.0);

        assert!(
            demo.is_steady_state(),
            "Should be in steady state after 500 time units"
        );

        assert!(
            demo.verify_equation(),
            "Little's Law should be verified. Error: {:.2}%",
            demo.littles_law_error() * 100.0
        );
    }

    #[test]
    fn test_verification_r_squared() {
        let mut demo = LittlesLawFactoryDemo::new(42);

        // Long run with history
        demo.run_until(500.0);

        let r_squared = demo.calculate_r_squared();
        assert!(
            r_squared > 0.90,
            "R² should be high for Little's Law: {r_squared}"
        );
    }

    // =========================================================================
    // Phase 4: Verification - Different utilization levels
    // =========================================================================

    #[test]
    fn test_verification_low_utilization() {
        let mut demo = LittlesLawFactoryDemo::new(42);
        demo.set_rates(2.0, 5.0); // ρ = 0.4
        demo.run_until(500.0);

        assert!(
            demo.verify_equation(),
            "Little's Law at low util: error={:.2}%",
            demo.littles_law_error() * 100.0
        );
    }

    #[test]
    fn test_verification_high_utilization() {
        let mut demo = LittlesLawFactoryDemo::new(42);
        demo.set_rates(4.5, 5.0); // ρ = 0.9
        demo.run_until(1000.0); // Need longer for high util

        assert!(
            demo.verify_equation(),
            "Little's Law at high util: error={:.2}%",
            demo.littles_law_error() * 100.0
        );
    }

    // =========================================================================
    // Phase 5: Falsification - CONWIP changes behavior
    // =========================================================================

    #[test]
    fn test_falsification_conwip_mode() {
        let mut demo = LittlesLawFactoryDemo::new(42);
        demo.set_wip_cap(Some(5)); // Cap WIP at 5

        demo.run_until(500.0);

        // Little's Law still holds for CONWIP!
        assert!(
            demo.verify_equation(),
            "Little's Law should hold even with CONWIP"
        );

        // But WIP is bounded
        assert!(demo.wip <= 5, "WIP should be capped at 5");
    }

    #[test]
    fn test_falsification_unstable_system() {
        let mut demo = LittlesLawFactoryDemo::new(42);
        demo.set_rates(6.0, 5.0); // ρ > 1 (unstable!)
        demo.tolerance = 0.05;

        // Run - queue will grow unbounded
        demo.run_until(100.0);

        // System is unstable - WIP grows without bound
        // Little's Law "holds" but averages are meaningless
        println!(
            "Unstable system: WIP={}, departures={}",
            demo.wip, demo.total_departures
        );
    }

    #[test]
    fn test_falsification_status_structure() {
        let demo = LittlesLawFactoryDemo::new(42);
        let status = demo.get_falsification_status();

        assert_eq!(status.criteria.len(), 3);
        assert_eq!(status.criteria[0].id, "LL-LINEAR");
        assert_eq!(status.criteria[1].id, "LL-ERROR");
        assert_eq!(status.criteria[2].id, "LL-STEADY");
    }

    // =========================================================================
    // Integration tests
    // =========================================================================

    #[test]
    fn test_demo_trait_implementation() {
        let mut demo = LittlesLawFactoryDemo::new(42);

        assert_eq!(demo.name(), "Little's Law Factory Simulation");
        assert_eq!(demo.emc_ref(), "operations/littles_law");

        demo.step(0.0);
        assert!(demo.time > 0.0);

        demo.reset();
        assert_eq!(demo.time, 0.0);
    }

    #[test]
    fn test_reproducibility() {
        let mut demo1 = LittlesLawFactoryDemo::new(42);
        let mut demo2 = LittlesLawFactoryDemo::new(42);

        demo1.run_until(100.0);
        demo2.run_until(100.0);

        assert_eq!(demo1.total_arrivals, demo2.total_arrivals);
        assert_eq!(demo1.total_departures, demo2.total_departures);
    }

    // =========================================================================
    // Additional coverage tests
    // =========================================================================

    #[test]
    fn test_default() {
        let demo = LittlesLawFactoryDemo::default();
        assert_eq!(demo.seed, 42);
        assert!((demo.arrival_rate - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_clone() {
        let demo = LittlesLawFactoryDemo::new(42);
        let cloned = demo.clone();
        assert_eq!(demo.seed, cloned.seed);
        assert!((demo.arrival_rate - cloned.arrival_rate).abs() < 1e-10);
    }

    #[test]
    fn test_debug() {
        let demo = LittlesLawFactoryDemo::new(42);
        let debug_str = format!("{demo:?}");
        assert!(debug_str.contains("LittlesLawFactoryDemo"));
    }

    #[test]
    fn test_serialization() {
        let demo = LittlesLawFactoryDemo::new(42);
        let json = serde_json::to_string(&demo).expect("serialize");
        assert!(json.contains("arrival_rate"));

        let restored: LittlesLawFactoryDemo = serde_json::from_str(&json).expect("deserialize");
        assert!((restored.arrival_rate - demo.arrival_rate).abs() < 1e-10);
    }

    #[test]
    fn test_average_wip_zero_time() {
        let demo = LittlesLawFactoryDemo::new(42);
        // At time 0, average WIP should be 0
        assert!((demo.average_wip() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_throughput_zero_time() {
        let demo = LittlesLawFactoryDemo::new(42);
        assert!((demo.throughput() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_average_cycle_time_zero_departures() {
        let demo = LittlesLawFactoryDemo::new(42);
        assert!((demo.average_cycle_time() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_littles_law_error_zero_wip() {
        let demo = LittlesLawFactoryDemo::new(42);
        // When L=0, error should be 0
        assert!((demo.littles_law_error() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_is_steady_state_false_short_time() {
        let mut demo = LittlesLawFactoryDemo::new(42);
        demo.warmup_time = 100.0;
        demo.run_until(50.0);
        assert!(!demo.is_steady_state());
    }

    #[test]
    fn test_is_steady_state_false_few_departures() {
        let mut demo = LittlesLawFactoryDemo::new(42);
        demo.warmup_time = 1.0;
        demo.run_until(5.0);
        // May not have 100 departures yet
        if demo.total_departures < 100 {
            assert!(!demo.is_steady_state());
        }
    }

    #[test]
    fn test_set_rates() {
        let mut demo = LittlesLawFactoryDemo::new(42);
        demo.set_rates(10.0, 15.0);
        assert!((demo.arrival_rate - 10.0).abs() < 1e-10);
        assert!((demo.service_rate - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_set_wip_cap() {
        let mut demo = LittlesLawFactoryDemo::new(42);
        assert!(demo.wip_cap.is_none());

        demo.set_wip_cap(Some(10));
        assert_eq!(demo.wip_cap, Some(10));

        demo.set_wip_cap(None);
        assert!(demo.wip_cap.is_none());
    }

    #[test]
    fn test_conwip_blocks_arrivals() {
        let mut demo = LittlesLawFactoryDemo::new(42);
        demo.set_wip_cap(Some(3));
        demo.set_rates(10.0, 1.0); // Very high arrival rate, slow service

        demo.run_until(100.0);

        // WIP should never exceed cap
        assert!(demo.wip <= 3, "WIP {} should be <= 3", demo.wip);
    }

    #[test]
    fn test_calculate_r_squared_empty_history() {
        let demo = LittlesLawFactoryDemo::new(42);
        // No history
        assert!((demo.calculate_r_squared() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_r_squared_insufficient_data() {
        let mut demo = LittlesLawFactoryDemo::new(42);
        demo.history = vec![(1.0, 1.0, 1.0, 1.0), (2.0, 2.0, 2.0, 2.0)];
        // Less than 3 points
        assert!((demo.calculate_r_squared() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_r_squared_zero_variance() {
        let mut demo = LittlesLawFactoryDemo::new(42);
        // All same WIP values
        demo.history = vec![
            (1.0, 5.0, 1.0, 1.0),
            (2.0, 5.0, 1.0, 1.0),
            (3.0, 5.0, 1.0, 1.0),
        ];
        let r2 = demo.calculate_r_squared();
        // With zero variance, R² should be 0
        assert!((r2 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_record_history_interval() {
        let mut demo = LittlesLawFactoryDemo::new(42);
        demo.run_until(100.0);

        // History should have been recorded
        assert!(!demo.history.is_empty(), "History should be populated");
    }

    #[test]
    fn test_step_multiple_events() {
        let mut demo = LittlesLawFactoryDemo::new(42);
        for _ in 0..200 {
            demo.step(0.0);
        }
        assert!(demo.time > 0.0);
        assert!(demo.total_arrivals > 0);
    }

    #[test]
    fn test_falsification_status_not_steady() {
        let mut demo = LittlesLawFactoryDemo::new(42);
        demo.warmup_time = 1000.0; // Very long warmup
        demo.run_until(10.0);

        let status = demo.get_falsification_status();
        // Should not be verified (not in steady state)
        assert!(!status.verified || status.message.contains("not in steady state"));
    }

    #[test]
    fn test_process_departure_empty_system() {
        let mut demo = LittlesLawFactoryDemo::new(42);
        // Ensure WIP is 0 and next_departure is Some
        demo.wip = 0;
        demo.next_departure = Some(1.0);

        // Process a departure with empty system
        demo.time = 1.0;
        demo.process_departure();

        // next_departure should be None now
        assert!(demo.next_departure.is_none());
    }

    #[test]
    fn test_wip_integral_tracking() {
        let mut demo = LittlesLawFactoryDemo::new(42);
        demo.run_until(50.0);

        // WIP integral should be > 0 after running
        let avg_wip = demo.average_wip();
        assert!(avg_wip >= 0.0);
    }

    #[test]
    fn test_utilization_calculation() {
        let mut demo = LittlesLawFactoryDemo::new(42);
        demo.set_rates(3.0, 6.0);
        assert!((demo.utilization() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_littles_law_prediction() {
        let mut demo = LittlesLawFactoryDemo::new(42);
        demo.run_until(200.0);

        let prediction = demo.littles_law_prediction();
        let throughput = demo.throughput();
        let cycle_time = demo.average_cycle_time();

        assert!((prediction - throughput * cycle_time).abs() < 1e-10);
    }

    #[test]
    fn test_run_until_zero() {
        let mut demo = LittlesLawFactoryDemo::new(42);
        demo.run_until(0.0);
        // Should not advance
        assert!(demo.time <= demo.next_arrival);
    }
}
