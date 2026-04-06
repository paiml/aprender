//! Heijunka (平準化) - Time-budget load leveling.
//!
//! Implements Toyota's Heijunka principle for consistent frame delivery:
//! - Time-budget per frame (16ms target for 60 FPS)
//! - Graceful quality degradation when budget exceeded
//! - Prevents Mura (unevenness) from O(N²) computation
//!
//! # Design Philosophy
//!
//! Rather than dropping frames or stuttering, Heijunka reduces simulation
//! fidelity (substeps) to maintain consistent visual delivery.
//!
//! # References
//!
//! [34] Liker, "The Toyota Way," McGraw-Hill, 2004.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use crate::error::SimResult;
use crate::orbit::physics::{NBodyState, YoshidaIntegrator};
use crate::orbit::units::OrbitTime;

/// Quality level for adaptive degradation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
pub enum QualityLevel {
    /// Minimum quality (fewest substeps).
    Minimum,
    /// Low quality.
    Low,
    /// Medium quality.
    Medium,
    /// High quality (default).
    #[default]
    High,
    /// Maximum quality (most substeps).
    Maximum,
}

impl QualityLevel {
    /// Degrade to next lower quality level.
    #[must_use]
    pub fn degrade(self) -> Self {
        match self {
            Self::Maximum => Self::High,
            Self::High => Self::Medium,
            Self::Medium => Self::Low,
            Self::Low | Self::Minimum => Self::Minimum,
        }
    }

    /// Upgrade to next higher quality level.
    #[must_use]
    pub fn upgrade(self) -> Self {
        match self {
            Self::Minimum => Self::Low,
            Self::Low => Self::Medium,
            Self::Medium => Self::High,
            Self::High | Self::Maximum => Self::Maximum,
        }
    }

    /// Get substep multiplier for this quality level.
    #[must_use]
    pub fn substep_multiplier(self) -> usize {
        contract_pre_iterator!();
        match self {
            Self::Minimum => 1,
            Self::Low => 2,
            Self::Medium => 4,
            Self::High => 8,
            Self::Maximum => 16,
        }
    }
}

/// Heijunka scheduler configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeijunkaConfig {
    /// Total frame budget in milliseconds (default: 16ms for 60 FPS).
    pub frame_budget_ms: f64,
    /// Physics budget as fraction of frame budget (default: 0.5).
    pub physics_budget_fraction: f64,
    /// Base time step in seconds.
    pub base_dt: f64,
    /// Maximum substeps per frame.
    pub max_substeps: usize,
    /// Minimum substeps per frame.
    pub min_substeps: usize,
    /// Auto-adjust quality based on performance.
    pub auto_adjust_quality: bool,
    /// Consecutive frames below budget needed to upgrade quality.
    pub upgrade_threshold: usize,
}

impl Default for HeijunkaConfig {
    fn default() -> Self {
        Self {
            frame_budget_ms: 16.0,        // 60 FPS
            physics_budget_fraction: 0.5, // 50% for physics
            base_dt: 3600.0,              // 1 hour simulation time per step
            max_substeps: 100,
            min_substeps: 1,
            auto_adjust_quality: true,
            upgrade_threshold: 10,
        }
    }
}

impl HeijunkaConfig {
    /// Get physics budget in milliseconds.
    #[must_use]
    pub fn physics_budget_ms(&self) -> f64 {
        self.frame_budget_ms * self.physics_budget_fraction
    }
}

/// Result of a Heijunka frame execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameResult {
    /// Number of substeps executed.
    pub substeps: usize,
    /// Time spent on physics in milliseconds.
    pub physics_time_ms: f64,
    /// Current quality level.
    pub quality: QualityLevel,
    /// Whether budget was exceeded.
    pub budget_exceeded: bool,
    /// Simulation time advanced.
    pub sim_time_advanced: f64,
}

/// Heijunka status for visualization.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HeijunkaStatus {
    /// Current frame budget (ms).
    pub budget_ms: f64,
    /// Time used in last frame (ms).
    pub used_ms: f64,
    /// Substeps in last frame.
    pub substeps: usize,
    /// Current quality level.
    pub quality: QualityLevel,
    /// Average physics time over recent frames (ms).
    pub avg_physics_ms: f64,
    /// Budget utilization (0.0 - 1.0+).
    pub utilization: f64,
}

/// Heijunka scheduler for load-leveled simulation.
#[derive(Debug, Clone)]
pub struct HeijunkaScheduler {
    config: HeijunkaConfig,
    quality: QualityLevel,
    integrator: YoshidaIntegrator,
    consecutive_under_budget: usize,
    physics_times: Vec<f64>,
    status: HeijunkaStatus,
}

impl HeijunkaScheduler {
    /// Create a new Heijunka scheduler.
    #[must_use]
    pub fn new(config: HeijunkaConfig) -> Self {
        Self {
            config,
            quality: QualityLevel::default(),
            integrator: YoshidaIntegrator::new(),
            consecutive_under_budget: 0,
            physics_times: Vec::with_capacity(100),
            status: HeijunkaStatus::default(),
        }
    }

    /// Get current quality level.
    #[must_use]
    pub fn quality(&self) -> QualityLevel {
        self.quality
    }

    /// Set quality level manually.
    pub fn set_quality(&mut self, quality: QualityLevel) {
        self.quality = quality;
    }

    /// Get current status for visualization.
    #[must_use]
    pub fn status(&self) -> &HeijunkaStatus {
        &self.status
    }

    /// Execute one frame of simulation with time-budget management.
    ///
    /// # Errors
    ///
    /// Returns error if physics integration fails.
    pub fn execute_frame(&mut self, state: &mut NBodyState) -> SimResult<FrameResult> {
        let budget_ms = self.config.physics_budget_ms();
        let budget_duration = Duration::from_secs_f64(budget_ms / 1000.0);

        let target_substeps = self
            .quality
            .substep_multiplier()
            .min(self.config.max_substeps)
            .max(self.config.min_substeps);

        let start = Instant::now();
        let mut substeps = 0;
        let mut sim_time_advanced = 0.0;

        // Execute substeps within budget
        while substeps < target_substeps {
            // Check if we have time for another step
            let elapsed = start.elapsed();
            if elapsed >= budget_duration && substeps > 0 {
                break;
            }

            // Execute one physics step
            let dt = OrbitTime::from_seconds(self.config.base_dt);
            self.integrator.step(state, dt)?;

            substeps += 1;
            sim_time_advanced += self.config.base_dt;
        }

        let physics_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        let budget_exceeded = physics_time_ms > budget_ms;

        // Track physics times for averaging
        self.physics_times.push(physics_time_ms);
        if self.physics_times.len() > 100 {
            self.physics_times.remove(0);
        }

        // Auto-adjust quality
        if self.config.auto_adjust_quality {
            self.adjust_quality(budget_exceeded, physics_time_ms, budget_ms);
        }

        // Update status
        let avg_physics_ms = if self.physics_times.is_empty() {
            0.0
        } else {
            self.physics_times.iter().sum::<f64>() / self.physics_times.len() as f64
        };

        self.status = HeijunkaStatus {
            budget_ms,
            used_ms: physics_time_ms,
            substeps,
            quality: self.quality,
            avg_physics_ms,
            utilization: physics_time_ms / budget_ms,
        };

        Ok(FrameResult {
            substeps,
            physics_time_ms,
            quality: self.quality,
            budget_exceeded,
            sim_time_advanced,
        })
    }

    fn adjust_quality(&mut self, budget_exceeded: bool, physics_time_ms: f64, budget_ms: f64) {
        if budget_exceeded {
            // Immediately degrade quality if over budget
            self.quality = self.quality.degrade();
            self.consecutive_under_budget = 0;
        } else if physics_time_ms < budget_ms * 0.5 {
            // Only upgrade if significantly under budget
            self.consecutive_under_budget += 1;
            if self.consecutive_under_budget >= self.config.upgrade_threshold {
                self.quality = self.quality.upgrade();
                self.consecutive_under_budget = 0;
            }
        } else {
            // Reset counter if close to budget
            self.consecutive_under_budget = 0;
        }
    }

    /// Estimate substeps possible within budget for given state.
    #[must_use]
    pub fn estimate_substeps(&self, _state: &NBodyState) -> usize {
        contract_pre_iterator!();
        // Use historical average to estimate
        if self.physics_times.is_empty() {
            return self.quality.substep_multiplier();
        }

        let avg_per_step = self.physics_times.iter().sum::<f64>()
            / self.physics_times.len() as f64
            / self.config.max_substeps.max(1) as f64;

        if avg_per_step > 0.0 {
            let budget_ms = self.config.physics_budget_ms();
            (budget_ms / avg_per_step) as usize
        } else {
            self.quality.substep_multiplier()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orbit::physics::OrbitBody;
    use crate::orbit::units::{OrbitMass, Position3D, Velocity3D, AU, EARTH_MASS, G, SOLAR_MASS};

    fn create_test_state() -> NBodyState {
        let v_circular = (G * SOLAR_MASS / AU).sqrt();
        let bodies = vec![
            OrbitBody::new(
                OrbitMass::from_kg(SOLAR_MASS),
                Position3D::zero(),
                Velocity3D::zero(),
            ),
            OrbitBody::new(
                OrbitMass::from_kg(EARTH_MASS),
                Position3D::from_au(1.0, 0.0, 0.0),
                Velocity3D::from_mps(0.0, v_circular, 0.0),
            ),
        ];
        NBodyState::new(bodies, 1e6)
    }

    #[test]
    fn test_quality_level_degrade() {
        assert_eq!(QualityLevel::Maximum.degrade(), QualityLevel::High);
        assert_eq!(QualityLevel::High.degrade(), QualityLevel::Medium);
        assert_eq!(QualityLevel::Medium.degrade(), QualityLevel::Low);
        assert_eq!(QualityLevel::Low.degrade(), QualityLevel::Minimum);
        assert_eq!(QualityLevel::Minimum.degrade(), QualityLevel::Minimum);
    }

    #[test]
    fn test_quality_level_upgrade() {
        assert_eq!(QualityLevel::Minimum.upgrade(), QualityLevel::Low);
        assert_eq!(QualityLevel::Low.upgrade(), QualityLevel::Medium);
        assert_eq!(QualityLevel::Medium.upgrade(), QualityLevel::High);
        assert_eq!(QualityLevel::High.upgrade(), QualityLevel::Maximum);
        assert_eq!(QualityLevel::Maximum.upgrade(), QualityLevel::Maximum);
    }

    #[test]
    fn test_quality_level_substep_multiplier() {
        assert_eq!(QualityLevel::Minimum.substep_multiplier(), 1);
        assert_eq!(QualityLevel::Low.substep_multiplier(), 2);
        assert_eq!(QualityLevel::Medium.substep_multiplier(), 4);
        assert_eq!(QualityLevel::High.substep_multiplier(), 8);
        assert_eq!(QualityLevel::Maximum.substep_multiplier(), 16);
    }

    #[test]
    fn test_heijunka_config_default() {
        let config = HeijunkaConfig::default();
        assert!((config.frame_budget_ms - 16.0).abs() < 1e-10);
        assert!((config.physics_budget_fraction - 0.5).abs() < 1e-10);
        assert!(config.auto_adjust_quality);
    }

    #[test]
    fn test_heijunka_config_physics_budget() {
        let config = HeijunkaConfig::default();
        let physics_budget = config.physics_budget_ms();
        assert!((physics_budget - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_heijunka_scheduler_creation() {
        let config = HeijunkaConfig::default();
        let scheduler = HeijunkaScheduler::new(config);
        assert_eq!(scheduler.quality(), QualityLevel::High);
    }

    #[test]
    fn test_heijunka_scheduler_set_quality() {
        let config = HeijunkaConfig::default();
        let mut scheduler = HeijunkaScheduler::new(config);

        scheduler.set_quality(QualityLevel::Low);
        assert_eq!(scheduler.quality(), QualityLevel::Low);
    }

    #[test]
    fn test_heijunka_execute_frame() {
        let mut config = HeijunkaConfig::default();
        config.frame_budget_ms = 1000.0; // Large budget for test
        config.max_substeps = 4;

        let mut scheduler = HeijunkaScheduler::new(config);
        let mut state = create_test_state();

        let result = scheduler.execute_frame(&mut state).expect("frame failed");

        assert!(result.substeps > 0);
        assert!(result.sim_time_advanced > 0.0);
        assert!(result.physics_time_ms >= 0.0);
    }

    #[test]
    fn test_heijunka_status_update() {
        let mut config = HeijunkaConfig::default();
        config.frame_budget_ms = 1000.0;
        config.max_substeps = 2;

        let mut scheduler = HeijunkaScheduler::new(config);
        let mut state = create_test_state();

        scheduler.execute_frame(&mut state).expect("frame failed");

        let status = scheduler.status();
        assert!((status.budget_ms - 500.0).abs() < 1e-10); // 50% of 1000
        assert!(status.used_ms >= 0.0);
        assert!(status.substeps > 0);
    }

    #[test]
    fn test_heijunka_frame_result() {
        let result = FrameResult {
            substeps: 4,
            physics_time_ms: 5.0,
            quality: QualityLevel::High,
            budget_exceeded: false,
            sim_time_advanced: 14400.0,
        };

        assert_eq!(result.substeps, 4);
        assert!(!result.budget_exceeded);
    }

    #[test]
    fn test_heijunka_estimate_substeps() {
        let config = HeijunkaConfig::default();
        let scheduler = HeijunkaScheduler::new(config);
        let state = create_test_state();

        let estimate = scheduler.estimate_substeps(&state);
        assert!(estimate > 0);
    }

    #[test]
    fn test_heijunka_auto_quality_disabled() {
        let mut config = HeijunkaConfig::default();
        config.auto_adjust_quality = false;
        config.frame_budget_ms = 1000.0;

        let mut scheduler = HeijunkaScheduler::new(config);
        let mut state = create_test_state();

        let initial_quality = scheduler.quality();

        for _ in 0..20 {
            scheduler.execute_frame(&mut state).expect("frame failed");
        }

        // Quality should not change when auto-adjust is disabled
        assert_eq!(scheduler.quality(), initial_quality);
    }
}
