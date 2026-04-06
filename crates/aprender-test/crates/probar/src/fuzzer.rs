//! Monte Carlo fuzzing for game input testing.
//!
//! Per spec Section 6.4: Monte Carlo simulation for edge case discovery.
//!
//! # Example
//!
//! ```ignore
//! let mut fuzzer = InputFuzzer::new(Seed::from_u64(12345));
//! for _ in 0..10_000 {
//!     let inputs = fuzzer.generate_valid_inputs();
//!     game.update(inputs);
//! }
//! ```

use crate::event::InputEvent;

/// Deterministic seed for reproducible fuzzing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Seed(u64);

impl Seed {
    /// Create a seed from a u64 value
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value)
    }

    /// Get the raw seed value
    #[must_use]
    pub const fn value(self) -> u64 {
        self.0
    }
}

/// Simple xorshift64 PRNG for deterministic fuzzing
#[derive(Debug, Clone)]
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    const fn new(seed: Seed) -> Self {
        // Ensure non-zero state
        let state = if seed.0 == 0 { 1 } else { seed.0 };
        Self { state }
    }

    const fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    #[allow(clippy::cast_precision_loss)]
    fn next_f32(&mut self) -> f32 {
        (self.next() as f32) / (u64::MAX as f32)
    }

    const fn next_range(&mut self, min: u64, max: u64) -> u64 {
        if min >= max {
            return min;
        }
        min + (self.next() % (max - min))
    }

    #[allow(clippy::suboptimal_flops)]
    fn next_f32_range(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_f32() * (max - min)
    }
}

/// Configuration for input fuzzing
#[derive(Debug, Clone)]
pub struct FuzzerConfig {
    /// Viewport width for coordinate generation
    pub viewport_width: f32,
    /// Viewport height for coordinate generation
    pub viewport_height: f32,
    /// Probability of generating a touch input (0.0-1.0)
    pub touch_probability: f32,
    /// Probability of generating a key input (0.0-1.0)
    pub key_probability: f32,
    /// Probability of generating a mouse input (0.0-1.0)
    pub mouse_probability: f32,
    /// Maximum swipe distance
    pub max_swipe_distance: f32,
    /// Maximum hold duration in ms
    pub max_hold_duration: u32,
}

impl Default for FuzzerConfig {
    fn default() -> Self {
        Self {
            viewport_width: 800.0,
            viewport_height: 600.0,
            touch_probability: 0.5,
            key_probability: 0.3,
            mouse_probability: 0.2,
            max_swipe_distance: 200.0,
            max_hold_duration: 1000,
        }
    }
}

impl FuzzerConfig {
    /// Set viewport dimensions
    #[must_use]
    pub const fn with_viewport(mut self, width: f32, height: f32) -> Self {
        self.viewport_width = width;
        self.viewport_height = height;
        self
    }
}

/// Input fuzzer for Monte Carlo game testing
///
/// Generates random but valid game inputs for stress testing.
#[derive(Debug, Clone)]
pub struct InputFuzzer {
    rng: Xorshift64,
    config: FuzzerConfig,
    inputs_generated: u64,
}

impl InputFuzzer {
    /// Create a new fuzzer with the given seed
    #[must_use]
    pub fn new(seed: Seed) -> Self {
        Self {
            rng: Xorshift64::new(seed),
            config: FuzzerConfig::default(),
            inputs_generated: 0,
        }
    }

    /// Create a fuzzer with custom configuration
    #[must_use]
    pub const fn with_config(seed: Seed, config: FuzzerConfig) -> Self {
        Self {
            rng: Xorshift64::new(seed),
            config,
            inputs_generated: 0,
        }
    }

    /// Generate a batch of valid random inputs
    #[must_use]
    pub fn generate_valid_inputs(&mut self) -> Vec<InputEvent> {
        let mut inputs = Vec::new();
        let roll = self.rng.next_f32();

        if roll < self.config.touch_probability {
            inputs.push(self.generate_touch_input());
        } else if roll < self.config.touch_probability + self.config.key_probability {
            inputs.push(self.generate_key_input());
        } else {
            inputs.push(self.generate_mouse_input());
        }

        self.inputs_generated += inputs.len() as u64;
        inputs
    }

    /// Generate a single random touch input
    fn generate_touch_input(&mut self) -> InputEvent {
        let x = self.rng.next_f32_range(0.0, self.config.viewport_width);
        let y = self.rng.next_f32_range(0.0, self.config.viewport_height);
        InputEvent::Touch { x, y }
    }

    /// Generate a single random key input
    #[allow(clippy::cast_possible_truncation)]
    fn generate_key_input(&mut self) -> InputEvent {
        const VALID_KEYS: &[&str] = &[
            "ArrowUp",
            "ArrowDown",
            "ArrowLeft",
            "ArrowRight",
            "Space",
            "Enter",
            "Escape",
            "KeyW",
            "KeyA",
            "KeyS",
            "KeyD",
        ];

        // Safe: VALID_KEYS.len() is small and fits in u64/usize
        let idx = self.rng.next_range(0, VALID_KEYS.len() as u64) as usize;
        InputEvent::key_press(VALID_KEYS[idx])
    }

    /// Generate a single random mouse input
    fn generate_mouse_input(&mut self) -> InputEvent {
        let x = self.rng.next_f32_range(0.0, self.config.viewport_width);
        let y = self.rng.next_f32_range(0.0, self.config.viewport_height);
        InputEvent::mouse_click(x, y)
    }

    /// Get the total number of inputs generated
    #[must_use]
    pub const fn inputs_generated(&self) -> u64 {
        self.inputs_generated
    }

    /// Reset the fuzzer to its initial state with a new seed
    pub const fn reset(&mut self, seed: Seed) {
        self.rng = Xorshift64::new(seed);
        self.inputs_generated = 0;
    }

    /// Get the current configuration
    #[must_use]
    pub const fn config(&self) -> &FuzzerConfig {
        &self.config
    }
}

/// Invariant checker for game state validation during fuzzing
#[derive(Debug, Clone, Default)]
pub struct InvariantChecker {
    checks: Vec<InvariantCheck>,
    violations: Vec<InvariantViolation>,
}

/// A single invariant check
#[derive(Debug, Clone)]
pub struct InvariantCheck {
    /// Name of the invariant
    pub name: String,
    /// Description of what the invariant checks
    pub description: String,
}

impl InvariantCheck {
    /// Create a new invariant check
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
        }
    }
}

/// A violation of an invariant
#[derive(Debug, Clone)]
pub struct InvariantViolation {
    /// Name of the violated invariant
    pub invariant_name: String,
    /// Description of the violation
    pub message: String,
    /// Step at which the violation occurred
    pub step: u64,
}

impl InvariantChecker {
    /// Create a new invariant checker
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an invariant check
    pub fn add_check(&mut self, check: InvariantCheck) {
        self.checks.push(check);
    }

    /// Record a violation
    pub fn record_violation(&mut self, invariant_name: &str, message: &str, step: u64) {
        self.violations.push(InvariantViolation {
            invariant_name: invariant_name.to_string(),
            message: message.to_string(),
            step,
        });
    }

    /// Check if any violations occurred
    #[must_use]
    pub fn has_violations(&self) -> bool {
        !self.violations.is_empty()
    }

    /// Get all violations
    #[must_use]
    pub fn violations(&self) -> &[InvariantViolation] {
        &self.violations
    }

    /// Get the number of checks
    #[must_use]
    pub fn check_count(&self) -> usize {
        self.checks.len()
    }

    /// Clear all violations
    pub fn clear_violations(&mut self) {
        self.violations.clear();
    }
}

/// Standard game invariants per spec Section 6.4
pub mod standard_invariants {
    use super::InvariantCheck;

    /// Health must be non-negative
    #[must_use]
    pub fn health_non_negative() -> InvariantCheck {
        InvariantCheck::new("health_non_negative", "Player health must be >= 0")
    }

    /// Entity count must not explode
    #[must_use]
    pub fn entity_count_bounded(max: usize) -> InvariantCheck {
        InvariantCheck::new(
            "entity_count_bounded",
            format!("Entity count must be < {max}"),
        )
    }

    /// Physics must remain stable
    #[must_use]
    pub fn physics_stable() -> InvariantCheck {
        InvariantCheck::new("physics_stable", "Physics simulation must remain stable")
    }

    /// Score must be valid
    #[must_use]
    pub fn score_valid() -> InvariantCheck {
        InvariantCheck::new("score_valid", "Score must be a valid number")
    }

    /// Positions must be within bounds
    #[must_use]
    pub fn positions_in_bounds() -> InvariantCheck {
        InvariantCheck::new(
            "positions_in_bounds",
            "All positions must be within world bounds",
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    mod seed_tests {
        use super::*;

        #[test]
        fn test_seed_from_u64() {
            let seed = Seed::from_u64(12345);
            assert_eq!(seed.value(), 12345);
        }

        #[test]
        fn test_seed_default() {
            let seed = Seed::default();
            assert_eq!(seed.value(), 0);
        }
    }

    mod xorshift_tests {
        use super::*;

        #[test]
        fn test_xorshift_deterministic() {
            let mut rng1 = Xorshift64::new(Seed::from_u64(42));
            let mut rng2 = Xorshift64::new(Seed::from_u64(42));

            for _ in 0..100 {
                assert_eq!(rng1.next(), rng2.next());
            }
        }

        #[test]
        fn test_xorshift_different_seeds() {
            let mut rng1 = Xorshift64::new(Seed::from_u64(1));
            let mut rng2 = Xorshift64::new(Seed::from_u64(2));

            // Should produce different sequences
            let seq1: Vec<u64> = (0..10).map(|_| rng1.next()).collect();
            let seq2: Vec<u64> = (0..10).map(|_| rng2.next()).collect();
            assert_ne!(seq1, seq2);
        }

        #[test]
        fn test_xorshift_f32_range() {
            let mut rng = Xorshift64::new(Seed::from_u64(42));

            for _ in 0..1000 {
                let value = rng.next_f32_range(10.0, 20.0);
                assert!((10.0..20.0).contains(&value));
            }
        }

        #[test]
        fn test_xorshift_range() {
            let mut rng = Xorshift64::new(Seed::from_u64(42));

            for _ in 0..1000 {
                let value = rng.next_range(5, 15);
                assert!((5..15).contains(&value));
            }
        }
    }

    mod fuzzer_tests {
        use super::*;

        #[test]
        fn test_fuzzer_deterministic() {
            let mut fuzzer1 = InputFuzzer::new(Seed::from_u64(12345));
            let mut fuzzer2 = InputFuzzer::new(Seed::from_u64(12345));

            for _ in 0..100 {
                let inputs1 = fuzzer1.generate_valid_inputs();
                let inputs2 = fuzzer2.generate_valid_inputs();

                assert_eq!(inputs1.len(), inputs2.len());
            }
        }

        #[test]
        fn test_fuzzer_generates_inputs() {
            let mut fuzzer = InputFuzzer::new(Seed::from_u64(42));

            for _ in 0..100 {
                let inputs = fuzzer.generate_valid_inputs();
                assert!(!inputs.is_empty());
            }
        }

        #[test]
        fn test_fuzzer_tracks_count() {
            let mut fuzzer = InputFuzzer::new(Seed::from_u64(42));
            assert_eq!(fuzzer.inputs_generated(), 0);

            for _ in 0..10 {
                let _ = fuzzer.generate_valid_inputs();
            }

            assert!(fuzzer.inputs_generated() >= 10);
        }

        #[test]
        fn test_fuzzer_reset() {
            let mut fuzzer = InputFuzzer::new(Seed::from_u64(42));

            // Generate some inputs
            for _ in 0..10 {
                let _ = fuzzer.generate_valid_inputs();
            }

            // Reset
            fuzzer.reset(Seed::from_u64(42));
            assert_eq!(fuzzer.inputs_generated(), 0);
        }

        #[test]
        fn test_fuzzer_with_config() {
            let config = FuzzerConfig::default().with_viewport(1920.0, 1080.0);
            let fuzzer = InputFuzzer::with_config(Seed::from_u64(42), config);

            assert!((fuzzer.config().viewport_width - 1920.0).abs() < f32::EPSILON);
            assert!((fuzzer.config().viewport_height - 1080.0).abs() < f32::EPSILON);
        }

        #[test]
        fn test_fuzzer_generates_all_input_types() {
            let mut fuzzer = InputFuzzer::new(Seed::from_u64(42));
            let mut has_touch = false;
            let mut has_key = false;
            let mut has_mouse = false;

            // Generate many inputs to cover all types
            for _ in 0..1000 {
                let inputs = fuzzer.generate_valid_inputs();
                for input in inputs {
                    match input {
                        InputEvent::Touch { .. } => has_touch = true,
                        InputEvent::KeyPress { .. } => has_key = true,
                        InputEvent::MouseClick { .. } => has_mouse = true,
                        _ => {}
                    }
                }
            }

            assert!(has_touch, "Should generate touch inputs");
            assert!(has_key, "Should generate key inputs");
            assert!(has_mouse, "Should generate mouse inputs");
        }

        #[test]
        fn test_fuzzer_touch_within_viewport() {
            let config = FuzzerConfig::default().with_viewport(800.0, 600.0);
            let mut fuzzer = InputFuzzer::with_config(Seed::from_u64(42), config);

            for _ in 0..1000 {
                let inputs = fuzzer.generate_valid_inputs();
                for input in inputs {
                    if let InputEvent::Touch { x, y, .. } = input {
                        assert!(
                            (0.0..=800.0).contains(&x),
                            "Touch x={x} should be within viewport"
                        );
                        assert!(
                            (0.0..=600.0).contains(&y),
                            "Touch y={y} should be within viewport"
                        );
                    }
                }
            }
        }
    }

    mod invariant_tests {
        use super::*;

        #[test]
        fn test_invariant_checker_new() {
            let checker = InvariantChecker::new();
            assert!(!checker.has_violations());
            assert_eq!(checker.check_count(), 0);
        }

        #[test]
        fn test_add_check() {
            let mut checker = InvariantChecker::new();
            checker.add_check(standard_invariants::health_non_negative());
            assert_eq!(checker.check_count(), 1);
        }

        #[test]
        fn test_record_violation() {
            let mut checker = InvariantChecker::new();
            checker.record_violation("test", "Test violation", 42);

            assert!(checker.has_violations());
            assert_eq!(checker.violations().len(), 1);
            assert_eq!(checker.violations()[0].step, 42);
        }

        #[test]
        fn test_clear_violations() {
            let mut checker = InvariantChecker::new();
            checker.record_violation("test", "Test violation", 1);
            assert!(checker.has_violations());

            checker.clear_violations();
            assert!(!checker.has_violations());
        }

        #[test]
        fn test_standard_invariants() {
            let health = standard_invariants::health_non_negative();
            assert_eq!(health.name, "health_non_negative");

            let entities = standard_invariants::entity_count_bounded(1000);
            assert!(entities.description.contains("1000"));

            let physics = standard_invariants::physics_stable();
            assert_eq!(physics.name, "physics_stable");

            let score = standard_invariants::score_valid();
            assert_eq!(score.name, "score_valid");

            let positions = standard_invariants::positions_in_bounds();
            assert_eq!(positions.name, "positions_in_bounds");
        }
    }

    mod monte_carlo_simulation_tests {
        use super::*;

        #[test]
        fn test_10k_steps_no_panic() {
            let mut fuzzer = InputFuzzer::new(Seed::from_u64(12345));
            let mut checker = InvariantChecker::new();
            checker.add_check(standard_invariants::health_non_negative());
            checker.add_check(standard_invariants::entity_count_bounded(2000));

            // Simulate 10,000 steps as per spec
            for step in 0..10_000 {
                let inputs = fuzzer.generate_valid_inputs();

                // Verify inputs are valid
                for input in &inputs {
                    match input {
                        InputEvent::Touch { x, y, .. } | InputEvent::MouseClick { x, y } => {
                            assert!(x.is_finite() && y.is_finite());
                        }
                        InputEvent::KeyPress { key } => {
                            assert!(!key.is_empty());
                        }
                        _ => {}
                    }
                }

                // Simulated game state checks
                let simulated_health = 100 - (step % 100) as i32;
                if simulated_health < 0 {
                    checker.record_violation(
                        "health_non_negative",
                        "Health dropped below zero",
                        step,
                    );
                }
            }

            assert_eq!(fuzzer.inputs_generated(), 10_000);
            // In this simulation, health is always >= 0
            assert!(!checker.has_violations());
        }
    }
}
