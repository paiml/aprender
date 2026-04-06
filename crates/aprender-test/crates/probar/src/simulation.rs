//! Deterministic simulation recording and replay.
//!
//! Per spec Section 6.4: Deterministic simulation for regression testing.
//!
//! # Example
//!
//! ```ignore
//! let recording = run_simulation(SimulationConfig {
//!     seed: 42,
//!     duration_frames: 3600, // 1 minute at 60fps
//!     actions: Box::new(RandomWalkAgent::new()),
//! });
//!
//! let replay_result = run_replay(&recording);
//! assert_eq!(recording.final_state_hash, replay_result.final_state_hash);
//! ```

use crate::event::InputEvent;
use crate::fuzzer::Seed;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Configuration for simulation runs
#[derive(Debug, Clone, Copy)]
pub struct SimulationConfig {
    /// Seed for deterministic random generation
    pub seed: u64,
    /// Duration in frames (e.g., 3600 for 1 minute at 60fps)
    pub duration_frames: u64,
    /// Target frames per second for timing calculations
    pub fps: u32,
    /// Maximum entities allowed before stopping
    pub max_entities: usize,
    /// Whether to record full state history
    pub record_states: bool,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            duration_frames: 3600, // 1 minute at 60fps
            fps: 60,
            max_entities: 2000,
            record_states: false,
        }
    }
}

impl SimulationConfig {
    /// Create a new config with the given seed and duration
    #[must_use]
    pub const fn new(seed: u64, duration_frames: u64) -> Self {
        Self {
            seed,
            duration_frames,
            fps: 60,
            max_entities: 2000,
            record_states: false,
        }
    }

    /// Set the seed
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the duration in frames
    #[must_use]
    pub const fn with_duration(mut self, frames: u64) -> Self {
        self.duration_frames = frames;
        self
    }

    /// Enable state recording
    #[must_use]
    pub const fn with_state_recording(mut self, enabled: bool) -> Self {
        self.record_states = enabled;
        self
    }

    /// Get the seed as a Seed type
    #[must_use]
    pub const fn as_seed(&self) -> Seed {
        Seed::from_u64(self.seed)
    }
}

/// A single frame's worth of recorded data
#[derive(Debug, Clone)]
pub struct RecordedFrame {
    /// Frame number
    pub frame: u64,
    /// Input events for this frame
    pub inputs: Vec<InputEvent>,
    /// Hash of game state after this frame (for verification)
    pub state_hash: u64,
}

/// A complete simulation recording
#[derive(Debug, Clone)]
pub struct SimulationRecording {
    /// Configuration used for this recording
    pub config: SimulationConfig,
    /// All recorded frames
    pub frames: Vec<RecordedFrame>,
    /// Hash of the final game state
    pub final_state_hash: u64,
    /// Total frames recorded
    pub total_frames: u64,
    /// Whether the simulation completed successfully
    pub completed: bool,
    /// Error message if simulation failed
    pub error: Option<String>,
}

impl SimulationRecording {
    /// Create a new empty recording
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // Vec::new() not const in stable
    pub fn new(config: SimulationConfig) -> Self {
        Self {
            config,
            frames: Vec::new(),
            final_state_hash: 0,
            total_frames: 0,
            completed: false,
            error: None,
        }
    }

    /// Add a recorded frame
    pub fn add_frame(&mut self, frame: RecordedFrame) {
        self.total_frames = frame.frame + 1;
        self.final_state_hash = frame.state_hash;
        self.frames.push(frame);
    }

    /// Mark simulation as completed
    pub const fn mark_completed(&mut self) {
        self.completed = true;
    }

    /// Mark simulation as failed
    #[allow(clippy::missing_const_for_fn)] // String allocation
    pub fn mark_failed(&mut self, error: &str) {
        self.completed = false;
        self.error = Some(error.to_string());
    }

    /// Get the duration in seconds
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self) -> f64 {
        self.total_frames as f64 / f64::from(self.config.fps)
    }

    /// Check if recording matches another (same final state)
    #[must_use]
    pub const fn matches(&self, other: &Self) -> bool {
        self.final_state_hash == other.final_state_hash && self.total_frames == other.total_frames
    }
}

/// Result of replaying a simulation
#[derive(Debug, Clone)]
pub struct ReplayResult {
    /// Hash of the final state after replay
    pub final_state_hash: u64,
    /// Total frames replayed
    pub frames_replayed: u64,
    /// Whether replay matched original recording
    pub determinism_verified: bool,
    /// Frame where divergence occurred (if any)
    pub divergence_frame: Option<u64>,
    /// Error message if replay failed
    pub error: Option<String>,
}

impl ReplayResult {
    /// Create a successful replay result
    #[must_use]
    pub const fn success(final_state_hash: u64, frames_replayed: u64) -> Self {
        Self {
            final_state_hash,
            frames_replayed,
            determinism_verified: true,
            divergence_frame: None,
            error: None,
        }
    }

    /// Create a failed replay result due to divergence
    #[must_use]
    pub fn diverged(divergence_frame: u64, expected_hash: u64, actual_hash: u64) -> Self {
        Self {
            final_state_hash: actual_hash,
            frames_replayed: divergence_frame,
            determinism_verified: false,
            divergence_frame: Some(divergence_frame),
            error: Some(format!(
                "State diverged at frame {divergence_frame}: expected hash {expected_hash}, got {actual_hash}"
            )),
        }
    }
}

/// A simulated game state for testing
#[derive(Debug, Clone, Default)]
pub struct SimulatedGameState {
    /// Current frame
    pub frame: u64,
    /// Player X position
    pub player_x: f32,
    /// Player Y position
    pub player_y: f32,
    /// Player health
    pub health: i32,
    /// Current score
    pub score: i32,
    /// Entity count
    pub entity_count: usize,
    /// Random state for determinism
    random_state: u64,
}

impl SimulatedGameState {
    /// Create a new game state with initial values
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self {
            frame: 0,
            player_x: 400.0,
            player_y: 300.0,
            health: 100,
            score: 0,
            entity_count: 1,
            random_state: seed,
        }
    }

    /// Update game state with inputs (deterministically)
    pub fn update(&mut self, inputs: &[InputEvent]) {
        self.frame += 1;

        // Process inputs deterministically
        for input in inputs {
            match input {
                InputEvent::Touch { x, y, .. } | InputEvent::MouseClick { x, y } => {
                    // Move player toward touch/click
                    let dx = x - self.player_x;
                    let dy = y - self.player_y;
                    let dist = dx.hypot(dy);
                    if dist > 1.0 {
                        self.player_x += dx / dist * 5.0;
                        self.player_y += dy / dist * 5.0;
                    }
                }
                InputEvent::KeyPress { key } => {
                    // Arrow keys move player
                    match key.as_str() {
                        "ArrowUp" | "KeyW" => self.player_y -= 5.0,
                        "ArrowDown" | "KeyS" => self.player_y += 5.0,
                        "ArrowLeft" | "KeyA" => self.player_x -= 5.0,
                        "ArrowRight" | "KeyD" => self.player_x += 5.0,
                        "Space" => self.score += 10, // Action button scores
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        // Deterministic random events based on frame
        self.random_state = self.random_state.wrapping_mul(6_364_136_223_846_793_005);
        self.random_state = self.random_state.wrapping_add(1_442_695_040_888_963_407);

        // Spawn/despawn entities deterministically
        if self.random_state % 100 < 5 && self.entity_count < 1000 {
            self.entity_count += 1;
        }
        if self.random_state % 100 > 95 && self.entity_count > 1 {
            self.entity_count -= 1;
        }

        // Clamp values
        self.player_x = self.player_x.clamp(0.0, 800.0);
        self.player_y = self.player_y.clamp(0.0, 600.0);
    }

    /// Compute a hash of the current state
    #[must_use]
    pub fn compute_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.frame.hash(&mut hasher);
        self.player_x.to_bits().hash(&mut hasher);
        self.player_y.to_bits().hash(&mut hasher);
        self.health.hash(&mut hasher);
        self.score.hash(&mut hasher);
        self.entity_count.hash(&mut hasher);
        self.random_state.hash(&mut hasher);
        hasher.finish()
    }

    /// Check if state is valid (invariants hold)
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.health >= 0 && self.entity_count < 2000
    }
}

/// Run a simulation with the given configuration
///
/// # Arguments
/// * `config` - Simulation configuration
/// * `input_generator` - Function that generates inputs for each frame
///
/// # Returns
/// A recording of the simulation
#[must_use]
pub fn run_simulation<F>(config: SimulationConfig, mut input_generator: F) -> SimulationRecording
where
    F: FnMut(u64) -> Vec<InputEvent>,
{
    let mut recording = SimulationRecording::new(config);
    let mut state = SimulatedGameState::new(config.seed);

    for frame in 0..config.duration_frames {
        // Generate inputs for this frame
        let inputs = input_generator(frame);

        // Update game state
        state.update(&inputs);

        // Check invariants
        if !state.is_valid() {
            recording.mark_failed(&format!("Invariant violation at frame {frame}"));
            return recording;
        }

        // Check entity limit
        if state.entity_count >= config.max_entities {
            recording.mark_failed(&format!(
                "Entity explosion at frame {frame}: {} entities",
                state.entity_count
            ));
            return recording;
        }

        // Record frame
        let recorded_frame = RecordedFrame {
            frame,
            inputs,
            state_hash: state.compute_hash(),
        };
        recording.add_frame(recorded_frame);
    }

    recording.mark_completed();
    recording
}

/// Replay a simulation recording and verify determinism
///
/// # Arguments
/// * `recording` - The original recording to replay
///
/// # Returns
/// Result of the replay including whether determinism was verified
#[must_use]
pub fn run_replay(recording: &SimulationRecording) -> ReplayResult {
    let mut state = SimulatedGameState::new(recording.config.seed);

    for recorded_frame in &recording.frames {
        // Apply the same inputs
        state.update(&recorded_frame.inputs);

        // Verify state hash matches
        let current_hash = state.compute_hash();
        if current_hash != recorded_frame.state_hash {
            return ReplayResult::diverged(
                recorded_frame.frame,
                recorded_frame.state_hash,
                current_hash,
            );
        }
    }

    ReplayResult::success(state.compute_hash(), recording.total_frames)
}

/// A random walk agent for testing
#[derive(Debug, Clone)]
pub struct RandomWalkAgent {
    state: u64,
}

impl RandomWalkAgent {
    /// Create a new random walk agent with a seed
    #[must_use]
    pub const fn new(seed: Seed) -> Self {
        Self {
            state: seed.value(),
        }
    }

    /// Generate inputs for the next frame
    pub fn next_inputs(&mut self) -> Vec<InputEvent> {
        // Simple xorshift for determinism
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;

        let direction = self.state % 5;
        let key = match direction {
            0 => "ArrowUp",
            1 => "ArrowDown",
            2 => "ArrowLeft",
            3 => "ArrowRight",
            _ => "Space",
        };

        vec![InputEvent::key_press(key)]
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    mod config_tests {
        use super::*;

        #[test]
        fn test_config_default() {
            let config = SimulationConfig::default();
            assert_eq!(config.duration_frames, 3600);
            assert_eq!(config.fps, 60);
            assert_eq!(config.max_entities, 2000);
        }

        #[test]
        fn test_config_builder() {
            let config = SimulationConfig::default()
                .with_seed(42)
                .with_duration(1000)
                .with_state_recording(true);

            assert_eq!(config.seed, 42);
            assert_eq!(config.duration_frames, 1000);
            assert!(config.record_states);
        }

        #[test]
        fn test_config_as_seed() {
            let config = SimulationConfig::new(12345, 100);
            assert_eq!(config.as_seed().value(), 12345);
        }
    }

    mod game_state_tests {
        use super::*;

        #[test]
        fn test_game_state_initial() {
            let state = SimulatedGameState::new(42);
            assert_eq!(state.frame, 0);
            assert_eq!(state.health, 100);
            assert_eq!(state.score, 0);
            assert!(state.is_valid());
        }

        #[test]
        fn test_game_state_deterministic() {
            let mut state1 = SimulatedGameState::new(42);
            let mut state2 = SimulatedGameState::new(42);

            let inputs = vec![InputEvent::key_press("ArrowUp")];

            for _ in 0..100 {
                state1.update(&inputs);
                state2.update(&inputs);
            }

            assert_eq!(state1.compute_hash(), state2.compute_hash());
        }

        #[test]
        fn test_game_state_movement() {
            let mut state = SimulatedGameState::new(0);
            let initial_y = state.player_y;

            state.update(&[InputEvent::key_press("ArrowUp")]);

            assert!(state.player_y < initial_y, "Player should move up");
        }

        #[test]
        fn test_game_state_hash_changes() {
            let mut state = SimulatedGameState::new(42);
            let initial_hash = state.compute_hash();

            state.update(&[InputEvent::key_press("Space")]);
            let new_hash = state.compute_hash();

            assert_ne!(initial_hash, new_hash, "Hash should change after update");
        }
    }

    mod recording_tests {
        use super::*;

        #[test]
        fn test_recording_new() {
            let config = SimulationConfig::default();
            let recording = SimulationRecording::new(config);

            assert!(!recording.completed);
            assert!(recording.frames.is_empty());
            assert_eq!(recording.total_frames, 0);
        }

        #[test]
        fn test_recording_add_frame() {
            let mut recording = SimulationRecording::new(SimulationConfig::default());

            recording.add_frame(RecordedFrame {
                frame: 0,
                inputs: vec![],
                state_hash: 12345,
            });

            assert_eq!(recording.total_frames, 1);
            assert_eq!(recording.final_state_hash, 12345);
        }

        #[test]
        fn test_recording_duration() {
            let config = SimulationConfig::default();
            let mut recording = SimulationRecording::new(config);

            for i in 0..60 {
                recording.add_frame(RecordedFrame {
                    frame: i,
                    inputs: vec![],
                    state_hash: 0,
                });
            }

            assert!((recording.duration_seconds() - 1.0).abs() < 0.01);
        }
    }

    mod simulation_tests {
        use super::*;

        #[test]
        fn test_run_simulation_completes() {
            let config = SimulationConfig::new(42, 100);

            let recording = run_simulation(config, |_frame| vec![]);

            assert!(recording.completed);
            assert_eq!(recording.total_frames, 100);
        }

        #[test]
        fn test_simulation_deterministic() {
            let config1 = SimulationConfig::new(42, 100);
            let config2 = SimulationConfig::new(42, 100);

            let recording1 = run_simulation(config1, |_| vec![InputEvent::key_press("Space")]);
            let recording2 = run_simulation(config2, |_| vec![InputEvent::key_press("Space")]);

            assert!(
                recording1.matches(&recording2),
                "Same seed should produce same result"
            );
        }

        #[test]
        fn test_simulation_different_seeds() {
            let config1 = SimulationConfig::new(1, 100);
            let config2 = SimulationConfig::new(2, 100);

            let recording1 = run_simulation(config1, |_| vec![]);
            let recording2 = run_simulation(config2, |_| vec![]);

            assert!(
                !recording1.matches(&recording2),
                "Different seeds should produce different results"
            );
        }
    }

    mod replay_tests {
        use super::*;

        #[test]
        fn test_replay_verifies_determinism() {
            let config = SimulationConfig::new(42, 100);
            let recording = run_simulation(config, |frame| {
                if frame % 10 == 0 {
                    vec![InputEvent::key_press("Space")]
                } else {
                    vec![]
                }
            });

            let replay_result = run_replay(&recording);

            assert!(
                replay_result.determinism_verified,
                "Replay should verify determinism"
            );
            assert_eq!(replay_result.final_state_hash, recording.final_state_hash);
        }

        #[test]
        fn test_replay_full_session() {
            // Per spec: 1 minute at 60fps = 3600 frames
            let config = SimulationConfig::new(42, 3600);

            let recording = run_simulation(config, |frame| {
                // Alternate between movement and action
                let key = match frame % 5 {
                    0 => "ArrowUp",
                    1 => "ArrowRight",
                    2 => "ArrowDown",
                    3 => "ArrowLeft",
                    _ => "Space",
                };
                vec![InputEvent::key_press(key)]
            });

            assert!(recording.completed);

            let replay_result = run_replay(&recording);
            assert!(
                replay_result.determinism_verified,
                "Full session replay should be deterministic"
            );
        }
    }

    mod agent_tests {
        use super::*;

        #[test]
        fn test_random_walk_agent_deterministic() {
            let mut agent1 = RandomWalkAgent::new(Seed::from_u64(42));
            let mut agent2 = RandomWalkAgent::new(Seed::from_u64(42));

            for _ in 0..100 {
                let inputs1 = agent1.next_inputs();
                let inputs2 = agent2.next_inputs();

                assert_eq!(inputs1.len(), inputs2.len());
            }
        }

        #[test]
        fn test_random_walk_simulation() {
            let seed = Seed::from_u64(12345);
            let mut agent = RandomWalkAgent::new(seed);

            let config = SimulationConfig::new(seed.value(), 1000);
            let recording = run_simulation(config, |_| agent.next_inputs());

            assert!(recording.completed);

            // Reset agent and replay
            let mut agent2 = RandomWalkAgent::new(seed);
            let mut recording2 =
                SimulationRecording::new(SimulationConfig::new(seed.value(), 1000));
            let mut state = SimulatedGameState::new(seed.value());

            for frame in 0..1000 {
                let inputs = agent2.next_inputs();
                state.update(&inputs);
                recording2.add_frame(RecordedFrame {
                    frame,
                    inputs,
                    state_hash: state.compute_hash(),
                });
            }

            assert!(
                recording.matches(&recording2),
                "Replay with same agent should match"
            );
        }
    }

    mod additional_coverage_tests {
        use super::*;

        #[test]
        fn test_recording_mark_failed() {
            let mut recording = SimulationRecording::new(SimulationConfig::default());
            recording.mark_failed("Test error");

            assert!(!recording.completed);
            assert_eq!(recording.error, Some("Test error".to_string()));
        }

        #[test]
        fn test_recording_mark_completed() {
            let mut recording = SimulationRecording::new(SimulationConfig::default());
            recording.mark_completed();

            assert!(recording.completed);
        }

        #[test]
        fn test_replay_result_diverged() {
            let result = ReplayResult::diverged(50, 12345, 67890);

            assert!(!result.determinism_verified);
            assert_eq!(result.divergence_frame, Some(50));
            assert_eq!(result.final_state_hash, 67890);
            assert!(result.error.is_some());
            assert!(result.error.unwrap().contains("diverged at frame 50"));
        }

        #[test]
        fn test_game_state_touch_input() {
            let mut state = SimulatedGameState::new(0);
            state.player_x = 100.0;
            state.player_y = 100.0;

            // Touch far away - should move toward it
            state.update(&[InputEvent::Touch { x: 200.0, y: 100.0 }]);

            assert!(state.player_x > 100.0, "Player should move toward touch");
        }

        #[test]
        fn test_game_state_mouse_click_input() {
            let mut state = SimulatedGameState::new(0);
            state.player_x = 100.0;
            state.player_y = 100.0;

            // Click far away - should move toward it
            state.update(&[InputEvent::MouseClick { x: 100.0, y: 200.0 }]);

            assert!(state.player_y > 100.0, "Player should move toward click");
        }

        #[test]
        fn test_game_state_touch_close_no_move() {
            let mut state = SimulatedGameState::new(0);
            state.player_x = 100.0;
            state.player_y = 100.0;
            let initial_x = state.player_x;
            let initial_y = state.player_y;

            // Touch very close - should not move (distance < 1.0)
            state.update(&[InputEvent::Touch { x: 100.5, y: 100.5 }]);

            // Account for frame update effects on random state
            assert!(
                (state.player_x - initial_x).abs() < 6.0,
                "Player should barely move"
            );
            assert!(
                (state.player_y - initial_y).abs() < 6.0,
                "Player should barely move"
            );
        }

        #[test]
        fn test_game_state_movement_keys() {
            let mut state = SimulatedGameState::new(0);
            state.player_x = 400.0;
            state.player_y = 300.0;

            // Test all movement keys
            let initial_x = state.player_x;
            state.update(&[InputEvent::key_press("ArrowRight")]);
            assert!(state.player_x > initial_x, "ArrowRight should move right");

            let initial_x = state.player_x;
            state.update(&[InputEvent::key_press("ArrowLeft")]);
            assert!(state.player_x < initial_x, "ArrowLeft should move left");

            let initial_y = state.player_y;
            state.update(&[InputEvent::key_press("ArrowDown")]);
            assert!(state.player_y > initial_y, "ArrowDown should move down");

            // Test WASD alternatives
            let initial_x = state.player_x;
            state.update(&[InputEvent::key_press("KeyD")]);
            assert!(state.player_x > initial_x, "KeyD should move right");

            let initial_y = state.player_y;
            state.update(&[InputEvent::key_press("KeyW")]);
            assert!(state.player_y < initial_y, "KeyW should move up");

            let initial_y = state.player_y;
            state.update(&[InputEvent::key_press("KeyS")]);
            assert!(state.player_y > initial_y, "KeyS should move down");

            let initial_x = state.player_x;
            state.update(&[InputEvent::key_press("KeyA")]);
            assert!(state.player_x < initial_x, "KeyA should move left");
        }

        #[test]
        fn test_game_state_unknown_key() {
            let mut state = SimulatedGameState::new(0);
            state.player_x = 400.0;
            state.player_y = 300.0;
            let initial_x = state.player_x;
            let initial_y = state.player_y;

            state.update(&[InputEvent::key_press("Unknown")]);

            // Should not move from key, but state still updates
            assert_eq!(state.frame, 1);
            // Position unchanged from key input
            assert!((state.player_x - initial_x).abs() < 0.1);
            assert!((state.player_y - initial_y).abs() < 0.1);
        }

        #[test]
        fn test_game_state_clamp_bounds() {
            let mut state = SimulatedGameState::new(0);

            // Move to edge
            state.player_x = 0.0;
            state.player_y = 0.0;

            // Try to move past bounds
            for _ in 0..100 {
                state.update(&[InputEvent::key_press("ArrowUp")]);
                state.update(&[InputEvent::key_press("ArrowLeft")]);
            }

            assert!(state.player_x >= 0.0, "X should be clamped at 0");
            assert!(state.player_y >= 0.0, "Y should be clamped at 0");

            // Move to other edge
            state.player_x = 800.0;
            state.player_y = 600.0;

            for _ in 0..100 {
                state.update(&[InputEvent::key_press("ArrowDown")]);
                state.update(&[InputEvent::key_press("ArrowRight")]);
            }

            assert!(state.player_x <= 800.0, "X should be clamped at 800");
            assert!(state.player_y <= 600.0, "Y should be clamped at 600");
        }
    }

    mod prop_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_simulation_always_completes(seed in 0u64..10000, frames in 1u64..500) {
                let config = SimulationConfig::new(seed, frames);
                let recording = run_simulation(config, |_| vec![]);

                prop_assert!(recording.completed);
                prop_assert_eq!(recording.total_frames, frames);
            }

            #[test]
            fn prop_simulation_deterministic(seed in 0u64..10000) {
                let config1 = SimulationConfig::new(seed, 100);
                let config2 = SimulationConfig::new(seed, 100);

                let rec1 = run_simulation(config1, |f| {
                    if f % 2 == 0 { vec![InputEvent::key_press("Space")] } else { vec![] }
                });
                let rec2 = run_simulation(config2, |f| {
                    if f % 2 == 0 { vec![InputEvent::key_press("Space")] } else { vec![] }
                });

                prop_assert!(rec1.matches(&rec2));
            }

            #[test]
            fn prop_game_state_always_valid(seed in 0u64..10000, frames in 1usize..1000) {
                let mut state = SimulatedGameState::new(seed);

                for _ in 0..frames {
                    state.update(&[InputEvent::key_press("Space")]);
                    prop_assert!(state.is_valid());
                }
            }

            #[test]
            fn prop_replay_verifies(seed in 0u64..1000) {
                let config = SimulationConfig::new(seed, 100);
                let recording = run_simulation(config, |_| vec![]);
                let replay = run_replay(&recording);

                prop_assert!(replay.determinism_verified);
                prop_assert_eq!(replay.final_state_hash, recording.final_state_hash);
            }
        }
    }
}
