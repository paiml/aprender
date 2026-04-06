//! Pong Simulation Demo - Deterministic Game Testing with PIXEL-001 v2.1
//!
//! Demonstrates using Probar's simulation framework to test
//! Pong game mechanics with deterministic replay and pixel-perfect
//! verification using PIXEL-001 v2.1 methodology.
//!
//! # Running
//!
//! ```bash
//! cargo run --example pong_simulation -p jugar-probar
//! ```
//!
//! # Features
//!
//! - Deterministic simulation of Pong physics
//! - Replay verification for regression testing
//! - Invariant checking (score bounds, paddle bounds)
//! - Random walk agent for fuzz testing
//! - **PIXEL-001 v2.1**: Pixel-perfect game element tracking

#![allow(
    clippy::uninlined_format_args,
    clippy::std_instead_of_core,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    dead_code
)]

use jugar_probar::pixel_coverage::{
    ConfidenceInterval, FalsifiabilityGate, FalsifiableHypothesis, OutputMode,
    PixelCoverageTracker, PixelRegion, ScoreBar,
};
use jugar_probar::{
    run_replay, run_simulation, Assertion, InputEvent, RandomWalkAgent, Seed, SimulationConfig,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn main() {
    println!("=== Probar Pong Simulation Demo (PIXEL-001 v2.1) ===\n");

    // Demo 1: Basic Pong simulation
    demo_pong_simulation();

    // Demo 2: Deterministic replay
    demo_deterministic_replay();

    // Demo 3: Fuzz testing with random agent
    demo_fuzz_testing();

    // Demo 4: Invariant checking
    demo_invariant_checking();

    // Demo 5: PIXEL-001 v2.1 - Pixel-Perfect Game Coverage
    demo_pixel_perfect_pong();

    println!("\n=== Pong Simulation Demo Complete (PIXEL-001 v2.1) ===");
}

/// Simulated Pong game state
#[derive(Debug, Clone)]
struct PongState {
    /// Ball X position
    ball_x: f32,
    /// Ball Y position
    ball_y: f32,
    /// Ball X velocity
    ball_vx: f32,
    /// Ball Y velocity
    ball_vy: f32,
    /// Paddle 1 Y position (left)
    paddle1_y: f32,
    /// Paddle 2 Y position (right)
    paddle2_y: f32,
    /// Player 1 score
    score1: u32,
    /// Player 2 score
    score2: u32,
    /// Current frame
    frame: u64,
}

impl PongState {
    const GAME_WIDTH: f32 = 800.0;
    const GAME_HEIGHT: f32 = 600.0;
    const PADDLE_HEIGHT: f32 = 100.0;
    const PADDLE_SPEED: f32 = 8.0;
    const BALL_SPEED: f32 = 6.0;
    const BALL_SIZE: f32 = 10.0;
    const PADDLE_MARGIN: f32 = 40.0;

    fn new() -> Self {
        Self {
            ball_x: Self::GAME_WIDTH / 2.0,
            ball_y: Self::GAME_HEIGHT / 2.0,
            ball_vx: Self::BALL_SPEED,
            ball_vy: Self::BALL_SPEED * 0.5,
            paddle1_y: Self::GAME_HEIGHT / 2.0,
            paddle2_y: Self::GAME_HEIGHT / 2.0,
            score1: 0,
            score2: 0,
            frame: 0,
        }
    }

    fn update(&mut self, inputs: &[InputEvent]) {
        self.frame += 1;

        // Process inputs
        for input in inputs {
            if let InputEvent::KeyPress { key } = input {
                match key.as_str() {
                    "KeyW" | "ArrowUp" => {
                        self.paddle1_y = (self.paddle1_y - Self::PADDLE_SPEED).clamp(
                            Self::PADDLE_HEIGHT / 2.0,
                            Self::GAME_HEIGHT - Self::PADDLE_HEIGHT / 2.0,
                        );
                    }
                    "KeyS" | "ArrowDown" => {
                        self.paddle1_y = (self.paddle1_y + Self::PADDLE_SPEED).clamp(
                            Self::PADDLE_HEIGHT / 2.0,
                            Self::GAME_HEIGHT - Self::PADDLE_HEIGHT / 2.0,
                        );
                    }
                    "ArrowLeft" => {
                        self.paddle2_y = (self.paddle2_y - Self::PADDLE_SPEED).clamp(
                            Self::PADDLE_HEIGHT / 2.0,
                            Self::GAME_HEIGHT - Self::PADDLE_HEIGHT / 2.0,
                        );
                    }
                    "ArrowRight" => {
                        self.paddle2_y = (self.paddle2_y + Self::PADDLE_SPEED).clamp(
                            Self::PADDLE_HEIGHT / 2.0,
                            Self::GAME_HEIGHT - Self::PADDLE_HEIGHT / 2.0,
                        );
                    }
                    _ => {}
                }
            }
        }

        // Update ball position
        self.ball_x += self.ball_vx;
        self.ball_y += self.ball_vy;

        // Bounce off top/bottom
        if self.ball_y <= Self::BALL_SIZE / 2.0
            || self.ball_y >= Self::GAME_HEIGHT - Self::BALL_SIZE / 2.0
        {
            self.ball_vy = -self.ball_vy;
            self.ball_y = self.ball_y.clamp(
                Self::BALL_SIZE / 2.0,
                Self::GAME_HEIGHT - Self::BALL_SIZE / 2.0,
            );
        }

        // Check paddle 1 collision (left)
        if self.ball_x <= Self::PADDLE_MARGIN + Self::BALL_SIZE / 2.0
            && self.ball_vx < 0.0
            && (self.ball_y - self.paddle1_y).abs() < Self::PADDLE_HEIGHT / 2.0
        {
            self.ball_vx = -self.ball_vx;
            // Add spin based on hit position
            let offset = (self.ball_y - self.paddle1_y) / (Self::PADDLE_HEIGHT / 2.0);
            self.ball_vy += offset * 2.0;
        }

        // Check paddle 2 collision (right)
        if self.ball_x >= Self::GAME_WIDTH - Self::PADDLE_MARGIN - Self::BALL_SIZE / 2.0
            && self.ball_vx > 0.0
            && (self.ball_y - self.paddle2_y).abs() < Self::PADDLE_HEIGHT / 2.0
        {
            self.ball_vx = -self.ball_vx;
            let offset = (self.ball_y - self.paddle2_y) / (Self::PADDLE_HEIGHT / 2.0);
            self.ball_vy += offset * 2.0;
        }

        // Score detection
        if self.ball_x < 0.0 {
            self.score2 += 1;
            self.reset_ball(1.0);
        } else if self.ball_x > Self::GAME_WIDTH {
            self.score1 += 1;
            self.reset_ball(-1.0);
        }
    }

    fn reset_ball(&mut self, direction: f32) {
        self.ball_x = Self::GAME_WIDTH / 2.0;
        self.ball_y = Self::GAME_HEIGHT / 2.0;
        self.ball_vx = Self::BALL_SPEED * direction;
        self.ball_vy = Self::BALL_SPEED * 0.3;
    }

    fn compute_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.ball_x.to_bits().hash(&mut hasher);
        self.ball_y.to_bits().hash(&mut hasher);
        self.ball_vx.to_bits().hash(&mut hasher);
        self.ball_vy.to_bits().hash(&mut hasher);
        self.paddle1_y.to_bits().hash(&mut hasher);
        self.paddle2_y.to_bits().hash(&mut hasher);
        self.score1.hash(&mut hasher);
        self.score2.hash(&mut hasher);
        self.frame.hash(&mut hasher);
        hasher.finish()
    }

    fn is_valid(&self) -> bool {
        // Check invariants
        self.score1 <= 100
            && self.score2 <= 100
            && self.paddle1_y >= 0.0
            && self.paddle1_y <= Self::GAME_HEIGHT
            && self.paddle2_y >= 0.0
            && self.paddle2_y <= Self::GAME_HEIGHT
    }
}

fn demo_pong_simulation() {
    println!("--- Demo 1: Pong Simulation ---\n");

    let mut state = PongState::new();

    println!("Initial state:");
    println!("  Ball: ({:.1}, {:.1})", state.ball_x, state.ball_y);
    println!(
        "  Paddles: P1={:.1}, P2={:.1}",
        state.paddle1_y, state.paddle2_y
    );
    println!("  Score: {} - {}", state.score1, state.score2);

    // Simulate 300 frames (5 seconds at 60fps)
    println!("\nSimulating 300 frames...");

    for frame in 0..300 {
        // Player 1 moves up, Player 2 tracks ball
        let mut inputs = vec![];

        if frame % 5 == 0 {
            inputs.push(InputEvent::key_press("KeyW"));
        }

        // AI: paddle 2 follows ball
        if state.ball_y < state.paddle2_y - 10.0 {
            inputs.push(InputEvent::key_press("ArrowLeft")); // Move up
        } else if state.ball_y > state.paddle2_y + 10.0 {
            inputs.push(InputEvent::key_press("ArrowRight")); // Move down
        }

        state.update(&inputs);
    }

    println!("\nFinal state after 300 frames:");
    println!("  Ball: ({:.1}, {:.1})", state.ball_x, state.ball_y);
    println!(
        "  Paddles: P1={:.1}, P2={:.1}",
        state.paddle1_y, state.paddle2_y
    );
    println!("  Score: {} - {}", state.score1, state.score2);
    println!("  State valid: {}", state.is_valid());
    println!();
}

fn demo_deterministic_replay() {
    println!("--- Demo 2: Deterministic Replay ---\n");

    // Use Probar's built-in simulation for determinism verification
    let config = SimulationConfig::new(42, 500);

    println!("Recording simulation (seed=42, frames=500)...");

    let recording = run_simulation(config, |frame| {
        // Alternate movement pattern
        let key = match frame % 4 {
            0 => "ArrowUp",
            1 => "ArrowRight",
            2 => "ArrowDown",
            _ => "Space",
        };
        vec![InputEvent::key_press(key)]
    });

    println!("  Completed: {}", recording.completed);
    println!("  Total frames: {}", recording.total_frames);
    println!("  Final hash: {}", recording.final_state_hash);

    // Replay and verify determinism
    println!("\nReplaying simulation...");
    let replay_result = run_replay(&recording);

    println!(
        "  Determinism verified: {}",
        replay_result.determinism_verified
    );
    println!("  Frames replayed: {}", replay_result.frames_replayed);
    println!(
        "  Hashes match: {}",
        replay_result.final_state_hash == recording.final_state_hash
    );

    let assertion = Assertion::equals(&replay_result.final_state_hash, &recording.final_state_hash);
    println!("  Assertion passed: {}", assertion.passed);
    println!();
}

fn demo_fuzz_testing() {
    println!("--- Demo 3: Fuzz Testing with Random Agent ---\n");

    let seed = Seed::from_u64(12345);
    let mut agent = RandomWalkAgent::new(seed);

    println!("Running fuzz test (seed=12345, frames=1000)...");

    let config = SimulationConfig::new(seed.value(), 1000);
    let recording = run_simulation(config, |_| agent.next_inputs());

    println!("  Completed without crash: {}", recording.completed);
    println!("  Total frames: {}", recording.total_frames);

    if let Some(err) = &recording.error {
        println!("  Error: {}", err);
    }

    // Run with different seed
    let seed2 = Seed::from_u64(67890);
    let mut agent2 = RandomWalkAgent::new(seed2);
    let config2 = SimulationConfig::new(seed2.value(), 1000);
    let recording2 = run_simulation(config2, |_| agent2.next_inputs());

    println!("\nSecond run (seed=67890):");
    println!("  Completed: {}", recording2.completed);
    println!("  Same result as first: {}", recording.matches(&recording2));
    println!();
}

fn demo_invariant_checking() {
    println!("--- Demo 4: Invariant Checking ---\n");

    let mut state = PongState::new();
    let mut violations = 0;
    let frames = 1000;

    println!("Running {} frames with invariant checks...", frames);

    for frame in 0..frames {
        // Random-ish input pattern
        let inputs = if frame % 3 == 0 {
            vec![InputEvent::key_press("KeyW")]
        } else if frame % 7 == 0 {
            vec![InputEvent::key_press("KeyS")]
        } else {
            vec![]
        };

        state.update(&inputs);

        if !state.is_valid() {
            violations += 1;
            println!(
                "  Violation at frame {}: score=({}, {}), paddles=({:.1}, {:.1})",
                frame, state.score1, state.score2, state.paddle1_y, state.paddle2_y
            );
        }
    }

    println!("\nInvariant check results:");
    println!("  Total frames: {}", frames);
    println!("  Violations: {}", violations);
    println!("  All invariants held: {}", violations == 0);

    // Check specific assertions
    let score_valid = Assertion::in_range(f64::from(state.score1), 0.0, 100.0);
    let paddle_valid = Assertion::in_range(
        f64::from(state.paddle1_y),
        0.0,
        f64::from(PongState::GAME_HEIGHT),
    );

    println!("\nFinal state assertions:");
    println!("  Score in range [0, 100]: {}", score_valid.passed);
    println!("  Paddle in game bounds: {}", paddle_valid.passed);
    println!();
}

fn demo_pixel_perfect_pong() {
    println!("--- Demo 5: PIXEL-001 v2.1 Pixel-Perfect Coverage ---\n");

    // Create pixel tracker for Pong game screen (800x600, 8x6 grid)
    let mut pixels = PixelCoverageTracker::builder()
        .resolution(800, 600)
        .grid_size(8, 6)
        .threshold(0.90)
        .build();

    println!("Pixel Tracker: 800x600 (8x6 grid, 90% threshold)\n");

    // Simulate a game with pixel tracking
    let mut state = PongState::new();
    println!("Tracking pixel coverage over 500 frames...");

    for frame in 0..500 {
        // Random-ish input pattern
        let inputs = if frame % 3 == 0 {
            vec![InputEvent::key_press("KeyW")]
        } else if frame % 7 == 0 {
            vec![InputEvent::key_press("KeyS")]
        } else if frame % 11 == 0 {
            vec![InputEvent::key_press("ArrowLeft")]
        } else {
            vec![]
        };

        state.update(&inputs);

        // Track ball position as pixel region (ball is 10x10)
        let ball_x = state.ball_x.clamp(0.0, 790.0) as u32;
        let ball_y = state.ball_y.clamp(0.0, 590.0) as u32;
        pixels.record_region(PixelRegion::new(ball_x, ball_y, 10, 10));

        // Track paddle 1 (left paddle)
        let paddle1_y = (state.paddle1_y - PongState::PADDLE_HEIGHT / 2.0).clamp(0.0, 500.0) as u32;
        pixels.record_region(PixelRegion::new(
            PongState::PADDLE_MARGIN as u32 - 10,
            paddle1_y,
            10,
            PongState::PADDLE_HEIGHT as u32,
        ));

        // Track paddle 2 (right paddle)
        let paddle2_y = (state.paddle2_y - PongState::PADDLE_HEIGHT / 2.0).clamp(0.0, 500.0) as u32;
        pixels.record_region(PixelRegion::new(
            (PongState::GAME_WIDTH - PongState::PADDLE_MARGIN) as u32,
            paddle2_y,
            10,
            PongState::PADDLE_HEIGHT as u32,
        ));
    }

    // Generate report
    let report = pixels.generate_report();
    println!("\nPixel Coverage Report:");
    println!("  Coverage: {:.1}%", report.overall_coverage * 100.0);
    println!("  Cells: {}/{}", report.covered_cells, report.total_cells);
    println!("  Meets Threshold: {}", report.meets_threshold);

    // Popperian Falsification
    println!("\nPopperian Falsification:");
    let gate = FalsifiabilityGate::new(15.0);

    let h1 = FalsifiableHypothesis::coverage_threshold("H0-PONG-PIX", 0.85)
        .evaluate(report.overall_coverage);
    let h2 = FalsifiableHypothesis::max_gap_size("H0-PONG-GAP", 10.0)
        .evaluate(report.uncovered_regions.len() as f32);

    println!(
        "  H0-PONG-PIX (≥85%): {}",
        if h1.falsified {
            "FALSIFIED"
        } else {
            "NOT FALSIFIED"
        }
    );
    println!(
        "  H0-PONG-GAP (≤10 gaps): {}",
        if h2.falsified {
            "FALSIFIED"
        } else {
            "NOT FALSIFIED"
        }
    );

    let gate_result = gate.evaluate(&h1);
    println!(
        "  FalsifiabilityGate: {}",
        if gate_result.is_passed() {
            "PASSED"
        } else {
            "FAILED"
        }
    );

    // Wilson Score Confidence Interval
    println!("\nWilson Score CI (95%):");
    let ci = ConfidenceInterval::wilson_score(report.covered_cells, report.total_cells, 0.95);
    println!("  [{:.1}%, {:.1}%]", ci.lower * 100.0, ci.upper * 100.0);

    // Score Bar
    println!("\nVisual Score:");
    let mode = OutputMode::from_env();
    let bar = ScoreBar::new("Pixel", report.overall_coverage, 0.90);
    println!("  {}", bar.render(mode));

    // Terminal Heatmap
    println!("\nPixel Heatmap (ball/paddle positions):");
    let heatmap = pixels.terminal_heatmap();
    for line in heatmap.render().lines() {
        println!("  {}", line);
    }

    // Final assertions
    let coverage_assertion = Assertion::in_range(report.overall_coverage as f64, 0.5, 1.0);
    println!(
        "\nAssertion (coverage in [50%, 100%]): {}",
        coverage_assertion.passed
    );
    println!();
}
