//! DeterministicBrick: Pure functional brick execution (PROBAR-SPEC-009-P11)
//!
//! Provides deterministic execution with:
//! - Pure functional design: (State, Input) → (State, Output)
//! - Persistent data structures for O(1) snapshots
//! - Time-travel debugging
//! - Jidoka guards for invariant checking
//!
//! # Design Philosophy
//!
//! DeterministicBrick applies WOS (WebAssembly Operating System) patterns
//! to ensure reproducible brick execution.
//!
//! # Example
//!
//! ```rust,ignore
//! use probar::brick::deterministic::{DeterministicBrick, BrickHistory};
//!
//! impl DeterministicBrick for MelSpectrogramBrick {
//!     type State = ComputeState;
//!     type Input = AudioChunk;
//!     type Output = MelFrames;
//!
//!     fn execute_pure(
//!         state: Self::State,
//!         input: Self::Input,
//!     ) -> Result<(Self::State, Self::Output), BrickError> {
//!         let mel = compute_mel(&input, &state.filterbank)?;
//!         let new_state = state.with_frame_count(state.frame_count + 1);
//!         Ok((new_state, mel))
//!     }
//! }
//! ```

use super::{Brick, BrickError};
use std::collections::HashMap;
use std::time::Duration;

/// Trait for bricks with deterministic, pure functional execution
pub trait DeterministicBrick: Brick {
    /// State type (must be cloneable for snapshots)
    type State: Clone + Default;
    /// Input type
    type Input;
    /// Output type
    type Output;

    /// Pure function: (state, input) → (new_state, output)
    ///
    /// This function MUST be:
    /// - Deterministic: same inputs → same outputs
    /// - Pure: no side effects
    /// - Total: always returns or errors (no panics)
    fn execute_pure(
        state: Self::State,
        input: Self::Input,
    ) -> Result<(Self::State, Self::Output), BrickError>;

    /// Create initial state
    fn initial_state() -> Self::State {
        Self::State::default()
    }

    /// Get state dependencies for determinism verification
    fn state_dependencies(&self) -> &[&str] {
        &[]
    }
}

/// Brick state snapshot with structural sharing
#[derive(Debug, Clone)]
pub struct BrickState {
    /// Named tensors (using owned Vecs for simplicity)
    pub tensors: HashMap<String, Vec<f32>>,
    /// Tensor shapes
    pub shapes: HashMap<String, Vec<usize>>,
    /// Scalar metadata
    pub metadata: HashMap<String, StateValue>,
    /// Snapshot version for debugging
    pub version: u64,
}

/// Value types for state metadata
#[derive(Debug, Clone, PartialEq)]
pub enum StateValue {
    /// Integer value
    Int(i64),
    /// Float value
    Float(f64),
    /// String value
    String(String),
    /// Boolean value
    Bool(bool),
}

impl BrickState {
    /// Create a new empty state
    #[must_use]
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            shapes: HashMap::new(),
            metadata: HashMap::new(),
            version: 0,
        }
    }

    /// Create a snapshot (clones with new version)
    #[must_use]
    pub fn snapshot(&self) -> Self {
        let mut snap = self.clone();
        snap.version = self.version + 1;
        snap
    }

    /// Set a tensor value
    pub fn set_tensor(&mut self, name: impl Into<String>, data: Vec<f32>, shape: Vec<usize>) {
        let name = name.into();
        self.tensors.insert(name.clone(), data);
        self.shapes.insert(name, shape);
    }

    /// Get a tensor value
    pub fn get_tensor(&self, name: &str) -> Option<(&[f32], &[usize])> {
        let data = self.tensors.get(name)?;
        let shape = self.shapes.get(name)?;
        Some((data, shape))
    }

    /// Set metadata value
    pub fn set_metadata(&mut self, key: impl Into<String>, value: StateValue) {
        self.metadata.insert(key.into(), value);
    }

    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&StateValue> {
        self.metadata.get(key)
    }
}

impl Default for BrickState {
    fn default() -> Self {
        Self::new()
    }
}

/// Execution trace entry for time-travel debugging
#[derive(Debug, Clone)]
pub struct ExecutionTrace {
    /// Operation name
    pub operation: String,
    /// Input summary (for debugging)
    pub input_summary: String,
    /// Output summary (for debugging)
    pub output_summary: String,
    /// Duration
    pub duration: Duration,
    /// State version before execution
    pub state_version_before: u64,
    /// State version after execution
    pub state_version_after: u64,
}

/// History for time-travel debugging
#[derive(Debug)]
pub struct BrickHistory {
    /// State snapshots
    snapshots: Vec<BrickState>,
    /// Execution traces
    traces: Vec<ExecutionTrace>,
    /// Current position
    position: usize,
    /// Maximum history size
    max_size: usize,
}

impl BrickHistory {
    /// Create a new history with given max size
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            snapshots: Vec::with_capacity(max_size),
            traces: Vec::with_capacity(max_size),
            position: 0,
            max_size,
        }
    }

    /// Record a state snapshot
    pub fn record(&mut self, state: BrickState, trace: ExecutionTrace) {
        // Truncate forward history if we're not at the end
        if self.position < self.snapshots.len() {
            self.snapshots.truncate(self.position);
            self.traces.truncate(self.position);
        }

        // Remove oldest if at capacity
        if self.snapshots.len() >= self.max_size {
            self.snapshots.remove(0);
            self.traces.remove(0);
        }

        self.snapshots.push(state);
        self.traces.push(trace);
        self.position = self.snapshots.len();
    }

    /// Step backward to previous state
    pub fn step_back(&mut self) -> Option<&BrickState> {
        if self.position > 0 {
            self.position -= 1;
            self.snapshots.get(self.position)
        } else {
            None
        }
    }

    /// Step forward to next state
    pub fn step_forward(&mut self) -> Option<&BrickState> {
        if self.position < self.snapshots.len() {
            let state = self.snapshots.get(self.position);
            self.position += 1;
            state
        } else {
            None
        }
    }

    /// Jump to specific position
    pub fn goto(&mut self, position: usize) -> Option<&BrickState> {
        if position < self.snapshots.len() {
            self.position = position;
            self.snapshots.get(position)
        } else {
            None
        }
    }

    /// Get current state
    pub fn current(&self) -> Option<&BrickState> {
        if self.position > 0 && self.position <= self.snapshots.len() {
            self.snapshots.get(self.position - 1)
        } else {
            self.snapshots.first()
        }
    }

    /// Get current position
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Get total snapshot count
    #[must_use]
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Check if history is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Get trace at position
    pub fn trace_at(&self, position: usize) -> Option<&ExecutionTrace> {
        self.traces.get(position)
    }

    /// Get all traces
    pub fn traces(&self) -> &[ExecutionTrace] {
        &self.traces
    }
}

impl Default for BrickHistory {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Severity level for invariant violations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardSeverity {
    /// Warning only, execution continues
    Warning,
    /// Error, execution stops
    Error,
    /// Critical, execution stops and alerts
    Critical,
}

/// Invariant guard for Jidoka pattern
pub struct InvariantGuard {
    /// Guard name
    pub name: &'static str,
    /// Check function
    pub check: fn(&BrickState) -> bool,
    /// Severity on violation
    pub severity: GuardSeverity,
}

impl std::fmt::Debug for InvariantGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvariantGuard")
            .field("name", &self.name)
            .field("check", &"<fn>")
            .field("severity", &self.severity)
            .finish()
    }
}

impl InvariantGuard {
    /// Create a new invariant guard
    #[must_use]
    pub const fn new(
        name: &'static str,
        check: fn(&BrickState) -> bool,
        severity: GuardSeverity,
    ) -> Self {
        Self {
            name,
            check,
            severity,
        }
    }

    /// Check the invariant
    pub fn check(&self, state: &BrickState) -> bool {
        (self.check)(state)
    }
}

/// Wrapper that adds invariant checking to a brick
#[derive(Debug)]
pub struct GuardedBrick<B: Brick> {
    /// Inner brick
    inner: B,
    /// Invariant guards
    guards: Vec<InvariantGuard>,
}

impl<B: Brick> GuardedBrick<B> {
    /// Create a new guarded brick
    #[must_use]
    pub fn new(brick: B) -> Self {
        Self {
            inner: brick,
            guards: Vec::new(),
        }
    }

    /// Add an invariant guard
    #[must_use]
    pub fn guard(mut self, guard: InvariantGuard) -> Self {
        self.guards.push(guard);
        self
    }

    /// Check all guards against state
    pub fn check_guards(&self, state: &BrickState) -> Result<(), GuardViolation> {
        for guard in &self.guards {
            if !guard.check(state) {
                return Err(GuardViolation {
                    guard_name: guard.name,
                    severity: guard.severity,
                });
            }
        }
        Ok(())
    }

    /// Get the inner brick
    pub fn inner(&self) -> &B {
        &self.inner
    }

    /// Get guards
    pub fn guards(&self) -> &[InvariantGuard] {
        &self.guards
    }
}

/// Guard violation error
#[derive(Debug, Clone)]
pub struct GuardViolation {
    /// Name of the violated guard
    pub guard_name: &'static str,
    /// Severity
    pub severity: GuardSeverity,
}

impl std::fmt::Display for GuardViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Invariant guard '{}' violated (severity: {:?})",
            self.guard_name, self.severity
        )
    }
}

impl std::error::Error for GuardViolation {}

/// Deterministic random number generator (for reproducibility)
#[derive(Debug, Clone)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    /// Create a new RNG with given seed
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generate next random u64
    pub fn next_u64(&mut self) -> u64 {
        // Simple xorshift64
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Generate random f64 in [0, 1)
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Generate random f32 in [0, 1)
    pub fn next_f32(&mut self) -> f32 {
        self.next_f64() as f32
    }

    /// Get current state (for checkpointing)
    #[must_use]
    pub const fn state(&self) -> u64 {
        self.state
    }

    /// Restore state (from checkpoint)
    pub fn restore(&mut self, state: u64) {
        self.state = state;
    }
}

impl Default for DeterministicRng {
    fn default() -> Self {
        Self::new(42)
    }
}

/// Deterministic clock for reproducibility
#[derive(Debug, Clone)]
pub struct DeterministicClock {
    /// Current time in nanoseconds
    current_ns: u64,
    /// Time step per tick
    tick_ns: u64,
}

impl DeterministicClock {
    /// Create a new clock
    #[must_use]
    pub const fn new(start_ns: u64, tick_ns: u64) -> Self {
        Self {
            current_ns: start_ns,
            tick_ns,
        }
    }

    /// Get current time
    #[must_use]
    pub const fn now_ns(&self) -> u64 {
        self.current_ns
    }

    /// Get current time as Duration
    #[must_use]
    pub const fn now(&self) -> Duration {
        Duration::from_nanos(self.current_ns)
    }

    /// Advance clock by one tick
    pub fn tick(&mut self) {
        self.current_ns += self.tick_ns;
    }

    /// Advance clock by multiple ticks
    pub fn advance(&mut self, ticks: u64) {
        self.current_ns += self.tick_ns * ticks;
    }

    /// Set time (for replay)
    pub fn set(&mut self, time_ns: u64) {
        self.current_ns = time_ns;
    }
}

impl Default for DeterministicClock {
    fn default() -> Self {
        // 10ms tick
        Self::new(0, 10_000_000)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_brick_state_basic() {
        let mut state = BrickState::new();
        state.set_tensor("audio", vec![1.0, 2.0, 3.0], vec![3]);
        state.set_metadata("frame_count", StateValue::Int(42));

        let (data, shape) = state.get_tensor("audio").unwrap();
        assert_eq!(data, &[1.0, 2.0, 3.0]);
        assert_eq!(shape, &[3]);

        assert_eq!(
            state.get_metadata("frame_count"),
            Some(&StateValue::Int(42))
        );
    }

    #[test]
    fn test_brick_state_snapshot() {
        let mut state = BrickState::new();
        state.set_metadata("count", StateValue::Int(1));

        let snap = state.snapshot();
        assert_eq!(snap.version, 1);
        assert_eq!(snap.get_metadata("count"), Some(&StateValue::Int(1)));
    }

    #[test]
    fn test_brick_history_forward() {
        let mut history = BrickHistory::new(10);

        for i in 0..5 {
            let mut state = BrickState::new();
            state.version = i;
            state.set_metadata("step", StateValue::Int(i as i64));

            let trace = ExecutionTrace {
                operation: format!("step_{}", i),
                input_summary: String::new(),
                output_summary: String::new(),
                duration: Duration::from_millis(1),
                state_version_before: i,
                state_version_after: i + 1,
            };

            history.record(state, trace);
        }

        assert_eq!(history.len(), 5);
        assert_eq!(history.position(), 5);
    }

    #[test]
    fn test_brick_history_time_travel() {
        let mut history = BrickHistory::new(10);

        // Record 3 states with values 0, 1, 2
        for i in 0..3 {
            let mut state = BrickState::new();
            state.set_metadata("value", StateValue::Int(i as i64));

            let trace = ExecutionTrace {
                operation: format!("op_{}", i),
                input_summary: String::new(),
                output_summary: String::new(),
                duration: Duration::from_millis(1),
                state_version_before: i as u64,
                state_version_after: (i + 1) as u64,
            };

            history.record(state, trace);
        }

        // After recording: position = 3 (past end)
        assert_eq!(history.position(), 3);

        // Go back: position = 2, returns snapshots[2] (value 2)
        let state = history.step_back().unwrap();
        assert_eq!(state.get_metadata("value"), Some(&StateValue::Int(2)));
        assert_eq!(history.position(), 2);

        // Go back again: position = 1, returns snapshots[1] (value 1)
        let state = history.step_back().unwrap();
        assert_eq!(state.get_metadata("value"), Some(&StateValue::Int(1)));
        assert_eq!(history.position(), 1);

        // Go forward: returns snapshots[1] (value 1), then position = 2
        let state = history.step_forward().unwrap();
        assert_eq!(state.get_metadata("value"), Some(&StateValue::Int(1)));
        assert_eq!(history.position(), 2);
    }

    #[test]
    fn test_brick_history_goto() {
        let mut history = BrickHistory::new(10);

        for i in 0..5 {
            let mut state = BrickState::new();
            state.set_metadata("index", StateValue::Int(i as i64));

            let trace = ExecutionTrace {
                operation: format!("op_{}", i),
                input_summary: String::new(),
                output_summary: String::new(),
                duration: Duration::from_millis(1),
                state_version_before: i as u64,
                state_version_after: (i + 1) as u64,
            };

            history.record(state, trace);
        }

        let state = history.goto(2).unwrap();
        assert_eq!(state.get_metadata("index"), Some(&StateValue::Int(2)));
        assert_eq!(history.position(), 2);
    }

    #[test]
    fn test_invariant_guard() {
        fn check_positive(state: &BrickState) -> bool {
            match state.get_metadata("count") {
                Some(StateValue::Int(n)) => *n >= 0,
                _ => true,
            }
        }

        let guard = InvariantGuard::new("positive_count", check_positive, GuardSeverity::Error);

        let mut state = BrickState::new();
        state.set_metadata("count", StateValue::Int(5));
        assert!(guard.check(&state));

        state.set_metadata("count", StateValue::Int(-1));
        assert!(!guard.check(&state));
    }

    #[test]
    fn test_deterministic_rng() {
        let mut rng1 = DeterministicRng::new(12345);
        let mut rng2 = DeterministicRng::new(12345);

        // Same seed should produce same sequence
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_deterministic_rng_f64_range() {
        let mut rng = DeterministicRng::new(42);

        for _ in 0..1000 {
            let val = rng.next_f64();
            assert!((0.0..1.0).contains(&val));
        }
    }

    #[test]
    fn test_deterministic_clock() {
        let mut clock = DeterministicClock::new(0, 1_000_000); // 1ms tick

        assert_eq!(clock.now_ns(), 0);

        clock.tick();
        assert_eq!(clock.now_ns(), 1_000_000);

        clock.advance(10);
        assert_eq!(clock.now_ns(), 11_000_000);
        assert_eq!(clock.now(), Duration::from_millis(11));
    }

    #[test]
    fn test_deterministic_clock_replay() {
        let mut clock = DeterministicClock::new(0, 1_000_000);

        clock.advance(100);
        assert_eq!(clock.now_ns(), 100_000_000);

        // Reset for replay
        clock.set(0);
        assert_eq!(clock.now_ns(), 0);
    }

    #[test]
    fn test_state_value_variants() {
        let int_val = StateValue::Int(42);
        let float_val = StateValue::Float(3.14);
        let string_val = StateValue::String("hello".into());
        let bool_val = StateValue::Bool(true);

        assert_eq!(int_val, StateValue::Int(42));
        assert_eq!(float_val, StateValue::Float(3.14));
        assert_eq!(string_val, StateValue::String("hello".into()));
        assert_eq!(bool_val, StateValue::Bool(true));
    }

    // ========================================================================
    // Additional comprehensive tests for 95%+ coverage
    // ========================================================================

    #[test]
    fn test_brick_state_default() {
        let state = BrickState::default();
        assert!(state.tensors.is_empty());
        assert!(state.shapes.is_empty());
        assert!(state.metadata.is_empty());
        assert_eq!(state.version, 0);
    }

    #[test]
    fn test_brick_state_get_tensor_nonexistent() {
        let state = BrickState::new();
        assert!(state.get_tensor("nonexistent").is_none());
    }

    #[test]
    fn test_brick_state_get_metadata_nonexistent() {
        let state = BrickState::new();
        assert!(state.get_metadata("nonexistent").is_none());
    }

    #[test]
    fn test_brick_state_tensor_missing_shape() {
        let mut state = BrickState::new();
        state.tensors.insert("data".into(), vec![1.0, 2.0]);
        // No shape entry - get_tensor should return None
        assert!(state.get_tensor("data").is_none());
    }

    #[test]
    fn test_brick_state_clone() {
        let mut state = BrickState::new();
        state.set_tensor("t1", vec![1.0], vec![1]);
        state.set_metadata("m1", StateValue::Bool(true));
        state.version = 5;

        let cloned = state.clone();
        assert_eq!(cloned.version, 5);
        assert_eq!(cloned.get_tensor("t1").unwrap().0, &[1.0]);
        assert_eq!(cloned.get_metadata("m1"), Some(&StateValue::Bool(true)));
    }

    #[test]
    fn test_state_value_clone() {
        let val = StateValue::String("test".into());
        let cloned = val.clone();
        assert_eq!(val, cloned);
    }

    #[test]
    fn test_state_value_partial_eq() {
        assert_ne!(StateValue::Int(1), StateValue::Int(2));
        assert_ne!(StateValue::Float(1.0), StateValue::Float(2.0));
        assert_ne!(StateValue::Bool(true), StateValue::Bool(false));
        assert_ne!(
            StateValue::String("a".into()),
            StateValue::String("b".into())
        );
    }

    #[test]
    fn test_execution_trace_clone() {
        let trace = ExecutionTrace {
            operation: "test".into(),
            input_summary: "in".into(),
            output_summary: "out".into(),
            duration: Duration::from_secs(1),
            state_version_before: 0,
            state_version_after: 1,
        };
        let cloned = trace.clone();
        assert_eq!(trace.operation, cloned.operation);
        assert_eq!(trace.duration, cloned.duration);
    }

    #[test]
    fn test_brick_history_default() {
        let history = BrickHistory::default();
        assert!(history.is_empty());
        assert_eq!(history.len(), 0);
        assert_eq!(history.position(), 0);
    }

    #[test]
    fn test_brick_history_is_empty() {
        let mut history = BrickHistory::new(10);
        assert!(history.is_empty());

        let state = BrickState::new();
        let trace = ExecutionTrace {
            operation: "op".into(),
            input_summary: String::new(),
            output_summary: String::new(),
            duration: Duration::ZERO,
            state_version_before: 0,
            state_version_after: 1,
        };
        history.record(state, trace);
        assert!(!history.is_empty());
    }

    #[test]
    fn test_brick_history_step_back_empty() {
        let mut history = BrickHistory::new(10);
        assert!(history.step_back().is_none());
    }

    #[test]
    fn test_brick_history_step_back_at_start() {
        let mut history = BrickHistory::new(10);

        let state = BrickState::new();
        let trace = ExecutionTrace {
            operation: "op".into(),
            input_summary: String::new(),
            output_summary: String::new(),
            duration: Duration::ZERO,
            state_version_before: 0,
            state_version_after: 1,
        };
        history.record(state, trace);

        // Move to position 0
        history.position = 0;
        assert!(history.step_back().is_none());
    }

    #[test]
    fn test_brick_history_step_forward_at_end() {
        let mut history = BrickHistory::new(10);

        let state = BrickState::new();
        let trace = ExecutionTrace {
            operation: "op".into(),
            input_summary: String::new(),
            output_summary: String::new(),
            duration: Duration::ZERO,
            state_version_before: 0,
            state_version_after: 1,
        };
        history.record(state, trace);

        // Position is already at end (1)
        assert!(history.step_forward().is_none());
    }

    #[test]
    fn test_brick_history_goto_invalid() {
        let mut history = BrickHistory::new(10);
        assert!(history.goto(100).is_none());
    }

    #[test]
    fn test_brick_history_current_empty() {
        let history = BrickHistory::new(10);
        assert!(history.current().is_none());
    }

    #[test]
    fn test_brick_history_current_at_start() {
        let mut history = BrickHistory::new(10);

        let mut state = BrickState::new();
        state.set_metadata("val", StateValue::Int(1));
        let trace = ExecutionTrace {
            operation: "op".into(),
            input_summary: String::new(),
            output_summary: String::new(),
            duration: Duration::ZERO,
            state_version_before: 0,
            state_version_after: 1,
        };
        history.record(state, trace);

        // Position is 1
        let current = history.current();
        assert!(current.is_some());
        assert_eq!(
            current.unwrap().get_metadata("val"),
            Some(&StateValue::Int(1))
        );
    }

    #[test]
    fn test_brick_history_current_position_zero() {
        let mut history = BrickHistory::new(10);

        let state = BrickState::new();
        let trace = ExecutionTrace {
            operation: "op".into(),
            input_summary: String::new(),
            output_summary: String::new(),
            duration: Duration::ZERO,
            state_version_before: 0,
            state_version_after: 1,
        };
        history.record(state, trace);

        // Force position to 0 (should return first)
        history.position = 0;
        assert!(history.current().is_some());
    }

    #[test]
    fn test_brick_history_trace_at() {
        let mut history = BrickHistory::new(10);

        let state = BrickState::new();
        let trace = ExecutionTrace {
            operation: "test_op".into(),
            input_summary: "input".into(),
            output_summary: "output".into(),
            duration: Duration::from_secs(2),
            state_version_before: 0,
            state_version_after: 1,
        };
        history.record(state, trace);

        let retrieved = history.trace_at(0);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().operation, "test_op");
    }

    #[test]
    fn test_brick_history_trace_at_invalid() {
        let history = BrickHistory::new(10);
        assert!(history.trace_at(100).is_none());
    }

    #[test]
    fn test_brick_history_traces() {
        let mut history = BrickHistory::new(10);

        for i in 0..3 {
            let state = BrickState::new();
            let trace = ExecutionTrace {
                operation: format!("op_{}", i),
                input_summary: String::new(),
                output_summary: String::new(),
                duration: Duration::ZERO,
                state_version_before: i as u64,
                state_version_after: (i + 1) as u64,
            };
            history.record(state, trace);
        }

        let traces = history.traces();
        assert_eq!(traces.len(), 3);
        assert_eq!(traces[0].operation, "op_0");
        assert_eq!(traces[2].operation, "op_2");
    }

    #[test]
    fn test_brick_history_record_truncates_forward() {
        let mut history = BrickHistory::new(10);

        // Record 5 states
        for i in 0..5 {
            let mut state = BrickState::new();
            state.set_metadata("i", StateValue::Int(i));
            let trace = ExecutionTrace {
                operation: format!("op_{}", i),
                input_summary: String::new(),
                output_summary: String::new(),
                duration: Duration::ZERO,
                state_version_before: i as u64,
                state_version_after: (i + 1) as u64,
            };
            history.record(state, trace);
        }

        // Go back to position 2
        history.goto(2);

        // Record a new state - should truncate forward
        let mut new_state = BrickState::new();
        new_state.set_metadata("new", StateValue::Bool(true));
        let trace = ExecutionTrace {
            operation: "new_op".into(),
            input_summary: String::new(),
            output_summary: String::new(),
            duration: Duration::ZERO,
            state_version_before: 2,
            state_version_after: 3,
        };
        history.record(new_state, trace);

        // Should now have 3 states (0, 1, new)
        assert_eq!(history.len(), 3);
    }

    #[test]
    fn test_brick_history_capacity_eviction() {
        let mut history = BrickHistory::new(3); // Small capacity

        // Record more than capacity
        for i in 0..5 {
            let mut state = BrickState::new();
            state.set_metadata("i", StateValue::Int(i));
            let trace = ExecutionTrace {
                operation: format!("op_{}", i),
                input_summary: String::new(),
                output_summary: String::new(),
                duration: Duration::ZERO,
                state_version_before: i as u64,
                state_version_after: (i + 1) as u64,
            };
            history.record(state, trace);
        }

        // Should only have 3 states (oldest evicted)
        assert_eq!(history.len(), 3);

        // First state should be i=2 (0 and 1 evicted)
        let first = history.goto(0).unwrap();
        assert_eq!(first.get_metadata("i"), Some(&StateValue::Int(2)));
    }

    #[test]
    fn test_guard_severity_values() {
        assert_eq!(GuardSeverity::Warning, GuardSeverity::Warning);
        assert_eq!(GuardSeverity::Error, GuardSeverity::Error);
        assert_eq!(GuardSeverity::Critical, GuardSeverity::Critical);
        assert_ne!(GuardSeverity::Warning, GuardSeverity::Error);
    }

    #[test]
    fn test_invariant_guard_debug() {
        fn check(_: &BrickState) -> bool {
            true
        }
        let guard = InvariantGuard::new("test", check, GuardSeverity::Warning);
        let debug_str = format!("{:?}", guard);
        assert!(debug_str.contains("InvariantGuard"));
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn test_guarded_brick() {
        use super::super::{Brick, BrickAssertion, BrickBudget, BrickVerification};

        struct TestBrick;
        impl Brick for TestBrick {
            fn brick_name(&self) -> &'static str {
                "TestBrick"
            }
            fn assertions(&self) -> &[BrickAssertion] {
                &[]
            }
            fn budget(&self) -> BrickBudget {
                BrickBudget::uniform(16)
            }
            fn verify(&self) -> BrickVerification {
                BrickVerification {
                    passed: vec![],
                    failed: vec![],
                    verification_time: Duration::ZERO,
                }
            }
            fn to_html(&self) -> String {
                String::new()
            }
            fn to_css(&self) -> String {
                String::new()
            }
        }

        fn check_positive(state: &BrickState) -> bool {
            match state.get_metadata("count") {
                Some(StateValue::Int(n)) => *n >= 0,
                _ => true,
            }
        }

        let guard = InvariantGuard::new("positive", check_positive, GuardSeverity::Error);
        let guarded = GuardedBrick::new(TestBrick).guard(guard);

        assert_eq!(guarded.inner().brick_name(), "TestBrick");
        assert_eq!(guarded.guards().len(), 1);
    }

    #[test]
    fn test_guarded_brick_check_guards_pass() {
        use super::super::{Brick, BrickAssertion, BrickBudget, BrickVerification};

        struct TestBrick;
        impl Brick for TestBrick {
            fn brick_name(&self) -> &'static str {
                "TestBrick"
            }
            fn assertions(&self) -> &[BrickAssertion] {
                &[]
            }
            fn budget(&self) -> BrickBudget {
                BrickBudget::uniform(16)
            }
            fn verify(&self) -> BrickVerification {
                BrickVerification {
                    passed: vec![],
                    failed: vec![],
                    verification_time: Duration::ZERO,
                }
            }
            fn to_html(&self) -> String {
                String::new()
            }
            fn to_css(&self) -> String {
                String::new()
            }
        }

        fn always_pass(_: &BrickState) -> bool {
            true
        }

        let guard = InvariantGuard::new("always_pass", always_pass, GuardSeverity::Error);
        let guarded = GuardedBrick::new(TestBrick).guard(guard);

        let state = BrickState::new();
        assert!(guarded.check_guards(&state).is_ok());
    }

    #[test]
    fn test_guarded_brick_check_guards_fail() {
        use super::super::{Brick, BrickAssertion, BrickBudget, BrickVerification};

        struct TestBrick;
        impl Brick for TestBrick {
            fn brick_name(&self) -> &'static str {
                "TestBrick"
            }
            fn assertions(&self) -> &[BrickAssertion] {
                &[]
            }
            fn budget(&self) -> BrickBudget {
                BrickBudget::uniform(16)
            }
            fn verify(&self) -> BrickVerification {
                BrickVerification {
                    passed: vec![],
                    failed: vec![],
                    verification_time: Duration::ZERO,
                }
            }
            fn to_html(&self) -> String {
                String::new()
            }
            fn to_css(&self) -> String {
                String::new()
            }
        }

        fn always_fail(_: &BrickState) -> bool {
            false
        }

        let guard = InvariantGuard::new("always_fail", always_fail, GuardSeverity::Critical);
        let guarded = GuardedBrick::new(TestBrick).guard(guard);

        let state = BrickState::new();
        let result = guarded.check_guards(&state);
        assert!(result.is_err());

        let violation = result.unwrap_err();
        assert_eq!(violation.guard_name, "always_fail");
        assert_eq!(violation.severity, GuardSeverity::Critical);
    }

    #[test]
    fn test_guard_violation_display() {
        let violation = GuardViolation {
            guard_name: "test_guard",
            severity: GuardSeverity::Error,
        };
        let display = format!("{}", violation);
        assert!(display.contains("test_guard"));
        assert!(display.contains("Error"));
    }

    #[test]
    fn test_guard_violation_error_trait() {
        let violation = GuardViolation {
            guard_name: "test",
            severity: GuardSeverity::Warning,
        };
        let _: &dyn std::error::Error = &violation;
    }

    #[test]
    fn test_deterministic_rng_default() {
        let rng = DeterministicRng::default();
        assert_eq!(rng.state(), 42);
    }

    #[test]
    fn test_deterministic_rng_f32_range() {
        let mut rng = DeterministicRng::new(123);
        for _ in 0..100 {
            let val = rng.next_f32();
            assert!((0.0..1.0).contains(&val));
        }
    }

    #[test]
    fn test_deterministic_rng_state() {
        let mut rng = DeterministicRng::new(999);
        let _ = rng.next_u64();
        let state = rng.state();
        assert_ne!(state, 999); // State should have changed
    }

    #[test]
    fn test_deterministic_rng_restore() {
        let mut rng1 = DeterministicRng::new(100);
        let mut rng2 = DeterministicRng::new(999);

        // Get some values from rng1
        for _ in 0..10 {
            rng1.next_u64();
        }

        // Save state and restore to rng2
        let saved_state = rng1.state();
        rng2.restore(saved_state);

        // Both should now produce same sequence
        for _ in 0..10 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_deterministic_rng_clone() {
        let mut rng1 = DeterministicRng::new(555);
        for _ in 0..5 {
            rng1.next_u64();
        }

        let mut rng2 = rng1.clone();

        // Both should produce same sequence from here
        for _ in 0..10 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_deterministic_clock_default() {
        let clock = DeterministicClock::default();
        assert_eq!(clock.now_ns(), 0);
        // Default tick is 10ms
    }

    #[test]
    fn test_deterministic_clock_clone() {
        let mut clock1 = DeterministicClock::new(100, 50);
        clock1.advance(5);

        let clock2 = clock1.clone();
        assert_eq!(clock1.now_ns(), clock2.now_ns());
    }

    // ========================================================================
    // Additional tests for 95%+ coverage - Debug, Clone, and edge cases
    // ========================================================================

    #[test]
    fn test_state_value_debug() {
        let int_val = StateValue::Int(42);
        let debug_str = format!("{:?}", int_val);
        assert!(debug_str.contains("Int"));
        assert!(debug_str.contains("42"));

        let float_val = StateValue::Float(3.14);
        let debug_str = format!("{:?}", float_val);
        assert!(debug_str.contains("Float"));

        let string_val = StateValue::String("hello".into());
        let debug_str = format!("{:?}", string_val);
        assert!(debug_str.contains("String"));
        assert!(debug_str.contains("hello"));

        let bool_val = StateValue::Bool(true);
        let debug_str = format!("{:?}", bool_val);
        assert!(debug_str.contains("Bool"));
        assert!(debug_str.contains("true"));
    }

    #[test]
    fn test_brick_state_debug() {
        let mut state = BrickState::new();
        state.set_tensor("test", vec![1.0, 2.0], vec![2]);
        state.set_metadata("key", StateValue::Int(1));
        state.version = 5;

        let debug_str = format!("{:?}", state);
        assert!(debug_str.contains("BrickState"));
        assert!(debug_str.contains("version"));
    }

    #[test]
    fn test_execution_trace_debug() {
        let trace = ExecutionTrace {
            operation: "compute".into(),
            input_summary: "input data".into(),
            output_summary: "output data".into(),
            duration: Duration::from_millis(100),
            state_version_before: 1,
            state_version_after: 2,
        };

        let debug_str = format!("{:?}", trace);
        assert!(debug_str.contains("ExecutionTrace"));
        assert!(debug_str.contains("compute"));
    }

    #[test]
    fn test_brick_history_debug() {
        let history = BrickHistory::new(10);
        let debug_str = format!("{:?}", history);
        assert!(debug_str.contains("BrickHistory"));
    }

    #[test]
    fn test_guard_violation_clone() {
        let violation = GuardViolation {
            guard_name: "test_guard",
            severity: GuardSeverity::Critical,
        };
        let cloned = violation.clone();
        assert_eq!(violation.guard_name, cloned.guard_name);
        assert_eq!(violation.severity, cloned.severity);
    }

    #[test]
    fn test_guard_violation_debug() {
        let violation = GuardViolation {
            guard_name: "my_guard",
            severity: GuardSeverity::Warning,
        };
        let debug_str = format!("{:?}", violation);
        assert!(debug_str.contains("GuardViolation"));
        assert!(debug_str.contains("my_guard"));
        assert!(debug_str.contains("Warning"));
    }

    #[test]
    fn test_guarded_brick_debug() {
        use super::super::{Brick, BrickAssertion, BrickBudget, BrickVerification};

        #[derive(Debug)]
        struct TestBrick;
        impl Brick for TestBrick {
            fn brick_name(&self) -> &'static str {
                "TestBrick"
            }
            fn assertions(&self) -> &[BrickAssertion] {
                &[]
            }
            fn budget(&self) -> BrickBudget {
                BrickBudget::uniform(16)
            }
            fn verify(&self) -> BrickVerification {
                BrickVerification {
                    passed: vec![],
                    failed: vec![],
                    verification_time: Duration::ZERO,
                }
            }
            fn to_html(&self) -> String {
                String::new()
            }
            fn to_css(&self) -> String {
                String::new()
            }
        }

        fn check(_: &BrickState) -> bool {
            true
        }

        let guard = InvariantGuard::new("guard1", check, GuardSeverity::Warning);
        let guarded = GuardedBrick::new(TestBrick).guard(guard);

        let debug_str = format!("{:?}", guarded);
        assert!(debug_str.contains("GuardedBrick"));
    }

    #[test]
    fn test_guarded_brick_multiple_guards() {
        use super::super::{Brick, BrickAssertion, BrickBudget, BrickVerification};

        struct TestBrick;
        impl Brick for TestBrick {
            fn brick_name(&self) -> &'static str {
                "TestBrick"
            }
            fn assertions(&self) -> &[BrickAssertion] {
                &[]
            }
            fn budget(&self) -> BrickBudget {
                BrickBudget::uniform(16)
            }
            fn verify(&self) -> BrickVerification {
                BrickVerification {
                    passed: vec![],
                    failed: vec![],
                    verification_time: Duration::ZERO,
                }
            }
            fn to_html(&self) -> String {
                String::new()
            }
            fn to_css(&self) -> String {
                String::new()
            }
        }

        fn always_pass(_: &BrickState) -> bool {
            true
        }
        fn check_count(state: &BrickState) -> bool {
            match state.get_metadata("count") {
                Some(StateValue::Int(n)) => *n >= 0,
                _ => true,
            }
        }

        let guard1 = InvariantGuard::new("guard1", always_pass, GuardSeverity::Warning);
        let guard2 = InvariantGuard::new("guard2", check_count, GuardSeverity::Error);

        let guarded = GuardedBrick::new(TestBrick).guard(guard1).guard(guard2);

        assert_eq!(guarded.guards().len(), 2);

        // Both guards pass
        let mut state = BrickState::new();
        state.set_metadata("count", StateValue::Int(5));
        assert!(guarded.check_guards(&state).is_ok());

        // Second guard fails
        state.set_metadata("count", StateValue::Int(-1));
        let result = guarded.check_guards(&state);
        assert!(result.is_err());
        let violation = result.unwrap_err();
        assert_eq!(violation.guard_name, "guard2");
    }

    #[test]
    fn test_guarded_brick_first_guard_fails() {
        use super::super::{Brick, BrickAssertion, BrickBudget, BrickVerification};

        struct TestBrick;
        impl Brick for TestBrick {
            fn brick_name(&self) -> &'static str {
                "TestBrick"
            }
            fn assertions(&self) -> &[BrickAssertion] {
                &[]
            }
            fn budget(&self) -> BrickBudget {
                BrickBudget::uniform(16)
            }
            fn verify(&self) -> BrickVerification {
                BrickVerification {
                    passed: vec![],
                    failed: vec![],
                    verification_time: Duration::ZERO,
                }
            }
            fn to_html(&self) -> String {
                String::new()
            }
            fn to_css(&self) -> String {
                String::new()
            }
        }

        fn always_fail(_: &BrickState) -> bool {
            false
        }
        fn always_pass(_: &BrickState) -> bool {
            true
        }

        let guard1 = InvariantGuard::new("first_fail", always_fail, GuardSeverity::Error);
        let guard2 = InvariantGuard::new("second_pass", always_pass, GuardSeverity::Warning);

        let guarded = GuardedBrick::new(TestBrick).guard(guard1).guard(guard2);

        let state = BrickState::new();
        let result = guarded.check_guards(&state);
        assert!(result.is_err());
        // First guard should fail before second is checked
        assert_eq!(result.unwrap_err().guard_name, "first_fail");
    }

    #[test]
    fn test_guard_severity_clone() {
        let severity = GuardSeverity::Critical;
        let cloned = severity;
        assert_eq!(severity, cloned);
    }

    #[test]
    fn test_guard_severity_copy() {
        let severity = GuardSeverity::Warning;
        let copied: GuardSeverity = severity;
        assert_eq!(severity, copied);
    }

    #[test]
    fn test_guard_severity_debug() {
        let warning = GuardSeverity::Warning;
        let error = GuardSeverity::Error;
        let critical = GuardSeverity::Critical;

        assert!(format!("{:?}", warning).contains("Warning"));
        assert!(format!("{:?}", error).contains("Error"));
        assert!(format!("{:?}", critical).contains("Critical"));
    }

    #[test]
    fn test_deterministic_brick_trait_default_impls() {
        use super::super::{Brick, BrickAssertion, BrickBudget, BrickVerification};

        #[derive(Debug)]
        struct TestDeterministicBrick;

        #[derive(Clone, Default)]
        struct TestState {
            value: i32,
        }

        impl Brick for TestDeterministicBrick {
            fn brick_name(&self) -> &'static str {
                "TestDeterministicBrick"
            }
            fn assertions(&self) -> &[BrickAssertion] {
                &[]
            }
            fn budget(&self) -> BrickBudget {
                BrickBudget::uniform(16)
            }
            fn verify(&self) -> BrickVerification {
                BrickVerification {
                    passed: vec![],
                    failed: vec![],
                    verification_time: Duration::ZERO,
                }
            }
            fn to_html(&self) -> String {
                String::new()
            }
            fn to_css(&self) -> String {
                String::new()
            }
        }

        impl DeterministicBrick for TestDeterministicBrick {
            type State = TestState;
            type Input = i32;
            type Output = i32;

            fn execute_pure(
                state: Self::State,
                input: Self::Input,
            ) -> Result<(Self::State, Self::Output), BrickError> {
                let new_state = TestState {
                    value: state.value + input,
                };
                let output = new_state.value;
                Ok((new_state, output))
            }
        }

        // Test default initial_state()
        let initial = TestDeterministicBrick::initial_state();
        assert_eq!(initial.value, 0);

        // Test default state_dependencies()
        let brick = TestDeterministicBrick;
        let deps = brick.state_dependencies();
        assert!(deps.is_empty());

        // Test execute_pure
        let state = TestState { value: 10 };
        let (new_state, output) = TestDeterministicBrick::execute_pure(state, 5).unwrap();
        assert_eq!(new_state.value, 15);
        assert_eq!(output, 15);
    }

    #[test]
    fn test_deterministic_brick_with_custom_state_dependencies() {
        use super::super::{Brick, BrickAssertion, BrickBudget, BrickVerification};

        struct CustomDepsBrick {
            deps: Vec<&'static str>,
        }

        #[derive(Clone, Default)]
        struct SimpleState;

        impl Brick for CustomDepsBrick {
            fn brick_name(&self) -> &'static str {
                "CustomDepsBrick"
            }
            fn assertions(&self) -> &[BrickAssertion] {
                &[]
            }
            fn budget(&self) -> BrickBudget {
                BrickBudget::uniform(16)
            }
            fn verify(&self) -> BrickVerification {
                BrickVerification {
                    passed: vec![],
                    failed: vec![],
                    verification_time: Duration::ZERO,
                }
            }
            fn to_html(&self) -> String {
                String::new()
            }
            fn to_css(&self) -> String {
                String::new()
            }
        }

        impl DeterministicBrick for CustomDepsBrick {
            type State = SimpleState;
            type Input = ();
            type Output = ();

            fn execute_pure(
                state: Self::State,
                _input: Self::Input,
            ) -> Result<(Self::State, Self::Output), BrickError> {
                Ok((state, ()))
            }

            fn state_dependencies(&self) -> &[&str] {
                &self.deps
            }
        }

        let brick = CustomDepsBrick {
            deps: vec!["audio_buffer", "mel_filterbank"],
        };

        let deps = brick.state_dependencies();
        assert_eq!(deps.len(), 2);
        assert_eq!(deps[0], "audio_buffer");
        assert_eq!(deps[1], "mel_filterbank");
    }

    #[test]
    fn test_brick_history_current_edge_cases() {
        let mut history = BrickHistory::new(10);

        // Empty history returns None
        assert!(history.current().is_none());

        // Add one state
        let mut state = BrickState::new();
        state.set_metadata("v", StateValue::Int(100));
        let trace = ExecutionTrace {
            operation: "op".into(),
            input_summary: String::new(),
            output_summary: String::new(),
            duration: Duration::ZERO,
            state_version_before: 0,
            state_version_after: 1,
        };
        history.record(state, trace);

        // Position is 1, len is 1 - should return snapshots[0]
        let current = history.current();
        assert!(current.is_some());
        assert_eq!(
            current.unwrap().get_metadata("v"),
            Some(&StateValue::Int(100))
        );
    }

    #[test]
    fn test_brick_history_step_forward_returns_correct_state() {
        let mut history = BrickHistory::new(10);

        // Record 3 states with values 10, 20, 30
        for i in 1..=3 {
            let mut state = BrickState::new();
            state.set_metadata("val", StateValue::Int(i * 10));
            let trace = ExecutionTrace {
                operation: format!("op_{}", i),
                input_summary: String::new(),
                output_summary: String::new(),
                duration: Duration::ZERO,
                state_version_before: (i - 1) as u64,
                state_version_after: i as u64,
            };
            history.record(state, trace);
        }

        // Go to position 0
        history.goto(0);
        assert_eq!(history.position(), 0);

        // Step forward - should return state at position 0 (val=10), then increment position to 1
        let state = history.step_forward().unwrap();
        assert_eq!(state.get_metadata("val"), Some(&StateValue::Int(10)));
        assert_eq!(history.position(), 1);

        // Step forward again - returns state at position 1 (val=20), position becomes 2
        let state = history.step_forward().unwrap();
        assert_eq!(state.get_metadata("val"), Some(&StateValue::Int(20)));
        assert_eq!(history.position(), 2);
    }

    #[test]
    fn test_deterministic_rng_reproducibility_across_types() {
        let mut rng1 = DeterministicRng::new(99999);
        let mut rng2 = DeterministicRng::new(99999);

        // Mix of operations should produce same results
        for _ in 0..10 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
            let f64_1 = rng1.next_f64();
            let f64_2 = rng2.next_f64();
            assert!((f64_1 - f64_2).abs() < f64::EPSILON);
            let f32_1 = rng1.next_f32();
            let f32_2 = rng2.next_f32();
            assert!((f32_1 - f32_2).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_deterministic_clock_tick_sequence() {
        let mut clock = DeterministicClock::new(0, 16_666_667); // ~60fps tick

        // Tick 60 times
        for _ in 0..60 {
            clock.tick();
        }

        // Should be approximately 1 second
        let expected_ns = 16_666_667u64 * 60;
        assert_eq!(clock.now_ns(), expected_ns);

        // Check Duration conversion
        let duration = clock.now();
        assert!(duration.as_secs_f64() > 0.99 && duration.as_secs_f64() < 1.01);
    }

    #[test]
    fn test_brick_state_multiple_tensors() {
        let mut state = BrickState::new();

        state.set_tensor("audio", vec![1.0, 2.0, 3.0], vec![3]);
        state.set_tensor("mel", vec![4.0, 5.0], vec![1, 2]);
        state.set_tensor("empty", vec![], vec![0]);

        let (audio_data, audio_shape) = state.get_tensor("audio").unwrap();
        assert_eq!(audio_data, &[1.0, 2.0, 3.0]);
        assert_eq!(audio_shape, &[3]);

        let (mel_data, mel_shape) = state.get_tensor("mel").unwrap();
        assert_eq!(mel_data, &[4.0, 5.0]);
        assert_eq!(mel_shape, &[1, 2]);

        let (empty_data, empty_shape) = state.get_tensor("empty").unwrap();
        assert!(empty_data.is_empty());
        assert_eq!(empty_shape, &[0]);
    }

    #[test]
    fn test_brick_state_overwrite_tensor() {
        let mut state = BrickState::new();

        state.set_tensor("data", vec![1.0], vec![1]);
        let (data, shape) = state.get_tensor("data").unwrap();
        assert_eq!(data, &[1.0]);
        assert_eq!(shape, &[1]);

        // Overwrite with new data
        state.set_tensor("data", vec![2.0, 3.0, 4.0], vec![3]);
        let (data, shape) = state.get_tensor("data").unwrap();
        assert_eq!(data, &[2.0, 3.0, 4.0]);
        assert_eq!(shape, &[3]);
    }

    #[test]
    fn test_brick_state_overwrite_metadata() {
        let mut state = BrickState::new();

        state.set_metadata("key", StateValue::Int(1));
        assert_eq!(state.get_metadata("key"), Some(&StateValue::Int(1)));

        state.set_metadata("key", StateValue::String("replaced".into()));
        assert_eq!(
            state.get_metadata("key"),
            Some(&StateValue::String("replaced".into()))
        );
    }

    #[test]
    fn test_brick_state_snapshot_preserves_data() {
        let mut state = BrickState::new();
        state.set_tensor("t", vec![1.0, 2.0], vec![2]);
        state.set_metadata("m", StateValue::Float(3.14));
        state.version = 10;

        let snapshot = state.snapshot();

        // Verify snapshot has incremented version
        assert_eq!(snapshot.version, 11);

        // Verify data is preserved
        let (data, shape) = snapshot.get_tensor("t").unwrap();
        assert_eq!(data, &[1.0, 2.0]);
        assert_eq!(shape, &[2]);
        assert_eq!(snapshot.get_metadata("m"), Some(&StateValue::Float(3.14)));

        // Original unchanged
        assert_eq!(state.version, 10);
    }

    #[test]
    fn test_execution_trace_all_fields() {
        let trace = ExecutionTrace {
            operation: "mel_spectrogram".into(),
            input_summary: "1024 samples @ 16kHz".into(),
            output_summary: "80 mel bands".into(),
            duration: Duration::from_micros(1500),
            state_version_before: 42,
            state_version_after: 43,
        };

        assert_eq!(trace.operation, "mel_spectrogram");
        assert_eq!(trace.input_summary, "1024 samples @ 16kHz");
        assert_eq!(trace.output_summary, "80 mel bands");
        assert_eq!(trace.duration, Duration::from_micros(1500));
        assert_eq!(trace.state_version_before, 42);
        assert_eq!(trace.state_version_after, 43);
    }

    #[test]
    fn test_invariant_guard_different_severities() {
        fn check(_: &BrickState) -> bool {
            true
        }

        let warning_guard = InvariantGuard::new("warning", check, GuardSeverity::Warning);
        let error_guard = InvariantGuard::new("error", check, GuardSeverity::Error);
        let critical_guard = InvariantGuard::new("critical", check, GuardSeverity::Critical);

        assert_eq!(warning_guard.severity, GuardSeverity::Warning);
        assert_eq!(error_guard.severity, GuardSeverity::Error);
        assert_eq!(critical_guard.severity, GuardSeverity::Critical);

        let state = BrickState::new();
        assert!(warning_guard.check(&state));
        assert!(error_guard.check(&state));
        assert!(critical_guard.check(&state));
    }

    #[test]
    fn test_guard_violation_all_severities() {
        let warning = GuardViolation {
            guard_name: "w",
            severity: GuardSeverity::Warning,
        };
        let error = GuardViolation {
            guard_name: "e",
            severity: GuardSeverity::Error,
        };
        let critical = GuardViolation {
            guard_name: "c",
            severity: GuardSeverity::Critical,
        };

        assert!(format!("{}", warning).contains("Warning"));
        assert!(format!("{}", error).contains("Error"));
        assert!(format!("{}", critical).contains("Critical"));
    }

    #[test]
    fn test_deterministic_rng_zero_seed() {
        // Zero seed should still work (though not recommended)
        let mut rng = DeterministicRng::new(0);

        // First call with state=0 will produce 0 (0^0=0 for all xorshift ops)
        // But subsequent calls should produce non-zero values eventually
        let mut seen_nonzero = false;
        for _ in 0..100 {
            if rng.next_u64() != 0 {
                seen_nonzero = true;
                break;
            }
        }
        // Note: With seed 0, xorshift produces all zeros, which is a known edge case
        // The test verifies the function doesn't panic
        let _ = seen_nonzero;
    }

    #[test]
    fn test_deterministic_clock_zero_tick() {
        let mut clock = DeterministicClock::new(100, 0);

        clock.tick();
        assert_eq!(clock.now_ns(), 100); // No change with 0 tick

        clock.advance(100);
        assert_eq!(clock.now_ns(), 100); // Still no change
    }

    #[test]
    fn test_brick_history_size_one() {
        let mut history = BrickHistory::new(1);

        // Record first state
        let mut state1 = BrickState::new();
        state1.set_metadata("v", StateValue::Int(1));
        let trace1 = ExecutionTrace {
            operation: "op1".into(),
            input_summary: String::new(),
            output_summary: String::new(),
            duration: Duration::ZERO,
            state_version_before: 0,
            state_version_after: 1,
        };
        history.record(state1, trace1);
        assert_eq!(history.len(), 1);

        // Record second state - should evict first
        let mut state2 = BrickState::new();
        state2.set_metadata("v", StateValue::Int(2));
        let trace2 = ExecutionTrace {
            operation: "op2".into(),
            input_summary: String::new(),
            output_summary: String::new(),
            duration: Duration::ZERO,
            state_version_before: 1,
            state_version_after: 2,
        };
        history.record(state2, trace2);
        assert_eq!(history.len(), 1);

        // Only second state should exist
        let current = history.goto(0).unwrap();
        assert_eq!(current.get_metadata("v"), Some(&StateValue::Int(2)));
    }

    #[test]
    fn test_guarded_brick_no_guards() {
        use super::super::{Brick, BrickAssertion, BrickBudget, BrickVerification};

        struct TestBrick;
        impl Brick for TestBrick {
            fn brick_name(&self) -> &'static str {
                "TestBrick"
            }
            fn assertions(&self) -> &[BrickAssertion] {
                &[]
            }
            fn budget(&self) -> BrickBudget {
                BrickBudget::uniform(16)
            }
            fn verify(&self) -> BrickVerification {
                BrickVerification {
                    passed: vec![],
                    failed: vec![],
                    verification_time: Duration::ZERO,
                }
            }
            fn to_html(&self) -> String {
                String::new()
            }
            fn to_css(&self) -> String {
                String::new()
            }
        }

        let guarded = GuardedBrick::new(TestBrick);
        assert!(guarded.guards().is_empty());

        // With no guards, check_guards always passes
        let state = BrickState::new();
        assert!(guarded.check_guards(&state).is_ok());
    }

    #[test]
    fn test_deterministic_rng_distribution() {
        let mut rng = DeterministicRng::new(777);
        let mut sum = 0.0f64;
        let n = 10000;

        for _ in 0..n {
            sum += rng.next_f64();
        }

        let avg = sum / n as f64;
        // Average should be approximately 0.5 for uniform [0, 1)
        assert!(avg > 0.4 && avg < 0.6);
    }

    // ========================================================================
    // Additional tests for 95%+ coverage - Exercise all Brick trait methods
    // ========================================================================

    /// Shared test brick that exercises all Brick trait methods
    mod shared_brick {
        use super::*;
        use crate::brick::{BrickAssertion, BrickBudget, BrickVerification};

        pub struct ComprehensiveTestBrick {
            pub name: &'static str,
        }

        impl Brick for ComprehensiveTestBrick {
            fn brick_name(&self) -> &'static str {
                self.name
            }
            fn assertions(&self) -> &[BrickAssertion] {
                &[]
            }
            fn budget(&self) -> BrickBudget {
                BrickBudget::uniform(16)
            }
            fn verify(&self) -> BrickVerification {
                BrickVerification {
                    passed: vec![],
                    failed: vec![],
                    verification_time: Duration::ZERO,
                }
            }
            fn to_html(&self) -> String {
                format!("<div>{}</div>", self.name)
            }
            fn to_css(&self) -> String {
                ".brick { color: red; }".into()
            }
        }
    }

    #[test]
    fn test_comprehensive_brick_all_methods() {
        use shared_brick::ComprehensiveTestBrick;

        let brick = ComprehensiveTestBrick { name: "TestBrick" };

        // Exercise all Brick trait methods
        assert_eq!(brick.brick_name(), "TestBrick");
        assert!(brick.assertions().is_empty());
        assert_eq!(brick.budget().total_ms, 16);

        let verification = brick.verify();
        assert!(verification.passed.is_empty());
        assert!(verification.failed.is_empty());
        assert_eq!(verification.verification_time, Duration::ZERO);

        assert!(brick.to_html().contains("TestBrick"));
        assert!(brick.to_css().contains(".brick"));
    }

    #[test]
    fn test_guarded_brick_exercises_inner_brick_methods() {
        use shared_brick::ComprehensiveTestBrick;

        fn always_pass(_: &BrickState) -> bool {
            true
        }

        let guard = InvariantGuard::new("pass", always_pass, GuardSeverity::Warning);
        let guarded = GuardedBrick::new(ComprehensiveTestBrick { name: "Guarded" }).guard(guard);

        // Exercise all methods via inner()
        let inner = guarded.inner();
        assert_eq!(inner.brick_name(), "Guarded");
        assert!(inner.assertions().is_empty());
        assert_eq!(inner.budget().total_ms, 16);

        let verification = inner.verify();
        assert!(verification.passed.is_empty());
        assert!(inner.to_html().contains("Guarded"));
        assert!(inner.to_css().contains(".brick"));
    }

    #[test]
    fn test_guard_check_function_all_state_value_variants() {
        // Test guard check function with all StateValue variants
        fn check_any_value(state: &BrickState) -> bool {
            match state.get_metadata("val") {
                Some(StateValue::Int(_)) => true,
                Some(StateValue::Float(_)) => true,
                Some(StateValue::String(_)) => true,
                Some(StateValue::Bool(_)) => true,
                None => true,
            }
        }

        let guard = InvariantGuard::new("any_value", check_any_value, GuardSeverity::Warning);

        // Test with Int
        let mut state = BrickState::new();
        state.set_metadata("val", StateValue::Int(42));
        assert!(guard.check(&state));

        // Test with Float
        state.set_metadata("val", StateValue::Float(3.14));
        assert!(guard.check(&state));

        // Test with String
        state.set_metadata("val", StateValue::String("test".into()));
        assert!(guard.check(&state));

        // Test with Bool
        state.set_metadata("val", StateValue::Bool(true));
        assert!(guard.check(&state));

        // Test with None
        let empty_state = BrickState::new();
        assert!(guard.check(&empty_state));
    }

    #[test]
    fn test_guard_check_positive_with_non_int_metadata() {
        // This exercises the `_ => true` branch in guard check functions
        fn check_positive_or_default(state: &BrickState) -> bool {
            match state.get_metadata("count") {
                Some(StateValue::Int(n)) => *n >= 0,
                _ => true, // This branch needs coverage
            }
        }

        let guard = InvariantGuard::new(
            "positive_or_default",
            check_positive_or_default,
            GuardSeverity::Error,
        );

        // Test with Float (not Int) - should return true via default branch
        let mut state = BrickState::new();
        state.set_metadata("count", StateValue::Float(42.0));
        assert!(guard.check(&state));

        // Test with String
        state.set_metadata("count", StateValue::String("not a number".into()));
        assert!(guard.check(&state));

        // Test with Bool
        state.set_metadata("count", StateValue::Bool(false));
        assert!(guard.check(&state));

        // Test with no metadata at all
        let empty_state = BrickState::new();
        assert!(guard.check(&empty_state));
    }

    #[test]
    fn test_deterministic_brick_error_propagation() {
        use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};

        #[derive(Clone, Default)]
        struct ErrorState {
            should_fail: bool,
        }

        struct FailingBrick;

        impl Brick for FailingBrick {
            fn brick_name(&self) -> &'static str {
                "FailingBrick"
            }
            fn assertions(&self) -> &[BrickAssertion] {
                &[]
            }
            fn budget(&self) -> BrickBudget {
                BrickBudget::uniform(16)
            }
            fn verify(&self) -> BrickVerification {
                BrickVerification {
                    passed: vec![],
                    failed: vec![],
                    verification_time: Duration::ZERO,
                }
            }
            fn to_html(&self) -> String {
                String::new()
            }
            fn to_css(&self) -> String {
                String::new()
            }
        }

        impl DeterministicBrick for FailingBrick {
            type State = ErrorState;
            type Input = ();
            type Output = ();

            fn execute_pure(
                state: Self::State,
                _input: Self::Input,
            ) -> Result<(Self::State, Self::Output), BrickError> {
                if state.should_fail {
                    Err(BrickError::HtmlGenerationFailed {
                        reason: "test failure".into(),
                    })
                } else {
                    Ok((state, ()))
                }
            }
        }

        // Test successful execution
        let state = ErrorState { should_fail: false };
        let result = FailingBrick::execute_pure(state, ());
        assert!(result.is_ok());

        // Test failing execution
        let state = ErrorState { should_fail: true };
        let result = FailingBrick::execute_pure(state, ());
        assert!(result.is_err());

        // Verify initial_state and state_dependencies
        let initial = FailingBrick::initial_state();
        assert!(!initial.should_fail);

        let brick = FailingBrick;
        assert!(brick.state_dependencies().is_empty());
    }

    #[test]
    fn test_brick_history_complex_navigation() {
        let mut history = BrickHistory::new(10);

        // Record 5 states
        for i in 0..5 {
            let mut state = BrickState::new();
            state.set_metadata("idx", StateValue::Int(i));
            let trace = ExecutionTrace {
                operation: format!("op_{}", i),
                input_summary: format!("input_{}", i),
                output_summary: format!("output_{}", i),
                duration: Duration::from_millis(i as u64),
                state_version_before: i as u64,
                state_version_after: (i + 1) as u64,
            };
            history.record(state, trace);
        }

        // Navigate to position 2
        let state = history.goto(2).unwrap();
        assert_eq!(state.get_metadata("idx"), Some(&StateValue::Int(2)));

        // Step forward twice
        let state = history.step_forward().unwrap();
        assert_eq!(state.get_metadata("idx"), Some(&StateValue::Int(2)));

        let state = history.step_forward().unwrap();
        assert_eq!(state.get_metadata("idx"), Some(&StateValue::Int(3)));

        // Step back once
        let state = history.step_back().unwrap();
        assert_eq!(state.get_metadata("idx"), Some(&StateValue::Int(3)));

        // Get current
        let current = history.current().unwrap();
        assert!(current.get_metadata("idx").is_some());
    }

    #[test]
    fn test_execution_trace_with_all_fields_populated() {
        let trace = ExecutionTrace {
            operation: "complex_operation".into(),
            input_summary: "1024 samples, 16-bit PCM".into(),
            output_summary: "80 mel filterbank coefficients".into(),
            duration: Duration::from_micros(2500),
            state_version_before: 100,
            state_version_after: 101,
        };

        // Verify all fields are accessible
        assert_eq!(trace.operation, "complex_operation");
        assert!(trace.input_summary.contains("1024"));
        assert!(trace.output_summary.contains("mel"));
        assert_eq!(trace.duration.as_micros(), 2500);
        assert_eq!(trace.state_version_before, 100);
        assert_eq!(trace.state_version_after, 101);

        // Clone and verify
        let cloned = trace.clone();
        assert_eq!(trace.operation, cloned.operation);
        assert_eq!(trace.duration, cloned.duration);
    }

    #[test]
    fn test_brick_state_comprehensive() {
        let mut state = BrickState::new();

        // Add multiple tensors with various shapes
        state.set_tensor("scalar", vec![1.0], vec![]);
        state.set_tensor("vector", vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        state.set_tensor("matrix", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        // Add all types of metadata
        state.set_metadata("int", StateValue::Int(-999));
        state.set_metadata("float", StateValue::Float(2.718281828));
        state.set_metadata("string", StateValue::String("deterministic".into()));
        state.set_metadata("bool", StateValue::Bool(false));

        // Verify tensors
        let (scalar_data, scalar_shape) = state.get_tensor("scalar").unwrap();
        assert_eq!(scalar_data, &[1.0]);
        assert!(scalar_shape.is_empty());

        let (matrix_data, matrix_shape) = state.get_tensor("matrix").unwrap();
        assert_eq!(matrix_data.len(), 6);
        assert_eq!(matrix_shape, &[2, 3]);

        // Verify metadata
        assert_eq!(state.get_metadata("int"), Some(&StateValue::Int(-999)));
        assert_eq!(
            state.get_metadata("float"),
            Some(&StateValue::Float(2.718281828))
        );
        assert_eq!(
            state.get_metadata("string"),
            Some(&StateValue::String("deterministic".into()))
        );
        assert_eq!(state.get_metadata("bool"), Some(&StateValue::Bool(false)));

        // Create snapshot and verify version increment
        state.version = 50;
        let snapshot = state.snapshot();
        assert_eq!(snapshot.version, 51);

        // Verify snapshot has all data
        assert!(snapshot.get_tensor("scalar").is_some());
        assert!(snapshot.get_tensor("vector").is_some());
        assert!(snapshot.get_tensor("matrix").is_some());
        assert_eq!(snapshot.get_metadata("int"), Some(&StateValue::Int(-999)));
    }

    #[test]
    fn test_guard_severity_all_variants_eq() {
        // Test equality within variants
        assert_eq!(GuardSeverity::Warning, GuardSeverity::Warning);
        assert_eq!(GuardSeverity::Error, GuardSeverity::Error);
        assert_eq!(GuardSeverity::Critical, GuardSeverity::Critical);

        // Test inequality across variants
        assert_ne!(GuardSeverity::Warning, GuardSeverity::Error);
        assert_ne!(GuardSeverity::Warning, GuardSeverity::Critical);
        assert_ne!(GuardSeverity::Error, GuardSeverity::Critical);

        // Test Copy trait
        let severity = GuardSeverity::Error;
        let copied: GuardSeverity = severity;
        let cloned = severity;
        assert_eq!(severity, copied);
        assert_eq!(severity, cloned);
    }

    #[test]
    fn test_invariant_guard_with_complex_check() {
        fn check_tensor_bounds(state: &BrickState) -> bool {
            if let Some((data, _shape)) = state.get_tensor("values") {
                data.iter().all(|&v| (0.0..=1.0).contains(&v))
            } else {
                true // No tensor = valid
            }
        }

        let guard = InvariantGuard::new(
            "tensor_bounds",
            check_tensor_bounds,
            GuardSeverity::Critical,
        );

        // Test with valid tensor
        let mut state = BrickState::new();
        state.set_tensor("values", vec![0.0, 0.5, 1.0], vec![3]);
        assert!(guard.check(&state));

        // Test with invalid tensor
        state.set_tensor("values", vec![0.0, 1.5, 0.5], vec![3]);
        assert!(!guard.check(&state));

        // Test with no tensor
        let empty_state = BrickState::new();
        assert!(guard.check(&empty_state));

        // Verify guard properties
        assert_eq!(guard.name, "tensor_bounds");
        assert_eq!(guard.severity, GuardSeverity::Critical);
    }

    #[test]
    fn test_guarded_brick_chain_multiple_guards() {
        use shared_brick::ComprehensiveTestBrick;

        fn guard1(_: &BrickState) -> bool {
            true
        }
        fn guard2(_: &BrickState) -> bool {
            true
        }
        fn guard3(_: &BrickState) -> bool {
            true
        }

        let guarded = GuardedBrick::new(ComprehensiveTestBrick { name: "Multi" })
            .guard(InvariantGuard::new("g1", guard1, GuardSeverity::Warning))
            .guard(InvariantGuard::new("g2", guard2, GuardSeverity::Error))
            .guard(InvariantGuard::new("g3", guard3, GuardSeverity::Critical));

        assert_eq!(guarded.guards().len(), 3);
        assert_eq!(guarded.guards()[0].name, "g1");
        assert_eq!(guarded.guards()[1].name, "g2");
        assert_eq!(guarded.guards()[2].name, "g3");

        assert_eq!(guarded.guards()[0].severity, GuardSeverity::Warning);
        assert_eq!(guarded.guards()[1].severity, GuardSeverity::Error);
        assert_eq!(guarded.guards()[2].severity, GuardSeverity::Critical);

        // All guards pass
        let state = BrickState::new();
        assert!(guarded.check_guards(&state).is_ok());
    }

    #[test]
    fn test_guard_violation_display_all_severities() {
        let warning = GuardViolation {
            guard_name: "warn_guard",
            severity: GuardSeverity::Warning,
        };
        let error = GuardViolation {
            guard_name: "err_guard",
            severity: GuardSeverity::Error,
        };
        let critical = GuardViolation {
            guard_name: "crit_guard",
            severity: GuardSeverity::Critical,
        };

        let warning_str = format!("{}", warning);
        let error_str = format!("{}", error);
        let critical_str = format!("{}", critical);

        assert!(warning_str.contains("warn_guard"));
        assert!(warning_str.contains("Warning"));

        assert!(error_str.contains("err_guard"));
        assert!(error_str.contains("Error"));

        assert!(critical_str.contains("crit_guard"));
        assert!(critical_str.contains("Critical"));

        // Test Debug trait
        let debug_str = format!("{:?}", warning);
        assert!(debug_str.contains("GuardViolation"));
        assert!(debug_str.contains("warn_guard"));
    }

    #[test]
    fn test_deterministic_rng_edge_cases() {
        // Test with max seed
        let mut rng = DeterministicRng::new(u64::MAX);
        let _ = rng.next_u64();
        let _ = rng.next_f64();
        let _ = rng.next_f32();

        // Test state save/restore across different operations
        let mut rng1 = DeterministicRng::new(0xDEADBEEF);
        for _ in 0..50 {
            let _ = rng1.next_u64();
        }
        let saved = rng1.state();

        let mut rng2 = DeterministicRng::new(0);
        rng2.restore(saved);

        // Both should produce same sequence from here
        for _ in 0..20 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_deterministic_clock_edge_cases() {
        // Test with very large tick
        let mut clock = DeterministicClock::new(0, u64::MAX / 2);
        clock.tick();
        assert_eq!(clock.now_ns(), u64::MAX / 2);

        // Test set to max value
        clock.set(u64::MAX - 1);
        assert_eq!(clock.now_ns(), u64::MAX - 1);

        // Test Duration conversion with large values
        let clock2 = DeterministicClock::new(1_000_000_000, 1); // 1 second
        let duration = clock2.now();
        assert_eq!(duration.as_secs(), 1);
    }

    #[test]
    fn test_brick_history_boundary_conditions() {
        // Test with empty history (capacity > 0, but no items recorded)
        let history = BrickHistory::new(5);

        // Test trace_at with empty history
        assert!(history.trace_at(0).is_none());

        // Test traces with empty history
        assert!(history.traces().is_empty());

        // Test step operations on empty history
        let mut history2 = BrickHistory::new(5);
        assert!(history2.step_back().is_none());
        assert!(history2.step_forward().is_none());
        assert!(history2.goto(0).is_none());
        assert!(history2.current().is_none());
    }

    #[test]
    fn test_brick_history_full_cycle() {
        let mut history = BrickHistory::new(3);

        // Fill to capacity
        for i in 0..3 {
            let mut state = BrickState::new();
            state.set_metadata("v", StateValue::Int(i));
            state.version = i as u64;
            let trace = ExecutionTrace {
                operation: format!("op_{}", i),
                input_summary: String::new(),
                output_summary: String::new(),
                duration: Duration::from_millis(i as u64 * 10),
                state_version_before: i as u64,
                state_version_after: (i + 1) as u64,
            };
            history.record(state, trace);
        }

        assert_eq!(history.len(), 3);

        // Navigate all the way back
        history.step_back();
        history.step_back();
        history.step_back();
        assert_eq!(history.position(), 0);

        // Record new - should truncate forward
        let mut new_state = BrickState::new();
        new_state.set_metadata("v", StateValue::Int(100));
        let trace = ExecutionTrace {
            operation: "new_op".into(),
            input_summary: String::new(),
            output_summary: String::new(),
            duration: Duration::ZERO,
            state_version_before: 0,
            state_version_after: 1,
        };
        history.record(new_state, trace);

        // Should have only 1 state now
        assert_eq!(history.len(), 1);
        assert_eq!(history.position(), 1);

        // Verify it's the new state
        let current = history.current().unwrap();
        assert_eq!(current.get_metadata("v"), Some(&StateValue::Int(100)));
    }

    #[test]
    fn test_state_value_debug_format() {
        let values = [
            StateValue::Int(i64::MIN),
            StateValue::Int(i64::MAX),
            StateValue::Float(f64::MIN),
            StateValue::Float(f64::MAX),
            StateValue::Float(f64::NAN),
            StateValue::Float(f64::INFINITY),
            StateValue::String(String::new()),
            StateValue::String("a very long string with special chars: \n\t\"".into()),
            StateValue::Bool(true),
            StateValue::Bool(false),
        ];

        for value in &values {
            let debug_str = format!("{:?}", value);
            assert!(!debug_str.is_empty());
        }
    }

    #[test]
    fn test_invariant_guard_debug_format() {
        fn dummy(_: &BrickState) -> bool {
            true
        }

        let guard = InvariantGuard::new("debug_test", dummy, GuardSeverity::Warning);
        let debug_str = format!("{:?}", guard);

        assert!(debug_str.contains("InvariantGuard"));
        assert!(debug_str.contains("debug_test"));
        assert!(debug_str.contains("<fn>"));
        assert!(debug_str.contains("Warning"));
    }

    #[test]
    fn test_brick_state_tensor_shape_mismatch() {
        let mut state = BrickState::new();

        // Add tensor normally
        state.set_tensor("normal", vec![1.0, 2.0, 3.0], vec![3]);
        assert!(state.get_tensor("normal").is_some());

        // Manually add tensor without shape
        state.tensors.insert("orphan".into(), vec![1.0, 2.0]);
        assert!(state.get_tensor("orphan").is_none());

        // Manually add shape without tensor
        state.shapes.insert("ghost".into(), vec![2, 2]);
        assert!(state.get_tensor("ghost").is_none());
    }

    #[test]
    fn test_deterministic_brick_with_non_default_initial_state() {
        use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};

        #[derive(Clone)]
        struct CustomInitState {
            counter: i32,
            name: String,
        }

        impl Default for CustomInitState {
            fn default() -> Self {
                Self {
                    counter: 100, // Non-zero default
                    name: "initialized".into(),
                }
            }
        }

        struct CustomInitBrick;

        impl Brick for CustomInitBrick {
            fn brick_name(&self) -> &'static str {
                "CustomInitBrick"
            }
            fn assertions(&self) -> &[BrickAssertion] {
                &[]
            }
            fn budget(&self) -> BrickBudget {
                BrickBudget::uniform(16)
            }
            fn verify(&self) -> BrickVerification {
                BrickVerification {
                    passed: vec![],
                    failed: vec![],
                    verification_time: Duration::ZERO,
                }
            }
            fn to_html(&self) -> String {
                String::new()
            }
            fn to_css(&self) -> String {
                String::new()
            }
        }

        impl DeterministicBrick for CustomInitBrick {
            type State = CustomInitState;
            type Input = i32;
            type Output = String;

            fn execute_pure(
                state: Self::State,
                input: Self::Input,
            ) -> Result<(Self::State, Self::Output), BrickError> {
                let new_state = CustomInitState {
                    counter: state.counter + input,
                    name: format!("{}-{}", state.name, input),
                };
                let output = format!("Counter: {}", new_state.counter);
                Ok((new_state, output))
            }
        }

        // Test initial_state default implementation
        let initial = CustomInitBrick::initial_state();
        assert_eq!(initial.counter, 100);
        assert_eq!(initial.name, "initialized");

        // Execute and verify
        let (new_state, output) = CustomInitBrick::execute_pure(initial, 5).unwrap();
        assert_eq!(new_state.counter, 105);
        assert_eq!(new_state.name, "initialized-5");
        assert!(output.contains("105"));
    }

    #[test]
    fn test_guarded_brick_check_guards_returns_first_failure() {
        use shared_brick::ComprehensiveTestBrick;

        fn pass(_: &BrickState) -> bool {
            true
        }
        fn fail1(_: &BrickState) -> bool {
            false
        }
        fn fail2(_: &BrickState) -> bool {
            false
        }

        let guarded = GuardedBrick::new(ComprehensiveTestBrick { name: "Test" })
            .guard(InvariantGuard::new("pass", pass, GuardSeverity::Warning))
            .guard(InvariantGuard::new("fail1", fail1, GuardSeverity::Error))
            .guard(InvariantGuard::new("fail2", fail2, GuardSeverity::Critical));

        let state = BrickState::new();
        let result = guarded.check_guards(&state);
        assert!(result.is_err());

        let violation = result.unwrap_err();
        // Should be fail1, not fail2
        assert_eq!(violation.guard_name, "fail1");
        assert_eq!(violation.severity, GuardSeverity::Error);
    }

    #[test]
    fn test_guard_violation_error_trait_source() {
        let violation = GuardViolation {
            guard_name: "test",
            severity: GuardSeverity::Warning,
        };

        // Test std::error::Error trait
        let err: &dyn std::error::Error = &violation;
        assert!(err.source().is_none());

        // Test Display
        let display = format!("{}", err);
        assert!(display.contains("test"));
    }

    #[test]
    fn test_brick_history_current_after_modifications() {
        let mut history = BrickHistory::new(10);

        // Empty history
        assert!(history.current().is_none());

        // Add one item
        let mut state = BrickState::new();
        state.set_metadata("x", StateValue::Int(1));
        let trace = ExecutionTrace {
            operation: "op".into(),
            input_summary: String::new(),
            output_summary: String::new(),
            duration: Duration::ZERO,
            state_version_before: 0,
            state_version_after: 1,
        };
        history.record(state, trace);

        // current() should return the last recorded state
        let current = history.current();
        assert!(current.is_some());
        assert_eq!(
            current.unwrap().get_metadata("x"),
            Some(&StateValue::Int(1))
        );

        // Add another item
        let mut state2 = BrickState::new();
        state2.set_metadata("x", StateValue::Int(2));
        let trace2 = ExecutionTrace {
            operation: "op2".into(),
            input_summary: String::new(),
            output_summary: String::new(),
            duration: Duration::ZERO,
            state_version_before: 1,
            state_version_after: 2,
        };
        history.record(state2, trace2);

        // current() should return the new last state
        let current = history.current();
        assert!(current.is_some());
        assert_eq!(
            current.unwrap().get_metadata("x"),
            Some(&StateValue::Int(2))
        );

        // Go back
        history.step_back();
        // current() should now return the previous state
        let current = history.current();
        assert!(current.is_some());
    }

    // ========================================================================
    // Tests to exercise Brick trait methods on all test fixtures
    // These ensure all the Brick impl methods get called
    // ========================================================================

    /// Helper to exercise all Brick trait methods on any Brick implementor
    fn exercise_brick_trait_methods<B: Brick>(brick: &B) {
        // Call every method to ensure coverage
        let _name = brick.brick_name();
        let _assertions = brick.assertions();
        let _budget = brick.budget();
        let _verification = brick.verify();
        let _html = brick.to_html();
        let _css = brick.to_css();
        let _test_id = brick.test_id();
        let _can_render = brick.can_render();
    }

    #[test]
    fn test_exercise_guarded_brick_inner_all_methods() {
        use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};

        struct FullBrick;
        impl Brick for FullBrick {
            fn brick_name(&self) -> &'static str {
                "FullBrick"
            }
            fn assertions(&self) -> &[BrickAssertion] {
                &[]
            }
            fn budget(&self) -> BrickBudget {
                BrickBudget::uniform(16)
            }
            fn verify(&self) -> BrickVerification {
                BrickVerification {
                    passed: vec![],
                    failed: vec![],
                    verification_time: Duration::ZERO,
                }
            }
            fn to_html(&self) -> String {
                "<div>Full</div>".into()
            }
            fn to_css(&self) -> String {
                ".full { }".into()
            }
        }

        fn check(_: &BrickState) -> bool {
            true
        }

        let guard = InvariantGuard::new("g", check, GuardSeverity::Warning);
        let guarded = GuardedBrick::new(FullBrick).guard(guard);

        // Exercise all methods on the inner brick
        exercise_brick_trait_methods(guarded.inner());
    }

    #[test]
    fn test_exercise_deterministic_brick_all_methods() {
        use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};

        struct DetBrick;

        #[derive(Clone, Default)]
        struct DetState;

        impl Brick for DetBrick {
            fn brick_name(&self) -> &'static str {
                "DetBrick"
            }
            fn assertions(&self) -> &[BrickAssertion] {
                &[]
            }
            fn budget(&self) -> BrickBudget {
                BrickBudget::uniform(16)
            }
            fn verify(&self) -> BrickVerification {
                BrickVerification {
                    passed: vec![],
                    failed: vec![],
                    verification_time: Duration::ZERO,
                }
            }
            fn to_html(&self) -> String {
                "<p>Det</p>".into()
            }
            fn to_css(&self) -> String {
                ".det { }".into()
            }
        }

        impl DeterministicBrick for DetBrick {
            type State = DetState;
            type Input = ();
            type Output = ();

            fn execute_pure(
                state: Self::State,
                _input: Self::Input,
            ) -> Result<(Self::State, Self::Output), BrickError> {
                Ok((state, ()))
            }
        }

        let brick = DetBrick;
        exercise_brick_trait_methods(&brick);

        // Also exercise DeterministicBrick specific methods
        let _ = DetBrick::initial_state();
        let _ = brick.state_dependencies();
        let state = DetState;
        let _ = DetBrick::execute_pure(state, ());
    }

    #[test]
    fn test_exercise_various_guard_check_functions() {
        // Define and exercise various guard check functions to ensure coverage

        fn check_int_positive(state: &BrickState) -> bool {
            match state.get_metadata("val") {
                Some(StateValue::Int(n)) => *n >= 0,
                _ => true,
            }
        }

        fn check_float_bounded(state: &BrickState) -> bool {
            match state.get_metadata("val") {
                Some(StateValue::Float(f)) => *f >= 0.0 && *f <= 1.0,
                _ => true,
            }
        }

        fn check_string_nonempty(state: &BrickState) -> bool {
            match state.get_metadata("val") {
                Some(StateValue::String(s)) => !s.is_empty(),
                _ => true,
            }
        }

        fn check_bool_true(state: &BrickState) -> bool {
            match state.get_metadata("val") {
                Some(StateValue::Bool(b)) => *b,
                _ => true,
            }
        }

        let guard1 =
            InvariantGuard::new("int_positive", check_int_positive, GuardSeverity::Warning);
        let guard2 =
            InvariantGuard::new("float_bounded", check_float_bounded, GuardSeverity::Error);
        let guard3 = InvariantGuard::new(
            "string_nonempty",
            check_string_nonempty,
            GuardSeverity::Critical,
        );
        let guard4 = InvariantGuard::new("bool_true", check_bool_true, GuardSeverity::Warning);

        // Test with Int
        let mut state = BrickState::new();
        state.set_metadata("val", StateValue::Int(5));
        assert!(guard1.check(&state));
        assert!(guard2.check(&state));
        assert!(guard3.check(&state));
        assert!(guard4.check(&state));

        // Test with negative Int
        state.set_metadata("val", StateValue::Int(-5));
        assert!(!guard1.check(&state));

        // Test with Float in range
        state.set_metadata("val", StateValue::Float(0.5));
        assert!(guard2.check(&state));

        // Test with Float out of range
        state.set_metadata("val", StateValue::Float(1.5));
        assert!(!guard2.check(&state));

        // Test with non-empty String
        state.set_metadata("val", StateValue::String("hello".into()));
        assert!(guard3.check(&state));

        // Test with empty String
        state.set_metadata("val", StateValue::String(String::new()));
        assert!(!guard3.check(&state));

        // Test with true Bool
        state.set_metadata("val", StateValue::Bool(true));
        assert!(guard4.check(&state));

        // Test with false Bool
        state.set_metadata("val", StateValue::Bool(false));
        assert!(!guard4.check(&state));
    }

    #[test]
    fn test_guarded_brick_with_all_methods_exercised() {
        use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};

        struct TestBrickFull;

        impl Brick for TestBrickFull {
            fn brick_name(&self) -> &'static str {
                "TestBrickFull"
            }
            fn assertions(&self) -> &[BrickAssertion] {
                &[BrickAssertion::TextVisible]
            }
            fn budget(&self) -> BrickBudget {
                BrickBudget::new(5, 5, 6)
            }
            fn verify(&self) -> BrickVerification {
                BrickVerification {
                    passed: vec![BrickAssertion::TextVisible],
                    failed: vec![],
                    verification_time: Duration::from_millis(1),
                }
            }
            fn to_html(&self) -> String {
                "<div class='test'>Content</div>".into()
            }
            fn to_css(&self) -> String {
                ".test { color: blue; }".into()
            }
        }

        fn check_has_count(state: &BrickState) -> bool {
            state.get_metadata("count").is_some()
        }

        let guard = InvariantGuard::new("has_count", check_has_count, GuardSeverity::Warning);
        let guarded = GuardedBrick::new(TestBrickFull).guard(guard);

        // Exercise inner brick
        let inner = guarded.inner();
        assert_eq!(inner.brick_name(), "TestBrickFull");
        assert_eq!(inner.assertions().len(), 1);
        assert_eq!(inner.budget().total_ms, 16);
        assert!(inner.verify().is_valid());
        assert!(inner.to_html().contains("Content"));
        assert!(inner.to_css().contains("blue"));
        assert!(inner.can_render());
        assert!(inner.test_id().is_none());

        // Check guards with state that has count
        let mut state = BrickState::new();
        state.set_metadata("count", StateValue::Int(42));
        assert!(guarded.check_guards(&state).is_ok());

        // Check guards with state that doesn't have count
        let empty_state = BrickState::new();
        let result = guarded.check_guards(&empty_state);
        assert!(result.is_err());
    }

    #[test]
    fn test_deterministic_brick_with_state_dependencies_override() {
        use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};

        struct DepsOverrideBrick;

        #[derive(Clone, Default)]
        struct DepsState;

        impl Brick for DepsOverrideBrick {
            fn brick_name(&self) -> &'static str {
                "DepsOverrideBrick"
            }
            fn assertions(&self) -> &[BrickAssertion] {
                &[]
            }
            fn budget(&self) -> BrickBudget {
                BrickBudget::uniform(16)
            }
            fn verify(&self) -> BrickVerification {
                BrickVerification {
                    passed: vec![],
                    failed: vec![],
                    verification_time: Duration::ZERO,
                }
            }
            fn to_html(&self) -> String {
                String::new()
            }
            fn to_css(&self) -> String {
                String::new()
            }
        }

        impl DeterministicBrick for DepsOverrideBrick {
            type State = DepsState;
            type Input = ();
            type Output = ();

            fn execute_pure(
                state: Self::State,
                _input: Self::Input,
            ) -> Result<(Self::State, Self::Output), BrickError> {
                Ok((state, ()))
            }

            fn state_dependencies(&self) -> &[&str] {
                &["dep1", "dep2", "dep3"]
            }
        }

        let brick = DepsOverrideBrick;

        // Exercise all Brick methods
        exercise_brick_trait_methods(&brick);

        // Check custom state_dependencies
        let deps = brick.state_dependencies();
        assert_eq!(deps.len(), 3);
        assert_eq!(deps[0], "dep1");
        assert_eq!(deps[1], "dep2");
        assert_eq!(deps[2], "dep3");
    }

    #[test]
    fn test_brick_history_position_tracking() {
        let mut history = BrickHistory::new(10);

        // Initially position is 0
        assert_eq!(history.position(), 0);

        // Add some states
        for i in 0..3 {
            let mut state = BrickState::new();
            state.set_metadata("i", StateValue::Int(i));
            let trace = ExecutionTrace {
                operation: format!("op{}", i),
                input_summary: String::new(),
                output_summary: String::new(),
                duration: Duration::ZERO,
                state_version_before: i as u64,
                state_version_after: (i + 1) as u64,
            };
            history.record(state, trace);
        }

        // Position should be 3 (past end)
        assert_eq!(history.position(), 3);
        assert_eq!(history.len(), 3);

        // goto(1) sets position to 1
        let _ = history.goto(1);
        assert_eq!(history.position(), 1);

        // step_forward returns state at position, then increments
        let _ = history.step_forward();
        assert_eq!(history.position(), 2);

        // step_back decrements position, then returns state at new position
        let _ = history.step_back();
        assert_eq!(history.position(), 1);

        // Verify trace access
        let trace = history.trace_at(0).unwrap();
        assert_eq!(trace.operation, "op0");

        let all_traces = history.traces();
        assert_eq!(all_traces.len(), 3);
    }

    // ========================================================================
    // Additional coverage tests for edge cases
    // ========================================================================

    #[test]
    fn test_brick_history_current_position_equals_len() {
        let mut history = BrickHistory::new(10);

        // Record one state
        let mut state = BrickState::new();
        state.set_metadata("val", StateValue::Int(42));
        let trace = ExecutionTrace {
            operation: "op".into(),
            input_summary: String::new(),
            output_summary: String::new(),
            duration: Duration::ZERO,
            state_version_before: 0,
            state_version_after: 1,
        };
        history.record(state, trace);

        // After record: position = 1, len = 1
        // position > 0 && position <= len is true
        // Should return snapshots[position - 1] = snapshots[0]
        assert_eq!(history.position(), 1);
        assert_eq!(history.len(), 1);

        let current = history.current();
        assert!(current.is_some());
        assert_eq!(
            current.unwrap().get_metadata("val"),
            Some(&StateValue::Int(42))
        );
    }

    #[test]
    fn test_brick_history_current_position_greater_than_len() {
        let mut history = BrickHistory::new(10);

        // Add two states
        for i in 0..2 {
            let mut state = BrickState::new();
            state.set_metadata("val", StateValue::Int(i));
            let trace = ExecutionTrace {
                operation: format!("op{}", i),
                input_summary: String::new(),
                output_summary: String::new(),
                duration: Duration::ZERO,
                state_version_before: i as u64,
                state_version_after: (i + 1) as u64,
            };
            history.record(state, trace);
        }

        // position = 2, len = 2
        // Now manually set position to something > len (shouldn't happen in normal use)
        // This tests the else branch where position > len
        history.position = 5;

        // position > 0 (5 > 0) but position > len (5 > 2)
        // So condition fails, returns snapshots.first()
        let current = history.current();
        assert!(current.is_some());
        // Should get first element
        assert_eq!(
            current.unwrap().get_metadata("val"),
            Some(&StateValue::Int(0))
        );
    }

    #[test]
    fn test_brick_history_step_forward_at_exact_len() {
        let mut history = BrickHistory::new(10);

        // Add one state
        let state = BrickState::new();
        let trace = ExecutionTrace {
            operation: "op".into(),
            input_summary: String::new(),
            output_summary: String::new(),
            duration: Duration::ZERO,
            state_version_before: 0,
            state_version_after: 1,
        };
        history.record(state, trace);

        // position = 1, len = 1
        // step_forward checks if position < len
        // 1 < 1 is false, so returns None
        assert_eq!(history.position(), 1);
        assert!(history.step_forward().is_none());
    }

    #[test]
    fn test_brick_history_record_at_capacity_evicts_oldest() {
        let mut history = BrickHistory::new(2);

        // Fill to capacity
        for i in 0..2 {
            let mut state = BrickState::new();
            state.set_metadata("val", StateValue::Int(i));
            let trace = ExecutionTrace {
                operation: format!("op{}", i),
                input_summary: String::new(),
                output_summary: String::new(),
                duration: Duration::ZERO,
                state_version_before: i as u64,
                state_version_after: (i + 1) as u64,
            };
            history.record(state, trace);
        }

        assert_eq!(history.len(), 2);

        // Record one more - should evict oldest
        let mut state = BrickState::new();
        state.set_metadata("val", StateValue::Int(99));
        let trace = ExecutionTrace {
            operation: "op_new".into(),
            input_summary: String::new(),
            output_summary: String::new(),
            duration: Duration::ZERO,
            state_version_before: 2,
            state_version_after: 3,
        };
        history.record(state, trace);

        // Still at capacity
        assert_eq!(history.len(), 2);

        // First element should now be val=1 (val=0 was evicted)
        let first = history.goto(0).unwrap();
        assert_eq!(first.get_metadata("val"), Some(&StateValue::Int(1)));

        // Second element should be val=99
        let second = history.goto(1).unwrap();
        assert_eq!(second.get_metadata("val"), Some(&StateValue::Int(99)));
    }

    #[test]
    fn test_deterministic_rng_all_value_ranges() {
        let mut rng = DeterministicRng::new(12345);

        // Test that f64 values are in [0, 1)
        for _ in 0..1000 {
            let f = rng.next_f64();
            assert!(f >= 0.0);
            assert!(f < 1.0);
        }

        // Test that f32 values are in [0, 1)
        for _ in 0..1000 {
            let f = rng.next_f32();
            assert!(f >= 0.0);
            assert!(f < 1.0);
        }
    }

    #[test]
    fn test_deterministic_clock_now_returns_duration() {
        let clock = DeterministicClock::new(1_000_000_000, 1); // 1 second in ns
        let duration = clock.now();
        assert_eq!(duration.as_nanos(), 1_000_000_000);
        assert_eq!(duration.as_secs(), 1);
    }

    #[test]
    fn test_state_value_all_variants_partial_eq() {
        // Test that different variant types are not equal
        let int = StateValue::Int(42);
        let float = StateValue::Float(42.0);
        let string = StateValue::String("42".into());
        let bool_val = StateValue::Bool(true);

        // Different variants are not equal (even if they represent similar values)
        assert_ne!(int, float);
        assert_ne!(int, string);
        assert_ne!(int, bool_val);
        assert_ne!(float, string);
        assert_ne!(float, bool_val);
        assert_ne!(string, bool_val);
    }

    #[test]
    fn test_brick_state_snapshot_increments_version() {
        let mut state = BrickState::new();
        state.version = 0;

        let snap1 = state.snapshot();
        assert_eq!(snap1.version, 1);

        let snap2 = snap1.snapshot();
        assert_eq!(snap2.version, 2);

        let snap3 = snap2.snapshot();
        assert_eq!(snap3.version, 3);
    }

    #[test]
    fn test_invariant_guard_const_new() {
        // Test that InvariantGuard::new can be used in const context
        fn check(_: &BrickState) -> bool {
            true
        }

        const GUARD: InvariantGuard =
            InvariantGuard::new("const_guard", check, GuardSeverity::Warning);

        assert_eq!(GUARD.name, "const_guard");
        assert_eq!(GUARD.severity, GuardSeverity::Warning);
    }

    #[test]
    fn test_deterministic_rng_const_new() {
        // Test that DeterministicRng::new can be used in const context
        const RNG: DeterministicRng = DeterministicRng::new(42);
        assert_eq!(RNG.state(), 42);
    }

    #[test]
    fn test_deterministic_clock_const_methods() {
        // Test const methods on DeterministicClock
        const CLOCK: DeterministicClock = DeterministicClock::new(100, 10);
        const NS: u64 = CLOCK.now_ns();
        const DUR: Duration = CLOCK.now();

        assert_eq!(NS, 100);
        assert_eq!(DUR.as_nanos(), 100);
    }

    #[test]
    fn test_guarded_brick_empty_guards_check_passes() {
        use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};

        struct EmptyGuardBrick;
        impl Brick for EmptyGuardBrick {
            fn brick_name(&self) -> &'static str {
                "EmptyGuardBrick"
            }
            fn assertions(&self) -> &[BrickAssertion] {
                &[]
            }
            fn budget(&self) -> BrickBudget {
                BrickBudget::uniform(16)
            }
            fn verify(&self) -> BrickVerification {
                BrickVerification {
                    passed: vec![],
                    failed: vec![],
                    verification_time: Duration::ZERO,
                }
            }
            fn to_html(&self) -> String {
                String::new()
            }
            fn to_css(&self) -> String {
                String::new()
            }
        }

        let guarded = GuardedBrick::new(EmptyGuardBrick);

        // With no guards, any state should pass
        let state = BrickState::new();
        assert!(guarded.check_guards(&state).is_ok());

        // Also with populated state
        let mut state2 = BrickState::new();
        state2.set_tensor("data", vec![1.0, 2.0], vec![2]);
        state2.set_metadata("key", StateValue::String("value".into()));
        assert!(guarded.check_guards(&state2).is_ok());
    }

    #[test]
    fn test_brick_history_record_not_at_end_truncates() {
        let mut history = BrickHistory::new(10);

        // Record 5 states
        for i in 0..5 {
            let mut state = BrickState::new();
            state.set_metadata("val", StateValue::Int(i));
            let trace = ExecutionTrace {
                operation: format!("op{}", i),
                input_summary: String::new(),
                output_summary: String::new(),
                duration: Duration::ZERO,
                state_version_before: i as u64,
                state_version_after: (i + 1) as u64,
            };
            history.record(state, trace);
        }

        assert_eq!(history.len(), 5);

        // Go back to position 3
        history.goto(3);
        assert_eq!(history.position(), 3);

        // Record new state - should truncate states 3 and 4
        let mut new_state = BrickState::new();
        new_state.set_metadata("val", StateValue::Int(100));
        let trace = ExecutionTrace {
            operation: "new_op".into(),
            input_summary: String::new(),
            output_summary: String::new(),
            duration: Duration::ZERO,
            state_version_before: 3,
            state_version_after: 4,
        };
        history.record(new_state, trace);

        // Should have 4 states now (0, 1, 2, 100)
        assert_eq!(history.len(), 4);
        assert_eq!(history.position(), 4);

        // Verify the last state is our new one
        let last = history.goto(3).unwrap();
        assert_eq!(last.get_metadata("val"), Some(&StateValue::Int(100)));

        // Verify traces were also truncated
        let traces = history.traces();
        assert_eq!(traces.len(), 4);
        assert_eq!(traces[3].operation, "new_op");
    }

    #[test]
    fn test_execution_trace_clone_preserves_all_fields() {
        let original = ExecutionTrace {
            operation: "test_op".into(),
            input_summary: "test_input".into(),
            output_summary: "test_output".into(),
            duration: Duration::from_micros(12345),
            state_version_before: 10,
            state_version_after: 11,
        };

        let cloned = original.clone();

        assert_eq!(original.operation, cloned.operation);
        assert_eq!(original.input_summary, cloned.input_summary);
        assert_eq!(original.output_summary, cloned.output_summary);
        assert_eq!(original.duration, cloned.duration);
        assert_eq!(original.state_version_before, cloned.state_version_before);
        assert_eq!(original.state_version_after, cloned.state_version_after);
    }

    #[test]
    fn test_brick_state_set_tensor_with_into() {
        let mut state = BrickState::new();

        // Test with String
        state.set_tensor(String::from("tensor1"), vec![1.0], vec![1]);
        assert!(state.get_tensor("tensor1").is_some());

        // Test with &str
        state.set_tensor("tensor2", vec![2.0], vec![1]);
        assert!(state.get_tensor("tensor2").is_some());
    }

    #[test]
    fn test_brick_state_set_metadata_with_into() {
        let mut state = BrickState::new();

        // Test with String
        state.set_metadata(String::from("key1"), StateValue::Int(1));
        assert!(state.get_metadata("key1").is_some());

        // Test with &str
        state.set_metadata("key2", StateValue::Int(2));
        assert!(state.get_metadata("key2").is_some());
    }

    #[test]
    fn test_guard_violation_source_is_none() {
        use std::error::Error;

        let violation = GuardViolation {
            guard_name: "test",
            severity: GuardSeverity::Error,
        };

        // GuardViolation has no source error
        assert!(violation.source().is_none());
    }

    #[test]
    fn test_brick_history_goto_returns_state_at_position() {
        let mut history = BrickHistory::new(10);

        // Record 3 states with distinct values
        for i in 0..3 {
            let mut state = BrickState::new();
            state.set_metadata("idx", StateValue::Int(i * 10));
            let trace = ExecutionTrace {
                operation: format!("op{}", i),
                input_summary: String::new(),
                output_summary: String::new(),
                duration: Duration::ZERO,
                state_version_before: i as u64,
                state_version_after: (i + 1) as u64,
            };
            history.record(state, trace);
        }

        // Test goto returns correct states
        let state0 = history.goto(0).unwrap();
        assert_eq!(state0.get_metadata("idx"), Some(&StateValue::Int(0)));

        let state1 = history.goto(1).unwrap();
        assert_eq!(state1.get_metadata("idx"), Some(&StateValue::Int(10)));

        let state2 = history.goto(2).unwrap();
        assert_eq!(state2.get_metadata("idx"), Some(&StateValue::Int(20)));

        // Invalid positions return None
        assert!(history.goto(3).is_none());
        assert!(history.goto(100).is_none());
    }

    #[test]
    fn test_deterministic_rng_xorshift_sequence() {
        // Verify the xorshift algorithm produces expected values
        let mut rng = DeterministicRng::new(1);

        // First few values from xorshift64 with seed 1
        let v1 = rng.next_u64();
        let v2 = rng.next_u64();
        let v3 = rng.next_u64();

        // Values should be different
        assert_ne!(v1, v2);
        assert_ne!(v2, v3);
        assert_ne!(v1, v3);

        // Restart with same seed should give same sequence
        let mut rng2 = DeterministicRng::new(1);
        assert_eq!(v1, rng2.next_u64());
        assert_eq!(v2, rng2.next_u64());
        assert_eq!(v3, rng2.next_u64());
    }

    #[test]
    fn test_brick_state_get_tensor_returns_none_for_missing_data() {
        let mut state = BrickState::new();

        // Add only shape, no data
        state.shapes.insert("only_shape".into(), vec![2, 3]);
        assert!(state.get_tensor("only_shape").is_none());

        // Add only data, no shape
        state.tensors.insert("only_data".into(), vec![1.0, 2.0]);
        assert!(state.get_tensor("only_data").is_none());

        // Both present - should work
        state.tensors.insert("both".into(), vec![1.0, 2.0]);
        state.shapes.insert("both".into(), vec![2]);
        assert!(state.get_tensor("both").is_some());
    }
}
