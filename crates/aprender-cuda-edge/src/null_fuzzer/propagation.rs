//! Null pointer propagation tracking.
//!
//! [`PropagationTracker`] records the call chain when a null pointer
//! propagates through kernel execution, enabling root-cause analysis
//! of where null checking failed.

use serde::{Deserialize, Serialize};

/// A single frame in a null-pointer propagation path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PropagationFrame {
    /// Kernel or function name.
    pub function: String,
    /// Argument index that received the null.
    pub arg_index: u32,
}

/// The full path a null pointer took through kernel execution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PropagationPath {
    /// Ordered list of frames from injection point to final use.
    pub frames: Vec<PropagationFrame>,
}

impl PropagationPath {
    /// Returns the depth of propagation (number of frames).
    #[must_use]
    pub fn depth(&self) -> usize {
        self.frames.len()
    }

    /// Returns the injection point (first frame), if any.
    #[must_use]
    pub fn injection_point(&self) -> Option<&PropagationFrame> {
        self.frames.first()
    }

    /// Returns the final use point (last frame), if any.
    #[must_use]
    pub fn final_use(&self) -> Option<&PropagationFrame> {
        self.frames.last()
    }
}

/// Outcome of a null propagation test.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropagationOutcome {
    /// The null was caught and handled gracefully.
    Caught {
        /// Where the null was caught.
        at_frame: usize,
    },
    /// The null propagated to the end without being caught.
    Uncaught,
    /// The null caused a crash.
    Crash {
        /// Error message from the crash.
        message: String,
    },
}

/// Tracks null pointer propagation through kernel call chains.
#[derive(Debug, Clone, Default)]
pub struct PropagationTracker {
    current_path: PropagationPath,
    completed_paths: Vec<(PropagationPath, PropagationOutcome)>,
}

impl PropagationTracker {
    /// Create a new propagation tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Begin tracking a new propagation chain.
    pub fn enter(&mut self, function: String, arg_index: u32) {
        self.current_path.frames.push(PropagationFrame { function, arg_index });
    }

    /// Exit the current frame (pop from the path).
    pub fn exit(&mut self) {
        self.current_path.frames.pop();
    }

    /// Record the outcome of the current propagation chain and reset.
    pub fn record(&mut self, outcome: PropagationOutcome) {
        let path = std::mem::take(&mut self.current_path);
        self.completed_paths.push((path, outcome));
    }

    /// Returns all completed propagation records.
    #[must_use]
    pub fn completed(&self) -> &[(PropagationPath, PropagationOutcome)] {
        &self.completed_paths
    }

    /// Returns the current path depth.
    #[must_use]
    pub fn current_depth(&self) -> usize {
        self.current_path.depth()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn propagation_path_depth() {
        let mut path = PropagationPath::default();
        assert_eq!(path.depth(), 0);
        path.frames.push(PropagationFrame { function: "kernel_a".into(), arg_index: 0 });
        assert_eq!(path.depth(), 1);
    }

    #[test]
    fn tracker_enter_exit() {
        let mut tracker = PropagationTracker::new();
        tracker.enter("kernel_a".into(), 0);
        assert_eq!(tracker.current_depth(), 1);
        tracker.enter("kernel_b".into(), 1);
        assert_eq!(tracker.current_depth(), 2);
        tracker.exit();
        assert_eq!(tracker.current_depth(), 1);
    }

    #[test]
    fn tracker_record_completes_path() {
        let mut tracker = PropagationTracker::new();
        tracker.enter("kernel_a".into(), 0);
        tracker.enter("kernel_b".into(), 1);
        tracker.record(PropagationOutcome::Uncaught);

        assert_eq!(tracker.completed().len(), 1);
        assert_eq!(tracker.current_depth(), 0);

        let (path, outcome) = &tracker.completed()[0];
        assert_eq!(path.depth(), 2);
        assert_eq!(outcome, &PropagationOutcome::Uncaught);
    }

    #[test]
    fn injection_and_final_use_points() {
        let path = PropagationPath {
            frames: vec![
                PropagationFrame { function: "entry".into(), arg_index: 0 },
                PropagationFrame { function: "deep".into(), arg_index: 2 },
            ],
        };
        assert_eq!(path.injection_point().unwrap().function, "entry");
        assert_eq!(path.final_use().unwrap().function, "deep");
    }
}
