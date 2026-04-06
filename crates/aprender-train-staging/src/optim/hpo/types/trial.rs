//! Trial types for HPO

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::parameter::ParameterValue;

/// A single trial (configuration + score)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trial {
    /// Trial ID
    pub id: usize,
    /// Parameter configuration
    pub config: HashMap<String, ParameterValue>,
    /// Objective score (lower is better by default)
    pub score: f64,
    /// Number of epochs/iterations used
    pub iterations: usize,
    /// Trial status
    pub status: TrialStatus,
}

impl Trial {
    /// Create a new trial
    pub fn new(id: usize, config: HashMap<String, ParameterValue>) -> Self {
        Self { id, config, score: f64::INFINITY, iterations: 0, status: TrialStatus::Pending }
    }

    /// Mark trial as complete with score
    pub fn complete(&mut self, score: f64, iterations: usize) {
        self.score = score;
        self.iterations = iterations;
        self.status = TrialStatus::Completed;
    }

    /// Mark trial as failed
    pub fn fail(&mut self) {
        self.status = TrialStatus::Failed;
    }
}

/// Trial status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrialStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Pruned,
}
