//! Experiment tracking — group training runs, compare metrics.
//!
//! An experiment is a named collection of training runs that can be compared.
//! This enables the iterate loop: train → eval → compare → retrain.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// An experiment (group of training runs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    pub id: String,
    pub name: String,
    pub description: String,
    pub run_ids: Vec<String>,
    pub created_at: u64,
}

/// Comparison of runs within an experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunComparison {
    pub experiment_id: String,
    pub runs: Vec<RunSummary>,
    pub best_run: Option<String>,
}

/// Summary of a single run for comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    pub id: String,
    pub method: String,
    pub status: String,
    pub final_loss: Option<f32>,
    pub total_steps: u64,
}

/// Experiment store with optional disk persistence.
pub struct ExperimentStore {
    experiments: RwLock<HashMap<String, Experiment>>,
    counter: std::sync::atomic::AtomicU64,
    data_dir: Option<std::path::PathBuf>,
}

impl ExperimentStore {
    /// Create in-memory experiment store.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            experiments: RwLock::new(HashMap::new()),
            counter: std::sync::atomic::AtomicU64::new(0),
            data_dir: None,
        })
    }

    /// Create experiment store with disk persistence.
    #[must_use]
    pub fn with_data_dir(dir: std::path::PathBuf) -> Arc<Self> {
        let _ = std::fs::create_dir_all(&dir);
        let mut experiments = HashMap::new();

        // Load existing experiments from disk
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                if entry.path().extension().is_some_and(|e| e == "json") {
                    if let Ok(data) = std::fs::read_to_string(entry.path()) {
                        if let Ok(exp) = serde_json::from_str::<Experiment>(&data) {
                            experiments.insert(exp.id.clone(), exp);
                        }
                    }
                }
            }
        }

        let count = experiments.len() as u64;
        if count > 0 {
            eprintln!("[banco] Loaded {count} experiments from {}", dir.display());
        }

        Arc::new(Self {
            experiments: RwLock::new(experiments),
            counter: std::sync::atomic::AtomicU64::new(count),
            data_dir: Some(dir),
        })
    }

    /// Create a new experiment.
    pub fn create(&self, name: &str, description: &str) -> Experiment {
        let seq = self.counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let exp = Experiment {
            id: format!("exp-{}-{seq}", epoch_secs()),
            name: name.to_string(),
            description: description.to_string(),
            run_ids: Vec::new(),
            created_at: epoch_secs(),
        };
        if let Ok(mut store) = self.experiments.write() {
            store.insert(exp.id.clone(), exp.clone());
        }
        self.persist(&exp);
        exp
    }

    /// Add a run to an experiment.
    pub fn add_run(&self, experiment_id: &str, run_id: &str) -> Result<(), ExperimentError> {
        let mut store = self.experiments.write().map_err(|_| ExperimentError::LockPoisoned)?;
        let exp = store
            .get_mut(experiment_id)
            .ok_or(ExperimentError::NotFound(experiment_id.to_string()))?;
        if !exp.run_ids.contains(&run_id.to_string()) {
            exp.run_ids.push(run_id.to_string());
        }
        let exp_clone = exp.clone();
        drop(store);
        self.persist(&exp_clone);
        Ok(())
    }

    /// Persist an experiment to disk (if data_dir is set).
    fn persist(&self, exp: &Experiment) {
        if let Some(dir) = &self.data_dir {
            let path = dir.join(format!("{}.json", exp.id));
            if let Ok(json) = serde_json::to_string_pretty(exp) {
                let _ = std::fs::write(path, json);
            }
        }
    }

    /// List all experiments.
    #[must_use]
    pub fn list(&self) -> Vec<Experiment> {
        let store = self.experiments.read().unwrap_or_else(|e| e.into_inner());
        let mut exps: Vec<Experiment> = store.values().cloned().collect();
        exps.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        exps
    }

    /// Get experiment by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<Experiment> {
        self.experiments.read().unwrap_or_else(|e| e.into_inner()).get(id).cloned()
    }

    /// Compare runs in an experiment using the training store.
    pub fn compare(
        &self,
        experiment_id: &str,
        training: &super::training::TrainingStore,
    ) -> Result<RunComparison, ExperimentError> {
        let exp =
            self.get(experiment_id).ok_or(ExperimentError::NotFound(experiment_id.to_string()))?;

        let mut summaries = Vec::new();
        let mut best_loss = f32::MAX;
        let mut best_id = None;

        for run_id in &exp.run_ids {
            if let Some(run) = training.get(run_id) {
                let final_loss = run.metrics.last().map(|m| m.loss);
                let total_steps = run.metrics.last().map(|m| m.step).unwrap_or(0);

                if let Some(loss) = final_loss {
                    if loss < best_loss {
                        best_loss = loss;
                        best_id = Some(run_id.clone());
                    }
                }

                summaries.push(RunSummary {
                    id: run_id.clone(),
                    method: format!("{:?}", run.method),
                    status: format!("{:?}", run.status),
                    final_loss,
                    total_steps,
                });
            }
        }

        Ok(RunComparison {
            experiment_id: experiment_id.to_string(),
            runs: summaries,
            best_run: best_id,
        })
    }
}

/// Experiment errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExperimentError {
    NotFound(String),
    LockPoisoned,
}

impl std::fmt::Display for ExperimentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "Experiment not found: {id}"),
            Self::LockPoisoned => write!(f, "Internal lock error"),
        }
    }
}

impl std::error::Error for ExperimentError {}

fn epoch_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
}
