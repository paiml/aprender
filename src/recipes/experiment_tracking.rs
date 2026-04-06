//! Experiment tracking recipe implementation.

use crate::experiment::{
    EnergyMetrics, ExperimentError, ExperimentRun, ExperimentStorage, ModelParadigm,
};
use crate::recipes::{ExperimentTrackingConfig, RecipeResult};

/// Experiment tracking recipe
#[derive(Debug)]
pub struct ExperimentTrackingRecipe {
    config: ExperimentTrackingConfig,
    current_run: Option<ExperimentRun>,
    start_time: Option<std::time::Instant>,
}

impl ExperimentTrackingRecipe {
    /// Create a new experiment tracking recipe
    pub fn new(config: ExperimentTrackingConfig) -> Self {
        Self { config, current_run: None, start_time: None }
    }

    /// Start a new experiment run
    pub fn start_run(&mut self, run_id: impl Into<String>) -> &mut ExperimentRun {
        let mut run = ExperimentRun::new(
            run_id,
            &self.config.experiment_name,
            self.config.paradigm,
            self.config.device.clone(),
        );
        run.platform = self.config.platform;
        run.tags = self.config.tags.clone();
        self.current_run = Some(run);
        self.start_time = Some(std::time::Instant::now());
        self.current_run.as_mut().expect("current_run was just set to Some")
    }

    /// Log a metric to the current run
    pub fn log_metric(
        &mut self,
        name: impl Into<String>,
        value: f64,
    ) -> Result<(), ExperimentError> {
        self.current_run
            .as_mut()
            .ok_or_else(|| ExperimentError::StorageError("No active run".to_string()))?
            .log_metric(name, value);
        Ok(())
    }

    /// Log a hyperparameter
    pub fn log_param(
        &mut self,
        name: impl Into<String>,
        value: serde_json::Value,
    ) -> Result<(), ExperimentError> {
        self.current_run
            .as_mut()
            .ok_or_else(|| ExperimentError::StorageError("No active run".to_string()))?
            .log_param(name, value);
        Ok(())
    }

    /// End the current run and calculate metrics
    pub fn end_run(&mut self, success: bool) -> Result<RecipeResult, ExperimentError> {
        let run = self
            .current_run
            .as_mut()
            .ok_or_else(|| ExperimentError::StorageError("No active run".to_string()))?;

        if success {
            run.complete();
        } else {
            run.fail();
        }

        let duration = self.start_time.take().map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);

        // Calculate energy metrics if enabled
        if self.config.track_energy {
            let power = self.config.device.estimated_power_watts() as f64;
            let energy_joules = power * duration;
            let mut energy = EnergyMetrics::new(energy_joules, power, power * 1.2, duration);

            if let Some(carbon_intensity) = self.config.carbon_intensity {
                energy = energy.with_carbon_intensity(carbon_intensity);
            }

            run.energy = Some(energy);
        }

        let mut result = RecipeResult::success("experiment-tracking");
        result = result.with_metric("duration_seconds", duration);

        if let Some(ref energy) = run.energy {
            result = result.with_metric("energy_joules", energy.total_joules);
            if let Some(co2) = energy.co2_grams {
                result = result.with_metric("co2_grams", co2);
            }
        }

        for (name, value) in &run.metrics {
            result = result.with_metric(format!("run_{}", name), *value);
        }

        Ok(result)
    }

    /// Get the current run
    pub fn current_run(&self) -> Option<&ExperimentRun> {
        self.current_run.as_ref()
    }

    /// Store the run to a backend
    pub fn store_run<S: ExperimentStorage>(&self, storage: &S) -> Result<(), ExperimentError> {
        if let Some(ref run) = self.current_run {
            storage.store_run(run)?;
        }
        Ok(())
    }
}
