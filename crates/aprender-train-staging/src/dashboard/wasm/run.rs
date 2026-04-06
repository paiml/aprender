//! WASM-compatible run wrapper.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wasm_bindgen::prelude::*;

use crate::storage::{ExperimentStorage, RunStatus};

use super::storage::IndexedDbStorage;

/// WASM-compatible run wrapper.
///
/// Provides a JavaScript-friendly API for training runs.
#[wasm_bindgen]
pub struct WasmRun {
    run_id: String,
    experiment_id: String,
    storage: Arc<Mutex<IndexedDbStorage>>,
    step_counters: HashMap<String, u64>,
    finished: bool,
}

#[wasm_bindgen]
impl WasmRun {
    /// Create a new run in a new experiment.
    #[wasm_bindgen(constructor)]
    pub fn new(experiment_name: &str) -> std::result::Result<WasmRun, JsValue> {
        let mut storage = IndexedDbStorage::new();

        let experiment_id = storage
            .create_experiment(experiment_name, None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let run_id =
            storage.create_run(&experiment_id).map_err(|e| JsValue::from_str(&e.to_string()))?;

        storage.start_run(&run_id).map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(Self {
            run_id,
            experiment_id,
            storage: Arc::new(Mutex::new(storage)),
            step_counters: HashMap::new(),
            finished: false,
        })
    }

    /// Log a metric value, auto-incrementing the step.
    pub fn log_metric(&mut self, key: &str, value: f64) -> std::result::Result<(), JsValue> {
        if self.finished {
            return Err(JsValue::from_str("Cannot log to finished run"));
        }

        let step = *self.step_counters.get(key).unwrap_or(&0);
        self.log_metric_at(key, step, value)?;
        self.step_counters.insert(key.to_string(), step + 1);
        Ok(())
    }

    /// Log a metric value at a specific step.
    pub fn log_metric_at(
        &mut self,
        key: &str,
        step: u64,
        value: f64,
    ) -> std::result::Result<(), JsValue> {
        if self.finished {
            return Err(JsValue::from_str("Cannot log to finished run"));
        }

        self.storage
            .lock()
            .map_err(|e| JsValue::from_str(&e.to_string()))?
            .log_metric(&self.run_id, key, step, value)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(())
    }

    /// Get all metrics as a JSON string.
    pub fn get_metrics_json(&self) -> std::result::Result<String, JsValue> {
        let storage = self.storage.lock().map_err(|e| JsValue::from_str(&e.to_string()))?;

        let keys = storage.list_metric_keys(&self.run_id);
        let mut metrics: HashMap<String, Vec<serde_json::Value>> = HashMap::new();

        for key in keys {
            if let Ok(points) = storage.get_metrics(&self.run_id, &key) {
                let values: Vec<serde_json::Value> = points
                    .iter()
                    .map(|p| {
                        serde_json::json!({
                            "step": p.step,
                            "value": p.value,
                            "timestamp": p.timestamp.to_rfc3339()
                        })
                    })
                    .collect();
                metrics.insert(key, values);
            }
        }

        serde_json::to_string(&metrics).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Subscribe to metric updates via a JavaScript callback.
    ///
    /// The callback receives (key: string, value: number) for each update.
    pub fn subscribe_metrics(&self, _callback: &js_sys::Function) {
        // In a full implementation, this would store the callback
        // and invoke it when metrics are logged.
        // For now, this is a placeholder showing the API.
    }

    /// Get the run ID.
    pub fn run_id(&self) -> String {
        self.run_id.clone()
    }

    /// Get the experiment ID.
    pub fn experiment_id(&self) -> String {
        self.experiment_id.clone()
    }

    /// Get current step for a metric key.
    pub fn current_step(&self, key: &str) -> u64 {
        *self.step_counters.get(key).unwrap_or(&0)
    }

    /// Check if the run is finished.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Finish the run with success status.
    pub fn finish(&mut self) -> std::result::Result<(), JsValue> {
        if self.finished {
            return Ok(());
        }

        self.storage
            .lock()
            .map_err(|e| JsValue::from_str(&e.to_string()))?
            .complete_run(&self.run_id, RunStatus::Success)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        self.finished = true;
        Ok(())
    }

    /// Finish the run with failed status.
    pub fn fail(&mut self) -> std::result::Result<(), JsValue> {
        if self.finished {
            return Ok(());
        }

        self.storage
            .lock()
            .map_err(|e| JsValue::from_str(&e.to_string()))?
            .complete_run(&self.run_id, RunStatus::Failed)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        self.finished = true;
        Ok(())
    }
}
