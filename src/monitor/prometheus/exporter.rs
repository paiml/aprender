//! Prometheus metrics exporter implementation.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

use super::types::{LabelSet, MetricDef, MetricType, MetricValue};

/// Prometheus metrics exporter for training monitoring
#[derive(Debug)]
pub struct PrometheusExporter {
    /// Default labels applied to all metrics
    default_labels: LabelSet,
    /// Metric definitions
    pub(crate) definitions: HashMap<String, MetricDef>,
    /// Current metric values
    values: RwLock<HashMap<String, Vec<MetricValue>>>,
    /// Total samples counter
    total_samples: AtomicU64,
}

impl PrometheusExporter {
    /// Create a new exporter with experiment and run labels
    pub fn new(experiment: &str, run: &str) -> Self {
        let default_labels = LabelSet::from_pairs(&[("experiment", experiment), ("run", run)]);

        let mut exporter = Self {
            default_labels,
            definitions: HashMap::new(),
            values: RwLock::new(HashMap::new()),
            total_samples: AtomicU64::new(0),
        };

        // Register default training metrics
        exporter.register_default_metrics();
        exporter
    }

    /// Create with custom default labels
    pub fn with_labels(labels: LabelSet) -> Self {
        let mut exporter = Self {
            default_labels: labels,
            definitions: HashMap::new(),
            values: RwLock::new(HashMap::new()),
            total_samples: AtomicU64::new(0),
        };

        exporter.register_default_metrics();
        exporter
    }

    /// Register default training metrics
    fn register_default_metrics(&mut self) {
        // Training metrics
        self.register(
            MetricDef::gauge("entrenar_epoch_loss", "Training loss per epoch")
                .with_labels(&["experiment", "run"]),
        );
        self.register(
            MetricDef::gauge("entrenar_validation_loss", "Validation loss per epoch")
                .with_labels(&["experiment", "run"]),
        );
        self.register(
            MetricDef::gauge("entrenar_learning_rate", "Current learning rate")
                .with_labels(&["experiment", "run"]),
        );
        self.register(
            MetricDef::gauge("entrenar_batch_throughput", "Batches processed per second")
                .with_labels(&["experiment", "run"]),
        );
        self.register(
            MetricDef::gauge("entrenar_validation_accuracy", "Validation accuracy")
                .with_labels(&["experiment", "run"]),
        );
        self.register(
            MetricDef::counter("entrenar_training_steps_total", "Total training steps completed")
                .with_labels(&["experiment", "run"]),
        );

        // GPU metrics
        self.register(
            MetricDef::gauge("entrenar_gpu_utilization", "GPU utilization percentage")
                .with_labels(&["experiment", "run", "device"]),
        );
        self.register(
            MetricDef::gauge("entrenar_gpu_memory_used_bytes", "GPU memory used in bytes")
                .with_labels(&["experiment", "run", "device"]),
        );
        self.register(
            MetricDef::gauge("entrenar_gpu_temperature_celsius", "GPU temperature in Celsius")
                .with_labels(&["experiment", "run", "device"]),
        );
        self.register(
            MetricDef::gauge("entrenar_gpu_power_watts", "GPU power draw in watts").with_labels(&[
                "experiment",
                "run",
                "device",
            ]),
        );

        // System metrics
        self.register(
            MetricDef::gauge("entrenar_memory_used_bytes", "Process memory usage in bytes")
                .with_labels(&["experiment", "run"]),
        );
    }

    /// Register a metric definition
    pub fn register(&mut self, def: MetricDef) {
        self.definitions.insert(def.name.clone(), def);
    }

    /// Record a metric value
    pub fn record(&self, name: &str, value: f64) {
        self.record_with_labels(name, value, self.default_labels.clone());
    }

    /// Record a metric value with additional labels
    pub fn record_with_labels(&self, name: &str, value: f64, mut labels: LabelSet) {
        // Merge default labels
        for (k, v) in &self.default_labels.values {
            if !labels.values.iter().any(|(lk, _)| lk == k) {
                labels.values.push((k.clone(), v.clone()));
            }
        }

        let metric_value = MetricValue { labels, value, timestamp: Some(current_timestamp_ms()) };

        if let Ok(mut values) = self.values.write() {
            values.entry(name.to_string()).or_default().push(metric_value);
        }

        self.total_samples.fetch_add(1, Ordering::Relaxed);
    }

    /// Record epoch metrics
    pub fn record_epoch(&self, epoch: u32, loss: f64, lr: f64) {
        self.record("entrenar_epoch_loss", loss);
        self.record("entrenar_learning_rate", lr);
        self.record("entrenar_training_steps_total", f64::from(epoch));
    }

    /// Record validation metrics
    pub fn record_validation(&self, loss: f64, accuracy: f64) {
        self.record("entrenar_validation_loss", loss);
        self.record("entrenar_validation_accuracy", accuracy);
    }

    /// Record batch throughput
    pub fn record_batch(&self, batches_per_second: f64) {
        self.record("entrenar_batch_throughput", batches_per_second);
    }

    /// Record GPU metrics for a device
    pub fn record_gpu(
        &self,
        device_id: u32,
        utilization: f64,
        memory_bytes: f64,
        temp: f64,
        power: f64,
    ) {
        let labels = self.default_labels.clone().add("device", &device_id.to_string());

        self.record_with_labels("entrenar_gpu_utilization", utilization, labels.clone());
        self.record_with_labels("entrenar_gpu_memory_used_bytes", memory_bytes, labels.clone());
        self.record_with_labels("entrenar_gpu_temperature_celsius", temp, labels.clone());
        self.record_with_labels("entrenar_gpu_power_watts", power, labels);
    }

    /// Record system memory usage
    pub fn record_memory(&self, used_bytes: f64) {
        self.record("entrenar_memory_used_bytes", used_bytes);
    }

    /// Get total samples recorded
    pub fn total_samples(&self) -> u64 {
        self.total_samples.load(Ordering::Relaxed)
    }

    /// Clear all recorded values
    pub fn clear(&self) {
        if let Ok(mut values) = self.values.write() {
            values.clear();
        }
    }

    /// Export metrics in Prometheus text format
    pub fn export(&self) -> String {
        let mut output = String::new();

        let values = match self.values.read() {
            Ok(v) => v,
            Err(_) => return output,
        };

        for (name, def) in &self.definitions {
            if let Some(metric_values) = values.get(name) {
                // Only export if we have values
                if metric_values.is_empty() {
                    continue;
                }

                // HELP line
                output.push_str(&format!("# HELP {} {}\n", name, def.help));

                // TYPE line
                let type_str = match def.metric_type {
                    MetricType::Gauge => "gauge",
                    MetricType::Counter => "counter",
                    MetricType::Histogram => "histogram",
                };
                output.push_str(&format!("# TYPE {name} {type_str}\n"));

                // Get latest value for each unique label set
                let mut latest: HashMap<String, &MetricValue> = HashMap::new();
                for mv in metric_values {
                    let key = mv.labels.format();
                    latest.insert(key, mv);
                }

                // Metric lines
                for mv in latest.values() {
                    let labels_str = mv.labels.format();
                    if let Some(ts) = mv.timestamp {
                        output.push_str(&format!("{}{} {} {}\n", name, labels_str, mv.value, ts));
                    } else {
                        output.push_str(&format!("{}{} {}\n", name, labels_str, mv.value));
                    }
                }
            }
        }

        output
    }

    /// Export metrics as JSON (for programmatic access)
    pub fn export_json(&self) -> String {
        let values = match self.values.read() {
            Ok(v) => v,
            Err(_) => return "{}".to_string(),
        };

        let mut metrics: HashMap<String, Vec<serde_json::Value>> = HashMap::new();

        for (name, metric_values) in values.iter() {
            let json_values: Vec<serde_json::Value> = metric_values
                .iter()
                .map(|mv| {
                    let mut obj = serde_json::Map::new();
                    for (k, v) in &mv.labels.values {
                        obj.insert(k.clone(), serde_json::Value::String(v.clone()));
                    }
                    obj.insert("value".to_string(), serde_json::json!(mv.value));
                    if let Some(ts) = mv.timestamp {
                        obj.insert("timestamp".to_string(), serde_json::json!(ts));
                    }
                    serde_json::Value::Object(obj)
                })
                .collect();

            metrics.insert(name.clone(), json_values);
        }

        serde_json::to_string_pretty(&metrics).unwrap_or_default()
    }
}

/// Get current timestamp in milliseconds
fn current_timestamp_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}
