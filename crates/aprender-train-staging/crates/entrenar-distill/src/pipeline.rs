//! Distillation pipeline execution (Heijunka - level scheduling).
//!
//! Orchestrates the complete distillation workflow from model fetching
//! through training and export.

use crate::config::{DistillConfig, WeightFormat};
use crate::weights::load_safetensors_weights;
use crate::MemoryEstimate;
use entrenar::distill::{save_student_checkpoint, DistillationCheckpoint, DistillationLoss};
use entrenar_common::{EntrenarError, Result};
use ndarray::{Array2, Axis};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Pipeline execution result.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Path to the output model
    pub output_path: PathBuf,
    /// Training metrics
    pub metrics: TrainingMetrics,
    /// Total execution time in seconds
    pub duration_seconds: f64,
}

/// Training metrics collected during distillation.
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    /// Initial loss at start of training
    pub initial_loss: f32,
    /// Final loss at end of training
    pub final_loss: f32,
    /// Best validation loss achieved
    pub best_loss: f32,
    /// Number of training steps completed
    pub steps_completed: u64,
    /// Average throughput (samples/second)
    pub throughput: f32,
}

impl TrainingMetrics {
    /// Calculate loss improvement ratio.
    pub fn improvement_ratio(&self) -> f32 {
        if self.initial_loss > 0.0 {
            1.0 - (self.final_loss / self.initial_loss)
        } else {
            0.0
        }
    }
}

/// Distillation pipeline orchestrator.
pub struct Pipeline<'a> {
    config: &'a DistillConfig,
}

impl<'a> Pipeline<'a> {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: &'a DistillConfig) -> Self {
        Self { config }
    }

    /// Execute the complete distillation pipeline.
    pub fn execute(&self) -> Result<PipelineResult> {
        let start = std::time::Instant::now();

        // Stage 1: Fetch/resolve models
        let teacher_path = self.fetch_teacher()?;
        let student_path = self.fetch_student()?;

        // Stage 2: Train with distillation loss
        let (metrics, student_weights, student_shapes) =
            self.train(&teacher_path, &student_path)?;

        // Stage 3: Export student checkpoint
        let output_path = self.export(&student_weights, &student_shapes, &metrics)?;

        Ok(PipelineResult { output_path, metrics, duration_seconds: start.elapsed().as_secs_f64() })
    }

    /// Estimate memory requirements for this configuration.
    pub fn estimate_memory(config: &DistillConfig) -> Result<MemoryEstimate> {
        let teacher_params = estimate_params_from_model_id(&config.teacher.model_id);
        let student_params = estimate_params_from_model_id(&config.student.model_id);

        let estimate = MemoryEstimate::new(
            student_params + teacher_params / 4,
            config.training.batch_size as usize,
            config.dataset.max_length,
            4096,
        );

        Ok(estimate)
    }

    /// Resolve teacher model to a local path.
    ///
    /// If the model_id looks like a local path, validates it exists.
    /// If it's a HuggingFace model ID (org/model), downloads via HfModelFetcher
    /// when the `hub` feature is enabled.
    fn fetch_teacher(&self) -> Result<PathBuf> {
        resolve_model_path(&self.config.teacher.model_id)
    }

    /// Resolve student model to a local path.
    fn fetch_student(&self) -> Result<PathBuf> {
        resolve_model_path(&self.config.student.model_id)
    }

    /// Run the distillation training loop.
    ///
    /// Loads teacher and student weights from SafeTensors files, computes
    /// distillation loss via `DistillationLoss`, and applies gradient updates
    /// to student parameters.
    ///
    /// Note: Without a full transformer forward pass (autograd backward ops
    /// are incomplete), logits are derived from loaded weight tensor slices
    /// as a demonstration of the real loss computation pipeline.
    fn train(
        &self,
        teacher_path: &Path,
        student_path: &Path,
    ) -> Result<(TrainingMetrics, HashMap<String, Vec<f32>>, HashMap<String, Vec<usize>>)> {
        // Load weights from both models
        let (teacher_weights, _teacher_shapes) = load_safetensors_weights(teacher_path)?;
        let (mut student_weights, student_shapes) = load_safetensors_weights(student_path)?;

        // Create distillation loss function
        let temperature = self.config.distillation.temperature;
        let alpha = self.config.distillation.alpha;
        let loss_fn = DistillationLoss::new(temperature, alpha);

        let lr = self.config.training.learning_rate as f32;

        // Derive synthetic logits from weight tensors for loss computation.
        // In a full pipeline, these would come from forward passes through
        // teacher and student models. Here we use weight slices as logits
        // and apply proper KD gradient descent in logit space.
        let batch_size = self.config.training.batch_size as usize;
        let num_classes = 32; // Synthetic vocab slice

        let teacher_logits = build_synthetic_logits(&teacher_weights, batch_size, num_classes);
        let labels: Vec<usize> = (0..batch_size).map(|i| i % num_classes).collect();

        // Maintain student logits as a mutable array for gradient descent
        let mut student_logits = build_synthetic_logits(&student_weights, batch_size, num_classes);

        let mut metrics = TrainingMetrics::default();
        let mut best_loss = f32::MAX;

        // Initial loss measurement
        let initial_loss = loss_fn.forward(&student_logits, &teacher_logits, &labels);
        metrics.initial_loss = initial_loss;
        best_loss = best_loss.min(initial_loss);

        let train_start = std::time::Instant::now();
        let mut step = 0u64;

        for _epoch in 0..self.config.training.epochs {
            let steps_this_epoch = (1000 / u64::from(self.config.training.batch_size)).max(1);

            for _s in 0..steps_this_epoch {
                // Compute distillation loss
                let loss = loss_fn.forward(&student_logits, &teacher_logits, &labels);
                best_loss = best_loss.min(loss);

                // Compute KD gradient: d(loss)/d(student_logits)
                let grad =
                    kd_gradient(&student_logits, &teacher_logits, &labels, temperature, alpha);

                // SGD update in logit space
                student_logits = &student_logits - &(grad * lr);

                step += 1;
            }
        }

        let elapsed = train_start.elapsed().as_secs_f32().max(1e-6);

        // Final loss measurement
        let final_loss = loss_fn.forward(&student_logits, &teacher_logits, &labels);

        metrics.final_loss = final_loss;
        metrics.best_loss = best_loss.min(final_loss);
        metrics.steps_completed = step;
        metrics.throughput = (step as f32 * batch_size as f32) / elapsed;

        // Write trained logits back into student weight tensor
        write_logits_to_weights(&mut student_weights, &student_logits, batch_size, num_classes);

        Ok((metrics, student_weights, student_shapes))
    }

    /// Export trained student model using `save_student_checkpoint`.
    fn export(
        &self,
        weights: &HashMap<String, Vec<f32>>,
        shapes: &HashMap<String, Vec<usize>>,
        metrics: &TrainingMetrics,
    ) -> Result<PathBuf> {
        std::fs::create_dir_all(&self.config.output.dir).map_err(|e| EntrenarError::Io {
            context: format!("creating output directory: {}", self.config.output.dir.display()),
            source: e,
        })?;

        let checkpoint = DistillationCheckpoint {
            teacher_model: self.config.teacher.model_id.clone(),
            temperature: self.config.distillation.temperature,
            alpha: self.config.distillation.alpha,
            final_loss: Some(metrics.final_loss),
            epoch: self.config.training.epochs as usize,
            step: metrics.steps_completed as usize,
        };

        let filename = match self.config.output.format {
            WeightFormat::SafeTensors => "model.safetensors",
            WeightFormat::Gguf => "model.gguf",
            WeightFormat::Apr => "model.json",
        };

        // Save SafeTensors checkpoint with distillation metadata sidecar
        let output_path = save_student_checkpoint(
            weights,
            shapes,
            &checkpoint,
            &self.config.output.dir,
            filename,
        )
        .map_err(|e| EntrenarError::Io {
            context: "saving student checkpoint".to_string(),
            source: e,
        })?;

        // For GGUF format with hub feature, also export via Exporter
        #[cfg(feature = "hub")]
        if self.config.output.format == WeightFormat::Gguf {
            let mw = crate::weights::weights_to_model_weights(weights.clone(), shapes.clone());
            let exporter = entrenar::hf_pipeline::Exporter::new()
                .output_dir(&self.config.output.dir)
                .gguf_quantization(entrenar::hf_pipeline::GgufQuantization::Q8_0);
            exporter.export(&mw, entrenar::hf_pipeline::ExportFormat::GGUF, filename).map_err(
                |e| EntrenarError::Internal { message: format!("GGUF export failed: {e}") },
            )?;
        }

        Ok(output_path)
    }
}

/// Resolve a model identifier to a local filesystem path.
///
/// - If it contains `/` or `.` and exists on disk, returns the path directly.
/// - If it looks like a HuggingFace model ID (org/model), uses HfModelFetcher
///   when the `hub` feature is enabled.
/// - Otherwise returns an error.
fn resolve_model_path(model_id: &str) -> Result<PathBuf> {
    let path = Path::new(model_id);

    // Check if it's a local path that exists
    if path.exists() {
        return Ok(path.to_path_buf());
    }

    // If it looks like a local path but doesn't exist, error
    if model_id.starts_with('/')
        || model_id.starts_with("./")
        || model_id.starts_with("../")
        || model_id.ends_with(".safetensors")
        || model_id.ends_with(".gguf")
    {
        return Err(EntrenarError::ModelNotFound { path: path.to_path_buf() });
    }

    // Looks like a HuggingFace model ID (org/model)
    #[cfg(feature = "hub")]
    {
        let fetcher = entrenar::hf_pipeline::HfModelFetcher::new().map_err(|e| {
            EntrenarError::HuggingFace { message: format!("failed to initialize HF fetcher: {e}") }
        })?;

        let artifact = fetcher
            .download_model(model_id, entrenar::hf_pipeline::FetchOptions::default())
            .map_err(|e| EntrenarError::HuggingFace {
                message: format!("failed to download '{model_id}': {e}"),
            })?;

        Ok(artifact.path)
    }

    #[cfg(not(feature = "hub"))]
    {
        if model_id.contains('/') {
            return Err(EntrenarError::HuggingFace {
                message: format!(
                    "'{model_id}' looks like a HuggingFace model ID, but the 'hub' feature is not enabled. \
                     Rebuild with: cargo build -p entrenar-distill --features hub"
                ),
            });
        }

        Err(EntrenarError::ModelNotFound { path: path.to_path_buf() })
    }
}

/// Build synthetic logits from model weights for loss computation.
///
/// Takes the first weight tensor large enough and reshapes a slice of it
/// into [batch_size, num_classes]. This is a placeholder for real forward
/// pass outputs until the autograd backward ops are complete.
fn build_synthetic_logits(
    weights: &HashMap<String, Vec<f32>>,
    batch_size: usize,
    num_classes: usize,
) -> Array2<f32> {
    let needed = batch_size * num_classes;

    // Find a weight tensor large enough
    for data in weights.values() {
        if data.len() >= needed {
            let slice = &data[..needed];
            return Array2::from_shape_vec((batch_size, num_classes), slice.to_vec())
                .expect("shape matches needed elements");
        }
    }

    // Fallback: generate small random-like logits from whatever weights exist
    let mut logits = Vec::with_capacity(needed);
    let all_data: Vec<f32> = weights.values().flat_map(|v| v.iter().copied()).collect();
    for i in 0..needed {
        logits.push(if all_data.is_empty() {
            (i as f32 * 0.1) % 3.0 - 1.0
        } else {
            all_data[i % all_data.len()]
        });
    }

    Array2::from_shape_vec((batch_size, num_classes), logits)
        .expect("shape matches needed elements")
}

/// Compute the knowledge distillation gradient with respect to student logits.
///
/// The gradient of the KD loss L = α·T²·KL(teacher_T || student_T) + (1-α)·CE(student, labels):
///
/// ∂L/∂z_student = α·T·(softmax(z_s/T) - softmax(z_t/T))
///               + (1-α)·(softmax(z_s) - one_hot(labels))
fn kd_gradient(
    student_logits: &Array2<f32>,
    teacher_logits: &Array2<f32>,
    labels: &[usize],
    temperature: f32,
    alpha: f32,
) -> Array2<f32> {
    let batch_size = student_logits.nrows();
    let num_classes = student_logits.ncols();

    // Soft target gradient: α·T·(softmax(student/T) - softmax(teacher/T))
    let student_soft = softmax_2d(&(student_logits / temperature));
    let teacher_soft = softmax_2d(&(teacher_logits / temperature));
    let soft_grad = (&student_soft - &teacher_soft) * (alpha * temperature);

    // Hard target gradient: (1-α)·(softmax(student) - one_hot(labels))
    let student_hard = softmax_2d(student_logits);
    let mut one_hot = Array2::zeros((batch_size, num_classes));
    for (i, &label) in labels.iter().enumerate() {
        if label < num_classes {
            one_hot[[i, label]] = 1.0;
        }
    }
    let hard_grad = (&student_hard - &one_hot) * (1.0 - alpha);

    // Combined gradient
    &soft_grad + &hard_grad
}

/// Compute softmax along the last axis of a 2D array.
fn softmax_2d(x: &Array2<f32>) -> Array2<f32> {
    let mut result = x.clone();
    for mut row in result.axis_iter_mut(Axis(0)) {
        let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        row.mapv_inplace(|v| (v - max_val).exp());
        let sum: f32 = row.sum();
        row.mapv_inplace(|v| v / sum);
    }
    result
}

/// Write trained logit values back into the first suitable weight tensor.
fn write_logits_to_weights(
    weights: &mut HashMap<String, Vec<f32>>,
    logits: &Array2<f32>,
    batch_size: usize,
    num_classes: usize,
) {
    let needed = batch_size * num_classes;
    let logit_data: Vec<f32> = logits.iter().copied().collect();

    for data in weights.values_mut() {
        if data.len() >= needed {
            data[..needed].copy_from_slice(&logit_data);
            return;
        }
    }
}

/// Known model size patterns: (substring, parameter count).
/// Ordered largest-first so "70b" matches before "7b".
const MODEL_SIZE_PATTERNS: &[(&str, u64)] = &[
    ("70b", 70_000_000_000),
    ("65b", 65_000_000_000),
    ("33b", 30_000_000_000),
    ("30b", 30_000_000_000),
    ("13b", 13_000_000_000),
    ("8b", 7_000_000_000),
    ("7b", 7_000_000_000),
    ("3b", 3_000_000_000),
    ("1.1b", 1_100_000_000),
    ("1b", 1_100_000_000),
    ("350m", 350_000_000),
    ("base", 350_000_000),
    ("125m", 125_000_000),
    ("small", 125_000_000),
];

/// Estimate parameter count from model ID.
fn estimate_params_from_model_id(model_id: &str) -> u64 {
    let lower = model_id.to_lowercase();

    MODEL_SIZE_PATTERNS
        .iter()
        .find(|(pattern, _)| lower.contains(pattern))
        .map_or(1_000_000_000, |&(_, count)| count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DistillConfig;

    #[test]
    fn test_estimate_params_from_model_id() {
        assert_eq!(estimate_params_from_model_id("meta-llama/Llama-2-7b"), 7_000_000_000);
        assert_eq!(estimate_params_from_model_id("TinyLlama/TinyLlama-1.1B"), 1_100_000_000);
        assert_eq!(estimate_params_from_model_id("microsoft/codebert-base"), 350_000_000);
    }

    #[test]
    fn test_training_metrics_improvement() {
        let metrics = TrainingMetrics {
            initial_loss: 2.0,
            final_loss: 1.0,
            best_loss: 0.9,
            steps_completed: 1000,
            throughput: 100.0,
        };

        assert!((metrics.improvement_ratio() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_memory_estimation() {
        let config = DistillConfig::minimal("meta-llama/Llama-2-7b", "TinyLlama/TinyLlama-1.1B");

        let estimate = Pipeline::estimate_memory(&config).expect("config should be valid");

        assert!(estimate.total_bytes > 10_000_000_000);
        assert!(estimate.recommended_batch_size > 0);
    }

    #[test]
    fn test_pipeline_result_has_duration() {
        let result = PipelineResult {
            output_path: PathBuf::from("/tmp/output"),
            metrics: TrainingMetrics::default(),
            duration_seconds: 100.0,
        };

        assert!(result.duration_seconds > 0.0);
    }

    #[test]
    fn test_build_synthetic_logits_shape() {
        let mut weights = HashMap::new();
        weights.insert("w".to_string(), vec![0.5; 256]);

        let logits = build_synthetic_logits(&weights, 4, 32);
        assert_eq!(logits.shape(), &[4, 32]);
    }

    #[test]
    fn test_build_synthetic_logits_empty_weights() {
        let weights = HashMap::new();
        let logits = build_synthetic_logits(&weights, 2, 8);
        assert_eq!(logits.shape(), &[2, 8]);
    }

    #[test]
    fn test_kd_gradient_reduces_loss() {
        let teacher = Array2::from_shape_vec((2, 4), vec![2.0, 1.0, 0.5, 0.1, 1.5, 1.2, 0.8, 0.3])
            .expect("operation should succeed");
        let mut student =
            Array2::from_shape_vec((2, 4), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2])
                .expect("operation should succeed");
        let labels = vec![0, 1];
        let loss_fn = DistillationLoss::new(4.0, 0.7);

        let initial_loss = loss_fn.forward(&student, &teacher, &labels);

        // Apply gradient steps
        for _ in 0..100 {
            let grad = kd_gradient(&student, &teacher, &labels, 4.0, 0.7);
            student = &student - &(grad * 0.5);
        }

        let final_loss = loss_fn.forward(&student, &teacher, &labels);
        assert!(
            final_loss < initial_loss,
            "KD gradient did not reduce loss: {initial_loss} -> {final_loss}"
        );
    }

    #[test]
    fn test_resolve_local_path_missing() {
        let result = resolve_model_path("/nonexistent/model.safetensors");
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_local_path_exists() {
        let tmp = tempfile::TempDir::new().expect("temp file creation should succeed");
        let path = tmp.path().join("model.safetensors");
        std::fs::write(&path, b"dummy").expect("file write should succeed");

        let resolved = resolve_model_path(path.to_str().expect("operation should succeed"))
            .expect("operation should succeed");
        assert_eq!(resolved, path);
    }

    #[cfg(not(feature = "hub"))]
    #[test]
    fn test_resolve_hf_model_without_hub_feature() {
        let result = resolve_model_path("meta-llama/Llama-2-7b");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("hub"));
    }

    #[test]
    fn test_pipeline_execute_with_real_safetensors() {
        use safetensors::tensor::{Dtype, TensorView};

        let tmp = tempfile::TempDir::new().expect("temp file creation should succeed");

        // Create teacher SafeTensors
        let teacher_data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
        let teacher_bytes: Vec<u8> = bytemuck::cast_slice(&teacher_data).to_vec();
        let teacher_views = vec![(
            "layer.weight",
            TensorView::new(Dtype::F32, vec![16, 16], &teacher_bytes)
                .expect("operation should succeed"),
        )];
        let teacher_path = tmp.path().join("teacher.safetensors");
        std::fs::write(
            &teacher_path,
            safetensors::serialize(teacher_views, None).expect("file write should succeed"),
        )
        .expect("operation should succeed");

        // Create student SafeTensors
        let student_data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.005).collect();
        let student_bytes: Vec<u8> = bytemuck::cast_slice(&student_data).to_vec();
        let student_views = vec![(
            "layer.weight",
            TensorView::new(Dtype::F32, vec![16, 16], &student_bytes)
                .expect("operation should succeed"),
        )];
        let student_path = tmp.path().join("student.safetensors");
        std::fs::write(
            &student_path,
            safetensors::serialize(student_views, None).expect("file write should succeed"),
        )
        .expect("operation should succeed");

        // Create config pointing to local files
        let output_dir = tmp.path().join("output");
        let mut config = DistillConfig::minimal(
            teacher_path.to_str().expect("config should be valid"),
            student_path.to_str().expect("config should be valid"),
        );
        config.output.dir = output_dir.clone();
        config.training.epochs = 2;
        config.training.batch_size = 4;

        let pipeline = Pipeline::new(&config);
        let result = pipeline.execute().expect("operation should succeed");

        assert!(result.output_path.exists());
        assert!(result.metrics.steps_completed > 0);
        assert!(result.metrics.initial_loss > 0.0);
        assert!(result.duration_seconds > 0.0);

        // Verify distillation metadata sidecar was created
        assert!(output_dir.join("distillation_metadata.json").exists());
    }

    /// Falsification: does training actually reduce loss?
    #[test]
    fn test_falsify_training_reduces_loss() {
        use safetensors::tensor::{Dtype, TensorView};

        let tmp = tempfile::TempDir::new().expect("temp file creation should succeed");

        // Teacher: higher magnitude weights (stronger signal)
        let teacher_data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.02 - 2.0).collect();
        let teacher_bytes: Vec<u8> = bytemuck::cast_slice(&teacher_data).to_vec();
        let teacher_views = vec![(
            "layer.weight",
            TensorView::new(Dtype::F32, vec![16, 16], &teacher_bytes)
                .expect("operation should succeed"),
        )];
        let teacher_path = tmp.path().join("teacher.safetensors");
        std::fs::write(
            &teacher_path,
            safetensors::serialize(teacher_views, None).expect("file write should succeed"),
        )
        .expect("operation should succeed");

        // Student: different initialization
        let student_data: Vec<f32> = (0..256).map(|i| (i as f32) * -0.01 + 1.0).collect();
        let student_bytes: Vec<u8> = bytemuck::cast_slice(&student_data).to_vec();
        let student_views = vec![(
            "layer.weight",
            TensorView::new(Dtype::F32, vec![16, 16], &student_bytes)
                .expect("operation should succeed"),
        )];
        let student_path = tmp.path().join("student.safetensors");
        std::fs::write(
            &student_path,
            safetensors::serialize(student_views, None).expect("file write should succeed"),
        )
        .expect("operation should succeed");

        let output_dir = tmp.path().join("output");
        let mut config = DistillConfig::minimal(
            teacher_path.to_str().expect("config should be valid"),
            student_path.to_str().expect("config should be valid"),
        );
        config.output.dir = output_dir;
        config.training.epochs = 5;
        config.training.batch_size = 4;
        config.training.learning_rate = 0.01;

        let pipeline = Pipeline::new(&config);
        let result = pipeline.execute().expect("operation should succeed");

        eprintln!(
            "initial_loss={}, final_loss={}, best_loss={}, steps={}",
            result.metrics.initial_loss,
            result.metrics.final_loss,
            result.metrics.best_loss,
            result.metrics.steps_completed
        );

        // FALSIFICATION: loss must actually decrease
        assert!(
            result.metrics.final_loss < result.metrics.initial_loss,
            "Training did NOT reduce loss! initial={} final={}",
            result.metrics.initial_loss,
            result.metrics.final_loss
        );
    }

    /// Falsification: does export produce valid re-loadable SafeTensors?
    #[test]
    fn test_falsify_export_roundtrip() {
        use safetensors::tensor::{Dtype, TensorView};

        let tmp = tempfile::TempDir::new().expect("temp file creation should succeed");

        // Create identical teacher/student so training doesn't matter
        let data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
        let data_bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        let views = vec![(
            "layer.weight",
            TensorView::new(Dtype::F32, vec![16, 16], &data_bytes)
                .expect("operation should succeed"),
        )];
        let model_path = tmp.path().join("model.safetensors");
        std::fs::write(
            &model_path,
            safetensors::serialize(views, None).expect("file write should succeed"),
        )
        .expect("file write should succeed");

        let output_dir = tmp.path().join("output");
        let mut config = DistillConfig::minimal(
            model_path.to_str().expect("config should be valid"),
            model_path.to_str().expect("config should be valid"),
        );
        config.output.dir = output_dir.clone();
        config.training.epochs = 1;
        config.training.batch_size = 4;

        let pipeline = Pipeline::new(&config);
        let result = pipeline.execute().expect("operation should succeed");

        // FALSIFICATION: can we re-load the exported file?
        let exported_data = std::fs::read(&result.output_path).expect("file read should succeed");
        let loaded = safetensors::SafeTensors::deserialize(&exported_data)
            .expect("exported SafeTensors file is not valid!");

        // Must contain the same tensor name
        assert!(
            loaded.names().contains(&"layer.weight"),
            "exported file missing 'layer.weight' tensor, has: {:?}",
            loaded.names()
        );

        // Check the data is f32 and has correct shape
        let tensor = loaded.tensor("layer.weight").expect("load should succeed");
        assert_eq!(tensor.dtype(), Dtype::F32);
        assert_eq!(tensor.shape(), &[16, 16]);

        // FALSIFICATION: metadata sidecar must parse as valid JSON
        let meta_path = output_dir.join("distillation_metadata.json");
        let meta_str = std::fs::read_to_string(&meta_path).expect("file read should succeed");
        let meta: entrenar::distill::DistillationCheckpoint =
            serde_json::from_str(&meta_str).expect("metadata sidecar is not valid JSON!");
        assert!(meta.temperature > 0.0);
    }

    /// Falsification: what happens with mismatched teacher/student tensor names?
    #[test]
    fn test_falsify_mismatched_tensor_names() {
        use safetensors::tensor::{Dtype, TensorView};

        let tmp = tempfile::TempDir::new().expect("temp file creation should succeed");

        // Teacher has "encoder.weight"
        let data: Vec<f32> = vec![1.0; 256];
        let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        let teacher_views = vec![(
            "encoder.weight",
            TensorView::new(Dtype::F32, vec![16, 16], &bytes).expect("encoding should succeed"),
        )];
        let teacher_path = tmp.path().join("teacher.safetensors");
        std::fs::write(
            &teacher_path,
            safetensors::serialize(teacher_views, None).expect("file write should succeed"),
        )
        .expect("operation should succeed");

        // Student has "decoder.weight" (completely different name)
        let student_views = vec![(
            "decoder.weight",
            TensorView::new(Dtype::F32, vec![16, 16], &bytes).expect("operation should succeed"),
        )];
        let student_path = tmp.path().join("student.safetensors");
        std::fs::write(
            &student_path,
            safetensors::serialize(student_views, None).expect("file write should succeed"),
        )
        .expect("operation should succeed");

        let output_dir = tmp.path().join("output");
        let mut config = DistillConfig::minimal(
            teacher_path.to_str().expect("config should be valid"),
            student_path.to_str().expect("config should be valid"),
        );
        config.output.dir = output_dir;
        config.training.epochs = 1;
        config.training.batch_size = 4;

        // Should NOT panic even with mismatched tensor names
        let pipeline = Pipeline::new(&config);
        let result = pipeline.execute();
        // This should succeed - gradient step just won't match any names
        assert!(result.is_ok(), "Pipeline panicked on mismatched tensors: {result:?}");
    }

    /// Falsification: single-element tensor edge case
    #[test]
    fn test_falsify_tiny_tensors() {
        use safetensors::tensor::{Dtype, TensorView};

        let tmp = tempfile::TempDir::new().expect("temp file creation should succeed");

        // Single element tensor - too small for batch_size * num_classes
        let data: Vec<f32> = vec![0.5];
        let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        let views = vec![(
            "w",
            TensorView::new(Dtype::F32, vec![1], &bytes).expect("operation should succeed"),
        )];
        let path = tmp.path().join("tiny.safetensors");
        std::fs::write(
            &path,
            safetensors::serialize(views, None).expect("file write should succeed"),
        )
        .expect("file write should succeed");

        let output_dir = tmp.path().join("output");
        let mut config = DistillConfig::minimal(
            path.to_str().expect("config should be valid"),
            path.to_str().expect("config should be valid"),
        );
        config.output.dir = output_dir;
        config.training.epochs = 1;
        config.training.batch_size = 2;

        let pipeline = Pipeline::new(&config);
        // Should NOT panic - should fall back to synthetic logits
        let result = pipeline.execute();
        assert!(result.is_ok(), "Pipeline panicked on tiny tensor: {result:?}");
    }
}
