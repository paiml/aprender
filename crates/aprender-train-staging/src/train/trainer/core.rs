//! Core Trainer struct and basic methods

use crate::io::{save_model, Model, ModelFormat, ModelMetadata, SaveConfig};
use crate::optim::Optimizer;
use crate::train::callback::{CallbackContext, CallbackManager, TrainerCallback};
use crate::train::{LossFn, MetricsTracker, TrainConfig};
use crate::Tensor;
use provable_contracts_macros::requires;
use std::path::Path;
use std::time::Instant;

/// High-level trainer that orchestrates the training loop
///
/// # Example
///
/// ```no_run
/// use entrenar::train::{Trainer, TrainConfig, Batch, MSELoss, EarlyStopping};
/// use entrenar::optim::Adam;
/// use entrenar::Tensor;
///
/// // Setup
/// let params = vec![Tensor::zeros(10, true)];
/// let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
/// let config = TrainConfig::default();
///
/// let mut trainer = Trainer::new(params, Box::new(optimizer), config);
/// trainer.set_loss(Box::new(MSELoss));
/// trainer.add_callback(EarlyStopping::new(5, 0.001));
///
/// // Training with callbacks
/// // let result = trainer.train(10, || batches.clone(), |x| x.clone());
/// ```
pub struct Trainer {
    /// Model parameters
    pub(crate) params: Vec<Tensor>,

    /// Optimizer
    pub(crate) optimizer: Box<dyn Optimizer>,

    /// Loss function
    pub(crate) loss_fn: Option<Box<dyn LossFn>>,

    /// Training configuration
    pub(crate) config: TrainConfig,

    /// Metrics tracker
    pub metrics: MetricsTracker,

    /// Callback manager
    pub(crate) callbacks: CallbackManager,

    /// Best loss achieved during training
    pub(crate) best_loss: Option<f32>,

    /// Training start time
    pub(crate) start_time: Option<Instant>,
}

impl Trainer {
    /// Create a new trainer
    pub fn new(params: Vec<Tensor>, optimizer: Box<dyn Optimizer>, config: TrainConfig) -> Self {
        Self {
            params,
            optimizer,
            loss_fn: None,
            config,
            metrics: MetricsTracker::new(),
            callbacks: CallbackManager::new(),
            best_loss: None,
            start_time: None,
        }
    }

    /// Set the loss function
    pub fn set_loss(&mut self, loss_fn: Box<dyn LossFn>) {
        self.loss_fn = Some(loss_fn);
    }

    /// Add a callback to the trainer
    pub fn add_callback<C: TrainerCallback + 'static>(&mut self, callback: C) {
        self.callbacks.add(callback);
    }

    /// Get current learning rate
    pub fn lr(&self) -> f32 {
        self.optimizer.lr()
    }

    /// Set learning rate
    pub fn set_lr(&mut self, lr: f32) {
        self.optimizer.set_lr(lr);
    }

    /// Get reference to model parameters
    pub fn params(&self) -> &[Tensor] {
        &self.params
    }

    /// Get mutable reference to model parameters
    pub fn params_mut(&mut self) -> &mut [Tensor] {
        &mut self.params
    }

    /// Get reference to callback manager
    pub fn callbacks(&self) -> &CallbackManager {
        &self.callbacks
    }

    /// Get mutable reference to callback manager
    pub fn callbacks_mut(&mut self) -> &mut CallbackManager {
        &mut self.callbacks
    }

    /// Build callback context from current state
    pub(crate) fn build_context(
        &self,
        epoch: usize,
        max_epochs: usize,
        step: usize,
        steps_per_epoch: usize,
        loss: f32,
        val_loss: Option<f32>,
    ) -> CallbackContext {
        CallbackContext {
            epoch,
            max_epochs,
            step,
            steps_per_epoch,
            global_step: self.metrics.steps,
            loss,
            lr: self.lr(),
            best_loss: self.best_loss,
            val_loss,
            elapsed_secs: self.start_time.map_or(0.0, |t| t.elapsed().as_secs_f64()),
        }
    }

    /// Save model parameters to a file
    ///
    /// This method persists the trained model weights to disk in SafeTensors format.
    /// Call this after training completes to preserve the learned parameters.
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path (should end in .safetensors)
    /// * `name` - Model name for metadata
    /// * `architecture` - Model architecture description
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use entrenar::train::{Trainer, TrainConfig};
    /// # use entrenar::optim::Adam;
    /// # use entrenar::Tensor;
    /// # let params = vec![Tensor::zeros(10, true)];
    /// # let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
    /// # let mut trainer = Trainer::new(params, Box::new(optimizer), TrainConfig::default());
    /// // After training...
    /// trainer.save("model.safetensors", "my-model", "linear").expect("save failed");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    #[requires(!self.params.is_empty())]
    pub fn save(
        &self,
        path: impl AsRef<Path>,
        name: &str,
        architecture: &str,
    ) -> crate::Result<()> {
        // Convert trainer params to io::Model format
        let params: Vec<(String, Tensor)> = self
            .params
            .iter()
            .enumerate()
            .map(|(i, t)| (format!("param_{i}"), t.clone()))
            .collect();

        let metadata = ModelMetadata::new(name, architecture);
        let model = Model::new(metadata, params);
        let config = SaveConfig::new(ModelFormat::SafeTensors);

        save_model(&model, path, &config)
    }

    /// Save model with custom parameter names
    ///
    /// Like `save()` but allows specifying custom names for each parameter tensor.
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path
    /// * `name` - Model name
    /// * `architecture` - Architecture description
    /// * `param_names` - Names for each parameter (must match params length)
    ///
    /// # Errors
    ///
    /// Returns an error if param_names length doesn't match params or file cannot be written.
    pub fn save_with_names(
        &self,
        path: impl AsRef<Path>,
        name: &str,
        architecture: &str,
        param_names: &[&str],
    ) -> crate::Result<()> {
        if param_names.len() != self.params.len() {
            return Err(crate::Error::InvalidParameter(format!(
                "param_names length {} doesn't match params length {}",
                param_names.len(),
                self.params.len()
            )));
        }

        let params: Vec<(String, Tensor)> = self
            .params
            .iter()
            .zip(param_names.iter())
            .map(|(t, name)| (name.to_string(), t.clone()))
            .collect();

        let metadata = ModelMetadata::new(name, architecture);
        let model = Model::new(metadata, params);
        let config = SaveConfig::new(ModelFormat::SafeTensors);

        save_model(&model, path, &config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::Adam;

    #[test]
    fn test_trainer_creation() {
        let params = vec![Tensor::zeros(10, true)];
        let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let trainer = Trainer::new(params, Box::new(optimizer), config);

        assert_eq!(trainer.params().len(), 1);
        assert_eq!(trainer.lr(), 0.001);
    }

    #[test]
    fn test_set_lr() {
        let params = vec![Tensor::zeros(10, true)];
        let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        assert_eq!(trainer.lr(), 0.001);

        trainer.set_lr(0.01);
        assert_eq!(trainer.lr(), 0.01);
    }

    #[test]
    fn test_params_mut() {
        let params = vec![Tensor::from_vec(vec![1.0, 2.0], true)];
        let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        let params = trainer.params_mut();
        assert_eq!(params.len(), 1);
        // Params should be mutable
        params[0] = Tensor::from_vec(vec![3.0, 4.0], true);
        assert_eq!(trainer.params()[0].data()[0], 3.0);
    }

    #[test]
    fn test_add_callback() {
        use crate::train::ProgressCallback;

        let params = vec![Tensor::zeros(10, true)];
        let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.add_callback(ProgressCallback::new(5));

        // Verify callback was added
        assert!(!trainer.callbacks().is_empty());
    }

    #[test]
    fn test_callbacks_mut() {
        use crate::train::ProgressCallback;

        let params = vec![Tensor::zeros(10, true)];
        let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        assert!(trainer.callbacks().is_empty());

        // Add callback via mutable ref
        trainer.callbacks_mut();
        trainer.add_callback(ProgressCallback::new(10));
        assert!(!trainer.callbacks().is_empty());
    }

    #[test]
    fn test_set_loss() {
        use crate::train::MSELoss;

        let params = vec![Tensor::zeros(10, true)];
        let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        assert!(trainer.loss_fn.is_none());

        trainer.set_loss(Box::new(MSELoss));
        assert!(trainer.loss_fn.is_some());
    }

    #[test]
    fn test_build_context() {
        let params = vec![Tensor::zeros(10, true)];
        let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.best_loss = Some(0.5);
        trainer.start_time = Some(Instant::now());

        let ctx = trainer.build_context(2, 10, 5, 100, 0.1, Some(0.2));

        assert_eq!(ctx.epoch, 2);
        assert_eq!(ctx.max_epochs, 10);
        assert_eq!(ctx.step, 5);
        assert_eq!(ctx.steps_per_epoch, 100);
        assert_eq!(ctx.loss, 0.1);
        assert_eq!(ctx.val_loss, Some(0.2));
        assert_eq!(ctx.best_loss, Some(0.5));
        // When start_time is set, elapsed_secs should be a finite non-negative number
        assert!(ctx.elapsed_secs.is_finite());
    }

    #[test]
    fn test_build_context_no_start_time() {
        let params = vec![Tensor::zeros(10, true)];
        let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let trainer = Trainer::new(params, Box::new(optimizer), config);
        // start_time is None

        let ctx = trainer.build_context(0, 5, 0, 50, 1.0, None);

        assert_eq!(ctx.epoch, 0);
        assert_eq!(ctx.elapsed_secs, 0.0);
        assert!(ctx.val_loss.is_none());
        assert!(ctx.best_loss.is_none());
    }

    #[test]
    fn test_save_with_names_length_mismatch() {
        let params = vec![Tensor::zeros(10, true), Tensor::zeros(20, true)];
        let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let trainer = Trainer::new(params, Box::new(optimizer), config);

        // Mismatch: 2 params, 3 names
        let result =
            trainer.save_with_names("/tmp/test.safetensors", "test", "linear", &["a", "b", "c"]);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("doesn't match"));
    }

    #[test]
    fn test_save() {
        let params = vec![Tensor::from_vec(vec![1.0, 2.0, 3.0], false)];
        let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let trainer = Trainer::new(params, Box::new(optimizer), config);

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_trainer_save.safetensors");

        let result = trainer.save(&path, "test-model", "linear");
        assert!(result.is_ok());

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_with_names() {
        let params = vec![
            Tensor::from_vec(vec![1.0, 2.0], false),
            Tensor::from_vec(vec![3.0, 4.0, 5.0], false),
        ];
        let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let trainer = Trainer::new(params, Box::new(optimizer), config);

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_trainer_save_names.safetensors");

        let result = trainer.save_with_names(&path, "test-model", "mlp", &["weights", "bias"]);
        assert!(result.is_ok());

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_trainer_metrics_tracker() {
        let params = vec![Tensor::zeros(10, true)];
        let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);

        // Metrics tracker should be accessible
        assert_eq!(trainer.metrics.steps, 0);
        trainer.metrics.steps = 100;
        assert_eq!(trainer.metrics.steps, 100);
    }
}
