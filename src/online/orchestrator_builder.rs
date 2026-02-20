
impl<M: OnlineLearner + std::fmt::Debug> OrchestratorBuilder<M> {
    /// Create a new builder
    pub fn new(model: M, n_features: usize) -> Self {
        Self {
            model,
            n_features,
            config: RetrainConfig::default(),
            delta: 0.002,
        }
    }

    /// Set minimum samples for retraining
    pub fn min_samples(mut self, min: usize) -> Self {
        self.config.min_samples = min;
        self
    }

    /// Set maximum buffer size
    pub fn max_buffer_size(mut self, max: usize) -> Self {
        self.config.max_buffer_size = max;
        self
    }

    /// Enable/disable incremental updates
    pub fn incremental_updates(mut self, enable: bool) -> Self {
        self.config.incremental_updates = enable;
        self
    }

    /// Enable/disable curriculum learning
    pub fn curriculum_learning(mut self, enable: bool) -> Self {
        self.config.curriculum_learning = enable;
        self
    }

    /// Set number of curriculum stages
    pub fn curriculum_stages(mut self, stages: usize) -> Self {
        self.config.curriculum_stages = stages;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.config.learning_rate = lr;
        self
    }

    /// Set number of retrain epochs
    pub fn retrain_epochs(mut self, epochs: usize) -> Self {
        self.config.retrain_epochs = epochs;
        self
    }

    /// Set ADWIN delta (sensitivity)
    pub fn adwin_delta(mut self, delta: f64) -> Self {
        self.delta = delta;
        self
    }

    /// Build the orchestrator with ADWIN detector
    pub fn build(self) -> RetrainOrchestrator<M, ADWIN> {
        let detector = ADWIN::with_delta(self.delta);
        RetrainOrchestrator::with_config(self.model, detector, self.n_features, self.config)
    }
}
