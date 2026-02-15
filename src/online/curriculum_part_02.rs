
impl<S: CurriculumScheduler + std::fmt::Debug> CurriculumTrainer<S> {
    /// Create a new curriculum trainer
    pub fn new(scheduler: S) -> Self {
        Self {
            scheduler,
            samples: Vec::new(),
        }
    }

    /// Add samples with difficulty scoring
    pub fn add_samples<D: DifficultyScorer>(
        &mut self,
        features: &[f64],
        targets: &[f64],
        n_features: usize,
        scorer: &D,
    ) -> Result<()> {
        if features.len() % n_features != 0 {
            return Err(AprenderError::dimension_mismatch(
                "features",
                n_features,
                features.len() % n_features,
            ));
        }

        let n_samples = features.len() / n_features;
        if n_samples != targets.len() {
            return Err(AprenderError::dimension_mismatch(
                "targets",
                n_samples,
                targets.len(),
            ));
        }

        for i in 0..n_samples {
            let feat = features[i * n_features..(i + 1) * n_features].to_vec();
            let target = targets[i];
            let difficulty = scorer.score(&feat, target);

            self.samples
                .push(ScoredSample::new(feat, target, difficulty));
        }

        // Sort by difficulty
        self.samples.sort_by(|a, b| {
            a.difficulty
                .partial_cmp(&b.difficulty)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(())
    }

    /// Get samples eligible for current stage
    pub fn eligible_samples(&self) -> Vec<&ScoredSample> {
        let threshold = self.scheduler.current_threshold();

        // Normalize difficulties to 0-1 range
        let (min_diff, max_diff) = self
            .samples
            .iter()
            .fold((f64::MAX, f64::MIN), |(min, max), s| {
                (min.min(s.difficulty), max.max(s.difficulty))
            });

        let range = (max_diff - min_diff).max(1e-10);

        self.samples
            .iter()
            .filter(|s| {
                let normalized = (s.difficulty - min_diff) / range;
                normalized <= threshold
            })
            .collect()
    }

    /// Get number of eligible samples
    pub fn n_eligible(&self) -> usize {
        self.eligible_samples().len()
    }

    /// Advance the curriculum
    pub fn advance(&mut self) {
        self.scheduler.advance();
    }

    /// Check if curriculum is complete
    pub fn is_complete(&self) -> bool {
        self.scheduler.is_complete()
    }

    /// Get current stage
    pub fn stage(&self) -> f64 {
        self.scheduler.stage()
    }

    /// Reset curriculum
    pub fn reset(&mut self) {
        self.scheduler.reset();
    }

    /// Get scheduler reference
    pub fn scheduler(&self) -> &S {
        &self.scheduler
    }
}
