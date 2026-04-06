//! Training result types

/// Result of a training run
#[derive(Debug, Clone)]
pub struct TrainResult {
    /// Final epoch reached
    pub final_epoch: usize,
    /// Final training loss
    pub final_loss: f32,
    /// Best loss achieved
    pub best_loss: f32,
    /// Whether training was stopped early
    pub stopped_early: bool,
    /// Total training time in seconds
    pub elapsed_secs: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_result_clone() {
        let result = TrainResult {
            final_epoch: 5,
            final_loss: 0.1,
            best_loss: 0.05,
            stopped_early: false,
            elapsed_secs: 10.0,
        };
        let cloned = result.clone();
        assert_eq!(result.final_epoch, cloned.final_epoch);
        assert_eq!(result.stopped_early, cloned.stopped_early);
    }
}
