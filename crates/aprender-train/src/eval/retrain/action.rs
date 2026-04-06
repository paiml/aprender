//! Actions taken by the AutoRetrainer.

/// Action taken by the AutoRetrainer
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    /// No action needed
    None,
    /// Warning logged but no retrain triggered
    WarningLogged,
    /// Retraining was triggered with given job ID
    RetrainTriggered(String),
}
