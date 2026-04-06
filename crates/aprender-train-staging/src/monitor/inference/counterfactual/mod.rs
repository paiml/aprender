//! Counterfactual Explanations (ENT-104)
//!
//! "What would have changed the decision?"

mod error;
mod explanation;
mod feature_change;

#[cfg(test)]
mod tests;

pub use error::CounterfactualError;
pub use explanation::Counterfactual;
pub use feature_change::FeatureChange;
