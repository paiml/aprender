//! Model Paradigm Classification (ENT-010)
//!
//! Provides classification of ML model paradigms with associated
//! memory and performance characteristics.

mod fine_tune;
mod model_paradigm;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use fine_tune::FineTuneMethod;
pub use model_paradigm::ModelParadigm;
