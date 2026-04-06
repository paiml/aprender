//! Core HPO types

mod parameter;
mod space;
mod strategy;
mod trial;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use parameter::{ParameterDomain, ParameterValue};
pub use space::HyperparameterSpace;
pub use strategy::{AcquisitionFunction, SearchStrategy, SurrogateModel};
pub use trial::{Trial, TrialStatus};
