//! Model I/O - Loading and saving models
//!
//! Provides functionality to save and load trained models, supporting
//! multiple serialization formats.

mod format;
mod load;
mod model;
mod ordered_metadata;
pub(crate) mod save;

#[cfg(test)]
mod tests;

pub use format::{ModelFormat, SaveConfig};
pub use load::load_model;
pub use model::{Model, ModelMetadata, ParameterInfo};
pub(crate) use ordered_metadata::serialize_with_ordered_metadata;
pub use save::save_model;
