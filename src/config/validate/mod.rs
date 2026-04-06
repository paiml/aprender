//! Configuration validation
//!
//! Validates training specifications for correctness before execution.

mod error;
pub mod json_schema;
mod validator;

#[cfg(test)]
mod proptests;
#[cfg(test)]
mod tests;

pub use error::ValidationError;
pub use validator::validate_config;
