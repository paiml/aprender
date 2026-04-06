//! HuggingFace Hub Publishing
//!
//! Publish trained models, model cards, and evaluation results to
//! HuggingFace Hub repositories.
//!
//! # Example
//!
//! ```ignore
//! use entrenar::hf_pipeline::publish::{HfPublisher, PublishConfig, ModelCard};
//!
//! let config = PublishConfig {
//!     repo_id: "username/my-model".to_string(),
//!     ..Default::default()
//! };
//! let publisher = HfPublisher::new(config)?;
//! let result = publisher.publish(&files, Some(&card))?;
//! println!("Published: {}", result.repo_url);
//! ```

pub mod config;
pub mod model_card;
pub mod publisher;
pub mod result;
pub mod submission;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod falsification_tests;

pub use config::{PublishConfig, RepoType};
pub use model_card::ModelCard;
pub use publisher::HfPublisher;
pub use result::{PublishError, PublishResult};
pub use submission::{format_submission_jsonl, format_submissions_jsonl};
