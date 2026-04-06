//! Generative Adversarial Network for Code Generation
//!
//! Implements a GAN architecture for generating valid Rust AST candidates:
//! - Generator: Maps latent vectors to Rust AST token sequences
//! - Discriminator: Classifies code as real (valid) or fake (invalid)
//!
//! # Architecture
//!
//! ```text
//! Latent Vector z ─┬─► Generator ─► AST Tokens ─┬─► Discriminator ─► Valid/Invalid
//!                  │                            │
//!                  │   Real AST Samples ────────┘
//!                  │
//!                  └── (sampled from N(0, I))
//! ```
//!
//! # Example
//!
//! ```rust
//! use entrenar::generative::{CodeGan, CodeGanConfig};
//!
//! let config = CodeGanConfig::default();
//! let mut gan = CodeGan::new(config);
//!
//! // Training loop would alternate between generator and discriminator updates
//! ```

mod config;
mod discriminator;
mod gan;
mod generator;
mod latent;

pub use config::{CodeGanConfig, DiscriminatorConfig, GeneratorConfig};
pub use discriminator::{sigmoid, Discriminator};
pub use gan::{CodeGan, CodeGanStats, TrainingResult};
pub use generator::Generator;
pub use latent::LatentCode;
