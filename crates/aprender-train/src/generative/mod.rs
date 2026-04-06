//! Generative Models for Code Synthesis
//!
//! This module implements Generative Adversarial Networks (GANs) for code generation where:
//! - Generator: Produces Rust AST candidates from latent space
//! - Discriminator: Validates syntax/semantics (approximates rustc)
//!
//! The GAN architecture enables:
//! - Latent space interpolation for handling novel Python constructs
//! - Learning the distribution of valid Rust code
//! - Generating diverse code translations

pub mod code_gan;

pub use code_gan::{
    sigmoid, CodeGan, CodeGanConfig, CodeGanStats, Discriminator, DiscriminatorConfig, Generator,
    GeneratorConfig, LatentCode, TrainingResult,
};
