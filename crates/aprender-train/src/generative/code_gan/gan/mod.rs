//! Code GAN main struct and training logic.

mod code_gan;
mod stats;
mod training_result;

#[cfg(test)]
mod tests;

pub use code_gan::CodeGan;
pub use stats::CodeGanStats;
pub use training_result::TrainingResult;
