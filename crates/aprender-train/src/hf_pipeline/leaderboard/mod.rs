//! HuggingFace Leaderboard Integration
//!
//! Read and parse HuggingFace open evaluation leaderboards (Open ASR, Open LLM v2,
//! MTEB, BigCodeBench) and compare your models against published results.
//!
//! # Example
//!
//! ```ignore
//! use entrenar::hf_pipeline::leaderboard::{LeaderboardClient, LeaderboardKind};
//!
//! let client = LeaderboardClient::new()?;
//! let hf = client.fetch(LeaderboardKind::OpenASR)?;
//! println!("Top model: {}", hf.entries[0].model_id);
//! ```

pub mod client;
pub mod parser;
pub mod types;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod falsification_tests;

pub use client::LeaderboardClient;
pub use parser::{column_to_metric, compare_with_leaderboard, to_leaderboard};
pub use types::{HfLeaderboard, LeaderboardEntry, LeaderboardKind};
