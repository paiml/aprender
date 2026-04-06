//! Coursera Reading Asset Generation
//!
//! Generates four types of Coursera "Reading" assets from transcripts:
//! - **Banner**: 1200x400 PNG with title and concept bubbles
//! - **Reflection**: Bloom's taxonomy questions with arXiv citations
//! - **Key Concepts**: Extracted concepts with definitions and code examples
//! - **Vocabulary**: Course-wide vocabulary organized by category

pub mod arxiv_db;
pub mod banner;
pub mod key_concepts;
pub mod reflection;
pub mod transcript;
pub mod types;
pub mod vocabulary;

pub use types::*;
