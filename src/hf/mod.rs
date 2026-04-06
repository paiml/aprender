//! HuggingFace Hub Integration
//!
//! Provides query, pull, and publish capabilities for the HuggingFace ecosystem.
//!
//! ## Commands
//!
//! - `batuta hf catalog` - Query 50+ HuggingFace ecosystem components
//! - `batuta hf search` - Search models, datasets, spaces on Hub
//! - `batuta hf info` - Get metadata for Hub assets
//! - `batuta hf course` - Query by Coursera course alignment
//! - `batuta hf tree` - Visualize HF ecosystem
//! - `batuta hf pull` - Download from Hub
//! - `batuta hf push` - Publish to Hub
//!
//! ## Security Features
//!
//! - SafeTensors enforcement (--safe-only default)
//! - Secret scanning before push
//! - Rate limit handling with exponential backoff

pub mod catalog;
pub mod client;
pub mod hub_client;
pub mod tree;
